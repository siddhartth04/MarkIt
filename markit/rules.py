"""The generic condition engine.

Rules are data, not code, so they can be edited live from the dashboard and
cover unrelated jobs (jam detection, product counting, alignment) with one
mechanism.
"""

import threading
import time
from collections import deque

# ----------------------------------------------------------------------------
# Scene rules — the generic condition engine
# ----------------------------------------------------------------------------
# Rules are data, not code, so they can be edited live from the dashboard and
# cover unrelated jobs (jam detection, product counting, alignment) with one
# mechanism. Each rule names a `when` type, the object `label` it applies to,
# and type-specific fields.
#
#   count    — how many of `label` are in view (or in `zone`);
#              fires when the count falls outside [min, max]
#   stalled  — objects of `label` moving slower than `speed_px` for `hold` s
#              (a conveyor jam: things are present but not advancing)
#   crossing — tally objects whose center crosses `line` (a counter, not an alert)
#   zone     — objects of `label` inside/outside a rectangle
#              (alignment: fires when a product drifts off the expected lane)
#   absent   — no `label` seen for `hold` s (starved line, missing operator)
#
# Coordinates are fractions of frame size (0-1) so a rule survives a resolution
# change or a different camera.

# Starts empty on purpose: a rule targeting a label the watch list does not
# detect fires constantly and teaches you to ignore alerts. Add rules from the
# dashboard once you can see what your camera actually detects.
DEFAULT_RULES = []

def _clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0

def _match(track, label):
    """A rule label of '' or 'any' applies to every object."""
    if not label or label.strip().lower() in ("any", "*"):
        return True
    return label.strip().lower() in str(track.label).strip().lower()

def _in_zone(track, zone, w, h):
    cx, cy = track.center()
    x1 = _clamp01(zone.get("x1")) * w
    x2 = _clamp01(zone.get("x2")) * w
    y1 = _clamp01(zone.get("y1")) * h
    y2 = _clamp01(zone.get("y2")) * h
    return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)

def _side_of_line(track, line, w, h):
    """Which side of the line the center is on: +1 / -1. Vertical or horizontal."""
    cx, cy = track.center()
    if line.get("axis", "x") == "x":
        return 1 if cx >= _clamp01(line.get("at", 0.5)) * w else -1
    return 1 if cy >= _clamp01(line.get("at", 0.5)) * h else -1

class SceneEngine:
    """Evaluates rules against the live tracks, holds state, and logs events.

    `hold` matters everywhere: a single frame crossing a threshold is usually
    noise (a lost feature point, one bad detection). Requiring a condition to
    persist is what separates a real event from a twitch.
    """
    def __init__(self, rules=None, max_log=60):
        self.rules = list(rules if rules is not None else DEFAULT_RULES)
        self.since = {}        # rule name -> when it first became true
        self.firing = set()    # rules past their hold time
        self.counts = {}       # rule name -> running tally (crossing rules)
        self.sides = {}        # (rule, track id) -> last side seen
        self.log = deque(maxlen=max_log)
        self.lock = threading.Lock()

    # -- rule management (called from HTTP handlers) -------------------------
    def set_rules(self, rules):
        with self.lock:
            self.rules = rules
            self.since.clear()
            self.firing.clear()
            self.sides.clear()
            # counts survive an edit so a shift tally is not lost by retuning
            # an unrelated rule

    def get_rules(self):
        with self.lock:
            return list(self.rules)

    def reset_counts(self):
        with self.lock:
            self.counts.clear()
            self.sides.clear()

    def _name(self, rule):
        return rule.get("name") or rule.get("when", "rule")

    def _fire(self, rule, detail):
        name = self._name(rule)
        if name not in self.firing:          # rising edge only: one log per event
            self.firing.add(name)
            self.log.appendleft({
                "t": time.strftime("%H:%M:%S"),
                "rule": name,
                "level": rule.get("level", "warn"),
                "detail": detail,
            })

    def _clear(self, rule):
        name = self._name(rule)
        self.since.pop(name, None)
        self.firing.discard(name)

    def _held(self, rule, now):
        """True once the condition has been continuously true for `hold` seconds."""
        start = self.since.setdefault(self._name(rule), now)
        return now - start >= float(rule.get("hold", 0) or 0)

    def update(self, tracks, w, h, now):
        """Returns the list of currently-firing rule names."""
        active = []
        for rule in self.get_rules():
            try:
                name = self._name(rule)
                kind = rule.get("when")
                label = rule.get("label", "")
                mine = [t for t in tracks if _match(t, label)]

                if kind == "count":
                    zone = rule.get("zone")
                    if zone:
                        mine = [t for t in mine if _in_zone(t, zone, w, h)]
                    n = len(mine)
                    lo, hi = rule.get("min"), rule.get("max")
                    bad = ((lo is not None and n < int(lo)) or
                           (hi is not None and n > int(hi)))
                    if bad and self._held(rule, now):
                        active.append(name)
                        self._fire(rule, f"{n} {label or 'objects'} in view")
                    elif not bad:
                        self._clear(rule)

                elif kind == "stalled":
                    thresh = float(rule.get("speed_px", 0.6))
                    slow = [t for t in mine if t.speed < thresh]
                    # Only a jam if something is there but not advancing; an
                    # empty belt is "absent", a different condition.
                    if mine and len(slow) == len(mine) and self._held(rule, now):
                        active.append(name)
                        avg = sum(t.speed for t in mine) / len(mine)
                        self._fire(rule, f"{len(mine)} stopped ({avg:.1f} px/f)")
                    elif not mine or len(slow) < len(mine):
                        self._clear(rule)

                elif kind == "absent":
                    if not mine and self._held(rule, now):
                        active.append(name)
                        self._fire(rule, f"no {label or 'objects'} seen")
                    elif mine:
                        self._clear(rule)

                elif kind == "zone":
                    zone = rule.get("zone") or {}
                    want_in = bool(rule.get("inside", True))
                    off = [t for t in mine if _in_zone(t, zone, w, h) != want_in]
                    if off and self._held(rule, now):
                        active.append(name)
                        where = "outside" if want_in else "inside"
                        self._fire(rule, f"{len(off)} {where} zone")
                    elif not off:
                        self._clear(rule)

                elif kind == "crossing":
                    line = rule.get("line") or {"axis": "x", "at": 0.5}
                    want = int(rule.get("direction", 1))
                    for t in mine:
                        key = (name, t.id)
                        side = _side_of_line(t, line, w, h)
                        prev = self.sides.get(key)
                        self.sides[key] = side
                        # Count on the transition only, one direction only, and
                        # once per track — so an item jittering on the line
                        # cannot inflate the tally.
                        if prev is not None and prev != side and side == want:
                            if name not in t.counted:
                                t.counted.add(name)
                                self.counts[name] = self.counts.get(name, 0) + 1
                                self.log.appendleft({
                                    "t": time.strftime("%H:%M:%S"), "rule": name,
                                    "level": "count",
                                    "detail": f"#{self.counts[name]} {t.label}",
                                })
                    live = {t.id for t in tracks}
                    stale = [k for k in self.sides if k[0] == name and k[1] not in live]
                    for k in stale:
                        self.sides.pop(k, None)
            except Exception as e:      # a bad rule must never kill the video
                print(f"rule error in {rule.get('name', '?')}: {e}")
        return active

scene = SceneEngine()
