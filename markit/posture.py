"""Turn MediaPipe landmarks into named posture conditions.

Kept separate from the object rules in `rules.py`: this one reasons about a
human body, that one reasons about detected objects on a line.
"""

import math
import time
from collections import deque

from .vision import mp_pose


L = mp_pose.PoseLandmark

def _mid(a, b):
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)

def _angle(a, b, c):
    """Interior angle at b, in degrees, from three (x, y) points."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1) or 1e-9
    n2 = math.hypot(*v2) or 1e-9
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

def pose_metrics(lm):
    """Reduce raw landmarks to a few human-meaningful numbers.

    Returns None when the key joints aren't confidently visible — a rule should
    go quiet rather than fire on landmarks MediaPipe is guessing at.
    """
    need = [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP]
    if any(lm[j.value].visibility < 0.5 for j in need):
        return None
    sh = _mid(lm[L.LEFT_SHOULDER.value],  lm[L.RIGHT_SHOULDER.value])
    hp = _mid(lm[L.LEFT_HIP.value],       lm[L.RIGHT_HIP.value])

    # Torso tilt away from vertical. Image y grows downward, hence -dy.
    lean = abs(math.degrees(math.atan2(sh[0] - hp[0], -(sh[1] - hp[1]))))

    m = {"lean": lean}

    # Knee/hip angles need the legs in frame; average whichever side is visible.
    knees, hips = [], []
    for side in ("LEFT", "RIGHT"):
        hip_j = lm[getattr(L, f"{side}_HIP").value]
        knee_j = lm[getattr(L, f"{side}_KNEE").value]
        ank_j = lm[getattr(L, f"{side}_ANKLE").value]
        shd_j = lm[getattr(L, f"{side}_SHOULDER").value]
        if min(hip_j.visibility, knee_j.visibility, ank_j.visibility) > 0.5:
            knees.append(_angle((hip_j.x, hip_j.y), (knee_j.x, knee_j.y), (ank_j.x, ank_j.y)))
        if min(shd_j.visibility, hip_j.visibility, knee_j.visibility) > 0.5:
            hips.append(_angle((shd_j.x, shd_j.y), (hip_j.x, hip_j.y), (knee_j.x, knee_j.y)))
    if knees:
        m["knee"] = sum(knees) / len(knees)
    if hips:
        m["hip"] = sum(hips) / len(hips)
    return m

# Each rule: name, a test over the metrics, how long it must hold, and severity.
# `hold` exists because a single frame crossing a threshold is usually noise —
# an arm swinging past the torso, or one bad landmark. Requiring the condition
# to persist turns a twitch into an event worth alerting on.
RULES = [
    {"name": "bending",      "hold": 1.0, "level": "warn",
     "test": lambda m: m["lean"] >= 45,
     "says": lambda m: f"torso at {m['lean']:.0f}deg"},
    {"name": "leaning",      "hold": 2.0, "level": "info",
     "test": lambda m: 25 <= m["lean"] < 45,
     "says": lambda m: f"torso at {m['lean']:.0f}deg"},
    {"name": "sitting",      "hold": 2.0, "level": "info",
     "test": lambda m: "hip" in m and m["hip"] < 120 and m["lean"] < 45,
     "says": lambda m: f"hip at {m['hip']:.0f}deg"},
    {"name": "crouching",    "hold": 1.0, "level": "warn",
     "test": lambda m: "knee" in m and m["knee"] < 100,
     "says": lambda m: f"knee at {m['knee']:.0f}deg"},
]

class RuleEngine:
    """Tracks which conditions currently hold, and logs alerts when they start."""
    def __init__(self, max_log=40):
        self.since = {}          # rule name -> when it first became true
        self.firing = set()      # rules that have passed their hold time
        self.log = deque(maxlen=max_log)

    def update(self, metrics, now):
        active = []
        if metrics is None:                  # nobody / not confident: clear state
            self.since.clear()
            self.firing.clear()
            return active
        for rule in RULES:
            name = rule["name"]
            try:
                held = bool(rule["test"](metrics))
            except (KeyError, TypeError):
                held = False
            if not held:
                self.since.pop(name, None)
                self.firing.discard(name)
                continue
            start = self.since.setdefault(name, now)
            if now - start >= rule["hold"]:
                active.append(name)
                if name not in self.firing:  # rising edge only: log once per event
                    self.firing.add(name)
                    self.log.appendleft({
                        "t": time.strftime("%H:%M:%S"),
                        "rule": name,
                        "level": rule["level"],
                        "detail": rule["says"](metrics),
                    })
        return active

engine = RuleEngine()

# Florence-2 needs seconds per cycle on CPU, so its boxes are stale the moment
# they arrive. Florence decides WHAT is in frame (prompt-driven); this tracker
# decides WHERE it is right now, following each box between detections at full
# framerate. Measured ~3 ms/frame for one box, ~16 ms for five.
# One stable BGR color per object type, so 'person' and 'phone' boxes never
# look alike. Keyed by label text, so a label keeps its color across cycles.
