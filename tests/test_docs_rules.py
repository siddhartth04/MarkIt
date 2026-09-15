"""Check docs/RULES.md against the rule engine it documents.

A reference doc that disagrees with the code is worse than none: readers trust
it. This fails when a field, rule type, or default drifts apart from rules.py.
"""
import _path  # noqa: F401  (adds the repo root to sys.path)

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from markit.rules import SceneEngine, _in_zone, _match  # noqa: E402

DOC = os.path.join("docs", "RULES.md")
doc = open(DOC, encoding="utf-8").read()
rules_src = open(os.path.join("markit", "rules.py"), encoding="utf-8").read()
web_src = open(os.path.join("markit", "web.py"), encoding="utf-8").read()

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")


class T:
    """Minimal stand-in for a Track."""
    _n = 0
    def __init__(self, label, cx=0, cy=0, speed=5.0):
        T._n += 1
        self.id, self.label, self.speed = T._n, label, speed
        self.counted, self._c = set(), (cx, cy)
    def center(self):
        return self._c


print("=== every rule type the server accepts is documented ===")
valid = set(re.search(r"valid = \{([^}]+)\}", web_src).group(1)
            .replace('"', "").replace(" ", "").split(","))
for kind in sorted(valid):
    check(f"'{kind}' has a section", f"## `{kind}`" in doc)
documented = set(re.findall(r"^## `(\w+)`", doc, re.M))
check("no section for a rule type that does not exist", documented <= valid,
      f"extra: {documented - valid}")

print("\n=== documented fields exist in the implementation ===")
for field in ["name", "when", "label", "hold", "level", "min", "max",
              "speed_px", "zone", "inside", "line", "direction"]:
    check(f"`{field}` is read by rules.py", f'"{field}"' in rules_src)

print("\n=== the doc's claims about behaviour actually hold ===")
W, H = 640, 480

# substring, case-insensitive label matching
check("'box' matches 'cardboard box'", _match(T("cardboard box"), "box"))
check("matching ignores case", _match(T("Box"), "BOX"))
check("'any' matches everything", _match(T("whatever"), "any"))
check("'*' matches everything", _match(T("whatever"), "*"))
check("blank matches everything", _match(T("whatever"), ""))
check("unrelated labels do not match", not _match(T("person"), "box"))

# zone edges inclusive, reversed coords normalised, out-of-range clamped
z = {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}
check("zone edges are inclusive", _in_zone(T("x", 160, 120), z, W, H))
check("just outside is excluded", not _in_zone(T("x", 159, 240), z, W, H))
check("reversed coords still work",
      _in_zone(T("x", 320, 240), {"x1": .75, "y1": .75, "x2": .25, "y2": .25}, W, H))
check("out-of-range coords clamp",
      _in_zone(T("x", 320, 240), {"x1": -5, "y1": -5, "x2": 99, "y2": 99}, W, H))

# hold timer resets on a single false frame, rather than accumulating
e = SceneEngine([{"name": "r", "when": "count", "label": "box", "max": 0, "hold": 1.0}])
t = 0.0
for _ in range(9):
    a = e.update([T("box", 10, 10)], W, H, t); t += 0.1
check("0.9s of a 1.0s hold does not fire", a == [])
e.update([], W, H, t); t += 0.1
for _ in range(9):
    a = e.update([T("box", 10, 10)], W, H, t); t += 0.1
check("timer resets on one false frame (not cumulative)", a == [])
for _ in range(3):
    a = e.update([T("box", 10, 10)], W, H, t); t += 0.1
check("fires once past the hold", a == ["r"])

# hold 0 fires immediately
e0 = SceneEngine([{"name": "z", "when": "count", "label": "box", "max": 0, "hold": 0}])
check("hold:0 fires on the first frame",
      e0.update([T("box", 1, 1)], W, H, 100.0) == ["z"])

# rising edge: one log per event, a new one after it clears
e2 = SceneEngine([{"name": "r", "when": "absent", "label": "box", "hold": 0.5}])
t = 0.0
for _ in range(10):
    e2.update([], W, H, t); t += 0.1
first = len(e2.log)
for _ in range(15):
    e2.update([], W, H, t); t += 0.1
check("a sustained condition logs once", len(e2.log) == first == 1)
for _ in range(5):
    e2.update([T("box", 1, 1)], W, H, t); t += 0.1
for _ in range(10):
    e2.update([], W, H, t); t += 0.1
check("clearing then recurring logs a second event", len(e2.log) == 2)

# stalled requires objects present
e3 = SceneEngine([{"name": "j", "when": "stalled", "label": "box",
                   "speed_px": 1.0, "hold": 0.2}])
t = 0.0
for _ in range(20):
    a = e3.update([], W, H, t); t += 0.1
check("an empty scene is not a jam", a == [])

# crossing: counted once per track, direction-sensitive, needs a prior side
line = {"name": "c", "when": "crossing", "label": "box",
        "line": {"axis": "x", "at": 0.5}, "direction": 1}
e4 = SceneEngine([line])
o = T("box", 100, 240)
t = 0.0
for x in list(range(100, 600, 25)) + list(range(575, 100, -25)) + list(range(100, 600, 25)):
    o._c = (x, 240)
    e4.update([o], W, H, t); t += 0.05
check("crossing counts once per track, even after returning",
      e4.counts.get("c") == 1, f"got {e4.counts.get('c')}")

e5 = SceneEngine([line])
spawned_past = T("box", 500, 240)
for _ in range(5):
    e5.update([spawned_past], W, H, t); t += 0.1
check("an object first seen past the line is not counted",
      e5.counts.get("c", 0) == 0)
check("doc warns about line placement near the frame edge",
      "already past the line" in doc)

e6 = SceneEngine([{**line, "direction": -1}])
o6 = T("box", 600, 240)
for x in range(600, 100, -25):
    o6._c = (x, 240)
    e6.update([o6], W, H, t); t += 0.05
check("direction -1 counts right-to-left", e6.counts.get("c") == 1)

print("\n=== doc does not contradict the shipped defaults ===")
check("says the rule list starts empty",
      "DEFAULT_RULES = []" in rules_src and "no rules" in doc.lower()
      or "DEFAULT_RULES = []" not in rules_src)

print("\n=== internal anchors resolve ===")
anchors = set(re.findall(r"\]\(#([\w-]+)\)", doc))
heads = {re.sub(r"[^\w\s-]", "", h).strip().lower().replace(" ", "-")
         for h in re.findall(r"^#{2,3} (.+)$", doc, re.M)}
check(f"all {len(anchors)} internal links resolve", anchors <= heads,
      f"broken: {sorted(anchors - heads)}")

print("\n=== README points at this document ===")
readme = open("README.md", encoding="utf-8").read()
check("README links docs/RULES.md", "docs/RULES.md" in readme)

print("\n" + ("FAILURES: " + ", ".join(fails) if fails else "RULES DOC VERIFIED"))
sys.exit(1 if fails else 0)
