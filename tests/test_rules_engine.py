"""Exercise every rule type with fake tracks. No camera, no model needed."""
import _path  # noqa: F401  (adds the repo root to sys.path)

import importlib.util
import markit.web, markit.rules, markit.tracking, markit.posture, markit.pipeline, markit.vision
import types
m = types.SimpleNamespace(**{**vars(markit.rules), **vars(markit.tracking), **vars(markit.posture), **vars(markit.vision), **vars(markit.pipeline), **vars(markit.web)})

W, H = 640, 480

class FakeTrack:
    _n = 0
    def __init__(self, label, cx, cy, speed=5.0):
        FakeTrack._n += 1
        self.id = FakeTrack._n
        self.label = label
        self.speed = speed
        self.counted = set()
        self.set(cx, cy)
    def set(self, cx, cy):
        self.box = [cx - 20, cy - 20, cx + 20, cy + 20]
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

fails = []
def check(name, got, want):
    ok = got == want
    if not ok:
        fails.append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")

print("=== stalled (conveyor jam) ===")
e = m.SceneEngine([{"name": "jam", "when": "stalled", "label": "box",
                    "speed_px": 1.0, "hold": 3.0, "level": "warn"}])
moving = [FakeTrack("box", 100, 100, speed=5.0)]
t = 1000.0
for _ in range(60):
    a = e.update(moving, W, H, t); t += 0.1
check("moving belt does not alert", a, [])

stopped = [FakeTrack("box", 100, 100, speed=0.1)]
for _ in range(20):     # 2.0s — under the 3s hold
    a = e.update(stopped, W, H, t); t += 0.1
check("stopped 2s: not yet", a, [])
for _ in range(15):     # past 3s
    a = e.update(stopped, W, H, t); t += 0.1
check("stopped 3.5s: alerts", a, ["jam"])
check("logged exactly once", len(e.log), 1)
for _ in range(30):
    e.update(stopped, W, H, t); t += 0.1
check("still stopped: no duplicate log", len(e.log), 1)

print("\n=== empty belt is 'absent', not a jam ===")
e2 = m.SceneEngine([{"name": "jam", "when": "stalled", "label": "box",
                     "speed_px": 1.0, "hold": 1.0}])
for _ in range(40):
    a = e2.update([], W, H, t); t += 0.1
check("no objects -> no jam alert", a, [])

print("\n=== crossing (product count) ===")
e3 = m.SceneEngine([{"name": "tally", "when": "crossing", "label": "box",
                     "line": {"axis": "x", "at": 0.5}, "direction": 1}])
item = FakeTrack("box", 100, 240)
for x in range(100, 600, 20):          # left -> right across x=320
    item.set(x, 240)
    e3.update([item], W, H, t); t += 0.1
check("one item crossing counts once", e3.counts.get("tally"), 1)

for x in range(580, 100, -20):         # travels back the wrong way
    item.set(x, 240)
    e3.update([item], W, H, t); t += 0.1
check("reverse direction not counted", e3.counts.get("tally"), 1)

# jitter right on the line must not inflate the count
j = FakeTrack("box", 318, 240)
for k in range(20):
    j.set(318 + (4 if k % 2 else -4), 240)
    e3.update([j], W, H, t); t += 0.1
check("jitter on the line counts at most once", e3.counts.get("tally") <= 2, True)

# three separate items
e4 = m.SceneEngine([{"name": "tally", "when": "crossing", "label": "box",
                     "line": {"axis": "x", "at": 0.5}, "direction": 1}])
items = [FakeTrack("box", 60, 200), FakeTrack("box", 40, 260), FakeTrack("box", 20, 300)]
for step in range(30):
    for it in items:
        cx, cy = it.center()
        it.set(cx + 20, cy)
    e4.update(items, W, H, t); t += 0.1
check("three items -> count 3", e4.counts.get("tally"), 3)

print("\n=== zone (alignment) ===")
e5 = m.SceneEngine([{"name": "off lane", "when": "zone", "label": "box",
                     "zone": {"x1": 0.3, "y1": 0.3, "x2": 0.7, "y2": 0.7},
                     "inside": True, "hold": 1.0}])
inside = [FakeTrack("box", 320, 240)]
for _ in range(30):
    a = e5.update(inside, W, H, t); t += 0.1
check("aligned product: quiet", a, [])
outside = [FakeTrack("box", 600, 60)]
for _ in range(30):
    a = e5.update(outside, W, H, t); t += 0.1
check("drifted product: alerts", a, ["off lane"])

print("\n=== count ===")
e6 = m.SceneEngine([{"name": "too many", "when": "count", "label": "box",
                     "max": 2, "hold": 1.0}])
two = [FakeTrack("box", 100, 100), FakeTrack("box", 200, 200)]
for _ in range(30):
    a = e6.update(two, W, H, t); t += 0.1
check("2 of max 2: quiet", a, [])
three = two + [FakeTrack("box", 300, 300)]
for _ in range(30):
    a = e6.update(three, W, H, t); t += 0.1
check("3 of max 2: alerts", a, ["too many"])

print("\n=== absent ===")
e7 = m.SceneEngine([{"name": "starved", "when": "absent", "label": "box", "hold": 2.0}])
for _ in range(15):
    a = e7.update([], W, H, t); t += 0.1
check("gone 1.5s: not yet", a, [])
for _ in range(10):
    a = e7.update([], W, H, t); t += 0.1
check("gone 2.5s: alerts", a, ["starved"])
a = e7.update([FakeTrack("box", 10, 10)], W, H, t)
check("object returns: clears", a, [])

print("\n=== label matching ===")
e8 = m.SceneEngine([{"name": "r", "when": "count", "label": "any", "max": 0, "hold": 0}])
check("'any' matches everything", e8.update([FakeTrack("widget", 5, 5)], W, H, t), ["r"])

print("\n=== a broken rule must not crash the loop ===")
e9 = m.SceneEngine([{"name": "bad", "when": "zone", "label": "box", "zone": None},
                    {"name": "good", "when": "absent", "label": "box", "hold": 0}])
a = e9.update([], W, H, t)
check("bad rule skipped, good rule still runs", "good" in a, True)

print("\n" + ("FAILURES: " + ", ".join(fails) if fails else "ALL SCENE RULE TESTS PASSED"))
raise SystemExit(1 if fails else 0)
