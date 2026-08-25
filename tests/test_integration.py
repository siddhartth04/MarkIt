"""End-to-end: synthetic video -> real Track objects -> real rule engine.

The rule-engine suite uses hand-built FakeTracks, which proves the decision
logic but skips everything that feeds it. This suite drives REAL Track objects
through REAL optical flow on generated video, so a break anywhere in
seed -> update -> speed -> rule evaluation shows up here.
"""
import _path  # noqa: F401  (adds the repo root to sys.path)

import sys

import cv2
import numpy as np

from markit.rules import SceneEngine
from markit.tracking import Track

W, H = 640, 480
rng = np.random.default_rng(3)
TEXTURE = rng.integers(0, 255, (70, 70, 3), dtype=np.uint8)

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")

def scene(positions):
    """Frame with a textured box at each (cx, cy)."""
    f = np.full((H, W, 3), 50, np.uint8)
    f[::13, :] = 78                       # background texture, so flow has context
    for cx, cy in positions:
        x, y = int(cx - 35), int(cy - 35)
        x, y = max(0, min(W - 70, x)), max(0, min(H - 70, y))
        f[y:y + 70, x:x + 70] = TEXTURE
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

def box_at(cx, cy):
    return [cx - 35, cy - 35, cx + 35, cy + 35]

def run(engine, path_fn, frames, dt=1 / 30, label="box"):
    """Drive real tracks along a path and feed them to the engine each frame."""
    g_prev = scene(path_fn(0))
    tracks = [Track(box_at(*p), label, g_prev, now=0.0) for p in path_fn(0)]
    tracks = [t for t in tracks if not t.lost]
    t_now, actives = 0.0, []
    for i in range(1, frames):
        g = scene(path_fn(i))
        for tr in tracks:
            tr.update(g_prev, g)
        tracks = [tr for tr in tracks if not tr.lost]
        g_prev = g
        t_now += dt
        actives = engine.update(tracks, W, H, t_now)
    return tracks, actives, t_now

print("=== moving belt: real tracks, real flow, jam rule stays quiet ===")
eng = SceneEngine([{"name": "jam", "when": "stalled", "label": "box",
                    "speed_px": 1.0, "hold": 2.0, "level": "warn"}])
# 70 frames keeps the object clear of the right edge; running it into the wall
# would stall it and decay the speed EMA, which is a property of the test scene,
# not of the tracker.
tracks, active, _ = run(eng, lambda i: [(80 + i * 6, 240)], 70)
check("track survived the whole run", len(tracks) == 1)
check(f"measured speed {tracks[0].speed:.2f} px/frame matches the true 6.0",
      abs(tracks[0].speed - 6.0) < 0.5)
check("moving belt does not fire the jam rule", active == [], f"got {active}")
check("nothing logged", len(eng.log) == 0)

print("\n=== stopped belt: same pipeline, jam rule fires ===")
eng2 = SceneEngine([{"name": "jam", "when": "stalled", "label": "box",
                     "speed_px": 1.0, "hold": 2.0, "level": "warn"}])
# move for a while, then stop dead
def stalling(i):
    return [(80 + min(i, 30) * 6, 240)]
tracks2, active2, _ = run(eng2, stalling, 150)
check("track survived", len(tracks2) == 1)
check(f"speed decayed to {tracks2[0].speed:.2f} px/frame", tracks2[0].speed < 1.0)
check("jam rule fired", "jam" in active2, f"got {active2}")
check("logged exactly one jam event", len(eng2.log) == 1)
if eng2.log:
    print(f"       logged: {dict(eng2.log[0])}")

print("\n=== product count: three real objects crossing a line ===")
eng3 = SceneEngine([{"name": "output", "when": "crossing", "label": "box",
                     "line": {"axis": "x", "at": 0.5}, "direction": 1}])
def three(i):
    return [(60 + i * 7, 150), (30 + i * 7, 250), (10 + i * 7, 350)]
tracks3, _, _ = run(eng3, three, 80)
got = eng3.counts.get("output", 0)
check(f"counted {got} of 3 objects", got == 3, f"tracks alive: {len(tracks3)}")

print("\n=== alignment: real object drifting out of its lane ===")
eng4 = SceneEngine([{"name": "off lane", "when": "zone", "label": "box",
                     "zone": {"x1": 0.1, "y1": 0.3, "x2": 0.9, "y2": 0.7},
                     "inside": True, "hold": 1.0}])
# starts centred in the lane, then drifts upward out of it
def drifting(i):
    return [(300, 240 - i * 4)]
_, active4, _ = run(eng4, drifting, 90)
check("zone rule caught the drift", "off lane" in active4, f"got {active4}")

print("\n=== alignment: object staying in its lane stays quiet ===")
eng5 = SceneEngine([{"name": "off lane", "when": "zone", "label": "box",
                     "zone": {"x1": 0.1, "y1": 0.3, "x2": 0.9, "y2": 0.7},
                     "inside": True, "hold": 1.0}])
_, active5, _ = run(eng5, lambda i: [(120 + i * 4, 240)], 90)
check("aligned product does not alert", active5 == [], f"got {active5}")

print("\n=== absent: belt empties out ===")
eng6 = SceneEngine([{"name": "starved", "when": "absent", "label": "box",
                     "hold": 1.0}])
# real tracks for a while, then feed an empty list (objects gone)
g0 = scene([(200, 240)])
tr = [Track(box_at(200, 240), "box", g0, now=0.0)]
t = 0.0
gp = g0
for i in range(1, 30):
    g = scene([(200 + i * 5, 240)])
    for x in tr:
        x.update(gp, g)
    gp = g
    t += 1 / 30
    a = eng6.update([x for x in tr if not x.lost], W, H, t)
check("present object: quiet", a == [], f"got {a}")
for _ in range(60):
    t += 1 / 30
    a = eng6.update([], W, H, t)
check("belt empty past hold: fires", "starved" in a, f"got {a}")

print("\n=== count rule over real tracks ===")
eng7 = SceneEngine([{"name": "too many", "when": "count", "label": "box",
                     "max": 2, "hold": 0.5}])
_, active7, _ = run(eng7, lambda i: [(60 + i * 5, 140), (60 + i * 5, 240),
                                     (60 + i * 5, 340)], 60)
check("3 objects trips max=2", "too many" in active7, f"got {active7}")

print("\n=== label filtering across mixed objects ===")
eng8 = SceneEngine([{"name": "phones only", "when": "count", "label": "phone",
                     "max": 0, "hold": 0.2}])
g0 = scene([(150, 200), (400, 300)])
mixed = [Track(box_at(150, 200), "person", g0, now=0.0),
         Track(box_at(400, 300), "phone", g0, now=0.0)]
mixed = [t for t in mixed if not t.lost]
tt = 0.0
gp = g0
for i in range(1, 30):
    g = scene([(150 + i * 3, 200), (400 + i * 3, 300)])
    for x in mixed:
        x.update(gp, g)
    gp = g
    tt += 1 / 30
    a8 = eng8.update(mixed, W, H, tt)
check("rule matched only the phone", "phones only" in a8, f"got {a8}")
eng9 = SceneEngine([{"name": "forklifts", "when": "count", "label": "forklift",
                     "max": 0, "hold": 0.2}])
a9 = []
for i in range(30):
    a9 = eng9.update(mixed, W, H, tt + i / 30)
check("rule ignored absent label", a9 == [], f"got {a9}")

print("\n" + ("FAILURES: " + ", ".join(fails) if fails
              else "ALL INTEGRATION TESTS PASSED"))
sys.exit(1 if fails else 0)
