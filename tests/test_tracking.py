"""Optical-flow tracking: does a box actually follow a moving object?"""
import _path  # noqa: F401  (adds the repo root to sys.path)

import sys

import cv2
import numpy as np

from markit.tracking import Track

rng = np.random.default_rng(1)
patch = rng.integers(0, 255, (90, 90, 3), dtype=np.uint8)

def frame_at(cx, cy):
    """A textured square on a faintly striped background."""
    f = np.full((480, 640, 3), 45, np.uint8)
    f[::11, :] = 80
    x, y = int(cx - 45), int(cy - 45)
    f[y:y + 90, x:x + 90] = patch
    return f

def gray_at(cx, cy):
    return cv2.cvtColor(frame_at(cx, cy), cv2.COLOR_BGR2GRAY)

def center(t):
    x1, y1, x2, y2 = t.box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")

print("=== follows horizontal motion ===")
g0 = gray_at(120, 240)
t = Track([75, 195, 165, 285], "mug", g0)
check("seeds on a textured object", not t.lost)
gp, worst = g0, 0.0
for i in range(1, 61):
    cx = 120 + i * 7
    g = gray_at(cx, 240)
    t.update(gp, g); gp = g
    if t.lost:
        break
    worst = max(worst, abs(center(t)[0] - cx))
check("survives 420 px of travel", not t.lost)
check(f"worst center error {worst:.1f} px", worst < 5)

print("\n=== follows diagonal motion ===")
g0 = gray_at(150, 150)
t = Track([105, 105, 195, 195], "mug", g0)
gp, worst = g0, 0.0
for i in range(1, 41):
    cx, cy = 150 + i * 6, 150 + i * 5
    g = gray_at(cx, cy)
    t.update(gp, g); gp = g
    if t.lost:
        break
    ex, ey = center(t)
    worst = max(worst, ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5)
check("survives diagonal travel", not t.lost)
check(f"worst center error {worst:.1f} px", worst < 5)

print("\n=== degrades safely ===")
flat = np.full((480, 640), 45, np.uint8)
check("featureless region refuses to seed", Track([100, 100, 200, 200], "blank", flat).lost)
g = gray_at(300, 240)
for box in ([-50, -50, 40, 40], [600, 400, 900, 900], [10, 10, 12, 12], [0, 0, 640, 480]):
    tr = Track(box, "edge", g)
    print(f"       box {str(box):22s} lost={tr.lost} draw={tr.draw_box(640, 480)}")
check("no crash on out-of-frame or degenerate boxes", True)

print("\n=== stale detection catches up (the whole point) ===")
# Florence sees frame 0 but its box only arrives LAG frames later.
SPEED, LAG = 6, 45
g_det = gray_at(120, 240)
det_box = [80, 200, 160, 280]
grays = [gray_at(120 + i * SPEED, 240) for i in range(LAG + 1)]
now_cx = 120 + LAG * SPEED

old_err = abs((det_box[0] + det_box[2]) / 2 - now_cx)   # box drawn as-is
t = Track(det_box, "mug", g_det)
t.update(g_det, grays[0])
for a, b in zip(grays, grays[1:]):
    t.update(a, b)
new_err = abs(center(t)[0] - now_cx)

print(f"       object moved {LAG * SPEED} px during the detection cycle")
print(f"       untracked box would be {old_err:.0f} px behind")
print(f"       tracked box is {new_err:.1f} px off")
check("tracker is not lost after catching up", not t.lost)
check(f"tracked error {new_err:.1f} px beats untracked {old_err:.0f} px", new_err < 5)

print("\n=== track identity and motion state ===")
g0 = gray_at(200, 240)
a = Track([155, 195, 245, 285], "box", g0)
b = Track([155, 195, 245, 285], "box", g0)
check("each track gets a distinct id", a.id != b.id)
check("speed starts at rest", a.speed == 0.0)
gp = g0
for i in range(1, 12):
    g = gray_at(200 + i * 8, 240)
    a.update(gp, g); gp = g
check(f"speed reflects motion ({a.speed:.1f} px/frame)", a.speed > 3)

print("\n" + ("FAILURES: " + ", ".join(fails) if fails else "ALL TRACKING TESTS PASSED"))
sys.exit(1 if fails else 0)
