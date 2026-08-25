"""Follow detected boxes between Florence-2 cycles.

Florence-2 needs seconds per cycle on CPU, so its boxes are stale the moment
they arrive. Florence decides WHAT is in frame (prompt-driven); this module
decides WHERE it is right now, following each box at full framerate.
Measured ~3 ms/frame for one box, ~16 ms for five.
"""

import math
import time

import cv2
import numpy as np

from .config import MAX_POINTS, MIN_POINTS, MAX_SCALE

PALETTE = [(0,177,255), (80,220,120), (255,140,80), (200,120,255),
           (80,200,255), (255,200,60), (140,255,200), (255,120,180)]
_color_of = {}
def label_color(label):
    key = str(label).strip().lower()
    if key not in _color_of:
        _color_of[key] = PALETTE[len(_color_of) % len(PALETTE)]
    return _color_of[key]

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

_next_track_id = 0

class Track:
    """One detected object: its label, current box, and the points we follow it by.

    Carries the state industrial rules need: a stable id (so an item is counted
    once, not once per frame), when it appeared, and recent motion (so a stalled
    conveyor is distinguishable from a moving one).
    """
    def __init__(self, box, label, gray, now=None):
        global _next_track_id
        _next_track_id += 1
        self.id = _next_track_id
        self.label = label
        self.box = [float(v) for v in box]
        self.pts = self._seed(gray, self.box)
        self.lost = self.pts is None
        self.born = now if now is not None else time.time()
        self.speed = 0.0            # px/frame, smoothed
        self.counted = set()        # names of count-rules that already consumed it
        self.prev_center = self.center()

    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def age(self, now):
        return now - self.born

    @staticmethod
    def _seed(gray, box):
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        roi = gray[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=MAX_POINTS,
                                      qualityLevel=0.01, minDistance=4)
        if pts is None or len(pts) < MIN_POINTS:
            return None
        pts = pts.astype(np.float32)
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1
        return pts

    def update(self, prev_gray, gray):
        """Advance the box by the median motion of its surviving feature points."""
        if self.lost or self.pts is None or len(self.pts) < MIN_POINTS:
            self.lost = True
            return
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.pts, None, **LK_PARAMS)
        if nxt is None or status is None:
            self.lost = True
            return
        keep = status.flatten() == 1
        if keep.sum() < MIN_POINTS:
            self.lost = True
            return
        old = self.pts[keep].reshape(-1, 2)
        new = nxt[keep].reshape(-1, 2)

        # Translate by median point motion (median resists a few bad vectors).
        dx, dy = np.median(new - old, axis=0)

        # Scale by how the point cloud's spread changed, guarded against blowup.
        spread_old = np.median(np.abs(old - old.mean(0))) + 1e-6
        spread_new = np.median(np.abs(new - new.mean(0))) + 1e-6
        scale = float(np.clip(spread_new / spread_old, 1 / MAX_SCALE, MAX_SCALE))
        if not (0.5 < scale < 2.0):     # per-frame change that large is bad flow, not motion
            scale = 1.0

        x1, y1, x2, y2 = self.box
        cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
        half_w = (x2 - x1) / 2 * scale
        half_h = (y2 - y1) / 2 * scale
        self.box = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
        self.pts = new.reshape(-1, 1, 2).astype(np.float32)

        # Exponentially-smoothed speed: one noisy frame shouldn't read as motion,
        # and a jam is "slow for a while" rather than "slow right now".
        c = self.center()
        step = math.hypot(c[0] - self.prev_center[0], c[1] - self.prev_center[1])
        self.speed = 0.7 * self.speed + 0.3 * step
        self.prev_center = c

    def draw_box(self, w, h):
        """Integer, in-frame box for drawing. None if it has left the view."""
        x1, y1, x2, y2 = (int(round(v)) for v in self.box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return None if x2 - x1 < 4 or y2 - y1 < 4 else (x1, y1, x2, y2)

# ----------------------------------------------------------------------------
