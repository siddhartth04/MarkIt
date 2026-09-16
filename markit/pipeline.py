"""The live pipeline: camera capture, tracking, rules, and the Florence worker.

Two threads share state behind one lock:

  capture_loop  sole owner of the camera and MediaPipe. Advances trackers,
                evaluates rules, draws overlays, publishes one annotated JPEG.
  worker        runs Florence-2 on the newest frame, independent of video rate.

HTTP handlers never touch the camera — they only read the last published JPEG —
so extra browser tabs cannot cause camera contention.
"""

import threading
import time

import cv2
import numpy as np
from PIL import Image

from .config import (CAM_INDEX, CAPTION_EVERY, DETECTION_PROMPT, SHOW_CAPTION)
from .posture import engine, pose_metrics, RULES
from .rules import _clamp01, scene
from .tracking import Track, label_color
from .vision import (florence, mp_draw, mp_pose, mp_styles, normalize_prompt, pose)

lock = threading.Lock()
latest_raw = None
shared = {"boxes": [], "labels": [], "caption": "", "florence_ms": 0, "fps": 0.0,
          "prompt": normalize_prompt(DETECTION_PROMPT), "jpeg": None,
          "tracking": 0, "pending": False, "active": [], "alerts": [], "metrics": {},
          "counts": {},
          # Live-toggleable from the dashboard. The caption is a SECOND Florence-2
          # pass on top of detection, so switching it off is the biggest single
          # cut to cycle time — and cycle time is the floor on how fast any rule
          # can react.
          "caption_on": SHOW_CAPTION}

# Handoff from the Florence worker to the capture loop. The worker publishes the
# detection ALONG WITH the grayscale frame it ran on, because by the time we see
# it the scene has moved on: the capture loop seeds trackers on that original
# frame, then rolls them forward to the present. Without the frame, boxes could
# only be pinned where the object used to be.
detection_result = None      # dict(boxes, labels, gray, prompt) or None

def worker():
    cycle = 0
    while True:
        with lock:
            frame = None if latest_raw is None else latest_raw.copy()
            prompt = shared["prompt"]
            caption_on = shared["caption_on"]
        if frame is None:
            time.sleep(0.05); continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t0 = time.time()
        boxes, labels = [], []
        caption = shared["caption"]
        try:
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            res = florence(pil, task, text_input=prompt)
            data = res.get(task, {})
            boxes = data.get("bboxes", []); labels = data.get("labels", [])
            if caption_on and cycle % CAPTION_EVERY == 0:
                cap_task = "<DETAILED_CAPTION>"
                caption = florence(pil, cap_task).get(cap_task, "")
            elif not caption_on:
                caption = ""        # clear it, so a stale description is not left on screen
        except Exception as e:
            print("Florence worker error:", e)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        global detection_result
        with lock:
            detection_result = {"boxes": boxes, "labels": labels,
                                "gray": gray, "prompt": prompt}
            shared["labels"] = labels
            shared["caption"] = caption
            shared["florence_ms"] = int((time.time() - t0) * 1000)
            shared["pending"] = False
        cycle += 1


# Video pipeline: ONE capture thread owns the camera + MediaPipe and publishes
# the latest annotated JPEG. HTTP clients only read that buffer, so any number
# of browser tabs / refreshes are safe (no camera or pose contention).
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():                 # fall back if DirectShow can't grab the device
    cap = cv2.VideoCapture(CAM_INDEX)

def placeholder(msg):
    img = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(img, msg, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    ok, jpg = cv2.imencode(".jpg", img)
    return jpg.tobytes()

def capture_loop():
    """Single owner of the camera + MediaPipe. Publishes the latest annotated JPEG."""
    global latest_raw, detection_result
    prev = time.time()
    fails = 0
    tracks = []          # live Track objects, updated every frame
    prev_gray = None     # previous frame's grayscale, for optical flow
    while True:
        if cap is None or not cap.isOpened():
            with lock:
                shared["jpeg"] = placeholder("No webcam found. Set CAM_INDEX (0/1/2) and restart.")
            time.sleep(0.5)
            continue
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 30:
                with lock:
                    shared["jpeg"] = placeholder("Camera read failed. Is another app using it?")
            time.sleep(0.03)
            continue
        fails = 0
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        with lock:
            latest_raw = frame.copy()
            fresh, detection_result = detection_result, None

        if fresh is not None:
            # New detection arrived. Seed trackers on the frame Florence actually
            # saw, then fast-forward them over the frames captured since, so the
            # boxes land on the object where it is NOW rather than where it was.
            tracks = [Track(b, l, fresh["gray"], now=time.time())
                      for b, l in zip(fresh["boxes"], fresh["labels"])]
            tracks = [t for t in tracks if not t.lost]
            for t in tracks:
                t.update(fresh["gray"], gray)
            tracks = [t for t in tracks if not t.lost]
        elif tracks and prev_gray is not None:
            for t in tracks:
                t.update(prev_gray, gray)
            tracks = [t for t in tracks if not t.lost]

        prev_gray = gray

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        metrics, active = None, []
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
            metrics = pose_metrics(res.pose_landmarks.landmark)
        tnow = time.time()
        active = engine.update(metrics, tnow)                 # posture rules
        active += scene.update(tracks, w, h, tnow)            # object rules
        merged = sorted(engine.log, key=lambda a: a["t"], reverse=True)
        merged = list(scene.log)[:12] + merged[:6]
        with lock:
            shared["active"] = active
            shared["metrics"] = {k: round(v, 1) for k, v in (metrics or {}).items()}
            shared["alerts"] = merged[:14]
            shared["counts"] = dict(scene.counts)
        seen = {}
        for t in tracks:
            xy = t.draw_box(w, h)
            if xy is None:
                continue
            x1, y1, x2, y2 = xy
            color = label_color(t.label)
            # Number repeats ("person 1", "person 2") so identical labels stay distinct.
            seen[t.label] = seen.get(t.label, 0) + 1
            tally = [tr.label for tr in tracks].count(t.label)
            text = f"{t.label} {seen[t.label]}" if tally > 1 else str(t.label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        for rule in scene.get_rules():
            try:
                z = rule.get("zone")
                if z:
                    zx1 = int(_clamp01(z.get("x1")) * w); zx2 = int(_clamp01(z.get("x2")) * w)
                    zy1 = int(_clamp01(z.get("y1")) * h); zy2 = int(_clamp01(z.get("y2")) * h)
                    hot = rule.get("name") in active
                    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2),
                                  (0, 60, 255) if hot else (120, 120, 120), 2)
                    cv2.putText(frame, str(rule.get("name", "")), (zx1 + 4, max(14, zy1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 60, 255) if hot else (150, 150, 150), 1, cv2.LINE_AA)
                ln = rule.get("line")
                if ln:
                    if ln.get("axis", "x") == "x":
                        lx = int(_clamp01(ln.get("at", 0.5)) * w)
                        cv2.line(frame, (lx, 0), (lx, h), (255, 220, 80), 2)
                        cv2.putText(frame, f"{rule.get('name','')}: {scene.counts.get(rule.get('name'),0)}",
                                    (lx + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 220, 80), 2, cv2.LINE_AA)
                    else:
                        ly = int(_clamp01(ln.get("at", 0.5)) * h)
                        cv2.line(frame, (0, ly), (w, ly), (255, 220, 80), 2)
                        cv2.putText(frame, f"{rule.get('name','')}: {scene.counts.get(rule.get('name'),0)}",
                                    (8, max(16, ly - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 220, 80), 2, cv2.LINE_AA)
            except Exception:
                pass

        if active:
            warn_names = {r.get("name") for r in scene.get_rules()
                          if r.get("level") == "warn"}
            warn_names |= {r["name"] for r in RULES if r["level"] == "warn"}
            warn = any(a in warn_names for a in active)
            color = (0, 60, 255) if warn else (0, 177, 255)
            text = "  ".join(a.upper() for a in active)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (0, 0), (tw + 24, th + 20), color, -1)
            cv2.putText(frame, text, (12, th + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6); prev = now
        ok, jpg = cv2.imencode(".jpg", frame)
        if ok:
            with lock:
                shared["jpeg"] = jpg.tobytes()
                shared["fps"] = round(fps, 1)
                shared["tracking"] = len(tracks)
        time.sleep(0.005)

def frames():
    """Light generator: just stream whatever the capture thread last published."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with lock:
            jpg = shared["jpeg"]
        if jpg is None:
            time.sleep(0.05); continue
        yield boundary + jpg + b"\r\n"
        time.sleep(0.03)   # ~30 fps stream cap
