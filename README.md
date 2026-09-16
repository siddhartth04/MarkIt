# MarkIt

**A local camera monitor you configure by typing what to watch for.**

Point it at a webcam, type `Box. Bottle. Person` into the watch list, and it finds those
things and follows them. Then add rules — *alert me if they stop moving*, *count them
as they pass this line*, *warn me if one drifts out of this area* — and it watches for
those conditions and logs them.

Everything runs on your own machine. Nothing is uploaded, and there is no API key or
account. The only network access is a one-time model download on first run.

---

## Contents

- [What it does](#what-it-does)
- [What it can do](#what-it-can-do) — the five rule types
- [What it uses](#what-it-uses) — models and libraries
- [How it does it](#how-it-does-it) — architecture
- [Install and run](#install-and-run)
- [Using the dashboard](#using-the-dashboard)
- [Writing rules](#writing-rules)
- [Limits worth knowing](#limits-worth-knowing)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

---

## What it does

Three things, in a chain:

**1. It finds what you asked for.** The watch list is plain language, not a fixed
list of classes. It starts on `person.`; type `hard hat. forklift.` and it looks for
those instead, or `red bottle cap.` and it looks for that. No retraining, no config
file — the model is open-vocabulary, so the prompt *is* the configuration.

**2. It follows what it found.** Detection is slow on a CPU — seconds per cycle. If
boxes were only drawn when detection finished, they would lag several seconds behind a
moving object. So a lightweight tracker follows each box between detection cycles at
full frame rate, and the boxes stay on their objects.

**3. It watches for conditions.** Rules run against the tracked objects every frame.
Objects stopped moving? Something left its lane? Fifteen items crossed the line? Each
rule that fires shows on the video, lights up the status bar, and lands in a timestamped
alert log.

The result is a dashboard in your browser: live annotated video, current detections,
active alerts, running counts, and an editor to change the rules without restarting.

---

## What it can do

Five rule types cover most monitoring jobs. Every rule targets a `label` — one of the
things in your watch list, or `any` for everything.

> Full reference with field tables, tuning advice, and a "why isn't my rule firing"
> checklist: **[docs/RULES.md](docs/RULES.md)**.

### `stalled` — things that should be moving, aren't

Fires when every matching object is moving slower than `speed_px` for `hold` seconds.

> **Conveyor jam.** Boxes are on the belt but not advancing.
> An *empty* belt does not fire this — that is `absent`, a different condition.

### `crossing` — count things passing a line

Tallies each object whose centre crosses a line you place on the frame. Counts on the
transition only, one direction only, once per object — so something wobbling on the line
cannot inflate the number.

> **Product count.** Items leaving the line, tallied per shift, resettable from the UI.

### `zone` — things that should be inside (or outside) an area

Fires when a matching object is on the wrong side of a rectangle for `hold` seconds.

> **Alignment.** A product has drifted off the expected lane.
> **Exclusion zone.** A person is inside an area they should not be (`inside: false`).

### `count` — too many, or too few

Fires when the number of matching objects in view — or inside a zone — falls outside
`[min, max]`.

> **Over/under-fill.** More than four items staged at a station.

### `absent` — nothing there at all

Fires when nothing matching has been seen for `hold` seconds.

> **Starved line.** Product stopped arriving.

### Plus: posture conditions

Because the pose model is already running, MarkIt also reports body angles for a person
in frame and fires on four built-in postures — useful for worker-safety monitoring
alongside the line rules:

| Condition | Fires when | Hold |
| --- | --- | --- |
| `bending` | Torso ≥ 45° from vertical | 1.0 s |
| `leaning` | Torso 25–45° from vertical | 2.0 s |
| `sitting` | Hip angle < 120° while upright | 2.0 s |
| `crouching` | Knee angle < 100° | 1.0 s |

These are defined in [`markit/posture.py`](markit/posture.py) and edited in code, not
from the dashboard.

---

## What it uses

| Component | What it is | Job here |
| --- | --- | --- |
| **[Florence-2](https://huggingface.co/microsoft/Florence-2-base)** (`base`, ~0.5 GB) | Microsoft's vision-language model | Open-vocabulary detection — finds objects from your typed prompt, and writes the scene caption |
| **[MediaPipe Pose](https://developers.google.com/mediapipe)** `0.10.14` | Google's on-device pose estimator | 33 body landmarks per frame, for the skeleton overlay and posture rules |
| **[OpenCV](https://opencv.org/)** | Computer-vision library | Camera capture, drawing, and the Lucas-Kanade optical flow that tracks boxes between detections |
| **[PyTorch](https://pytorch.org/)** | ML runtime | Runs Florence-2 on the CPU |
| **[Flask](https://flask.palletsprojects.com/)** | Web framework | Serves the dashboard, the MJPEG video stream, and the JSON API |

The dashboard itself is plain HTML, CSS, and JavaScript in a single file — no build
step, no npm, no framework.

> **`transformers` is pinned to `4.44.2` on purpose.** Florence-2's model code is
> downloaded from the Hub and breaks on newer versions, because the KV-cache API
> changed. If you unpin it, confirm Florence-2 still loads before relying on it.

### Why these choices

**Florence-2 over YOLO or a custom model.** A conventional detector recognises a fixed
list of classes decided at training time. Adding a new product means labelling data and
retraining. Florence-2 is open-vocabulary: you change what it looks for by changing a
sentence. For a factory where the thing on the belt changes, that is the difference
between a config edit and a project.

**Optical flow over a tracking model.** Boxes need updating every frame, so the tracker
has a hard cost ceiling. Measured on this machine:

| Method | 1 object | 5 objects |
| --- | --- | --- |
| OpenCV `TrackerMIL` | 76 ms/frame | 349 ms/frame → **3 fps** |
| **Lucas-Kanade optical flow** | **3 ms/frame** | **16 ms/frame → 64 fps** |

Optical flow is ~20× faster and sub-pixel accurate on textured objects, so that is what
runs. (OpenCV 5.0 removed KCF, CSRT, and MOSSE; `TrackerMIL` was the only built-in left
that needs no extra model files, and it was far too slow.)

---

## How it does it

### The core problem

Florence-2 takes **seconds** per cycle on a CPU. A box computed from a frame captured
three seconds ago is useless on a moving object — it marks where the thing *was*.

### The solution: split "what" from "where"

**Florence-2 decides *what* is in frame. Optical flow decides *where* it is now.**

The detection worker publishes each result *together with the grayscale frame it ran
on*. The capture loop then seeds trackers against that original frame and fast-forwards
them over every frame captured since, so the box lands on the object's current position.

Measured on synthetic motion — an object travelling 270 px during one detection cycle:

| | Box centre error |
| --- | --- |
| Box drawn as detection returned it | 270 px behind |
| **Seeded and fast-forwarded** | **0 px** |

### Threads

Two background threads share state behind one lock:

| Thread | Responsibility |
| --- | --- |
| `capture_loop` | Sole owner of the camera and MediaPipe. Advances trackers, evaluates rules, draws overlays, publishes one annotated JPEG. |
| `worker` | Runs Florence-2 on the newest frame in a loop, independent of the video rate. |

HTTP handlers never touch the camera — they only read the last published JPEG. Any
number of browser tabs can watch without causing camera contention.

### Why rules use a hold time

A single frame crossing a threshold is usually noise: one lost feature point, one bad
detection, an arm swinging past a torso. Every rule (except counting) requires its
condition to stay true for `hold` seconds. That is what separates an event from a twitch.

Alerts log on the **rising edge** only, so a jam lasting ten minutes produces one log
entry, not thousands. The condition clearing and recurring logs a second one.

### Frame-relative coordinates

Zone and line positions are fractions of the frame (`0`–`1`), not pixels, so a rule
survives a resolution change or a swap to a different camera.

---

## Install and run

**Requirements:** Python 3.10–3.12 (`mediapipe 0.10.14` does not support 3.13+), a
webcam, and ~2 GB free disk for model weights.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements_web.txt
```

```bash
python app.py                   # or: python -m markit
```

Open <http://localhost:5000>. Stop with `Ctrl+C`.

The first launch downloads the Florence-2 weights (~0.5 GB) and takes a few minutes.
Later launches load from cache in about 20 seconds. **Wait for `Florence-2 ready.` in
the terminal** before opening the page.

Starting the app **stops any previous instance first**. You can relaunch freely without
hunting for leftover processes.

> This matters more than it sounds. Two instances share one webcam, which makes the
> driver hand back the same frame repeatedly — video freezes around 1 fps and the boxes
> look like they are lagging badly, when the tracker is actually working fine on frames
> that are not changing. Only a process listening on MarkIt's own port is stopped; other
> Python work is left alone.

---

## Using the dashboard

**Watch list** — one object per phrase: `person. phone.` Press **Set**; it applies on
the next detection cycle, no restart. Commas and "and" work too — the input is
normalized before it reaches the model.

> This matters more than it looks. Florence-2 splits the watch list on periods, so
> `person . phone` (with a space before the period) becomes the phrases `"person "` and
> `" phone"`. In testing that returned a **spurious third box** for a two-object scene.
> MarkIt normalizes the input so the format cannot bite you, but the phrasing still has
> to name distinct things.

**Detections** — what was found, with duplicates collapsed (`person x2`). Each object
type gets its own colour in the video; repeats of one type are numbered (`person 1`,
`person 2`).

**Rules** — add, rename, retune, and delete conditions live. Changes take effect
immediately. Zones and count lines are drawn on the video and turn red while their rule
is firing, so you can see what each rule is watching.

**Scene** — a sentence describing the whole frame. The switch in its header turns it
off, and that is worth knowing about: the caption is a *second* Florence-2 pass, and on
this machine it was the difference between a steady ~4.6 s detection cycle and one
spiking to ~10.5 s. Cycle time is the floor on how fast any rule can react, so turn the
caption off when responsiveness matters more than the description.

**Alerts** — running counts as large tiles, and a timestamped log below.

**Status pill** (top right) — turns red and names the firing rule, so a problem is
visible at a glance without reading the log.

**Telemetry** — skeleton FPS (the smooth video loop), Florence cycle time (the slow
detection loop), and how many boxes are currently tracked.

---

## Writing rules

```jsonc
// conveyor jam: boxes on the belt that stop moving for 5s
{"name": "belt jam", "when": "stalled", "label": "box",
 "speed_px": 0.6, "hold": 5, "level": "warn"}

// product count: tally boxes moving left-to-right past mid-frame
{"name": "output", "when": "crossing", "label": "box",
 "line": {"axis": "x", "at": 0.5}, "direction": 1}

// alignment: alert if a box leaves the expected lane for 2s
{"name": "off lane", "when": "zone", "label": "box",
 "zone": {"x1": 0.2, "y1": 0.35, "x2": 0.8, "y2": 0.65},
 "inside": true, "hold": 2}

// starved line: no boxes seen for 20s
{"name": "line starved", "when": "absent", "label": "box", "hold": 20}
```

### Fields

| Field | Applies to | Meaning |
| --- | --- | --- |
| `name` | all | Shown in alerts and on the video. Also the key for its counter. |
| `when` | all | `stalled` · `crossing` · `zone` · `count` · `absent` |
| `label` | all | Which detections it applies to. `any` or blank matches everything. Substring match. |
| `hold` | all but `crossing` | Seconds the condition must persist before alerting |
| `level` | all | `warn` (red) or `info` |
| `speed_px` | `stalled` | Below this many px/frame counts as stopped |
| `min` / `max` | `count` | Allowed range; outside it fires |
| `zone` | `zone`, `count` | `{x1, y1, x2, y2}` as fractions of the frame |
| `inside` | `zone` | `true` = alert when an object leaves; `false` = alert when one enters |
| `line` | `crossing` | `{axis: "x"\|"y", at: 0.0–1.0}` |
| `direction` | `crossing` | `1` or `-1` — which way across the line counts |

MarkIt starts with **no rules**. That is deliberate: a rule targeting a label your watch
list does not detect fires constantly, and an alert panel that is always red teaches you
to ignore it. Watch the Detections panel first to see what your camera actually reports,
then add rules against those exact labels.

The startup watch list is `person.` — change it in the dashboard, or set
`DETECTION_PROMPT` in [`markit/config.py`](markit/config.py) to whatever your line runs.

---

## Limits worth knowing

**Counting is rate-limited by detection speed.** An item must be visible for at least
one Florence-2 cycle to be counted at all. On a fast belt, items can pass between cycles
and be missed entirely. **Verify the tally against a known batch before trusting it for
production numbers.** For high-rate counting, a purpose-built detector or a GPU is the
right answer — this is a monitoring aid, not a certified counter.

The `stalled`, `zone`, `absent`, and `count` rules are far less sensitive to this,
because they describe slow conditions rather than instantaneous events.

**The tracker needs visual texture.** Lucas-Kanade follows corners and edges. A plain
white product on a white belt has few features to lock onto, and its box will drop until
the next detection cycle refreshes it. Better lighting helps; lowering `MIN_POINTS`
keeps boxes alive longer at the cost of more drift.

**Grounding can hallucinate.** `CAPTION_TO_PHRASE_GROUNDING` grounds phrases into the
image rather than deciding whether the thing exists. Ask for `forklift` in a room with
no forklift and it may box something anyway. Keep the watch list to things that are
plausibly present.

**One camera, one process.** There is no multi-camera support, no recording, and no
persistence — rules and counts live in memory and reset when the process stops.

---

## Configuration

Edit [`markit/config.py`](markit/config.py), then restart.

| Setting | Default | Meaning |
| --- | --- | --- |
| `DETECTION_PROMPT` | `person. hard hat. safety vest. forklift.` | Initial watch list |
| `FLORENCE_MODEL` | `microsoft/Florence-2-base` | Use `-large` for better, slower results |
| `SHOW_CAPTION` | `True` | Off spends every cycle on detection instead |
| `CAPTION_EVERY` | `3` | Caption once per N detection cycles |
| `CAM_INDEX` | `0` | Try `1` or `2` if the wrong camera opens |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Server binding |
| `MAX_POINTS` | `24` | Feature points seeded per box |
| `MIN_POINTS` | `4` | Below this a box is considered lost |
| `MAX_SCALE` | `4.0` | Rejects implausible box growth from bad flow |

---

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The dashboard |
| `GET /video_feed` | MJPEG stream of the annotated video |
| `GET /status` | Detections, active rules, alert log, counts, telemetry |
| `POST /set_prompt` | Change the watch list — `{"prompt": "box. bottle."}` |
| `GET /rules` | Current rules and counter tallies |
| `POST /rules` | Replace the rule list — `{"rules": [...]}`. Invalid rules are rejected `400` without disturbing the running set. |
| `POST /set_caption` | Turn the scene caption on or off live — `{"on": false}` |
| `POST /reset_counts` | Zero the crossing tallies (start of shift) |

---

## Project layout

```
app.py                  entry point — python app.py
markit/
  config.py             settings you are likely to change
  vision.py             Florence-2 + MediaPipe loading and inference
  tracking.py           optical-flow box tracking between detection cycles
  rules.py              the generic condition engine
  posture.py            body-angle conditions from pose landmarks
  pipeline.py           camera capture, worker threads, shared state
  web.py                Flask routes
  singleton.py          stops a leftover instance so the camera is not shared
  templates/
    dashboard.html      the dashboard (plain HTML/CSS/JS, no build step)
tests/
  run_all.py            run every suite:  python tests/run_all.py
```

Importing `markit.vision` loads Florence-2, which takes ~20 s. That is why the tests run
as separate processes, and why the server prints `Florence-2 ready.` before listening.

---

## Tests

```bash
python tests/run_all.py
```

No camera or network needed — the model loads from the local cache.

| Suite | Covers |
| --- | --- |
| `test_prompt.py` | Watch-list normalization across separator styles |
| `test_tracking.py` | Tracking accuracy, loss handling, stale-detection catch-up |
| `test_posture.py` | Body angles, hold-time debouncing, rising-edge logging |
| `test_rules_engine.py` | Every rule type's decision logic, using synthetic tracks |
| `test_integration.py` | Generated video → real trackers → real rules, end to end |
| `test_api.py` | HTTP endpoints and rule validation |
| `test_dashboard.py` | Every element the page's JS touches actually exists |
| `test_singleton.py` | Startup takes over the port from a leftover instance |
| `test_readme.py` | This README's factual claims, checked against the code |
| `test_docs_rules.py` | `docs/RULES.md` against the rule engine it documents |

`test_readme.py` re-checks this document against the source on every run — config
defaults, rule types, routes, posture thresholds, dependency pins, and file paths. If a
value here stops matching the code, the suite fails.

### What the tests prove, and what they don't

`test_integration.py` drives **real** `Track` objects through **real** optical flow on
generated video and into the **real** rule engine, so a break anywhere in
seed → update → speed → rule evaluation is caught. Tracking measures exactly 6.00
px/frame against a known 6 px/frame motion, with 0 px drift over 59 frames.

What no test covers, because it needs your hardware and your line:

- **Florence-2 actually detecting your products.** The tests bypass Florence and feed
  the tracker known boxes. Whether the model finds *your* items, from *your* watch-list
  wording, under *your* lighting, is the one thing only your camera can answer.
- **Whether your thresholds are right.** `speed_px`, `hold`, and zone bounds depend on
  camera distance, belt speed, and frame rate. Tune them live in the dashboard.
- **Counting accuracy at your line's rate.** See [Limits](#limits-worth-knowing) —
  verify against a known batch.

---

## Troubleshooting

**"No webcam found"** — change `CAM_INDEX` to `1` or `2` in `markit/config.py` and
restart.

**"Camera read failed. Is another app using it?"** — close Zoom, Teams, or anything else
holding the device.

**Florence-2 fails to load** — confirm `transformers==4.44.2` is what is actually
installed (`pip show transformers`). Newer versions break the model's Hub code.

**Port already in use** — an older instance is still running. On Windows:

```powershell
Get-NetTCPConnection -LocalPort 5000 -State Listen | Select-Object OwningProcess
Stop-Process -Id <that-id>
```

**Only one object detected when I asked for two** — check the separator. Use
`person. phone.`, not `person . phone`. MarkIt normalizes this for you, but the phrasing
still has to name two distinct things.

**Slow detection** — expected on CPU. `Florence-2-base` is already the lighter model;
set `SHOW_CAPTION = False` to spend every cycle on detection. Note the *boxes* still
track at full frame rate — only the labels refresh at cycle speed.

**Boxes drop off quickly** — the tracker needs visual texture. Plain, uniform, or
motion-blurred objects lose their feature points fast. Improve the lighting, or lower
`MIN_POINTS` to keep boxes alive longer at the cost of more drift.

**A rule never fires** — check the `label` matches a detection actually appearing in the
Detections panel (it is a substring match, so `box` matches `cardboard box`). Then check
`hold` is not longer than the condition lasts.
