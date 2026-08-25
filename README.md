# MarkIt

A local webcam vision monitor. Florence-2 (open-vocabulary object detection + scene
captioning) and MediaPipe (pose skeleton) run on your machine and stream an annotated
video feed to a browser dashboard.

Nothing leaves your computer at runtime. The only network access is the one-time
download of the Florence-2 weights (~0.5 GB) on first run.

![dashboard: live feed on the left, watch-list / scene / detections / telemetry on the right]

## Requirements

- Python 3.10–3.12 (`mediapipe` 0.10.14 does not support 3.13+)
- A webcam
- ~2 GB free disk for model weights

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements_web.txt
```

`transformers` is pinned to `4.44.2` on purpose — Florence-2's remote Hub code
breaks on newer versions because the KV-cache API changed. If you unpin it, verify
Florence-2 still loads before relying on it.

## Run

```bash
python app.py          # or: python -m markit
```

Then open <http://localhost:5000>. Stop with `Ctrl+C`.

If the port is already in use, an older instance is still running — stop that
first, or the new one will fail to bind and you will be looking at a stale page.

The first launch is slow: it downloads and loads the model before the server starts.
Wait for `Florence-2 ready.` in the terminal.

## Using it

- **Watch list** — one object per phrase, separated by periods
  (`person. phone.`) then press **Set**. It applies to the next detection cycle;
  no restart needed.

  Commas and "and" work too — the input is normalized before it reaches the model.
  This matters: Florence-2 splits on periods, so a stray space (`person . phone`)
  produces the phrases `"person "` and `" phone"`, which measurably degrades
  grounding — in testing it returned a spurious third box for a two-object scene.

- **Multiple objects** — every match gets its own box and its own tracker. Each
  object type gets a distinct color, repeats of one type are numbered in the video
  (`person 1`, `person 2`), and the Detections panel collapses them to `person x2`.
- **Rules** — conditions evaluated against the tracked objects every frame. Add,
  rename, retune, and delete them live; changes take effect immediately. See
  [Rules](#rules) below.

- **Scene** — a sentence describing the whole frame, refreshed every 3rd cycle.
- **Detections** — labels Florence-2 grounded in the current frame.
- **Telemetry** — skeleton FPS (the smooth video loop) and Florence cycle time
  (the slow detection loop).

Detection takes seconds per cycle on CPU, but the boxes still follow moving objects
in real time: Florence-2 decides *what* is in frame, and an optical-flow tracker keeps
each box on *where* that thing is now, updating every frame. The **tracking** telemetry
row shows how many boxes are currently being followed.

A box is dropped when the tracker loses it — the object leaves frame, is occluded, or
has too little texture to lock onto. It comes back on the next detection cycle.

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
  templates/
    dashboard.html      the dashboard (plain HTML/CSS/JS, no build step)
tests/
  run_all.py            run every suite:  python tests/run_all.py
```

Importing `markit.vision` loads Florence-2, which takes ~20 s. That is why the
tests run as separate processes and why the server prints `Florence-2 ready.`
before it starts listening.

## Rules

A rule is data, not code: one mechanism covers jam detection, product counting, and
alignment. Every rule has a `name`, a `when` type, and a `label` naming which detected
objects it applies to (`any` or blank matches everything).

| `when` | Fires when | Key fields |
| --- | --- | --- |
| `stalled` | Objects are present but not moving — a conveyor jam | `speed_px`, `hold` |
| `crossing` | Tallies objects crossing a line (a counter, not an alarm) | `line.axis`, `line.at`, `direction` |
| `zone` | An object drifts outside (or into) a rectangle — alignment | `zone.x1/y1/x2/y2`, `inside`, `hold` |
| `count` | The number in view falls outside a range | `min`, `max`, `hold` |
| `absent` | Nothing of that label has been seen — a starved line | `hold` |

Zone and line coordinates are **fractions of the frame** (0–1), so a rule survives a
resolution change or a different camera.

`hold` is how many seconds the condition must persist before it alerts. This is what
separates a real event from a one-frame glitch — a lost feature point or a single bad
detection. Counting rules have no `hold`: they fire on the transition itself.

Alerts log on the **rising edge** only, so a jam that lasts ten minutes produces one
entry rather than thousands. The condition clearing and re-occurring logs a second one.

### Worked examples

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
```

Zones and count lines are drawn on the video feed, turning red while their rule is
firing, so you can see what a rule is actually watching.

### A real constraint on counting

Florence-2 takes **seconds per detection cycle on CPU**. Counting works because the
optical-flow tracker follows each object between cycles at full framerate — Florence
only supplies the labels. In practice this means:

- **Items must be visible for at least one detection cycle** to be counted at all. Fast
  items on a quick belt may pass between cycles and be missed entirely.
- **Verify the tally against a known batch** before trusting it for production numbers.
- For high-rate counting, a purpose-built detector (or a GPU) is the right answer; this
  is a monitoring aid, not a certified counter.

The `stalled`, `zone`, `absent`, and `count` rules are far less sensitive to this,
because they describe slow conditions rather than instantaneous events.

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `GET /rules` | Current rules and counter tallies |
| `POST /rules` | Replace the whole rule list — `{"rules": [...]}`; invalid rules are rejected 400 without disturbing the running set |
| `POST /reset_counts` | Zero the crossing tallies (start of shift) |
| `GET /status` | Detections, active rules, alert log, counts, telemetry |
| `POST /set_prompt` | Change the watch list |

## How it works

Three concurrent pieces share state behind a single lock:

| Piece | Role |
| --- | --- |
| `capture_loop` | Sole owner of the camera and MediaPipe. Advances the trackers, draws skeleton + boxes, publishes one annotated JPEG. |
| `worker` | Runs Florence-2 on the newest raw frame in a loop, independent of the video rate. |
| `Track` | One object's box plus the feature points it is followed by (Lucas-Kanade optical flow). Carries a stable id, age, and smoothed speed. |
| `SceneEngine` | Evaluates the rules against live tracks each frame; holds debounce state and the alert log. (`markit/rules.py`) |
| Flask | `/video_feed` (MJPEG), `/status` (JSON, polled once a second), `/set_prompt` (POST). |

The worker hands off each detection **together with the grayscale frame it ran on**.
The capture loop seeds trackers against that original frame and then fast-forwards them
over every frame captured since, so a box lands on the object's current position rather
than where it was when detection started. Measured on synthetic motion: an object that
travels 270 px during one detection cycle lands with the box 0 px off, versus 270 px off
without the tracker.

HTTP handlers never touch the camera — they only read the last published JPEG — so
extra browser tabs and refreshes cannot cause camera contention.

The dashboard is a single HTML string in `app.py`, substituted with plain
`str.replace()` rather than a template engine so CSS and JS braces can never collide
with template syntax.

## Configuration

Constants at the top of [`app.py`](app.py):

| Name | Default | Meaning |
| --- | --- | --- |
| `DETECTION_PROMPT` | `person. hard hat. safety vest. forklift.` | Initial watch list |
| `FLORENCE_MODEL` | `microsoft/Florence-2-base` | Use `-large` for better, slower results |
| `CAM_INDEX` | `0` | Try `1` or `2` if the wrong camera opens |
| `SHOW_CAPTION` | `True` | Turn off to spend every cycle on detection |
| `CAPTION_EVERY` | `3` | Caption once per N detection cycles |

Tracker tuning lives just above the `Track` class: `MAX_POINTS` (features per box),
`MIN_POINTS` (below this a box is dropped), and `LK_PARAMS` (`winSize` — raise it for
fast motion, lower it for speed).

## Tests

```bash
python tests/run_all.py
```

144 assertions, no camera or network needed (the model loads from the local
cache). Suites:

| Suite | Covers |
| --- | --- |
| `test_prompt.py` | Watch-list normalization across separator styles |
| `test_tracking.py` | Box tracking accuracy, loss handling, stale-detection catch-up |
| `test_posture.py` | Body angles, hold-time debouncing, rising-edge logging |
| `test_rules_engine.py` | Every rule type, plus counting correctness |
| `test_api.py` | HTTP endpoints and rule validation |
| `test_dashboard.py` | Every element the page's JS touches actually exists |

## Troubleshooting

**"No webcam found"** — change `CAM_INDEX` to `1` or `2` and restart.

**"Camera read failed. Is another app using it?"** — close Zoom, Teams, or anything
else holding the device.

**Florence-2 fails to load** — confirm `transformers==4.44.2` is what is actually
installed (`pip show transformers`). Newer versions break the model's Hub code.

**Slow detection** — expected on CPU. `Florence-2-base` is already the lighter model;
set `SHOW_CAPTION = False` to spend every cycle on detection instead. Note the *boxes*
still track at full framerate; only the labels refresh at cycle speed.

**Only one object detected when I asked for two** — check the separator. Use
`person. phone.`, not `person . phone`. The app normalizes this for you now, but the
phrasing still has to name two distinct things.

**Boxes drop off quickly** — the tracker needs visual texture to hold on. Plain,
uniform, or motion-blurred objects lose their feature points fast. Improve the lighting,
or lower `MIN_POINTS` to keep boxes alive longer at the cost of more drift.
