# Rules — complete reference

How each rule type decides, what every field does, and how to set one up for a real
job. For the short version, see the [README](../README.md#what-it-can-do).

- [The shared machinery](#the-shared-machinery) — read this first
- [`count`](#count--how-many-are-in-view)
- [`stalled`](#stalled--present-but-not-advancing)
- [`absent`](#absent--nothing-there-at-all)
- [`zone`](#zone--inside-or-outside-an-area)
- [`crossing`](#crossing--tally-items-passing-a-line)
- [Choosing a rule](#choosing-a-rule)
- [Setting one up](#setting-one-up-a-worked-session)
- [Field reference](#field-reference)
- [Why a rule is not firing](#why-a-rule-is-not-firing)

---

## The shared machinery

MarkIt starts with **no rules**. That is deliberate: a rule targeting a label your watch
list does not detect fires constantly, and an alert panel that is always red teaches you
to ignore it. Get detection working first, then add rules against the labels you can
actually see in the Detections panel.

Every rule is a JSON object evaluated **once per video frame** against the list of
currently-tracked objects. Four mechanisms apply to all of them.

### 1. Label matching

```python
mine = [t for t in tracks if _match(t, label)]
```

The rule's `label` is matched as a **case-insensitive substring** of what the detector
actually returned:

| Detected label | Rule label | Matches |
| --- | --- | --- |
| `cardboard box` | `box` | yes |
| `Box` | `box` | yes |
| `box` | `BOX` | yes |
| `person` | `box` | no |
| anything | `` (blank) | yes |
| anything | `any` or `*` | yes |

Substring matching is deliberate: Florence-2 often returns a longer phrase than you
typed. But it cuts both ways — a rule labelled `box` also matches `box cutter`. Keep
labels specific enough to be unambiguous in your scene.

### 2. The hold timer

`hold` is how many **seconds the condition must be continuously true** before the rule
fires.

```python
start = self.since.setdefault(name, now)
return now - start >= float(rule.get("hold", 0) or 0)
```

The important word is *continuously*. The timer records when the condition first became
true, and **a single frame where it is false resets it to zero** — elapsed time does not
accumulate across interruptions:

```
0.9s violating  -> quiet (hold is 1.0s)
1 frame OK      -> timer reset
0.9s violating  -> still quiet   (not 1.8s cumulative)
0.2s more       -> FIRES
```

This is what separates an event from a twitch: one lost feature point, one bad detection
frame, or an object briefly clipped at the frame edge will not raise an alarm.

`hold: 0` fires on the very first frame the condition holds.

### 3. Rising-edge logging

A rule that fires is added to a `firing` set and **will not log again until it clears**.
A jam lasting ten minutes produces one log entry, not thousands.

If the condition clears and later recurs, that is a genuinely separate event and logs
again. The `active` list (what turns the status pill red and highlights the rule) tracks
the live state every frame; the log records transitions.

### 4. Crash isolation

Each rule is evaluated inside `try/except`. A malformed rule prints an error to the
terminal and is skipped — the video, and every other rule, keeps running. You cannot
take the app down with a bad rule.

### Coordinates

All zone and line positions are **fractions of the frame, 0–1**, never pixels:

```python
x1 = _clamp01(zone.get("x1")) * w
```

`0.5` is the middle whether the camera is 640×480 or 1920×1080, so a rule survives a
resolution change or a different camera. Values outside 0–1 are clamped rather than
raising. Zone edges are **inclusive**, and reversed coordinates (`x1 > x2`) are
normalised — the rectangle works either way round.

---

## `count` — how many are in view

**Fires when the number of matching objects falls outside an allowed range.**

```python
n = len(mine)
lo, hi = rule.get("min"), rule.get("max")
bad = ((lo is not None and n < int(lo)) or
       (hi is not None and n > int(hi)))
```

`min` and `max` are **independent and both optional**. Set only the bound you care
about. If neither is set, the rule can never fire.

| Intent | `min` | `max` | Fires at |
| --- | --- | --- | --- |
| Too many accumulating | — | `4` | 5 or more |
| Something is missing | `1` | — | 0 |
| Exactly two expected | `2` | `2` | any count ≠ 2 |
| Acceptable band | `2` | `5` | below 2, or above 5 |

```jsonc
// no more than four cartons staged at the station
{"name": "staging full", "when": "count", "label": "carton",
 "max": 4, "hold": 5, "level": "warn"}
```

### Restricting to an area

Add a `zone` and only objects whose centre is inside it are counted. This is how one
camera covers several stations:

```jsonc
{"name": "inbound queue", "when": "count", "label": "box",
 "zone": {"x1": 0.0, "y1": 0.4, "x2": 0.35, "y2": 1.0},
 "max": 6, "hold": 5}
```

### Choosing `hold`

The count fluctuates naturally — detection runs every few seconds, and an object briefly
leaving frame or losing tracker texture drops it for a moment. Watch the Detections
panel (it shows `box x3`) for a minute before deciding. If the number is jumpy while the
scene is steady, raise `hold` rather than loosening `min`/`max`.

> A `min` rule fires on an empty scene, overlapping with `absent`. Use `min: 1` for
> "there should be product here"; use `absent` when the concern is specifically that the
> line has gone dead — it reads more clearly in the log.

---

## `stalled` — present but not advancing

**Fires when matching objects are there, but none of them are moving.**

```python
slow = [t for t in mine if t.speed < thresh]
if mine and len(slow) == len(mine) and self._held(rule, now):
```

Two guards define it:

- **`mine` must be non-empty.** An empty belt is not a jam — that is `absent`.
- **Every** matching object must be slow. If one item is still moving, the line is
  running and the rule stays quiet.

`speed_px` is in **pixels per frame**, measured as an exponentially-smoothed average of
each object's movement. The smoothing means one noisy frame cannot read as motion, and
a jam is "slow for a while" rather than "slow right now."

```jsonc
{"name": "belt jam", "when": "stalled", "label": "box",
 "speed_px": 0.6, "hold": 5, "level": "warn"}
```

### Choosing `speed_px`

It depends entirely on camera distance and belt speed — there is no universal value. A
distant camera sees a fast belt as slow motion in pixels.

To calibrate: run the line normally and watch the **tracking** telemetry. Pick a
threshold well below the speed you observe during normal operation, so ordinary
variation never trips it. Default is `0.6`.

Use a generous `hold` (5 s or more). Products legitimately pause at stations.

---

## `absent` — nothing there at all

**Fires when nothing matching has been seen for `hold` seconds.**

```python
if not mine and self._held(rule, now):
```

This is the complement to `stalled`. Together they cover the whole space:

| Situation | Objects present? | Moving? | Fires |
| --- | --- | --- | --- |
| Running normally | yes | yes | *nothing* |
| Jammed | yes | no | `stalled` |
| Empty | no | — | `absent` |

Without `absent`, a starved line is silent: there is nothing present to be "stopped," so
`stalled` deliberately cannot fire.

```jsonc
{"name": "line starved", "when": "absent", "label": "box", "hold": 30}
```

### What it catches

- Upstream stopped feeding product
- Nobody at a station that should be staffed
- Camera moved, was covered, or lost focus
- **Your watch list is wrong** — if this fires immediately on a running line, the label
  does not match what the detector returns

### Choosing `hold`

**Set it high.** This is the rule most prone to false alarms — normal gaps between
items, a slow detection cycle, or a brief tracker loss all produce momentary emptiness.
If your line has a 30-second gap between batches, `hold` must exceed that.

---

## `zone` — inside or outside an area

**Fires when a matching object is on the wrong side of a rectangle.**

```python
want_in = bool(rule.get("inside", True))
off = [t for t in mine if _in_zone(t, zone, w, h) != want_in]
```

The `inside` flag inverts the entire meaning:

| `inside` | Objects should be | Fires when one | Use for |
| --- | --- | --- | --- |
| `true` (default) | inside the box | **leaves** | alignment, staying on a lane |
| `false` | outside the box | **enters** | exclusion zones, safety |

```jsonc
// alignment: product drifted off the expected lane
{"name": "off lane", "when": "zone", "label": "box",
 "zone": {"x1": 0.2, "y1": 0.35, "x2": 0.8, "y2": 0.65},
 "inside": true, "hold": 2}

// safety: a person entered the press area
{"name": "keep out", "when": "zone", "label": "person",
 "zone": {"x1": 0.6, "y1": 0.0, "x2": 1.0, "y2": 0.5},
 "inside": false, "hold": 1, "level": "warn"}
```

Membership is tested on the **centre point** of the box, not any overlap. An object
half-in and half-out counts as wherever its centre is.

Zones are drawn on the live video and turn red while their rule is firing, so you can
see exactly what is being watched and adjust the bounds by eye.

---

## `crossing` — tally items passing a line

**Counts objects crossing a line. This one does not alert** — it never appears in the
active list, never turns the status pill red, and ignores `hold`. It produces a running
tally shown as a large tile on the dashboard.

```python
side = _side_of_line(t, line, w, h)
prev = self.sides.get(key)
if prev is not None and prev != side and side == want:
    if name not in t.counted:
        t.counted.add(name)
        self.counts[name] = self.counts.get(name, 0) + 1
```

Three guards make the tally trustworthy:

1. **`prev != side`** — counts the *transition*, not presence. An item resting on the
   line does not accumulate.
2. **`side == want`** — one direction only. An item drifting back the other way is not
   counted and does not decrement.
3. **`name not in t.counted`** — once per tracked object, permanently. An item that
   crosses, returns, and crosses again still counts **once**.

### Line geometry

| Field | Meaning |
| --- | --- |
| `line.axis: "x"` | a **vertical** line, crossed by horizontal (left/right) motion |
| `line.axis: "y"` | a **horizontal** line, crossed by vertical (up/down) motion |
| `line.at` | position as a fraction, `0.0`–`1.0` |
| `direction: 1` | counts motion toward increasing x (rightward) or y (downward) |
| `direction: -1` | counts the opposite direction |

```jsonc
// items moving left-to-right past mid-frame
{"name": "output", "when": "crossing", "label": "box",
 "line": {"axis": "x", "at": 0.5}, "direction": 1}
```

### Where to put the line

**An object that is already past the line when first detected is never counted** — there
is no previous side to compare against. This is a real constraint on placement.

Put the line where objects have been visible and tracked for a while before they reach
it. A line at `at: 0.9`, right at the frame edge, will undercount: items may not be
detected until they are already past it. Something nearer the middle, with clear
approach room, counts reliably.

Counts reset with **Reset counts** in the dashboard, or `POST /reset_counts`. Editing
other rules does not reset a tally.

---

## Choosing a rule

| You want to know | Rule |
| --- | --- |
| Did the belt jam? | `stalled` |
| How many did we produce? | `crossing` |
| Did product stop arriving? | `absent` |
| Is it lined up correctly? | `zone` with `inside: true` |
| Did someone enter a danger area? | `zone` with `inside: false` |
| Are too many piling up? | `count` with `max` |
| Is the station empty? | `count` with `min`, or `absent` |

---

## Setting one up: a worked session

Counting boxes on a belt, from scratch.

**1. Get the detection working first.** Set the watch list to `box.` and press **Set**.
Wait for a detection cycle, then read the **Detections** panel. If it shows nothing, no
rule will ever fire — try different wording (`cardboard box.`, `carton.`) until the
panel shows your product consistently.

**2. Note the exact label.** The panel shows what the detector actually returned. If it
says `cardboard box`, your rule label can be `box` (substring match) but not `carton`.

**3. Add the rule.** Pick `crossing` from the dropdown, press **+ Add rule**. It appears
with defaults and a yellow line is drawn at mid-frame on the video.

**4. Set the fields.** `label` to `box`, `line.at` where items have clear approach room,
`direction` to match travel (`1` for left-to-right).

**5. Watch it run.** Each count logs as `#1 box`, `#2 box`. Compare the tile against a
batch you count by hand.

**6. Add a jam rule.** Pick `stalled`, label `box`, `hold` `5`. Watch the tracking
telemetry during normal running to pick `speed_px` — well below observed normal speed.

**7. Add a starvation rule.** Pick `absent`, label `box`, `hold` longer than your
longest legitimate gap between batches.

Start every `hold` too high and lower it if you miss real events. Missing a few while
calibrating is recoverable; an alert panel that cries wolf all shift gets ignored
permanently.

---

## Field reference

| Field | Applies to | Type | Meaning |
| --- | --- | --- | --- |
| `name` | all | string | Shown in alerts and on the video. Also the key for a counter. Must be unique. |
| `when` | all | enum | `count` · `stalled` · `crossing` · `zone` · `absent` |
| `label` | all | string | Which detections it applies to. Case-insensitive substring. Blank / `any` / `*` matches everything. |
| `hold` | all but `crossing` | seconds | How long the condition must persist continuously before firing |
| `level` | all | `warn` \| `info` | `warn` shows red |
| `min` | `count` | integer | Fires below this. Optional. |
| `max` | `count` | integer | Fires above this. Optional. |
| `speed_px` | `stalled` | px/frame | Below this counts as stopped |
| `zone` | `zone`, `count` | object | `{x1, y1, x2, y2}` as fractions, 0–1 |
| `inside` | `zone` | boolean | `true` = alert when an object leaves; `false` = alert when one enters |
| `line` | `crossing` | object | `{axis: "x" \| "y", at: 0.0–1.0}` |
| `direction` | `crossing` | `1` \| `-1` | Which way across the line counts |

Unknown `when` values are rejected with HTTP 400 and the running rule set is left
untouched. Missing `name` and `level` are filled in with defaults.

---

## Why a rule is not firing

Work down this list in order.

**1. Is the object detected at all?** Check the Detections panel. If your product is not
listed there, the problem is the watch list, not the rule. Try different wording.

**2. Does the label match?** It is a substring match against the detected label. `box`
matches `cardboard box`; `carton` matches neither. Copy the wording from the Detections
panel.

**3. Is `hold` longer than the condition lasts?** A 5-second hold never fires on a
3-second event. Remember the timer resets completely on any frame where the condition is
false — flickering detection can keep it permanently reset.

**4. For `count`: are `min`/`max` both empty?** Then nothing can fire. Set at least one.

**5. For `stalled`: is anything detected?** It requires at least one matching object.
An empty belt is `absent`, not `stalled`.

**6. For `stalled`: is `speed_px` too low?** If it is below the residual jitter of a
stationary object, the object never reads as "slow." Raise it.

**7. For `zone`: is the centre actually outside?** Membership uses the box's centre
point, not overlap. Watch the drawn rectangle on the video.

**8. For `crossing`: was the object already past the line when detected?** It needs to
be seen on both sides. Move the line away from the frame edge.

**9. Check the terminal.** A malformed rule prints `rule error in <name>: ...` and is
skipped silently in the UI.
