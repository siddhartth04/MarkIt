"""
MarkIt — local web dashboard
-----------------------------
Florence-2 (object detection + scene caption) + MediaPipe (skeleton), served as a local
web dashboard. The Python process owns the webcam and streams the annotated video to your
browser, with a control panel to change the watch-list live and read detections/caption.

Fully local: the model runs on your CPU, and nothing is sent to any server at runtime.
First run downloads the Florence-2 weights (~0.5 GB) once — needs internet that one time.

Run in VS Code:  python app.py
Then open:       http://localhost:5000     (open ONE browser tab)
Stop:            Ctrl+C in the terminal
"""

import threading, time
import cv2
import numpy as np
from PIL import Image
from flask import Flask, Response, request, jsonify

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DETECTION_PROMPT = "person. hard hat. safety vest. forklift."
FLORENCE_MODEL   = "microsoft/Florence-2-base"   # 'base' light / 'large' better+slower
CAM_INDEX        = 0
SHOW_CAPTION     = True
CAPTION_EVERY    = 3

# ----------------------------------------------------------------------------
# Load Florence-2 (CPU / no-flash-attn workaround)
# ----------------------------------------------------------------------------
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def _fixed_get_imports(filename):
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

print("Loading Florence-2 (first run downloads weights)...")
with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL, trust_remote_code=True, torch_dtype=torch.float32
    ).to("cpu").eval()
print("Florence-2 ready.")

def florence(image_pil, task_prompt, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image_pil, return_tensors="pt")
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=256, num_beams=1, do_sample=False,
        )
    text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        text, task=task_prompt, image_size=(image_pil.width, image_pil.height))

# ----------------------------------------------------------------------------
# MediaPipe Pose
# ----------------------------------------------------------------------------
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ----------------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------------
lock = threading.Lock()
latest_raw = None
shared = {"boxes": [], "labels": [], "caption": "", "florence_ms": 0, "fps": 0.0,
          "prompt": DETECTION_PROMPT, "jpeg": None}

def worker():
    cycle = 0
    while True:
        with lock:
            frame = None if latest_raw is None else latest_raw.copy()
            prompt = shared["prompt"]
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
            if SHOW_CAPTION and cycle % CAPTION_EVERY == 0:
                cap_task = "<DETAILED_CAPTION>"
                caption = florence(pil, cap_task).get(cap_task, "")
        except Exception as e:
            print("Florence worker error:", e)
        with lock:
            shared["boxes"] = boxes; shared["labels"] = labels
            shared["caption"] = caption
            shared["florence_ms"] = int((time.time() - t0) * 1000)
        cycle += 1

# ----------------------------------------------------------------------------
# Video pipeline: ONE capture thread owns the camera + MediaPipe and publishes
# the latest annotated JPEG. HTTP clients only read that buffer, so any number
# of browser tabs / refreshes are safe (no camera or pose contention).
# ----------------------------------------------------------------------------
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
    global latest_raw
    prev = time.time()
    fails = 0
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
        with lock:
            latest_raw = frame.copy()
            boxes = list(shared["boxes"]); labels = list(shared["labels"])

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
        for box, lab in zip(boxes, labels):
            try:
                x1, y1, x2, y2 = [int(v) for v in box]
            except (TypeError, ValueError):
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 177, 255), 2)
            cv2.putText(frame, str(lab), (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 177, 255), 2, cv2.LINE_AA)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6); prev = now
        ok, jpg = cv2.imencode(".jpg", frame)
        if ok:
            with lock:
                shared["jpeg"] = jpg.tobytes()
                shared["fps"] = round(fps, 1)
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

# ----------------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------------
app = Flask(__name__)

PAGE = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MarkIt — local vision monitor</title>
<style>
  :root{
    --graphite:#1a1d21; --panel:#23272e; --panel-2:#20242a; --line:#343b44;
    --amber:#ffb100; --amber-dim:#8a6a1f; --live:#4ade80;
    --ink:#e6e8eb; --muted:#8b929c;
    --sans:"Segoe UI",system-ui,-apple-system,sans-serif;
    --mono:"Consolas","JetBrains Mono",ui-monospace,monospace;
  }
  *{box-sizing:border-box} html,body{margin:0}
  body{background:var(--graphite);color:var(--ink);font-family:var(--sans);
       min-height:100vh;padding:20px;}
  .bar{display:flex;align-items:center;justify-content:space-between;
       padding-bottom:16px;border-bottom:1px solid var(--line);margin-bottom:20px;}
  .brand{font-weight:700;letter-spacing:.14em;font-size:20px}
  .brand b{color:var(--amber)} .brand span{color:var(--muted);font-weight:400}
  .sub{font-family:var(--mono);font-size:11px;color:var(--muted);
       text-transform:uppercase;letter-spacing:.18em;margin-top:3px}
  .live{display:flex;align-items:center;gap:8px;font-family:var(--mono);
        font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--live)}
  @media (prefers-reduced-motion:no-preference){
    .dot{animation:pulse 1.6s ease-in-out infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  }
  .grid{display:flex;gap:20px;align-items:flex-start}
  .feed{flex:2 1 640px;min-width:320px;background:#000;border:1px solid var(--line);
        border-radius:10px;overflow:hidden;position:relative}
  .feed img{display:block;width:100%;height:auto}
  .feed .tag{position:absolute;top:10px;left:10px;font-family:var(--mono);font-size:11px;
             letter-spacing:.14em;text-transform:uppercase;color:var(--amber);
             background:rgba(0,0,0,.55);padding:4px 8px;border-radius:4px}
  .rail{flex:1 1 320px;min-width:280px;display:flex;flex-direction:column;gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;
           color:var(--muted);margin:0 0 12px}
  .field{display:flex;gap:8px}
  .field input{flex:1;background:var(--panel-2);border:1px solid var(--line);color:var(--ink);
               font-family:var(--mono);font-size:13px;padding:10px;border-radius:6px}
  .field input:focus{outline:none;border-color:var(--amber)}
  .field button{background:var(--amber);color:#1a1d21;border:0;font-weight:700;
                font-family:var(--sans);padding:0 16px;border-radius:6px;cursor:pointer;
                letter-spacing:.02em}
  .field button:active{transform:translateY(1px)}
  .hint{color:var(--muted);font-size:12px;margin-top:9px;line-height:1.5}
  .scene{font-family:var(--mono);font-size:13px;line-height:1.6;color:var(--ink);min-height:20px}
  .scene.empty{color:var(--muted)}
  .chips{display:flex;flex-wrap:wrap;gap:8px;min-height:20px}
  .chip{font-family:var(--mono);font-size:12px;letter-spacing:.03em;color:var(--amber);
        border:1px solid var(--amber-dim);background:rgba(255,177,0,.08);
        padding:4px 10px;border-radius:4px}
  .chips .none{color:var(--muted);border:0;background:none;padding:0;font-family:var(--mono);font-size:12px}
  .tele{display:grid;grid-template-columns:1fr auto;gap:8px 12px;font-family:var(--mono);font-size:13px}
  .tele .k{color:var(--muted)} .tele .v{color:var(--ink);text-align:right}
  .tele .v.amber{color:var(--amber)}
  .foot{color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.08em;
        text-transform:uppercase;margin-top:20px;text-align:center}
  @media(max-width:820px){.grid{flex-direction:column}.feed,.rail{width:100%}}
</style></head><body>
  <div class="bar">
    <div><div class="brand"><b>MARK</b>//IT <span>vision monitor</span></div></div>
    <div class="live"><span class="dot"></span> live</div>
  </div>

  <div class="grid">
    <div class="feed">
      <span class="tag">camera 01</span>
      <img src="/video_feed" alt="live feed">
    </div>

    <div class="rail">
      <div class="card">
        <p class="eyebrow">Watch list</p>
        <div class="field">
          <input id="prompt" value="__PROMPT__" spellcheck="false">
          <button onclick="setPrompt()">Set</button>
        </div>
        <p class="hint">Type what to detect, separated by periods. Detection refreshes
          every few seconds on CPU — this is Florence-2 thinking, not a freeze.</p>
      </div>

      <div class="card">
        <p class="eyebrow">Scene</p>
        <div id="scene" class="scene empty">waiting for first read…</div>
      </div>

      <div class="card">
        <p class="eyebrow">Detections</p>
        <div id="chips" class="chips"><span class="none">none yet</span></div>
      </div>

      <div class="card">
        <p class="eyebrow">Telemetry</p>
        <div class="tele">
          <span class="k">skeleton</span><span id="fps" class="v amber">– fps</span>
          <span class="k">florence cycle</span><span id="ms" class="v">– ms</span>
          <span class="k">model</span><span class="v">__MODEL__</span>
        </div>
      </div>
    </div>
  </div>
  <div class="foot">running locally · nothing leaves this machine</div>

<script>
async function setPrompt(){
  const v=document.getElementById('prompt').value;
  await fetch('/set_prompt',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:v})});
}
async function poll(){
  try{
    const r=await fetch('/status'); const s=await r.json();
    const scene=document.getElementById('scene');
    if(s.caption){scene.textContent=s.caption;scene.classList.remove('empty');}
    const chips=document.getElementById('chips');
    if(s.labels && s.labels.length){
      chips.innerHTML=s.labels.map(l=>`<span class="chip">${l}</span>`).join('');
    }else{chips.innerHTML='<span class="none">none yet</span>';}
    document.getElementById('fps').textContent=(s.fps??'–')+' fps';
    document.getElementById('ms').textContent=(s.florence_ms??'–')+' ms';
  }catch(e){}
}
setInterval(poll,1000); poll();
</script>
</body></html>
"""

def _esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))

@app.route("/")
def index():
    with lock:
        p = shared["prompt"]
    # plain string substitution (no template engine) so CSS/JS braces can never collide
    html = PAGE.replace("__PROMPT__", _esc(p)).replace("__MODEL__", _esc(FLORENCE_MODEL.split("/")[-1]))
    return Response(html, mimetype="text/html")

@app.route("/video_feed")
def video_feed():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with lock:
        return jsonify(caption=shared["caption"], labels=shared["labels"],
                       fps=shared["fps"], florence_ms=shared["florence_ms"])

@app.route("/set_prompt", methods=["POST"])
def set_prompt():
    data = request.get_json(force=True, silent=True) or {}
    p = (data.get("prompt") or "").strip().lower()
    if p:
        with lock:
            shared["prompt"] = p
    return jsonify(ok=True, prompt=p)

if __name__ == "__main__":
    # start the two background threads once (camera+pose, and Florence)
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    print("Open http://localhost:5000  (Ctrl+C to stop)")
    # debug=False so the model isn't loaded twice; threaded so stream + status run together
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
