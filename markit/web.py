"""Flask routes. The dashboard is a static template with two substitutions.

Plain str.replace() rather than a template engine, so the CSS and JS braces in
the page can never collide with template syntax.
"""

import os

from flask import Flask, Response, jsonify, request

from .config import FLORENCE_MODEL, HOST, PORT
from .pipeline import capture_loop, frames, lock, shared, worker
from .rules import scene
from .vision import normalize_prompt

app = Flask(__name__)

_TEMPLATE = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
_cache = {}

def _page():
    """Read the template once, then serve from memory."""
    if "html" not in _cache:
        with open(_TEMPLATE, encoding="utf-8") as fh:
            _cache["html"] = fh.read()
    return _cache["html"]

def _esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))

@app.route("/")
def index():
    with lock:
        p = shared["prompt"]
    # plain string substitution (no template engine) so CSS/JS braces can never collide
    html = _page().replace("__PROMPT__", _esc(p)).replace("__MODEL__", _esc(FLORENCE_MODEL.split("/")[-1]))
    return Response(html, mimetype="text/html")

@app.route("/video_feed")
def video_feed():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with lock:
        return jsonify(caption=shared["caption"], labels=shared["labels"],
                       fps=shared["fps"], florence_ms=shared["florence_ms"],
                       tracking=shared["tracking"], pending=shared["pending"],
                       active=shared["active"], alerts=shared["alerts"],
                       metrics=shared["metrics"], counts=shared["counts"],
                       caption_on=shared["caption_on"])

@app.route("/set_prompt", methods=["POST"])
def set_prompt():
    data = request.get_json(force=True, silent=True) or {}
    p = normalize_prompt((data.get("prompt") or "").strip())
    if p:
        with lock:
            shared["prompt"] = p
            shared["pending"] = True      # UI shows "reading…" until the next cycle lands
    return jsonify(ok=True, prompt=p)

@app.route("/rules", methods=["GET", "POST"])
def rules():
    """Read or replace the whole rule list. Kept simple: the dashboard sends
    the full set, so there is no partial-update path to get out of sync."""
    if request.method == "GET":
        return jsonify(rules=scene.get_rules(), counts=dict(scene.counts))
    data = request.get_json(force=True, silent=True) or {}
    incoming = data.get("rules")
    if not isinstance(incoming, list):
        return jsonify(ok=False, error="expected a 'rules' list"), 400
    clean, errors = [], []
    valid = {"count", "stalled", "crossing", "zone", "absent"}
    for i, r in enumerate(incoming):
        if not isinstance(r, dict):
            errors.append(f"rule {i}: not an object"); continue
        if r.get("when") not in valid:
            errors.append(f"rule {i}: 'when' must be one of {sorted(valid)}")
            continue
        r.setdefault("name", f"{r['when']} {i+1}")
        r.setdefault("level", "warn")
        clean.append(r)
    if errors:
        return jsonify(ok=False, error="; ".join(errors)), 400
    scene.set_rules(clean)
    return jsonify(ok=True, rules=scene.get_rules())

@app.route("/set_caption", methods=["POST"])
def set_caption():
    """Turn the scene caption on or off while running.

    The caption is a second Florence-2 pass, so switching it off is the single
    biggest cut to detection cycle time — which sets the floor on how quickly
    any rule can react.
    """
    data = request.get_json(force=True, silent=True) or {}
    want = bool(data.get("on"))
    with lock:
        shared["caption_on"] = want
        if not want:
            shared["caption"] = ""
    return jsonify(ok=True, caption_on=want)

@app.route("/reset_counts", methods=["POST"])
def reset_counts():
    scene.reset_counts()
    return jsonify(ok=True)


def serve():
    """Start the background threads, then run the server."""
    import threading

    # app.py clears a leftover instance before importing anything, because the
    # camera is grabbed at import time. This is the fallback for `python -m
    # markit` and for callers that import serve() directly; by now the camera
    # is already open, so it only rescues the port.
    from .singleton import free_port
    free_port(PORT)

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    print(f"Open http://{HOST}:{PORT}  (Ctrl+C to stop)")
    # debug=False so the model is not loaded twice; threaded so the video
    # stream and the status polling can run at the same time.
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
