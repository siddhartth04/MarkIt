"""
MarkIt — local prompt-driven vision monitoring
----------------------------------------------
Florence-2 (open-vocabulary detection + scene caption) and MediaPipe (pose),
served as a local web dashboard with a live rule engine for conditions like
conveyor jams, product counts, and alignment.

Fully local: the models run on your CPU, and nothing is sent to any server at
runtime. First run downloads the Florence-2 weights (~0.5 GB) once.

Run:   python app.py          (or: python -m markit)
Open:  http://localhost:5000
Stop:  Ctrl+C

Code lives in the markit/ package:
    config.py    settings you are likely to change
    vision.py    Florence-2 + MediaPipe model loading and inference
    tracking.py  optical-flow box tracking between detection cycles
    rules.py     the generic condition engine (jam / count / zone / absent)
    posture.py   body-angle conditions from pose landmarks
    pipeline.py  camera capture, worker threads, shared state
    web.py       Flask routes
"""

from markit.web import serve

if __name__ == "__main__":
    serve()
