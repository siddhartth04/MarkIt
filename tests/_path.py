"""Put the repo root on sys.path so `import markit` works when a test is run
directly (python tests/test_x.py), not just via the runner."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
