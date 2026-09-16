"""Startup must take over the port from a leftover instance.

Two instances sharing one webcam makes the driver return duplicate frames —
video freezes and the boxes look like they are lagging, when in fact the
tracker is working fine on frames that are not changing. This is the guard
against that recurring.
"""
import _path  # noqa: F401  (adds the repo root to sys.path)

import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from markit.singleton import _pids_on_port, free_port  # noqa: E402

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")


def spare_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


print("=== an idle port reports nothing and needs no action ===")
p = spare_port()
check("no pids on an unused port", _pids_on_port(p) == [])
check("free_port succeeds on an unused port", free_port(p) is True)

print("\n=== a listener is detected, stopped, and the port comes back free ===")
port = spare_port()
# A bare listener process: enough to hold the port, with no model loading.
code = (f"import socket,time\n"
        f"s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"
        f"s.bind(('127.0.0.1',{port})); s.listen(5)\n"
        f"time.sleep(120)\n")
proc = subprocess.Popen([sys.executable, "-c", code],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):                      # wait for it to actually listen
        if _pids_on_port(port):
            break
        time.sleep(0.1)

    found = _pids_on_port(port)
    # Not necessarily proc.pid: on Windows the launcher can spawn an
    # intermediate, so the listening pid is a child. What matters is that
    # *something* is found, and that it is not us.
    check("listener is detected", len(found) >= 1, f"saw {found}")
    check("the listener is not this process", os.getpid() not in found)

    t0 = time.time()
    freed = free_port(port)
    took = time.time() - t0
    check("free_port reports success", freed is True)
    check("port is actually released", _pids_on_port(port) == [])
    check(f"it waits for release rather than returning instantly ({took:.1f}s)",
          took >= 1.0)

    print("\n=== the port is rebindable afterwards ===")
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        check("a fresh bind succeeds", True)
    except OSError as e:
        check("a fresh bind succeeds", False, str(e))
    finally:
        s.close()
finally:
    if proc.poll() is None:
        proc.kill()

print("\n=== it never targets this process ===")
own = spare_port()
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", own))
s.listen(1)
try:
    for _ in range(30):
        if _pids_on_port(own):
            break
        time.sleep(0.1)
    check("our own pid is excluded from the kill list",
          os.getpid() not in _pids_on_port(own))
finally:
    s.close()

print("\n=== startup clears the port before importing the camera ===")
app_src = open("app.py", encoding="utf-8").read()
i_free = app_src.find("free_port(PORT)")
i_web = app_src.find("from markit.web import serve")
check("app.py calls free_port", i_free != -1)
check("free_port runs before markit.web is imported",
      i_free != -1 and i_web != -1 and i_free < i_web,
      "the camera opens at import time, so a later call is too late")

print("\n" + ("FAILURES: " + ", ".join(fails) if fails else "ALL SINGLETON TESTS PASSED"))
sys.exit(1 if fails else 0)
