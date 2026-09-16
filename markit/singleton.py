"""Make sure only one MarkIt owns the camera and the port.

Two instances is the failure mode that looks like a bug in everything else: the
second process cannot bind the port, and both fight over the webcam, so the
driver hands back the same frame repeatedly. Video freezes at ~1 fps and the
boxes appear to lag, while the tracker is working perfectly on frames that are
not changing.

So on startup we find whatever is already listening on our port and stop it.
Deliberately narrow: only a process holding *this* port is touched, never
every python.exe, because that would kill unrelated work.
"""

import os
import subprocess
import sys
import time


def _pids_on_port(port):
    """PIDs listening on `port`. Empty list if nothing is, or if we can't tell."""
    pids = set()
    try:
        if sys.platform == "win32":
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                 capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                parts = line.split()
                # proto  local            foreign          state       pid
                if len(parts) >= 5 and parts[3].upper() == "LISTENING":
                    local = parts[1]
                    if local.rsplit(":", 1)[-1] == str(port):
                        pids.add(int(parts[4]))
        else:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True, timeout=10).stdout
            pids.update(int(p) for p in out.split() if p.strip().isdigit())
    except (OSError, ValueError, subprocess.SubprocessError):
        return []          # can't inspect: let the bind fail loudly instead
    pids.discard(os.getpid())
    return sorted(pids)


def _describe(pid):
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            if out and "," in out:
                return out.split(",")[0].strip('"')
        else:
            return subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _kill(pid):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=10)
        else:
            os.kill(pid, 15)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def free_port(port, wait=6.0):
    """Stop anything listening on `port`. Returns True if the port ends up free.

    Waits for the OS to actually release the socket — a killed process does not
    free its port instantly, and binding too early fails with the same confusing
    "address in use" the whole exercise is meant to avoid.
    """
    stale = _pids_on_port(port)
    if not stale:
        return True

    for pid in stale:
        name = _describe(pid)
        print(f"Port {port} is held by {name} (pid {pid}) — stopping it.")
        _kill(pid)

    deadline = time.time() + wait
    while time.time() < deadline:
        if not _pids_on_port(port):
            # The listener is gone, but the camera handle can lag slightly
            # behind the socket. A short settle avoids grabbing a half-released
            # device and getting duplicate frames.
            time.sleep(1.0)
            print("Previous instance stopped.")
            return True
        time.sleep(0.25)

    print(f"Warning: something is still listening on port {port}.")
    return False
