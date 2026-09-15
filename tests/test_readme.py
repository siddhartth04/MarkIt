"""Check the README's factual claims against the actual code.

Docs drift silently: a default changes, a route is renamed, and the README keeps
asserting the old value. This suite fails when that happens.
"""
import _path  # noqa: F401  (adds the repo root to sys.path)

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
readme = open("README.md", encoding="utf-8").read()

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")

print("=== config table matches markit/config.py ===")
cfg = open("markit/config.py", encoding="utf-8").read()
for name, val in [("DETECTION_PROMPT", "person."),
                  ("FLORENCE_MODEL", "microsoft/Florence-2-base"),
                  ("CAM_INDEX", "0"), ("SHOW_CAPTION", "True"),
                  ("CAPTION_EVERY", "3"), ("PORT", "5000"),
                  ("MAX_POINTS", "24"), ("MIN_POINTS", "4"), ("MAX_SCALE", "4.0")]:
    in_code = re.search(rf"^{name}\s*=\s*(.+?)(?:\s*#|$)", cfg, re.M)
    actual = in_code.group(1).strip().strip('"') if in_code else None
    check(f"{name} = {val}", actual == val, f"code says {actual!r}")
    check(f"{name} documented", name in readme)

print("\n=== rule types match the server's validator ===")
web = open("markit/web.py", encoding="utf-8").read()
valid = set(re.search(r'valid = \{([^}]+)\}', web).group(1).replace('"', '').replace(' ', '').split(','))
documented = {t for t in ["stalled", "crossing", "zone", "count", "absent"] if f"`{t}`" in readme}
check(f"all 5 rule types documented: {sorted(valid)}", valid == documented,
      f"code={sorted(valid)} readme={sorted(documented)}")

print("\n=== routes match markit/web.py ===")
routes = set(re.findall(r'@app\.route\("([^"]+)"', web))
for r in routes:
    check(f"{r} documented", r in readme)

print("\n=== posture thresholds match markit/posture.py ===")
post = open("markit/posture.py", encoding="utf-8").read()
check("bending >= 45 deg", 'm["lean"] >= 45' in post and "45°" in readme)
check("leaning 25-45", '25 <= m["lean"] < 45' in post and "25–45°" in readme)
check("sitting hip < 120", 'm["hip"] < 120' in post and "120°" in readme)
check("crouching knee < 100", 'm["knee"] < 100' in post and "100°" in readme)

print("\n=== default rules match markit/rules.py ===")
rules = open("markit/rules.py", encoding="utf-8").read()
empty = re.search(r"DEFAULT_RULES\s*=\s*\[\s*\]", rules) is not None
check("ships with no preloaded rules", empty)
check("README says the rule list starts empty",
      "starts with no rules" in readme.lower() or "no rules" in readme.lower())

print("\n=== dependencies match requirements_web.txt ===")
reqs = open("requirements_web.txt", encoding="utf-8").read()
check("transformers pinned to 4.44.2", "transformers==4.44.2" in reqs
      and "4.44.2" in readme)
check("mediapipe 0.10.14", "mediapipe==0.10.14" in reqs and "0.10.14" in readme)
for lib in ["torch", "opencv-python", "flask"]:
    check(f"{lib} in requirements", lib in reqs)

print("\n=== project layout matches the filesystem ===")
# only the files listed under the markit/ block, not the tests/ block
block = readme.split("markit/" + chr(10))[1].split("tests/")[0]
for f in re.findall(r"^\s{2}(\w+\.py)\s{2,}", block, re.M):
    check(f"markit/{f} exists", os.path.exists(f"markit/{f}"))
check("templates/dashboard.html exists", os.path.exists("markit/templates/dashboard.html"))
check("tests/run_all.py exists", os.path.exists("tests/run_all.py"))

print("\n=== test suites listed actually exist ===")
listed = set(re.findall(r"`(test_\w+\.py)`", readme))
present = {f for f in os.listdir("tests") if f.startswith("test_")}
check(f"all {len(listed)} listed suites exist", listed <= present,
      f"missing: {listed - present}")
check("no suite left undocumented", present <= listed,
      f"undocumented: {present - listed}")

print("\n=== internal anchors resolve ===")
anchors = set(re.findall(r"\]\(#([\w-]+)\)", readme))
heads = {re.sub(r"[^\w\s-]", "", h).strip().lower().replace(" ", "-")
         for h in re.findall(r"^#{2,3} (.+)$", readme, re.M)}
check(f"all {len(anchors)} TOC links resolve", anchors <= heads,
      f"broken: {sorted(anchors - heads)}")

print("\n=== file links resolve ===")
for path in set(re.findall(r"\]\((markit/[\w./]+|tests/[\w./]+)\)", readme)):
    check(f"link {path}", os.path.exists(path))

print("\n=== every suite is registered in the runner ===")
# A suite that exists but is never run is worse than no suite: it looks like
# coverage while silently rotting.
runner = open(os.path.join("tests", "run_all.py"), encoding="utf-8").read()
on_disk = {f for f in os.listdir("tests")
           if f.startswith("test_") and f.endswith(".py")}
for f in sorted(on_disk):
    check(f"{f} is in run_all.py's ORDER", f'"{f}"' in runner)

print("\n=== no test-count claims to go stale ===")
# Counts of assertions or suites are noise to a reader and drift on every edit.
# Describe what is covered instead; the table above does that.
stale = re.findall(r"\d+\s+(?:assertions?|test cases?|checks)\b", readme)
check("README makes no numeric test claims", not stale, f"found: {stale}")

print("\n" + ("FAILURES: " + ", ".join(fails) if fails else "README FULLY VERIFIED"))
sys.exit(1 if fails else 0)
