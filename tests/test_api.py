import _path  # noqa: F401  (adds the repo root to sys.path)

import importlib.util, json
import markit.web, markit.rules, markit.tracking, markit.posture, markit.pipeline, markit.vision
import types
m = types.SimpleNamespace(**{**vars(markit.rules), **vars(markit.tracking), **vars(markit.posture), **vars(markit.vision), **vars(markit.pipeline), **vars(markit.web)})
c = m.app.test_client()

fails = []
def check(name, cond):
    if not cond:
        fails.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")

print("=== GET /rules ===")
j = c.get("/rules").get_json()
print("  ", json.dumps(j)[:160])
check("returns a rules list", isinstance(j.get("rules"), list))

print("\n=== POST /rules (valid) ===")
payload = {"rules": [
    {"name": "belt jam", "when": "stalled", "label": "box", "speed_px": 0.6,
     "hold": 5, "level": "warn"},
    {"name": "output", "when": "crossing", "label": "box",
     "line": {"axis": "x", "at": 0.5}, "direction": 1},
    {"name": "lane", "when": "zone", "label": "box",
     "zone": {"x1": .2, "y1": .2, "x2": .8, "y2": .8}, "inside": True, "hold": 2},
]}
r = c.post("/rules", json=payload)
j = r.get_json()
check("accepted", r.status_code == 200 and j["ok"])
check("all three stored", len(j["rules"]) == 3)
check("engine sees them", len(m.scene.get_rules()) == 3)

print("\n=== POST /rules (invalid 'when') ===")
r = c.post("/rules", json={"rules": [{"name": "x", "when": "nonsense"}]})
print("   ->", r.status_code, r.get_json())
check("rejected with 400", r.status_code == 400)
check("did not clobber existing rules", len(m.scene.get_rules()) == 3)

print("\n=== POST /rules (not a list) ===")
r = c.post("/rules", json={"rules": "nope"})
check("rejected with 400", r.status_code == 400)

print("\n=== defaults are filled in ===")
r = c.post("/rules", json={"rules": [{"when": "absent", "label": "box"}]})
j = r.get_json()
print("  ", j["rules"])
check("name auto-generated", bool(j["rules"][0].get("name")))
check("level defaulted", j["rules"][0].get("level") == "warn")

print("\n=== /status carries the new fields ===")
s = c.get("/status").get_json()
print("  keys:", sorted(s.keys()))
for k in ("active", "alerts", "counts", "metrics", "tracking", "labels"):
    check(f"status has '{k}'", k in s)

print("\n=== /set_caption toggles live ===")
s0 = c.get("/status").get_json()
check("status exposes caption_on", "caption_on" in s0)
r = c.post("/set_caption", json={"on": False})
check("turning it off works", r.get_json() == {"ok": True, "caption_on": False})
check("status reflects off", c.get("/status").get_json()["caption_on"] is False)
check("caption text is cleared", c.get("/status").get_json()["caption"] == "")
r = c.post("/set_caption", json={"on": True})
check("turning it on works", r.get_json()["caption_on"] is True)
check("status reflects on", c.get("/status").get_json()["caption_on"] is True)
check("a missing body defaults to off",
      c.post("/set_caption", json={}).get_json()["caption_on"] is False)
c.post("/set_caption", json={"on": True})

print("\n=== /reset_counts ===")
m.scene.counts["output"] = 42
r = c.post("/reset_counts")
check("cleared", r.get_json()["ok"] and not m.scene.counts)

print("\n=== dashboard renders the rule editor ===")
html = c.get("/").data
for token in (b"id=\"rules\"", b"addRule", b"FIELDS", b"resetCounts", b"id=\"counts\""):
    check(f"page contains {token.decode()}", token in html)
check("no unrendered placeholder", b"__PROMPT__" not in html and b"__MODEL__" not in html)

print("\n" + ("FAILURES: " + ", ".join(fails) if fails else "ALL API TESTS PASSED"))
raise SystemExit(1 if fails else 0)
