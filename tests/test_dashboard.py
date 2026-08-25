"""Verify the redesigned page keeps every contract the old one had."""
import _path  # noqa: F401  (adds the repo root to sys.path)

import importlib.util, re
import markit.web, markit.rules, markit.tracking, markit.posture, markit.pipeline, markit.vision
import types
m = types.SimpleNamespace(**{**vars(markit.rules), **vars(markit.tracking), **vars(markit.posture), **vars(markit.vision), **vars(markit.pipeline), **vars(markit.web)})
c = m.app.test_client()
html = c.get("/").data.decode("utf-8")

fails = []
def check(name, cond, extra=""):
    if not cond:
        fails.append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + extra) if extra and not cond else ''}")

print("=== every element the JS touches exists ===")
# ids referenced via getElementById in the script
ids_used = set(re.findall(r"getElementById\('([^']+)'\)", html))
ids_present = set(re.findall(r'id="([^"]+)"', html))
for i in sorted(ids_used):
    check(f"#{i} present", i in ids_present)
check("no orphan ids referenced", ids_used <= ids_present,
      f"missing: {ids_used - ids_present}")

print("\n=== handlers referenced by onclick are defined ===")
handlers = set(re.findall(r'onclick="(\w+)\(', html))
for h in sorted(handlers):
    check(f"{h}() defined", re.search(rf"function {h}\s*\(", html) is not None)

print("\n=== the button setPrompt() grabs still matches the markup ===")
# setPrompt does querySelector('.field .btn')
check("querySelector target exists",
      re.search(r'class="field"', html) and re.search(r'class="btn"', html))
check("selector is '.field .btn'", ".field .btn" in html)

print("\n=== server-side placeholders substituted ===")
check("no __PROMPT__ left", "__PROMPT__" not in html)
check("no __MODEL__ left", "__MODEL__" not in html)
check("model name rendered", "Florence-2-base" in html)

print("\n=== rule editor machinery intact ===")
for token in ["FIELDS", "RULES_STATE", "renderRules", "editRule", "delRule",
              "addRule", "saveRules", "loadRules", "getPath", "setPath"]:
    check(f"{token} present", token in html)
for kind in ["count", "stalled", "crossing", "zone", "absent"]:
    check(f"rule type '{kind}' offered", f'value="{kind}"' in html)

print("\n=== endpoints the page calls ===")
for ep in ["/status", "/set_prompt", "/rules", "/reset_counts", "/video_feed"]:
    check(f"{ep} referenced", ep in html)
    # and actually routable
    rules = {str(r) for r in m.app.url_map.iter_rules()}
    check(f"{ep} routed", any(ep == str(r) for r in m.app.url_map.iter_rules()))

print("\n=== status fields the page reads are all served ===")
s = c.get("/status").get_json()
for f in ["caption", "labels", "fps", "florence_ms", "pending", "tracking",
          "active", "alerts", "counts", "metrics"]:
    check(f"status.{f}", f in s)

print("\n=== html sanity ===")
check("single <html>", html.count("<html") == 1)
check("single <body>", html.count("<body") == 1)
check("style block closed", html.count("<style") == html.count("</style>") == 1)
check("script block closed", html.count("<script") == html.count("</script>") == 1)
check("no stray triple-quote", '"""' not in html)
opens = len(re.findall(r"<div\b", html)); closes = html.count("</div>")
check(f"div balance ({opens} open / {closes} close)", opens == closes)

print("\n=== XSS: labels and rule names are escaped ===")
check("esc() helper defined", "const esc=" in html)
check("chips use esc", "esc(l)" in html)
check("rule names use esc", "esc(r.name" in html)
check("alert fields use esc", "esc(a.rule)" in html and "esc(a.detail)" in html)

print("\n=== accessibility basics ===")
check("img has alt", re.search(r'<img[^>]+alt="[^"]+"', html) is not None)
check("bare inputs labelled", html.count("aria-label") >= 3)
check("lang set", 'lang="en"' in html)
check("viewport meta", "viewport" in html)

print("\n=== responsive + theme ===")
check("has media queries", html.count("@media") >= 3)
check("respects reduced motion", "prefers-reduced-motion" in html)
check("uses css custom properties", html.count("var(--") > 50)

print("\n" + ("FAILURES: " + ", ".join(fails) if fails else "ALL UI TESTS PASSED"))
raise SystemExit(1 if fails else 0)
