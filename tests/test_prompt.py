"""Watch-list normalization: any separator style must reach Florence-2 as 'a. b.'"""
import _path  # noqa: F401  (adds the repo root to sys.path)

import sys

from markit.vision import normalize_prompt as n

cases = [
    ("person . phone",         "person. phone."),   # spaced period (a real user typo)
    ("person. phone.",         "person. phone."),
    ("person,phone",           "person. phone."),
    ("person and phone",       "person. phone."),
    ("  person .  phone  ",    "person. phone."),
    ("person",                 "person."),
    ("person.",                "person."),
    ("person. phone. laptop.", "person. phone. laptop."),
    ("",                       ""),
    ("...",                    ""),
    ("  ",                     ""),
    ("hard hat. safety vest.", "hard hat. safety vest."),
]

fails = 0
for inp, want in cases:
    got = n(inp)
    ok = got == want
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {inp!r:26s} -> {got!r}")

# 'and' is a separator, but must not be eaten inside a word
got = n("sandwich. bandana")
ok = got == "sandwich. bandana."
fails += not ok
print(f"\n  {'ok  ' if ok else 'FAIL'} word safety: {got!r}")

print("\n" + (f"FAILURES: {fails}" if fails else "ALL PROMPT TESTS PASSED"))
sys.exit(1 if fails else 0)
