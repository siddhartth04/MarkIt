"""Test the rule engine with synthetic landmarks — no camera, no model."""
import _path  # noqa: F401  (adds the repo root to sys.path)

import importlib.util, math, types, sys
import markit.web, markit.rules, markit.tracking, markit.posture, markit.pipeline, markit.vision
import types
m = types.SimpleNamespace(**{**vars(markit.rules), **vars(markit.tracking), **vars(markit.posture), **vars(markit.vision), **vars(markit.pipeline), **vars(markit.web)})

class LM:
    def __init__(self,x,y,v=1.0): self.x,self.y,self.visibility=x,y,v

def body(lean_deg=0.0, hip_deg=180.0, knee_deg=180.0, vis=1.0):
    """Build a 33-landmark body with the requested torso lean / joint angles."""
    P = m.L
    lms=[LM(0.5,0.5,vis) for _ in range(33)]
    hipy, hx = 0.60, 0.5
    r=0.30; a=math.radians(lean_deg)
    sx, sy = hx + r*math.sin(a), hipy - r*math.cos(a)
    for j,(x,y) in {
        P.LEFT_SHOULDER:(sx-0.08,sy), P.RIGHT_SHOULDER:(sx+0.08,sy),
        P.LEFT_HIP:(hx-0.06,hipy),    P.RIGHT_HIP:(hx+0.06,hipy),
    }.items(): lms[j.value]=LM(x,y,vis)
    # knee placed so hip angle = hip_deg measured from the torso direction
    for side,sgn in (("LEFT",-1),("RIGHT",1)):
        hxx = hx+sgn*0.06
        ha = a + math.radians(180-hip_deg)
        kx, ky = hxx + 0.22*math.sin(ha), hipy + 0.22*math.cos(ha)
        lms[getattr(P,f"{side}_KNEE").value]=LM(kx,ky,vis)
        ka = ha + math.radians(180-knee_deg)
        lms[getattr(P,f"{side}_ANKLE").value]=LM(kx+0.22*math.sin(ka), ky+0.22*math.cos(ka), vis)
    return lms

print("=== metrics read back correctly ===")
for want in (0, 30, 45, 60, 90):
    got = m.pose_metrics(body(lean_deg=want))
    print(f"  lean {want:3d} -> measured {got['lean']:5.1f}")
    assert abs(got['lean']-want) < 1.5, "lean math wrong"

got = m.pose_metrics(body(hip_deg=90))
print(f"  sitting hip 90 -> measured hip {got['hip']:.0f}")
assert abs(got['hip']-90) < 3

print("\n=== low visibility returns None (rules stay quiet) ===")
print("  ", m.pose_metrics(body(vis=0.1)))
assert m.pose_metrics(body(vis=0.1)) is None

print("\n=== hold time debounces a twitch ===")
e = m.RuleEngine()
t = 1000.0
# brief 45-deg spike for 0.3s: must NOT alert (bending needs 1.0s)
for i in range(3):
    a = e.update(m.pose_metrics(body(lean_deg=50)), t); t += 0.1
print(f"  after 0.3s bent: active={a} log={len(e.log)}  (expect [] and 0)")
assert a == [] and len(e.log) == 0, "fired too early"

# hold it past 1.0s: must alert exactly once
for i in range(12):
    a = e.update(m.pose_metrics(body(lean_deg=50)), t); t += 0.1
print(f"  after 1.5s bent: active={a} log={len(e.log)}  (expect ['bending'] and 1)")
assert "bending" in a and len(e.log) == 1, "should have fired once"

# keep holding: no duplicate log entries
for i in range(20):
    e.update(m.pose_metrics(body(lean_deg=50)), t); t += 0.1
print(f"  still bent 2s later: log={len(e.log)} (expect 1 — rising edge only)")
assert len(e.log) == 1, "logged repeatedly instead of once"

# stand up, bend again -> second event
for i in range(10): e.update(m.pose_metrics(body(lean_deg=0)), t); t += 0.1
for i in range(15): a = e.update(m.pose_metrics(body(lean_deg=50)), t); t += 0.1
print(f"  bent again: log={len(e.log)} (expect 2)")
assert len(e.log) == 2

print("\n=== person leaves frame clears state ===")
e.update(None, t)
print(f"   active={e.update(None,t)} firing={e.firing}")
assert not e.firing

print("\n=== rule selection across poses ===")
for name,kw in {"upright":{}, "lean 30":{"lean_deg":30}, "bend 60":{"lean_deg":60},
                "sitting":{"hip_deg":90}, "crouch":{"knee_deg":70}}.items():
    e2 = m.RuleEngine(); tt=2000.0
    for i in range(40): act = e2.update(m.pose_metrics(body(**kw)), tt); tt += 0.1
    print(f"  {name:10s} -> {act}")

print("\nALL RULE TESTS PASSED")
