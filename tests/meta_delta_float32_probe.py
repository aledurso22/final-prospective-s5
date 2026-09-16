"""Production float32 probe for the two-sided candidate. x64 OFF.

Declared smoke-check constants (not accuracy certificates):
  nesting logits relative <= 1e-4; shared gradients within
  max(2e-3 |g|, 1e3 eps32 G) per leaf; raw_r JVP vs central differences at
  h = 1e-2 and 3e-3, 2e-2 relative at EACH, resolvability
  |jvp| >= 100 eps32 max(|f|, 1) / h_min; raw_tau gradient at rho = 1
  <= 1e-3 max|dL/draw_r|; projected float32 points certified in float64.
"""

import math
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")
import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as onp                                               # noqa: E402

assert not jax.config.read("jax_enable_x64")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optax                                                       # noqa: E402
from experiments.meta_delta import calibrate as CAL                # noqa: E402
from experiments.meta_delta import dynamics as MD                  # noqa: E402
from experiments.meta_delta import model as MM                     # noqa: E402
from experiments.nested_memory import task as TK                   # noqa: E402

assert not jax.config.read("jax_enable_x64"), "an import enabled x64"
fails = []
eps32 = float(onp.finfo(onp.float32).eps)
rs = onp.random.RandomState(7)
slot = CAL.configurations()["slots"]["adaptive_delta/A"]
pd = MM.init_params("adaptive_delta", 11, init_coeffs=slot)
pd = dict(pd, gate_u=jnp.asarray(0.5 * rs.randn(31), jnp.float32),
          gate_b=jnp.asarray(0.5 * rs.randn(7), jnp.float32))
pc = MM.delta_to_two_sided(pd)
for tag, p in (("delta", pd), ("candidate", pc)):
    for k, v in p.items():
        if onp.asarray(v).dtype != onp.float32:
            fails.append(f"{tag} leaf {k} is {onp.asarray(v).dtype}")
b = TK.generate_batch(9500, 1)
ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event", "label")}
q = ep["event"] == TK.QUERY


def loss(rule):
    def f(p):
        out = MM.rollout(rule, p, ep)
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(jnp.maximum(ep["label"], 0),
                                          TK.N_VALUES)) * q
        return jnp.sum(ce) / jnp.sum(q)
    return f


yd = onp.asarray(MM.rollout("adaptive_delta", pd, ep)["logits"], onp.float64)
yc = onp.asarray(MM.rollout("gp_two_sided", pc, ep)["logits"], onp.float64)
rel = onp.linalg.norm(yc - yd) / onp.linalg.norm(yd)
print(f"  nesting logits rel {rel:.2e}")
if not rel <= 1e-4:
    fails.append(f"nesting logits rel {rel:.2e}")
gd = jax.grad(loss("adaptive_delta"))(pd)
gc = jax.grad(loss("gp_two_sided"))(pc)
G = math.sqrt(sum(float(onp.sum(onp.asarray(v, onp.float64) ** 2))
                  for v in gd.values()))
for k in gd:
    d = float(onp.linalg.norm(onp.asarray(gc[k], onp.float64)
                              - onp.asarray(gd[k], onp.float64)))
    n = float(onp.linalg.norm(onp.asarray(gd[k], onp.float64)))
    if not d <= max(2e-3 * n, 1e3 * eps32 * G):
        fails.append(f"shared grad {k}: abs {d:.2e} norm {n:.2e}")

f = loss("gp_two_sided")
fr = lambda r: f(dict(pc, raw_r=jnp.asarray([r], jnp.float32)))  # noqa: E731
jvp = float(jax.jvp(fr, (jnp.float32(0.0),), (jnp.float32(1.0),))[1])
fval = float(fr(jnp.float32(0.0)))
if not abs(jvp) >= 100 * eps32 * max(abs(fval), 1.0) / 3e-3:
    fails.append(f"raw_r tangent not resolvable: jvp {jvp:.3e}")
for h in (1e-2, 3e-3):
    fd = (float(fr(jnp.float32(h))) - float(fr(jnp.float32(-h)))) / (2 * h)
    e = abs(jvp - fd) / max(abs(fd), 1e-30)
    print(f"  raw_r jvp {jvp:.5e} fd(h={h}) {fd:.5e} rel {e:.2e}")
    if not e < 2e-2:
        fails.append(f"raw_r tangent h={h}: rel {e:.2e}")
gtau = abs(float(gc["raw_tau"][0])); gr = abs(float(gc["raw_r"][0]))
print(f"  start |dL/draw_r| {gr:.3e} |dL/draw_tau| {gtau:.3e}")
if not (gr > 0 and gtau <= 1e-3 * gr):
    fails.append("start derivatives: raw_r absent or raw_tau not ~0")

n_ok = 0
for i in range(400):
    p = dict(raw_eta=jnp.asarray([rs.uniform(-4, 3)], jnp.float32),
             raw_tau=jnp.asarray([rs.uniform(-4, 3)], jnp.float32),
             raw_r=jnp.asarray([rs.uniform(-2, 8)], jnp.float32))
    pq, _ = MD.project_two_sided(p)
    rep = MD.domain_report(pq)
    n_ok += int(rep["passed"])
    if not rep["passed"]:
        fails.append(f"projected float32 point not certified: {rep}")
        break
print(f"  projection: {n_ok} float32 projected points certified in float64")
# the ill-conditioned edge x -> 1+ (U ~ e/(x-1) sensitivity), explicitly
n_edge = 0
for dx in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1.5e-7):
    for jitter in range(-3, 4):
        eta = jnp.asarray([(1.0 + dx) / 2.0], jnp.float32)
        eta = eta * jnp.asarray(1.0 + jitter * eps32, jnp.float32)
        p = dict(raw_eta=jnp.log(eta), raw_tau=jnp.asarray([0.0], jnp.float32),
                 raw_r=jnp.asarray([30.0], jnp.float32))
        pq, _ = MD.project_two_sided(p)
        rep = MD.domain_report(pq)
        n_edge += 1
        if not rep["passed"]:
            fails.append(f"edge x-1={dx} jitter {jitter}: {rep}")
print(f"  projection edge x -> 1+: {n_edge} points checked")

if fails:
    print("FAILURES:")
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("  float32 probe: all checks passed")
