"""Production float32 probe for the two-sided candidate. x64 OFF.

Declared smoke-check constants (not accuracy certificates):
  nesting logits relative <= 1e-4; shared gradients within
  max(2e-3 |g|, 1e3 eps32 G) per leaf; raw_r JVP vs central differences at
  h = 1e-2 and 3e-3, 2e-2 relative at EACH, resolvability
  |jvp| >= 100 eps32 max(|f|, 1) / h_min; raw_tau gradient at rho = 1
  <= 1e-3 max|dL/draw_r|; projected float32 points certified in float64.

The first block uses a nonzero-gate STRESS fixture. Review R1 (535fb02) adds:
  * the ACTUAL initialized tree (zero gate, rho = 1, tau = 1): raw_r tangent
    and vanishing raw_tau tangent; if the raw_r derivative is below the float32
    resolvability threshold that is REPORTED as a limitation of float32
    finite differences, not as a failure or a dead parameter;
  * an executed float32 WIDER-REGION sequence (key changes, write, idle) at a
    point safely above rho = 1 and one at the projection margin, against an
    independent float64 dense-ODE reference for the same rounded inputs, at
    the existing float32 trajectory tolerance TRAJ32 = 2e-5;
  * a raw_r directional derivative at an INTERIOR wider-region point with
    h such that r +- h stays strictly inside the projection bound;
  * dtype assertions on coefficients, generator, carries and outputs.
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
TRAJ32, FD32, REL32 = 2e-5, (1e-2, 3e-3), 2e-2


# ---- finite-before-resolution guards (review F1, f13295c) ----------------
def all_finite(*arrs):
    return all(bool(onp.all(onp.isfinite(onp.asarray(a, onp.float64))))
               for a in arrs)


def update_worst(worst, err):
    """Aggregate an error WITHOUT letting a NaN be discarded by max():
    a non-finite error makes the aggregate +inf, which fails any tolerance."""
    if not onp.isfinite(err):
        return float("inf")
    return max(worst, err)


def tangent_decision(label, jvp, fval, perturbed, rel_tol=None):
    """Decide a directional-derivative check in the declared order:
    (1) every loss, perturbed loss, JVP, FD and error must be FINITE, else a
        failure - non-finite arithmetic is never "not resolvable";
    (2) only a FINITE derivative below the resolvability threshold receives
        the reported limitation;
    (3) otherwise each declared step must meet the relative tolerance.
    `perturbed` maps h -> (f(x + h), f(x - h)). Returns (failures, limited)."""
    rel_tol = REL32 if rel_tol is None else rel_tol
    out = []
    fds = {h: (fp - fm) / (2 * h) for h, (fp, fm) in perturbed.items()}
    errs = {h: abs(jvp - fd) / max(abs(fd), 1e-30) for h, fd in fds.items()}
    if not all_finite(jvp, fval, [v for pair in perturbed.values()
                                  for v in pair], list(fds.values()),
                      list(errs.values())):
        return [f"{label}: non-finite loss, JVP, perturbed loss, FD or error "
                f"(jvp={jvp}, f={fval}, perturbed={perturbed})"], False
    thr = 100 * eps32 * max(abs(fval), 1.0) / min(perturbed)
    for h, e in errs.items():                # printed in EVERY finite case
        print(f"  {label} jvp {jvp:.5e} fd(h={h}) {fds[h]:.5e} rel {e:.2e} "
              f"(threshold {thr:.2e})")
    if abs(jvp) < thr:
        return [], True
    for h, e in errs.items():
        if not e < rel_tol:
            out.append(f"{label} h={h}: rel {e:.2e}")
    return out, False


# lightweight rejection regressions for the guards themselves
if update_worst(1e-9, float("nan")) < TRAJ32:
    fails.append("guard regression: a NaN trajectory error was not rejected")
_f, _lim = tangent_decision("guard regression", float("nan"), 1.0,
                            {1e-2: (1.0, 1.0), 3e-3: (1.0, 1.0)})
if not _f or _lim:
    fails.append("guard regression: a NaN JVP was labelled a limitation")
_f, _lim = tangent_decision("guard regression", 1e-12, 1.0,
                            {1e-2: (1.0, float("inf")), 3e-3: (1.0, 1.0)})
if not _f or _lim:
    fails.append("guard regression: a non-finite perturbed loss passed")
# a FINITE derivative below the threshold is a reported limitation, not a
# failure (dispatch-1 correction); f = 1 gives threshold ~4e-3 at h = 3e-3
_f, _lim = tangent_decision("guard regression", 1e-6, 1.0,
                            {1e-2: (1.0 + 1e-8, 1.0 - 1e-8),
                             3e-3: (1.0 + 3e-9, 1.0 - 3e-9)})
if _f or not _lim:
    fails.append("guard regression: a finite below-threshold derivative was "
                 f"not reported as a limitation (failures {_f})")

slot = CAL.configurations()["slots"]["adaptive_delta/A"]
# ---- STRESS fixture: nonzero random gate (not the trained start)
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
# dispatch-1 correction: the SAME finite-before-resolution rule as every other
# tangent block; both declared perturbations, unchanged tolerance/threshold
pert_s = {h: (float(fr(jnp.float32(h))), float(fr(jnp.float32(-h))))
          for h in FD32}
f_, lim_ = tangent_decision("STRESS raw_r", jvp, fval, pert_s)
fails.extend(f_)
if lim_:
    print("  LIMITATION: stress-fixture raw_r derivative is FINITE and below "
          "float32 resolvability; errors printed above; not a failure")
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


# ---- ACTUAL START (review R1): unchanged initialized tree ----------------
slots = CAL.configurations()["slots"]
pc0 = MM.init_params("gp_two_sided", 300, init_coeffs=slots["gp_two_sided/A"])
pd0 = MM.init_params("adaptive_delta", 300, init_coeffs=slots["adaptive_delta/A"])
assert float(onp.max(onp.abs(onp.asarray(pc0["gate_u"])))) == 0.0
f0 = loss("gp_two_sided")
fr0 = lambda r: f0(dict(pc0, raw_r=jnp.asarray([r], jnp.float32)))  # noqa: E731
jvp0 = float(jax.jvp(fr0, (jnp.float32(0.0),), (jnp.float32(1.0),))[1])
fv0 = float(fr0(jnp.float32(0.0)))
thr0 = 100 * eps32 * max(abs(fv0), 1.0) / min(FD32)
g0 = jax.grad(f0)(pc0)
print(f"  ACTUAL START raw_r jvp {jvp0:.5e} (float32 resolvability {thr0:.2e}"
      f")  raw_tau grad {float(g0['raw_tau'][0]):.3e}")
pert0 = {h: (float(fr0(jnp.float32(h))), float(fr0(jnp.float32(-h))))
         for h in FD32}
f_, lim_ = tangent_decision("ACTUAL START raw_r", jvp0, fv0, pert0)
fails.extend(f_)
if lim_:
    print("  LIMITATION: actual-start raw_r derivative is FINITE and below the "
          "float32 finite-difference resolvability threshold; the float64 "
          "check is the evidence for this tangent. Not a dead parameter.")
g_tau0, g_r0 = float(g0["raw_tau"][0]), float(g0["raw_r"][0])
if not all_finite(g_tau0, g_r0):
    fails.append(f"actual start gradients non-finite: tau {g_tau0} r {g_r0}")
elif not abs(g_tau0) <= 1e-3 * max(abs(jvp0), abs(g_r0), 1e-30) + 1e3 * eps32:
    fails.append("actual start raw_tau tangent not ~0")

# ---- WIDER REGION, executed float32 vs float64 dense reference ----------
from experiments.adaptive_memory import dynamics as AD            # noqa: E402
from scipy.linalg import expm as sp_expm                          # noqa: E402


def dense_ref(W, Z, k, v, w, eta, tau, rho):
    """Independent float64 reference: the full (W, Z) matrix ODE,
    Wdot = -nu w (W K - v k^T) - Z/tau, Zdot = -nu (1-rho) w (W K - v k^T) - Z/tau,
    exponentiated as an augmented affine system."""
    dv, dk = W.shape
    n = dv * dk
    nu = eta / rho
    K = onp.outer(k, k)
    WK = onp.kron(K.T, onp.eye(dv))
    vk = onp.reshape(onp.outer(v, k), -1, order="F")
    A = onp.zeros((2 * n + 1, 2 * n + 1))
    A[:n, :n] = -nu * w * WK
    A[:n, n:2 * n] = -onp.eye(n) / tau
    A[:n, 2 * n] = nu * w * vk
    A[n:2 * n, :n] = -nu * (1 - rho) * w * WK
    A[n:2 * n, n:2 * n] = -onp.eye(n) / tau
    A[n:2 * n, 2 * n] = nu * (1 - rho) * w * vk
    z = onp.concatenate([onp.reshape(W, -1, order="F"),
                         onp.reshape(Z, -1, order="F"), [1.0]])
    out = sp_expm(A) @ z
    return (onp.reshape(out[:n], (dv, dk), order="F"),
            onp.reshape(out[n:2 * n], (dv, dk), order="F"))


eta_w, tau_w = jnp.float32(0.9), jnp.float32(1.0)
b_w = float(MD.log_rho_upper(eta_w[None], tau_w[None], onp.float32)[0])
points = dict(safely_above=onp.float32(0.3 * b_w),
              at_projection_margin=onp.float32(b_w))
rsw = onp.random.RandomState(99)
seq = []
for s_ in range(12):
    kk = rsw.randn(4); kk /= onp.linalg.norm(kk)
    w = 0.0 if s_ in (3, 4, 8, 9, 10) else float(rsw.uniform(0.2, 2.0))
    seq.append((kk.astype(onp.float32), rsw.randn(3).astype(onp.float32),
                onp.float32(w)))
for name, r in points.items():
    pr = dict(raw_eta=jnp.log(eta_w)[None], raw_tau=jnp.log(tau_w)[None],
              raw_r=jnp.asarray([r], jnp.float32))
    pr, _ = MD.project_two_sided(pr)
    c = MD.two_sided_response(pr)
    for k_, v_ in c.items():
        if onp.asarray(v_).dtype != onp.float32:
            fails.append(f"coefficient {k_} dtype {onp.asarray(v_).dtype}")
    eta_e, tau_e, rho_e = c["eta"][0], c["tau"][0], c["rho"][0]
    if not all_finite(*[onp.asarray(v_) for v_ in c.values()]):
        fails.append(f"{name}: non-finite executed coefficients")
        continue
    if not float(rho_e) > 1.0:
        fails.append(f"{name}: rho {float(rho_e)} not above 1")
    W32 = jnp.zeros((3, 4), jnp.float32); Z32 = jnp.zeros((3, 4), jnp.float32)
    W64 = onp.zeros((3, 4)); Z64 = onp.zeros((3, 4))
    worst = 0.0
    for kk, vv, w in seq:
        G = MD.two_sided_generator(eta_e, tau_e, rho_e, jnp.asarray(w))
        F = AD.expm2(G)
        a0 = jnp.exp(-AD.H / tau_e)
        for arr, lab in ((G, "generator"), (F, "F"), (a0, "a0")):
            if onp.asarray(arr).dtype != onp.float32:
                fails.append(f"{name}: {lab} dtype {onp.asarray(arr).dtype}")
        W32, Z32 = AD.two_state_step(W32, Z32, jnp.asarray(kk), jnp.asarray(vv),
                                     F, a0)
        if W32.dtype != onp.float32 or Z32.dtype != onp.float32:
            fails.append(f"{name}: carry dtype {W32.dtype}/{Z32.dtype}")
        W64, Z64 = dense_ref(W64, Z64, kk.astype(onp.float64),
                             vv.astype(onp.float64), float(w),
                             float(eta_e), float(tau_e), float(rho_e))
        if not all_finite(G, F, a0, W32, Z32):
            fails.append(f"{name}: non-finite executed G/F/a0/W/Z")
            worst = float("inf")
            break
        if not all_finite(W64, Z64):
            fails.append(f"{name}: non-finite float64 reference W/Z")
            worst = float("inf")
            break
        scale = max(1.0, float(onp.max(onp.abs(W64))),
                    float(onp.max(onp.abs(Z64))))
        for got, ref in ((W32, W64), (Z32, Z64)):
            err = float(onp.max(onp.abs(onp.asarray(got, onp.float64) - ref))) \
                / scale
            worst = update_worst(worst, err)
    print(f"  WIDER REGION {name}: rho {float(rho_e):.6f} float32 vs float64 "
          f"dense worst rel {worst:.2e} (tol {TRAJ32:.0e})")
    if not worst < TRAJ32:
        fails.append(f"wider region {name}: {worst:.2e}")

# interior wider-region derivative of the smooth forward law, via the model
r_int = 0.3 * b_w
h_max = max(FD32)
if not r_int + h_max < b_w:
    fails.append("interior point too close to the bound for the declared h")
pci = MM.delta_to_two_sided(pd0)
pci = dict(pci, raw_eta=jnp.log(eta_w)[None], raw_tau=jnp.log(tau_w)[None])
fi = loss("gp_two_sided")
fri = lambda r: fi(dict(pci, raw_r=jnp.asarray([r], jnp.float32)))  # noqa: E731
out_i = MM.rollout("gp_two_sided", dict(pci, raw_r=jnp.asarray([r_int],
                                                               jnp.float32)), ep)
if out_i["logits"].dtype != onp.float32:
    fails.append(f"output dtype {out_i['logits'].dtype}")
if not all_finite(out_i["logits"]):
    fails.append("wider interior: non-finite logits")
for carry in out_i["final_carry"]:
    if carry.dtype != onp.float32:
        fails.append(f"final carry dtype {carry.dtype}")
    if not all_finite(carry):
        fails.append("wider interior: non-finite final carry")
jvp_i = float(jax.jvp(fri, (jnp.float32(r_int),), (jnp.float32(1.0),))[1])
fv_i = float(fri(jnp.float32(r_int)))
thr_i = 100 * eps32 * max(abs(fv_i), 1.0) / min(FD32)
print(f"  WIDER INTERIOR raw_r={r_int:.4f} jvp {jvp_i:.5e} threshold "
      f"{thr_i:.2e}")
pert_i = {h: (float(fri(jnp.float32(r_int + h))),
              float(fri(jnp.float32(r_int - h)))) for h in FD32}
f_, lim_ = tangent_decision("WIDER INTERIOR raw_r", jvp_i, fv_i, pert_i)
fails.extend(f_)
if lim_:
    print("  LIMITATION: wider-interior raw_r derivative is FINITE and below "
          "float32 resolvability; reported, not a failure")

if fails:
    print("FAILURES:")
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("  float32 probe: all checks passed")
