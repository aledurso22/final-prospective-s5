"""Production float32 probe for the prospective Momentum DeltaNet study.
x64 OFF. Requires PM_SOURCE_DIR (the restored development checkpoint).

Declared constants (reused from the completed studies; smoke checks, not
accuracy certificates):
  TRAJ32 = 2e-5  relative: logits and both carries, kappa = 0 / g = 1 vs
                 native, nonzero incoming carries, chunk boundary;
  shared parameter and incoming-carry gradients within
                 max(2e-3 |g|, 1e3 eps32 G) per leaf (the mixed criterion);
  FD32 = (1e-2, 3e-3), REL32 = 2e-2, resolvability
                 |jvp| >= 100 eps32 max(|f|, 1) / h_min, finite-before-
                 resolution ordering; a FINITE unresolvable derivative is a
                 reported limitation, never a fabricated dead parameter;
  ENTRY32 = 2e-5 relative: executed float32 transition entries vs A_kappa
                 evaluated in float64 on the SAME rounded gates.
The kappa derivative at the start (kappa = 0) is a derivative of the smooth
forward law, which is not clipped; kappa - h < 0 lies outside the PROJECTED
domain only, not outside the forward function's domain.
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
from experiments.nested_memory import model as NM                  # noqa: E402
from experiments.nested_memory import task as TK                   # noqa: E402
from experiments.prospective_momentum import dynamics as PD        # noqa: E402
from experiments.prospective_momentum import model as PM           # noqa: E402
from experiments.prospective_momentum import source as SRC         # noqa: E402

assert not jax.config.read("jax_enable_x64"), "an import enabled x64"
fails = []
eps32 = float(onp.finfo(onp.float32).eps)
rs = onp.random.RandomState(7)
TRAJ32, FD32, REL32, ENTRY32 = 2e-5, (1e-2, 3e-3), 2e-2, 2e-5
F32 = onp.float32


# ---- finite-before-resolution guards (same policy as the meta-delta probe)
def all_finite(*arrs):
    return all(bool(onp.all(onp.isfinite(onp.asarray(a, onp.float64))))
               for a in arrs)


def update_worst(worst, err):
    if not onp.isfinite(err):
        return float("inf")
    return max(worst, err)


def tangent_decision(label, jvp, fval, perturbed, rel_tol=None):
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
    for h, e in errs.items():
        print(f"  {label} jvp {jvp:.5e} fd(h={h}) {fds[h]:.5e} rel {e:.2e} "
              f"(threshold {thr:.2e})")
    if abs(jvp) < thr:
        return [], True
    for h, e in errs.items():
        if not e < rel_tol:
            out.append(f"{label} h={h}: rel {e:.2e}")
    return out, False


if update_worst(1e-9, float("nan")) < TRAJ32:
    fails.append("guard regression: a NaN trajectory error was not rejected")
_f, _lim = tangent_decision("guard regression", float("nan"), 1.0,
                            {1e-2: (1.0, 1.0), 3e-3: (1.0, 1.0)})
if not _f or _lim:
    fails.append("guard regression: a NaN JVP was labelled a limitation")
_f, _lim = tangent_decision("guard regression", 1e-6, 1.0,
                            {1e-2: (1.0 + 1e-8, 1.0 - 1e-8),
                             3e-3: (1.0 + 3e-9, 1.0 - 3e-9)})
if _f or not _lim:
    fails.append("guard regression: finite below-threshold derivative not a "
                 "limitation")


def rel(a, b):
    a, b = onp.asarray(a, onp.float64), onp.asarray(b, onp.float64)
    if not all_finite(a, b):
        return float("inf")
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def check_dtype(label, *arrs):
    for a in arrs:
        if onp.asarray(a).dtype != F32:
            fails.append(f"dtype {label}: {onp.asarray(a).dtype}")


# ---- restored development checkpoint -------------------------------------
src_dir = os.environ.get("PM_SOURCE_DIR")
if not src_dir:
    print("FAIL: PM_SOURCE_DIR not set; the restored source is required")
    sys.exit(1)
st, sel = SRC.load_metadata(src_dir)
SRC.verify_metadata("momentum_delta", st, sel)
pn = SRC.restore("momentum_delta", src_dir)
check_dtype("restored leaves", *pn.values())

b = TK.generate_batch(9600, 1)
eps = [{k: jnp.asarray(b[k][i]) for k in ("key_id", "val_id", "event",
                                          "label")} for i in range(2)]


def loss_fn(rule, ep):
    q = ep["event"] == TK.QUERY

    def f(p, c0):
        out = PM.rollout(rule, p, ep, carry0=c0)
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(jnp.maximum(ep["label"], 0),
                                          TK.N_VALUES)) * q
        return jnp.sum(ce) / jnp.sum(q)
    return f


worst = 0.0
for rule in ("prospective_momentum", "gain_momentum"):
    pc = PM.convert_momentum(pn, rule)
    check_dtype(f"{rule} leaves", *pc.values())
    for ep in eps:
        c0 = (jnp.asarray(0.3 * rs.randn(NM.D_V, NM.D_K), jnp.float32),
              jnp.asarray(0.3 * rs.randn(NM.D_V, NM.D_K), jnp.float32))
        on = PM.rollout("momentum_delta", pn, ep, carry0=c0)
        oc = PM.rollout(rule, pc, ep, carry0=c0)
        check_dtype(f"{rule} outputs", oc["logits"], *oc["final_carry"],
                    *oc["gates"])
        worst = update_worst(worst, rel(oc["logits"], on["logits"]))
        for x, y in zip(oc["final_carry"], on["final_carry"]):
            worst = update_worst(worst, rel(x, y))
        o1 = PM.rollout(rule, pc, {k: v[:29] for k, v in ep.items()},
                        carry0=c0)
        o2 = PM.rollout(rule, pc, {k: v[29:] for k, v in ep.items()},
                        carry0=o1["final_carry"])
        worst = update_worst(worst, rel(jnp.concatenate(
            [o1["logits"], o2["logits"]]), oc["logits"]))
        gn = jax.grad(loss_fn("momentum_delta", ep), argnums=(0, 1))(pn, c0)
        gc = jax.grad(loss_fn(rule, ep), argnums=(0, 1))(pc, c0)
        G = math.sqrt(sum(float(onp.sum(onp.asarray(v, onp.float64) ** 2))
                          for v in gn[0].values()))
        pairs = [(k, gc[0][k], gn[0][k]) for k in gn[0]] + \
                [(f"carry0[{i}]", gc[1][i], gn[1][i]) for i in range(2)]
        for k, x, y in pairs:
            x64, y64 = onp.asarray(x, onp.float64), onp.asarray(y, onp.float64)
            if not all_finite(x64, y64):
                fails.append(f"{rule} grad {k} non-finite")
                continue
            d = float(onp.linalg.norm(x64 - y64))
            n = float(onp.linalg.norm(y64))
            if not d <= max(2e-3 * n, 1e3 * eps32 * G):
                fails.append(f"{rule} grad {k}: abs {d:.2e} norm {n:.2e}")
print(f"  restored checkpoint: worst native-nesting relative error "
      f"{worst:.2e} (TRAJ32 {TRAJ32})")
if not worst <= TRAJ32:
    fails.append(f"native nesting worst {worst:.2e}")

# ---- kappa derivative: actual start and an interior point ----------------
pk = PM.convert_momentum(pn, "prospective_momentum")
a, bb, mu, eta = PD.table_gates(pk, PD.write_table())
check_dtype("table gates", a, bb, mu, eta)
a64, b64, mu64, e64 = (onp.asarray(x, onp.float64) for x in (a, bb, mu, eta))
aq = a64 * b64 * e64
kb64 = float(onp.min(onp.where(aq > 0, ((1 + a64) * (1 + mu64) - aq)
                               / (2 * onp.where(aq > 0, aq, 1)), onp.inf)))
gb64 = float(onp.min(onp.where(aq > 0, (1 + a64) * (1 + mu64)
                               / onp.where(aq > 0, aq, 1), onp.inf)))
print(f"  source frozen-token bounds (float64 on float32 gates): kappa < "
      f"{kb64:.6e}, g < {gb64:.6e}")
fk_loss = loss_fn("prospective_momentum", eps[0])
for kappa0, label in ((0.0, "START kappa"), (0.5 * kb64, "INTERIOR kappa")):
    if label.startswith("INTERIOR") and not (
            onp.isfinite(kb64) and kappa0 - max(FD32) > 0
            and kappa0 + max(FD32) < kb64 * (1 - PD.PROJ_REL_MARGIN)):
        print(f"  LIMITATION: {label} fixture kappa0={kappa0} +- h is not "
              "strictly inside (0, cap) for the float32 steps; the float64 "
              "interior check (h <= 1e-5) is the evidence. Not a failure.")
        continue
    fk = lambda kk: fk_loss(dict(pk, kappa=jnp.asarray(  # noqa: E731
        [kk], jnp.float32)), None)
    jvp = float(jax.jvp(fk, (jnp.float32(kappa0),), (jnp.float32(1.0),))[1])
    fval = float(fk(jnp.float32(kappa0)))
    pert = {h: (float(fk(jnp.float32(kappa0 + h))),
                float(fk(jnp.float32(kappa0 - h)))) for h in FD32}
    f_, lim_ = tangent_decision(label, jvp, fval, pert)
    fails.extend(f_)
    if lim_:
        print(f"  LIMITATION: {label} derivative is FINITE and below float32 "
              "resolvability; the float64 check is the evidence. Not a dead "
              "parameter.")

# ---- projection and executed transitions in float32 ----------------------
pq, tel = PD.project(dict(pk, kappa=jnp.asarray([1e12], jnp.float32)))
cap32 = float(pq["kappa"][0])
check_dtype("projected kappa", pq["kappa"])
eff = 1.0 - cap32 / kb64 if onp.isfinite(kb64) else float("inf")
print(f"  projected float32 cap {cap32:.6e}; effective relative margin to the "
      f"float64 bound {eff:.3e} (declared {PD.PROJ_REL_MARGIN})")
if not (all_finite(cap32) and 0 <= cap32 < kb64 and eff >= 0.5 * PD.PROJ_REL_MARGIN):
    fails.append(f"float32 projection cap {cap32} vs bound {kb64}")
pg, _ = PD.project(dict(PM.convert_momentum(pn, "gain_momentum"),
                        log_g=jnp.asarray([50.0], jnp.float32)))
gcap = math.exp(float(pg["log_g"][0]))
print(f"  projected gain cap {gcap:.6e} (bound {gb64:.6e})")
if not (all_finite(gcap) and 0 < gcap < gb64):
    fails.append(f"float32 gain cap {gcap} vs bound {gb64}")

cases = (("native kappa=0", PD.prospective_step, 0.0, 0.0),
         ("candidate at projected cap", PD.prospective_step, cap32, cap32),
         ("gain at projected cap", PD.gain_step, gcap, gcap))
for label, step, scalar, s64 in cases:
    A = onp.asarray(PD.executed_transitions(step, (a, bb, mu, eta),
                                            jnp.asarray(scalar, jnp.float32),
                                            jnp.float32))
    check_dtype(f"transition {label}", A)
    counts, worst_e = {}, 0.0
    for i in range(A.shape[0]):
        lab, _ = PD.classify(A[i])
        counts[lab] = counts.get(lab, 0) + 1
        ref = (PD.analytic_transition(a64[i], b64[i], mu64[i], e64[i], s64)
               if step is PD.prospective_step else
               PD.analytic_gain_transition(a64[i], b64[i], mu64[i], e64[i],
                                           s64))
        err = float(onp.max(onp.abs(A[i] - ref)
                            / onp.maximum(1.0, onp.abs(ref))))
        worst_e = update_worst(worst_e, err)
    print(f"  executed float32 transitions, {label}: {counts}; worst entry "
          f"error {worst_e:.2e}")
    if counts.get("unstable") or counts.get("nonfinite"):
        fails.append(f"{label}: {counts}")
    if not worst_e <= ENTRY32:
        fails.append(f"{label}: transition entries {worst_e:.2e}")

# ---- a real float32 training step from the restored source --------------
from experiments.prospective_momentum import study as ST           # noqa: E402
p_in = dict(pk, kappa=jnp.asarray([0.5 * cap32], jnp.float32))
opt = ST.TX.init(p_in)
bt = {k: jnp.asarray(v) for k, v in TK.generate_batch(9700, 2).items()
      if k in ("key_id", "val_id", "event", "label")}
o = ST.train_step("prospective_momentum", p_in, opt, bt,
                  jnp.asarray(0.01, jnp.float32))
p1 = o[0]
check_dtype("trained leaves", *p1.values())
rep1 = PD.transition_report(p1, "prospective_momentum")
cap_step = float(o[8]["cap"])
cap_upd = rep1["kappa_bound_f64"] * (1 - PD.PROJ_REL_MARGIN)
print(f"  training step: kappa {float(p1['kappa'][0]):.6e}, step cap "
      f"{cap_step:.6e}, cap of UPDATED gates {cap_upd:.6e}, incoming cap "
      f"{cap32:.6e}, kappa/bound {rep1['kappa_over_bound']:.6f}, "
      f"classification {rep1['classification']}, grad {float(o[9]):.3e}")
if not all_finite(float(o[2]), float(o[9]), cap_step):
    fails.append("training step: non-finite loss, kappa gradient or cap")
elif abs(cap_step - cap_upd) > 1e-5 * cap_upd:
    fails.append(f"training step cap {cap_step} is not the updated-gate cap "
                 f"{cap_upd}")
why = PD.transition_failure(rep1)
if why:
    fails.append(f"training step: {why}")
# outward proposal on these UPDATED trained gates, then an inward move
pq1, t1 = PD.project(dict(p1, kappa=jnp.asarray([2.0 * cap_upd], jnp.float32)))
pin1, t2 = PD.project(dict(pq1, kappa=0.9 * pq1["kappa"]))
if int(t1["n_projected"]) != 1 or int(t2["n_projected"]) != 0 or \
        PD.transition_failure(PD.transition_report(
            pq1, "prospective_momentum")):
    fails.append("outward proposal / inward move on trained gates")

print("FLOAT32 PROBE:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
