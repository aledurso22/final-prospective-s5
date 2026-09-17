"""Production float32 probe for the TSS containment study. x64 OFF.

The focused-check module runs in float64, while the study refuses to start
unless x64 is disabled. This probe exercises the NEW paths in the production
precision: the five-carry processing rollout (executed coefficient form) and
its streaming, the two exact points, the executed-coefficient gate (including
points the declared numerical gaps accept but the rounded coefficients must
refuse), the feasibility repair, coefficient derivatives against an
independent float64 original-law reference whose PRIMAL carries and loss are
also checked, and the masked optimizer path of both processing arms.

Review R5 (7613c86): every piece of derivative evidence - production loss and
JVP, reference loss, reference derivative, reference carries, perturbed loss,
forward difference and error - must be finite BEFORE any tolerance or
resolvability decision; injected-NaN regressions prove it.

Declared constants, unchanged from the completed studies:
  TRAJ32 = 2e-5  relative: logits, carries and streaming;
  REF32  = 2e-2  relative: a float32 JVP against an INDEPENDENT float64
                 sequential sensitivity of the same rounded inputs (a
                 cross-precision comparison, so the float32 smoke tolerance
                 applies, not a float64 identity tolerance);
  resolvability |jvp| >= 100 eps32 max(|f|, 1) / h, finite BEFORE resolution;
                 a FINITE unresolvable derivative is a reported limitation,
                 never a fabricated dead parameter.

Bitwise equality is asserted only for stored constants that must not change.
Requires PM_SOURCE_RUN (the read-only replication run).
"""

import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")
import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as onp                                               # noqa: E402

assert not jax.config.read("jax_enable_x64")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optax                                                       # noqa: E402
from experiments.nested_memory import dynamics as NMD              # noqa: E402
from experiments.nested_memory import model as NM                  # noqa: E402
from experiments.nested_memory import task as TK                   # noqa: E402
from experiments.prospective_momentum import filtered as FL        # noqa: E402
from experiments.prospective_momentum import model as PM           # noqa: E402
from experiments.prospective_momentum import ordinary as OD        # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import study as ST           # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

assert not jax.config.read("jax_enable_x64"), "an import enabled x64"
fails = []
eps32 = float(onp.finfo(onp.float32).eps)
F32 = onp.float32
TRAJ32, REF32 = 2e-5, 2e-2
H = FL.H


def all_finite(*arrs):
    return all(bool(onp.all(onp.isfinite(onp.asarray(a, onp.float64))))
               for a in arrs)


def rel(a, b):
    a, b = onp.asarray(a, onp.float64), onp.asarray(b, onp.float64)
    if not all_finite(a, b):
        return float("inf")
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def finite_tree(*trees):
    return all(bool(onp.all(onp.isfinite(onp.asarray(v, onp.float64))))
               for t in trees
               for v in jax.tree_util.tree_leaves(t)
               if onp.issubdtype(onp.asarray(v).dtype, onp.inexact))


def moved_finite(after, before):
    if not all_finite(after, before):
        return False
    return float(after) != float(before)


def close(label, got, ref, rel_tol, floor_eps=1e3 * eps32):
    if not all_finite(got, ref):
        fails.append(f"{label}: non-finite ({got} vs {ref})")
        return False
    a = abs(float(got) - float(ref))
    r = a / max(abs(float(ref)), 1e-30)
    print(f"  {label}: {float(got):.9g} vs {float(ref):.9g} abs {a:.3e} rel "
          f"{r:.3e} (tolerance {rel_tol}, floor {floor_eps:.2e})")
    if a <= max(rel_tol * abs(float(ref)), floor_eps * max(abs(float(ref)),
                                                           1.0)):
        return True
    fails.append(f"{label}: abs {a:.3e} rel {r:.3e}")
    return False


def check(label, cond):
    print(f"  {label}: {'ok' if cond else 'FAILED'}")
    if not cond:
        fails.append(label)


# guard regressions
if moved_finite(float("nan"), 0.3) or moved_finite(0.3, 0.3):
    fails.append("guard regression: NaN or unchanged counted as movement")
if finite_tree({"a": onp.array([onp.nan])}):
    fails.append("guard regression: a NaN tree was called finite")


def check_dtype(label, *arrs):
    for a in arrs:
        if onp.asarray(a).dtype != F32:
            fails.append(f"dtype {label}: {onp.asarray(a).dtype}")


run = os.environ.get("PM_SOURCE_RUN", TC.SOURCE_RUN)
assert os.path.isdir(RS.sources_dir(run)), f"source run {run} is REQUIRED"
with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
    manifest = json.load(fh)
pn = RS.restore_source(run, RS.find_entry(manifest, TC.SOURCE_DEV,
                                          TC.SOURCE_FAMILY))
check_dtype("restored source", *pn.values())


def tree(M, gam, T):
    dt = pn["A_log"].dtype
    return dict(pn, fil_M=jnp.full((1,), M, dtype=dt),
                fil_gamma=jnp.full((1,), gam, dtype=dt),
                fil_T=jnp.full((1,), T, dtype=dt))


def ep(seed, i=0):
    b = TK.generate_batch(seed, 2)
    return {k: jnp.asarray(b[k][i]) for k in ("key_id", "val_id", "event",
                                              "label")}


print("--- 1. five-carry rollout, streaming and dtypes (float32) ---")
p0 = PM.add_extension(pn, FL.FILTERED)
check_dtype("start leaves", *p0.values())
e0 = ep(9750)
full = PM.rollout(FL.FILTERED, p0, e0)
a = PM.rollout(FL.FILTERED, p0, {k: v[:29] for k, v in e0.items()})
b = PM.rollout(FL.FILTERED, p0, {k: v[29:] for k, v in e0.items()},
               carry0=a["final_carry"])
check("carries, logits and gates finite",
      all_finite(full["logits"], *full["final_carry"], *full["gates"],
                 full["w_norm"], full["aux_norm"]))
check("five carries", len(full["final_carry"]) == 5)
check_dtype("rollout", full["logits"], *full["final_carry"])
close("streaming logits", rel(jnp.concatenate([a["logits"], b["logits"]]),
                              full["logits"]), 0.0, TRAJ32, floor_eps=TRAJ32)
for i, (x, y) in enumerate(zip(b["final_carry"], full["final_carry"])):
    close(f"streaming carry {i}", rel(x, y), 0.0, TRAJ32, floor_eps=TRAJ32)

print("--- 2. the two exact points, at trajectory tolerances ---")
nat_logits = PM.rollout("momentum_delta", pn, e0)["logits"]
rec = PM.rollout(FL.FILTERED, TC.native_point_tree(pn), e0)
close("native point vs native rule (logits)", rel(rec["logits"], nat_logits),
      0.0, TRAJ32, floor_eps=TRAJ32)
for i in range(2):
    close(f"native point vs native rule (carry {i})",
          rel(rec["final_carry"][i],
              PM.rollout("momentum_delta", pn, e0)["final_carry"][i]),
          0.0, TRAJ32, floor_eps=TRAJ32)
# the boundary against the two-tap operator at kappa = 1 (same executed dtype)
op = dict(pn, kappa=jnp.ones((1,), pn["A_log"].dtype))
close("TSS boundary at T = h vs the two-tap operator at kappa = 1",
      rel(PM.rollout(FL.FILTERED, tree(0.0, 0.0, H), e0)["logits"],
          PM.rollout(OD.ORDINARY, op, e0)["logits"]), 0.0, TRAJ32,
      floor_eps=TRAJ32)
check("the TSS start is NOT the native function",
      rel(full["logits"], nat_logits) > 1e-4)

print("--- 3. the executed-coefficient gate in float32 ---")
e3 = ep(9749)
for M, gam, T, want in ((0.0, 0.0, H, True), (0.0, H, 0.0, True),
                        (0.4, 0.2, 1.1, True), (1.0, 0.0, 0.0, False),
                        (1e18, 0.0, FL.G_MIN, False),
                        (3.4e38, 0.0, 0.6, False)):
    # the coefficients THIS compiled rollout executed, not a recomputation
    out3 = TC.rollout_outputs(FL.FILTERED, tree(M, gam, T), e3)
    rep = FL.filter_report(FL.coefficient_values(out3["coeff"]))
    guard = FL.in_loop_guard(out3["coeff"])
    ok = FL.filter_failure(rep) is None
    print(f"  (M={M:g}, gamma={gam:g}, T={T:g}) a={rep['a']:.9g} "
          f"b={rep['b']:.9g} c={rep['c']:.9g} d={rep['d']:.9g} "
          f"{rep['classification']} accepted={ok} guard={bool(guard['ok'])} "
          f"gaps {rep['gap_gamma_plus_T']:.3e}/{rep['gap_filter']:.3e}")
    check(f"executed gate verdict at (M={M:g}, gamma={gam:g}, T={T:g})",
          ok == want and bool(guard["ok"]) == want)
nat3 = FL.coefficient_values(TC.rollout_outputs(
    FL.FILTERED, TC.native_point_tree(pn), e3)["coeff"])
check("executed native-point coefficients are exactly a=b=d=0, c=1",
      (nat3["a"], nat3["b"], nat3["c"], nat3["d"]) == (0.0, 0.0, 1.0, 0.0))
check("the declared gaps alone would have accepted the refused large mass",
      FL.report_from_tree(tree(1e18, 0.0, FL.G_MIN))["gap_gamma_plus_T"]
      >= 0.0)

print("--- 4. the feasibility repair in float32 ---")
rs = onp.random.RandomState(3)
for _ in range(8):
    p = tree(float(rs.uniform(-1, 2)), float(rs.uniform(-1, 1)),
             float(rs.uniform(-1, 3)))
    once, tel = FL.repair(p)
    twice, tel2 = FL.repair(once)
    check_dtype("repair", *(once[k] for k in FL.LEAVES))
    if not (all(float(once[k][0]) >= 0.0 for k in FL.LEAVES)
            and all(onp.array_equal(onp.asarray(once[k]),
                                    onp.asarray(twice[k]))
                    for k in FL.LEAVES)
            and int(tel2["n_repaired"]) == 0
            and FL.filter_failure(FL.report_from_tree(once)) is None):
        fails.append(f"repair failed for {[float(p[k][0]) for k in FL.LEAVES]}"
                     f" -> {[float(once[k][0]) for k in FL.LEAVES]}")
for M, gam, T in (FL.tss_boundary(), FL.native_point()):
    out, tel = FL.repair(tree(M, gam, T))
    check(f"exact point ({M:g}, {gam:g}, {T:g}) is a fixed point of the repair",
          int(tel["n_repaired"]) == 0
          and all(onp.array_equal(onp.asarray(out[k]),
                                  onp.asarray(tree(M, gam, T)[k]))
                  for k in FL.LEAVES))
out, _ = FL.repair(tree(1.0, 0.0, 0.0))
check("the missing-damping counterexample is repaired to a stable filter",
      FL.filter_failure(FL.report_from_tree(out)) is None
      and float(out["fil_gamma"][0] + out["fil_T"][0]) > 0.0)

print("--- 5. coefficient sensitivity against an independent float64 "
      "reference ---")


def independent_sensitivity(p, ep_, direction):
    """Independently coded sequential float64 forward sensitivity of the
    ORIGINAL form of the law; only the coefficient-independent inputs
    (preprocessing, gates, readout) are shared. Returns the loss, the
    derivative and the final W, U, y carries."""
    from experiments.adaptive_memory import model as AM
    e = AM.sanitize_episode(ep_)
    key_id, val_id, event = e["key_id"], e["val_id"], e["event"]
    k_all, k_valid = NMD.safe_normalize(p["key_raw"])
    keys = onp.asarray(k_all[key_id], onp.float64)
    has_v = (val_id >= 0).astype(jnp.float32)
    vals = onp.asarray(p["value_table"][jnp.maximum(val_id, 0)]
                       * has_v[:, None], onp.float64)
    mask = onp.asarray((event == TK.WRITE).astype(jnp.float32)
                       * k_valid[key_id], onp.float64)
    gx = NM.gate_features(key_id, val_id, event).astype(jnp.float32)
    al, be, mu, eta = (onp.asarray(x, onp.float64)
                       for x in NM._momentum_gates(p, gx))
    Hm = onp.asarray(p["readout_W"], onp.float64)
    bias = onp.asarray(p["readout_b"], onp.float64)
    lab = onp.maximum(onp.asarray(e["label"]), 0)
    qq = (onp.asarray(e["event"]) == TK.QUERY).astype(onp.float64)
    M, gam, T = (float(onp.asarray(p[k]).ravel()[0]) for k in FL.LEAVES)
    dM, dgam, dT = direction
    A = M + H * (gam + T)
    dA = dM + H * (dgam + dT)
    d_v, d_k = vals.shape[1], keys.shape[1]
    zero = lambda: onp.zeros((d_v, d_k))                        # noqa: E731
    W, U, y, y_prev, Rp = zero(), zero(), zero(), zero(), zero()
    dW, dU, dy, dy_prev, dRp = zero(), zero(), zero(), zero(), zero()
    L = keys.shape[0]
    logits = onp.zeros((L, Hm.shape[0]))
    dlogits = onp.zeros_like(logits)
    for t in range(L):
        k, v, m = keys[t], vals[t], mask[t]
        Wb, dWb = al[t] * W, al[t] * dW
        R = m * onp.outer(Wb @ k - v, k)
        dR = m * onp.outer(dWb @ k, k)
        N = M * (y - y_prev) + H * H * (R - y) + H * T * (R - Rp)
        dN = (dM * (y - y_prev) + M * (dy - dy_prev) + H * H * (dR - dy)
              + H * dT * (R - Rp) + H * T * (dR - dRp))
        y_new = y + N / A
        dy_new = dy + dN / A - N * dA / (A * A)
        U_new = mu[t] * U + eta[t] * y_new
        dU_new = mu[t] * dU + eta[t] * dy_new
        W_new = Wb - be[t] * U_new
        dW_new = dWb - be[t] * dU_new
        y_prev, dy_prev = y, dy
        W, U, y, Rp = W_new, U_new, y_new, R
        dW, dU, dy, dRp = dW_new, dU_new, dy_new, dR
        logits[t] = Hm @ (W @ k) + bias
        dlogits[t] = Hm @ (dW @ k)
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - onp.log(onp.exp(z).sum(axis=1))[:, None]
    pr = onp.exp(logp)
    ce = -logp[onp.arange(L), lab] * qq
    dce = -(dlogits[onp.arange(L), lab] - (pr * dlogits).sum(axis=1)) * qq
    n = max(qq.sum(), 1.0)
    return float(ce.sum() / n), float(dce.sum() / n), W, U, y


def loss_of(p, ep_):
    q = (ep_["event"] == TK.QUERY)
    lab = jnp.maximum(ep_["label"], 0)

    def f(coeffs):
        pp = dict(p, **{k: jnp.asarray([coeffs[i]], dtype=p[k].dtype)
                        for i, k in enumerate(FL.LEAVES)})
        out = PM.rollout(FL.FILTERED, pp, ep_)
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(lab, TK.N_VALUES,
                                          dtype=out["logits"].dtype)) * q
        return jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0)
    return f


#: absolute floor factor for a DECLARED analytic zero (amendment after the
#: failed dispatch 20260917-152713); eps is that of the dtype that computed
#: each derivative: float32 for the production JVP, float64 for the reference
ZERO_FLOOR = 1e3
EPS64 = float(onp.finfo(onp.float64).eps)


def sensitivity_decision(label, val, jvp, ref_val, ref_deriv, ref_carries,
                         prod_carries, fwd_val, hstep, expected_zero=False):
    """FINITE FIRST for every piece of evidence, then the primal agreement
    (TRAJ32, cross precision), then the DECISIVE derivative decision, then
    the diagnostic forward difference, which must be finite too.

    `expected_zero` is DECLARED by the fixture from an analytic identity,
    never inferred from a measured magnitude: the production JVP and the
    independent reference are then checked SEPARATELY against their own
    dtype's absolute floor. Otherwise the reference must be nondegenerate and
    the JVP must agree with it at REF32. Returns (failures, info)."""
    pieces = dict(loss=val, jvp=jvp, reference_loss=ref_val,
                  reference_derivative=ref_deriv, perturbed_loss=fwd_val)
    bad = [k for k, x in pieces.items() if not all_finite(x)]
    bad += [f"reference carry {i}" for i, c in enumerate(ref_carries)
            if not all_finite(c)]
    bad += [f"production carry {i}" for i, c in enumerate(prod_carries)
            if not all_finite(c)]
    if bad:
        return [f"{label}: non-finite evidence {bad}"], None
    fd = (float(fwd_val) - float(val)) / hstep
    loss_err = abs(float(val) - ref_val) / max(abs(ref_val), 1e-30)
    carry_err = [rel(pc, rc) for pc, rc in zip(prod_carries, ref_carries)]
    if not all_finite(fd, loss_err, carry_err):
        return [f"{label}: non-finite FD or primal error"], None
    fails = []
    if loss_err > TRAJ32 or max(carry_err) > TRAJ32:
        fails.append(f"{label}: independent primal differs (loss "
                     f"{loss_err:.2e}, carries {carry_err})")
    resolvable = abs(float(jvp)) >= 100 * eps32 * max(abs(float(val)),
                                                      1.0) / hstep
    if expected_zero:
        prod_floor = ZERO_FLOOR * eps32 * max(abs(float(val)), 1.0)
        ref_floor = ZERO_FLOOR * EPS64 * max(abs(ref_val), 1.0)
        if abs(float(jvp)) > prod_floor:
            fails.append(f"{label}: declared analytic zero, but float32 JVP "
                         f"{float(jvp):.3e} > floor {prod_floor:.3e}")
        if abs(ref_deriv) > ref_floor:
            fails.append(f"{label}: declared analytic zero, but float64 "
                         f"reference {ref_deriv:.3e} > floor {ref_floor:.3e}")
        return fails, dict(fd=fd, err=None, loss_err=loss_err,
                           carry_err=carry_err, resolvable=bool(resolvable),
                           production_floor=prod_floor,
                           reference_floor=ref_floor,
                           verdict=("analytic zero verified within tolerance"
                                    if not fails else
                                    "analytic zero NOT verified"))
    err = abs(float(jvp) - ref_deriv) / max(abs(ref_deriv), 1e-30)
    if not all_finite(err):
        return [f"{label}: non-finite derivative error"], None
    if abs(ref_deriv) <= 1e-12:
        fails.append(f"FIXTURE DEFECT: degenerate sensitivity for {label}")
    elif err > REF32:
        fails.append(f"{label}: derivative vs float64 reference {err:.2e}")
    return fails, dict(fd=fd, err=err, loss_err=loss_err,
                       carry_err=carry_err, resolvable=bool(resolvable),
                       verdict="agrees with the independent reference")


# injected-NaN regressions for the decision itself
_z = [onp.zeros((2, 2))] * 3
_nan = float("nan")
for what, args in (
        ("reference derivative", (1.0, 0.5, 1.0, _nan, _z, _z, 1.0)),
        ("reference primal loss", (1.0, 0.5, _nan, 0.5, _z, _z, 1.0)),
        ("reference primal carry", (1.0, 0.5, 1.0, 0.5,
                                    [onp.full((2, 2), _nan)] * 3, _z, 1.0)),
        ("perturbed loss", (1.0, 0.5, 1.0, 0.5, _z, _z, _nan))):
    f_, _ = sensitivity_decision("regression", *args, 1e-3)
    check(f"NaN {what} is rejected, not accepted", bool(f_))
f_, info_ = sensitivity_decision("regression", 1.0, 0.5, 1.0, 0.5, _z, _z,
                                 1.0 + 0.5e-3, 1e-3)
check("finite consistent evidence is accepted", not f_ and info_ is not None)
# amendment regressions: a DECLARED analytic zero is rejected when either
# derivative exceeds its own dtype's floor or is non-finite
_f32_floor = ZERO_FLOOR * eps32
_f64_floor = ZERO_FLOOR * EPS64
for what, jv, rf, reject in (
        ("zero production and roundoff reference", 0.0, -1.1e-17, False),
        ("production above the float32 floor", 3 * _f32_floor, 0.0, True),
        ("reference above the float64 floor", 0.0, 3 * _f64_floor, True),
        ("reference below the float32 but above the float64 floor", 0.0,
         0.5 * _f32_floor, True),
        ("non-finite production", _nan, 0.0, True),
        ("non-finite reference", 0.0, float("inf"), True)):
    f_, info_ = sensitivity_decision("zero regression", 1.0, jv, 1.0, rf, _z,
                                     _z, 1.0, 1e-3, expected_zero=True)
    check(f"declared zero: {what} is "
          f"{'rejected' if reject else 'accepted'}", bool(f_) == reject)

e1 = ep(9751)
# the ONE declared analytic zero: for M = 0, gamma = h the law is
# y_next = R_t + T/(h+T) (y - R_prev); matched (zero) initialization gives
# y = R_prev on every token, so the trajectory is native for every fixed T on
# this line and dL/dT = 0 exactly, for any data
for point, direction, name, expected_zero in (
        ((0.0, 0.0, H), (1.0, 0.0, 0.0), "M inward at the TSS boundary",
         False),
        ((0.0, 0.0, H), (0.0, 1.0, 0.0), "gamma inward at the TSS boundary",
         False),
        ((0.0, 0.0, H), (0.0, 0.0, 1.0), "T on the TSS boundary", False),
        ((0.0, H, 0.0), (0.0, 0.0, 1.0), "T inward at the native point",
         True),
        ((0.4, 0.3, 1.5), (0.0, 0.0, 1.0), "T at an interior point", False)):
    p = tree(*point)
    ref_val, ref, Wr, Ur, yr = independent_sensitivity(p, e1, direction)
    prod = PM.rollout(FL.FILTERED, p, e1)["final_carry"]
    f = loss_of(p, e1)
    c0 = jnp.asarray(point, jnp.float32)
    dvec = jnp.asarray(direction, jnp.float32)
    val, jvp = jax.jvp(f, (c0,), (dvec,))
    hstep = 1e-3
    fwd = f(c0 + hstep * dvec)                    # INWARD points only
    f_, info = sensitivity_decision(name, val, jvp, ref_val, ref,
                                    (Wr, Ur, yr), prod[:3], fwd, hstep,
                                    expected_zero=expected_zero)
    print(f"  {name}: float32 jvp {float(jvp):.6e} independent float64 "
          f"{ref:.6e} {info}")
    fails.extend(f_)
    if expected_zero:
        if info is not None:
            print(f"  {name}: {info['verdict']} (float32 floor "
                  f"{info['production_floor']:.2e}, float64 floor "
                  f"{info['reference_floor']:.2e}; forward-difference "
                  f"diagnostic {info['fd']:.3e})")
    elif info is not None and not info["resolvable"]:
        print("  LIMITATION: finite but below float32 forward-difference "
              "resolvability; the independent float64 reference is the "
              "evidence, not a dead parameter")

print("--- 6. the masked optimizer path of both processing arms (float32) ---")
bt = {k: jnp.asarray(v) for k, v in TK.generate_batch(9752, 2).items()
      if k in ("key_id", "val_id", "event", "label")}
lr = jnp.asarray(0.01, jnp.float32)
for arm in (TC.TSS, TC.GEN):
    p = tree(0.0 if arm == TC.TSS else 0.25, 0.0 if arm == TC.TSS else 0.15,
             1.2)
    opt = ST.TX.init(p)
    o = TC.train_step_filtered(FL.FILTERED, TC.FROZEN_LEAVES[arm], p, opt, bt,
                               lr)
    p2, opt2, tel, grads = o[0], o[1], o[8], o[9]
    check_dtype(f"{arm} updated leaves", *p2.values())
    scalars = [float(x) for x in o[2:8]] + [float(v) for v in grads.values()]
    if not (finite_tree(p2, opt2) and all_finite(*scalars)):
        fails.append(f"{arm}: non-finite updated tree, optimizer state or "
                     f"returned scalars")
        continue
    check(f"{arm}: the executed-coefficient gate accepted the forward pass",
          bool(tel["gate_ok"]))
    check(f"{arm}: stored constants unchanged (bitwise)",
          TC.frozen_leaf_differences(p2, p, arm) == {})
    mask = TC.leaf_mask(p, TC.FROZEN_LEAVES[arm])
    check(f"{arm}: the mask routes exactly the frozen leaves",
          all(float(mask[k]) == 0.0 for k in TC.FROZEN_LEAVES[arm])
          and all(float(mask[k]) == 1.0 for k in p
                  if k not in TC.FROZEN_LEAVES[arm]))
    moved = [k for k in FL.LEAVES if moved_finite(p2[k][0], p[k][0])]
    check(f"{arm}: the trainable coefficients moved finitely ({moved})",
          bool(moved) and all(k not in TC.FROZEN_LEAVES[arm] for k in moved))
    check(f"{arm}: the backbone moved",
          moved_finite(onp.asarray(p2["a_proj"]).ravel()[0],
                       onp.asarray(p["a_proj"]).ravel()[0]))
    m6, sets6 = TC.evaluate_arm(arm, p2, TK.generate_batch(9753, 4))
    bad = TC.checkpoint_failure(arm, p2, opt2, p, m6, sets6)
    check(f"{arm}: the updated tree passes checkpoint acceptance on the "
          f"coefficients its evaluation executed ({bad})", bad is None)
    print(f"  {arm}: loss {float(o[2]):.6f} acc {float(o[3]):.4f} grads "
          f"{ {k: float(v) for k, v in grads.items()} } repaired "
          f"{int(tel['n_repaired'])} jury_min "
          f"{float(tel['gate_jury_min']):.3e}")

print("--- 7. a NaN update is refused, not counted as movement ---")
nan_tree = tree(float("nan"), 0.0, H)
nan_exec = FL.coefficient_values(TC.rollout_outputs(FL.FILTERED, nan_tree,
                                                    e0)["coeff"])
check("a NaN coefficient fails the executed gate",
      FL.filter_failure(FL.filter_report(nan_exec)) is not None)
check("a NaN coefficient fails arm validation",
      TC.validate(TC.GEN, nan_tree, [nan_exec]) is not None)
check("a NaN tree is not finite", not finite_tree(nan_tree))

print()
if fails:
    print("FLOAT32 PROBE FAILURES:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(4)
print("FLOAT32 PROBE OK")
