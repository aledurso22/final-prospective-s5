"""Production float32 probe for the TSS containment study. x64 OFF.

The focused-check module runs in float64, while the study refuses to start
unless x64 is disabled. This probe exercises the NEW paths in the production
precision: the five-carry processing rollout and its streaming, the two exact
points, the executed filter gate (including a point the declared numerical
gaps accept but the rounded polynomial must refuse), the feasibility repair
and the masked optimizer path of both processing arms.

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

print("--- 3. the executed filter gate in float32 ---")
for M, gam, T, want in ((0.0, 0.0, H, True), (0.0, H, 0.0, True),
                        (0.4, 0.2, 1.1, True), (1.0, 0.0, 0.0, False),
                        (1e18, 0.0, FL.G_MIN, False),
                        (3.4e38, 0.0, 0.6, False)):
    rep = FL.filter_report(tree(M, gam, T))
    ok = FL.filter_failure(rep) is None
    print(f"  (M={M:g}, gamma={gam:g}, T={T:g}) c1={rep['executed_c1']:.9g} "
          f"c0={rep['executed_c0']:.9g} A={rep['A']:.9g} "
          f"{rep['classification']} accepted={ok} gaps "
          f"{rep['gap_gamma_plus_T']:.3e}/{rep['gap_filter']:.3e}")
    check(f"executed gate verdict at (M={M:g}, gamma={gam:g}, T={T:g})",
          ok == want)
check("the declared gaps alone would have accepted the refused large mass",
      FL.filter_report(tree(1e18, 0.0, FL.G_MIN))["gap_gamma_plus_T"] >= 0.0)

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
            and FL.filter_failure(FL.filter_report(once)) is None):
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
      FL.filter_failure(FL.filter_report(out)) is None
      and float(out["fil_gamma"][0] + out["fil_T"][0]) > 0.0)

print("--- 5. coefficient sensitivity against an independent float64 "
      "reference ---")


def independent_sensitivity(p, ep_, direction):
    """Independently coded sequential float64 forward sensitivity; only the
    coefficient-independent inputs (preprocessing, gates, readout) are
    shared."""
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
    return float(ce.sum() / n), float(dce.sum() / n)


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


e1 = ep(9751)
for point, direction, name in (
        ((0.0, 0.0, H), (1.0, 0.0, 0.0), "M inward at the TSS boundary"),
        ((0.0, 0.0, H), (0.0, 1.0, 0.0), "gamma inward at the TSS boundary"),
        ((0.0, 0.0, H), (0.0, 0.0, 1.0), "T on the TSS boundary"),
        ((0.4, 0.3, 1.5), (0.0, 0.0, 1.0), "T at an interior point")):
    p = tree(*point)
    fval64, ref = independent_sensitivity(p, e1, direction)
    f = loss_of(p, e1)
    val, jvp = jax.jvp(f, (jnp.asarray(point, jnp.float32),),
                       (jnp.asarray(direction, jnp.float32),))
    if not all_finite(val, jvp):
        fails.append(f"{name}: non-finite float32 loss or JVP")
        continue
    if abs(ref) <= 1e-12:
        fails.append(f"FIXTURE DEFECT: degenerate analytic sensitivity for "
                     f"{name} ({ref})")
        continue
    err = abs(float(jvp) - ref) / max(abs(ref), 1e-30)
    hstep = 1e-3
    fwd = (float(f(jnp.asarray(point, jnp.float32)
                   + hstep * jnp.asarray(direction, jnp.float32)))
           - float(val)) / hstep
    resolvable = abs(float(jvp)) >= 100 * eps32 * max(abs(float(val)),
                                                      1.0) / hstep
    print(f"  {name}: float32 jvp {float(jvp):.6e} independent float64 "
          f"{ref:.6e} rel {err:.2e} (REF32 {REF32}) inward FD {fwd:.6e} "
          f"resolvable={bool(resolvable)}")
    if err > REF32:
        fails.append(f"{name}: derivative vs float64 reference {err:.2e}")
    if not resolvable:
        print("  LIMITATION: finite but below float32 forward-difference "
              "resolvability; the independent float64 reference above is the "
              "evidence, not a dead parameter. Only INWARD points were "
              "evaluated: negative mass or damping is not an admissible "
              "model.")

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
    check(f"{arm}: the executed filter gate accepted the update",
          bool(tel["guard_ok"]))
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
    bad = TC.validate(arm, p2)
    check(f"{arm}: the updated tree validates ({bad})", bad is None)
    print(f"  {arm}: loss {float(o[2]):.6f} acc {float(o[3]):.4f} grads "
          f"{ {k: float(v) for k, v in grads.items()} } repaired "
          f"{int(tel['n_repaired'])} jury_min "
          f"{float(tel['guard_jury_min']):.3e}")

print("--- 7. a NaN update is refused, not counted as movement ---")
nan_tree = tree(float("nan"), 0.0, H)
check("a NaN coefficient fails the executed gate",
      FL.filter_failure(FL.filter_report(nan_tree)) is not None)
check("a NaN coefficient fails arm validation",
      TC.validate(TC.GEN, nan_tree) is not None)
check("a NaN tree is not finite", not finite_tree(nan_tree))

print()
if fails:
    print("FLOAT32 PROBE FAILURES:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(4)
print("FLOAT32 PROBE OK")
