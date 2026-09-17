"""Focused checks for the TSS containment study. CLUSTER-ONLY.

Scope: the new residual-processing law, its two exact points, the executed
filter gate, the feasibility repair, the coefficient gradients and the frozen
selection/deployment logic. The completed studies' suites are not re-run.

PREDECLARED TOLERANCES, unchanged from the completed studies:
    ID64    1e-9   relative: float64 identities between recurrences
    GRAD64  1e-8   relative per leaf, absolute floor 1e3 eps64 G
    SENS64  1e-6   relative: coefficient derivative vs the INDEPENDENT
                   sequential float64 sensitivity
    EXACT64 1e-12  relative: executed transition versus its analytic form

Comparison policy (review R3): bitwise equality is asserted only for stored
leaves that must not change and for genuinely shared executed operations.
Everything computed by two separately compiled recurrences is compared
finite-first at the tolerances above. Derivatives are NOT required to be
nonzero: they must AGREE with the independent sensitivity, including
legitimate zeros, and a fixture that is analytically degenerate is reported
as a fixture defect.

PM_SOURCE_RUN must point at the completed replication run (read-only); a
missing source is a FAILURE, never a skip.
"""

import math
import os
import sys

import jax
import numpy as onp
import pytest
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optax                                                       # noqa: E402
from experiments.nested_memory import dynamics as NMD             # noqa: E402
from experiments.nested_memory import model as NM                 # noqa: E402
from experiments.nested_memory import task as TK                  # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import filtered as FL       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import ordinary as OD       # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import study as ST          # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

ID64, GRAD64, SENS64, EXACT64 = 1e-9, 1e-8, 1e-6, 1e-12
F64 = onp.float64
EPS64 = float(onp.finfo(F64).eps)
H = FL.H


def _finite_tree(*trees):
    return all(bool(onp.all(onp.isfinite(onp.asarray(v, F64))))
               for t in trees
               for v in jax.tree_util.tree_leaves(t)
               if onp.issubdtype(onp.asarray(v).dtype, onp.inexact))


def _moved_finite(after, before):
    a, b = float(onp.asarray(after)), float(onp.asarray(before))
    return bool(onp.isfinite(a) and onp.isfinite(b) and a != b)


def _rel(a, b):
    a, b = onp.asarray(a, F64), onp.asarray(b, F64)
    if not (onp.all(onp.isfinite(a)) and onp.all(onp.isfinite(b))):
        return float("inf")
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def _close(label, got, ref, rel_tol, floor=1e3 * EPS64):
    g, r = float(onp.asarray(got)), float(onp.asarray(ref))
    assert onp.isfinite(g) and onp.isfinite(r), (label, g, r)
    a = abs(g - r)
    print(f"  {label}: {g:.12g} vs {r:.12g} abs {a:.3e} rel "
          f"{a / max(abs(r), 1e-30):.3e} (tolerance {rel_tol})")
    assert a <= max(rel_tol * abs(r), floor * max(abs(r), 1.0)), (label, a)


def _f64(p):
    return {k: jnp.asarray(onp.asarray(v), dtype=jnp.float64)
            for k, v in p.items()}


def _source_native_f64():
    run = os.environ.get("PM_SOURCE_RUN", TC.SOURCE_RUN)
    assert os.path.isdir(RS.sources_dir(run)), \
        f"read-only source run {run} is REQUIRED"
    import json
    with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
        manifest = json.load(fh)
    e = RS.find_entry(manifest, TC.SOURCE_DEV, TC.SOURCE_FAMILY)
    return _f64(RS.restore_source(run, e))


def _ep(seed, i=0):
    b = TK.generate_batch(seed, 2)
    return {k: jnp.asarray(b[k][i]) for k in ("key_id", "val_id", "event",
                                              "label")}


def _tree(pn, M, gam, T):
    dt = pn["A_log"].dtype
    return dict(pn, fil_M=jnp.full((1,), M, dtype=dt),
                fil_gamma=jnp.full((1,), gam, dtype=dt),
                fil_T=jnp.full((1,), T, dtype=dt))


def _schedule(rs, n, dv=3, dk=4):
    keys = rs.randn(n, dk)
    keys /= onp.linalg.norm(keys, axis=1)[:, None]
    return dict(k=keys, v=rs.randn(n, dv),
                m=(rs.rand(n) < 0.6).astype(float),
                alpha=rs.uniform(0.2, 1.0, n), beta=rs.uniform(0.1, 0.9, n),
                eta=rs.uniform(0.8, 1.9, n), mu=rs.uniform(0.3, 0.9, n))


def _run_filtered(sched, M, gam, T, dv=3, dk=4):
    """The PRODUCTION step over a schedule, returning W, U, y and the masked
    residuals it actually formed (closed loop: R depends on W)."""
    A = M + H * (gam + T)
    coeff = tuple(jnp.asarray(x, jnp.float64) for x in (M, gam, T, A))
    carry = tuple(jnp.zeros((dv, dk), jnp.float64) for _ in range(5))
    W, U, Y, R = [], [], [], []
    for t in range(len(sched["k"])):
        carry = FL.filtered_step(
            carry, jnp.asarray(sched["k"][t]), jnp.asarray(sched["v"][t]),
            jnp.asarray(sched["m"][t]), sched["alpha"][t], sched["beta"][t],
            sched["mu"][t], sched["eta"][t], coeff)
        W.append(onp.asarray(carry[0])); U.append(onp.asarray(carry[1]))
        Y.append(onp.asarray(carry[2])); R.append(onp.asarray(carry[4]))
    return onp.array(W), onp.array(U), onp.array(Y), onp.array(R), carry


def _run_other(step, sched, scalar, n_carry, dv=3, dk=4):
    carry = tuple(jnp.zeros((dv, dk), jnp.float64) for _ in range(n_carry))
    W, U = [], []
    for t in range(len(sched["k"])):
        carry = step(carry, jnp.asarray(sched["k"][t]),
                     jnp.asarray(sched["v"][t]), jnp.asarray(sched["m"][t]),
                     sched["alpha"][t], sched["beta"][t], sched["mu"][t],
                     sched["eta"][t], scalar)
        W.append(onp.asarray(carry[0])); U.append(onp.asarray(carry[1]))
    return onp.array(W), onp.array(U), carry


def _native_run(sched, dv=3, dk=4):
    carry = (jnp.zeros((dv, dk), jnp.float64), jnp.zeros((dv, dk),
                                                         jnp.float64))
    W, U = [], []
    for t in range(len(sched["k"])):
        carry = NMD.momentum_delta_step(
            carry, jnp.asarray(sched["k"][t]), jnp.asarray(sched["v"][t]),
            jnp.asarray(sched["m"][t]), sched["alpha"][t], sched["beta"][t],
            sched["mu"][t], sched["eta"][t])
        W.append(onp.asarray(carry[0])); U.append(onp.asarray(carry[1]))
    return onp.array(W), onp.array(U)


# ======================= 1. the exact points of the placement ===============
@pytest.mark.parametrize("T", [0.6, 1.0, 4.0])
def test_boundary_is_literal_tss_processing(T):
    """M = gamma = 0: the executed processing state equals literal TSS Eq.
    (17) driven by the SAME closed-loop masked residuals, at the same clock,
    initialization and same-token output convention."""
    rs = onp.random.RandomState(1)
    s = _schedule(rs, 40)
    _, _, Y, R, _ = _run_filtered(s, 0.0, 0.0, T)
    ref = OD.tss_processing_reference(list(R), tau=T, h=H)
    assert _rel(Y, onp.array(ref)) < ID64, T


def test_native_point_is_exactly_native_momentum():
    """(M, gamma, T) = (0, h, 0): y_next = R_t, so the Momentum update below
    is the native one. Separately compiled recurrences, so this is the
    declared float64 identity tolerance, never bitwise."""
    rs = onp.random.RandomState(2)
    s = _schedule(rs, 45)
    M, gam, T = FL.native_point()
    Wf, Uf, Y, R, _ = _run_filtered(s, M, gam, T)
    Wn, Un = _native_run(s)
    assert _rel(Wf, Wn) < ID64
    assert _rel(Uf, Un) < ID64
    assert _rel(Y, R) < ID64                   # y_next = R_t


@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_two_tap_mapping_inside_the_family(kappa):
    """M = 0, gamma + T = h reproduces the two-tap operator with kappa = T/h,
    for kappa <= 1 (gamma >= 0)."""
    rs = onp.random.RandomState(3)
    s = _schedule(rs, 35)
    M, gam, T = FL.two_tap_mapping(kappa)
    assert gam >= 0.0 and FL.mapping_admissible(kappa)
    Wf, Uf, _, _, _ = _run_filtered(s, M, gam, T)
    Wo, Uo, _ = _run_other(OD.ordinary_step, s, kappa, 3)
    assert _rel(Wf, Wo) < ID64, kappa
    assert _rel(Uf, Uo) < ID64, kappa


@pytest.mark.parametrize("kappa", [1.87, 2.14, 2.61])
def test_learned_operator_horizons_are_outside_the_family(kappa):
    """The completed study learned kappa in 1.87-2.61; those map to negative
    gamma and are REFUSED, not silently accepted: the operator stays a
    separate comparator, outside the containment claim."""
    M, gam, T = FL.two_tap_mapping(kappa)
    assert gam < 0.0 and not FL.mapping_admissible(kappa)
    pn = _source_native_f64()
    bad = _tree(pn, M, gam, T)
    assert FL.filter_failure(FL.filter_report(bad)) is not None
    repaired, tel = FL.repair(bad)
    assert float(repaired["fil_gamma"][0]) == 0.0       # clamped, not learned
    assert int(tel["n_repaired"]) >= 1
    # the repaired point is NOT the two-tap operator any more
    rs = onp.random.RandomState(4)
    s = _schedule(rs, 20)
    Wf, _, _, _, _ = _run_filtered(
        s, *(float(repaired[k][0]) for k in FL.LEAVES))
    Wo, _, _ = _run_other(OD.ordinary_step, s, kappa, 3)
    assert _rel(Wf, Wo) > 1e-3


def test_T_equals_h_is_both_tss_and_the_kappa_one_operator():
    rs = onp.random.RandomState(5)
    s = _schedule(rs, 25)
    _, _, Y, R, _ = _run_filtered(s, 0.0, 0.0, H)
    prev = onp.zeros_like(R[0])
    for t in range(len(R)):
        assert _rel(Y[t], 2.0 * R[t] - prev) < ID64, t
        prev = R[t]
    assert _rel(Y, onp.array(OD.tss_processing_reference(list(R), tau=H,
                                                         h=H))) < ID64


def test_episode_start_values():
    """y = y_prev = R_prev = 0 at episode start, so the first token gives
    (h^2 + hT)/A R_0, i.e. (1 + h/T) R_0 on the boundary and 2 R_0 at T = h."""
    rs = onp.random.RandomState(6)
    s = _schedule(rs, 3)
    s["m"][0] = 1.0
    for M, gam, T in ((0.0, 0.0, H), (0.0, 0.0, 4.0), (0.3, 0.2, 2.0)):
        _, _, Y, R, _ = _run_filtered(s, M, gam, T)
        A = M + H * (gam + T)
        assert _rel(Y[0], (H * H + H * T) / A * R[0]) < ID64, (M, gam, T)
    _, _, Y, R, _ = _run_filtered(s, 0.0, 0.0, H)
    assert _rel(Y[0], 2.0 * R[0]) < ID64


# ======================= 2. rollout, streaming and counts ===================
def test_rollout_streaming_and_carry_counts():
    pn = _source_native_f64()
    p = PM.add_extension(pn, FL.FILTERED)
    ep = _ep(9700)
    full = PM.rollout(FL.FILTERED, p, ep)
    a = PM.rollout(FL.FILTERED, p, {k: v[:29] for k, v in ep.items()})
    b = PM.rollout(FL.FILTERED, p, {k: v[29:] for k, v in ep.items()},
                   carry0=a["final_carry"])
    assert _rel(jnp.concatenate([a["logits"], b["logits"]]),
                full["logits"]) < ID64
    for x, y in zip(b["final_carry"], full["final_carry"]):
        assert _rel(x, y) < ID64
    assert len(full["final_carry"]) == 5
    assert FL.CARRY_EXECUTED == 320 == PD.CARRY[FL.FILTERED]
    assert FL.CARRY_MINIMAL_M_ZERO == 256
    counts = TC.parameter_counts(TC.GEN, p)
    assert counts["stored"] == 572 and counts["trainable"] == 572
    tss = TC.parameter_counts(TC.TSS, p)
    assert tss["stored"] == 572 and tss["trainable"] == 570
    assert tss["frozen_constants"] == 2
    assert counts["carry_real_numbers_executed"] == 320
    assert counts["carry_real_numbers_minimal_if_M_zero"] == 256


def test_declared_start_is_literal_tss_not_native():
    pn = _source_native_f64()
    p = PM.add_extension(pn, FL.FILTERED)
    assert float(p["fil_M"][0]) == 0.0 and float(p["fil_gamma"][0]) == 0.0
    assert float(p["fil_T"][0]) == FL.T0 == H
    ep = _ep(9701)
    start = PM.rollout(FL.FILTERED, p, ep)["logits"]
    native = PM.rollout("momentum_delta", pn, ep)["logits"]
    assert _rel(start, native) > 1e-4, ("the TSS start must NOT be the native "
                                        "function")
    nat_pt = TC.native_point_tree(pn)
    assert _rel(PM.rollout(FL.FILTERED, nat_pt, ep)["logits"], native) < ID64


def test_recovery_tolerance_policy_is_finite_first_and_bounded():
    def metrics(acc, ce):
        cat = {c: dict(accuracy=acc, cross_entropy=ce, n=100)
               for c in TK.CATEGORIES}
        return {f: dict(by_category=cat) for f in TK.FAMILIES}
    base = metrics(0.5, 1.0)
    assert TC.recovery_differences(base, metrics(0.5, 1.0))[0] == []
    ok = metrics(0.5 + TC.RECOVERY_QUERY_TOL / 100.0,
                 1.0 * (1 + TC.RECOVERY_CE_REL / 2))
    assert TC.recovery_differences(base, ok)[0] == []
    bad = metrics(0.5 + 5.0 / 100.0, 1.0)
    assert TC.recovery_differences(base, bad)[0]
    nan = metrics(float("nan"), 1.0)
    assert TC.recovery_differences(base, nan)[0]


# ======================= 3. the executed filter gate ========================
@pytest.mark.parametrize("M,gam,T", [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                                     (0.5, 0.3, 2.0), (2.0, 0.0, 0.7),
                                     (0.0, 0.25, 0.75)])
def test_executed_transition_is_the_declared_polynomial(M, gam, T):
    pn = _source_native_f64()
    rep = FL.filter_report(_tree(pn, M, gam, T))
    A = M + H * (gam + T)
    assert abs(rep["A"] - A) <= EXACT64 * max(1.0, A)
    _close("c1", rep["executed_c1"], 1.0 + (M - H * H) / A, EXACT64)
    _close("c0", rep["executed_c0"], M / A, EXACT64)
    strict = (A > 0 and gam + T > 0
              and 4 * M + 2 * H * (gam + T) > H * H)
    assert (rep["classification"] == "stable") == strict, rep


def test_missing_damping_counterexample_is_refused_and_repaired():
    """h = 1, M = 1, gamma = T = 0 passes A > 0 and 4M + 2h(gamma+T) > h^2,
    but gamma + T = 0 gives z^2 - z + 1: roots on the unit circle."""
    pn = _source_native_f64()
    bad = _tree(pn, 1.0, 0.0, 0.0)
    rep = FL.filter_report(bad)
    assert rep["A"] > 0 and 4 * rep["M"] + 2 * H * (rep["gamma"] + rep["T"]) \
        > H * H
    assert rep["classification"] != "stable"
    assert FL.filter_failure(rep) is not None
    fixed, tel = FL.repair(bad)
    assert float(fixed["fil_gamma"][0] + fixed["fil_T"][0]) > 0.0
    assert float(tel["gap_gamma_plus_T"]) >= 0.0
    good = FL.filter_report(fixed)
    assert good["classification"] == "stable", good
    assert FL.filter_failure(good) is None


def test_repair_clamps_restores_and_is_idempotent():
    pn = _source_native_f64()
    rs = onp.random.RandomState(7)
    for _ in range(12):
        p = _tree(pn, rs.uniform(-1, 2), rs.uniform(-1, 1),
                  rs.uniform(-1, 3))
        once, tel = FL.repair(p)
        twice, tel2 = FL.repair(once)
        for k in FL.LEAVES:
            assert float(once[k][0]) >= 0.0
            assert onp.array_equal(onp.asarray(once[k]),
                                   onp.asarray(twice[k]))
        assert int(tel2["n_repaired"]) == 0
        assert float(tel["gap_gamma_plus_T"]) >= 0.0
        assert float(tel["gap_filter"]) >= 0.0
        assert FL.filter_failure(FL.filter_report(once)) is None


def test_repair_fixes_the_two_exact_points_and_respects_frozen_leaves():
    pn = _source_native_f64()
    for M, gam, T in (FL.tss_boundary(), FL.native_point()):
        p = _tree(pn, M, gam, T)
        out, tel = FL.repair(p)
        for k in FL.LEAVES:
            assert onp.array_equal(onp.asarray(out[k]), onp.asarray(p[k])), k
        assert int(tel["n_repaired"]) == 0
    # a frozen leaf is restored bitwise even when the proposal is negative
    p = _tree(pn, -0.5, -0.25, 0.9)
    out, _ = FL.repair(p, frozen=("fil_M", "fil_gamma"))
    assert float(out["fil_M"][0]) == -0.5
    assert float(out["fil_gamma"][0]) == -0.25


def test_gate_refuses_a_mass_that_rounds_the_filter_onto_the_circle():
    """Clearance s1: an unbounded M makes A round to M and M/A round to one.
    The executed polynomial is then refused; the declared gaps alone would
    not have caught it."""
    pn = _source_native_f64()
    huge = _tree(pn, 1e18, 0.0, FL.G_MIN)
    rep = FL.filter_report(huge)
    assert rep["gap_gamma_plus_T"] >= 0.0        # the declared gap holds
    assert rep["executed_c0"] == 1.0             # but the rounded det is one
    assert rep["classification"] != "stable"
    assert FL.filter_failure(rep) is not None


def test_in_loop_guard_reads_the_same_executed_coefficients():
    """Genuinely shared executed operation: the guard and the report call the
    same `executed_filter_transition` on the same coefficients, so equality
    here is exact by construction (not a cross-compilation claim)."""
    pn = _source_native_f64()
    for M, gam, T in ((0.0, 0.0, 1.0), (0.4, 0.1, 1.2), (1.0, 0.0, 0.0)):
        p = _tree(pn, M, gam, T)
        g = FL.in_loop_guard(FL.coefficients(p), p["fil_M"].dtype)
        rep = FL.filter_report(p)
        assert float(g["c1"]) == rep["executed_c1"]
        assert float(g["c0"]) == rep["executed_c0"]
        assert bool(g["ok"]) == (FL.filter_failure(rep) is None)


def test_validate_refuses_an_unstable_arm_and_a_moved_stored_constant():
    pn = _source_native_f64()
    assert TC.validate(TC.GEN, _tree(pn, 0.0, 0.0, 1.0)) is None
    assert TC.validate(TC.TSS, _tree(pn, 0.0, 0.0, 1.0)) is None
    assert TC.validate(TC.GEN, _tree(pn, 1.0, 0.0, 0.0)) is not None
    moved = TC.validate(TC.TSS, _tree(pn, 0.2, 0.0, 1.0))
    assert moved is not None and "boundary" in moved


# ======================= 4. gradients against an independent reference ======
def _independent_sensitivity(p, ep, direction):
    """INDEPENDENTLY CODED sequential float64 forward sensitivity of the query
    cross-entropy with respect to (M, gamma, T) in the given direction. It
    reimplements the rollout's recurrence and its tangent; only the
    coefficient-independent inputs (preprocessing, gates, readout) are shared.
    `direction` is (dM, dgamma, dT)."""
    from experiments.adaptive_memory import model as AM
    e = AM.sanitize_episode(ep)
    key_id, val_id, event = e["key_id"], e["val_id"], e["event"]
    k_all, k_valid = NMD.safe_normalize(p["key_raw"])
    keys = onp.asarray(k_all[key_id], F64)
    has_v = (val_id >= 0).astype(p["key_raw"].dtype)
    vals = onp.asarray(p["value_table"][jnp.maximum(val_id, 0)]
                       * has_v[:, None], F64)
    mask = onp.asarray((event == TK.WRITE).astype(p["key_raw"].dtype)
                       * k_valid[key_id], F64)
    gx = NM.gate_features(key_id, val_id, event).astype(p["key_raw"].dtype)
    al, be, mu, eta = (onp.asarray(x, F64) for x in NM._momentum_gates(p, gx))
    Hm = onp.asarray(p["readout_W"], F64)
    bias = onp.asarray(p["readout_b"], F64)
    lab = onp.maximum(onp.asarray(e["label"]), 0)
    qq = (onp.asarray(e["event"]) == TK.QUERY).astype(F64)
    M, gam, T = (float(onp.asarray(p[k]).ravel()[0]) for k in FL.LEAVES)
    dM, dgam, dT = direction
    h = H
    A = M + h * (gam + T)
    dA = dM + h * (dgam + dT)
    d_v, d_k = vals.shape[1], keys.shape[1]
    Z = lambda: onp.zeros((d_v, d_k))                          # noqa: E731
    W, U, y, y_prev, Rp = Z(), Z(), Z(), Z(), Z()
    dW, dU, dy, dy_prev, dRp = Z(), Z(), Z(), Z(), Z()
    L = keys.shape[0]
    logits = onp.zeros((L, Hm.shape[0]))
    dlogits = onp.zeros_like(logits)
    for t in range(L):
        k, v, m = keys[t], vals[t], mask[t]
        Wb, dWb = al[t] * W, al[t] * dW
        R = m * onp.outer(Wb @ k - v, k)
        dR = m * onp.outer(dWb @ k, k)
        N = M * (y - y_prev) + h * h * (R - y) + h * T * (R - Rp)
        dN = (dM * (y - y_prev) + M * (dy - dy_prev)
              + h * h * (dR - dy) + h * dT * (R - Rp)
              + h * T * (dR - dRp))
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
    dce = -(dlogits[onp.arange(L), lab]
            - (pr * dlogits).sum(axis=1)) * qq
    n = max(qq.sum(), 1.0)
    return float(ce.sum() / n), float(dce.sum() / n)


def _loss_fn(p, ep):
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)

    def f(coeffs):
        pp = dict(p, **{k: jnp.asarray([coeffs[i]], dtype=p[k].dtype)
                        for i, k in enumerate(FL.LEAVES)})
        out = PM.rollout(FL.FILTERED, pp, ep)
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(lab, TK.N_VALUES,
                                          dtype=out["logits"].dtype)) * q
        return jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0)
    return f


@pytest.mark.parametrize("point,direction,name", [
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "M inward at the TSS boundary"),
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "gamma inward at the TSS boundary"),
    ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), "T on the TSS boundary"),
    ((0.4, 0.3, 1.5), (1.0, 0.0, 0.0), "M at an interior point"),
    ((0.4, 0.3, 1.5), (0.0, 1.0, 0.0), "gamma at an interior point"),
    ((0.4, 0.3, 1.5), (0.0, 0.0, 1.0), "T at an interior point"),
])
def test_coefficient_derivatives_match_the_independent_sensitivity(
        point, direction, name):
    """Agreement with the independent sensitivity is the criterion; a zero is
    accepted when the reference is zero too. At the boundary the derivative is
    taken INWARD and analytically: no differentiation through the feasibility
    repair, and no finite-difference point with negative mass or damping."""
    pn = _source_native_f64()
    ep = _ep(9702)
    p = _tree(pn, *point)
    fval, ref = _independent_sensitivity(p, ep, direction)
    coeffs = jnp.asarray(point, jnp.float64)
    f = _loss_fn(p, ep)
    val, jvp = jax.jvp(f, (coeffs,), (jnp.asarray(direction, jnp.float64),))
    assert onp.isfinite(float(val)) and onp.isfinite(float(jvp))
    _close(f"loss ({name})", val, fval, ID64)
    assert abs(ref) > 1e-12, (f"FIXTURE DEFECT: the analytic sensitivity for "
                              f"{name} is degenerate ({ref}); choose another "
                              f"episode, not a looser test")
    _close(f"dL/d({name})", jvp, ref, SENS64)
    # inward forward difference, admissible points only
    hstep = 1e-6
    inward = jnp.asarray(point, jnp.float64) + hstep * jnp.asarray(
        direction, jnp.float64)
    fd = (float(f(inward)) - float(val)) / hstep
    resolvable = abs(jvp) >= 100 * EPS64 * max(abs(float(val)), 1.0) / hstep
    print(f"  {name}: jvp {float(jvp):.6e} reference {ref:.6e} inward FD "
          f"{fd:.6e} resolvable={bool(resolvable)}")
    if resolvable:
        assert abs(fd - float(jvp)) <= 1e-3 * max(abs(float(jvp)), 1e-12)
    else:
        print("  LIMITATION: finite but below float64 forward-difference "
              "resolvability; the independent sensitivity above is the "
              "evidence, not a dead parameter")


def test_backbone_gradients_are_finite_and_flow_with_full_bptt():
    pn = _source_native_f64()
    ep = _ep(9703)
    p = PM.add_extension(pn, FL.FILTERED)

    def loss(pp):
        q = (ep["event"] == TK.QUERY)
        lab = jnp.maximum(ep["label"], 0)
        out = PM.rollout(FL.FILTERED, pp, ep)
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(lab, TK.N_VALUES,
                                          dtype=out["logits"].dtype)) * q
        return jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0)
    g = jax.grad(loss)(p)
    assert _finite_tree(g)
    G = math.sqrt(sum(float(onp.sum(onp.asarray(v, F64) ** 2))
                      for v in g.values()))
    assert onp.isfinite(G) and G > 0.0
    for k in ("key_raw", "value_table", "readout_W", "a_proj", "e_proj"):
        assert onp.all(onp.isfinite(onp.asarray(g[k], F64)))


# ======================= 5. the training step and its freezing ==============
def test_train_step_freezes_stored_constants_and_moves_the_rest():
    pn = _source_native_f64()
    eps = {k: jnp.asarray(v) for k, v in TK.generate_batch(9704, 2).items()
           if k in ("key_id", "val_id", "event", "label")}
    lr = jnp.asarray(0.01, dtype=jnp.float64)
    for arm in (TC.TSS, TC.GEN):
        p = _tree(pn, 0.2 if arm == TC.GEN else 0.0,
                  0.1 if arm == TC.GEN else 0.0, 1.3)
        opt = ST.TX.init(p)
        o = TC.train_step_filtered(FL.FILTERED, TC.FROZEN_LEAVES[arm], p, opt,
                                   eps, lr)
        p2, opt2, tel, grads = o[0], o[1], o[8], o[9]
        assert _finite_tree(p2, opt2), arm
        for x in list(o[2:8]):
            assert onp.isfinite(float(x)), arm
        assert bool(tel["guard_ok"]), arm
        assert TC.frozen_leaf_differences(p2, p, arm) == {}
        if arm == TC.TSS:
            for k in ("fil_M", "fil_gamma"):
                assert onp.array_equal(onp.asarray(p2[k]), onp.asarray(p[k]))
            assert onp.isfinite(float(grads["fil_T"]))
            assert _moved_finite(p2["fil_T"][0], p["fil_T"][0])
        else:
            assert any(_moved_finite(p2[k][0], p[k][0]) for k in FL.LEAVES)
        assert _moved_finite(onp.asarray(p2["a_proj"]).ravel()[0],
                             onp.asarray(p["a_proj"]).ravel()[0])
        mask = TC.leaf_mask(p, TC.FROZEN_LEAVES[arm])
        assert all(float(mask[k]) == 0.0 for k in TC.FROZEN_LEAVES[arm])
        assert all(float(mask[k]) == 1.0 for k in p
                   if k not in TC.FROZEN_LEAVES[arm])


def test_nan_updates_are_rejected_not_counted_as_movement():
    assert not _moved_finite(float("nan"), 0.3)
    assert not _moved_finite(0.3, 0.3)
    assert _moved_finite(0.4, 0.3)
    assert not _finite_tree({"a": jnp.asarray([jnp.nan])})
    pn = _source_native_f64()
    nan_tree = _tree(pn, float("nan"), 0.0, 1.0)
    assert FL.filter_failure(FL.filter_report(nan_tree)) is not None
    assert TC.validate(TC.GEN, nan_tree) is not None


# ======================= 6. protocol wiring: work, streams, selection =======
def test_streams_are_fresh_and_work_fits_the_declared_budget():
    assert TC.stream_overlaps() == []
    w = TC.planned_work()
    assert w["trained_runs_development"] == 8
    assert w["trained_runs_final"] == 12
    assert w["max_total_updates"] == 4000
    assert w["development_checkpoints_per_family"] == 9
    assert TC.VAL_AT == (0, 25, 50, 100, 200)
    assert TC.UPDATES == 200 and dict(TC.LRS) == {"A": 0.003, "B": 0.01}
    assert TC.LAW_OF[TC.TSS] == TC.LAW_OF[TC.GEN] == FL.FILTERED
    assert TC.FROZEN_LEAVES[TC.TSS] == ("fil_M", "fil_gamma")
    assert TC.FROZEN_LEAVES[TC.GEN] == ()
    assert TC.LAW_OF[TC.OPERATOR] == OD.ORDINARY


def _dev_row(arm, config, lr, vals):
    return dict(rule=arm, config=config, lr=lr,
                validation=[dict(update=u, primary=p, revision_ce=ce,
                                 retention=rt, recall=rc)
                            for u, p, ce, rt, rc in vals])


def _dev_rows(spec):
    rows = []
    for arm, per_config in spec.items():
        for config, lr, vals in per_config:
            rows.append(_dev_row(arm, config, lr, vals))
    return rows


def _uniform(primary, ce=1.0, ret=0.5, rec=0.5):
    return [(u, primary, ce, ret, rec) for u in TC.VAL_AT]


def test_selection_is_native_first_then_matched_retention():
    status = {}
    spec = {
        TC.NATIVE: [("A", 0.003, _uniform(0.50, ret=0.60, rec=0.70)),
                    ("B", 0.01, _uniform(0.55, ret=0.58, rec=0.70))],
        # feasible only at update 50, and worse elsewhere
        TC.GEN: [("A", 0.003, [(0, 0.40, 1.0, 0.58, 0.70),
                               (25, 0.60, 1.0, 0.50, 0.70),
                               (50, 0.58, 1.0, 0.60, 0.72),
                               (100, 0.70, 1.0, 0.40, 0.70),
                               (200, 0.72, 1.0, 0.30, 0.70)]),
                 ("B", 0.01, _uniform(0.45, ret=0.40, rec=0.40))],
        TC.TSS: [("A", 0.003, _uniform(0.52, ret=0.62, rec=0.72)),
                 ("B", 0.01, _uniform(0.51, ret=0.62, rec=0.72))],
        TC.OPERATOR: [("A", 0.003, _uniform(0.80, ret=0.10, rec=0.10)),
                      ("B", 0.01, _uniform(0.79, ret=0.10, rec=0.10))],
    }
    sel, plan = TC.select(_dev_rows(spec), status)
    assert sel is not None
    # native: highest primary regardless of retention
    assert sel[TC.NATIVE]["primary"] == 0.55 and sel[TC.NATIVE]["lr"] == 0.01
    assert status["selection"]["r_native"] == 0.58
    assert status["selection"]["c_native"] == 0.70
    # generalized: only the update-50 checkpoint is feasible
    assert sel[TC.GEN]["feasible"] and not sel[TC.GEN]["diagnostic"]
    assert sel[TC.GEN]["update"] == 50 and sel[TC.GEN]["primary"] == 0.58
    # literal TSS: feasible, ties broken by fewer updates then lower lr
    assert sel[TC.TSS]["feasible"] and sel[TC.TSS]["update"] == 0
    assert sel[TC.TSS]["lr"] == 0.003
    # the operator has no feasible checkpoint: flagged diagnostic, NOT dropped
    assert not sel[TC.OPERATOR]["feasible"]
    assert sel[TC.OPERATOR]["diagnostic"]
    assert sel[TC.OPERATOR]["primary"] == 0.80


def test_update_zero_checkpoint_is_deduplicated():
    spec = {TC.NATIVE: [("A", 0.003, _uniform(0.5)),
                        ("B", 0.01, _uniform(0.5))]}
    rows = _dev_rows(spec)
    cps = TC.checkpoints(rows, TC.NATIVE)
    assert len(cps) == 9
    assert sum(1 for c in cps if c["update"] == 0) == 1
    assert all("heldout" not in c for c in cps)


def test_selection_refuses_non_finite_checkpoints():
    spec = {
        # the NaN sits at update 25, i.e. NOT on the deduplicated
        # update-zero checkpoint, so it really reaches the ranking
        TC.NATIVE: [("A", 0.003, _uniform(0.5)),
                    ("B", 0.01, [(u, (float("nan") if u == 25 else 0.5),
                                  1.0, 0.5, 0.5) for u in TC.VAL_AT])],
        TC.GEN: [("A", 0.003, _uniform(0.5)), ("B", 0.01, _uniform(0.5))],
        TC.TSS: [("A", 0.003, _uniform(0.5)), ("B", 0.01, _uniform(0.5))],
        TC.OPERATOR: [("A", 0.003, _uniform(0.5)),
                      ("B", 0.01, _uniform(0.5))],
    }
    sel, plan = TC.select(_dev_rows(spec), {})
    assert sel is None and plan is None


def test_deployment_fallbacks_are_explicit_and_never_relabelled():
    # TSS infeasible: the generalized arm gets NO TSS fallback, and a native
    # choice is never wrapped in an M = gamma = 0 map
    sel = {TC.NATIVE: dict(primary=0.60, feasible=True, diagnostic=False),
           TC.TSS: dict(primary=0.75, feasible=False, diagnostic=True),
           TC.GEN: dict(primary=0.58, feasible=True, diagnostic=False),
           TC.OPERATOR: dict(primary=0.90, feasible=False, diagnostic=True)}
    status = {}
    plan = TC.deployment_plan(sel, status)
    assert [f["kind"] for f in plan[TC.GEN]["fallbacks"]] == ["native"]
    # trained generalized (0.58) does NOT beat the native fallback (0.60)
    assert plan[TC.GEN]["choice"] == "native"
    assert plan[TC.GEN]["evaluate_arm"] == TC.NATIVE
    # an infeasible extension can never be deployed as itself
    assert plan[TC.OPERATOR]["choice"] == "native"
    assert plan[TC.TSS]["choice"] == "native"
    out = TC.deployment_outcome(
        [dict(rule=TC.NATIVE, heldout=dict(primary=0.6))], plan)
    assert out[TC.GEN]["counted_as_improvement"] is False
    assert "NOT literal TSS" in out[TC.GEN]["label"]
    # now TSS is feasible and better: it becomes an available fallback
    sel2 = dict(sel, **{TC.TSS: dict(primary=0.75, feasible=True,
                                     diagnostic=False)})
    plan2 = TC.deployment_plan(sel2, {})
    assert [f["kind"] for f in plan2[TC.GEN]["fallbacks"]] == ["native", "tss"]
    assert plan2[TC.GEN]["choice"] == "tss"
    assert plan2[TC.GEN]["evaluate_arm"] == TC.TSS
    # a trained endpoint must beat the best fallback STRICTLY
    sel3 = dict(sel2, **{TC.GEN: dict(primary=0.75, feasible=True,
                                      diagnostic=False)})
    assert TC.deployment_plan(sel3, {})[TC.GEN]["choice"] == "tss"
    sel4 = dict(sel2, **{TC.GEN: dict(primary=0.76, feasible=True,
                                      diagnostic=False)})
    assert TC.deployment_plan(sel4, {})[TC.GEN]["choice"] == "trained"


def _final(arm, seed, primary, ret, rec):
    return dict(rule=arm, seed=seed,
                heldout=dict(primary=primary,
                             retention_revision_untouched=ret,
                             recall_overall=rec))


def test_screen_labels_availability_and_requires_no_measured_decrease():
    rows = []
    for i, s in enumerate(TC.SOURCE_FINAL):
        rows += [_final(TC.GEN, s, 0.70 + 0.01 * i, 0.60, 0.70),
                 _final(TC.TSS, s, 0.60 + 0.01 * i, 0.60, 0.70),
                 _final(TC.NATIVE, s, 0.55 + 0.01 * i, 0.605, 0.70),
                 _final(TC.OPERATOR, s, 0.72 + 0.01 * i, 0.50, 0.65),
                 _final(TC.ANCHOR, s, 0.40, 0.62, 0.70)]
    sel = {a: dict(feasible=True, diagnostic=False)
           for a in (TC.NATIVE,) + TC.EXTENSION_ARMS}
    sc = TC.screen(rows, sel)
    by = {c["name"]: c for c in sc["comparisons"]}
    gt = by["extension_versus_literal_tss"]
    assert gt["promising_matched_retention"] is True
    assert gt["constrained_screen_available"] is True
    assert gt["kind"] == "scientific"
    # retention loss against native: the -1 pp safeguard passes but this
    # study's no-measured-decrease condition does not
    gn = by["generalized_versus_native"]
    assert gn["safeguard_passed_minus_one_pp"] is True
    assert gn["no_measured_decrease"] is False
    assert gn["promising_matched_retention"] is False
    # a diagnostic endpoint makes the constrained screen unavailable
    sel2 = dict(sel, **{TC.TSS: dict(feasible=False, diagnostic=True)})
    sc2 = TC.screen(rows, sel2)
    g2 = {c["name"]: c for c in sc2["comparisons"]}[
        "extension_versus_literal_tss"]
    assert g2["constrained_screen_available"] is False
    assert g2["promising_matched_retention"] is False
    assert TC.TSS in g2["constraint_failing_endpoints"]
    assert "INFEASIBLE" in g2["availability_note"]
    assert by["generalized_versus_learned_operator"]["kind"] == "outside_claim"
    assert by["generalized_versus_frozen_source"]["kind"] == "descriptive"


def test_declared_comparisons_and_scope_statements():
    names = [c[0] for c in TC.COMPARISONS]
    assert "extension_versus_literal_tss" in names
    assert "generalized_versus_learned_operator" in names
    assert TC.ABSENT_LITERATURE == ("gated_delta",)
    rep = FL.filter_report(_tree(_source_native_f64(), 0.0, 0.0, 1.0))
    assert "isolated" in rep["note"]
