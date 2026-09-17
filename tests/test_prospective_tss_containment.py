"""Focused checks for the TSS containment study. CLUSTER-ONLY.

Scope: the processing law in its executed coefficient form, its exact points
against separately coded original-law references, the executed-coefficient
gate, the feasibility repair, coefficient gradients against an independent
sensitivity, checkpoint validation and persistence in the actual runner
(including zero-update endpoints and an invalid intermediate checkpoint), and
the frozen selection/deployment logic. The completed studies' suites are not
re-run.

PREDECLARED TOLERANCES, unchanged from the completed studies:
    ID64    1e-9   relative: float64 identities between recurrences
    SENS64  1e-6   relative: coefficient derivative vs the INDEPENDENT
                   sequential float64 sensitivity
    EXACT64 1e-12  relative: coefficients recomputed in another context

Comparison policy (review R3 of bdc1c19 and R2/R5 of 7613c86): bitwise
equality only for stored leaves that must not change, genuinely shared
executed operations, and coefficients whose IEEE arithmetic is exact at the
named points. Everything else is compared FINITE FIRST at the tolerances
above. Derivatives are not required to be nonzero: they must agree with the
independent sensitivity; a degenerate analytic fixture is a fixture defect.

PM_SOURCE_RUN must point at the completed replication run (read-only); a
missing source is a FAILURE, never a skip.
"""

import json
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

ID64, SENS64, EXACT64 = 1e-9, 1e-6, 1e-12
F64 = onp.float64
EPS64 = float(onp.finfo(F64).eps)
H = FL.H


def _all_finite(*xs):
    return all(bool(onp.all(onp.isfinite(onp.asarray(x, F64)))) for x in xs)


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
    with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
        manifest = json.load(fh)
    e = RS.find_entry(manifest, TC.SOURCE_DEV, TC.SOURCE_FAMILY)
    return _f64(RS.restore_source(run, e))


def _ep(seed, i=0):
    b = TK.generate_batch(seed, 2)
    return {k: jnp.asarray(b[k][i]) for k in ("key_id", "val_id", "event",
                                              "label")}


def _leaves(M, gam, T, dtype=jnp.float64):
    return {"fil_M": jnp.full((1,), M, dtype=dtype),
            "fil_gamma": jnp.full((1,), gam, dtype=dtype),
            "fil_T": jnp.full((1,), T, dtype=dtype)}


def _tree(pn, M, gam, T):
    return dict(pn, **_leaves(M, gam, T, pn["A_log"].dtype))


def _executed(p):
    """Coefficients of a tree recomputed in this context (a diagnostic
    stand-in where no compiled program's record is at hand)."""
    return [FL.coefficient_values(FL.coefficients(p))]


def _schedule(rs, n, dv=3, dk=4):
    keys = rs.randn(n, dk)
    keys /= onp.linalg.norm(keys, axis=1)[:, None]
    return dict(k=keys, v=rs.randn(n, dv),
                m=(rs.rand(n) < 0.6).astype(float),
                alpha=rs.uniform(0.2, 1.0, n), beta=rs.uniform(0.1, 0.9, n),
                eta=rs.uniform(0.8, 1.9, n), mu=rs.uniform(0.3, 0.9, n))


def _run_filtered(sched, M, gam, T, dv=3, dk=4):
    """The PRODUCTION coefficient-form step over a schedule, returning W, U,
    y and the masked residuals it actually formed (closed loop)."""
    coeff = FL.coefficients(_leaves(M, gam, T))
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


# ======================= 1. the executed form and the exact points ==========
@pytest.mark.parametrize("M,gam,T", [(0.0, 0.0, 0.6), (0.0, 0.0, 4.0),
                                     (0.5, 0.3, 2.0), (2.0, 0.0, 0.7),
                                     (0.0, 1.0, 0.0), (0.3, 0.25, 0.75)])
def test_coefficient_form_is_the_original_law(M, gam, T):
    """Algebraic equivalence, checked against the SEPARATELY CODED original
    form `FL.filtered_reference` on the same closed-loop residuals."""
    rs = onp.random.RandomState(1)
    s = _schedule(rs, 40)
    _, _, Y, R, _ = _run_filtered(s, M, gam, T)
    ref = FL.filtered_reference(list(R), M, gam, T, h=H)
    assert _rel(Y, onp.array(ref)) < ID64, (M, gam, T)
    A = M + H * (gam + T)
    c = FL.coefficient_values(FL.coefficients(_leaves(M, gam, T)))
    _close("a", c["a"], (2 * M + H * (gam + T) - H * H) / A, EXACT64)
    _close("b", c["b"], M / A, EXACT64)
    _close("c", c["c"], (H * H + H * T) / A, EXACT64)
    _close("d", c["d"], H * T / A, EXACT64)


@pytest.mark.parametrize("T", [0.6, 1.0, 4.0])
def test_boundary_is_literal_tss_processing(T):
    rs = onp.random.RandomState(2)
    s = _schedule(rs, 40)
    _, _, Y, R, _ = _run_filtered(s, 0.0, 0.0, T)
    ref = OD.tss_processing_reference(list(R), tau=T, h=H)
    assert _rel(Y, onp.array(ref)) < ID64, T


def test_exact_coefficients_at_the_two_points():
    """IEEE arithmetic is exact at these points, so bitwise equality is a
    valid expectation here (not a cross-compilation claim)."""
    nat = FL.coefficient_values(FL.coefficients(_leaves(*FL.native_point())))
    assert (nat["a"], nat["b"], nat["c"], nat["d"]) == (0.0, 0.0, 1.0, 0.0)
    tss = FL.coefficient_values(FL.coefficients(_leaves(0.0, 0.0, H)))
    assert (tss["a"], tss["b"], tss["c"], tss["d"]) == (0.0, 0.0, 2.0, 1.0)


def test_native_point_is_native_momentum():
    rs = onp.random.RandomState(3)
    s = _schedule(rs, 45)
    Wf, Uf, Y, R, _ = _run_filtered(s, *FL.native_point())
    Wn, Un = _native_run(s)
    assert _rel(Wf, Wn) < ID64 and _rel(Uf, Un) < ID64
    assert _rel(Y, R) < ID64


@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_two_tap_mapping_inside_the_family(kappa):
    rs = onp.random.RandomState(4)
    s = _schedule(rs, 35)
    M, gam, T = FL.two_tap_mapping(kappa)
    assert gam >= 0.0 and FL.mapping_admissible(kappa)
    Wf, Uf, _, _, _ = _run_filtered(s, M, gam, T)
    Wo, Uo, _ = _run_other(OD.ordinary_step, s, kappa, 3)
    assert _rel(Wf, Wo) < ID64 and _rel(Uf, Uo) < ID64


@pytest.mark.parametrize("kappa", [1.87, 2.14, 2.61])
def test_learned_operator_horizons_are_outside_the_family(kappa):
    M, gam, T = FL.two_tap_mapping(kappa)
    assert gam < 0.0 and not FL.mapping_admissible(kappa)
    pn = _source_native_f64()
    bad = _tree(pn, M, gam, T)
    assert FL.filter_failure(FL.report_from_tree(bad)) is not None
    repaired, tel = FL.repair(bad)
    assert float(repaired["fil_gamma"][0]) == 0.0
    assert int(tel["n_repaired"]) >= 1
    rs = onp.random.RandomState(5)
    s = _schedule(rs, 20)
    Wf, _, _, _, _ = _run_filtered(
        s, *(float(repaired[k][0]) for k in FL.LEAVES))
    Wo, _, _ = _run_other(OD.ordinary_step, s, kappa, 3)
    assert _rel(Wf, Wo) > 1e-3


def test_episode_start_values():
    rs = onp.random.RandomState(6)
    s = _schedule(rs, 3)
    s["m"][0] = 1.0
    for M, gam, T in ((0.0, 0.0, H), (0.0, 0.0, 4.0), (0.3, 0.2, 2.0)):
        _, _, Y, R, _ = _run_filtered(s, M, gam, T)
        A = M + H * (gam + T)
        assert _rel(Y[0], (H * H + H * T) / A * R[0]) < ID64, (M, gam, T)
    _, _, Y, R, _ = _run_filtered(s, 0.0, 0.0, H)
    assert _rel(Y[0], 2.0 * R[0]) < ID64


# ======================= 2. rollout, streaming, counts and loss =============
def test_rollout_streaming_carries_and_counts():
    pn = _source_native_f64()
    p = PM.add_extension(pn, FL.FILTERED)
    ep = _ep(9700)
    full = PM.rollout(FL.FILTERED, p, ep)
    a = PM.rollout(FL.FILTERED, p, {k: v[:29] for k, v in ep.items()})
    b = PM.rollout(FL.FILTERED, p, {k: v[29:] for k, v in ep.items()},
                   carry0=a["final_carry"])
    assert _all_finite(full["logits"], *full["final_carry"],
                       full["proc_max_abs"])
    assert _rel(jnp.concatenate([a["logits"], b["logits"]]),
                full["logits"]) < ID64
    for x, y in zip(b["final_carry"], full["final_carry"]):
        assert _rel(x, y) < ID64
    assert len(full["final_carry"]) == 5
    assert set(full["coeff"]) == set(FL.COEFF_NAMES)
    assert FL.CARRY_EXECUTED == 320 == PD.CARRY[FL.FILTERED]
    counts = TC.parameter_counts(TC.GEN, p)
    assert counts["stored"] == 572 and counts["trainable"] == 572
    assert counts["carry_real_numbers_implemented"] == 320
    tss = TC.parameter_counts(TC.TSS, p)
    assert tss["stored"] == 572 and tss["trainable"] == 570
    assert tss["carry_real_numbers_implemented"] == 320


def test_other_rules_keep_their_rollout_outputs():
    pn = _source_native_f64()
    ep = _ep(9704)
    for law in ("prospective_momentum", OD.ORDINARY):
        out = PM.rollout(law, PM.add_extension(pn, law), ep)
        assert "proc_max_abs" not in out
        assert out["logits"].shape[0] == ep["key_id"].shape[0]


def test_study_loss_is_the_unchanged_loss():
    pn = _source_native_f64()
    p = _tree(pn, 0.3, 0.1, 1.4)
    eps = {k: jnp.asarray(v) for k, v in TK.generate_batch(9705, 2).items()
           if k in ("key_id", "val_id", "event", "label")}
    lf, auxf = TC.batch_loss_f(FL.FILTERED, p, eps)
    ls, _ = ST.batch_loss(FL.FILTERED, p, eps)
    _close("study loss vs the shared loss", lf, ls, ID64)
    assert _all_finite(*auxf["coeff"], auxf["proc_max_abs"])


def test_declared_start_is_literal_tss_not_native():
    pn = _source_native_f64()
    p = PM.add_extension(pn, FL.FILTERED)
    assert float(p["fil_M"][0]) == 0.0 and float(p["fil_gamma"][0]) == 0.0
    assert float(p["fil_T"][0]) == FL.T0 == H
    ep = _ep(9701)
    native = PM.rollout("momentum_delta", pn, ep)["logits"]
    assert _rel(PM.rollout(FL.FILTERED, p, ep)["logits"], native) > 1e-4
    assert _rel(PM.rollout(FL.FILTERED, TC.native_point_tree(pn),
                           ep)["logits"], native) < ID64


def test_count_differences_use_integer_counts():
    def metrics(correct, n, ce):
        cat = {c: dict(accuracy=correct / n, cross_entropy=ce, n=n)
               for c in TK.CATEGORIES}
        return {f: dict(by_category=cat) for f in TK.FAMILIES}
    base = metrics(50, 100, 1.0)
    same = TC.count_differences(base, metrics(50, 100, 1.0))
    assert same["identical_within_declared_identity_tolerance"]
    assert same["decisive"] is False and not same["malformed"]
    # the review's case: 0.5 vs 0.51 at n = 100 is exactly ONE count
    one = TC.count_differences(base, metrics(51, 100, 1.0))
    assert not one["malformed"]
    assert all(v["count_difference"] == 1
               for v in one["per_category"].values())
    assert not one["identical_within_declared_identity_tolerance"]
    two = TC.count_differences(base, metrics(52, 100, 1.0))
    assert all(v["count_difference"] == 2
               for v in two["per_category"].values())
    assert TC.count_differences(base, metrics(50, 100,
                                              float("nan")))["malformed"]
    assert TC.count_differences(base, metrics(50, 99, 1.0))["malformed"]
    broken = metrics(50, 100, 1.0)
    for c in TK.CATEGORIES:
        broken["recall"]["by_category"][c]["accuracy"] = 0.505
    assert TC.count_differences(base, broken)["malformed"]


# ======================= 3. the executed-coefficient gate ===================
@pytest.mark.parametrize("M,gam,T", [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                                     (0.5, 0.3, 2.0), (2.0, 0.0, 0.7),
                                     (0.0, 0.25, 0.75), (1.0, 0.0, 0.0),
                                     (0.0, 0.2, 0.2)])
def test_classification_matches_the_strict_conditions(M, gam, T):
    rep = FL.report_from_tree(_leaves(M, gam, T))
    A = M + H * (gam + T)
    strict = (A > 0 and gam + T > 0 and 4 * M + 2 * H * (gam + T) > H * H)
    assert (rep["classification"] == "stable") == strict, rep


def test_missing_damping_counterexample_is_refused_and_repaired():
    rep = FL.report_from_tree(_leaves(1.0, 0.0, 0.0))
    assert rep["A"] > 0 and 4 * rep["M"] > H * H
    assert (rep["a"], rep["b"]) == (1.0, 1.0)           # z^2 - z + 1
    assert rep["classification"] == "neutral"
    assert FL.filter_failure(rep) is not None
    fixed, tel = FL.repair(_leaves(1.0, 0.0, 0.0))
    assert float(fixed["fil_gamma"][0] + fixed["fil_T"][0]) > 0.0
    good = FL.report_from_tree(fixed)
    assert good["classification"] == "stable"
    assert FL.filter_failure(good) is None


def test_gate_refuses_rounded_coefficients_the_gaps_would_accept():
    rep = FL.report_from_tree(_leaves(1e18, 0.0, FL.G_MIN))
    assert rep["gap_gamma_plus_T"] >= 0.0 and rep["gap_filter"] >= 0.0
    assert rep["b"] == 1.0
    assert rep["classification"] != "stable"
    assert FL.filter_failure(rep) is not None


def test_nonfinite_executed_quantities_are_refused():
    base = FL.coefficient_values(FL.coefficients(_leaves(0.2, 0.1, 1.0)))
    for k in FL.COEFF_NAMES:
        bad = dict(base, **{k: float("nan")})
        assert FL.filter_failure(FL.filter_report(bad)) is not None, k
        g = FL.in_loop_guard(tuple(jnp.asarray(bad[n]) for n in
                                   FL.COEFF_NAMES))
        assert not bool(g["ok"]), k


def test_guard_and_report_classify_the_rollouts_own_values():
    """The acceptance record is taken from the coefficients a compiled
    program RETURNED; the host guard and the exact report agree on them."""
    pn = _source_native_f64()
    ep = _ep(9702)
    for M, gam, T in ((0.0, 0.0, 1.0), (0.4, 0.1, 1.2), (1.0, 0.0, 0.0)):
        out = TC.rollout_outputs(FL.FILTERED, _tree(pn, M, gam, T), ep)
        vals = FL.coefficient_values(out["coeff"])
        g = FL.in_loop_guard(out["coeff"])
        rep = FL.filter_report(vals)
        assert bool(g["ok"]) == (FL.filter_failure(rep) is None), (M, gam, T)


def test_repair_clamps_restores_and_is_idempotent():
    rs = onp.random.RandomState(7)
    for _ in range(12):
        p = _leaves(rs.uniform(-1, 2), rs.uniform(-1, 1), rs.uniform(-1, 3))
        once, tel = FL.repair(p)
        twice, tel2 = FL.repair(once)
        for k in FL.LEAVES:
            assert float(once[k][0]) >= 0.0
            assert onp.array_equal(onp.asarray(once[k]),
                                   onp.asarray(twice[k]))
        assert int(tel2["n_repaired"]) == 0
        assert float(tel["gap_gamma_plus_T"]) >= 0.0
        assert float(tel["gap_filter"]) >= 0.0
        assert FL.filter_failure(FL.report_from_tree(once)) is None


def test_repair_fixes_the_exact_points_and_respects_frozen_leaves():
    for M, gam, T in (FL.tss_boundary(), FL.native_point()):
        p = _leaves(M, gam, T)
        out, tel = FL.repair(p)
        for k in FL.LEAVES:
            assert onp.array_equal(onp.asarray(out[k]), onp.asarray(p[k]))
        assert int(tel["n_repaired"]) == 0
    out, _ = FL.repair(_leaves(-0.5, -0.25, 0.9),
                       frozen=("fil_M", "fil_gamma"))
    assert float(out["fil_M"][0]) == -0.5
    assert float(out["fil_gamma"][0]) == -0.25


def test_validate_requires_executed_records_and_the_boundary():
    pn = _source_native_f64()
    ok = _tree(pn, 0.0, 0.0, 1.0)
    assert TC.validate(TC.GEN, ok, _executed(ok)) is None
    assert TC.validate(TC.TSS, ok, _executed(ok)) is None
    assert "no executed" in TC.validate(TC.GEN, ok, None)
    bad = _tree(pn, 1.0, 0.0, 0.0)
    assert TC.validate(TC.GEN, bad, _executed(bad)) is not None
    moved = _tree(pn, 0.2, 0.0, 1.0)
    msg = TC.validate(TC.TSS, moved, _executed(moved))
    assert msg is not None and "boundary" in msg


# ======================= 4. gradients against an independent reference ======
def _independent_sensitivity(p, ep, direction):
    """INDEPENDENTLY CODED sequential float64 forward sensitivity in the
    ORIGINAL form of the law. Returns (loss, dloss, final W, U, y)."""
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
    A = M + H * (gam + T)
    dA = dM + H * (dgam + dT)
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


#: absolute floor factor for a DECLARED analytic zero (amendment after the
#: failed dispatch 20260917-152713): |derivative| <= ZERO_FLOOR x eps x
#: max(|loss|, 1), with eps of the dtype that computed that derivative
ZERO_FLOOR = 1e3


def sensitivity_decision(label, val, jvp, ref_val, ref_deriv, fwd_val, hstep,
                         rel_tol, expected_zero=False, prod_eps=EPS64):
    """Finite FIRST for every piece of evidence (review R5), then the primal
    agreement, then the decisive derivative decision, then the diagnostic
    forward difference (which must be finite).

    `expected_zero` is DECLARED by the fixture from an analytic identity,
    never inferred from a measured magnitude. For such a fixture the
    production JVP and the independent reference are each checked SEPARATELY
    against the absolute floor of the dtype that computed it; otherwise the
    reference must be nondegenerate and the JVP must agree with it."""
    fails = []
    pieces = dict(loss=val, jvp=jvp, reference_loss=ref_val,
                  reference_derivative=ref_deriv, perturbed_loss=fwd_val)
    nonfinite = [k for k, x in pieces.items() if not _all_finite(x)]
    if nonfinite:
        return [f"{label}: non-finite {nonfinite}"], None
    fd = (float(fwd_val) - float(val)) / hstep
    if not _all_finite(fd):
        return [f"{label}: non-finite FD"], None
    if abs(float(val) - ref_val) > ID64 * max(abs(ref_val), 1.0):
        fails.append(f"{label}: reference primal loss differs")
    resolvable = abs(float(jvp)) >= 100 * EPS64 * max(abs(float(val)),
                                                      1.0) / hstep
    if expected_zero:
        prod_floor = ZERO_FLOOR * prod_eps * max(abs(float(val)), 1.0)
        ref_floor = ZERO_FLOOR * EPS64 * max(abs(ref_val), 1.0)
        if abs(float(jvp)) > prod_floor:
            fails.append(f"{label}: declared analytic zero, but production "
                         f"JVP {float(jvp):.3e} > floor {prod_floor:.3e}")
        if abs(ref_deriv) > ref_floor:
            fails.append(f"{label}: declared analytic zero, but reference "
                         f"{ref_deriv:.3e} > floor {ref_floor:.3e}")
        return fails, dict(fd=fd, err=None, resolvable=bool(resolvable),
                           production_floor=prod_floor,
                           reference_floor=ref_floor,
                           verdict=("analytic zero verified within tolerance"
                                    if not fails else "analytic zero NOT "
                                    "verified"))
    err = abs(float(jvp) - ref_deriv) / max(abs(ref_deriv), 1e-30)
    if not _all_finite(err):
        return [f"{label}: non-finite error"], None
    if abs(ref_deriv) <= 1e-12:
        fails.append(f"FIXTURE DEFECT: degenerate sensitivity for {label}")
    elif err > rel_tol:
        fails.append(f"{label}: derivative error {err:.2e}")
    return fails, dict(fd=fd, err=err, resolvable=bool(resolvable),
                       verdict="agrees with the independent sensitivity")


def test_sensitivity_decision_rejects_nan_evidence():
    nan = float("nan")
    assert sensitivity_decision("x", 1.0, 0.5, 1.0, nan, 1.0, 1e-6,
                                SENS64)[0]
    assert sensitivity_decision("x", 1.0, 0.5, nan, 0.5, 1.0, 1e-6,
                                SENS64)[0]
    assert sensitivity_decision("x", 1.0, 0.5, 1.0, 0.5, nan, 1e-6,
                                SENS64)[0]
    fails, info = sensitivity_decision("x", 1.0, 0.5, 1.0, 0.5,
                                       1.0 + 0.5e-6, 1e-6, SENS64)
    assert not fails and info is not None


def test_expected_zero_decision_rejects_above_floor_and_nonfinite():
    """Amendment regressions: a declared analytic zero is rejected when
    EITHER derivative exceeds its own dtype's floor or is non-finite."""
    floor = ZERO_FLOOR * EPS64 * 1.0
    args = dict(hstep=1e-6, rel_tol=SENS64, expected_zero=True)
    ok, info = sensitivity_decision("z", 1.0, 0.0, 1.0, -1.1e-17, 1.0,
                                    **args)
    assert not ok and info["verdict"] == ("analytic zero verified within "
                                          "tolerance")
    assert sensitivity_decision("z", 1.0, 0.5 * floor, 1.0, 0.5 * floor, 1.0,
                                **args)[0] == []
    assert sensitivity_decision("z", 1.0, 3 * floor, 1.0, 0.0, 1.0,
                                **args)[0]                 # production
    assert sensitivity_decision("z", 1.0, 0.0, 1.0, 3 * floor, 1.0,
                                **args)[0]                 # reference
    assert sensitivity_decision("z", 1.0, float("nan"), 1.0, 0.0, 1.0,
                                **args)[0]
    assert sensitivity_decision("z", 1.0, 0.0, 1.0, float("inf"), 1.0,
                                **args)[0]
    # the float32 floor is looser for the PRODUCTION derivative only
    f32 = float(onp.finfo(onp.float32).eps)
    assert sensitivity_decision("z", 1.0, 3 * floor, 1.0, 0.0, 1.0,
                                prod_eps=f32, **args)[0] == []
    assert sensitivity_decision("z", 1.0, 0.0, 1.0, 3 * floor, 1.0,
                                prod_eps=f32, **args)[0]


#: the ONE declared analytic zero. For M = 0, gamma = h the law is
#: y_next = R_t + T/(h+T) (y - R_prev); with matched (zero) initialization
#: y = R_prev on every token by induction, so the whole trajectory is native
#: for EVERY fixed T on this line and dL/dT = 0 exactly, for any data.
EXPECTED_ZERO = "T inward at the native point"


@pytest.mark.parametrize("point,direction,name,expected_zero", [
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "M inward at the TSS boundary", False),
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "gamma inward at the TSS boundary",
     False),
    ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), "T on the TSS boundary", False),
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), "M inward at the native point", False),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), EXPECTED_ZERO, True),
    ((0.4, 0.3, 1.5), (1.0, 0.0, 0.0), "M at an interior point", False),
    ((0.4, 0.3, 1.5), (0.0, 1.0, 0.0), "gamma at an interior point", False),
    ((0.4, 0.3, 1.5), (0.0, 0.0, 1.0), "T at an interior point", False),
])
def test_coefficient_derivatives_match_the_independent_sensitivity(
        point, direction, name, expected_zero):
    """Derivatives flow through the coefficients with no boundary branch;
    at the boundaries they are taken INWARD, never through the repair and
    never at a negative-mass or negative-damping point. `expected_zero` is a
    declaration from the analytic identity above, not a measured outcome."""
    pn = _source_native_f64()
    ep = _ep(9703)
    p = _tree(pn, *point)
    ref_val, ref, Wr, Ur, yr = _independent_sensitivity(p, ep, direction)
    out = PM.rollout(FL.FILTERED, p, ep)
    for label, got, want in (("W", out["final_carry"][0], Wr),
                             ("U", out["final_carry"][1], Ur),
                             ("y", out["final_carry"][2], yr)):
        assert _rel(got, want) < ID64, (name, label)
    f = _loss_fn(p, ep)
    coeffs = jnp.asarray(point, jnp.float64)
    val, jvp = jax.jvp(f, (coeffs,), (jnp.asarray(direction, jnp.float64),))
    hstep = 1e-6
    fwd = f(coeffs + hstep * jnp.asarray(direction, jnp.float64))
    fails, info = sensitivity_decision(name, val, jvp, ref_val, ref, fwd,
                                       hstep, SENS64,
                                       expected_zero=expected_zero)
    print(f"  {name}: jvp {float(jvp):.6e} reference {ref:.6e} {info}")
    assert not fails, fails
    if expected_zero:
        print(f"  {name}: {info['verdict']} (production floor "
              f"{info['production_floor']:.2e}, reference floor "
              f"{info['reference_floor']:.2e}; forward-difference diagnostic "
              f"{info['fd']:.3e})")
    elif info["resolvable"]:
        assert abs(info["fd"] - float(jvp)) <= 1e-3 * max(abs(float(jvp)),
                                                          1e-12)
    else:
        print("  LIMITATION: finite but below float64 forward-difference "
              "resolvability; the independent sensitivity is the evidence")


# ======================= 5. training step, freezing and the runner ==========
def test_train_step_freezes_constants_and_gates_the_forward_pass():
    pn = _source_native_f64()
    eps = {k: jnp.asarray(v) for k, v in TK.generate_batch(9706, 2).items()
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
        assert _all_finite(*o[2:8], *grads.values())
        assert bool(tel["gate_ok"]), arm
        assert TC.frozen_leaf_differences(p2, p, arm) == {}
        if arm == TC.TSS:
            for k in ("fil_M", "fil_gamma"):
                assert onp.array_equal(onp.asarray(p2[k]), onp.asarray(p[k]))
            assert _moved_finite(p2["fil_T"][0], p["fil_T"][0])
        else:
            assert any(_moved_finite(p2[k][0], p[k][0]) for k in FL.LEAVES)
        assert _moved_finite(onp.asarray(p2["a_proj"]).ravel()[0],
                             onp.asarray(p["a_proj"]).ravel()[0])
        mask = TC.leaf_mask(p, TC.FROZEN_LEAVES[arm])
        assert all(float(mask[k]) == 0.0 for k in TC.FROZEN_LEAVES[arm])
        assert all(float(mask[k]) == 1.0 for k in p
                   if k not in TC.FROZEN_LEAVES[arm])
    # the gate is on the FORWARD pass: an unstable tree is refused
    bad = _tree(pn, 1.0, 0.0, 0.0)
    o = TC.train_step_filtered(FL.FILTERED, (), bad, ST.TX.init(bad), eps, lr)
    assert not bool(o[8]["gate_ok"])


def test_step_failure_is_checked_at_every_step():
    rec = dict(update=3, executed_filter_ok=True, jury_min=0.5,
               proc_max_abs=1.0, executed={},
               **{k: 0.1 for k in FL.LEAVES},
               **{"grad_" + k: 0.0 for k in FL.LEAVES})
    good = {k: 0.1 for k in ST.SCALAR_NAMES}
    assert TC.step_failure(TC.GEN, good, rec) is None
    assert TC.step_failure(TC.GEN, dict(good, **{ST.SCALAR_NAMES[0]:
                                                 float("nan")}), rec)
    assert TC.step_failure(TC.GEN, good, dict(rec, executed_filter_ok=False))
    assert TC.step_failure(TC.GEN, good, dict(rec, grad_fil_T=float("inf")))


def _small_val():
    return TK.generate_batch(9707, 4)


def _runner(arm, source_p, updates, out, val_at=None, status=None):
    status = {"incomplete": []} if status is None else status
    return TC.run_one(arm, "A", 0.003, TC.SOURCE_DEV, source_p, _small_val(),
                      updates, str(out), 1e18, 30.0, status, "fixture",
                      "fixture-source", None, val_at), status


@pytest.mark.parametrize("arm", [TC.TSS, TC.GEN, TC.NATIVE, TC.OPERATOR])
def test_zero_update_endpoint_is_valid(arm, tmp_path):
    """Review R1: a selected update-zero endpoint is a valid named-family
    endpoint - evaluated, validated and persisted - not a failure and not a
    frozen-source anchor."""
    pn = _source_native_f64()
    (rec, p), status = _runner(arm, TC.start_tree(arm, pn), 0, tmp_path)
    assert rec["invalid"] is None, rec["invalid"]
    assert rec["updates"] == 0
    assert rec["endpoint_kind"] == "zero_update_named_family_endpoint"
    assert rec["regime"] == "full"
    assert len(rec["validation"]) == 1 and rec["validation"][0]["accepted"]
    assert os.path.isfile(rec["validation"][0]["params_file"])
    assert rec["validation"][0]["opt_file"] is not None


def test_zero_update_endpoint_with_nonfinite_state_fails(tmp_path):
    pn = _source_native_f64()
    bad = dict(TC.start_tree(TC.GEN, pn),
               readout_b=jnp.full_like(pn["readout_b"], jnp.nan))
    (rec, _), _ = _runner(TC.GEN, bad, 0, tmp_path)
    assert rec["invalid"] is not None


def test_every_checkpoint_is_validated_persisted_and_reproducible(tmp_path):
    """Review R3: each selectable checkpoint is accepted and saved before
    the next update, and an earlier checkpoint's saved tree reproduces its
    recorded metrics."""
    pn = _source_native_f64()
    start = TC.start_tree(TC.GEN, pn)
    (rec, _), status = _runner(TC.GEN, start, 2, tmp_path, val_at=(0, 1, 2))
    assert rec["invalid"] is None, rec["invalid"]
    assert [v["update"] for v in rec["validation"]] == [0, 1, 2]
    assert all(v["accepted"] for v in rec["validation"])
    assert len(status["checkpoint_log"]) == 3
    v1 = rec["validation"][1]
    p1 = TC.load_params(v1["params_file"], start)
    m, sets = TC.evaluate_arm(TC.GEN, p1, _small_val())
    for key, got in (("primary", m["primary"]),
                     ("revision_ce", m["revision_ce"]),
                     ("retention", m["retention_revision_untouched"]),
                     ("recall", m["recall_overall"])):
        _close(f"saved update-1 checkpoint {key}", got, v1[key], ID64)
    assert TC.validate(TC.GEN, p1, sets) is None


def test_an_invalid_intermediate_checkpoint_cannot_be_hidden(tmp_path,
                                                            monkeypatch):
    pn = _source_native_f64()
    real = TC.evaluate_arm
    calls = {"n": 0}

    def flaky(arm, p, eps_np):
        m, sets = real(arm, p, eps_np)
        calls["n"] += 1
        if calls["n"] == 2:                       # the update-1 checkpoint
            m = dict(m, primary=float("nan"))
            m["revision"] = dict(m["revision"], macro_accuracy=float("nan"),
                                 accuracy=float("nan"))
        return m, sets
    monkeypatch.setattr(TC, "evaluate_arm", flaky)
    (rec, _), status = _runner(TC.GEN, TC.start_tree(TC.GEN, pn), 2,
                               tmp_path, val_at=(0, 1, 2))
    assert rec["invalid"] is not None and "update 1" in rec["invalid"]
    assert [v["update"] for v in rec["validation"]] == [0, 1]
    assert rec["final_validation"] is None
    assert status["checkpoint_log"][-1]["accepted"] is False


# ======================= 6. protocol wiring: work, streams, selection =======
def test_streams_are_fresh_and_work_fits_the_declared_budget():
    assert TC.stream_overlaps() == []
    w = TC.planned_work()
    assert w["trained_runs_development"] == 8
    assert w["named_family_final_endpoints"] == 12
    assert w["max_total_updates"] == 4000
    assert w["development_checkpoints_per_family"] == 9
    assert TC.VAL_AT == (0, 25, 50, 100, 200)
    assert TC.FROZEN_LEAVES[TC.TSS] == ("fil_M", "fil_gamma")
    assert TC.LAW_OF[TC.OPERATOR] == OD.ORDINARY


def _dev_rows(spec, reject=None):
    rows = []
    for arm, per_config in spec.items():
        for config, lr, vals in per_config:
            rows.append(dict(rule=arm, config=config, lr=lr, validation=[
                dict(update=u, primary=p, revision_ce=ce, retention=rt,
                     recall=rc,
                     accepted=not (reject and (arm, config, u) == reject),
                     params_file=f"{arm}_{config}_u{u}.msgpack")
                for u, p, ce, rt, rc in vals]))
    return rows


def _uniform(primary, ce=1.0, ret=0.5, rec=0.5):
    return [(u, primary, ce, ret, rec) for u in TC.VAL_AT]


def _spec():
    return {
        TC.NATIVE: [("A", 0.003, _uniform(0.50, ret=0.60, rec=0.70)),
                    ("B", 0.01, _uniform(0.55, ret=0.58, rec=0.70))],
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


def test_selection_is_native_first_then_matched_retention():
    status = {}
    sel, plan = TC.select(_dev_rows(_spec()), status)
    assert sel is not None
    assert sel[TC.NATIVE]["primary"] == 0.55 and sel[TC.NATIVE]["lr"] == 0.01
    assert status["selection"]["r_native"] == 0.58
    assert sel[TC.GEN]["feasible"] and sel[TC.GEN]["update"] == 50
    assert sel[TC.GEN]["params_file"] == f"{TC.GEN}_A_u50.msgpack"
    assert sel[TC.TSS]["feasible"] and sel[TC.TSS]["update"] == 0
    assert sel[TC.TSS]["lr"] == 0.003
    assert not sel[TC.OPERATOR]["feasible"] and sel[TC.OPERATOR]["diagnostic"]


def test_an_unaccepted_intermediate_checkpoint_blocks_selection():
    rows = _dev_rows(_spec(), reject=(TC.GEN, "B", 25))
    assert TC.select(rows, {}) == (None, None)


def test_update_zero_checkpoint_is_deduplicated():
    rows = _dev_rows({TC.NATIVE: [("A", 0.003, _uniform(0.5)),
                                  ("B", 0.01, _uniform(0.5))]})
    cps = TC.checkpoints(rows, TC.NATIVE)
    assert len(cps) == 9 and sum(c["update"] == 0 for c in cps) == 1
    assert all("heldout" not in c for c in cps)


def test_selection_refuses_non_finite_checkpoints():
    spec = _spec()
    spec[TC.NATIVE][1] = ("B", 0.01, [(u, (float("nan") if u == 25 else 0.5),
                                       1.0, 0.5, 0.5) for u in TC.VAL_AT])
    assert TC.select(_dev_rows(spec), {}) == (None, None)


def _s(primary, feasible=True, ce=1.0, update=50, lr=0.003, config="A"):
    return dict(primary=primary, feasible=feasible, diagnostic=not feasible,
                revision_ce=ce, update=update, lr=lr, config=config,
                params_file=f"p_{primary}_{update}.msgpack")


def test_deployment_fallbacks_use_the_full_ordering_and_never_relabel():
    sel = {TC.NATIVE: _s(0.60), TC.TSS: _s(0.75, feasible=False),
           TC.GEN: _s(0.58), TC.OPERATOR: _s(0.90, feasible=False)}
    plan = TC.deployment_plan(sel, {})
    assert [f["kind"] for f in plan[TC.GEN]["fallbacks"]] == ["native"]
    assert plan[TC.GEN]["choice"] == "native"
    assert plan[TC.GEN]["executed_family"] == "momentum_delta"
    assert plan[TC.GEN]["chosen_checkpoint"]["arm"] == TC.NATIVE
    assert "NOT a point of the literal-TSS family" in \
        plan[TC.TSS]["fallbacks"][0]["map"]
    assert "(0, h, 0)" in plan[TC.GEN]["fallbacks"][0]["map"]
    assert plan[TC.OPERATOR]["choice"] == "native"
    out = TC.deployment_outcome(
        [dict(rule=TC.NATIVE, seed=501, heldout=dict(primary=0.6),
              endpoint_params_file="x")], plan)
    assert out[TC.GEN]["counted_as_improvement"] is False
    assert "NOT literal TSS" in out[TC.GEN]["label"]
    assert out[TC.GEN]["chosen_checkpoint"]["arm"] == TC.NATIVE
    # feasible TSS becomes a fallback; equal primary resolved by CE, not by
    # insertion order
    sel2 = dict(sel, **{TC.TSS: _s(0.60, ce=0.9, update=100)})
    plan2 = TC.deployment_plan(sel2, {})
    assert plan2[TC.GEN]["choice"] == "tss"
    assert plan2[TC.GEN]["executed_family"] == FL.FILTERED
    sel3 = dict(sel, **{TC.TSS: _s(0.60, ce=1.1)})
    assert TC.deployment_plan(sel3, {})[TC.GEN]["choice"] == "native"
    # a full tie on all four keys uses the DECLARED tie rule
    sel4 = dict(sel, **{TC.TSS: _s(0.60)})
    assert TC.deployment_plan(sel4, {})[TC.GEN]["choice"] == "native"
    # strict improvement over the best fallback
    sel5 = dict(sel2, **{TC.GEN: _s(0.60)})
    assert TC.deployment_plan(sel5, {})[TC.GEN]["choice"] == "tss"
    sel6 = dict(sel2, **{TC.GEN: _s(0.61)})
    assert TC.deployment_plan(sel6, {})[TC.GEN]["choice"] == "trained"


def _final(arm, seed, primary, ret, rec, kind="trained_named_family_endpoint"):
    return dict(rule=arm, seed=seed, endpoint_kind=kind,
                heldout=dict(primary=primary,
                             retention_revision_untouched=ret,
                             recall_overall=rec))


def test_screen_labels_availability_and_requires_no_measured_decrease():
    rows = []
    for i, s in enumerate(TC.SOURCE_FINAL):
        rows += [_final(TC.GEN, s, 0.70 + 0.01 * i, 0.60, 0.70),
                 _final(TC.TSS, s, 0.60 + 0.01 * i, 0.60, 0.70,
                        kind="zero_update_named_family_endpoint"),
                 _final(TC.NATIVE, s, 0.55 + 0.01 * i, 0.605, 0.70),
                 _final(TC.OPERATOR, s, 0.72 + 0.01 * i, 0.50, 0.65),
                 _final(TC.ANCHOR, s, 0.40, 0.62, 0.70,
                        kind="frozen_source_anchor")]
    sel = {a: dict(feasible=True, diagnostic=False)
           for a in (TC.NATIVE,) + TC.EXTENSION_ARMS}
    by = {c["name"]: c for c in TC.screen(rows, sel)["comparisons"]}
    gt = by["extension_versus_literal_tss"]
    assert gt["promising_matched_retention"] is True
    assert gt["endpoint_kinds"][TC.TSS] == [
        "zero_update_named_family_endpoint"]
    gn = by["generalized_versus_native"]
    assert gn["safeguard_passed_minus_one_pp"] is True
    assert gn["no_measured_decrease"] is False
    assert gn["promising_matched_retention"] is False
    sel2 = dict(sel, **{TC.TSS: dict(feasible=False, diagnostic=True)})
    g2 = {c["name"]: c for c in TC.screen(rows, sel2)["comparisons"]}[
        "extension_versus_literal_tss"]
    assert g2["constrained_screen_available"] is False
    assert g2["promising_matched_retention"] is False
    assert "INFEASIBLE" in g2["availability_note"]
    assert by["generalized_versus_learned_operator"]["kind"] == "outside_claim"


def test_direct_recovery_policy_is_decisive_and_finite_first():
    pn = _source_native_f64()
    val = _small_val()
    fails, rows = TC.direct_recovery(pn, val)
    assert fails == [], fails
    assert len(rows) == 2 * TC.RECOVERY_EPISODES_PER_FAMILY
    for r in rows:
        assert (r["executed"]["a"], r["executed"]["b"], r["executed"]["c"],
                r["executed"]["d"]) == (0.0, 0.0, 1.0, 0.0)
    broken = dict(pn, readout_b=jnp.full_like(pn["readout_b"], jnp.nan))
    fails2, _ = TC.direct_recovery(broken, val)
    assert fails2
