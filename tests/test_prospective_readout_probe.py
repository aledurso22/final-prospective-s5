"""Checks for the corrected domain, the prior-art identities, the Stage A
readout algebra and the Stage B probe. CLUSTER-ONLY.

The prior-art and Stage A identities were verified in EXACT rational
arithmetic while the documents were written; here they are reproduced
numerically in float64 against the production code, as the brief asks.
Tolerance ID64 = 1e-9 relative, with the established near-zero floor.
"""

import json
import os
import sys

import jax
import numpy as onp
import pytest
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.nested_memory import dynamics as NMD             # noqa: E402
from experiments.prospective_momentum import filtered as FL       # noqa: E402
from experiments.prospective_momentum import readout_probe as RB  # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import retention_aware as RA  # noqa
from experiments.prospective_momentum import temporal_task as TT  # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

ID64 = 1e-9
H = FL.H
F64 = onp.float64


def _close(label, got, ref, tol=ID64):
    g, r = float(got), float(ref)
    assert onp.isfinite(g) and onp.isfinite(r), (label, g, r)
    assert abs(g - r) <= max(tol * abs(r), 1e-12), (label, g, r)


def _coeffs(M, g, T):
    p = {"fil_M": jnp.full((1,), M, dtype=jnp.float64),
         "fil_gamma": jnp.full((1,), g, dtype=jnp.float64),
         "fil_T": jnp.full((1,), T, dtype=jnp.float64)}
    return FL.coefficient_values(FL.coefficients(p))


def _source_native_f64():
    run = os.environ.get("PM_SOURCE_RUN", RB.SOURCE_RUN)
    assert os.path.isdir(RS.sources_dir(run)), f"source run {run} REQUIRED"
    with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
        manifest = json.load(fh)
    e = RS.find_entry(manifest, RA.SOURCE_DEV, TC.SOURCE_FAMILY)
    return {k: jnp.asarray(onp.asarray(v), dtype=jnp.float64)
            for k, v in RS.restore_source(run, e).items()}


# ===================== 1. prior art, reproduced in float64 ==================
def test_two_tap_at_kappa_mu_is_nesterov_gradient_correction():
    rs = onp.random.RandomState(1)
    mu, eta, beta = 0.6, 1.75, 2.0 / 3.0
    gs = rs.randn(40)
    # executed operator with alpha = 1 (no forgetting), R = gradient
    U = x = 0.0
    gp = 0.0
    xs = []
    for g in gs:
        U = mu * U + eta * ((1 + mu) * g - mu * gp)
        x = x - beta * U
        gp = g
        xs.append(x)
    s = beta * eta
    y = ym = 0.0
    gp = 0.0
    ys = []
    for g in gs:
        yn = y + mu * (y - ym) - s * g - mu * s * (g - gp)
        ym, y, gp = y, yn, g
        ys.append(y)
    assert max(abs(a - b) for a, b in zip(xs, ys)) <= 1e-12 * max(
        1.0, max(abs(v) for v in ys))
    # off the line the two differ
    U = x = 0.0
    gp = 0.0
    off = []
    for g in gs:
        U = mu * U + eta * ((1 + mu + 0.25) * g - (mu + 0.25) * gp)
        x = x - beta * U
        gp = g
        off.append(x)
    assert max(abs(a - b) for a, b in zip(off, ys)) > 1e-6


def test_operator_is_momentum_plus_an_averaged_gradient_difference():
    rs = onp.random.RandomState(2)
    mu, eta, kappa = 0.55, 1.3, 1.25
    gs = rs.randn(30)
    U = I = D = 0.0
    gp = 0.0
    worst = 0.0
    for g in gs:
        U = mu * U + eta * ((1 + kappa) * g - kappa * gp)
        I = mu * I + eta * g
        D = mu * D + eta * (g - gp)
        worst = max(worst, abs(U - (I + kappa * D)))
        gp = g
    assert worst <= 1e-12


@pytest.mark.parametrize("M,g,T", [(0.0, 0.0, 1.0), (0.25, 0.0, 0.5),
                                   (0.5, -0.5, 2.0), (0.0, 1.5, 0.5),
                                   (1.0, -0.25, 1.0)])
def test_filter_decomposition_and_velocity_weight(M, g, T):
    c = _coeffs(M, g, T)
    _close("a+c == 1+b+d", c["a"] + c["c"], 1 + c["b"] + c["d"])
    _close("total dR weight", ((c["c"] - 1) + c["b"]) / (1 - c["a"] + c["b"]),
           (H - g) / H)
    rs = onp.random.RandomState(3)
    R = rs.randn(25)
    y1 = y2 = E1 = E2 = Rp = Rpp = 0.0
    worst = 0.0
    for r in R:
        y = c["a"] * y1 - c["b"] * y2 + c["c"] * r - c["d"] * Rp
        E = (c["a"] * E1 - c["b"] * E2 + (c["c"] - 1) * (r - Rp)
             + c["b"] * (Rp - Rpp))
        worst = max(worst, abs(y - (r + E)))
        y2, y1, E2, E1, Rpp, Rp = y1, y, E1, E, Rp, r
    assert worst <= 1e-12


def test_pm_pole_cancellation_is_a_first_order_delta_rule():
    mu, eta, beta = 0.6, 1.75, 2.0 / 3.0
    kappa = mu / (1 - mu)
    rs = onp.random.RandomState(4)
    Rs = rs.randn(20)
    Q = 0.0
    got, want = [], []
    for R in Rs:
        Qn = mu * Q + eta * R
        Q = Qn - kappa * eta * ((1 - mu) / mu) * R
        got.append(-(beta * Qn + kappa * beta * eta * R))
        want.append(-(beta * eta / (1 - mu)) * R)
    assert max(abs(a - b) for a, b in zip(got, want)) <= 1e-12


def test_M_equals_gamma_T_cancels_in_the_continuous_law_only():
    for g, T in ((2.0, 3.0), (0.5, 4.0)):
        M = g * T
        s = -1.0 / T
        _close("continuous denominator at -1/T", M * s * s + (g + T) * s + 1,
               0.0, tol=0.0)
        c = _coeffs(M, g, T)
        z = c["d"] / c["c"]
        assert abs(z * z - c["a"] * z + c["b"]) > 1e-3   # NOT cancelled


# ===================== 2. the corrected coefficient domain =================
@pytest.mark.parametrize("M,g,T", [(0.0, -1.0, 2.0), (0.0, -1.5, 2.5),
                                   (0.25, -0.5, 1.0), (0.0, 0.0, 1.0),
                                   (0.0, 1.0, 0.0), (1.0, -0.25, 0.5),
                                   (0.0, 0.2, 0.2), (1.0, 0.0, 0.0)])
def test_corrected_domain_matches_the_jury_conditions(M, g, T):
    c = _coeffs(M, g, T)
    A = M + H * (g + T)
    strict = (A > 0 and (g + T) > 0 and 4 * M + 2 * H * (g + T) > H * H)
    rep = FL.filter_report(c)
    assert (rep["classification"] == "stable") == strict, (M, g, T, rep)
    assert FL.admissible(M, g, T) == bool(strict and M >= 0 and T >= 0)
    ok = FL.filter_failure_corrected(rep) is None
    assert ok == bool(strict and M >= 0 and T >= 0)


def test_two_tap_line_is_fir_and_admissible_above_kappa_one():
    for kappa in (0.5, 1.0, 1.87, 2.16, 3.0):
        M, g, T = 0.0, H * (1 - kappa), kappa * H
        c = _coeffs(M, g, T)
        assert c["a"] == 0.0 and c["b"] == 0.0            # FIR
        _close("c", c["c"], 1 + kappa)
        _close("d", c["d"], kappa)
        assert FL.admissible(M, g, T)
        assert FL.filter_failure_corrected(FL.filter_report(c)) is None
        if kappa > 1:
            assert g < 0
            # the PASSIVE domain of the completed studies still refuses it
            assert FL.filter_failure(FL.filter_report(c)) is not None


def test_corrected_repair_never_clamps_gamma_and_restores_the_gaps():
    rs = onp.random.RandomState(5)
    for _ in range(12):
        M, g, T = rs.uniform(-1, 2), rs.uniform(-2, 2), rs.uniform(-1, 3)
        p = {"fil_M": jnp.full((1,), M), "fil_gamma": jnp.full((1,), g),
             "fil_T": jnp.full((1,), T)}
        out, tel = FL.repair_corrected(p)
        assert float(out["fil_gamma"][0]) == g                 # never clamped
        assert float(out["fil_M"][0]) >= 0 and float(out["fil_T"][0]) >= 0
        assert float(tel["gap_gamma_plus_T"]) >= 0
        assert float(tel["gap_filter"]) >= 0
        rep = FL.report_from_tree(out)
        assert FL.filter_failure_corrected(rep) is None
        again, tel2 = FL.repair_corrected(out)
        assert int(tel2["n_repaired"]) == 0
    # the passive repair of the completed studies is unchanged
    p = {"fil_M": jnp.full((1,), 0.0), "fil_gamma": jnp.full((1,), -0.5),
         "fil_T": jnp.full((1,), 2.0)}
    assert float(FL.repair(p)[0]["fil_gamma"][0]) == 0.0
    assert float(FL.repair_corrected(p)[0]["fil_gamma"][0]) == -0.5


# ===================== 3. Stage A readout algebra ==========================
def _X(M, g, T, Ws):
    c = _coeffs(M, g, T)
    X1 = X2 = Wp = 0.0
    out = []
    for W in Ws:
        X = c["a"] * X1 - c["b"] * X2 + c["c"] * W - c["d"] * Wp
        out.append(X)
        X2, X1, Wp = X1, X, W
    return out


def test_readout_recovers_native_and_literal_tss():
    rs = onp.random.RandomState(6)
    Ws = rs.randn(30)
    assert max(abs(x - w) for x, w in zip(_X(0.0, H, 0.0, Ws), Ws)) == 0.0
    T = 2.5
    ref, X, Wp = [], 0.0, 0.0
    for W in Ws:
        X = X + (H / T) * (W - X) + (W - Wp)
        ref.append(X)
        Wp = W
    assert max(abs(a - b) for a, b in zip(_X(0.0, 0.0, T, Ws), ref)) <= 1e-12


def test_two_tap_readout_and_lookahead_differ_by_the_decay_term():
    rs = onp.random.RandomState(7)
    Ws = rs.randn(20)
    kappa = 1.75
    two = _X(0.0, H * (1 - kappa), kappa * H, Ws)
    want = [(1 + kappa) * W - kappa * (Ws[i - 1] if i else 0.0)
            for i, W in enumerate(Ws)]
    assert max(abs(a - b) for a, b in zip(two, want)) <= 1e-12
    alpha, beta, U, Wprev = 0.9, 0.5, 1.4, 1.5
    W = alpha * Wprev - beta * U
    dW = W - Wprev
    _close("dW split", dW, (alpha - 1) * Wprev - beta * U)
    lam = 1.25
    _close("two-tap minus lookahead", (W + lam * dW) - (W - lam * beta * U),
           lam * (alpha - 1) * Wprev)


def test_readout_never_feeds_back_into_the_backbone():
    """Frozen backbone: W and U are bitwise identical across readout arms,
    because the probe reads ONE trajectory (Stage A s3)."""
    pn = _source_native_f64()
    batch = TT.generate_batch(991_101, 4)
    fails, tr = RB.trajectories(pn, batch)
    assert fails == [], fails
    base_W, base_U = tr["W"].copy(), tr["U"].copy()
    for arm in (RB.arms()[0], RB.arms()[3], RB.arms()[-1]):
        X = RB.readout(arm, tr)
        assert onp.array_equal(tr["W"], base_W)
        assert onp.array_equal(tr["U"], base_U)
        assert onp.all(onp.isfinite(X))
    # the native-identity arm reproduces the trajectory exactly
    assert onp.array_equal(RB.readout(RB.arms()[0], tr), base_W)


# ===================== 4. the probe's measurement ==========================
def test_roll_forward_target_is_the_native_idle_recurrence():
    """The primary target rolls (W_t, U_t) forward over IDLE tokens only.
    Where an episode's next k tokens really are idle, it must reproduce the
    actual trajectory."""
    pn = _source_native_f64()
    batch = TT.generate_batch(991_104, 4)
    fails, tr = RB.trajectories(pn, batch)
    assert fails == [], fails
    targets = RB.roll_forward(tr)
    g = tr["idle_gates"]
    W, U = tr["W"], tr["U"]
    u, w = U.copy(), W.copy()
    for step in range(1, max(RB.K_OFFSETS) + 1):
        u = g["mu"] * u
        w = g["alpha"] * w - g["beta"] * u
        if step in RB.K_OFFSETS:
            assert onp.allclose(targets[step], w, rtol=0, atol=0)
    ev = onp.asarray(batch["event"])
    from experiments.nested_memory.task import IDLE
    checked = 0
    for k in (1, 2, 4):
        for b in range(ev.shape[0]):
            for t in range(ev.shape[1] - k):
                if onp.all(ev[b, t + 1:t + 1 + k] == IDLE):
                    a = targets[k][b, t].astype(F64)
                    c = W[b, t + k].astype(F64)
                    assert onp.linalg.norm(a - c) <= 1e-4 * max(
                        1.0, onp.linalg.norm(c)), (k, b, t)
                    checked += 1
    assert checked > 0


def test_identity_arm_scores_exactly_one_on_every_reported_key():
    pn = _source_native_f64()
    batch = TT.generate_batch(991_102, 4)
    _, tr = RB.trajectories(pn, batch)
    keys = onp.asarray(NMD.safe_normalize(pn["key_raw"])[0], F64)
    half = RB.split_halves(batch)
    m = RB.arm_metrics(RB.readout(RB.arms()[0], tr), tr, batch,
                       RB.roll_forward(tr), keys, half)
    assert any(v["n"] > 0 for v in m.values())
    for key, v in m.items():
        if v["mean"] is not None:
            _close(key, v["mean"], 1.0, tol=1e-9)
    # the primary key is declared: rolled-forward target, key projection
    assert RB.primary_key("all", "both", "confirmation", 4) in m
    assert (RB.PRIMARY_TARGET, RB.PRIMARY_PROJECTION) == ("rollforward",
                                                          "key")


def test_halves_are_declared_and_balanced():
    batch = TT.generate_batch(991_105, 8)
    half = RB.split_halves(batch)
    fam, cond = onp.asarray(batch["family"]), onp.asarray(batch["condition"])
    for f in onp.unique(fam):
        for c in onp.unique(cond):
            idx = (fam == f) & (cond == c)
            sel = (half[idx] == "selection").sum()
            assert sel == idx.sum() // 2, (f, c, sel, idx.sum())


def test_component_split_is_exact_and_budgets_are_matched():
    pn = _source_native_f64()
    batch = TT.generate_batch(991_103, 4)
    _, tr = RB.trajectories(pn, batch)
    cs = RB.component_split(tr, batch)
    # the split is exact in real arithmetic; the residual here is the float32
    # rounding of the production update, recomputed in float64
    assert cs["identity_max_abs"] <= 1e-4
    assert cs["both/write_tokens"]["n"] > 0
    arm_list = RB.arms()
    names = [a["name"] for a in arm_list]
    assert len(names) == len(set(names))
    sizes = {}
    for a in arm_list:
        sizes[a["family"]] = sizes.get(a["family"], 0) + 1
    for fam in RB.MATCHED_FAMILIES:
        assert sizes[fam] == RB.N_MATCHED, (fam, sizes)
    assert sizes["identity"] == 1 and sizes["generalized_extended"] > 0
    assert set(RB.BASELINE_FAMILIES) == {"literal_tss", "lookahead"}
    fails, rows = RB.arm_admissibility(arm_list)
    assert fails == [], fails
    assert any(a.get("gamma", 0) < 0 for a in arm_list)      # kappa > 1
    assert RB.K_OFFSETS == (1, 2, 4, 8, 16)


def _rec(family, sel, conf):
    m = {}
    for k in RB.K_OFFSETS:
        m[RB.primary_key(half="selection", k=k)] = dict(mean=sel)
        m[RB.primary_key(half="confirmation", k=k)] = dict(mean=conf)
    return dict(family=family, metrics=m)


def test_stopping_rule_uses_the_best_carry_free_baseline():
    """The interior must beat the BEST of the literal-TSS line and the
    U-lookahead, on the confirmation half, at matched budget."""
    beats_tss_only = {
        "tss@T=1.0": _rec("literal_tss", 0.95, 0.95),
        "lookahead@lambda=1.0": _rec("lookahead", 0.80, 0.80),
        "generalized@x": _rec("generalized_interior", 0.90, 0.90)}
    sr = RB.stopping_rule(beats_tss_only)
    assert sr["offsets_won"] == 0 and not sr["passes"]
    assert sr["baseline_families"] == ["literal_tss", "lookahead"]
    beats_both = dict(beats_tss_only,
                      **{"generalized@x": _rec("generalized_interior", 0.70,
                                               0.70)})
    assert RB.stopping_rule(beats_both)["passes"]
    # selection picks the arm, confirmation decides: a selection-only winner
    # whose confirmation number is bad does not pass
    mirage = {
        "tss@T=1.0": _rec("literal_tss", 0.90, 0.90),
        "lookahead@lambda=1.0": _rec("lookahead", 0.92, 0.92),
        "generalized@lucky": _rec("generalized_interior", 0.70, 0.95),
        "generalized@steady": _rec("generalized_interior", 0.88, 0.88)}
    sr2 = RB.stopping_rule(mirage)
    assert sr2["per_offset"]["k1"]["interior"]["arm"] == "generalized@lucky"
    assert sr2["offsets_won"] == 0 and not sr2["passes"]
    edge = dict(beats_tss_only,
                **{"generalized@x": _rec("generalized_interior", 0.79, 0.79)})
    assert RB.stopping_rule(edge)["passes"]        # exactly the margin


def test_overall_verdict_counts_checkpoints_not_a_pooled_mean():
    def seed(passes):
        return dict(stopping_rule=dict(passes=passes))
    ov = RB.overall_verdict({"500": seed(True), "501": seed(True),
                             "502": seed(True), "503": seed(False)})
    assert ov["interior_beats_baseline"] and len(ov["checkpoints_passed"]) == 3
    ov2 = RB.overall_verdict({"500": seed(True), "501": seed(True),
                              "502": seed(False), "503": seed(False)})
    assert not ov2["interior_beats_baseline"]
    assert "STOP" in ov2["verdict"]


def test_probe_stream_is_fresh():
    lo = RB.STREAM["probe"]
    for a, b in RA.previous_ranges() + list(RA.new_ranges().values()):
        assert not (a <= lo <= b), (a, b)
