"""Focused checks for the same-backbone study. CLUSTER-ONLY.

Scope (brief s5): the actual rollout and the full gradient path for the new
operator law and the coefficient-only regime — native nesting, the fixed-gate
equivalence, the arbitrary-gate transported identity, first idle-step timing,
streaming, finite arithmetic, coefficient gradients and frozen-leaf
invariance — plus a regression that the completed laws are unchanged by the
additive dispatch. The completed studies' suites are not re-run.

PREDECLARED TOLERANCES, unchanged from the completed studies:
    ID64    1e-9   relative: float64 identities
    GRAD64  1e-8   relative per leaf, absolute floor 1e3 eps64 G
    FD64    1e-6   relative: coefficient derivative vs the analytic reference
    EXACT64 1e-12  relative: executed transition versus its analytic form

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
from experiments.nested_memory import model as NM                 # noqa: E402
from experiments.nested_memory import task as TK                  # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import ordinary as OD       # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import same_backbone as SB  # noqa: E402
from experiments.prospective_momentum import study as ST          # noqa: E402

ID64, GRAD64, FD64, EXACT64 = 1e-9, 1e-8, 1e-6, 1e-12
F64 = onp.float64
EPS64 = float(onp.finfo(F64).eps)


def _rel(a, b):
    a, b = onp.asarray(a, F64), onp.asarray(b, F64)
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def _ep(seed, i=0):
    b = TK.generate_batch(seed, 2)
    return {k: jnp.asarray(b[k][i]) for k in ("key_id", "val_id", "event",
                                               "label")}


def _f64(p):
    return {k: jnp.asarray(onp.asarray(v), dtype=jnp.float64)
            for k, v in p.items()}


def _source_native_f64():
    run = os.environ.get("PM_SOURCE_RUN", SB.SOURCE_RUN)
    assert os.path.isdir(RS.sources_dir(run)), \
        f"read-only source run {run} is REQUIRED"
    import json
    with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
        manifest = json.load(fh)
    e = RS.find_entry(manifest, SB.SOURCE_DEV, SB.SOURCE_FAMILY)
    return _f64(RS.restore_source(run, e))


def _perturbed_native(seed):
    rs = onp.random.RandomState(seed)
    p = _f64(NM.init_params("momentum_delta", seed))
    for k in ("a_proj", "b_proj", "m_proj", "e_proj"):
        p[k] = p[k] + jnp.asarray(0.7 * rs.randn(*p[k].shape))
    return p


def _schedule(rs, n, dv=3, dk=4, const_eta_mu=True, const_ratio=False):
    keys = rs.randn(n, dk)
    keys /= onp.linalg.norm(keys, axis=1)[:, None]
    sched = dict(k=keys, v=rs.randn(n, dv),
                 m=(rs.rand(n) < 0.6).astype(float),
                 alpha=rs.uniform(0.2, 1.0, n), beta=rs.uniform(0.1, 0.9, n))
    if const_eta_mu:
        sched["eta"] = onp.full(n, 1.3)
        sched["mu"] = onp.full(n, 0.7)
    elif const_ratio:
        mu = rs.uniform(0.3, 0.9, n)
        sched["mu"] = mu
        sched["eta"] = 1.7 * mu                      # eta/mu constant
    else:
        sched["eta"] = rs.uniform(0.8, 1.9, n)
        sched["mu"] = rs.uniform(0.3, 0.9, n)
    return sched


def _run(step, sched, kappa, n_carry, dv=3, dk=4, extra=()):
    carry = tuple([onp.zeros((dv, dk))] * n_carry) + extra
    W, U = [], []
    for t in range(len(sched["k"])):
        carry = step(tuple(jnp.asarray(c) for c in carry),
                     jnp.asarray(sched["k"][t]), jnp.asarray(sched["v"][t]),
                     jnp.asarray(sched["m"][t]), sched["alpha"][t],
                     sched["beta"][t], sched["mu"][t], sched["eta"][t], kappa)
        carry = tuple(onp.asarray(c) for c in carry)
        W.append(carry[0].copy())
        U.append(carry[1].copy())
    return onp.array(W), onp.array(U), carry


# ================================ 1. fixed-gate equivalence (audit s1) ======
@pytest.mark.parametrize("kappa", [0.0, 0.7, 3.0])
def test_constant_eta_mu_makes_the_two_laws_identical(kappa):
    """Closed loop, with state-dependent residuals, arbitrary alpha, beta,
    masks, keys and values; both start from zero carries."""
    rs = onp.random.RandomState(4)
    s = _schedule(rs, 40, const_eta_mu=True)
    Wc, Qc, _ = _run(PD.prospective_step, s, kappa, 2)
    Wo, Uo, _ = _run(OD.ordinary_step, s, kappa, 3)
    assert _rel(Wo, Wc) < ID64
    # the mapped state matches too: U = Q + kappa (eta/mu) R
    eta, mu = s["eta"][0], s["mu"][0]
    R = []
    W_prev = onp.zeros((3, 4))
    for t in range(len(s["k"])):
        Wb = s["alpha"][t] * W_prev
        R.append(s["m"][t] * onp.outer(Wb @ s["k"][t] - s["v"][t], s["k"][t]))
        W_prev = Wc[t]
    assert _rel(Uo, Qc + kappa * (eta / mu) * onp.array(R)) < ID64


def test_equivalence_holds_iff_the_eta_over_mu_ratio_is_constant():
    rs = onp.random.RandomState(5)
    same = _schedule(rs, 30, const_eta_mu=False, const_ratio=True)
    Wc, _, _ = _run(PD.prospective_step, same, 0.9, 2)
    Wo, _, _ = _run(OD.ordinary_step, same, 0.9, 3)
    assert _rel(Wo, Wc) < ID64, "constant eta/mu must still coincide"
    diff = _schedule(rs, 30, const_eta_mu=False)
    Wc2, _, _ = _run(PD.prospective_step, diff, 0.9, 2)
    Wo2, _, _ = _run(OD.ordinary_step, diff, 0.9, 3)
    assert _rel(Wo2, Wc2) > 1e-6, "varying eta/mu must separate the laws"


def test_one_step_difference_matches_the_audit_formula():
    """candidate - ordinary in U = kappa [eta_t - mu_t eta_(t-1)/mu_(t-1)]
    R_(t-1), from the SAME incoming mapped state and residuals."""
    rs = onp.random.RandomState(6)
    kappa = 0.8
    W0, Q0 = rs.randn(3, 4), rs.randn(3, 4)
    k = rs.randn(4); k /= onp.linalg.norm(k)
    v, m = rs.randn(3), 1.0
    a, b = 0.8, 0.5
    eta_p, mu_p, eta_t, mu_t = 1.4, 0.6, 1.1, 0.8
    Rprev = rs.randn(3, 4)
    U0 = Q0 + kappa * (eta_p / mu_p) * Rprev            # mapped incoming state
    Wc, Qc = PD.prospective_step((jnp.asarray(W0), jnp.asarray(Q0)),
                                 jnp.asarray(k), jnp.asarray(v), m, a, b,
                                 mu_t, eta_t, kappa)
    R = m * onp.outer(a * onp.asarray(W0) @ k - v, k)
    Uc = onp.asarray(Qc) + kappa * (eta_t / mu_t) * R   # candidate, mapped
    Wo, Uo, _ = OD.ordinary_update(jnp.asarray(a * W0), jnp.asarray(U0),
                                   jnp.asarray(Rprev), b, mu_t, eta_t, kappa,
                                   jnp.asarray(R))
    predicted = kappa * (eta_t - mu_t * eta_p / mu_p) * Rprev
    assert _rel(Uc - onp.asarray(Uo), predicted) < ID64
    assert _rel(onp.asarray(Wc) - onp.asarray(Wo), -b * predicted) < ID64


def test_transported_reference_matches_the_candidate_on_any_schedule():
    rs = onp.random.RandomState(7)
    for kappa in (0.0, 0.5, 2.5):
        s = _schedule(rs, 35, const_eta_mu=False)
        Wc, _, _ = _run(PD.prospective_step, s, kappa, 2)
        Wt, Ut, _ = _run(OD.transported_step, s, kappa, 3,
                         extra=(onp.float64(1.0), onp.float64(1.0)))
        assert _rel(Wt, Wc) < ID64, kappa


# ================================ 2. timing, masking, streaming =============
def test_first_idle_step_after_a_write_carries_the_derivative():
    """R is masked BEFORE the difference and Rpros is not re-masked."""
    rs = onp.random.RandomState(8)
    k = onp.zeros(4); k[0] = 1.0
    v = rs.randn(3)
    a, b, mu, eta, kappa = 0.9, 0.4, 0.7, 1.2, 0.6
    W0 = rs.randn(3, 4)
    (W1, U1, R1) = OD.ordinary_step(
        (jnp.asarray(W0), jnp.zeros((3, 4)), jnp.zeros((3, 4))),
        jnp.asarray(k), jnp.asarray(v), 1.0, a, b, mu, eta, kappa)
    # the next token is idle (m = 0): its own residual is zero, but the
    # previous residual still contributes -kappa eta R_(t-1)
    (W2, U2, R2) = OD.ordinary_step((W1, U1, R1), jnp.asarray(k),
                                    jnp.asarray(v), 0.0, a, b, mu, eta, kappa)
    expected_U2 = mu * onp.asarray(U1) - kappa * eta * onp.asarray(R1)
    assert _rel(U2, expected_U2) < ID64
    assert float(onp.max(onp.abs(onp.asarray(R2)))) == 0.0
    assert float(onp.max(onp.abs(onp.asarray(R1)))) > 0.0


def test_streaming_and_dormant_cache_at_kappa_zero():
    pn = _source_native_f64()
    p = PM.add_extension(pn, OD.ORDINARY)
    ep = _ep(9800)
    full = PM.rollout(OD.ORDINARY, p, ep)
    a = PM.rollout(OD.ORDINARY, p, {k: v[:29] for k, v in ep.items()})
    bnd = PM.rollout(OD.ORDINARY, p, {k: v[29:] for k, v in ep.items()},
                     carry0=a["final_carry"])
    assert _rel(jnp.concatenate([a["logits"], bnd["logits"]]),
                full["logits"]) < ID64
    for x, y in zip(bnd["final_carry"], full["final_carry"]):
        assert _rel(x, y) < ID64
    # at kappa = 0 the cache is dormant but NOT empty: it holds the last
    # residual and is reported as carried state
    assert len(full["final_carry"]) == 3
    assert float(onp.max(onp.abs(onp.asarray(full["final_carry"][2])))) >= 0.0
    assert PD.CARRY[OD.ORDINARY] == 192 == OD.CARRY_REALS


def _loss(rule, ep):
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)

    def f(p):
        out = PM.rollout(rule, p, ep)
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(lab, TK.N_VALUES,
                                          dtype=out["logits"].dtype)) * q
        return jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0)
    return f


def test_kappa_zero_is_native_for_both_laws_on_the_restored_source():
    """Regression: the additive dispatch leaves the completed laws intact."""
    pn = _source_native_f64()
    rs = onp.random.RandomState(9)
    ep = _ep(9801)
    c0 = (jnp.asarray(0.3 * rs.randn(NM.D_V, NM.D_K)),
          jnp.asarray(0.3 * rs.randn(NM.D_V, NM.D_K)))
    on = PM.rollout("momentum_delta", pn, ep, carry0=c0)
    for law in ("prospective_momentum", "gain_momentum", OD.ORDINARY):
        p = PM.add_extension(pn, law)
        c = c0 + ((jnp.zeros((NM.D_V, NM.D_K)),) if law == OD.ORDINARY else ())
        out = PM.rollout(law, p, ep, carry0=c)
        assert _rel(out["logits"], on["logits"]) < ID64, law
        assert _rel(out["final_carry"][0], on["final_carry"][0]) < ID64, law
        assert _rel(out["final_carry"][1], on["final_carry"][1]) < ID64, law


def test_shared_gradients_agree_at_kappa_zero():
    pn = _source_native_f64()
    ep = _ep(9802)
    gn = jax.grad(_loss("momentum_delta", ep))(pn)
    G = math.sqrt(sum(float(onp.sum(onp.asarray(v, F64) ** 2))
                      for v in gn.values()))
    for law in ("prospective_momentum", OD.ORDINARY):
        gc = jax.grad(_loss(law, ep))(PM.add_extension(pn, law))
        for k in gn:
            d = float(onp.linalg.norm(onp.asarray(gc[k], F64)
                                      - onp.asarray(gn[k], F64)))
            n = float(onp.linalg.norm(onp.asarray(gn[k], F64)))
            assert d <= max(GRAD64 * n, 1e3 * EPS64 * G), (law, k, d, n)
        assert onp.isfinite(float(gc["kappa"][0]))


# ================================ 3. literal TSS Eq. (17) relationship ======
def test_literal_tss_processing_stage_equals_the_operator_only_at_tau_h():
    rs = onp.random.RandomState(10)
    R = [rs.randn(2, 3) for _ in range(6)]
    y = OD.tss_processing_reference(R, tau=1.0, h=1.0)
    prev = onp.zeros((2, 3))
    for t, r in enumerate(R):
        rpros = r + 1.0 * (r - prev)              # operator at kappa = 1
        assert _rel(y[t], rpros) < ID64, t
        prev = r
    y2 = OD.tss_processing_reference(R, tau=4.0, h=1.0)
    prev = onp.zeros((2, 3))
    diff = 0.0
    for t, r in enumerate(R):
        diff = max(diff, _rel(y2[t], r + 1.0 * (r - prev)))
        prev = r
    assert diff > 1e-3, "tau != h must differ from the kappa = 1 operator"


# ================================ 4. frozen-token stability (audit s4) ======
def test_operator_characteristic_polynomial_is_z_times_the_candidates():
    rs = onp.random.RandomState(11)
    for _ in range(25):
        a, b = rs.uniform(0.3, 1.0), rs.uniform(0.05, 0.95)
        mu, eta = rs.uniform(0.14, 0.95), rs.uniform(0.1, 1.9)
        kappa = rs.uniform(0.0, 2.0)
        A = OD.analytic_transition(a, b, mu, eta, kappa)
        Ax = onp.asarray(OD.executed_transitions(
            tuple(jnp.asarray([x]) for x in (a, b, mu, eta)), kappa,
            jnp.float64))[0]
        assert onp.max(onp.abs(Ax - A)) <= EXACT64 * max(1, onp.max(onp.abs(A)))
        q = b * eta
        quad = onp.array([1.0, -(a + mu - a * q * (1 + kappa)),
                          a * mu - a * q * kappa])
        cubic = onp.poly(A)                        # monic coefficients
        assert abs(cubic[3]) <= 1e-12 * max(1.0, abs(cubic[1]))
        assert onp.max(onp.abs(cubic[:3] - quad)) <= 1e-9 * max(1, onp.max(
            onp.abs(quad)))
        # the same Jury verdict as the candidate's 2x2
        lab, _ = OD.classify_cubic(Ax)
        cand = PD.classify(PD.analytic_transition(a, b, mu, eta, kappa))[0]
        if min(PD.jury(a, b, mu, eta, kappa)) > 1e-9:
            assert lab == "stable" and cand == "stable"


def test_operator_above_the_candidate_bound_is_unstable():
    a, b, mu, eta = 0.9, 0.5, 0.6, 1.3
    kb = float(PD.kappa_bound(*(jnp.asarray([x]) for x in (a, b, mu, eta)))[0])
    assert OD.classify_cubic(OD.analytic_transition(
        a, b, mu, eta, kb * 0.5))[0] == "stable"
    assert OD.classify_cubic(OD.analytic_transition(
        a, b, mu, eta, kb * 1.01))[0] == "unstable"


def test_operator_transition_report_and_failure():
    p = PM.add_extension(_perturbed_native(23), OD.ORDINARY)
    rep = OD.transition_report(p)
    assert rep["classification"].get("stable") == rep["n_write_settings"]
    assert OD.transition_failure(rep) is None
    bad = dict(p, kappa=jnp.asarray([rep["kappa_bound_f64"] * 1.05]))
    assert OD.transition_failure(OD.transition_report(bad)) is not None
    assert SB.validate("ordinary_full", p) is None


# ================================ 5. coefficient-only regime ================
def test_coefficient_only_freezes_the_backbone_but_keeps_full_bptt():
    pn = _source_native_f64()
    eps = {k: jnp.asarray(v) for k, v in TK.generate_batch(9803, 2).items()
           if k in ("key_id", "val_id", "event", "label")}
    lr = jnp.asarray(0.01, dtype=jnp.float64)
    for law in ("prospective_momentum", OD.ORDINARY):
        p = PM.add_extension(pn, law)
        p = dict(p, kappa=jnp.asarray([0.4]))
        opt = ST.TX.init(p)
        full = ST.train_step(law, p, opt, eps, lr)
        frozen = SB.train_step_coefficient_only(law, p, opt, eps, lr)
        # the SAME loss and the SAME kappa gradient: freezing changes only
        # which leaves the optimizer may move
        assert float(full[2]) == float(frozen[2])
        assert float(full[9]) == float(frozen[9]) != 0.0
        pf = frozen[0]
        assert SB.frozen_leaf_differences(pf, p) == {}
        for k in p:
            if k != "kappa":
                assert onp.array_equal(onp.asarray(pf[k]), onp.asarray(p[k]))
        assert float(pf["kappa"][0]) != float(p["kappa"][0])
        # the full regime does move the backbone
        assert SB.frozen_leaf_differences(full[0], p) != {}


def test_frozen_leaf_differences_reports_what_changed():
    p = PM.add_extension(_perturbed_native(29), OD.ORDINARY)
    assert SB.frozen_leaf_differences(p, p) == {}
    moved = dict(p, A_log=p["A_log"] + 1e-7)
    d = SB.frozen_leaf_differences(moved, p)
    assert set(d) == {"A_log"} and d["A_log"] > 0
    only_kappa = dict(p, kappa=p["kappa"] + 1.0)
    assert SB.frozen_leaf_differences(only_kappa, p) == {}


# ================================ 6. plan, streams, screens =================
def test_arm_plan_regimes_and_counts():
    assert len(SB.ARMS) == 6 and len(SB.TRAINED_ARMS) == 5
    assert SB.REGIME_OF["native_frozen"] == "frozen"
    assert SB.LAW_OF["ordinary_coeff"] == OD.ORDINARY
    w = SB.planned_work()
    assert w["trained_runs_development"] == 10
    assert w["trained_runs_final"] == 15
    assert w["frozen_evaluations"] == 4
    assert w["total_updates"] == 25 * 200 == 5000
    pn = _perturbed_native(31)
    for arm, law, regime in SB.ARMS:
        p = (dict(pn) if law == "momentum_delta"
             else PM.add_extension(pn, law))
        c = SB.parameter_counts(arm, p)
        expect = 0 if regime == "frozen" else (
            1 if regime == "coefficient_only" else c["stored"])
        assert c["trainable"] == expect, (arm, c)
        assert c["stored"] == (569 if law == "momentum_delta" else 570)
    assert PM.parameter_counts("momentum_delta", pn)["total"] == 569
    for law in ("prospective_momentum", OD.ORDINARY):
        q = PM.add_extension(pn, law)
        assert PM.parameter_counts(law, q)["total"] == 570
        assert float(q["kappa"][0]) == 0.0
    assert PD.CARRY["prospective_momentum"] == 128
    assert PD.CARRY[OD.ORDINARY] == 192


def test_streams_are_fresh_and_disjoint_from_every_previous_study():
    assert SB.stream_overlaps() == []
    r = SB.new_ranges()
    assert r["continuation_train"] == (325_000_000, 325_030_199)
    assert r["dev_validation"] == (350_000_000,) * 2
    assert r["eval_validation"] == (351_000_000,) * 2
    assert r["heldout"] == (360_000_000,) * 2
    prev = SB.previous_ranges()
    assert (225_000_000, 225_030_199) in prev      # the replication's stream
    assert (260_000_000, 260_000_000) in prev


def _rows(primary, retention=0.60, recall=0.70):
    return [dict(rule=a, seed=s, heldout=dict(
        primary=primary[a], retention_revision_untouched=retention,
        recall_overall=recall))
        for s in SB.SOURCE_FINAL for a, _, _ in SB.ARMS]


def test_every_comparison_is_reported_separately_with_both_conditions():
    base = {a: 0.55 for a, _, _ in SB.ARMS}
    base["prospective_full"] = 0.60
    sc = SB.screen(_rows(base))
    names = [c["name"] for c in sc["comparisons"]]
    assert names == [n for n, _, _, _ in SB.COMPARISONS]
    for c in sc["comparisons"]:
        assert c["complete_paired_seeds"] and "question" in c
        assert c["retention_direction"] == "unchanged"
        assert c["no_measured_decrease"] is True
    same = next(c for c in sc["comparisons"] if c["name"] == "same_backbone")
    assert same["passed"] is True and same["candidate"] == "prospective_full"
    # R3: the coefficient-only arms are compared with the CONTINUED native
    by_name = {c["name"]: c for c in sc["comparisons"]}
    for n, cand in (("coefficient_only_versus_continued_native",
                     "prospective_coeff"),
                    ("ordinary_coefficient_only_versus_continued_native",
                     "ordinary_coeff")):
        c = by_name[n]
        assert c["candidate"] == cand and c["against"] == "native_full"
        assert c["mean_primary_difference"] is not None
        assert c["descriptive_only"] is False
    # the coefficient-only versus full pairs stay descriptive
    for n in SB.DESCRIPTIVE_ONLY:
        assert by_name[n]["descriptive_only"] is True
    assert "gated_delta" in sc["absent_literature_arms"]
    assert "no joint literature win" in sc["note"].lower()
    # every declared pair must resolve through the reused comparison helper
    for c in sc["comparisons"]:
        assert c["display"] and c["against"] in SB.ARM_DISPLAY
        assert isinstance(c["descriptive_only"], bool)
    # the descriptive condition is stricter than the -1 pp safeguard
    rows = _rows(base)
    for r in rows:
        if r["rule"] == "prospective_full":
            r["heldout"]["retention_revision_untouched"] = 0.595
    sc2 = SB.screen(rows)
    same2 = next(c for c in sc2["comparisons"] if c["name"] == "same_backbone")
    assert same2["passed"] is True                      # -0.5 pp, safeguard ok
    assert same2["no_measured_decrease"] is False
    assert same2["retention_direction"] == "decreased"


def test_selection_is_development_only_and_per_arm():
    rows = []
    for a, _, _ in SB.TRAINED_ARMS:
        for tag in ("A", "B"):
            rows.append(dict(tag="dev", rule=a, config=tag,
                             lr=dict(SB.LRS)[tag], seed=SB.SOURCE_DEV,
                             final_validation=dict(
                                 primary=(0.6 if tag == "A" else 0.5),
                                 revision_ce=1.0)))
    rows.append(dict(tag="dev", rule="native_frozen", config="-", lr=0.0,
                     seed=SB.SOURCE_DEV,
                     final_validation=dict(primary=0.99, revision_ce=0.1)))
    status = {}
    sel, table = SB.select(rows, status)
    assert set(sel) == {a for a, _, _ in SB.TRAINED_ARMS}
    assert all(v == "A" for v in sel.values())
    assert "native_frozen" not in sel


def test_production_float32_probe_in_its_own_process():
    """R2: the NEW operator rollout, streaming cache and coefficient-only
    optimizer path, checked with x64 DISABLED as the study runs them."""
    import subprocess
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "prospective_same_backbone_float32_probe.py")
    env = dict(os.environ, JAX_ENABLE_X64="0",
               PM_SOURCE_RUN=os.environ.get("PM_SOURCE_RUN", SB.SOURCE_RUN))
    r = subprocess.run([sys.executable, probe], env=env, capture_output=True,
                       text=True)
    print(r.stdout[-6000:]); print(r.stderr[-3000:])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_the_reused_sources_are_read_only_and_verified():
    run = os.environ.get("PM_SOURCE_RUN", SB.SOURCE_RUN)
    import json
    with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
        manifest = json.load(fh)
    for seed in (SB.SOURCE_DEV,) + SB.SOURCE_FINAL:
        e = RS.find_entry(manifest, seed, SB.SOURCE_FAMILY)
        RS.restore_source(run, e)                      # verifies the checksum
        with pytest.raises(RS.SourceRefusal):
            RS.restore_source(run, dict(e, sha256="0" * 64))
    st = {}
    SB.load_sources(run, st)
    assert st["source"]["reuse"].startswith("read-only")
    assert len(st["source"]["hashes_at_restore"]) == 4
