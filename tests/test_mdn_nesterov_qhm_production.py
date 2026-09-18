"""float64 checks of the PRODUCTION rules and ladder machinery against the
exact reference forms (docs/MDN_NESTEROV_QHM_AUDIT.md). Runs on the cluster.

The identities themselves are proved in exact arithmetic by
`test_mdn_nesterov_qhm_identities.py`; this file checks that the executed
JAX code implements those forms, in the production shell, causally, and
that the completed arms still execute their documented laws.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                          # noqa: E402
import numpy as onp                                              # noqa: E402
import pytest                                                    # noqa: E402

from experiments.nested_memory import dynamics as NMD            # noqa: E402
from experiments.nested_memory import model as NM                # noqa: E402
from experiments.nested_memory.task import QUERY, WRITE          # noqa: E402
from experiments.prospective_momentum import dynamics as PD      # noqa: E402
from experiments.prospective_momentum import filtered as FL      # noqa: E402
from experiments.prospective_momentum import model as PM         # noqa: E402
from experiments.prospective_momentum import nesterov_ladder as NL  # noqa
from experiments.prospective_momentum import optimizer_forms as OF  # noqa
from experiments.prospective_momentum import ordinary as OD      # noqa: E402
from experiments.prospective_momentum import temporal_task as TT  # noqa: E402

F64 = onp.float64
TOL = 1e-12


def lst(x):
    return onp.asarray(x, F64).tolist()


def params(seed=7, nu=None):
    p = NM.init_params("momentum_delta", seed, dtype=F64)
    if nu is not None:
        p = dict(p, **{NL.QHM_LEAF: jnp.asarray([nu], dtype=F64)})
    return p


def episodes(seed=123, n=4):
    return TT.generate_batch(seed, n)


def ep_at(batch, i):
    return {k: jnp.asarray(batch[k][i]) for k in TT.MODEL_INPUTS}


def shell_inputs(p, ep):
    """The keys, values and masks the shell feeds the step (the SAME helper
    calls as the production shell), as float64 host lists."""
    k_all, k_valid = NMD.safe_normalize(p["key_raw"])
    key_id, val_id, event = (onp.asarray(ep[k]) for k in ("key_id", "val_id",
                                                           "event"))
    val_id = onp.where(event == WRITE, val_id, -1)    # the shell's contract
    keys = onp.asarray(k_all, F64)[key_id]
    valid = onp.asarray(k_valid)[key_id]
    has_v = (val_id >= 0).astype(F64)
    vals = onp.asarray(p["value_table"], F64)[onp.maximum(val_id, 0)] \
        * has_v[:, None]
    mask = (event == WRITE).astype(F64) * valid.astype(F64)
    return [(keys[t].tolist(), vals[t].tolist(), float(mask[t]))
            for t in range(len(key_id))]


def gates_of(out):
    a, b, mu, eta = (onp.asarray(g, F64) for g in out["gates"])
    return [(float(a[t]), float(b[t]), float(mu[t]), float(eta[t]))
            for t in range(len(a))]


def ref_logits(p, Ws, toks):
    R, b = onp.asarray(p["readout_W"], F64), onp.asarray(p["readout_b"], F64)
    return onp.stack([R @ (onp.asarray(W, F64) @ onp.asarray(k, F64)) + b
                      for W, (k, _, _) in zip(Ws, toks)])


def maxrel(a, b):
    a, b = onp.asarray(a, F64), onp.asarray(b, F64)
    return float(onp.max(onp.abs(a - b)) / max(1.0, onp.max(onp.abs(b))))


# ---------------------------------------------------------------- steps ----
def test_hand_values_are_reproduced_bitwise():
    """The exact 1x1 values of the identity tests (dyadic, so exact in
    float64): Nesterov W = 1/2, 3/4, 13/16; QHM(nu = 1/2) W = 1/2, 3/4, 27/32."""
    one = jnp.ones((1, 1), F64)
    z = jnp.zeros((1, 1), F64)
    k, v, m = jnp.ones((1,), F64), jnp.ones((1,), F64), jnp.asarray(1.0, F64)
    h = jnp.asarray(0.5, F64)
    c, ws = (z, z), []
    for _ in range(3):
        c = NL.nesterov_step(c, k, v, m, h, h, h, one[0, 0])
        ws.append(float(c[0][0, 0]))
    assert ws == [0.5, 0.75, 0.8125]
    c, ws = (z, z), []
    for _ in range(3):
        c = NL.qhm_step(c, k, v, m, h, h, h, one[0, 0], h)
        ws.append(float(c[0][0, 0]))
    assert ws == [0.5, 0.75, 0.84375]


def test_steps_match_the_reference_forms():
    rng = onp.random.RandomState(0)
    for _ in range(50):
        W, U = rng.randn(8, 8), rng.randn(8, 8)
        k = rng.randn(8)
        k /= onp.linalg.norm(k)
        v = rng.randn(8)
        m = float(rng.randint(0, 2))
        g = (rng.rand(), rng.rand(), rng.rand(), 2 * rng.rand())
        nu = rng.rand()
        J = [jnp.asarray(x, F64) for x in (W, U, k, v, m) + g]
        a = NL.nesterov_step((J[0], J[1]), J[2], J[3], J[4], *J[5:])
        r = OF.nesterov_step((lst(W), lst(U)), (lst(k), lst(v), m), g)
        assert maxrel(a[0], r[0]) <= TOL and maxrel(a[1], r[1]) <= TOL
        a = NL.qhm_step((J[0], J[1]), J[2], J[3], J[4], *J[5:],
                        jnp.asarray(nu, F64))
        r = OF.qhm_step((lst(W), lst(U)), (lst(k), lst(v), m), g, nu)
        assert maxrel(a[0], r[0]) <= TOL and maxrel(a[1], r[1]) <= TOL


# -------------------------------------------------------------- rollouts --
@pytest.mark.parametrize("rule", [NL.NESTEROV, NL.QHM])
def test_rollout_equals_a_direct_unrolled_reference(rule):
    p = params(nu=0.37 if rule == NL.QHM else None)
    batch = episodes()
    for i in range(0, batch["event"].shape[0], 3):
        ep = ep_at(batch, i)
        out = NL.rollout(rule, p, ep, trace=True)
        toks, gs = shell_inputs(p, ep), gates_of(out)
        z = OF.zeros(NM.D_V, NM.D_K, 0.0)
        if rule == NL.NESTEROV:
            ref = OF.run(OF.nesterov_step, (z, z), toks, gs)
        else:
            ref = OF.run(OF.qhm_step, (z, z), toks, gs, 0.37)
        Ws = [c[0] for c in ref]
        assert maxrel(out["W_trace"], Ws) <= TOL
        assert maxrel(out["logits"], ref_logits(p, Ws, toks)) <= TOL
        assert onp.all(onp.isfinite(onp.asarray(out["logits"])))


@pytest.mark.parametrize("rule", [NL.NESTEROV, NL.QHM])
def test_no_future_token_or_gate_enters_an_update(rule):
    p = params(nu=0.6 if rule == NL.QHM else None)
    batch = episodes(seed=321)
    ep = ep_at(batch, 0)
    L = int(ep["event"].shape[0])
    for cut in (1, L // 2, L - 2):
        ep2 = dict(ep)
        rng = onp.random.RandomState(cut)
        for name, hi in (("key_id", 32), ("val_id", 8), ("event", 3)):
            arr = onp.asarray(ep[name]).copy()
            arr[cut + 1:] = rng.randint(0, hi, size=L - cut - 1)
            ep2[name] = jnp.asarray(arr)
        a = onp.asarray(NL.rollout(rule, p, ep)["logits"])
        b = onp.asarray(NL.rollout(rule, p, ep2)["logits"])
        assert onp.array_equal(a[:cut + 1], b[:cut + 1])


def test_the_ladder_shell_is_the_production_shell():
    """The native step in this module's shell equals the production native
    rollout, and QHM at nu = 1 is the native function."""
    p = params()
    batch = episodes(seed=55)
    for i in range(batch["event"].shape[0]):
        ep = ep_at(batch, i)
        ref = PM.rollout("momentum_delta", p, ep)
        sh = NL.rollout(NL.NATIVE_SHELL, p, ep)
        q1 = NL.rollout(NL.QHM, NL.start_tree(NL.QHM_ARM, p), ep)
        assert maxrel(sh["logits"], ref["logits"]) <= 1e-13
        assert maxrel(q1["logits"], ref["logits"]) <= 1e-13
        for a, b in zip(sh["final_carry"], ref["final_carry"]):
            assert maxrel(a, b) <= 1e-13


def test_completed_arms_still_execute_their_documented_laws():
    """Native, the two-tap operator and the processing law (literal-TSS and
    interior points) against the exact reference forms, in float64."""
    p = params(seed=11)
    batch = episodes(seed=77)
    z = OF.zeros(NM.D_V, NM.D_K, 0.0)
    p_ord = dict(PM.convert_momentum(p, OD.ORDINARY),
                 kappa=jnp.asarray([0.3], F64))
    p_tss = dict(p, **FL.initial_leaves(F64, 2.0))
    p_gen = dict(p, fil_M=jnp.asarray([0.3], F64),
                 fil_gamma=jnp.asarray([0.2], F64),
                 fil_T=jnp.asarray([1.5], F64))
    for i in range(0, batch["event"].shape[0], 2):
        ep = ep_at(batch, i)
        nat = PM.rollout("momentum_delta", p, ep)
        toks = shell_inputs(p, ep)
        gs = gates_of(nat)
        ref = OF.run(OF.native_step, (z, z), toks, gs)
        assert maxrel(nat["logits"], ref_logits(p, [c[0] for c in ref],
                                                toks)) <= TOL
        o = PM.rollout(OD.ORDINARY, p_ord, ep, trace=True)
        ref = OF.run(OF.two_tap_step, (z, z, z), toks, gs, 0.3)
        assert maxrel(o["W_trace"], [c[0] for c in ref]) <= TOL
        for pp, (M, gam, T) in ((p_tss, (0.0, 0.0, 2.0)),
                                (p_gen, (0.3, 0.2, 1.5))):
            f = PM.rollout(FL.FILTERED, pp, ep, trace=True)
            coeff = OF.filter_coefficients(M, gam, T)
            ref = OF.run(OF.filtered_step, (z,) * 5, toks, gs, coeff)
            assert maxrel(f["W_trace"], [c[0] for c in ref]) <= TOL


def test_outputs_stay_finite_in_the_declared_stable_region():
    """Long write-only sequences at fixed gates inside each rule's
    frozen-token stable region (Nesterov: (q-1)(alpha+mu+alpha mu) < 1)."""
    rng = onp.random.RandomState(4)

    def run(step, g, scalar, L=256):
        ks = rng.randn(L, NM.D_K)
        ks /= onp.linalg.norm(ks, axis=1, keepdims=True)
        vs = rng.uniform(-1, 1, (L, NM.D_V))

        def body(c, x):
            k, v = x
            c = step(c, k, v, jnp.asarray(1.0, F64), *g, scalar)
            return c, jnp.max(jnp.abs(c[0]))
        z = jnp.zeros((NM.D_V, NM.D_K), F64)
        _, mx = jax.lax.scan(body, (z, z), (jnp.asarray(ks), jnp.asarray(vs)))
        return onp.asarray(mx)

    n = 0
    while n < 8:
        a, b, mu, eta = rng.rand(), rng.rand(), rng.rand(), 2 * rng.rand()
        if (b * eta - 1) * (a + mu + a * mu) >= 1:
            continue
        g = tuple(jnp.asarray(x, F64) for x in (a, b, mu, eta))
        assert onp.all(onp.isfinite(run(NL.nesterov_step, g,
                                        jnp.asarray(0.0, F64))))
        assert onp.all(onp.isfinite(run(NL.qhm_step, g,
                                        jnp.asarray(rng.rand(), F64))))
        n += 1


# ------------------------------------------------ training and start trees --
def test_train_step_projects_nu_and_stays_finite():
    eps = {k: jnp.asarray(v) for k, v in episodes(seed=9).items()
           if k in TT.MODEL_INPUTS}
    lr = jnp.asarray(0.01, F64)
    for nu0, lo, hi in ((1.5, 0.0, 1.0), (-0.2, 0.0, 1.0)):
        p = params(nu=nu0)
        o = NL.train_step(NL.QHM, p, NL.ST.TX.init(p), eps, lr)
        nu = float(onp.asarray(o[0][NL.QHM_LEAF])[0])
        assert lo <= nu <= hi and int(o[8]["n_projected"]) == 1
        assert all(onp.isfinite(float(x)) for x in o[2:8])
    p = params()
    o = NL.train_step(NL.NESTEROV, p, NL.ST.TX.init(p), eps, lr)
    assert all(onp.isfinite(float(x)) for x in o[2:8])
    assert NL.project(p)[0] is p


def test_start_trees():
    p = params()
    nag = NL.start_tree(NL.NAG_ARM, p)
    assert set(nag) == set(p) and all(
        onp.array_equal(onp.asarray(nag[k]), onp.asarray(p[k])) for k in p)
    q = NL.start_tree(NL.QHM_ARM, p)
    assert float(onp.asarray(q[NL.QHM_LEAF])[0]) == 1.0
    with pytest.raises(ValueError):
        NL.start_tree(NL.NAG_ARM, dict(p, extra=jnp.zeros(1)))


def test_frozen_token_table_matches_the_closed_forms():
    settings = onp.array([[0.9, 1.0, 0.9, 1.9], [0.5, 0.5, 0.5, 1.0],
                          [0.3, 0.8, 0.95, 1.7]], F64)
    gates = tuple(jnp.asarray(settings[:, i]) for i in range(4))
    A = onp.asarray(PD.executed_transitions(NL.nesterov_step, gates,
                                            jnp.asarray(0.0, F64), F64))
    for i, s in enumerate(settings):
        assert onp.allclose(A[i], OF.key_aligned_2x2("nesterov", *s),
                            atol=1e-15)
    assert PD.classify(A[0])[0] == "unstable"
    A = onp.asarray(PD.executed_transitions(NL.qhm_step, gates,
                                            jnp.asarray(0.25, F64), F64))
    for i, s in enumerate(settings):
        assert onp.allclose(A[i], OF.key_aligned_2x2("qhm", *s, nu=0.25),
                            atol=1e-15)
        assert PD.classify(A[i])[0] == "stable"


# --------------------------------------------------------- paired analysis --
def test_groups_are_cell_balanced_and_the_jackknife_is_paired():
    batch = TT.generate_batch(4242, 128)
    n = batch["event"].shape[0]
    g = NL.groups_of(n)
    for j in range(NL.N_GROUPS):
        sel = g == j
        for f in onp.unique(batch["family"]):
            for c in (0, 1):
                assert onp.sum(sel & (batch["family"] == f)
                               & (batch["condition"] == c)) == 2
    rng = onp.random.RandomState(1)
    q = (batch["event"] == QUERY).astype(F64)
    correct = (rng.rand(*q.shape) < 0.7) * q
    ce = rng.rand(*q.shape) * q
    full, loo = NL.grouped_aggregates(correct, ce, batch)
    ref = TT.aggregate(correct, ce, batch)
    for f in NL.METRICS.values():
        assert f(full) == f(ref)
    same = NL.paired_seed(full, loo, full, loo)
    assert all(v["difference"] == 0 and v["se"] == 0 for v in same.values())


def test_verdict_rule_uses_the_paired_difference():
    V = NL.verdict
    assert V(0.05, 0.01, [0.04, 0.05, 0.06]) == "BETTER"
    assert V(-0.05, 0.01, [-0.04, -0.05, -0.06]) == "WORSE"
    assert V(0.0, 0.001, [0.001, -0.001, 0.0]) == "EQUIVALENT_WITHIN_MARGIN"
    assert V(0.005, 0.001, [0.004, 0.005, 0.006]) == \
        "EQUIVALENT_WITHIN_MARGIN"                   # precedence, inside margin
    assert V(0.008, 0.003, [0.007, 0.008, 0.009]) == "BETTER_BELOW_MARGIN"
    assert V(0.03, 0.05, [0.1, -0.02, 0.01]) == "INDETERMINATE"
    assert V(0.05, 0.01, [0.1, 0.1, -0.05]) != "BETTER"   # seed consistency


def _comp(labels):
    """labels: {"a_vs_b": {metric: label}} in the ladder orientation."""
    out = {}
    for key, lab in labels.items():
        out[key] = {m: dict(label=lab.get(m, "INDETERMINATE"))
                    for m in NL.METRICS}
    return out


def _gen_vs_all(**labs):
    return {f"{NL.GEN}_vs_{c}": dict(labs) for c in NL.CONTROLS}


def test_recommendation_mapping_uses_each_control_individually():
    everything_better = _comp(_gen_vs_all(revision="BETTER",
                                          immediate_revised="BETTER"))
    assert NL.recommend(everything_better, [])[0] == \
        "GENERALIZED_GP_RETAINS_DISTINCT_ADVANTAGE"
    # beating all but ONE control is not enough (no strongest-comparator
    # choice from results)
    one_short = _gen_vs_all(revision="BETTER")
    one_short[f"{NL.GEN}_vs_{NL.OPERATOR}"] = dict(revision="INDETERMINATE")
    assert NL.recommend(_comp(one_short), [])[0] != \
        "GENERALIZED_GP_RETAINS_DISTINCT_ADVANTAGE"
    # an unavailable Nesterov is not an applicable control
    no_nag = _gen_vs_all(revision="BETTER")
    del no_nag[f"{NL.GEN}_vs_{NL.NAG_ARM}"]
    assert NL.recommend(_comp(no_nag), [NL.NAG_ARM])[0] == \
        "GENERALIZED_GP_RETAINS_DISTINCT_ADVANTAGE"
    nag_best = _comp({
        f"{NL.NAG_ARM}_vs_{NL.NATIVE}": dict(revision="BETTER"),
        f"{NL.NAG_ARM}_vs_{NL.OPERATOR}": dict(revision="BETTER"),
        f"{NL.NAG_ARM}_vs_{NL.TSS}": dict(revision="EQUIVALENT_WITHIN_MARGIN"),
        f"{NL.QHM_ARM}_vs_{NL.NAG_ARM}": dict(revision="WORSE"),
        f"{NL.GEN}_vs_{NL.NAG_ARM}": dict(revision="WORSE_BELOW_MARGIN")})
    assert NL.recommend(nag_best, [])[0] == "RUN_LITERAL_NESTEROV_AT_SCALE"
    reduces = _comp({f"{NL.GEN}_vs_{NL.OPERATOR}": dict(
        revision="EQUIVALENT_WITHIN_MARGIN",
        immediate_revised="EQUIVALENT_WITHIN_MARGIN")})
    assert NL.recommend(reduces, [])[0] == \
        "GENERALIZED_GP_REDUCES_TO_KNOWN_OPTIMIZER"
    assert NL.recommend(_comp({}), [])[0] == "NO_GO"
    labels = {NL.recommend(c, [])[0]
              for c in (everything_better, nag_best, reduces, _comp({}))}
    assert "REDUNDANT_CONTROL_CONFIRMED" not in labels


def test_immediate_claim_needs_every_applicable_control():
    ok = _comp(_gen_vs_all(immediate_revised="BETTER"))
    assert NL.immediate_claim(ok, [])["claim_holds"]
    short = _gen_vs_all(immediate_revised="BETTER")
    short[f"{NL.GEN}_vs_{NL.NAG_ARM}"] = dict(
        immediate_revised="BETTER_BELOW_MARGIN")
    assert not NL.immediate_claim(_comp(short), [])["claim_holds"]


def test_all_fifteen_pairs_and_orientation():
    assert NL.LADDER == (NL.NATIVE, NL.OPERATOR, NL.TSS, NL.NAG_ARM,
                         NL.QHM_ARM, NL.GEN)
    pairs = {frozenset(p) for p in NL.COMPARISONS}
    assert len(NL.COMPARISONS) == 15 and len(pairs) == 15
    comp = _comp({f"{NL.QHM_ARM}_vs_{NL.NAG_ARM}": dict(revision="WORSE")})
    assert NL.label(comp, NL.NAG_ARM, NL.QHM_ARM, "revision") == "BETTER"


def test_combine_reports_every_seed_sign():
    per = {s: {m: dict(difference=d, se=0.001) for m in NL.METRICS}
           for s, d in zip((501, 502, 503), (0.02, -0.01, 0.0))}
    c = NL.combine(per)["immediate_revised"]
    assert c["per_seed_sign"] == {"501": "+", "502": "-", "503": "0"}


# ----------------------------------------- Nesterov applicability records --
def test_episode_violations_and_the_common_stable_subset():
    L = 6
    event = onp.array([[WRITE, QUERY, WRITE, 2, WRITE, QUERY]] * 8)
    gates = onp.zeros((8, L, 4))
    gates[..., 0], gates[..., 1], gates[..., 2], gates[..., 3] = \
        0.5, 0.5, 0.5, 1.0                           # stable: lhs < 0
    gates[5, 2] = (0.9, 1.0, 0.9, 1.9)               # a violating WRITE
    gates[1, 1] = (0.9, 1.0, 0.9, 1.9)               # on a QUERY: ignored
    n_bad, _ = NL.episode_violations(gates, event)
    assert n_bad.tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    keep = NL.stable_subset(8, n_bad)
    assert keep.tolist() == [True] * 4 + [False] * 4  # whole block 1 excluded
    assert NL.stable_subset(8, None).all()


def test_grouped_aggregates_on_a_subset_of_whole_blocks():
    batch = TT.generate_batch(99, 128)
    n = batch["event"].shape[0]
    rng = onp.random.RandomState(2)
    q = (batch["event"] == QUERY).astype(F64)
    correct = (rng.rand(*q.shape) < 0.6) * q
    ce = rng.rand(*q.shape) * q
    viol = onp.zeros(n, dtype=int)
    viol[[3, 50, 130]] = 1
    keep = NL.stable_subset(n, viol)
    assert int((~keep).sum()) == 12
    full, loo = NL.grouped_aggregates(correct, ce, batch, keep)
    sub = {k: v[keep] for k, v in batch.items()}
    ref = TT.aggregate(correct[keep], ce[keep], sub)
    for f in NL.METRICS.values():
        assert f(full) == f(ref) and onp.isfinite(f(full))
    rec = NL.exclusion_record(keep, viol, batch)
    assert rec["excluded_episodes"] == 12 and rec["excluded_blocks"] == 3


# ------------------------------- the completed two-tap arm, unchanged ------
#: sha256 of experiments/prospective_momentum/ordinary.py as executed by the
#: completed temporal-response (f3227df) and retention-aware (84389be) runs
ORDINARY_SHA256 = ("a8f0f31370e1d5bb27bf58cbd8460a268ade34b8eac77960a145d5cc"
                   "8ebdb51c")


def test_two_tap_arm_is_the_completed_implementation():
    import hashlib
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "prospective_momentum",
        "ordinary.py")
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == \
        ORDINARY_SHA256
    assert NL.LAW[NL.OPERATOR] == OD.ORDINARY
    p = params()
    a, b = NL.start_tree(NL.OPERATOR, p), NL.TC.start_tree(NL.OPERATOR, p)
    assert set(a) == set(b) and all(onp.array_equal(onp.asarray(a[k]),
                                                    onp.asarray(b[k]))
                                    for k in a)
    assert float(onp.asarray(a["kappa"])[0]) == 0.0


def test_two_tap_hand_values_are_reproduced_bitwise():
    """The exact 1x1 two-tap values (kappa = 1/2): W = 3/4, 31/32, 259/256."""
    z = jnp.zeros((1, 1), F64)
    k, v, m = jnp.ones((1,), F64), jnp.ones((1,), F64), jnp.asarray(1.0, F64)
    h, one = jnp.asarray(0.5, F64), jnp.asarray(1.0, F64)
    c, ws = (z, z, z), []
    for _ in range(3):
        c = OD.ordinary_step(c, k, v, m, h, h, h, one, h)
        ws.append(float(c[0][0, 0]))
    assert ws == [0.75, 0.96875, 1.01171875]


def test_two_tap_ladder_evaluation_equals_the_completed_evaluation():
    """The ladder evaluates the two-tap arm with EXACTLY the completed
    study's batch function: identical metrics, bit for bit."""
    from experiments.prospective_momentum import temporal_response as TRm
    p = dict(NL.start_tree(NL.OPERATOR, params(seed=13)),
             kappa=jnp.asarray([0.4], F64))
    batch = TT.generate_batch(31, 8)
    m_new, _ = NL.evaluate(NL.OPERATOR, p, batch)
    m_old, _ = TRm.evaluate_arm(NL.OPERATOR, p, batch)
    for f in NL.METRICS.values():
        assert f(m_new) == f(m_old)
    assert m_new["state_norms"] == m_old["state_norms"]
