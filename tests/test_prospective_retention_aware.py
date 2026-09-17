"""Targeted checks for the retention-aware study. CLUSTER-ONLY.

Only what this study adds: the category weighting of the loss terms, the
frozen reference (no gradient), lambda = 0 recovery of the existing training
steps within existing tolerances, finite-first handling of the loss terms,
the constrained-selection wiring, and the reference mapping from saved
artifacts. Everything else is covered by the existing suites, which the
launcher also runs.

Tolerances, unchanged: ID64 = 1e-9 relative; GRAD64 = 1e-8 relative per leaf
with the absolute floor 1e3 eps64 G.
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

from experiments.nested_memory.task import QUERY                  # noqa: E402
from experiments.prospective_momentum import filtered as FL       # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import retention_aware as RA  # noqa
from experiments.prospective_momentum import study as ST          # noqa: E402
from experiments.prospective_momentum import temporal_task as TT  # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

ID64, GRAD64 = 1e-9, 1e-8
EPS64 = float(onp.finfo(onp.float64).eps)


def _source_native_f64():
    run = os.environ.get("PM_SOURCE_RUN", RA.SOURCE_RUN)
    assert os.path.isdir(RS.sources_dir(run)), f"source run {run} REQUIRED"
    with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
        manifest = json.load(fh)
    e = RS.find_entry(manifest, RA.SOURCE_DEV, TC.SOURCE_FAMILY)
    return {k: jnp.asarray(onp.asarray(v), dtype=jnp.float64)
            for k, v in RS.restore_source(run, e).items()}


def _batch(seed=991_001):
    b = TT.generate_batch(seed, 8)
    return b, ST.to_jax(b), {k: jnp.asarray(b[k]) for k in RA.META_FIELDS}


def _perturbed(p, scale=0.05, seed=3):
    """Perturb backbone leaves only; the filter coefficients stay in their
    domain."""
    rs = onp.random.RandomState(seed)
    return {k: (v if k in FL.LEAVES
                else v + jnp.asarray(scale * rs.randn(*v.shape)))
            for k, v in p.items()}


# ============================ 1. category weighting ==========================
def test_loss_categories_use_the_evaluation_weighting():
    b, eps, meta = _batch()
    rs = onp.random.RandomState(1)
    q = b["event"] == QUERY
    ce = rs.rand(*q.shape) * q
    cu, cr, n_min = RA.category_ce(jnp.asarray(ce), jnp.asarray(q), meta)
    host = TT.aggregate(onp.zeros(q.shape), ce, b)
    want_u = 0.5 * (host["revision"]["ce_parts"]["untouched_probe"]
                    + host["revision"]["ce_parts"]["late_untouched"])
    want_r = host["recall"]["macro_cross_entropy"]
    assert abs(float(cu) - want_u) <= ID64 * max(abs(want_u), 1.0)
    assert abs(float(cr) - want_r) <= ID64 * max(abs(want_r), 1.0)
    assert float(n_min) == 4.0             # 8 per family -> 4 per block cell
    assert not set(RA.META_FIELDS) & set(eps)


# ============================ 2. the frozen reference ========================
def test_reference_receives_no_gradient_and_penalties_are_hinges():
    pn = _source_native_f64()
    ref = pn
    p = _perturbed(TC.start_tree(RA.GEN, pn))
    _, eps, meta = _batch(991_002)
    g_ref = jax.grad(lambda r: RA.ra_batch_loss(FL.FILTERED, 1.0, p, r, eps,
                                                meta)[0])(ref)
    assert all(float(onp.max(onp.abs(onp.asarray(v)))) == 0.0
               for v in g_ref.values())
    total, aux = RA.ra_batch_loss(FL.FILTERED, 1.0, p, ref, eps, meta)
    t = {k: float(v) for k, v in aux["terms"].items()}
    assert t["penalty_untouched"] == max(0.0, t["ce_untouched"]
                                         - t["reference_untouched"])
    assert t["penalty_recall"] == max(0.0, t["ce_recall"]
                                      - t["reference_recall"])
    want = t["base"] + t["penalty_untouched"] + t["penalty_recall"]
    assert abs(float(total) - want) <= ID64 * max(abs(want), 1.0)
    # the candidate's gradient equals base + penalty with the reference
    # terms held as CONSTANTS (independently assembled)
    ru, rr = t["reference_untouched"], t["reference_recall"]
    q = eps["event"] == QUERY

    def manual(pp):
        base, a = TC.batch_loss_f(FL.FILTERED, pp, eps)
        cu, cr, _ = RA.category_ce(a["ce"], q, meta)
        return (base + jnp.maximum(cu - ru, 0.0)
                + jnp.maximum(cr - rr, 0.0))
    g_ra = jax.grad(lambda pp: RA.ra_batch_loss(FL.FILTERED, 1.0, pp, ref,
                                                eps, meta)[0])(p)
    g_man = jax.grad(manual)(p)
    G = math.sqrt(sum(float(onp.sum(onp.asarray(v) ** 2))
                      for v in g_man.values()))
    for k in g_man:
        d = float(onp.linalg.norm(onp.asarray(g_ra[k]) - onp.asarray(g_man[k])))
        n = float(onp.linalg.norm(onp.asarray(g_man[k])))
        assert d <= max(GRAD64 * n, 1e3 * EPS64 * G), (k, d, n)


# ============================ 3. lambda = 0 recovery =========================
@pytest.mark.parametrize("arm", [RA.GEN, RA.NATIVE])
def test_lambda_zero_recovers_the_existing_step(arm):
    pn = _source_native_f64()
    p = TC.start_tree(arm, pn)
    opt = ST.TX.init(p)
    _, eps, meta = _batch(991_003)
    lr = jnp.asarray(0.01, dtype=jnp.float64)
    rule = RA.LAW_OF[arm]
    new = RA.train_step_ra(rule, RA.FROZEN_LEAVES[arm], 0.0, p, opt,
                           _perturbed(pn), eps, meta, lr)
    old = (TC.train_step_filtered(rule, RA.FROZEN_LEAVES[arm], p, opt, eps,
                                  lr)
           if rule == FL.FILTERED else ST.train_step(rule, p, opt, eps, lr))
    for x in (new[2], old[2]):
        assert onp.isfinite(float(x))
    assert abs(float(new[2]) - float(old[2])) <= ID64 * abs(float(old[2]))
    for k in old[0]:
        a, b = onp.asarray(new[0][k]), onp.asarray(old[0][k])
        assert onp.all(onp.isfinite(a)) and onp.all(onp.isfinite(b))
        d, n = float(onp.linalg.norm(a - b)), float(onp.linalg.norm(b))
        assert d <= max(ID64 * n, 1e3 * EPS64), (arm, k, d, n)
    assert float(new[10]["base"]) == pytest.approx(float(old[2]), rel=ID64)


# ============================ 4. finite-first loss handling =================
def _rec(**terms):
    base = dict(base=1.0, ce_untouched=1.0, ce_recall=1.0,
                reference_untouched=0.9, reference_recall=0.9,
                penalty_untouched=0.1, penalty_recall=0.1, min_cell_count=4.0)
    base.update(terms)
    return dict(update=7, executed_filter_ok=True, terms=base)


def test_step_failure_rejects_nonfinite_or_empty_loss_terms():
    good = {k: 0.5 for k in ST.SCALAR_NAMES}
    assert RA.step_failure_ra(RA.NATIVE, good, _rec()) is None
    for k in ("base", "ce_untouched", "ce_recall", "reference_untouched",
              "reference_recall", "penalty_untouched", "penalty_recall"):
        assert RA.step_failure_ra(RA.NATIVE, good,
                                  _rec(**{k: float("nan")})), k
    assert RA.step_failure_ra(RA.NATIVE, good, _rec(min_cell_count=0.0))
    missing = _rec()
    del missing["terms"]["reference_recall"]
    assert RA.step_failure_ra(RA.NATIVE, good, missing)
    assert RA.step_failure_ra(RA.NATIVE, dict(good, loss=float("inf")),
                              _rec())


def test_nonfinite_reference_propagates_to_a_rejected_term():
    pn = _source_native_f64()
    bad_ref = dict(pn, readout_b=jnp.full_like(pn["readout_b"], jnp.nan))
    _, eps, meta = _batch(991_004)
    _, aux = RA.ra_batch_loss("momentum_delta", 1.0, pn, bad_ref, eps, meta)
    t = {k: float(v) for k, v in aux["terms"].items()}
    assert not onp.isfinite(t["reference_untouched"])
    assert RA.step_failure_ra(RA.NATIVE, {k: 0.5 for k in ST.SCALAR_NAMES},
                              dict(update=0, executed_filter_ok=True,
                                   terms=t))


# ============================ 5. constrained-selection wiring ===============
def _rows(spec):
    rows = []
    for arm, slots in spec.items():
        for tag, vals in slots:
            rows.append(dict(rule=arm, config=tag, lr=RA.LR, validation=[
                dict(update=u, primary=pr, revision_ce=1.0, retention=rt,
                     recall=rc, immediate_revision=0.5, later=0.5,
                     accepted=True, params_file=f"{arm}{tag}{u}")
                for u, pr, rt, rc in vals]))
    return rows


def _flat(pr, rt, rc):
    return [(u, pr, rt, rc) for u in RA.VAL_AT]


def _spec():
    return {
        RA.NATIVE: [("lambda0", _flat(0.70, 0.69, 0.79)),
                    ("lambda1", _flat(0.69, 0.70, 0.80))],
        RA.TSS: [("lambda0", _flat(0.72, 0.68, 0.79)),
                 ("lambda1", [(0, .60, .72, .80), (25, .70, .69, .79),
                              (50, .71, .70, .79), (100, .705, .70, .79),
                              (200, .73, .68, .78)])],
        RA.GEN: [("lambda0", _flat(0.74, 0.60, 0.70)),
                 ("lambda1", _flat(0.73, 0.65, 0.75))],
        RA.OPERATOR: [("lambda0", _flat(0.71, 0.70, 0.80)),
                      ("lambda1", _flat(0.70, 0.71, 0.81))],
    }


def test_constrained_selection_records_infeasibility_and_controls():
    status = {}
    endpoints, plan = RA.select_ra(_rows(_spec()), status)
    ref = endpoints[RA.NATIVE]["reference"]
    assert (ref["config"], ref["primary"]) == ("lambda0", 0.70)
    assert status["selection"]["r_native"] == 0.69
    tss = endpoints[RA.TSS]
    # feasible needs retention >= .69 and recall >= .79: lambda1 u50 (0.71)
    assert (tss["constrained"]["config"], tss["constrained"]["update"]) == (
        "lambda1", 50)
    assert (tss["unconstrained"]["config"], tss["unconstrained"]["update"]) \
        == ("lambda1", 200)
    assert tss["lambda0"]["config"] == "lambda0"
    assert tss["lambda0"]["update"] == 0          # all ties -> fewer updates
    assert endpoints[RA.GEN]["constrained"] is None       # INFEASIBLE
    assert endpoints[RA.GEN]["unconstrained"]["config"] == "lambda0"
    assert endpoints[RA.OPERATOR]["constrained"]["config"] == "lambda0"
    # infeasible generalized: a DEPLOYMENT fallback (here the feasible literal
    # TSS checkpoint, higher than native on development), never a trained win
    assert plan[RA.GEN]["choice"] == "tss"
    assert plan[RA.GEN]["trained_feasible"] is False
    table = {r["arm"]: r for r in status["selection"]["table"]}
    assert table[RA.GEN]["infeasible"] is True


def test_update_zero_is_shared_and_unaccepted_checkpoints_block():
    rows = _rows(_spec())
    cps = RA.checkpoints_ra(rows, RA.GEN)
    assert len(cps) == 9 and sum(c["update"] == 0 for c in cps) == 1
    assert [c["config"] for c in cps if c["update"] == 0] == ["lambda0"]
    rows[5]["validation"][3]["accepted"] = False
    assert RA.select_ra(rows, {}) == (None, None)


def test_final_plan_and_primary_screen_availability():
    endpoints, _ = RA.select_ra(_rows(_spec()), {})
    plan = RA.final_plan_ra(endpoints)
    by = {(t["arm"], t["config"]): t for t in plan}
    assert by[(RA.TSS, "lambda1")]["keep_at"] == [50, 200]
    assert (RA.role_id(RA.GEN, "constrained"), 0) not in \
        by.get((RA.GEN, "lambda0"), {}).get("endpoints", [])
    assert len(plan) <= len(RA.TRAINED_ARMS) * len(RA.LAMBDAS)
    assert RA.PLANNED["max_total_updates"] == 6400
    sc = RA.screen_ra([], endpoints)
    gt = [c for c in sc["primary"]
          if c["name"] == "generalized_versus_literal_tss"][0]
    assert gt["available"] is False and gt["aggregate_screen_passed"] is False
    assert "INFEASIBLE" in gt["reason"]
    assert RA.stream_overlaps() == []


# ============================ 6. references from saved artifacts ===========
def _doc(run):
    def pf(name):
        return os.path.join(run, "params", name)
    sel = dict(config="B", lr=0.01, update=200,
               params_file=pf(RA.REFERENCE_FILES[500]), primary=0.72,
               retention=0.71, recall=0.79, immediate_revision=0.7,
               later=0.75)
    final = [dict(rule=RA.NATIVE, seed=s, config="B", lr=0.01, update=200,
                  endpoint_kind="trained_named_family_endpoint",
                  endpoint_params_file=pf(RA.REFERENCE_FILES[s]),
                  final_validation_summary=dict(
                      primary=0.7, retention=0.7, recall=0.8,
                      immediate_revision=0.7, later=0.74))
             for s in RA.SOURCE_FINAL]
    return dict(complete=True, failed=None, study_status="PASS",
                selection=dict(selected={RA.NATIVE: sel}),
                frozen_before_finals=dict(selection={RA.NATIVE: dict(sel)}),
                final=final)


def test_reference_mapping_is_derived_and_never_substituted(tmp_path):
    run = str(tmp_path)
    m = RA.reference_mapping(run, _doc(run))
    assert sorted(m) == [500, 501, 502, 503]
    assert m[500]["stream"] == "dev_validation"
    assert m[502]["path"].endswith(RA.REFERENCE_FILES[502])
    bad = _doc(run)
    bad["final"] = [r for r in bad["final"] if r["seed"] != 502]
    with pytest.raises(RA.ReferenceRefusal):
        RA.reference_mapping(run, bad)
    swapped = _doc(run)
    swapped["final"][0]["endpoint_params_file"] = os.path.join(
        run, "params", RA.REFERENCE_FILES[502])
    with pytest.raises(RA.ReferenceRefusal):
        RA.reference_mapping(run, swapped)
    changed = _doc(run)
    changed["frozen_before_finals"]["selection"][RA.NATIVE]["update"] = 100
    with pytest.raises(RA.ReferenceRefusal):
        RA.reference_mapping(run, changed)
    recipe = _doc(run)
    for d in (recipe["selection"]["selected"][RA.NATIVE],
              recipe["frozen_before_finals"]["selection"][RA.NATIVE]):
        d["config"] = "A"
    with pytest.raises(RA.ReferenceRefusal):
        RA.reference_mapping(run, recipe)
    failed = _doc(run)
    failed["study_status"] = "FAILED"
    with pytest.raises(RA.ReferenceRefusal):
        RA.reference_mapping(run, failed)


# ============================ 7. preflight covers both slots (review R1) ====
def _stub_preflight(tmp_path, fail=None, retrace=None):
    """Runs the REAL preflight orchestration with stubbed numerics: every
    (family, slot) must be exercised, and a lambda = 0 failure or retrace
    must surface under that slot."""
    calls, counter = [], {"n": 0}

    def factory(lam, refs):
        def step(arm, p, opt, seed, u, lr, hist):
            calls.append((arm, lam, u))
            if retrace == (arm, lam) and u >= 2:
                counter["n"] += 1
            terms = dict(base=1.0, ce_untouched=1.0, ce_recall=1.0,
                         reference_untouched=1.0, reference_recall=1.0,
                         penalty_untouched=0.0, penalty_recall=0.0,
                         min_cell_count=4.0)
            if fail == (arm, lam):
                terms["penalty_recall"] = float("nan")
            rec = dict(update=u, executed_filter_ok=True, jury_min=0.5,
                       proc_max_abs=1.0, executed={}, terms=terms,
                       **{k: 0.1 for k in FL.LEAVES},
                       **{"grad_" + k: 0.0 for k in FL.LEAVES})
            hist.append(rec)
            return p, opt, {k: 0.5 for k in ST.SCALAR_NAMES}, rec
        return step
    status = {}
    total, retraced, failures = RA.preflight_ra(
        {RA.SOURCE_DEV: {}}, {}, None, str(tmp_path), status,
        step_factory=factory, evaluate=lambda arm, p, v: ({}, None),
        checkpoint_failure=lambda *a: None,
        save_tree=lambda path, tree: None,
        cache_size=lambda: counter["n"],
        start_tree=lambda arm, p0: {"w": jnp.zeros((1,))})
    return calls, status, total, retraced, failures


def test_preflight_exercises_every_family_and_slot(tmp_path):
    calls, status, total, retraced, failures = _stub_preflight(tmp_path)
    seen = {(a, lam) for a, lam, _ in calls}
    assert seen == {(a, lam) for a, _, _, _ in RA.TRAINED_ARMS
                    for _, lam in RA.LAMBDAS}
    assert all(sum(1 for c in calls if c[:2] == k) == 7 for k in seen)
    assert len(status["preflight"]["rows"]) == 8
    assert failures == [] and retraced is False and onp.isfinite(total)
    assert "not charged again" in status["preflight"]["coverage"]


def test_a_lambda_zero_failure_or_retrace_cannot_be_hidden(tmp_path):
    _, status, _, _, failures = _stub_preflight(
        tmp_path, fail=(RA.GEN, 0.0))
    assert any(f.startswith(f"{RA.GEN}/lambda0") for f in failures)
    assert not any(f.startswith(f"{RA.GEN}/lambda1") for f in failures)
    _, status, _, retraced, failures = _stub_preflight(
        tmp_path, retrace=(RA.NATIVE, 0.0))
    assert retraced is True
    rows = {(r["arm"], r["config"]): r for r in status["preflight"]["rows"]}
    assert rows[(RA.NATIVE, "lambda0")]["retraced"] is True
    assert rows[(RA.NATIVE, "lambda1")]["retraced"] is False


# ============================ 8. digest pairing (review D1) =================
def test_digest_pairs_saved_full_precision_values():
    from experiments.prospective_momentum import retention_aware_summary as RS_
    a = dict(primary=0.712345678, retention_revision_untouched=0.70001,
             recall_overall=0.79)
    b = dict(primary=0.702345678, retention_revision_untouched=0.70002,
             recall_overall=0.79)
    d = RS_.paired_differences({("x", 501): a, ("y", 501): b}, "x", "y",
                               [501, 502])
    assert d[501]["retention"] == a["retention_revision_untouched"] - \
        b["retention_revision_untouched"]
    assert d[501]["revision"] == a["primary"] - b["primary"]
    assert d[502]["missing"] == ["x", "y"]
