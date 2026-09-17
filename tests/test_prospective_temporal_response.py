"""Focused checks for the temporal-response study. CLUSTER-ONLY.

Scope: the new task (schedule, exact balance, oracle, the two distinct times,
model-input isolation), the declared metric weighting, the mechanism
diagnostic (exact impulse responses, open loop, closed-loop first-write
agreement), the selection and matched-operating-point rules, the final plan,
the aggregate screen, the runner's checkpoints and kept endpoints, and the
opt-in state trace. The law, gate, repair, training steps and recovery are
covered unchanged by the TSS-containment checks, which the launcher also runs.

Tolerances, unchanged: ID64 = 1e-9 relative. PM_SOURCE_RUN must point at the
read-only replication run; a missing source is a FAILURE.
"""

import json
import os
import sys
from fractions import Fraction

import jax
import numpy as onp
import pytest
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.nested_memory.task import IDLE, QUERY, WRITE     # noqa: E402
from experiments.prospective_momentum import filtered as FL       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import study as ST          # noqa: E402
from experiments.prospective_momentum import temporal_diagnostic as TD  # noqa
from experiments.prospective_momentum import temporal_response as TR  # noqa
from experiments.prospective_momentum import temporal_task as TT  # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

ID64 = 1e-9
F64 = onp.float64


def _rel(a, b):
    a, b = onp.asarray(a, F64), onp.asarray(b, F64)
    assert onp.all(onp.isfinite(a)) and onp.all(onp.isfinite(b))
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def _source_native_f64():
    run = os.environ.get("PM_SOURCE_RUN", TR.SOURCE_RUN)
    assert os.path.isdir(RS.sources_dir(run)), \
        f"read-only source run {run} is REQUIRED"
    with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
        manifest = json.load(fh)
    e = RS.find_entry(manifest, TR.SOURCE_DEV, TC.SOURCE_FAMILY)
    return {k: jnp.asarray(onp.asarray(v), dtype=jnp.float64)
            for k, v in RS.restore_source(run, e).items()}


# =========================== 1. the task ====================================
def test_schedule_is_exactly_balanced_and_the_oracle_exact():
    b = TT.generate_batch(987_001, 8)
    c = TT.structure_check(b)
    assert c["length"] == 64 and c["queries_per_sequence"] == [20]
    assert c["oracle_exact"] and c["block_cells_balanced"]
    assert c["block_cell_size"] == [2]           # 8 per family / 4
    assert c["query_value_field_is_absent"] and c["idle_value_field_is_absent"]
    assert c["every_query_has_a_label"] and c["no_label_outside_queries"]
    assert c["writes_per_sequence_by_condition"] == {
        "idle_gap": [18], "intervening_writes": [44]}
    assert c["idle_per_sequence_by_condition"] == {
        "idle_gap": [26], "intervening_writes": [0]}
    assert set(c["late_counts"].values()) == {20}  # 4 episodes x 5
    assert TT.N_FILL == 26 and TT.LATE_START == 54 and TT.BLOCK_START == 13


def test_every_block_has_the_declared_shape():
    b = TT.generate_batch(987_002, 4)
    for e in range(b["event"].shape[0]):
        cond = TT.CONDITIONS[b["condition"][e]]
        seen = []
        for d in TT.DELAYS:
            q = onp.flatnonzero((b["delay"][e] == d))
            assert len(q) == 2 and q[1] == q[0] + 1
            r = q[0] - d
            assert b["event"][e, r] == WRITE
            assert sorted(b["kind"][e, q].tolist()) == [0, 1]
            assert b["key_id"][e, q[b["kind"][e, q] == 0][0]] == \
                b["key_id"][e, r]
            fill = b["event"][e, r + 1:r + d]
            assert onp.all(fill == (IDLE if cond == "idle_gap" else WRITE))
            seen.append(r)
        assert sorted(seen)[0] == TT.BLOCK_START
        assert onp.all(b["event"][e, TT.LATE_START:] == QUERY)


def test_since_revision_and_age_are_distinct_times():
    b = TT.generate_batch(987_003, 8)
    q = b["event"] == QUERY
    rev = q & (b["kind"] == 0)
    unt = q & (b["kind"] == 1)
    assert onp.array_equal(b["age"][rev], b["since_revision"][rev])
    assert onp.all(b["age"][unt] > b["since_revision"][unt])
    assert onp.all(b["since_revision"][rev] == b["delay"][rev]
                   + b["offset"][rev])


def test_offsets_are_balanced_within_every_cell():
    b = TT.generate_batch(987_004, 8)
    counts = TT.cell_counts(b)
    assert set(v for k, v in counts.items() if "/d" in k) == {2}


def test_families_rewrite_as_declared():
    b = TT.generate_batch(987_005, 8)
    for e in range(b["event"].shape[0]):
        first = {}
        for t in range(TT.N_TARGETS):
            first[int(b["key_id"][e, t])] = int(b["val_id"][e, t])
        for t in onp.flatnonzero(b["kind"][e] == 0):
            key, lab = int(b["key_id"][e, t]), int(b["label"][e, t])
            if b["family"][e] == 0:                   # recall
                assert lab == first[key]
            else:                                     # revision
                assert lab != first[key]


def test_batch_size_must_allow_exact_balance():
    with pytest.raises(ValueError):
        TT.generate_batch(1, 6)


def test_only_model_inputs_reach_the_model(monkeypatch):
    assert TT.MODEL_INPUTS == ("key_id", "val_id", "event", "label")
    b = TT.generate_batch(987_006, 4)
    assert set(ST.to_jax(b)) <= set(TT.MODEL_INPUTS)
    seen = []
    real = ST.eval_batch

    def spy(rule, p, eps):
        seen.append(set(eps))
        return real(rule, p, eps)
    monkeypatch.setattr(ST, "eval_batch", spy)
    TR.evaluate_arm(TR.NATIVE, _source_native_f64(), b)
    assert seen and all(s == set(TT.MODEL_INPUTS) for s in seen)


# =========================== 2. the metric weighting ========================
def test_equal_cell_weighting_and_declared_aggregates():
    b = TT.generate_batch(987_007, 8)
    q = b["event"] == QUERY
    ce = onp.zeros(b["event"].shape)
    correct = (q & (b["kind"] == 0) & (b["delay"] == 1)).astype(float)
    m = TT.aggregate(correct, ce, b)
    for fam in ("recall", "revision"):
        assert m[fam]["parts"]["revised_probe"] == pytest.approx(0.2)
        assert m[fam]["parts"]["untouched_probe"] == 0.0
        assert m[fam]["macro_accuracy"] == pytest.approx(0.05)
    assert m["primary"] == pytest.approx(0.05)
    assert m["immediate_revision"] == 1.0
    assert m["later"] == 0.0 and m["retention_revision_untouched"] == 0.0
    allc = q.astype(float)
    m1 = TT.aggregate(allc, ce, b)
    for k in TT.REQUIRED_SCALARS:
        if k != "revision_ce":
            assert m1[k] == 1.0, k
    # recall later = mean(block d>=4 of both kinds, late macro)
    cor = (q & (b["family"] == 0)[:, None] & (b["kind"] >= 2)).astype(float)
    m2 = TT.aggregate(cor, ce, b)
    assert m2["later_parts"]["recall_later"] == pytest.approx(0.5)
    assert TT.metrics_finite(m1)


def test_metrics_finite_rejects_nan_and_empty_cells():
    b = TT.generate_batch(987_008, 4)
    q = b["event"] == QUERY
    m = TT.aggregate(q.astype(float), onp.zeros(q.shape), b)
    assert TT.metrics_finite(m)
    name = next(iter(m["revision"]["cells"]))
    bad = json.loads(json.dumps(m))
    bad["revision"]["cells"][name]["accuracy"] = float("nan")
    assert not TT.metrics_finite(bad)
    empty = json.loads(json.dumps(m))
    empty["recall"]["cells"][name] = dict(accuracy=None, cross_entropy=None,
                                          n=0)
    assert not TT.metrics_finite(empty)


# =========================== 3. the mechanism diagnostic ====================
def test_exact_impulse_responses_are_the_declared_ones():
    tss = TD.exact_impulse(0, 0, 1, 6)
    gen = TD.exact_impulse(Fraction(1, 4), 0, Fraction(1, 2), 6)
    assert tss == [2, -1, 0, 0, 0, 0]
    assert gen == [2, Fraction(-2, 3), Fraction(-2, 3), Fraction(2, 9),
                   Fraction(2, 9), Fraction(-2, 27)]
    kg = TD.exact_coefficients(Fraction(1, 4), 0, Fraction(1, 2))
    assert (kg["A"], kg["a"], kg["b"], kg["c"], kg["d"]) == (
        Fraction(3, 4), 0, Fraction(1, 3), 2, Fraction(2, 3))
    # equal immediate amplitude and unit DC gain, different dynamics
    for M, g, T in ((0, 0, 1), (Fraction(1, 4), 0, Fraction(1, 2))):
        k = TD.exact_coefficients(M, g, T)
        assert k["c"] == 2
        assert (k["c"] - k["d"]) / (1 - k["a"] + k["b"]) == 1


def test_open_loop_production_matches_the_exact_response():
    fails, rows = TD.open_loop(jnp.float64)
    assert fails == [], fails
    assert rows["tss_T1"]["executed_response"][:3] == [2.0, -1.0, 0.0]
    for name, r in rows.items():
        print(f"  {name}: max open-loop error {r['max_abs_error']:.3e}")


def test_closed_loop_first_write_agrees_before_later_readouts():
    pn = _source_native_f64()
    b = TT.generate_batch(987_009, 4)
    fails, rep = TD.closed_loop(pn, b)
    assert fails == [], fails
    fw = rep["first_write"]
    print(f"  first-write agreement {fw}")
    assert fw["max_relative_difference"] <= TD.TRAJ32
    assert fw["nonzero_first_write"]
    d = rep["differences"]["generalized_M1q_T1h_minus_tss_T1"]
    for g, curves in d.items():
        p0 = abs(curves["revised_label_probability"][0])
        assert p0 <= 1e-9, (g, p0)             # identical after the write
    later = max(abs(x) for g in d.values()
                for x in g["revised_label_probability"][1:])
    print(f"  largest later revised-probability difference {later:.3e}")
    assert later > 0.0                          # the dynamics then differ
    assert rep["executed_coefficients"]["tss_T1"]["c"] == 2.0
    assert rep["executed_coefficients"]["generalized_M1q_T1h"]["c"] == 2.0


def test_trace_is_opt_in_and_shares_the_scan():
    pn = _source_native_f64()
    b = TT.generate_batch(987_010, 4)
    ep = {k: jnp.asarray(b[k][0]) for k in TT.MODEL_INPUTS}
    p = PM.add_extension(pn, FL.FILTERED)
    plain = PM.rollout(FL.FILTERED, p, ep)
    traced = PM.rollout(FL.FILTERED, p, ep, trace=True)
    assert "W_trace" not in plain
    assert _rel(traced["logits"], plain["logits"]) < ID64
    assert onp.array_equal(onp.asarray(traced["W_trace"][-1]),
                           onp.asarray(traced["final_carry"][0]))
    with pytest.raises(ValueError):
        PM.rollout("momentum_delta", pn, ep, trace=True)


# =========================== 4. selection and the matched rule ==============
def _rows(spec):
    rows = []
    for arm, configs in spec.items():
        for config, lr, vals in configs:
            rows.append(dict(rule=arm, config=config, lr=lr, validation=[
                dict(update=u, primary=pr, revision_ce=1.0, retention=rt,
                     recall=rc, immediate_revision=im, later=la,
                     accepted=True, params_file=f"{arm}{config}{u}")
                for u, pr, rt, rc, im, la in vals]))
    return rows


def _flat(pr, rt=0.6, rc=0.7, im=0.5, la=0.5):
    return [(u, pr, rt, rc, im, la) for u in TR.VAL_AT]


def _spec():
    return {
        TR.NATIVE: [("A", 0.003, _flat(0.50)), ("B", 0.01, _flat(0.55))],
        TR.TSS: [("A", 0.003, _flat(0.60, im=0.80, la=0.40)),
                 ("B", 0.01, [(0, 0.50, .6, .7, .50, .5),
                              (25, 0.62, .5, .7, .81, .41),
                              (50, 0.63, .5, .7, .82, .42),
                              (100, 0.66, .5, .7, .85, .43),
                              (200, 0.70, .55, .69, .90, .44)])],
        TR.GEN: [("A", 0.003, [(0, 0.50, .6, .7, .50, .5),
                               (25, 0.61, .6, .7, .90, .47),
                               (50, 0.64, .6, .7, .905, .49),
                               (100, 0.66, .6, .7, .92, .46),
                               (200, 0.69, .6, .7, .95, .48)]),
                 ("B", 0.01, _flat(0.72, im=0.99, la=0.30))],
        TR.OPERATOR: [("A", 0.003, _flat(0.71)), ("B", 0.01, _flat(0.73))],
    }


def test_main_selection_is_unconstrained_and_identical_for_every_family():
    status = {}
    sel, plan = TR.select_main(_rows(_spec()), status)
    assert sel[TR.NATIVE]["primary"] == 0.55
    assert sel[TR.TSS]["update"] == 200 and sel[TR.TSS]["lr"] == 0.01
    assert sel[TR.GEN]["lr"] == 0.01          # 0.72 at every update B...
    assert sel[TR.GEN]["update"] == 25        # ...ties -> fewer updates
    assert sel[TR.OPERATOR]["primary"] == 0.73
    # deployment feasibility is separate: TSS loses retention on development
    assert sel[TR.TSS]["feasible"] is False
    assert "no retention or recall constraint" in status["selection"]["rule"]
    assert plan[TR.TSS]["choice"] == "native"


def test_matched_operating_point_rule_is_frozen_and_never_relaxed():
    status = {}
    sel, _ = TR.select_main(_rows(_spec()), status)
    # reference: selected TSS immediate 0.90 -> band [0.90, 0.91]
    m = TR.select_matched(_rows(_spec()), sel, status)
    rec = status["matched_operating_point"]
    assert rec["available"] and rec["n_feasible"] == 2
    assert (m["config"], m["update"]) == ("A", 50)   # later 0.49 > 0.47
    # nothing in the band: reported unavailable, no relaxation
    spec = _spec()
    spec[TR.GEN] = [("A", 0.003, _flat(0.69, im=0.95)),
                    ("B", 0.01, _flat(0.72, im=0.99))]
    st2 = {}
    sel2, _ = TR.select_main(_rows(spec), st2)
    assert TR.select_matched(_rows(spec), sel2, st2) is None
    assert st2["matched_operating_point"]["available"] is False


def test_an_unaccepted_checkpoint_blocks_selection():
    rows = _rows(_spec())
    rows[3]["validation"][2]["accepted"] = False
    assert TR.select_main(rows, {}) == (None, None)


def test_final_plan_merges_learning_rates_and_bounds_the_work():
    sel, _ = TR.select_main(_rows(_spec()), {})
    same = dict(sel[TR.GEN], update=0)
    plan = TR.final_plan(sel, same)
    gen = [t for t in plan if t["arm"] == TR.GEN]
    assert len(gen) == 1 and gen[0]["keep_at"] == [0, 25]
    other = dict(sel[TR.GEN], config="A", lr=0.003, update=50)
    plan2 = TR.final_plan(sel, other)
    assert len([t for t in plan2 if t["arm"] == TR.GEN]) == 2
    work = sum(t["updates"] for t in plan2) * len(TR.SOURCE_FINAL)
    assert work <= 3000
    assert TR.planned_work()["max_total_updates"] == 4600
    assert TR.stream_overlaps() == []


# =========================== 5. screens =====================================
def _heldout(b, frac):
    q = b["event"] == QUERY
    rs = onp.random.RandomState(int(frac * 1e6))
    correct = (q & (rs.rand(*q.shape) < frac)).astype(float)
    return TT.aggregate(correct, onp.zeros(q.shape), b)


def test_aggregate_screen_and_secondary_cells():
    b = TT.generate_batch(987_011, 8)
    rows = []
    for s in TR.SOURCE_FINAL:
        rows += [dict(rule=TR.GEN, seed=s, heldout=_heldout(b, 0.9)),
                 dict(rule=TR.TSS, seed=s, heldout=_heldout(b, 0.5)),
                 dict(rule=TR.NATIVE, seed=s, heldout=_heldout(b, 0.5)),
                 dict(rule=TR.OPERATOR, seed=s, heldout=_heldout(b, 0.95)),
                 dict(rule=TR.ANCHOR, seed=s, heldout=_heldout(b, 0.4))]
    sc = {c["name"]: c for c in TR.screen(rows)["comparisons"]}
    g = sc["generalized_versus_literal_tss"]
    manual = (g["complete_paired_seeds"]
              and g["mean_primary_difference"] >= 0.01
              and g["positive_in_all_seeds"]
              and g["retention_difference"] >= 0
              and g["recall_difference"] >= 0)
    assert g["aggregate_screen_passed"] == bool(manual) is True
    assert sc["generalized_versus_learned_operator"][
        "aggregate_screen_passed"] is False
    assert g["secondary_cells"]["revision"]["by_delay"]["revised_probe"]


# =========================== 6. the runner ==================================
def _small():
    return TT.generate_batch(987_012, 4)


def test_zero_update_endpoint_is_kept_and_valid(tmp_path):
    pn = _source_native_f64()
    status = {"incomplete": []}
    r = TR.run_one(TR.GEN, "A", 0.003, TR.SOURCE_DEV,
                   TC.start_tree(TR.GEN, pn), _small(), 0, str(tmp_path),
                   1e18, 30.0, status, "fixture", "src", None, None, (0,))
    rec, p, kept = r
    assert rec["invalid"] is None, rec["invalid"]
    assert 0 in kept and rec["validation"][0]["accepted"]


def test_kept_intermediate_endpoint_matches_its_saved_checkpoint(tmp_path):
    pn = _source_native_f64()
    start = TC.start_tree(TR.GEN, pn)
    status = {"incomplete": []}
    rec, p, kept = TR.run_one(TR.GEN, "A", 0.003, TR.SOURCE_DEV, start,
                              _small(), 2, str(tmp_path), 1e18, 30.0, status,
                              "fixture", "src", None, (0, 1, 2), (1,))
    assert rec["invalid"] is None, rec["invalid"]
    v1 = [v for v in rec["validation"] if v["update"] == 1][0]
    saved = TC.load_params(v1["params_file"], start)
    for k in start:
        assert onp.array_equal(onp.asarray(saved[k]), onp.asarray(kept[1][k]))
    m, _ = TR.evaluate_arm(TR.GEN, kept[1], _small())
    for key, want in (("primary", v1["primary"]),
                      ("immediate_revision", v1["immediate_revision"]),
                      ("later", v1["later"])):
        assert abs(m[key] - want) <= ID64 * max(abs(want), 1.0), key


def test_keep_at_must_be_a_checkpoint(tmp_path):
    pn = _source_native_f64()
    with pytest.raises(ValueError):
        TR.run_one(TR.NATIVE, "A", 0.003, TR.SOURCE_DEV, dict(pn), _small(),
                   2, str(tmp_path), 1e18, 30.0, {"incomplete": []},
                   "fixture", "src", None, (0, 2), (1,))
