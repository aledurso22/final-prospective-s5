"""Digest for a temporal-response run. Written BEFORE the run.

Read-only: saved JSON only; no JAX, no model, no recomputation.

    python -m experiments.prospective_momentum.temporal_response_summary <run_dir>
"""

import json
import os
import sys

from .summary import f

FAMS = ("recall", "revision")
OFFSETS = (0, 1, 2, 3, 5, 9, 16, 17)


def pct(x):
    return "   n/a" if x is None else f"{100 * x:6.2f}"


def agg_line(m):
    return (f"primary {pct(m.get('primary'))} retention "
            f"{pct(m.get('retention_revision_untouched', m.get('retention')))}"
            f" recall {pct(m.get('recall_overall', m.get('recall')))} "
            f"immediate {pct(m.get('immediate_revision'))} later "
            f"{pct(m.get('later'))}")


def delay_lines(m, indent="      "):
    for fam in FAMS:
        bd = (m.get(fam) or {}).get("by_delay") or {}
        for kind, curve in bd.items():
            print(f"{indent}{fam:<8} {kind:<16} nominal " + " ".join(
                f"d{d}:{pct(v)}" for d, v in curve.items()))
        bc = (m.get(fam) or {}).get("by_condition") or {}
        for kind, cc in bc.items():
            print(f"{indent}{fam:<8} {kind:<16} " + " ".join(
                f"{c}:{pct(v)}" for c, v in cc.items()))


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')} [{st.get('study')}] complete="
          f"{st.get('complete')} failed={st.get('failed')} incomplete="
          f"{st.get('incomplete')} heldout_opened={st.get('heldout_opened')} "
          f"wall={st.get('wall_s')} source_unchanged="
          f"{st.get('source_unchanged')}")
    print(f"computation_status={st.get('computation_status')} study_status="
          f"{st.get('study_status')} integrity_verified="
          f"{st.get('integrity_verified')}")
    print("(the launcher's terminal verdict is merged AFTER this digest)")
    print(f"task {st.get('task')} match band {st.get('match_band')}")
    print(f"planned {st.get('planned_work')} completed "
          f"{st.get('work_completed')}")
    print(f"streams {st.get('streams')} ranges {st.get('stream_ranges')}")

    tc = st.get("task_checks") or {}
    print("\n--- task checks ---")
    for name in ("dev_validation", "eval_validation", "diagnostic",
                 "first_training_batch", "heldout"):
        c = tc.get(name)
        if c:
            print(f"  {name}: n {c.get('n')} queries "
                  f"{c.get('queries_per_sequence')} writes "
                  f"{c.get('writes_per_sequence_by_condition')} idle "
                  f"{c.get('idle_per_sequence_by_condition')} balanced "
                  f"{c.get('block_cells_balanced')} cell "
                  f"{c.get('block_cell_size')} oracle {c.get('oracle_exact')}")
            print(f"    times {c.get('times')}")

    print("\n--- start points ---")
    for seed, r in (st.get("start_points") or {}).items():
        print(f"  seed {seed}: storage identity "
              f"{r.get('storage_identity_of_the_two_processing_start_trees')}"
              f" recovery failures "
              f"{r.get('direct_native_point_recovery_failures')} start "
              f"acceptance {r.get('start_acceptance_failure')}")

    dg = st.get("mechanism_diagnostic") or {}
    print("\n--- mechanism diagnostic (diagnostic only) ---")
    print(f"  failures {dg.get('failures')} wall {dg.get('wall_s')}")
    for name, r in (dg.get("open_loop") or {}).items():
        print(f"  OPEN LOOP {name}: exact {r.get('exact_coefficients')} "
              f"executed {r.get('executed_coefficients')}")
        print(f"    exact response {r.get('exact_response')[:6]}")
        print(f"    executed response "
              f"{[round(x, 6) for x in r.get('executed_response')[:6]]} "
              f"max error {r.get('max_abs_error')}")
    cl = dg.get("closed_loop") or {}
    if cl and cl.get("stage"):
        print(f"  CLOSED LOOP stopped at stage {cl.get('stage')} "
              f"(see failures above)")
    if cl and not cl.get("stage"):
        print(f"  finite-first: {cl.get('finite_first')}")
        print(f"  processing-state max |entry| "
              f"{cl.get('processing_state_max_abs')}")
        print(f"  CLOSED LOOP incoming: {cl.get('incoming_state')}")
        print(f"  native-point logit agreement "
              f"{cl.get('native_point_logit_agreement')}")
        print(f"  first write {cl.get('first_write')}")
        print(f"  offsets: {cl.get('offset_meaning')}")
        for pair, groups in (cl.get("differences") or {}).items():
            print(f"  {pair}")
            for g, c in groups.items():
                for k in ("revised_label_probability",
                          "untouched_label_probability"):
                    print(f"    {g:<30} {k:<28} " + " ".join(
                        f"{o}:{c[k][o]:+.4f}" for o in OFFSETS
                        if o < len(c[k])))
        print(f"  scope: {cl.get('scope')}")

    pf = st.get("preflight") or {}
    print("\n--- preflight ---")
    for r in pf.get("rows", []):
        print(f"  {r.get('arm'):<24} step {f(1e3 * r.get('step_s', 0), 2)}ms "
              f"checkpoint {f(r.get('checkpoint_s'), 2)}s eval "
              f"{f(r.get('evaluation_s'), 2)}s steps checked "
              f"{r.get('measured_steps_checked')} failure "
              f"{r.get('acceptance_failure')}")
    print(f"  projected {pf.get('projected_remaining_s')} failures "
          f"{pf.get('failures')} retraced {pf.get('retraced_any')}")
    print(f"  coverage: {pf.get('coverage')}")

    print("\n--- development (source 500) ---")
    for r in st.get("development", []):
        print(f"  {r['rule']:<24} {r['config']} lr={r['lr']} invalid="
              f"{r['invalid']}")
        for v in r.get("validation", []):
            print(f"    u{v['update']:>3} {agg_line(v)} accepted="
                  f"{v.get('accepted')}")

    sel = st.get("selection") or {}
    print("\n--- frozen selection ---")
    print(f"  rule: {sel.get('rule')}")
    print(f"  deployment feasibility: {sel.get('deployment_feasibility')}")
    for arm, s in (sel.get("selected") or {}).items():
        print(f"  {arm:<24} {s.get('config')} lr {s.get('lr')} update "
              f"{s.get('update')} {agg_line(s)} deployable={s.get('feasible')}")
    mo = st.get("matched_operating_point") or {}
    print(f"  matched operating point: {mo}")
    print(f"  final plan: {st.get('final_plan')}")
    for arm, pl in (st.get("deployment_plan") or {}).items():
        print(f"  deployment {arm:<22} choice={pl.get('choice')} evaluate="
              f"{pl.get('evaluate_arm')} map: {pl.get('parameter_map')}")

    rows = st.get("final", [])
    print("\n--- held-out endpoints per seed ---")
    for r in rows:
        h = r.get("heldout")
        if not h:
            continue
        print(f"  {r['rule']:<42} seed {r['seed']} {r.get('config')} u"
              f"{r.get('update')} ({r.get('endpoint_kind')}) {agg_line(h)}")
        delay_lines(h)
        for ex in h.get("executed_coefficient_sets") or []:
            print(f"      executed {ex}")

    sc = st.get("screen") or {}
    if sc:
        print("\n--- comparisons (primary aggregate screen) ---")
        print(f"  {sc.get('rule')}")
        print(f"  {sc.get('secondary')}")
        for c in sc.get("comparisons", []):
            print(f"\n  [{c['name']}] ({c['kind']}) aggregate_passed="
                  f"{c['aggregate_screen_passed']} safeguard(-1pp)="
                  f"{c.get('safeguard_passed_minus_one_pp')}")
            print(f"    revision {f(c['mean_primary_difference'])} paired "
                  f"{c['paired_primary_differences']} all-positive "
                  f"{c['positive_in_all_seeds']}")
            print(f"    retention {f(c['retention_difference'])} "
                  f"({c['retention_direction']}) recall "
                  f"{f(c['recall_difference'])} ({c['recall_direction']})")
            sec = c.get("secondary_cells") or {}
            for key in ("immediate_revision", "later"):
                if key in sec:
                    print(f"    secondary {key}: {sec[key]}")
            for fam in FAMS:
                bd = (sec.get(fam) or {}).get("by_delay") or {}
                for kind, curve in bd.items():
                    print(f"    secondary {fam:<8} {kind:<16} " + " ".join(
                        f"d{d}:{100 * v['mean']:+.2f}"
                        for d, v in curve.items()))

    ta = st.get("timing_analysis") or {}
    if ta:
        print("\n--- timing analysis (secondary) ---")
        for part in ("main_endpoints", "matched_endpoints"):
            t = ta.get(part)
            if not t:
                continue
            d = t.get("differences") or {}
            print(f"  {part}: {t.get('label')}")
            print(f"    descriptive flag ({t.get('descriptive_flag_meaning')})"
                  f": {t.get('later_improved_at_no_worse_immediate')}")
            hm = t.get("heldout_immediate_match") or {}
            print(f"    held-out immediate match (band {hm.get('band')}): "
                  f"per seed {hm.get('per_seed_in_band')} all seeds "
                  f"{hm.get('all_seeds_in_band')} mean in band "
                  f"{hm.get('mean_in_band')}")
            if t.get("heldout_match_statement"):
                print(f"    {t['heldout_match_statement']}")
            print(f"    immediate {d.get('immediate_revision')}")
            print(f"    later {d.get('later')}")
            for k in ("later_part/revised_later", "later_part/retention",
                      "later_part/recall_later"):
                print(f"    {k} {d.get(k)}")
            print(f"    executed immediate amplitude c "
                  f"{t.get('executed_immediate_amplitude_c')}")
        print(f"  matched rule record {ta.get('matched_operating_point')}")
        print(f"  attribution: {ta.get('attribution')}")
        print(f"  scope: {ta.get('scope')}")
    for arm, o in (st.get("deployment_outcome") or {}).items():
        print(f"  deployment outcome {arm:<22} -> {o.get('label')} held-out "
              f"primary {f(o.get('heldout_primary_mean'))} "
              f"counted_as_improvement={o.get('counted_as_improvement')}")


if __name__ == "__main__":
    main(sys.argv[1])
