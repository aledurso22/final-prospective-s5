"""Digest for a TSS containment run. Written BEFORE the run.

Read-only: saved JSON only; no JAX, no model, no recomputation.

    python -m experiments.prospective_momentum.tss_containment_summary <run_dir>
"""

import json
import os
import sys

from .summary import CATS, f, g, metrics_line

ARM_ORDER = ("generalized_processing", "tss_processing", "native_full",
             "operator_full", "native_frozen")


def categories_full(m):
    out = []
    for fam in ("revision", "recall"):
        cs = g(m, fam, "by_category", default={}) or {}
        out.append(fam + ": " + ", ".join(
            f"{c} {f(g(cs, c, 'accuracy'))}" for c in CATS))
    return " | ".join(out)


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')} [{st.get('study')}] complete="
          f"{st.get('complete')} failed={st.get('failed')} incomplete="
          f"{st.get('incomplete')} heldout_opened={st.get('heldout_opened')} "
          f"wall={st.get('wall_s')} source_unchanged="
          f"{st.get('source_unchanged')}")
    print(f"computation_status={st.get('computation_status')} study_status="
          f"{st.get('study_status')} integrity_verified="
          f"{st.get('integrity_verified')} integrity_failures="
          f"{st.get('integrity_failures')} runtime_failures="
          f"{st.get('runtime_failures')} heldout_evaluation_complete="
          f"{st.get('heldout_evaluation_complete')}")
    print("(the launcher's terminal verdict is merged AFTER this digest; see "
          "status.json `terminal` and logs/terminal.json)")
    print(f"specification {st.get('specification')} protocol "
          f"{st.get('protocol')}")
    print(f"reused sources: {st.get('source_run')} (READ ONLY) seeds "
          f"{st.get('source_seeds')}")
    print(f"planned {st.get('planned_work')} completed "
          f"{st.get('work_completed')}")
    print(f"streams {st.get('streams')} ranges {st.get('stream_ranges')}")
    print(f"arm law {st.get('arm_law')}")
    print(f"frozen coefficient leaves {st.get('arm_frozen_leaves')}")
    print(f"executed form: {st.get('executed_form')}")
    print(f"filter constants {st.get('filter_constants')}")
    print(f"recovery policy {st.get('recovery_policy')}")
    print(f"acceptance gate: {st.get('acceptance_gate')}")
    print(f"carry {st.get('carry')}")

    src = st.get("source", {})
    print("\n--- reused independent Momentum sources ---")
    for seed, e in (src.get("entries") or {}).items():
        print(f"  seed {seed}: {e.get('file')} sha256 "
              f"{str(e.get('sha256'))[:16]}... source-run primary "
              f"{f(e.get('metrics_end'))}")

    sp = st.get("start_points", {})
    print("\n--- start points (each check labelled with its scope) ---")
    for k, v in (sp.get("scope") or {}).items():
        print(f"  scope {k}: {v}")
    print(f"  tolerances {sp.get('tolerances')}")
    for seed, r in (sp.get("rows") or {}).items():
        print(f"  seed {seed}: storage identity="
              f"{r.get('storage_identity_of_the_two_processing_start_trees')}"
              f" DECISIVE direct recovery failures="
              f"{r.get('direct_native_point_recovery_failures')}")
        for row in r.get("direct_native_point_recovery") or []:
            print(f"    episode {row.get('episode')} relative errors "
                  f"{row.get('relative_errors')}")
        agg = r.get("aggregate_native_point_differences") or {}
        print(f"    aggregate (diagnostic): malformed={agg.get('malformed')} "
              f"identical={agg.get('identical_within_declared_identity_tolerance')}")
        for name, d in (agg.get("per_category") or {}).items():
            print(f"      {name}: count difference {d.get('count_difference')}"
                  f" CE rel {d.get('cross_entropy_relative')}")
        for ef in r.get("start_executed_filter") or []:
            print(f"    start executed a={f(ef.get('a'), 6)} b="
                  f"{f(ef.get('b'), 6)} c={f(ef.get('c'), 6)} d="
                  f"{f(ef.get('d'), 6)} {ef.get('classification')}")
        print(f"    acceptance failures: start "
              f"{r.get('start_acceptance_failure')} native point "
              f"{r.get('native_point_acceptance_failure')}")

    pf = st.get("preflight", {})
    print("\n--- preflight (disposable state) ---")
    for r in pf.get("rows", []):
        print(f"  {r.get('arm'):<24} step {f(1e3 * r.get('step_s', 0), 2)}ms "
              f"checkpoint {f(r.get('checkpoint_s', 0), 2)}s evaluation "
              f"{f(r.get('evaluation_s', 0), 2)}s steps checked "
              f"{r.get('measured_steps_checked')} compile "
              f"{f(r.get('compile_s_incurred'), 1)}s stored "
              f"{g(r, 'params', 'stored')} trainable "
              f"{g(r, 'params', 'trainable')} carry {r.get('carry')} failure "
              f"{r.get('acceptance_failure')}")
    print(f"  projected {pf.get('projected_remaining_s')} (host allowance "
          f"{pf.get('host_allowance_s')}) failures {pf.get('failures')} "
          f"retraced {pf.get('retraced_any')}")

    print(f"\n--- development (source {g(st, 'source_seeds', 'development')})"
          " ---")
    for r in st.get("development", []):
        print(f"  {r['rule']:<24} {r['config']:<2} lr={r['lr']:<6} updates "
              f"{r.get('updates')} stored {g(r, 'params', 'stored')} "
              f"trainable {g(r, 'params', 'trainable')} source "
              f"{r.get('source')}")
        for v in r.get("validation", []):
            print(f"  {'':<24}   update {v['update']:>3} primary "
                  f"{f(v['primary'])} revision_ce {f(v['revision_ce'], 4)} "
                  f"retention {f(v['retention'])} recall {f(v['recall'])} "
                  f"accepted={v.get('accepted')} saved {v.get('params_file')}")
            if v.get("failure"):
                print(f"  {'':<24}     FAILURE {v['failure']}")
        print(f"  {'':<24} invalid={r['invalid']} gains "
              f"{r.get('training_gain')}")
        if r.get("frozen_leaf_differences"):
            print(f"  {'':<24} STORED CONSTANTS CHANGED "
                  f"{r['frozen_leaf_differences']}")

    sel = st.get("selection", {})
    print("\n--- frozen development selection ---")
    print(f"  rule: {sel.get('rule')}")
    print(f"  R_native {f(sel.get('r_native'))} C_native "
          f"{f(sel.get('c_native'))}")
    for row in sel.get("table", []) or []:
        # the feasibility flags live in `selected`, not in the table row;
        # display only - nothing is recomputed
        c = dict(row.get("chosen", {}),
                 **((sel.get("selected") or {}).get(row.get("arm")) or {}))
        print(f"  {row.get('arm'):<24} update {c.get('update')} lr "
              f"{c.get('lr')} primary {f(c.get('primary'))} retention "
              f"{f(c.get('retention'))} recall {f(c.get('recall'))} "
              f"feasible={c.get('feasible')} diagnostic={c.get('diagnostic')} "
              f"tree {c.get('params_file')} "
              + (f"({row.get('n_feasible')} of {row.get('n_candidates')} "
                 f"feasible)" if row.get("n_feasible") is not None else
                 f"(native reference: selected unconstrained from "
                 f"{row.get('n_candidates')} checkpoints)"))
        print(f"  {'':<24} selected under: {c.get('selected_under')}")
    print(f"  selection unchanged through finals: "
          f"{st.get('selection_frozen') == sel.get('selected')}")

    print("\n--- deployment plan (separate from the scientific comparison) ---")
    for arm, pl in (st.get("deployment_plan") or {}).items():
        print(f"  {arm:<24} choice={pl.get('choice')} evaluate="
              f"{pl.get('evaluate_arm')} family {pl.get('executed_family')} "
              f"map: {pl.get('parameter_map')}")
        print(f"  {'':<24} checkpoint {pl.get('chosen_checkpoint')}")
        print(f"  {'':<24} trained development primary "
              f"{f(pl.get('trained_development_primary'))} feasible="
              f"{pl.get('trained_feasible')} best fallback "
              f"{pl.get('best_fallback')}")
    for arm, o in (st.get("deployment_outcome") or {}).items():
        print(f"  {arm:<24} -> {o.get('label')} held-out primary "
              f"{f(o.get('heldout_primary_mean'))} counted_as_improvement="
              f"{o.get('counted_as_improvement')}")

    rows = st.get("final", [])
    by = {(r["rule"], r["seed"]): r for r in rows}
    seeds = st.get("final_seeds", [])
    print("\n--- final runs: held-out per source seed ---")
    for a in ARM_ORDER:
        hs = []
        for s in seeds:
            r = by.get((a, s))
            if not r:
                print(f"  {a:<24} {s} missing")
                continue
            h = r.get("heldout")
            if h is None:
                print(f"  {a:<24} {s} no held-out (invalid={r['invalid']})")
                continue
            hs.append(h)
            print(f"  {a:<24} {s} {metrics_line(h)} updates "
                  f"{r.get('updates')} ({r.get('endpoint_kind')}) wall "
                  f"{f(r['wall_s'], 1)}s source "
                  f"{r.get('source')}")
            print(f"  {'':<24}    {categories_full(h)}")
            print(f"  {'':<24}    state norms {h.get('state_norms')}")
        if len(hs) == len(seeds) and hs:
            n = len(hs)
            m = {k: sum(x[k] for x in hs) / n for k in
                 ("primary", "retention_revision_untouched", "recall_overall",
                  "revision_ce")}
            print(f"  {a:<24} MEAN {metrics_line(m)}")

    sc = st.get("screen")
    if sc:
        print("\n--- declared comparisons, each reported separately ---")
        print(f"  criterion: {sc.get('rule')}")
        print(f"  {sc.get('safeguard_rule_reported_separately')}")
        for c in sc.get("comparisons", []):
            print(f"\n  [{c['name']}] ({c.get('kind')}) {c.get('question')}")
            print(f"    {c['candidate']} minus {c['against']}: "
                  f"promising={c.get('promising_matched_retention')} "
                  f"no_measured_decrease={c.get('no_measured_decrease')} "
                  f"safeguard(-1pp)={c.get('safeguard_passed_minus_one_pp')} "
                  f"constrained_screen_available="
                  f"{c.get('constrained_screen_available')} endpoint kinds "
                  f"{c.get('endpoint_kinds')}")
            print(f"    mean primary {f(c['mean_primary_difference'])} paired "
                  f"{c['paired_primary_differences']} all-positive "
                  f"{c['positive_in_all_seeds']}")
            print(f"    retention {f(c['retention_difference'])} "
                  f"({c.get('retention_direction')})  recall "
                  f"{f(c['recall_difference'])} ({c.get('recall_direction')})")
            if c.get("availability_note"):
                print(f"    {c['availability_note']}")
            for s in seeds:
                x, y = by.get((c["candidate"], s)), by.get((c["against"], s))
                if x and y and x.get("heldout") and y.get("heldout"):
                    hx, hy = x["heldout"], y["heldout"]
                    print(f"    seed {s}: primary "
                          f"{100 * (hx['primary'] - hy['primary']):+.2f} "
                          f"retention {100 * (hx['retention_revision_untouched'] - hy['retention_revision_untouched']):+.2f}"
                          f" recall "
                          f"{100 * (hx['recall_overall'] - hy['recall_overall']):+.2f}")
        print(f"\n  note: {sc.get('note')}")

    print("\n--- final runs: executed coefficients, gate and repair ---")
    for r in rows:
        cf = r.get("coefficients_final") or {}
        efs = cf.get("executed_filter")
        print(f"  {r['rule']:<24} seed {r['seed']} {r['config']} law "
              f"{r.get('law')} {r.get('endpoint_kind')} stored "
              f"{g(r, 'params', 'stored')} trainable "
              f"{g(r, 'params', 'trainable')} carry {r['carry']} invalid "
              f"{r['invalid']} endpoint tree {r.get('endpoint_params_file')}")
        if efs:
            for ef in efs:
                print(f"      executed M {f(ef.get('M'), 6)} gamma "
                      f"{f(ef.get('gamma'), 6)} T {f(ef.get('T'), 6)} A "
                      f"{f(ef.get('A'), 6)} a {f(ef.get('a'), 6)} b "
                      f"{f(ef.get('b'), 6)} c {f(ef.get('c'), 6)} d "
                      f"{f(ef.get('d'), 6)}")
                print(f"      {ef.get('classification')} Jury slacks "
                      f"{ef.get('jury_slacks_exact')}; declared gaps gamma+T "
                      f"{f(ef.get('gap_gamma_plus_T'), 6)} filter "
                      f"{f(ef.get('gap_filter'), 6)}; TSS boundary="
                      f"{ef.get('at_tss_boundary')} native point="
                      f"{ef.get('at_native_point')}")
            print(f"      {efs[0].get('note')}")
            print(f"      {cf.get('carry_note')}")
            print(f"      {cf.get('scope_note')}")
        elif cf.get("table_transition"):
            t = cf["table_transition"]
            label = ("OPERATOR" if r.get("law") == "ordinary_prospective"
                     else "NATIVE MOMENTUM")
            print(f"      {label} frozen-token TABLE transitions "
                  f"{t.get('classification')}"
                  f" kappa {f(t.get('kappa'), 6)} kappa/bound "
                  f"{f(t.get('kappa_over_bound'), 4)}")
            if cf.get("containment_note"):
                print(f"      {cf.get('containment_note')}")
        hist = r.get("coefficient_history") or []
        for h in hist[::25] + hist[-1:]:
            print(f"      history {h}")


if __name__ == "__main__":
    main(sys.argv[1])
