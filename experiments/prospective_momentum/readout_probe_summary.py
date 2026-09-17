"""Digest for a Stage B readout-probe run. Written BEFORE the run.

Read-only: saved JSON only; no JAX, no model, no recomputation.

    python -m experiments.prospective_momentum.readout_probe_summary <run_dir>
"""

import json
import os
import sys

from .summary import f


def fmt(v, nd=4):
    return "  n/a" if v is None else f"{v:.{nd}f}"


def table(metrics, arms, families, kinds, conditions, offsets, half):
    """One block per query kind and condition: arms as rows, offsets as
    columns, best per column marked. Values are the declared primary metric
    (rolled-forward target, key projection) on the named half."""
    for kind in kinds:
        for cond in conditions:
            keys = [f"rollforward/key/{kind}/{cond}/{half}/k{k}"
                    for k in offsets]
            best = {}
            for key in keys:
                vals = [(n, metrics[n][key]["mean"]) for n in arms
                        if metrics[n][key]["mean"] is not None]
                best[key] = min(vals, key=lambda x: x[1])[0] if vals else None
            print(f"\n  [{kind} queries | {cond} | {half} half]"
                  "  (1.0 = no better than native)")
            print("    " + "arm".ljust(38)
                  + "".join(f"k={k}".rjust(10) for k in offsets))
            for n in arms:
                row = "".join(
                    (("*" if best[key] == n else " ")
                     + fmt(metrics[n][key]["mean"]).rjust(9)) for key in keys)
                print("    " + n.ljust(38) + row)


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')} [{st.get('study')}] complete="
          f"{st.get('complete')} failed={st.get('failed')} incomplete="
          f"{st.get('incomplete')} wall={st.get('wall_s')} source_unchanged="
          f"{st.get('source_unchanged')}")
    print(f"computation_status={st.get('computation_status')} study_status="
          f"{st.get('study_status')} integrity_verified="
          f"{st.get('integrity_verified')}")
    print(f"training: {st.get('training')}")
    print(f"measurement: {st.get('measurement')}")
    print(f"checkpoints {st.get('checkpoint_run')} seeds {st.get('seeds')}")
    print(f"grids {st.get('grids')}")
    moved = [r for r in (st.get("arm_table") or []) if r.get("repair_moved")]
    print(f"arms whose proposed point the repair would move: {len(moved)} "
          f"{[r['name'] for r in moved]}")
    neg = [r for r in (st.get("arm_table") or [])
           if (r.get("proposed") or {}).get("gamma", 0) < 0]
    print(f"arms with gamma < 0 exercised on the production path: {len(neg)}")
    print(f"arms {st.get('n_arms')} stream {st.get('stream')} episodes/family "
          f"{st.get('episodes_per_family')} offsets {st.get('offsets')}")
    tc = st.get("task_check") or {}
    print(f"task: queries {tc.get('queries_per_sequence')} balanced "
          f"{tc.get('block_cells_balanced')} oracle {tc.get('oracle_exact')} "
          f"digest {st.get('task_digest')}")
    ref = st.get("reference") or {}
    print("\n--- frozen checkpoints (read-only) ---")
    for s, path in (ref.get("files") or {}).items():
        print(f"  seed {s}: {path}")
    print(f"  reproduction ok: "
          f"{all(v['ok'] for r in (ref.get('reproduction') or {}).values() for v in r.values())}")

    arms = st.get("arms", [])
    offsets = st.get("offsets", [])
    print(f"\nmeasurement: {st.get('measurement')}")
    print(f"baseline: {st.get('baseline')}")
    print(f"split: {st.get('split')}")
    ov = st.get("overall") or {}
    print("\n=== PREDECLARED VERDICT ===")
    print(f"  checkpoints passed {ov.get('checkpoints_passed')} of "
          f"{ov.get('checkpoints_required')} required, failed "
          f"{ov.get('checkpoints_failed')}, indeterminate "
          f"{ov.get('checkpoints_indeterminate')} -> "
          f"interior_beats_baseline={ov.get('interior_beats_baseline')}")
    print(f"  {ov.get('verdict')}")
    print(f"  decisive target per checkpoint {ov.get('decisive_target')}")
    print(f"  outcome on BOTH targets per checkpoint "
          f"{ov.get('outcome_on_both_targets')}")
    for seed, stmt in (ov.get("branch_statement") or {}).items():
        print(f"  branch, checkpoint {seed}: {stmt}")
    print(f"  {ov.get('scope')}")
    print(f"  {ov.get('note')}")
    proj = st.get("projection") or {}
    print(f"\nprojection {proj.get('seconds_per_arm_by_class')} -> "
          f"{fmt(proj.get('projected_remaining_s'), 1)}s of "
          f"{fmt(proj.get('remaining_s'), 1)}s remaining")

    detail = {}
    mpath = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(mpath):
        detail = json.load(open(mpath))
    for seed, r in (st.get("per_seed") or {}).items():
        sr = r.get("stopping_rule") or {}
        print(f"\n=== checkpoint {seed} (wall {f(r.get('wall_s'), 1)}s, "
              f"native-point agreement {r.get('native_point_logit_agreement')}"
              f") ===")
        cf = r.get("closed_form") or {}
        print(f"  idle gates {cf.get('idle_gates') or r.get('idle_gates')}")
        print(f"  closed-form target coefficients "
              f"{cf.get('closed_form_coefficients')}")
        print(f"  gate variation {cf.get('gate_variation')}")
        print(f"  best lookahead on the primary target (selection half) "
              f"{cf.get('best_lookahead_on_primary')}")
        print(f"  near-exact at {cf.get('offsets_near_exact')} offsets "
              f"{cf.get('near_exact_offsets')} (threshold "
              f"{cf.get('near_exact_threshold')}) -> DECISIVE TARGET "
              f"{cf.get('decisive_target')}")
        print(f"  BRANCH: {cf.get('branch_statement')}")
        print(f"  {cf.get('branch_scope')}")
        print(f"  {cf.get('note')}")
        und = r.get("underpowered_cells") or []
        print(f"  underpowered cells: {len(und)}"
              + (f" (first: {und[:3]})" if und else ""))
        # BOTH target rules, always, decisive one first and labelled
        rules = r.get("rules") or {}
        if not rules:                      # older records: rebuild the pair
            other = r.get("rule_on_the_other_target") or {}
            rules = {x.get("target"): x for x in (sr, other) if x}
        order = [t for t in (cf.get("decisive_target"),) if t in rules]
        order += [t for t in sorted(rules) if t not in order]
        for tname in order:
            row = rules[tname] or {}
            mark = ("DECISIVE, carries the verdict"
                    if tname == cf.get("decisive_target")
                    else "reported beside it, not discarded")
            print(f"  --- rule on target '{tname}' [{mark}] ---")
            print(f"    observed pooled standard error per offset "
                  "(read BEFORE the verdict; an offset with SE above the "
                  f"margin {row.get('se_indeterminate_above')} is "
                  "INDETERMINATE, neither pass nor fail):")
            for key, o in (row.get("per_offset") or {}).items():
                print(f"      {key}: SE {fmt(o.get('worst_pooled_sem'))} "
                      f"per-arm {o.get('pooled_sem')} n {o.get('pooled_n')} "
                      f"-> {o.get('status')}")
            print(f"    offsets won {row.get('offsets_won')} of "
                  f"{row.get('offsets_required')} required, lost "
                  f"{row.get('offsets_lost')}, indeterminate "
                  f"{row.get('offsets_indeterminate')} -> outcome "
                  f"{row.get('outcome')} (passes={row.get('passes')}, margin "
                  f"{row.get('margin')}, baseline "
                  f"{row.get('baseline_families')})")
            for key, o in (row.get("per_offset") or {}).items():
                print(f"      {key}: interior {o.get('interior')}")
                print(f"          baseline {o.get('baseline')} difference "
                      f"{fmt(o.get('difference'))} status {o.get('status')}"
                      + (f" [{o.get('reason')}]" if o.get("reason") else ""))
        print(f"  dW components: {r.get('component_split')}")
        for key, v in (r.get("argmins_matched") or {}).items():
            if key.endswith("/confirmation/k4") or key.endswith(
                    "/confirmation/k16"):
                print(f"    argmin (matched budget) {key:<48} "
                      f"{v and v['arm']:<38} {fmt(v and v['mean'])}")
        seed_detail = detail.get(seed) or {}
        if seed_detail.get("metrics"):
            table(seed_detail["metrics"], arms, None,
                  ("all", "revised", "untouched"),
                  ("both", "idle_gap", "intervening_writes"), offsets,
                  "confirmation")
            sec = (seed_detail.get("secondary") or {}).get("native_identity")
            if sec:
                print(f"    secondary (native identity) revised "
                      f"{[round(x, 4) for x in sec['revised_label_probability'][:6]]}"
                      f" untouched "
                      f"{[round(x, 4) for x in sec['untouched_label_probability'][:6]]}")


if __name__ == "__main__":
    main(sys.argv[1])
