"""Digest for a retention-aware run. Written BEFORE the run.

Read-only: saved JSON only; no JAX, no model, no recomputation.

    python -m experiments.prospective_momentum.retention_aware_summary <run_dir>
"""

import json
import os
import sys

from .summary import f
from .temporal_response_summary import agg_line, delay_lines


def comparison_lines(c):
    if c.get("available") is False:
        print(f"  [{c['name']}] {c['candidate']} vs {c['against']}: "
              f"UNAVAILABLE - {c.get('reason')}")
        return
    print(f"  [{c['name']}] ({c['kind']}) {c['candidate']} minus "
          f"{c['against']}: aggregate_passed={c['aggregate_screen_passed']}")
    print(f"    revision {f(c['mean_primary_difference'])} paired "
          f"{c['paired_primary_differences']} all-positive "
          f"{c['positive_in_all_seeds']}")
    print(f"    retention {f(c['retention_difference'])} "
          f"({c.get('retention_direction')}) recall "
          f"{f(c['recall_difference'])} ({c.get('recall_direction')})")
    sec = c.get("secondary_cells") or {}
    for k in ("immediate_revision", "later"):
        if k in sec:
            print(f"    secondary {k}: {sec[k]}")


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')} [{st.get('study')}] complete="
          f"{st.get('complete')} failed={st.get('failed')} incomplete="
          f"{st.get('incomplete')} wall={st.get('wall_s')} source_unchanged="
          f"{st.get('source_unchanged')}")
    print(f"computation_status={st.get('computation_status')} study_status="
          f"{st.get('study_status')} integrity_verified="
          f"{st.get('integrity_verified')}")
    print(f"loss: {st.get('loss')}")
    print(f"lambda slots {st.get('lambda_slots')} lr {st.get('learning_rate')}")
    print(f"planned {st.get('planned_work')} completed "
          f"{st.get('work_completed')}")
    print(f"streams {st.get('stream_ranges')}")

    ref = st.get("reference") or {}
    print("\n--- fixed native references (read-only) ---")
    print(f"  run {ref.get('run')}")
    for s, path in (ref.get("files") or {}).items():
        print(f"  seed {s} ({(ref.get('stages') or {}).get(s)}): {path}")
        print(f"    reproduction {(ref.get('reproduction') or {}).get(s)}")
    print(f"  hashes {ref.get('hashes')}")

    print("\n--- start points ---")
    for s, r in (st.get("start_points") or {}).items():
        print(f"  seed {s}: {r}")

    pf = st.get("preflight") or {}
    print("\n--- preflight ---")
    for r in pf.get("rows", []):
        print(f"  {r.get('arm'):<24} step {f(1e3 * r.get('step_s', 0), 2)}ms "
              f"checkpoint {f(r.get('checkpoint_s'), 2)}s compile "
              f"{f(r.get('compile_s_incurred'), 1)}s steps checked "
              f"{r.get('measured_steps_checked')} failure "
              f"{r.get('acceptance_failure')}")
        print(f"    last loss terms {r.get('last_terms')}")
    print(f"  projected {pf.get('projected_remaining_s')} failures "
          f"{pf.get('failures')} retraced {pf.get('retraced_any')}")
    print(f"  coverage: {pf.get('coverage')}")

    print("\n--- development (source 500) ---")
    for r in st.get("development", []):
        print(f"  {r['rule']:<24} {r['config']} invalid={r['invalid']}")
        for v in r.get("validation", []):
            print(f"    u{v['update']:>3} {agg_line(v)} accepted="
                  f"{v.get('accepted')}")
        hist = r.get("coefficient_history") or []
        for h in hist[::50] + hist[-1:]:
            print(f"    terms u{h.get('update')}: {h.get('terms')}")

    sel = st.get("selection") or {}
    print("\n--- frozen selection ---")
    print(f"  rule: {sel.get('rule')}")
    print(f"  native reference R {f(sel.get('r_native'))} C "
          f"{f(sel.get('c_native'))}: {sel.get('native_reference')}")
    for arm, eps in (sel.get("endpoints") or {}).items():
        for role, c in eps.items():
            print(f"  {arm:<24} {role:<14} " + (
                "INFEASIBLE" if c is None else
                f"{c['config']} u{c['update']} {agg_line(c)}"))
    for row in sel.get("table", []):
        if "n_feasible" in row:
            print(f"  {row['arm']}: {row['n_feasible']} of "
                  f"{row['n_candidates']} feasible")
    print(f"  final plan {st.get('final_plan')}")

    print("\n--- held-out endpoints ---")
    for r in st.get("final", []):
        h = r.get("heldout")
        if not h:
            continue
        print(f"  {r['rule']:<40} seed {r['seed']} {r.get('config')} u"
              f"{r.get('update')} {agg_line(h)}")
        delay_lines(h, indent="      ")
        for ex in h.get("executed_coefficient_sets") or []:
            print(f"      executed {ex}")

    sc = st.get("screen") or {}
    if sc:
        print(f"\n--- PRIMARY verdicts (constrained endpoints) ---\n  "
              f"{sc.get('rule')}")
        for c in sc.get("primary", []):
            comparison_lines(c)
        print("\n--- descriptive comparisons ---")
        for c in sc.get("descriptive", []):
            comparison_lines(c)
        print(f"\n  note: {sc.get('note')}")
    for arm, o in (st.get("deployment_outcome") or {}).items():
        print(f"  deployment {arm:<22} {o}")


if __name__ == "__main__":
    main(sys.argv[1])
