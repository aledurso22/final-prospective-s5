"""Digest for a same-backbone run. Written BEFORE the run.

Read-only: saved JSON only; no JAX, no model, no recomputation.

    python -m experiments.prospective_momentum.same_backbone_summary <run_dir>
"""

import json
import os
import sys

from .summary import CATS, f, g, metrics_line, print_coefficients


def categories_full(m):
    """Full category names (review: no abbreviated `late` labels), for
    per-seed rows as well as means. Historical digests are unchanged."""
    out = []
    for fam in ("revision", "recall"):
        cs = g(m, fam, "by_category", default={}) or {}
        out.append(fam + ": " + ", ".join(
            f"{c} {f(g(cs, c, 'accuracy'))}" for c in CATS))
    return " | ".join(out)


ARM_ORDER = ("prospective_full", "prospective_coeff", "ordinary_full",
             "ordinary_coeff", "native_full", "native_frozen")


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')} [{st.get('study')}] complete="
          f"{st.get('complete')} failed={st.get('failed')} incomplete="
          f"{st.get('incomplete')} heldout_opened={st.get('heldout_opened')} "
          f"wall={st.get('wall_s')} source_unchanged={st.get('source_unchanged')}")
    print(f"computation_status={st.get('computation_status')} study_status="
          f"{st.get('study_status')} integrity_verified="
          f"{st.get('integrity_verified')} integrity_failures="
          f"{st.get('integrity_failures')} runtime_failures="
          f"{st.get('runtime_failures')} heldout_evaluation_complete="
          f"{st.get('heldout_evaluation_complete')}")
    print("(the launcher's terminal verdict is merged AFTER this digest; see "
          "status.json `terminal` and logs/terminal.json)")
    print(f"reused sources: {st.get('source_run')} (READ ONLY) seeds "
          f"{st.get('source_seeds')}")
    print(f"planned {st.get('planned_work')} completed "
          f"{st.get('work_completed')}")
    print(f"streams {st.get('streams')} ranges {st.get('stream_ranges')}")
    print(f"arm law {st.get('arm_law')}")
    print(f"arm regime {st.get('arm_regime')}")
    print(f"trainable leaf when frozen: "
          f"{st.get('trainable_leaf_when_frozen')}; literature arms ABSENT "
          f"from this batch: {st.get('absent_literature_arms')}")

    src = st.get("source", {})
    print("\n--- reused independent Momentum sources ---")
    for seed, e in (src.get("entries") or {}).items():
        print(f"  seed {seed}: {e.get('file')} sha256 {str(e.get('sha256'))[:16]}"
              f"... source-run primary {f(e.get('metrics_end'))}")
    print(f"  update-zero identity (all laws at kappa = 0): "
          f"{st.get('update_zero_identity')}")

    pf = st.get("preflight", {})
    print("\n--- preflight (both regimes, disposable state) ---")
    for r in pf.get("rows", []):
        print(f"  {r.get('arm'):<20} {r.get('regime'):<17} step "
              f"{f(1e3 * r.get('step_s', 0), 2)}ms eval "
              f"{f(1e3 * r.get('eval_s', 0), 1)}ms report "
              f"{f(r.get('report_s', 0), 2)}s compile "
              f"{f(r.get('compile_s_incurred'), 1)}s params "
              f"{g(r, 'params', 'total')} carry {r.get('carry')} failure "
              f"{r.get('acceptance_failure')}")
    print(f"  projected {pf.get('projected_remaining_s')} (host allowance "
          f"{pf.get('host_allowance_s')}) failures {pf.get('failures')} "
          f"retraced {pf.get('retraced_any')}")

    print(f"\n--- development (source {g(st, 'source_seeds', 'development')})"
          " ---")
    for r in st.get("development", []):
        print(f"  {r['rule']:<20} {r['config']:<2} lr={r['lr']:<6} updates "
              f"{r.get('updates')} params {g(r, 'params', 'stored')} stored / "
              f"{g(r, 'params', 'trainable')} trainable source "
              f"{r.get('source')}")
        print(f"  {'':<20} START {metrics_line(r['start_validation'])}")
        print(f"  {'':<20} END   {metrics_line(r['final_validation'])} "
              f"invalid={r['invalid']} gains {r.get('training_gain')}")
        if r.get("frozen_leaf_differences"):
            print(f"  {'':<20} FROZEN LEAVES CHANGED "
                  f"{r['frozen_leaf_differences']}")
        if r.get("extension_summary"):
            print(f"  {'':<20} extension {r['extension_summary']}")
    print("selected:", g(st, "selection", "selected"),
          "frozen:", st.get("selection_frozen"))
    for row in g(st, "selection", "table", default=[]) or []:
        print(f"  {row}")

    rows = st.get("final", [])
    by = {(r["rule"], r["seed"]): r for r in rows}
    seeds = st.get("final_seeds", [])
    print("\n--- final runs: held-out per source seed ---")
    for a in ARM_ORDER:
        hs = []
        for s in seeds:
            r = by.get((a, s))
            if not r:
                print(f"  {a:<20} {s} missing")
                continue
            h = r.get("heldout")
            if h is None:
                print(f"  {a:<20} {s} no held-out (invalid={r['invalid']})")
                continue
            hs.append(h)
            print(f"  {a:<20} {s} {metrics_line(h)} updates "
                  f"{r.get('updates')} wall {f(r['wall_s'], 1)}s source "
                  f"{r.get('source')}")
            print(f"  {'':<20}    {categories_full(h)}")
            print(f"  {'':<20}    state norms {h.get('state_norms')}")
            if r.get("frozen_leaf_differences"):
                print(f"  {'':<20}    FROZEN LEAVES CHANGED "
                      f"{r['frozen_leaf_differences']}")
        if len(hs) == len(seeds) and hs:
            n = len(hs)
            m = {k: sum(x[k] for x in hs) / n for k in
                 ("primary", "retention_revision_untouched", "recall_overall",
                  "revision_ce")}
            print(f"  {a:<20} MEAN {metrics_line(m)}")
            cm = []
            for fam in ("revision", "recall"):
                cm.append(fam + ": " + ", ".join(
                    f"{c} {f(sum(x[fam]['by_category'][c]['accuracy'] for x in hs) / n)}"
                    for c in CATS))
            print(f"  {'':<20}    MEAN categories {' | '.join(cm)}")

    sc = st.get("screen")
    if sc:
        print("\n--- declared comparisons, each reported separately ---")
        print(f"  safeguard rule: {sc.get('rule')}")
        print(f"  descriptive condition: {sc.get('descriptive_condition')}")
        for c in sc.get("comparisons", []):
            print(f"\n  [{c['name']}] {c.get('question')}")
            print(f"    {c['candidate']} minus {c['against']}: "
                  f"safeguard_passed={c['passed']} no_measured_decrease="
                  f"{c['no_measured_decrease']} descriptive_only="
                  f"{c.get('descriptive_only')}")
            print(f"    mean primary {f(c['mean_primary_difference'])} paired "
                  f"{c['paired_primary_differences']} all-positive "
                  f"{c['positive_in_all_seeds']}")
            print(f"    retention {f(c['retention_difference'])} "
                  f"({c.get('retention_direction')})  recall "
                  f"{f(c['recall_difference'])} ({c.get('recall_direction')})")
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

    print("\n--- final runs: coefficients, projection and frozen leaves ---")
    for r in rows:
        print(f"  {r['rule']:<20} seed {r['seed']} {r['config']} law "
              f"{r.get('law')} regime {r.get('regime')} params "
              f"{g(r, 'params', 'total')} carry {r['carry']} invalid "
              f"{r['invalid']}")
        print(f"      parameters stored {g(r, 'params', 'stored')} trainable "
              f"{g(r, 'params', 'trainable')} ({g(r, 'params', 'regime')}); "
              f"frozen_leaf_differences {r.get('frozen_leaf_differences')}")
        for c in r.get("curve", []):
            print(f"      curve {c}")
        if r.get("extension_summary"):
            print(f"      extension summary {r['extension_summary']}")
            hist = r.get("extension_history") or []
            for h in hist[::25] + hist[-1:]:
                print(f"      history {h}")
        cf = r.get("coefficients_final") or {}
        if cf.get("table_transition") and cf.get("law") == "ordinary_prospective":
            t = cf["table_transition"]
            print(f"      OPERATOR TABLE transitions {t.get('classification')}"
                  f" min Jury {t.get('min_jury_expression')} kappa "
                  f"{t.get('kappa')} kappa/bound {t.get('kappa_over_bound')}")
            print(f"      {t.get('bound_note')}")
            print(f"      {cf.get('carry_note')}")
            print(f"      {cf.get('observed_rollout_gates_note')}")
            gt = cf.get("gate_table", {})
            for k in ("alpha", "beta", "mu", "eta", "q"):
                print(f"      table {k}: {gt.get(k)}")
            if "sector_9" in gt:
                print(f"      sector (9): {gt['sector_9']}")
        else:
            print_coefficients(cf)


if __name__ == "__main__":
    main(sys.argv[1])
