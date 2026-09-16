"""Digest for an independent-source replication run. Written BEFORE the run.

Read-only: saved JSON only; no JAX, no model, no recomputation. It reuses the
completed study's printers where the records are identical.

    python -m experiments.prospective_momentum.replication_summary <run_dir>
"""

import json
import os
import sys

from .summary import ARMS, CATS, categories, f, g, metrics_line, \
    print_coefficients


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
    print("(the launcher's terminal verdict, including verification of the "
          "NEW source checkpoints, is merged AFTER this digest; see "
          "status.json `terminal` and logs/terminal.json)")
    print(f"reference: completed study {st.get('completed_study_reference')}")
    print(f"planned work {st.get('planned_work')} completed "
          f"{st.get('work_completed')}")
    print(f"source seeds {st.get('source_seeds')} recipe "
          f"{st.get('source_recipe')}")
    print(f"streams {st.get('streams')} ranges {st.get('stream_ranges')}")
    print(f"arm -> source family {st.get('arm_source_family')}")

    src = st.get("source", {})
    man = (src.get("manifest") or {}).get("entries", [])
    print(f"\n--- independently pretrained sources ({len(man)}) ---")
    for e in man:
        ms, me = e.get("metrics_start", {}), e.get("metrics_end", {})
        print(f"  {e['key']:<28} primary {f(ms.get('primary'))} -> "
              f"{f(me.get('primary'))} ret {f(me.get('retention_revision_untouched'))} "
              f"rec {f(me.get('recall_overall'))} revCE "
              f"{f(me.get('revision_ce'))} wall {f(e.get('wall_s'), 1)}s")
        print(f"  {'':<28} file {e['file']} sha256 {e['sha256'][:16]}... "
              f"opt {e['opt_file']} restore_bitwise "
              f"{e.get('restore_bitwise_equal')} restore_failures "
              f"{e.get('restore_reproduction_failures')} invalid "
              f"{e.get('invalid')}")
        print(f"  {'':<28} init {e.get('initializer')}")
        print(f"  {'':<28} recipe {e.get('recipe')} stream {e.get('stream')}")
        if e.get("extension_summary"):
            print(f"  {'':<28} extension {e['extension_summary']}")
        for c in e.get("curve", []):
            print(f"  {'':<28} curve {c}")
    print(f"  update-zero identity per source seed: {src.get('identity')}")
    print(f"  baseline hashes: {src.get('hashes_at_restore')}")

    pf = st.get("preflight", {})
    print("\n--- preflight (source AND continuation paths, disposable) ---")
    for r in pf.get("rows", []):
        print(f"  {r.get('path'):<18} {r.get('rule'):<22} step "
              f"{f(1e3 * r.get('step_s', 0), 2)}ms eval "
              f"{f(1e3 * (r.get('eval_s') or 0), 1)}ms report "
              f"{f(r.get('report_s') or 0, 2)}s compile "
              f"{f(r.get('compile_s_incurred'), 1)}s failure "
              f"{r.get('acceptance_failure')}")
    print(f"  projected {pf.get('projected_remaining_s')} = sources "
          f"{pf.get('source_s')} + identity {pf.get('identity_s')} + "
          f"continuation {pf.get('continuation_s')} + host "
          f"{pf.get('host_allowance_s')}; save/checksum/restore "
          f"{pf.get('save_checksum_restore_s')}; failures {pf.get('failures')}"
          f" retraced {pf.get('retraced_any')}")

    print("\n--- development (source 500): start -> update 200 ---")
    for r in st.get("development", []):
        print(f"  {r['rule']:<22} {r['config']} lr={r['lr']:<6} source "
              f"{r.get('source')}")
        print(f"  {'':<22} START {metrics_line(r['start_validation'])}")
        print(f"  {'':<22} END   {metrics_line(r['final_validation'])} "
              f"invalid={r['invalid']} gains {r.get('training_gain')}")
        if r.get("extension_summary"):
            print(f"  {'':<22} extension {r['extension_summary']}")
    print("selected:", g(st, "selection", "selected"),
          "frozen:", st.get("selection_frozen"))
    for row in g(st, "selection", "table", default=[]) or []:
        print(f"  {row}")

    rows = st.get("final", [])
    by = {(r["rule"], r["seed"]): r for r in rows}
    seeds = st.get("final_seeds", [])
    print("\n--- final runs: held-out per independent source seed ---")
    means = {}
    for a in ARMS:
        hs = []
        for s in seeds:
            r = by.get((a, s))
            if not r:
                print(f"  {a:<22} {s} missing")
                continue
            h = r.get("heldout")
            if h is None:
                print(f"  {a:<22} {s} no held-out (invalid={r['invalid']})")
                continue
            hs.append(h)
            print(f"  {a:<22} {s} source {r.get('source')}")
            print(f"  {'':<22}    {metrics_line(h)} wall {f(r['wall_s'], 1)}s")
            print(f"  {'':<22}    {categories(h)}")
            print(f"  {'':<22}    state norms {h.get('state_norms')}")
            if h.get("observed_rollout_gates"):
                print(f"  {'':<22}    OBSERVED-ROLLOUT gates "
                      + json.dumps(h["observed_rollout_gates"]))
        if len(hs) == len(seeds) and hs:
            n = len(hs)
            m = {k: sum(x[k] for x in hs) / n for k in
                 ("primary", "retention_revision_untouched", "recall_overall",
                  "revision_ce")}
            means[a] = m
            print(f"  {a:<22} MEAN {metrics_line(m)}")
            cm = []
            for fam in ("revision", "recall"):
                cm.append(fam + " " + " ".join(
                    f"{c[:4]}={f(sum(x[fam]['by_category'][c]['accuracy'] for x in hs) / n)}"
                    for c in CATS))
            print(f"  {'':<22}    categories {' | '.join(cm)}")

    print("\n--- per-seed paired differences vs the candidate (pp) ---")
    cand = "prospective_momentum"
    for a in ARMS:
        if a == cand:
            continue
        for s in seeds:
            x, y = by.get((cand, s)), by.get((a, s))
            if not (x and y and x.get("heldout") and y.get("heldout")):
                continue
            hx, hy = x["heldout"], y["heldout"]
            print(f"  vs {a:<22} seed {s} primary "
                  f"{100 * (hx['primary'] - hy['primary']):+.2f} retention "
                  f"{100 * (hx['retention_revision_untouched'] - hy['retention_revision_untouched']):+.2f}"
                  f" recall "
                  f"{100 * (hx['recall_overall'] - hy['recall_overall']):+.2f}")

    print("\n--- final runs: training, extension and coefficients ---")
    for r in rows:
        print(f"  {r['rule']:<22} seed {r['seed']} {r['config']} source "
              f"{r.get('source')}")
        print(f"      start {metrics_line(r['start_validation'])}")
        print(f"      end(val) {metrics_line(r['final_validation'])} gains "
              f"{r.get('training_gain')}")
        print(f"      params {g(r, 'params', 'total')} carry {r['carry']} "
              f"invalid {r['invalid']}")
        for c in r.get("curve", []):
            print(f"      curve {c}")
        if r.get("extension_summary"):
            print(f"      extension summary {r['extension_summary']}")
            hist = r.get("extension_history") or []
            for h in hist[::25] + hist[-1:]:
                print(f"      history {h}")
        print_coefficients(r.get("coefficients_final"))

    sc = st.get("screen")
    if sc:
        print("\n--- screens (rule: " + sc.get("rule", "") + ") ---")

        def show(name, c):
            print(f"  {name} vs {c['against']}: passed={c['passed']} mean "
                  f"{f(c['mean_primary_difference'])} paired "
                  f"{c['paired_primary_differences']} all-positive "
                  f"{c['positive_in_all_seeds']} retention "
                  f"{f(c['retention_difference'])} ({c.get('retention_direction')})"
                  f" recall {f(c['recall_difference'])} "
                  f"({c.get('recall_direction')})")
        for c in sc["literature"]:
            show("LITERATURE", c)
        print(f"  LITERATURE SCREEN (joint): {sc['literature_screen_passed']}")
        show("GAIN CONTROL", sc["gain_control"])
        show("OLD GENERALIZED", sc["old_generalized"])
        show("TSS Eq.(17) direct, applicability-limited", sc["tss_eq17"])
        print("  note: passing a -1 pp safeguard does not mean zero measured "
              "loss; the directions above are stated explicitly.")
        print(f"  {sc.get('note')}")


if __name__ == "__main__":
    main(sys.argv[1])
