"""Read a prospective-momentum run directory and print everything the report
needs. Written BEFORE the run. Read-only: saved JSON only; no JAX, no model,
no recomputation.

    python -m experiments.prospective_momentum.summary <run_dir>

Raw logarithmic leaves and exponentiated coefficients are printed under
separate labels (`raw_log_g` vs `executed_g` [production float32 exp] vs
`reference_g_f64`; `raw_log_leaves` vs `exponentiated`). kappa is stored
directly, not as a logarithm. Extension summaries state their units: kappa
relative margins; log_g and raw_r log slacks (plus the gain's relative
margin). TABLE and OBSERVED-ROLLOUT gate coverage are labelled separately.
"""

import json
import os
import sys

ARMS = ("prospective_momentum", "momentum_delta", "gain_momentum",
        "gated_delta", "gp_two_sided", "tss_eq17")
CATS = ("immediate_selected", "middle_untouched", "late_selected",
        "late_untouched")


def f(v, nd=4):
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def metrics_line(m):
    return (f"primary {f(m.get('primary'))} ret "
            f"{f(m.get('retention_revision_untouched'))} rec "
            f"{f(m.get('recall_overall'))} revCE {f(m.get('revision_ce'))}")


def categories(m):
    out = []
    for fam in ("revision", "recall"):
        cs = g(m, fam, "by_category", default={}) or {}
        out.append(fam + " " + " ".join(
            f"{c[:4]}={f(g(cs, c, 'accuracy'))}" for c in CATS))
    return " | ".join(out)


def print_coefficients(c):
    if not c:
        return
    rule = c.get("rule")
    if "table_transition" in c:
        t = c["table_transition"]
        print(f"      TABLE-coverage transitions: {t.get('classification')} min Jury "
              f"{t.get('min_jury_expression')} kappa_bound_f64 "
              f"{t.get('kappa_bound_f64')} gain_bound_f64 "
              f"{t.get('gain_bound_f64')}")
        if rule == "prospective_momentum":
            print(f"      kappa (stored directly) {t.get('kappa')} "
                  f"kappa/bound {t.get('kappa_over_bound')}")
        if rule == "gain_momentum":
            print(f"      raw_log_g {c.get('raw_log_g')}  executed_g "
                  f"{t.get('executed_g')} ({t.get('executed_g_dtype')})  "
                  f"reference_g_f64 {t.get('reference_g_f64')}  executed "
                  f"g/bound {t.get('executed_g_over_bound')}")
        gt = c.get("gate_table", {})
        for k in ("alpha", "beta", "mu", "eta", "q"):
            print(f"      table {k}: {gt.get(k)}")
        print(f"      rounded: alpha=1 {gt.get('alpha_rounded_to_one')} mu=1 "
              f"{gt.get('mu_rounded_to_one')} beta=0 "
              f"{gt.get('beta_rounded_to_zero')} mu at clamp "
              f"{gt.get('mu_at_clamp')}")
        if "sector_9" in gt:
            s = gt["sector_9"]
            print(f"      sector (9): fraction of write settings passive "
                  f"{s.get('fraction_of_write_settings_in_passive_sector')} "
                  f"threshold mu/(1-mu) {s.get('passive_threshold_mu_over_1_minus_mu')}"
                  f" QHM nu {s.get('qhm_nu')}")
        print("      (observed-rollout gates: see each evaluation's "
              "observed_rollout_gates below)")
    else:
        if "raw_log_leaves" in c:
            print(f"      raw_log_leaves {c['raw_log_leaves']}")
            print(f"      exponentiated  {c.get('exponentiated')}")
        if "domain" in c:
            d = c["domain"]
            print(f"      domain side {d.get('side')} certified "
                  f"{d.get('certified')} executed rho {d.get('executed_rho')}"
                  f" T {d.get('executed_T')} tau {d.get('executed_tau')}")
        if c.get("gates"):
            print(f"      gates {json.dumps(c['gates'])}")


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')} complete={st.get('complete')} "
          f"failed={st.get('failed')} incomplete={st.get('incomplete')} "
          f"heldout_opened={st.get('heldout_opened')} wall={st.get('wall_s')}"
          f" source_unchanged={st.get('source_unchanged')}")
    print(f"computation_status={st.get('computation_status')} study_status="
          f"{st.get('study_status')} integrity_verified="
          f"{st.get('integrity_verified')} integrity_failures="
          f"{st.get('integrity_failures')} runtime_failures="
          f"{st.get('runtime_failures')} heldout_evaluation_complete="
          f"{st.get('heldout_evaluation_complete')}")
    print("(the launcher's terminal verdict is merged AFTER this digest; see "
          "status.json `terminal` and logs/terminal.json)")
    print(f"streams {st.get('streams')} ranges {st.get('stream_ranges')}")
    print(f"projection relative margin {st.get('projection_relative_margin')}")

    src = st.get("source", {})
    print("\n--- source restoration and reproduction (dev val 60M) ---")
    for fam in ("momentum_delta", "gated_delta", "gp_two_sided", "tss_eq17"):
        r = src.get(fam)
        if not r:
            print(f"  {fam}: not restored")
            continue
        print(f"  {fam} {r.get('file')} failures "
              f"{r.get('reproduction_failures')} params "
              f"{g(r, 'params', 'total')}")
        for row in r.get("reproduction", []):
            if "metric" in row:
                print(f"    {row['metric']:<30} saved {f(row['saved'], 6)} "
                      f"measured {f(row['measured'], 6)}")
            else:
                print(f"    {row['family']:<8} {row['category']:<20} dq "
                      f"{f(row['query_difference'], 3)} dCE "
                      f"{row['ce_relative_difference']:.2e}")
    for r, t in (src.get("momentum_source_transitions") or {}).items():
        print(f"  source transitions {r}: {t.get('classification')} "
              f"kappa_bound_f64 {t.get('kappa_bound_f64')} gain_bound_f64 "
              f"{t.get('gain_bound_f64')}")
    print(f"  source hashes at restore {src.get('hashes_at_restore')}")
    print(f"  update-zero identity {st.get('update_zero_identity')}")

    pf = st.get("preflight", {})
    print("\n--- preflight ---")
    for r in pf.get("rows", []):
        print(f"  {r['rule']:<22} compile {f(r['compile_s_incurred'], 1)}s "
              f"step {f(1e3 * r['step_s'], 2)}ms eval "
              f"{f(1e3 * r['eval_s'], 1)}ms report {f(r['report_s'], 2)}s arm "
              f"{f(r['arm_remaining_s'], 1)}s params {g(r, 'params', 'total')}"
              f" carry {r['carry']} failure {r['acceptance_failure']}")
    print(f"  projected {pf.get('projected_remaining_s')} failures "
          f"{pf.get('failures')} retraced {pf.get('retraced_any')}")

    print("\n--- development (seed %s): start -> update 200 ---"
          % st.get("dev_seed"))
    for r in st.get("development", []):
        print(f"  {r['rule']:<22} {r['config']} lr={r['lr']:<6} START "
              f"{metrics_line(r['start_validation'])}")
        print(f"  {'':<22}           END   "
              f"{metrics_line(r['final_validation'])}  invalid={r['invalid']}")
        print(f"  {'':<22}           gains {r.get('training_gain')}")
        if r.get("extension_summary"):
            print(f"  {'':<22}           extension {r['extension_summary']}")
    print("selected:", g(st, "selection", "selected"))
    for row in g(st, "selection", "table", default=[]) or []:
        print(f"  {row}")

    rows = st.get("final", [])
    by = {(r["rule"], r["seed"]): r for r in rows}
    seeds = st.get("final_seeds", [])
    print("\n--- final runs: held-out per seed ---")
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
            print(f"  {a:<22} {s} {metrics_line(h)} wall "
                  f"{f(r['wall_s'], 1)}s")
            print(f"  {'':<22}      {categories(h)}")
            print(f"  {'':<22}      state norms {h.get('state_norms')}")
            if h.get("observed_rollout_gates"):
                print(f"  {'':<22}      OBSERVED-ROLLOUT gates "
                      + json.dumps(h["observed_rollout_gates"]))
        if len(hs) == len(seeds) and hs:
            n = len(hs)
            mean = {k: sum(h[k] for h in hs) / n for k in
                    ("primary", "retention_revision_untouched",
                     "recall_overall", "revision_ce")}
            print(f"  {a:<22} MEAN {metrics_line(mean)}")
            cm = []
            for fam in ("revision", "recall"):
                cm.append(fam + " " + " ".join(
                    f"{c[:4]}={f(sum(h[fam]['by_category'][c]['accuracy'] for h in hs) / n)}"
                    for c in CATS))
            print(f"  {'':<22}      categories {' | '.join(cm)}")

    print("\n--- final runs: training, extension and coefficients ---")
    for r in rows:
        print(f"  {r['rule']:<22} seed {r['seed']} {r['config']} start "
              f"{metrics_line(r['start_validation'])}")
        print(f"      end(val) {metrics_line(r['final_validation'])} gains "
              f"{r.get('training_gain')}")
        print(f"      params {g(r, 'params', 'total')} carry {r['carry']} "
              f"invalid {r['invalid']}")
        for c in r.get("curve", []):
            print(f"      curve {c}")
        og = (r.get("final_validation") or {}).get("observed_rollout_gates")
        if og:
            print("      OBSERVED-ROLLOUT gates (final validation) "
                  + json.dumps(og))
        for v in r.get("validation", []):
            print(f"      val u={v['update']} {f(v['primary'])} norms "
                  f"{v.get('state_norms')}")
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
                  f"{f(c['retention_difference'])} recall "
                  f"{f(c['recall_difference'])}")
        for c in sc["literature"]:
            show("LITERATURE", c)
        print(f"  LITERATURE SCREEN (joint): {sc['literature_screen_passed']}")
        show("GAIN CONTROL", sc["gain_control"])
        show("OLD GENERALIZED", sc["old_generalized"])
        show("TSS Eq.(17) direct, applicability-limited", sc["tss_eq17"])
        print(f"  note: {sc.get('note')}")


if __name__ == "__main__":
    main(sys.argv[1])
