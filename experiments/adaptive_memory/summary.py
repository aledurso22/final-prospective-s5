"""Read a completed adaptive-memory run and print everything the report needs.

Read-only: it opens the saved JSON and writes nothing. No model, JAX or
numerical recomputation - every number printed was produced by the run.

    python -m experiments.adaptive_memory.summary <run_dir> [<log_dir>]
"""

import json
import os
import sys

import numpy as onp

ARMS = ("adaptive_prospective", "adaptive_inertial", "adaptive_delta",
        "tss_prospective", "ideal_projection", "gated_delta",
        "momentum_delta")
SHORT = {"adaptive_prospective": "generalized prosp.",
         "adaptive_inertial": "inertial control",
         "adaptive_delta": "first-order delta",
         "tss_prospective": "TSS prospective",
         "ideal_projection": "ideal equilibrium",
         "gated_delta": "Gated DeltaNet",
         "momentum_delta": "Momentum DeltaNet"}
CATS = ("immediate_selected", "middle_untouched", "late_selected",
        "late_untouched")
FAMS = ("recall", "revision")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def comparison_block(title, rows):
    print(f"\n--- {title} ---")
    for c in rows:
        ps = c["paired_primary_differences"]
        print(f"  vs {SHORT.get(c['against'], c['against'])}: "
              f"PASSED={c['passed']}")
        print(f"     mean primary {c['mean_primary_candidate']:.4f} vs "
              f"{c['mean_primary_other']:.4f}  "
              f"diff {c['mean_primary_difference']:+.4f}  "
              f"(+1 point met: {c['meets_plus_one_point']})")
        print("     paired primary diff per seed: " +
              "  ".join(f"{s}:{v:+.4f}" for s, v in sorted(ps.items())) +
              f"  all positive: {c['positive_in_all_seeds']}  "
              f"complete pairs: {c['complete_paired_seeds']}")
        print(f"     retention diff {c['retention_difference']:+.4f} "
              f"(>= -0.01: {c['retention_within_one_point']})   "
              f"recall diff {c['recall_difference']:+.4f} "
              f"(>= -0.01: {c['recall_within_one_point']})")


def main(run_dir, log_dir=None):
    st = load(os.path.join(run_dir, "status.json"))
    dev = load(os.path.join(run_dir, "development.json"))
    fin = load(os.path.join(run_dir, "final.json"))
    by = {(r["rule"], r["seed"]): r for r in fin}
    seeds = st["final_seeds"]

    print(f"run {st['run_id']}  complete={st.get('complete')}  "
          f"heldout_opened={st.get('heldout_opened')}  "
          f"wall={st.get('wall_s', float('nan')):.0f}s  "
          f"failed={st.get('failed')}  incomplete={st.get('incomplete')}")
    cal = st["calibration"]
    print(f"beta_star={cal['beta_star']:.12f}  nu recovery abs err "
          f"{cal['reference_recovery']['absolute_error']:.2e}")
    for k in sorted(cal["slots"]):
        s = cal["slots"][k]
        vals = " ".join(f"{n}={s[n]:.6g}" for n in
                        ("nu", "eta", "tau", "rho", "tau_m", "epsilon", "M",
                         "T") if n in s)
        print(f"  {k:<28} lr={s['lr']:<6g} {vals}")
    for rec in cal["records"]:
        print(f"  solve {rec['kind']:<11} tau={rec['tau']} ratio="
              f"{rec.get('ratio')} rate={rec['rate']:.10g} bracket="
              f"{[round(b, 6) for b in rec['bracket']]} err="
              f"{rec['observable_error']:.1e} grid pts "
              f"{rec.get('grid_points_evaluated')}")

    pf = st.get("preflight", {})
    print(f"\npreflight: incurred compilation "
          f"{pf.get('incurred_compilation_s', float('nan')):.1f}s, projected "
          f"remaining {pf.get('projected_remaining_s', float('nan')):.1f}s, "
          f"retrace={pf.get('retraced_any')}")

    print("\n--- development (seed 200, update 200, validation) ---")
    for rule in ARMS:
        for r in [d for d in dev if d["rule"] == rule]:
            v = r["final_validation"]
            print(f"  {SHORT[rule]:<20} {r['config']} lr={r['lr']:<6g} "
                  f"primary {v['primary']:.4f}  revCE "
                  f"{v['revision_ce']:.4f}  retention "
                  f"{v['retention_revision_untouched']:.4f}  recall "
                  f"{v['recall_overall']:.4f}")
    print("selected: " + ", ".join(f"{SHORT[r]}={c}" for r, c in
                                   st["selection"]["selected"].items()))

    print("\n--- held-out per seed (primary / revision untouched retention "
          "/ recall / revision CE) ---")
    for rule in ARMS:
        for sd in seeds:
            h = by[(rule, sd)]["heldout"]
            print(f"  {SHORT[rule]:<20} {sd}  {h['primary']:.4f}  "
                  f"{h['retention_revision_untouched']:.4f}  "
                  f"{h['recall_overall']:.4f}  {h['revision_ce']:.4f}")
        hs = [by[(rule, sd)]["heldout"] for sd in seeds]
        print(f"  {SHORT[rule]:<20} mean  "
              f"{onp.mean([h['primary'] for h in hs]):.4f}  "
              f"{onp.mean([h['retention_revision_untouched'] for h in hs]):.4f}  "
              f"{onp.mean([h['recall_overall'] for h in hs]):.4f}  "
              f"{onp.mean([h['revision_ce'] for h in hs]):.4f}")

    print("\n--- held-out categories, mean over seeds (accuracy) ---")
    for fam in FAMS:
        print(f"  [{fam}]  " + "  ".join(f"{c:>18}" for c in CATS))
        for rule in ARMS:
            vals = [onp.mean([by[(rule, sd)]["heldout"][fam]["by_category"]
                              [c]["accuracy"] for sd in seeds]) for c in CATS]
            print(f"  {SHORT[rule]:<20}" +
                  "".join(f"{v:>20.4f}" for v in vals))

    sc = st["screen"]
    comparison_block("LITERATURE SCREEN: "
                     f"{'PASSED' if sc['literature_screen_passed'] else 'FAILED'}",
                     sc["comparisons"])
    comparison_block("ORDINARY-PROSPECTIVITY SCREEN: "
                     f"{'PASSED' if sc['ordinary_prospectivity_screen_passed'] else 'FAILED'}",
                     sc["ordinary_prospectivity"])
    print(f"\n--- ATTRIBUTION (independent of both screens): prospective "
          f"term credited = {sc['prospective_term_credited']} ---")
    for a in sc["attribution"]:
        ps = a["paired_primary_differences"]
        print(f"  vs {SHORT[a['against']]}: exceeds={a['exceeds']}  paired "
              + "  ".join(f"{s}:{v:+.4f}" for s, v in sorted(ps.items())) +
              f"  retention {a['retention_difference']:+.4f}  recall "
              f"{a['recall_difference']:+.4f}")

    print("\n--- learned coefficients and gates (final seeds) ---")
    for rule in ARMS:
        for sd in seeds:
            c = by[(rule, sd)]["coefficients_final"]
            keep = {k: v for k, v in c.items()
                    if isinstance(v, (int, float)) and k != "rule"}
            sw = c.get("source_weight")
            gates = c.get("gates")
            print(f"  {SHORT[rule]:<20} {sd} " +
                  " ".join(f"{k}={v:.4g}" if isinstance(v, float)
                           else f"{k}={v}" for k, v in keep.items()) +
                  (f"  a[min/med/max]={sw['min']:.3f}/{sw['median']:.3f}/"
                   f"{sw['max']:.3f}" if sw else "") +
                  (f"  gates={json.dumps(gates)[:160]}" if gates else ""))

    print("\n--- training gain (validation, update 0 -> 200) and costs ---")
    for rule in ARMS:
        for sd in seeds:
            r = by[(rule, sd)]
            g = r["training_gain"]
            c = r["curve"][-1] if r["curve"] else {}
            print(f"  {SHORT[rule]:<20} {sd} gain primary "
                  f"{g['primary']:+.4f}  wall {r['wall_s']:.1f}s  params "
                  f"{r['params']['total']}  carry {r['carry']}  last grad "
                  f"{c.get('grad_norm', float('nan')):.3g} W "
                  f"{c.get('w_norm', float('nan')):.3g} aux "
                  f"{c.get('aux_norm', float('nan')):.3g}")

    if log_dir:
        path = os.path.join(log_dir, "measured_errors.tsv")
        if os.path.exists(path):
            rows = [ln.split("\t") for ln in open(path).read().splitlines()
                    if ln.strip()]
            print(f"\n--- measured numerical errors: {len(rows)} recorded, "
                  f"{sum(r[-1] != 'ok' for r in rows)} outside tolerance ---")
            for kind in sorted({r[0] for r in rows}):
                worst = max((r for r in rows if r[0] == kind),
                            key=lambda r: float(r[2]))
                print(f"  {kind:<16} worst {worst[1]:<28} {float(worst[2]):.3e}"
                      f"  (tol {worst[3]})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
