"""Read a completed meta-delta run and print what the report needs.

Read-only: opens saved JSON only; no JAX, no model, no recomputation.

    python -m experiments.meta_delta.summary <run_dir>
"""

import json
import os
import sys

ARMS = ("gp_two_sided", "adaptive_delta", "heavy_ball_same_mass", "tss_eq17",
        "gated_delta", "momentum_delta")


def f(v, nd=4):
    return "None" if v is None else (f"{v:.{nd}f}" if isinstance(v, float)
                                     else str(v))


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')} complete={st.get('complete')} "
          f"failed={st.get('failed')} incomplete={st.get('incomplete')} "
          f"heldout_opened={st.get('heldout_opened')} wall={st.get('wall_s')}")
    cal = st["calibration"]
    print(f"beta*={cal['beta_star']:.10f} eta0={cal['eta0']:.10f} "
          f"tau0={cal['tau0']} eq17 T0={cal['eq17_T0']} "
          f"eta0={cal['eq17_eta0']:.10f} initial rho_max="
          f"{cal['initial_upper_rho']:.6f}")

    print("\n--- development (seed %s, update 200) ---" % st["dev_seed"])
    for r in st.get("development", []):
        v = r["final_validation"]
        print(f"  {r['rule']:<22} {r['config']} lr={r['lr']:<6} primary "
              f"{v['primary']:.4f} revCE {v['revision_ce']:.4f} retention "
              f"{v['retention_revision_untouched']:.4f} recall "
              f"{v['recall_overall']:.4f}")
    print("selected:", st.get("selection", {}).get("selected"))

    rows = st.get("final", [])
    by = {(r["rule"], r["seed"]): r for r in rows}
    seeds = st["final_seeds"]
    print("\n--- held-out per seed: primary / retention / recall / revCE ---")
    for a in ARMS:
        vals = []
        for s in seeds:
            h = by[(a, s)]["heldout"]
            vals.append(h)
            print(f"  {a:<22} {s}  {h['primary']:.4f}  "
                  f"{h['retention_revision_untouched']:.4f}  "
                  f"{h['recall_overall']:.4f}  {h['revision_ce']:.4f}")
        n = len(vals)
        print(f"  {a:<22} mean  "
              f"{sum(h['primary'] for h in vals) / n:.4f}  "
              f"{sum(h['retention_revision_untouched'] for h in vals) / n:.4f}  "
              f"{sum(h['recall_overall'] for h in vals) / n:.4f}  "
              f"{sum(h['revision_ce'] for h in vals) / n:.4f}")

    print("\n--- held-out categories (mean over seeds) ---")
    cats = ("immediate_selected", "middle_untouched", "late_selected",
            "late_untouched")
    for fam in ("recall", "revision"):
        print(f"  [{fam}] " + " ".join(f"{c:>18}" for c in cats))
        for a in ARMS:
            m = [sum(by[(a, s)]["heldout"][fam]["by_category"][c]["accuracy"]
                     for s in seeds) / len(seeds) for c in cats]
            print(f"  {a:<22}" + "".join(f"{x:>19.4f}" for x in m))

    sc = st.get("screen", {})

    def block(title, c):
        dm, dr, dc = (c["mean_primary_difference"], c["retention_difference"],
                      c["recall_difference"])
        print(f"\n--- {title}: passed={c['passed']} ---")
        print(f"  mean primary diff {dm!r}  condition >= +0.01: "
              f"{dm is not None and dm >= 0.01}")
        print(f"  paired primary {c['paired_primary_differences']}  "
              f"condition all > 0: {c['positive_in_all_seeds']}  complete "
              f"pairs: {c['complete_paired_seeds']}")
        print(f"  retention diff {dr!r} (>= -0.01: "
              f"{dr is not None and dr >= -0.01})  recall diff {dc!r} "
              f"(>= -0.01: {dc is not None and dc >= -0.01})")
    for c in sc.get("literature", []):
        block(f"LITERATURE vs {c['against']}", c)
    block("MATCHED DELTA", sc["matched_delta"])
    for c in sc.get("eq17_direct_fast_weight", []):
        block("TSS Eq.(17) direct fast weight (applicability-limited)", c)
    block("HEAVY-BALL family (not causal attribution)", sc["heavy_ball_family"])

    print("\n--- validation gain (update 0 -> 200), final runs, and cost ---")
    for a in ARMS:
        for s in seeds:
            r = by[(a, s)]
            v0, v1 = r["validation"][0], r["validation"][-1]
            print(f"  {a:<22} {s} primary {v0['primary']:.4f} -> "
                  f"{v1['primary']:.4f} (gain {r['training_gain']['primary']:+.4f})"
                  f"  revCE {v0['revision_ce']:.4f} -> {v1['revision_ce']:.4f}"
                  f"  retention {v0['retention']:.4f} -> {v1['retention']:.4f}"
                  f"  recall {v0['recall']:.4f} -> {v1['recall']:.4f}"
                  f"  params {r['params']['total']} carry {r['carry']} "
                  f"wall {r['wall_s']:.1f}s lr {r['lr']} config {r['config']}")
    pf = st.get("preflight", {})
    for row in pf.get("rows", []):
        print(f"  preflight {row['rule']:<22} step {row['step_s']*1e3:.2f}ms "
              f"eval {row['eval_s']*1e3:.1f}ms compile "
              f"{row['compile_s_incurred']:.1f}s")

    print("\n--- learned coefficients (final seeds) ---")
    for a in ARMS:
        for s in seeds:
            r = by[(a, s)]
            cf = r["coefficients_final"]
            dom = cf.get("domain")
            extra = ""
            if dom:
                extra = (f"eta {dom.get('executed_eta'):.5f} tau "
                         f"{dom.get('executed_tau'):.5f} rho "
                         f"{dom.get('executed_rho'):.5f} gamma "
                         f"{dom.get('executed_gamma'):.5f} M "
                         f"{dom.get('executed_M'):.5f} T "
                         f"{dom.get('executed_T'):.5f} side {dom.get('side')} "
                         f"certified {dom.get('certified')} dL/g^2 "
                         f"{dom.get('certificate_f64_dL_over_gamma2')}")
            else:
                extra = " ".join(f"{k}={v:.5f}" for k, v in cf.items()
                                 if isinstance(v, float))
            sw = cf.get("source_weight")
            gates = cf.get("gates")
            print(f"  {a:<22} {s} {extra}"
                  + (f" gate[min/med/max] {sw['min']:.3f}/{sw['median']:.3f}/"
                     f"{sw['max']:.3f}" if sw else "")
                  + (f" gates {json.dumps(gates)[:140]}" if gates else ""))
            if r.get("projection_telemetry") and a == "gp_two_sided":
                print(f"      telemetry {r['projection_telemetry']}  wall "
                      f"{r['wall_s']:.1f}s  gain {r['training_gain']}")


if __name__ == "__main__":
    main(sys.argv[1])
