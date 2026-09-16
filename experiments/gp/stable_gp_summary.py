"""Read a completed stable-GP continuation run and print what the report needs.

Read-only: it opens the saved JSON and writes nothing. No JAX, no model, no
recomputation - every number was produced by the run itself.

    python -m experiments.gp.stable_gp_summary <run_dir>
"""

import json
import os
import sys


def fmt(v, nd=4):
    return "None" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')}  complete={st.get('complete')}  "
          f"failed={st.get('failed')}  incomplete={st.get('incomplete')}  "
          f"wall={fmt(st.get('wall_s'))}s")
    ch = st.get("source", {}).get("chosen", {})
    print(f"source {ch.get('run_dir')} epoch {ch.get('best_epoch')} "
          f"commit {ch.get('source_commit')}")
    for n, h in (ch.get("file_sha256") or {}).items():
        print(f"   sha256 {n} {h}")
    print("reproduction", st.get("source_reproduction"))

    g = st.get("identity_gate", {})
    for k in ("B_learned_input_vs_A_rawat",
              "C_stable_generalized_vs_B_learned_input"):
        r = g.get(k, {})
        pg, ig = r.get("param_grads", {}), r.get("input_grads", {})
        print(f"\n[gate] {k}: logits {fmt(r.get('logits_rel'))} argmax "
              f"{fmt(r.get('argmax_agreement'))} passed {r.get('passed')}")
        print(f"   params within mixed tol {pg.get('passed')} over "
              f"{pg.get('n_leaves')} leaves, abs_tol {fmt(pg.get('abs_tol'))}, "
              f"worst {pg.get('worst_leaf')} -> "
              f"{pg.get('leaves', {}).get(pg.get('worst_leaf'), {})}")
        print(f"   inputs within mixed tol {ig.get('passed')}  routing "
              f"{fmt(r.get('routing', {}).get('worst'))}")
        print(f"   independent first update (reported): "
              f"{r.get('independent_first_update')}")
    print("added-leaf grad norms at start:",
          json.dumps(g.get("added_leaf_grad_norms_per_layer")))
    print("epoch0:", st.get("epoch0"))

    pf = st.get("preflight", {})
    print(f"\npreflight projected {fmt(pf.get('projected_remaining_s'))}s "
          f"failures {pf.get('failures')} retraced {pf.get('retraced_any')}")

    sc = st.get("screen") or {}
    if sc:
        print(f"\nDEVELOPMENT SUCCESS: {sc.get('development_success')}")
        for k, c in sc.get("comparisons", {}).items():
            print(f"  {k}: mean acc {fmt(c.get('mean_acc_pp'))} pp  mean dCE "
                  f"{fmt(c.get('mean_ce_diff'))}  passed {c.get('passed')}")
            for p in c.get("per_seed", []):
                print(f"     seed {p['seed']}: acc {p['acc_pp']:+.4f} pp  "
                      f"dCE {fmt(p.get('ce'))}")

    for r in st.get("results", []):
        print(f"\n== {r['arm']} seed {r['seed']}: endpoint acc "
              f"{r['endpoint_val_accuracy']:.5f} ce "
              f"{r['endpoint_val_cross_entropy']:.5f}  best(descriptive) "
              f"{r['best_val_descriptive']}  wall {r['wall_s']:.1f}s  "
              f"params {r['params'].get('total')} added {r['added_parameters']}")
        print("   carry", r.get("state_counts"))
        print("   telemetry", r.get("projection_telemetry"))
        for lay, v in (r.get("executed_coefficients_final") or {}).items():
            print(f"   {lay} {v.get('summary')}")
        for L in (r.get("final_domain") or {}).get("layers", []):
            print(f"   domain {L['layer']} passed {L['passed']} passive "
                  f"{L.get('n_passive_rho_le_1')}/{L.get('n_modes')} real "
                  f"{L.get('n_real_modes')} min_rel_margin "
                  f"{fmt(L.get('min_relative_margin_complex'))} maxRe "
                  f"{fmt(L.get('max_real_part_executed_generator'))} minS "
                  f"{fmt(L.get('min_S'))}")
        for rc in r.get("response_change", []):
            print(f"   L{rc['layer']} current {rc['current_tap_start']:.4f}->"
                  f"{rc['current_tap_end']:.4f} (rel {rc['current_tap_rel_change']:.4f})"
                  f"  history {rc['history_start']:.4f}->{rc['history_end']:.4f}"
                  f" (rel {rc['history_rel_change']:.4f})")


if __name__ == "__main__":
    main(sys.argv[1])
