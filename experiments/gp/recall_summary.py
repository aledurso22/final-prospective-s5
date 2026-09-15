"""Summarize a completed recall study. Read-only: parses results.json."""

import json
import os
import sys

import numpy as onp

ARMS = ("ordinary", "rawat", "professor", "gp_rho", "gp_rho_frozen",
        "ordinary_2x")
DELAYS = ("8", "32", "64", "96")


def main():
    d = sys.argv[1]
    rows = json.load(open(os.path.join(d, "results.json")))
    st = json.load(open(os.path.join(d, "status.json")))
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["arm"]): r for r in rows}

    print("=== primary: mean accuracy at trained delays 32 and 64 ===")
    print(f"  {'arm':16s} " + "  ".join(f"seed{s}" for s in seeds) + "    mean")
    for a in ARMS:
        v = [by[(s, a)]["primary_mean_acc_32_64"] for s in seeds]
        print(f"  {a:16s} " + "  ".join(f"{x:.4f}" for x in v)
              + f"   {onp.mean(v):.4f}")

    print("\n=== paired differences (percentage points), each seed shown ===")
    for ref in ("ordinary", "rawat", "gp_rho_frozen", "ordinary_2x"):
        v = [(by[(s, "gp_rho")]["primary_mean_acc_32_64"]
              - by[(s, ref)]["primary_mean_acc_32_64"]) * 100 for s in seeds]
        sign = "all +" if all(x > 0 for x in v) else (
            "all -" if all(x < 0 for x in v) else "MIXED")
        print(f"  gp_rho - {ref:16s} " + "  ".join(f"{x:+7.3f}" for x in v)
              + f"   mean {onp.mean(v):+7.3f}   {sign}")

    print("\n=== accuracy by delay (mean over seeds); 96 is HELD OUT ===")
    print(f"  {'arm':16s} " + "  ".join(f"d{x:>4s}" for x in DELAYS))
    for a in ARMS:
        v = [onp.mean([by[(s, a)]["per_delay"][x]["accuracy"] for s in seeds])
             for x in DELAYS]
        print(f"  {a:16s} " + "  ".join(f"{x:.3f}" for x in v))

    print("\n=== cost ===")
    print(f"  {'arm':16s} {'params':>8s} {'phys':>6s} {'aux':>5s} {'buf':>5s} "
          f"{'s/run':>7s}")
    for a in ARMS:
        r = by[(seeds[0], a)]
        c = r["state_counts"]
        phys = sum(x["physical_real"] for x in c)
        aux = sum(x["auxiliary_real"] for x in c)
        buf = sum(x.get("previous_input_buffer", 0) for x in c)
        wall = onp.mean([by[(s, a)]["wall_s"] for s in seeds])
        print(f"  {a:16s} {r['params']:8d} {phys:6d} {aux:5d} {buf:5d} "
              f"{wall:7.1f}")

    print("\n=== paired interventions (mean over seeds) ===")
    print(f"  {'arm':16s} {'base acc':>9s} {'cue-changed acc':>16s} "
          f"{'|dlogit| cue':>13s} {'|dlogit| distr':>15s} {'distr flip':>11s}")
    for a in ARMS:
        i = [by[(s, a)]["interventions"] for s in seeds]
        print(f"  {a:16s} {onp.mean([q['base_accuracy'] for q in i]):9.3f} "
              f"{onp.mean([q['cue_changed_accuracy'] for q in i]):16.3f} "
              f"{onp.mean([q['cue_change_logit_l2'] for q in i]):13.3f} "
              f"{onp.mean([q['distractor_change_logit_l2'] for q in i]):15.3f} "
              f"{onp.mean([q['distractor_change_pred_flip'] for q in i]):11.3f}")

    print("\n=== learned rho (gp_rho), per seed, all layers ===")
    for s in seeds:
        r = by[(s, "gp_rho")]
        init = onp.concatenate([onp.asarray(v) for v in r["rho_init"].values()])
        fin = onp.concatenate([onp.asarray(v) for v in r["rho_final"].values()])
        mu = 5.0 * fin
        print(f"  seed {s}: rho init {init.min():.6f}  final min {fin.min():.6f}"
              f"  med {onp.median(fin):.6f}  max {fin.max():.6f}"
              f"   mu med {onp.median(mu):.4f}  n_fell={int((fin < init).sum())}"
              f"/{fin.size}")
    print("\n=== effective clock (exp(log_step)) medians ===")
    for a in ("ordinary", "gp_rho", "gp_rho_frozen"):
        for s in seeds[:1]:
            c = by[(s, a)]["effective_clock"]
            v = onp.concatenate([onp.asarray(x) for x in c.values()])
            print(f"  {a:16s} seed{s}: min {v.min():.5f} med "
                  f"{onp.median(v):.5f} max {v.max():.5f}")

    g = st.get("gate", {})
    print(f"\n=== initialization gate (tol {g.get('tolerance')}) ===")
    for a, v in (g.get("arms") or {}).items():
        print(f"  {a:16s} worst signal-only core impulse rel "
              f"{v['worst_impulse_rel']:.4e}   query-logit rel "
              f"{v['query_logits']['rel']:.4e}")
    print(f"  passed={g.get('passed')}  worst_gp={g.get('gp_worst_impulse_rel')}")
    print(f"\nwall {st.get('wall_s'):.0f}s  complete={st.get('complete')}  "
          f"incomplete={st.get('incomplete')}")


if __name__ == "__main__":
    main()
