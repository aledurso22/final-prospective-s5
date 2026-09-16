"""Read a completed nested-memory run and print everything the report needs.

Read-only: it opens the saved JSON and writes nothing.

    python -m experiments.nested_memory.summary <run_dir>
"""

import json
import os
import sys

import numpy as onp

ARMS = ("prospective_memory", "inertial_memory", "delta_matched_write",
        "gated_delta", "momentum_delta")
LIT = ("gated_delta", "momentum_delta")
CATS = ("immediate_selected", "middle_untouched", "late_selected",
        "late_untouched")
CHANCE = 0.125


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    rs = json.load(open(os.path.join(run_dir, "results.json")))
    by = {(r["rule"], r["seed"]): r for r in rs}
    seeds = st["seeds"]

    print(f"run {st['run_id']}  complete={st.get('complete')}  "
          f"heldout_opened={st.get('heldout_opened')}  "
          f"wall={st.get('wall_s', float('nan')):.0f}s")
    pf = st.get("preflight", {})
    print(f"preflight: incurred compilation "
          f"{pf.get('incurred_compilation_s', float('nan')):.1f}s, projected "
          f"remaining {pf.get('projected_remaining_s', float('nan')):.1f}s, "
          f"retrace={pf.get('retraced_any')}")

    print("\n--- held-out, per seed and mean (chance = 0.125) ---")
    print(f"{'arm':<30}{'seed':>6}{'primary':>9}{'retention':>11}"
          f"{'recall':>9}{'rev CE':>9}")
    mean = {}
    for a in ARMS:
        prim, ret, rec = [], [], []
        for s in seeds:
            h = by[(a, s)]["heldout"]
            prim.append(h["primary"]); ret.append(h["retention_revision_untouched"])
            rec.append(h["recall_overall"])
            print(f"{a:<30}{s:>6}{h['primary']:>9.4f}"
                  f"{h['retention_revision_untouched']:>11.4f}"
                  f"{h['recall_overall']:>9.4f}"
                  f"{h['revision']['cross_entropy']:>9.4f}")
        mean[a] = dict(primary=onp.mean(prim), retention=onp.mean(ret),
                       recall=onp.mean(rec), per_seed_primary=prim,
                       per_seed_retention=ret, per_seed_recall=rec)
        print(f"{a:<30}{'mean':>6}{mean[a]['primary']:>9.4f}"
              f"{mean[a]['retention']:>11.4f}{mean[a]['recall']:>9.4f}")

    print("\n--- learning: validation primary at updates 0 / 100 / 200 ---")
    for a in ARMS:
        row = []
        for s in seeds:
            v = {x["update"]: x["primary"] for x in by[(a, s)]["validation"]}
            row.append((v.get(0), v.get(100), v.get(200)))
        m0 = onp.mean([r[0] for r in row]); m2 = onp.mean([r[2] for r in row])
        print(f"{a:<30} " + "  ".join(
            f"{r[0]:.4f}/{r[1]:.4f}/{r[2]:.4f}" for r in row)
            + f"   mean gain {100*(m2-m0):+.2f} pp (from {m0:.4f})")

    print("\n--- the PREDECLARED screening rule, applied ---")
    P = mean["prospective_memory"]
    ok = True
    for a in LIT:
        d = [100 * (x - y) for x, y in zip(P["per_seed_primary"],
                                           mean[a]["per_seed_primary"])]
        dm = onp.mean(d)
        dret = 100 * (P["retention"] - mean[a]["retention"])
        drec = 100 * (P["recall"] - mean[a]["recall"])
        pass_mean = dm >= 1.0
        pass_all = all(x > 0 for x in d)
        pass_reg = dret >= -1.0 and drec >= -1.0
        ok &= pass_mean and pass_all and pass_reg
        print(f"  vs {a:<22} primary per seed {['%+.2f' % x for x in d]} "
              f"mean {dm:+.2f} pp")
        print(f"  {'':25} >= +1 pp: {pass_mean}   all seeds positive: "
              f"{pass_all}   regressions <= 1 pp: {pass_reg} "
              f"(retention {dret:+.2f}, recall {drec:+.2f})")
    print(f"  PROMISING DEVELOPMENT SIGNAL: {ok}")

    print("\n--- attribution: against the same-state ablation and the delta ---")
    for a in ("inertial_memory", "delta_matched_write"):
        d = [100 * (x - y) for x, y in zip(P["per_seed_primary"],
                                           mean[a]["per_seed_primary"])]
        print(f"  prospective - {a:<22} {['%+.2f' % x for x in d]} "
              f"mean {onp.mean(d):+.2f} pp")

    print("\n--- held-out accuracy by query category (mean over seeds) ---")
    print(f"{'arm':<30}{'family':<10}" + "".join(f"{c:>20}" for c in CATS))
    for a in ARMS:
        for fam in ("revision", "recall"):
            vals = []
            for c in CATS:
                vals.append(onp.mean([by[(a, s)]["heldout"][fam]
                                      ["by_category"][c]["accuracy"]
                                      for s in seeds]))
            print(f"{a:<30}{fam:<10}" + "".join(f"{v:>20.4f}" for v in vals))

    print("\n--- counts, gates and cost ---")
    for a in ARMS:
        r = by[(a, seeds[0])]
        g = r.get("gates")
        gs = ""
        if g:
            gs = "  alpha[%.3f,%.3f] beta[%.3f,%.3f]" % (
                g["alpha"]["min"], g["alpha"]["max"],
                g["beta"]["min"], g["beta"]["max"])
            if "mu" in g:
                gs += " mu[%.3f,%.3f] eta[%.3f,%.3f] at-clamp %.3f" % (
                    g["mu"]["min"], g["mu"]["max"], g["eta"]["min"],
                    g["eta"]["max"], g["fraction_of_tokens_at_the_mu_clamp"])
        print(f"{a:<30} trainable={r['params']['total']:>4} "
              f"(gate {r['params']['gate']:>3})  carry={r['carry']:>3}  "
              f"wall={onp.mean([by[(a,s)]['wall_s'] for s in seeds]):5.1f}s{gs}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
