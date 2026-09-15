"""Summarize a completed recall study. Read-only: parses results.json."""

import json
import os
import sys

import numpy as onp

ARMS = ("ordinary", "rawat", "professor", "gp_rho", "gp_rho_frozen",
        "ordinary_2x")
DELAYS = ("8", "32", "64", "96")


def rho_values(record):
    """Read a rho record in EITHER schema, without rewriting old artifacts.

    old: {"layer/path": [rho, ...]}            (executed rho only, clipped)
    new: {"layer/path": {"raw": [...], "rho": [...], "n_at_upper": ...}}

    The completed study was written in the old schema and is left untouched;
    this reader simply understands both.
    """
    if not record:
        return None
    rho, raw, occ = [], [], dict(n_at_upper=0, n_at_lower=0, n_outside_raw=0,
                                 n_modes=0, schema=None)
    for v in record.values():
        if isinstance(v, dict):
            occ["schema"] = "new"
            rho.append(onp.asarray(v["rho"], dtype=float))
            if "raw" in v:
                raw.append(onp.asarray(v["raw"], dtype=float))
            for k in ("n_at_upper", "n_at_lower", "n_outside_raw", "n_modes"):
                if k in v:
                    occ[k] += int(v[k])
        else:
            occ["schema"] = "old (executed rho only; raw not recorded)"
            rho.append(onp.asarray(v, dtype=float))
    return dict(rho=onp.concatenate(rho) if rho else None,
                raw=onp.concatenate(raw) if raw else None, occupancy=occ)


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

    print("\n=== stored versus trainable parameters ===")
    for a in ARMS:
        r = by[(seeds[0], a)]
        tc = r.get("trainable_params")
        if tc:
            print(f"  {a:16s} stored {tc['stored']:6d}  trainable "
                  f"{tc['trainable']:6d}  rho leaves {tc['rho_leaves']:3d}")
        else:
            print(f"  {a:16s} stored {r['params']:6d}  trainable: not recorded "
                  f"(old artifact)")

    print("\n=== learned rho (gp_rho), per seed, all layers ===")
    for s in seeds:
        r = by[(s, "gp_rho")]
        ini = rho_values(r.get("rho_init"))
        fin = rho_values(r.get("rho_final"))
        if fin is None:
            print(f"  seed {s}: no rho record")
            continue
        f, i = fin["rho"], (ini["rho"] if ini else None)
        mu = 5.0 * f
        line = (f"  seed {s}: final min {f.min():.6f}  med {onp.median(f):.6f}"
                f"  max {f.max():.6f}   mu med {onp.median(mu):.4f}")
        if i is not None:
            line += f"   n_fell={int((f < i).sum())}/{f.size}"
        print(line)
        occ = fin["occupancy"]
        print(f"      schema: {occ['schema']}")
        if occ["schema"] == "new":
            print(f"      bound occupancy: at upper {occ['n_at_upper']}, at "
                  f"lower {occ['n_at_lower']}, raw outside "
                  f"{occ['n_outside_raw']} of {occ['n_modes']}")
        else:
            print("      bound occupancy: NOT RECORDED in this artifact; the "
                  "exact count at the margin cannot be recovered")
        if fin["raw"] is not None:
            print(f"      raw log-rho: min {fin['raw'].min():.6f} max "
                  f"{fin['raw'].max():.6f}")

    print("\n=== projection telemetry (new artifacts only) ===")
    any_tel = False
    for s in seeds:
        for a in ("gp_rho", "gp_rho_frozen"):
            t = by[(s, a)].get("projection_telemetry")
            if not t:
                continue
            any_tel = True
            print(f"  seed {s} {a:14s} events {t['n_projection_events']:6d}  "
                  f"max overshoot {t['max_raw_overshoot_beyond_bound']:.4e}  "
                  f"mean |g_rho| {t['mean_rho_grad_norm']:.3e}  "
                  f"mean |u_rho| {t['mean_rho_update_norm']:.3e}")
    if not any_tel:
        print("  none recorded (old artifact: the run predates the projection "
              "repair, and its raw overshoot cannot be recovered)")
    print("\n=== effective clock (exp(log_step)) medians, ALL seeds ===")
    for a in ("ordinary", "gp_rho", "gp_rho_frozen"):
        for s in seeds:
            c = by[(s, a)]["effective_clock"]
            v = onp.concatenate([onp.asarray(x) for x in c.values()])
            print(f"  {a:16s} seed{s}: min {v.min():.5f} med "
                  f"{onp.median(v):.5f} max {v.max():.5f}")

    gates = st.get("gates") or ([st["gate"]] if st.get("gate") else [])
    print(f"\n=== initialization gates ({len(gates)} recorded; "
          f"{len(seeds)} seeds) ===")
    if len(gates) < len(seeds):
        print("  NOTE: fewer gates than seeds - the executed run gated only "
              "the first seed, so the others were never probed")
    for g in gates:
        print(f"  seed {g.get('seed')}  passed={g.get('passed')}  "
              f"worst impulse {g.get('gp_worst_impulse_rel')}  "
              f"worst frequency {g.get('gp_worst_frequency_rel', 'NOT ENFORCED')}")
        for a, v in (g.get("arms") or {}).items():
            fr = [l.get("freq_rel") for l in v["layers"]]
            fr = [x for x in fr if x is not None]
            print(f"    {a:16s} impulse {v['worst_impulse_rel']:.4e}   "
                  f"frequency {(max(fr) if fr else float('nan')):.4e}   "
                  f"query-logit {v['query_logits']['rel']:.4e}")
        if g.get("failures"):
            print(f"    failures: {g['failures']}")
    print(f"\nwall {st.get('wall_s'):.0f}s  complete={st.get('complete')}  "
          f"incomplete={st.get('incomplete')}")


if __name__ == "__main__":
    main()
