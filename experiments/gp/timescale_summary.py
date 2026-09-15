"""Read a completed learned-timescale run and print the telemetry that decides
how its result should be read.

The endpoint table alone cannot distinguish "the new coordinate moved and did
not help" from "the new coordinate never moved". Those are different findings
with different follow-ups, so this prints the executed T and rho distributions,
their boundary occupancy, the projection events and the gradient/update norms
for BOTH response coordinates, plus T's epoch-by-epoch path.

Read-only: it opens the saved JSON and writes nothing.

    python -m experiments.gp.timescale_summary <run_dir>
"""

import json
import os
import sys

import numpy as onp


def _st(v):
    v = onp.asarray(v, dtype=float).ravel()
    return (f"{v.min():.4f}/{onp.median(v):.4f}/{v.max():.4f}")


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    rs = json.load(open(os.path.join(run_dir, "results.json")))

    print(f"run      : {st.get('run_id')}  seed {st.get('seed')}  "
          f"epochs {st.get('epochs')}  steps/epoch {st.get('steps_per_epoch')}")
    print(f"status   : complete={st.get('complete')}  "
          f"incomplete={st.get('incomplete')}")
    pf = st.get("preflight", {})
    print(f"preflight: projected {pf.get('projected_total_s', float('nan')):.1f}s")
    print(f"wall     : {st.get('wall_s', float('nan')):.0f}s")

    print("\n--- endpoint (the predeclared primary), and counts ---")
    print(f"{'arm':<18}{'endpoint':>10}{'CE':>10}{'best':>9}"
          f"{'stored':>9}{'trainable':>11}{'state':>8}")
    for r in rs:
        tot = r["state_counts"]["total_over_layers"]
        state = tot.get("total_real", tot.get("physical_real"))
        print(f"{r['arm']:<18}{r['primary_endpoint_val_accuracy']:>10.4f}"
              f"{r['primary_endpoint_val_cross_entropy']:>10.5f}"
              f"{r['best_val']['accuracy']:>9.4f}"
              f"{r['counts']['stored']:>9}{r['counts']['trainable']:>11}"
              f"{state:>8}")

    cmp_ = st.get("comparisons", {})
    print("\n--- comparisons, percentage points ---")
    iso = cmp_.get("isolating", {})
    print(f"  ISOLATING {iso.get('name')}: {iso.get('endpoint_pp')}")
    for k in ("vs_matched_ordinary", "vs_rawat", "vs_native",
              "fixed_vs_matched_ordinary", "fixed_vs_rawat"):
        if k in cmp_:
            print(f"  {k:<28}{cmp_[k]}")

    gate = st.get("gate") or {}
    print("\n--- initialization check ---")
    print(f"  arms identical: max|dlogit| = "
          f"{gate.get('max_abs_logit_difference')}  "
          f"digest equal = {gate.get('param_digest_equal')}  "
          f"passed = {gate.get('passed')}")
    print(f"  T0 = {gate.get('T_at_init')}   rho0 = {gate.get('rho_at_init')}")
    for arm, rows in (gate.get("signal_difference_from_matched_ordinary")
                      or {}).items():
        rel = [lay.get("vs_matched_ordinary", {}).get("impulse_rel")
               for lay in rows]
        frq = [lay.get("vs_matched_ordinary", {}).get("freq_rel")
               for lay in rows]
        print(f"  {arm}: signal-only impulse rel vs matched ordinary "
              f"{['%.3e' % v if v is not None else 'n/a' for v in rel]}")
        print(f"  {arm}: frequency rel                              "
              f"{['%.3e' % v if v is not None else 'n/a' for v in frq]}")

    print("\n--- executed response, per arm that carries one ---")
    for r in rs:
        rf = r.get("response_final")
        if not rf:
            continue
        print(f"\n== {r['arm']}")
        tel = r.get("projection_telemetry") or {}
        for coord in ("rho", "T"):
            t = tel.get(coord, {})
            if not t:
                continue
            ev, eu = t.get("n_projection_events"), t.get("entry_updates")
            share = (100.0 * ev / eu) if ev is not None and eu else float("nan")
            print(f"   {coord:<4} projection events {ev} of {eu} "
                  f"entry-updates ({share:.2f} %)   "
                  f"max proposed overshoot {t.get('max_proposed_overshoot'):.3e}")
            print(f"        mean |grad| {t.get('mean_grad_norm'):.4e}   "
                  f"mean |update| {t.get('mean_update_norm'):.4e}")
        for lay in rf["layers"]:
            T = onp.asarray(lay["T"], dtype=float)
            rho = onp.asarray(lay["rho"], dtype=float)
            mu = onp.asarray(lay["mu"], dtype=float)
            tj = onp.asarray(lay["T_times_j"]["abs"], dtype=float)
            print(f"   L{lay['layer']}  T {_st(T)}   rho {_st(rho)}   "
                  f"mu {_st(mu)}   |T*j| med {onp.median(tj):.4f}")
        print("   raw-leaf occupancy:")
        for k, v in rf["raw_leaves"].items():
            print(f"     {k.split('/')[-2] if '/' in k else k}: "
                  + ", ".join(f"{kk}={vv}" for kk, vv in v.items()
                              if kk != "raw"))
        traj = r.get("response_trajectory") or []
        path = []
        for e in traj:
            resp = e.get("response")
            if not resp:
                continue
            T0 = onp.asarray(resp["layers"][0]["T"], dtype=float)
            r0 = onp.asarray(resp["layers"][0]["rho"], dtype=float)
            path.append((e["epoch"], float(onp.median(T0)),
                         float(onp.median(r0)), float(T0.max() - T0.min())))
        if path:
            print("   layer 0 by epoch (epoch: median T, median rho, spread T):")
            print("     " + "  ".join(f"{ep}:{t:.4f}/{rr:.4f}/{sp:.4f}"
                                      for ep, t, rr, sp in path))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
