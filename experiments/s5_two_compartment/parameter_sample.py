"""Per-mode (T_j, rho_j, gamma_j) from a trained generalized checkpoint.

The result tables report the MEDIAN over modes, which hides the distribution
the arm actually learned: whether every mode moved together, whether some
collapsed back to the initialization, and where the fast pole ended up per
mode. This prints the raw per-mode values.

Reads checkpoints the same way `prospective_drift` does -- msgpack_restore
into nested dicts, NO model built, no data cache, no device requirement -- so
it runs anywhere the 3.2 MiB files can be copied.

    python -m experiments.s5_two_compartment.parameter_sample \\
        --run-root /Users/durso/s5-runs/s5-two-compartment-factored/20260921-175242 \\
        --arm generalized_prospective_s5 --seeds 301 302 303 --sample 12
"""

import argparse
import glob
import json
import os

import numpy as onp

from experiments.s5_two_compartment.prospective_drift import (
    INITIAL, _layer_groups, _read)

RHO_MIN = 1e-4
H = 1.0


def softplus(x):
    return onp.log1p(onp.exp(-onp.abs(x))) + onp.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + onp.exp(-x))


def leaves(group):
    """(T, rho, gamma, M) per mode, from the raw leaves. Mirrors
    `response_mass_gamma` exactly, in numpy so no JAX import is needed."""
    T = softplus(onp.asarray(group["generalized_T_raw"], dtype=onp.float64))
    rho = RHO_MIN + (1.0 - RHO_MIN) * sigmoid(
        onp.asarray(group["generalized_rho_raw"], dtype=onp.float64))
    gamma = softplus(onp.asarray(group["generalized_gamma_raw"],
                                 dtype=onp.float64))
    return T, rho, gamma, rho * gamma * T


def quantiles(values):
    q = onp.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return dict(zip(("min", "p5", "p25", "median", "p75", "p95", "max"),
                    (float(x) for x in q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--arm", default="generalized_prospective_s5")
    ap.add_argument("--seeds", nargs="+", type=int, default=[301, 302, 303])
    ap.add_argument("--sample", type=int, default=12,
                    help="how many individual modes to print per layer")
    ap.add_argument("--layer", type=int, default=0,
                    help="which layer to print modes from")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print(f"initialization, identical for every mode: "
          f"T={INITIAL['T']}  rho={INITIAL['rho']}  gamma={INITIAL['gamma']}"
          f"  ->  M = rho*gamma*T = {INITIAL['rho']*INITIAL['gamma']*INITIAL['T']:.5f}\n")

    report = {"run_root": a.run_root, "arm": a.arm, "seeds": {}}
    for seed in a.seeds:
        pattern = os.path.join(a.run_root, a.arm, str(seed),
                               "checkpoint_epoch_*.msgpack")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"seed {seed}: no checkpoints at {pattern}")
            continue
        params = _read(files[-1])
        groups = _layer_groups(params)
        names = sorted(groups)
        print(f"=== seed {seed}   {os.path.basename(files[-1])}   "
              f"{len(names)} layers, {len(leaves(groups[names[0]])[0])} modes each")

        allT, allR, allG, allM = [], [], [], []
        per_layer = {}
        for name in names:
            T, rho, gamma, M = leaves(groups[name])
            allT.append(T); allR.append(rho); allG.append(gamma); allM.append(M)
            per_layer[name] = {"T": quantiles(T), "rho": quantiles(rho),
                               "gamma": quantiles(gamma), "M": quantiles(M)}

        # individual modes from one layer
        name = names[min(a.layer, len(names) - 1)]
        T, rho, gamma, M = leaves(groups[name])
        fast = (M + H * T) / (M + H * T + H * gamma)
        order = onp.argsort(-gamma)                     # most-damped first
        idx = order[onp.linspace(0, len(order) - 1, min(a.sample, len(order))
                                 ).astype(int)]
        print(f"  layer {name}, {len(idx)} modes spanning the gamma range:")
        print(f"  {'mode':>5} {'T_j':>9} {'rho_j':>8} {'gamma_j':>9} "
              f"{'M_j':>9} {'fast pole':>10}")
        for j in idx:
            print(f"  {int(j):5d} {T[j]:9.5f} {rho[j]:8.4f} {gamma[j]:9.4f} "
                  f"{M[j]:9.5f} {fast[j]:10.5f}")

        T = onp.concatenate(allT); rho = onp.concatenate(allR)
        gamma = onp.concatenate(allG); M = onp.concatenate(allM)
        print(f"\n  all {T.size} modes, all layers:")
        for label, v in (("T", T), ("rho", rho), ("gamma", gamma), ("M", M)):
            q = quantiles(v)
            print(f"    {label:>5}  min {q['min']:9.5f}  p25 {q['p25']:9.5f}  "
                  f"med {q['median']:9.5f}  p75 {q['p75']:9.5f}  "
                  f"max {q['max']:9.5f}")
        moved = {"T": int((onp.abs(T - INITIAL['T']) > 0.1 * INITIAL['T']).sum()),
                 "rho": int((onp.abs(rho - INITIAL['rho']) > 0.1 * INITIAL['rho']).sum()),
                 "gamma": int((onp.abs(gamma - INITIAL['gamma']) > 0.1 * INITIAL['gamma']).sum())}
        print(f"    modes more than 10% away from their init: "
              f"T {moved['T']}/{T.size}   rho {moved['rho']}/{T.size}   "
              f"gamma {moved['gamma']}/{T.size}\n")
        report["seeds"][seed] = {"checkpoint": os.path.basename(files[-1]),
                                 "per_layer": per_layer, "moved": moved}

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
