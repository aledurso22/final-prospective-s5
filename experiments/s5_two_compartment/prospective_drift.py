"""Did training exploit the exact prospective correction, or retreat from it?

Gamma_k = T + gamma h / (T (1 - lambda_bar)) diverges as lambda_bar -> 1,
and the modes Native S5 uses for its longest memory are exactly the ones
where the matched correction is largest. So gradient descent has three
ways to weaken it:

    lambda_bar -> less extreme      (give up the long memory)
    gamma      -> 0                 (give up the damping that needs
                                     compensating in the first place)
    T          -> values that shrink Gamma_k

and one way to exploit it: keep modes with |lambda_bar| near 1 AND keep a
substantial Gamma_k. That is the intended regime -- very slow internal
memory with a very fast prospective response -- and this script measures
which of the four happened, per layer and over the whole inventory, by
comparing the trained checkpoint against the initialization.

NO MODEL IS BUILT. The first version called `runner.init_state` three
times purely to obtain a parameter template, which pulled in optax, the
dataloaders and a full Flax model construction for what is a read-only
inspection of two files. `flax.serialization.msgpack_restore` reads a
checkpoint into nested dicts with no template at all, so this needs only
JAX and Flax, runs in seconds, needs no data cache, and does not care which
device it is on. The checkpoints are 3.2 MiB each, so it can equally be run
away from the cluster on copies of them.

WHAT IT COMPARES. The FIRST and LAST checkpoints present, which is the
honest "did training move it" question, plus the exact scalar values every
mode starts from -- T = 0.05, rho = 0.5, gamma = 1.0 -- which are known
without building anything.

    python -m experiments.s5_two_compartment.prospective_drift \\
        --run-root /Users/durso/s5-runs/s5-two-compartment-factored/<stamp> \\
        --arm matched_lag_prospective_s5 --out drift.json
"""

import argparse
import glob
import json
import os

import jax.numpy as np
import numpy as onp
from flax import serialization
from flax.traverse_util import flatten_dict

from s5 import matched_lag_ssm as ML
from s5.generalized_prospective_ssm import response_mass_gamma
from s5.ssm import discretize_zoh

H = 1.0
#: what every mode starts from, from runner.T_INIT / RHO_INIT / gamma_init
INITIAL = {"T": 0.05, "rho": 0.5, "gamma": 1.0}


def _layer_groups(params):
    """Every SSM layer's parameters, keyed by module path.

    Located by `generalized_T_raw`, so the grouping follows the actual
    tree rather than an assumed layout. Works on the plain nested dicts
    `msgpack_restore` returns as well as on a live parameter tree.
    """
    flat = flatten_dict(params)
    groups = {}
    for path in flat:
        if path[-1] == "generalized_T_raw":
            groups["/".join(path[:-1])] = {}
    for path, value in flat.items():
        prefix = "/".join(path[:-1])
        if prefix in groups:
            groups[prefix][path[-1]] = np.asarray(value)
    return groups


def _read(path):
    """A checkpoint as nested dicts. No template, no model, no device."""
    with open(path, "rb") as handle:
        restored = serialization.msgpack_restore(handle.read())
    for key in ("params", "target"):
        if key in restored:
            return restored[key]
    return restored


def _mode_quantities(group, clip_eigs=True):
    """lambda_bar, T, gamma and Gamma_k for one layer's parameters."""
    value = (np.clip(group["Lambda_re"], None, -1e-4) if clip_eigs
             else group["Lambda_re"]) + 1j * group["Lambda_im"]
    b_tilde = group["B"][..., 0] + 1j * group["B"][..., 1]
    lambda_bar, _ = discretize_zoh(value, b_tilde,
                                   np.exp(group["log_step"][:, 0]))
    T, mass, gamma, rho = response_mass_gamma(group["generalized_T_raw"],
                                              group["generalized_rho_raw"],
                                              group["generalized_gamma_raw"])
    gamma_k = ML.matched_gamma(lambda_bar, T, gamma, H)
    return (onp.asarray(lambda_bar), onp.asarray(T), onp.asarray(gamma),
            onp.asarray(rho), onp.asarray(gamma_k))


def _summary(label, lam, T, gamma, rho, gamma_k):
    magnitude = onp.abs(lam)
    return {
        "stage": label,
        "lambda_abs": {"median": float(onp.median(magnitude)),
                       "max": float(magnitude.max()),
                       "above_0.99": int((magnitude > 0.99).sum()),
                       "above_0.999": int((magnitude > 0.999).sum())},
        "one_minus_lambda_abs_median": float(onp.median(onp.abs(1.0 - lam))),
        "T": {"median": float(onp.median(T)), "min": float(T.min()),
              "max": float(T.max())},
        "gamma": {"median": float(onp.median(gamma)),
                  "min": float(gamma.min()), "max": float(gamma.max())},
        "rho_median": float(onp.median(rho)),
        "gamma_k_abs": {"median": float(onp.median(onp.abs(gamma_k))),
                        "p90": float(onp.percentile(onp.abs(gamma_k), 90)),
                        "max": float(onp.abs(gamma_k).max())},
    }


def _joint(lam, gamma_k):
    """The interesting positive: slow modes that KEPT a large correction.

    Reported as the median |Gamma_k| among the slowest tenth of modes,
    against the median over all of them. A ratio well above one means the
    slow modes are the ones carrying the correction, which is the intended
    regime rather than an accident.
    """
    magnitude = onp.abs(lam)
    if magnitude.size == 0:
        return {}
    cutoff = onp.percentile(magnitude, 90)
    slow = onp.abs(gamma_k)[magnitude >= cutoff]
    everything = onp.abs(gamma_k)
    return {
        "slowest_decile_lambda_abs_min": float(cutoff),
        "slowest_decile_gamma_k_median": float(onp.median(slow)),
        "all_modes_gamma_k_median": float(onp.median(everything)),
        "concentration_ratio": float(onp.median(slow)
                                     / max(float(onp.median(everything)),
                                           1e-30)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arm", default="matched_lag_prospective_s5")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[301, 302, 303])
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    report = {"schema": "s5-two-compartment/prospective-drift-v2",
              "arm": arguments.arm, "run_root": arguments.run_root,
              "initial_scalars": INITIAL, "seeds": {}}
    for seed in arguments.seeds:
        checkpoints = sorted(glob.glob(os.path.join(
            arguments.run_root, arguments.arm, str(seed),
            "checkpoint_epoch_*.msgpack")))
        if len(checkpoints) < 2:
            print(f"seed {seed}: {len(checkpoints)} checkpoint(s), need 2",
                  flush=True)
            continue
        stages = {"first": _read(checkpoints[0]),
                  "last": _read(checkpoints[-1])}

        pieces = {k: _layer_groups(v) for k, v in stages.items()}
        per_layer, pooled = {}, {"first": [], "last": []}
        for name in sorted(pieces["first"]):
            row = {}
            for stage in ("first", "last"):
                values = _mode_quantities(pieces[stage][name])
                pooled[stage].append(values)
                row[stage] = _summary(stage, *values)
            row["last"]["joint"] = _joint(pooled["last"][-1][0],
                                          pooled["last"][-1][4])
            per_layer[name] = row
        stacked = {stage: [onp.concatenate(c)
                           for c in zip(*pooled[stage])]
                   for stage in ("first", "last")}
        summary = {stage: _summary(stage, *stacked[stage])
                   for stage in ("first", "last")}
        summary["last"]["joint"] = _joint(stacked["last"][0],
                                          stacked["last"][4])
        report["seeds"][str(seed)] = {
            "first_checkpoint": os.path.basename(checkpoints[0]),
            "last_checkpoint": os.path.basename(checkpoints[-1]),
            "epochs_seen": len(checkpoints),
            "per_layer": per_layer, "pooled": summary}
        before, after = summary["first"], summary["last"]
        print(f"seed {seed}  epochs {len(checkpoints):2d}  "
              f"|lam| med {before['lambda_abs']['median']:.5f}"
              f" -> {after['lambda_abs']['median']:.5f}   "
              f"gamma {before['gamma']['median']:.4f}"
              f" -> {after['gamma']['median']:.4f}   "
              f"T {before['T']['median']:.4f}"
              f" -> {after['T']['median']:.4f}   "
              f"|Gk| med {before['gamma_k_abs']['median']:.1f}"
              f" -> {after['gamma_k_abs']['median']:.1f}", flush=True)

    verdicts = {}
    for seed, detail in report["seeds"].items():
        before, after = detail["pooled"]["first"], detail["pooled"]["last"]
        verdicts[seed] = {
            "retreated_lambda": bool(after["one_minus_lambda_abs_median"]
                                     > 1.2 * before["one_minus_lambda_abs_median"]),
            "gamma_vs_initial_1.0": after["gamma"]["median"],
            "T_vs_initial_0.05": after["T"]["median"],
            "retreated_gamma_k": bool(after["gamma_k_abs"]["median"]
                                      < 0.5 * before["gamma_k_abs"]["median"]),
            "modes_above_0.99": after["lambda_abs"]["above_0.99"],
            "modes_above_0.999": after["lambda_abs"]["above_0.999"],
            "concentration_ratio": after["joint"]["concentration_ratio"],
        }
    report["verdicts"] = verdicts
    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
