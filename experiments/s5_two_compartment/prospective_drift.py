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

from s5 import matched_lag_ssm as ML
from s5.generalized_prospective_ssm import response_mass_gamma
from s5.ssm import discretize_zoh
from experiments.s5_two_compartment.certify_inventory import _layer_groups

H = 1.0


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

    from experiments.s5_three_arm_full import runner

    report = {"schema": "s5-two-compartment/prospective-drift-v1",
              "arm": arguments.arm, "run_root": arguments.run_root,
              "seeds": {}}
    for seed in arguments.seeds:
        template = runner.init_state(arguments.arm, seed)
        checkpoints = sorted(glob.glob(os.path.join(
            arguments.run_root, arguments.arm, str(seed),
            "checkpoint_epoch_*.msgpack")))
        if not checkpoints:
            print(f"seed {seed}: no checkpoint yet", flush=True)
            continue
        with open(checkpoints[-1], "rb") as handle:
            trained = serialization.from_bytes(template, handle.read())

        pieces = {"initial": _layer_groups(template.params),
                  "trained": _layer_groups(trained.params)}
        per_layer, pooled = {}, {"initial": [], "trained": []}
        for name in sorted(pieces["initial"]):
            row = {}
            for stage in ("initial", "trained"):
                values = _mode_quantities(pieces[stage][name])
                pooled[stage].append(values)
                row[stage] = _summary(stage, *values)
            row["trained"]["joint"] = _joint(
                *[pooled["trained"][-1][i] for i in (0, 4)])
            per_layer[name] = row
        stacked = {}
        for stage in ("initial", "trained"):
            columns = list(zip(*pooled[stage]))
            stacked[stage] = [onp.concatenate(c) for c in columns]
        summary = {stage: _summary(stage, *stacked[stage])
                   for stage in ("initial", "trained")}
        summary["trained"]["joint"] = _joint(stacked["trained"][0],
                                             stacked["trained"][4])
        report["seeds"][str(seed)] = {"checkpoint": checkpoints[-1],
                                      "per_layer": per_layer,
                                      "pooled": summary}
        before, after = summary["initial"], summary["trained"]
        print(f"seed {seed}  |lam| med {before['lambda_abs']['median']:.5f}"
              f" -> {after['lambda_abs']['median']:.5f}   "
              f"gamma med {before['gamma']['median']:.4f}"
              f" -> {after['gamma']['median']:.4f}   "
              f"T med {before['T']['median']:.4f}"
              f" -> {after['T']['median']:.4f}   "
              f"|Gk| med {before['gamma_k_abs']['median']:.1f}"
              f" -> {after['gamma_k_abs']['median']:.1f}", flush=True)

    # the verdict the question asks for, in one place
    verdicts = {}
    for seed, detail in report["seeds"].items():
        before, after = detail["pooled"]["initial"], detail["pooled"]["trained"]
        verdicts[seed] = {
            "retreated_lambda": bool(after["one_minus_lambda_abs_median"]
                                     > 1.2 * before["one_minus_lambda_abs_median"]),
            "retreated_gamma": bool(after["gamma"]["median"]
                                    < 0.5 * before["gamma"]["median"]),
            "retreated_gamma_k": bool(after["gamma_k_abs"]["median"]
                                      < 0.5 * before["gamma_k_abs"]["median"]),
            "slow_modes_kept": int(after["lambda_abs"]["above_0.99"]),
            "concentration_ratio": after["joint"]["concentration_ratio"],
        }
    report["verdicts"] = verdicts
    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
