"""What regime did the second-order model actually learn?

Diagnostics only: it reads checkpoints that training already wrote and
changes nothing. The theoretical picture under test is

    one SLOW pole carrying long SSM memory
    + one FASTER WWJ/response pole

so the question is whether training converged to that, or to something
qualitatively different, if the arm becomes competitive.

Per seed, per requested epoch, per layer, with quantiles:

    learned T, gamma, rho
    |r1|, |r2| from r^2 - a1 r - a2 = 0, the SLOW root named by magnitude
    the ratio of the two, which is the separation the theory predicts
    optionally R_pros, the prospective-drive ratio on one validation batch

and the epoch-by-epoch validation curves of every seed beside Native's, so
same-epoch CE and accuracy are read rather than extrapolated.

No model is built: checkpoints are read with `msgpack_restore`, so this
needs only jax.numpy, flax and numpy unless --data-cache is given.

    python -m experiments.s5_two_compartment.regime_report \\
        --run-root .../20260921-175242 --arm generalized_prospective_s5 \\
        --native-run .../20260919-224525 --epochs 3 6 10 15 --out regime.json
"""

import argparse
import json
import os

import jax.numpy as np
import numpy as onp
from flax import serialization
from flax.traverse_util import flatten_dict

from s5.discrete_recurrence import generalized_coefficients
from s5.factored_recurrence import companion_roots
from s5.generalized_prospective_ssm import response_mass_gamma
from s5.ssm import discretize_zoh

H = 1.0
QUANTILES = (10, 50, 90)


def _read(path):
    with open(path, "rb") as handle:
        restored = serialization.msgpack_restore(handle.read())
    for key in ("params", "target"):
        if key in restored:
            return restored[key]
    return restored


def _layer_groups(params):
    flat = flatten_dict(params)
    groups = {"/".join(p[:-1]): {} for p in flat
              if p[-1] == "generalized_T_raw"}
    for path, value in flat.items():
        prefix = "/".join(path[:-1])
        if prefix in groups:
            groups[prefix][path[-1]] = np.asarray(value)
    return groups


def _stats(values):
    """median / p10 / p90 / max, as the request asks for."""
    array = onp.asarray(values, dtype=onp.float64)
    p10, p50, p90 = onp.percentile(array, QUANTILES)
    return {"p10": float(p10), "median": float(p50), "p90": float(p90),
            "max": float(array.max()), "min": float(array.min())}


def _layer(group, clip_eigs=True):
    """Everything the regime question needs, for one layer."""
    value = (np.clip(group["Lambda_re"], None, -1e-4) if clip_eigs
             else group["Lambda_re"]) + 1j * group["Lambda_im"]
    b_tilde = group["B"][..., 0] + 1j * group["B"][..., 1]
    lambda_bar, b_bar = discretize_zoh(value, b_tilde,
                                       np.exp(group["log_step"][:, 0]))
    T, mass, gamma, rho = response_mass_gamma(group["generalized_T_raw"],
                                              group["generalized_rho_raw"],
                                              group["generalized_gamma_raw"])
    a1, a2, _, _ = generalized_coefficients(lambda_bar, b_bar, T, mass, gamma)
    # r^2 - a1 r - a2 = 0; companion_roots returns (major, minor) BY
    # MAGNITUDE, so the first is the slow one by construction
    slow, fast = companion_roots(a1, a2)
    slow_abs = onp.abs(onp.asarray(slow))
    fast_abs = onp.abs(onp.asarray(fast))
    return {
        "modes": int(slow_abs.size),
        "T": _stats(T), "gamma": _stats(gamma), "rho": _stats(rho),
        "lambda_abs": _stats(onp.abs(onp.asarray(lambda_bar))),
        "slow_root_abs": _stats(slow_abs),
        "fast_root_abs": _stats(fast_abs),
        "separation_slow_over_fast": _stats(
            slow_abs / onp.maximum(fast_abs, 1e-30)),
        # the regime claim, as a count rather than an impression
        "modes_slow_above_0.99": int((slow_abs > 0.99).sum()),
        "modes_fast_below_0.5": int((fast_abs < 0.5).sum()),
        "modes_in_predicted_regime": int(((slow_abs > 0.99)
                                          & (fast_abs < 0.5)).sum()),
    }


def _prospective_ratio(group, inputs, clip_eigs=True):
    """R_pros = RMS[(T/h) Delta(Kx)] / RMS[Kx], per mode, on one batch.

    Taken on the PROJECTED signal, because that is what the drive
    differences: d = Kx + (T/h)(Kx_t - Kx_{t-1}).
    """
    value = (np.clip(group["Lambda_re"], None, -1e-4) if clip_eigs
             else group["Lambda_re"]) + 1j * group["Lambda_im"]
    b_tilde = group["B"][..., 0] + 1j * group["B"][..., 1]
    lambda_bar, b_bar = discretize_zoh(value, b_tilde,
                                       np.exp(group["log_step"][:, 0]))
    T, mass, gamma, _ = response_mass_gamma(group["generalized_T_raw"],
                                            group["generalized_rho_raw"],
                                            group["generalized_gamma_raw"])
    q = mass + H * (gamma + T)
    gain = (H * T / q).astype(b_bar.dtype)[..., None] * b_bar
    projected = np.einsum("blh,ph->blp", inputs.astype(np.float32),
                          gain.real) \
        + 1j * np.einsum("blh,ph->blp", inputs.astype(np.float32), gain.imag)
    difference = projected[:, 1:] - projected[:, :-1]
    top = onp.asarray(np.sqrt(np.mean(np.abs(
        (T / H) * difference) ** 2, axis=(0, 1))))
    bottom = onp.asarray(np.sqrt(np.mean(np.abs(projected) ** 2, axis=(0, 1))))
    return _stats(top / onp.maximum(bottom, 1e-30))


def _curve(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arm", default="generalized_prospective_s5")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[301, 302, 303])
    parser.add_argument("--epochs", type=int, nargs="+",
                        default=[3, 6, 10, 15])
    parser.add_argument("--native-run",
                        help="the three-arm run root, for the curve column")
    parser.add_argument("--data-cache",
                        help="enables R_pros on one validation batch")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    inputs = None
    if arguments.data_cache:
        # the same call the runner makes, so the batch is the real thing
        from experiments.s5_three_arm_full import data as EXPERIMENT_DATA
        from experiments.s5_three_arm_full.runner import BATCH_SIZE
        # ("train", "val") is the call the runner makes and the one the
        # cache manifest is validated against; asking for "val" alone is
        # untested and this is a diagnostic, not a place to find out.
        cache = EXPERIMENT_DATA.load_official_raw(arguments.data_cache,
                                                  ("train", "val"))[0]
        inputs = np.asarray(cache["val"][0][:BATCH_SIZE])
        print(f"R_pros on one validation batch, shape {inputs.shape}",
              flush=True)

    report = {"schema": "s5-two-compartment/regime-v1", "arm": arguments.arm,
              "run_root": arguments.run_root, "epochs": arguments.epochs,
              "h": H, "seeds": {}, "curves": {}}

    for seed in arguments.seeds:
        seed_root = os.path.join(arguments.run_root, arguments.arm, str(seed))
        report["curves"][str(seed)] = _curve(
            os.path.join(seed_root, "metrics.jsonl"))
        snapshots = {}
        for epoch in arguments.epochs:
            path = os.path.join(seed_root,
                                f"checkpoint_epoch_{epoch:02d}.msgpack")
            if not os.path.exists(path):
                continue
            groups = _layer_groups(_read(path))
            layers = {}
            for name in sorted(groups):
                layers[name] = _layer(groups[name])
                if inputs is not None:
                    layers[name]["R_pros"] = _prospective_ratio(groups[name],
                                                                inputs)
            snapshots[str(epoch)] = layers
            first = layers[sorted(layers)[0]]
            print(f"seed {seed} epoch {epoch:2d}  "
                  f"T {first['T']['median']:.4f}  "
                  f"gamma {first['gamma']['median']:.3f}  "
                  f"rho {first['rho']['median']:.3f}  "
                  f"|r_slow| {first['slow_root_abs']['median']:.5f}  "
                  f"|r_fast| {first['fast_root_abs']['median']:.4f}  "
                  f"in-regime {first['modes_in_predicted_regime']:2d}/"
                  f"{first['modes']}", flush=True)
        report["seeds"][str(seed)] = snapshots

    if arguments.native_run:
        report["curves"]["native_301"] = _curve(os.path.join(
            arguments.native_run, "native_matched_s5", "301",
            "metrics.jsonl"))

    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)

    native = report["curves"].get("native_301", [])
    print()
    print(f"{'epoch':>6} {'Native':>16} " + " ".join(
        f"{'seed ' + s:>16}" for s in map(str, arguments.seeds)))
    for index in range(max([len(v) for v in report["curves"].values()] or [0])):
        row = [f"{index + 1:6d}"]
        for key in ["native_301"] + [str(s) for s in arguments.seeds]:
            curve = report["curves"].get(key, [])
            if index < len(curve):
                validation = curve[index]["validation"]
                row.append(f"{validation['accuracy'] * 100:7.2f}% "
                           f"{validation['cross_entropy']:7.4f}")
            else:
                row.append(f"{'':>16}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
