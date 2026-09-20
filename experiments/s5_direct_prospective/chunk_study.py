"""Chunk-size study for the block scan, on REAL initialized S5 modes.

Measures, for both orders and every chunk size, exactly what decides whether
the block scan can replace the rejected full-sequence doubling scan:

  * float32 forward error versus float32 sequential (the fair baseline);
  * float32 forward error versus float64 sequential (the truth);
  * gradient error, same two baselines;
  * recurrence residual, which is zero for a sequential solution by
    construction and nonzero for anything that reassociates;
  * determinism (same inputs twice, bitwise);
  * runtime and peak GPU memory, compilation timed separately;
  * behaviour of the compiled executable at sequence length 16000.

THE CHUNK SIZE IS NOT CHOSEN BY SPEED. A cell passes only if its float32
error and gradient error are comparable to the ordinary sequential float32
recurrence; the fastest cell that fails that is rejected.

Prediction to be tested, from emulated float32 off-cluster
(docs/analysis/direct_prospective_stable_cells.txt): the block scan MATCHES
the sequential recurrence to within 2-5x at the well-conditioned cells
(tau = 2, 5, 10, where |H^64| <= 4.6) and FAILS at the nearly marginal ones
(tau = 100 and 1000, where |H^64| is 3.7e2 and 9.4e2). |H^C| is the
predictor, and it is recorded for every row.

Runs no training and submits nothing.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy
from flax.traverse_util import flatten_dict

from experiments.s5_three_arm_full import runner as RUNNER
from s5 import direct_prospective as DP
from s5.ssm import discretize_zoh

CHUNKS = (16, 32, 64, 128, 256)
ORDERS = {"professor_tss": 2, "wwj_generalized_tss": 3}
#: every cell the cluster measured as stable, NOT only the weak tau = 1000
#: limit: (tau, measured radius at eps = 0, measured radius at eps = 1/4)
CLUSTER_STABLE_CELLS = (
    (2.0, 0.9512748075, 0.9923729495),
    (5.0, 0.8294413371, 0.8776972719),
    (10.0, 0.8990310402, 0.8851554058),
    (50.0, 0.9799557162, 0.9662813133),
    (100.0, 0.9899888330, 0.9819794261),
    (1000.0, 0.9989998876, 0.9980477225),
)
#: |H^C| is what decides whether a chunk size is usable in float32: off
#: cluster, |H^64| = 2.0 at tau = 5 (block matches sequential to 4x) and
#: 9.4e2 at tau = 1000 (block fails). The rule below is applied in code.
TRANSITION_NORM_CEILING = 10.0


def production_modes(seed=301, layer=0):
    state = RUNNER.init_state("native_matched_s5", seed)
    flat = flatten_dict(state.params)
    keys = [key for key in flat if key[-1] == "Lambda_re"]
    prefix = sorted(keys)[layer][:-1]
    lambda_continuous = (jnp.clip(flat[prefix + ("Lambda_re",)], None, -1e-4)
                         + 1j * flat[prefix + ("Lambda_im",)])
    b = flat[prefix + ("B",)]
    step = jnp.exp(flat[prefix + ("log_step",)][:, 0])
    return discretize_zoh(lambda_continuous, b[..., 0] + 1j * b[..., 1], step)


def coefficients(model, lambda_bar, b_bar, dtype, tau_value):
    tau = jnp.full((lambda_bar.shape[0],), tau_value, dtype=dtype)
    if model == "professor_tss":
        return DP.professor_tss_state_coefficients(
            lambda_bar, b_bar, tau, DP.PROFESSOR_LINEAR_TARGET)
    mass = DP.mass_from_eps(tau, jnp.asarray(0.25, dtype=dtype))
    return DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass,
                                         DP.PROFESSOR_LINEAR_TARGET)


def transition_norm(coefficients_A, chunk):
    """max |H^C| per mode: the predictor of float32 usability."""
    transition = DP.chunk_transition(coefficients_A, chunk)
    return float(jnp.max(jnp.abs(transition)))


def relative(candidate, reference):
    candidate = numpy.asarray(candidate, dtype=numpy.complex128)
    reference = numpy.asarray(reference, dtype=numpy.complex128)
    scale = float(numpy.max(numpy.abs(reference)))
    if not numpy.isfinite(scale) or scale == 0.0:
        return float("inf")
    return float(numpy.max(numpy.abs(candidate - reference))) / scale


def residual(coefficients_A, drive, states):
    """max |s_t - (sum_i A_i s_{t-i} + d_t)| / scale. Zero for a sequential
    solution by construction."""
    order = len(coefficients_A)
    predicted = drive
    for index, coefficient in enumerate(coefficients_A):
        shifted = jnp.concatenate(
            (jnp.zeros((index + 1,) + states.shape[1:], states.dtype),
             states[:-(index + 1)]), axis=0)
        predicted = predicted + coefficient * shifted
    scale = float(jnp.max(jnp.abs(states)))
    if not numpy.isfinite(scale) or scale == 0.0:
        return float("inf")
    return float(jnp.max(jnp.abs(states - predicted))) / scale


def peak_bytes():
    try:
        return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
    except Exception:
        return None


def study(model, lambda_bar, b_bar, length, tau_value, steps=3):
    single = jnp.complex64
    double = jnp.complex128
    inputs32 = jax.random.normal(jax.random.PRNGKey(0),
                                 (length, b_bar.shape[1])).astype(jnp.float32)
    A32, C32 = coefficients(model, lambda_bar.astype(single),
                            b_bar.astype(single), jnp.float32, tau_value)
    drive32 = DP.input_drive(C32, inputs32)
    A64, C64 = coefficients(model, lambda_bar.astype(double),
                            b_bar.astype(double), jnp.float64, tau_value)
    drive64 = DP.input_drive(C64, inputs32.astype(jnp.float64))

    sequential32 = DP.sequential_scan_jax(A32, drive32)
    sequential64 = DP.sequential_scan_jax(A64, drive64)
    baseline = {
        "sequential_float32_vs_float64": relative(sequential32, sequential64),
        "sequential_float32_residual": residual(A32, drive32, sequential32),
    }

    def gradient(coefficients_A, drive, scan_kind, chunk):
        def loss(values):
            states = DP.run_scan(values, drive, scan_kind, chunk)
            return jnp.sum(jnp.abs(states) ** 2).real
        return jax.grad(loss)(tuple(coefficients_A))

    reference_gradient = gradient(A64, drive64, "sequential", 0)
    rows = []
    for chunk in CHUNKS:
        compiled = jax.jit(lambda d, c=chunk: DP.block_scan(A32, d, c))
        started = time.perf_counter()
        states = compiled(drive32)
        jax.block_until_ready(states)
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(steps):
            states = compiled(drive32)
        jax.block_until_ready(states)
        seconds = (time.perf_counter() - started) / steps
        again = compiled(drive32)
        grads = gradient(A32, drive32, "block", chunk)
        rows.append({
            "model": model, "order": ORDERS[model], "chunk": chunk,
            "length": length, "tau": tau_value,
            "transition_norm": transition_norm(A32, chunk),
            "float32_vs_sequential_float32": relative(states, sequential32),
            "float32_vs_sequential_float64": relative(states, sequential64),
            "gradient_vs_sequential_float64":
                max(relative(g, r) for g, r in zip(grads,
                                                   reference_gradient)),
            "residual": residual(A32, drive32, states),
            "deterministic": bool(jnp.all(states == again)),
            "finite": bool(jnp.all(jnp.isfinite(states))),
            "compile_seconds": compile_seconds,
            "seconds_per_call": seconds,
            "peak_gpu_bytes": peak_bytes(),
        })
    # the rejected full-sequence doubling scan, for contrast only
    doubling = DP.companion_doubling_scan(A32, drive32)
    rows.append({
        "model": model, "order": ORDERS[model], "chunk": None,
        "length": length, "tau": tau_value,
        "scan": "doubling (REJECTED for float32)",
        "float32_vs_sequential_float32": relative(doubling, sequential32),
        "float32_vs_sequential_float64": relative(doubling, sequential64),
        "finite": bool(jnp.all(jnp.isfinite(doubling))),
    })
    return baseline, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[1000, 16000])
    parser.add_argument("--taus", type=float, nargs="*", default=[],
                        help="restrict to these tau values; default is every "
                             "cluster-measured stable cell")
    args = parser.parse_args()
    lambda_bar, b_bar = production_modes(args.seed)
    report = {"schema": "s5-direct-prospective/chunk-study-v1",
              "cell": {"tau": TAU, "eps": EPS,
                       "target": DP.PROFESSOR_LINEAR_TARGET},
              "chunks": list(CHUNKS), "results": []}
    for model in ORDERS:
        for tau_value, radius_zero, radius_critical in CLUSTER_STABLE_CELLS:
            if args.taus and tau_value not in args.taus:
                continue
            for length in args.lengths:
                baseline, rows = study(model, lambda_bar, b_bar, length,
                                       tau_value)
                report["results"].append({
                    "model": model, "tau": tau_value, "length": length,
                    "cluster_radius": (radius_zero if model == "professor_tss"
                                       else radius_critical),
                    "baseline": baseline, "rows": rows})
    # the decision rule, applied by code
    acceptable = []
    for block in report["results"]:
        tolerance = 10.0 * block["baseline"]["sequential_float32_vs_float64"]
        for row in block["rows"]:
            if row.get("chunk") is None:
                continue
            if (row["finite"] and row["deterministic"]
                    and row["transition_norm"] <= TRANSITION_NORM_CEILING
                    and row["float32_vs_sequential_float64"] <= tolerance
                    and row["gradient_vs_sequential_float64"] <= tolerance):
                acceptable.append(dict(row, model=block["model"],
                                       tau=block["tau"]))
    report["acceptance_rule"] = (
        f"|H^C| <= {TRANSITION_NORM_CEILING}, and float32 forward AND "
        "gradient error within 10x the ordinary sequential float32 error, "
        "finite and deterministic. The chunk size is never chosen by speed.")
    report["transition_norm_ceiling"] = TRANSITION_NORM_CEILING
    report["acceptable_cells"] = acceptable
    report["status"] = "BLOCK_SCAN_USABLE" if acceptable else \
        "BLOCK_SCAN_NOT_USABLE_IN_FLOAT32"
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"status": report["status"],
                      "acceptable_cells": acceptable,
                      "results": report["results"]}, indent=2))
    if not acceptable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
