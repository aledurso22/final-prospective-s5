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
(docs/analysis/direct_prospective_block.txt): the block scan will NOT match
sequential float32 at the tau = 1000 cell, because applying H^C injects
about eps*|H^C| per boundary and |H^C| reaches 1.7e2 at C = 16 and 2.9e3 at
C = 256 while the states are O(1). If that is what the GPU shows, the block
scan is not the fix either and the finding must be reported as such.

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
#: the accepted stable cell
TAU, EPS = 1000.0, 0.25


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


def coefficients(model, lambda_bar, b_bar, dtype):
    tau = jnp.full((lambda_bar.shape[0],), TAU, dtype=dtype)
    if model == "professor_tss":
        return DP.professor_tss_state_coefficients(
            lambda_bar, b_bar, tau, DP.PROFESSOR_LINEAR_TARGET)
    mass = DP.mass_from_eps(tau, jnp.asarray(EPS, dtype=dtype))
    return DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass,
                                         DP.PROFESSOR_LINEAR_TARGET)


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


def study(model, lambda_bar, b_bar, length, steps=3):
    single = jnp.complex64
    double = jnp.complex128
    inputs32 = jax.random.normal(jax.random.PRNGKey(0),
                                 (length, b_bar.shape[1])).astype(jnp.float32)
    A32, C32 = coefficients(model, lambda_bar.astype(single),
                            b_bar.astype(single), jnp.float32)
    drive32 = DP.input_drive(C32, inputs32)
    A64, C64 = coefficients(model, lambda_bar.astype(double),
                            b_bar.astype(double), jnp.float64)
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
            "length": length,
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
        "length": length, "scan": "doubling (REJECTED for float32)",
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
    args = parser.parse_args()
    lambda_bar, b_bar = production_modes(args.seed)
    report = {"schema": "s5-direct-prospective/chunk-study-v1",
              "cell": {"tau": TAU, "eps": EPS,
                       "target": DP.PROFESSOR_LINEAR_TARGET},
              "chunks": list(CHUNKS), "results": []}
    for model in ORDERS:
        for length in args.lengths:
            baseline, rows = study(model, lambda_bar, b_bar, length)
            report["results"].append({"model": model, "length": length,
                                      "baseline": baseline, "rows": rows})
    # the decision rule, applied by code
    acceptable = []
    for block in report["results"]:
        tolerance = 10.0 * block["baseline"]["sequential_float32_vs_float64"]
        for row in block["rows"]:
            if row.get("chunk") is None:
                continue
            if (row["finite"] and row["deterministic"]
                    and row["float32_vs_sequential_float64"] <= tolerance
                    and row["gradient_vs_sequential_float64"] <= tolerance):
                acceptable.append(row)
    report["acceptance_rule"] = ("float32 forward and gradient error within "
                                 "10x the ordinary sequential float32 error, "
                                 "finite and deterministic")
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
