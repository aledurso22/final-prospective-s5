"""Native versus one-stage versus two-stage, at the production shape.

Measures what the cascade costs: the S5 scan is unchanged and each stage
adds one scalar associative scan of the same shape, so the expectation is
near-Native throughput and a small constant memory increase. The expectation
is not assumed -- it is measured, with compilation timed separately.

Runs no training and submits nothing.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp

from experiments.s5_three_arm_full import runner as RUNNER
from s5 import modal_prospective as MP

LENGTH = RUNNER.SEQ_LEN
MODES = 64
FEATURES = RUNNER.D_MODEL
REPEATS = 5


def peak_bytes():
    try:
        return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
    except Exception:
        return None


def fixture(seed, dtype=jnp.complex64):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    magnitude = 0.2 + 0.75 * jax.random.uniform(keys[0], (MODES,))
    angle = jax.random.uniform(keys[1], (MODES,), minval=-3.1, maxval=3.1)
    lambda_bar = (magnitude * jnp.exp(1j * angle)).astype(dtype)
    b_bar = jax.random.normal(keys[2], (MODES, FEATURES)).astype(dtype)
    inputs = jax.random.normal(jax.random.PRNGKey(seed + 1),
                               (LENGTH, FEATURES)).astype(jnp.float32)
    return lambda_bar, b_bar, inputs


def stages_for(count, dtype=jnp.complex64):
    pairs = ((1.0, 3.0), (0.5, 2.0))[:count]
    return [(jnp.full((MODES,), d, dtype=dtype),
             jnp.full((MODES,), n, dtype=dtype)) for d, n in pairs]


def measure(label, stages, lambda_bar, b_bar, inputs):
    def forward(b):
        states = MP.native_states(lambda_bar, b, inputs)
        return MP.apply_cascade(states, stages)

    def loss(b):
        return jnp.sum(jnp.abs(forward(b)) ** 2).real

    compiled_forward = jax.jit(forward)
    compiled_grad = jax.jit(jax.grad(loss))

    started = time.perf_counter()
    jax.block_until_ready(compiled_forward(b_bar))
    forward_compile = time.perf_counter() - started
    started = time.perf_counter()
    jax.block_until_ready(compiled_grad(b_bar))
    backward_compile = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(REPEATS):
        out = compiled_forward(b_bar)
    jax.block_until_ready(out)
    forward_seconds = (time.perf_counter() - started) / REPEATS

    started = time.perf_counter()
    for _ in range(REPEATS):
        grads = compiled_grad(b_bar)
    jax.block_until_ready(grads)
    backward_seconds = (time.perf_counter() - started) / REPEATS

    return {"arm": label, "stages": len(stages),
            "forward_compile_seconds": forward_compile,
            "backward_compile_seconds": backward_compile,
            "forward_seconds": forward_seconds,
            "forward_plus_backward_seconds": backward_seconds,
            "tokens_per_second": LENGTH / forward_seconds,
            "peak_gpu_bytes": peak_bytes(),
            "finite": bool(jnp.all(jnp.isfinite(out)))
                      and bool(jnp.all(jnp.isfinite(grads))),
            "added_poles": [float(value) for value in
                            jnp.ravel(MP.added_poles(stages).real)][:4]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=301)
    args = parser.parse_args()
    lambda_bar, b_bar, inputs = fixture(args.seed)
    rows = [measure(label, stages_for(count), lambda_bar, b_bar, inputs)
            for label, count in (("native", 0), ("one_stage", 1),
                                 ("two_stage", 2))]
    native = rows[0]
    for row in rows:
        row["forward_ratio_to_native"] = (
            native["forward_seconds"] / row["forward_seconds"])
        row["memory_ratio_to_native"] = (
            row["peak_gpu_bytes"] / native["peak_gpu_bytes"]
            if row["peak_gpu_bytes"] and native["peak_gpu_bytes"] else None)
    report = {"schema": "s5-modal/benchmark-v1",
              "shape": {"length": LENGTH, "modes": MODES,
                        "features": FEATURES, "precision": "float32/complex64"},
              "rows": rows,
              "note": "every added pole is d/(h+d) < 1 by construction; the "
                      "S5 scan itself is unchanged"}
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
