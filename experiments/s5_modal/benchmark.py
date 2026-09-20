"""Native versus one-stage versus two-stage, at the production shape.

MEASURED, and it corrected the expectation. A first run gave 0.58x Native
for one stage and 0.46x for two -- each stage costs about one more scan of
the same size, which is what three scans against one should cost. The
earlier "near-Native throughput" prediction was wrong and is withdrawn.

Two things that first run got wrong, fixed here:

  * PEAK MEMORY WAS NOT ISOLATED. `peak_bytes_in_use` is a process-wide
    high-water mark, so all three arms reported the identical figure and
    the memory ratio of 1.0 was an artifact, not a measurement. Each arm is
    now measured in its OWN process, and `--arm` runs exactly one.
  * THE SCAN WAS MEASURED IN ISOLATION. The layer also does the readout,
    the norm and the GLU, so a scan-level ratio overstates the model-level
    cost. A layer-level comparison is measured too, and it is the one that
    decides feasibility.

Runs no training and submits nothing.
"""

import argparse
import json
import subprocess
import sys
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


ARMS = {"native": 0, "one_stage": 1, "two_stage": 2}


def layer_comparison(seed):
    """The LAYER, not the scan alone: readout, norm and GLU included.

    This is the ratio that decides feasibility; the scan-level ratio
    overstates the cost because the scan is only part of the layer.
    """
    import numpy

    from s5.modal_prospective_ssm import init_modal_prospective_S5SSM
    from s5.ssm import init_S5SSM

    # `init_S5SSM` takes `bidirectional` as a REQUIRED argument, which the
    # first version omitted; both constructors are given the same kwargs so
    # the comparison is like for like
    kwargs = dict(Lambda_re_init=-0.5 * numpy.ones(MODES),
                  Lambda_im_init=numpy.linspace(0.1, 30.0, MODES),
                  V=numpy.eye(MODES, dtype=numpy.complex64),
                  Vinv=numpy.eye(MODES, dtype=numpy.complex64),
                  H=FEATURES, P=MODES, C_init="lecun_normal",
                  discretization="zoh", dt_min=0.001, dt_max=0.1,
                  conj_sym=False, clip_eigs=True, bidirectional=False)
    inputs = jax.random.normal(jax.random.PRNGKey(seed),
                               (LENGTH, FEATURES)).astype(jnp.float32)
    out = {}
    for label, constructor in (("native_layer", init_S5SSM(**kwargs)),
                               ("modal_layer",
                                init_modal_prospective_S5SSM(**kwargs))):
        model = constructor()
        # both layers are the real production classes, not stand-ins
        variables = model.init(jax.random.PRNGKey(0), inputs)
        forward = jax.jit(lambda params, m=model: m.apply(params, inputs))
        jax.block_until_ready(forward(variables))
        started = time.perf_counter()
        for _ in range(REPEATS):
            values = forward(variables)
        jax.block_until_ready(values)
        out[label] = {"seconds": (time.perf_counter() - started) / REPEATS,
                      "finite": bool(jnp.all(jnp.isfinite(values)))}
    out["modal_ratio_to_native"] = (out["native_layer"]["seconds"]
                                    / out["modal_layer"]["seconds"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--arm", choices=sorted(ARMS),
                        help="measure ONE arm, so peak memory is this "
                             "process's alone; the orchestrator uses it")
    parser.add_argument("--layer-only", action="store_true")
    args = parser.parse_args()

    if args.arm:
        lambda_bar, b_bar, inputs = fixture(args.seed)
        row = measure(args.arm, stages_for(ARMS[args.arm]), lambda_bar,
                      b_bar, inputs)
        with open(args.out, "w") as handle:
            json.dump(row, handle, indent=2)
        print(json.dumps(row, indent=2))
        return

    if args.layer_only:
        report = {"schema": "s5-modal/benchmark-layer-v1",
                  "layer": layer_comparison(args.seed)}
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
        return

    # each arm in its OWN process, so peak_bytes_in_use is not a shared
    # high-water mark across arms
    rows = []
    for label in ("native", "one_stage", "two_stage"):
        path = f"{args.out}.{label}.json"
        subprocess.run([sys.executable, "-m",
                        "experiments.s5_modal.benchmark", "--out", path,
                        "--seed", str(args.seed), "--arm", label],
                       check=True)
        with open(path) as handle:
            rows.append(json.load(handle))
    native = rows[0]
    for row in rows:
        row["forward_ratio_to_native"] = (
            native["forward_seconds"] / row["forward_seconds"])
        row["memory_ratio_to_native"] = (
            row["peak_gpu_bytes"] / native["peak_gpu_bytes"]
            if row["peak_gpu_bytes"] and native["peak_gpu_bytes"] else None)
    report = {"schema": "s5-modal/benchmark-v2",
              "shape": {"length": LENGTH, "modes": MODES,
                        "features": FEATURES, "precision": "float32/complex64"},
              "rows": rows,
              "layer": layer_comparison(args.seed),
              "measurement_notes": [
                  "each arm ran in its own process, so peak_bytes_in_use is "
                  "not shared between arms",
                  "the row ratios compare the SCAN alone; the layer entry "
                  "compares the whole layer, which is the feasibility "
                  "number",
                  "every added pole is d/(h+d) < 1 by construction; the S5 "
                  "scan itself is unchanged"]}
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
