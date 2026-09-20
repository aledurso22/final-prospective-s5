"""Production-shaped Native-versus-WWJ benchmark, and the performance gate.

The previous generalized arm was not wrong, it was unusable: about 2.98
training steps per minute, roughly 154.7 hours left in a wave, against about
5.6 hours of allocation. Being merely faster than that is not the bar. This
script measures both arms under IDENTICAL production conditions -- same GPU,
batch size 16, sequence length 16000, width 96, depth 6, bidirectional, the
same optimizer update and precision -- and applies the declared gate.

THE GATE (all of it must hold, or full training is not authorized):

  1. the production-shaped forward, backward and optimizer update complete;
  2. peak GPU memory stays below MEMORY_CEILING_FRACTION of the device;
  3. loss, gradients and states are finite;
  4. the parallel scan matches the sequential oracle at production width;
  5. the projected wall clock for 15 epochs fits the requested allocation
     with >= TIME_MARGIN to spare, for the plan that `--waves` declares:
     one arm's wave (three seeds concurrent on three GPUs) or both WWJ arms
     as two sequential waves. A one-arm projection is never reported as the
     wall time of the whole WWJ experiment;
  6. WWJ throughput is at least THROUGHPUT_FLOOR of Native throughput.

The floor is 0.80, not the earlier 0.50, because the architecture changed:
the expensive part is now the EXISTING optimized Native S5 scan and the WWJ
addition is an O(L*P) three-tap, so throughput should be near Native.
Failing 6 means the implementation needs optimizing, not that training
should start.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp

from experiments.s5_three_arm_full import runner as RUNNER
from experiments.s5_wwj.wwj_model import (create_wwj_train_state,
                                          set_wwj_learning_rates,
                                          wwj_diagnostics, wwj_model)

#: gate constants, declared before the measurement
MEMORY_CEILING_FRACTION = 0.80
TIME_MARGIN = 0.25
THROUGHPUT_FLOOR = 0.80
#: the protocol this benchmark projects
EPOCHS = 15
TRAIN_EXAMPLES = 30769


def _finite(tree):
    return bool(all(bool(jnp.all(jnp.isfinite(value)))
                    for value in jax.tree_util.tree_leaves(tree)
                    if hasattr(value, "dtype")))


def _peak_bytes():
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return None
    return stats.get("peak_bytes_in_use")


def _device_bytes():
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return None
    return stats.get("bytes_limit")


def _batch(seed):
    rng = jax.random.PRNGKey(seed)
    x = jax.random.normal(rng, (RUNNER.BATCH_SIZE, RUNNER.SEQ_LEN,
                                RUNNER.INPUT_DIM))
    y = jax.random.randint(rng, (RUNNER.BATCH_SIZE,), 0, 10)
    return x, y


def measure(label, state, model, eval_model, steps, seed):
    """Compilation, steady-state throughput, forward and backward timings."""
    x, y = _batch(seed)

    def update(state, step):
        return RUNNER.train_one_batch_observable(
            state, jax.random.PRNGKey(seed * 1000 + step), x, y, model, step,
            1, EPOCHS)

    started = time.perf_counter()
    state, loss, accuracy, grad_norm, gradients_finite = update(state, 0)
    jax.block_until_ready(state.params)
    compile_seconds = time.perf_counter() - started
    finite = bool(gradients_finite) and _finite(state.params) and \
        bool(jnp.isfinite(loss))

    started = time.perf_counter()
    for step in range(1, steps + 1):
        state, loss, accuracy, grad_norm, gradients_finite = update(state, step)
    jax.block_until_ready(state.params)
    steady_seconds = time.perf_counter() - started
    finite = finite and bool(gradients_finite) and bool(jnp.isfinite(loss))

    # forward and backward are timed on the EVALUATION model, so dropout
    # randomness and batch-statistics updates do not enter the timing
    timesteps = jnp.ones((x.shape[0], RUNNER.SEQ_LEN))
    forward = jax.jit(lambda params: jnp.sum(eval_model.apply(
        {"params": params, "batch_stats": state.batch_stats},
        x, timesteps) ** 2))
    value = forward(state.params)
    jax.block_until_ready(value)
    started = time.perf_counter()
    for _ in range(3):
        value = forward(state.params)
    jax.block_until_ready(value)
    forward_seconds = (time.perf_counter() - started) / 3.0

    backward = jax.jit(jax.grad(forward))
    grads = backward(state.params)
    jax.block_until_ready(grads)
    started = time.perf_counter()
    for _ in range(3):
        grads = backward(state.params)
    jax.block_until_ready(grads)
    forward_backward_seconds = (time.perf_counter() - started) / 3.0

    per_step = steady_seconds / steps
    return state, {
        "arm": label,
        "compile_seconds": compile_seconds,
        "measured_steps": steps,
        "seconds_per_step": per_step,
        "steps_per_minute": 60.0 / per_step,
        "forward_seconds": forward_seconds,
        "backward_seconds": max(forward_backward_seconds - forward_seconds,
                                0.0),
        "forward_backward_seconds": forward_backward_seconds,
        "peak_gpu_bytes": _peak_bytes(),
        "device_bytes": _device_bytes(),
        "finite": finite,
        "loss_after_measurement": float(loss),
        "gradient_norm": float(grad_norm),
    }


def scan_matches_oracle(seed=301, length=256, tau=0.05, eps=0.25):
    """The optimized path against the frozen sequential oracle, at production
    width -- and against the AUGMENTED recurrent realization, which is the
    claim that matters: Native scan plus three-tap is not an approximation of
    it."""
    from s5.wwj_operator import k_and_m, wwj_states
    from tests import wwj_operator_reference as ORACLE

    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    radius = 0.2 + 0.7 * jax.random.uniform(keys[0], (64,))
    angle = jax.random.uniform(keys[1], (64,), minval=-2.0, maxval=2.0)
    lambda_bar = (radius * jnp.exp(1j * angle)).astype(jnp.complex64)
    b_bar = jax.random.normal(keys[2], (64, RUNNER.D_MODEL)).astype(
        jnp.complex64)
    inputs = jax.random.normal(keys[0], (length, RUNNER.D_MODEL))
    k, m = k_and_m(jnp.asarray(tau, dtype=jnp.float32),
                   jnp.asarray(eps, dtype=jnp.float32))
    fast = wwj_states(lambda_bar, b_bar, inputs, k, m)
    slow = ORACLE.wwj_sequential(lambda_bar, b_bar, inputs, k, m)
    augmented = ORACLE.augmented_sequential(lambda_bar, b_bar, inputs, k, m)
    scale = float(jnp.maximum(jnp.max(jnp.abs(slow)), 1.0))
    oracle_error = float(jnp.max(jnp.abs(fast - slow))) / scale
    augmented_error = float(jnp.max(jnp.abs(fast - augmented))) / scale
    return {"relative_error_versus_oracle": oracle_error,
            "relative_error_versus_augmented_realization": augmented_error,
            "tolerance": 2e-4,
            "matches": max(oracle_error, augmented_error) <= 2e-4}


def project(steps_per_minute, compile_seconds, waves):
    """Wall clock for 15 epochs, and WHAT THAT NUMBER COVERS.

    Three seeds of one arm run CONCURRENTLY on three GPUs, so one arm's wave
    costs one seed's time, not three. It does NOT cover the other WWJ arm:
    `wwj_critical_s5` and `wwj_passive_s5` are two waves. `waves` states
    which plan is being projected, and the gate is applied to
    `hours_planned`, never to the one-arm number when two waves are planned.
    """
    steps_per_epoch = TRAIN_EXAMPLES // RUNNER.BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    hours = total_steps / steps_per_minute / 60.0 + compile_seconds / 3600.0
    scope = {1: "ONE arm: three seeds concurrent on three GPUs. The other "
                "WWJ arm needs its own allocation and is NOT included.",
             2: "BOTH WWJ arms, run as two sequential waves in one "
                "allocation, three seeds concurrent within each wave."}
    return {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "hours_one_seed": hours,
            "hours_one_arm_wave_three_concurrent_seeds": hours,
            "hours_both_wwj_arms_sequential": 2.0 * hours,
            "waves_planned": waves, "hours_planned": waves * hours,
            "covers": scope[waves],
            "excludes": "the Native and Zucchet waves, which this "
                        "measurement does not project"}


def gate(native, wwj, oracle, projection, allocation_hours):
    problems = []
    if not wwj["finite"]:
        problems.append("WWJ forward/backward/update produced a nonfinite value")
    if not oracle["matches"]:
        problems.append(
            "the optimized path does not match the sequential oracle and "
            "the augmented realization: relative errors "
            f"{oracle['relative_error_versus_oracle']:.3e} / "
            f"{oracle['relative_error_versus_augmented_realization']:.3e}")
    peak, limit = wwj["peak_gpu_bytes"], wwj["device_bytes"]
    if peak is None or limit is None:
        problems.append("GPU memory could not be measured")
    elif peak > MEMORY_CEILING_FRACTION * limit:
        problems.append(f"peak GPU memory {peak / 2**30:.2f} GiB exceeds "
                        f"{MEMORY_CEILING_FRACTION:.0%} of "
                        f"{limit / 2**30:.2f} GiB")
    budget = allocation_hours * (1.0 - TIME_MARGIN)
    if projection["hours_planned"] > budget:
        problems.append(
            f"projected {projection['hours_planned']:.2f} h for "
            f"{projection['waves_planned']} wave(s) exceeds the "
            f"{budget:.2f} h budget ({allocation_hours:.1f} h minus a "
            f"{TIME_MARGIN:.0%} margin)")
    ratio = wwj["steps_per_minute"] / native["steps_per_minute"]
    if ratio < THROUGHPUT_FLOOR:
        problems.append(f"WWJ throughput is {ratio:.2f}x Native, below the "
                        f"{THROUGHPUT_FLOOR:.2f}x floor: optimize the scan "
                        "before training")
    return {"authorized": not problems, "problems": problems,
            "throughput_ratio_to_native": ratio}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", default="wwj_critical_s5")
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tau-init", type=float, default=0.25)
    parser.add_argument("--eps-init", type=float, default=0.0625)
    parser.add_argument("--allocation-hours", type=float, required=True)
    parser.add_argument("--waves", type=int, choices=(1, 2), default=2,
                        help="1: this arm only, in its own allocation. "
                             "2 (default): both WWJ arms sequentially in one "
                             "allocation. The gate uses this.")
    args = parser.parse_args()

    native_state = RUNNER.init_state("native_matched_s5", args.seed)
    native_model = RUNNER.model_for("native_matched_s5", True)
    _, native = measure("native_matched_s5", native_state, native_model,
                        RUNNER.model_for("native_matched_s5", False),
                        args.steps, args.seed)
    del native_state

    wwj_state = create_wwj_train_state(args.arm, args.seed,
                                       tau_init=args.tau_init,
                                       eps_init=args.eps_init)
    wwj_state = set_wwj_learning_rates(wwj_state, RUNNER.LR, RUNNER.SSM_LR)
    model = wwj_model(args.arm, True, tau_init=args.tau_init,
                      eps_init=args.eps_init)
    eval_model = wwj_model(args.arm, False, tau_init=args.tau_init,
                           eps_init=args.eps_init)
    wwj_state, wwj = measure(args.arm, wwj_state, model, eval_model,
                             args.steps, args.seed)

    oracle = scan_matches_oracle(args.seed)
    projection = project(wwj["steps_per_minute"], wwj["compile_seconds"],
                         args.waves)
    report = {
        "schema": "s5-wwj/benchmark-v1",
        "protocol": {"epochs": EPOCHS, "batch_size": RUNNER.BATCH_SIZE,
                     "seq_len": RUNNER.SEQ_LEN, "d_model": RUNNER.D_MODEL,
                     "n_layers": RUNNER.N_LAYERS,
                     "bidirectional": RUNNER.SHARED_CONFIG.bidirectional},
        "native": native, "wwj": wwj, "scan_versus_oracle": oracle,
        "projection": projection,
        "diagnostics": wwj_diagnostics(wwj_state, args.arm),
        "gate": gate(native, wwj, oracle, projection, args.allocation_hours),
        "gate_constants": {"memory_ceiling_fraction": MEMORY_CEILING_FRACTION,
                           "time_margin": TIME_MARGIN,
                           "throughput_floor": THROUGHPUT_FLOOR},
        "historical_reference": {
            "old_generalized_steps_per_minute": 2.98,
            "old_generalized_projected_hours": 154.73,
            "rejected_mixed_stencil":
                "unstable on S5 modes (max companion radius ~1.705, float32 "
                "NaN, float64 states ~1e81); not benchmarked"},
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"native_steps_per_minute": native["steps_per_minute"],
                      "wwj_steps_per_minute": wwj["steps_per_minute"],
                      "gate": report["gate"],
                      "projection": projection}, indent=2))
    if not report["gate"]["authorized"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
