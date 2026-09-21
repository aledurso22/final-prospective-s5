"""Throughput and peak-memory benchmark for the two-compartment scans.

THE QUESTION. The production run used `scan_companion_sequential` and
measured about 2.98 optimizer steps per minute at length 16,000, which
projects to roughly 155 hours for a three-seed wave. That is a sequential
implementation over 16,000 tokens with a rematerialized backward pass; it
is not evidence about the equation. This measures whether a correct
parallel scan is materially faster, and reports the answer either way.

EACH ARM RUNS IN ITS OWN PROCESS. Peak device memory is a per-process high
water mark, so measuring several arms in one process reports the maximum
over all of them and attributes it to whichever ran last. The parent
spawns one child per arm and collects JSON.

    python -m experiments.s5_two_compartment.benchmark --out bench.json
    python -m experiments.s5_two_compartment.benchmark --arm factored
"""

import argparse
import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

#: (label, production arm, scan implementation)
ARMS = (
    ("native", "native_matched_s5", None),
    ("sequential", "generalized_prospective_s5", "sequential"),
    ("companion", "generalized_prospective_s5", "companion"),
    ("factored", "generalized_prospective_s5", "factored"),
)
#: the wave the projection is for
EPOCHS_PROJECTED = 15
SEEDS_PROJECTED = 3


def peak_device_bytes():
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return None
    return stats.get("peak_bytes_in_use")


def patched_factory(runner, production, implementation):
    """Build the arm's SSM with the requested scan, through the FACTORY.

    REGRESSION, and it silently voided three rows of the first benchmark.
    The first version set the class attribute:

        GeneralizedProspectiveS5SSM.implementation = implementation

    A Flax `nn.Module` is turned into a dataclass at CLASS DEFINITION time,
    so `__init__` has already captured the field defaults; assigning to the
    class attribute afterwards changes nothing an instance sees. Every
    generalized row therefore ran `scan_companion_sequential`, and the
    giveaway was `peak_device_bytes` identical to the BYTE between
    `sequential` and `companion` (8957147904 both), which two different
    algorithms cannot produce.

    The implementation now goes through `init_generalized_prospective_S5SSM`,
    which threads it into the partial, and `verify` below reads it back off
    the constructed partial rather than trusting that it took.
    """
    from s5.three_arm_factory import init_generalized_prospective_S5SSM

    original = runner.ssm_factory

    def factory(arm):
        if arm == "generalized_prospective_s5" and implementation is not None:
            return init_generalized_prospective_S5SSM(
                response_init=runner.T_INIT, rho_init=runner.RHO_INIT,
                gamma_init=1.0, implementation=implementation,
                **runner.ssm_kwargs(arm))
        return original(arm)

    runner.ssm_factory = factory
    return original


def verify(runner, production, implementation):
    """Read the scan back off the constructed module. Never assume it took.

    Returns the implementation actually in force, so the benchmark row
    carries evidence rather than an intention.
    """
    built = runner.ssm_factory(production)
    actual = getattr(built, "keywords", {}).get("implementation")
    if production != "generalized_prospective_s5":
        return None
    if actual is None:
        from s5.factored_recurrence import DEFAULT_IMPLEMENTATION
        actual = DEFAULT_IMPLEMENTATION
    if implementation is not None and actual != implementation:
        raise SystemExit(
            f"asked for {implementation!r} but the constructed module has "
            f"{actual!r}; refusing to report a row that measures something "
            f"else")
    return actual


def measure(label, warmup, steps):
    """One arm, in this process. Returns a plain dict."""
    from experiments.s5_three_arm_full import runner

    production, implementation = dict(
        (name, (arm, choice)) for name, arm, choice in ARMS)[label]
    original = patched_factory(runner, production, implementation)
    actual = verify(runner, production, implementation)

    state = runner.init_state(production, runner.SEEDS[0])
    model = runner.model_cls_for(production)(training=True)
    batch = jnp.zeros((runner.BATCH_SIZE, runner.SEQ_LEN, runner.INPUT_DIM),
                      dtype=jnp.float32)
    labels = jnp.zeros((runner.BATCH_SIZE,), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    steps_per_epoch = 64

    def one(state, index):
        return runner.train_one_batch(state, key, batch, labels, model,
                                      index, steps_per_epoch,
                                      EPOCHS_PROJECTED)[0]

    for index in range(warmup):
        state = one(state, index)
    jax.block_until_ready(state.params)
    began = time.time()
    for index in range(steps):
        state = one(state, warmup + index)
    jax.block_until_ready(state.params)
    elapsed = time.time() - began
    runner.ssm_factory = original

    per_step = elapsed / steps
    return {
        "arm": label, "production_arm": production,
        "implementation_requested": implementation,
        "implementation_in_force": actual,
        "batch": int(runner.BATCH_SIZE), "length": int(runner.SEQ_LEN),
        "steps_measured": steps, "warmup": warmup,
        "seconds_per_step": per_step,
        "steps_per_minute": 60.0 / per_step,
        "peak_device_bytes": peak_device_bytes(),
        "peak_device_gib": (peak_device_bytes() / 2 ** 30
                            if peak_device_bytes() else None),
        "recurrent_state_size": runner.recurrent_state_size(production),
    }


def project(row, steps_per_epoch):
    """The full wave, from measured throughput.

    STATE CONCURRENCY ASSUMPTION, stated rather than buried: the three
    seeds are assumed to run SEQUENTIALLY on one device, which is what the
    serialized launcher does. Running them concurrently would divide the
    wall time by three only if three devices were free and the peak memory
    of each fitted independently.
    """
    seconds = row["seconds_per_step"] * steps_per_epoch * EPOCHS_PROJECTED
    return {
        "steps_per_epoch_assumed": steps_per_epoch,
        "epochs": EPOCHS_PROJECTED, "seeds": SEEDS_PROJECTED,
        "hours_per_seed": seconds / 3600.0,
        "hours_three_seeds_sequential": seconds * SEEDS_PROJECTED / 3600.0,
        "concurrency_assumption": "three seeds sequential on one device",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--arm", choices=[name for name, _, _ in ARMS])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=536,
                        help="Speech Commands batches per epoch at batch 16")
    arguments = parser.parse_args()

    if arguments.arm:
        print(json.dumps(measure(arguments.arm, arguments.warmup,
                                 arguments.steps)))
        return

    rows = {}
    for label, _, _ in ARMS:
        command = [sys.executable, "-m",
                   "experiments.s5_two_compartment.benchmark",
                   "--arm", label, "--warmup", str(arguments.warmup),
                   "--steps", str(arguments.steps)]
        print(f"--- {label} ---", flush=True)
        finished = subprocess.run(command, capture_output=True, text=True,
                                  env=dict(os.environ))
        if finished.returncode != 0:
            rows[label] = {"arm": label, "failed": True,
                           "stderr": finished.stderr[-4000:]}
            print(finished.stderr[-4000:], flush=True)
            continue
        row = json.loads(finished.stdout.strip().splitlines()[-1])
        row["projection"] = project(row, arguments.steps_per_epoch)
        rows[label] = row
        print(json.dumps(row, indent=2), flush=True)

    ok = {k: v for k, v in rows.items() if not v.get("failed")}
    # GUARD. Two different scans cannot allocate identically to the byte.
    # The first benchmark reported sequential and companion both at exactly
    # 8957147904 because the implementation never changed, so identical
    # peaks are now called out rather than tabulated.
    peaks = {}
    for name, row in ok.items():
        peaks.setdefault(row.get("peak_device_bytes"), []).append(name)
    suspicious = {int(peak): names for peak, names in peaks.items()
                  if peak and len(names) > 1}
    report = {"schema": "s5-two-compartment/benchmark-v2", "arms": rows,
              "implementations_in_force": {
                  k: v.get("implementation_in_force") for k, v in ok.items()},
              "identical_peak_memory_suspicious": suspicious}
    if "sequential" in ok:
        base = ok["sequential"]["seconds_per_step"]
        report["speedup_over_sequential"] = {
            k: base / v["seconds_per_step"] for k, v in ok.items()}
    if "native" in ok:
        base = ok["native"]["seconds_per_step"]
        report["cost_relative_to_native"] = {
            k: v["seconds_per_step"] / base for k, v in ok.items()}
    if arguments.out:
        with open(arguments.out, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in report if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
