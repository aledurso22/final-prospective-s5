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


def measure(label, warmup, steps):
    """One arm, in this process. Returns a plain dict."""
    from experiments.s5_three_arm_full import runner

    arm = dict((name, (production, implementation))
               for name, production, implementation in ARMS)[label]
    production, implementation = arm

    if implementation is not None:
        # the ONLY thing that changes between the generalized rows: which
        # scan evaluates the identical recurrence
        import s5.generalized_prospective_ssm as GP
        original = GP.GeneralizedProspectiveS5SSM.implementation
        GP.GeneralizedProspectiveS5SSM.implementation = implementation
    else:
        original = None

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

    if original is not None:
        import s5.generalized_prospective_ssm as GP
        GP.GeneralizedProspectiveS5SSM.implementation = original

    per_step = elapsed / steps
    return {
        "arm": label, "production_arm": production,
        "implementation": implementation,
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
    report = {"schema": "s5-two-compartment/benchmark-v1", "arms": rows}
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
