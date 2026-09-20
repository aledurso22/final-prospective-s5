"""Bounded DEVELOPMENTAL check of a WWJ arm. Not a scientific result.

What it does, and nothing more: production-shaped initialization and
compilation, two genuine optimizer updates, a checkpoint saved and reloaded,
a short FIXED training subset, and validation. It never touches the test
split -- only the finalizer of a real experiment ever does, and this is not
one. Its output is written to `dev_gate.json` and must be reported
separately from any 15-epoch comparison.

It passes only when loss, gradients, states and parameters stay finite, the
loss on the fixed subset actually falls, the per-mode companion radii stay
inside the declared ceiling, the checkpoint restores to a finite state, and
the process reaches the end by itself.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
from flax import serialization

from dataloaders import speech_commands10 as SC
from experiments.s5_three_arm_full import data as EXPERIMENT_DATA
from experiments.s5_three_arm_full import runner as RUNNER
from s5.wwj_prospective_ssm import WWJ_SCIENTIFIC_NAMES
from experiments.s5_wwj.wwj_model import (create_wwj_train_state,
                                          set_wwj_learning_rates,
                                          wwj_diagnostics, wwj_model)

EPOCHS = 15
#: the loss on the fixed subset must fall by at least this fraction
REQUIRED_LOSS_REDUCTION = 0.02
#: companion radii above this are reported as a failure of the gate
RADIUS_CEILING = 1.05


def _finite(tree):
    return bool(all(bool(jnp.all(jnp.isfinite(value)))
                    for value in jax.tree_util.tree_leaves(tree)
                    if hasattr(value, "dtype")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", default="wwj_critical_s5")
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--subset-steps", type=int, default=20)
    parser.add_argument("--tau-init", type=float, default=0.25)
    parser.add_argument("--eps-init", type=float, default=0.0625)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # train and validation only; the test split is never requested here
    cache = EXPERIMENT_DATA.load_official_raw(args.data_cache,
                                              ("train", "val"))[0]
    train, val = cache["train"], cache["val"]

    started = time.perf_counter()
    state = create_wwj_train_state(args.arm, args.seed,
                                   tau_init=args.tau_init,
                                   eps_init=args.eps_init)
    state = set_wwj_learning_rates(state, RUNNER.LR, RUNNER.SSM_LR)
    model = wwj_model(args.arm, True, tau_init=args.tau_init,
                      eps_init=args.eps_init)
    eval_model = wwj_model(args.arm, False, tau_init=args.tau_init,
                           eps_init=args.eps_init)
    initialization_seconds = time.perf_counter() - started

    steps_per_epoch = len(train[1]) // RUNNER.BATCH_SIZE
    fixed_batches = []
    for index, indices in enumerate(
            SC.epoch_batches(len(train[1]), RUNNER.BATCH_SIZE, args.seed, 0,
                             drop_last=True)):
        fixed_batches.append(RUNNER.batch_arrays(*train, indices))
        if len(fixed_batches) == args.subset_steps:
            break

    record = {"schema": "s5-wwj/dev-gate-v1", "arm": args.arm,
              "scientific_name": WWJ_SCIENTIFIC_NAMES[args.arm],
              "seed": args.seed, "epochs_requested_if_trained": EPOCHS,
              "tau_init": args.tau_init, "eps_init": args.eps_init,
              "subset_steps": len(fixed_batches),
              "initialization_seconds": initialization_seconds,
              "steps": [], "status": "INCOMPLETE"}

    # ---- two genuine optimizer updates, timed and checked
    compile_started = time.perf_counter()
    for step in range(2):
        xb, yb = fixed_batches[step % len(fixed_batches)]
        state, loss, accuracy, grad_norm, gradients_finite = \
            RUNNER.train_one_batch_observable(
                state, jax.random.PRNGKey(args.seed * 1000 + step), xb, yb,
                model, step, steps_per_epoch, EPOCHS)
        record["steps"].append(
            {"step": step, "loss": float(loss), "accuracy": float(accuracy),
             "gradient_norm": float(grad_norm),
             "gradients_finite": bool(gradients_finite),
             "state_finite": _finite(state.params),
             "seconds": time.perf_counter() - compile_started})
        if not bool(gradients_finite) or not _finite(state.params):
            record["status"] = "NONFINITE_UPDATE"
            break

    # ---- the fixed subset, repeated, so learning is demonstrable
    if record["status"] == "INCOMPLETE":
        first_pass, last_pass = None, None
        for sweep in range(3):
            losses = []
            for index, (xb, yb) in enumerate(fixed_batches):
                step = 2 + sweep * len(fixed_batches) + index
                state, loss, accuracy, grad_norm, gradients_finite = \
                    RUNNER.train_one_batch_observable(
                        state, jax.random.PRNGKey(args.seed * 1000 + step),
                        xb, yb, model, step, steps_per_epoch, EPOCHS)
                losses.append(float(loss))
                if not bool(gradients_finite) or not _finite(state.params):
                    record["status"] = "NONFINITE_SUBSET_STEP"
                    break
            if record["status"] != "INCOMPLETE":
                break
            mean_loss = sum(losses) / len(losses)
            record.setdefault("subset_mean_loss", []).append(mean_loss)
            first_pass = mean_loss if first_pass is None else first_pass
            last_pass = mean_loss
        record["loss_reduction"] = (
            None if first_pass is None or last_pass is None
            else (first_pass - last_pass) / abs(first_pass))

    # ---- checkpoint save and reload
    checkpoint = os.path.join(args.out, "dev_gate_checkpoint.msgpack")
    with open(checkpoint, "wb") as handle:
        handle.write(serialization.to_bytes(state.params))
    with open(checkpoint, "rb") as handle:
        restored = serialization.from_bytes(state.params, handle.read())
    record["checkpoint"] = checkpoint
    record["checkpoint_restored"] = _finite(restored)

    # ---- validation, on the validation split only
    xb = jnp.asarray(val[0][:RUNNER.BATCH_SIZE])
    yb = jnp.asarray(val[1][:RUNNER.BATCH_SIZE])
    losses, accuracies, _ = RUNNER.eval_step(
        xb, yb, jnp.ones((xb.shape[0], RUNNER.SEQ_LEN)), state, eval_model,
        True)
    record["validation"] = {"cross_entropy": float(jnp.mean(losses)),
                            "accuracy": float(jnp.mean(accuracies)),
                            "n": int(xb.shape[0])}

    diagnostics = wwj_diagnostics(state, args.arm)
    record["diagnostics"] = diagnostics
    record["peak_gpu_bytes"] = (jax.devices()[0].memory_stats() or {}).get(
        "peak_bytes_in_use")

    problems = []
    if record["status"] != "INCOMPLETE":
        problems.append(record["status"])
    if not record["checkpoint_restored"]:
        problems.append("checkpoint did not restore to a finite state")
    if record.get("loss_reduction") is None:
        problems.append("no loss trajectory was produced")
    elif record["loss_reduction"] < REQUIRED_LOSS_REDUCTION:
        problems.append(
            f"subset loss fell by {record['loss_reduction']:.3%}, below the "
            f"required {REQUIRED_LOSS_REDUCTION:.1%}")
    radius = diagnostics["max_companion_spectral_radius"]
    if not (radius == radius) or radius > RADIUS_CEILING:
        problems.append(f"max companion spectral radius {radius} exceeds "
                        f"{RADIUS_CEILING}")
    record["status"] = "DEV_GATE_PASS" if not problems else "DEV_GATE_FAIL"
    record["problems"] = problems
    record["developmental_only"] = True

    with open(os.path.join(args.out, "dev_gate.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({"status": record["status"], "problems": problems,
                      "loss_reduction": record.get("loss_reduction"),
                      "validation": record["validation"],
                      "max_companion_spectral_radius": radius}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
