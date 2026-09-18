"""Open the test split once and finalize nine completed array tasks."""

import argparse
import json
import os

import jax
import jax.numpy as jnp
from flax import serialization

from experiments.s5_three_arm_full import data as EXPERIMENT_DATA
from experiments.s5_three_arm_full.runner import (ARM_ORDER, BATCH_SIZE,
                                                   EPOCHS, SEQ_LEN, SEEDS,
                                                   arm_summary, evaluate,
                                                   init_state, paired_summary)


def main(args):
    rows = []
    for arm in ARM_ORDER:
        for seed in SEEDS:
            path = os.path.join(args.task_root, arm, str(seed),
                                "task_result.json")
            if not os.path.exists(path):
                raise SystemExit(f"missing completed task: {path}")
            with open(path) as handle:
                row = json.load(handle)
            if row["code_identifier"] != arm or row["seed"] != seed:
                raise SystemExit(f"task identity mismatch: {path}")
            if row["selected_epoch"] >= EPOCHS - 2:
                row["status"] = "POSSIBLY_UNDERTRAINED"
                row["status_reason"] = (
                    "best validation checkpoint is in the final three epochs")
            else:
                row["status"] = "OK"
            rows.append(row)

    test = EXPERIMENT_DATA.load_official_raw(
        args.data_cache, ("test",))[0]["test"]
    for row in rows:
        row["test"] = restore_and_evaluate(row, test)

    result = {
        "schema": "s5-three-arm-full-training/v2",
        "scientific_arms": [
            "Native S5 recurrence under the shared stability constraint",
            "Zucchet prospective S5 recurrence",
            "Generalized prospective S5 recurrence (M,γ,T)"],
        "code_identifiers": list(ARM_ORDER), "seeds": list(SEEDS),
        "schedule": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                      "lr": 0.008, "ssm_lr": 0.002, "weight_decay": 0.04,
                      "lr_schedule": "one_epoch_linear_warmup_then_cosine",
                      "checkpoint_selection": "validation_accuracy_then_cross_entropy"},
        "rows": rows, "arm_summary": arm_summary(rows),
        "paired_differences": paired_summary(rows),
        "test_opened_after_selection": True,
        "test_opened_once_by_finalizer": True,
        "data_manifest": os.path.join(args.data_cache, "manifest.json"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(args.out, "artifact_manifest.json"), "w") as handle:
        json.dump({"schema": "s5-three-arm-full-training/artifacts-v2",
                   "results": "results.json", "test_opened_once": True,
                   "task_root": os.path.abspath(args.task_root)},
                  handle, indent=2)


def restore_and_evaluate(row, test):
    from experiments.s5_three_arm_full.runner import model_for
    model = model_for(row["code_identifier"], False)
    template = init_state(row["code_identifier"], row["seed"])
    with open(row["selected_checkpoint"], "rb") as handle:
        restored = serialization.from_bytes(template, handle.read())
    return evaluate(restored, model, *test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
