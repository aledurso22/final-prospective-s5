"""Production-shaped step timing: the real model, Native versus modal.

The layer benchmark measured ONE layer at batch 1 and gave 0.593x Native.
Real training is batch 16, six layers, bidirectional, plus the optimizer and
the readout, so that ratio is an extrapolation. This measures the thing
itself: full `BatchClassificationModel` at the production configuration, a
handful of genuine optimizer updates, compilation timed separately, and a
15-epoch projection against Native's OWN recorded wall time.

It trains nothing that is kept, touches no dataset, and submits nothing.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
from functools import partial

from dataloaders import speech_commands10 as SC
from experiments.s5_three_arm_full import runner as RUNNER
from s5.modal_prospective_ssm import init_modal_prospective_S5SSM
from s5.seq_model import BatchClassificationModel
from s5.train_helpers import create_train_state

#: the production protocol this projects
EPOCHS = 15
TRAIN_EXAMPLES = 30769


def model_cls(arm):
    """The real classification model, with either SSM constructor."""
    kwargs = RUNNER.ssm_kwargs("native_matched_s5")
    ssm = (RUNNER.ssm_factory("native_matched_s5") if arm == "native"
           else init_modal_prospective_S5SSM(**kwargs))
    return partial(BatchClassificationModel, ssm=ssm,
                   d_output=len(SC.WORDS), d_model=RUNNER.D_MODEL,
                   n_layers=RUNNER.N_LAYERS, padded=False,
                   activation="half_glu1", dropout=RUNNER.SHARED_CONFIG.dropout,
                   mode="pool", prenorm=True, batchnorm=True, bn_momentum=0.95)


def measure(arm, seed, steps):
    state = create_train_state(
        model_cls(arm), jax.random.PRNGKey(seed), padded=False,
        retrieval=False, in_dim=RUNNER.INPUT_DIM, bsz=RUNNER.BATCH_SIZE,
        seq_len=RUNNER.SEQ_LEN, weight_decay=RUNNER.WEIGHT_DECAY,
        batchnorm=True, opt_config="noBCdecay", ssm_lr=RUNNER.SSM_LR,
        lr=RUNNER.LR, dt_global=False)
    model = model_cls(arm)(training=True)
    x = jax.random.normal(jax.random.PRNGKey(seed),
                          (RUNNER.BATCH_SIZE, RUNNER.SEQ_LEN,
                           RUNNER.INPUT_DIM))
    y = jax.random.randint(jax.random.PRNGKey(seed + 1),
                           (RUNNER.BATCH_SIZE,), 0, len(SC.WORDS))

    def update(state, step):
        return RUNNER.train_one_batch_observable(
            state, jax.random.PRNGKey(seed * 1000 + step), x, y, model, step,
            1, EPOCHS)

    started = time.perf_counter()
    state, loss, _, grad_norm, finite = update(state, 0)
    jax.block_until_ready(state.params)
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for step in range(1, steps + 1):
        state, loss, _, grad_norm, finite = update(state, step)
    jax.block_until_ready(state.params)
    seconds = (time.perf_counter() - started) / steps

    peak = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
    return {"arm": arm, "compile_seconds": compile_seconds,
            "seconds_per_step": seconds, "steps_per_minute": 60.0 / seconds,
            "peak_gpu_bytes": peak, "loss": float(loss),
            "gradient_norm": float(grad_norm),
            "gradients_finite": bool(finite)}


def project(steps_per_minute):
    steps_per_epoch = TRAIN_EXAMPLES // RUNNER.BATCH_SIZE
    total = steps_per_epoch * EPOCHS
    return {"steps_per_epoch": steps_per_epoch, "total_steps": total,
            "hours_one_seed": total / steps_per_minute / 60.0,
            "note": "three seeds run concurrently on three GPUs, so a wave "
                    "costs one seed's time"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--native-hours", type=float, default=5.51,
                        help="Native's OWN recorded 15-epoch wall time")
    args = parser.parse_args()
    rows = [measure(arm, args.seed, args.steps) for arm in ("native", "modal")]
    native, modal = rows
    ratio = native["seconds_per_step"] / modal["seconds_per_step"]
    report = {
        "schema": "s5-modal/production-step-v1",
        "configuration": {"batch": RUNNER.BATCH_SIZE,
                          "seq_len": RUNNER.SEQ_LEN,
                          "d_model": RUNNER.D_MODEL,
                          "n_layers": RUNNER.N_LAYERS,
                          "bidirectional": RUNNER.SHARED_CONFIG.bidirectional,
                          "epochs_projected": EPOCHS},
        "rows": rows,
        "modal_throughput_ratio_to_native": ratio,
        "projection": {"native": project(native["steps_per_minute"]),
                       "modal": project(modal["steps_per_minute"])},
        "native_recorded_hours": args.native_hours,
        "modal_hours_from_measured_ratio":
            args.native_hours / ratio if ratio else None,
        "layer_level_ratio_for_comparison": 0.593,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
