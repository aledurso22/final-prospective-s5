"""Full-training, exactly-three-arm Speech Commands S5 comparison."""

import argparse
import hashlib
import json
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict
from jax.scipy.linalg import block_diag

from dataloaders import speech_commands10 as SC
from experiments.s5_three_arm_full import data as EXPERIMENT_DATA
from s5.gp_ssm import init_gp_ssm
from s5.gp_second_order import init_second_order_ssm
from s5.seq_model import BatchClassificationModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.train_helpers import (constant_lr, cosine_annealing, create_train_state,
                              linear_warmup, train_step,
                              update_learning_rate_per_step, eval_step)


SCIENTIFIC_NAMES = {
    "native_matched_s5": "Native matched S5",
    "zucchet_prospective_s5": "Zucchet prospective S5 recurrence",
    "generalized_prospective_s5": "Generalized prospective S5 recurrence (M,γ,T)",
}
ARM_ORDER = tuple(SCIENTIFIC_NAMES)
SEEDS = (301, 302, 303)
D_MODEL, SSM_SIZE_BASE, BLOCKS, N_LAYERS = 96, 128, 16, 6
SEQ_LEN, INPUT_DIM = 16000, 1
BATCH_SIZE, EPOCHS = 16, 40
LR, SSM_LR, LR_FINAL, WEIGHT_DECAY = 0.008, 0.002, 1e-6, 0.04
GRAD_CLIP, WARMUP_END = 1.0, 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ssm_kwargs(arm):
    block = SSM_SIZE_BASE // BLOCKS
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block)
    block //= 2
    P = SSM_SIZE_BASE // 2
    Lambda, V = Lambda[:block], V[:, :block]
    Vc = V.conj().T
    Lambda = (Lambda * np.ones((BLOCKS, block))).ravel()
    V = block_diag(*([V] * BLOCKS))
    Vc = block_diag(*([Vc] * BLOCKS))
    return dict(
        H=D_MODEL, P=P, Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag, V=V, Vinv=Vc,
        C_init="trunc_standard_normal", discretization="zoh",
        dt_min=0.001, dt_max=0.1, conj_sym=True,
        clip_eigs=True, bidirectional=True,
    )


def ssm_factory(arm):
    kw = ssm_kwargs(arm)
    if arm == "native_matched_s5":
        return init_S5SSM(**kw)
    if arm == "zucchet_prospective_s5":
        return init_gp_ssm(mechanism="gp_diagonal", gp_init_scale=0.05, **kw)
    if arm == "generalized_prospective_s5":
        return init_second_order_ssm(gp_init_scale=0.3,
                                     mu_ratio_init=0.5, **kw)
    raise ValueError(arm)


def model_for(arm, training):
    return BatchClassificationModel(
        ssm=ssm_factory(arm), d_output=len(SC.WORDS), d_model=D_MODEL,
        n_layers=N_LAYERS, padded=False, activation="half_glu1",
        dropout=0.1, training=training, mode="pool", prenorm=True,
        batchnorm=True, bn_momentum=0.95,
    )


def model_cls_for(arm):
    return partial(
        BatchClassificationModel, ssm=ssm_factory(arm),
        d_output=len(SC.WORDS), d_model=D_MODEL, n_layers=N_LAYERS,
        padded=False, activation="half_glu1", dropout=0.1, mode="pool",
        prenorm=True, batchnorm=True, bn_momentum=0.95)


def init_state(arm, seed):
    return create_train_state(
        model_cls_for(arm), jax.random.PRNGKey(seed), padded=False,
        retrieval=False, in_dim=INPUT_DIM, bsz=BATCH_SIZE, seq_len=SEQ_LEN,
        weight_decay=WEIGHT_DECAY, batchnorm=True, opt_config="noBCdecay",
        ssm_lr=SSM_LR, lr=LR, dt_global=False)


def shared_parameter_digest(params):
    flat = flatten_dict(params)
    return {"/".join(k): hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
            for k, v in flat.items()}


def parameter_count(params):
    return int(sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(params)))


def recurrent_state_size(arm):
    per_layer = SSM_SIZE_BASE if arm != "generalized_prospective_s5" else 2 * SSM_SIZE_BASE
    return {"complex_modes": SSM_SIZE_BASE // 2,
            "real_values_total": N_LAYERS * per_layer,
            "per_layer_real_values": per_layer}


def batch_arrays(x, y, indices):
    return jnp.asarray(x[indices]), jnp.asarray(y[indices])


def evaluate(state, model, x, y):
    losses, correct, count = [], 0, 0
    for start in range(0, len(y), BATCH_SIZE):
        xb = jnp.asarray(x[start:start + BATCH_SIZE])
        yb = jnp.asarray(y[start:start + BATCH_SIZE])
        batch_loss, batch_acc, logits = eval_step(
            xb, yb, jnp.ones((xb.shape[0], SEQ_LEN)), state, model, True)
        losses.append(float(jnp.sum(batch_loss)))
        correct += int(jnp.sum(batch_acc))
        count += len(yb)
    return {"cross_entropy": float(sum(losses) / count),
            "accuracy": float(correct / count), "n": count}


def train_arm(arm, seed, data, checkpoint_dir):
    state = init_state(arm, seed)
    train_model, eval_model = model_for(arm, True), model_for(arm, False)
    step = 0
    best = None
    start_time = time.perf_counter()
    for epoch in range(EPOCHS):
        decay = linear_warmup if epoch < WARMUP_END else cosine_annealing
        end_step = (len(data["train"][1]) // BATCH_SIZE) * EPOCHS
        for indices in SC.epoch_batches(len(data["train"][1]), BATCH_SIZE,
                                        seed, epoch, drop_last=True):
            xb, yb = batch_arrays(*data["train"], indices)
            rng = jax.random.PRNGKey(seed * 1000 + step)
            state, _ = train_step(
                state, rng, xb, yb, jnp.ones((xb.shape[0], SEQ_LEN)),
                train_model, True)
            state, step = update_learning_rate_per_step(
                (decay, SSM_LR, LR, step, end_step, "noBCdecay", LR_FINAL),
                state)
        val = evaluate(state, eval_model, *data["val"])
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch + 1:02d}.msgpack")
        with open(checkpoint_path, "wb") as handle:
            handle.write(serialization.to_bytes({
                "params": state.params, "batch_stats": state.batch_stats}))
        with open(os.path.join(checkpoint_dir, "metrics.jsonl"), "a") as handle:
            handle.write(json.dumps({"epoch": epoch + 1,
                                     "validation": val}) + "\n")
        candidate = (val["accuracy"], -val["cross_entropy"], -epoch)
        if best is None or candidate > best["selection"]:
            best = {"selection": candidate, "epoch": epoch + 1,
                    "state": state, "val": val,
                    "checkpoint": checkpoint_path}
    elapsed = time.perf_counter() - start_time
    return {"scientific_name": SCIENTIFIC_NAMES[arm],
            "code_identifier": arm, "seed": seed,
            "selected_epoch": best["epoch"], "validation": best["val"],
            "selected_checkpoint": best["checkpoint"],
            "parameter_count": parameter_count(state.params),
            "recurrent_state": recurrent_state_size(arm),
            "training_seconds": elapsed,
            "examples_per_second": EPOCHS * len(data["train"][1]) / elapsed}


def paired_summary(rows):
    def summarize(values):
        values = np.asarray(values, dtype=np.float64)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(values.size))
        half = 4.302652729 * se
        return {"mean": mean, "standard_error": se,
                "ci95": [mean - half, mean + half]}

    out = {}
    for left, right in (("generalized_prospective_s5", "zucchet_prospective_s5"),
                        ("generalized_prospective_s5", "native_matched_s5")):
        pairs = []
        for seed in SEEDS:
            l = next(r for r in rows if r["seed"] == seed and r["code_identifier"] == left)
            r = next(r for r in rows if r["seed"] == seed and r["code_identifier"] == right)
            pairs.append({"seed": seed,
                          "test_accuracy": l["test"]["accuracy"] - r["test"]["accuracy"],
                          "test_cross_entropy": l["test"]["cross_entropy"] - r["test"]["cross_entropy"]})
        out[f"{left}_minus_{right}"] = {
            "per_seed": pairs,
            "test_accuracy": summarize([p["test_accuracy"] for p in pairs]),
            "test_cross_entropy": summarize(
                [p["test_cross_entropy"] for p in pairs]),
        }
    return out


def arm_summary(rows):
    summary = {}
    for arm in ARM_ORDER:
        selected = [row for row in rows if row["code_identifier"] == arm]
        summary[arm] = {
            "scientific_name": SCIENTIFIC_NAMES[arm],
            "mean_test_accuracy": float(np.mean(
                [row["test"]["accuracy"] for row in selected])),
            "mean_test_cross_entropy": float(np.mean(
                [row["test"]["cross_entropy"] for row in selected])),
            "mean_selected_epoch": float(np.mean(
                [row["selected_epoch"] for row in selected])),
            "mean_training_seconds": float(np.mean(
                [row["training_seconds"] for row in selected])),
            "mean_examples_per_second": float(np.mean(
                [row["examples_per_second"] for row in selected])),
        }
    return summary


def run_task(args):
    if args.arm not in ARM_ORDER or args.seed not in SEEDS:
        raise ValueError(f"invalid task identity: {args.arm}, {args.seed}")
    cache = EXPERIMENT_DATA.load_official_raw(args.data_cache, ("train", "val"))[0]
    data = {"train": cache["train"], "val": cache["val"]}
    row = train_arm(args.arm, args.seed, data, args.out)
    with open(os.path.join(args.out, "task_result.json"), "w") as handle:
        json.dump(row, handle, indent=2)
    print(json.dumps(row, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=ARM_ORDER, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run_task(parser.parse_args())


if __name__ == "__main__":
    main()
