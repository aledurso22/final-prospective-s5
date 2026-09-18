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
import optax
from flax.training import train_state
from flax.traverse_util import flatten_dict

from dataloaders import speech_commands10 as SC
from s5.gp_ssm import init_gp_ssm
from s5.gp_second_order import init_second_order_ssm
from s5.seq_model import BatchClassificationModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO


SCIENTIFIC_NAMES = {
    "native_matched_s5": "Native matched S5",
    "zucchet_prospective_s5": "Zucchet prospective S5 recurrence",
    "generalized_prospective_s5": "Generalized prospective S5 recurrence (M,γ,T)",
}
ARM_ORDER = tuple(SCIENTIFIC_NAMES)
SEEDS = (301, 302, 303)
D_MODEL, SSM_SIZE, N_LAYERS = 32, 32, 4
BATCH_SIZE, EPOCHS = 32, 40
LR, LR_FINAL, WEIGHT_DECAY, CLIP = 1e-3, 1e-6, 1e-4, 1.0


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ssm_kwargs():
    block = SSM_SIZE // 1
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block)
    block //= 2
    P = SSM_SIZE // 2
    Lambda, V = Lambda[:block], V[:, :block]
    Vc = V.conj().T
    Lambda = Lambda[:block]
    return dict(
        H=D_MODEL, P=P, Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag, V=V, Vinv=Vc,
        C_init="trunc_standard_normal", discretization="zoh",
        dt_min=0.001, dt_max=0.1, conj_sym=True,
        clip_eigs=True, bidirectional=False,
    )


def ssm_factory(arm):
    kw = ssm_kwargs()
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
        dropout=0.0, training=training, mode="pool", prenorm=True,
        batchnorm=False, bn_momentum=0.95,
    )


def init_state(arm, seed):
    model = model_for(arm, True)
    key = jax.random.PRNGKey(seed)
    x = jnp.zeros((BATCH_SIZE, SC.N_FRAMES, SC.N_MFCC), dtype=jnp.float32)
    steps = jnp.ones((BATCH_SIZE, SC.N_FRAMES), dtype=jnp.float32)
    variables = model.init({"params": key}, x, steps)
    schedule = optax.cosine_decay_schedule(LR, EPOCHS * 1, LR_FINAL)
    tx = optax.chain(optax.clip_by_global_norm(CLIP),
                     optax.adamw(schedule, weight_decay=WEIGHT_DECAY))
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=variables["params"], tx=tx)


def shared_parameter_digest(params):
    flat = flatten_dict(params)
    return {"/".join(k): hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
            for k, v in flat.items()}


def parameter_count(params):
    return int(sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(params)))


def recurrent_state_size(arm):
    per_layer = SSM_SIZE if arm != "generalized_prospective_s5" else 2 * SSM_SIZE
    return {"complex_modes": SSM_SIZE // 2,
            "real_values_total": N_LAYERS * per_layer,
            "per_layer_real_values": per_layer}


def batch_arrays(x, y, indices):
    return jnp.asarray(x[indices]), jnp.asarray(y[indices])


def loss_and_metrics(state, model, x, y, train, rng=None):
    variables = {"params": state.params}
    logits = model.apply(variables, x, jnp.ones((x.shape[0], SC.N_FRAMES)),
                         rngs={"dropout": rng} if rng is not None else None)
    loss = -jnp.mean(logits[jnp.arange(y.shape[0]), y])
    return loss, (logits, jnp.mean(jnp.argmax(logits, axis=-1) == y))


def make_train_step(model):
    @jax.jit
    def step(state, x, y):
        def objective(params):
            logits = model.apply({"params": params}, x,
                                 jnp.ones((x.shape[0], SC.N_FRAMES)))
            return -jnp.mean(logits[jnp.arange(y.shape[0]), y])
        loss, grads = jax.value_and_grad(objective)(state.params)
        return state.apply_gradients(grads=grads), loss
    return step


def evaluate(state, model, x, y):
    losses, correct, count = [], 0, 0
    for start in range(0, len(y), BATCH_SIZE):
        xb = jnp.asarray(x[start:start + BATCH_SIZE])
        yb = jnp.asarray(y[start:start + BATCH_SIZE])
        logits = model.apply({"params": state.params}, xb,
                             jnp.ones((xb.shape[0], SC.N_FRAMES)))
        losses.append(float(-jnp.sum(logits[jnp.arange(yb.shape[0]), yb])))
        correct += int(jnp.sum(jnp.argmax(logits, axis=-1) == yb))
        count += len(yb)
    return {"cross_entropy": float(sum(losses) / count),
            "accuracy": float(correct / count), "n": count}


def train_arm(arm, seed, data):
    state = init_state(arm, seed)
    train_model, eval_model = model_for(arm, True), model_for(arm, False)
    step = make_train_step(train_model)
    best = None
    start_time = time.perf_counter()
    for epoch in range(EPOCHS):
        for indices in SC.epoch_batches(len(data["train"][1]), BATCH_SIZE,
                                        seed, epoch, drop_last=True):
            xb, yb = batch_arrays(*data["train"], indices)
            state, _ = step(state, xb, yb)
        val = evaluate(state, eval_model, *data["val"])
        candidate = (val["accuracy"], -val["cross_entropy"], -epoch)
        if best is None or candidate > best["selection"]:
            best = {"selection": candidate, "epoch": epoch + 1,
                    "state": state, "val": val}
    elapsed = time.perf_counter() - start_time
    return {"scientific_name": SCIENTIFIC_NAMES[arm],
            "code_identifier": arm, "seed": seed,
            "selected_epoch": best["epoch"], "validation": best["val"],
            "_selected_state": best["state"],
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


def run(args):
    cache = SC.load_splits(args.data_cache, ("train", "val"))[0]
    data = {"train": cache["train"], "val": cache["val"]}
    rows = []
    for seed in SEEDS:
        for arm in ARM_ORDER:
            print(f"TRAIN seed={seed} arm={SCIENTIFIC_NAMES[arm]}", flush=True)
            rows.append(train_arm(arm, seed, data))
    test = SC.load_splits(args.data_cache, ("test",))[0]["test"]
    for row in rows:
        state = row.pop("_selected_state")
        eval_model = model_for(row["code_identifier"], False)
        row["test"] = evaluate(state, eval_model, *test)
    result = {"schema": "s5-three-arm-full-training/v1",
              "scientific_arms": [SCIENTIFIC_NAMES[a] for a in ARM_ORDER],
              "code_identifiers": list(ARM_ORDER), "seeds": list(SEEDS),
              "schedule": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                           "lr": LR, "lr_final": LR_FINAL,
                           "weight_decay": WEIGHT_DECAY, "grad_clip": CLIP},
              "rows": rows, "arm_summary": arm_summary(rows),
              "paired_differences": paired_summary(rows),
              "test_opened_after_selection": True,
              "data_cache": os.path.abspath(args.data_cache),
              "data_manifest_sha256": sha256_file(os.path.join(args.data_cache,
                                                                 "manifest.json"))}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(args.out, "artifact_manifest.json"), "w") as handle:
        json.dump({"schema": "s5-three-arm-full-training/artifacts-v1",
                   "results": "results.json", "protocol": args.protocol,
                   "git_commit": args.commit, "arms": list(SCIENTIFIC_NAMES.items())},
                  handle, indent=2)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--commit", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
