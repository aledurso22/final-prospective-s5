"""Full-training, exactly-three-arm Speech Commands S5 comparison."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict
from jax.scipy.linalg import block_diag

from dataloaders import speech_commands10 as SC
from experiments.s5_three_arm_full import data as EXPERIMENT_DATA
from experiments.s5_three_arm_full import gpu_telemetry as GPU_TELEMETRY
from s5.seq_model import BatchClassificationModel
from s5.ssm_init import make_DPLR_HiPPO
from s5.matched_lag_ssm import init_matched_lag_prospective_S5SSM
from s5.three_arm_factory import (init_S5SSM,
                                  init_generalized_prospective_S5SSM,
                                  init_prospective_S5SSM)
from s5.discrete_recurrence import companion_radius, generalized_coefficients, zucchet_coefficients
from s5.factored_recurrence import DEFAULT_IMPLEMENTATION, SCAN_IMPLEMENTATIONS
from s5.ssm import discretize_zoh
from s5.train_helpers import (cosine_annealing, create_train_state,
                              linear_warmup, train_step, train_step_observable,
                              train_step_telemetry,
                              update_learning_rate_per_step, eval_step)


SCIENTIFIC_NAMES = {
    "native_matched_s5": "Native S5",
    "zucchet_prospective_s5": "Zucchet prospective dynamics — finite-difference realization",
    "generalized_prospective_s5": "generalized prospective dynamics (M,γ,T) — finite-difference realization",
    "matched_lag_prospective_s5": "theory-matched zero-first-order-lag prospective dynamics (M,γ,Γ_k) — finite-difference realization",
}
#: PINNED to the original three. The fourth arm is a theory test, not part
#: of the three-arm comparison, and `finalize.py` and `arm_summary` iterate
#: ARM_ORDER -- extending it would make the finalizer demand artifacts for
#: an arm the three-arm protocol never runs.
ARM_ORDER = ("native_matched_s5", "zucchet_prospective_s5",
             "generalized_prospective_s5")
#: everything `--arm` accepts
RUNNABLE_ARMS = tuple(SCIENTIFIC_NAMES)
SEEDS = (301, 302, 303)
#: WHICH SCAN evaluates the generalized arm's recurrence. The equation, the
#: coefficients and the parameterization are identical for every choice --
#: verified to 1.83e-13 in float64 over the whole production inventory -- so
#: this changes evaluation order and nothing else. It defaults to the
#: original sequential scan, so an existing command is unchanged.
GENERALIZED_IMPLEMENTATION = DEFAULT_IMPLEMENTATION
T_INIT, RHO_INIT = 0.05, 0.5


class NumericalTrainingFailure(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(json.dumps(record))


@dataclass(frozen=True)
class SharedS5Config:
    d_model: int = 96
    ssm_size: int = 128
    blocks: int = 16
    n_layers: int = 6
    input_dim: int = 1
    seq_len: int = 16000
    batch_size: int = 16
    epochs: int = 40
    lr: float = 0.008
    ssm_lr: float = 0.002
    lr_final: float = 1e-6
    weight_decay: float = 0.04
    warmup_end: int = 1
    dropout: float = 0.1
    c_init: str = "lecun_normal"
    discretization: str = "zoh"
    conj_sym: bool = True
    clip_eigs: bool = True
    bidirectional: bool = True


SHARED_CONFIG = SharedS5Config()
D_MODEL, SSM_SIZE_BASE, BLOCKS, N_LAYERS = (SHARED_CONFIG.d_model,
                                               SHARED_CONFIG.ssm_size,
                                               SHARED_CONFIG.blocks,
                                               SHARED_CONFIG.n_layers)
SEQ_LEN, INPUT_DIM = SHARED_CONFIG.seq_len, SHARED_CONFIG.input_dim
BATCH_SIZE, EPOCHS = SHARED_CONFIG.batch_size, SHARED_CONFIG.epochs
LR, SSM_LR, LR_FINAL, WEIGHT_DECAY = (SHARED_CONFIG.lr, SHARED_CONFIG.ssm_lr,
                                       SHARED_CONFIG.lr_final,
                                       SHARED_CONFIG.weight_decay)
WARMUP_END = SHARED_CONFIG.warmup_end

ARM_CONFIGS = {
    "native_matched_s5": {"shared": SHARED_CONFIG,
                           "recurrence_constructor": "init_S5SSM",
                           "extra_parameters": (), "state_dimension": 128},
    "zucchet_prospective_s5": {"shared": SHARED_CONFIG,
                                "recurrence_constructor": "init_prospective_S5SSM",
                                "extra_parameters": ("T",),
                                "state_dimension": 128},
    "generalized_prospective_s5": {"shared": SHARED_CONFIG,
                                    "recurrence_constructor": "init_generalized_prospective_S5SSM",
                                    "extra_parameters": ("T", "rho", "gamma"),
                                    "state_dimension": 256},
    "matched_lag_prospective_s5": {"shared": SHARED_CONFIG,
                                    "recurrence_constructor": "init_matched_lag_prospective_S5SSM",
                                    "extra_parameters": ("T", "rho", "gamma"),
                                    "state_dimension": 256},
}


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
        C_init=SHARED_CONFIG.c_init,
        discretization=SHARED_CONFIG.discretization,
        dt_min=0.001, dt_max=0.1, conj_sym=True,
        clip_eigs=SHARED_CONFIG.clip_eigs,
        bidirectional=SHARED_CONFIG.bidirectional,
    )


def ssm_factory(arm):
    kw = ssm_kwargs(arm)
    if arm == "native_matched_s5":
        return init_S5SSM(**kw)
    if arm == "zucchet_prospective_s5":
        return init_prospective_S5SSM(response_init=T_INIT, **kw)
    if arm == "generalized_prospective_s5":
        return init_generalized_prospective_S5SSM(
            response_init=T_INIT, rho_init=RHO_INIT, gamma_init=1.0,
            implementation=GENERALIZED_IMPLEMENTATION, **kw)
    if arm == "matched_lag_prospective_s5":
        # IDENTICAL initialization to the generalized arm, deliberately:
        # the comparison must differ only in the prospective drive.
        return init_matched_lag_prospective_S5SSM(
            response_init=T_INIT, rho_init=RHO_INIT, gamma_init=1.0,
            implementation=GENERALIZED_IMPLEMENTATION, **kw)
    raise ValueError(arm)


def model_for(arm, training):
    return BatchClassificationModel(
        ssm=ssm_factory(arm), d_output=len(SC.WORDS), d_model=D_MODEL,
        n_layers=N_LAYERS, padded=False, activation="half_glu1",
        dropout=SHARED_CONFIG.dropout, training=training, mode="pool", prenorm=True,
        batchnorm=True, bn_momentum=0.95,
    )


def model_cls_for(arm):
    return partial(
        BatchClassificationModel, ssm=ssm_factory(arm),
        d_output=len(SC.WORDS), d_model=D_MODEL, n_layers=N_LAYERS,
        padded=False, activation="half_glu1", dropout=SHARED_CONFIG.dropout,
        mode="pool",
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
    two_compartment = ("generalized_prospective_s5",
                       "matched_lag_prospective_s5")
    per_layer = 2 * SSM_SIZE_BASE if arm in two_compartment else SSM_SIZE_BASE
    return {"complex_modes": SSM_SIZE_BASE // 2,
            "real_values_total": N_LAYERS * per_layer,
            "per_layer_real_values": per_layer}


def batch_arrays(x, y, indices):
    return jnp.asarray(x[indices]), jnp.asarray(y[indices])


def learning_rate_at_step(step, steps_per_epoch):
    warmup_steps = steps_per_epoch * WARMUP_END
    cosine_steps = steps_per_epoch * (EPOCHS - WARMUP_END)
    if step < warmup_steps:
        schedule = (linear_warmup, step, warmup_steps)
    else:
        schedule = (cosine_annealing, step - warmup_steps, cosine_steps)
    function, local_step, end_step = schedule
    return (float(function(local_step, LR, end_step, LR_FINAL)),
            float(function(local_step, SSM_LR, end_step, LR_FINAL)))


def apply_scheduled_learning_rate(state, step, steps_per_epoch, epochs=EPOCHS):
    """Set both optimizer rates for the update at ``step``.

    ``epochs`` is the number of epochs this run will actually train for. The
    schedule is one warm-up epoch and cosine decay over the remaining
    ``epochs - WARMUP_END``, so a shorter run decays over its own horizon
    instead of the 40-epoch one. The default keeps the original protocol.
    """
    warmup_steps = steps_per_epoch * WARMUP_END
    if step < warmup_steps:
        decay, schedule_step, end_step = linear_warmup, step, warmup_steps
    else:
        decay, schedule_step, end_step = (
            cosine_annealing, step - warmup_steps,
            steps_per_epoch * (epochs - WARMUP_END))
    return update_learning_rate_per_step(
        (decay, SSM_LR, LR, schedule_step, end_step, "noBCdecay", LR_FINAL),
        state)[0]


def train_one_batch(state, rng, xb, yb, model, step, steps_per_epoch,
                    epochs=EPOCHS):
    state = apply_scheduled_learning_rate(state, step, steps_per_epoch,
                                          epochs)
    return train_step(state, rng, xb, yb,
                      jnp.ones((xb.shape[0], SEQ_LEN)), model, True)


def train_one_batch_telemetry(state, rng, xb, yb, model, step, steps_per_epoch,
                              epochs=EPOCHS):
    state = apply_scheduled_learning_rate(state, step, steps_per_epoch,
                                          epochs)
    return train_step_telemetry(
        state, rng, xb, yb, jnp.ones((xb.shape[0], SEQ_LEN)), model, True)


def train_one_batch_observable(state, rng, xb, yb, model, step, steps_per_epoch,
                               epochs=EPOCHS):
    state = apply_scheduled_learning_rate(state, step, steps_per_epoch,
                                          epochs)
    return train_step_observable(
        state, rng, xb, yb, jnp.ones((xb.shape[0], SEQ_LEN)), model, True)


def _all_finite(tree):
    return bool(all(bool(jnp.all(jnp.isfinite(value)))
                    for value in jax.tree_util.tree_leaves(tree)
                    if hasattr(value, "dtype")))


def _gpu_memory_stats():
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return {key: int(value) for key, value in stats.items()
            if key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit",
                       "peak_bytes_used")}


def companion_spectral_radius(state, arm):
    flat = flatten_dict(state.params)
    radii = []
    for key in flat:
        if key[-1] != "Lambda_re":
            continue
        prefix = key[:-1]
        lam_re, lam_im = flat[key], flat[prefix + ("Lambda_im",)]
        step = jnp.exp(flat[prefix + ("log_step",)][..., 0])
        lam = jnp.clip(lam_re, None, -1e-4) + 1j * lam_im
        b = jnp.zeros((lam.shape[0], 1), dtype=lam.dtype)
        abar, bbar = discretize_zoh(lam, b, step)
        if arm == "native_matched_s5":
            radii.append(jnp.max(jnp.abs(abar)))
            continue
        if arm == "zucchet_prospective_s5":
            raw = flat[prefix + ("prospective_T_raw",)]
            coeff = zucchet_coefficients(abar, bbar, jax.nn.softplus(raw))
        else:
            t = jax.nn.softplus(flat[prefix + ("generalized_T_raw",)])
            rho = 1e-4 + (1.0 - 1e-4) * jax.nn.sigmoid(
                flat[prefix + ("generalized_rho_raw",)])
            gamma = jax.nn.softplus(flat[prefix + ("generalized_gamma_raw",)])
            coeff = generalized_coefficients(abar, bbar, t, rho * gamma * t, gamma)
        radii.append(jnp.max(companion_radius(coeff[0], coeff[1])))
    return float(jnp.max(jnp.asarray(radii)))


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


#: file written BEFORE any JAX import by `gpu_telemetry`; it carries the job
#: identity and a NODE/NVML GPU INVENTORY, which is not device assignment
PRE_JAX_TELEMETRY = "gpu_telemetry.json"
#: file written AFTER CUDA initialization and the production update: the only
#: record that binds THIS process id to a physical GPU
PROCESS_GPU_BINDING = "gpu_process_binding.json"
_PRE_JAX_EMITTED = set()
#: out -> process-binding record, cached so artifacts share one resolution
_PROCESS_GPU = {}


def _execution_identity():
    """Job, array member and GPU allocation, from the environment."""
    return {key: os.environ.get(key) for key in GPU_TELEMETRY.SLURM_KEYS}


def record_lifecycle(out, event, **fields):
    record = {"event": event, "pid": os.getpid(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "slurm": _execution_identity(),
              "gpu_memory": _gpu_memory_stats(), **fields}
    # the PRE-JAX record is attached once per process. It identifies the job,
    # not the device: its GPU list is a node inventory, kept under its own key
    # so it is never read as this task's assignment.
    if out not in _PRE_JAX_EMITTED:
        pre_jax = GPU_TELEMETRY.load(os.path.join(out, PRE_JAX_TELEMETRY))
        if pre_jax is not None:
            record["pre_jax_telemetry"] = pre_jax
        _PRE_JAX_EMITTED.add(out)
    # the process-specific binding, once it exists: this, and only this, says
    # which physical GPU this runner ran on
    if out in _PROCESS_GPU:
        record["process_gpu"] = GPU_TELEMETRY.summary(_PROCESS_GPU[out])
    with open(os.path.join(out, "lifecycle.jsonl"), "a") as handle:
        handle.write(json.dumps(record) + "\n")


def bind_process_gpu(out):
    """Bind THIS runner's pid to a physical GPU and persist it.

    Called after CUDA has initialized and the production update has run, so
    the process is an active compute application, and BEFORE the smoke
    training steps, so the artifact survives a kill during them. The result
    is `resolved`, `multiple` or `unresolved`; nothing is inferred from the
    node inventory when it does not resolve.
    """
    binding = GPU_TELEMETRY.write_process_binding(
        os.path.join(out, PROCESS_GPU_BINDING))
    _PROCESS_GPU[out] = binding
    return binding


def train_arm(arm, seed, data, checkpoint_dir, state=None, train_model=None,
              epochs=EPOCHS):
    if state is None:
        state = init_state(arm, seed)
    if train_model is None:
        train_model = model_for(arm, True)
    eval_model = model_for(arm, False)
    step = 0
    best = None
    start_time = time.perf_counter()
    steps_per_epoch = len(data["train"][1]) // BATCH_SIZE
    for epoch in range(epochs):
        for indices in SC.epoch_batches(len(data["train"][1]), BATCH_SIZE,
                                        seed, epoch, drop_last=True):
            xb, yb = batch_arrays(*data["train"], indices)
            rng = jax.random.PRNGKey(seed * 1000 + step)
            if epoch == 0 and step == 0:
                record_lifecycle(checkpoint_dir, "before_first_actual_training_step")
            state, loss, accuracy, grad_norm, gradients_finite = train_one_batch_observable(
                state, rng, xb, yb, train_model, step, steps_per_epoch, epochs)
            if epoch == 0 and step == 0:
                record_lifecycle(checkpoint_dir, "after_first_actual_training_step")
            state_finite = _all_finite(state)
            nonfinite = not (bool(gradients_finite) and state_finite and
                             bool(jnp.isfinite(loss)) and bool(jnp.isfinite(accuracy)) and
                             bool(jnp.isfinite(grad_norm)))
            telemetry = {"arm": SCIENTIFIC_NAMES[arm], "code_identifier": arm,
                         "seed": seed, "epoch": epoch + 1, "step": step,
                         "loss": float(loss), "accuracy": float(accuracy),
                         "state_norm": float(jnp.sqrt(sum(
                             jnp.sum(jnp.abs(v) ** 2) for v in jax.tree_util.tree_leaves(state)
                             if hasattr(v, "dtype")))),
                         "gradient_norm": float(grad_norm),
                         "gradients_finite": bool(gradients_finite),
                         "state_finite": state_finite, "nonfinite": nonfinite,
                         "companion_spectral_radius": companion_spectral_radius(state, arm)}
            with open(os.path.join(checkpoint_dir, "step_metrics.jsonl"), "a") as handle:
                handle.write(json.dumps(telemetry) + "\n")
            if nonfinite:
                raise NumericalTrainingFailure(telemetry)
            step += 1
        val = evaluate(state, eval_model, *data["val"])
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch + 1:02d}.msgpack")
        with open(checkpoint_path, "wb") as handle:
            handle.write(serialization.to_bytes(state))
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
            "epochs_requested": epochs,
            "selected_epoch": best["epoch"], "validation": best["val"],
            "selected_checkpoint": best["checkpoint"],
            "parameter_count": parameter_count(state.params),
            "recurrent_state": recurrent_state_size(arm),
            "training_seconds": elapsed,
            "examples_per_second": epochs * len(data["train"][1]) / elapsed}


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
    if args.arm not in RUNNABLE_ARMS or args.seed not in SEEDS:
        raise ValueError(f"invalid task identity: {args.arm}, {args.seed}")
    # a caller that never heard of --epochs gets the original protocol
    epochs = getattr(args, "epochs", EPOCHS)
    os.makedirs(args.out, exist_ok=True)
    cache = EXPERIMENT_DATA.load_official_raw(args.data_cache, ("train", "val"))[0]
    data = {"train": cache["train"], "val": cache["val"]}
    record_lifecycle(args.out, "before_training_initialization")
    initial_state = init_state(args.arm, args.seed)
    train_model = model_for(args.arm, True)
    record_lifecycle(args.out, "after_training_initialization")
    first_indices = next(SC.epoch_batches(len(data["train"][1]), BATCH_SIZE,
                                           args.seed, 0, drop_last=True))
    first_batch = batch_arrays(*data["train"], first_indices)
    production_check(args.arm, args.seed, args.out, first_batch,
                     initial_state, train_model, epochs=epochs)
    binding = bind_process_gpu(args.out)
    record_lifecycle(args.out, "after_production_check_before_full_path",
                     process_gpu=GPU_TELEMETRY.summary(binding))
    if args.smoke:
        result = full_path_smoke(args.arm, args.seed, data, args.out,
                                 initial_state, train_model,
                                 epochs=epochs)
        print(json.dumps(result, indent=2))
        return
    row = train_arm(args.arm, args.seed, data, args.out,
                    state=initial_state, train_model=train_model,
                    epochs=epochs)
    with open(os.path.join(args.out, "task_result.json"), "w") as handle:
        json.dump(row, handle, indent=2)
    print(json.dumps(row, indent=2))


def production_check(arm, seed, out, batch, state, model, epochs=EPOCHS):
    """Fail closed on one exact-shape finite update, then allow training."""
    xb, yb = batch
    started = time.perf_counter()
    memory_before = _gpu_memory_stats()
    checked, loss, accuracy, grad_norm, gradients_finite = train_one_batch_observable(
        state, jax.random.PRNGKey(seed * 1000), xb, yb, model, 0, 1, epochs)
    result = {"arm": SCIENTIFIC_NAMES[arm], "code_identifier": arm, "seed": seed,
              "epochs_requested": epochs,
              # provenance: which scan evaluated the recurrence. Recorded in
              # every run's artifacts so a result can never be read without
              # knowing how it was computed.
              "scan_implementation": (GENERALIZED_IMPLEMENTATION
                                      if arm in ("generalized_prospective_s5",
                                                 "matched_lag_prospective_s5")
                                      else None),
              "slurm": _execution_identity(),
              "loss": float(loss), "accuracy": float(accuracy),
              "gradient_norm": float(grad_norm),
              "gradients_finite": bool(gradients_finite),
              "state_finite": _all_finite(checked),
              "gpu_memory_before_update": memory_before,
              "seconds": time.perf_counter() - started}
    with open(os.path.join(out, "production_check.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    record_lifecycle(out, "after_production_check_update",
                     gradients_finite=bool(gradients_finite),
                     state_finite=result["state_finite"])
    if not result["gradients_finite"] or not result["state_finite"]:
        raise NumericalTrainingFailure({
            "arm": SCIENTIFIC_NAMES[arm], "code_identifier": arm,
            "seed": seed, "epoch": 0, "step": 0,
            "failure": "production check nonfinite", **result})


def full_path_smoke(arm, seed, data, out, state, train_model,
                    epochs=EPOCHS):
    """Exercise production lifecycle through train, validation, save, reload."""
    steps_per_epoch = len(data["train"][1]) // BATCH_SIZE
    step = 0
    for indices in SC.epoch_batches(len(data["train"][1]), BATCH_SIZE,
                                    seed, 0, drop_last=True):
        xb, yb = batch_arrays(*data["train"], indices)
        state, loss, accuracy, grad_norm, gradients_finite = train_one_batch_observable(
            state, jax.random.PRNGKey(seed * 1000 + step), xb, yb,
            train_model, step, steps_per_epoch, epochs)
        if not gradients_finite or not _all_finite(state):
            raise NumericalTrainingFailure({
                "arm": SCIENTIFIC_NAMES[arm], "code_identifier": arm,
                "seed": seed, "epoch": 1, "step": step,
                "failure": "smoke training step nonfinite"})
        step += 1
        if step == 2:
            break
    record_lifecycle(out, "after_smoke_training_steps", steps=step)
    eval_model = model_for(arm, False)
    xb = jnp.asarray(data["val"][0][:BATCH_SIZE])
    yb = jnp.asarray(data["val"][1][:BATCH_SIZE])
    losses, accuracies, _ = eval_step(
        xb, yb, jnp.ones((xb.shape[0], SEQ_LEN)), state, eval_model, True)
    checkpoint = os.path.join(out, "smoke_checkpoint.msgpack")
    with open(checkpoint, "wb") as handle:
        handle.write(serialization.to_bytes(state))
    with open(checkpoint, "rb") as handle:
        restored = serialization.from_bytes(state, handle.read())
    pre_jax = GPU_TELEMETRY.load(os.path.join(out, PRE_JAX_TELEMETRY))
    binding = _PROCESS_GPU.get(out) or GPU_TELEMETRY.load(
        os.path.join(out, PROCESS_GPU_BINDING))
    result = {"status": "SMOKE_PASS", "scientific_name": SCIENTIFIC_NAMES[arm],
              "code_identifier": arm, "seed": seed, "training_steps": step,
              "epochs_requested": epochs,
              "slurm": _execution_identity(),
              # the runner-pid -> physical GPU binding; the inventory is kept
              # beside it, labelled, and is never used to claim identity
              "process_gpu": GPU_TELEMETRY.summary(binding),
              "gpu_inventory_uuids": GPU_TELEMETRY.inventory_uuids(pre_jax),
              "validation_accuracy": float(jnp.mean(accuracies)),
              "validation_cross_entropy": float(jnp.mean(losses)),
              "checkpoint": checkpoint, "checkpoint_restored": _all_finite(restored),
              "production_check": os.path.join(out, "production_check.json")}
    if not result["checkpoint_restored"]:
        raise RuntimeError("smoke checkpoint restoration produced nonfinite state")
    with open(os.path.join(out, "smoke_result.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=RUNNABLE_ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--smoke", action="store_true")
    # the number of epochs this run trains for. The default is the original
    # protocol, so an existing command is unchanged; the schedule (one warm-up
    # epoch then cosine decay) is recomputed over whatever is requested.
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    # the scan used by the generalized arm. Default unchanged, so every
    # existing command behaves exactly as before.
    parser.add_argument("--implementation", choices=sorted(SCAN_IMPLEMENTATIONS),
                        default=DEFAULT_IMPLEMENTATION)
    args = parser.parse_args()
    if args.epochs <= WARMUP_END:
        parser.error(f"--epochs must exceed the {WARMUP_END}-epoch warm-up")
    global GENERALIZED_IMPLEMENTATION
    GENERALIZED_IMPLEMENTATION = args.implementation
    try:
        run_task(args)
    except Exception as error:
        os.makedirs(args.out, exist_ok=True)
        failure = {"scientific_name": SCIENTIFIC_NAMES.get(args.arm, args.arm),
                   "code_identifier": args.arm, "seed": args.seed,
                   "epochs_requested": getattr(args, "epochs", EPOCHS),
                   "slurm": _execution_identity(),
                   "process_gpu": GPU_TELEMETRY.summary(
                       _PROCESS_GPU.get(args.out) or GPU_TELEMETRY.load(
                           os.path.join(args.out, PROCESS_GPU_BINDING))),
                   # the exception class, so a reader can tell the declared
                   # numerical failure from any other kind of failure
                   "error_type": type(error).__name__,
                   "failure": str(error)}
        if isinstance(error, NumericalTrainingFailure):
            failure["record"] = error.record
        if isinstance(error, NumericalTrainingFailure):
            with open(os.path.join(args.out, "failure.json"), "w") as handle:
                json.dump(failure, handle, indent=2)
            return
        raise


if __name__ == "__main__":
    main()
