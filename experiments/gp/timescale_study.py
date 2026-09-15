"""Learned response TIMESCALE: bounded five-arm Speech Commands screen.

One seed, ten epochs, five arms, from scratch, full BPTT:

    native_s5         native input gain, no pole clipping (literature baseline)
    gain_clip_s5      alpha-scaled input + clipped poles, ordinary one-tap
                      recurrence (isolates the substrate change)
    alpha_p_s5        Rawat prospective-INPUT two-tap, fixed input horizon 5
    gp_rho_T_fixed    generalized prospective recurrence, learned rho, T frozen
    gp_rho_T          the same, with a learned per-mode horizon T_i

`gp_rho_T` vs `gp_rho_T_fixed` is the comparison that isolates learning the new
timescale; the two are initialized IDENTICALLY (rho_0 = 0.75, eta_0 = 0, so
T = 5 exactly) and differ only in whether eta receives optimizer updates.

This is a ONE-SEED development screen. It cannot establish a robust gain, and
nothing here escalates to a larger batch automatically.

One process, one deadline, GPU only. The test arrays are never opened.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dataloaders import speech_commands10 as SC                    # noqa: E402
from s5 import timescale_response as TR                            # noqa: E402
from s5.checkpointing import provenance                            # noqa: E402
from s5.rawat_model import RawatClassifier, parameter_report       # noqa: E402
from s5.rawat_s5 import (LOG_RHO_BOUNDS, LOG_T_BOUNDS,             # noqa: E402
                         RHO_INIT_TIMESCALE, RHO_ONLY_PARAM_NAME,
                         T_BOUNDS, T_ONLY_PARAM_NAME, T_REFERENCE,
                         init_substrate_ssm)
from s5.physical_coefficients import SYMMETRIC_REFERENCE           # noqa: E402
from s5.response_projection import (project_response_leaves,       # noqa: E402
                                    projection_telemetry)
import experiments.gp.rawat_benchmark as RB                        # noqa: E402

# ---- Stage 2 architecture and recipe. Verified against the saved Stage 2
#      config at startup when one is supplied; a discrepancy REFUSES to run.
D_MODEL = 32
SSM_SIZE = 32
N_LAYERS = 4
BLOCKS = RB.HIPPO_BLOCKS          # 8
BATCH = 32
EPOCHS = 10
LR = 1e-3
LR_FINAL = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
SEED = 100

ARMS = ("native_s5", "gain_clip_s5", "alpha_p_s5", "gp_rho_T_fixed",
        "gp_rho_T")
#: the two generalized arms, which must be identical at initialization
GENERALIZED = ("gp_rho_T_fixed", "gp_rho_T")
#: the matched ordinary substrate the response differences are reported against
MATCHED_ORDINARY = "gain_clip_s5"
#: the arm whose common parameter tree every other arm is cloned from
CLONE_SOURCE = "gain_clip_s5"

IMPULSE_LAGS = 128
N_FREQ = 65
IMPULSE_BANDS = ((0, 3), (4, 15), (16, 63), (64, 127))
#: equality of the two generalized arms at initialization: they are the SAME
#: function by construction, so this is a tight identity check, not a tolerance
INIT_EQUALITY_TOL = 1e-6

_MODEL_CACHE = {}


def build(arm, training):
    """One module per (arm, training), reused across calls so the jit cache
    is hit. Two separately constructed modules are not equal - the `ssm` field
    is a `functools.partial`, compared by identity - so rebuilding would
    recompile."""
    key = (arm, training)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = RB.build_model(arm, D_MODEL, SSM_SIZE, N_LAYERS,
                                           training)
    return _MODEL_CACHE[key]


_SINGLE_CACHE = {}


def build_single(arm):
    """UNBATCHED view for `method=` reads; `nn.vmap` cannot map non-arrays.

    Cached: the per-epoch response snapshots read four layers for two arms
    every epoch, and reconstructing the module each time re-runs `setup` on
    every call for no benefit.
    """
    if arm not in _SINGLE_CACHE:
        _SINGLE_CACHE[arm] = _make_single(arm)
    return _SINGLE_CACHE[arm]


def _make_single(arm):
    return RawatClassifier(
        ssm=init_substrate_ssm(arm, physical=SYMMETRIC_REFERENCE,
                               **RB.ssm_kwargs(D_MODEL, SSM_SIZE)),
        d_model=D_MODEL, n_layers=N_LAYERS, d_output=RB.D_OUTPUT,
        training=False)


def init_variables(arm, seed):
    m = build(arm, True)
    dummy = jnp.zeros((2, SC.N_FRAMES, SC.N_MFCC))
    v = m.init({"params": jax.random.PRNGKey(seed),
                "dropout": jax.random.PRNGKey(seed + 1)},
               dummy, jnp.ones((2, SC.N_FRAMES)), None)
    return m, v["params"], v.get("batch_stats")


def clone_common(src, dst):
    """Copy every parameter the two trees SHARE; a shape mismatch is an error.

    The arms differ only in static configuration (input gain, pole clipping,
    recurrence) plus the generalized arms' two response leaves, so every
    ordinary parameter is shared. A silently unshared parameter would break the
    pairing, so a mismatch raises rather than being skipped.
    """
    from flax.traverse_util import flatten_dict, unflatten_dict
    a, b = flatten_dict(src), flatten_dict(dst)
    out, copied, kept = {}, [], []
    for k, v in b.items():
        if k in a:
            if a[k].shape != v.shape:
                raise ValueError(f"shape mismatch for {'/'.join(k)}: "
                                 f"{a[k].shape} vs {v.shape}")
            out[k] = a[k]; copied.append("/".join(k))
        else:
            out[k] = v; kept.append("/".join(k))
    return unflatten_dict(out), copied, kept


def tree_digest(tree, skip=()):
    from flax.traverse_util import flatten_dict
    if tree is None:
        return None
    flat = flatten_dict(tree)
    h = hashlib.sha256()
    for k in sorted(flat):
        if k[-1] in skip:
            continue
        v = onp.asarray(flat[k])
        h.update("/".join(k).encode()); h.update(str(v.dtype).encode())
        h.update(str(v.shape).encode()); h.update(v.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------- training --
def make_optimizer(steps_per_epoch, epochs, freeze_T=False):
    """The comparator's optimizer policy, identical for every arm.

    No response-specific learning rate and no weight-decay exemption: both
    response coordinates sit in the same AdamW group as the native parameters.
    The fixed-T arm freezes eta by ZEROING its updates, which removes decoupled
    decay along with the gradient step, rather than by removing the leaf - so
    the two generalized arms keep identical parameter trees.
    """
    total = max(1, steps_per_epoch * epochs)
    sched = optax.cosine_decay_schedule(init_value=LR, decay_steps=total,
                                        alpha=LR_FINAL / LR)
    adamw = optax.adamw(sched, weight_decay=WEIGHT_DECAY)
    if not freeze_T:
        tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), adamw)
    else:
        def label(path, _):
            name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
            return "frozen" if name == T_ONLY_PARAM_NAME else "train"
        tx = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP),
            optax.multi_transform(
                {"train": adamw, "frozen": optax.set_to_zero()},
                lambda p: jax.tree_util.tree_map_with_path(label, p)))
    desc = ("AdamW(wd=%g) on ALL trainable parameters, cosine %g -> %g over %d "
            "steps, global clip %g, label smoothing %g%s"
            % (WEIGHT_DECAY, LR, LR_FINAL, total, GRAD_CLIP, LABEL_SMOOTHING,
               "; log T frozen (updates zeroed, so no decay either)"
               if freeze_T else ""))
    return tx, desc


def loss_fn(params, batch_stats, model, x, y, rng):
    variables = {"params": params}
    if batch_stats is not None:
        variables["batch_stats"] = batch_stats
    ts = jnp.ones(x.shape[:2])
    logits, updates = model.apply(variables, x, ts, None,
                                  rngs={"dropout": rng},
                                  mutable=["batch_stats"])
    onehot = jax.nn.one_hot(y, RB.D_OUTPUT)
    loss = optax.softmax_cross_entropy(
        logits, optax.smooth_labels(onehot, LABEL_SMOOTHING)).mean()
    return loss, (logits, updates.get("batch_stats"))


@partial(jax.jit, static_argnums=(0, 1))
def train_step(model, tx, params, opt_state, batch_stats, x, y, rng):
    """One optimizer update, THEN the feasible-set projection of BOTH raw
    response coordinates. Optimizer state is untouched by the projection."""
    (loss, (logits, new_bs)), g = jax.value_and_grad(loss_fn, has_aux=True)(
        params, batch_stats, model, x, y, rng)
    upd, opt_state = tx.update(g, opt_state, params)
    raw = optax.apply_updates(params, upd)
    params = project_response_leaves(raw)
    tel = dict(
        rho=projection_telemetry(raw, params, g, upd,
                                 names=(RHO_ONLY_PARAM_NAME,)),
        T=projection_telemetry(raw, params, g, upd,
                               names=(T_ONLY_PARAM_NAME,)))
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return (params, opt_state, (new_bs if new_bs is not None else batch_stats),
            loss, acc, optax.global_norm(g), tel)


@partial(jax.jit, static_argnums=(0,))
def eval_batch(model, params, batch_stats, x, y, mask):
    """`mask` zeroes the padded rows of a ragged final batch.

    It is a traced argument, so every evaluation batch has the SAME shape and
    the evaluation program compiles exactly once per arm. A ragged final batch
    would otherwise compile a second program on the first epoch and again
    whenever the split size changed.
    """
    variables = {"params": params}
    if batch_stats is not None:
        variables["batch_stats"] = batch_stats
    logits = model.apply(variables, x, jnp.ones(x.shape[:2]), None)
    correct = jnp.sum((jnp.argmax(logits, -1) == y) * mask)
    ce = jnp.sum(optax.softmax_cross_entropy(
        logits, jax.nn.one_hot(y, RB.D_OUTPUT)) * mask)
    return correct, ce, jnp.sum(mask)


def evaluate_split(model, params, batch_stats, X, Y, batch=256):
    """Score a split at a FIXED batch shape, padding the final batch."""
    correct, ce, seen = 0.0, 0.0, 0.0
    n = X.shape[0]
    for i in range(0, n, batch):
        xb = onp.asarray(X[i:i + batch]); yb = onp.asarray(Y[i:i + batch])
        m = xb.shape[0]
        mask = onp.zeros((batch,), dtype=onp.float32)
        mask[:m] = 1.0
        if m < batch:
            pad = batch - m
            xb = onp.concatenate([xb, onp.repeat(xb[-1:], pad, 0)], 0)
            yb = onp.concatenate([yb, onp.repeat(yb[-1:], pad, 0)], 0)
        c, l, k = eval_batch(model, params, batch_stats, jnp.asarray(xb),
                             jnp.asarray(yb), jnp.asarray(mask))
        correct += float(c); ce += float(l); seen += float(k)
    return dict(accuracy=correct / seen, cross_entropy=ce / seen, n=int(seen))


# ------------------------------------------------------- response reading --
def read_core(arm, params, layer):
    from s5 import substrate_diagnostics as SD
    return SD.read_core(build_single(arm), {"params": params}, layer)


def response_state(arm, params):
    """Raw and EXECUTED rho and T per layer, derived mass, boundary occupancy.

    Everything here is read from the bound module's executed coefficients, so a
    learned-timescale arm reports its learned per-mode T rather than the static
    `physical.T` field.
    """
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    raws = {"/".join(k): onp.asarray(v) for k, v in flat.items()
            if k[-1] in (RHO_ONLY_PARAM_NAME, T_ONLY_PARAM_NAME)}
    if not raws:
        return None
    rows = []
    for layer in range(N_LAYERS):
        core = read_core(arm, params, layer)
        ex = core["executed_response"]
        rho = onp.asarray(ex["rho"]).ravel()
        T = onp.asarray(onp.broadcast_to(onp.asarray(ex["T"]), rho.shape))
        a = onp.asarray(core["a"]).ravel()
        Tj = T * (-a)
        rows.append(dict(
            layer=layer, kind=ex["kind"], T_is_per_mode=bool(ex["T_is_per_mode"]),
            rho=rho.tolist(), T=T.tolist(), mu=(rho * T).tolist(),
            # the EXECUTED clock, Delta = exp(log_step); log_step itself is
            # recoverable as log(Delta) and both are recorded by name
            Delta=onp.asarray(core["Delta"]).ravel().tolist(),
            log_step=onp.log(onp.asarray(core["Delta"])).ravel().tolist(),
            T_times_j=dict(abs=onp.abs(Tj).tolist(),
                           real=onp.real(Tj).tolist(),
                           imag=onp.imag(Tj).tolist()),
            summary=TR.summarize(T, rho),
            components_check=TR.check_identities(T, rho)))
    lo_r, hi_r = LOG_RHO_BOUNDS
    lo_t, hi_t = LOG_T_BOUNDS
    occupancy = {}
    for name, raw in raws.items():
        lo, hi = ((lo_t, hi_t) if name.endswith(T_ONLY_PARAM_NAME)
                  else (lo_r, hi_r))
        occupancy[name] = dict(
            raw=raw.tolist(), n_modes=int(raw.size),
            n_at_upper=int(onp.sum(raw >= hi - 1e-12)),
            n_at_lower=int(onp.sum(raw <= lo + 1e-12)),
            n_outside_raw=int(onp.sum((raw > hi) | (raw < lo))))
    return dict(layers=rows, raw_leaves=occupancy)


def signal_response_probe(arm, params, reference=None):
    """Signal-only impulse and frequency probe, native D removed.

    Reported, never gated: the generalized arms start at rho = 0.75 and are
    MEANT to differ from the ordinary substrate, so the old 1 % match gate does
    not apply to them. When `reference` is given the relative differences from
    that arm's cores are recorded alongside the band energies.
    """
    from s5 import substrate_diagnostics as SD
    rows = []
    for layer in range(N_LAYERS):
        core = read_core(arm, params, layer)
        K = SD.impulse_matrices(core, IMPULSE_LAGS).copy()
        D = onp.diag(onp.asarray(core["D"]))
        K[0] -= D
        w, Hf = SD.frequency_response(core, N_FREQ)
        Hf = Hf - D[None]
        row = dict(layer=layer, window_lags=IMPULSE_LAGS, n_freq=N_FREQ,
                   tail_beyond_window="unknown (not bounded here)",
                   **SD.band_energy(K, IMPULSE_BANDS))
        if reference is not None:
            rcore = read_core(reference[0], reference[1], layer)
            Kr = SD.impulse_matrices(rcore, IMPULSE_LAGS).copy()
            Dr = onp.diag(onp.asarray(rcore["D"]))
            Kr[0] -= Dr
            _, Hr = SD.frequency_response(rcore, N_FREQ)
            Hr = Hr - Dr[None]
            nk = float(onp.sqrt(onp.sum(Kr ** 2)))
            nh = float(onp.sqrt(onp.sum(onp.abs(Hr) ** 2)))
            dk = float(onp.sqrt(onp.sum((K - Kr) ** 2)))
            dh = float(onp.sqrt(onp.sum(onp.abs(Hf - Hr) ** 2)))
            row["vs_matched_ordinary"] = dict(
                reference_arm=reference[0],
                impulse_abs_diff=dk, impulse_ref_norm=nk,
                impulse_rel=(dk / nk if nk > 1e-12 else None),
                freq_abs_diff=dh, freq_ref_norm=nh,
                freq_rel=(dh / nh if nh > 1e-12 else None),
                note=("RECORDED, not gated: these arms start at rho = 0.75 and "
                      "are intended to have a different response"))
        rows.append(row)
    return rows


def state_counts(arm, params):
    m = build_single(arm)
    per_layer = m.apply({"params": params},
                        method=lambda mm: [mm.encoder.layers[i].seq
                                           .state_counts()
                                           for i in range(N_LAYERS)])
    total = {}
    for k, v in per_layer[0].items():
        total[k] = (v if isinstance(v, (bool, str)) or v is None
                    else int(v) * N_LAYERS)
    return dict(per_layer=per_layer, total_over_layers=total,
                n_layers=N_LAYERS)


def counts(params, arm):
    """STORED values and TRAINABLE degrees of freedom are different numbers.

    `gp_rho_T_fixed` stores the eta leaves but never updates them, so it has
    the same stored count as `gp_rho_T` and 64 fewer trainable coefficients.
    More trainable coefficients do not imply more carried state: both
    generalized arms carry identical physical and auxiliary state.
    """
    from flax.traverse_util import flatten_dict
    rep = parameter_report(params)
    flat = flatten_dict(params)
    rho = sum(int(v.size) for k, v in flat.items()
              if k[-1] == RHO_ONLY_PARAM_NAME)
    eta = sum(int(v.size) for k, v in flat.items()
              if k[-1] == T_ONLY_PARAM_NAME)
    frozen = eta if arm == "gp_rho_T_fixed" else 0
    return dict(stored=rep["total"], ssm=rep["ssm"], other=rep["other"],
                rho_leaf_values=rho, log_T_leaf_values=eta,
                frozen_values=frozen, trainable=rep["total"] - frozen,
                note=("gp_rho_T_fixed stores log T but excludes it from every "
                      "update and from decay" if frozen else None))


def jsonable(o):
    if isinstance(o, onp.integer):
        return int(o)
    if isinstance(o, onp.floating):
        return float(o)
    if isinstance(o, onp.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


def write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)


def save_tree(path, tree):
    from flax import serialization
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(serialization.to_bytes(tree))
    return path


# ------------------------------------------------------------- one arm -----
def run_arm(arm, seed, params, batch_stats, data, steps_per_epoch, epochs,
            out, deadline, reserve_s, status):
    """Train ONE arm for `epochs` epochs. Returns the row, or None if the
    budget stopped it. A partial arm is recorded as incomplete, never scored."""
    (Xtr, Ytr), (Xva, Yva) = data["train"], data["val"]
    train_model, eval_model = build(arm, True), build(arm, False)
    tx, opt_desc = make_optimizer(steps_per_epoch, epochs,
                                  freeze_T=(arm == "gp_rho_T_fixed"))
    opt_state = tx.init(params)
    rng = jax.random.PRNGKey(seed + 2)      # the SAME stream for every arm
    epochs_rec, best = [], dict(accuracy=-1.0, cross_entropy=float("inf"),
                                epoch=-1)
    trajectory = [dict(epoch=-1, when="initialization",
                       response=response_state(arm, params))]
    tot = {c: dict(events=0, max_over=0.0, g=0.0, u=0.0) for c in ("rho", "T")}
    n_steps = 0
    t_arm = time.time()
    for epoch in range(epochs):
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"arm {arm}: stopped at epoch {epoch} (budget); NOT scored")
            return None
        t0 = time.time()
        tot_loss = tot_acc = nb = 0.0
        for idx in SC.epoch_batches(Xtr.shape[0], BATCH, seed, epoch):
            sel = onp.sort(idx)
            xb = jnp.asarray(onp.asarray(Xtr[sel]))
            yb = jnp.asarray(Ytr[sel])
            rng, sub = jax.random.split(rng)
            (params, opt_state, batch_stats, loss, acc, gn,
             tel) = train_step(train_model, tx, params, opt_state,
                               batch_stats, xb, yb, sub)
            tot_loss += float(loss); tot_acc += float(acc); nb += 1
            n_steps += 1
            for c in ("rho", "T"):
                tot[c]["events"] += int(tel[c]["n_projected"])
                tot[c]["max_over"] = max(tot[c]["max_over"],
                                         float(tel[c]["max_overshoot"]))
                tot[c]["g"] += float(tel[c]["response_grad_norm"])
                tot[c]["u"] += float(tel[c]["response_update_norm"])
        val = evaluate_split(eval_model, params, batch_stats, Xva, Yva)
        if val["accuracy"] > best["accuracy"]:
            best = dict(accuracy=val["accuracy"],
                        cross_entropy=val["cross_entropy"], epoch=epoch)
            save_tree(os.path.join(out, "params",
                                   f"best_{arm}.msgpack"),
                      dict(params=params, batch_stats=batch_stats))
        rec = dict(epoch=epoch, train_loss=tot_loss / nb,
                   train_acc=tot_acc / nb, val_accuracy=val["accuracy"],
                   val_cross_entropy=val["cross_entropy"],
                   grad_norm=float(gn), epoch_s=time.time() - t0,
                   peak_memory_bytes=RB.peak_memory_bytes())
        epochs_rec.append(rec)
        trajectory.append(dict(epoch=epoch, when="end of epoch",
                               response=response_state(arm, params)))
        print(f"  {arm:16s} epoch {epoch:>2}  "
              f"train {rec['train_loss']:.4f}/{rec['train_acc']:.4f}  "
              f"val {val['accuracy']:.4f}  best {best['accuracy']:.4f}  "
              f"{rec['epoch_s']:.1f}s")
        write(os.path.join(out, "status.json"), status)
    save_tree(os.path.join(out, "params", f"final_{arm}.msgpack"),
              dict(params=params, batch_stats=batch_stats))
    save_tree(os.path.join(out, "params", f"optimizer_{arm}.msgpack"),
              opt_state)
    n = max(n_steps, 1)
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    sizes = dict(
        rho=sum(int(v.size) for k, v in flat.items()
                if k[-1] == RHO_ONLY_PARAM_NAME),
        T=sum(int(v.size) for k, v in flat.items()
              if k[-1] == T_ONLY_PARAM_NAME))
    telemetry = {c: dict(
        n_projection_events=tot[c]["events"],
        entry_updates=n * sizes[c],
        max_proposed_overshoot=tot[c]["max_over"],
        mean_grad_norm=tot[c]["g"] / n, mean_update_norm=tot[c]["u"] / n,
        steps=n_steps,
        note=("n_projection_events counts (entry, update) EVENTS, not distinct "
              "modes, steps or time at the boundary; max_proposed_overshoot is "
              "the PRE-projection excursion the optimizer proposed and the "
              "projection rejected, never an executed coefficient"))
        for c in ("rho", "T")}
    return dict(
        arm=arm, seed=seed, optimizer=opt_desc,
        primary_endpoint_val_accuracy=epochs_rec[-1]["val_accuracy"],
        primary_endpoint_val_cross_entropy=epochs_rec[-1]["val_cross_entropy"],
        primary_endpoint_epoch=epochs_rec[-1]["epoch"],
        best_val=best, epochs=epochs_rec, wall_s=time.time() - t_arm,
        params=parameter_report(params), counts=counts(params, arm),
        state_counts=state_counts(arm, params),
        response_trajectory=trajectory,
        response_final=response_state(arm, params),
        signal_probe_endpoint=signal_response_probe(arm, params),
        projection_telemetry=telemetry,
        peak_memory_bytes=RB.peak_memory_bytes(),
        final_param_digest=tree_digest(params))


# ------------------------------------------------------------- preflight ---
def preflight(data, steps_per_epoch, epochs, seed, status):
    """Measure the cost of ALL FIVE arms, not one.

    Compilation is counted once per arm because the modules are cached and
    reused; validation is extrapolated from a measured partial pass. Preflight
    updates are DISCARDED: the comparative runs re-initialize parameters,
    optimizer state and the RNG stream from scratch.
    """
    (Xtr, Ytr), (Xva, Yva) = data["train"], data["val"]
    idx = onp.sort(next(iter(SC.epoch_batches(Xtr.shape[0], BATCH, 0, 0))))
    xb = jnp.asarray(onp.asarray(Xtr[idx])); yb = jnp.asarray(Ytr[idx])
    rows, total = [], 0.0
    for arm in ARMS:
        tx, _ = make_optimizer(steps_per_epoch, epochs,
                               freeze_T=(arm == "gp_rho_T_fixed"))
        m, p, bs = init_variables(arm, seed)
        RB.assert_response_policy(p, arm, N_LAYERS, SSM_SIZE // 2)
        opt = tx.init(p)
        rng = jax.random.PRNGKey(0)
        t0 = time.time()
        q, o2, b2, loss, _, _, _ = train_step(m, tx, p, opt, bs, xb, yb, rng)
        loss.block_until_ready()
        compile_s = time.time() - t0
        t1 = time.time()
        for _ in range(3):
            q, o2, b2, loss, _, _, _ = train_step(m, tx, q, o2, b2, xb, yb, rng)
        loss.block_until_ready()
        step_s = (time.time() - t1) / 3
        # Evaluation: the COMPILE is a one-off and the per-sample cost is not.
        # Timing one 512-sample call and scaling it by n_val/512 multiplies the
        # evaluation compile by that same factor. On the first attempt that
        # turned a ~4 s compile into a claimed 43.7 s per epoch, projected
        # 2,790 s and refused a batch that in fact fits. The two are now
        # measured separately.
        eval_model = build(arm, False)
        t2 = time.time()
        evaluate_split(eval_model, p, bs, Xva[:512], Yva[:512])
        eval_compile_s = time.time() - t2
        t3 = time.time()
        evaluate_split(eval_model, p, bs, Xva[:512], Yva[:512])
        val_steady_s = (time.time() - t3) * (Xva.shape[0] / 512.0)
        arm_s = (compile_s + eval_compile_s
                 + epochs * (step_s * steps_per_epoch + val_steady_s))
        total += arm_s
        rows.append(dict(arm=arm, compile_s=compile_s,
                         eval_compile_s=eval_compile_s, step_s=step_s,
                         epoch_s=step_s * steps_per_epoch,
                         val_pass_s=val_steady_s, arm_total_s=arm_s))
        print(f"[preflight] {arm:16s} compile {compile_s:6.1f}s"
              f" (+eval {eval_compile_s:5.1f}s)  step {step_s * 1e3:6.2f}ms  "
              f"epoch {step_s * steps_per_epoch:6.1f}s  val {val_steady_s:5.2f}s"
              f"  arm {arm_s:6.1f}s")
        del q, o2, b2, opt
    # initialization gate, the per-epoch response snapshots and checkpoint
    # writes. Host-side and measured only as an allowance, not extrapolated.
    host_s = 60.0
    total += host_s
    status["preflight"] = dict(
        rows=rows, epochs=epochs, arms=list(ARMS),
        host_allowance_s=host_s, projected_total_s=total,
        scope=("training AND evaluation compilation once per arm, plus every "
               "arm x epoch, the steady-state validation pass, the "
               "initialization gate and the per-epoch response snapshots. "
               "Evaluation compile and steady cost are measured SEPARATELY; "
               "scaling a compile-dominated timing per sample is what made "
               "the first attempt project 2790 s."))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cache", required=True)
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/timescale")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--stage2_config", default=None,
                    help="saved Stage 2 config.json; a discrepancy REFUSES")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=1200.0)
    ap.add_argument("--reserve_s", type=float, default=40.0)
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    left = lambda: deadline - time.time() - args.reserve_s          # noqa: E731

    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    # train and val ONLY: the test arrays are never opened, not merely unscored
    data, manifest = SC.load_splits(args.data_cache, splits=("train", "val"))
    steps_per_epoch = data["train"][0].shape[0] // BATCH
    seed = args.seed

    cfg = dict(d_model=D_MODEL, ssm_size=SSM_SIZE, n_layers=N_LAYERS,
               hippo_blocks=BLOCKS, conj_sym=True, discretization="zoh",
               bidirectional=False, batch=BATCH, epochs=args.epochs, lr=LR,
               lr_final=LR_FINAL, weight_decay=WEIGHT_DECAY,
               grad_clip=GRAD_CLIP, label_smoothing=LABEL_SMOOTHING)
    config_check = verify_against_stage2(args.stage2_config, cfg)
    if config_check.get("discrepancies"):
        raise SystemExit(
            "REFUSING: this configuration differs from the saved Stage 2 "
            f"config: {config_check['discrepancies']}. Exposed rather than "
            "reconciled; resolve it before launching.")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)

    status = dict(
        run_id=run_id, out=out, backend=backend, seed=seed,
        arms=list(ARMS), epochs=args.epochs, batch=BATCH,
        steps_per_epoch=steps_per_epoch, config=cfg,
        stage2_config_check=config_check,
        design=dict(
            primary="validation accuracy at the fixed tenth epoch",
            isolating_comparison="gp_rho_T minus gp_rho_T_fixed",
            literature_comparisons=["native_s5", "alpha_p_s5"],
            substrate_control="gain_clip_s5",
            seeds=1,
            caveat=("ONE-seed development screen. It cannot establish a robust "
                    "gain; a positive result is a candidate for confirmation, "
                    "not a benchmark improvement or proof of physical "
                    "causation. No imported +0.3 pp threshold, no automatic "
                    "escalation, no checkpoint or seed selection.")),
        response_policy=dict(
            parameterization="per-mode rho and per-mode T; gamma_n = 1",
            learned_leaves=[RHO_ONLY_PARAM_NAME, T_ONLY_PARAM_NAME],
            T_parameterization="T_i = %g * exp(eta_i)" % T_REFERENCE,
            rho_init=RHO_INIT_TIMESCALE, eta_init=0.0,
            T_init=T_REFERENCE,
            rho_bounds=[float(onp.exp(LOG_RHO_BOUNDS[0])),
                        float(onp.exp(LOG_RHO_BOUNDS[1]))],
            T_bounds=list(T_BOUNDS),
            mass="derived at every forward call, mu_i = rho_i T_i",
            within_sequence_adaptation=False,
            circuit_map=dict(c_star=TR.C_STAR,
                             identities="kappa=c*/T, tau_d=T rho, "
                                        "gamma_phys=T, M_phys=T^2 rho"),
            decay_note=("AdamW decoupled decay shrinks eta toward 0, which "
                        "biases T toward its REFERENCE value 5, not toward 0. "
                        "Log-coordinate decay is not coordinate invariant."),
            bounds_note=("T bounds are DECLARED NUMERICAL guardrails for this "
                         "bounded study, not physiological limits, and were "
                         "not selected from accuracy.")),
        budget_s=args.budget_s, provenance=provenance(),
        data_manifest=dict(
            counts=manifest["counts"],
            file_list_sha256=manifest["file_list_sha256"],
            feature_sha256={k: v for k, v in manifest["feature_sha256"].items()
                            if k in ("train", "val")},
            mfcc=manifest["mfcc"], split_seed=manifest["split_seed"],
            splits_opened=["train", "val"]),
        gate=None, results=[], incomplete=[])
    print(f"[*] out={out} backend={backend} seed={seed} "
          f"epochs={args.epochs} steps/epoch={steps_per_epoch}")
    write(os.path.join(out, "status.json"), status)

    proj = preflight(data, steps_per_epoch, args.epochs, seed, status)
    write(os.path.join(out, "status.json"), status)
    if args.preflight_only:
        print(f"TIMESCALE_STATUS=PREFLIGHT_ONLY out={out}")
        return 0
    if proj > left():
        status["incomplete"].append(
            f"projected {proj:.0f}s > remaining {left():.0f}s; comparative "
            f"training NOT started. No epoch, arm or control was reduced and "
            f"the cap was not widened.")
        write(os.path.join(out, "status.json"), status)
        print(f"[!] {status['incomplete'][-1]}")
        print(f"TIMESCALE_STATUS=INCOMPLETE out={out}")
        return 3

    # ---- one common initialization, cloned into every arm
    _, p_src, bs_src = init_variables(CLONE_SOURCE, seed)
    inits = {}
    for arm in ARMS:
        _, p_a, bs_a = init_variables(arm, seed)
        p_a, copied, kept = clone_common(p_src, p_a)
        RB.assert_response_policy(p_a, arm, N_LAYERS, SSM_SIZE // 2)
        inits[arm] = (p_a, bs_a, dict(n_cloned=len(copied),
                                      n_arm_specific=len(kept),
                                      arm_specific=kept))
    skip = (RHO_ONLY_PARAM_NAME, T_ONLY_PARAM_NAME)
    common_digests = {a: tree_digest(inits[a][0], skip=skip) for a in ARMS}
    bs_digests = {a: tree_digest(inits[a][1]) for a in ARMS}
    status["initialization"] = dict(
        clone_source=CLONE_SOURCE,
        common_param_digests=common_digests,
        batch_stats_digests=bs_digests,
        common_params_identical=len(set(common_digests.values())) == 1,
        batch_stats_identical=len(set(bs_digests.values())) == 1,
        arm_specific={a: inits[a][2] for a in ARMS},
        note=("the native-vs-alpha input gain and the pole-clipping difference "
              "are STATIC configuration, not parameters: they are intentional "
              "and remain explicit"))
    if not (status["initialization"]["common_params_identical"]
            and status["initialization"]["batch_stats_identical"]):
        status["incomplete"].append(
            "common initialization is NOT identical across arms; refusing to "
            "report a paired comparison")
        write(os.path.join(out, "status.json"), status)
        print(f"TIMESCALE_STATUS=FAILED out={out}")
        return 4

    # ---- initialization check: the two generalized arms are the SAME function
    #
    # NOT the old 1 % match-to-ordinary gate: these arms start at rho = 0.75
    # and are MEANT to differ from the ordinary substrate. What must hold is
    # that the fixed-T and learned-T arms are identical, so their difference
    # after training is attributable to learning eta and to nothing else.
    a_fix, a_lrn = GENERALIZED
    xprobe = jnp.asarray(onp.asarray(data["val"][0][:32]))
    y_fix = onp.asarray(build(a_fix, False).apply(
        {"params": inits[a_fix][0], "batch_stats": inits[a_fix][1]},
        xprobe, jnp.ones(xprobe.shape[:2]), None))
    y_lrn = onp.asarray(build(a_lrn, False).apply(
        {"params": inits[a_lrn][0], "batch_stats": inits[a_lrn][1]},
        xprobe, jnp.ones(xprobe.shape[:2]), None))
    max_logit_diff = float(onp.max(onp.abs(y_fix - y_lrn)))
    resp_fix = response_state(a_fix, inits[a_fix][0])
    resp_lrn = response_state(a_lrn, inits[a_lrn][0])
    T0 = onp.asarray(resp_lrn["layers"][0]["T"])
    rho0 = onp.asarray(resp_lrn["layers"][0]["rho"])
    gate = dict(
        kind="equality of the two generalized arms at initialization",
        arms=[a_fix, a_lrn], max_abs_logit_difference=max_logit_diff,
        tolerance=INIT_EQUALITY_TOL,
        param_digest_equal=(tree_digest(inits[a_fix][0])
                            == tree_digest(inits[a_lrn][0])),
        T_at_init=dict(min=float(T0.min()), max=float(T0.max()),
                       expected=T_REFERENCE),
        rho_at_init=dict(min=float(rho0.min()), max=float(rho0.max()),
                         expected=RHO_INIT_TIMESCALE),
        # RECORDED, not gated
        signal_difference_from_matched_ordinary={
            a: signal_response_probe(
                a, inits[a][0],
                reference=(MATCHED_ORDINARY, inits[MATCHED_ORDINARY][0]))
            for a in GENERALIZED},
        note=("the superseded 1 % function-match-to-ordinary gate does NOT "
              "apply to these arms: rho_0 = 0.75 gives them a deliberately "
              "different response. The differences from the matched ordinary "
              "substrate are recorded, not gated, and no label or score is "
              "read here."))
    gate["passed"] = bool(
        max_logit_diff <= INIT_EQUALITY_TOL and gate["param_digest_equal"]
        and abs(float(T0.max()) - T_REFERENCE) < 1e-5
        and abs(float(rho0.max()) - RHO_INIT_TIMESCALE) < 1e-5)
    status["gate"] = gate
    write(os.path.join(out, "status.json"), status)
    print(f"[gate] generalized arms identical at init: "
          f"max|dlogit| = {max_logit_diff:.3e} <= {INIT_EQUALITY_TOL:.0e}, "
          f"T0 = {float(T0.max()):.6f}, rho0 = {float(rho0.max()):.6f} -> "
          f"{'PASS' if gate['passed'] else 'FAIL'}")
    if not gate["passed"]:
        status["incomplete"].append(f"initialization check FAILED: {gate}")
        write(os.path.join(out, "status.json"), status)
        print("[STOP] the two generalized arms are NOT identical at "
              "initialization. Stopping and reporting; rho and T are not "
              "adjusted to make this pass.")
        print(f"TIMESCALE_STATUS=GATE_STOP out={out}")
        return 3

    rows = []
    for arm in ARMS:
        if left() < 30:
            status["incomplete"].append(f"arm {arm} not started (budget)")
            continue
        p_a, bs_a, _ = inits[arm]
        row = run_arm(arm, seed, p_a, bs_a, data, steps_per_epoch,
                      args.epochs, out, deadline, args.reserve_s, status)
        if row is None:
            write(os.path.join(out, "status.json"), status)
            continue
        row["initial_param_digest"] = tree_digest(p_a)
        rows.append(row)
        status["results"] = rows
        write(os.path.join(out, "results.json"), rows)
        write(os.path.join(out, "status.json"), status)

    by = {r["arm"]: r for r in rows}

    def delta(a, b, field="primary_endpoint_val_accuracy"):
        if a not in by or b not in by:
            return None
        return 100.0 * (by[a][field] - by[b][field])

    status["comparisons"] = dict(
        primary="validation accuracy at the fixed tenth epoch, percentage "
                "points, learned-T minus each control",
        isolating=dict(name="gp_rho_T - gp_rho_T_fixed",
                       endpoint_pp=delta("gp_rho_T", "gp_rho_T_fixed"),
                       note="the ONLY difference between these two arms is "
                            "whether eta receives optimizer updates"),
        vs_matched_ordinary=delta("gp_rho_T", "gain_clip_s5"),
        vs_rawat=delta("gp_rho_T", "alpha_p_s5"),
        vs_native=delta("gp_rho_T", "native_s5"),
        fixed_vs_matched_ordinary=delta("gp_rho_T_fixed", "gain_clip_s5"),
        fixed_vs_rawat=delta("gp_rho_T_fixed", "alpha_p_s5"),
        best_checkpoint_secondary={
            a: (by[a]["best_val"] if a in by else None) for a in ARMS},
        note=("best-validation scores are SECONDARY and reported separately; "
              "they do not replace the predeclared fixed-epoch endpoint, and "
              "the endpoint was not changed after seeing the ordering"))
    status["wall_s"] = time.time() - t0
    status["complete"] = len(rows) == len(ARMS) and not status["incomplete"]
    write(os.path.join(out, "status.json"), status)

    print("\n  arm                 endpoint val acc   endpoint CE   best")
    for a in ARMS:
        if a not in by:
            print(f"  {a:18s} NOT COMPLETED")
            continue
        r = by[a]
        print(f"  {a:18s} {r['primary_endpoint_val_accuracy']:.4f}"
              f"             {r['primary_endpoint_val_cross_entropy']:.5f}"
              f"      {r['best_val']['accuracy']:.4f}")
    iso = status["comparisons"]["isolating"]["endpoint_pp"]
    if iso is not None:
        print(f"\n  [isolating] gp_rho_T - gp_rho_T_fixed = {iso:+.3f} pp")
    print(f"[*] wall {status['wall_s']:.0f}s  arms {len(rows)}/{len(ARMS)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[INCOMPLETE] {m}")
        print(f"TIMESCALE_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"TIMESCALE_STATUS=COMPLETE out={out}")
    return 0


def verify_against_stage2(path, cfg):
    """Compare this configuration with the SAVED Stage 2 config.

    Architectural defaults are not inferred: if a saved config is supplied,
    every field both carry must agree, and a discrepancy is EXPOSED and refuses
    to run rather than being silently reconciled. If none is supplied that is
    recorded too, so the report cannot imply a verification that did not occur.
    """
    if not path:
        return dict(checked=False, config=cfg,
                    note="no saved Stage 2 config supplied; the values above "
                         "are the committed protocol's and were NOT verified "
                         "against a saved run in this execution")
    with open(path) as fh:
        saved = json.load(fh)
    flat = saved.get("args", saved)
    alias = dict(d_model="d_model", ssm_size="ssm_size", n_layers="n_layers",
                 batch="batch", lr="lr", lr_final="lr_final",
                 weight_decay="weight_decay", grad_clip="grad_clip",
                 label_smoothing="label_smoothing")
    diffs = {}
    for ours, theirs in alias.items():
        if theirs in flat and flat[theirs] != cfg[ours]:
            diffs[ours] = dict(ours=cfg[ours], saved=flat[theirs])
    return dict(checked=True, path=path, config=cfg, saved_subset=flat,
                discrepancies=diffs)


if __name__ == "__main__":
    sys.exit(main())
