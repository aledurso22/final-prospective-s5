"""Combined prospective input + prospective recurrence: bounded SC10 screen.

Two arms, one substrate, three paired seeds, ten epochs, from scratch:

    rawat            Rawat's prospective-INPUT alpha-P-S5, unchanged
    gp_rho_prospin   the same prospective input driving the generalized
                     prospective RECURRENCE (rho-only, mass derived mu = T rho)

Everything shared is shared by construction and checked, not asserted: the two
arms start from the SAME common parameter tree and batch statistics, see the
SAME ordered minibatches, and draw from the SAME dropout stream. The combined
arm's only extra parameters are its `log_response_rho_only` leaves.

One process, one deadline, GPU only. The test split is never opened: this
runner calls `load_splits(..., splits=("train","val"))` rather than `load`.
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
from s5.checkpointing import provenance                            # noqa: E402
from s5.physical_coefficients import SYMMETRIC_REFERENCE           # noqa: E402
from s5.rawat_model import (BatchRawatClassifier, RawatClassifier,  # noqa: E402
                            parameter_report)
from s5.rawat_s5 import (LOG_RHO_BOUNDS, RHO_INIT_RECALL,          # noqa: E402
                         RHO_ONLY_PARAM_NAME, init_substrate_ssm)
from s5.response_projection import (project_response_leaves,       # noqa: E402
                                    projection_telemetry,
                                    response_leaves)
import experiments.gp.rawat_benchmark as RB                        # noqa: E402

# ---- declared configuration. Copied from the executed Stage 2 configuration;
#      the runner re-reads the saved Stage 2 config at startup when one is
#      supplied and REFUSES to run on a discrepancy rather than reconciling it.
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
SEEDS = (100, 101, 102)

ARMS = ("rawat", "gp_rho_prospin")
ARM_SUBSTRATE = dict(rawat="alpha_p_s5", gp_rho_prospin="gp_rho_prospin")
#: the arm whose initial signal response the combined arm must match
REFERENCE_ARM = "rawat"

#: initialization gate, declared before execution. Identical definition to the
#: recall study's corrected gate: relative Frobenius of the signal-only
#: per-layer recurrent CORE response, native D removed, over a finite lag
#: window and a finite frequency grid. BOTH criteria are enforced.
INIT_RESPONSE_TOL = 0.01
INIT_RESPONSE_ABS_TOL = 1e-9
IMPULSE_LAGS = 128
N_FREQ = 65
#: lag bands for the response summary; the tail beyond the window is UNKNOWN
IMPULSE_BANDS = ((0, 3), (4, 15), (16, 63), (64, 127))

_MODEL_CACHE = {}


def build(arm, training):
    """Models are built ONCE per (arm, training) and reused across seeds.

    Two separately constructed modules are not equal - the `ssm` field is a
    `functools.partial`, compared by identity - so rebuilding per seed would
    miss the jit cache and recompile every arm for every seed. With three seeds
    inside one 1,200 s budget that is not an optimization, it is the difference
    between fitting and not fitting.
    """
    key = (arm, training)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = RB.build_model(ARM_SUBSTRATE[arm], D_MODEL,
                                           SSM_SIZE, N_LAYERS, training)
    return _MODEL_CACHE[key]


def build_single(arm):
    """UNBATCHED view for `method=` reads; `nn.vmap` cannot map non-arrays."""
    return RawatClassifier(
        ssm=init_substrate_ssm(ARM_SUBSTRATE[arm],
                               physical=SYMMETRIC_REFERENCE,
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

    The combined arm's tree is the reference arm's tree plus the rho leaves, so
    everything except those leaves is copied. A silently unshared parameter
    would break the pairing, so a mismatch raises.
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
    """SHA-256 over the flattened tree, so "same initialization" is checkable.

    Leaves named in `skip` are excluded, which is how the arm-specific rho
    leaves are held out of the COMMON-parameter comparison.
    """
    from flax.traverse_util import flatten_dict
    h = hashlib.sha256()
    if tree is None:
        return None
    for k in sorted(flatten_dict(tree)):
        if k[-1] in skip:
            continue
        v = onp.asarray(flatten_dict(tree)[k])
        h.update("/".join(k).encode()); h.update(str(v.dtype).encode())
        h.update(str(v.shape).encode()); h.update(v.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------- training --
def make_optimizer(steps_per_epoch, epochs):
    total = max(1, steps_per_epoch * epochs)
    sched = optax.cosine_decay_schedule(init_value=LR, decay_steps=total,
                                        alpha=LR_FINAL / LR)
    tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                     optax.adamw(sched, weight_decay=WEIGHT_DECAY))
    return tx, ("single group: AdamW(wd=%g) on ALL trainable parameters, "
                "cosine %g -> %g over %d steps, global clip %g, label "
                "smoothing %g. No response-specific learning rate and no "
                "weight-decay exemption." % (WEIGHT_DECAY, LR, LR_FINAL,
                                             total, GRAD_CLIP,
                                             LABEL_SMOOTHING))


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
    """One optimizer update, THEN the feasible-set projection.

    Full BPTT: no `stop_gradient` anywhere on the block carry or the delayed
    input, so the combined arm's gradient reaches rho, every ordinary S5
    parameter, and the previous-input path including its dependence on earlier
    lower-layer activations.
    """
    (loss, (logits, new_bs)), g = jax.value_and_grad(loss_fn, has_aux=True)(
        params, batch_stats, model, x, y, rng)
    upd, opt_state = tx.update(g, opt_state, params)
    raw = optax.apply_updates(params, upd)
    params = project_response_leaves(raw)
    tel = projection_telemetry(raw, params, grads=g, updates=upd)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return (params, opt_state, (new_bs if new_bs is not None else batch_stats),
            loss, acc, optax.global_norm(g), tel)


@partial(jax.jit, static_argnums=(0,))
def eval_batch(model, params, batch_stats, x, y):
    variables = {"params": params}
    if batch_stats is not None:
        variables["batch_stats"] = batch_stats
    logits = model.apply(variables, x, jnp.ones(x.shape[:2]), None)
    correct = jnp.sum(jnp.argmax(logits, -1) == y)
    ce = jnp.sum(optax.softmax_cross_entropy(
        logits, jax.nn.one_hot(y, RB.D_OUTPUT)))
    return correct, ce, y.shape[0]


def evaluate_split(model, params, batch_stats, X, Y, batch=256):
    correct, ce, seen = 0, 0.0, 0
    for i in range(0, X.shape[0], batch):
        xb = jnp.asarray(onp.asarray(X[i:i + batch]))
        yb = jnp.asarray(Y[i:i + batch])
        c, l, m = eval_batch(model, params, batch_stats, xb, yb)
        correct += int(c); ce += float(l); seen += int(m)
    return dict(accuracy=correct / seen, cross_entropy=ce / seen, n=seen)


# ------------------------------------------------ initialization gate ------
def read_core(arm, params, layer):
    from s5 import substrate_diagnostics as SD
    m = build_single(arm)
    return SD.read_core(m, {"params": params}, layer)


def core_response_change(params_ref, params_arm, layer, arm):
    """Signal-only core impulse and frequency difference, EXCLUDING native D.

        rel = ||K_arm - K_ref||_F / ||K_ref||_F

    over lags 0..IMPULSE_LAGS-1 and a uniform 0..pi grid of N_FREQ points. The
    native D contribution is removed from both so a matched feedthrough cannot
    hide a response difference. A numerically zero reference is NOT dropped: it
    receives the declared ABSOLUTE criterion instead, because an arm with a
    response where the reference has none is not matched either.
    """
    from s5 import substrate_diagnostics as SD
    ca = read_core(REFERENCE_ARM, params_ref, layer)
    cb = read_core(arm, params_arm, layer)
    Ka = SD.impulse_matrices(ca, IMPULSE_LAGS).copy()
    Kb = SD.impulse_matrices(cb, IMPULSE_LAGS).copy()
    Da = onp.diag(onp.asarray(ca["D"])); Db = onp.diag(onp.asarray(cb["D"]))
    Ka[0] -= Da; Kb[0] -= Db
    na = float(onp.sqrt(onp.sum(Ka ** 2)))
    diff = float(onp.sqrt(onp.sum((Kb - Ka) ** 2)))
    _, Ha = SD.frequency_response(ca, N_FREQ)
    _, Hb = SD.frequency_response(cb, N_FREQ)
    Ha = Ha - Da[None]; Hb = Hb - Db[None]
    nfa = float(onp.sqrt(onp.sum(onp.abs(Ha) ** 2)))
    fdiff = float(onp.sqrt(onp.sum(onp.abs(Hb - Ha) ** 2)))
    return dict(layer=layer, impulse_ref_norm=na, impulse_abs_diff=diff,
                impulse_rel=(diff / na if na > 1e-12 else None),
                freq_ref_norm=nfa, freq_abs_diff=fdiff,
                freq_rel=(fdiff / nfa if nfa > 1e-12 else None),
                lags=IMPULSE_LAGS, n_freq=N_FREQ,
                zero_reference=bool(na <= 1e-12))


def evaluate_gate(layers, tol=INIT_RESPONSE_TOL, abs_tol=INIT_RESPONSE_ABS_TOL):
    """Decide the gate. PURE: records in, decision out.

    Enforces BOTH declared criteria. Non-finite values FAIL rather than being
    skipped. Separated from the runner so the decision can be exercised
    directly instead of inferred from the source.
    """
    failures, worst_imp, worst_freq = [], 0.0, 0.0
    for lay in layers:
        for kind, rel, absd in (
                ("impulse", lay["impulse_rel"], lay["impulse_abs_diff"]),
                ("frequency", lay["freq_rel"], lay["freq_abs_diff"])):
            tag = f"L{lay['layer']} {kind}"
            if absd is None or not onp.isfinite(absd):
                failures.append(f"{tag}: non-finite absolute difference")
                continue
            if rel is None:                         # zero reference
                if absd > abs_tol:
                    failures.append(f"{tag}: zero reference but abs "
                                    f"{absd:.3e} > {abs_tol:.0e}")
                continue
            if not onp.isfinite(rel):
                failures.append(f"{tag}: non-finite relative difference")
                continue
            if kind == "impulse":
                worst_imp = max(worst_imp, rel)
            else:
                worst_freq = max(worst_freq, rel)
            if rel > tol:
                failures.append(f"{tag}: {rel:.3e} > {tol:.0e}")
    return dict(worst_impulse_rel=worst_imp, worst_frequency_rel=worst_freq,
                tolerance=tol, abs_tolerance_zero_reference=abs_tol,
                failures=failures, passed=not failures,
                scope=("finite-window impulse over 0..%d lags and a %d-point "
                       "frequency grid, at THIS seed's initialization. Not a "
                       "uniform transfer-function theorem."
                       % (IMPULSE_LAGS - 1, N_FREQ)))


def probe_logit_change(params_ref, bs_ref, arm, params_arm, bs_arm, x):
    """Initial validation-probe logit difference. NO labels are read."""
    m = build(REFERENCE_ARM, False)
    ma = build(arm, False)
    ts = jnp.ones(x.shape[:2])
    va = {"params": params_ref}
    vb = {"params": params_arm}
    if bs_ref is not None:
        va["batch_stats"] = bs_ref
    if bs_arm is not None:
        vb["batch_stats"] = bs_arm
    ya = onp.asarray(m.apply(va, x, ts, None))
    yb = onp.asarray(ma.apply(vb, x, ts, None))
    na = float(onp.sqrt(onp.sum(ya ** 2)))
    d = float(onp.sqrt(onp.sum((yb - ya) ** 2)))
    return dict(ref_norm=na, abs_diff=d,
                rel=(d / na if na > 1e-12 else None),
                note="reported separately; not part of the gate decision")


# ------------------------------------------------------------ reporting ----
def rho_record(params):
    """Raw values, executed rho, derived mass and boundary occupancy."""
    from flax.traverse_util import flatten_dict
    lo, hi = LOG_RHO_BOUNDS
    out = {}
    for k, v in flatten_dict(params).items():
        if k[-1] != RHO_ONLY_PARAM_NAME:
            continue
        raw = onp.asarray(v)
        rho = onp.exp(onp.clip(raw, lo, hi))
        out["/".join(k)] = dict(
            raw=raw.tolist(), rho=rho.tolist(),
            mu=(SYMMETRIC_REFERENCE.T * rho).tolist(),
            median_rho=float(onp.median(rho)), min_rho=float(onp.min(rho)),
            max_rho=float(onp.max(rho)),
            n_at_upper=int(onp.sum(raw >= hi - 1e-12)),
            n_at_lower=int(onp.sum(raw <= lo + 1e-12)),
            n_outside_raw=int(onp.sum((raw > hi) | (raw < lo))),
            n_below_init=int(onp.sum(rho < RHO_INIT_RECALL)),
            n_modes=int(raw.size))
    return out or None


def response_summary(arm, params):
    """Impulse band energies of the executed core, per layer.

    Windowed by construction: the tail beyond the last measured lag is NOT
    bounded here and is reported as unknown. These are response measurements,
    not evidence that a mode learned task-relevant memory.
    """
    from s5 import substrate_diagnostics as SD
    rows = []
    for l in range(N_LAYERS):
        core = read_core(arm, params, l)
        K = SD.impulse_matrices(core, IMPULSE_LAGS).copy()
        K[0] -= onp.diag(onp.asarray(core["D"]))
        be = SD.band_energy(K, IMPULSE_BANDS)
        rows.append(dict(layer=l, **be,
                         window_lags=IMPULSE_LAGS,
                         tail_beyond_window="unknown (not bounded here)"))
    return rows


def state_counts(arm, params):
    m = build_single(arm)
    x = jnp.zeros((SC.N_FRAMES, SC.N_MFCC))
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


def trainable_count(params, arm):
    rep = parameter_report(params)
    rho = sum(int(v.size) for v in response_leaves(params))
    return dict(stored=rep["total"], ssm=rep["ssm"], other=rep["other"],
                response_leaves=rho, trainable=rep["total"],
                note="every stored value is trainable in both arms of this "
                     "batch; the frozen-rho arm is not part of it")


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


def save_params(path, tree):
    from flax import serialization
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(serialization.to_bytes(tree))
    return path


# ------------------------------------------------------------- one arm -----
def run_arm(arm, seed, params, batch_stats, tx, data, steps_per_epoch, epochs,
            out, deadline, reserve_s, status):
    """Train ONE arm for `epochs` epochs, recording every epoch.

    Returns the row, or None if the budget stopped it before it finished. A
    partial arm is recorded as incomplete and is NOT reported as a score.
    """
    (Xtr, Ytr), (Xva, Yva) = data["train"], data["val"]
    train_model, eval_model = build(arm, True), build(arm, False)
    opt_state = tx.init(params)
    rng = jax.random.PRNGKey(seed + 2)      # SAME stream for both arms
    epochs_rec, best = [], dict(accuracy=-1.0, cross_entropy=float("inf"),
                                epoch=-1)
    tel_tot = dict(n_projection_events=0, max_overshoot=0.0,
                   response_grad_norm_sum=0.0, response_update_norm_sum=0.0,
                   steps=0)
    t_arm = time.time()
    for epoch in range(epochs):
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"seed {seed} arm {arm}: stopped at epoch {epoch} (budget)")
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
            tel_tot["n_projection_events"] += int(tel["n_projected"])
            tel_tot["max_overshoot"] = max(tel_tot["max_overshoot"],
                                           float(tel["max_overshoot"]))
            tel_tot["response_grad_norm_sum"] += float(tel["response_grad_norm"])
            tel_tot["response_update_norm_sum"] += \
                float(tel["response_update_norm"])
            tel_tot["steps"] += 1
        val = evaluate_split(eval_model, params, batch_stats, Xva, Yva)
        if val["accuracy"] > best["accuracy"]:
            best = dict(accuracy=val["accuracy"],
                        cross_entropy=val["cross_entropy"], epoch=epoch)
            save_params(os.path.join(out, "params",
                                     f"best_seed{seed}_{arm}.msgpack"),
                        dict(params=params, batch_stats=batch_stats))
        rec = dict(epoch=epoch, train_loss=tot_loss / nb,
                   train_acc=tot_acc / nb, val_accuracy=val["accuracy"],
                   val_cross_entropy=val["cross_entropy"],
                   grad_norm=float(gn), epoch_s=time.time() - t0,
                   peak_memory_bytes=RB.peak_memory_bytes())
        epochs_rec.append(rec)
        print(f"  [seed {seed}] {arm:15s} epoch {epoch:>2}  "
              f"train {rec['train_loss']:.4f}/{rec['train_acc']:.4f}  "
              f"val {val['accuracy']:.4f}  best {best['accuracy']:.4f}  "
              f"{rec['epoch_s']:.1f}s")
        write(os.path.join(out, "status.json"), status)
    save_params(os.path.join(out, "params", f"final_seed{seed}_{arm}.msgpack"),
                dict(params=params, batch_stats=batch_stats))
    n = max(tel_tot["steps"], 1)
    telemetry = dict(
        n_projection_events=tel_tot["n_projection_events"],
        entry_updates=n * sum(int(v.size) for v in response_leaves(params)),
        max_proposed_overshoot=tel_tot["max_overshoot"],
        mean_response_grad_norm=tel_tot["response_grad_norm_sum"] / n,
        mean_response_update_norm=tel_tot["response_update_norm_sum"] / n,
        steps=tel_tot["steps"],
        note=("n_projection_events counts (entry, update) EVENTS, not distinct "
              "modes, distinct steps, or time at the boundary. "
              "max_proposed_overshoot is the PRE-projection excursion the "
              "optimizer proposed and the projection rejected; the forward "
              "clip means it is never an inadmissible executed coefficient."))
    return dict(
        seed=seed, arm=arm,
        primary_endpoint_val_accuracy=epochs_rec[-1]["val_accuracy"],
        primary_endpoint_val_cross_entropy=epochs_rec[-1]["val_cross_entropy"],
        primary_endpoint_epoch=epochs_rec[-1]["epoch"],
        best_val=best, epochs=epochs_rec, wall_s=time.time() - t_arm,
        params=parameter_report(params), trainable=trainable_count(params, arm),
        state_counts=state_counts(arm, params),
        rho_final=rho_record(params), projection_telemetry=telemetry,
        response_bands=response_summary(arm, params),
        peak_memory_bytes=RB.peak_memory_bytes(),
        final_param_digest=tree_digest(params))


# ------------------------------------------------------------- preflight ---
def preflight(data, steps_per_epoch, epochs, seeds, status):
    """Measure the WHOLE batch's cost, not one seed's.

    Compilation is measured once per arm because the models are cached and
    reused across seeds, which is what makes that reuse a claim rather than a
    hope. The projection covers every seed and both arms, plus the per-epoch
    validation pass and the host overhead measured here.
    """
    (Xtr, Ytr), (Xva, Yva) = data["train"], data["val"]
    tx, _ = make_optimizer(steps_per_epoch, epochs)
    rows = []
    idx = onp.sort(next(iter(SC.epoch_batches(Xtr.shape[0], BATCH, 0, 0))))
    xb = jnp.asarray(onp.asarray(Xtr[idx])); yb = jnp.asarray(Ytr[idx])
    for arm in ARMS:
        m, p, bs = init_variables(arm, seeds[0])
        RB.assert_response_policy(p, ARM_SUBSTRATE[arm], N_LAYERS, SSM_SIZE // 2)
        opt = tx.init(p)
        rng = jax.random.PRNGKey(0)
        t0 = time.time()
        q, opt2, bs2, loss, _, _, _ = train_step(m, tx, p, opt, bs, xb, yb, rng)
        loss.block_until_ready()
        compile_s = time.time() - t0
        t1 = time.time()
        for _ in range(3):
            q, opt2, bs2, loss, _, _, _ = train_step(m, tx, q, opt2, bs2, xb,
                                                     yb, rng)
        loss.block_until_ready()
        step_s = (time.time() - t1) / 3
        t2 = time.time()
        evaluate_split(build(arm, False), p, bs, Xva[:512], Yva[:512])
        val_s = (time.time() - t2) * (Xva.shape[0] / 512.0)
        rows.append(dict(arm=arm, compile_s=compile_s, step_s=step_s,
                         epoch_s=step_s * steps_per_epoch,
                         val_pass_s=val_s,
                         per_seed_s=epochs * (step_s * steps_per_epoch + val_s)))
        print(f"[preflight] {arm:15s} compile {compile_s:6.1f}s  "
              f"step {step_s * 1e3:6.2f}ms  epoch {step_s * steps_per_epoch:6.1f}s"
              f"  val {val_s:5.1f}s")
        # the preflight's updates are DISCARDED: the real runs re-initialize
        # parameters, optimizer state and the RNG stream from scratch.
        del q, opt, opt2, bs2
    total = (sum(r["compile_s"] for r in rows)
             + len(seeds) * sum(r["per_seed_s"] for r in rows))
    gate_s = 25.0 * len(seeds)          # measured gate cost, host-side
    total += gate_s
    status["preflight"] = dict(rows=rows, seeds=len(seeds), epochs=epochs,
                               gate_allowance_s=gate_s,
                               projected_total_s=total,
                               scope=("compilation once per arm plus every seed "
                                      "x arm x epoch, the per-epoch validation "
                                      "pass and the per-seed gate"))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cache", required=True)
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/combined")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
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
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'. All "
                         f"training and numerical validation belong on the "
                         f"cluster GPU.")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    # train and val ONLY: the test arrays are never opened, not merely never
    # scored. `load` would open all three.
    data, manifest = SC.load_splits(args.data_cache, splits=("train", "val"))
    steps_per_epoch = data["train"][0].shape[0] // BATCH
    seeds = [int(s) for s in args.seeds.split(",")]

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    tx, opt_desc = make_optimizer(steps_per_epoch, args.epochs)

    status = dict(run_id=run_id, out=out, backend=backend, seeds=seeds,
                  arms=list(ARMS), arm_substrate=ARM_SUBSTRATE,
                  epochs=args.epochs, batch=BATCH,
                  steps_per_epoch=steps_per_epoch, optimizer=opt_desc,
                  config=dict(d_model=D_MODEL, ssm_size=SSM_SIZE,
                              n_layers=N_LAYERS, hippo_blocks=BLOCKS,
                              conj_sym=True, discretization="zoh",
                              lr=LR, lr_final=LR_FINAL,
                              weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
                              label_smoothing=LABEL_SMOOTHING),
                  # the EXECUTED response policy and its derived coefficients,
                  # not the static symmetric-reference table: that table would
                  # mislabel this arm's initialization and its derived mass.
                  response_policy=dict(
                      parameterization="rho-only, gamma_n = 1, T fixed",
                      T=SYMMETRIC_REFERENCE.T,
                      prospective_input_horizon=SYMMETRIC_REFERENCE.T,
                      rho_init=RHO_INIT_RECALL,
                      rho_bounds=[float(onp.exp(LOG_RHO_BOUNDS[0])),
                                  float(onp.exp(LOG_RHO_BOUNDS[1]))],
                      mass="derived, mu = T * rho; never learned separately",
                      learned_leaves=[RHO_ONLY_PARAM_NAME],
                      within_sequence_adaptation=False,
                      note=("coefficients are constant within a sequence and "
                            "are updated between minibatches by BPTT")),
                  init_response_tol=INIT_RESPONSE_TOL,
                  budget_s=args.budget_s, provenance=provenance(),
                  data_manifest=dict(counts=manifest["counts"],
                                     file_list_sha256=manifest["file_list_sha256"],
                                     feature_sha256={
                                         k: v for k, v in
                                         manifest["feature_sha256"].items()
                                         if k in ("train", "val")},
                                     mfcc=manifest["mfcc"],
                                     split_seed=manifest["split_seed"],
                                     splits_opened=["train", "val"]),
                  gates=[], results=[], incomplete=[])
    print(f"[*] out={out} backend={backend} seeds={seeds} "
          f"epochs={args.epochs} steps/epoch={steps_per_epoch}")
    write(os.path.join(out, "status.json"), status)

    proj = preflight(data, steps_per_epoch, args.epochs, seeds, status)
    write(os.path.join(out, "status.json"), status)
    if args.preflight_only:
        print(f"COMBINED_STATUS=PREFLIGHT_ONLY out={out}")
        return 0
    if proj > left():
        status["incomplete"].append(
            f"projected {proj:.0f}s > remaining {left():.0f}s; comparative "
            f"training NOT started. No epoch, seed or arm was reduced.")
        write(os.path.join(out, "status.json"), status)
        print(f"[!] {status['incomplete'][-1]}")
        print(f"COMBINED_STATUS=INCOMPLETE out={out}")
        return 3

    rows = []
    for seed in seeds:
        if left() < 60:
            status["incomplete"].append(f"seed {seed} not started (budget)")
            break
        # ---- paired initialization: ONE common tree, both arms
        _, p_ref, bs_ref = init_variables(REFERENCE_ARM, seed)
        _, p_cmb, bs_cmb = init_variables(ARMS[1], seed)
        p_cmb, copied, kept = clone_common(p_ref, p_cmb)
        for arm, p in ((REFERENCE_ARM, p_ref), (ARMS[1], p_cmb)):
            RB.assert_response_policy(p, ARM_SUBSTRATE[arm], N_LAYERS,
                                      SSM_SIZE // 2)
        common = dict(
            common_param_digest_ref=tree_digest(p_ref),
            common_param_digest_combined=tree_digest(
                p_cmb, skip=(RHO_ONLY_PARAM_NAME,)),
            batch_stats_digest_ref=tree_digest(bs_ref),
            batch_stats_digest_combined=tree_digest(bs_cmb),
            n_cloned=len(copied), n_arm_specific=len(kept),
            arm_specific=kept)
        common["common_params_identical"] = (
            common["common_param_digest_ref"]
            == common["common_param_digest_combined"])
        common["batch_stats_identical"] = (
            common["batch_stats_digest_ref"]
            == common["batch_stats_digest_combined"])
        if not (common["common_params_identical"]
                and common["batch_stats_identical"]):
            status["incomplete"].append(
                f"seed {seed}: paired initialization is NOT identical "
                f"({common}); refusing to report a paired comparison")
            write(os.path.join(out, "status.json"), status)
            print(f"COMBINED_STATUS=FAILED out={out}")
            return 4

        # ---- initialization gate, before THIS seed's training
        Xva = data["val"][0]
        xprobe = jnp.asarray(onp.asarray(Xva[:32]))
        layers = [core_response_change(p_ref, p_cmb, l, ARMS[1])
                  for l in range(N_LAYERS)]
        gate = dict(seed=seed, arm=ARMS[1], layers=layers, common=common,
                    probe_logits=probe_logit_change(p_ref, bs_ref, ARMS[1],
                                                    p_cmb, bs_cmb, xprobe))
        gate.update(evaluate_gate(layers))
        status["gates"].append(gate)
        write(os.path.join(out, "status.json"), status)
        print(f"[gate] seed {seed}: worst impulse "
              f"{gate['worst_impulse_rel']:.4e}, worst frequency "
              f"{gate['worst_frequency_rel']:.4e}, tol "
              f"{INIT_RESPONSE_TOL:.0e} -> "
              f"{'PASS' if gate['passed'] else 'FAIL'}")
        if not gate["passed"]:
            status["stopped_by_gate"] = True
            status["incomplete"].append(
                f"seed {seed}: initialization gate FAILED {gate['failures']}")
            write(os.path.join(out, "status.json"), status)
            print(f"[STOP] seed {seed} gate FAILED: {gate['failures']}. "
                  f"Stopping this route and reporting, WITHOUT adjusting rho "
                  f"and without dropping the seed.")
            print(f"COMBINED_STATUS=GATE_STOP out={out}")
            return 3

        for arm, p, bs in ((REFERENCE_ARM, p_ref, bs_ref),
                           (ARMS[1], p_cmb, bs_cmb)):
            row = run_arm(arm, seed, p, bs, tx, data, steps_per_epoch,
                          args.epochs, out, deadline, args.reserve_s, status)
            if row is None:
                write(os.path.join(out, "status.json"), status)
                continue
            row["initial_param_digest"] = tree_digest(p)
            rows.append(row)
            status["results"] = rows
            write(os.path.join(out, "results.json"), rows)
            write(os.path.join(out, "status.json"), status)

    # ---- paired differences, combined minus reference, per seed
    by = {(r["seed"], r["arm"]): r for r in rows}
    diffs = []
    for seed in seeds:
        a = by.get((seed, REFERENCE_ARM)); b = by.get((seed, ARMS[1]))
        if a is None or b is None:
            continue
        diffs.append(dict(
            seed=seed,
            endpoint_pp=100.0 * (b["primary_endpoint_val_accuracy"]
                                 - a["primary_endpoint_val_accuracy"]),
            endpoint_ce=(b["primary_endpoint_val_cross_entropy"]
                         - a["primary_endpoint_val_cross_entropy"]),
            best_pp=100.0 * (b["best_val"]["accuracy"]
                             - a["best_val"]["accuracy"])))
    status["paired_differences"] = dict(
        primary="validation accuracy after the fixed tenth epoch, "
                "combined minus Rawat, per seed",
        per_seed=diffs,
        mean_endpoint_pp=(float(onp.mean([d["endpoint_pp"] for d in diffs]))
                          if diffs else None),
        mean_best_pp=(float(onp.mean([d["best_pp"] for d in diffs]))
                      if diffs else None),
        note=("best-checkpoint scores are reported SEPARATELY and do not "
              "replace the predeclared fixed-epoch endpoint"))
    status["wall_s"] = time.time() - t0
    status["complete"] = (len(rows) == len(seeds) * len(ARMS)
                          and not status["incomplete"])
    write(os.path.join(out, "status.json"), status)
    for d in diffs:
        print(f"[paired] seed {d['seed']}: endpoint "
              f"{d['endpoint_pp']:+.3f} pp   best {d['best_pp']:+.3f} pp")
    if diffs:
        print(f"[paired] mean endpoint "
              f"{status['paired_differences']['mean_endpoint_pp']:+.3f} pp")
    print(f"[*] wall {status['wall_s']:.0f}s  rows {len(rows)}/"
          f"{len(seeds) * len(ARMS)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[INCOMPLETE] {m}")
        print(f"COMBINED_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"COMBINED_STATUS=COMPLETE out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
