"""Paired continuation screen: Rawat, learned input horizon, stable generalized.

Authoritative brief: NEXT_PROSPECTIVE_S5_CODING_BRIEF_2026_09_16.md.
Protocol, frozen before execution: docs/STABLE_GP_CONTINUATION_PROTOCOL.md.

Three arms continue from ONE saved, validation-selected Rawat checkpoint:

    A  rawat_unchanged      alpha_p_s5, exactly as saved
    B  rawat_learned_input  the same, with a per-mode input horizon
                            T_in = 5 exp(q), q = 0 at the start
    C  sgp_learned_input    the SAME learned input horizon PLUS the stable
                            generalized recurrence, rho = exp(r) and
                            T = 10 exp(t), r = t = 0 at the start
                            (recurrent reference 10, review R0)

This is a WEIGHT WARM-START, not an exact resume: optimizer state is reset in
every arm and a new ten-epoch cosine schedule starts. Seeds 201/202/203 are
paired continuation STREAMS (data order and dropout) from one source model, not
three independently trained initial models.

One process, one deadline, GPU only. The test split is never opened.
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
from s5 import stable_gp as SG                                     # noqa: E402
from s5 import substrate_diagnostics as SD                         # noqa: E402
from s5.checkpointing import provenance                            # noqa: E402
from s5.physical_coefficients import SYMMETRIC_REFERENCE           # noqa: E402
from s5.rawat_model import RawatClassifier, parameter_report       # noqa: E402
from s5.rawat_s5 import STABLE_GP_RESPONSES, init_substrate_ssm    # noqa: E402
import experiments.gp.rawat_benchmark as RB                        # noqa: E402

# ---- declared configuration: the Stage 2 architecture and common policy.
#      The saved source config is re-read and a discrepancy REFUSES the run.
D_MODEL, SSM_SIZE, N_LAYERS = 32, 32, 4
BATCH = 32
EPOCHS = 10
LR, LR_FINAL = 1e-3, 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
SEEDS = (201, 202, 203)
SOURCE_ARM = "alpha_p_s5"
#: Stage 2's declared candidate order for its learning-rate slots
SOURCE_LR_ORDER = (1e-3, 3e-4)

ARMS = ("A_rawat", "B_learned_input", "C_stable_generalized")
ARM_SUBSTRATE = {"A_rawat": "alpha_p_s5",
                 "B_learned_input": "rawat_learned_input",
                 "C_stable_generalized": "sgp_learned_input"}
ARM_ADDED_LEAVES = {"A_rawat": (),
                    "B_learned_input": STABLE_GP_RESPONSES["rawat_learned_input"],
                    "C_stable_generalized":
                        STABLE_GP_RESPONSES["sgp_learned_input"]}

# ---- DECLARED TOLERANCES for the production-dtype identity gate, frozen
#      before execution. The EXACT identities are established separately in
#      float64 (tests/test_stable_gp.py). In production float32 the B-vs-C
#      comparison is a closed-form diagonal exponential against a Pade 4x4
#      block exponential; the repository's declared float32 coefficient
#      resolution for that block is F32 = 2e-4 (timescale probe), and these
#      gates allow for its propagation through four layers.
SOURCE_CE_TOL = 1e-5            # restored vs saved validation CE, as Stage 2
LOGIT_REL_TOL = 5e-4            # relative Frobenius, validation probe
GRAD_REL_TOL = 2e-3             # shared-leaf and input gradients, per leaf
#: AMENDED before execution (review R2): the ABSOLUTE branch for near-zero
#: leaves. A leaf passes iff its values are finite and
#:     ||g_x - g_ref||_F <= max(GRAD_REL_TOL ||g_ref||_F,
#:                              GRAD_ABS_EPS_FACTOR * eps(dtype) * G_ref)
#: with G_ref the global norm of the reference gradient over ALL compared
#: leaves. In float32 the absolute floor is 1.2e-4 G_ref; in float64 it is
#: 2.2e-13 G_ref. Needed because training-mode batch normalization removes a
#: constant shift of the encoder Dense bias, so that leaf's exact gradient is
#: zero while floating reductions leave residuals, and a purely relative
#: criterion is undefined there. No leaf is dropped: both errors, magnitudes
#: and finiteness are logged. Not tuned against any checkpoint.
GRAD_ABS_EPS_FACTOR = 1e3
#: optimizer ROUTING identity (review R2): the SAME shared gradient values are
#: copied into both trees, genuinely extra leaves frozen, and the first
#: updates compared entrywise as max |u_x - u_ref| / lr_0.
UPDATE_MAX_TOL = 2e-3
#: coordinates with |g| below this are reported as near-zero in the separate,
#: NON-gating comparison of independently computed first updates
NEAR_ZERO_GRAD = 1e-7
PROBE_N = 256
#: preflight: measured host-path training steps after warm-up
PREFLIGHT_TIMED_STEPS = 40
PREFLIGHT_VAL_BATCHES = 32
# ---- declared performance screen
SCREEN_MIN_GAIN_PP = 0.3
IMPULSE_LAGS = 128

_MODELS = {}


# ------------------------------------------------------------- utilities ---
def build(arm, training):
    key = (arm, training)
    if key not in _MODELS:
        _MODELS[key] = RB.build_model(ARM_SUBSTRATE[arm], D_MODEL, SSM_SIZE,
                                      N_LAYERS, training)
    return _MODELS[key]


def build_single(arm):
    return RawatClassifier(
        ssm=init_substrate_ssm(ARM_SUBSTRATE[arm],
                               physical=SYMMETRIC_REFERENCE,
                               **RB.ssm_kwargs(D_MODEL, SSM_SIZE)),
        d_model=D_MODEL, n_layers=N_LAYERS, d_output=RB.D_OUTPUT,
        training=False)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


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
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)
    os.replace(tmp, path)


def save_tree(path, tree):
    from flax import serialization
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(serialization.to_bytes(tree))


def leaf_name(path):
    return path[-1].key if hasattr(path[-1], "key") else str(path[-1])


# ------------------------------------------------------ source checkpoint --
def select_source(stage2_dir):
    """Apply Stage 2's ORIGINAL rule to its alpha_p_s5 runs, before training.

    Within a run the saved `best` checkpoint is the first strict maximum of
    validation accuracy. Across the run's learning-rate slots: highest
    validation accuracy, then lower validation cross entropy, then Stage 2's
    declared candidate order (1e-3 before 3e-4). No new score is used.
    """
    manifest = os.path.join(stage2_dir, "manifest.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit(f"BLOCKER: Stage 2 manifest not found at {manifest}")
    cands = []
    with open(manifest) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("arm") != SOURCE_ARM or rec.get("exit") != 0:
                continue
            rd = rec["run_dir"]
            met = os.path.join(rd, "metrics.jsonl")
            if not (os.path.exists(met)
                    and os.path.exists(os.path.join(rd, "best.msgpack"))):
                continue
            with open(met) as mf:
                ms = [json.loads(l) for l in mf if l.strip()]
            ms = [m for m in ms if m.get("mode") == "train"]
            if not ms:
                continue
            best = None
            for m in ms:
                if best is None or m["val_accuracy"] > best["val_accuracy"]:
                    best = m
            cands.append(dict(run_dir=rd, lr=float(rec["lr"]),
                              seed=rec.get("seed"), best=best,
                              epochs=len(ms)))
    if not cands:
        raise SystemExit(f"BLOCKER: no completed {SOURCE_ARM} run with a best "
                         f"checkpoint under {stage2_dir}")

    def order(c):
        lr_rank = (SOURCE_LR_ORDER.index(c["lr"]) if c["lr"] in SOURCE_LR_ORDER
                   else len(SOURCE_LR_ORDER))
        return (-c["best"]["val_accuracy"], c["best"]["val_cross_entropy"],
                lr_rank)
    ranked = sorted(cands, key=order)
    chosen = ranked[0]
    rd = chosen["run_dir"]
    hashes = {n: sha256_file(os.path.join(rd, n))
              for n in ("best.msgpack", "best.meta.json", "config.json",
                        "metrics.jsonl") if os.path.exists(os.path.join(rd, n))}
    with open(os.path.join(rd, "config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(rd, "best.meta.json")) as fh:
        meta = json.load(fh)
    return dict(
        rule=("within run: first strict maximum of validation accuracy "
              "(the saved best checkpoint); across alpha_p_s5 slots: validation "
              "accuracy, then validation CE, then Stage 2 candidate order "
              "1e-3 before 3e-4"),
        candidates=[dict(run_dir=c["run_dir"], lr=c["lr"], seed=c["seed"],
                         best_epoch=c["best"]["epoch"],
                         val_accuracy=c["best"]["val_accuracy"],
                         val_cross_entropy=c["best"]["val_cross_entropy"])
                    for c in ranked],
        chosen=dict(run_dir=rd, lr=chosen["lr"], seed=chosen["seed"],
                    best_epoch=chosen["best"]["epoch"],
                    saved_val_accuracy=chosen["best"]["val_accuracy"],
                    saved_val_cross_entropy=chosen["best"]["val_cross_entropy"],
                    meta_epoch=meta.get("epoch"),
                    source_commit=(cfg.get("provenance") or {}).get("commit"),
                    file_sha256=hashes),
        config=cfg)


def check_source_config(cfg):
    """REFUSE rather than reconcile a discrepancy with the declared screen."""
    c = cfg.get("config", {})
    want = dict(arm=SOURCE_ARM, d_model=D_MODEL, ssm_size=SSM_SIZE,
                n_layers=N_LAYERS, batch=BATCH, weight_decay=WEIGHT_DECAY,
                grad_clip=GRAD_CLIP, label_smoothing=LABEL_SMOOTHING,
                ssm_weight_decay=None, lr_final=LR_FINAL)
    bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
    if bad:
        raise SystemExit(f"BLOCKER: source checkpoint config differs from the "
                         f"declared screen: {bad}")
    return want


def restore_source(chosen, cfg, data):
    """Restore params AND batch statistics; reproduce the saved metrics."""
    from experiments.gp.stage2_diagnostic import restore_readonly
    c = dict(cfg["config"])
    c["_steps_per_epoch"] = (cfg.get("extra") or {}).get("steps_per_epoch",
                                                          843)
    L = restore_readonly(chosen["run_dir"], c, "best")
    Xva, Yva = data["val"]
    res = RB.evaluate_split(L["params"], L["batch_stats"],
                            build("A_rawat", False), Xva, Yva, BATCH)
    n = res["n"]
    saved_correct = int(round(chosen["saved_val_accuracy"] * n))
    got_correct = int(round(res["accuracy"] * n))
    ce_diff = abs(res["cross_entropy"] - chosen["saved_val_cross_entropy"])
    rec = dict(n=n, saved_correct=saved_correct, restored_correct=got_correct,
               counts_match=bool(saved_correct == got_correct),
               saved_ce=chosen["saved_val_cross_entropy"],
               restored_ce=res["cross_entropy"], ce_abs_diff=ce_diff,
               ce_tol=SOURCE_CE_TOL, ce_ok=bool(ce_diff <= SOURCE_CE_TOL),
               checkpoint_epoch=L["meta"].get("epoch"), dtypes=L["casts"])
    rec["passed"] = rec["counts_match"] and rec["ce_ok"]
    return L, rec, res


# ----------------------------------------------------------- warm start ----
def warm_start(arm, src_params, src_bs):
    """Clone every common leaf from the source; whitelist the added leaves.

    Only the arm's DECLARED added leaves may be missing from the source, they
    must have shape (P,) and be exactly zero, and nothing may be missing in the
    other direction. Anything else refuses the run.
    """
    from flax.traverse_util import flatten_dict, unflatten_dict
    m = build(arm, True)
    v = m.init({"params": jax.random.PRNGKey(0),
                "dropout": jax.random.PRNGKey(1)},
               jnp.zeros((2, SC.N_FRAMES, SC.N_MFCC)),
               jnp.ones((2, SC.N_FRAMES)), None)
    tmpl, src = flatten_dict(v["params"]), flatten_dict(src_params)
    allowed = set(ARM_ADDED_LEAVES[arm])
    out, added = {}, []
    for k, val in tmpl.items():
        if k in src:
            if src[k].shape != val.shape or src[k].dtype != val.dtype:
                raise SystemExit(f"BLOCKER: {arm} leaf {'/'.join(k)} "
                                 f"{val.shape}/{val.dtype} vs source "
                                 f"{src[k].shape}/{src[k].dtype}")
            out[k] = src[k]
        else:
            if k[-1] not in allowed:
                raise SystemExit(f"BLOCKER: {arm} has undeclared leaf "
                                 f"{'/'.join(k)} absent from the source")
            if val.shape != (SSM_SIZE // 2,) or float(
                    onp.max(onp.abs(onp.asarray(val)))) != 0.0:
                raise SystemExit(f"BLOCKER: added leaf {'/'.join(k)} is not a "
                                 f"zero ({SSM_SIZE // 2},) vector")
            out[k] = val
            added.append("/".join(k))
    missing = [k for k in src if k not in tmpl]
    if missing:
        raise SystemExit(f"BLOCKER: source leaves not in {arm}: {missing}")
    if len(added) != N_LAYERS * len(allowed):
        raise SystemExit(f"BLOCKER: {arm} added {len(added)} leaves, expected "
                         f"{N_LAYERS * len(allowed)}")
    bs = jax.tree_util.tree_map(lambda x: x, src_bs)
    return unflatten_dict(out), bs, added


# ------------------------------------------------------------- optimizer ---
def label_tree(params):
    return jax.tree_util.tree_map_with_path(
        lambda p, _: ("response" if leaf_name(p) in SG.ADDED_LEAVES
                      else "common"), params)


def make_optimizer(steps_per_epoch, epochs):
    total = max(1, steps_per_epoch * epochs)
    sched = optax.cosine_decay_schedule(init_value=LR, decay_steps=total,
                                        alpha=LR_FINAL / LR)
    tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.multi_transform(
            {"common": optax.adamw(sched, weight_decay=WEIGHT_DECAY),
             "response": optax.adam(sched)},
            label_tree))
    desc = dict(
        labels=dict(common="AdamW(b1=0.9, b2=0.999, eps=1e-8, weight_decay="
                           f"{WEIGHT_DECAY}) on every inherited parameter",
                    response="Adam(b1=0.9, b2=0.999, eps=1e-8), NO weight "
                             "decay, on log_T_in / log_rho_rec / log_T_rec"),
        schedule=f"cosine {LR} -> {LR_FINAL} over {total} steps, both groups",
        clipping=f"global norm {GRAD_CLIP} over ALL gradients, before the "
                 f"groups",
        label_smoothing=LABEL_SMOOTHING,
        optimizer_state="RESET in every arm: weight warm-start, not resume",
        total_steps=total)
    return tx, desc


def loss_fn(params, batch_stats, model, x, y, rng):
    variables = {"params": params, "batch_stats": batch_stats}
    logits, updates = model.apply(variables, x, jnp.ones(x.shape[:2]), None,
                                  rngs={"dropout": rng},
                                  mutable=["batch_stats"])
    loss = optax.softmax_cross_entropy(
        logits, optax.smooth_labels(jax.nn.one_hot(y, RB.D_OUTPUT),
                                    LABEL_SMOOTHING)).mean()
    return loss, (logits, updates["batch_stats"])


def _added_norm(tree):
    vs = [v for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]
          if leaf_name(p) in SG.ADDED_LEAVES]
    if not vs:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(v.astype(jnp.float32) ** 2) for v in vs))


@partial(jax.jit, static_argnums=(0, 1))
def train_step(model, tx, params, opt_state, batch_stats, x, y, rng):
    (loss, (logits, new_bs)), g = jax.value_and_grad(
        loss_fn, has_aux=True)(params, batch_stats, model, x, y, rng)
    upd, opt_state = tx.update(g, opt_state, params)
    proposed = optax.apply_updates(params, upd)
    # post-update projection of r onto the stable interior, from the UPDATED
    # complete layer; optimizer state untouched; a no-op for A and B
    params, tel = SG.project_stable_domain(proposed)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return (params, opt_state, new_bs, loss, acc, optax.global_norm(g), tel,
            _added_norm(g), _added_norm(upd))


@partial(jax.jit, static_argnums=(0,))
def grads_at(model, params, batch_stats, x, y, rng):
    """Loss, logits, parameter gradients AND input gradients, training mode."""
    (loss, (logits, _)), (g, gx) = jax.value_and_grad(
        loss_fn, argnums=(0, 3), has_aux=True)(params, batch_stats, model, x,
                                                y, rng)
    return loss, logits, g, gx


@partial(jax.jit, static_argnums=(0,))
def logits_eval(model, params, batch_stats, x):
    return model.apply({"params": params, "batch_stats": batch_stats}, x,
                       jnp.ones(x.shape[:2]), None)


# ---------------------------------------------------- identity gate --------
def _flat(tree):
    from flax.traverse_util import flatten_dict
    return flatten_dict(tree) if isinstance(tree, dict) else {("x",): tree}


def _rel(a, b):
    a = onp.asarray(a, dtype=onp.float64); b = onp.asarray(b, dtype=onp.float64)
    d = float(onp.sqrt(onp.sum((a - b) ** 2)))
    n = float(onp.sqrt(onp.sum(b ** 2)))
    return d / n if n > 0 else (0.0 if d == 0 else float("inf"))


def compare_shared(gx, gref, dtype, rel_tol=GRAD_REL_TOL,
                   abs_factor=GRAD_ABS_EPS_FACTOR):
    """Mixed absolute/relative identity over the PAIR-SPECIFIC shared leaves.

    The shared set is the intersection of the two trees, so C vs B includes
    the shared input horizon q, and B vs A includes every inherited leaf.
    Accepts trees or a single array (input gradients). Every leaf is recorded:
    absolute and relative error, magnitude, finiteness, and which branch
    decided it. Nothing is dropped.
    """
    fx, fr = _flat(gx), _flat(gref)
    keys = sorted(set(fx) & set(fr))
    G = float(onp.sqrt(sum(float(onp.sum(onp.asarray(fr[k], onp.float64) ** 2))
                           for k in keys)))
    abs_tol = abs_factor * float(onp.finfo(onp.dtype(dtype)).eps) * G
    rows, ok = {}, True
    for k in keys:
        x = onp.asarray(fx[k], onp.float64); r = onp.asarray(fr[k], onp.float64)
        finite = bool(onp.all(onp.isfinite(x)) and onp.all(onp.isfinite(r)))
        d = float(onp.sqrt(onp.sum((x - r) ** 2))) if finite else float("inf")
        n = float(onp.sqrt(onp.sum(r ** 2))) if finite else float("nan")
        rel = d / n if (finite and n > 0) else None
        passed = bool(finite and d <= max(rel_tol * n, abs_tol))
        branch = ("relative" if (finite and rel_tol * n >= abs_tol)
                  else "absolute")
        ok = ok and passed
        rows["/".join(map(str, k))] = dict(abs_err=d, rel_err=rel,
                                           ref_norm=n, finite=finite,
                                           branch=branch, passed=passed)
    worst = max(rows, key=lambda n: (not rows[n]["passed"],
                                     rows[n]["abs_err"])) if rows else None
    return dict(passed=ok, n_leaves=len(rows), reference_global_norm=G,
                abs_tol=abs_tol, rel_tol=rel_tol, worst_leaf=worst,
                leaves=rows)


def compare_updates(ua, ub):
    fa, fb = _flat(ua), _flat(ub)
    keys = sorted(set(fa) & set(fb))
    rows = {"/".join(k): float(onp.max(onp.abs(
        onp.asarray(fa[k], dtype=onp.float64)
        - onp.asarray(fb[k], dtype=onp.float64)))) / LR for k in keys}
    return dict(worst=max(rows.values()) if rows else 0.0,
                worst_leaf=max(rows, key=rows.get) if rows else None,
                n=len(rows))


def routed_gradients(params_x, g_ref):
    """Copy the reference's gradient VALUES into x's tree on shared leaves;
    genuinely extra leaves of x get zero (frozen)."""
    from flax.traverse_util import flatten_dict, unflatten_dict
    fr = flatten_dict(g_ref)
    return unflatten_dict({k: (fr[k] if k in fr else jnp.zeros_like(v))
                           for k, v in flatten_dict(params_x).items()})


def first_update(tx, params, g):
    st = tx.init(params)
    upd, _ = tx.update(g, st, params)
    return upd


def routing_identity(tx, p_x, p_ref, g_ref):
    """Optimizer-routing identity (review R2): SAME gradient values, so any
    difference is routing, grouping or decay - not gradient rounding."""
    u_ref = first_update(tx, p_ref, routed_gradients(p_ref, g_ref))
    u_x = first_update(tx, p_x, routed_gradients(p_x, g_ref))
    c = compare_updates(u_x, u_ref)
    c["passed"] = bool(c["worst"] <= UPDATE_MAX_TOL)
    return c


def independent_update_report(tx, model, p_x, bs_x, g_x, p_ref, bs_ref, g_ref,
                              model_ref, xprobe):
    """NOT a gate: actual first updates from independently computed gradients,
    their near-zero coordinates, and the post-update functional discrepancy."""
    u_x = first_update(tx, p_x, g_x)
    u_ref = first_update(tx, p_ref, g_ref)
    c = compare_updates(u_x, u_ref)
    fr = _flat(g_ref)
    near = sum(int(onp.sum(onp.abs(onp.asarray(v)) < NEAR_ZERO_GRAD))
               for v in fr.values())
    q_x = optax.apply_updates(p_x, u_x)
    q_ref = optax.apply_updates(p_ref, u_ref)
    post = _rel(logits_eval(model, q_x, bs_x, xprobe),
                logits_eval(model_ref, q_ref, bs_ref, xprobe))
    return dict(max_update_diff_over_lr=c["worst"], worst_leaf=c["worst_leaf"],
                n_near_zero_reference_coordinates=near,
                near_zero_threshold=NEAR_ZERO_GRAD,
                post_update_logits_rel=post,
                note=("reported only; a finite-precision forward identity does "
                      "not imply identical training trajectories"))


def identity_gate(start, data, tx):
    """Initial predictions, shared gradients (parameters and input), optimizer
    routing, and C's initial executed domain - BEFORE any arm trains. Any gated
    disagreement stops the screen."""
    Xtr, Ytr = data["train"]
    Xva, _ = data["val"]
    xp = jnp.asarray(onp.asarray(Xva[:PROBE_N]))
    idx = onp.sort(next(iter(SC.epoch_batches(Xtr.shape[0], BATCH,
                                              SEEDS[0], 0))))
    xb = jnp.asarray(onp.asarray(Xtr[idx])); yb = jnp.asarray(Ytr[idx])
    rng = jax.random.PRNGKey(SEEDS[0])
    out = dict(probe_n=PROBE_N, tolerances=dict(
        logits_rel=LOGIT_REL_TOL, grads_rel=GRAD_REL_TOL,
        grads_abs_eps_factor=GRAD_ABS_EPS_FACTOR,
        routing_update_max_over_lr=UPDATE_MAX_TOL))
    logits, grads, xgrads = {}, {}, {}
    for arm in ARMS:
        p, bs = start[arm]
        logits[arm] = onp.asarray(logits_eval(build(arm, False), p, bs, xp))
        _, _, g, gx = grads_at(build(arm, True), p, bs, xb, yb, rng)
        grads[arm], xgrads[arm] = g, gx
    pairs = (("B_learned_input", "A_rawat"),
             ("C_stable_generalized", "B_learned_input"))
    ok = True
    dtype = onp.float32
    for x, ref in pairs:
        lr = _rel(logits[x], logits[ref])
        agree = float(onp.mean(onp.argmax(logits[x], -1)
                               == onp.argmax(logits[ref], -1)))
        gp = compare_shared(grads[x], grads[ref], dtype)
        gi = compare_shared(xgrads[x], xgrads[ref], dtype)
        route = routing_identity(tx, start[x][0], start[ref][0], grads[ref])
        indep = independent_update_report(
            tx, build(x, False), start[x][0], start[x][1], grads[x],
            start[ref][0], start[ref][1], grads[ref], build(ref, False), xp)
        row = dict(logits_rel=lr, argmax_agreement=agree,
                   param_grads=gp, input_grads=gi, routing=route,
                   independent_first_update=indep)
        row["passed"] = bool(lr <= LOGIT_REL_TOL and gp["passed"]
                             and gi["passed"] and route["passed"])
        ok = ok and row["passed"]
        out[f"{x}_vs_{ref}"] = row
    # added-leaf gradients at the start, for the record: q is active in B and
    # C; r is generally nonzero at T != T_in; t's task gradient vanishes at
    # rho = 1
    added = {}
    for arm in ARMS[1:]:
        for k, v in _flat(grads[arm]).items():
            if k[-1] in SG.ADDED_LEAVES:
                added.setdefault(arm, {}).setdefault(k[-1], []).append(
                    float(onp.sqrt(onp.sum(onp.asarray(v) ** 2))))
    out["added_leaf_grad_norms_per_layer"] = added
    out["C_initial_domain"] = SG.executed_domain_report(
        start["C_stable_generalized"][0])
    out["B_initial_domain"] = SG.executed_domain_report(
        start["B_learned_input"][0])
    ok = (ok and out["C_initial_domain"]["passed"]
          and out["B_initial_domain"]["passed"])
    out["passed"] = bool(ok)
    return out


# ------------------------------------------------------- diagnostics -------
def response_profile(arm, params):
    """Per-layer current tap ||K_0 - diag D|| and history ||K_1..K_127||."""
    rows = []
    m = build_single(arm)
    for l in range(N_LAYERS):
        core = SD.read_core(m, {"params": params}, l)
        K = SD.impulse_matrices(core, IMPULSE_LAGS).copy()
        K[0] -= onp.diag(onp.asarray(core["D"]))
        rows.append(dict(layer=l,
                         current_tap=float(onp.sqrt(onp.sum(K[0] ** 2))),
                         history=float(onp.sqrt(onp.sum(K[1:] ** 2))),
                         K=K))
    return rows


def profile_change(start_rows, end_rows):
    out = []
    for s, e in zip(start_rows, end_rows):
        out.append(dict(
            layer=s["layer"],
            current_tap_start=s["current_tap"], current_tap_end=e["current_tap"],
            history_start=s["history"], history_end=e["history"],
            current_tap_rel_change=_rel(e["K"][0], s["K"][0]),
            history_rel_change=_rel(e["K"][1:], s["K"][1:]),
            window_lags=IMPULSE_LAGS,
            tail_beyond_window="not bounded here"))
    return out


def added_leaf_record(params):
    """Executed-dtype T_in, rho, T, M per layer (review R4), with summaries."""
    rec = SG.executed_coefficients(params)
    for layer, v in rec.items():
        v["summary"] = {k: dict(min=float(onp.min(v[k])),
                                median=float(onp.median(v[k])),
                                max=float(onp.max(v[k])))
                        for k in ("T_in", "rho", "T", "M") if k in v}
    return rec


def tree_finite(tree):
    return bool(all(onp.all(onp.isfinite(onp.asarray(v)))
                    for v in jax.tree_util.tree_leaves(tree)
                    if onp.issubdtype(onp.asarray(v).dtype, onp.inexact)))


def state_counts(arm, params):
    m = build_single(arm)
    per = m.apply({"params": params},
                  method=lambda mm: mm.encoder.layers[0].seq.state_counts())
    return {k: (v if isinstance(v, (bool, str)) or v is None
                else int(v) * N_LAYERS) for k, v in per.items()}


# ------------------------------------------------- the shared step loop ---
def step_loop(arm, tx, state, Xtr, Ytr, seed, epoch_index, tel,
              max_steps=None):
    """The ONE host path used by training AND by preflight timing (review R3):
    host slicing, device transfer, RNG split, the jitted step with projection,
    and synchronization of EVERY scalar the run records."""
    params, opt_state, bs, rng = state
    model = build(arm, True)
    tl = ta = nb = 0.0
    gn = 0.0
    for i, idx in enumerate(SC.epoch_batches(Xtr.shape[0], BATCH, seed,
                                             epoch_index)):
        if max_steps is not None and i >= max_steps:
            break
        sel = onp.sort(idx)
        xb = jnp.asarray(onp.asarray(Xtr[sel])); yb = jnp.asarray(Ytr[sel])
        rng, sub = jax.random.split(rng)
        (params, opt_state, bs, loss, acc, gnorm, t, agn,
         aun) = train_step(model, tx, params, opt_state, bs, xb, yb, sub)
        tl += float(loss); ta += float(acc); nb += 1
        gn = float(gnorm)
        tel["events"] += int(t["n_projected"])
        tel["max_overshoot"] = max(tel["max_overshoot"],
                                   float(t["max_overshoot"]))
        tel["min_log_margin"] = min(tel["min_log_margin"],
                                    float(t["min_log_margin"]))
        tel["added_grad_norm_sum"] += float(agn)
        tel["added_update_norm_sum"] += float(aun)
        tel["steps"] += 1
    return (params, opt_state, bs, rng), dict(
        train_loss=tl / max(nb, 1), train_acc=ta / max(nb, 1),
        last_grad_norm=gn, steps=int(nb))


def new_telemetry():
    return dict(events=0, max_overshoot=0.0, min_log_margin=float("inf"),
                added_grad_norm_sum=0.0, added_update_norm_sum=0.0, steps=0)


def epoch_acceptance(arm, params, opt_state, bs, rec):
    """Finiteness of parameters, optimizer state, normalization state and
    every reported scalar, plus the executed-domain report for B and C."""
    scal = [rec["train_loss"], rec["train_acc"], rec["last_grad_norm"],
            rec["val_accuracy"], rec["val_cross_entropy"]]
    checks = dict(params_finite=tree_finite(params),
                  optimizer_state_finite=tree_finite(opt_state),
                  batch_stats_finite=tree_finite(bs),
                  scalars_finite=bool(onp.all(onp.isfinite(scal))))
    dom = (SG.executed_domain_report(params) if arm != "A_rawat" else None)
    checks["domain_passed"] = True if dom is None else dom["passed"]
    return all(checks.values()), checks, dom


# --------------------------------------------------------------- one run ---
def run_one(arm, seed, start, tx, data, steps_per_epoch, out, deadline,
            reserve_s, status, epoch0):
    (Xtr, Ytr), (Xva, Yva) = data["train"], data["val"]
    eval_model = build(arm, False)
    params, bs = start[arm]
    state = (params, tx.init(params), bs,
             jax.random.PRNGKey(seed))          # SAME dropout stream per arm
    epochs = [dict(epoch=0, **epoch0[arm], note="shared start, measured once")]
    tel = new_telemetry()
    partial_path = os.path.join(out, "runs", f"seed{seed}_{arm}.json")
    t_run = time.time()

    def persist(extra=None):
        write(partial_path, dict(arm=arm, seed=seed, epochs=epochs,
                                 projection_telemetry=dict(tel),
                                 elapsed_s=time.time() - t_run,
                                 **(extra or {})))

    for epoch in range(1, EPOCHS + 1):
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{arm} seed {seed}: stopped before epoch {epoch} (budget)")
            persist(dict(stopped="budget"))
            return None
        t0 = time.time()
        # data order is a pure function of (seed, epoch index from zero)
        state, st = step_loop(arm, tx, state, Xtr, Ytr, seed, epoch - 1, tel)
        params, opt_state, bs, _ = state
        val = RB.evaluate_split(params, bs, eval_model, Xva, Yva, BATCH)
        rec = dict(epoch=epoch, **st, val_accuracy=val["accuracy"],
                   val_cross_entropy=val["cross_entropy"], val_n=val["n"],
                   epoch_s=time.time() - t0)
        ok, checks, dom = epoch_acceptance(arm, params, opt_state, bs, rec)
        rec["acceptance"] = checks
        if dom is not None:
            rec["domain"] = dom
            rec["n_passive"] = sum(r.get("n_passive_rho_le_1", 0)
                                   for r in dom["layers"])
        epochs.append(rec)
        persist()                     # every completed epoch, immediately
        if not ok:
            status["failed"] = (f"{arm} seed {seed} epoch {epoch}: acceptance "
                                f"failed {checks}")
            persist(dict(failed=status["failed"]))
            return "FAILED"
        print(f"  [seed {seed}] {arm:22s} epoch {epoch:>2}  train "
              f"{rec['train_loss']:.4f}/{rec['train_acc']:.4f}  val "
              f"{val['accuracy']:.4f} ce {val['cross_entropy']:.4f}  "
              f"{rec['epoch_s']:.1f}s")
    params, opt_state, bs, _ = state
    save_tree(os.path.join(out, "params", f"final_seed{seed}_{arm}.msgpack"),
              dict(params=params, batch_stats=bs))
    n = max(tel["steps"], 1)
    best = max(epochs[1:], key=lambda e: e["val_accuracy"])
    row = dict(
        arm=arm, substrate=ARM_SUBSTRATE[arm], seed=seed,
        endpoint_epoch=EPOCHS,
        endpoint_val_accuracy=epochs[-1]["val_accuracy"],
        endpoint_val_cross_entropy=epochs[-1]["val_cross_entropy"],
        best_val_descriptive=dict(epoch=best["epoch"],
                                  accuracy=best["val_accuracy"],
                                  note="descriptive only; never replaces the "
                                       "epoch-10 endpoint"),
        epochs=epochs, wall_s=time.time() - t_run,
        params=parameter_report(params),
        added_parameters=sum(int(onp.asarray(v).size)
                             for p, v in jax.tree_util.tree_flatten_with_path(
                                 params)[0]
                             if leaf_name(p) in SG.ADDED_LEAVES),
        state_counts=state_counts(arm, params),
        executed_coefficients_final=added_leaf_record(params),
        projection_telemetry=dict(
            events=tel["events"],
            entry_updates=n * (N_LAYERS * (SSM_SIZE // 2)
                               if arm == "C_stable_generalized" else 0),
            max_proposed_overshoot=tel["max_overshoot"],
            min_post_projection_log_margin=tel["min_log_margin"],
            mean_added_grad_norm=tel["added_grad_norm_sum"] / n,
            mean_added_update_norm=tel["added_update_norm_sum"] / n,
            steps=tel["steps"],
            note=("events count (entry, update) pairs whose PROPOSED r "
                  "exceeded the stable-interior bound; optimizer state was "
                  "left untouched")),
        final_param_digest=tree_digest(params))
    if arm != "A_rawat":
        row["final_domain"] = SG.executed_domain_report(params)
    row["response_change"] = profile_change(start["_profile"][arm],
                                            response_profile(arm, params))
    for r in row["response_change"]:
        r.pop("K", None)
    persist(dict(complete=True))
    return row


# ------------------------------------------------------------- preflight ---
def preflight(start, tx, data, steps_per_epoch, status, out):
    """Time what training EXECUTES (review R3), per arm:

      * steps through `step_loop`, the same host path as training, after a
        warm-up that absorbs compilation (incurred, not projected);
      * a validation pass, scaled from PREFLIGHT_VAL_BATCHES full batches;
      * per-epoch acceptance on the ACTUAL measured preflight state and
        metrics (review F2): its verdict is ENFORCED, not only timed;
      * per-epoch persistence of the run record;
      * final serialization and response diagnostics.
    """
    (Xtr, Ytr), (Xva, Yva) = data["train"], data["val"]
    rows, total, retraced, failures = [], 0.0, False, []
    n_val_batches = -(-Xva.shape[0] // BATCH)
    for arm in ARMS:
        p, bs = start[arm]
        opt = tx.init(p)
        state = (p, opt, bs, jax.random.PRNGKey(0))
        t0 = time.time()
        state, _ = step_loop(arm, tx, state, Xtr, Ytr, 0, 0, new_telemetry(),
                             max_steps=2)
        warm_s = time.time() - t0
        n0 = train_step._cache_size()
        tel = new_telemetry()
        t1 = time.time()
        state, st = step_loop(arm, tx, state, Xtr, Ytr, 0, 1, tel,
                              max_steps=PREFLIGHT_TIMED_STEPS)
        step_s = (time.time() - t1) / st["steps"]
        arm_retraced = train_step._cache_size() != n0
        retraced = retraced or arm_retraced
        q, o2, b2, _ = state
        RB.evaluate_split(q, b2, build(arm, False), Xva[:BATCH * 2],
                          Yva[:BATCH * 2], BATCH)            # warm
        t2 = time.time()
        val = RB.evaluate_split(q, b2, build(arm, False),
                                Xva[:BATCH * PREFLIGHT_VAL_BATCHES],
                                Yva[:BATCH * PREFLIGHT_VAL_BATCHES], BATCH)
        val_s = (time.time() - t2) / PREFLIGHT_VAL_BATCHES * n_val_batches
        t3 = time.time()
        # F2: the ACTUAL measured step metrics and validation result
        rec = dict(train_loss=st["train_loss"], train_acc=st["train_acc"],
                   last_grad_norm=st["last_grad_norm"],
                   val_accuracy=val["accuracy"],
                   val_cross_entropy=val["cross_entropy"],
                   val_n_preflight=val["n"])
        acc_ok, checks, dom = epoch_acceptance(arm, q, o2, b2, rec)
        write(os.path.join(out, "preflight", f"persist_{arm}.json"),
              dict(epochs=[rec] * EPOCHS, telemetry=tel,
                   acceptance=checks, domain=dom))
        epoch_host_s = time.time() - t3
        t4 = time.time()
        save_tree(os.path.join(out, "preflight", f"final_{arm}.msgpack"),
                  dict(params=q, batch_stats=b2))
        response_profile(arm, q)
        state_counts(arm, q)
        tree_digest(q)
        final_s = time.time() - t4
        per_run = (EPOCHS * (steps_per_epoch * step_s + val_s + epoch_host_s)
                   + final_s)
        arm_total = len(SEEDS) * per_run
        total += arm_total
        timing = dict(host_path_step_s=step_s, val_pass_s=val_s,
                      epoch_acceptance_and_persist_s=epoch_host_s,
                      final_serialize_diagnostics_s=final_s,
                      per_run_s=per_run, arm_total_s=arm_total)
        timing_ok = all(onp.isfinite(v) and v >= 0 for v in timing.values())
        if not acc_ok:
            failures.append(f"{arm}: preflight acceptance failed {checks}")
        if not timing_ok:
            failures.append(f"{arm}: non-finite or negative measured timing "
                            f"{timing}")
        rows.append(dict(arm=arm, warmup_s_incurred=warm_s,
                         timed_steps=st["steps"], **timing,
                         measured_metrics=rec, acceptance=checks,
                         acceptance_passed=acc_ok, domain=dom,
                         timing_finite_nonnegative=timing_ok,
                         retraced=arm_retraced))
        print(f"[preflight] {arm:22s} warm {warm_s:6.1f}s  step "
              f"{step_s * 1e3:6.2f}ms  epoch {steps_per_epoch * step_s:6.1f}s  "
              f"val {val_s:5.1f}s  accept {epoch_host_s:4.2f}s  final "
              f"{final_s:4.1f}s  x{len(SEEDS)} = {arm_total:6.1f}s")
    status["preflight"] = dict(
        rows=rows, projected_remaining_s=total, retraced_any=retraced,
        failures=failures,
        scope=("9 runs x 10 epochs x steps through the SAME host path as "
               "training (data access, transfer, RNG split, projection "
               "telemetry, synchronization), the per-epoch validation pass, "
               "acceptance and persistence, and final serialization and "
               "diagnostics. Warm-up compilation is incurred, not projected."))
    print(f"PREFLIGHT_PROJECTED_REMAINING_S={total:.1f}")
    return total, retraced, failures


def decide_after_preflight(proj, retraced, failures, left_s):
    """Pure decision (review F2). Invalid numerical state is FAILED (4); a valid
    but untrustworthy or over-budget projection is INCOMPLETE (3); None means
    the nine runs may start."""
    if failures:
        return 4, "FAILED", f"preflight acceptance/timing failed: {failures}"
    if proj is None or not onp.isfinite(proj) or proj < 0:
        return 4, "FAILED", f"non-finite or negative projection {proj!r}"
    if retraced:
        return 3, "INCOMPLETE", ("retrace during preflight step timing; "
                                 "projection untrustworthy, not started")
    if proj > left_s:
        return 3, "INCOMPLETE", (f"projected {proj:.0f}s > remaining "
                                 f"{left_s:.0f}s; the screen was NOT started. "
                                 f"No arm, seed or epoch was reduced.")
    return None


def execute_screen(start, tx, data, steps_per_epoch, out, deadline, reserve_s,
                   status, epoch0, preflight_fn=None):
    """Preflight, the ENFORCED decision, then the nine runs and the screen.
    Returns (exit code, label). No run starts unless the decision is None."""
    preflight_fn = preflight if preflight_fn is None else preflight_fn
    proj, retraced, failures = preflight_fn(start, tx, data, steps_per_epoch,
                                            status, out)
    write(os.path.join(out, "status.json"), status)
    left_s = deadline - time.time() - reserve_s
    d = decide_after_preflight(proj, retraced, failures, left_s)
    if d is not None:
        code, label, why = d
        if code == 4:
            status["failed"] = why
        else:
            status["incomplete"].append(why)
        print(f"[!] {why}")
        return code, label
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            r = run_one(arm, seed, start, tx, data, steps_per_epoch, out,
                        deadline, reserve_s, status, epoch0)
            if r == "FAILED":
                status["results"] = rows
                return 4, "FAILED"
            if r is None:
                status["results"] = rows
                return 3, "INCOMPLETE"
            rows.append(r)
            status["results"] = rows
            write(os.path.join(out, "results.json"), rows)
            write(os.path.join(out, "status.json"), status)
    status["screen"] = screen(rows)
    sc = status["screen"]
    for k, c in sc["comparisons"].items():
        per = "  ".join(f"{p['seed']}:{p['acc_pp']:+.2f}pp"
                        for p in c["per_seed"])
        print(f"[screen] {k}: mean {c['mean_acc_pp']:+.3f} pp  {per}"
              + (f"  mean dCE {c['mean_ce_diff']:+.5f}  passed={c['passed']}"
                 if "passed" in c else ""))
    print(f"[screen] DEVELOPMENT SUCCESS: {sc['development_success']}")
    status["complete"] = True
    return 0, "PASS"


# ---------------------------------------------------------------- screen ---
def screen(rows):
    by = {(r["seed"], r["arm"]): r for r in rows}
    res = dict(rule=dict(
        min_mean_gain_pp=SCREEN_MIN_GAIN_PP,
        paired=("C minus A and C minus B validation accuracy positive in "
                "EVERY stream"),
        ce="mean unsmoothed validation CE of C lower than BOTH A and B",
        endpoint="epoch 10; best-validation epochs are descriptive only"))
    comps, passed = {}, True
    for ref in ("A_rawat", "B_learned_input"):
        per = []
        for s in SEEDS:
            c, r = by.get((s, "C_stable_generalized")), by.get((s, ref))
            if c is None or r is None:
                continue
            per.append(dict(seed=s,
                            acc_pp=100.0 * (c["endpoint_val_accuracy"]
                                            - r["endpoint_val_accuracy"]),
                            ce=(c["endpoint_val_cross_entropy"]
                                - r["endpoint_val_cross_entropy"])))
        complete = len(per) == len(SEEDS)
        mean_pp = float(onp.mean([p["acc_pp"] for p in per])) if per else None
        mean_ce = float(onp.mean([p["ce"] for p in per])) if per else None
        ok = bool(complete and mean_pp >= SCREEN_MIN_GAIN_PP
                  and all(p["acc_pp"] > 0 for p in per) and mean_ce < 0)
        comps[f"C_vs_{ref}"] = dict(per_seed=per, mean_acc_pp=mean_pp,
                                    mean_ce_diff=mean_ce,
                                    complete_pairs=complete, passed=ok)
        passed = passed and ok
    per = []
    for s in SEEDS:
        b, a = by.get((s, "B_learned_input")), by.get((s, "A_rawat"))
        if b and a:
            per.append(dict(seed=s, acc_pp=100.0 * (b["endpoint_val_accuracy"]
                                                    - a["endpoint_val_accuracy"]),
                            ce=(b["endpoint_val_cross_entropy"]
                                - a["endpoint_val_cross_entropy"])))
    comps["B_vs_A_rawat_descriptive"] = dict(
        per_seed=per,
        mean_acc_pp=float(onp.mean([p["acc_pp"] for p in per])) if per else None)
    res["comparisons"] = comps
    res["development_success"] = bool(passed)
    res["note"] = ("a development screen on one source checkpoint and three "
                   "paired continuation streams; not a significance test, "
                   "not an independent-initialization result, not SOTA")
    return res


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cache", required=True)
    ap.add_argument("--stage2_dir", required=True)
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/stable-gp")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=1200.0)
    ap.add_argument("--reserve_s", type=float, default=40.0)
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    left = lambda: deadline - time.time() - args.reserve_s          # noqa: E731
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    status = dict(run_id=run_id, out=out, backend=backend, arms=list(ARMS),
                  arm_substrate=ARM_SUBSTRATE, seeds=list(SEEDS),
                  epochs=EPOCHS, batch=BATCH, provenance=provenance(),
                  kind="weight warm-start continuation, optimizer state reset",
                  incomplete=[], results=[])

    def finish(code, label):
        status["wall_s"] = time.time() - t0
        write(os.path.join(out, "status.json"), status)
        print(f"STABLE_GP_STATUS={label} out={out}")
        return code

    # ---- data: train and validation only; test arrays never opened
    data, manifest = SC.load_splits(args.data_cache, splits=("train", "val"))
    steps_per_epoch = data["train"][0].shape[0] // BATCH
    status["data_manifest"] = dict(
        counts=manifest["counts"],
        feature_sha256={k: v for k, v in manifest["feature_sha256"].items()
                        if k in ("train", "val")},
        splits_opened=["train", "val"])

    # ---- source checkpoint, chosen by the ORIGINAL rule, then reproduced
    src = select_source(args.stage2_dir)
    status["source"] = {k: v for k, v in src.items() if k != "config"}
    check_source_config(src["config"])
    write(os.path.join(out, "status.json"), status)
    L, restore, _ = restore_source(src["chosen"], src["config"], data)
    status["source_reproduction"] = restore
    print(f"[source] {src['chosen']['run_dir']} epoch "
          f"{restore['checkpoint_epoch']}: saved {restore['saved_correct']} "
          f"restored {restore['restored_correct']} / {restore['n']}, ce diff "
          f"{restore['ce_abs_diff']:.2e}")
    if not restore["passed"]:
        status["failed"] = "source checkpoint did not reproduce its metrics"
        return finish(4, "FAILED")

    # ---- warm start: explicit clone, whitelisted added leaves
    start, clone = {}, {}
    for arm in ARMS:
        p, bs, added = warm_start(arm, L["params"], L["batch_stats"])
        start[arm] = (p, bs)
        clone[arm] = dict(added_leaves=added,
                          common_digest=tree_digest(p, skip=SG.ADDED_LEAVES),
                          batch_stats_digest=tree_digest(bs))
    digests = {c["common_digest"] for c in clone.values()}
    bsd = {c["batch_stats_digest"] for c in clone.values()}
    status["warm_start"] = dict(per_arm=clone,
                                common_identical=len(digests) == 1,
                                batch_stats_identical=len(bsd) == 1,
                                source_digest=tree_digest(L["params"]))
    if len(digests) != 1 or len(bsd) != 1:
        status["failed"] = "common weights or normalization state differ"
        return finish(4, "FAILED")

    tx, opt_desc = make_optimizer(steps_per_epoch, EPOCHS)
    status["optimizer"] = opt_desc

    # ---- identity gate BEFORE any arm trains
    gate = identity_gate(start, data, tx)
    status["identity_gate"] = gate
    write(os.path.join(out, "status.json"), status)
    for k in ("B_learned_input_vs_A_rawat",
              "C_stable_generalized_vs_B_learned_input"):
        g = gate[k]
        print(f"[gate] {k}: logits {g['logits_rel']:.2e}  param grads "
              f"{'PASS' if g['param_grads']['passed'] else 'FAIL'} "
              f"(worst {g['param_grads']['worst_leaf']})  input grads "
              f"{'PASS' if g['input_grads']['passed'] else 'FAIL'}  routing "
              f"{g['routing']['worst']:.2e}  independent update "
              f"{g['independent_first_update']['max_update_diff_over_lr']:.2e}"
              f" (reported)  argmax {g['argmax_agreement']:.4f}  -> "
              f"{'PASS' if g['passed'] else 'FAIL'}")
    print(f"[gate] initial executed domain: C "
          f"{'PASS' if gate['C_initial_domain']['passed'] else 'FAIL'}, B "
          f"{'PASS' if gate['B_initial_domain']['passed'] else 'FAIL'}")
    if not gate["passed"]:
        status["failed"] = ("baseline identity or initial stable domain "
                            "disagreed; the screen is stopped, no tolerance "
                            "or baseline was changed")
        return finish(4, "FAILED")

    # ---- epoch-0 validation for every arm, and response profiles
    epoch0 = {}
    start["_profile"] = {}
    for arm in ARMS:
        p, bs = start[arm]
        v = RB.evaluate_split(p, bs, build(arm, False), *data["val"], BATCH)
        epoch0[arm] = dict(val_accuracy=v["accuracy"],
                           val_cross_entropy=v["cross_entropy"], val_n=v["n"])
        start["_profile"][arm] = response_profile(arm, p)
    status["epoch0"] = epoch0
    print(f"[epoch0] " + "  ".join(f"{a} {epoch0[a]['val_accuracy']:.4f}"
                                   for a in ARMS))
    write(os.path.join(out, "status.json"), status)

    # ---- preflight (enforced), then the nine runs and the screen
    code, label = execute_screen(start, tx, data, steps_per_epoch, out,
                                 deadline, args.reserve_s, status, epoch0)
    return finish(code, label)

if __name__ == "__main__":
    sys.exit(main())
