"""Paired continuation screen: Rawat, learned input horizon, stable generalized.

Authoritative brief: NEXT_PROSPECTIVE_S5_CODING_BRIEF_2026_09_16.md.
Protocol, frozen before execution: docs/STABLE_GP_CONTINUATION_PROTOCOL.md.

Three arms continue from ONE saved, validation-selected Rawat checkpoint:

    A  rawat_unchanged      alpha_p_s5, exactly as saved
    B  rawat_learned_input  the same, with a per-mode input horizon
                            T_in = 5 exp(q), q = 0 at the start
    C  sgp_learned_input    the SAME learned input horizon PLUS the stable
                            generalized recurrence, rho = exp(r) and
                            T = 5 exp(t), r = t = 0 at the start

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
#      float64 (tests/test_stable_gp.py, 1e-9 relative). In production float32
#      the B-vs-C comparison is a closed-form diagonal exponential against a
#      Pade 4x4 block exponential; the repository's declared float32
#      coefficient resolution for that block is F32 = 2e-4 (timescale probe),
#      and these gates allow for its propagation through four layers.
SOURCE_CE_TOL = 1e-5            # restored vs saved validation CE, as Stage 2
LOGIT_REL_TOL = 5e-4            # relative Frobenius, validation probe
GRAD_REL_TOL = 2e-3             # common-leaf gradients, per leaf, relative
#: common-leaf FIRST optimizer update with the added leaves' gradients frozen,
#: measured as max |u_x - u_ref| / lr_0 over every common entry. Adam's first
#: step is ~ -lr g/(|g| + eps) entrywise, so a per-leaf relative norm would
#: magnify float noise in near-zero entries; lr_0 is the natural scale.
UPDATE_MAX_TOL = 2e-3
PROBE_N = 256
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
    (loss, (logits, _)), g = jax.value_and_grad(
        loss_fn, has_aux=True)(params, batch_stats, model, x, y, rng)
    return loss, logits, g


@partial(jax.jit, static_argnums=(0,))
def logits_eval(model, params, batch_stats, x):
    return model.apply({"params": params, "batch_stats": batch_stats}, x,
                       jnp.ones(x.shape[:2]), None)


# ---------------------------------------------------- identity gate --------
def _common(tree):
    from flax.traverse_util import flatten_dict
    return {k: v for k, v in flatten_dict(tree).items()
            if k[-1] not in SG.ADDED_LEAVES}


def _rel(a, b):
    a = onp.asarray(a, dtype=onp.float64); b = onp.asarray(b, dtype=onp.float64)
    d = float(onp.sqrt(onp.sum((a - b) ** 2)))
    n = float(onp.sqrt(onp.sum(b ** 2)))
    return d / n if n > 0 else (0.0 if d == 0 else float("inf"))


def compare_common(ga, gb):
    ca, cb = _common(ga), _common(gb)
    if set(ca) != set(cb):
        return dict(worst=float("inf"), note="common leaf sets differ")
    rows = {"/".join(k): _rel(ca[k], cb[k]) for k in ca}
    worst = max(rows.values()) if rows else 0.0
    return dict(worst=worst, worst_leaf=max(rows, key=rows.get), n=len(rows))


def compare_updates(ua, ub):
    ca, cb = _common(ua), _common(ub)
    if set(ca) != set(cb):
        return dict(worst=float("inf"), note="common leaf sets differ")
    rows = {"/".join(k): float(onp.max(onp.abs(
        onp.asarray(ca[k], dtype=onp.float64)
        - onp.asarray(cb[k], dtype=onp.float64)))) / LR for k in ca}
    return dict(worst=max(rows.values()), worst_leaf=max(rows, key=rows.get),
                n=len(rows))


def frozen_update(tx, params, g):
    """One optimizer update with the ADDED leaves' gradients frozen to zero.

    Adding those gradients to the global clip can legitimately change common
    updates even when common raw gradients agree, so the baseline comparison
    of UPDATES is made with them frozen, as the brief specifies.
    """
    g0 = jax.tree_util.tree_map_with_path(
        lambda p, v: (jnp.zeros_like(v) if leaf_name(p) in SG.ADDED_LEAVES
                      else v), g)
    st = tx.init(params)
    upd, _ = tx.update(g0, st, params)
    return upd


def identity_gate(start, data, tx):
    """Initial predictions, shared gradients and frozen-extra updates, BEFORE
    any arm trains. Any disagreement stops the screen."""
    Xtr, Ytr = data["train"]
    Xva, _ = data["val"]
    xp = jnp.asarray(onp.asarray(Xva[:PROBE_N]))
    idx = onp.sort(next(iter(SC.epoch_batches(Xtr.shape[0], BATCH,
                                              SEEDS[0], 0))))
    xb = jnp.asarray(onp.asarray(Xtr[idx])); yb = jnp.asarray(Ytr[idx])
    rng = jax.random.PRNGKey(SEEDS[0])
    out = dict(probe_n=PROBE_N, tolerances=dict(
        logits_rel=LOGIT_REL_TOL, grads_rel=GRAD_REL_TOL,
        updates_max_over_lr=UPDATE_MAX_TOL))
    logits, grads, updates = {}, {}, {}
    for arm in ARMS:
        p, bs = start[arm]
        logits[arm] = onp.asarray(logits_eval(build(arm, False), p, bs, xp))
        _, _, g = grads_at(build(arm, True), p, bs, xb, yb, rng)
        grads[arm] = g
        updates[arm] = frozen_update(tx, p, g)
    pairs = (("B_learned_input", "A_rawat"),
             ("C_stable_generalized", "B_learned_input"))
    ok = True
    for x, ref in pairs:
        key = f"{x}_vs_{ref}"
        lr = _rel(logits[x], logits[ref])
        agree = float(onp.mean(onp.argmax(logits[x], -1)
                               == onp.argmax(logits[ref], -1)))
        gc = compare_common(grads[x], grads[ref])
        uc = compare_updates(updates[x], updates[ref])
        row = dict(logits_rel=lr, argmax_agreement=agree,
                   grads_common_worst_rel=gc["worst"],
                   grads_worst_leaf=gc.get("worst_leaf"),
                   updates_frozen_extra_max_over_lr=uc["worst"],
                   updates_worst_leaf=uc.get("worst_leaf"))
        row["passed"] = bool(lr <= LOGIT_REL_TOL and gc["worst"] <= GRAD_REL_TOL
                             and uc["worst"] <= UPDATE_MAX_TOL)
        ok = ok and row["passed"]
        out[key] = row
    # added-leaf gradients at the start, for the record: q is active in B and
    # C; r is generally nonzero; t's task gradient vanishes at rho = 1
    from flax.traverse_util import flatten_dict
    added = {}
    for arm in ARMS[1:]:
        for k, v in flatten_dict(grads[arm]).items():
            if k[-1] in SG.ADDED_LEAVES:
                added.setdefault(arm, {}).setdefault(k[-1], []).append(
                    float(onp.sqrt(onp.sum(onp.asarray(v) ** 2))))
    out["added_leaf_grad_norms_per_layer"] = added
    out["C_initial_domain"] = SG.executed_domain_report(
        start["C_stable_generalized"][0])
    ok = ok and out["C_initial_domain"]["passed"]
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
    from flax.traverse_util import flatten_dict
    out = {}
    for k, v in flatten_dict(params).items():
        if k[-1] not in SG.ADDED_LEAVES:
            continue
        raw = onp.asarray(v, dtype=onp.float64)
        if k[-1] == SG.LEAF_RHO:
            val = onp.exp(raw); name = "rho"
        elif k[-1] == SG.LEAF_T:
            val = SG.RECURRENT_T_REFERENCE * onp.exp(raw); name = "T"
        else:
            val = SG.INPUT_T_REFERENCE * onp.exp(raw); name = "T_in"
        out["/".join(k)] = dict(quantity=name,
                                min=float(val.min()),
                                median=float(onp.median(val)),
                                max=float(val.max()),
                                raw_abs_max=float(onp.abs(raw).max()),
                                values=val.tolist())
    return out


def state_counts(arm, params):
    m = build_single(arm)
    per = m.apply({"params": params},
                  method=lambda mm: mm.encoder.layers[0].seq.state_counts())
    return {k: (v if isinstance(v, (bool, str)) or v is None
                else int(v) * N_LAYERS) for k, v in per.items()}


# --------------------------------------------------------------- one run ---
def run_one(arm, seed, start, tx, data, steps_per_epoch, out, deadline,
            reserve_s, status, epoch0):
    (Xtr, Ytr), (Xva, Yva) = data["train"], data["val"]
    train_model, eval_model = build(arm, True), build(arm, False)
    params, bs = start[arm]
    opt_state = tx.init(params)
    rng = jax.random.PRNGKey(seed)          # SAME dropout stream in every arm
    epochs = [dict(epoch=0, **epoch0[arm], note="shared start, measured once")]
    tel = dict(events=0, max_overshoot=0.0, min_log_margin=float("inf"),
               added_grad_norm_sum=0.0, added_update_norm_sum=0.0, steps=0)
    t_run = time.time()
    for epoch in range(1, EPOCHS + 1):
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{arm} seed {seed}: stopped before epoch {epoch} (budget)")
            return None
        t0 = time.time()
        tl = ta = nb = 0.0
        # data order is a pure function of (seed, epoch index from zero)
        for idx in SC.epoch_batches(Xtr.shape[0], BATCH, seed, epoch - 1):
            sel = onp.sort(idx)
            xb = jnp.asarray(onp.asarray(Xtr[sel])); yb = jnp.asarray(Ytr[sel])
            rng, sub = jax.random.split(rng)
            (params, opt_state, bs, loss, acc, gn, t, agn,
             aun) = train_step(train_model, tx, params, opt_state, bs, xb, yb,
                               sub)
            tl += float(loss); ta += float(acc); nb += 1
            tel["events"] += int(t["n_projected"])
            tel["max_overshoot"] = max(tel["max_overshoot"],
                                       float(t["max_overshoot"]))
            tel["min_log_margin"] = min(tel["min_log_margin"],
                                        float(t["min_log_margin"]))
            tel["added_grad_norm_sum"] += float(agn)
            tel["added_update_norm_sum"] += float(aun)
            tel["steps"] += 1
        val = RB.evaluate_split(params, bs, eval_model, Xva, Yva, BATCH)
        rec = dict(epoch=epoch, train_loss=tl / nb, train_acc=ta / nb,
                   val_accuracy=val["accuracy"],
                   val_cross_entropy=val["cross_entropy"], val_n=val["n"],
                   last_grad_norm=float(gn), epoch_s=time.time() - t0)
        if arm == "C_stable_generalized":
            dom = SG.executed_domain_report(params)
            rec["domain_passed"] = dom["passed"]
            rec["n_passive"] = sum(r["n_passive_rho_le_1"]
                                   for r in dom["layers"])
            if not dom["passed"]:
                status["failed"] = (f"{arm} seed {seed} epoch {epoch}: "
                                    f"executed coefficients left the stable "
                                    f"domain or became non-finite")
                status["failed_domain"] = dom
                return "FAILED"
        if not all(onp.isfinite([rec["train_loss"], rec["val_cross_entropy"]])):
            status["failed"] = f"{arm} seed {seed} epoch {epoch}: non-finite"
            return "FAILED"
        epochs.append(rec)
        print(f"  [seed {seed}] {arm:22s} epoch {epoch:>2}  train "
              f"{rec['train_loss']:.4f}/{rec['train_acc']:.4f}  val "
              f"{val['accuracy']:.4f} ce {val['cross_entropy']:.4f}  "
              f"{rec['epoch_s']:.1f}s")
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
        added_leaves_final=added_leaf_record(params),
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
    if arm == "C_stable_generalized":
        row["final_domain"] = SG.executed_domain_report(params)
    row["response_change"] = profile_change(start["_profile"][arm],
                                            response_profile(arm, params))
    for r in row["response_change"]:
        r.pop("K", None)
    return row


# ------------------------------------------------------------- preflight ---
def preflight(start, tx, data, steps_per_epoch, status, diag_s):
    (Xtr, Ytr), (Xva, _) = data["train"], data["val"]
    idx = onp.sort(next(iter(SC.epoch_batches(Xtr.shape[0], BATCH, 0, 0))))
    xb = jnp.asarray(onp.asarray(Xtr[idx])); yb = jnp.asarray(Ytr[idx])
    rows, total, retraced = [], 0.0, False
    n_val_batches = -(-Xva.shape[0] // BATCH)
    for arm in ARMS:
        m = build(arm, True)
        p, bs = start[arm]
        opt = tx.init(p)
        rng = jax.random.PRNGKey(0)
        t0 = time.time()
        out = train_step(m, tx, p, opt, bs, xb, yb, rng)
        out[3].block_until_ready()
        compile_s = time.time() - t0
        q, o2, b2 = out[0], out[1], out[2]
        out = train_step(m, tx, q, o2, b2, xb, yb, rng)
        q, o2, b2 = out[0], out[1], out[2]
        out[3].block_until_ready()
        n0 = train_step._cache_size()
        t1 = time.time()
        for _ in range(5):
            out = train_step(m, tx, q, o2, b2, xb, yb, rng)
            q, o2, b2 = out[0], out[1], out[2]
        out[3].block_until_ready()
        step_s = (time.time() - t1) / 5
        arm_retraced = train_step._cache_size() != n0
        retraced = retraced or arm_retraced
        t2 = time.time()
        RB.evaluate_split(p, bs, build(arm, False), Xva[:BATCH * 8],
                          data["val"][1][:BATCH * 8], BATCH)
        val_s = (time.time() - t2) / 8 * n_val_batches
        per_run = EPOCHS * (steps_per_epoch * step_s + val_s)
        arm_total = len(SEEDS) * per_run
        total += arm_total
        rows.append(dict(arm=arm, compile_s_incurred=compile_s, step_s=step_s,
                         val_pass_s=val_s, per_run_s=per_run,
                         arm_total_s=arm_total, retraced=arm_retraced))
        print(f"[preflight] {arm:22s} compile {compile_s:6.1f}s  step "
              f"{step_s * 1e3:6.2f}ms  epoch {steps_per_epoch * step_s:6.1f}s  "
              f"val {val_s:5.1f}s  x{len(SEEDS)} = {arm_total:6.1f}s")
    host = 3.0 * len(SEEDS) * len(ARMS) + len(SEEDS) * len(ARMS) * diag_s
    total += host
    status["preflight"] = dict(rows=rows, host_and_diagnostics_s=host,
                               measured_diagnostic_s_per_run=diag_s,
                               projected_remaining_s=total,
                               retraced_any=retraced,
                               scope=("9 runs x 10 epochs x steps, one "
                                      "validation pass per epoch, final "
                                      "diagnostics and serialization. "
                                      "Compilation measured here is incurred, "
                                      "not projected again."))
    print(f"PREFLIGHT_PROJECTED_REMAINING_S={total:.1f}")
    return total, retraced


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
        print(f"[gate] {k}: logits {g['logits_rel']:.2e}  grads "
              f"{g['grads_common_worst_rel']:.2e}  frozen-extra updates "
              f"{g['updates_frozen_extra_max_over_lr']:.2e}  argmax "
              f"{g['argmax_agreement']:.4f}  -> "
              f"{'PASS' if g['passed'] else 'FAIL'}")
    print(f"[gate] C initial stable domain: "
          f"{'PASS' if gate['C_initial_domain']['passed'] else 'FAIL'}")
    if not gate["passed"]:
        status["failed"] = ("baseline identity or initial stable domain "
                            "disagreed; the screen is stopped, no tolerance "
                            "or baseline was changed")
        return finish(4, "FAILED")

    # ---- epoch-0 validation for every arm, and response profiles
    epoch0 = {}
    t_diag = time.time()
    start["_profile"] = {}
    for arm in ARMS:
        p, bs = start[arm]
        v = RB.evaluate_split(p, bs, build(arm, False), *data["val"], BATCH)
        epoch0[arm] = dict(val_accuracy=v["accuracy"],
                           val_cross_entropy=v["cross_entropy"], val_n=v["n"])
        start["_profile"][arm] = response_profile(arm, p)
    diag_s = (time.time() - t_diag) / len(ARMS)
    status["epoch0"] = epoch0
    print(f"[epoch0] " + "  ".join(f"{a} {epoch0[a]['val_accuracy']:.4f}"
                                   for a in ARMS))
    write(os.path.join(out, "status.json"), status)

    # ---- preflight on the complete declared screen
    proj, retraced = preflight(start, tx, data, steps_per_epoch, status,
                               diag_s)
    write(os.path.join(out, "status.json"), status)
    if retraced:
        status["incomplete"].append("retrace during preflight step timing; "
                                    "projection untrustworthy, not started")
        return finish(3, "INCOMPLETE")
    if proj > left():
        status["incomplete"].append(
            f"projected {proj:.0f}s > remaining {left():.0f}s; the screen was "
            f"NOT started. No arm, seed or epoch was reduced.")
        print(f"[!] {status['incomplete'][-1]}")
        return finish(3, "INCOMPLETE")

    # ---- nine runs, grouped by stream so a budget stop leaves whole pairs
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            r = run_one(arm, seed, start, tx, data, steps_per_epoch, out,
                        deadline, args.reserve_s, status, epoch0)
            if r == "FAILED":
                status["results"] = rows
                return finish(4, "FAILED")
            if r is None:
                status["results"] = rows
                return finish(3, "INCOMPLETE")
            rows.append(r)
            status["results"] = rows
            write(os.path.join(out, "results.json"), rows)
            write(os.path.join(out, "status.json"), status)

    status["screen"] = screen(rows)
    sc = status["screen"]
    for k, c in sc["comparisons"].items():
        per = "  ".join(f"{p['seed']}:{p['acc_pp']:+.2f}pp" for p in c["per_seed"])
        print(f"[screen] {k}: mean {c['mean_acc_pp']:+.3f} pp  {per}"
              + (f"  mean dCE {c['mean_ce_diff']:+.5f}  passed={c['passed']}"
                 if "passed" in c else ""))
    print(f"[screen] DEVELOPMENT SUCCESS: {sc['development_success']}")
    status["complete"] = True
    return finish(0, "PASS")


if __name__ == "__main__":
    sys.exit(main())
