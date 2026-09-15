"""Speech Commands 10 benchmark runner: five arms, one substrate, GPU only.

Modes
-----
``integration``  one initialization plus two updates, then a throughput and
                 memory measurement. Establishes EXECUTION, not accuracy.
``train``        train with the published recipe; write an atomic best
                 checkpoint (highest validation accuracy) and a resumable last
                 checkpoint every epoch. Never touches the test split.
``evaluate``     load a selected checkpoint and score ONE split. Test scoring
                 is refused unless `--confirm_test` is passed, so a test number
                 cannot be produced as a side effect of a training job.

The published recipe (Rawat et al. Appendix E.4) is implemented as stated:
AdamW, batch 32, learning rate 1e-3 and weight decay 1e-4 on ALL S5 parameters
including Lambda, B_tilde, C_tilde and the log steps, global gradient norm
clipped at 1.0, cross entropy with label smoothing 0.1 on mean-pooled logits,
cosine annealing to 1e-6 over at most 300 epochs, early stopping after 20
epochs without improved validation accuracy, a checkpoint each epoch, and test
accuracy read from the highest-validation-accuracy checkpoint.

NOTE, recorded rather than silently reconciled: applying weight decay to the
SSM parameters differs from the upstream S5 convention, which excludes them.
We follow the PAPER because this is a reproduction. `--ssm_weight_decay` can
restore the upstream convention for a separately labelled run; it is not used
by the predeclared protocol.

The added physical coefficients are NOT parameters here. They live in frozen
configuration (`s5.physical_coefficients`), so no gradient, optimizer slot or
weight-decay term can reach them; `--assert_fixed` verifies this at startup and
aborts rather than warning.
"""

import argparse
import json
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import optax
from flax.training import train_state
from jax.scipy.linalg import block_diag

from dataloaders import speech_commands10 as SC
from s5.checkpointing import (append_metrics, checkpoint_exists, make_run_dir,
                              provenance, restore_checkpoint, save_checkpoint,
                              write_config)
from s5.physical_coefficients import SYMMETRIC_REFERENCE, coefficient_table
from s5.rawat_model import BatchRawatClassifier, parameter_report
from s5.rawat_s5 import (ARMS, RESPONSE_PARAM_NAMES, RHO_ONLY_PARAM_NAME,
                         T_ONLY_PARAM_NAME, init_substrate_ssm)
from s5.response_projection import ALL_RESPONSE_LEAF_NAMES
from s5.response_projection import \
    project_response_leaves as shared_project_response_leaves
from s5.ssm_init import make_DPLR_HiPPO

#: Appendix E.3 defaults for the depth-4 width-32 MFCC row of Table 6
HIPPO_BLOCKS = 8
D_OUTPUT = 10


class TrainState(train_state.TrainState):
    batch_stats: dict = None


def require_gpu(allow_cpu=False):
    """Explicit backend guard. Training on CPU is refused, not warned about."""
    backend = jax.default_backend()
    if backend != "gpu" and not allow_cpu:
        raise SystemExit(
            f"REFUSING TO RUN: backend is {backend!r}, not 'gpu'.\n"
            f"All training and numerical validation for this study belong on "
            f"the cluster GPU. Pass --allow_cpu ONLY for a structural smoke "
            f"test whose output must never be reported as evidence.")
    return backend


def ssm_kwargs(d_model, ssm_size, blocks=HIPPO_BLOCKS, conj_sym=True):
    blk = ssm_size // blocks
    Lam, _, _, V, _ = make_DPLR_HiPPO(blk)
    if conj_sym:
        blk //= 2
        P = ssm_size // 2
    else:
        P = ssm_size
    Lam, V = Lam[:blk], V[:, :blk]
    Vc = V.conj().T
    Lam = (Lam * jnp.ones((blocks, blk))).ravel()
    return dict(H=d_model, P=P, Lambda_re_init=Lam.real,
                Lambda_im_init=Lam.imag, V=block_diag(*([V] * blocks)),
                Vinv=block_diag(*([Vc] * blocks)),
                C_init="trunc_standard_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=conj_sym,
                bidirectional=False)


def build_model(arm, d_model, ssm_size, n_layers, training):
    ssm = init_substrate_ssm(arm, physical=SYMMETRIC_REFERENCE,
                             **ssm_kwargs(d_model, ssm_size))
    return BatchRawatClassifier(ssm=ssm, d_model=d_model, n_layers=n_layers,
                                d_output=D_OUTPUT, training=training)


def make_optimizer(args, steps_per_epoch):
    total = max(1, steps_per_epoch * args.epochs)
    sched = optax.cosine_decay_schedule(
        init_value=args.lr, decay_steps=total, alpha=args.lr_final / args.lr)
    if args.ssm_weight_decay is None:
        tx = optax.chain(optax.clip_by_global_norm(args.grad_clip),
                         optax.adamw(sched, weight_decay=args.weight_decay))
        groups = "single group: AdamW(wd=%g) on ALL parameters (paper)" \
            % args.weight_decay
    else:
        def label(path, _):
            return "ssm" if any(p.key == "seq" for p in path
                                if hasattr(p, "key")) else "reg"
        tx = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.multi_transform(
                {"ssm": optax.adamw(sched, weight_decay=args.ssm_weight_decay),
                 "reg": optax.adamw(sched, weight_decay=args.weight_decay)},
                lambda params: jax.tree_util.tree_map_with_path(label, params)))
        groups = ("two groups: SSM wd=%g, other wd=%g (UPSTREAM convention, "
                  "NOT the paper)" % (args.ssm_weight_decay, args.weight_decay))
    return tx, sched, groups


#: which response leaves each arm is allowed to carry, per layer. Every arm
#: absent from this table keeps the blanket prohibition: the historical
#: fixed-coefficient arms must stay fixed, and the prohibition is NOT relaxed
#: globally just because a newer arm learns a response.
ARM_RESPONSE_LEAVES = {
    "gp_learned_response": RESPONSE_PARAM_NAMES,
    "gp_rho": (RHO_ONLY_PARAM_NAME,),
    "gp_rho_frozen": (RHO_ONLY_PARAM_NAME,),
    "gp_rho_prospin": (RHO_ONLY_PARAM_NAME,),
    # both learned-timescale arms STORE both leaves so their trees are
    # identical; the fixed arm simply never updates eta. Stored is not
    # trainable, and the runner reports the two counts separately.
    "gp_rho_T": (RHO_ONLY_PARAM_NAME, T_ONLY_PARAM_NAME),
    "gp_rho_T_fixed": (RHO_ONLY_PARAM_NAME, T_ONLY_PARAM_NAME),
}


def assert_response_policy(params, arm, n_layers, P):
    """Enforce the coefficient policy EXACTLY, per arm.

    The allowed leaves come from `ARM_RESPONSE_LEAVES`, which is keyed by arm.
    The earlier version hard-coded the two superseded leaf names, so the
    rho-only arms would have tripped the blanket prohibition while a stray
    `log_response_rho_only` on a FIXED arm would have gone unnoticed - it was
    not in the set being searched for. Both directions are now covered.
    """
    from flax.traverse_util import flatten_dict
    flat = {"/".join(k): v for k, v in flatten_dict(params).items()}
    banned = ("gp_response", "response_raw", "mu_ratio", "rho_raw", "horizon",
              "so_")
    allowed = ARM_RESPONSE_LEAVES.get(arm, ())
    found = {n: v for n, v in flat.items()
             if n.rsplit("/", 1)[-1] in ALL_RESPONSE_LEAF_NAMES}
    other = [n for n in flat if any(tok in n.lower() for tok in banned)]
    if other:
        raise SystemExit(
            f"REFUSING TO RUN arm {arm!r}: unexpected response-like "
            f"parameters {other}.")
    stray = sorted(n for n in found if n.rsplit("/", 1)[-1] not in allowed)
    if stray:
        raise SystemExit(
            f"REFUSING TO RUN arm {arm!r}: response leaves {stray} are not "
            f"declared for this arm (allowed: {list(allowed)}). The added "
            f"physical response must stay frozen configuration wherever it is "
            f"not explicitly declared learnable.")
    if allowed:
        want = n_layers * len(allowed)
        if len(found) != want:
            raise SystemExit(
                f"REFUSING: arm {arm!r} must carry exactly {want} response "
                f"leaves ({list(allowed)} per layer), found {sorted(found)}.")
        for n, v in found.items():
            if v.shape != (P,):
                raise SystemExit(
                    f"REFUSING: response leaf {n} has shape {v.shape}, "
                    f"expected {(P,)} (one scalar per STORED complex mode, "
                    f"shared with its conjugate partner).")
        added = sum(int(v.size) for v in found.values())
        print(f"[*] response policy: learned {list(allowed)}, {len(found)} "
              f"leaves, {added} added real parameters")
    return len(flat)


#: Imported, NOT redefined. The local copy here recognized only the two
#: superseded leaf names and would have left `log_response_rho_only` stranded
#: outside its interval with zero task gradient - the exact lockout that
#: reversed the recall study's one positive finding. One whitelist, one module.
project_response_leaves = shared_project_response_leaves


def loss_and_logits(params, batch_stats, model, x, y, rng, training,
                    label_smoothing):
    variables = {"params": params}
    if batch_stats is not None:
        variables["batch_stats"] = batch_stats
    ts = jnp.ones(x.shape[:2])
    if training:
        logits, updates = model.apply(variables, x, ts, None,
                                      rngs={"dropout": rng},
                                      mutable=["batch_stats"])
        new_bs = updates.get("batch_stats")
    else:
        logits = model.apply(variables, x, ts, None)
        new_bs = None
    onehot = jax.nn.one_hot(y, D_OUTPUT)
    loss = optax.softmax_cross_entropy(
        logits, optax.smooth_labels(onehot, label_smoothing)).mean()
    return loss, (logits, new_bs)


@partial(jax.jit, static_argnums=(4, 6))
def train_step(state, x, y, rng, model, label_smoothing, _unused=None):
    def fn(p):
        return loss_and_logits(p, state.batch_stats, model, x, y, rng, True,
                               label_smoothing)
    (loss, (logits, new_bs)), grads = jax.value_and_grad(fn, has_aux=True)(
        state.params)
    state = state.apply_gradients(grads=grads)
    # declared projection policy; a no-op for every arm without response leaves
    state = state.replace(params=project_response_leaves(state.params))
    if new_bs is not None:
        state = state.replace(batch_stats=new_bs)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return state, loss, acc, optax.global_norm(grads)


@partial(jax.jit, static_argnums=(3,))
def eval_step(params, batch_stats, batch, model):
    x, y = batch
    loss, (logits, _) = loss_and_logits(params, batch_stats, model, x, y, None,
                                        False, 0.0)
    correct = jnp.sum(jnp.argmax(logits, -1) == y)
    # unsmoothed cross entropy, for the declared selection tie-break
    ce = optax.softmax_cross_entropy(
        logits, jax.nn.one_hot(y, D_OUTPUT)).sum()
    return correct, ce, y.shape[0]


def evaluate_split(params, batch_stats, model, X, Y, batch_size):
    n = X.shape[0]
    correct = ce = seen = 0
    for i in range(0, n, batch_size):
        xb = jnp.asarray(onp.asarray(X[i:i + batch_size]))
        yb = jnp.asarray(Y[i:i + batch_size])
        c, l, m = eval_step(params, batch_stats, (xb, yb), model)
        correct += int(c); ce += float(l); seen += int(m)
    return dict(accuracy=correct / seen, cross_entropy=ce / seen, n=seen)


def peak_memory_bytes():
    try:
        st = jax.devices()[0].memory_stats()
        return None if st is None else int(st.get("peak_bytes_in_use", 0)) or None
    except Exception:
        return None


def init_everything(args, steps_per_epoch):
    model = build_model(args.arm, args.d_model, args.ssm_size, args.n_layers,
                        training=True)
    eval_model = build_model(args.arm, args.d_model, args.ssm_size,
                             args.n_layers, training=False)
    dummy = jnp.zeros((2, SC.N_FRAMES, SC.N_MFCC))
    variables = model.init({"params": jax.random.PRNGKey(args.seed),
                            "dropout": jax.random.PRNGKey(args.seed + 1)},
                           dummy, jnp.ones((2, SC.N_FRAMES)), None)
    params = variables["params"]
    assert_response_policy(params, args.arm, args.n_layers, args.ssm_size // 2)
    tx, sched, groups = make_optimizer(args, steps_per_epoch)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx,
                              batch_stats=variables.get("batch_stats"))
    return model, eval_model, state, groups


def arm_state_counts(args):
    """Executed carry sizes, read from the bound module, not assumed."""
    m = build_model(args.arm, args.d_model, args.ssm_size, args.n_layers, True)
    v = m.init({"params": jax.random.PRNGKey(0), "dropout":
                jax.random.PRNGKey(1)},
               jnp.zeros((2, SC.N_FRAMES, SC.N_MFCC)),
               jnp.ones((2, SC.N_FRAMES)), None)

    def read(mm):
        return mm.encoder.layers[0].seq.state_counts()
    per_layer = m.apply(v, method=read)
    # Only NUMERIC entries scale with depth. A layer's state_counts may also
    # carry descriptive entries - the prospective-recurrence arm reports why it
    # is memoryless - and int()-ing those crashed the preflight on the second
    # arm. Non-numeric entries are passed through unscaled.
    scaled = {}
    for k, x in per_layer.items():
        if isinstance(x, (bool, str)) or x is None:
            scaled[k] = x
        else:
            scaled[k] = int(x) * args.n_layers
    return scaled | {"per_layer": per_layer, "n_layers": args.n_layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train",
                    choices=["integration", "train", "evaluate"])
    ap.add_argument("--arm", default="alpha_p_s5", choices=sorted(ARMS))
    ap.add_argument("--data_cache", required=True)
    ap.add_argument("--outdir", default="results/gp/rawat_sc10")
    ap.add_argument("--run_dir", default=None,
                    help="existing run directory to resume or evaluate")
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--ssm_size", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--max_epochs_this_job", type=int, default=None,
                    help="stop after this many epochs and stay resumable")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_final", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ssm_weight_decay", type=float, default=None)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--confirm_test", action="store_true",
                    help="required to score the test split; a test number is "
                         "never produced as a side effect")
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    backend = require_gpu(args.allow_cpu)
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    data, manifest = SC.load(args.data_cache)
    Xtr, Ytr = data["train"]
    steps_per_epoch = Xtr.shape[0] // args.batch
    model, eval_model, state, groups = init_everything(args, steps_per_epoch)
    n_params = parameter_report(state.params)
    counts = arm_state_counts(args)

    run_dir = args.run_dir or make_run_dir(args.outdir, tag=args.arm)
    os.makedirs(run_dir, exist_ok=True)
    meta_extra = dict(arm=args.arm, arm_definition=ARMS[args.arm],
                      backend=backend, params=n_params, state_counts=counts,
                      optimizer_groups=groups,
                      steps_per_epoch=steps_per_epoch,
                      physical_coefficients=coefficient_table(),
                      data_manifest=dict(
                          counts=manifest["counts"],
                          file_list_sha256=manifest["file_list_sha256"],
                          feature_sha256=manifest["feature_sha256"],
                          mfcc=manifest["mfcc"],
                          split_seed=manifest["split_seed"]))
    write_config(run_dir, args, extra=meta_extra)
    print(f"[*] arm={args.arm} backend={backend} run_dir={run_dir}")
    print(f"[*] params={n_params} state={counts}")

    if args.mode == "evaluate":
        if args.split == "test" and not args.confirm_test:
            raise SystemExit(
                "REFUSING: scoring the test split requires --confirm_test, and "
                "may be done ONCE after every training and configuration "
                "choice is fixed.")
        state, bs, meta = restore_checkpoint(run_dir, args.checkpoint, state,
                                             state.batch_stats)
        X, Y = data[args.split]
        res = evaluate_split(state.params, bs, eval_model, X, Y, args.batch)
        rec = dict(mode="evaluate", split=args.split,
                   checkpoint=args.checkpoint, arm=args.arm,
                   checkpoint_epoch=meta.get("epoch"), **res)
        append_metrics(run_dir, rec)
        print(f"[RESULT] {json.dumps(rec)}")
        return rec

    rng = jax.random.PRNGKey(args.seed + 2)
    start_epoch, best = 0, dict(accuracy=-1.0, cross_entropy=float("inf"),
                                epoch=-1)
    since_improved = 0
    if checkpoint_exists(run_dir, "last"):
        state, bs, meta = restore_checkpoint(run_dir, "last", state,
                                             state.batch_stats, rng)
        state = state.replace(batch_stats=bs)
        ls = meta.get("loop_state") or {}
        start_epoch = int(ls.get("next_epoch", 0))
        best = ls.get("best", best)
        since_improved = int(ls.get("since_improved", 0))
        if "rng" in meta:
            rng = meta["rng"]
        print(f"[*] resumed from epoch {start_epoch} (best acc "
              f"{best['accuracy']:.4f} at epoch {best['epoch']})")

    if args.mode == "integration":
        t0 = time.time()
        idx = next(iter(SC.epoch_batches(Xtr.shape[0], args.batch,
                                         args.seed, 0)))
        xb = jnp.asarray(onp.asarray(Xtr[onp.sort(idx)]))
        yb = jnp.asarray(Ytr[onp.sort(idx)])
        state, loss, acc, gn = train_step(state, xb, yb, rng, model,
                                          args.label_smoothing)
        loss.block_until_ready()
        compile_s = time.time() - t0
        t1 = time.time()
        for _ in range(2):
            state, loss, acc, gn = train_step(state, xb, yb, rng, model,
                                              args.label_smoothing)
        loss.block_until_ready()
        steady = (time.time() - t1) / 2
        rec = dict(mode="integration", arm=args.arm, loss=float(loss),
                   acc=float(acc), grad_norm=float(gn),
                   compile_plus_first_step_s=compile_s,
                   steady_state_step_s=steady,
                   throughput_seq_per_s=args.batch / steady,
                   peak_memory_bytes=peak_memory_bytes(),
                   params=n_params, state_counts=counts, backend=backend)
        append_metrics(run_dir, rec)
        print(f"[RESULT] {json.dumps(rec)}")
        return rec

    Xva, Yva = data["val"]
    stop_after = (args.epochs if args.max_epochs_this_job is None
                  else min(args.epochs, start_epoch + args.max_epochs_this_job))
    for epoch in range(start_epoch, stop_after):
        t0 = time.time()
        tot_loss = tot_acc = nb = 0.0
        for idx in SC.epoch_batches(Xtr.shape[0], args.batch, args.seed, epoch):
            sel = onp.sort(idx)
            xb = jnp.asarray(onp.asarray(Xtr[sel]))
            yb = jnp.asarray(Ytr[sel])
            rng, sub = jax.random.split(rng)
            state, loss, acc, gn = train_step(state, xb, yb, sub, model,
                                              args.label_smoothing)
            tot_loss += float(loss); tot_acc += float(acc); nb += 1
        val = evaluate_split(state.params, state.batch_stats, eval_model,
                             Xva, Yva, args.batch)
        improved = val["accuracy"] > best["accuracy"]
        if improved:
            best = dict(accuracy=val["accuracy"],
                        cross_entropy=val["cross_entropy"], epoch=epoch)
            since_improved = 0
        else:
            since_improved += 1
        rec = dict(mode="train", arm=args.arm, epoch=epoch,
                   train_loss=tot_loss / nb, train_acc=tot_acc / nb,
                   val_accuracy=val["accuracy"],
                   val_cross_entropy=val["cross_entropy"],
                   grad_norm=float(gn), epoch_s=time.time() - t0,
                   peak_memory_bytes=peak_memory_bytes(),
                   best_val_accuracy=best["accuracy"],
                   since_improved=since_improved)
        append_metrics(run_dir, rec)
        print(f"  epoch {epoch:>3}  train {rec['train_loss']:.4f}/"
              f"{rec['train_acc']:.4f}  val {val['accuracy']:.4f}  "
              f"best {best['accuracy']:.4f}  {rec['epoch_s']:.1f}s")

        loop = dict(next_epoch=epoch + 1, best=best,
                    since_improved=since_improved, data_seed=args.seed,
                    rng=rng,
                    note="a resumed epoch restarts at its first batch")
        save_checkpoint(run_dir, "last", state, epoch, int(state.step),
                        config=args, batch_stats=state.batch_stats,
                        data_seed=args.seed, loop_state=loop)
        if improved:
            save_checkpoint(run_dir, "best", state, epoch, int(state.step),
                            config=args, batch_stats=state.batch_stats,
                            data_seed=args.seed, loop_state=loop,
                            notes=f"highest validation accuracy "
                                  f"{best['accuracy']:.6f} at epoch {epoch}")
        if since_improved >= args.patience:
            print(f"[*] early stop: {args.patience} epochs without improvement")
            break

    summary = dict(arm=args.arm, best=best, params=n_params,
                   state_counts=counts, epochs_run=stop_after - start_epoch,
                   completed=(since_improved >= args.patience
                              or stop_after >= args.epochs),
                   backend=backend, provenance=provenance())
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"[RESULT] {json.dumps({k: v for k, v in summary.items() if k != 'provenance'})}")
    return summary


if __name__ == "__main__":
    main()
