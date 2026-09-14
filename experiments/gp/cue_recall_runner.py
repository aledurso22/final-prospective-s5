"""Runner for the controlled cue/recall experiment.

Full BPTT throughout. Reuses the existing SSM factories, the existing encoder
and the existing optimizer parameter-group rule, so mechanisms differ only in
the SSM.

    python -m experiments.gp.cue_recall_runner --mechanism gp_diagonal \
        --updates 200 --seed 0
"""

import argparse
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

from jax.scipy.linalg import block_diag                                # noqa: E402
from s5.checkpointing import append_metrics, make_run_dir, provenance, \
    save_checkpoint, write_config                                      # noqa: E402
from s5.gp_ssm import init_gp_ssm                                      # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                                # noqa: E402
from s5.tokenwise_model import BatchTokenwiseDualHead                  # noqa: E402
from s5.train_helpers import map_nested_fn                             # noqa: E402
from tasks import cue_recall as T                                      # noqa: E402

#: frozen model shape for this experiment (see docs/GP_EXPERIMENT_PROTOCOL.md)
D_MODEL = 32
SSM_SIZE_BASE = 32
N_LAYERS = 2
BLOCKS = 2                      # HiPPO initialization blocks, NOT response blocks


def ssm_kwargs(d_model=D_MODEL, ssm_size=SSM_SIZE_BASE, blocks=BLOCKS,
               conj_sym=True, clip_eigs=True):
    block = ssm_size // blocks
    Lam, _, _, V, _ = make_DPLR_HiPPO(block)
    if conj_sym:
        block //= 2
        P = ssm_size // 2
    else:
        P = ssm_size
    Lam, V = Lam[:block], V[:, :block]
    Vc = V.conj().T
    Lam = (Lam * jnp.ones((blocks, block))).ravel()
    return dict(H=d_model, P=P, Lambda_re_init=Lam.real,
                Lambda_im_init=Lam.imag, V=block_diag(*([V] * blocks)),
                Vinv=block_diag(*([Vc] * blocks)),
                C_init="trunc_standard_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=conj_sym,
                clip_eigs=clip_eigs, bidirectional=False)


def build_model(mechanism, gp_init_scale, training=True, clip_eigs=True):
    if mechanism == "modal_ssm":
        from s5.modal_ssm import init_modal_ssm
        fn = init_modal_ssm(gp_init_scale=gp_init_scale,
                            **ssm_kwargs(clip_eigs=clip_eigs))
    else:
        kw = dict(mechanism=mechanism, **ssm_kwargs(clip_eigs=clip_eigs))
        if mechanism != "plain":
            kw["gp_init_scale"] = gp_init_scale
        fn = init_gp_ssm(**kw)
    return BatchTokenwiseDualHead(ssm=fn, d_model=D_MODEL, n_layers=N_LAYERS,
                                  n_recall=T.N_PAYLOAD, training=training)


def make_state(model, key, batch, lr, ssm_lr, weight_decay=0.01,
               freeze_response=False):
    """Same parameter-group rule as production, plus an optional frozen
    response group for the control that trains everything EXCEPT t."""
    variables = model.init({"params": key, "dropout": key},
                           batch["x"], jnp.ones(batch["x"].shape[:2]))
    params = variables["params"]

    if freeze_response:
        label = map_nested_fn(
            lambda k, _: "frozen" if k in ("gp_response_raw", "z_raw")
            else ("ssm" if k in ("B", "Lambda_re", "Lambda_im", "log_step",
                                 "norm", "alpha_re", "alpha_im")
                  else "regular"))
    else:
        label = map_nested_fn(
            lambda k, _: "ssm" if k in ("B", "Lambda_re", "Lambda_im",
                                        "log_step", "norm", "gp_response_raw",
                                        "alpha_re", "alpha_im", "z_raw")
            else "regular")

    tx = optax.multi_transform(
        {"frozen": optax.set_to_zero(),
         "ssm": optax.adam(learning_rate=ssm_lr),
         "regular": optax.adamw(learning_rate=lr, weight_decay=weight_decay)},
        label)
    return params, tx, tx.init(params)


def loss_fn(params, model, batch):
    r, c = model.apply({"params": params}, batch["x"],
                       jnp.ones(batch["x"].shape[:2]))
    return T.losses(r, c, batch)["loss"], (r, c)


# Both `model` and `tx` are non-array callables: model is a Flax module and
# `tx` is an optax GradientTransformation (a NamedTuple of functions). Both
# must be static, or jit tries to abstractify them.
@partial(jax.jit, static_argnums=(3, 4))
def _update(params, opt_state, batch, model, tx):
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, model, batch)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, grads


@partial(jax.jit, static_argnums=(1,))
def _eval_batch(params, model, b):
    r, c = model.apply({"params": params}, b["x"],
                       jnp.ones(b["x"].shape[:2]))
    return T.metrics(r, c, b)


def evaluate(params, model, batches):
    acc = None
    for b in batches:
        m = _eval_batch(params, model, b)
        acc = m if acc is None else {k: acc[k] + m[k] for k in m}
    return {k: float(v) / len(batches) for k, v in acc.items()}


def response_stats(params):
    """Initial/final response values and derived quantities, for logging."""
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    out = {}
    for k, v in flat.items():
        if k[-1] in ("gp_response_raw", "z_re", "z_im"):
            t = onp.asarray(jax.nn.softplus(v)) if k[-1] == "gp_response_raw" \
                else onp.asarray(v)
            name = "/".join(k)
            out[name] = dict(mean=float(onp.mean(t)), min=float(onp.min(t)),
                             max=float(onp.max(t)))
    return out


def _peak_memory_bytes():
    """Device peak bytes where the backend exposes it; None on CPU."""
    try:
        st = jax.devices()[0].memory_stats()
        return None if st is None else int(st.get("peak_bytes_in_use", 0)) or None
    except Exception:
        return None


def spectral_stats(params):
    """Per-layer stability diagnostics required by docs/GP_EXPERIMENT_PROTOCOL.md.

    Reports, per SSM layer: how many raw poles the `clip_eigs=True` guard would
    actually move (the clipping statistic), the effective pole real parts
    Re(a_eff) = (sigma - t|a|^2)/|m|^2, and t_i |a_i|. For `modal_ssm` the
    pole is `alpha` directly and there is no prospective coupling, so t|a| is
    not defined and only the pole statistics are reported.
    """
    from flax.traverse_util import flatten_dict
    flat = {"/".join(k): onp.asarray(v) for k, v in flatten_dict(params).items()}
    layers = sorted({k.rsplit("/", 1)[0] for k in flat if k.endswith("Lambda_re")
                     or k.endswith("alpha_re")})
    out = {}
    for pre in layers:
        if pre + "/Lambda_re" in flat:                      # S5-parameterized
            re_raw = flat[pre + "/Lambda_re"]
            im = flat[pre + "/Lambda_im"]
            step = onp.exp(flat[pre + "/log_step"][:, 0])
            lam = onp.minimum(re_raw, -1e-4) + 1j * im      # clip_eigs=True
            a = step * lam
            n_clipped = int(onp.sum(re_raw > -1e-4))
            raw_key = pre + "/gp_response_raw"
            if raw_key in flat:
                t = onp.logaddexp(0.0, flat[raw_key])       # softplus
            else:
                t = onp.zeros_like(onp.real(a))
            m = 1.0 - t * a
            a_eff = a / m
            out[pre] = dict(
                n_modes=int(a.size), n_raw_poles_clipped=n_clipped,
                clipped_fraction=float(n_clipped) / float(a.size),
                eff_pole_re_max=float(onp.max(onp.real(a_eff))),
                eff_pole_re_mean=float(onp.mean(onp.real(a_eff))),
                abs_abar_max=float(onp.max(onp.abs(onp.exp(a_eff)))),
                t_abs_a_max=float(onp.max(t * onp.abs(a))),
                t_abs_a_mean=float(onp.mean(t * onp.abs(a))))
        else:                                                # modal_ssm
            # ModalSSM applies the SAME clip in coefficients() when
            # clip_eigs=True. Read the CLIPPED pole, or this reports a
            # different system than the one that ran (review item R3).
            from s5.modal_ssm import ALPHA_CLIP
            re_raw = flat[pre + "/alpha_re"]
            alpha = onp.minimum(re_raw, ALPHA_CLIP) + 1j * flat[pre + "/alpha_im"]
            n_clipped = int(onp.sum(re_raw > ALPHA_CLIP))
            z = flat[pre + "/z_re"] + 1j * flat[pre + "/z_im"]
            out[pre] = dict(
                n_modes=int(alpha.size), n_raw_poles_clipped=n_clipped,
                clipped_fraction=float(n_clipped) / float(alpha.size),
                eff_pole_re_max=float(onp.max(onp.real(alpha))),
                eff_pole_re_mean=float(onp.mean(onp.real(alpha))),
                abs_abar_max=float(onp.max(onp.abs(onp.exp(alpha)))),
                # z is a free static gain here, NOT a prospective coupling:
                # t|a| is undefined for this arm and |z| is reported instead.
                t_abs_a_max=None, t_abs_a_mean=None,
                abs_z_max=float(onp.max(onp.abs(z))),
                abs_z_mean=float(onp.mean(onp.abs(z))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", default="gp_diagonal")
    ap.add_argument("--gp_init_scale", type=float, default=0.05)
    ap.add_argument("--updates", type=int, default=750)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ssm_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=4)
    ap.add_argument("--freeze_response", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--matmul_precision", default="highest")
    args = ap.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    cfg = T.TaskConfig(); cfg.validate()

    model = build_model(args.mechanism, args.gp_init_scale, training=True)
    dev = [T.generate(T.split_key(args.seed, T.Split.DEV, i), args.batch, cfg)
           for i in range(args.eval_batches)]
    first = T.generate(T.split_key(args.seed, T.Split.TRAIN, 0), args.batch, cfg)

    params, tx, opt_state = make_state(
        model, jax.random.PRNGKey(args.seed), first, args.lr, args.ssm_lr,
        freeze_response=args.freeze_response)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    run_dir = make_run_dir(args.outdir or "results/gp/cue_recall",
                           tag=args.mechanism)
    write_config(run_dir, args, extra=dict(
        n_params=n_params, d_model=D_MODEL, ssm_size_base=SSM_SIZE_BASE,
        n_layers=N_LAYERS, blocks=BLOCKS, seq_len=cfg.seq_len,
        matmul_precision=args.matmul_precision,
        n_state_coords=2 * SSM_SIZE_BASE * N_LAYERS,
        response_initial=response_stats(params),
        spectral_initial=spectral_stats(params)))
    print(f"[*] {args.mechanism}  params={n_params}  run_dir={run_dir}")
    print(f"[*] backend={jax.default_backend()} devices={jax.devices()} "
          f"matmul_precision={args.matmul_precision}")

    initial_response = response_stats(params)
    initial_spectral = spectral_stats(params)
    prev_params = params
    best = dict(joint_bce=float("inf"))
    t0 = time.time()
    for step in range(1, args.updates + 1):
        batch = T.generate(T.split_key(args.seed, T.Split.TRAIN, step),
                           args.batch, cfg)
        prev_params = params
        params, opt_state, loss, grads = _update(params, opt_state, batch,
                                                 model, tx)
        if step % args.eval_every == 0 or step == args.updates:
            m = evaluate(params, model, dev)
            gnorm = float(optax.global_norm(grads))
            unorm = float(optax.global_norm(
                jax.tree_util.tree_map(lambda a, b: a - b, params, prev_params)))
            rec = dict(step=step, train_loss=float(loss), grad_norm=gnorm,
                       update_norm=unorm, wall_s=time.time() - t0, **m)
            append_metrics(run_dir, rec)
            print(f"  step {step:>5}  train {float(loss):.4f}  "
                  f"dev joint {m['joint_bce']:.4f}  recall_bit "
                  f"{m['recall_bit_acc']:.4f}  exact8 {m['recall_exact8_acc']:.4f}"
                  f"  current {m['current_acc']:.4f}")
            if m["joint_bce"] < best["joint_bce"]:
                best = dict(rec)
    summary = dict(mechanism=args.mechanism, n_params=n_params, best=best,
                   n_state_coords=2 * SSM_SIZE_BASE * N_LAYERS,
                   response_initial=initial_response,
                   final_response=response_stats(params),
                   spectral_initial=initial_spectral,
                   spectral_final=spectral_stats(params),
                   peak_memory_bytes=_peak_memory_bytes(),
                   wall_s=time.time() - t0, provenance=provenance(),
                   config=vars(args))
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"[*] best dev joint BCE {best['joint_bce']:.4f} at step {best['step']}")
    print(f"[*] wrote {run_dir}/summary.json")
    return summary


if __name__ == "__main__":
    main()
