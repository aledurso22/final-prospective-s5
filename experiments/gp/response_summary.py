"""Read the learned response leaves from a saved run and summarize them.

Read-only. Restores the checkpoint, reads log_response_gamma/log_response_rho,
applies the same clip the forward pass applies, and reports the distributions
of gamma_n, rho, the DERIVED mu and the physical components they realize.

    python -m experiments.gp.response_summary <run_dir> [--checkpoint best]
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from s5 import constrained_response as CR                         # noqa: E402
from s5.checkpointing import restore_checkpoint                   # noqa: E402
from s5.rawat_s5 import (LOG_GAMMA_BOUNDS, LOG_RHO_BOUNDS,        # noqa: E402
                         RESPONSE_PARAM_NAMES)
from experiments.gp import rawat_benchmark as RB                  # noqa: E402
import dataloaders.speech_commands10 as SC                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "config.json")) as fh:
        blob = json.load(fh)
    cfg = blob["config"]
    spe = blob.get("extra", {}).get("steps_per_epoch", 843)

    class A:
        pass
    a = A()
    for k, v in cfg.items():
        setattr(a, k, v)
    model = RB.build_model(a.arm, a.d_model, a.ssm_size, a.n_layers, True)
    dummy = jnp.zeros((2, SC.N_FRAMES, SC.N_MFCC))
    variables = model.init({"params": jax.random.PRNGKey(a.seed),
                            "dropout": jax.random.PRNGKey(a.seed + 1)},
                           dummy, jnp.ones((2, SC.N_FRAMES)), None)
    tx, _, _ = RB.make_optimizer(a, spe)
    state = RB.TrainState.create(apply_fn=model.apply,
                                 params=variables["params"], tx=tx,
                                 batch_stats=variables.get("batch_stats"))
    state, _, meta = restore_checkpoint(args.run_dir, args.checkpoint, state,
                                        state.batch_stats)

    from flax.traverse_util import flatten_dict
    flat = {"/".join(k): onp.asarray(v)
            for k, v in flatten_dict(state.params).items()}
    init = {"/".join(k): onp.asarray(v)
            for k, v in flatten_dict(variables["params"]).items()}
    layers = sorted({n.rsplit("/", 1)[0] for n in flat
                     if n.rsplit("/", 1)[-1] in RESPONSE_PARAM_NAMES})
    if not layers:
        print(f"[*] {args.arm if hasattr(args,'arm') else cfg['arm']}: no "
              f"response leaves (fixed-response arm)")
        return

    print(f"[*] {cfg['arm']}  checkpoint {args.checkpoint} "
          f"(epoch {meta.get('epoch')})")
    allg, allr = [], []
    out = dict(arm=cfg["arm"], checkpoint=args.checkpoint,
               epoch=meta.get("epoch"), layers={})
    for pre in layers:
        eta = onp.clip(flat[f"{pre}/log_response_gamma"], *LOG_GAMMA_BOUNDS)
        zet = onp.clip(flat[f"{pre}/log_response_rho"], *LOG_RHO_BOUNDS)
        eta0 = init[f"{pre}/log_response_gamma"]
        zet0 = init[f"{pre}/log_response_rho"]
        g_n, rho = onp.exp(eta), onp.exp(zet)
        mu = CR.T_HORIZON * g_n * rho
        allg.append(g_n); allr.append(rho)
        moved = dict(
            gamma_max_abs_change=float(onp.max(onp.abs(eta - eta0))),
            rho_max_abs_change=float(onp.max(onp.abs(zet - zet0))),
            n_at_gamma_bound=int(onp.sum((eta <= LOG_GAMMA_BOUNDS[0] + 1e-9)
                                         | (eta >= LOG_GAMMA_BOUNDS[1] - 1e-9))),
            n_at_rho_bound=int(onp.sum((zet <= LOG_RHO_BOUNDS[0] + 1e-9)
                                       | (zet >= LOG_RHO_BOUNDS[1] - 1e-9))))
        s = CR.summarize(g_n, rho)
        out["layers"][pre] = dict(summary=s, movement=moved)
        print(f"  {pre}")
        print(f"    gamma_n  min {s['gamma_n']['min']:.4f}  med "
              f"{s['gamma_n']['median']:.4f}  max {s['gamma_n']['max']:.4f}"
              f"   (init 1.0)")
        print(f"    rho      min {s['rho']['min']:.4f}  med "
              f"{s['rho']['median']:.4f}  max {s['rho']['max']:.4f}"
              f"   (init 0.75)")
        print(f"    mu       min {s['mu']['min']:.4f}  med "
              f"{s['mu']['median']:.4f}  max {s['mu']['max']:.4f}"
              f"   (init 3.75)")
        print(f"    moved: |dlog gamma| max {moved['gamma_max_abs_change']:.4f}"
              f"  |dlog rho| max {moved['rho_max_abs_change']:.4f}"
              f"  at bounds: {moved['n_at_gamma_bound']}/"
              f"{moved['n_at_rho_bound']}")
    g_all = onp.concatenate(allg); r_all = onp.concatenate(allr)
    tot = CR.summarize(g_all, r_all)
    out["all_layers"] = tot
    ident = CR.check_identities(g_all, r_all, tol=1e-9)
    out["component_identities"] = ident
    print("  --- all layers ---")
    for k in ("gamma_n", "rho", "mu", "tau_d", "c_d", "G_s", "h", "g_L"):
        v = tot[k]
        print(f"    {k:9s} min {v['min']:10.5f} med {v['median']:10.5f} "
              f"max {v['max']:10.5f} std {v['std']:9.5f}")
    print(f"    component identities hold: {ident['passed']} "
          f"(max residual {ident['max_residual']:.2e}, all positive "
          f"{ident['all_positive']})")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"  wrote {args.out}")
    print("RESPONSE_SUMMARY_DONE")


if __name__ == "__main__":
    main()
