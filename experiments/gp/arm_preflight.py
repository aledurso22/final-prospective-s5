"""Generic arm preflight: two real updates, measured cost, projection.

Used by both the constrained-response and the (superseded) adaptive batches;
the arms are supplied with --arms, so nothing here is specific to either.

GPU only. Does NOT train. Prints a machine-readable projection so the launcher
can decide whether the bounded batch fits the remaining budget.
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import dataloaders.speech_commands10 as SC                        # noqa: E402
from experiments.gp import rawat_benchmark as RB                  # noqa: E402

NEW_ARMS = ("gp_learned_response", "prospective_recurrence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cache", default="/Users/durso/s5-runs/sc10_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--arms", default=",".join(NEW_ARMS))
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    backend = RB.require_gpu(args.allow_cpu)
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    data, manifest = SC.load_splits(args.data_cache, splits=("train", "val"))
    Xtr, Ytr = data["train"]
    n_val = data["val"][0].shape[0]
    steps_per_epoch = Xtr.shape[0] // args.batch
    val_batches = -(-n_val // args.batch)

    idx = onp.sort(onp.random.RandomState(0).permutation(
        Xtr.shape[0])[:args.batch])
    xb = jnp.asarray(onp.asarray(Xtr[idx]))
    yb = jnp.asarray(Ytr[idx])

    rows = []
    for arm in args.arms.split(","):
        arm = arm.strip()
        if not arm:
            continue

        class A:
            pass
        a = A()
        a.arm, a.d_model, a.ssm_size, a.n_layers = arm, 32, 32, 4
        a.batch, a.epochs, a.lr, a.lr_final = args.batch, args.epochs, 1e-3, 1e-6
        a.weight_decay, a.ssm_weight_decay = 1e-4, None
        a.grad_clip, a.label_smoothing, a.seed = 1.0, 0.1, 100
        model, eval_model, state, groups = RB.init_everything(a, steps_per_epoch)
        counts = RB.arm_state_counts(a)
        params = RB.parameter_report(state.params)
        rng = jax.random.PRNGKey(a.seed + 2)

        t0 = time.time()
        state, loss, acc, gn = RB.train_step(state, xb, yb, rng, model,
                                             a.label_smoothing)
        loss.block_until_ready()
        compile_s = time.time() - t0
        t1 = time.time()
        state, loss, acc, gn = RB.train_step(state, xb, yb, rng, model,
                                             a.label_smoothing)
        loss.block_until_ready()
        steady = time.time() - t1

        ok = bool(onp.isfinite(float(loss)) and onp.isfinite(float(gn))
                  and float(gn) > 0.0)
        rows.append(dict(
            arm=arm, params=params, state_counts=counts,
            loss=float(loss), grad_norm=float(gn),
            compile_plus_first_step_s=compile_s,
            steady_state_step_s=steady,
            peak_memory_bytes=RB.peak_memory_bytes(),
            finite_and_learning=ok,
            projected_train_s=steady * steps_per_epoch * args.epochs,
            projected_eval_s=steady * val_batches * args.epochs / 3.0,
            projected_total_s=(compile_s + steady * steps_per_epoch * args.epochs
                               + steady * val_batches * args.epochs / 3.0)))
        print(f"  {arm:22s} step={steady*1000:7.2f} ms  compile={compile_s:5.1f}s"
              f"  loss={float(loss):.4f} gnorm={float(gn):.3e}"
              f"  proj={rows[-1]['projected_total_s']:7.1f}s  ok={ok}")

    total = sum(r["projected_total_s"] for r in rows)
    out = dict(backend=backend, steps_per_epoch=steps_per_epoch,
               val_batches=val_batches, epochs=args.epochs,
               arms=rows, projected_batch_total_s=total,
               all_finite_and_learning=all(r["finite_and_learning"]
                                           for r in rows),
               data=dict(train=manifest["feature_sha256"]["train"],
                         val=manifest["feature_sha256"]["val"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    print(f"PREFLIGHT_OK={out['all_finite_and_learning']}")
    return 0 if out["all_finite_and_learning"] else 5


if __name__ == "__main__":
    sys.exit(main())
