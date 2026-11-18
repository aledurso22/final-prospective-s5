"""Measured full-shape memory and steady-state step time for one arm, on the GPU.

Runs the production training update (train_one_batch_observable: batch 16,
16,000 tokens, six layers, the arm's real recurrence) on synthetic inputs for
a few steps from the production float32 initialization, and records the
device allocator's peak bytes and the per-step wall time after compilation.
No dataset is read and no checkpoint is written; the state is discarded.
Compiler buffer accounting is recorded separately by ``runner --profile``.
"""
import argparse
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_proximal_inertia import runner as PIR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-gpu", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"FAIL: refusing existing output {out}")
    if jax.config.read("jax_enable_x64"):
        raise SystemExit("FAIL: production precision requires JAX_ENABLE_X64=0")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise SystemExit("FAIL: GPU backend required")
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(R.BATCH_SIZE, R.SEQ_LEN, R.INPUT_DIM)).astype(np.float32))
    y = jnp.asarray(np.arange(R.BATCH_SIZE) % 10)
    steps_per_epoch = 1000
    times, losses = [], []
    with PIR.configured(args.arm), H.registered_history():
        state = R.init_state(args.arm, args.seed)
        model = R.model_for(args.arm, True)
        for i in range(args.steps):
            start = time.perf_counter()
            state, loss, acc, gnorm, finite = R.train_one_batch_observable(
                state, jax.random.PRNGKey(i), x, y, model, i, steps_per_epoch, 15)
            loss = float(jax.block_until_ready(loss))
            times.append(time.perf_counter() - start)
            losses.append(loss)
            if not (math.isfinite(loss) and bool(finite)):
                raise SystemExit(f"FAIL: nonfinite update at step {i}")
    stats = jax.devices()[0].memory_stats() or {}
    steady = times[2:] if len(times) > 2 else times[1:]
    report = {"arm": args.arm, "mass_ratio": PIR.ARMS[args.arm], "backend": jax.default_backend(),
              "device": str(jax.devices()[0]), "batch_size": R.BATCH_SIZE,
              "sequence_length": R.SEQ_LEN, "layers": R.N_LAYERS, "steps": args.steps,
              "first_step_seconds_including_compile": times[0],
              "steady_step_seconds": {"median": float(np.median(steady)), "max": float(np.max(steady)),
                                      "n": len(steady)},
              "losses": losses,
              "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
              "bytes_limit": stats.get("bytes_limit"),
              "peak_gib": None if stats.get("peak_bytes_in_use") is None
              else stats["peak_bytes_in_use"] / 2 ** 30,
              "data_read": False, "checkpoint_written": False,
              "scope": "allocator peak for this process (init + compiled updates); synthetic inputs"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("arm", "backend", "peak_gib", "steady_step_seconds")}),
          flush=True)
    return report


if __name__ == "__main__":
    main()
