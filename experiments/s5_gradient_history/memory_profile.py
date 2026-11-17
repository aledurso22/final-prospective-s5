"""Compile-only production update memory accounting, without loading data.

Uses abstract float32 TrainState/input values and the original training update.
The result is compiler buffer accounting for THIS backend, not measured peak
device memory. Runtime libraries, live initialization buffers and allocator
reservation/fragmentation can make actual peak memory higher. No training step,
checkpoint write, BN update, dataset read or optimizer update is executed.
"""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp

from experiments.s5_three_arm_full import runner as R
from experiments.s5_gradient_history import runner as H
from s5 import train_helpers as TH


ARMS = ("native_matched_s5", "calibrated_old_s5", H.ARM)
KINDS = ("observable", "telemetry")
HISTORY_SCANS = ("parallel", "parallel_reference")


@contextmanager
def history_scan_context(arm, scan):
    """Select only the research implementation, restoring process globals."""
    previous = H.SCAN
    if arm == H.ARM:
        H.SCAN = scan
    try:
        yield
    finally:
        H.SCAN = previous


def abstract_state(arm, seed):
    """The production state shapes, without materializing sequence activations."""
    # Build static SSM initialization arrays outside eval_shape tracing.
    cls = R.model_cls_for(arm)
    rng = jax.random.PRNGKey(seed)
    return jax.eval_shape(lambda: TH.create_train_state(
        cls, rng, padded=False, retrieval=False, in_dim=R.INPUT_DIM,
        bsz=R.BATCH_SIZE, seq_len=R.SEQ_LEN, weight_decay=R.WEIGHT_DECAY,
        batchnorm=True, opt_config="noBCdecay", ssm_lr=R.SSM_LR,
        lr=R.LR, dt_global=False))


def memory_fields(stats):
    keys = ("argument_size_in_bytes", "output_size_in_bytes",
            "alias_size_in_bytes", "temp_size_in_bytes",
            "generated_code_size_in_bytes", "host_argument_size_in_bytes",
            "host_output_size_in_bytes", "host_alias_size_in_bytes",
            "host_temp_size_in_bytes", "host_generated_code_size_in_bytes")
    out = {k: int(getattr(stats, k)) for k in keys if hasattr(stats, k)}
    required = ("argument_size_in_bytes", "output_size_in_bytes",
                "alias_size_in_bytes", "temp_size_in_bytes")
    if all(k in out for k in required):
        out["buffer_accounting_total_bytes"] = (out[required[0]] + out[required[1]]
                                                 - out[required[2]] + out[required[3]])
    return out


def compile_profile(arm, seed=301, kinds=KINDS, history_scan=None):
    if arm not in ARMS or any(k not in KINDS for k in kinds):
        raise ValueError("unsupported arm or update kind")
    if jax.config.read("jax_enable_x64"):
        raise ValueError("memory profile requires production JAX_ENABLE_X64=0")
    history_scan = H.SCAN if history_scan is None else history_scan
    if history_scan not in HISTORY_SCANS:
        raise ValueError("unsupported history scan")
    root = Path(__file__).resolve().parents[2]
    files = ("s5/gradient_history.py", "s5/train_helpers.py", "s5/layers.py",
             "experiments/s5_gradient_history/runner.py")
    source_hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}
    rows = []
    with history_scan_context(arm, history_scan), (H.registered_history() if arm == H.ARM else nullcontext()):
        state = abstract_state(arm, seed)
        model = R.model_for(arm, True)
        args = (state, jax.ShapeDtypeStruct((2,), jnp.uint32),
                jax.ShapeDtypeStruct((R.BATCH_SIZE, R.SEQ_LEN, R.INPUT_DIM), jnp.float32),
                jax.ShapeDtypeStruct((R.BATCH_SIZE,), jnp.int32),
                jax.ShapeDtypeStruct((R.BATCH_SIZE, R.SEQ_LEN), jnp.float32), model, True)
        for kind in kinds:
            fn = TH.train_step_observable if kind == "observable" else TH.train_step_telemetry
            start = time.perf_counter()
            compiled = fn.lower(*args).compile()
            stats = compiled.memory_analysis()
            rows.append({"kind": kind, "compile_seconds": time.perf_counter() - start,
                         "memory": None if stats is None else memory_fields(stats)})
            del compiled
    after = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}
    if after != source_hashes:
        raise RuntimeError("implementation sources changed during memory profiling")
    return {"arm": arm, "seed_for_shape_construction": seed,
            "history_scan": history_scan if arm == H.ARM else None,
            "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
            "jax": jax.__version__, "batch_size": R.BATCH_SIZE,
            "sequence_length": R.SEQ_LEN, "layers": R.N_LAYERS,
            "d_model": R.D_MODEL, "ssm_real_size": R.SSM_SIZE_BASE,
            "x64": False, "training_executed": False, "data_read": False,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_worktree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()),
            "source_sha256": source_hashes,
            "scope": "Compiler buffer accounting on the named backend; not measured peak GPU use or CPU-to-GPU extrapolation.",
            "updates": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=ARMS, default=H.ARM)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    ap.add_argument("--history-scan", choices=HISTORY_SCANS, default="parallel",
                    help="parallel is the exact custom adjoint; parallel_reference is ordinary scan autodiff")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"refusing existing output: {out}")
    result = compile_profile(args.arm, args.seed, args.kinds, args.history_scan)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    for row in result["updates"]:
        memory = row["memory"] or {}
        print(args.arm, row["kind"], result["backend"],
              "compiler total GiB", memory.get("buffer_accounting_total_bytes", 0) / 2**30,
              "temporary GiB", memory.get("temp_size_in_bytes", 0) / 2**30, flush=True)
    return result


if __name__ == "__main__":
    main()
