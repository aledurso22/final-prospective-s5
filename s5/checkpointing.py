"""Opt-in checkpointing and full-precision JSON metrics for S5 runs.

Design constraints, all deliberate:

* **Opt-in only.** Nothing here runs unless `--checkpoint_dir` is given, so the
  frozen one-epoch plain configuration and its historical results are
  unchanged.
* **No randomness consumed.** Serialization never splits or draws a PRNG key,
  so enabling checkpointing cannot alter a training trajectory.
* **Full precision.** Metrics are written as native Python floats via json, not
  rounded console text, and not via any wandb summary file (offline wandb 0.29
  does not write one).

DECLARED RESUME BOUNDARY
------------------------
Restore is supported and checked **at an epoch boundary, in-process, on the
same host and backend**. Under those conditions a restored state reproduces a
fixed-batch evaluation and the next optimizer update exactly.

DEFERRED, and explicitly NOT part of Milestone A:
  * a wired epoch-resume entrypoint (`--resume_from`) does not exist; this
    module loads and stores state, it does not restart the training loop;
  * the training RNG and the dataloader's RNG state are not captured - only
    the seed is recorded, and a seed is not a current state;
  * training-loop scheduling/selection counters (best_acc, early-stop count,
    lr step) are not captured.

NOT claimed, and not checked:
  * mid-epoch resume (the dataloader's within-epoch position is not captured);
  * cross-process or cross-host continuation, which additionally requires
    deterministic kernels (see F-002: set
    XLA_FLAGS=--xla_gpu_deterministic_ops=true);
  * resume across a different JAX/CUDA build.
"""

import datetime
import json
import os
import subprocess

import jax
import numpy as np
from flax import serialization


def _git(*args, cwd=None):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def provenance(repo_root=None):
    """Commit, dirty-tree identity, host, backend, devices, timestamp."""
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    status = _git("status", "--porcelain", cwd=root) or ""
    # `git diff` alone sees neither staged nor untracked content, so its hash
    # cannot identify every dirty tree. Hash the WORKING TREE against HEAD
    # (includes staged) and add the bytes of every untracked file.
    import hashlib
    h = hashlib.sha256()
    h.update((_git("diff", "HEAD", cwd=root) or "").encode())
    untracked = [ln[3:] for ln in status.splitlines() if ln.startswith("??")]
    for rel in sorted(untracked):
        full = os.path.join(root, rel)
        h.update(rel.encode())
        if os.path.isfile(full):
            with open(full, "rb") as fh:
                h.update(fh.read())
        elif os.path.isdir(full):
            for dirpath, _, names in os.walk(full):
                for nm in sorted(names):
                    fp = os.path.join(dirpath, nm)
                    h.update(os.path.relpath(fp, root).encode())
                    with open(fp, "rb") as fh:
                        h.update(fh.read())
    return dict(
        timestamp=datetime.datetime.now().astimezone().isoformat(),
        commit=_git("rev-parse", "HEAD", cwd=root),
        branch=_git("rev-parse", "--abbrev-ref", "HEAD", cwd=root),
        dirty=bool(status),
        dirty_files=status.splitlines(),
        untracked_files=untracked,
        # covers unstaged + staged + untracked content
        dirty_manifest_sha256=h.hexdigest() if status else None,
        hostname=os.uname().nodename,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        jax_version=jax.__version__,
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        xla_flags=os.environ.get("XLA_FLAGS"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def make_run_dir(base, tag="run"):
    """Genuinely unique run directory.

    A timestamp + pid name collides when the same tag is created twice within
    one second in one process. `mkdtemp` creates the directory atomically and
    never returns an existing one.
    """
    import tempfile
    os.makedirs(base, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tempfile.mkdtemp(prefix=f"{stamp}-{tag}-", dir=base)


def _to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def write_config(run_dir, args, extra=None):
    payload = dict(config={k: _to_jsonable(v) for k, v in vars(args).items()},
                   provenance=provenance())
    if extra:
        payload["extra"] = {k: _to_jsonable(v) for k, v in extra.items()}
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_to_jsonable)
    return path


def append_metrics(run_dir, record):
    """Append one full-precision metrics record (JSON Lines)."""
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps({k: _to_jsonable(v) for k, v in record.items()},
                            default=_to_jsonable) + "\n")
    return path


def save_checkpoint(run_dir, name, state, epoch, step, config=None,
                    batch_stats=None, data_seed=None, notes=None):
    """Serialize params, optimizer state, batch stats, position and config.

    Uses flax msgpack for the pytrees. Does NOT touch any PRNG.
    """
    os.makedirs(run_dir, exist_ok=True)
    payload = dict(params=state.params, opt_state=state.opt_state,
                   step=np.asarray(state.step))
    if batch_stats is not None:
        payload["batch_stats"] = batch_stats
    blob = serialization.to_bytes(payload)
    ckpt = os.path.join(run_dir, f"{name}.msgpack")
    with open(ckpt, "wb") as fh:
        fh.write(blob)
    meta = dict(name=name, epoch=int(epoch), step=int(step),
                data_seed=None if data_seed is None else int(data_seed),
                has_batch_stats=batch_stats is not None,
                resume_boundary="epoch; in-process; same host and backend",
                notes=notes, provenance=provenance())
    if config is not None:
        meta["config"] = {k: _to_jsonable(v) for k, v in vars(config).items()}
    with open(os.path.join(run_dir, f"{name}.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=_to_jsonable)
    return ckpt


def restore_checkpoint(run_dir, name, state, batch_stats=None):
    """Restore into an existing state object of the right structure."""
    with open(os.path.join(run_dir, f"{name}.msgpack"), "rb") as fh:
        blob = fh.read()
    target = dict(params=state.params, opt_state=state.opt_state,
                  step=np.asarray(state.step))
    if batch_stats is not None:
        target["batch_stats"] = batch_stats
    restored = serialization.from_bytes(target, blob)
    # `step` must be restored too: optax schedules and bias corrections read it,
    # and leaving it at 0 silently rewinds the optimizer's notion of time.
    new_state = state.replace(params=restored["params"],
                              opt_state=restored["opt_state"],
                              step=int(np.asarray(restored["step"])))
    with open(os.path.join(run_dir, f"{name}.meta.json")) as fh:
        meta = json.load(fh)
    return new_state, restored.get("batch_stats"), meta
