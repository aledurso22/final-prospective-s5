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

WHAT THIS MODULE DOES, PRECISELY
--------------------------------
It **serializes and restores training state**: parameters, optimizer state,
mutable batch statistics and `TrainState.step`.

Since the 2026-09-15 cluster brief it ALSO supports loop resume, through the
optional `loop_state` argument to `save_checkpoint`:

  * epoch/step counters, the dropout RNG key, and the selection and
    early-stopping counters are stored;
  * data order does not need an RNG snapshot because it is a pure function of
    `(data_seed, epoch)` - see `dataloaders.speech_commands10.epoch_batches` -
    so restarting an interrupted epoch reproduces its permutation exactly. A
    RESUMED EPOCH RESTARTS AT ITS FIRST BATCH; mid-epoch batch position is not
    restored, and the meta records this;
  * checkpoint writes are ATOMIC (temp file plus `os.replace`), so a killed
    job cannot leave a truncated file that later loads as garbage.

`meta["resumable"]` distinguishes the two cases. A checkpoint written WITHOUT
`loop_state` is still serialization-only, and its `scope` field says so.

What is checked: given externally supplied inputs and randomness, a restored
state reproduces a fixed-batch evaluation and the next optimizer update
exactly, in-process, on the same host and backend.

STILL NOT CAPTURED, stated rather than implied:
  * mid-epoch batch position (a resume replays the interrupted epoch);
  * any RNG consumed outside the recorded dropout key;
  * cross-host or cross-backend bitwise equality.

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
    """Convert to JSON-safe types, RECURSING into containers.

    The non-recursive version stringified nested dicts, so a resumable
    checkpoint's `loop_state["best"]` came back as the text "{'accuracy': ...}"
    and the resume path crashed with `TypeError: string indices must be
    integers`. Anything that survives a save/restore round trip has to recurse.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
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
                    batch_stats=None, data_seed=None, notes=None,
                    loop_state=None):
    """Serialize params, optimizer state, batch stats, position and config.

    Uses flax msgpack for the pytrees. Writes ATOMICALLY: a temporary file in
    the same directory followed by `os.replace`, so a job killed mid-write
    cannot leave a half-written checkpoint that later loads as garbage.

    `loop_state` carries everything a training loop needs to continue rather
    than merely to re-evaluate: epoch/step counters, the dropout RNG, the data
    order identity, the early-stopping and selection counters. Pass it and the
    recorded scope changes from "serialization only" to "resumable".
    """
    os.makedirs(run_dir, exist_ok=True)
    payload = dict(params=state.params, opt_state=state.opt_state,
                   step=np.asarray(state.step))
    if batch_stats is not None:
        payload["batch_stats"] = batch_stats
    if loop_state is not None and "rng" in loop_state:
        payload["rng"] = np.asarray(loop_state["rng"])
    blob = serialization.to_bytes(payload)
    ckpt = os.path.join(run_dir, f"{name}.msgpack")
    _atomic_write(ckpt, blob)
    resumable = loop_state is not None
    meta = dict(name=name, epoch=int(epoch), step=int(step),
                data_seed=None if data_seed is None else int(data_seed),
                has_batch_stats=batch_stats is not None,
                resumable=resumable,
                loop_state=(None if loop_state is None else
                            {k: _to_jsonable(v) for k, v in loop_state.items()
                             if k != "rng"}),
                scope=(("resumable training state: params, optimizer state, "
                        "batch stats, step/epoch counters, dropout RNG, data "
                        "order identity and selection/early-stopping counters. "
                        "Data order is a pure function of (data_seed, epoch), "
                        "so no dataloader RNG has to be reconstructed.")
                       if resumable else
                       ("state serialization only; NOT a training-loop resume. "
                        "Reproduces a fixed-batch eval and the next update given "
                        "externally supplied inputs and randomness, in-process, "
                        "same host and backend.")),
                notes=notes, provenance=provenance())
    if config is not None:
        meta["config"] = {k: _to_jsonable(v) for k, v in vars(config).items()}
    _atomic_write(os.path.join(run_dir, f"{name}.meta.json"),
                  json.dumps(meta, indent=2, default=_to_jsonable).encode())
    return ckpt


def _atomic_write(path, blob):
    """Write via a same-directory temp file and rename. Never a partial file."""
    import tempfile
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def checkpoint_exists(run_dir, name):
    return os.path.exists(os.path.join(run_dir, f"{name}.msgpack"))


def restore_checkpoint(run_dir, name, state, batch_stats=None, rng=None):
    """Restore into an existing state object of the right structure.

    Returns `(state, batch_stats, meta)`. When the checkpoint was written with
    a `loop_state`, `meta["loop_state"]` carries the counters and
    `meta["rng"]` the restored dropout key, so a caller can continue the loop
    rather than only re-evaluate.
    """
    with open(os.path.join(run_dir, f"{name}.msgpack"), "rb") as fh:
        blob = fh.read()
    target = dict(params=state.params, opt_state=state.opt_state,
                  step=np.asarray(state.step))
    if batch_stats is not None:
        target["batch_stats"] = batch_stats
    if rng is not None:
        target["rng"] = np.asarray(rng)
    restored = serialization.from_bytes(target, blob)
    # `step` must be restored too: schedules and any step-dependent logic read
    # it, and leaving it at 0 silently rewinds the training loop's notion of
    # time. NOTE: optax Adam keeps its OWN count inside opt_state, so restoring
    # opt_state is what preserves Adam's bias correction; restoring
    # TrainState.step is a separate, also necessary, fix.
    new_state = state.replace(params=restored["params"],
                              opt_state=restored["opt_state"],
                              step=int(np.asarray(restored["step"])))
    with open(os.path.join(run_dir, f"{name}.meta.json")) as fh:
        meta = json.load(fh)
    if "rng" in restored:
        meta = dict(meta, rng=restored["rng"])
    return new_state, restored.get("batch_stats"), meta
