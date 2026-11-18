"""Resume the interrupted fixed-alpha history study at an epoch boundary.

This is full-state continuation, not a weights-only warm start. Source files
are read only; all inherited records and new epochs live in a new run. The
original 15-epoch schedule, batch order, dropout keys and selection rule stay
in force. The source GPU smoke is checked against its original identity; a
new restored-state GPU preflight is then performed and discarded.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import jax
import numpy as np
from flax import serialization

from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R

SOURCE_COMMIT = "54bd025ca0d55b9b3e2d3ce184e30c2550ccfd9a"
# This file has the optional continuation hook. Every other source used by
# the interrupted run must still be byte-identical to the source manifest.
RESUME_CHANGED_SOURCES = {"experiments/s5_three_arm_full/runner.py"}
REPRODUCTION_CE_ATOL = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _rows(path, *, allow_incomplete=False):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.endswith("\n"):
                if allow_incomplete:
                    break
                raise ValueError(f"incomplete JSONL record: {path}")
            if line.strip():
                rows.append(json.loads(line))
    return rows


def original_source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", SOURCE_COMMIT],
        cwd=root, text=True).splitlines()
    names = [p for p in names if p.endswith(".py")
             and p.split("/", 1)[0] in {"s5", "dataloaders", "experiments"}]
    return {p: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{p}"], cwd=root)).hexdigest()
            for p in names}


def _check_source_code(identity):
    if identity.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("source commit is not the pinned interrupted history implementation")
    expected = original_source_hashes()
    if identity.get("source_sha256") != expected:
        raise ValueError("source manifest does not match its pinned git commit")
    root = Path(__file__).resolve().parents[2]
    changed = {}
    for name, digest in expected.items():
        current = sha256(root / name)
        if current != digest:
            if name not in RESUME_CHANGED_SOURCES:
                raise ValueError(f"history training implementation changed: {name}")
            changed[name] = {"source_sha256": digest, "resume_sha256": current}
    return changed


def inspect_source(source_task, data_cache, epoch):
    source = Path(source_task).resolve()
    identity = _json(source / "history_protocol.json")
    seed = identity.get("seed")
    expected = {"arm": H.ARM, "alpha": H.ALPHA, "epochs": H.EPOCHS,
                "batch_size": R.BATCH_SIZE, "scan": H.SCAN,
                "jax_enable_x64": False, "jax_version": jax.__version__,
                "data_cache": str(Path(data_cache).resolve()),
                "data_manifest_sha256": sha256(Path(data_cache) / "manifest.json")}
    for name, value in expected.items():
        if identity.get(name) != value:
            raise ValueError(f"source protocol mismatch: {name}")
    if seed not in (301, 302) or source.name != str(seed) or source.parent.name != H.ARM:
        raise ValueError("source task path, arm and seed disagree")
    if not isinstance(epoch, int) or not 1 <= epoch < H.EPOCHS:
        raise ValueError("resume epoch must be completed and below the original horizon")
    if (source / "task_result.json").exists():
        raise ValueError("source task is already completed")
    if (source / "failure.json").exists():
        raise ValueError("source has a recorded failure; review it before continuation")
    changed = _check_source_code(identity)
    rows = _rows(source / "metrics.jsonl")
    if [r.get("epoch") for r in rows] != list(range(1, epoch + 1)):
        raise ValueError("source must contain exactly the completed epochs through resume epoch")
    for row in rows:
        value = row["validation"]
        if (not math.isfinite(value["accuracy"]) or not 0 <= value["accuracy"] <= 1
                or not math.isfinite(value["cross_entropy"]) or value["cross_entropy"] < 0
                or not isinstance(value["n"], int) or value["n"] <= 0):
            raise ValueError("source validation record is invalid")
        checkpoint = source / f"checkpoint_epoch_{row['epoch']:02d}.msgpack"
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise ValueError(f"missing completed-epoch checkpoint: {checkpoint}")
    if len({r["validation"]["n"] for r in rows}) != 1:
        raise ValueError("source validation counts changed")
    smoke = source.parents[1] / "smoke" / H.ARM / str(seed) / "smoke_result.json"
    H.check_smoke_gate(_json(smoke), identity)
    return {"source": source, "identity": identity, "seed": seed,
            "rows": rows, "smoke": smoke, "changed_sources": changed}


def _snapshot(source):
    return {str(p.relative_to(source)): sha256(p)
            for p in sorted(source.rglob("*")) if p.is_file()}


def _write_rows(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _restore(template, raw, expected_step):
    state = serialization.from_bytes(template, raw)
    old_leaves, old_tree = jax.tree_util.tree_flatten(template)
    new_leaves, new_tree = jax.tree_util.tree_flatten(state)
    if old_tree != new_tree or len(old_leaves) != len(new_leaves):
        raise ValueError("checkpoint state structure differs from history template")
    for expected, actual in zip(old_leaves, new_leaves):
        if np.shape(expected) != np.shape(actual):
            raise ValueError("checkpoint parameter/optimizer shape mismatch")
        if hasattr(expected, "dtype") and np.dtype(expected.dtype) != np.asarray(actual).dtype:
            raise ValueError("checkpoint parameter/optimizer precision mismatch")
    if int(state.step) != expected_step:
        raise ValueError("checkpoint update count differs from completed epochs")
    if not R._all_finite(state):
        raise ValueError("checkpoint contains nonfinite model or optimizer state")
    return state


def restored_preflight(template, raw, data, seed, epoch, expected_validation):
    """Read-only validation reproduction, then two discarded scheduled updates."""
    steps_per_epoch = len(data["train"][1]) // R.BATCH_SIZE
    step = epoch * steps_per_epoch
    state = _restore(template, raw, step)
    observed = R.evaluate(state, R.model_for(H.ARM, False), *data["val"])
    exact = observed == expected_validation
    if (observed["n"] != expected_validation["n"]
            or observed["accuracy"] != expected_validation["accuracy"]
            or abs(observed["cross_entropy"] - expected_validation["cross_entropy"]) > REPRODUCTION_CE_ATOL):
        raise ValueError(f"saved checkpoint validation failed reproduction: {observed} vs {expected_validation}")
    train_model = R.model_for(H.ARM, True)
    checks = []
    for offset, indices in enumerate(R.SC.epoch_batches(
            len(data["train"][1]), R.BATCH_SIZE, seed, epoch, drop_last=True)):
        if offset == 2:
            break
        batch = R.batch_arrays(*data["train"], indices)
        state, loss, accuracy, norm, finite = R.train_one_batch_observable(
            state, jax.random.PRNGKey(seed * 1000 + step + offset),
            *batch, train_model, step + offset, steps_per_epoch, H.EPOCHS)
        record = {"step": step + offset, "loss": float(loss),
                  "accuracy": float(accuracy), "gradient_norm": float(norm),
                  "gradients_finite": bool(finite), "state_finite": R._all_finite(state)}
        if (not record["gradients_finite"] or not record["state_finite"]
                or not all(math.isfinite(record[k]) for k in ("loss", "accuracy", "gradient_norm"))):
            raise ValueError("restored-state GPU preflight produced nonfinite values")
        checks.append(record)
    if len(checks) != 2:
        raise ValueError("resume preflight requires two full training batches")
    poles = H.history_pole_summary(state.params)
    if poles["nonfinite_modes"]:
        raise ValueError("restored-state preflight has nonfinite poles")
    return {"status": "RESUME_PREFLIGHT_PASS", "backend": jax.default_backend(),
            "checkpoint_validation": observed, "recorded_validation": expected_validation,
            "validation_exact": exact, "cross_entropy_atol": REPRODUCTION_CE_ATOL,
            "discarded_training_steps": checks, "poles_after_discarded_steps": poles,
            "policy": "finite instability is recorded, not clipped, matching original training"}


def resume_task(source_task, out, data_cache, epoch=9, *, require_gpu=True):
    source, out = Path(source_task).resolve(), Path(out).resolve()
    original_run = source.parents[1]
    if out == original_run or original_run in out.parents or out in source.parents:
        raise ValueError("resume output must be outside the source run")
    if out.exists() and any(out.iterdir()):
        raise ValueError("resume output must be new or empty")
    if jax.config.read("jax_enable_x64"):
        raise ValueError("history continuation requires JAX_ENABLE_X64=0")
    if require_gpu and (jax.default_backend() != "gpu" or len(jax.devices()) != 1):
        raise ValueError("history continuation requires exactly one allocated GPU")
    info = inspect_source(source, data_cache, epoch)
    before = _snapshot(source)
    smoke_hash = sha256(info["smoke"])
    out.mkdir(parents=True, exist_ok=True)
    seed = info["seed"]
    provenance = {"source_task": str(source), "source_commit": SOURCE_COMMIT,
                  "source_epoch": epoch, "remaining_epochs": H.EPOCHS - epoch,
                  "total_schedule_epochs": H.EPOCHS, "seed": seed, "arm": H.ARM,
                  "source_files_sha256": before, "source_smoke": str(info["smoke"]),
                  "source_smoke_sha256": smoke_hash, "source_protocol": info["identity"],
                  "audited_changed_sources": info["changed_sources"],
                  "checkpoint_state": "parameters, optimizer moments/counts/rates, BatchNorm, TrainState.step",
                  "source_training_seconds": None,
                  "runtime_note": "original elapsed training time was not checkpointed; new timing covers resumed epochs only",
                  "test_split_read": False}
    try:
        current_identity = H.protocol_identity(data_cache, seed)
        provenance["resume_protocol"] = current_identity
        (out / "history_protocol.json").write_text(json.dumps(current_identity, indent=2) + "\n")
        (out / "resume_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        data = R.EXPERIMENT_DATA.load_official_raw(data_cache, ("train", "val"))[0]
        steps_per_epoch = len(data["train"][1]) // R.BATCH_SIZE
        if any(r["validation"]["n"] != len(data["val"][1]) for r in info["rows"]):
            raise ValueError("source validation count disagrees with data cache")
        expected_step = epoch * steps_per_epoch
        old_steps = _rows(source / "step_metrics.jsonl", allow_incomplete=True)
        completed_steps = [r for r in old_steps if r.get("epoch", epoch + 1) <= epoch]
        if ([r.get("step") for r in completed_steps] != list(range(expected_step))
                or any(r.get("nonfinite", True) for r in completed_steps)):
            raise ValueError("source completed-epoch update records are missing or nonfinite")
        old_poles = _rows(source / "pole_diagnostics.jsonl", allow_incomplete=True)
        completed_poles = [r for r in old_poles if r.get("epoch", epoch + 1) <= epoch]
        if [r.get("epoch") for r in completed_poles] != list(range(epoch + 1)):
            raise ValueError("source completed-epoch pole records are missing")
        raw = (source / f"checkpoint_epoch_{epoch:02d}.msgpack").read_bytes()
        with H.registered_history():
            R.record_lifecycle(str(out), "before_resume_initialization")
            template = R.init_state(H.ARM, seed)
            print(f"RESUME_PREFLIGHT {H.ARM}/{seed} after epoch {epoch}", flush=True)
            preflight = restored_preflight(template, raw, data, seed, epoch,
                                           info["rows"][-1]["validation"])
            (out / "resume_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
            R.bind_process_gpu(str(out))
            # Reload after the discarded checks. The original scheduler mutates
            # dictionaries inside opt_state; reusing a checked state is unsafe.
            state = _restore(template, raw, expected_step)
            for row in info["rows"]:
                name = f"checkpoint_epoch_{row['epoch']:02d}.msgpack"
                shutil.copyfile(source / name, out / name)
            _write_rows(out / "metrics.jsonl", info["rows"])
            _write_rows(out / "step_metrics.jsonl", completed_steps)
            _write_rows(out / "pole_diagnostics.jsonl", completed_poles)
            selected = max(info["rows"], key=lambda r: (
                r["validation"]["accuracy"], -r["validation"]["cross_entropy"], -r["epoch"]))
            previous_best = {"epoch": selected["epoch"], "val": selected["validation"],
                "selection": (selected["validation"]["accuracy"],
                              -selected["validation"]["cross_entropy"], -(selected["epoch"] - 1)),
                "checkpoint": str(out / f"checkpoint_epoch_{selected['epoch']:02d}.msgpack")}
            provenance["restored_step"] = expected_step
            provenance["selection_before_resume"] = previous_best
            provenance["discarded_partial_epoch_steps"] = len(old_steps) - len(completed_steps)
            R.record_lifecycle(str(out), "before_resumed_training", epoch=epoch, step=expected_step)
            print(f"RESUME_TRAIN {H.ARM}/{seed}: epochs {epoch + 1}..{H.EPOCHS}, step {expected_step}", flush=True)
            result = R.train_arm(H.ARM, seed, data, str(out), state=state,
                                 epochs=H.EPOCHS, start_epoch=epoch, previous_best=previous_best)
        if _snapshot(source) != before or sha256(info["smoke"]) != smoke_hash:
            raise RuntimeError("source artifacts changed during continuation")
        provenance["source_unchanged_after_resume"] = True
        (out / "resume_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        result.update(seed=seed, scan_implementation="history_stationary_block", history_alpha=H.ALPHA,
                      resumed_from_epoch=epoch, resumed_epochs=H.EPOCHS - epoch,
                      resumed_training_seconds=result["training_seconds"],
                      training_seconds_scope="resumed epochs only; original timing unavailable",
                      source_task=str(source), source_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                      source_unchanged=True, selection_includes_source_epochs=True,
                      resume_preflight_passed=True, test_split_read=False)
        (out / "task_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as error:
        (out / "failure.json").write_text(json.dumps({"arm": H.ARM, "seed": seed,
            "error_type": type(error).__name__, "failure": str(error),
            "source_unchanged": _snapshot(source) == before,
            "resume_from_epoch": epoch}, indent=2) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--epoch", type=int, default=9)
    args = parser.parse_args(argv)
    resume_task(args.source_task, args.out, args.data_cache, args.epoch)


if __name__ == "__main__":
    main()
