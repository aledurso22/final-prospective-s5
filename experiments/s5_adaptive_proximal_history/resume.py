"""Continue interrupted adaptive/frozen P from a validated full-state checkpoint.

The pinned implementation, original 15-epoch schedule, dropout keys and epoch
order are preserved. Source artifacts are read only. Restored validation and two
discarded updates gate continuation; the checkpoint is restored again afterward.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import jax

from experiments.s5_adaptive_proximal_history import runner as AP
from experiments.s5_gradient_history import runner as H
from experiments.s5_gradient_history import resume as HR
from experiments.s5_three_arm_full import runner as R

SOURCE_COMMIT = "4372acd76c9f53e8d83d4f929d5bbd4745ace3ad"
REPO = Path(__file__).resolve().parents[2]
sha256 = HR.sha256


@lru_cache(maxsize=1)
def original_source_hashes():
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", SOURCE_COMMIT],
        cwd=REPO, text=True).splitlines()
    names = [name for name in names if name.endswith(".py") and
             name.split("/", 1)[0] in {"s5", "dataloaders", "experiments"}]
    return {name: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{name}"], cwd=REPO)).hexdigest()
            for name in names}


def check_source_code(identity):
    if identity.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("source commit is not the pinned adaptive implementation")
    expected = original_source_hashes()
    if identity.get("source_sha256") != expected:
        raise ValueError("source manifest does not match its pinned commit")
    for name, digest in expected.items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"original training implementation changed: {name}")
    return len(expected)


def inspect_source(source_task, data_cache, epoch, arm):
    """CPU-only source/protocol gate; no data arrays or checkpoint are changed."""
    AP.validate_arm(arm)
    source = Path(source_task).resolve()
    identity = HR._json(source / "history_protocol.json")
    seed = identity.get("seed")
    if seed not in (301, 302) or source.name != str(seed) or source.parent.name != arm:
        raise ValueError("source task path, arm and seed disagree")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch < H.EPOCHS:
        raise ValueError("resume epoch must be completed and below the original horizon")
    if (source / "task_result.json").exists():
        raise ValueError("source task is already completed")
    if (source / "failure.json").exists():
        raise ValueError("source has a recorded failure; review before continuation")
    count = check_source_code(identity)
    with AP.configured(arm):
        current = H.protocol_identity(data_cache, seed)
        # New recovery files and the new commit are recorded in current identity;
        # all other protocol fields must reproduce the original experiment.
        expected = {k: v for k, v in current.items() if k not in ("git_commit", "source_sha256")}
        observed = {k: v for k, v in identity.items() if k not in ("git_commit", "source_sha256")}
        if expected != observed:
            differing = sorted(k for k in set(expected) | set(observed)
                               if expected.get(k) != observed.get(k))
            raise ValueError(f"source protocol mismatch: {', '.join(differing)}")
        rows = HR._rows(source / "metrics.jsonl")
        if [row.get("epoch") for row in rows] != list(range(1, epoch + 1)):
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
        if len({row["validation"]["n"] for row in rows}) != 1:
            raise ValueError("source validation counts changed")
        smoke = source.parents[1] / "smoke" / arm / str(seed) / "smoke_result.json"
        H.check_smoke_gate(HR._json(smoke), identity)
    return {"source": source, "identity": identity, "seed": seed,
            "rows": rows, "smoke": smoke, "verified_source_file_count": count}


def _output_path(source, out):
    original_run = source.parents[1]
    if out == original_run or original_run in out.parents or out in source.parents:
        raise ValueError("resume output must be outside the source run")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("resume output must be new or empty")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resume_task(source_task, out, data_cache, epoch=3, *, arm,
                require_gpu=True, preflight_only=False):
    AP.validate_arm(arm)
    source, out = Path(source_task).resolve(), Path(out).resolve()
    _output_path(source, out)
    if jax.config.read("jax_enable_x64"):
        raise ValueError("adaptive continuation requires JAX_ENABLE_X64=0")
    if require_gpu and (jax.default_backend() != "gpu" or len(jax.devices()) != 1):
        raise ValueError("adaptive continuation requires exactly one allocated GPU")
    info = inspect_source(source, data_cache, epoch, arm)
    before = HR._snapshot(source)
    smoke_hash = sha256(info["smoke"])
    out.mkdir(parents=True, exist_ok=True)
    seed = info["seed"]

    def unchanged():
        return HR._snapshot(source) == before and sha256(info["smoke"]) == smoke_hash

    provenance = {
        "source_task": str(source), "source_commit": SOURCE_COMMIT,
        "source_epoch": epoch, "total_schedule_epochs": H.EPOCHS,
        "remaining_epochs": H.EPOCHS - epoch, "arm": arm, "seed": seed,
        "source_files_sha256": before, "source_smoke": str(info["smoke"]),
        "source_smoke_sha256": smoke_hash, "source_protocol": info["identity"],
        "verified_original_source_file_count": info["verified_source_file_count"],
        "checkpoint_state": "parameters including horizons, optimizer moments/counts/rates, BatchNorm, TrainState.step",
        "source_training_seconds": None, "preflight_only": preflight_only,
        "runtime_note": "timings cover this invocation only; original duration was not checkpointed",
        "test_split_read": False,
    }
    try:
        with AP.configured(arm):
            current = H.protocol_identity(data_cache, seed)
            provenance["resume_protocol"] = current
            _write_json(out / "history_protocol.json", current)
            _write_json(out / "resume_provenance.json", provenance)
            data = R.EXPERIMENT_DATA.load_official_raw(data_cache, ("train", "val"))[0]
            steps_per_epoch = len(data["train"][1]) // R.BATCH_SIZE
            expected_step = epoch * steps_per_epoch
            if any(row["validation"]["n"] != len(data["val"][1]) for row in info["rows"]):
                raise ValueError("source validation count disagrees with data cache")
            old_steps = HR._rows(source / "step_metrics.jsonl", allow_incomplete=True)
            completed_steps = [row for row in old_steps if row.get("epoch", epoch + 1) <= epoch]
            if ([row.get("step") for row in completed_steps] != list(range(expected_step))
                    or any(row.get("nonfinite", True) for row in completed_steps)
                    or any(row.get("code_identifier") != arm or row.get("seed") != seed
                           or row.get("epoch") != row["step"] // steps_per_epoch + 1
                           for row in completed_steps)):
                raise ValueError("source completed-epoch updates are missing or nonfinite")
            old_poles = HR._rows(source / "pole_diagnostics.jsonl", allow_incomplete=True)
            completed_poles = [row for row in old_poles if row.get("epoch", epoch + 1) <= epoch]
            if [row.get("epoch") for row in completed_poles] != list(range(epoch + 1)):
                raise ValueError("source completed-epoch pole records are missing")
            raw = (source / f"checkpoint_epoch_{epoch:02d}.msgpack").read_bytes()
            checkpoint_hash = hashlib.sha256(raw).hexdigest()
            with H.registered_history():
                template = R.init_state(arm, seed)
                restored = HR._restore(template, raw, expected_step)
                restored_poles = AP.pole_summary(restored.params, arm)
                AP.horizon_gate(restored_poles, arm)
                if restored_poles["nonfinite_modes"]:
                    raise ValueError("restored checkpoint has nonfinite poles")
                R.record_lifecycle(str(out), "before_resume_preflight", epoch=epoch, step=expected_step)
                print(f"RESUME_PREFLIGHT {arm}/{seed} after epoch {epoch}", flush=True)
                preflight = HR.restored_preflight(template, raw, data, seed, epoch,
                                                  info["rows"][-1]["validation"])
                AP.horizon_gate(preflight["poles_after_discarded_steps"], arm)
                preflight.update(arm=arm, seed=seed, source_task=str(source),
                                 source_epoch=epoch, source_checkpoint_sha256=checkpoint_hash,
                                 restored_poles=restored_poles, source_unchanged=unchanged())
                if not preflight["source_unchanged"]:
                    raise RuntimeError("source artifacts changed during preflight")
                _write_json(out / "resume_preflight.json", preflight)
                provenance.update(restored_step=expected_step,
                                  source_checkpoint_sha256=checkpoint_hash,
                                  discarded_partial_epoch_steps=len(old_steps) - len(completed_steps))
                if preflight_only:
                    provenance["source_unchanged_after_resume"] = True
                    _write_json(out / "resume_provenance.json", provenance)
                    print(f"RESUME_PREFLIGHT_PASS {arm}/{seed}", flush=True)
                    return preflight
                R.bind_process_gpu(str(out))
                # Scheduled updates mutate optimizer hyperparameter dictionaries;
                # discard the checked state and restore all checkpoint bytes again.
                state = HR._restore(template, raw, expected_step)
                for row in info["rows"]:
                    name = f"checkpoint_epoch_{row['epoch']:02d}.msgpack"
                    shutil.copyfile(source / name, out / name)
                HR._write_rows(out / "metrics.jsonl", info["rows"])
                HR._write_rows(out / "step_metrics.jsonl", completed_steps)
                HR._write_rows(out / "pole_diagnostics.jsonl", completed_poles)
                selected = max(info["rows"], key=lambda row: (
                    row["validation"]["accuracy"], -row["validation"]["cross_entropy"], -row["epoch"]))
                previous_best = {
                    "epoch": selected["epoch"], "val": selected["validation"],
                    "selection": (selected["validation"]["accuracy"],
                                  -selected["validation"]["cross_entropy"], -(selected["epoch"] - 1)),
                    "checkpoint": str(out / f"checkpoint_epoch_{selected['epoch']:02d}.msgpack"),
                }
                provenance["selection_before_resume"] = previous_best
                _write_json(out / "resume_provenance.json", provenance)
                R.record_lifecycle(str(out), "before_resumed_training", epoch=epoch, step=expected_step)
                print(f"RESUME_TRAIN {arm}/{seed}: epochs {epoch + 1}..{H.EPOCHS}, step {expected_step}", flush=True)
                result = R.train_arm(arm, seed, data, str(out), state=state,
                                     epochs=H.EPOCHS, start_epoch=epoch, previous_best=previous_best)
                final_raw = (out / f"checkpoint_epoch_{H.EPOCHS:02d}.msgpack").read_bytes()
                final_state = HR._restore(template, final_raw, H.EPOCHS * steps_per_epoch)
                AP.horizon_gate(AP.pole_summary(final_state.params, arm), arm)
                result.update(AP.parameter_counts(final_state.params, arm))
            if not unchanged():
                raise RuntimeError("source artifacts changed during continuation")
            provenance["source_unchanged_after_resume"] = True
            _write_json(out / "resume_provenance.json", provenance)
            extra = R.N_LAYERS * (R.SSM_SIZE_BASE // 2)
            result.update(
                scan_implementation="history_stationary_block", history_alpha=AP.ALPHA,
                proximal_ratio=AP.PROXIMAL_RATIO, trainable_horizon=arm == AP.ARM,
                extra_parameter_count=extra,
                extra_trainable_parameters=extra if arm == AP.ARM else 0,
                extra_frozen_parameters=0 if arm == AP.ARM else extra,
                resumed_from_epoch=epoch, resumed_epochs=H.EPOCHS - epoch,
                resumed_training_seconds=result["training_seconds"],
                training_seconds_scope="resumed epochs only; original timing unavailable",
                source_task=str(source), source_checkpoint_sha256=checkpoint_hash,
                source_unchanged=True, selection_includes_source_epochs=True,
                resume_preflight_passed=True, test_split_read=False)
            _write_json(out / "task_result.json", result)
            print(f"FINISHED {arm}/{seed}: selected epoch {result['selected_epoch']}; "
                  f"accuracy {100*result['validation']['accuracy']:.4f}%", flush=True)
            return result
    except Exception as error:
        _write_json(out / "failure.json", {
            "arm": arm, "seed": seed, "error_type": type(error).__name__,
            "failure": str(error), "source_unchanged": unchanged(),
            "resume_from_epoch": epoch, "preflight_only": preflight_only})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=AP.ARMS, required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    resume_task(args.source_task, args.out, args.data_cache, args.epoch,
                arm=args.arm, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
