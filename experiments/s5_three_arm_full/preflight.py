"""Exact-shape, no-dataset preflight for the three production arms."""

import argparse
import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

from experiments.s5_three_arm_full import runner


PREFLIGHT_STAGE_ORDER = (
    "telemetry",
    "normal_compile",
    "normal_steady_state",
)
PREFLIGHT_RESULT_FIELDS = frozenset(
    {
        "scientific_name", "code_identifier", "seed", "batch_size",
        "sequence_length", "stage_order", "telemetry_compile_seconds",
        "normal_compile_seconds", "steady_step_seconds", "peak_vram_bytes",
        "finite_gradients", "finite_state",
    }
)


def _block(value):
    return value.block_until_ready() if hasattr(value, "block_until_ready") else value


def _block_tree(tree):
    return jax.tree_util.tree_map(_block, tree)


def _peak_vram_bytes():
    stats = jax.devices()[0].memory_stats()
    if not stats:
        raise RuntimeError("GPU memory telemetry is unavailable")
    for key in ("peak_bytes_in_use", "peak_bytes_used"):
        if key in stats:
            return int(stats[key])
    raise RuntimeError(f"GPU peak-memory telemetry is unavailable: {stats}")


def _record(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with open(temporary_path, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _validate_result(arm, result):
    if not isinstance(result, dict):
        raise RuntimeError(f"malformed preflight result for {arm}: not an object")
    if set(result) != PREFLIGHT_RESULT_FIELDS:
        raise RuntimeError(
            f"malformed preflight result for {arm}: fields={sorted(result)}")
    if result["code_identifier"] != arm:
        raise RuntimeError(f"preflight result arm mismatch: {result['code_identifier']}")
    if result["scientific_name"] != runner.SCIENTIFIC_NAMES[arm]:
        raise RuntimeError(f"preflight scientific name mismatch for {arm}")
    if result["stage_order"] != list(PREFLIGHT_STAGE_ORDER):
        raise RuntimeError(f"preflight stage order mismatch for {arm}")
    if result["finite_gradients"] is not True or result["finite_state"] is not True:
        raise RuntimeError(f"nonfinite preflight result for {arm}")
    for field in (
        "telemetry_compile_seconds",
        "normal_compile_seconds",
        "steady_step_seconds",
    ):
        value = result[field]
        if not isinstance(value, (int, float)) or not jnp.isfinite(value) or value < 0:
            raise RuntimeError(f"invalid {field} for {arm}: {value}")
    if not isinstance(result["peak_vram_bytes"], int) or result["peak_vram_bytes"] <= 0:
        raise RuntimeError(f"invalid peak VRAM for {arm}")


def _run_all_arms(output_dir):
    results = []
    for arm in runner.ARM_ORDER:
        result_path = os.path.join(output_dir, f"{arm}.json")
        if os.path.exists(result_path):
            os.unlink(result_path)
        command = [
            sys.executable,
            "-m",
            "experiments.s5_three_arm_full.preflight",
            "--out",
            output_dir,
            "--arm",
            arm,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"preflight child failed for {arm}: returncode={completed.returncode}")
        if not os.path.isfile(result_path):
            raise RuntimeError(f"preflight child produced no result for {arm}")
        try:
            with open(result_path) as handle:
                result = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read preflight result for {arm}") from exc
        _validate_result(arm, result)
        results.append(result)
    aggregate = {"test_split_opened": False, "arms": results}
    _record(os.path.join(output_dir, "preflight.json"), aggregate)
    print(json.dumps(results, indent=2))
    return aggregate


def run_arm(arm, output_dir):
    if runner.BATCH_SIZE != 16 or runner.SEQ_LEN != 16000:
        raise RuntimeError("preflight requires batch_size=16 and seq_len=16000")
    state = runner.init_state(arm, 301)
    model = runner.model_for(arm, True)
    x = jnp.ones((16, 16000, 1), dtype=jnp.float32)
    y = jnp.arange(16, dtype=jnp.int32) % 10
    _block_tree(state)
    progress_path = os.path.join(output_dir, f"{arm}.progress.json")
    progress = {"code_identifier": arm, "stage_order": []}
    print(f"[{arm}] telemetry compile+execute", flush=True)
    telemetry_start = time.perf_counter()
    state, loss, gradients_finite = runner.train_one_batch_telemetry(
        state, jax.random.PRNGKey(301), x, y, model, 0, 1)
    _block_tree(state)
    telemetry_compile_seconds = time.perf_counter() - telemetry_start
    progress.update({
        "stage_order": ["telemetry"],
        "telemetry_compile_seconds": telemetry_compile_seconds,
        "finite_gradients": bool(gradients_finite),
    })
    _record(progress_path, progress)
    print(f"[{arm}] telemetry complete in {telemetry_compile_seconds:.3f}s", flush=True)
    if not bool(gradients_finite):
        raise RuntimeError(f"non-finite gradients for {arm}")

    print(f"[{arm}] normal compile+execute", flush=True)
    normal_compile_start = time.perf_counter()
    state, loss = runner.train_one_batch(
        state, jax.random.PRNGKey(302), x, y, model, 1, 1)
    _block_tree(state)
    normal_compile_seconds = time.perf_counter() - normal_compile_start
    progress.update({
        "stage_order": ["telemetry", "normal_compile"],
        "normal_compile_seconds": normal_compile_seconds,
    })
    _record(progress_path, progress)
    print(f"[{arm}] normal compile+execute complete in "
          f"{normal_compile_seconds:.3f}s", flush=True)

    print(f"[{arm}] normal steady-state step", flush=True)
    steady_start = time.perf_counter()
    state, loss = runner.train_one_batch(
        state, jax.random.PRNGKey(303), x, y, model, 2, 1)
    _block_tree(state)
    steady_step_seconds = time.perf_counter() - steady_start
    finite_state = all(bool(jnp.all(jnp.isfinite(value)))
                       for value in jax.tree_util.tree_leaves(state))
    if not finite_state or not bool(jnp.isfinite(loss)):
        raise RuntimeError(f"non-finite update for {arm}")
    progress.update({
        "stage_order": list(PREFLIGHT_STAGE_ORDER),
        "steady_step_seconds": steady_step_seconds,
        "finite_state": finite_state,
    })
    _record(progress_path, progress)
    print(f"[{arm}] steady-state step complete in {steady_step_seconds:.3f}s",
          flush=True)
    result = {"scientific_name": runner.SCIENTIFIC_NAMES[arm],
              "code_identifier": arm, "seed": 301,
              "batch_size": 16, "sequence_length": 16000,
              "stage_order": list(PREFLIGHT_STAGE_ORDER),
              "telemetry_compile_seconds": telemetry_compile_seconds,
              "normal_compile_seconds": normal_compile_seconds,
              "steady_step_seconds": steady_step_seconds,
              "peak_vram_bytes": _peak_vram_bytes(),
              "finite_gradients": bool(gradients_finite),
              "finite_state": finite_state}
    _record(os.path.join(output_dir, f"{arm}.json"), result)
    print(f"[{arm}] complete: {json.dumps(result)}", flush=True)
    return result


def main(args):
    os.makedirs(args.out, exist_ok=True)
    if args.arm is None:
        if len(jax.devices("gpu")) != 1:
            raise SystemExit("preflight requires exactly one visible GPU")
        _run_all_arms(args.out)
        return
    if len(jax.devices("gpu")) != 1:
        raise SystemExit("preflight requires exactly one visible GPU")
    result = run_arm(args.arm, args.out)
    _validate_result(args.arm, result)
    _record(os.path.join(args.out, "preflight.json"),
            {"test_split_opened": False, "arms": [result]})
    print(json.dumps([result], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=runner.ARM_ORDER, default=None)
    main(parser.parse_args())
