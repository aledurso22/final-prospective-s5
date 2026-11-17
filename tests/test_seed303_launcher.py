"""Allocation wiring only: no dataset, GPU, or training computation required."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NAME = "allocation_s5_seed303_four_gpu.sh"
ARMS = ("native_matched_s5", "generalized_prospective_s5",
        "calibrated_old_s5", "calibrated_prediction_s5")


@pytest.fixture
def allocation(tmp_path):
    """An isolated committed checkout and fake interpreter for shell guards."""
    repo = tmp_path / "repo"
    scripts = repo / "bin/run_experiments"
    scripts.mkdir(parents=True)
    for name in (NAME, "cluster_env.sh"):
        shutil.copy2(ROOT / "bin/run_experiments" / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
                   check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    interpreter = venv / "bin/python"
    interpreter.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
a = sys.argv[1:]
if a and a[0] == "-":
    if a[4].endswith("preflight.json"):
        pathlib.Path(a[4]).write_text("{}")
        sys.exit(0)
    os.execv(sys.executable, [sys.executable] + a)
module = a[a.index("-m") + 1]
out = pathlib.Path(a[a.index("--out") + 1])
if module.endswith("gpu_telemetry"):
    out.write_text("{}")
    sys.exit(0)
arm = a[a.index("--arm") + 1]
if "--smoke" in a:
    smoke = {"status": "SMOKE_PASS", "telemetry_check": {
        "legacy_seconds": .01, "fused_seconds": .001, "values_agree": True}}
    prod = {"scan_implementation": None if arm == "native_matched_s5" else "factored",
            "x64_enabled": False, "initial_poles": {"production_precision": {
                "stable": 384, "numerical_boundary": 0, "unstable": 0,
                "min_margin_production": .0001}}}
    if os.environ.get("FAIL_SMOKE") == arm:
        smoke["status"] = "SMOKE_FAIL"
    (out / "smoke_result.json").write_text(json.dumps(smoke))
    (out / "production_check.json").write_text(json.dumps(prod))
else:
    (out / "task_result.json").write_text("{}")
    if os.environ.get("FAIL_TRAIN") == arm:
        sys.exit(7)
''')
    interpreter.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    env = dict(os.environ, PROSPECTIVE_REPO=str(repo), PROSPECTIVE_VENV=str(venv),
               PROSPECTIVE_RUNS=str(tmp_path / "runs"), S5_THREE_ARM_DATA=str(data),
               EXPECTED_COMMIT=commit, SLURM_JOB_ID="1234", CUDA_VISIBLE_DEVICES="0,1,2,3",
               JAX_ENABLE_X64="0", DRY_RUN="1", STAGE="train", RUN_ID="test")
    # Keep fixture behavior independent of the user's launch-shell settings.
    for key in ("OUT_ROOT", "IMPLEMENTATION", "FAIL_SMOKE", "FAIL_TRAIN"):
        env.pop(key, None)
    return repo, scripts / NAME, env


def run(allocation, **overrides):
    repo, script, env = allocation
    return subprocess.run(["bash", str(script)], cwd=repo, env={**env, **overrides},
                          text=True, capture_output=True, timeout=20)


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "bin/run_experiments" / NAME)], check=True)


def test_dry_run_has_four_smokes_then_four_training_tasks(allocation):
    r = run(allocation)
    assert r.returncode == 0, r.stderr
    calls = [line for line in r.stdout.splitlines() if " -m experiments.s5_three_arm_full.runner " in line]
    assert len(calls) == 8
    for i, arm in enumerate(ARMS):
        for offset in (0, 4):
            assert f"--arm {arm} --seed 303 --epochs 15 --implementation factored" in calls[offset + i]
        assert "--smoke" in calls[i]
        assert "--smoke" not in calls[4 + i]
        caches = [line.strip() for line in r.stdout.splitlines()
                  if f"CUDA_VISIBLE_DEVICES={i} " in line]
        assert len(caches) == 2 and caches[0] == caches[1]
    assert "epochs / batch: 15 / 16" in r.stdout
    assert "s5-timescale-seed303/test" in r.stdout
    assert not (Path(allocation[2]["PROSPECTIVE_RUNS"])).exists()


@pytest.mark.parametrize("changes, message", [
    ({"CUDA_VISIBLE_DEVICES": "0,1,2"}, "need 4 distinct GPU tokens"),
    ({"CUDA_VISIBLE_DEVICES": "0,1,2,2"}, "need 4 distinct GPU tokens"),
    ({"JAX_ENABLE_X64": "1"}, "production training is float32"),
    ({"EXPECTED_COMMIT": "0" * 40}, "is not"),
    ({"STAGE": "test"}, "STAGE must be smoke or train"),
])
def test_guards_before_launch(allocation, changes, message):
    r = run(allocation, **changes)
    assert r.returncode != 0
    assert message in r.stderr
    assert "DRY_RUN child" not in r.stdout


def test_dirty_checkout_refused(allocation):
    allocation[1].write_text(allocation[1].read_text() + "\n# local edit\n")
    r = run(allocation)
    assert r.returncode != 0 and "refusing a modified worktree" in r.stderr


def test_existing_output_refused(allocation):
    (Path(allocation[2]["PROSPECTIVE_RUNS"]) / "s5-timescale-seed303/test").mkdir(parents=True)
    r = run(allocation, DRY_RUN="0")
    assert r.returncode != 0 and "run root exists" in r.stderr


def test_smoke_failure_blocks_every_training_task(allocation):
    r = run(allocation, DRY_RUN="0", FAIL_SMOKE=ARMS[2])
    assert r.returncode != 0, r.stdout
    assert "smoke gate" in r.stderr
    root = Path(allocation[2]["PROSPECTIVE_RUNS"]) / "s5-timescale-seed303/test"
    assert len(list(root.glob("smoke/*/303/smoke_result.json"))) == 4
    assert not list(root.glob("*/303/task_result.json"))


def test_nonzero_training_exit_fails_even_if_result_exists(allocation):
    r = run(allocation, DRY_RUN="0", FAIL_TRAIN=ARMS[2])
    assert r.returncode != 0, r.stdout
    assert "INCOMPLETE:" in r.stderr and "exit 7" in r.stderr
    root = Path(allocation[2]["PROSPECTIVE_RUNS"]) / "s5-timescale-seed303/test"
    assert len(list(root.glob("*/303/task_result.json"))) == 4


def test_smoke_only_stops_after_gate(allocation):
    r = run(allocation, DRY_RUN="0", STAGE="smoke")
    assert r.returncode == 0, r.stderr
    assert "Nothing trained" in r.stdout
    root = Path(allocation[2]["PROSPECTIVE_RUNS"]) / "s5-timescale-seed303/test"
    assert json.loads((root / "smoke_gate.json").read_text())["failures"] == []
    assert not list(root.glob("*/303/task_result.json"))
