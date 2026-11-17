"""History allocation wiring, fail-closed gates and output isolation; no GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAME = "allocation_s5_history_two_gpu.sh"


@pytest.fixture
def allocation(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "bin/run_experiments"
    scripts.mkdir(parents=True)
    for name in (NAME, "cluster_env.sh"):
        shutil.copy2(ROOT / "bin/run_experiments" / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin/python"
    py.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys, types
a = sys.argv[1:]
if a and a[0] == "-":
    code = sys.stdin.read()
    if len(a) > 2 and a[2].endswith("preflight.json"):
        pathlib.Path(a[2]).write_text("{}")
        if os.environ.get("FAIL_PREFLIGHT"): sys.exit(9)
        sys.exit(0)
    if len(a) == 1:
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
        if os.environ.get("FAIL_DEVICE"): sys.exit(8)
        sys.exit(0)
    # The actual history gate is covered in integration tests. Inject only
    # its interface here to exercise this launcher's additional checks.
    package = types.ModuleType("experiments")
    sub = types.ModuleType("experiments.s5_gradient_history")
    h = types.ModuleType("experiments.s5_gradient_history.runner")
    h.ARM = "gradient_history_s5"
    h.protocol_identity = lambda cache, seed: {"seed": seed}
    def gate(report, identity):
        assert report["protocol_identity"] == identity
        assert report["status"] == "SMOKE_PASS"
        assert report["backend"] == "gpu"
        assert report["telemetry_check"]["values_agree"]
    h.check_smoke_gate = gate
    sub.runner = h
    package.s5_gradient_history = sub
    sys.modules.update({"experiments": package, "experiments.s5_gradient_history": sub,
                        "experiments.s5_gradient_history.runner": h})
    sys.argv = ["-"] + a[1:]
    exec(compile(code, "<launcher-gate>", "exec"))
    sys.exit(0)
module = a[a.index("-m") + 1]
out = pathlib.Path(a[a.index("--out") + 1])
if module.endswith("gpu_telemetry"):
    out.write_text("{}")
    sys.exit(0)
seed = int(a[a.index("--seed") + 1])
# This emulates the real history CLI's stronger output refusal. Logs/tmp
# or telemetry pre-created inside a task directory would break the job.
assert not out.exists(), "task output was pre-created"
out.mkdir(parents=True)
pp = {"unstable": 0, "nonfinite": 0, "max_radius_production": .999,
      "max_radius_float64": .999}
poles = {"n_modes": 384, "nonfinite_modes": 0, "production_precision": pp}
if "--smoke" in a:
    smoke = {"status": "SMOKE_PASS", "backend": "gpu", "checkpoint_restored": True,
             "code_identifier": "gradient_history_s5", "seed": seed, "training_steps": 2,
             "validation_accuracy": .5, "validation_cross_entropy": .8,
             "protocol_identity": {"seed": seed}, "scan_implementation": "history_stationary_block",
             "telemetry_check": {"values_agree": True}, "poles_after_smoke_steps": poles}
    prod = {"scan_implementation": "history_stationary_block", "x64_enabled": False,
            "gradients_finite": True, "state_finite": True, "initial_poles": poles,
            "loss": .8, "accuracy": .5, "gradient_norm": 2.}
    corruption = os.environ.get("BAD_SMOKE") if seed == 302 else None
    if corruption == "status": smoke["status"] = "FAIL"
    if corruption == "unstable": pp["unstable"] = 1
    if corruption == "identity": smoke["protocol_identity"] = {"seed": 999}
    if corruption == "x64": prod["x64_enabled"] = True
    if corruption == "nan": prod["gradient_norm"] = float("nan")
    (out / "smoke_result.json").write_text(json.dumps(smoke))
    (out / "production_check.json").write_text(json.dumps(prod))
    if corruption == "exit": sys.exit(7)
else:
    assert "--train" in a and "--smoke-report" in a
    assert pathlib.Path(a[a.index("--smoke-report") + 1]).is_file()
    if os.environ.get("MISSING_RESULT") != str(seed):
        (out / "task_result.json").write_text("{}")
    if os.environ.get("FAIL_TRAIN") == str(seed): sys.exit(7)
''')
    py.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    env = dict(os.environ, PROSPECTIVE_REPO=str(repo), PROSPECTIVE_VENV=str(venv),
               PROSPECTIVE_RUNS=str(tmp_path / "runs"), S5_THREE_ARM_DATA=str(data),
               EXPECTED_COMMIT=sha, SLURM_JOB_ID="1234", CUDA_VISIBLE_DEVICES="4,5",
               JAX_ENABLE_X64="0", DRY_RUN="1", STAGE="train")
    for key in ("HISTORY_RUN_ROOT", "FAIL_PREFLIGHT", "FAIL_DEVICE", "BAD_SMOKE",
                "FAIL_TRAIN", "MISSING_RESULT"):
        env.pop(key, None)
    return repo, scripts / NAME, env


def run(allocation, **overrides):
    repo, script, env = allocation
    return subprocess.run(["bash", str(script)], cwd=repo, env={**env, **overrides},
                          text=True, capture_output=True, timeout=20)


def output_root(allocation):
    return Path(allocation[2]["PROSPECTIVE_RUNS"]) / "s5-history-pilot/job-1234"


def test_shell_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "bin/run_experiments" / NAME)], check=True)


def test_dry_plan_two_smokes_then_two_fresh_trainings(allocation):
    r = run(allocation)
    assert r.returncode == 0, r.stderr
    calls = [s for s in r.stdout.splitlines() if " -m experiments.s5_gradient_history.runner " in s]
    assert len(calls) == 4
    for i, seed in enumerate((301, 302)):
        assert f"--seed {seed} --smoke" in calls[i]
        assert f"--seed {seed} --train --smoke-report" in calls[i + 2]
        caches = [s.split(" cache ", 1)[1] for s in r.stdout.splitlines()
                  if s.startswith("DRY_RUN") and f"seed {seed} " in s]
        assert len(caches) == 2 and caches[0] == caches[1]
    assert not output_root(allocation).exists()


@pytest.mark.parametrize("env,message", [
    ({"CUDA_VISIBLE_DEVICES": "4"}, "exactly 2 distinct"),
    ({"CUDA_VISIBLE_DEVICES": "4,4"}, "exactly 2 distinct"),
    ({"CUDA_VISIBLE_DEVICES": "0,1,2"}, "exactly 2 distinct"),
    ({"JAX_ENABLE_X64": "1"}, "JAX_ENABLE_X64=0"),
    ({"EXPECTED_COMMIT": "wrong"}, "not EXPECTED_COMMIT"),
    ({"STAGE": "test"}, "STAGE must"),
])
def test_allocation_guards(allocation, env, message):
    r = run(allocation, **env)
    assert r.returncode != 0 and message in r.stderr


@pytest.mark.parametrize("bad", ["status", "unstable", "identity", "x64", "nan", "exit"])
def test_one_bad_smoke_blocks_both_trainings(allocation, bad):
    r = run(allocation, DRY_RUN="0", BAD_SMOKE=bad)
    assert r.returncode != 0, r.stdout
    assert not list(output_root(allocation).glob("gradient_history_s5/*/task_result.json"))


def test_success_and_outside_task_logs(allocation):
    r = run(allocation, DRY_RUN="0")
    assert r.returncode == 0, r.stdout + r.stderr
    root = output_root(allocation)
    assert json.loads((root / "smoke_gate.json").read_text())["all_passed"]
    assert len(list(root.glob("gradient_history_s5/*/task_result.json"))) == 2
    assert len(list(root.glob("inventories/*.json"))) == 4
    assert (root / "logs/smoke-301.log").exists()
    assert (root / "logs/train-302.log").exists()


@pytest.mark.parametrize("env", [{"FAIL_PREFLIGHT": "1"}, {"FAIL_DEVICE": "1"},
                                  {"FAIL_TRAIN": "302"}, {"MISSING_RESULT": "302"}])
def test_failed_stage_never_reports_completion(allocation, env):
    r = run(allocation, DRY_RUN="0", **env)
    assert r.returncode != 0
    assert "Completed history pilot" not in r.stdout


def test_smoke_only_and_existing_output_guards(allocation):
    r = run(allocation, DRY_RUN="0", STAGE="smoke")
    assert r.returncode == 0, r.stderr
    assert "Nothing trained" in r.stdout
    assert not list(output_root(allocation).glob("gradient_history_s5/*/task_result.json"))
    r = run(allocation)
    assert r.returncode != 0 and "run root exists" in r.stderr


def test_dirty_checkout_refused(allocation):
    allocation[1].write_text(allocation[1].read_text() + "\n# modified\n")
    r = run(allocation)
    assert r.returncode != 0 and "modified worktree" in r.stderr
