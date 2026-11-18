"""Execute the shell workflow with fake GPU tasks; never submit a real job."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = "cluster_s5_proximal_memory_mix.sh"
SUBMIT = "submit_s5_proximal_memory_mix.sh"


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "bin/run_experiments"
    scripts.mkdir(parents=True)
    for name in (CLUSTER, SUBMIT, "cluster_env.sh"):
        shutil.copy2(ROOT / "bin/run_experiments" / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    data = tmp_path / "official-cache"
    data.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin/python"
    py.write_text(r'''#!/usr/bin/env python3
import contextlib, json, os, pathlib, sys, types
a = sys.argv[1:]
if a and a[0] == "-u": a = a[1:]
arm = "proximal_memory_mix_s5"
pp = {"unstable": 0, "nonfinite": 0, "stable": 384,
      "max_radius_float64": .9999, "max_radius_production": .9999}
if a[0] == "-":
    code = sys.stdin.read()
    assert code.strip(), "background execution must retain stdin"
    sys.argv = a
    if len(a) == 3 and a[2].endswith("preflight.json"):
        if os.environ.get("FAIL_PREFLIGHT"): sys.exit(8)
        pathlib.Path(a[2]).write_text("{}")
    elif "devices=jax.devices()" in code and "allocator_limit_bytes" in code:
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
        limit = int(os.environ.get("GPU_LIMIT", str(24 * 2**30)))
        pathlib.Path(a[1]).write_text(json.dumps({"backend": "gpu", "device_count": 1,
            "x64": False, "allocator_limit_bytes": limit if limit else None,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}))
    elif "Allocated devices:" in code:
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 2
    elif "M.check_smoke_gate" in code:
        exp = types.ModuleType("experiments")
        h_pkg = types.ModuleType("experiments.s5_gradient_history")
        m_pkg = types.ModuleType("experiments.s5_proximal_memory_mix")
        h = types.SimpleNamespace(protocol_identity=lambda cache, seed: {"seed": seed})
        def gate(report, production, identity):
            assert report["status"] == "SMOKE_PASS"
            assert report["seed"] == identity["seed"]
            assert report["backend"] == "gpu"
            assert report["checkpoint_restored"]
        m = types.SimpleNamespace(ARM=arm, configured=lambda **kw: contextlib.nullcontext(), check_smoke_gate=gate)
        h_pkg.runner = h; m_pkg.runner = m
        sys.modules.update({"experiments": exp, "experiments.s5_gradient_history": h_pkg,
                            "experiments.s5_proximal_memory_mix": m_pkg})
        exec(compile(code, "<smoke-gate>", "exec"))
    else:
        # Exercise the real compiler-memory gate, final-result checks and
        # all-seed gate aggregation, which need only Python's standard library.
        exec(compile(code, "<launcher-check>", "exec"))
    sys.exit(0)
module = a[a.index("-m") + 1]
assert module == "experiments.s5_proximal_memory_mix.runner"
out = pathlib.Path(a[a.index("--out") + 1])
seed = int(a[a.index("--seed") + 1])
kind = next(k for k in ("profile", "smoke", "train") if "--" + k in a)
assert a[a.index("--alpha") + 1] == "1"
assert a[a.index("--fast-ratio") + 1] == "0.1"
assert a[a.index("--slow-ratio") + 1] == "1"
assert abs(float(a[a.index("--slow-weight") + 1]) - 4/9) < 1e-15
assert a[a.index("--proximal-ratio") + 1] == "0.5"
assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
record = {"kind": kind, "seed": seed, "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "out": str(out)}
with open(os.environ["CALLS"], "a") as f: f.write(json.dumps(record) + "\n")
assert not out.exists(), "outputs must be fresh"
if kind == "profile":
    total = int(os.environ.get("COMPILER_BYTES", str(2**30)))
    memory = {"buffer_accounting_total_bytes": total} if total else None
    out.write_text(json.dumps({"arm": arm, "seed_for_shape_construction": seed,
       "backend": os.environ.get("PROFILE_BACKEND", "gpu"), "devices": ["CudaDevice(id=0)"],
       "git_commit": os.environ["EXPECTED_COMMIT"], "git_worktree_dirty": False,
       "history_alpha": 1., "fast_ratio": .1, "slow_ratio": 1., "slow_weight": 4/9, "proximal_ratio": .5,
       "x64": False, "batch_size": 16, "sequence_length": 16000, "layers": 6,
       "updates": [{"kind": k, "memory": memory} for k in ("observable", "telemetry")]}))
    sys.exit(0)
out.mkdir(parents=True)
if kind == "smoke":
    if os.environ.get("FAIL_SMOKE") == str(seed): sys.exit(6)
    if os.environ.get("UNSTABLE_INIT") == str(seed): pp["unstable"] = 1
    (out / "production_check.json").write_text(json.dumps({"code_identifier": arm, "seed": seed,
       "x64_enabled": False, "gradients_finite": True, "state_finite": True,
       "scan_implementation": "history_stationary_block", "initial_poles": {"production_precision": pp}}))
    (out / "smoke_result.json").write_text(json.dumps({"status": "SMOKE_PASS", "seed": seed,
       "backend": "gpu", "checkpoint_restored": True}))
    sys.exit(0)
assert pathlib.Path(a[a.index("--smoke-report") + 1]).is_file()
if os.environ.get("FAIL_TRAIN") == str(seed): sys.exit(7)
if os.environ.get("MISSING_RESULT") != str(seed):
    (out / "task_result.json").write_text(json.dumps({"code_identifier": arm, "seed": seed,
       "epochs_requested": 15, "selected_epoch": 14,
       "validation": {"accuracy": .97, "cross_entropy": .1}}))
''')
    py.chmod(0o755)
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    sbatch = fakebin / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE/args"\ncat > "$CAPTURE/payload"\necho "Submitted batch job 123"\n')
    sbatch.chmod(0o755)
    env = {**os.environ, "EXPECTED_COMMIT": sha, "PROSPECTIVE_VENV": str(venv),
           "PROSPECTIVE_RUNS": str(tmp_path / "runs"), "S5_THREE_ARM_DATA": str(data),
           "SLURM_JOB_ID": "123", "JAX_ENABLE_X64": "0", "STAGE": "train",
           "CUDA_VISIBLE_DEVICES": "GPU-allocated-four,GPU-allocated-seven", "DRY_RUN": "0",
           "CALLS": str(tmp_path / "calls"), "CAPTURE": str(tmp_path),
           "PATH": f"{fakebin}:{os.environ['PATH']}"}
    for key in ("FAIL_PREFLIGHT", "FAIL_SMOKE", "FAIL_TRAIN", "MISSING_RESULT", "UNSTABLE_INIT",
                "COMPILER_BYTES", "GPU_LIMIT", "PROFILE_BACKEND"):
        env.pop(key, None)
    return repo, scripts, env, tmp_path


def run(setup, name=CLUSTER, **changes):
    repo, scripts, env, _ = setup
    return subprocess.run(["bash", str(scripts / name)], cwd=repo, env={**env, **changes},
                          capture_output=True, text=True, timeout=30)


def records(setup):
    path = setup[3] / "calls"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_shell_syntax_and_isolated_submission(setup):
    for name in (CLUSTER, SUBMIT):
        subprocess.run(["bash", "-n", str(ROOT / "bin/run_experiments" / name)], check=True)
    result = run(setup, SUBMIT)
    assert result.returncode == 0, result.stderr
    args = (setup[3] / "args").read_text().splitlines()
    for arg in ("--partition=pgi14", "--nodelist=pgi14-gpu1", "--gres=gpu:nvidia_geforce_rtx_4090:2",
                "--cpus-per-task=16", "--mem=100G", "--time=08:00:00"):
        assert arg in args
    payload = (setup[3] / "payload").read_text()
    assert 'TASK_REPO="/Local/durso/s5-proximal-memory-mix-${SLURM_JOB_ID}"' in payload
    assert 'git checkout --detach "$S5_MEMORY_MIX_COMMIT"' in payload
    assert 'exec bash bin/run_experiments/cluster_s5_proximal_memory_mix.sh' in payload
    subprocess.run(["bash", "-n", str(setup[3] / "payload")], check=True)


def test_dry_run_submits_and_writes_nothing(setup):
    for name in (SUBMIT, CLUSTER):
        result = run(setup, name, DRY_RUN="1")
        assert result.returncode == 0 and "DRY_RUN" in result.stdout, result.stderr
    assert not (setup[3] / "payload").exists()
    assert not (setup[3] / "runs").exists()


def test_both_smokes_gate_training_on_assigned_tokens(setup):
    result = run(setup)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = records(setup)
    assert len(calls) == 6
    for seed, token in ((301, "GPU-allocated-four"), (302, "GPU-allocated-seven")):
        mine = [r for r in calls if r["seed"] == seed]
        assert [r["kind"] for r in mine] == ["profile", "smoke", "train"]
        assert {r["gpu"] for r in mine} == {token}
    first_train = next(i for i, r in enumerate(calls) if r["kind"] == "train")
    assert sum(r["kind"] == "smoke" for r in calls[:first_train]) == 2
    out = setup[3] / "runs/s5-proximal-memory-mix/job-123"
    assert len(list(out.glob("proximal_memory_mix_s5/*/task_result.json"))) == 2
    gate = json.loads((out / "smoke_gate.json").read_text())
    assert gate["status"] == "SMOKE_PASS"


def test_smoke_stage_never_trains(setup):
    result = run(setup, STAGE="smoke")
    assert result.returncode == 0, result.stderr
    assert {r["kind"] for r in records(setup)} == {"profile", "smoke"}


@pytest.mark.parametrize("changes", [
    {"FAIL_SMOKE": "301"}, {"UNSTABLE_INIT": "302"},
    {"COMPILER_BYTES": str(30 * 2**30)}, {"PROFILE_BACKEND": "cpu"},
])
def test_failed_gate_prevents_all_training(setup, changes):
    result = run(setup, **changes)
    assert result.returncode != 0
    assert not any(r["kind"] == "train" for r in records(setup))
    assert "Completed proximal memory mix" not in result.stdout


def test_missing_memory_accounting_still_requires_runtime_smoke(setup):
    result = run(setup, COMPILER_BYTES="0", GPU_LIMIT="0")
    assert result.returncode == 0, result.stderr
    out = setup[3] / "runs/s5-proximal-memory-mix/job-123/gates/memory-301.json"
    rows = json.loads(out.read_text())["rows"]
    assert all(not r["fit_check_available"] for r in rows)
    assert sum(r["kind"] == "smoke" for r in records(setup)) == 2


@pytest.mark.parametrize("changes", [
    {"CUDA_VISIBLE_DEVICES": "0"}, {"CUDA_VISIBLE_DEVICES": "0,0"},
    {"CUDA_VISIBLE_DEVICES": "0, 1"}, {"CUDA_VISIBLE_DEVICES": "0,1,2"},
    {"JAX_ENABLE_X64": "1"}, {"EXPECTED_COMMIT": "bad"},
    {"STAGE": "other"}, {"DRY_RUN": "true"}, {"SLURM_JOB_ID": ""},
])
def test_invalid_allocation_refused_before_tasks(setup, changes):
    result = run(setup, **changes)
    assert result.returncode != 0
    assert not records(setup)


def test_preflight_failure_prevents_tasks(setup):
    result = run(setup, FAIL_PREFLIGHT="1")
    assert result.returncode != 0 and not records(setup)


@pytest.mark.parametrize("changes", [{"FAIL_TRAIN": "301"}, {"MISSING_RESULT": "302"}])
def test_incomplete_training_is_not_reported_complete(setup, changes):
    result = run(setup, **changes)
    assert result.returncode != 0
    assert "Completed proximal memory mix" not in result.stdout
    assert sum(r["kind"] == "train" for r in records(setup)) == 2


def test_existing_outputs_and_untracked_code_refused(setup):
    out = setup[3] / "runs/s5-proximal-memory-mix/job-123"
    out.mkdir(parents=True)
    result = run(setup)
    assert result.returncode != 0 and "already exists" in result.stderr
    (setup[0] / "unexpected.py").write_text("# untracked executable source\n")
    for name in (CLUSTER, SUBMIT):
        result = run(setup, name)
        assert result.returncode != 0 and "clean" in result.stderr


def test_cleanup_does_not_target_other_jobs():
    source = (ROOT / "bin/run_experiments" / CLUSTER).read_text()
    assert "kill -TERM \"$pid\"" in source and "kill -TERM \"$child\"" in source
    for unsafe in ("kill 0", "kill -0", "kill -- -", "pkill", "killall", "scancel"):
        assert unsafe not in source
