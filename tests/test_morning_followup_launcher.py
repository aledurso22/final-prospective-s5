"""Exercise the actual shell queues with fake training processes, never Slurm/GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOCATION = "allocation_s5_morning_followup.sh"
SUBMIT = "submit_s5_morning_followup.sh"


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "bin/run_experiments"
    scripts.mkdir(parents=True)
    for name in (ALLOCATION, SUBMIT, "cluster_env.sh"):
        shutil.copy2(ROOT / "bin/run_experiments" / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    source = tmp_path / "source/gradient_history_s5"
    for seed in (301, 302):
        task = source / str(seed)
        task.mkdir(parents=True)
        for name in ("checkpoint_epoch_09.msgpack", "metrics.jsonl", "history_protocol.json"):
            (task / name).write_text("source data, never modified\n")
    data = tmp_path / "data"
    data.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin/python"
    py.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
a = sys.argv[1:]
if a and a[0] == "-u": a = a[1:]
if a[0] == "-":
    code = sys.stdin.read()
    assert code.strip(), "stdin lost by background process"
    if len(a) == 3 and a[2].endswith("preflight.json"):
        if os.environ.get("FAIL_PREFLIGHT"): sys.exit(8)
        pathlib.Path(a[2]).write_text("{}")
    elif len(a) == 2 and a[1].endswith("task_result.json"):
        sys.argv = a
        exec(compile(code, "<result-check>", "exec"))
    else:
        count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        assert count == (8 if "len(devices) == 8" in code else 1)
    sys.exit(0)
module = a[a.index("-m") + 1]
out = pathlib.Path(a[a.index("--out") + 1])
if module.endswith("gpu_telemetry"):
    out.write_text("{}")
    sys.exit(0)
resume = module.endswith(".resume")
seed = int(pathlib.Path(a[a.index("--source-task") + 1]).name) if resume else int(a[a.index("--seed") + 1])
arm = "gradient_history_s5" if resume else a[a.index("--arm") + 1]
kind = "resume" if resume else next(k for k in ("profile", "smoke", "train") if "--" + k in a)
record = {"kind": kind, "seed": seed, "arm": arm, "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "out": str(out)}
with open(os.environ["CALLS"], "a") as f: f.write(json.dumps(record) + "\\n")
assert not out.exists(), "task outputs must not be precreated"
if kind == "profile":
    out.write_text("{}")
    sys.exit(0)
out.mkdir(parents=True)
if kind == "resume":
    assert a[a.index("--epoch") + 1] == "9"
    assert pathlib.Path(a[a.index("--source-task") + 1], "checkpoint_epoch_09.msgpack").is_file()
    if os.environ.get("FAIL_RESUME") == str(seed): sys.exit(7)
elif kind == "smoke":
    (out / "smoke_result.json").write_text("{}")
    if os.environ.get("FAIL_SMOKE") == str(seed) and arm == "native_matched_s5": sys.exit(6)
    sys.exit(0)
else:
    assert pathlib.Path(a[a.index("--smoke-report") + 1]).is_file()
if os.environ.get("MISSING_RESULT") != str(seed):
    (out / "task_result.json").write_text(json.dumps({"code_identifier": arm, "seed": seed,
        "epochs_requested": 15, "selected_epoch": 14, "validation": {"accuracy": .97, "cross_entropy": .1}}))
''')
    py.chmod(0o755)
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    sbatch = fakebin / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE/args"\ncat > "$CAPTURE/payload"\necho "Submitted batch job 123"\n')
    sbatch.chmod(0o755)
    env = {**os.environ, "EXPECTED_COMMIT": sha, "PROSPECTIVE_VENV": str(venv),
           "PROSPECTIVE_RUNS": str(tmp_path / "runs"), "S5_THREE_ARM_DATA": str(data),
           "HISTORY_RESUME_SOURCE_ROOT": str(source), "SLURM_JOB_ID": "123", "JAX_ENABLE_X64": "0",
           "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "DRY_RUN": "0", "CALLS": str(tmp_path / "calls"),
           "PATH": f"{fakebin}:{os.environ['PATH']}", "CAPTURE": str(tmp_path)}
    for key in ("FAIL_PREFLIGHT", "FAIL_RESUME", "FAIL_SMOKE", "MISSING_RESULT", "MORNING_RUN_ROOT"):
        env.pop(key, None)
    return repo, scripts, env, tmp_path


def run(setup, name=ALLOCATION, **changes):
    repo, scripts, env, _ = setup
    return subprocess.run(["bash", str(scripts / name)], cwd=repo, env={**env, **changes},
                          capture_output=True, text=True, timeout=30)


def records(setup):
    p = setup[3] / "calls"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def test_shell_syntax():
    for name in (ALLOCATION, SUBMIT):
        subprocess.run(["bash", "-n", str(ROOT / "bin/run_experiments" / name)], check=True)


def test_single_eight_gpu_submission(setup):
    r = run(setup, SUBMIT)
    assert r.returncode == 0, r.stdout + r.stderr
    args = (setup[3] / "args").read_text().splitlines()
    assert "--gres=gpu:nvidia_geforce_rtx_4090:8" in args
    assert "--cpus-per-task=32" in args and "--mem=200G" in args
    assert "--time=08:00:00" in args
    body = (setup[3] / "payload").read_text()
    assert "git checkout --detach" in body
    assert "exec bash bin/run_experiments/allocation_s5_morning_followup.sh" in body
    assert subprocess.run(["bash", "-n", str(setup[3] / "payload")]).returncode == 0


def test_dry_runs_submit_nothing_and_allocate_no_outputs(setup):
    for name in (SUBMIT, ALLOCATION):
        r = run(setup, name, DRY_RUN="1")
        assert r.returncode == 0 and "DRY_RUN" in r.stdout, r.stderr
    assert not (setup[3] / "payload").exists()
    assert not (setup[3] / "runs").exists()


def test_all_ten_tasks_have_correct_order_gpu_and_isolated_results(setup):
    source = Path(setup[2]["HISTORY_RESUME_SOURCE_ROOT"])
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    r = run(setup)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = records(setup)
    actual = [(x["gpu"], x["kind"], x["arm"], x["seed"]) for x in calls if x["kind"] in ("train", "resume")]
    expected = {
        ("0", "resume", "gradient_history_s5", 301), ("0", "train", "proximal_history_s5", 303),
        ("1", "resume", "gradient_history_s5", 302), ("1", "train", "proximal_history_s5", 302),
        ("2", "train", "native_matched_s5", 304), ("3", "train", "calibrated_prediction_s5", 304),
        ("4", "train", "proximal_history_s5", 304), ("5", "train", "native_matched_s5", 305),
        ("6", "train", "calibrated_prediction_s5", 305), ("7", "train", "proximal_history_s5", 305),
    }
    assert len(actual) == 10 and set(actual) == expected
    for gpu in ("0", "1"):
        queue = [x for x in actual if x[0] == gpu]
        assert [x[1] for x in queue] == ["resume", "train"]
    for x in calls:
        if x["kind"] == "train":
            earlier = calls[:calls.index(x)]
            assert any(y["kind"] == "smoke" and (y["arm"], y["seed"], y["gpu"]) ==
                       (x["arm"], x["seed"], x["gpu"]) for y in earlier)
    out = Path(setup[2]["PROSPECTIVE_RUNS"]) / "s5-morning-followup/job-123"
    assert len(list(out.glob("**/task_result.json"))) == 10
    assert (out / "history/gradient_history_s5/301/task_result.json").exists()
    assert (out / "fresh/proximal_history_s5/alpha0p1/302/task_result.json").exists()
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_resume_failure_does_not_waste_queued_independent_replication(setup):
    r = run(setup, FAIL_RESUME="301")
    assert r.returncode != 0 and "INCOMPLETE" in r.stderr
    calls = records(setup)
    assert any(x["kind"] == "train" and x["arm"] == "proximal_history_s5" and x["seed"] == 303 for x in calls)
    assert "Completed all ten" not in r.stdout


def test_smoke_failure_blocks_that_training_only(setup):
    r = run(setup, FAIL_SMOKE="304")
    assert r.returncode != 0
    calls = records(setup)
    assert not any(x["kind"] == "train" and x["arm"] == "native_matched_s5" and x["seed"] == 304 for x in calls)
    assert any(x["kind"] == "train" and x["arm"] == "native_matched_s5" and x["seed"] == 305 for x in calls)


@pytest.mark.parametrize("changes", [
    {"CUDA_VISIBLE_DEVICES": "0,1"}, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,6"},
    {"JAX_ENABLE_X64": "1"}, {"EXPECTED_COMMIT": "wrong"}, {"DRY_RUN": "invalid"},
])
def test_bad_allocation_refused_before_work(setup, changes):
    r = run(setup, **changes)
    assert r.returncode != 0
    assert not records(setup)


def test_preflight_failure_prevents_all_worker_tasks(setup):
    r = run(setup, FAIL_PREFLIGHT="1")
    assert r.returncode != 0 and not records(setup)


def test_missing_source_checkpoint_prevents_submission_and_execution(setup):
    source = Path(setup[2]["HISTORY_RESUME_SOURCE_ROOT"])
    (source / "301/checkpoint_epoch_09.msgpack").unlink()
    for name in (SUBMIT, ALLOCATION):
        r = run(setup, name)
        assert r.returncode != 0 and "missing source" in r.stderr
    assert not (setup[3] / "payload").exists()
    assert not records(setup)


def test_successful_process_without_final_result_is_not_completion(setup):
    r = run(setup, MISSING_RESULT="301")
    assert r.returncode != 0 and "missing result" in r.stderr
    assert "Completed all ten" not in r.stdout
    assert any(x["kind"] == "train" and x["seed"] == 303 for x in records(setup))


def test_existing_output_and_source_overlap_refused(setup):
    out = Path(setup[2]["PROSPECTIVE_RUNS"]) / "s5-morning-followup/job-123"
    out.mkdir(parents=True)
    r = run(setup)
    assert r.returncode != 0 and "already exists" in r.stderr
    source = Path(setup[2]["HISTORY_RESUME_SOURCE_ROOT"])
    r = run(setup, MORNING_RUN_ROOT=str(source / "new-output"))
    assert r.returncode != 0 and "overlap" in r.stderr
    assert not (source / "new-output").exists()
