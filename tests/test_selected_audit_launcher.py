"""Exercise allocation orchestration with stub diagnostics, never real data."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NAME = "allocation_s5_selected_audit_four_gpu.sh"
ARMS = ("native_matched_s5", "generalized_prospective_s5",
        "calibrated_old_s5", "calibrated_prediction_s5")


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
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    interpreter = venv / "bin/python"
    interpreter.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
a = sys.argv[1:]
if a and a[0] == "-":
    source = sys.stdin.read()
    if "import jax" in source:
        assert os.environ["JAX_ENABLE_X64"] == "0"
        assert os.environ["JAX_PLATFORMS"] == "cuda"
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
        if os.environ.get("FAIL_DEVICE") == os.environ["CUDA_VISIBLE_DEVICES"]:
            sys.exit(11)
        sys.exit(0)
    sys.argv = ["-"] + a[1:]
    exec(compile(source, "<inline>", "exec"))
    sys.exit(0)
module = a[a.index("-m") + 1]
arms = ("native_matched_s5", "generalized_prospective_s5",
        "calibrated_old_s5", "calibrated_prediction_s5")
if module.endswith("selected_checkpoint_audit"):
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    if os.environ.get("FAIL_CPU"):
        sys.exit(8)
    out = pathlib.Path(a[a.index("--out") + 1])
    out.mkdir(parents=True)
    tasks = [{"arm": arm, "seed": seed} for arm in arms for seed in (301, 302)]
    if os.environ.get("BAD_ORDER"):
        tasks.reverse()
    (out / "manifest.json").write_text(json.dumps({"tasks": tasks}))
    sys.exit(0)
assert module.endswith("selected_bn_audit")
if "--aggregate" in a:
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    root = pathlib.Path(a[a.index("--aggregate") + 1])
    files = list(root.glob("*/3*/selected_summary.json"))
    assert len(files) == 8
    assert all(json.loads(f.read_text())["replay"] for f in files)
    (root / "selected_bn_summary.json").write_text("{}")
    sys.exit(0)
index = int(a[a.index("--task-index") + 1])
token = os.environ["CUDA_VISIBLE_DEVICES"]
assert token == str(index % 4)
assert os.environ["JAX_PLATFORMS"] == "cuda"
out = pathlib.Path(a[a.index("--out") + 1])
assert out.parts[-2:] == (arms[index // 2], str(301 + index % 2))
out.mkdir(parents=True)
fd = os.open(os.environ["CALLS"], os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
os.write(fd, (json.dumps({"index": index, "gpu": token}) + "\\n").encode())
os.close(fd)
(out / "selected_summary.json").write_text(json.dumps({
    "replay": os.environ.get("BAD_REPLAY") != str(index)}))
if os.environ.get("FAIL_TASK") == str(index):
    sys.exit(9)
''')
    interpreter.chmod(0o755)
    source = tmp_path / "source"
    source.mkdir()
    (source / "sentinel").write_text("original result")
    data = tmp_path / "data"
    data.mkdir()
    env = dict(os.environ, PROSPECTIVE_VENV=str(venv), PROSPECTIVE_RUNS=str(tmp_path / "runs"),
               EXPECTED_COMMIT=commit, SLURM_JOB_ID="5678", CUDA_VISIBLE_DEVICES="0,1,2,3",
               JAX_ENABLE_X64="0", DRY_RUN="0", SOURCE_RUN=str(source),
               S5_THREE_ARM_DATA=str(data), CALLS=str(tmp_path / "calls.jsonl"))
    for key in ("AUDIT_ROOT", "FAIL_CPU", "BAD_ORDER", "FAIL_DEVICE", "FAIL_TASK", "BAD_REPLAY"):
        env.pop(key, None)
    return repo, scripts / NAME, env


def run(allocation, **changes):
    repo, script, env = allocation
    return subprocess.run(["bash", str(script)], cwd=repo, env={**env, **changes},
                          text=True, capture_output=True, timeout=20)


def audit_root(allocation):
    return Path(allocation[2]["PROSPECTIVE_RUNS"]) / "s5-selected-audit/job-5678"


def test_syntax_and_dry_run_without_source_or_data(allocation):
    subprocess.run(["bash", "-n", str(allocation[1])], check=True)
    r = run(allocation, DRY_RUN="1", SOURCE_RUN="/does-not-exist", S5_THREE_ARM_DATA="/missing")
    assert r.returncode == 0, r.stderr
    assert "nothing started" in r.stdout
    assert r.stdout.count("task-index") == 8
    assert not audit_root(allocation).exists()


@pytest.mark.parametrize("changes, message", [
    ({"CUDA_VISIBLE_DEVICES": "0,1,2"}, "need 4 distinct GPU tokens"),
    ({"CUDA_VISIBLE_DEVICES": "0,1,2,2"}, "need 4 distinct GPU tokens"),
    ({"JAX_ENABLE_X64": "1"}, "production replay requires"),
    ({"SLURM_JOB_ID": "bad"}, "is not numeric"),
    ({"EXPECTED_COMMIT": "0" * 40}, "is not"),
])
def test_guards_before_work(allocation, changes, message):
    r = run(allocation, **changes)
    assert r.returncode != 0 and message in r.stderr
    assert not audit_root(allocation).exists()


def test_source_output_refused_before_writing(allocation):
    source = Path(allocation[2]["SOURCE_RUN"])
    r = run(allocation, AUDIT_ROOT=str(source / "bad"))
    assert r.returncode != 0 and "outside the original run" in r.stderr
    assert sorted(p.name for p in source.iterdir()) == ["sentinel"]


def test_existing_output_refused(allocation):
    audit_root(allocation).mkdir(parents=True)
    r = run(allocation)
    assert r.returncode != 0 and "refusing existing output" in r.stderr


def test_four_workers_cover_eight_tasks_and_preserve_source(allocation):
    r = run(allocation)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [json.loads(line) for line in Path(allocation[2]["CALLS"]).read_text().splitlines()]
    assert sorted(c["index"] for c in calls) == list(range(8))
    for slot in range(4):
        assert [c["index"] for c in calls if c["gpu"] == str(slot)] == [slot, slot + 4]
    assert (audit_root(allocation) / "bn/selected_bn_summary.json").exists()
    source = Path(allocation[2]["SOURCE_RUN"])
    assert sorted(p.name for p in source.iterdir()) == ["sentinel"]
    assert (source / "sentinel").read_text() == "original result"


@pytest.mark.parametrize("changes", [{"FAIL_CPU": "1"}, {"BAD_ORDER": "1"}])
def test_cpu_failure_or_wrong_manifest_blocks_gpu_work(allocation, changes):
    r = run(allocation, **changes)
    assert r.returncode != 0
    assert not Path(allocation[2]["CALLS"]).exists()


@pytest.mark.parametrize("changes", [{"FAIL_TASK": "2"}, {"FAIL_DEVICE": "2"}])
def test_worker_failure_blocks_aggregation(allocation, changes):
    r = run(allocation, **changes)
    assert r.returncode != 0, r.stdout
    assert "No aggregate produced" in r.stderr
    assert not (audit_root(allocation) / "bn/selected_bn_summary.json").exists()
    calls = [json.loads(line) for line in Path(allocation[2]["CALLS"]).read_text().splitlines()]
    assert 6 not in [c["index"] for c in calls]  # worker 2 never reaches its second task


def test_bad_replay_cannot_produce_aggregate(allocation):
    r = run(allocation, BAD_REPLAY="3")
    assert r.returncode != 0
    assert not (audit_root(allocation) / "bn/selected_bn_summary.json").exists()
