"""Execute the launcher's phase control without training or Slurm access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "bin/run_experiments"
CLUSTER = "cluster_s5_adaptive_proximal_history.sh"
SUBMIT = "submit_s5_adaptive_proximal_history.sh"
ARMS = ("adaptive_proximal_history_s5", "adaptive_proximal_history_control_s5")


@pytest.fixture
def launcher_env(tmp_path):
    """Use a committed disposable copy so real checkout edits cannot mask guards."""
    repo = tmp_path / "repo"
    scripts = repo / "bin/run_experiments"
    scripts.mkdir(parents=True)
    for name in (CLUSTER, SUBMIT, "cluster_env.sh"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    for args in (("init", "-q"), ("config", "user.name", "Launcher Test"),
                 ("config", "user.email", "launcher-test@example.invalid"),
                 ("add", "."), ("commit", "-qm", "launcher fixture")):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    fake_bin = tmp_path / "venv/bin"
    fake_bin.mkdir(parents=True)
    fake_python = fake_bin / "python"
    fake_python.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys
from pathlib import Path
args = [a for a in sys.argv[1:] if a != "-u"]
if not args or args[0] != "-m":
    # Device check, preflight, profile binding check and report-only summary
    # are not model executions in these orchestration-only tests.
    sys.exit(0)
module = args[1]
def val(flag):
    return args[args.index(flag) + 1]
if module.endswith(".audit"):
    if os.environ.get("FAIL_AUDIT") == "1":
        sys.exit(2)
    out = Path(val("--out")); out.mkdir(parents=True)
    (out / "block_audit.json").write_text("{}")
    (out / "summary.txt").write_text("audit completed\n")
    sys.exit(0)
assert module == "experiments.s5_adaptive_proximal_history.runner"
arm, seed, out = val("--arm"), val("--seed"), Path(val("--out"))
stage = next(x for x in ("profile", "smoke", "train") if "--" + x in args)
event = dict(stage=stage, arm=arm, seed=seed,
             gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
             backend=os.environ.get("JAX_PLATFORMS"))
fd = os.open(os.environ["EVENTS"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(fd, (json.dumps(event) + "\n").encode()); os.close(fd)
if stage == "profile":
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{}")
elif stage == "smoke":
    if os.environ.get("FAILING_SMOKE") == arm + "/" + seed:
        sys.exit(3)
    out.mkdir(parents=True)
    (out / "smoke_result.json").write_text("{}")
else:
    smoke = Path(val("--smoke-report"))
    assert smoke.is_file() and smoke.parent.name == seed
    out.mkdir(parents=True)
    (out / "task_result.json").write_text(json.dumps(dict(
        selected_epoch=15, validation=dict(accuracy=.9, cross_entropy=.3))))
''')
    fake_python.chmod(0o755)
    runs = tmp_path / "runs"
    data = tmp_path / "cache"
    runs.mkdir(); data.mkdir()
    events = tmp_path / "events.jsonl"
    env = {**os.environ, "EXPECTED_COMMIT": sha, "DRY_RUN": "0", "STAGE": "train",
           "SLURM_JOB_ID": "987654", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d",
           "PROSPECTIVE_VENV": str(fake_bin.parent), "PROSPECTIVE_RUNS": str(runs),
           "S5_THREE_ARM_DATA": str(data), "JAX_ENABLE_X64": "0", "EVENTS": str(events)}
    # No inherited test switches should modify the disposable worker behavior.
    for key in ("FAIL_AUDIT", "FAILING_SMOKE"):
        env.pop(key, None)
    return repo, env, runs / "s5-adaptive-proximal-history/job-987654", events


def run_launcher(fixture, *, script=CLUSTER, **overrides):
    repo, env, _, _ = fixture
    return subprocess.run(["bash", str(repo / "bin/run_experiments" / script)],
                          cwd=repo, env={**env, **overrides}, capture_output=True,
                          text=True, timeout=30)


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_scripts_parse_and_dryrun_writes_nothing(launcher_env):
    repo, _, out, events = launcher_env
    for name in (CLUSTER, SUBMIT):
        subprocess.run(["bash", "-n", str(repo / "bin/run_experiments" / name)], check=True)
        r = run_launcher(launcher_env, script=name, DRY_RUN="1")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "DRY_RUN" in r.stdout
        assert "adaptive_proximal_history_control_s5" in r.stdout
    assert not out.exists() and not events.exists()


@pytest.mark.parametrize("devices", ("0,1", "0,1,2,2", "", "0,1,2,3,4"))
def test_wrong_gpu_allocation_refuses_before_any_output(launcher_env, devices):
    r = run_launcher(launcher_env, CUDA_VISIBLE_DEVICES=devices)
    assert r.returncode != 0 and "exactly four" in r.stderr
    assert not launcher_env[2].exists() and not launcher_env[3].exists()


@pytest.mark.parametrize("override, message", [
    ({"JAX_ENABLE_X64": "1"}, "x64"),
    ({"STAGE": "test"}, "invalid STAGE"),
    ({"EXPECTED_COMMIT": "0" * 40}, "wrong commit"),
])
def test_protocol_guards(launcher_env, override, message):
    r = run_launcher(launcher_env, **override)
    assert r.returncode != 0 and message in r.stderr
    assert not launcher_env[2].exists()


def test_existing_output_is_never_reused(launcher_env):
    out = launcher_env[2]
    out.mkdir(parents=True)
    marker = out / "preserve.txt"
    marker.write_text("original")
    r = run_launcher(launcher_env)
    assert r.returncode != 0 and "already exists" in r.stderr
    assert marker.read_text() == "original" and not launcher_env[3].exists()


def test_submit_pins_node_gpu_request_and_isolated_checkout(launcher_env, tmp_path):
    repo, env, _, _ = launcher_env
    capture = tmp_path / "submission.json"
    fake_sbatch = tmp_path / "submission-bin"
    fake_sbatch.mkdir()
    executable = fake_sbatch / "sbatch"
    executable.write_text(f"#!{sys.executable}\n" +
        "import json, os, sys\n" +
        "with open(os.environ['SUBMISSION_CAPTURE'], 'w') as f:\n" +
        "    json.dump({'args': sys.argv[1:], 'script': sys.stdin.read()}, f)\n")
    executable.chmod(0o755)
    r = run_launcher(launcher_env, script=SUBMIT, STAGE="smoke",
                     PATH=str(fake_sbatch) + os.pathsep + env["PATH"],
                     SUBMISSION_CAPTURE=str(capture))
    assert r.returncode == 0, r.stdout + r.stderr
    submitted = json.loads(capture.read_text())
    assert "--nodelist=pgi14-gpu1" in submitted["args"]
    assert "--partition=pgi14" in submitted["args"]
    assert "--gres=gpu:nvidia_geforce_rtx_4090:4" in submitted["args"]
    assert "--cpus-per-task=32" in submitted["args"] and "--mem=160G" in submitted["args"]
    assert f"--export=ALL,S5_AP_COMMIT={env['EXPECTED_COMMIT']},S5_AP_STAGE=smoke" in submitted["args"]
    assert 'git checkout --detach "$S5_AP_COMMIT"' in submitted["script"]
    assert '/Local/durso/s5-adaptive-proximal-history-${SLURM_JOB_ID}' in submitted["script"]
    assert '[[ ! -e "$REPO" ]]' in submitted["script"]
    assert 'exec bash bin/run_experiments/' + CLUSTER in submitted["script"]


def test_any_failed_smoke_blocks_all_training(launcher_env):
    r = run_launcher(launcher_env, FAILING_SMOKE=ARMS[0] + "/301")
    assert r.returncode != 0 and "NO training started" in r.stderr
    events = read_events(launcher_env[3])
    assert sum(x["stage"] == "smoke" for x in events) == 4
    assert not any(x["stage"] == "train" for x in events)
    assert (launcher_env[2] / "status" / (ARMS[0] + "-301")).read_text().strip() == "FAIL:smoke"


def test_all_smokes_precede_training_and_each_worker_keeps_its_gpu(launcher_env):
    r = run_launcher(launcher_env)
    assert r.returncode == 0, r.stdout + r.stderr
    events = read_events(launcher_env[3])
    smoke_positions = [i for i, e in enumerate(events) if e["stage"] == "smoke"]
    train_positions = [i for i, e in enumerate(events) if e["stage"] == "train"]
    assert len(smoke_positions) == len(train_positions) == 4
    assert max(smoke_positions) < min(train_positions)
    for arm, seed, gpu in ((ARMS[0], "301", "GPU-a"), (ARMS[0], "302", "GPU-b"),
                           (ARMS[1], "301", "GPU-c"), (ARMS[1], "302", "GPU-d")):
        task = [e for e in events if e["arm"] == arm and e["seed"] == seed]
        assert [e["stage"] for e in task] == ["profile", "smoke", "train"]
        assert all(e["gpu"] == gpu and e["backend"] == "cuda" for e in task)
    assert (launcher_env[2] / "status/audit").read_text().strip() == "PASS"


def test_smoke_only_never_trains_or_audits(launcher_env):
    r = run_launcher(launcher_env, STAGE="smoke")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(e["stage"] == "train" for e in read_events(launcher_env[3]))
    assert not (launcher_env[2] / "audit").exists()


def test_audit_failure_preserves_and_identifies_completed_training(launcher_env):
    r = run_launcher(launcher_env, FAIL_AUDIT="1")
    assert r.returncode != 0
    assert "AUDIT FAILED after all four trainings completed" in r.stderr
    assert "rerun only the audit" in r.stderr
    out = launcher_env[2]
    for arm in ARMS:
        for seed in (301, 302):
            assert (out / arm / str(seed) / "task_result.json").is_file()
            assert (out / "status" / f"{arm}-{seed}").read_text().strip() == "TRAINED"
    assert (out / "status/audit").read_text().strip() == "FAIL"
