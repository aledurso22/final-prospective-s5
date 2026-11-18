"""Exercise recovery orchestration without Slurm or GPU access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "bin/run_experiments"
CLUSTER = "cluster_s5_adaptive_history_resume.sh"
SUBMIT = "submit_s5_adaptive_history_resume.sh"
ARMS = ("adaptive_proximal_history_s5", "adaptive_proximal_history_control_s5")


@pytest.fixture
def launcher_env(tmp_path):
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
import hashlib, json, os, sys
from pathlib import Path
args = [a for a in sys.argv[1:] if a != "-u"]
if not args or args[0] != "-m":
    script = sys.stdin.read()
    if "import jax" in script:
        sys.exit(0)  # Orchestration tests do not claim GPU validation.
    sys.argv = args
    exec(compile(script, "<launcher-inline>", "exec"), {"__name__": "__main__"})
    sys.exit(0)
module = args[1]
def val(flag):
    return args[args.index(flag) + 1]
def event(**fields):
    fd = os.open(os.environ["EVENTS"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.write(fd, (json.dumps(fields) + "\n").encode()); os.close(fd)
if module.endswith(".audit"):
    event(stage="audit", backend=os.environ.get("JAX_PLATFORMS"),
          gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if os.environ.get("FAIL_AUDIT") == "1":
        sys.exit(2)
    out = Path(val("--out")); out.mkdir(parents=True)
    (out / "block_audit.json").write_text("{}")
    (out / "summary.txt").write_text("audit completed\n")
    sys.exit(0)
assert module == "experiments.s5_adaptive_proximal_history.resume"
arm, source, out = val("--arm"), Path(val("--source-task")), Path(val("--out"))
seed = source.name
assert source.parent.name == arm and val("--epoch") == "3"
stage = "preflight" if "--preflight-only" in args else "resume"
event(stage=stage, arm=arm, seed=seed, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
      backend=os.environ.get("JAX_PLATFORMS"))
if os.environ.get("FAILING_PREFLIGHT") == arm + "/" + seed and stage == "preflight":
    sys.exit(3)
if os.environ.get("FAILING_RESUME") == arm + "/" + seed and stage == "resume":
    sys.exit(4)
out.mkdir(parents=True)
if stage == "preflight":
    report = dict(status="RESUME_PREFLIGHT_PASS", backend="gpu", arm=arm, seed=int(seed),
                  source_task=str(source), source_epoch=3, source_unchanged=True,
                  source_checkpoint_sha256=hashlib.sha256(
                      (source / "checkpoint_epoch_03.msgpack").read_bytes()).hexdigest())
    if os.environ.get("BAD_BINDING") == arm + "/" + seed:
        report["seed"] = 999
    (out / "resume_preflight.json").write_text(json.dumps(report))
else:
    (out / "task_result.json").write_text(json.dumps(dict(
        selected_epoch=15, validation=dict(accuracy=.9, cross_entropy=.3))))
''')
    fake_python.chmod(0o755)
    commands = {
        "hostname": 'echo "${FAKE_HOST:-pgi14-gpu1}"',
        "scontrol": 'echo "JobId=$SLURM_JOB_ID ${ISOLATION_FIELDS:-OverSubscribe=NO} NodeList=pgi14-gpu1"',
        "squeue": 'echo "$SLURM_JOB_ID"; if [ -n "${OTHER_JOB:-}" ]; then echo "$OTHER_JOB"; fi',
    }
    for name, body in commands.items():
        script = fake_bin / name
        script.write_text("#!/bin/bash\n" + body + "\n")
        script.chmod(0o755)
    runs, data, source = tmp_path / "runs", tmp_path / "cache", tmp_path / "source"
    runs.mkdir(); data.mkdir()
    for arm in ARMS:
        for seed in (301, 302):
            task = source / arm / str(seed)
            task.mkdir(parents=True)
            (task / "checkpoint_epoch_03.msgpack").write_bytes(b"fixture checkpoint")
    events = tmp_path / "events.jsonl"
    env = {**os.environ, "EXPECTED_COMMIT": sha, "DRY_RUN": "0", "STAGE": "train",
           "SLURM_JOB_ID": "987654", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f,GPU-g,GPU-h",
           "PROSPECTIVE_VENV": str(fake_bin.parent), "PROSPECTIVE_RUNS": str(runs),
           "AP_RESUME_SOURCE_ROOT": str(source), "S5_THREE_ARM_DATA": str(data),
           "JAX_ENABLE_X64": "0", "EVENTS": str(events),
           "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]}
    for key in ("FAIL_AUDIT", "FAILING_PREFLIGHT", "FAILING_RESUME", "BAD_BINDING",
                "FAKE_HOST", "ISOLATION_FIELDS", "OTHER_JOB"):
        env.pop(key, None)
    return repo, env, runs / "s5-adaptive-history-resume/job-987654", events


def launch(fixture, script=CLUSTER, **overrides):
    repo, env, _, _ = fixture
    return subprocess.run(["bash", str(repo / "bin/run_experiments" / script)], cwd=repo,
                          env={**env, **overrides}, capture_output=True, text=True, timeout=30)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_scripts_parse_and_dryrun_creates_nothing(launcher_env):
    repo, _, out, event_path = launcher_env
    for script in (CLUSTER, SUBMIT):
        subprocess.run(["bash", "-n", str(repo / "bin/run_experiments" / script)], check=True)
        result = launch(launcher_env, script=script, DRY_RUN="1")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "DRY_RUN" in result.stdout and ARMS[1] in result.stdout
    assert not out.exists() and not event_path.exists()


@pytest.mark.parametrize("devices", ("", "0,1,2,3", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,7,8"))
def test_wrong_gpu_allocation_refuses(launcher_env, devices):
    result = launch(launcher_env, CUDA_VISIBLE_DEVICES=devices)
    assert result.returncode != 0 and "exactly eight" in result.stderr
    assert not launcher_env[2].exists() and not launcher_env[3].exists()


@pytest.mark.parametrize("fields", ("OverSubscribe=OK", "OverSubscribe=YES", "Shared=1", "Exclusive=USER",
                                   "Unknown=1", "OverSubscribe=NO Shared=1"))
def test_nonexclusive_or_unconfirmed_allocation_refuses(launcher_env, fields):
    result = launch(launcher_env, ISOLATION_FIELDS=fields)
    assert result.returncode != 0 and "allocation" in result.stderr
    assert not launcher_env[2].exists() and not launcher_env[3].exists()


@pytest.mark.parametrize("fields", ("OverSubscribe=NO", "Shared=0", "Exclusive=NODE"))
def test_known_exclusive_formats_pass(launcher_env, fields):
    result = launch(launcher_env, ISOLATION_FIELDS=fields, STAGE="preflight")
    assert result.returncode == 0, result.stdout + result.stderr
    assert fields in (launcher_env[2] / "slurm_job.txt").read_text()
    assert len(events(launcher_env[3])) == 4


@pytest.mark.parametrize("override, expected", [
    ({"FAKE_HOST": "pgi15-head1"}, "pgi14-gpu1"),
    ({"OTHER_JOB": "12345"}, "another active job"),
    ({"EXPECTED_COMMIT": "0" * 40}, "wrong commit"),
    ({"JAX_ENABLE_X64": "1"}, "x64"),
    ({"STAGE": "test"}, "invalid STAGE"),
])
def test_protocol_and_isolation_guards(launcher_env, override, expected):
    result = launch(launcher_env, **override)
    assert result.returncode != 0 and expected in result.stderr
    assert not launcher_env[2].exists()


def test_source_checkpoint_required_before_work(launcher_env):
    source = Path(launcher_env[1]["AP_RESUME_SOURCE_ROOT"])
    (source / ARMS[0] / "301/checkpoint_epoch_03.msgpack").unlink()
    result = launch(launcher_env)
    assert result.returncode != 0 and "missing source epoch-3" in result.stderr
    assert not launcher_env[2].exists() and not launcher_env[3].exists()


def test_existing_output_never_reused(launcher_env):
    out = launcher_env[2]
    out.mkdir(parents=True)
    marker = out / "preserve.txt"
    marker.write_text("original")
    result = launch(launcher_env)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert marker.read_text() == "original" and not launcher_env[3].exists()


def test_submission_reserves_whole_node_and_pins_clone(launcher_env, tmp_path):
    repo, env, _, _ = launcher_env
    capture = tmp_path / "submitted.json"
    fake_sbatch = Path(env["PROSPECTIVE_VENV"]) / "bin/sbatch"
    fake_sbatch.write_text(f"#!{sys.executable}\n" +
                          "import json, os, sys\n" +
                          "with open(os.environ['CAPTURE'], 'w') as f:\n" +
                          "    json.dump({'args': sys.argv[1:], 'script': sys.stdin.read(), " +
                          "'source': os.environ['AP_RESUME_SOURCE_ROOT']}, f)\n")
    fake_sbatch.chmod(0o755)
    result = launch(launcher_env, script=SUBMIT, CAPTURE=str(capture))
    assert result.returncode == 0, result.stdout + result.stderr
    submitted = json.loads(capture.read_text())
    for arg in ("--exclusive", "--partition=pgi14", "--nodelist=pgi14-gpu1",
                "--gres=gpu:nvidia_geforce_rtx_4090:8", "--cpus-per-task=32", "--mem=160G"):
        assert arg in submitted["args"]
    assert submitted["source"] == env["AP_RESUME_SOURCE_ROOT"]
    assert f"--export=ALL,S5_AP_RESUME_COMMIT={env['EXPECTED_COMMIT']},S5_AP_RESUME_STAGE=train" in submitted["args"]
    assert 'git checkout --detach "$S5_AP_RESUME_COMMIT"' in submitted["script"]
    assert '/Local/durso/s5-adaptive-history-resume-${SLURM_JOB_ID}' in submitted["script"]
    assert '[[ ! -e "$REPO" ]]' in submitted["script"]
    assert 'exec bash bin/run_experiments/' + CLUSTER in submitted["script"]


@pytest.mark.parametrize("failure", ("FAILING_PREFLIGHT", "BAD_BINDING"))
def test_any_preflight_failure_gates_all_continuations(launcher_env, failure):
    result = launch(launcher_env, **{failure: ARMS[0] + "/301"})
    assert result.returncode != 0 and "NO continuation started" in result.stderr
    observed = events(launcher_env[3])
    assert len(observed) == 4 and all(e["stage"] == "preflight" for e in observed)


def test_phase_order_device_assignment_audit_and_source_preservation(launcher_env):
    source = Path(launcher_env[1]["AP_RESUME_SOURCE_ROOT"])
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = launch(launcher_env)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = events(launcher_env[3])
    assert [e["stage"] for e in observed] == ["preflight"] * 4 + ["resume"] * 4 + ["audit"]
    for arm, seed, gpu in ((ARMS[0], "301", "GPU-a"), (ARMS[0], "302", "GPU-b"),
                           (ARMS[1], "301", "GPU-c"), (ARMS[1], "302", "GPU-d")):
        rows = [e for e in observed if e.get("arm") == arm and e.get("seed") == seed]
        assert len(rows) == 2 and all(e["gpu"] == gpu and e["backend"] == "cuda" for e in rows)
    assert observed[-1]["backend"] == "cpu" and observed[-1]["gpu"] == ""
    assert (launcher_env[2] / "status/audit").read_text().strip() == "PASS"
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert "selected_epoch" in result.stdout


def test_failed_resume_skips_audit_without_deleting_successes(launcher_env):
    result = launch(launcher_env, FAILING_RESUME=ARMS[0] + "/301")
    assert result.returncode != 0
    assert not any(e["stage"] == "audit" for e in events(launcher_env[3]))
    assert (launcher_env[2] / "status/audit").read_text().strip() == "SKIPPED:incomplete_training"
    assert (launcher_env[2] / ARMS[1] / "302/task_result.json").exists()


def test_audit_failure_preserves_all_completed_results(launcher_env):
    result = launch(launcher_env, FAIL_AUDIT="1")
    assert result.returncode != 0 and "AUDIT FAILED after all four trainings completed" in result.stderr
    for arm in ARMS:
        for seed in (301, 302):
            assert (launcher_env[2] / arm / str(seed) / "task_result.json").exists()
            assert (launcher_env[2] / "status" / f"{arm}-{seed}").read_text().strip() == "TRAINED"
    assert (launcher_env[2] / "status/audit").read_text().strip() == "FAIL"
