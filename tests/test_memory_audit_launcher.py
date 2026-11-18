"""Validate submission without contacting Slurm or the cluster."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path("bin/run_experiments/submit_s5_memory_audit.sh")


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path / "repo"
    (repo / SCRIPT.parent).mkdir(parents=True)
    shutil.copy2(ROOT / SCRIPT, repo / SCRIPT)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    sbatch = fakebin / "sbatch"
    sbatch.write_text('#!/bin/bash\nset -eu\n'
                     'for v in SBATCH_GRES SBATCH_GPUS SBATCH_GPUS_PER_NODE SBATCH_GPUS_PER_TASK SBATCH_GPUS_PER_SOCKET; do\n'
                     '  if printenv "$v" >/dev/null; then exit 99; fi\n'
                     'done\n'
                     'printf "%s\\n" "$@" > "$CAPTURE/args"\n'
                     'cat > "$CAPTURE/payload"\n'
                     'echo "Submitted batch job 123"\n')
    sbatch.chmod(0o755)
    env = {**os.environ, "EXPECTED_COMMIT": sha, "DRY_RUN": "0",
           "PATH": f"{fakebin}:{os.environ['PATH']}", "CAPTURE": str(tmp_path)}
    return repo, env, tmp_path


def run(fixture, *args, **changes):
    repo, env, _ = fixture
    return subprocess.run(["bash", str(repo / SCRIPT), *args], cwd=repo,
                          env={**env, **changes}, capture_output=True, text=True, timeout=30)


def test_cpu_submission_isolated_pinned_and_all_nine_inputs(fixture):
    result = run(fixture, SBATCH_GRES="gpu:8", SBATCH_GPUS="8", SBATCH_GPUS_PER_NODE="8",
                 SBATCH_GPUS_PER_TASK="8", SBATCH_GPUS_PER_SOCKET="4")
    assert result.returncode == 0, result.stdout + result.stderr
    out = fixture[2]
    args = (out / "args").read_text().splitlines()
    for arg in ("--partition=pgi14", "--nodelist=pgi14-gpu1", "--cpus-per-task=4", "--mem=24G", "--time=02:00:00"):
        assert arg in args
    assert not any(arg.startswith(("--gres", "--gpu")) for arg in args)
    payload = (out / "payload").read_text()
    assert 'TASK_REPO="/Local/durso/s5-memory-audit-${SLURM_JOB_ID}"' in payload
    assert 'git checkout --detach "$S5_MEMORY_AUDIT_COMMIT"' in payload
    assert 'JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 CUDA_VISIBLE_DEVICES=""' in payload
    assert 'OUT="$RUNS/s5-memory-audit/job-${SLURM_JOB_ID}"' in payload
    assert '--all-epochs --out "$OUT"' in payload
    for arm in ("native_matched_s5", "generalized_prospective_s5", "calibrated_old_s5"):
        for seed in (301, 302):
            assert f'"$BASE/{arm}/{seed}"' in payload
        assert f'"$THIRD/{arm}/303"' in payload
    assert "20261117-230628" in payload and "20261117-223126" not in payload
    assert "--train" not in payload and "--test" not in payload
    assert "S5_THREE_ARM_DATA" not in payload
    assert "git push" not in payload
    subprocess.run(["bash", "-n", str(out / "payload")], check=True)


def test_dry_run_submits_nothing(fixture):
    result = run(fixture, DRY_RUN="1")
    assert result.returncode == 0, result.stderr
    assert "CPU-only" in result.stdout and "no GPUs" in result.stdout
    assert not (fixture[2] / "payload").exists()
    assert not (fixture[2] / "args").exists()


@pytest.mark.parametrize("sha", ["", "a" * 40, "not-a-commit"])
def test_wrong_or_missing_commit_refused(fixture, sha):
    result = run(fixture, EXPECTED_COMMIT=sha)
    assert result.returncode != 0
    assert not (fixture[2] / "payload").exists()


def test_dirty_checkout_refused(fixture):
    (fixture[0] / "untracked.txt").write_text("unsaved work")
    result = run(fixture)
    assert result.returncode != 0 and "clean" in result.stderr
    assert not (fixture[2] / "payload").exists()


def test_invalid_dry_run_or_arguments_refused(fixture):
    assert run(fixture, DRY_RUN="yes").returncode != 0
    assert run(fixture, "--train").returncode != 0
    assert not (fixture[2] / "payload").exists()


def test_shell_syntax():
    subprocess.run(["bash", "-n", str(ROOT / SCRIPT)], check=True)
