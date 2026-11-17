"""Exercise submission with an intercepted sbatch; never contact Slurm."""
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bin/run_experiments/submit_s5_history_pilot.sh"


@pytest.fixture
def shell(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    git = fake / "git"
    git.write_text('#!/bin/bash\nshift 2\nif [[ "$1" == rev-parse ]]; then echo pinned; fi\n')
    git.chmod(0o755)
    sbatch = fake / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE/args"\ncat > "$CAPTURE/payload"\necho "Submitted batch job 123"\n')
    sbatch.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "CAPTURE": str(tmp_path),
           "EXPECTED_COMMIT": "pinned", "HISTORY_AFTER_JOB": "987654321"}
    env.pop("DRY_RUN", None)
    return tmp_path, env


def test_submits_only_two_gpu_pilot_with_audit_dependency(shell):
    tmp, env = shell
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = (tmp / "args").read_text().splitlines()
    assert "--dependency=afterok:987654321" in args
    assert "--gres=gpu:nvidia_geforce_rtx_4090:2" in args
    assert "--cpus-per-task=16" in args
    payload = (tmp / "payload").read_text()
    assert "exec bash bin/run_experiments/allocation_s5_history_two_gpu.sh" in payload
    assert "submit_s5_followup" not in payload
    assert "git checkout --detach" in payload
    assert subprocess.run(["bash", "-n", str(tmp / "payload")]).returncode == 0


def test_dry_run_does_not_submit(shell):
    tmp, env = shell
    result = subprocess.run(["bash", str(SCRIPT)], env={**env, "DRY_RUN": "1"}, capture_output=True, text=True)
    assert result.returncode == 0 and "DRY_RUN" in result.stdout
    assert not (tmp / "payload").exists()


@pytest.mark.parametrize("change", [{"HISTORY_AFTER_JOB": "bad;job"}, {"EXPECTED_COMMIT": "wrong"}])
def test_invalid_submission_refused(shell, change):
    tmp, env = shell
    result = subprocess.run(["bash", str(SCRIPT)], env={**env, **change}, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp / "payload").exists()
