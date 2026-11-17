"""Retry only the interrupted seed303 allocation, leaving the audit intact."""
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bin/run_experiments/submit_s5_followup.sh"


@pytest.fixture
def intercepted(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    git = fake / "git"
    git.write_text('#!/bin/bash\nshift 2\nif [[ "$1" == rev-parse ]]; then echo pinned; fi\n')
    git.chmod(0o755)
    sbatch = fake / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$CAPTURE/args"\ncat >> "$CAPTURE/payload"\necho submitted\n')
    sbatch.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}",
           "EXPECTED_COMMIT": "pinned", "CAPTURE": str(tmp_path)}
    env.pop("DRY_RUN", None)
    return tmp_path, env


def test_seed303_only(intercepted):
    tmp, env = intercepted
    p = subprocess.run(["bash", str(SCRIPT), "--seed303-only"], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    args = (tmp / "args").read_text()
    assert args.count("--job-name=") == 1
    assert "--job-name=s5-followup-seed303" in args
    assert "S5_FOLLOWUP_TASK=selected-audit" not in args
    assert "--gres=gpu:nvidia_geforce_rtx_4090:4" in args


def test_default_still_submits_original_two_jobs(intercepted):
    tmp, env = intercepted
    p = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (tmp / "args").read_text().count("--job-name=") == 2


def test_unknown_argument_never_submits(intercepted):
    tmp, env = intercepted
    p = subprocess.run(["bash", str(SCRIPT), "--typo"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and not (tmp / "args").exists()
