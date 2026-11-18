"""End-of-job bookkeeping: robust summary and explicit trained-P audit status.

    python -m pytest tests/test_proximal_inertia_job_report.py -q
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.s5_proximal_inertia import job_report as JR

ROOT = Path(__file__).resolve().parents[1]


def _run_root(tmp_path, peak):
    root = tmp_path / "job"
    (root / "status").mkdir(parents=True)
    (root / "profiles").mkdir()
    for arm in JR.ARMS:
        for seed in JR.SEEDS:
            (root / "status" / f"{arm}-{seed}").write_text("TRAINED\n")
            (root / "profiles" / f"{arm}-{seed}-gpu.json").write_text(json.dumps(
                {"peak_gib": peak, "steady_step_seconds": {"median": None if peak is None else 0.5}}))
            task = root / arm / str(seed)
            task.mkdir(parents=True)
            (task / "task_result.json").write_text(json.dumps(
                {"selected_epoch": 14, "validation": {"accuracy": 0.97, "cross_entropy": 0.09}}))
    return root


def test_summary_tolerates_missing_allocator_telemetry(tmp_path):
    text = JR.summary(_run_root(tmp_path, None))
    assert text.count("peak n/a GiB  step n/a s") == 4
    assert text.count("val acc 97.0000%") == 4          # training results still reported
    text = JR.summary(_run_root(tmp_path / "b", 3.25))
    assert "peak 3.25 GiB  step 0.500 s" in text


def test_summary_tolerates_unreadable_and_malformed_files(tmp_path):
    root = _run_root(tmp_path, None)
    (root / "profiles" / "proximal_inertial_s5-301-gpu.json").write_text("{not json")
    (root / "proximal_inertial_s5" / "302" / "task_result.json").write_text('{"validation": null}')
    (root / "profiles" / "proximal_inertial_s5-302-gpu.json").write_text('{"steady_step_seconds": 7}')
    text = JR.summary(root)
    assert "gpu profile unreadable" in text and "val acc n/a%" in text
    assert JR.main(["summary", str(tmp_path / "does-not-exist")]) == 0


def _report(path, tasks, unavailable):
    path.write_text(json.dumps({"tasks": [{}] * tasks, "unavailable": [{}] * unavailable}))
    return path


@pytest.mark.parametrize("tasks,unavailable,status,code", [
    (7, 0, "PASS audited=7 unavailable=0", 0),
    (5, 2, "PARTIAL audited=5 unavailable=2", 0),
    (0, 7, "FAIL audited=0 unavailable=7", 1)])
def test_p_audit_status(tmp_path, tasks, unavailable, status, code, capsys):
    path = _report(tmp_path / "r.json", tasks, unavailable)
    assert JR.p_audit_status(path) == status
    assert JR.main(["p-audit-status", str(path)]) == code
    assert capsys.readouterr().out.strip() == status


def test_p_audit_status_fails_on_missing_or_corrupt_report(tmp_path):
    assert JR.p_audit_status(tmp_path / "missing.json").startswith("FAIL")
    (tmp_path / "bad.json").write_text("{}")
    assert JR.p_audit_status(tmp_path / "bad.json").startswith("FAIL")


def _status_block():
    """The shell's post-wait P-audit decision, extracted verbatim from the launcher."""
    text = (ROOT / "bin/run_experiments/cluster_s5_proximal_inertia.sh").read_text()
    start = text.index('P_AUDIT_STATUS="$(cat')
    return text[start:text.index("esac", start) + 4]


@pytest.mark.parametrize("content,failed", [
    ("PASS audited=7 unavailable=0", "0"), ("PARTIAL audited=5 unavailable=2", "0"),
    ("FAIL block_audit_exit=1", "1"), ("", "1"), (None, "1")])
def test_launcher_marks_job_failed_unless_p_audit_passes(tmp_path, content, failed):
    (tmp_path / "status").mkdir()
    if content is not None:
        (tmp_path / "status" / "p-audit").write_text(content + "\n")
    script = f'set -euo pipefail\nRUN_ROOT="{tmp_path}"\nFAILED=0\n{_status_block()}\necho "FAILED=$FAILED"\n'
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().endswith(f"FAILED={failed}")
    if failed == "1":
        assert "FAIL: trained-P audit" in done.stderr
    assert "exit 1" in (ROOT / "bin/run_experiments/cluster_s5_proximal_inertia.sh").read_text()
