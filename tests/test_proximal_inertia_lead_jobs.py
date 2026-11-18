"""Baseline-reuse gate, paired report and cluster scripts for the sigma=0 ablation.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest tests/test_proximal_inertia_lead_jobs.py -q
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import pytest

from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_inertia import baseline_check as BC
from experiments.s5_proximal_inertia import job_report as JR
from experiments.s5_proximal_inertia import runner as PIR
from tests.test_block_audit import _task

ROOT = Path(__file__).resolve().parents[1]
CFG = {"history_alpha": .1, "proximal_ratio": .5, "mass_ratio": 1.5}


@pytest.fixture(autouse=True)
def float32_only():
    assert not jax.config.read("jax_enable_x64")


def _baseline(tmp_path, arm="proximal_inertial_s5", config=CFG, mutate=None):
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.json").write_text('{"official": true}')
    root = tmp_path / "job-76984"
    root.mkdir()
    (root / "run_metadata.txt").write_text(f"commit={BC.BASELINE_COMMIT}\njob=76984\n")
    for seed in (301, 302):
        t = _task(tmp_path / f"stage{seed}", arm, seed=seed, config=config)
        (root / "proximal_inertial_s5").mkdir(exist_ok=True)
        t.rename(root / "proximal_inertial_s5" / str(seed))
        with PIR.configured(arm if arm in PIR.ARMS else "proximal_inertial_s5"):
            ident = H.protocol_identity(cache, seed)
        ident.pop("input_lead", None)                       # acb4d85 identities had no input_lead
        ident["git_commit"] = BC.BASELINE_COMMIT
        if mutate:
            mutate(ident)
        (root / "proximal_inertial_s5" / str(seed) / "history_protocol.json").write_text(json.dumps(ident))
    return root, cache


def test_matched_completed_baseline_supports_reuse(tmp_path):
    root, cache = _baseline(tmp_path)
    rep = BC.check(root, cache)
    assert rep["reuse_supported"], rep["reasons"]
    assert rep["differs_by_design"] == ["arm", "input_lead"]
    assert all(t["config"]["mass_ratio"] == 1.5 for t in rep["tasks"])


@pytest.mark.parametrize("mutate,reason", [
    (lambda i: i.update(data_manifest_sha256="0" * 64), "data_manifest_sha256"),
    (lambda i: i.update(jax_version="0.0.1"), "jax_version"),
    (lambda i: i.update(batch_size=32), "batch_size"),
    (lambda i: i.update(git_commit="deadbeef"), "trained at")])
def test_protocol_mismatch_blocks_reuse(tmp_path, mutate, reason):
    root, cache = _baseline(tmp_path, mutate=mutate)
    rep = BC.check(root, cache)
    assert not rep["reuse_supported"] and any(reason in r for r in rep["reasons"])


def test_control_or_unfinished_baseline_blocks_reuse(tmp_path):
    root, cache = _baseline(tmp_path / "a", config={**CFG, "mass_ratio": 1.0})
    rep = BC.check(root, cache)
    assert not rep["reuse_supported"] and any("matched" in r for r in rep["reasons"])
    root, cache = _baseline(tmp_path / "b")
    (root / "proximal_inertial_s5" / "302" / "task_result.json").unlink()
    rep = BC.check(root, cache)
    assert not rep["reuse_supported"] and any("302" in r for r in rep["reasons"])
    with pytest.raises(SystemExit):
        BC.main(["--baseline-root", str(root), "--data-cache", str(cache), "--out", str(tmp_path / "x.json")])


def test_paired_report(tmp_path):
    new, base = tmp_path / "new", tmp_path / "base"
    for root, arm, acc in ((new, "proximal_inertial_nolead_s5", 3600), (base, "proximal_inertial_s5", 3590)):
        for seed in (301, 302):
            d = root / arm / str(seed); d.mkdir(parents=True)
            (d / "task_result.json").write_text(json.dumps({"selected_epoch": 14, "validation": {
                "accuracy": acc / 3703, "cross_entropy": 0.1, "n": 3703}}))
    text = JR.paired(new, base)
    assert text.count("(+10)") == 2 and "No accuracy claim" in text
    (new / "proximal_inertial_nolead_s5" / "302" / "task_result.json").unlink()
    assert "missing" in JR.paired(new, base) and JR.main(["paired", str(new), str(base)]) == 0


@pytest.mark.parametrize("name,tokens,wrong", [("probe", "0", "0,1"), ("train", "0,1", "0")])
def test_scripts_syntax_dryrun_and_gpu_count(tmp_path, name, tokens, wrong):
    path = ROOT / f"bin/run_experiments/cluster_s5_proximal_inertia_lead_{name}.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        pytest.skip("scripts refuse a modified checkout; run after committing")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env = {**os.environ, "EXPECTED_COMMIT": sha, "DRY_RUN": "1", "SLURM_JOB_ID": "4242",
           "CUDA_VISIBLE_DEVICES": tokens, "PROSPECTIVE_VENV": sys.prefix,
           "PROSPECTIVE_RUNS": str(tmp_path), "S5_THREE_ARM_DATA": str(tmp_path), "JAX_ENABLE_X64": "0"}
    done = subprocess.run(["bash", str(path)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "DRY_RUN" in done.stdout and "job-76984" in done.stdout
    assert not (tmp_path / "s5-proximal-inertia-lead").exists()
    bad = subprocess.run(["bash", str(path)], cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": wrong},
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "exactly" in bad.stderr
