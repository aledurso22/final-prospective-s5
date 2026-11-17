"""checkpoint_compare: provenance, pairing, refusal, and matched-init identity.

Synthetic run directories laid out exactly as the S5 launchers write them,
holding the float32 production init as the "trained" checkpoint (so every
trained/init ratio must be exactly 1), plus one perturbed copy.

    python -m pytest tests/test_checkpoint_compare.py -q
"""

import json
import os

import numpy as onp
import pytest
from flax import serialization
from flax.traverse_util import unflatten_dict

from experiments.s5_predictor_corrector import checkpoint_compare as CC
from experiments.s5_predictor_corrector.production_init import (
    production_init_flat)


def _write_task(root, arm, seed, epoch=15, flat=None, result=True, rid=None):
    task = os.path.join(root, arm, str(seed))
    os.makedirs(task, exist_ok=True)
    flat = production_init_flat(arm, seed) if flat is None else flat
    state = {"step": 0, "params": unflatten_dict(flat), "opt_state": {},
             "batch_stats": {}}
    with open(os.path.join(task, f"checkpoint_epoch_{epoch:02d}.msgpack"), "wb") as fh:
        fh.write(serialization.msgpack_serialize(state))
    if result:
        with open(os.path.join(task, "task_result.json"), "w") as fh:
            json.dump({"code_identifier": rid or arm, "seed": seed,
                       "selected_epoch": 11}, fh)
    with open(os.path.join(root, "run_metadata.txt"), "w") as fh:
        fh.write("authoritative_commit=abc123\nscan_implementation=factored\n")
    return task


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("runs")
    nat = _write_task(str(base / "native_run"), CC.NATIVE, 301)
    gen = _write_task(str(base / "gen_run"), CC.GENERALIZED, 301)
    return base, nat, gen


def test_matched_init_checkpoints_give_identity(runs, tmp_path):
    base, nat, gen = runs
    rep = CC.main(["--native", nat, "--generalized", gen, "--epoch", "15",
                   "--out", str(tmp_path)])
    s = rep["seeds"][0]
    assert s["seed"] == 301 and s["reconstructed_ssm_init_identical_across_arms"]
    a = s["actual"]
    for arm, keys in (("native", ("Delta", "decay")),
                      ("generalized", ("Delta", "T_over_gamma"))):
        for k in keys:
            d = a[arm]["trained_over_init"][k]
            assert d["p0"] == pytest.approx(1) and d["p100"] == pytest.approx(1)
    assert a["paired_trained"] == a["paired_init"]
    cf = s["counterfactual_at_generalized_learned_params"]
    assert cf["A"]["max_abs_diff_vs_production_coefficients"] < 1e-6
    assert cf["A"]["stable"] == 384 and cf["A"]["unstable"] == 0   # 6 x 64
    assert "dc_delay_tokens_stable_modes_only" in cf["C"]
    # matched init really is the training init: T/gamma = 0.05 there
    assert a["generalized"]["matched_init"]["T_over_gamma"]["p50"] == pytest.approx(0.05, rel=1e-5)
    prov = rep["inputs"]["native"][0]
    assert prov["epoch"] == 15 and prov["seed"] == 301
    assert prov["run_metadata"]["authoritative_commit"] == "abc123"
    assert len(prov["checkpoint_sha256"]) == 64
    assert os.path.exists(tmp_path / "checkpoint_compare.json")


def test_trained_drift_is_measured_on_effective_poles(tmp_path):
    flat = dict(production_init_flat(CC.NATIVE, 301))
    for k in list(flat):
        if k[-1] == "log_step":
            flat[k] = flat[k] + onp.float32(onp.log(2.0))   # Delta doubled
    nat = _write_task(str(tmp_path / "n"), CC.NATIVE, 301, flat=flat)
    gen = _write_task(str(tmp_path / "g"), CC.GENERALIZED, 301)
    rep = CC.main(["--native", nat, "--generalized", gen, "--epoch", "15",
                   "--out", str(tmp_path / "o")])
    a = rep["seeds"][0]["actual"]
    assert a["native"]["trained_over_init"]["Delta"]["p50"] == pytest.approx(2, rel=1e-5)
    # Re(Lambda) is the same for every mode, so decay time halves exactly
    assert a["native"]["trained_over_init"]["decay"]["p50"] == pytest.approx(0.5, rel=1e-4)


def test_selected_epoch_is_read_from_task_result(runs, tmp_path):
    base, nat, gen = runs
    with pytest.raises(CC.CheckpointError, match="checkpoint_epoch_11"):
        CC.main(["--native", nat, "--generalized", gen, "--epoch", "selected",
                 "--out", str(tmp_path)])


def test_unmatched_seeds_fail(runs, tmp_path):
    base, nat, gen = runs
    gen302 = _write_task(str(tmp_path / "g"), CC.GENERALIZED, 302,
                         flat=production_init_flat(CC.GENERALIZED, 301))
    with pytest.raises(CC.CheckpointError, match="unmatched seeds"):
        CC.main(["--native", nat, "--generalized", gen302, "--epoch", "15",
                 "--out", str(tmp_path)])


def test_native_checkpoint_passed_as_generalized_fails(tmp_path):
    fake = _write_task(str(tmp_path / "g"), CC.GENERALIZED, 301,
                       flat=production_init_flat(CC.NATIVE, 301))
    nat = _write_task(str(tmp_path / "n"), CC.NATIVE, 301)
    with pytest.raises(CC.CheckpointError, match="no generalized_"):
        CC.main(["--native", nat, "--generalized", fake, "--epoch", "15",
                 "--out", str(tmp_path / "o")])


def test_wrong_arm_directory_or_identity_fails(runs, tmp_path):
    base, nat, gen = runs
    with pytest.raises(CC.CheckpointError, match="parent directory"):
        CC.main(["--native", gen, "--generalized", gen, "--epoch", "15",
                 "--out", str(tmp_path)])
    bad = _write_task(str(tmp_path / "x"), CC.NATIVE, 301,
                      rid="zucchet_prospective_s5")
    with pytest.raises(CC.CheckpointError, match="code_identifier"):
        CC.main(["--native", bad, "--generalized", gen, "--epoch", "15",
                 "--out", str(tmp_path)])


def test_missing_checkpoint_and_task_result_fail(runs, tmp_path):
    base, nat, gen = runs
    with pytest.raises(CC.CheckpointError, match="missing"):
        CC.main(["--native", nat, "--generalized", gen, "--epoch", "14",
                 "--out", str(tmp_path)])
    bare = _write_task(str(tmp_path / "b"), CC.NATIVE, 301, result=False)
    with pytest.raises(CC.CheckpointError, match="selected_epoch"):
        CC.main(["--native", bare, "--generalized", gen, "--epoch", "selected",
                 "--out", str(tmp_path)])


def test_existing_output_is_never_overwritten(runs, tmp_path):
    base, nat, gen = runs
    out = tmp_path / "evidence"
    out.mkdir()
    (out / "checkpoint_compare.json").write_text('{"preserved": true}')
    with pytest.raises(CC.CheckpointError, match="refusing to overwrite"):
        CC.main(["--native", nat, "--generalized", gen, "--epoch", "15",
                 "--out", str(out)])
    assert (out / "checkpoint_compare.json").read_text() == '{"preserved": true}'


def test_unstable_modes_are_counted_and_excluded(tmp_path):
    """Push T/gamma up on the generalized checkpoint until some counterfactual
    modes go unstable: they must be counted, and response statistics must be
    over stable modes only."""
    flat = dict(production_init_flat(CC.GENERALIZED, 301))
    for k in list(flat):
        if k[-1] == "generalized_gamma_raw":
            flat[k] = onp.full_like(flat[k], -6.0)       # gamma ~ 2.5e-3
        if k[-1] == "generalized_T_raw":
            flat[k] = onp.full_like(flat[k], 1.0)        # T ~ 1.31
    gen = _write_task(str(tmp_path / "g"), CC.GENERALIZED, 301, flat=flat)
    nat = _write_task(str(tmp_path / "n"), CC.NATIVE, 301)
    rep = CC.main(["--native", nat, "--generalized", gen, "--epoch", "15",
                   "--out", str(tmp_path / "o")])
    cf = rep["seeds"][0]["counterfactual_at_generalized_learned_params"]
    unstable = {v: cf[v]["unstable"] for v in "ABCD"}
    assert any(n > 0 for n in unstable.values()), unstable
    for v in "ABCD":
        assert cf[v]["stable"] + cf[v]["unstable"] == cf[v]["n_modes"] == 384
        if cf[v]["unstable"]:
            assert cf[v]["max_radius"] > 1
            assert cf[v]["unstable_radius"]["p0"] >= 1 - 1e-12
