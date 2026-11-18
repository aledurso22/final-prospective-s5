"""Runner registration and checkpoint audit for the sigma=0 ablation (float32).

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest tests/test_proximal_inertia_lead_integration.py -q
"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_inertia import block_audit as BA
from experiments.s5_proximal_inertia import runner as PIR
from experiments.s5_three_arm_full import runner as R
from tests.test_block_audit import _task

BASE, ABL = "proximal_inertial_s5", "proximal_inertial_nolead_s5"
CFG = {"history_alpha": .1, "proximal_ratio": .5, "mass_ratio": 1.5}


@pytest.fixture(autouse=True)
def float32_only():
    assert not jax.config.read("jax_enable_x64")


@pytest.fixture
def tiny(monkeypatch):
    for name, value in {"D_MODEL": 8, "SSM_SIZE_BASE": 16, "BLOCKS": 2,
                        "N_LAYERS": 2, "SEQ_LEN": 32, "BATCH_SIZE": 2}.items():
        monkeypatch.setattr(R, name, value)


def test_ablation_differs_from_matched_baseline_only_in_input_lead(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with PIR.configured(BASE):
        a = H.protocol_identity(tmp_path, 301)
    with PIR.configured(ABL):
        b = H.protocol_identity(tmp_path, 301)
    assert {k for k in a if a[k] != b[k]} == {"arm", "input_lead"}
    assert a["mass_ratio"] == b["mass_ratio"] == 1.5          # matched to m=1.5 tau, NOT the control
    assert (a["input_lead"], b["input_lead"]) == (1.0, 0.0)
    with PIR.configured(ABL), H.registered_history():
        f = R.ssm_factory(ABL)
        assert f.keywords["input_lead"] == 0.0 and f.keywords["mass_ratio"] == 1.5


def test_native_parameters_identical_and_cpu_smoke(tiny, tmp_path):
    native = R.init_state("native_matched_s5", 302)
    rng = np.random.default_rng(1)
    train = (rng.normal(size=(8, R.SEQ_LEN, 1)).astype(np.float32), np.arange(8) % 10)
    val = (rng.normal(size=(4, R.SEQ_LEN, 1)).astype(np.float32), np.arange(4) % 10)
    with PIR.configured(ABL), H.registered_history():
        state = R.init_state(ABL, 302)
        a, b = flatten_dict(native.params), flatten_dict(state.params)
        assert set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
        poles = R.pole_summary(state.params, ABL)
        assert poles["input_lead"] == 0.0 and poles["production_precision"]["unstable"] == 0
        model = R.model_for(ABL, True)
        R.production_check(ABL, 302, str(tmp_path), (jnp.asarray(train[0][:2]),
                           jnp.asarray(train[1][:2])), state, model, epochs=15)
        smoke = R.full_path_smoke(ABL, 302, {"train": train, "val": val}, str(tmp_path),
                                  state, model, epochs=15)
    assert smoke["checkpoint_restored"] and smoke["training_steps"] == 2
    assert ABL not in R.RUNNABLE_ARMS


@pytest.fixture(scope="module")
def audited(tmp_path_factory):
    root = tmp_path_factory.mktemp("lead")
    task = _task(root, BASE, config=CFG)                 # legacy result: no input_lead key
    rep = BA.main(["--task", str(task), "--out", str(root / "a"), "--horizon", "512",
                   "--compare-input-lead", "0"])
    same = BA.main(["--task", str(task), "--out", str(root / "b"), "--horizon", "512",
                    "--compare-input-lead", "1"])
    return rep, same


def test_counterfactual_changes_only_the_input_column(audited):
    rep, same = audited
    t = rep["tasks"][0]
    assert t["config"]["input_lead"] == 1.0                 # legacy baseline read as sigma=1
    for stage in t["stages"].values():
        for layer in stage["layers"]:
            cf = layer["input_lead_counterfactual"]
            assert cf["input_lead"] == 0.0 and "counterfactual" in cf["label"]
            assert cf["max_abs_Ad_difference"] < 1e-12      # homogeneous dynamics unchanged
            assert cf["max_abs_bd_difference"] > 1e-4
            for d in ("forward", "backward"):
                kd = cf["kernel_difference"][d]
                assert kd["relative_difference_energy"] > 0
                assert "cross_pole_interference_fraction_of_mode_energy" in cf["directions"][d]
                assert "cross_pole_interference_fraction_of_mode_energy" in layer["directions"][d]
    for stage in same["tasks"][0]["stages"].values():
        for layer in stage["layers"]:
            cf = layer["input_lead_counterfactual"]
            assert cf["max_abs_bd_difference"] == 0
            assert cf["kernel_difference"]["forward"]["relative_difference_energy"] == 0


def test_ablation_result_must_declare_its_input_lead(tmp_path):
    t = _task(tmp_path, ABL, config=CFG)
    with pytest.raises(BA.AuditError, match="input_lead"):
        BA.main(["--task", str(t), "--out", str(tmp_path / "o"), "--horizon", "64"])
    t2 = _task(tmp_path / "ok", ABL, config={**CFG, "input_lead": 0.0})
    rep = BA.main(["--task", str(t2), "--out", str(tmp_path / "o2"), "--horizon", "64"])
    assert rep["tasks"][0]["config"]["input_lead"] == 0.0
    with pytest.raises(BA.AuditError, match="proximal inertial"):
        p = _task(tmp_path / "p", "proximal_history_s5", config={"history_alpha": 1.0, "proximal_ratio": .5})
        BA.main(["--task", str(p), "--out", str(tmp_path / "o3"), "--horizon", "64",
                 "--compare-input-lead", "0"])
