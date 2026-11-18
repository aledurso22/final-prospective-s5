"""Scoped proximal inertial runner integration, production float32 semantics.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest tests/test_proximal_inertia_runner.py -q
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_inertia import runner as PIR
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full import step_telemetry as ST

ROOT = Path(__file__).resolve().parents[1]
CAND, CTRL = "proximal_inertial_s5", "proximal_inertial_control_s5"


@pytest.fixture(autouse=True)
def float32_only():
    assert not jax.config.read("jax_enable_x64")


@pytest.fixture
def tiny(monkeypatch):
    for name, value in {"D_MODEL": 8, "SSM_SIZE_BASE": 16, "BLOCKS": 2,
                        "N_LAYERS": 2, "SEQ_LEN": 32, "BATCH_SIZE": 2}.items():
        monkeypatch.setattr(R, name, value)


def test_scoped_registration_is_restored_even_after_error():
    hs = {k: getattr(H, k) for k in ("ARM", "ALPHA", "NormalizedGradientHistoryS5SSM",
          "production_transitions", "history_pole_summary", "protocol_identity", "registered_history")}
    rs = {k: getattr(R, k) for k in ("RUNNABLE_ARMS", "ssm_factory", "radius_array", "pole_summary",
          "production_check", "ARM_CONFIGS", "SCIENTIFIC_NAMES", "recurrent_state_size")}
    old_st = ST.radius_array
    with pytest.raises(RuntimeError, match="intentional"):
        with PIR.configured(CAND):
            with H.registered_history():
                f = R.ssm_factory(CAND)
                assert f.keywords["mass_ratio"] == 1.5 and f.keywords["proximal_ratio"] == .5
                assert f.keywords["history_alpha"] == .1
                assert R.recurrent_state_size(CAND)["states_per_complex_mode"] == 2
                assert R.ARM_CONFIGS[CAND]["recurrence_constructor"] == "ProximalInertialS5SSM"
                raise RuntimeError("intentional")
    assert all(getattr(H, k) is v for k, v in hs.items())
    assert all(getattr(R, k) is v for k, v in rs.items())
    assert ST.radius_array is old_st
    with pytest.raises(ValueError):
        with PIR.configured("proximal_history_s5"):
            pass


def test_control_differs_only_in_mass_ratio(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with PIR.configured(CAND):
        a = H.protocol_identity(tmp_path, 301)
    with PIR.configured(CTRL):
        b = H.protocol_identity(tmp_path, 301)
    assert a["mass_ratio"] == 1.5 and b["mass_ratio"] == 1.0
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"arm", "mass_ratio"}
    assert a["alpha"] == .1 and a["proximal_ratio"] == .5 and a["epochs"] == 15 and a["batch_size"] == 16
    assert "s5/proximal_inertia.py" in a["source_sha256"]
    with PIR.configured(CAND):
        with pytest.raises(ValueError, match="protocol"):
            H.check_smoke_gate({"protocol_identity": b}, a)


@pytest.mark.parametrize("arm", [CAND, CTRL])
def test_native_parameters_and_optimizer_identical(tiny, arm):
    native = R.init_state("native_matched_s5", 301)
    with PIR.configured(arm), H.registered_history():
        state = R.init_state(arm, 301)
    a, b = flatten_dict(native.params), flatten_dict(state.params)
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    ta, tb = jax.tree_util.tree_flatten(native.opt_state), jax.tree_util.tree_flatten(state.opt_state)
    assert ta[1] == tb[1]
    for x, y in zip(ta[0], tb[0]):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("arm", [CAND, CTRL])
def test_cpu_smoke_poles_and_telemetry(tiny, tmp_path, arm):
    rng = np.random.default_rng(5)
    train = (rng.normal(size=(8, R.SEQ_LEN, 1)).astype(np.float32), np.arange(8) % 10)
    val = (rng.normal(size=(4, R.SEQ_LEN, 1)).astype(np.float32), np.arange(4) % 10)
    with PIR.configured(arm), H.registered_history():
        state = R.init_state(arm, 301)
        poles = R.pole_summary(state.params, arm)
        assert poles["n_modes"] == 16 and poles["block_width"] == 2
        assert poles["production_precision"]["unstable"] == 0
        assert float(H.history_radius(state.params)) == pytest.approx(
            poles["production_precision"]["max_radius_production"], abs=2e-6)
        model = R.model_for(arm, True)
        R.production_check(arm, 301, str(tmp_path), (jnp.asarray(train[0][:2]),
                           jnp.asarray(train[1][:2])), state, model, epochs=15)
        smoke = R.full_path_smoke(arm, 301, {"train": train, "val": val}, str(tmp_path),
                                  state, model, epochs=15)
    assert smoke["checkpoint_restored"] and smoke["telemetry_check"]["values_agree"]
    assert smoke["training_steps"] == 2 and smoke["code_identifier"] == arm
    assert smoke["poles_after_smoke_steps"]["mass_ratio"] == PIR.ARMS[arm]


def test_main_smoke_binds_protocol_and_restores(tiny, tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    data = {"train": (rng.normal(size=(8, 32, 1)).astype(np.float32), np.arange(8) % 10),
            "val": (rng.normal(size=(4, 32, 1)).astype(np.float32), np.arange(4) % 10)}
    cache = tmp_path / "data"; cache.mkdir(); (cache / "manifest.json").write_text("{}")
    requested = []
    monkeypatch.setattr(R.EXPERIMENT_DATA, "load_official_raw",
                        lambda path, splits: (requested.append(splits) or (data, {})))
    out = tmp_path / "smoke"
    PIR.main(["--arm", CTRL, "--seed", "302", "--out", str(out), "--data-cache", str(cache), "--smoke"])
    smoke = json.loads((out / "smoke_result.json").read_text())
    assert smoke["code_identifier"] == CTRL and smoke["mass_ratio"] == 1.0 and smoke["seed"] == 302
    assert smoke["protocol_identity"]["mass_ratio"] == 1.0
    assert smoke["backend"] == "cpu" and smoke["checkpoint_restored"]
    assert requested == [("train", "val")]           # the test split is never requested
    assert H.ARM == "gradient_history_s5" and CTRL not in R.RUNNABLE_ARMS
    with pytest.raises(ValueError, match="GPU smoke"):    # training refuses a CPU smoke
        PIR.main(["--arm", CTRL, "--seed", "302", "--out", str(tmp_path / "t"),
                  "--data-cache", str(cache), "--train", "--smoke-report",
                  str(out / "smoke_result.json")])


def test_compile_profile_and_gpu_profile_cpu(tiny, tmp_path):
    from experiments.s5_gradient_history import memory_profile as MP
    from experiments.s5_proximal_inertia import gpu_profile as GP
    old = MP.ARMS
    rep = PIR.memory_profile(CAND, 301, tmp_path / "profile.json")
    assert rep["mass_ratio"] == 1.5 and rep["training_executed"] is False
    assert all(r["memory"]["buffer_accounting_total_bytes"] > 0 for r in rep["updates"])
    assert MP.ARMS is old and CAND not in R.RUNNABLE_ARMS
    g = GP.main(["--arm", CTRL, "--steps", "3", "--out", str(tmp_path / "gpu.json")])
    assert g["backend"] == "cpu" and len(g["losses"]) == 3 and all(np.isfinite(g["losses"]))
    with pytest.raises(SystemExit):
        GP.main(["--arm", CTRL, "--steps", "3", "--out", str(tmp_path / "x.json"), "--require-gpu"])


def test_launchers_syntax_dryrun_and_gpu_count(tmp_path):
    scripts = [ROOT / "bin/run_experiments" / f"{s}_s5_proximal_inertia.sh" for s in ("cluster", "submit")]
    for path in scripts:
        subprocess.run(["bash", "-n", str(path)], check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env = {**os.environ, "EXPECTED_COMMIT": sha, "DRY_RUN": "1", "STAGE": "train",
           "SLURM_JOB_ID": "987654", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
           "PROSPECTIVE_VENV": sys.prefix, "PROSPECTIVE_RUNS": str(tmp_path),
           "S5_THREE_ARM_DATA": str(tmp_path), "JAX_ENABLE_X64": "0"}
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, text=True).strip()
    if dirty:
        pytest.skip("launchers refuse a modified checkout; run after committing")
    for path in scripts:
        done = subprocess.run(["bash", str(path)], cwd=ROOT, env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "proximal_inertial_control_s5" in done.stdout or "candidate/control" in done.stdout
    assert not (tmp_path / "s5-proximal-inertia").exists()
    done = subprocess.run(["bash", str(scripts[0])], cwd=ROOT,
                          env={**env, "CUDA_VISIBLE_DEVICES": "0,1"}, capture_output=True, text=True)
    assert done.returncode != 0 and "exactly four" in done.stderr
