"""Production-float32 integration for the isolated proximal memory mixture."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_history import runner as P
from experiments.s5_proximal_memory_mix import runner as M
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full import step_telemetry as ST


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def production_float32():
    assert not jax.config.read("jax_enable_x64"), "run this module in its own float32 pytest process"


@pytest.fixture
def tiny(monkeypatch):
    for name, value in {"D_MODEL": 4, "SSM_SIZE_BASE": 8, "BLOCKS": 1,
                        "N_LAYERS": 2, "SEQ_LEN": 16, "BATCH_SIZE": 2}.items():
        monkeypatch.setattr(R, name, value)


def _assert_trees_identical(left, right):
    a, ta = jax.tree_util.tree_flatten(left)
    b, tb = jax.tree_util.tree_flatten(right)
    assert ta == tb
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_scoped_registration_restores_every_reference_on_exception():
    hs = {key: getattr(H, key) for key in (
        "ARM", "ALPHA", "NormalizedGradientHistoryS5SSM", "production_transitions",
        "history_pole_summary", "protocol_identity", "registered_history", "SCAN")}
    rs = {key: getattr(R, key) for key in (
        "RUNNABLE_ARMS", "ssm_factory", "radius_array", "pole_summary", "production_check",
        "ARM_CONFIGS", "SCIENTIFIC_NAMES", "SEEDS", "recurrent_state_size")}
    old_telemetry = ST.radius_array
    with pytest.raises(RuntimeError, match="intentional"):
        with M.configured(seed=1234):
            with H.registered_history():
                assert 1234 in R.SEEDS
                assert M.ARM in R.RUNNABLE_ARMS
                factory = R.ssm_factory(M.ARM)
                assert factory.keywords["history_alpha"] == 1
                assert factory.keywords["slow_weight"] == 4/9
                assert R.ARM_CONFIGS[M.ARM]["extra_parameters"] == ()
                assert R.recurrent_state_size(M.ARM)["states_per_complex_mode"] == 3
                raise RuntimeError("intentional")
    assert all(getattr(H, key) is value for key, value in hs.items())
    assert all(getattr(R, key) is value for key, value in rs.items())
    assert ST.radius_array is old_telemetry


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0}, {"alpha": float("nan")}, {"fast_ratio": 0},
    {"fast_ratio": 2}, {"slow_ratio": float("inf")},
    {"slow_weight": -0.1}, {"slow_weight": 1.1}, {"proximal_ratio": 0.49},
    {"proximal_ratio": float("nan")}, {"seed": 0}, {"seed": True},
])
def test_invalid_or_unstable_configuration_refused(kwargs):
    with pytest.raises(ValueError):
        with M.configured(**kwargs):
            pytest.fail("invalid configuration entered")


def test_bound_accepts_exact_frontier_without_changing_coefficients():
    assert M.validate_config() == pytest.approx(.5, abs=1e-15)
    ratio = .5-2e-16
    with M.configured(proximal_ratio=ratio):
        with H.registered_history():
            assert R.ARM_CONFIGS[M.ARM]["proximal_ratio"] == ratio
    with pytest.raises(ValueError, match="stability"):
        M.validate_config(proximal_ratio=.5-1e-8)


def test_identity_binds_all_memory_settings_sources_and_seed(tmp_path):
    (tmp_path/"manifest.json").write_text("{}")
    identities = []
    for kwargs, seed in (({}, 301), ({"alpha": .1}, 301),
                         ({"fast_ratio": .05}, 301), ({"slow_ratio": .9}, 301),
                         ({"slow_weight": .3}, 301), ({"proximal_ratio": .6}, 301),
                         ({}, 991)):
        with M.configured(**kwargs, seed=seed):
            identity = H.protocol_identity(tmp_path, seed)
            identities.append(identity)
            assert identity["arm"] == M.ARM and identity["epochs"] == 15
            assert identity["batch_size"] == 16 and identity["outer_optimizer"] == "unchanged"
            assert identity["scan_implementation"] == M.SCAN
            assert "s5/proximal_memory_mix.py" in identity["source_sha256"]
            assert "experiments/s5_proximal_memory_mix/runner.py" in identity["source_sha256"]
    assert len({json.dumps(x, sort_keys=True) for x in identities}) == len(identities)


def test_full_native_and_proximal_backbone_optimizer_identity(tiny):
    native = R.init_state("native_matched_s5", 304)
    with P.configured(1):
        with H.registered_history():
            proximal = R.init_state(P.ARM, 304)
    with M.configured(seed=304):
        with H.registered_history():
            mixed = R.init_state(M.ARM, 304)
    for other in (proximal, mixed):
        _assert_trees_identical(native.params, other.params)
        _assert_trees_identical(native.opt_state, other.opt_state)
        _assert_trees_identical(native.batch_stats, other.batch_stats)


def test_actual_pole_summary_uses_mixture_not_previous_history(tiny):
    with M.configured():
        with H.registered_history():
            state = R.init_state(M.ARM, 301)
            poles = R.pole_summary(state.params, M.ARM)
            assert poles["n_modes"] == 8
            assert poles["production_precision"]["unstable"] == 0
            assert poles["production_precision"]["nonfinite"] == 0
            assert poles["max_expm_vs_generator_radius_difference"] < 1e-12
            assert float(H.history_radius(state.params)) == pytest.approx(
                poles["production_precision"]["max_radius_production"], abs=2e-6)
            actual = np.asarray(M.transitions(state.params))
    previous = np.asarray(P.transitions(state.params, 1, .5))
    assert actual.shape == previous.shape
    assert np.max(abs(np.sort_complex(np.linalg.eigvals(actual))-
                      np.sort_complex(np.linalg.eigvals(previous)))) > 1e-3


def test_real_two_step_smoke_checkpoint_and_strict_gpu_gate(tiny, tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    data = {
        "train": (rng.normal(size=(8, 16, 1)).astype(np.float32), np.arange(8)%10),
        "val": (rng.normal(size=(4, 16, 1)).astype(np.float32), np.arange(4)%10),
    }
    cache = tmp_path/"data"; cache.mkdir(); (cache/"manifest.json").write_text("{}")
    loads = []
    def load(path, splits):
        loads.append(splits)
        return data, {}
    monkeypatch.setattr(R.EXPERIMENT_DATA, "load_official_raw", load)
    out = tmp_path/"smoke"
    argv = ["--seed", "304", "--out", str(out), "--data-cache", str(cache), "--smoke"]
    M.main(argv)
    report = json.loads((out/"smoke_result.json").read_text())
    production = json.loads((out/"production_check.json").read_text())
    identity = json.loads((out/"memory_mix_protocol.json").read_text())
    assert report["code_identifier"] == M.ARM and report["seed"] == 304
    assert report["checkpoint_restored"] and report["training_steps"] == 2
    assert report["backend"] == "cpu" and loads == [("train", "val")]
    assert production["x64_enabled"] is False and production["epochs_requested"] == 15
    assert report["protocol_identity"] == identity
    assert report["slow_weight"] == 4/9 and report["proximal_ratio"] == .5
    # CPU smoke cannot authorize training. A declared GPU copy exercises the
    # remaining gate checks without pretending that GPU execution was tested.
    with pytest.raises(ValueError, match="GPU smoke"):
        M.check_smoke_gate(report, production, identity)
    pretend = copy.deepcopy(report); pretend["backend"] = "gpu"
    M.check_smoke_gate(pretend, production, identity)
    for field, value in (("seed", 305), ("training_steps", 1), ("scan_implementation", "wrong"),
                         ("slow_weight", .1), ("epochs_requested", 14),
                         ("validation_accuracy", float("nan"))):
        wrong = copy.deepcopy(pretend); wrong[field] = value
        with pytest.raises(ValueError):
            M.check_smoke_gate(wrong, production, identity)
    for field, value in (("x64_enabled", True), ("gradients_finite", False),
                         ("fast_ratio", .2), ("epochs_requested", 14)):
        wrong = copy.deepcopy(production); wrong[field] = value
        with pytest.raises(ValueError):
            M.check_smoke_gate(pretend, wrong, identity)
    wrong = copy.deepcopy(production)
    wrong["initial_poles"]["production_precision"]["unstable"] = 1
    with pytest.raises(ValueError, match="unstable"):
        M.check_smoke_gate(pretend, wrong, identity)
    with pytest.raises(SystemExit):
        M.main(argv)
    assert H.ARM == "gradient_history_s5" and M.ARM not in R.RUNNABLE_ARMS
    assert R.SEEDS == (301, 302, 303)


def test_profile_compiles_without_data_and_restores_registration(tiny, tmp_path):
    from experiments.s5_gradient_history import memory_profile as MP
    old = MP.ARMS
    out = tmp_path/"profile.json"
    M.main(["--seed", "777", "--out", str(out), "--profile"])
    result = json.loads(out.read_text())
    assert result["arm"] == M.ARM and result["seed_for_shape_construction"] == 777
    assert result["training_executed"] is False and result["data_read"] is False
    assert result["slow_weight"] == 4/9 and result["history_alpha"] == 1
    assert len(result["updates"]) == 2
    assert all(x["memory"]["buffer_accounting_total_bytes"] > 0 for x in result["updates"])
    assert "s5/proximal_memory_mix.py" in result["source_sha256"]
    assert MP.ARMS is old and M.ARM not in R.RUNNABLE_ARMS


def test_cli_refuses_float64_before_profile_execution(tmp_path):
    out = tmp_path/"profile.json"
    process = subprocess.run([
        sys.executable, "-m", "experiments.s5_proximal_memory_mix.runner",
        "--seed", "301", "--out", str(out), "--profile",
    ], cwd=ROOT, env={**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"},
        text=True, capture_output=True)
    assert process.returncode != 0 and "JAX_ENABLE_X64=0" in process.stderr
    assert not out.exists()
