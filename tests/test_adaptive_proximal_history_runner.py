"""Integration checks for the single learned-horizon intervention."""
import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from experiments.s5_adaptive_proximal_history import runner as A
from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_history import runner as P
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full import step_telemetry as ST
from s5 import train_helpers as TH


@pytest.fixture(autouse=True)
def float32_only():
    assert not jax.config.read("jax_enable_x64")


@pytest.fixture
def tiny(monkeypatch):
    for name, value in {"D_MODEL": 8, "SSM_SIZE_BASE": 16, "BLOCKS": 2,
                        "N_LAYERS": 2, "SEQ_LEN": 16, "BATCH_SIZE": 2}.items():
        monkeypatch.setattr(R, name, value)


def test_scoped_changes_restore_after_error():
    names = ("ARM", "ALPHA", "NormalizedGradientHistoryS5SSM", "production_transitions",
             "history_pole_summary", "protocol_identity", "registered_history", "check_smoke_gate")
    old = {k: getattr(H, k) for k in names}
    old_r = {k: getattr(R, k) for k in ("init_state", "ssm_factory", "RUNNABLE_ARMS", "radius_array",
                                      "production_check", "pole_summary", "ARM_CONFIGS")}
    group, radius = TH.no_bc_decay_group, ST.radius_array
    with pytest.raises(RuntimeError, match="intentional"):
        with A.configured(A.ARM):
            with H.registered_history():
                assert H.ALPHA == 1 and H.ARM == A.ARM
                assert TH.no_bc_decay_group(A.HORIZON_KEY) == TH.no_bc_decay_group("log_step") == "ssm"
                assert R.ssm_factory(A.ARM).keywords["trainable_horizon"] is True
                raise RuntimeError("intentional")
    assert all(getattr(H, k) is v for k, v in old.items())
    assert all(getattr(R, k) is v for k, v in old_r.items())
    assert TH.no_bc_decay_group is group and ST.radius_array is radius
    with pytest.raises(ValueError, match="unsupported"):
        with A.configured("not_an_arm"):
            pass


def test_initialization_and_actual_optimizer_updates_match_old_p(tiny):
    with P.configured(1):
        with H.registered_history():
            original = R.init_state(P.ARM, 301)
            old_updated = original.apply_gradients(grads=jax.tree_util.tree_map(jnp.ones_like, original.params))
    base, after_base = flatten_dict(original.params), flatten_dict(old_updated.params)
    previous = None
    for arm in A.ARMS:
        with A.configured(arm):
            with H.registered_history():
                state = R.init_state(arm, 301)
                flat = flatten_dict(state.params)
                extra = [k for k in flat if k[-1] == A.HORIZON_KEY]
                assert sum(flat[k].size for k in extra) == 16
                assert set(flat) - set(extra) == set(base)
                for k in base:
                    np.testing.assert_array_equal(flat[k], base[k])
                    assert TH.no_bc_decay_group(k[-1]) == ("ssm" if k[-1] in TH.NO_BC_DECAY_SSM_KEYS else "regular")
                for k in extra:
                    np.testing.assert_array_equal(flat[k], np.zeros_like(flat[k]))
                if previous is not None:
                    for k in flat:
                        np.testing.assert_array_equal(flat[k], previous[k])
                previous = flat
                updated = state.apply_gradients(grads=jax.tree_util.tree_map(jnp.ones_like, state.params))
                after = flatten_dict(updated.params)
                for k in base:
                    np.testing.assert_array_equal(after[k], after_base[k])
                for k in extra:
                    if arm == A.ARM:
                        step_key = k[:-1] + ("log_step",)
                        expected = after[step_key][:, 0] - flat[step_key][:, 0]
                        np.testing.assert_allclose(after[k], expected, atol=3e-7, rtol=0)
                        assert np.any(after[k] != 0)
                    else:
                        np.testing.assert_array_equal(after[k], np.zeros_like(after[k]))


def test_identity_and_gate_bind_trainable_choice(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    identities = []
    for arm in A.ARMS:
        with A.configured(arm):
            identities.append(H.protocol_identity(tmp_path, 301))
    candidate, control = identities
    assert candidate != control
    assert candidate["trainable_horizon"] is True and control["trainable_horizon"] is False
    assert candidate["horizon_optimizer_group"] == "ssm"
    assert control["horizon_optimizer_group"] == "none"
    assert candidate["alpha"] == 1 and candidate["proximal_ratio"] == .5
    assert candidate["epochs"] == 15 and candidate["batch_size"] == 16
    assert "s5/adaptive_proximal_history.py" in candidate["source_sha256"]
    with A.configured(A.ARM):
        with pytest.raises(ValueError, match="protocol"):
            H.check_smoke_gate({"protocol_identity": control}, candidate)


def _example_params():
    return {"seq": {"Lambda_re": jnp.array([-.1, -.3], jnp.float32),
                    "Lambda_im": jnp.array([.2, 1.], jnp.float32),
                    "log_step": jnp.array([[-1.], [-2.]], jnp.float32),
                    A.HORIZON_KEY: jnp.zeros(2, jnp.float32)}}


def test_diagnostics_use_learned_horizons_and_handle_invalid_parameters():
    initial = _example_params()
    changed = copy.deepcopy(initial)
    changed["seq"][A.HORIZON_KEY] = jnp.array([.6, -.4], jnp.float32)
    a, b = A.pole_summary(initial), A.pole_summary(changed)
    assert not np.allclose(A.transitions(initial), A.transitions(changed))
    assert a["horizon_multiplier"]["p50"] == 1
    assert b["horizon_multiplier"]["max"] > 1 and b["horizon_multiplier"]["min"] < 1
    assert b["production_precision"]["max_abs_radius_difference_production_vs_float64"] < 2e-6
    assert b["production_precision"]["unstable"] == 0
    np.testing.assert_allclose(b["log_horizon_scale_values"]["seq"], [.6, -.4], atol=1e-7)
    A.horizon_gate(a, A.CONTROL)
    with pytest.raises(ValueError, match="moved"):
        A.horizon_gate(b, A.CONTROL)
    with pytest.raises(ValueError, match="did not update"):
        A.horizon_gate(a, A.ARM, after_updates=True)
    A.horizon_gate(b, A.ARM, after_updates=True)
    bad = copy.deepcopy(initial)
    bad["seq"][A.HORIZON_KEY] = jnp.array([float("nan"), 0], jnp.float32)
    summary = A.pole_summary(bad)
    assert summary["nonfinite_modes"] == 1 and not summary["positive_finite_horizons"]
    with pytest.raises(ValueError, match="positive and finite"):
        A.horizon_gate(summary, A.ARM)


@pytest.mark.parametrize("arm", A.ARMS)
def test_actual_cpu_smoke_preserves_freeze_and_labels(tiny, tmp_path, monkeypatch, arm):
    rng = np.random.default_rng(6)
    data = {"train": (rng.normal(size=(8, 16, 1)).astype(np.float32), np.arange(8) % 10),
            "val": (rng.normal(size=(4, 16, 1)).astype(np.float32), np.arange(4) % 10)}
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    requested = []

    def load(path, splits):
        requested.append(splits)
        return data, {}

    monkeypatch.setattr(R.EXPERIMENT_DATA, "load_official_raw", load)
    out = tmp_path / "smoke"
    A.main(["--arm", arm, "--seed", "301", "--out", str(out),
            "--data-cache", str(cache), "--smoke"])
    smoke = json.loads((out / "smoke_result.json").read_text())
    assert smoke["code_identifier"] == arm
    assert smoke["checkpoint_restored"] and smoke["telemetry_check"]["values_agree"]
    assert smoke["history_alpha"] == 1 and smoke["proximal_ratio"] == .5
    assert smoke["trainable_horizon"] == (arm == A.ARM)
    assert smoke["extra_trainable_parameters"] == (16 if arm == A.ARM else 0)
    assert smoke["frozen_parameter_count"] == (0 if arm == A.ARM else 16)
    assert smoke["stored_parameter_count"] - smoke["trainable_parameter_count"] == smoke["frozen_parameter_count"]
    A.horizon_gate(smoke["poles_after_smoke_steps"], arm, after_updates=True)
    assert requested == [("train", "val")]
    assert H.ARM == "gradient_history_s5" and arm not in R.RUNNABLE_ARMS


def test_profile_is_abstract_and_scoped(tiny, tmp_path):
    from experiments.s5_gradient_history import memory_profile as MP
    original = MP.ARMS
    out = tmp_path / "profile.json"
    report = A.memory_profile(A.ARM, 301, out)
    assert report["arm"] == A.ARM and report["trainable_horizon"]
    assert not report["training_executed"] and not report["data_read"]
    assert report["extra_parameter_count"] == 16
    assert all(row["memory"]["buffer_accounting_total_bytes"] > 0 for row in report["updates"])
    assert "s5/adaptive_proximal_history.py" in report["source_sha256"]
    assert MP.ARMS is original and A.ARM not in R.RUNNABLE_ARMS
    with pytest.raises(ValueError, match="existing output"):
        A.memory_profile(A.ARM, 301, out)
