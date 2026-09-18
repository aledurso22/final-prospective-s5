import os
import sys
import inspect

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.s5_three_arm_full import data
from experiments.s5_three_arm_full import runner
from experiments.s5_three_arm_full.finalize import restore_and_evaluate
from s5.generalized_prospective_ssm import generalized_zoh_coefficients
from s5.prospective_ssm import _clocked_coefficients
from s5.ssm import discretize_zoh
from s5.train_helpers import no_bc_decay_group


def test_protocol_uses_upstream_initializer_and_shared_stability_constraint():
    assert all(runner.ssm_kwargs(arm)["C_init"] == "lecun_normal"
               for arm in runner.ARM_ORDER)
    assert all(runner.ssm_kwargs(arm)["clip_eigs"] is True
               for arm in runner.ARM_ORDER)
    assert runner.SCIENTIFIC_NAMES["native_matched_s5"] == (
        "Native S5 recurrence under the shared stability constraint")


def test_raw_normalization_is_one_statistic_per_channel():
    train = np.asarray([[[0.0], [2.0]], [[4.0], [6.0]]], dtype=np.float32)
    values = np.asarray([[[1.0], [3.0]]], dtype=np.float32)
    normalized, mean, std = data.normalize_raw_audio(train, values)
    np.testing.assert_equal(mean.shape, (1, 1, 1))
    np.testing.assert_equal(std.shape, (1, 1, 1))
    np.testing.assert_allclose(mean, [[[3.0]]])
    np.testing.assert_allclose(std, [[[2.5819888]]], rtol=1e-6)
    np.testing.assert_allclose(normalized, [[[-0.774593], [0.0]]], rtol=1e-5)


def test_learning_rate_boundaries_are_fail_closed():
    steps = 10
    warmup_start, _ = runner.learning_rate_at_step(0, steps)
    warmup_end, _ = runner.learning_rate_at_step(steps - 1, steps)
    cosine_start, _ = runner.learning_rate_at_step(steps, steps)
    cosine_end, _ = runner.learning_rate_at_step(steps * runner.EPOCHS, steps)
    np.testing.assert_allclose(warmup_start, runner.LR / steps)
    np.testing.assert_allclose(warmup_end, runner.LR)
    np.testing.assert_allclose(cosine_start, runner.LR)
    np.testing.assert_allclose(cosine_end, runner.LR_FINAL)


def test_all_prospective_response_leaves_use_the_ssm_group():
    for key in ("prospective_T_raw", "generalized_T_raw",
                "generalized_M_raw"):
        assert no_bc_decay_group(key) == "ssm"


def test_prospective_T_zero_recovers_native_coefficients():
    lam = jnp.asarray([-0.3 + 0.1j, -0.6 - 0.2j])
    b = jnp.asarray([[0.2 + 0.1j], [0.1 - 0.3j]])
    step = jnp.asarray([0.4, 0.7])
    a_bar, b_bar = discretize_zoh(lam, b, step)
    got_a, got_b, got_d = _clocked_coefficients(
        lam, b, step, jnp.zeros(2))
    np.testing.assert_allclose(got_a, a_bar, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got_b, b_bar, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got_d, 0.0, atol=1e-7)


def test_generalized_small_mass_recovers_prospective_dynamics():
    lam = jnp.asarray([-0.3 + 0.1j])
    b = jnp.asarray([[0.2 + 0.1j]])
    step = jnp.asarray([0.4])
    t = jnp.asarray([0.7])
    a_bar, b_bar, d_x = _clocked_coefficients(
        lam, b, step, t)
    A_bar, B_bar = generalized_zoh_coefficients(
        lam, b, step, t, jnp.asarray([1e-3]))
    x = jnp.asarray([0.5])
    state = jnp.zeros((1, 2), dtype=lam.dtype)
    for value in (0.5, -0.2, 0.7):
        state = (A_bar @ state[..., None])[..., 0] + B_bar @ jnp.asarray([value])
    prospective = jnp.zeros(1, dtype=lam.dtype)
    for value in (0.5, -0.2, 0.7):
        prospective = a_bar * prospective + b_bar[:, 0] * value
    np.testing.assert_allclose(state[:, 0], prospective + d_x[:, 0] * 0.7,
                               rtol=3e-2, atol=3e-3)


def test_runner_has_no_historical_gp_production_imports():
    source = inspect.getsource(runner)
    assert "s5.gp_ssm" not in source
    assert "s5.gp_coefficients" not in source
    assert "s5.gp_second_order" not in source
    assert set(runner.ARM_CONFIGS) == set(runner.ARM_ORDER)
    assert len({config["shared"] for config in runner.ARM_CONFIGS.values()}) == 1


def test_one_actual_training_step_checkpoint_restore_and_finalizer_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "SEQ_LEN", 8)
    x = jnp.ones((2, 8, 1), dtype=jnp.float32)
    y = jnp.asarray([0, 1])
    for arm in runner.ARM_ORDER:
        state = runner.init_state(arm, 301)
        model = runner.model_for(arm, True)
        state, loss = runner.train_step(
            state, jax.random.PRNGKey(77), x, y, jnp.ones((2, 8)), model, True)
        assert np.isfinite(float(loss))
        checkpoint = tmp_path / f"{arm}.msgpack"
        from flax import serialization
        checkpoint.write_bytes(serialization.to_bytes(state))
        row = {"code_identifier": arm, "seed": 301,
               "selected_checkpoint": str(checkpoint)}
        result = restore_and_evaluate(row, (x, y))
        assert result["n"] == 2
        assert 0.0 <= result["accuracy"] <= 1.0
