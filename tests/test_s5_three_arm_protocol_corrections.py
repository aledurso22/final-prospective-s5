import json
import os
import sys
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.s5_three_arm_full import data
from experiments.s5_three_arm_full import preflight
from experiments.s5_three_arm_full import runner
from experiments.s5_three_arm_full.finalize import restore_and_evaluate
from s5.generalized_prospective_ssm import (RHO_MIN,
                                            generalized_zoh_coefficients,
                                            response_and_mass)
from s5.prospective_ssm import _clocked_coefficients
from s5.ssm import discretize_zoh
from s5.train_helpers import no_bc_decay_group
from s5.three_arm_recurrences import generalized_prospective_s5_zoh


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
                "generalized_rho_raw"):
        assert no_bc_decay_group(key) == "ssm"
    assert runner.T_INIT == 0.05
    assert runner.ssm_factory("zucchet_prospective_s5")().response_init == runner.T_INIT
    assert runner.ssm_factory("generalized_prospective_s5")().response_init == runner.T_INIT


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


def test_extreme_mass_ratio_raw_values_are_bounded():
    raw = jnp.asarray([-100.0, -10.0, 0.0, 10.0, 100.0])
    T, M = response_and_mass(raw, raw)
    assert bool(jnp.all(T > 0))
    assert bool(jnp.all(M > 0))
    assert bool(jnp.all(M <= T))
    assert bool(jnp.all(jnp.asarray(RHO_MIN) <= M / T))


def test_generalized_spectral_radius_is_stable_for_extreme_ratios():
    modes = jnp.asarray([-0.2 + 0.4j, -0.7 - 0.2j])
    B = jnp.ones((2, 1), dtype=jnp.complex128)
    step = jnp.ones(2)
    for raw in (-4.0, 0.0, 4.0):
        T, M = response_and_mass(jnp.full(2, raw), jnp.full(2, raw))
        A_bar, _ = generalized_zoh_coefficients(modes, B, step, T, M)
        radius = jnp.max(jnp.abs(jnp.linalg.eigvals(A_bar)))
        assert float(radius) < 1.0


def test_four_by_four_zoh_matches_large_reference_and_gradients():
    modes = jnp.asarray([-0.3 + 0.1j, -0.6 - 0.2j])
    B = jnp.asarray([[0.2 + 0.1j], [0.1 - 0.3j]])
    step = jnp.asarray([0.4, 0.7])

    def exact(T, M):
        return generalized_zoh_coefficients(modes, B, step, T, M)

    def reference(T, M):
        a, b = modes * step, step[:, None] * B
        A_bar, B_bar = generalized_prospective_s5_zoh(
            a, b, M, jnp.ones_like(M), T)
        return A_bar, B_bar

    T, M = jnp.asarray([0.05, 0.08]), jnp.asarray([0.025, 0.04])
    got = exact(T, M)
    want = reference(T, M)
    for left, right in zip(got, want):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    for index in (0, 1):
        grad_exact = jax.grad(lambda t, m: jnp.real(exact(t, m)[index]).sum(),
                              argnums=(0, 1))(T, M)
        grad_reference = jax.grad(
            lambda t, m: jnp.real(reference(t, m)[index]).sum(),
            argnums=(0, 1))(T, M)
        for left, right in zip(grad_exact, grad_reference):
            np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)


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
        state, loss = runner.train_one_batch(
            state, jax.random.PRNGKey(77), x, y, model, 0, 10)
        assert np.isfinite(float(loss))
        checkpoint = tmp_path / f"{arm}.msgpack"
        from flax import serialization
        checkpoint.write_bytes(serialization.to_bytes(state))
        row = {"code_identifier": arm, "seed": 301,
               "selected_checkpoint": str(checkpoint)}
        result = restore_and_evaluate(row, (x, y))
        assert result["n"] == 2
        assert 0.0 <= result["accuracy"] <= 1.0


def test_actual_train_step_sees_scheduled_optimizer_rates(monkeypatch):
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "SEQ_LEN", 8)
    state = runner.init_state("native_matched_s5", 301)
    model = runner.model_for("native_matched_s5", True)
    x = jnp.ones((2, 8, 1), dtype=jnp.float32)
    y = jnp.asarray([0, 1])
    captured = []

    def capture(current, *args):
        captured.append(float(current.opt_state.inner_states[
            "regular"].inner_state.hyperparams["learning_rate"]))
        return current, jnp.asarray(0.0)

    monkeypatch.setattr(runner, "train_step", capture)
    steps_per_epoch = 10
    for step in (0, steps_per_epoch - 1, steps_per_epoch,
                 steps_per_epoch * runner.EPOCHS - 1):
        runner.train_one_batch(state, jax.random.PRNGKey(step), x, y,
                               model, step, steps_per_epoch)
    expected = [runner.learning_rate_at_step(step, steps_per_epoch)[0]
                for step in (0, 9, 10, 399)]
    np.testing.assert_allclose(captured, expected, rtol=1e-6, atol=1e-8)


def test_telemetry_step_matches_normal_step(monkeypatch):
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    monkeypatch.setattr(runner, "SEQ_LEN", 8)
    state = runner.init_state("native_matched_s5", 301)
    model = runner.model_for("native_matched_s5", True)
    x = jnp.ones((2, 8, 1), dtype=jnp.float32)
    y = jnp.asarray([0, 1])
    normal_state, normal_loss = runner.train_one_batch(
        state, jax.random.PRNGKey(91), x, y, model, 0, 10)
    telemetry_state, telemetry_loss, finite = runner.train_one_batch_telemetry(
        state, jax.random.PRNGKey(91), x, y, model, 0, 10)
    assert bool(finite)
    np.testing.assert_allclose(normal_loss, telemetry_loss, rtol=1e-6, atol=1e-7)
    for left, right in zip(jax.tree_util.tree_leaves(normal_state),
                           jax.tree_util.tree_leaves(telemetry_state)):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)


def test_preflight_pins_three_stage_order_and_result_fields():
    assert preflight.PREFLIGHT_STAGE_ORDER == (
        "telemetry", "normal_compile", "normal_steady_state")
    required = {
        "stage_order",
        "telemetry_compile_seconds",
        "normal_compile_seconds",
        "steady_step_seconds",
        "peak_vram_bytes",
        "finite_gradients",
        "finite_state",
    }
    assert required <= preflight.PREFLIGHT_RESULT_FIELDS
    assert "step_seconds" not in preflight.PREFLIGHT_RESULT_FIELDS
    source = inspect.getsource(preflight.run_arm)
    assert source.index("train_one_batch_telemetry") < source.index(
        "normal compile+execute")
    assert source.index("normal compile+execute") < source.index(
        "normal steady-state step")


def test_full_preflight_uses_separate_children_and_fixed_order(monkeypatch, tmp_path):
    calls = []

    def child(command, check):
        assert check is False
        arm = command[-1]
        calls.append(arm)
        result = {
            "scientific_name": runner.SCIENTIFIC_NAMES[arm],
            "code_identifier": arm,
            "seed": 301,
            "batch_size": 16,
            "sequence_length": 16000,
            "stage_order": list(preflight.PREFLIGHT_STAGE_ORDER),
            "telemetry_compile_seconds": 1.0,
            "normal_compile_seconds": 2.0,
            "steady_step_seconds": 0.1,
            "peak_vram_bytes": 123,
            "finite_gradients": True,
            "finite_state": True,
        }
        with open(tmp_path / f"{arm}.json", "w") as handle:
            json.dump(result, handle)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", child)
    aggregate = preflight._run_all_arms(str(tmp_path))
    assert calls == list(runner.ARM_ORDER)
    assert [row["code_identifier"] for row in aggregate["arms"]] == list(
        runner.ARM_ORDER)
    with open(tmp_path / "preflight.json") as handle:
        assert json.load(handle) == aggregate


def test_full_preflight_propagates_child_failure(monkeypatch, tmp_path):
    def child(command, check):
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(preflight.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="child failed"):
        preflight._run_all_arms(str(tmp_path))
