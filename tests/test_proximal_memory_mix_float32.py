"""Production-precision checks; run in a separate process with x64 disabled."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_memory_mix import runner as M
from experiments.s5_three_arm_full import runner as R
from s5 import train_helpers as TH
from s5.gradient_history import scan_history_blocks
from s5.proximal_memory_mix import positive_memory_zoh


@pytest.fixture(autouse=True)
def production_precision():
    assert not jax.config.read("jax_enable_x64"), "run this file alone with JAX_ENABLE_X64=0"
    yield
    jax.clear_caches()


def test_complete_bn_dropout_adam_update_matches_reference(monkeypatch):
    for key, value in {"D_MODEL": 8, "SSM_SIZE_BASE": 16, "BLOCKS": 2,
                       "N_LAYERS": 2, "SEQ_LEN": 65, "BATCH_SIZE": 4}.items():
        monkeypatch.setattr(R, key, value)
    rng = np.random.default_rng(20261118)
    inputs = jnp.asarray(rng.normal(size=(4, 65, 1)), jnp.float32)
    labels = jnp.asarray([0, 2, 4, 1], jnp.int32)
    times = jnp.ones((4, 65), jnp.float32)
    key = jax.random.PRNGKey(301005)
    with M.configured() as arm, H.registered_history():
        monkeypatch.setattr(H, "SCAN", "parallel_reference")
        state = R.init_state(arm, 301)
        reference_model = R.model_for(arm, True)
        monkeypatch.setattr(H, "SCAN", "parallel")
        production_model = R.model_for(arm, True)
        assert reference_model.dropout == production_model.dropout == .1
        before = jax.device_get(state)
        reference = TH.train_step_observable(
            state, key, inputs, labels, times, reference_model, True)
        production = TH.train_step_observable(
            state, key, inputs, labels, times, production_model, True)
        reference, production = jax.device_get((reference, production))
    for expected, actual in zip(jax.tree_util.tree_leaves(reference),
                                jax.tree_util.tree_leaves(production)):
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=3e-6)
    assert int(reference[0].step) == int(production[0].step) == int(before.step) + 1
    assert reference[-1] and production[-1]
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(before.batch_stats),
        jax.tree_util.tree_leaves(production[0].batch_stats)))


def test_16000_tokens_64_modes_forward_and_backward_finite():
    # Very slow nearly rotating modes as well as strongly contracting modes.
    real = jnp.geomspace(1e-5, .4, 64, dtype=jnp.float32)
    imag = jnp.geomspace(1e-3, 3., 64, dtype=jnp.float32)
    rng = np.random.default_rng(318)
    drive = jnp.asarray(rng.normal(size=(16000, 64)), jnp.complex64)

    def loss(re, im, values):
        A, b, _ = positive_memory_zoh(re + 1j*im)
        states = scan_history_blocks(A, b, values)
        return jnp.mean(jnp.abs(states[..., 0])**2)

    value, gradients = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(
        real, imag, drive)
    value, gradients = jax.device_get((value, gradients))
    assert np.isfinite(value)
    assert all(np.isfinite(g).all() for g in gradients)
    assert all(g.dtype in (np.float32, np.complex64) for g in gradients)
    assert all(np.linalg.norm(g) > 0 for g in gradients)
