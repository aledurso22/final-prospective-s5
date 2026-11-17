"""The memory fix preserves a complete BN/dropout/Adam training update."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from s5 import train_helpers as TH


def test_full_update_matches_reference_with_same_batch_and_rng(monkeypatch):
    assert not jax.config.read("jax_enable_x64"), "run this integration check in production float32"
    for key, value in {"D_MODEL": 8, "SSM_SIZE_BASE": 16, "BLOCKS": 2,
                       "N_LAYERS": 2, "SEQ_LEN": 65, "BATCH_SIZE": 4}.items():
        monkeypatch.setattr(R, key, value)
    rng = np.random.default_rng(20261117)
    inputs = jnp.asarray(rng.normal(size=(4, 65, 1)), jnp.float32)
    labels = jnp.asarray([0, 2, 4, 1], jnp.int32)
    times = jnp.ones((4, 65), jnp.float32)
    key = jax.random.PRNGKey(301005)
    with H.registered_history():
        monkeypatch.setattr(H, "SCAN", "parallel_reference")
        state = R.init_state(H.ARM, 301)
        old_model = R.model_for(H.ARM, True)
        monkeypatch.setattr(H, "SCAN", "parallel")
        new_model = R.model_for(H.ARM, True)
        assert old_model.dropout == new_model.dropout == .1
        before = jax.device_get(state)
        old = TH.train_step_observable(state, key, inputs, labels, times, old_model, True)
        new = TH.train_step_observable(state, key, inputs, labels, times, new_model, True)
        old, new = jax.device_get((old, new))
    # Check full optimizer state and BN state as well as metrics/weights.
    for left, right in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new)):
        assert np.isfinite(left).all() and np.isfinite(right).all()
        np.testing.assert_allclose(left, right, rtol=3e-5, atol=2e-6)
    assert int(old[0].step) == int(new[0].step) == int(before.step) + 1
    assert old[-1] and new[-1]
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(before.batch_stats), jax.tree_util.tree_leaves(new[0].batch_stats)))
