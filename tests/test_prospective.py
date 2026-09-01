import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

from s5.layers import SequenceLayer
from s5.prospective import apply_parallel, step
from s5.seq_model import StackedEncoderModel


def _stream(sequence, alpha, reset_mask=None):
    outputs = []
    cache = None
    if reset_mask is None:
        reset_mask = np.zeros(sequence.shape[0], dtype=bool)
    for token, reset in zip(sequence, reset_mask):
        output, cache = step(token, cache, alpha, reset=reset)
        outputs.append(output)
    return jnp.stack(outputs)


def test_alpha_zero_is_exact_identity():
    sequence = jnp.asarray([[1.0, -2.0], [3.0, 5.0], [-7.0, 11.0]])
    actual = apply_parallel(sequence, 0.0)
    np.testing.assert_array_equal(actual, sequence)


def test_parallel_matches_streaming_with_resets():
    sequence = jnp.arange(24, dtype=jnp.float32).reshape(8, 3)
    reset_mask = jnp.asarray([True, False, False, True, False, False, True, False])
    alpha = 0.5

    parallel = apply_parallel(sequence, alpha, reset_mask)
    streaming = _stream(sequence, alpha, reset_mask)

    np.testing.assert_allclose(parallel, streaming, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(parallel[reset_mask], sequence[reset_mask])


def test_future_changes_do_not_affect_past_outputs():
    original = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
    changed = original.at[7:].set(-999.0)

    original_output = apply_parallel(original, 0.75)
    changed_output = apply_parallel(changed, 0.75)

    np.testing.assert_array_equal(original_output[:7], changed_output[:7])


def test_jit_and_gradient():
    sequence = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    reset_mask = jnp.asarray([True, False, True, False])

    compiled = jax.jit(apply_parallel)(sequence, 0.25, reset_mask)
    eager = apply_parallel(sequence, 0.25, reset_mask)
    np.testing.assert_allclose(compiled, eager)

    gradient = jax.grad(
        lambda alpha: jnp.sum(apply_parallel(sequence, alpha, reset_mask))
    )(0.25)
    expected = jnp.sum(sequence[1] - sequence[0]) + jnp.sum(sequence[3] - sequence[2])
    np.testing.assert_allclose(gradient, expected)


def test_empty_sequence_is_supported():
    sequence = jnp.empty((0, 3), dtype=jnp.float32)
    assert apply_parallel(sequence, 0.5).shape == (0, 3)


class _FakeSSM(nn.Module):
    step_rescale: float = 1.0

    @nn.compact
    def __call__(self, sequence):
        return 2.0 * sequence


def test_sequence_layer_fixed_alpha_zero_matches_off_mode():
    sequence = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    common = dict(
        ssm=_FakeSSM,
        dropout=0.0,
        d_model=3,
        activation="gelu",
        training=False,
        prenorm=True,
        batchnorm=False,
    )
    baseline = SequenceLayer(**common, prospective_mode="off")
    zero_lead = SequenceLayer(
        **common, prospective_mode="lead", prospective_alpha=0.0
    )
    key = jax.random.PRNGKey(7)
    baseline_variables = baseline.init(key, sequence)
    lead_variables = zero_lead.init(key, sequence)

    baseline_output = baseline.apply(baseline_variables, sequence)
    lead_output = zero_lead.apply(lead_variables, sequence)
    np.testing.assert_array_equal(lead_output, baseline_output)


def test_bidirectional_prospective_layer_is_rejected():
    sequence = jnp.ones((2, 3))
    model = SequenceLayer(
        ssm=_FakeSSM,
        dropout=0.0,
        d_model=3,
        training=False,
        prospective_mode="lead",
        prospective_alpha=0.25,
        bidirectional=True,
    )
    with np.testing.assert_raises_regex(ValueError, "unidirectional"):
        model.init(jax.random.PRNGKey(0), sequence)


def test_last_layer_placement_creates_one_learned_alpha():
    sequence = jnp.ones((4, 3))
    model = StackedEncoderModel(
        ssm=_FakeSSM,
        d_model=3,
        n_layers=3,
        dropout=0.0,
        training=False,
        prospective_mode="lead",
        prospective_alpha=0.25,
        prospective_alpha_learned=True,
        prospective_layers="last",
    )
    variables = model.init(jax.random.PRNGKey(0), sequence, None)
    flat_params = flatten_dict(variables["params"])
    alpha_paths = [path for path in flat_params if path[-1] == "alpha_logit"]
    assert len(alpha_paths) == 1
    assert "layers_2" in alpha_paths[0]
