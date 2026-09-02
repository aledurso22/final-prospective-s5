"""G2a - authoritative deterministic alpha=0 identity gate.

Freezes the requirement

    F_PC-S5,alpha=0(u; theta) == F_S5(u; theta)

for fixed, non-trainable alpha=0 under identical parameters, input, masks, RNG
keys, optimizer state and precision.

Every assertion here is EXACT (`assert_array_equal`, atol=0, rtol=0). There is
no tolerance band that counts as a pass. A non-zero discrepancy of any size is
a blocking finding, not a pass -- see docs/GATES.md.

Numbered checks correspond to the G2a specification in docs/GATES.md.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import flatten_dict

from s5.prospective import ProspectiveLead, apply_parallel, step
from s5.seq_model import BatchClassificationModel
from tests.s5_reference import make_ssm_init_fn

# The documented smoke configuration (docs/BASELINE.md, E2-001/E2-002).
SMOKE = dict(d_model=64, ssm_size=64, blocks=2)
SMOKE_PARAM_COUNT = 26058
SEQ_LEN = 784
BSZ = 2
N_LAYERS = 2

# A smaller model for the many-assertion checks, same code paths.
SMALL = dict(d_model=8, ssm_size=8, blocks=2)
SMALL_SEQ = 32


def _model(cfg, training=False, **prospective):
    ssm_init_fn, _ = make_ssm_init_fn(**cfg)
    return BatchClassificationModel(
        ssm=ssm_init_fn, d_output=10, d_model=cfg["d_model"], n_layers=N_LAYERS,
        padded=False, activation="half_glu1", dropout=0.0, training=training,
        mode="pool", prenorm=True, batchnorm=False, **prospective,
    )


OFF = dict(prospective_mode="off")
ZERO = dict(prospective_mode="lead", prospective_alpha=0.0,
            prospective_alpha_learned=False, prospective_layers="all")


def _pair(cfg, seq_len):
    """Build off / alpha=0 models plus identical inputs and identical params."""
    rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
    x = jax.random.normal(jax.random.PRNGKey(42), (BSZ, seq_len, 1))
    ts = jnp.ones((BSZ, seq_len))
    off, zero = _model(cfg, **OFF), _model(cfg, **ZERO)
    return off, zero, off.init(rngs, x, ts), zero.init(rngs, x, ts), x, ts


def _assert_trees_exactly_equal(a, b, what):
    flat_a, flat_b = flatten_dict(a), flatten_dict(b)
    assert sorted(flat_a) == sorted(flat_b), f"{what}: differing key paths"
    for k in flat_a:
        assert flat_a[k].shape == flat_b[k].shape, f"{what}: shape at {k}"
        assert flat_a[k].dtype == flat_b[k].dtype, f"{what}: dtype at {k}"
        np.testing.assert_array_equal(flat_a[k], flat_b[k], err_msg=f"{what} at {k}")


# ---------------------------------------------- (1)(2)(3) parameter tree


def test_1_parameter_tree_structure_identical():
    _, _, v_off, v_zero, _, _ = _pair(SMALL, SMALL_SEQ)
    _assert_trees_exactly_equal(v_off["params"], v_zero["params"], "param tree")


def test_2_smoke_config_parameter_count_is_26058():
    _, _, v_off, v_zero, _, _ = _pair(SMOKE, SEQ_LEN)
    n_off = sum(p.size for p in jax.tree_util.tree_leaves(v_off["params"]))
    n_zero = sum(p.size for p in jax.tree_util.tree_leaves(v_zero["params"]))
    assert n_off == SMOKE_PARAM_COUNT, n_off
    assert n_zero == SMOKE_PARAM_COUNT, n_zero


def test_3_fixed_zero_introduces_no_alpha_parameter():
    _, _, _, v_zero, _, _ = _pair(SMALL, SMALL_SEQ)
    alphas = [p for p in flatten_dict(v_zero["params"]) if "alpha" in p[-1]]
    assert alphas == [], f"fixed alpha=0 allocated parameters: {alphas}"


# ------------------------------------------------ (4)(9) operator level


def test_4_operator_output_identity_exact():
    x = jax.random.normal(jax.random.PRNGKey(3), (16, 5))
    np.testing.assert_array_equal(apply_parallel(x, 0.0), x)


def test_9_parallel_path_identity_exact():
    """Parallel path at alpha=0 is the identity, with and without resets."""
    x = jax.random.normal(jax.random.PRNGKey(4), (16, 5))
    mask = jnp.zeros(16, dtype=bool).at[jnp.array([0, 5, 11])].set(True)
    np.testing.assert_array_equal(apply_parallel(x, 0.0, mask), x)
    np.testing.assert_array_equal(jax.jit(apply_parallel)(x, 0.0, mask), x)


# ------------------------------------------------------ (5) model logits


def test_5_model_logits_identical_exact():
    off, zero, v_off, v_zero, x, ts = _pair(SMALL, SMALL_SEQ)
    np.testing.assert_array_equal(zero.apply(v_zero, x, ts), off.apply(v_off, x, ts))


def test_5_model_logits_identical_under_jit():
    off, zero, v_off, v_zero, x, ts = _pair(SMALL, SMALL_SEQ)
    np.testing.assert_array_equal(
        jax.jit(zero.apply)(v_zero, x, ts), jax.jit(off.apply)(v_off, x, ts)
    )


# -------------------------------------------------- (6) fixed-batch loss


def _loss_fn(model, x, ts, labels):
    def loss(params):
        logits = model.apply({"params": params}, x, ts)
        return -jnp.mean(logits[jnp.arange(labels.shape[0]), labels])
    return loss


def test_6_fixed_batch_loss_identical_exact():
    off, zero, v_off, v_zero, x, ts = _pair(SMALL, SMALL_SEQ)
    labels = jnp.array([3, 7])
    l_off = _loss_fn(off, x, ts, labels)(v_off["params"])
    l_zero = _loss_fn(zero, x, ts, labels)(v_zero["params"])
    np.testing.assert_array_equal(l_zero, l_off)


# ----------------------------------------------------- (7) gradients


def test_7_gradients_identical_for_every_ordinary_parameter():
    off, zero, v_off, v_zero, x, ts = _pair(SMALL, SMALL_SEQ)
    labels = jnp.array([3, 7])
    g_off = jax.grad(_loss_fn(off, x, ts, labels))(v_off["params"])
    g_zero = jax.grad(_loss_fn(zero, x, ts, labels))(v_zero["params"])
    _assert_trees_exactly_equal(g_off, g_zero, "gradients")
    # gradients must be non-trivial, else the test proves nothing
    assert sum(float(jnp.sum(jnp.abs(g))) for g in
               jax.tree_util.tree_leaves(g_off)) > 0.0


# --------------------------------------------- (8) one optimizer update


def test_8_one_optimizer_update_identical_including_state():
    off, zero, v_off, v_zero, x, ts = _pair(SMALL, SMALL_SEQ)
    labels = jnp.array([3, 7])
    tx = optax.adam(1e-3)

    def one_step(model, params):
        opt_state = tx.init(params)
        grads = jax.grad(_loss_fn(model, x, ts, labels))(params)
        updates, new_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state

    p_off, s_off = one_step(off, v_off["params"])
    p_zero, s_zero = one_step(zero, v_zero["params"])

    _assert_trees_exactly_equal(p_off, p_zero, "params after one update")
    for a, b in zip(jax.tree_util.tree_leaves(s_off),
                    jax.tree_util.tree_leaves(s_zero)):
        np.testing.assert_array_equal(a, b, err_msg="optimizer state")


# --------------------------------------------------- (10) streaming path


def test_10_streaming_output_identity_exact():
    """Token-by-token streaming at alpha=0 returns the input unchanged."""
    x = jax.random.normal(jax.random.PRNGKey(5), (24, 6))
    cache, outputs = None, []
    for token in x:
        out, cache = step(token, cache, 0.0, reset=False)
        outputs.append(out)
    np.testing.assert_array_equal(jnp.stack(outputs), x)


def test_10_streaming_cache_is_observationally_irrelevant_at_alpha_zero():
    """Any cache value whatsoever leaves the alpha=0 output unchanged.

    This is the operational meaning of 'the cache must be observationally
    irrelevant at alpha=0': the output does not depend on it.
    """
    token = jnp.array([1.0, -2.0, 3.0])
    for cache in [None, jnp.zeros(3), jnp.full((3,), 1e9), -token, token * 7.0]:
        out, new_cache = step(token, cache, 0.0, reset=False)
        np.testing.assert_array_equal(out, token)
        np.testing.assert_array_equal(new_cache, token)


def test_10_streaming_matches_parallel_at_alpha_zero():
    x = jax.random.normal(jax.random.PRNGKey(6), (24, 6))
    cache, outputs = None, []
    for token in x:
        out, cache = step(token, cache, 0.0, reset=False)
        outputs.append(out)
    np.testing.assert_array_equal(jnp.stack(outputs), apply_parallel(x, 0.0))


# ------------------------------------------- (11) reset / boundary cases


def test_11_t0_boundary_identity():
    x = jax.random.normal(jax.random.PRNGKey(7), (8, 4))
    np.testing.assert_array_equal(apply_parallel(x, 0.0)[0], x[0])
    out, _ = step(x[0], None, 0.0)
    np.testing.assert_array_equal(out, x[0])


def test_11_length_one_sequence_identity():
    x = jax.random.normal(jax.random.PRNGKey(8), (1, 4))
    np.testing.assert_array_equal(apply_parallel(x, 0.0), x)
    np.testing.assert_array_equal(
        apply_parallel(x, 0.0, jnp.array([True])), x
    )


def test_11_reset_restart_identity():
    x = jax.random.normal(jax.random.PRNGKey(9), (10, 4))
    for reset in (True, False):
        out, _ = step(x[3], x[2], 0.0, reset=reset)
        np.testing.assert_array_equal(out, x[3])


def test_11_packed_segment_boundary_identity():
    x = jax.random.normal(jax.random.PRNGKey(10), (12, 4))
    mask = jnp.zeros(12, dtype=bool).at[jnp.array([0, 4, 9])].set(True)
    np.testing.assert_array_equal(apply_parallel(x, 0.0, mask), x)


def test_11_stale_cache_identity():
    """A cache left over from a previous segment cannot perturb alpha=0."""
    stale = jnp.array([1e6, -1e6, 5.0, 0.0])
    token = jnp.array([1.0, 2.0, 3.0, 4.0])
    for reset in (True, False):
        out, _ = step(token, stale, 0.0, reset=reset)
        np.testing.assert_array_equal(out, token)


# ------------------------------- documented hazard: non-finite inputs


def test_alpha_zero_identity_holds_only_for_finite_inputs():
    """Documents a REAL limitation of the current no-bypass implementation.

    At alpha=0 the operator computes x + 0.0 * (x - previous). For finite x
    this is exact. For non-finite x, 0.0 * (+/-inf) = NaN, so:

      - the non-finite position becomes NaN rather than propagating inf; and
      - the FOLLOWING timestep is also corrupted, because previous[t+1] is the
        non-finite value, whereas plain S5 leaves that position clean.

    This test asserts the current behaviour so any change is deliberate. It is
    the concrete evidence for the static-zero bypass that theory prefers; see
    the open question in docs/GATES.md. It is NOT a pass of G2a for
    non-finite inputs.
    """
    x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    contaminated = x.at[2, 1].set(jnp.inf)
    out = np.asarray(apply_parallel(contaminated, 0.0))

    # plain S5 would emit inf here and leave t=3 untouched
    assert np.isnan(out[2, 1]), "expected 0*inf -> NaN at the contaminated step"
    assert np.isnan(out[3, 1]), "expected forward contamination to t+1"
    np.testing.assert_array_equal(out[3, 0], x[3, 0])  # other channels clean

    # finite inputs remain exactly identical
    np.testing.assert_array_equal(apply_parallel(x, 0.0), x)
