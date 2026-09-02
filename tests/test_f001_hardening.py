"""F-001 hardening: static-zero bypass in ProspectiveLead.

At fixed, non-trainable ``alpha == 0`` the module bypasses the correction
arithmetic and the cache entirely, while still constructing and executing the
prospective module and its wiring (so the end-to-end code-path control is
preserved).

The module-level ``apply_parallel``/``step`` functions are deliberately NOT
bypassed: they remain the tested definition of the operator arithmetic, and
``tests/test_g2a_identity.py`` continues to exercise them at alpha=0 for
finite inputs.  This file tests the difference between the two layers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

import s5.prospective as prospective
from s5.prospective import ProspectiveLead
from s5.seq_model import BatchClassificationModel
from tests.s5_reference import make_ssm_init_fn

SEQ, H = 16, 4


def _lead(**kw):
    module = ProspectiveLead(**kw)
    variables = module.init(jax.random.PRNGKey(0), jnp.ones((SEQ, H)))
    return module, variables


# ------------------------------------------------ the guard itself


@pytest.mark.parametrize("kw,expected", [
    (dict(alpha=0.0, learned=False), True),
    (dict(alpha=0, learned=False), True),
    (dict(alpha=0.0, learned=True), False),      # learned: alpha is a parameter
    (dict(alpha=0.25, learned=False), False),
    (dict(alpha=1e-30, learned=False), False),   # tiny but not zero
])
def test_static_zero_guard_predicate(kw, expected):
    assert ProspectiveLead(**kw)._is_static_zero is expected


def test_static_zero_guard_rejects_bool():
    """bool is a subclass of int; False must not masquerade as a fixed zero."""
    assert ProspectiveLead(alpha=False, learned=False)._is_static_zero is False


# ------------------------------------- (1) finite exact identity preserved


def test_1_bypass_preserves_finite_exact_identity():
    module, variables = _lead(alpha=0.0, learned=False)
    x = jax.random.normal(jax.random.PRNGKey(1), (SEQ, H))
    np.testing.assert_array_equal(module.apply(variables, x), x)
    np.testing.assert_array_equal(jax.jit(module.apply)(variables, x), x)


def test_1_bypass_preserves_identity_with_reset_mask():
    module, variables = _lead(alpha=0.0, learned=False)
    x = jax.random.normal(jax.random.PRNGKey(2), (SEQ, H))
    mask = jnp.zeros(SEQ, dtype=bool).at[jnp.array([0, 5, 9])].set(True)
    np.testing.assert_array_equal(module.apply(variables, x, mask), x)


def test_1_full_model_logits_still_identical_to_off():
    """The end-to-end code-path control still holds after the patch."""
    ssm_init_fn, _ = make_ssm_init_fn(d_model=8, ssm_size=8, blocks=2)
    common = dict(ssm=ssm_init_fn, d_output=10, d_model=8, n_layers=2,
                  padded=False, activation="half_glu1", dropout=0.0,
                  training=False, mode="pool", prenorm=True, batchnorm=False)
    rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 32, 1))
    ts = jnp.ones((2, 32))

    off = BatchClassificationModel(**common, prospective_mode="off")
    zero = BatchClassificationModel(
        **common, prospective_mode="lead", prospective_alpha=0.0,
        prospective_alpha_learned=False, prospective_layers="all")
    v_off, v_zero = off.init(rngs, x, ts), zero.init(rngs, x, ts)

    np.testing.assert_array_equal(zero.apply(v_zero, x, ts), off.apply(v_off, x, ts))
    # and still allocates no alpha parameter
    assert [p for p in flatten_dict(v_zero["params"]) if "alpha" in p[-1]] == []


# ------------------------- (2) Inf/NaN no longer contaminates at alpha=0


@pytest.mark.parametrize("bad", [jnp.inf, -jnp.inf, jnp.nan])
def test_2_nonfinite_no_longer_contaminated_by_module(bad):
    """The exact defect in F-001, now fixed at the module level."""
    module, variables = _lead(alpha=0.0, learned=False)
    x = jnp.arange(SEQ * H, dtype=jnp.float32).reshape(SEQ, H).at[2, 1].set(bad)
    out = np.asarray(module.apply(variables, x))
    ref = np.asarray(x)

    # exact identity including the non-finite entry itself
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
    finite = np.isfinite(ref)
    np.testing.assert_array_equal(out[finite], ref[finite])
    np.testing.assert_array_equal(np.isposinf(out), np.isposinf(ref))
    np.testing.assert_array_equal(np.isneginf(out), np.isneginf(ref))

    # the specific F-001 symptom: the FOLLOWING timestep stayed clean
    assert np.isfinite(out[3, 1]), "t+1 contaminated - F-001 has regressed"
    np.testing.assert_array_equal(out[3, 1], ref[3, 1])


def test_2_free_function_still_shows_the_arithmetic_hazard():
    """The un-bypassed arithmetic is unchanged and still under test.

    This is intentional: `apply_parallel` remains the definition of the
    operator. Only ProspectiveLead bypasses it at fixed zero.
    """
    x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3).at[2, 1].set(jnp.inf)
    out = np.asarray(prospective.apply_parallel(x, 0.0))
    assert np.isnan(out[2, 1]) and np.isnan(out[3, 1])


# --------------------------------- (3) stale cache observationally irrelevant


def test_3_step_neither_reads_nor_writes_cache_at_fixed_zero():
    module, variables = _lead(alpha=0.0, learned=False)
    token = jnp.array([1.0, -2.0, 3.0, 4.0])
    for stale in [None, jnp.zeros(H), jnp.full((H,), 1e9), -token,
                  jnp.full((H,), jnp.nan), jnp.full((H,), jnp.inf)]:
        out, new_cache = module.apply(
            variables, token, stale, False, method=ProspectiveLead.step)
        np.testing.assert_array_equal(out, token)
        # cache is returned untouched, never overwritten with the token
        if stale is None:
            assert new_cache is None
        else:
            np.testing.assert_array_equal(new_cache, stale)


def test_3_step_identity_holds_across_reset_flag():
    module, variables = _lead(alpha=0.0, learned=False)
    token = jnp.array([5.0, 6.0, 7.0, 8.0])
    for reset in (True, False):
        out, _ = module.apply(variables, token, jnp.full((H,), 1e9), reset,
                              method=ProspectiveLead.step)
        np.testing.assert_array_equal(out, token)


# ----------------------------------- (4) alpha > 0 behaviour unchanged


@pytest.mark.parametrize("alpha", [0.125, 0.25, 0.5, 1.0, 1e-30])
def test_4_alpha_positive_matches_free_function_exactly(alpha):
    """Non-zero alpha must not take the bypass; results identical to before."""
    module, variables = _lead(alpha=alpha, learned=False)
    x = jax.random.normal(jax.random.PRNGKey(4), (SEQ, H))
    mask = jnp.zeros(SEQ, dtype=bool).at[jnp.array([0, 7])].set(True)
    np.testing.assert_array_equal(
        module.apply(variables, x, mask), prospective.apply_parallel(x, alpha, mask))


@pytest.mark.parametrize("alpha", [0.25, 1.0])
def test_4_alpha_positive_step_still_caches(alpha):
    module, variables = _lead(alpha=alpha, learned=False)
    token = jnp.array([1.0, 2.0, 3.0, 4.0])
    cache = jnp.zeros(H)
    out, new_cache = module.apply(variables, token, cache, False,
                                  method=ProspectiveLead.step)
    expected_out, expected_cache = prospective.step(token, cache, alpha, False)
    np.testing.assert_array_equal(out, expected_out)
    np.testing.assert_array_equal(new_cache, expected_cache)
    # alpha>0 DOES write the cache
    np.testing.assert_array_equal(new_cache, token)


def test_4_learned_alpha_never_bypasses_even_when_initialised_at_zero():
    module = ProspectiveLead(alpha=0.0, learned=True, alpha_max=1.0)
    variables = module.init(jax.random.PRNGKey(0), jnp.ones((SEQ, H)))
    assert [p for p in flatten_dict(variables["params"]) if "alpha" in p[-1]]
    x = jax.random.normal(jax.random.PRNGKey(5), (SEQ, H))
    alpha = module.apply(variables, method=ProspectiveLead.effective_alpha)
    np.testing.assert_array_equal(
        module.apply(variables, x), prospective.apply_parallel(x, alpha))


def test_4_alpha_positive_parallel_still_matches_streaming():
    """The parallel/streaming equivalence for alpha>0 is untouched."""
    module, variables = _lead(alpha=0.5, learned=False)
    x = jax.random.normal(jax.random.PRNGKey(6), (SEQ, H))
    cache, outs = None, []
    for token in x:
        out, cache = module.apply(variables, token, cache, False,
                                  method=ProspectiveLead.step)
        outs.append(out)
    np.testing.assert_array_equal(jnp.stack(outs), module.apply(variables, x))
