"""Explicit history adjoint: independent reverse-mode correctness checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from s5.gradient_history import (NormalizedGradientHistoryS5SSM,
    history_zoh, normalized_history_zoh, scan_history_blocks)
from s5.ssm_init import make_DPLR_HiPPO

F64 = jax.config.read("jax_enable_x64")
DTYPE = jnp.complex128 if F64 else jnp.complex64
TOL = 2e-10 if F64 else 8e-5


def blocks():
    k = jnp.asarray([.001+.02j, .03+.2j, .07+.8j], DTYPE)
    return normalized_history_zoh(k)[:2]


@pytest.mark.parametrize("length", [1, 2, 31, 32, 65])
@pytest.mark.parametrize("reverse", [False, True])
def test_full_complex_pullback_matches_sequential_and_old_tape(length, reverse):
    A, b = blocks()
    rng = np.random.default_rng(21)
    u = jnp.asarray(rng.normal(size=(length, 3)) + 1j*rng.normal(size=(length, 3)), DTYPE)
    g = jnp.asarray(rng.normal(size=(length, 3, 3)) + 1j*rng.normal(size=(length, 3, 3)), DTYPE)
    def pullback(method):
        out, back = jax.vjp(lambda aa, bb, uu: scan_history_blocks(
            aa, bb, uu, reverse=reverse, implementation=method), A, b, u)
        return out, back(g)
    value, grads = pullback("parallel")
    for method in ("sequential", "parallel_reference"):
        expected, expected_grads = pullback(method)
        if method == "parallel_reference":
            np.testing.assert_array_equal(value, expected)
        else:
            np.testing.assert_allclose(value, expected, rtol=TOL, atol=TOL)
        for got, ref in zip(grads, expected_grads):
            assert np.isfinite(got).all()
            np.testing.assert_allclose(got, ref, rtol=5*TOL, atol=5*TOL)


def test_vmap_accumulates_shared_transition_and_input_gradients():
    A, b = blocks()
    rng = np.random.default_rng(9)
    u = jnp.asarray(rng.normal(size=(4, 61, 3)) + 1j*rng.normal(size=(4, 61, 3)), DTYPE)
    def loss(aa, bb, uu, method):
        x = jax.vmap(lambda one: scan_history_blocks(aa, bb, one, implementation=method))(uu)
        return jnp.mean(abs(x)**2) + jnp.mean(jnp.sin(x.real))
    got = jax.grad(lambda aa, bb, uu: loss(aa, bb, uu, "parallel"), argnums=(0, 1, 2))(A, b, u)
    ref = jax.grad(lambda aa, bb, uu: loss(aa, bb, uu, "sequential"), argnums=(0, 1, 2))(A, b, u)
    for left, right in zip(got, ref):
        np.testing.assert_allclose(left, right, rtol=5*TOL, atol=5*TOL)


def test_real_drive_cotangent_and_repeated_pole():
    # The pure realization has a repeated pole at kappa=0,tau=ell=1.
    A, b = history_zoh(jnp.zeros(1, DTYPE), jnp.ones(1), jnp.ones(1))
    u = jnp.arange(33, dtype=A.real.dtype)[:, None] / 33
    def loss(aa, bb, uu, method):
        return jnp.mean(abs(scan_history_blocks(aa, bb, uu, implementation=method))**2)
    got = jax.grad(lambda aa, bb, uu: loss(aa, bb, uu, "parallel"), argnums=(0, 1, 2))(A, b, u)
    ref = jax.grad(lambda aa, bb, uu: loss(aa, bb, uu, "sequential"), argnums=(0, 1, 2))(A, b, u)
    assert got[2].dtype == u.dtype
    for left, right in zip(got, ref):
        assert np.isfinite(left).all()
        np.testing.assert_allclose(left, right, rtol=10*TOL, atol=10*TOL)


def test_bidirectional_parameter_gradient_tree_matches_old_tape():
    Lambda, _, _, V, _ = make_DPLR_HiPPO(8)
    Lambda, V = Lambda[:4], V[:, :4]
    kw = dict(H=3, P=4, Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
              V=V, Vinv=V.conj().T, C_init="lecun_normal", discretization="zoh",
              dt_min=.001, dt_max=.1, conj_sym=True, clip_eigs=True,
              bidirectional=True)
    fast = NormalizedGradientHistoryS5SSM(**kw)
    ref = NormalizedGradientHistoryS5SSM(**kw, implementation="parallel_reference")
    u = jnp.asarray(np.random.default_rng(29).normal(size=(49, 3)), jnp.float32)
    variables = fast.init(jax.random.PRNGKey(301), u)
    np.testing.assert_array_equal(fast.apply(variables, u), ref.apply(variables, u))
    def value_grad(model):
        return jax.value_and_grad(lambda p: jnp.mean(model.apply({"params": p}, u)**2))(variables["params"])
    actual, expected = value_grad(fast), value_grad(ref)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=20*TOL, atol=20*TOL)


def test_long_near_unit_mode_adjoint_matches_sequential():
    # Uses arbitrary, nonzero output cotangents at 16000 steps, so the reverse
    # recurrence and shared-transition reduction are both exercised.
    A, b = normalized_history_zoh(jnp.asarray([.0005+.002j], DTYPE))[:2]
    rng = np.random.default_rng(18)
    u = jnp.asarray(rng.normal(size=(16000, 1)), DTYPE)
    def loss(aa, bb, method):
        x = scan_history_blocks(aa, bb, u, implementation=method)
        return jnp.mean(abs(x[..., 0])**2)
    results = [jax.jit(jax.value_and_grad(lambda aa, bb: loss(aa, bb, m), argnums=(0, 1)))(A, b)
               for m in ("parallel", "sequential")]
    for got, ref in zip(jax.tree_util.tree_leaves(results[0]), jax.tree_util.tree_leaves(results[1])):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, ref, rtol=2e-8 if F64 else .002, atol=.002)
