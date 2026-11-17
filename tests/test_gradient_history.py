"""Numerical checks of the separate gradient-history constructor.

Run separately under JAX_ENABLE_X64=0 and =1. The float32 run tests actual
complex64 expm/gradients; it does not construct matrices in float64 first.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from s5.gradient_history import (NormalizedGradientHistoryS5SSM,
    history_generator, history_zoh, normalized_history_zoh, scan_history_blocks)
from s5.ssm import S5SSM
from s5.ssm_init import make_DPLR_HiPPO

F64 = jax.config.read("jax_enable_x64")
DTYPE = jnp.complex128 if F64 else jnp.complex64
TOL = 2e-10 if F64 else 5e-5


def _physical_reference(kappa, tau, ell):
    A = np.array([[0, 1, 0], [-(1+tau/ell)*kappa/tau, -1/tau, 1/ell],
                  [kappa/ell, 0, -1/ell]], complex)
    b = np.array([0, (1+tau/ell)/tau, -1/ell], complex)
    aug = np.zeros((4, 4), complex)
    aug[:3, :3], aug[:3, 3] = A, b
    E = expm(aug)
    S = np.diag([1, tau, tau])
    return S @ E[:3, :3] @ np.linalg.inv(S), S @ E[:3, 3]


def _modes():
    # Initialized S5 covers these four phase classes; vary magnitudes over
    # the measured initialized range, including slow/highly rotating modes.
    phase = np.array([.707376, 1.320752, 1.477682, 1.545622])
    k = (np.geomspace(.000688, 1.973, 12)[:, None] * np.exp(1j*phase)).ravel()
    return jnp.asarray(k, DTYPE)


def _kwargs(bidirectional=True, conj_sym=True):
    block = 8
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block)
    if conj_sym:
        Lambda, V = Lambda[:block//2], V[:, :block//2]
    return dict(H=3, P=len(Lambda), Lambda_re_init=Lambda.real,
                Lambda_im_init=Lambda.imag, V=V, Vinv=V.conj().T,
                C_init="lecun_normal", discretization="zoh", dt_min=.001,
                dt_max=.1, conj_sym=conj_sym, clip_eigs=True,
                bidirectional=bidirectional)


def test_exact_zoh_against_independent_physical_realization():
    k = _modes()
    A, b, horizon = normalized_history_zoh(k)
    assert A.dtype == DTYPE and b.dtype == DTYPE
    refs = [_physical_reference(ki, hi, hi) for ki, hi in zip(np.array(k), np.array(horizon))]
    np.testing.assert_allclose(A, np.stack([r[0] for r in refs]), rtol=TOL, atol=TOL/10)
    np.testing.assert_allclose(b, np.stack([r[1] for r in refs]), rtol=TOL, atol=TOL/10)


def test_continuous_transfer_and_native_dc_gain():
    k = np.asarray(_modes())
    tau, ell = .1/abs(k), .05/abs(k)
    A, b = map(np.asarray, history_generator(jnp.asarray(k), jnp.asarray(tau), jnp.asarray(ell)))
    for omega in (0., .1, 2.):
        p = 1j*omega
        got = np.linalg.solve(p*np.eye(3)[None]-A, b[..., None])[:, 0, 0]
        L, d = tau+ell, tau*ell
        want = (1+L*p)/(d*p**3+L*p**2+(1+k*L)*p+k)
        np.testing.assert_allclose(got, want, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("reverse", [False, True])
def test_parallel_sequential_values_and_parameter_gradients(reverse):
    k = jnp.array([.01+.02j, .03+.2j, .07+.8j], DTYPE)
    rng = np.random.default_rng(73)
    drive = jnp.asarray(rng.normal(size=(131, 3)) + 1j*rng.normal(size=(131, 3)), DTYPE)
    def output(reals, method):
        kk = reals + 1j*k.imag
        A, b, _ = normalized_history_zoh(kk)
        return scan_history_blocks(A, b, drive, reverse=reverse, implementation=method)
    a, b = output(k.real, "parallel"), output(k.real, "sequential")
    np.testing.assert_allclose(a, b, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(output(k.real, "associative"), b, rtol=TOL, atol=TOL)
    losses = [lambda re, m=m: jnp.mean(abs(output(re, m)[..., 0])**2)
              for m in ("parallel", "sequential", "associative")]
    ga, gb, gc = [jax.grad(f)(k.real) for f in losses]
    assert np.isfinite(ga).all() and np.isfinite(gb).all() and np.isfinite(gc).all()
    np.testing.assert_allclose(ga, gb, rtol=5*TOL, atol=5*TOL)
    np.testing.assert_allclose(ga, gc, rtol=5*TOL, atol=5*TOL)
    # An external central difference tests expm AD, not only scan agreement.
    eps = 1e-5 if F64 else 2e-4
    numeric = []
    for i in range(len(k)):
        bump = jnp.zeros_like(k.real).at[i].set(eps)
        numeric.append(float((losses[0](k.real+bump)-losses[0](k.real-bump))/(2*eps)))
    np.testing.assert_allclose(ga, numeric, rtol=3e-5 if F64 else .01, atol=.01 if not F64 else 1e-4)


@pytest.mark.parametrize("length", [1, 2, 3, 16, 18])
def test_stationary_scan_even_odd_and_single_token(length):
    k = jnp.asarray([.1+.2j, .2+.4j], DTYPE)
    A, b, _ = normalized_history_zoh(k)
    drive = jnp.asarray(np.random.default_rng(2).normal(size=(length, 2)), DTYPE)
    actual = scan_history_blocks(A, b, drive)
    expected = scan_history_blocks(A, b, drive, implementation="sequential")
    np.testing.assert_allclose(actual, expected, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("kappa", [0., 1e-7])
def test_repeated_or_near_repeated_poles_have_finite_gradients(kappa):
    # kappa=0,tau=ell=1 has a repeated continuous pole -1 (and pole 0).
    drive = jnp.ones((33, 1), DTYPE)
    def loss(k, implementation):
        A, b = history_zoh(jnp.asarray([k], DTYPE), jnp.ones(1), jnp.ones(1))
        states = scan_history_blocks(A, b, drive, implementation=implementation)
        return jnp.mean(abs(states[..., 0])**2)
    vals = [jax.value_and_grad(lambda k: loss(k, m))(jnp.asarray(kappa))
            for m in ("parallel", "sequential")]
    assert np.isfinite(np.array(vals)).all()
    np.testing.assert_allclose(vals[0], vals[1], rtol=TOL, atol=10*TOL)


def test_positive_ell_limit_approaches_native():
    k = jnp.asarray([.1+.2j, .2+.4j], DTYPE)
    drive = jnp.asarray(np.random.default_rng(5).normal(size=(60, 2)), DTYPE)
    native_A, native_b = jnp.exp(-k), -jnp.expm1(-k)/k
    def step(s, u):
        s = native_A*s + native_b*u
        return s, s
    _, reference = jax.lax.scan(step, jnp.zeros_like(k), drive)
    errors = []
    for ell in (.1, .01):
        A, b = history_zoh(k, jnp.ones(2)*.5, jnp.ones(2)*ell)
        values = scan_history_blocks(A, b, drive)[..., 0]
        errors.append(float(jnp.linalg.norm(values-reference)))
    assert errors[1] < .2*errors[0]


@pytest.mark.parametrize("bidirectional,conj_sym", [(False, True), (True, True), (False, False)])
def test_backbone_preservation_native_control_and_direction_reset(bidirectional, conj_sym):
    kw = _kwargs(bidirectional, conj_sym)
    x = jnp.asarray(np.random.default_rng(5).normal(size=(29, kw["H"])), dtype=jnp.float32)
    key = jax.random.PRNGKey(301)
    native = S5SSM(**kw)
    control = NormalizedGradientHistoryS5SSM(**kw, history_alpha=0)
    history = NormalizedGradientHistoryS5SSM(**kw)
    variables = native.init(key, x)
    for model in (control, history):
        params = model.init(key, x)
        assert jax.tree_util.tree_structure(variables) == jax.tree_util.tree_structure(params)
        for u, v in zip(jax.tree_util.tree_leaves(variables), jax.tree_util.tree_leaves(params)):
            assert np.array_equal(u, v)
    assert np.array_equal(native.apply(variables, x), control.apply(variables, x))
    sequential = NormalizedGradientHistoryS5SSM(**kw, implementation="sequential")
    np.testing.assert_allclose(history.apply(variables, x), sequential.apply(variables, x), rtol=TOL, atol=TOL)
    gradients = jax.grad(lambda p: jnp.mean(history.apply({"params": p}, x)**2))(variables["params"])
    reference_gradients = jax.grad(lambda p: jnp.mean(sequential.apply({"params": p}, x)**2))(variables["params"])
    assert all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(gradients))
    for actual, expected in zip(jax.tree_util.tree_leaves(gradients),
                                jax.tree_util.tree_leaves(reference_gradients)):
        np.testing.assert_allclose(actual, expected, rtol=5*TOL, atol=5*TOL)
    # A single token has the same forward/backward zero-history state.
    bound = history.bind(variables)
    drive = jax.vmap(lambda u: bound.history_B@u)(x[:1])
    a = scan_history_blocks(bound.history_Ad, bound.history_bd, drive)
    b = scan_history_blocks(bound.history_Ad, bound.history_bd, drive, reverse=True)
    assert np.array_equal(a, b)


def test_invalid_static_options_refused():
    x = jnp.zeros((2, 3))
    kw = _kwargs()
    for alpha in (-.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            NormalizedGradientHistoryS5SSM(**kw, history_alpha=alpha).init(jax.random.PRNGKey(0), x)
    with pytest.raises(ValueError):
        NormalizedGradientHistoryS5SSM(**dict(kw, discretization="bilinear")).init(jax.random.PRNGKey(0), x)


def test_16000_token_float32_constructor_and_scan_accuracy():
    # Always uses genuine complex64 construction, even in the x64 test job.
    k = _modes().astype(jnp.complex64)
    A, b, h = normalized_history_zoh(k)
    rng = np.random.default_rng(41)
    drive = (rng.normal(size=(16000, len(k))) + 1j*rng.normal(size=(16000, len(k)))).astype(np.complex64)
    actual = np.asarray(jax.jit(scan_history_blocks)(A, b, jnp.asarray(drive)))[..., 0]
    refs = [_physical_reference(ki, hi, hi) for ki, hi in zip(np.array(k), np.array(h))]
    Ar, br = np.stack([r[0] for r in refs]), np.stack([r[1] for r in refs])
    state = np.zeros_like(br)
    expected = np.empty_like(actual, dtype=np.complex128)
    for i, u in enumerate(drive):
        state = np.einsum("pij,pj->pi", Ar, state) + br*u[:, None]
        expected[i] = state[:, 0]
    assert np.isfinite(actual).all()
    error = np.linalg.norm(actual-expected, axis=0)/np.linalg.norm(expected, axis=0)
    assert error.max() < 8e-4, error
