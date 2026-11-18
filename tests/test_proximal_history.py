"""Proximal history checks; run x64=0 and x64=1 in separate processes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from s5.gradient_history import (NormalizedGradientHistoryS5SSM,
    normalized_history_zoh, scan_history_blocks)
from s5.proximal_history import ProximalGradientHistoryS5SSM, proximal_history_zoh
from s5.ssm import S5SSM
from s5.ssm_init import make_DPLR_HiPPO

F64 = jax.config.read("jax_enable_x64")
DTYPE = jnp.complex128 if F64 else jnp.complex64
TOL = 2e-10 if F64 else 8e-5


def _physical(kappa, alpha, ratio):
    tau = alpha/abs(kappa)
    denominator = 1+ratio*tau*kappa
    effective = kappa/denominator
    A = np.array([[0, 1, 0], [-2*effective/tau, -1/tau, 1/tau],
                  [effective/tau, 0, -1/tau]], complex)
    b = np.array([0, 2/tau, -1/tau], complex)/denominator
    scaling = np.diag([1, tau, tau])
    aug = np.zeros((4, 4), complex)
    aug[:3, :3], aug[:3, 3] = A, b
    discrete = expm(aug)
    return (scaling@A@np.linalg.inv(scaling), scaling@b,
            scaling@discrete[:3, :3]@np.linalg.inv(scaling), scaling@discrete[:3, 3])


def _kwargs(bidirectional=True):
    lam, _, _, V, _ = make_DPLR_HiPPO(8)
    lam, V = lam[:4], V[:, :4]
    return dict(H=3, P=4, Lambda_re_init=lam.real, Lambda_im_init=lam.imag,
                V=V, Vinv=V.conj().T, C_init="lecun_normal", discretization="zoh",
                dt_min=.001, dt_max=.1, conj_sym=True, clip_eigs=True,
                bidirectional=bidirectional)


@pytest.mark.parametrize("alpha", [.1, 1.])
def test_physical_zoh_and_input_scale(alpha):
    k = jnp.asarray([.001+.9j, .03-.2j, .4+3j], DTYPE)
    Ad, bd, horizon = proximal_history_zoh(k, alpha)
    assert Ad.dtype == bd.dtype == DTYPE
    np.testing.assert_allclose(horizon, alpha/np.abs(k), rtol=TOL, atol=TOL)
    reference = [_physical(ki, alpha, .5) for ki in np.asarray(k)]
    np.testing.assert_allclose(Ad, np.stack([r[2] for r in reference]), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(bd, np.stack([r[3] for r in reference]), rtol=TOL, atol=TOL)


def test_zero_ratio_exactly_existing_history():
    k = jnp.asarray([.01+.2j, .1-.3j], DTYPE)
    actual = proximal_history_zoh(k, .1, 0.)
    expected = normalized_history_zoh(k, .1)
    for a, b in zip(actual, expected):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("alpha", [.1, 1.])
def test_continuous_transfer_native_dc_and_added_delay(alpha):
    for k in (.01+.2j, .1-.4j):
        A, b, _, _ = _physical(k, alpha, .5)
        tau = alpha/abs(k)
        mu = .5*tau
        def transfer(p):
            return (1+2*tau*p)/((1+mu*k)*p*(tau*p+1)**2+k*(1+2*tau*p))
        for p in (0., .2j, 1.1j):
            actual = np.linalg.solve(p*np.eye(3)-A, b)[0]
            np.testing.assert_allclose(actual, transfer(p), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(transfer(0), 1/k, rtol=1e-12)
        eps = 1e-7*abs(k)
        delay = -(transfer(eps)-transfer(-eps))/(2*eps*transfer(0))
        np.testing.assert_allclose(delay, 1/k+mu, rtol=2e-7, atol=2e-7)


def test_positive_real_stability_and_sharp_half_threshold():
    # Broad dimensionless region, including modes very close to pure rotation.
    for ratio in (.5, .7, 2.):
        for magnitude in np.geomspace(1e-4, 1e3, 31):
            for phase in np.linspace(-np.pi/2+1e-6, np.pi/2-1e-6, 23):
                q = magnitude*np.exp(1j*phase)
                d = 1+ratio*q
                roots = np.roots([d, 2*d, d+2*q, q])
                assert np.max(roots.real) < 0, (ratio, q, roots)
    w = np.geomspace(1e-6, 1e6, 1001)
    F = .5+(2j*w+1)/(1j*w*(1j*w+1)**2)
    np.testing.assert_allclose(F.real, .5-2*w*w/(1+w*w)**2, atol=2e-15)
    assert np.min(F.real) >= -2e-15
    # For every nu<1/2, q=-1/F_nu(i) lies in Re(q)>0 and has root z=i.
    for ratio in (0., .2, .49):
        q = -1/((ratio-.5)-1j)
        assert q.real > 0
        d = 1+ratio*q
        assert abs(np.polyval([d, 2*d, d+2*q, q], 1j)) < 2e-15


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("alpha", [.1, 1.])
def test_scans_and_parameter_gradients(reverse, alpha):
    k = jnp.asarray([.02+.3j, .07-.8j], DTYPE)
    rng = np.random.default_rng(71)
    drive = jnp.asarray(rng.normal(size=(65, 2))+1j*rng.normal(size=(65, 2)), DTYPE)
    def output(re, im, method):
        A, b, _ = proximal_history_zoh(re+1j*im, alpha)
        return scan_history_blocks(A, b, drive, reverse=reverse, implementation=method)
    values, gradients = [], []
    for method in ("parallel", "sequential", "associative"):
        values.append(output(k.real, k.imag, method))
        gradients.append(jax.grad(lambda re, im: jnp.mean(abs(output(re, im, method)[..., 0])**2),
                                  argnums=(0, 1))(k.real, k.imag))
    for value in values[1:]:
        np.testing.assert_allclose(values[0], value, rtol=TOL, atol=TOL)
    for grad in gradients:
        assert all(np.isfinite(g).all() for g in grad)
        for g, expected in zip(grad, gradients[0]):
            np.testing.assert_allclose(g, expected, rtol=10*TOL, atol=10*TOL)
    eps = 1e-5 if F64 else 2e-4
    fun = lambda re: jnp.mean(abs(output(re, k.imag, "parallel")[..., 0])**2)
    numeric = []
    for i in range(2):
        bump = jnp.zeros_like(k.real).at[i].set(eps)
        numeric.append(float((fun(k.real+bump)-fun(k.real-bump))/(2*eps)))
    np.testing.assert_allclose(gradients[0][0], numeric,
                               rtol=5e-5 if F64 else .02, atol=1e-4 if F64 else .02)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_backbone_native_control_zero_ratio_and_full_layer_gradients(bidirectional):
    kw = _kwargs(bidirectional)
    x = jnp.asarray(np.random.default_rng(6).normal(size=(23, 3)), jnp.float32)
    key = jax.random.PRNGKey(301)
    native = S5SSM(**kw)
    control = ProximalGradientHistoryS5SSM(**kw, history_alpha=0)
    history = ProximalGradientHistoryS5SSM(**kw, history_alpha=1.)
    old_control = ProximalGradientHistoryS5SSM(**kw, history_alpha=.1, proximal_ratio=0)
    old = NormalizedGradientHistoryS5SSM(**kw, history_alpha=.1)
    sequential = ProximalGradientHistoryS5SSM(**kw, history_alpha=1., implementation="sequential")
    variables = native.init(key, x)
    for model in (control, history, old_control):
        other = model.init(key, x)
        assert jax.tree_util.tree_structure(variables) == jax.tree_util.tree_structure(other)
        for a, b in zip(jax.tree_util.tree_leaves(variables), jax.tree_util.tree_leaves(other)):
            assert np.array_equal(a, b)
    assert np.array_equal(native.apply(variables, x), control.apply(variables, x))
    assert np.array_equal(old.apply(variables, x), old_control.apply(variables, x))
    np.testing.assert_allclose(history.apply(variables, x), sequential.apply(variables, x), rtol=TOL, atol=TOL)
    gradients = [jax.grad(lambda p: jnp.mean(model.apply({"params": p}, x)**2))(variables["params"])
                 for model in (history, sequential)]
    for a, b in zip(*[jax.tree_util.tree_leaves(g) for g in gradients]):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=10*TOL, atol=10*TOL)
    bound = history.bind(variables)
    drive = jax.vmap(lambda u: bound.history_B@u)(x[:1])
    forward = scan_history_blocks(bound.history_Ad, bound.history_bd, drive)
    backward = scan_history_blocks(bound.history_Ad, bound.history_bd, drive, reverse=True)
    assert np.array_equal(forward, backward)


def test_invalid_static_options_refused():
    x = jnp.zeros((2, 3))
    for alpha in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ProximalGradientHistoryS5SSM(**_kwargs(), history_alpha=alpha).init(jax.random.PRNGKey(0), x)
    for ratio in (-.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ProximalGradientHistoryS5SSM(**_kwargs(), proximal_ratio=ratio).init(jax.random.PRNGKey(0), x)
    with pytest.raises(ValueError):
        ProximalGradientHistoryS5SSM(**dict(_kwargs(), discretization="bilinear")).init(jax.random.PRNGKey(0), x)


def test_long_float32_scan_against_physical_float64_reference():
    k = jnp.asarray([.0005+.8j, .01-.2j, .2+1.3j], jnp.complex64)
    Ad, bd, _ = proximal_history_zoh(k, 1.)
    assert Ad.dtype == bd.dtype == jnp.complex64
    rng = np.random.default_rng(42)
    drive = (rng.normal(size=(16000, 3))+1j*rng.normal(size=(16000, 3))).astype(np.complex64)
    actual = np.asarray(jax.jit(scan_history_blocks)(Ad, bd, jnp.asarray(drive)))[..., 0]
    refs = [_physical(complex(ki), 1., .5) for ki in np.asarray(k)]
    A, b = np.stack([r[2] for r in refs]), np.stack([r[3] for r in refs])
    state = np.zeros_like(b)
    expected = np.empty_like(actual, dtype=np.complex128)
    for t, u in enumerate(drive):
        state = np.einsum("pij,pj->pi", A, state)+b*u[:, None]
        expected[t] = state[:, 0]
    assert np.isfinite(actual).all()
    error = np.linalg.norm(actual-expected, axis=0)/np.linalg.norm(expected, axis=0)
    assert error.max() < .002, error
