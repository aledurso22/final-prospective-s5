"""Focused adaptive-P checks, suitable for separate float32/float64 runs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from s5.adaptive_proximal_history import (
    AdaptiveProximalHistoryS5SSM, adaptive_proximal_history_zoh)
from s5.gradient_history import scan_history_blocks
from s5.proximal_history import ProximalGradientHistoryS5SSM, proximal_history_zoh
from s5.ssm_init import make_DPLR_HiPPO


F64 = jax.config.read("jax_enable_x64")
DTYPE = jnp.complex128 if F64 else jnp.complex64
REAL = jnp.float64 if F64 else jnp.float32
TOL = 2e-10 if F64 else 8e-5


def _kwargs(bidirectional=True):
    lam, _, _, V, _ = make_DPLR_HiPPO(8)
    lam, V = lam[:4], V[:, :4]
    return dict(H=3, P=4, Lambda_re_init=lam.real, Lambda_im_init=lam.imag,
                V=V, Vinv=V.conj().T, C_init="lecun_normal", discretization="zoh",
                dt_min=.001, dt_max=.1, conj_sym=True, clip_eigs=True,
                bidirectional=bidirectional)


def _reference(kappa, horizon, ratio=.5):
    # Independent physical coordinates (s, v, q), followed by the similarity
    # transform to the implementation's (s, tau*v, tau*q).
    d = 1+ratio*horizon*kappa
    ke = kappa/d
    A = np.array([[0, 1, 0], [-2*ke/horizon, -1/horizon, 1/horizon],
                  [ke/horizon, 0, -1/horizon]], complex)
    b = np.array([0, 2/horizon, -1/horizon], complex)/d
    aug = np.zeros((4, 4), complex)
    aug[:3, :3], aug[:3, 3] = A, b
    discrete = expm(aug)
    scale = np.diag([1, horizon, horizon])
    return (scale@discrete[:3, :3]@np.linalg.inv(scale), scale@discrete[:3, 3])


@pytest.mark.parametrize("alpha", [.1, 1.])
@pytest.mark.parametrize("ratio", [0., .5])
def test_zero_multiplier_exactly_existing_p_blocks(alpha, ratio):
    k = jnp.asarray([.01+.2j, .2-.8j, .4+.001j], DTYPE)
    expected = proximal_history_zoh(k, alpha, ratio)
    actual = adaptive_proximal_history_zoh(k, jnp.zeros(3, REAL), alpha, ratio)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_zero_multiplier_preserves_native_leaves_output_and_common_gradients(bidirectional):
    x = jnp.asarray(np.random.default_rng(6).normal(size=(19, 3)), jnp.float32)
    old = ProximalGradientHistoryS5SSM(**_kwargs(bidirectional), history_alpha=1.)
    new = AdaptiveProximalHistoryS5SSM(**_kwargs(bidirectional))
    frozen = AdaptiveProximalHistoryS5SSM(**_kwargs(bidirectional), trainable_horizon=False)
    key = jax.random.PRNGKey(301)
    original = old.init(key, x)["params"]
    params = new.init(key, x)["params"]
    frozen_params = frozen.init(key, x)["params"]
    assert set(params) == set(original) | {"log_horizon_scale"}
    assert params["log_horizon_scale"].shape == (4,)
    np.testing.assert_array_equal(params["log_horizon_scale"], 0)
    for name, value in original.items():
        np.testing.assert_array_equal(params[name], value)
    for name in params:
        np.testing.assert_array_equal(params[name], frozen_params[name])
    outputs = [model.apply({"params": p}, x)
               for model, p in ((old, original), (new, params), (frozen, params))]
    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])
    gradients = [jax.grad(lambda p: jnp.mean(model.apply({"params": p}, x)**2))(p)
                 for model, p in ((old, original), (new, params), (frozen, params))]
    for name in original:
        np.testing.assert_allclose(gradients[1][name], gradients[0][name],
                                   rtol=10*TOL, atol=10*TOL)
        np.testing.assert_array_equal(gradients[2][name], gradients[1][name])
    assert np.isfinite(gradients[1]["log_horizon_scale"]).all()
    assert np.linalg.norm(gradients[1]["log_horizon_scale"]) > 1e-8
    np.testing.assert_array_equal(gradients[2]["log_horizon_scale"], 0)


@pytest.mark.parametrize("reverse", [False, True])
def test_horizon_gradient_matches_reference_scan_and_finite_differences(reverse):
    k = jnp.asarray([.02+.3j, .07-.8j], DTYPE)
    xi = jnp.asarray([-.7, .45], REAL)
    rng = np.random.default_rng(71)
    drive = jnp.asarray(rng.normal(size=(33, 2))+1j*rng.normal(size=(33, 2)), DTYPE)

    def loss(log_scale, implementation):
        A, b, _ = adaptive_proximal_history_zoh(k, log_scale)
        states = scan_history_blocks(A, b, drive, reverse=reverse, implementation=implementation)
        return jnp.mean(abs(states[..., 0])**2)

    grads = [jax.grad(lambda p: loss(p, implementation))(xi)
             for implementation in ("parallel", "parallel_reference", "sequential")]
    assert np.isfinite(grads[0]).all()
    for grad in grads[1:]:
        np.testing.assert_allclose(grads[0], grad, rtol=10*TOL, atol=10*TOL)
    eps = 1e-5 if F64 else 2e-3
    numeric = []
    for i in range(len(xi)):
        bump = jnp.zeros_like(xi).at[i].set(eps)
        numeric.append(float((loss(xi+bump, "parallel")-loss(xi-bump, "parallel"))/(2*eps)))
    np.testing.assert_allclose(grads[0], numeric,
                               rtol=5e-5 if F64 else .015, atol=2e-7 if F64 else 2e-4)


def test_free_horizons_match_independent_zoh_and_preserve_equilibrium():
    k = jnp.asarray([.002+.4j, .03-.2j, .4+3j], DTYPE)
    xi = jnp.asarray([-1.2, .4, 1.5], REAL)
    Ad, bd, horizon = adaptive_proximal_history_zoh(k, xi)
    assert Ad.dtype == bd.dtype == DTYPE
    refs = [_reference(complex(ki), float(h)) for ki, h in zip(k, horizon)]
    np.testing.assert_allclose(Ad, np.stack([r[0] for r in refs]), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(bd, np.stack([r[1] for r in refs]), rtol=TOL, atol=TOL)
    for A, b, ki in zip(np.asarray(Ad), np.asarray(bd), np.asarray(k)):
        equilibrium = np.linalg.solve(np.eye(3)-A, b)
        np.testing.assert_allclose(equilibrium[0], 1/ki, rtol=10*TOL, atol=10*TOL)
        np.testing.assert_allclose(equilibrium[1:], 0, atol=10*TOL)


def test_horizon_changes_dimensionless_pole_shape_not_just_clock():
    # In continuous dimensionless coordinates z=tau*p the coefficients
    # depend on chi=tau*kappa. A uniform clock rescaling cannot change z.
    k = .02+.3j
    shapes = []
    for xi in (-.7, 0., .8):
        tau = np.exp(xi)/abs(k)
        chi = tau*k
        d = 1+.5*chi
        roots = np.roots([d, 2*d, d+2*chi, chi])
        shapes.append(np.sort_complex(roots))
    assert np.max(abs(shapes[0]-shapes[1])) > .05
    assert np.max(abs(shapes[1]-shapes[2])) > .05


def test_continuous_stability_covers_independent_horizons():
    # The theorem is dimensionless and independent of how tau was chosen.
    # This screen includes slow/fast and nearly rotational complex modes.
    for magnitude in np.geomspace(1e-3, 1e3, 19):
        for phase in np.linspace(-np.pi/2+1e-5, np.pi/2-1e-5, 17):
            chi = magnitude*np.exp(1j*phase)
            d = 1+.5*chi
            roots = np.roots([d, 2*d, d+2*chi, chi])
            assert max(roots.real) < 0, (chi, roots)


def test_bad_static_configuration_is_rejected():
    k = jnp.asarray([.1+.2j], DTYPE)
    for alpha in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            adaptive_proximal_history_zoh(k, jnp.zeros(1), alpha=alpha)
    for ratio in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            adaptive_proximal_history_zoh(k, jnp.zeros(1), proximal_ratio=ratio)
    with pytest.raises(ValueError):
        adaptive_proximal_history_zoh(k, jnp.asarray([1j], DTYPE))
