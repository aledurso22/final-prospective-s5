"""Input-lead ablation (sigma) of proximal_inertial_s5. Run in both precisions:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest tests/test_proximal_inertia_lead.py -q
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_proximal_inertia_lead.py -q
"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict
from scipy.linalg import expm

from experiments.s5_three_arm_full import runner as R
from s5 import proximal_inertia as PI

X64 = jax.config.read("jax_enable_x64")
CDT = np.complex128 if X64 else np.complex64
f64 = pytest.mark.skipif(not X64, reason="float64 reference")


def _kappas(n=24, seed=0):
    rng = np.random.default_rng(seed)
    return (np.exp(rng.uniform(-6, 1, n)) + 1j * rng.normal(size=n) * np.exp(rng.uniform(-3, 3, n))).astype(CDT)


def _H(p, k, t, mu, m, sigma):
    return (1 + sigma * t * p) / ((1 + mu * k) * (m * p * p + p) + k * (1 + t * p))


@f64
@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_transfer_identity_dc_gain_and_delay(sigma):
    for k in _kappas(seed=1):
        k = complex(k)
        t = .1 / abs(k); mu, m = .5 * t, 1.5 * t
        A, g = PI.inertia_generator(jnp.asarray([k]), t, mu, m, sigma)
        A, g = np.asarray(A[0]), np.asarray(g[0])
        for p in (0.3j / t, (1 + 2j) / t, 0.01 / t, 0.0):
            Hs = np.linalg.solve(p * np.eye(2) - A, g)[0]
            assert abs(Hs - _H(p, k, t, mu, m, sigma)) <= 1e-9 * abs(_H(p, k, t, mu, m, sigma))
        assert abs(np.linalg.solve(-A, g)[0] - 1 / k) <= 1e-9 / abs(k)       # DC gain native
        eps = 1e-6 / max(abs(k), 1 / t)
        T = -(_H(eps, k, t, mu, m, sigma) - _H(-eps, k, t, mu, m, sigma)) / (2 * eps * _H(0, k, t, mu, m, sigma))
        assert abs(T - (1 / k + mu + (1 - sigma) * t)) <= 1e-5 * abs(1 / k + mu + t)  # sigma=0 adds tau


def test_generator_unchanged_and_only_input_column_moves():
    kappa = jnp.asarray(_kappas(seed=2))
    tau = .1 / jnp.abs(kappa)
    A1, g1 = PI.inertia_generator(kappa, tau, .5 * tau, 1.5 * tau, 1.0)
    A0, g0 = PI.inertia_generator(kappa, tau, .5 * tau, 1.5 * tau, 0.0)
    assert np.array_equal(np.asarray(A1), np.asarray(A0))                     # bitwise
    d = 1 + .5 * tau * kappa
    np.testing.assert_allclose(np.asarray(g0), np.stack([0 * d, 1 / d], -1), rtol=1e-6 if not X64 else 1e-14)
    Ad1, bd1, _ = PI.proximal_inertia_zoh(kappa, input_lead=1.0)
    Ad0, bd0, _ = PI.proximal_inertia_zoh(kappa, input_lead=0.0)
    z1 = np.sort_complex(np.linalg.eigvals(np.asarray(Ad1).astype(np.complex128)).ravel())
    z0 = np.sort_complex(np.linalg.eigvals(np.asarray(Ad0).astype(np.complex128)).ravel())
    np.testing.assert_allclose(z0, z1, rtol=0, atol=1e-12 if X64 else 2e-6)   # same poles
    assert np.max(np.abs(np.asarray(bd0) - np.asarray(bd1))) > 1e-3          # residues change


def _physical(kappa, tau, mu, m, b, jump_sigma):
    """Independent (s, v=s') simulation; m dv = jump_sigma tau db/d at token boundaries."""
    d = 1 + mu * kappa
    k = kappa / d
    E = expm(np.array([[0, 1, 0], [-k / m, -(1 + tau * k) / m, 1 / (m * d)], [0, 0, 0]], complex))
    x, prev, out = np.zeros(2, complex), 0j, []
    for bt in b:
        x = x + np.array([0, jump_sigma * tau * (bt - prev) / (d * m)])
        x = (E @ np.r_[x, bt])[:2]
        prev = bt
        out.append(x[0])
    return np.asarray(out)


@f64
@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_independent_zoh_with_sigma_jump(sigma):
    kappa = _kappas(10, seed=3)
    rng = np.random.default_rng(4)
    b = rng.normal(size=(250, 10)) + 1j * rng.normal(size=(250, 10))
    Ad, bd, tau = PI.proximal_inertia_zoh(jnp.asarray(kappa), input_lead=sigma)
    got = np.asarray(PI.scan_blocks(Ad, bd, jnp.asarray(b), implementation="sequential"))[..., 0]
    tau = np.asarray(tau)
    for p in range(10):
        args = (kappa[p], tau[p], .5 * tau[p], 1.5 * tau[p], b[:, p])
        want = _physical(*args, jump_sigma=sigma)
        np.testing.assert_allclose(got[:, p], want, rtol=1e-9, atol=1e-11 * abs(want).max())
        wrong = _physical(*args, jump_sigma=1 - sigma)   # the other policy's jump
        assert np.abs(got[:, p] - wrong).max() > 1e-4 * abs(want).max()


def test_gradients_sigma0_custom_adjoint_equals_autodiff():
    kw = {**R.ssm_kwargs("native_matched_s5"), "step_rescale": 1.0}
    x = jnp.asarray(np.random.default_rng(5).normal(size=(257, kw["H"])), jnp.float32)
    params = PI.ProximalInertialS5SSM(**kw, input_lead=0.0).init(jax.random.PRNGKey(1), x)
    loss = lambda impl: (lambda p: jnp.sum(PI.ProximalInertialS5SSM(
        **kw, input_lead=0.0, implementation=impl).apply(p, x) ** 2))
    g1, g0 = jax.grad(loss("parallel"))(params), jax.grad(loss("parallel_reference"))(params)
    for (k, a), b in zip(flatten_dict(g1).items(), flatten_dict(g0).values()):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all(), k
        assert np.abs(a - b).max() <= (1e-8 if X64 else 3e-3) * max(np.abs(b).max(), 1e-30), k


@f64
def test_gradient_sigma0_matches_finite_difference():
    kw = {**R.ssm_kwargs("native_matched_s5"), "step_rescale": 1.0}
    x = jnp.asarray(np.random.default_rng(6).normal(size=(64, kw["H"])))
    layer = PI.ProximalInertialS5SSM(**kw, input_lead=0.0)
    params = layer.init(jax.random.PRNGKey(2), x)
    loss = lambda p: jnp.sum(layer.apply(p, x) ** 2)
    g, flat = flatten_dict(jax.grad(loss)(params)), flatten_dict(params)
    for leaf in ("Lambda_re", "Lambda_im", "log_step", "B"):
        key = [k for k in flat if k[-1] == leaf][0]
        idx = (5,) + (0,) * (flat[key].ndim - 1)
        f = lambda e: float(loss(unflatten_dict({**flat, key: flat[key].at[idx].add(e)})))
        fd = (f(1e-6) - f(-1e-6)) / 2e-6
        assert abs(fd - float(g[key][idx])) <= 1e-5 * max(1.0, abs(fd)), leaf


def test_sigma1_is_bitwise_the_trained_baseline_code():
    from experiments.s5_proximal_inertia import lead_identity as LI
    rep = LI.compare(length=2048)
    s1 = rep["sigma1_vs_legacy"]
    assert s1["all_bitwise_identical"], json.dumps(s1)[:2000]
    assert rep["sigma0_vs_sigma1"]["bd"]["max_abs_difference"] > 0


def test_input_lead_refuses_nonstatic_or_nonfinite():
    for bad in (float("nan"), True, "0"):
        with pytest.raises(ValueError):
            PI.proximal_inertia_zoh(jnp.asarray(_kappas(2)), input_lead=bad)
