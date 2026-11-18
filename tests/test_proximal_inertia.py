"""Proximal inertial S5 recurrence. Run separately in both precisions:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest tests/test_proximal_inertia.py -q
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_proximal_inertia.py -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from scipy.linalg import expm

from experiments.s5_three_arm_full import runner as R
from s5 import proximal_inertia as PI
from s5.ssm import S5SSM

X64 = jax.config.read("jax_enable_x64")
RTOL = 1e-10 if X64 else 2e-5
CDT = np.complex128 if X64 else np.complex64


def _kappas(n=96, seed=0):
    rng = np.random.default_rng(seed)
    re = np.exp(rng.uniform(-6, 1, n))
    im = rng.normal(size=n) * np.exp(rng.uniform(-3, 3, n))
    return (re + 1j * im).astype(CDT)


def _first_order(kappa, mu, b):
    """s_t = e^{-k} s_{t-1} + (1-e^{-k})/kappa * b_t, k = kappa/(1+mu kappa)."""
    k = kappa / (1 + mu * kappa)
    a, c = np.exp(-k), (1 - np.exp(-k)) / kappa
    s, out = np.zeros_like(kappa), []
    for bt in b:
        s = a * s + c * bt
        out.append(s)
    return np.asarray(out)


def _physical_reference(kappa, tau, mu, m, b, jump=True):
    """INDEPENDENT construction in physical (s, v=s') coordinates.

    Inside a token the input is constant, so r_mu' = k v and
    m v' = -v - r_mu - tau k v. At each token boundary the total input
    derivative is an impulse: integrating m s'' + tau r_mu' over the jump
    gives m dv = tau db/d, i.e. the velocity jumps; s is continuous.
    Post-update sampling, zero prehistory (b_{-1}=0)."""
    d = 1 + mu * kappa
    k = kappa / d
    A = np.array([[0, 1, 0], [-k / m, -(1 + tau * k) / m, 1 / (m * d)], [0, 0, 0]], complex)
    E = expm(A)
    x, prev, out = np.zeros(2, complex), 0j, []
    for bt in b:
        if jump:
            x = x + np.array([0, tau * (bt - prev) / (d * m)])
        y = E @ np.r_[x, bt]
        x, prev = y[:2], bt
        out.append(x[0])
    return np.asarray(out)


def test_control_decouples_exactly_and_is_first_order_proximal_flow():
    kappa = jnp.asarray(_kappas())
    Ad, bd, tau = PI.proximal_inertia_zoh(kappa, mass_ratio=1.0)
    assert np.all(np.asarray(Ad[:, 1, 0]) == 0) and np.all(np.asarray(bd[:, 1]) == 0)
    rng = np.random.default_rng(1)
    b = (rng.normal(size=(400, 96)) + 1j * rng.normal(size=(400, 96))).astype(CDT)
    states = np.asarray(PI.scan_blocks(Ad, bd, jnp.asarray(b), implementation="parallel"))
    assert np.all(states[..., 1] == 0)          # w stays exactly zero
    want = _first_order(np.asarray(kappa, np.complex128), np.asarray(.5 * tau, np.float64), b)
    np.testing.assert_allclose(states[..., 0], want, rtol=RTOL * 20, atol=RTOL * np.abs(want).max())


def test_native_limit_is_exact_and_alpha_limit_converges():
    kappa = jnp.asarray(_kappas())
    Ad, bd, _ = PI.proximal_inertia_zoh(kappa, alpha=0.37, mass_ratio=1.0, proximal_ratio=0.0)
    k = np.asarray(kappa, np.complex128)
    np.testing.assert_allclose(Ad[:, 0, 0], np.exp(-k), rtol=RTOL * 10)
    np.testing.assert_allclose(bd[:, 0], (1 - np.exp(-k)) / k, rtol=RTOL * 10)
    if X64:   # candidate -> native as alpha -> 0, first order in alpha
        errs = []
        for a in (1e-2, 1e-3):
            Ad, bd, _ = PI.proximal_inertia_zoh(kappa, alpha=a)
            b = np.r_[1, np.zeros(63)][:, None] * np.ones((1, 96))
            h = np.asarray(PI.scan_blocks(Ad, bd, jnp.asarray(b), implementation="sequential"))[..., 0]
            errs.append(np.max(abs(h - _first_order(k, 0, b))))
        assert errs[1] < errs[0] / 5


def test_module_native_limit_matches_native_s5_layer():
    kw = R.ssm_kwargs("native_matched_s5")
    x = jnp.asarray(np.random.default_rng(2).normal(size=(64, kw["H"])), jnp.float32)
    native = S5SSM(**kw, step_rescale=1.0)
    params = native.init(jax.random.PRNGKey(3), x)
    limit = PI.ProximalInertialS5SSM(**kw, step_rescale=1.0, mass_ratio=1.0,
                                     proximal_ratio=0.0, history_alpha=0.3)
    assert sorted(flatten_dict(limit.init(jax.random.PRNGKey(3), x))) == sorted(flatten_dict(params))
    y0, y1 = native.apply(params, x), limit.apply(params, x)
    np.testing.assert_allclose(y1, y0, rtol=0, atol=(1e-9 if X64 else 2e-4) * float(jnp.abs(y0).max()))


@pytest.mark.skipif(not X64, reason="float64 reference")
@pytest.mark.parametrize("mass_ratio", [1.5, 1.0])
def test_independent_physical_reference_including_input_jumps(mass_ratio):
    kappa = _kappas(12, seed=4)
    rng = np.random.default_rng(5)
    b = rng.normal(size=(300, 12)) + 1j * rng.normal(size=(300, 12))
    Ad, bd, tau = PI.proximal_inertia_zoh(jnp.asarray(kappa), mass_ratio=mass_ratio)
    got = np.asarray(PI.scan_blocks(Ad, bd, jnp.asarray(b), implementation="sequential"))[..., 0]
    tau = np.asarray(tau)
    for p in range(12):
        want = _physical_reference(kappa[p], tau[p], .5 * tau[p], mass_ratio * tau[p], b[:, p])
        np.testing.assert_allclose(got[:, p], want, rtol=1e-9, atol=1e-11 * abs(want).max())
    # dropping the input-derivative jump gives a materially different response
    p = int(np.argmax(np.abs(kappa)))
    keep = _physical_reference(kappa[p], tau[p], .5 * tau[p], mass_ratio * tau[p], b[:, p])
    drop = _physical_reference(kappa[p], tau[p], .5 * tau[p], mass_ratio * tau[p], b[:, p], jump=False)
    assert np.abs(keep - drop).max() > 1e-3 * np.abs(keep).max()


@pytest.mark.skipif(not X64, reason="float64 reference")
def test_transfer_identity_and_cancellation():
    for k in _kappas(20, seed=6):
        k = complex(k)
        for mr in (1.5, 1.0):
            t = .1 / abs(k); mu, m = .5 * t, mr * t
            A, g = PI.inertia_generator(jnp.asarray([k]), t, mu, m)
            A, g = np.asarray(A[0]), np.asarray(g[0])
            for p in (0.3j / t, (1 + 2j) / t, 0.01 / t):
                Hs = np.linalg.solve(p * np.eye(2) - A, g)[0]
                Ht = (1 + t * p) / ((1 + mu * k) * (m * p * p + p) + k * (1 + t * p))
                assert abs(Hs - Ht) <= 1e-9 * abs(Ht)
            D = (1 + mu * k) * (m / t ** 2 - 1 / t) + k * 0
            if mr == 1.0:
                assert abs(D) < 1e-9 * abs(1 + mu * k) / t ** 2       # pole cancelled
            else:
                assert abs(D) > 0.1 * abs(1 + mu * k) / t                # pole released


@pytest.mark.skipif(not X64, reason="float64 reference")
def test_sharp_stability_bound():
    theta = np.linspace(-np.pi / 2 + 1e-7, np.pi / 2 - 1e-7, 4001)
    for mag in (1e-4, 1e-2, 1.0, 30.0):
        k = mag * np.exp(1j * theta)
        t = .1 / mag
        A, _ = PI.inertia_generator(jnp.asarray(k), t, .5 * t, 1.5 * t)
        assert np.asarray(np.linalg.eigvals(np.asarray(A))).real.max() < 0
        Ad, _, _ = PI.proximal_inertia_zoh(jnp.asarray(k))
        assert np.abs(np.linalg.eigvals(np.asarray(Ad))).max() < 1
    kc = 0.00023385579949444118 - 0.05009230721178928j      # mu=0.4 < m-tau=0.5
    A, _ = PI.inertia_generator(jnp.asarray([kc]), 1.0, 0.4, 1.5)
    assert np.linalg.eigvals(np.asarray(A[0])).real.max() > 0
    A, _ = PI.inertia_generator(jnp.asarray([kc]), 1.0, 0.5, 1.5)
    assert np.linalg.eigvals(np.asarray(A[0])).real.max() < 0


@pytest.mark.parametrize("impl", ["parallel_reference", "associative", "sequential"])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_agreement_full_length(impl, reverse):
    kappa = jnp.asarray(_kappas(8, seed=7))
    Ad, bd, _ = PI.proximal_inertia_zoh(kappa)
    rng = np.random.default_rng(8)
    b = jnp.asarray((rng.normal(size=(16000, 8)) + 1j * rng.normal(size=(16000, 8))).astype(CDT))
    ref = PI.scan_blocks(Ad, bd, b, reverse=reverse, implementation="parallel")
    got = PI.scan_blocks(Ad, bd, b, reverse=reverse, implementation=impl)
    scale = float(jnp.abs(ref).max())
    assert float(jnp.abs(got - ref).max()) <= (1e-11 if X64 else 5e-5) * scale
    if reverse:
        fwd = PI.scan_blocks(Ad, bd, b[::-1], implementation="sequential")[::-1]
        assert float(jnp.abs(fwd - ref).max()) <= (1e-11 if X64 else 5e-5) * scale


def _layer_loss(impl, params, x, module_kwargs):
    m = PI.ProximalInertialS5SSM(**module_kwargs, implementation=impl)
    return jnp.sum(m.apply(params, x) ** 2)


def test_gradients_custom_adjoint_equals_autodiff_and_are_finite():
    kw = {**R.ssm_kwargs("native_matched_s5"), "step_rescale": 1.0}
    x = jnp.asarray(np.random.default_rng(9).normal(size=(257, kw["H"])), jnp.float32)
    params = PI.ProximalInertialS5SSM(**kw).init(jax.random.PRNGKey(1), x)
    g1 = jax.grad(lambda p: _layer_loss("parallel", p, x, kw))(params)
    g0 = jax.grad(lambda p: _layer_loss("parallel_reference", p, x, kw))(params)
    for (k, a), b in zip(flatten_dict(g1).items(), flatten_dict(g0).values()):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all(), k
        assert np.abs(a - b).max() <= (1e-8 if X64 else 3e-3) * max(np.abs(b).max(), 1e-30), k
    assert any(np.abs(np.asarray(v)).max() > 0 for k, v in flatten_dict(g1).items()
               if k[-1] in ("Lambda_re", "Lambda_im", "log_step"))


@pytest.mark.skipif(not X64, reason="finite differences need float64")
def test_gradient_matches_finite_difference():
    kw = {**R.ssm_kwargs("native_matched_s5"), "step_rescale": 1.0}
    x = jnp.asarray(np.random.default_rng(10).normal(size=(64, kw["H"])))
    params = PI.ProximalInertialS5SSM(**kw).init(jax.random.PRNGKey(2), x)
    g = flatten_dict(jax.grad(lambda p: _layer_loss("parallel", p, x, kw))(params))
    flat = flatten_dict(params)
    from flax.traverse_util import unflatten_dict
    for leaf in ("Lambda_re", "Lambda_im", "log_step"):
        key = [k for k in flat if k[-1] == leaf][0]
        idx = (3,) + (0,) * (flat[key].ndim - 1)
        eps = 1e-6
        def f(delta):
            f2 = dict(flat); f2[key] = flat[key].at[idx].add(delta)
            return float(_layer_loss("parallel", unflatten_dict(f2), x, kw))
        fd = (f(eps) - f(-eps)) / (2 * eps)
        assert abs(fd - float(g[key][idx])) <= 1e-5 * max(1.0, abs(fd)), leaf


def test_no_parameters_added_and_bad_configs_refused():
    kw = {**R.ssm_kwargs("native_matched_s5"), "step_rescale": 1.0}
    x = jnp.zeros((8, kw["H"]))
    native = flatten_dict(S5SSM(**kw).init(jax.random.PRNGKey(0), x))
    ours = flatten_dict(PI.ProximalInertialS5SSM(**kw).init(jax.random.PRNGKey(0), x))
    assert sorted(native) == sorted(ours)
    for k in native:
        np.testing.assert_array_equal(native[k], ours[k])
    for bad in ({"history_alpha": 0.0}, {"mass_ratio": -1.0}, {"proximal_ratio": float("nan")},
                {"implementation": "eigen"}):
        with pytest.raises(ValueError):
            PI.ProximalInertialS5SSM(**{**kw, **bad}).init(jax.random.PRNGKey(0), x)


def test_production_init_poles_stable_in_this_precision():
    from experiments.s5_three_arm_full.runner import ssm_kwargs
    kw = ssm_kwargs("native_matched_s5")
    lam = kw["Lambda_re_init"] + 1j * kw["Lambda_im_init"]
    for delta in (1e-3, 1e-2, 1e-1):
        kappa = jnp.asarray((-lam * delta).astype(CDT))
        for mr in (1.5, 1.0):
            Ad, _, _ = PI.proximal_inertia_zoh(kappa, mass_ratio=mr)
            r = np.abs(np.linalg.eigvals(np.asarray(Ad).astype(np.complex128))).max()
            assert np.isfinite(r) and r < 1
