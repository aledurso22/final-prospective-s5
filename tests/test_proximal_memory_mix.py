"""Physical-law, stability and implementation checks; run each precision alone."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from s5.gradient_history import scan_history_blocks
from s5.proximal_memory_mix import (ProximalMemoryMixS5SSM,
    positive_memory_generator, positive_memory_zoh)
from s5.ssm import S5SSM
from s5.ssm_init import make_DPLR_HiPPO


F64 = jax.config.read("jax_enable_x64")
DTYPE = jnp.complex128 if F64 else jnp.complex64
TOL = 3e-10 if F64 else 1e-4


def _reference(kappa, alpha, fast=.1, slow=1., weight=4/9, proximal=.5):
    h = alpha/abs(kappa)
    tf, ts, mu = fast*h, slow*h, proximal*h
    R = 1+mu*kappa
    A = np.array([[0, 1-weight, weight], [-kappa/(R*tf), -1/tf, 0],
                  [-kappa/(R*ts), 0, -1/ts]], complex)
    b = np.array([0, 1/(R*tf), 1/(R*ts)], complex)
    augmented = np.zeros((4, 4), complex)
    augmented[:3, :3], augmented[:3, 3] = A, b
    discrete = expm(augmented)
    S = np.diag([1, h, h])
    return A, b, S@discrete[:3, :3]@np.linalg.inv(S), S@discrete[:3, 3], S


def _kwargs(bidirectional=True):
    lam, _, _, V, _ = make_DPLR_HiPPO(8)
    lam, V = lam[:4], V[:, :4]
    return dict(H=3, P=4, Lambda_re_init=lam.real, Lambda_im_init=lam.imag,
                V=V, Vinv=V.conj().T, C_init="lecun_normal", discretization="zoh",
                dt_min=.001, dt_max=.1, conj_sym=True, clip_eigs=True,
                bidirectional=bidirectional)


@pytest.mark.parametrize("alpha", [.1, 1.])
def test_zoh_matches_independent_physical_matrix_exponential(alpha):
    k = jnp.asarray([.001+.9j, .03-.2j, .4+3j], DTYPE)
    Ad, bd, h = positive_memory_zoh(k, alpha)
    assert Ad.dtype == bd.dtype == DTYPE
    assert h.dtype == k.real.dtype
    np.testing.assert_allclose(h, alpha/np.abs(k), rtol=TOL, atol=TOL)
    refs = [_reference(complex(ki), alpha) for ki in np.asarray(k)]
    np.testing.assert_allclose(Ad, np.stack([r[2] for r in refs]), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(bd, np.stack([r[3] for r in refs]), rtol=TOL, atol=TOL)


def test_generator_and_nonzero_initial_zoh_against_literal_ode():
    k, alpha, w = .08+.6j, .7, 4/9
    h = alpha/abs(k)
    tf, ts, mu = .1*h, h, .5*h
    R = 1+mu*k
    A, b = positive_memory_generator(jnp.asarray(k, DTYPE), tf, ts, mu, w)
    x0 = np.array([.3+.1j, -.2+.4j, .7-.3j])
    physical_input = .9-.5j
    def ode(_, x):
        direction = (physical_input-k*x[0])/R
        return np.array([(1-w)*x[1]+w*x[2],
                         (direction-x[1])/tf, (direction-x[2])/ts])
    np.testing.assert_allclose(A@x0+b*physical_input, ode(0, x0), rtol=TOL, atol=TOL)
    sol = solve_ivp(ode, (0., 1.), x0, rtol=2e-12, atol=2e-13)
    Ad, bd, _ = positive_memory_zoh(jnp.asarray(k, DTYPE), alpha)
    scale = np.array([1., h, h])
    actual = np.asarray(Ad)@(scale*x0)+np.asarray(bd)*physical_input
    np.testing.assert_allclose(actual, scale*sol.y[:, -1], rtol=TOL, atol=TOL)


@pytest.mark.parametrize("weight", [0., 4/9, 1.])
@pytest.mark.parametrize("equal_times", [False, True])
def test_endpoint_weights_equal_times_and_repeated_filter_poles(weight, equal_times):
    k = jnp.asarray([.07+.4j, .2-.3j], DTYPE)
    fast, slow = .2, .2 if equal_times else 1.
    ratio = (1-weight)*fast+weight*slow
    Ad, bd, _ = positive_memory_zoh(k, .5, fast, slow, weight, ratio)
    refs = [_reference(complex(ki), .5, fast, slow, weight, ratio) for ki in np.asarray(k)]
    np.testing.assert_allclose(Ad, np.stack([r[2] for r in refs]), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(bd, np.stack([r[3] for r in refs]), rtol=TOL, atol=TOL)
    gradient = jax.grad(lambda re: jnp.sum(abs(positive_memory_zoh(
        re+1j*k.imag, .5, fast, slow, weight, ratio)[0])**2))(k.real)
    assert np.isfinite(gradient).all()


def test_constant_direction_and_native_dc_gain_added_delay():
    # kappa=0 leaves a constant external direction; both filters approach d.
    A, b = positive_memory_generator(jnp.asarray(0j, DTYPE), .2, 1., .5, 4/9)
    d = .3-.7j
    steady_velocity = np.array([0, d, d])
    np.testing.assert_allclose(A@steady_velocity+b*d, [d, 0, 0], atol=TOL)
    for k in (.01+.2j, .1-.4j):
        tf, ts, w, mu = .2, 1., 4/9, .7
        A, b = positive_memory_generator(jnp.asarray(k, DTYPE), tf, ts, mu, w)
        def transfer(p):
            K = (1-w)/(1+tf*p)+w/(1+ts*p)
            return K/((1+mu*k)*p+k*K)
        for p in (0., .2j, 1.1j):
            actual = np.linalg.solve(p*np.eye(3)-np.asarray(A), np.asarray(b))[0]
            np.testing.assert_allclose(actual, transfer(p), rtol=TOL, atol=TOL)
        np.testing.assert_allclose(transfer(0), 1/k, rtol=1e-12)
        eps = 1e-7*abs(k)
        delay = -(transfer(eps)-transfer(-eps))/(2*eps*transfer(0))
        np.testing.assert_allclose(delay, 1/k+mu, rtol=2e-7, atol=2e-7)


def test_uniform_stability_bound_and_below_bound_counterexample():
    tf, ts, w = .1, 1., 4/9
    bound = (1-w)*tf+w*ts
    assert bound == .5
    for mu in (bound, 1.2*bound):
        for mag in np.geomspace(1e-4, 1e3, 19):
            for phi in np.linspace(-np.pi/2+1e-5, np.pi/2-1e-5, 17):
                k = mag*np.exp(1j*phi)
                R = 1+mu*k
                roots = np.roots([R*tf*ts, R*(tf+ts),
                                 R+k*((1-w)*ts+w*tf), k])
                assert roots.real.max() < 0, (mu, k, roots)
    omega = .1
    p = 1j*omega
    K = (1-w)/(1+tf*p)+w/(1+ts*p)
    mu = .4
    F = mu+K/p
    k = -1/F
    assert k.real > 0
    A, _ = positive_memory_generator(jnp.asarray(k, DTYPE), tf, ts, mu, w)
    assert np.min(abs(np.linalg.eigvals(np.asarray(A))-p)) < TOL
    frequencies = np.geomspace(1e-4, 1e3, 99)
    F = bound+((1-w)/(1+tf*1j*frequencies)+w/(1+ts*1j*frequencies))/(1j*frequencies)
    expected = bound-(1-w)*tf/(1+tf**2*frequencies**2)-w*ts/(1+ts**2*frequencies**2)
    np.testing.assert_allclose(F.real, expected, atol=3e-15)


@pytest.mark.parametrize("reverse", [False, True])
def test_scan_outputs_and_full_parameter_input_gradients(reverse):
    k = jnp.asarray([.02+.3j, .07-.8j], DTYPE)
    rng = np.random.default_rng(71)
    drive = jnp.asarray(rng.normal(size=(33, 2))+1j*rng.normal(size=(33, 2)), DTYPE)
    def output(re, im, u, method):
        A, b, _ = positive_memory_zoh(re+1j*im)
        return scan_history_blocks(A, b, u, reverse=reverse, implementation=method)
    methods = ("parallel", "sequential", "associative", "parallel_reference")
    values = [output(k.real, k.imag, drive, method) for method in methods]
    gradients = [jax.grad(lambda re, im, u: jnp.mean(abs(output(re, im, u, method)[..., 0])**2),
                          argnums=(0, 1, 2))(k.real, k.imag, drive) for method in methods]
    for value in values[1:]:
        np.testing.assert_allclose(values[0], value, rtol=TOL, atol=TOL)
    for grad in gradients:
        for g, expected in zip(grad, gradients[0]):
            assert np.isfinite(g).all()
            np.testing.assert_allclose(g, expected, rtol=10*TOL, atol=10*TOL)
    eps = 1e-5 if F64 else 3e-4
    fun = lambda re: jnp.mean(abs(output(re, k.imag, drive, "parallel")[..., 0])**2)
    numeric = []
    for i in range(2):
        bump = jnp.zeros_like(k.real).at[i].set(eps)
        numeric.append(float((fun(k.real+bump)-fun(k.real-bump))/(2*eps)))
    np.testing.assert_allclose(gradients[0][0], numeric,
                               rtol=8e-5 if F64 else .025,
                               atol=1e-4 if F64 else .02)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_unchanged_parameter_tree_native_branch_and_layer_gradients(bidirectional):
    kw = _kwargs(bidirectional)
    x = jnp.asarray(np.random.default_rng(6).normal(size=(17, 3)), jnp.float32)
    key = jax.random.PRNGKey(301)
    native = S5SSM(**kw)
    control = ProximalMemoryMixS5SSM(**kw, history_alpha=0, proximal_ratio=0)
    mix = ProximalMemoryMixS5SSM(**kw)
    sequential = ProximalMemoryMixS5SSM(**kw, implementation="sequential")
    variables = native.init(key, x)
    for model in (control, mix):
        other = model.init(key, x)
        assert jax.tree_util.tree_structure(variables) == jax.tree_util.tree_structure(other)
        for a, b in zip(jax.tree_util.tree_leaves(variables), jax.tree_util.tree_leaves(other)):
            assert np.array_equal(a, b)
    assert np.array_equal(native.apply(variables, x), control.apply(variables, x))
    np.testing.assert_allclose(mix.apply(variables, x), sequential.apply(variables, x), rtol=TOL, atol=TOL)
    gradients = [jax.grad(lambda p: jnp.mean(model.apply({"params": p}, x)**2))(variables["params"])
                 for model in (mix, sequential)]
    for a, b in zip(*[jax.tree_util.tree_leaves(g) for g in gradients]):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=10*TOL, atol=10*TOL)
    bound = mix.bind(variables)
    drive = jax.vmap(lambda u: bound.history_B@u)(x[:1])
    forward = scan_history_blocks(bound.history_Ad, bound.history_bd, drive)
    backward = scan_history_blocks(bound.history_Ad, bound.history_bd, drive, reverse=True)
    assert np.array_equal(forward, backward)


@pytest.mark.parametrize("change", [
    {"history_alpha": -1.}, {"history_alpha": float("nan")},
    {"fast_ratio": 0.}, {"fast_ratio": float("inf")},
    {"slow_ratio": .05}, {"slow_ratio": float("nan")},
    {"slow_weight": -.1}, {"slow_weight": 1.1},
    {"proximal_ratio": .49}, {"proximal_ratio": float("inf")},
    {"discretization": "bilinear"}, {"implementation": "unknown"},
])
def test_constructor_refuses_invalid_or_below_bound_options(change):
    kwargs = dict(_kwargs(), **change)
    with pytest.raises(ValueError):
        ProximalMemoryMixS5SSM(**kwargs).init(jax.random.PRNGKey(0), jnp.zeros((2, 3)))


def test_native_branch_is_distinct_from_singular_normalized_blocks():
    with pytest.raises(ValueError):
        positive_memory_zoh(jnp.asarray(.1+.2j, DTYPE), alpha=0)


def test_bound_comparison_roundoff_does_not_clip_the_supplied_coefficient():
    ratio = .5-2e-16
    model = ProximalMemoryMixS5SSM(**_kwargs(), proximal_ratio=ratio)
    variables = model.init(jax.random.PRNGKey(0), jnp.zeros((2, 3)))
    assert model.bind(variables).proximal_ratio == ratio
    assert model.bind(variables).proximal_ratio < .5
    with pytest.raises(ValueError):
        ProximalMemoryMixS5SSM(**_kwargs(), proximal_ratio=.5-1e-8).init(
            jax.random.PRNGKey(0), jnp.zeros((2, 3)))
