"""Independent mass / damping, zero ordinary drag: equations and scans.

The completed experiment is the oracle: on the constrained family M = rho
gamma T the research constructor must reproduce GeneralizedProspectiveS5SSM.
Every stability claim is checked against explicit roots, because the
per-mode coefficients are COMPLEX and real-coefficient tests do not apply.
"""

import math

import jax
import numpy as onp
import pytest
import jax.numpy as np  # noqa: E402

from s5 import factored_recurrence as FR  # noqa: E402
from s5 import independent_inertia as II  # noqa: E402
from s5.discrete_recurrence import (generalized_coefficients,  # noqa: E402
                                    scan_companion, scan_companion_sequential)
from s5.generalized_prospective_ssm import (  # noqa: E402
    GeneralizedProspectiveS5SSM, response_mass_gamma)
from s5.ssm_init import make_DPLR_HiPPO  # noqa: E402

RNG = onp.random.default_rng(20261116)


@pytest.fixture(autouse=True)
def _require_x64():
    """Same convention as test_factored_recurrence_float64.py: refuse to run
    in float32 rather than flip a global flag other test files would see."""
    assert jax.config.read("jax_enable_x64"), (
        "x64 is OFF. Re-run as: JAX_ENABLE_X64=1 python -m pytest "
        "tests/test_independent_inertia.py -q")


def _modes(n=400):
    """lambda_bar inside the disc, spread like the production inventory."""
    mag = 1.0 - 10 ** RNG.uniform(-3.5, -0.5, n)
    ang = RNG.uniform(-1.5, 1.5, n)
    return mag * onp.exp(1j * ang)


def _s5_kwargs(H=4, block=8):
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block)
    half = block // 2
    Lambda, V = Lambda[:half], V[:, :half]
    return dict(H=H, P=half, Lambda_re_init=Lambda.real,
                Lambda_im_init=Lambda.imag, V=V, Vinv=V.conj().T,
                C_init="lecun_normal", discretization="zoh", dt_min=0.001,
                dt_max=0.1, conj_sym=True, clip_eigs=True, bidirectional=True)


# ------------------------------------------------------- 1. the mapping ---
@pytest.mark.parametrize("h", [1.0, 0.37])
def test_closed_form_operator_is_the_production_recurrence(h):
    lam = _modes()
    T, M, g = (RNG.uniform(0.01, 3, lam.size), RNG.uniform(0, 3, lam.size),
               RNG.uniform(0, 3, lam.size))
    a1, a2, c1, c2 = generalized_coefficients(
        np.asarray(lam), np.ones((lam.size, 1), complex), np.asarray(T),
        np.asarray(M), np.asarray(g), h)
    c2_, c1_, c0_ = II.operator_polynomial(lam, T, M, g, h)
    assert onp.allclose(onp.asarray(a1), -c1_ / c2_, atol=1e-13)
    assert onp.allclose(onp.asarray(a2), -c0_ / c2_, atol=1e-13)
    if h == 1.0:   # the h = 1 closed form quoted in the docs
        kap = II.stiffness(lam, T)
        assert onp.allclose(c2_, M + T + g)
        assert onp.allclose(c1_, -2 * (M + T) - g + kap * (1 + T))
        assert onp.allclose(c0_, (M + T) - kap * T)
    # drive: c1 z + c2 = G((beta+delta) z - delta), zero at T/(h+T)
    z0 = II.transfer_zero(T, h)
    assert onp.allclose(onp.asarray(c1[:, 0]) * z0 + onp.asarray(c2[:, 0]), 0,
                        atol=1e-13)


def test_backward_difference_form_with_parasitic_mass_T():
    """(M+T) D^2 s_t + gamma D s_t + kappa (1 + T D) s_{t-1} = G (1+T D) x_t."""
    # a STABLE case, so an absolute tolerance is meaningful; gamma != 0 so
    # every term of the operator is exercised
    lam, T, M, g = 0.97 * onp.exp(0.3j), 0.4, 0.2, 0.7
    co = II.coefficients_numpy(lam, T, M, g)
    assert max(abs(r) for r in II.discrete_roots(lam, T, M, g)) < 1
    x = RNG.normal(size=300)
    s = II.recurrence_impulse(*co, 300, x=x)
    s = onp.concatenate([onp.zeros(3), s.ravel()])
    xp = onp.concatenate([onp.zeros(3), x])
    t = onp.arange(3, 303)
    D = lambda v, k: v[k] - v[k - 1]
    D2 = lambda v, k: v[k] - 2 * v[k - 1] + v[k - 2]
    kap, G = T * (1 - lam), T
    lhs = ((M + T) * D2(s, t) + g * D(s, t)
           + kap * (s[t - 1] + T * D(s, t - 1)))
    rhs = G * (xp[t] + T * D(xp, t))
    assert onp.allclose(lhs, rhs, atol=1e-11)


# --------------------------------------- 2. recovery and exact zero drag ---
def test_recovers_generalized_arm_on_the_constrained_family():
    kw = _s5_kwargs()
    x = jax.random.normal(jax.random.PRNGKey(1), (64, kw["H"]))
    gen = GeneralizedProspectiveS5SSM(**kw, implementation="sequential")
    T, M, g, _ = response_mass_gamma(*(np.full(kw["P"], v, np.float32) for v in
                                       (math.log(math.expm1(0.05)), 0.0,
                                        math.log(math.expm1(1.0)))))
    ind = II.IndependentInertiaS5SSM(
        **kw, response_init=float(T[0]), mass_init=float(M[0]),
        gamma_init=float(g[0]), implementation="sequential")
    pg = gen.init(jax.random.PRNGKey(0), x)
    pi = ind.init(jax.random.PRNGKey(0), x)
    shared = ("Lambda_re", "Lambda_im", "B", "C1", "C2", "D", "log_step")
    for name in shared:   # identical backbone, same key, same names
        assert onp.array_equal(pg["params"][name], pi["params"][name]), name
    got = ind.apply(pi, x)
    want = gen.apply(pg, x)
    assert onp.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_zero_damping_is_exactly_zero_with_positive_mass():
    kw = _s5_kwargs()
    x = jax.random.normal(jax.random.PRNGKey(1), (16, kw["H"]))
    mod = II.IndependentInertiaS5SSM(**kw, mass_init=0.3, zero_damping=True)
    params = mod.init(jax.random.PRNGKey(0), x)["params"]
    assert "inertia_gamma_raw" not in params
    T, M, gamma = II.independent_mass_damping(
        params["inertia_T_raw"], params["inertia_M_raw"], None)
    assert bool(np.all(gamma == 0.0)) and bool(np.all(M > 0))
    assert onp.allclose(M, 0.3, rtol=1e-6)
    # and the gradient cannot reintroduce it: no parameter feeds gamma
    grads = jax.grad(lambda p: np.sum(mod.apply({"params": p}, x) ** 2))(params)
    assert set(grads) == set(params)


# ------------------------------------------- 3. stability tests are right ---
def test_schur_cohn_matches_explicit_roots_for_complex_coefficients():
    lam = _modes(3000)
    T = 10 ** RNG.uniform(-2, 3, lam.size)
    M = onp.where(RNG.uniform(size=lam.size) < 0.2, 0, 10 ** RNG.uniform(-6, 4, lam.size))
    g = onp.where(RNG.uniform(size=lam.size) < 0.5, 0, 10 ** RNG.uniform(-3, 1, lam.size))
    r1, r2 = II.discrete_roots(lam, T, M, g)
    by_roots = onp.maximum(abs(r1), abs(r2)) < 1
    sc = II.schur_cohn_stable(lam, T, M, g)
    margin = abs(onp.maximum(abs(r1), abs(r2)) - 1) > 1e-9
    assert onp.array_equal(sc[margin], by_roots[margin])
    assert 0.05 < by_roots.mean() < 0.95     # both outcomes are exercised


def test_complex_hurwitz_matches_explicit_roots():
    a = RNG.normal(size=4000) + 1j * RNG.normal(size=4000)
    b = RNG.normal(size=4000) + 1j * RNG.normal(size=4000)
    root = onp.sqrt(a * a - 4 * b)
    by_roots = onp.maximum(((-a + root) / 2).real, ((-a - root) / 2).real) < 0
    assert onp.array_equal(II.hurwitz_stable_complex(a, b), by_roots)


def test_zero_drag_critical_mass_is_the_continuous_boundary():
    lam = _modes()
    T = 10 ** RNG.uniform(-2, 2, lam.size)
    Mc = II.zero_drag_critical_mass(lam, T)
    assert onp.all(II.continuous_stable(lam, T, 0.99 * Mc, 0.0))
    assert not onp.any(II.continuous_stable(lam, T, 1.01 * Mc, 0.0))


def test_real_mode_zero_drag_is_continuously_stable_for_every_mass():
    lam = onp.linspace(0.5, 0.9999, 50) + 0j
    for M in (1e-6, 1.0, 1e6):
        assert onp.all(II.continuous_stable(lam, 0.05, M, 0.0))


# ------------------------------------------- 4. structure of the transfer ---
def test_cancellation_happens_exactly_on_M_equals_T_gamma_minus_one():
    lam = _modes()
    T = RNG.uniform(0.01, 2, lam.size)
    g = RNG.uniform(1, 4, lam.size)
    assert onp.allclose(II.cancellation_residual(lam, T, T * (g - 1), g), 0,
                        atol=1e-12)
    res = II.cancellation_residual(lam, T, T * (g - 1) + 0.1, g)
    assert onp.allclose(res, 0.1 / (1 + T) ** 2)


def test_M0_gamma1_is_the_first_order_F_recurrence():
    """Recurrence order 2, minimal state dimension 1, pole F = 1 - kappa."""
    lam, T = 0.993 * onp.exp(0.2j), 0.05
    co = II.coefficients_numpy(lam, T, 0.0, 1.0)
    x = RNG.normal(size=500)
    s = II.recurrence_impulse(*co, 500, x=x).ravel()
    F, G = 1 - T * (1 - lam), T     # beta + delta = 1 at M=0, gamma=1
    ref, prev = [], 0
    for v in x:
        prev = F * prev + G * v
        ref.append(prev)
    assert onp.allclose(s, ref, atol=1e-12)


def test_dc_gain_is_native_and_delay_formula_holds():
    lam = _modes(50)
    for (T, M, g) in [(0.05, 0.025, 1.0), (0.05, 0.025, 0.0), (2.0, 5.0, 0.3)]:
        co = II.coefficients_numpy(lam, T, M, g)
        dc = (co[2] + co[3]) / (1 - co[0] - co[1])
        assert onp.allclose(dc, 1 / (1 - lam), rtol=1e-10)
        w = 1e-6
        z = onp.exp(1j * w)
        Hw = (co[2] + co[3] / z) / (1 - co[0] / z - co[1] / z ** 2)
        delay = -onp.angle(Hw / dc) / w
        assert onp.allclose(delay, II.low_frequency_delay(lam, T, M, g),
                            rtol=1e-3, atol=1e-3)


def test_double_unit_pole_is_defective_and_observable_not_cancelled():
    """kappa -> 0, gamma = 0: (M+T)(z-1)^2 with the zero at T/(1+T) != 1."""
    T, M = 0.05, 0.025
    c2, c1, c0 = II.operator_polynomial(1.0 + 0j, T, M, 0.0)
    assert onp.allclose(onp.roots([c2, c1, c0]), 1.0, atol=1e-7)
    assert abs(II.transfer_zero(T) - 1.0) > 0.9
    co = II.coefficients_numpy(1.0 + 0j, T, M, 0.0)
    step = II.recurrence_impulse(*co, 200, x=onp.ones(200)).ravel()
    # quadratic growth of a step: a true double integrator in H(z)
    second = onp.diff(step, 2)[50:]
    assert onp.allclose(second, second[0], rtol=1e-9) and abs(second[0]) > 0


# -------------------------------------------- 5. poles vs trajectories ---
def test_trajectory_is_the_partial_fraction_of_the_predicted_poles():
    lam = _modes(20)
    for (T, M, g) in [(0.05, 0.025, 1.0), (0.05, 0.025, 0.0), (50, 100, 0.0)]:
        co = II.coefficients_numpy(lam, T, M, g)
        r1, r2 = II.discrete_roots(lam, T, M, g)
        imp = II.recurrence_impulse(*co, 60)
        t = onp.arange(60)[:, None]
        # s_t = A r1^t + B r2^t for t >= 1, fit on t = 1, 2 then predict
        A = (imp[2] - r2 * imp[1]) / (r1 * (r1 - r2))
        B = (imp[1] - A * r1) / r2
        pred = A * r1 ** t + B * r2 ** t
        scale = abs(imp).max(0)
        assert onp.all(abs(pred[1:] - imp[1:]).max(0) / scale < 1e-8)


# ---------------------------------------------------------- 6. the scans ---
def _random_case(key, n_modes, gamma0):
    k = jax.random.split(key, 4)
    lam = (1 - 10 ** jax.random.uniform(k[0], (n_modes,), minval=-3, maxval=-1)
           ) * np.exp(1j * jax.random.uniform(k[1], (n_modes,), minval=-1, maxval=1))
    b = (jax.random.normal(k[2], (n_modes, 3))
         + 1j * jax.random.normal(k[3], (n_modes, 3))) * 0.3
    return lam, b


@pytest.mark.parametrize("gamma", [1.0, 0.0])
def test_factored_equals_sequential_values_and_gradients(gamma):
    lam, b = _random_case(jax.random.PRNGKey(3), 6, gamma)
    x = jax.random.normal(jax.random.PRNGKey(4), (120, 3))

    def loss(scan, T, M, g, lam_, b_):
        co = generalized_coefficients(lam_, b_, T, M, g)
        return np.sum(np.abs(scan(*co, x)) ** 2)

    args = (np.full(6, 0.05), np.full(6, 0.025), np.full(6, gamma), lam, b)
    want_v = loss(scan_companion_sequential, *args)
    want_g = jax.grad(loss, argnums=(1, 2, 3))(scan_companion_sequential, *args)
    for scan in (scan_companion, FR.scan_factored):
        assert onp.isclose(loss(scan, *args), want_v, rtol=1e-10)
        got = jax.grad(loss, argnums=(1, 2, 3))(scan, *args)
        for w, g_ in zip(want_g, got):
            assert onp.allclose(g_, w, rtol=1e-8, atol=1e-8 * onp.abs(w).max())


@pytest.mark.parametrize("delta", [1e-14, 1e-6, 1e-3])
def test_near_coincident_poles_values_and_gradients(delta):
    x = jax.random.normal(jax.random.PRNGKey(0), (300, 2))
    c1 = np.array([[0.3 + 0.1j, -0.2j]])
    c2 = np.array([[0.1, 0.05j]])

    def loss(scan, rr, ri):
        r = rr + 1j * ri
        a1, a2 = FR.roots_to_coefficients(np.atleast_1d(r),
                                          np.atleast_1d(r * (1 + delta)))
        return np.sum(np.abs(scan(a1, a2, c1, c2, x)) ** 2)

    p = (0.999 * math.cos(0.01), 0.999 * math.sin(0.01))
    want = onp.array(jax.grad(lambda a, b: loss(scan_companion_sequential, a, b),
                              argnums=(0, 1))(*p))
    got = onp.array(jax.grad(lambda a, b: loss(FR.scan_factored, a, b),
                             argnums=(0, 1))(*p))
    assert onp.allclose(got, want, rtol=1e-8)


def test_FINDING_factored_gradient_is_nan_when_discriminant_rounds_to_zero():
    """Recorded, not fixed: the forward value at a repeated root is exact
    (as documented), but d sqrt(disc) is infinite at disc == 0, so the
    factored scan's gradient is NaN there while the oracle's is finite.
    Measured: float64 at relative root separation 0 and 1e-10; float32 at 0.
    Irrelevant to the completed run (roots ~0.999 and ~0.05)."""
    x = jax.random.normal(jax.random.PRNGKey(0), (100, 1))
    c1, c2 = np.array([[1.0 + 0j]]), np.array([[0.0 + 0j]])

    def loss(scan, rr):
        r = np.atleast_1d(rr + 0j)
        a1, a2 = FR.roots_to_coefficients(r, r)
        return np.sum(np.abs(scan(a1, a2, c1, c2, x)) ** 2)

    want = jax.grad(lambda r: loss(scan_companion_sequential, r))(0.99)
    got = jax.grad(lambda r: loss(FR.scan_factored, r))(0.99)
    assert onp.isfinite(want)
    assert onp.isnan(got)


# ------------------------------------- 7. the separate exact-ZOH scheme ---
def test_exact_zoh_poles_dc_and_state_space_simulation():
    lam = np.asarray(_modes(12))
    b = np.ones((12, 1), complex)
    for (T, M, g) in [(0.05, 0.025, 1.0), (5.0, 0.1, 0.0), (0.05, 1e-7, 0.0)]:
        full = lambda v: np.full(12, v)
        a1, a2, c1, c2 = map(onp.asarray, II.exact_zoh_coefficients(
            lam, b, full(T), full(M), full(g)))
        s1, s2 = II.continuous_roots(onp.asarray(lam), T, M, g)
        e1, e2 = onp.exp(s1), onp.exp(s2)
        assert onp.allclose(a1, e1 + e2, atol=1e-10)
        assert onp.allclose(a2, -e1 * e2, atol=1e-10)
        dc = (c1[:, 0] + c2[:, 0]) / (1 - a1 - a2)
        assert onp.allclose(dc * (1 - onp.asarray(lam)), 1, rtol=1e-6)
    # direct two-state simulation: M xi'' + (g + T k) xi' + k xi = G x, s = xi + T xi'
    from scipy.linalg import expm
    lam1, T, M, g = complex(lam[0]), 2.0, 0.3, 0.0
    kap = T * (1 - lam1)
    A = onp.array([[0, 1], [-kap / M, -(g + T * kap) / M]])
    aug = onp.zeros((3, 3), complex)
    aug[:2, :2], aug[1, 2] = A, 1 / M
    E = expm(aug)
    xs = RNG.normal(size=80)
    xi, ref = onp.zeros(2, complex), []
    for v in xs:
        xi = E[:2, :2] @ xi + E[:2, 2] * v
        ref.append((xi[0] + T * xi[1]) * T)
    co = tuple(onp.asarray(c)[0] for c in II.exact_zoh_coefficients(
        np.asarray([lam1]), np.ones((1, 1), complex), np.full(1, T),
        np.full(1, M), np.full(1, g)))
    got = II.recurrence_impulse(co[0], co[1], co[2][0], co[3][0], 80, x=xs)
    assert onp.allclose(got.ravel(), ref, atol=1e-10)


def test_zero_drag_memory_time_collapses_to_T():
    """Continuous: slow time constant is T + gamma/kappa (as M -> 0), so at
    gamma = 0 it is T whatever lambda_bar is; the exact scheme shows it."""
    lam = np.asarray(_modes(20))
    for T in (0.05, 5.0):
        s1, s2 = II.continuous_roots(onp.asarray(lam), T, 1e-9, 0.0)
        slow = onp.where(abs(s1) < abs(s2), s1, s2)
        assert onp.allclose(slow, -1 / T, rtol=1e-3)
