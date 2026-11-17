"""Predictor-corrector form of the generalized S5: identities and scans (f64).

The completed generalized arm is the oracle for variant A. Every coefficient
formula is checked against a LITERAL evaluation of the predictor-corrector
equations, never only against itself. Stability claims use explicit roots,
since the per-mode coefficients are complex.

    JAX_ENABLE_X64=1 python -m pytest tests/test_predictor_corrector.py -q
"""

import math

import jax
import jax.numpy as np
import numpy as onp
import pytest

from s5 import factored_recurrence as FR
from s5 import predictor_corrector as PC
from s5.discrete_recurrence import (generalized_coefficients, scan_companion,
                                    scan_companion_sequential)
from s5.generalized_prospective_ssm import (GeneralizedProspectiveS5SSM,
                                            response_mass_gamma)
from s5.ssm_init import make_DPLR_HiPPO

RNG = onp.random.default_rng(20261117)
ALL = list(PC.VARIANTS.items())


@pytest.fixture(autouse=True)
def _require_x64():
    assert jax.config.read("jax_enable_x64"), (
        "x64 is OFF. Re-run as: JAX_ENABLE_X64=1 python -m pytest "
        "tests/test_predictor_corrector.py -q")


def _modes(n=300):
    mag = 1.0 - 10 ** RNG.uniform(-3.5, -0.5, n)
    return mag * onp.exp(1j * RNG.uniform(-3.0, 3.0, n))


def _params(n):
    T = 10 ** RNG.uniform(-2, 1, n)
    g = 10 ** RNG.uniform(-1.5, 1, n)
    rho = RNG.uniform(1e-4, 1, n)
    return T, rho * g * T, g


def _s5_kwargs(H=4, block=8, bidirectional=True):
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block)
    half = block // 2
    Lambda, V = Lambda[:half], V[:, :half]
    return dict(H=H, P=half, Lambda_re_init=Lambda.real,
                Lambda_im_init=Lambda.imag, V=V, Vinv=V.conj().T,
                C_init="lecun_normal", discretization="zoh", dt_min=0.001,
                dt_max=0.1, conj_sym=True, clip_eigs=True,
                bidirectional=bidirectional)


# ----------------------------------------- 1. recovery of the existing arm ---
def test_variant_A_coefficients_are_generalized_coefficients():
    lam = _modes(); T, M, g = _params(lam.size)
    b = RNG.normal(size=(lam.size, 3)) + 1j * RNG.normal(size=(lam.size, 3))
    want = generalized_coefficients(np.asarray(lam), np.asarray(b),
                                    np.asarray(T), np.asarray(M), np.asarray(g))
    got = PC.pc_coefficients(np.asarray(lam), np.asarray(b), np.asarray(T),
                             np.asarray(M), np.asarray(g), "legacy", "old")
    for w, v in zip(want, got):
        assert onp.allclose(v, w, rtol=1e-13, atol=1e-15)


def test_residual_form_coefficients():
    T, M, g = _params(50)
    for scale in PC.TARGET_SCALES:
        k = PC.target_scale(T, g, scale)
        mu, eta, zeta = PC.residual_coefficients(T, M, g, k)
        Q = M + g + T
        assert onp.allclose(mu, (M + T) / Q) and onp.allclose(eta, k / Q)
        assert onp.allclose(zeta, k * T / Q)
        ratio = eta / (1 - mu)
        assert onp.allclose(ratio, T / g if scale == "legacy" else 1.0)


def test_variant_A_module_is_the_generalized_module():
    kw = _s5_kwargs()
    x = jax.random.normal(jax.random.PRNGKey(1), (64, kw["H"]))
    gen = GeneralizedProspectiveS5SSM(**kw, implementation="sequential")
    pca = PC.PredictorCorrectorS5SSM(**kw, scale="legacy", corrector="old",
                                     implementation="sequential")
    pg = gen.init(jax.random.PRNGKey(0), x)
    pp = pca.init(jax.random.PRNGKey(0), x)
    assert jax.tree_util.tree_structure(pg) == jax.tree_util.tree_structure(pp)
    for a, b in zip(jax.tree_util.tree_leaves(pg), jax.tree_util.tree_leaves(pp)):
        assert onp.array_equal(a, b)                    # same values, names
    # not bitwise: the raw parameters are float32 in BOTH modules, so
    # (T, M, gamma) carry float32 rounding through two different groupings
    # of the same algebra; agreement is at that level
    assert onp.allclose(gen.apply(pg, x), pca.apply(pp, x), rtol=1e-6, atol=1e-7)
    # and the very same parameters drive every variant (matched init)
    for scale, corr in PC.VARIANTS.values():
        m = PC.PredictorCorrectorS5SSM(**kw, scale=scale, corrector=corr)
        pv = m.init(jax.random.PRNGKey(0), x)
        for a, b in zip(jax.tree_util.tree_leaves(pg),
                        jax.tree_util.tree_leaves(pv)):
            assert onp.array_equal(a, b)


# ------------------------------- 2. literal equations vs expanded coeffs ---
@pytest.mark.parametrize("name,variant", ALL)
def test_direct_predictor_corrector_equals_expanded_recurrence(name, variant):
    scale, corr = variant
    P_, H_, L_ = 7, 3, 80
    lam = _modes(P_); T, M, g = _params(P_)
    b = RNG.normal(size=(P_, H_)) + 1j * RNG.normal(size=(P_, H_))
    x = RNG.normal(size=(L_, H_))
    want = PC.direct_rollout(lam, b, T, M, g, x, scale, corr)
    co = PC.pc_coefficients(np.asarray(lam), np.asarray(b), np.asarray(T),
                            np.asarray(M), np.asarray(g), scale, corr)
    for scan in (scan_companion_sequential, scan_companion):
        got = onp.asarray(scan(*co, np.asarray(x)))
        assert onp.allclose(got, want, rtol=1e-11, atol=1e-11 * abs(want).max())


@pytest.mark.parametrize("name,variant", ALL)
def test_expanded_form_with_nonzero_prehistory(name, variant):
    """Initial conditions: the literal equations with s_{-1}, s_{-2}, x_{-1}
    set equal the expanded recurrence started from the same prehistory."""
    scale, corr = variant
    lam = _modes(5); T, M, g = _params(5)
    b = RNG.normal(size=(5, 2)) + 1j * RNG.normal(size=(5, 2))
    x = RNG.normal(size=(30, 2)); xp = RNG.normal(size=2)
    s1, s2 = RNG.normal(size=5) + 1j, RNG.normal(size=5) - 1j
    want = PC.direct_rollout(lam, b, T, M, g, x, scale, corr, (s1, s2), xp)
    a1, a2, c1, c2 = PC.pc_coefficients(lam, b, T, M, g, scale, corr)
    out, p1, p2, up = [], s1, s2, xp
    for xt in x:
        s = a1 * p1 + a2 * p2 + c1 @ xt + c2 @ up
        out.append(s); p2, p1, up = p1, s, xt
    assert onp.allclose(onp.stack(out), want, rtol=1e-11, atol=1e-11)


def test_closed_form_coefficients_old_and_new():
    lam = _modes(200); T, M, g = _params(200)
    for scale in PC.TARGET_SCALES:
        k = PC.target_scale(T, g, scale)
        mu, eta, zeta = PC.residual_coefficients(T, M, g, k)
        K = 1 - lam; P = 1 - eta * K; L = mu - zeta * K
        o = PC.scalar_coefficients(lam, T, M, g, scale, "old")
        n = PC.scalar_coefficients(lam, T, M, g, scale, "prediction")
        for got, want in zip(o, (P + L, -L, eta + zeta, -zeta)):
            assert onp.allclose(got, want)
        for got, want in zip(n, (P * (1 + L), -P * L, eta + zeta * P, -zeta * P)):
            assert onp.allclose(got, want)


def test_new_equals_old_correction_plus_P_times_displacement():
    """new: s = s1 - eta r_t(s1) + P d ; old: s = s1 - eta r_t(s1) + d."""
    lam = 0.97 * onp.exp(-0.4j); T, M, g = 0.3, 0.1, 0.8
    mu, eta, zeta = PC.residual_coefficients(T, M, g, T)
    K = 1 - lam; P = 1 - eta * K
    s1, s2, u, up = 0.3 + 0.2j, -0.1 + 0.5j, 0.7 - 0.1j, -0.2 + 0.3j
    r = lambda s, uu: K * s - uu
    d = mu * (s1 - s2) - zeta * (r(s1, u) - r(s2, up))
    y = s1 + d
    assert onp.isclose(y - eta * r(s1, u), s1 - eta * r(s1, u) + d)
    assert onp.isclose(y - eta * r(y, u), s1 - eta * r(s1, u) + P * d)


# ---------------------------------------------------- 3. the identities ---
@pytest.mark.parametrize("name,variant", ALL)
def test_dc_gain_is_native(name, variant):
    lam = _modes(); T, M, g = _params(lam.size)
    co = PC.scalar_coefficients(lam, T, M, g, *variant)
    assert onp.allclose(PC.dc_gain(*co), 1 / (1 - lam), rtol=1e-9)


@pytest.mark.parametrize("name,variant", ALL)
def test_dc_group_delay_closed_form_and_numerical(name, variant):
    lam = _modes(60); T, M, g = _params(60)
    co = PC.scalar_coefficients(lam, T, M, g, *variant)
    analytic = PC.group_delay_dc(*co)
    assert onp.allclose(analytic, PC.predicted_delay(lam, T, M, g, *variant),
                        rtol=1e-9, atol=1e-9)
    w = 1e-6   # central difference of the unwrapped phase at DC
    ph = lambda om: onp.angle(PC.transfer(*co, onp.exp(1j * om)))
    numeric = -(onp.angle(onp.exp(1j * (ph(w) - ph(-w))))) / (2 * w)
    stable = PC.schur_cohn(co[0], co[1])
    assert onp.allclose(numeric[stable], analytic[stable], rtol=1e-4, atol=1e-4)


def test_prediction_corrector_adds_exactly_mu_tokens_of_delay():
    lam = _modes(); T, M, g = _params(lam.size)
    for scale in PC.TARGET_SCALES:
        mu, _, _ = PC.residual_coefficients(T, M, g, PC.target_scale(T, g, scale))
        d_old = PC.group_delay_dc(*PC.scalar_coefficients(lam, T, M, g, scale, "old"))
        d_new = PC.group_delay_dc(*PC.scalar_coefficients(lam, T, M, g, scale, "prediction"))
        assert onp.allclose(d_new - d_old, mu, rtol=1e-8, atol=1e-8)


def test_calibrated_old_has_exactly_native_dc_delay():
    lam = _modes(); T, M, g = _params(lam.size)
    native = PC.group_delay_dc(*PC.native_coefficients(lam))
    calib = PC.group_delay_dc(*PC.scalar_coefficients(lam, T, M, g, "calibrated", "old"))
    assert onp.allclose(calib, native, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("corrector", PC.CORRECTORS)
@pytest.mark.parametrize("scale", PC.TARGET_SCALES)
def test_slow_pole_first_order_in_K(scale, corrector):
    """z_slow = 1 - [eta/(1-mu)] K + O(K^2): the error shrinks as K^2."""
    T, M, g = 0.4, 0.1, 0.9
    mu, eta, _ = PC.residual_coefficients(T, M, g, PC.target_scale(T, g, scale))
    errs = []
    for eps in (1e-2, 1e-3, 1e-4):
        K = eps * onp.exp(0.7j); lam = 1 - K
        z1, _ = PC.roots(*PC.scalar_coefficients(lam, T, M, g, scale, corrector)[:2])
        errs.append(abs(z1 - (1 - eta / (1 - mu) * K)))
    assert errs[1] / errs[0] < 0.02 and errs[2] / errs[1] < 0.02   # ~ 1e-2


def test_reference_point_old_reproduces_native_exactly():
    """k = gamma, gamma = 2, M = T (rho = 1/2): poles {lambda_bar, mu},
    numerator B_bar (1 - mu z^-1) -> the mu-mode cancels, H = native."""
    lam = _modes(40); T = 10 ** RNG.uniform(-2, 1, 40); g = 2.0 * onp.ones(40)
    M = 0.5 * g * T
    co = PC.scalar_coefficients(lam, T, M, g, "calibrated", "old")
    mu, _, _ = PC.residual_coefficients(T, M, g, g)
    z1, z2 = PC.roots(co[0], co[1])
    poles = onp.sort_complex(onp.stack([z1, z2], -1))
    assert onp.allclose(poles, onp.sort_complex(onp.stack([lam, mu + 0j], -1)),
                        atol=1e-10)
    x = RNG.normal(size=500)
    native = PC.simulate(*PC.native_coefficients(lam), x)
    assert onp.allclose(PC.simulate(*co, x), native, rtol=1e-10, atol=1e-10)
    new = PC.simulate(*PC.scalar_coefficients(lam, T, M, g, "calibrated",
                                              "prediction"), x)
    assert onp.max(abs(new - native)) > 1e-3 * onp.max(abs(native))


def test_calibrated_scale_is_redundant_rank_two():
    """eta = 1 - mu identically under k = gamma: the map (T, rho, gamma) ->
    (mu, eta, zeta) has Jacobian rank 2, versus 3 for the legacy scale."""
    def coeffs(raw, scale):
        T, M, g, _ = response_mass_gamma(raw[0], raw[1], raw[2])
        return np.stack(PC.residual_coefficients(
            T, M, g, PC.target_scale(T, g, scale)))
    raw = np.array([-2.97, 0.0, 0.54])
    for scale, rank in (("legacy", 3), ("calibrated", 2)):
        J = onp.asarray(jax.jacfwd(lambda r: coeffs(r, scale))(raw))
        sv = onp.linalg.svd(J, compute_uv=False)
        assert (sv > 1e-10 * sv[0]).sum() == rank, (scale, sv)


# --------------------------------------------------------- 4. stability ---
def test_schur_cohn_matches_explicit_roots():
    lam = _modes(4000); T, M, g = _params(4000)
    for variant in PC.VARIANTS.values():
        a1, a2, *_ = PC.scalar_coefficients(lam, T, M, g, *variant)
        z1, z2 = PC.roots(a1, a2)
        explicit = onp.maximum(abs(z1), abs(z2)) < 1
        margin = abs(onp.maximum(abs(z1), abs(z2)) - 1) > 1e-9
        assert onp.array_equal(PC.schur_cohn(a1, a2)[margin], explicit[margin])


def test_P_inside_disc_does_not_imply_stability_stable_old_unstable_new():
    """A near-Nyquist mode, legacy scale, low drag: |P| = 0.91 < 1, the
    OLD recurrence is stable (radius 0.925), the NEW one is not (1.887)."""
    lam = -0.9565550204271533 - 0.14460018638677533j
    T, rho, g = 0.1405803640432392, 0.018400380421423577, 0.004244009715520318
    M = rho * g * T
    mu, eta, zeta = PC.residual_coefficients(T, M, g, T)
    assert abs(1 - eta * (1 - lam)) < 1
    ro = max(map(abs, PC.roots(*PC.scalar_coefficients(lam, T, M, g, "legacy", "old")[:2])))
    rn = max(map(abs, PC.roots(*PC.scalar_coefficients(lam, T, M, g, "legacy", "prediction")[:2])))
    assert ro < 0.93 and rn > 1.8
    co = PC.scalar_coefficients(lam, T, M, g, "legacy", "prediction")
    h = PC.simulate(*co, onp.r_[1.0, onp.zeros(60)])
    assert abs(h[-1]) > 1e10          # it really diverges, nothing clips it


def test_partial_fractions_match_trajectories():
    lam = _modes(50); T, M, g = _params(50)
    for variant in PC.VARIANTS.values():
        co = PC.scalar_coefficients(lam, T, M, g, *variant)
        z1, z2, R1, R2 = PC.residues(*co)
        stable = onp.maximum(abs(z1), abs(z2)) < 1
        t = onp.arange(40)[:, None]
        pf = R1 * z1 ** t + R2 * z2 ** t
        h = PC.simulate(*co, onp.r_[1.0, onp.zeros(39)])
        sep = abs(z1 - z2) > 1e-3
        ok = stable & sep
        assert onp.allclose(h[:, ok], pf[:, ok], rtol=1e-8, atol=1e-8)


# ------------------------------------------------------------ 5. scans ---
def _loss(scan, variant, lam, b, T, M, g, x):
    co = PC.pc_coefficients(lam, b, T, M, g, *variant)
    return np.sum(np.abs(scan(*co, x)) ** 2)


@pytest.mark.parametrize("name,variant", ALL)
def test_scans_agree_values_and_gradients(name, variant):
    n = 6
    lam = np.asarray(_modes(n)); T, M, g = map(np.asarray, (0.05 * onp.ones(n),
                                                          0.025 * onp.ones(n),
                                                          onp.ones(n)))
    b = np.asarray(RNG.normal(size=(n, 3)) + 1j * RNG.normal(size=(n, 3)))
    x = jax.random.normal(jax.random.PRNGKey(4), (150, 3))
    f = lambda scan: (lambda T_, M_, g_, l_: _loss(scan, variant, l_, b, T_, M_, g_, x))
    want_v = f(scan_companion_sequential)(T, M, g, lam)
    want_g = jax.grad(f(scan_companion_sequential), argnums=(0, 1, 2, 3))(T, M, g, lam)
    for scan in (scan_companion, FR.scan_factored):
        assert onp.isclose(f(scan)(T, M, g, lam), want_v, rtol=1e-10)
        got = jax.grad(f(scan), argnums=(0, 1, 2, 3))(T, M, g, lam)
        for w, v in zip(want_g, got):
            assert onp.allclose(v, w, rtol=1e-7, atol=1e-8 * onp.abs(w).max())


def _repeated_root_mode(T, M, g, scale):
    """lambda_bar at which the NEW corrector's roots coincide exactly:
    P (1 + L)^2 = 4 L is a cubic in K; take a root with |1 - K| < 1."""
    mu, eta, zeta = PC.residual_coefficients(T, M, g, PC.target_scale(T, g, scale))
    P = onp.poly1d([-eta, 1.0]); L = onp.poly1d([-zeta, mu])
    for K in (P * (1 + L) ** 2 - 4 * L).roots:
        if abs(1 - K) < 1:
            return 1 - K
    raise AssertionError("no in-disc repeated-root mode")


@pytest.mark.parametrize("delta", [0.0, 1e-9, 1e-5])
def test_repeated_roots_sequential_vs_companion(delta):
    """At (near-)repeated roots the companion block scan is the trusted
    parallel reference; the factored scan is NOT relied upon there (its
    gradient is singular at a zero discriminant, see test_independent_inertia)."""
    T, M, g = 0.05, 0.025, 1.0
    lam0 = _repeated_root_mode(T, M, g, "calibrated")
    a1, a2, *_ = PC.scalar_coefficients(lam0, T, M, g, "calibrated", "prediction")
    assert abs(a1 * a1 + 4 * a2) < 1e-12 * abs(a1) ** 2
    lam = lam0 * (1 + delta)
    b = np.asarray([[0.4 + 0.2j, -0.3j]])
    x = jax.random.normal(jax.random.PRNGKey(2), (200, 2))
    args = tuple(np.asarray([v]) for v in (T, M, g))

    def f(scan):
        def loss(lr, li):
            l_ = np.atleast_1d(lr + 1j * li)
            return _loss(scan, ("calibrated", "prediction"), l_, b, *args, x)
        return loss

    p = (lam.real, lam.imag)
    want = onp.array(jax.grad(f(scan_companion_sequential), argnums=(0, 1))(*p))
    got = onp.array(jax.grad(f(scan_companion), argnums=(0, 1))(*p))
    assert onp.all(onp.isfinite(want)) and onp.all(onp.isfinite(got))
    assert onp.allclose(got, want, rtol=1e-8)
    assert onp.isclose(f(scan_companion)(*p), f(scan_companion_sequential)(*p),
                       rtol=1e-10)


# ------------------------------------------------- 6. bidirectional/init ---
@pytest.mark.parametrize("name,variant", ALL)
def test_bidirectional_is_two_zero_initialized_directions(name, variant):
    kw = _s5_kwargs(H=3)
    x = jax.random.normal(jax.random.PRNGKey(5), (40, kw["H"]))
    mod = PC.PredictorCorrectorS5SSM(**kw, scale=variant[0], corrector=variant[1])
    params = mod.init(jax.random.PRNGKey(0), x)
    co = mod.apply(params, method=lambda m: (m.pc_a1, m.pc_a2, m.pc_c1, m.pc_c2))
    xn = onp.asarray(x)
    lam, Bb = _lambda_b(mod, params)
    p = params["params"]
    T, M, g, _ = response_mass_gamma(p["generalized_T_raw"], p["generalized_rho_raw"],
                                     p["generalized_gamma_raw"])
    T, M, g = (onp.asarray(v, float) for v in (T, M, g))
    fwd = PC.direct_rollout(lam, Bb, T, M, g, xn, *variant)
    bwd = PC.direct_rollout(lam, Bb, T, M, g, xn[::-1], *variant)[::-1]
    states = onp.concatenate([fwd, bwd], -1)
    C = onp.asarray(p["C1"][..., 0] + 1j * p["C1"][..., 1])
    C2 = onp.asarray(p["C2"][..., 0] + 1j * p["C2"][..., 1])
    Ct = onp.concatenate([C, C2], -1)
    want = 2 * (states @ Ct.T).real + onp.asarray(p["D"]) * xn
    got = onp.asarray(mod.apply(params, x))
    # float32 raw parameters: the module forms mu, eta, zeta in float32
    assert onp.allclose(got, want, rtol=1e-6, atol=1e-6)
    assert co[0].shape == (kw["P"],)


def _lambda_b(mod, params):
    from s5.ssm import discretize_zoh
    def f(m):
        b = m.B[..., 0] + 1j * m.B[..., 1]
        step = m.step_rescale * np.exp(m.log_step[:, 0])
        return discretize_zoh(m.Lambda, b, step)
    lam, b = mod.apply(params, method=f)
    return onp.asarray(lam), onp.asarray(b)


# ------------------------------------- 7. second-order pole and families ---
@pytest.mark.parametrize("scale", PC.TARGET_SCALES)
def test_second_order_slow_pole_and_corrector_shift(scale):
    """z_slow = 1 - c K + c2 K^2 + O(K^3), c = eta/(1-mu), with
         old: c2 = -c (c - eta - zeta)/(1-mu)
         new: c2 = -c (c - eta (1+mu) - zeta)/(1-mu)
    so the correction point moves the principal pole by mu c^2 K^2."""
    T, M, g = 0.05, 0.025, 1.0
    mu, eta, zeta = PC.residual_coefficients(T, M, g, PC.target_scale(T, g, scale))
    c = eta / (1 - mu)
    want = {"old": -c * (c - eta - zeta) / (1 - mu),
            "prediction": -c * (c - eta * (1 + mu) - zeta) / (1 - mu)}
    assert onp.isclose(want["prediction"] - want["old"], mu * c * c)
    for corr, c2 in want.items():
        for K in (1e-3 * onp.exp(0.4j), 1e-3 * onp.exp(-1.2j)):
            z, _ = PC.roots(*PC.scalar_coefficients(1 - K, T, M, g, scale, corr)[:2])
            assert abs((z - (1 - c * K)) / K ** 2 - c2) < 5e-3 * max(abs(c2), 1e-3)


def test_calibrated_old_is_exactly_native_on_M_eq_T_gamma_minus_1():
    """mu = zeta  <=>  M = T (gamma - 1): then (z - lambda_bar)(z - mu) with
    numerator B_bar (1 - mu z^-1) for ANY gamma > 1, i.e. rho = 1 - 1/gamma.
    gamma = 2, M = T is the one point with rho = 1/2."""
    lam = _modes(30)
    x = RNG.normal(size=400)
    native = PC.simulate(*PC.native_coefficients(lam), x)
    for g in (1.2, 2.0, 7.0):
        T = 0.3; M = T * (g - 1)
        co = PC.scalar_coefficients(lam, T, M, g, "calibrated", "old")
        assert onp.allclose(PC.simulate(*co, x), native, rtol=1e-10, atol=1e-10)
