"""Focused checks for the adaptive associative-memory study. CLUSTER-ONLY.

PREDECLARED TOLERANCES AND STEPS, frozen before any numerical check ran:

    EXACT64   1e-9   one rank-one (W, Z) step vs an independently assembled
                     dense augmented-matrix exponential of the FULL matrix ODE
    TRAJ32    2e-5   the same over a 64-token genuine float32 trajectory
    GRAD64    1e-6   relative, JVP vs central differences, float64, away from
                     a near-zero reference
    NEAR64    1e-9   ABSOLUTE bound on the discrepancy when the reference
                     directional derivative is below 1e-6; this bounds the
                     difference, it does not excuse a small analytic value
    GRAD32    2e-2   production float32, enforced SEPARATELY at BOTH declared
                     perturbations, with an absolute fallback NEAR32 = 3e-3
                     that bounds the actual discrepancy
    IDENT64   1e-10  analytic limits and reductions of the law
    REPRO64   1e-10  the completed study's trajectories at the reference
                     coefficients with gate = 1, under Z = -P/gamma
    STREAM32  2e-5   chunked carry vs an unsplit sequence
    CAL       1e-10  calibration observable; 1e-8 on recovering nu = 4/3

    GRAD_PERTURBATIONS   float64 (1e-5, 1e-6);  float32 (1e-2, 3e-3)

No step is chosen after seeing a result, and no tolerance is relaxed to make a
check pass; a needed repair is recorded in the protocol instead.

INITIAL-CONDITION CONVENTION. Every derivative check holds the initial carry
`(W0, Z0)` fixed and parameter-independent. `Z0` and `P0 = -gamma Z0` define
DIFFERENT initial-condition dependencies once gamma is learned, so parameter
gradients across those two graphs need not agree and are never compared here.

Float64 checks run on cluster CPU as independent references; the production
path is float32 on GPU. Each is labelled where it matters.
"""

import os
import sys

import jax
import numpy as onp
import pytest
from scipy.linalg import expm as sp_expm

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.adaptive_memory import calibrate as CAL           # noqa: E402
from experiments.adaptive_memory import dynamics as AD             # noqa: E402
from experiments.adaptive_memory import gate as AG                 # noqa: E402
from experiments.adaptive_memory import model as AM                # noqa: E402
from experiments.nested_memory import dynamics as ND               # noqa: E402
from experiments.nested_memory import model as NM                  # noqa: E402
from experiments.nested_memory import task as TK                   # noqa: E402

EXACT64, TRAJ32, GRAD64, NEAR64 = 1e-9, 2e-5, 1e-6, 1e-9
GRAD32, NEAR32, IDENT64, REPRO64 = 2e-2, 3e-3, 1e-10, 1e-10
STREAM32, CAL_TOL, NU_TOL = 2e-5, 1e-10, 1e-8
FD64 = (1e-5, 1e-6)
FD32 = (1e-2, 3e-3)


# --------------------------- independent dense-ODE references --------------
def _dense(W, Z, k, v, w, kind, coeff, h=1.0):
    """Advance the FULL matrix ODE by an augmented affine matrix exponential.

    Assembled from the CONTINUOUS law, not from the 2x2 generator or the
    rank-one algebra:

      prospective   Wdot = -nu w (W K - v k^T) - Z/tau
                    Zdot = -nu (1-rho) w (W K - v k^T) - Z/tau
      inertial      Wdot = -Z/tau
                    Zdot = +eta w (W K - v k^T) - Z/tau

    with `K = k k^T`. Exponentiates `[[A, c], [0, 0]]` via `scipy.linalg.expm`.
    """
    dv, dk = W.shape
    n = dv * dk
    K = onp.outer(k, k)
    WK = onp.kron(K.T, onp.eye(dv))          # vec(W K) = (K^T kron I) vec(W)
    vk = onp.reshape(onp.outer(v, k), -1, order="F")
    A = onp.zeros((2 * n + 1, 2 * n + 1))
    if kind == "prospective":
        nu, tau, rho = coeff["nu"], coeff["tau"], coeff["rho"]
        g1, g2, inv = -nu * w, -nu * (1.0 - rho) * w, 1.0 / tau
        A[:n, :n] = g1 * WK; A[:n, 2 * n] = -g1 * vk
        A[:n, n:2 * n] = -inv * onp.eye(n)
    else:
        eta, tau = coeff["eta"], coeff["tau"]
        g1, g2, inv = 0.0, eta * w, 1.0 / tau
        A[:n, n:2 * n] = -inv * onp.eye(n)
    A[n:2 * n, :n] = g2 * WK
    A[n:2 * n, 2 * n] = -g2 * vk
    A[n:2 * n, n:2 * n] = -inv * onp.eye(n)
    z = onp.concatenate([onp.reshape(W, -1, order="F"),
                         onp.reshape(Z, -1, order="F"), [1.0]])
    out = sp_expm(A * h) @ z
    return (onp.reshape(out[:n], (dv, dk), order="F"),
            onp.reshape(out[n:2 * n], (dv, dk), order="F"))


def _gen(kind, coeff, w, dt=onp.float64):
    """The generator, with EVERY coefficient in the requested dtype.

    R5. The previous revision built default-dtype (float64, with x64 on)
    coefficients and cast only the finished exponential, so the "float32
    trajectory" test exercised float32 STATE arithmetic against float64
    COEFFICIENTS. The coefficient inputs, the generator, the exponential and
    the idle factor are now all genuinely the requested dtype, each asserted.
    """
    if kind == "prospective":
        G = AD.prospective_generator(jnp.asarray(coeff["nu"], dt),
                                     jnp.asarray(coeff["tau"], dt),
                                     jnp.asarray(coeff["rho"], dt),
                                     jnp.asarray(float(w), dt))
    else:
        G = AD.inertial_generator(jnp.asarray(coeff["eta"], dt),
                                  jnp.asarray(coeff["tau"], dt),
                                  jnp.asarray(float(w), dt))
    assert G.dtype == dt, (kind, G.dtype, dt)
    return G


def _step(kind, coeff, W, Z, k, v, w, dt=onp.float64):
    G = _gen(kind, coeff, w, dt)
    F = AD.expm2(G)
    a0 = jnp.exp(-jnp.asarray(AD.H, dt) / jnp.asarray(coeff["tau"], dt))
    assert F.dtype == dt and a0.dtype == dt, (kind, F.dtype, a0.dtype, dt)
    W2, Z2 = AD.two_state_step(jnp.asarray(W, dt), jnp.asarray(Z, dt),
                               jnp.asarray(k, dt), jnp.asarray(v, dt), F, a0)
    assert W2.dtype == dt and Z2.dtype == dt, (kind, W2.dtype, dt)
    return onp.asarray(W2), onp.asarray(Z2)


COEFFS = [
    ("prospective", dict(nu=4.0 / 3.0, tau=0.75, rho=0.75)),
    ("prospective", dict(nu=0.4, tau=32.0, rho=0.25)),
    ("prospective", dict(nu=3.0, tau=2.0, rho=0.9)),
    ("inertial", dict(eta=1.0, tau=0.75)),
    ("inertial", dict(eta=0.2, tau=32.0)),
    ("inertial", dict(eta=4.0, tau=0.3)),        # complex poles: 1/tau^2 < 4*eta
]


@pytest.mark.parametrize("case", range(12))
def test_the_exact_step_matches_an_independent_dense_ODE(case):
    """Nonzero carries, nonorthogonal key switches, changing gate weights,
    idle (query) intervals, both rules, several coefficient sets. float64."""
    kind, coeff = COEFFS[case % len(COEFFS)]
    rs = onp.random.RandomState(case)
    dv, dk = (3, 4) if case % 2 else (4, 3)
    W = rs.randn(dv, dk); Z = rs.randn(dv, dk)
    worst = 0.0
    for t in range(5):
        k = rs.randn(dk); k /= onp.linalg.norm(k)
        v = rs.randn(dv)
        w = 0.0 if t == 3 else float(2.0 * rs.rand())   # one query interval
        Wr, Zr = _dense(W, Z, k, v, w, kind, coeff)
        Wg, Zg = _step(kind, coeff, W, Z, k, v, w)
        scale = max(1.0, onp.abs(Wr).max(), onp.abs(Zr).max())
        worst = max(worst, onp.abs(Wg - Wr).max() / scale,
                    onp.abs(Zg - Zr).max() / scale)
        W, Z = Wr, Zr
    assert worst < EXACT64, (case, kind, coeff, worst)


@pytest.mark.parametrize("case", range(len(COEFFS)))
def test_the_exact_step_holds_over_a_64_token_float32_trajectory(case):
    """Production dtype, a full episode length. Reference in float64."""
    kind, coeff = COEFFS[case]
    rs = onp.random.RandomState(100 + case)
    dv = dk = 8
    W64 = onp.zeros((dv, dk)); Z64 = onp.zeros((dv, dk))
    W32 = onp.zeros((dv, dk), onp.float32); Z32 = onp.zeros((dv, dk), onp.float32)
    worst = 0.0
    for t in range(64):
        k = rs.randn(dk); k /= onp.linalg.norm(k)
        v = rs.randn(dv)
        w = 0.0 if t % 4 == 3 else float(2.0 * rs.rand())
        W64, Z64 = _dense(W64, Z64, k, v, w, kind, coeff)
        W32, Z32 = _step(kind, coeff, W32, Z32, k, v, w, dt=onp.float32)
        assert W32.dtype == onp.float32 and Z32.dtype == onp.float32
        worst = max(worst, onp.abs(W32 - W64).max()
                    / max(1.0, onp.abs(W64).max()))
    print(f"  {kind} 64-token float32 vs float64 dense: {worst:.3e}")
    assert worst < TRAJ32, (kind, worst)


# ------------------------------- the 2x2 exponential ------------------------
#: R4 witnesses. Ordinary, confluent, oscillatory and STIFF-STABLE generators,
#: plus both sides of the actual float32 series switch. `diag(-200, -1)` is the
#: coordinator's witness: the true entries are e^-200 and e^-1, but the
#: unscaled form forms cosh(99.5), which exceeds the float32 range.
EXPM_CASES = [
    ("ordinary", [[-1.0, 2.0], [0.5, -3.0]]),
    ("confluent", [[-2.0, 1.0], [0.0, -2.0]]),
    ("confluent_exact", [[-2.0, 0.0], [0.0, -2.0]]),
    ("oscillatory", [[-0.5, 1.0], [-4.0, -0.5]]),
    ("oscillatory_fast", [[-0.1, 1.0], [-400.0, -0.1]]),
    ("stiff_diag", [[-200.0, 0.0], [0.0, -1.0]]),
    ("stiff_witness", [[-200.0, 1.0], [0.0, -1.0]]),
    ("stiff_prospective", [[-onp.exp(9.0), -4.0 / 3.0],
                           [-onp.exp(9.0) / 4.0, -4.0 / 3.0]]),
    ("grid_top_prospective", [[-onp.exp(12.0), -4.0 / 3.0],
                              [-onp.exp(12.0) / 4.0, -4.0 / 3.0]]),
]


def _switch_pair(dt):
    """`m` just inside and just outside the declared switch, as a generator.

    `[[0, 1], [m, 0]]` has trace 0 and discriminant/4 exactly `m`, so it puts
    the switch itself under test rather than a nearby proxy.
    """
    sw = AD._switch(dt)
    return [("below_switch", 0.5 * sw), ("at_switch", sw),
            ("above_switch", 2.0 * sw), ("below_switch_neg", -0.5 * sw),
            ("above_switch_neg", -2.0 * sw)]


@pytest.mark.parametrize("name,G", EXPM_CASES)
def test_expm2_matches_scipy_in_float64_with_recorded_error(name, G):
    """Quantitative error against an independent reference, not `isfinite`."""
    G = onp.asarray(G, dtype=onp.float64)
    ref = sp_expm(G)
    got = onp.asarray(AD.expm2(jnp.asarray(G)))
    assert onp.all(onp.isfinite(got)), (name, got)
    err = onp.abs(got - ref).max() / max(1.0, onp.abs(ref).max())
    print(f"  expm2 f64 {name:<22} rel {err:.3e}")
    assert err < EXACT64, (name, err, got, ref)


@pytest.mark.parametrize("name,G", EXPM_CASES)
def test_expm2_does_not_overflow_in_float32(name, G):
    """The coordinator's witness class: a finite answer must not pass through
    an infinite intermediate. Reference computed in float64."""
    G64 = onp.asarray(G, dtype=onp.float64)
    ref = sp_expm(G64)
    got = onp.asarray(AD.expm2(jnp.asarray(G64.astype(onp.float32))))
    assert got.dtype == onp.float32
    assert onp.all(onp.isfinite(got)), (name, "overflow", got)
    err = onp.abs(got - ref).max() / max(1.0, onp.abs(ref).max())
    print(f"  expm2 f32 {name:<22} rel {err:.3e}")
    assert err < TRAJ32, (name, err)


@pytest.mark.parametrize("dt", [onp.float32, onp.float64])
def test_expm2_is_accurate_on_both_sides_of_the_actual_series_switch(dt):
    """The switch is dtype-dependent, so it is probed at its real location in
    each dtype, on both sides and in both branches."""
    for name, m in _switch_pair(dt):
        G = onp.array([[0.0, 1.0], [m, 0.0]], dtype=onp.float64)
        ref = sp_expm(G)
        got = onp.asarray(AD.expm2(jnp.asarray(G.astype(dt))))
        assert got.dtype == dt
        err = onp.abs(got - ref).max() / max(1.0, onp.abs(ref).max())
        tol = EXACT64 if dt is onp.float64 else TRAJ32
        print(f"  switch {onp.dtype(dt).name} {name:<20} m={m:+.3e} "
              f"rel {err:.3e}")
        assert err < tol, (dt, name, m, err)


@pytest.mark.parametrize("dt", [onp.float32, onp.float64])
def test_expm2_derivatives_are_finite_and_correct_across_the_switch(dt):
    """Finiteness alone does not establish accuracy: the derivative is checked
    against central differences of an INDEPENDENT reference on both sides of
    the switch, at the confluent root, and on the complex-pole branch."""
    sw = AD._switch(dt)
    probes = [0.0, 0.5 * sw, sw, 2.0 * sw, -0.5 * sw, -2.0 * sw, 1.0, -4.0,
              4.0]
    fd_h = FD64[0] if dt is onp.float64 else FD32[0]
    tol = GRAD64 if dt is onp.float64 else GRAD32
    for m in probes:
        f = lambda x: jnp.sum(AD.expm2(                       # noqa: E731
            jnp.stack([jnp.stack([jnp.asarray(0.0, dt), jnp.asarray(1.0, dt)]),
                       jnp.stack([x, jnp.asarray(0.0, dt)])])))
        g = float(jax.grad(f)(jnp.asarray(m, dt)))
        assert onp.isfinite(g), (dt, m, g)

        def ref(x):
            return float(sp_expm(onp.array([[0.0, 1.0], [x, 0.0]],
                                           dtype=onp.float64)).sum())
        fd = (ref(m + fd_h) - ref(m - fd_h)) / (2.0 * fd_h)
        rel = abs(g - fd) / max(abs(fd), 1e-3)
        print(f"  d/dm {onp.dtype(dt).name} m={m:+.4e} auto={g:+.6e} "
              f"ref={fd:+.6e} rel={rel:.2e}")
        assert rel < tol, (dt, m, g, fd, rel)


def _shifted_fixture(m):
    """F1. A STABLE, representable family spanning the same discriminants.

        G_m = [[-c, 1], [m, -c]] ,   c = 1 + sqrt(max(m, 0))

    For `m > 0` the eigenvalues are `-1` and `-1 - 2 sqrt(m)`; for `m < 0` they
    are `-1 +- i sqrt(|m|)`; at `m = 0` the root is confluent. So it still
    exercises the positive, negative, confluent and inactive branches at
    extreme discriminants, but its exponential is always representable.

    The previous fixture `[[0, 1], [m, 0]]` at `m = 1e6` has eigenvalues
    +-1000, so `exp(G)` and its derivatives genuinely EXCEED float64 range.
    That is honest overflow in the ACTIVE branch of an unstable matrix, not
    poisoning by an unused one, and no correct implementation can return a
    finite answer there. The assertion, not the implementation, was wrong.

    `c` is computed ONCE from the base `m` and held FIXED while differentiating
    any entry, so the family stays a fixed base matrix rather than a moving
    target that would silently re-stabilize itself under perturbation.
    """
    c = 1.0 + onp.sqrt(max(m, 0.0))
    return c, onp.array([[-c, 1.0], [m, -c]], dtype=onp.float64)


SHIFTED_M = (-1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6)


@pytest.mark.parametrize("m", SHIFTED_M)
def test_expm2_on_the_stable_shifted_family_matches_scipy(m):
    """Quantitative accuracy on the same fixed-base matrix used below."""
    _, G = _shifted_fixture(m)
    ref = sp_expm(G)
    assert onp.all(onp.isfinite(ref)), (m, "the fixture must be representable")
    got = onp.asarray(AD.expm2(jnp.asarray(G)))
    err = onp.abs(got - ref).max() / max(1.0, onp.abs(ref).max())
    print(f"  shifted m={m:+.1e} rel {err:.3e}")
    assert err < EXACT64, (m, err)


@pytest.mark.parametrize("m", SHIFTED_M)
def test_expm2_has_no_nan_from_an_inactive_branch(m):
    """Every branch is evaluated on where-guarded SAFE inputs, so an inactive
    branch must not poison the backward pass through a zero cotangent.

    Differentiated entry by entry about the FIXED shifted base, whose true
    derivatives are representable — so a non-finite result here is a real
    defect and not an unrepresentable answer.
    """
    c, G0 = _shifted_fixture(m)
    base = [-c, 1.0, m, -c]
    for entry in range(4):
        def f(x):
            g = [jnp.asarray(v) for v in base]
            g[entry] = x                      # c stays FIXED, by construction
            return jnp.sum(AD.expm2(jnp.stack([jnp.stack([g[0], g[1]]),
                                               jnp.stack([g[2], g[3]])])))
        v = float(jax.grad(f)(jnp.asarray(base[entry])))
        assert onp.isfinite(v), (m, entry, v)
        # and it is the RIGHT derivative, against the independent reference
        E = onp.zeros((2, 2)); E[entry // 2, entry % 2] = 1.0
        h = FD64[1] * max(1.0, abs(base[entry]))
        fd = (sp_expm(G0 + h * E).sum() - sp_expm(G0 - h * E).sum()) / (2 * h)
        assert onp.isfinite(fd), (m, entry)
        if abs(fd) < 1e-6:
            assert abs(v - fd) < NEAR64, (m, entry, v, fd)
        else:
            rel = abs(v - fd) / abs(fd)
            print(f"  d/dg{entry} m={m:+.1e} auto={v:+.6e} ref={fd:+.6e} "
                  f"rel={rel:.2e}")
            assert rel < GRAD64, (m, entry, v, fd, rel)


# ------------------------- reduction to the completed study ----------------
def test_gate_one_reproduces_the_completed_prospective_trajectory():
    """nu = 4/3, tau = 3/4, rho = 3/4 with a = 1 is the completed study's law,
    in the coordinates Z = -P/gamma."""
    c = ND.prospective_constants()                       # M=3/4, gamma=T=h=1
    coeff = dict(nu=ND.T_REF / ND.M_REF, tau=ND.M_REF / ND.GAMMA_REF,
                 rho=ND.M_REF / (ND.GAMMA_REF * ND.T_REF))
    assert abs(coeff["nu"] - 4 / 3) < IDENT64 and abs(coeff["rho"] - .75) < IDENT64
    rs = onp.random.RandomState(11)
    dv = dk = 6
    Wo = rs.randn(dv, dk); Po = rs.randn(dv, dk)
    Wn, Zn = Wo.copy(), -Po / ND.GAMMA_REF
    worst = 0.0
    for t in range(24):
        k = rs.randn(dk); k /= onp.linalg.norm(k)
        v = rs.randn(dv); m = float(t % 5 != 4)          # jumps AND idle motion
        Wo2, Po2 = ND.two_state_step(
            (jnp.asarray(Wo), jnp.asarray(Po)), jnp.asarray(k), jnp.asarray(v),
            jnp.asarray(m), jnp.asarray(c["F"]), c["a0"], c["b0"])
        Wo, Po = onp.asarray(Wo2), onp.asarray(Po2)
        Wn, Zn = _step("prospective", coeff, Wn, Zn, k, v, m)
        worst = max(worst, onp.abs(Wn - Wo).max(),
                    onp.abs(Zn + Po / ND.GAMMA_REF).max())
    print(f"  prospective reproduction: {worst:.3e}")
    assert worst < REPRO64, worst


def test_gate_one_reproduces_the_completed_inertial_trajectory():
    """eta = 1/gamma, tau = M/gamma with a = 1 is the completed inertial
    ablation, again under Z = -P/gamma (there P = M Wdot)."""
    c = ND.inertial_constants()
    coeff = dict(eta=1.0 / ND.GAMMA_REF, tau=ND.M_REF / ND.GAMMA_REF)
    rs = onp.random.RandomState(12)
    dv = dk = 6
    Wo = rs.randn(dv, dk); Po = rs.randn(dv, dk)
    Wn, Zn = Wo.copy(), -Po / ND.GAMMA_REF
    worst = 0.0
    for t in range(24):
        k = rs.randn(dk); k /= onp.linalg.norm(k)
        v = rs.randn(dv); m = float(t % 5 != 4)
        Wo2, Po2 = ND.two_state_step(
            (jnp.asarray(Wo), jnp.asarray(Po)), jnp.asarray(k), jnp.asarray(v),
            jnp.asarray(m), jnp.asarray(c["F"]), c["a0"], c["b0"])
        Wo, Po = onp.asarray(Wo2), onp.asarray(Po2)
        Wn, Zn = _step("inertial", coeff, Wn, Zn, k, v, m)
        worst = max(worst, onp.abs(Wn - Wo).max(),
                    onp.abs(Zn + Po / ND.GAMMA_REF).max())
    print(f"  inertial reproduction: {worst:.3e}")
    assert worst < REPRO64, worst


def test_the_rho_one_boundary_is_the_first_order_delta_rule():
    """An EXPLICIT boundary evaluation at rho = 1 with Z0 = 0, not an
    unreachable finite sigmoid parameter. Source weights change each step."""
    rs = onp.random.RandomState(13)
    dv = dk = 5
    for nu in (0.3, 4.0 / 3.0, 5.0):
        for tau in (0.75, 32.0):
            coeff = dict(nu=nu, tau=tau, rho=1.0)
            W = rs.randn(dv, dk); Wd = W.copy(); Z = onp.zeros((dv, dk))
            worst = 0.0
            for t in range(10):
                k = rs.randn(dk); k /= onp.linalg.norm(k)
                v = rs.randn(dv)
                w = 0.0 if t == 4 else float(2.0 * rs.rand())
                W, Z = _step("prospective", coeff, W, Z, k, v, w)
                beta = -onp.expm1(-AD.H * nu * w)
                Wd = onp.asarray(AD.delta_step(
                    jnp.asarray(Wd), jnp.asarray(k), jnp.asarray(v),
                    jnp.asarray(beta)))
                worst = max(worst, onp.abs(W - Wd).max(), onp.abs(Z).max())
            assert worst < IDENT64, (nu, tau, worst)


def test_the_delta_arm_uses_the_gradient_flow_beta_and_freezes_when_idle():
    for eta in (0.5, 1.0, 3.0):
        for w in (0.0, 0.25, 1.0, 2.0):
            b = -onp.expm1(-AD.H * eta * w)
            assert abs(b - (1.0 - onp.exp(-eta * w))) < IDENT64
        assert abs(-onp.expm1(-AD.H * eta * 0.0)) < IDENT64


# ---------------------------------- derivatives ----------------------------
def _probe(kind, raw, dt=onp.float64, n=6, seed=21, W0=None, Z0=None):
    """A nondegenerate differentiable scalar of a short trajectory.

    The initial carry is FIXED and parameter-independent (see the module
    docstring's initial-condition convention).
    """
    rs = onp.random.RandomState(seed)
    dv = dk = 4
    ks = []
    for _ in range(n):
        x = rs.randn(dk); ks.append(x / onp.linalg.norm(x))
    vs = [rs.randn(dv) for _ in range(n)]
    ws = [0.0 if t == 2 else float(0.3 + 1.4 * rs.rand()) for t in range(n)]
    qq = rs.randn(dk); q = jnp.asarray(qq / onp.linalg.norm(qq), dt)
    proj = jnp.asarray(rs.randn(dv), dt)
    W0 = jnp.zeros((dv, dk), dt) if W0 is None else jnp.asarray(W0, dt)
    Z0 = jnp.zeros((dv, dk), dt) if Z0 is None else jnp.asarray(Z0, dt)

    def f(r):
        if kind == "prospective":
            nu, tau = jnp.exp(r[0]), jnp.exp(r[1])
            rho = jax.nn.sigmoid(r[2])
            G = lambda w: AD.prospective_generator(nu, tau, rho, w)  # noqa: E731
        else:
            eta, tau = jnp.exp(r[0]), jnp.exp(r[1])
            G = lambda w: AD.inertial_generator(eta, tau, w)         # noqa: E731
        a0 = jnp.exp(-AD.H / tau)
        W, Z = W0, Z0
        acc = jnp.zeros((), dt)
        for t in range(n):
            wt = jnp.asarray(ws[t], dt) * (1.0 + 0.1 * r[-1])
            F = AD.expm2(G(wt)).astype(dt)
            W, Z = AD.two_state_step(W, Z, jnp.asarray(ks[t], dt),
                                     jnp.asarray(vs[t], dt), F, a0.astype(dt))
            acc = acc + jnp.dot(proj, W @ q)
        return acc
    return f


RAWS = {"prospective": onp.array([onp.log(4 / 3), onp.log(0.75),
                                  onp.log(3.0), 0.0]),
        "inertial": onp.array([0.0, onp.log(0.75), 0.0])}


def _directional(f, r, d, h):
    return (f(r + h * d) - f(r - h * d)) / (2.0 * h)


@pytest.mark.parametrize("kind", ["prospective", "inertial"])
@pytest.mark.parametrize("di", range(4))
def test_float64_jvp_matches_central_differences(kind, di):
    """The principal derivative check: automatic differentiation against the
    exact trajectory, compared with central differences at BOTH declared
    float64 steps."""
    r0 = RAWS[kind]
    rs = onp.random.RandomState(300 + di)
    d = rs.randn(r0.size); d /= onp.linalg.norm(d)
    f = _probe(kind, r0)
    jv = float(jax.jvp(lambda x: f(x), (jnp.asarray(r0),),
                       (jnp.asarray(d),))[1])
    assert onp.isfinite(jv)
    for h in FD64:
        fd = float(_directional(lambda x: f(jnp.asarray(x)), r0, d, h))
        if abs(fd) < 1e-6:
            assert abs(jv - fd) < NEAR64, (kind, di, h, jv, fd)
        else:
            rel = abs(jv - fd) / abs(fd)
            assert rel < GRAD64, (kind, di, h, jv, fd, rel)


@pytest.mark.parametrize("kind", ["prospective", "inertial"])
def test_float32_jvp_at_both_declared_perturbations(kind):
    """Production float32 has its OWN documented finite-difference floor and
    BOTH declared steps are enforced separately. This is not a float64 label
    on a float32 path."""
    r0 = RAWS[kind].astype(onp.float32)
    rs = onp.random.RandomState(400)
    d = rs.randn(r0.size).astype(onp.float32); d /= onp.linalg.norm(d)
    f = _probe(kind, r0, dt=onp.float32)
    jv = float(jax.jvp(lambda x: f(x), (jnp.asarray(r0, onp.float32),),
                       (jnp.asarray(d, onp.float32),))[1])
    worst = []
    for h in FD32:
        fd = float(_directional(lambda x: f(jnp.asarray(x, onp.float32)),
                                r0, d, onp.float32(h)))
        worst.append((h, fd, abs(jv - fd),
                      abs(jv - fd) / abs(fd) if abs(fd) > 1e-4 else None))
    print(f"  float32 {kind}: jvp={jv:.6g} " +
          " ".join(f"h={h:g} fd={fd:.6g} abs={a:.3g}"
                   for h, fd, a, _ in worst))
    for h, fd, a, rel in worst:
        assert a < NEAR32 or (rel is not None and rel < GRAD32), \
            (kind, h, jv, fd, a, rel)


@pytest.mark.parametrize("rule", list(AM.NEW_RULES))
def test_gradients_reach_the_response_gate_and_embedding_leaves(rule):
    """A real optimizer step, on a nondegenerate fixture: finite raw gradients
    and nonzero updates on every response leaf, both gate leaves and the
    embeddings. No clipping lockout and no host conversion of a learned leaf."""
    import optax
    slot = CAL.configurations()["slots"][f"{rule}/A"]
    p = AM.init_params(rule, 77, init_coeffs=slot, dtype=onp.float64)
    b = TK.generate_batch(4321, 2)
    ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event")}
    lab = jnp.maximum(jnp.asarray(b["label"][0]), 0)
    qm = jnp.asarray(b["event"][0] == TK.QUERY)

    def loss(q):
        o = AM.rollout(rule, q, ep)
        ce = optax.softmax_cross_entropy(
            o["logits"], jax.nn.one_hot(lab, TK.N_VALUES, dtype=o["logits"].dtype))
        return jnp.sum(ce * qm) / jnp.sum(qm)

    g = jax.grad(loss)(p)
    must = ["key_raw", "value_table", "readout_W"]
    must += [k for k in p if k.startswith("raw_") or k.startswith("gate_")]
    if rule == "ideal_projection":
        # deliberately no gate and no response leaf: a positive scalar source
        # weight changes neither the constraint nor the update, so gate
        # parameters here would have identically zero gradients
        assert not any(k.startswith("gate_") or k.startswith("raw_")
                       for k in p), rule
    for k in must:
        gk = onp.asarray(g[k])
        assert onp.all(onp.isfinite(gk)), (rule, k)
        assert onp.abs(gk).max() > 0.0, (rule, k, "zero gradient")
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adam(3e-3, b1=.9, b2=.999, eps=1e-8))
    st = tx.init(p)
    upd, _ = tx.update(g, st, p)
    p2 = optax.apply_updates(p, upd)
    for k in must:
        assert onp.abs(onp.asarray(p2[k]) - onp.asarray(p[k])).max() > 0.0, \
            (rule, k, "update did not move the leaf")
        assert isinstance(p2[k], jax.Array), (rule, k, "host conversion")


# ------------------------------- storage inequality ------------------------
def test_the_incremental_storage_inequality_holds_for_the_prospective_law():
    """Identical exogenous weighted inputs, FIXED prospective parameters, two
    initial states. V is non-increasing across exact steps.

    Declared scope: this is the analytic result for the prospective law only.
    It is NOT applied to the inertial control, and not to coefficients that
    change inside an episode.
    """
    rs = onp.random.RandomState(31)
    dv = dk = 5
    for nu, tau, rho in [(4 / 3, .75, .75), (0.4, 32.0, .25), (3.0, 2.0, .9)]:
        eta, kappa = nu * rho, tau * nu * (1.0 - rho)
        coeff = dict(nu=nu, tau=tau, rho=rho)
        W1, Z1 = rs.randn(dv, dk), rs.randn(dv, dk)
        W2, Z2 = rs.randn(dv, dk), rs.randn(dv, dk)
        V = lambda W1, Z1, W2, Z2: (                              # noqa: E731
            onp.sum((W1 - Z1 - (W2 - Z2)) ** 2) / (2 * eta)
            + tau * onp.sum((Z1 - Z2) ** 2) / (2 * kappa))
        prev = V(W1, Z1, W2, Z2)
        assert prev > 0
        for t in range(20):
            k = rs.randn(dk); k /= onp.linalg.norm(k)
            v = rs.randn(dv)
            w = 0.0 if t % 6 == 5 else float(2.0 * rs.rand())
            W1, Z1 = _step("prospective", coeff, W1, Z1, k, v, w)
            W2, Z2 = _step("prospective", coeff, W2, Z2, k, v, w)
            cur = V(W1, Z1, W2, Z2)
            assert cur <= prev + 1e-12, (nu, tau, rho, t, prev, cur)
            prev = cur


# ----------------------------------- the shell -----------------------------
def _ep(batch, i):
    return {k: jnp.asarray(batch[k][i]) for k in ("key_id", "val_id", "event")}


@pytest.mark.parametrize("rule", list(AD.RULES))
def test_streaming_matches_the_unsplit_sequence(rule):
    """Chunked carry against one pass, and a zero/reset carry."""
    slot = CAL.configurations()["slots"].get(f"{rule}/A", {})
    p = AM.init_params(rule, 55, init_coeffs=slot)
    b = TK.generate_batch(4400, 1)
    ep = _ep(b, 0)
    full = AM.rollout(rule, p, ep)
    L = ep["event"].shape[0]
    carry, parts = None, []
    for lo in range(0, L, 16):
        sub = {k: v[lo:lo + 16] for k, v in ep.items()}
        o = AM.rollout(rule, p, sub, carry0=carry)
        carry = o["final_carry"]
        parts.append(onp.asarray(o["logits"]))
    got = onp.concatenate(parts)
    ref = onp.asarray(full["logits"])
    assert onp.abs(ref).max() > 1e-3, "an all-zero fixture proves nothing"
    assert onp.abs(got - ref).max() < STREAM32, (rule,
                                                 onp.abs(got - ref).max())


@pytest.mark.parametrize("rule", list(AD.RULES))
def test_batched_equals_per_example(rule):
    slot = CAL.configurations()["slots"].get(f"{rule}/A", {})
    p = AM.init_params(rule, 56, init_coeffs=slot)
    b = TK.generate_batch(4401, 2)
    eps = {k: jnp.asarray(b[k]) for k in ("key_id", "val_id", "event")}
    batched = jax.vmap(lambda e: AM.rollout(rule, p, e)["logits"])(eps)
    for i in range(eps["event"].shape[0]):
        one = AM.rollout(rule, p, _ep(b, i))["logits"]
        assert onp.abs(onp.asarray(batched[i]) - onp.asarray(one)).max() \
            < STREAM32, (rule, i)


@pytest.mark.parametrize("rule", list(AD.RULES))
def test_a_later_token_cannot_change_an_earlier_logit(rule):
    slot = CAL.configurations()["slots"].get(f"{rule}/A", {})
    p = AM.init_params(rule, 57, init_coeffs=slot)
    b = TK.generate_batch(4402, 1)
    ep = _ep(b, 0)
    L = ep["event"].shape[0]
    cut = L // 2
    ref = onp.asarray(AM.rollout(rule, p, ep)["logits"])
    alt = dict(ep)
    kk = onp.asarray(ep["key_id"]).copy(); vv = onp.asarray(ep["val_id"]).copy()
    kk[cut:] = (kk[cut:] + 7) % TK.N_KEYS
    vv[cut:] = onp.where(vv[cut:] >= 0, (vv[cut:] + 3) % TK.N_VALUES, vv[cut:])
    alt["key_id"] = jnp.asarray(kk); alt["val_id"] = jnp.asarray(vv)
    got = onp.asarray(AM.rollout(rule, p, alt)["logits"])
    assert onp.abs(got[:cut] - ref[:cut]).max() < 1e-6, rule
    assert onp.abs(got[cut:] - ref[cut:]).max() > 1e-4, (rule, "inert fixture")


@pytest.mark.parametrize("rule", list(AD.RULES))
@pytest.mark.parametrize("dt", [onp.float32, onp.float64])
def test_the_executed_dtype_follows_the_parameters(rule, dt):
    slot = CAL.configurations()["slots"].get(f"{rule}/A", {})
    p = AM.init_params(rule, 58, init_coeffs=slot, dtype=dt)
    assert p["key_raw"].dtype == dt, "x64 must be enabled for the f64 route"
    out = AM.rollout(rule, p, _ep(TK.generate_batch(4403, 1), 0))
    assert out["logits"].dtype == dt
    for c in out["final_carry"]:
        assert c.dtype == dt, (rule, dt, c.dtype)


def test_every_parameter_leaf_is_explicitly_typed():
    """A weakly typed leaf retraces a step that consumes its own output."""
    cfg = CAL.configurations()["slots"]
    for rule in AD.RULES:
        p = AM.init_params(rule, 59, init_coeffs=cfg.get(f"{rule}/A", {}))
        weak = [k for k, v in p.items() if jnp.asarray(v).weak_type]
        assert not weak, (rule, weak)


def test_the_generator_supplies_no_query_value_on_valid_episodes():
    """The valid input has no query value field at all. This is the no-leakage
    property; it is a statement about the DATA, checked before any arm runs."""
    for seed in (4404, 4408, 4409):
        b = TK.generate_batch(seed, 2)
        q = (b["event"] == TK.QUERY)
        assert q.any()
        assert onp.all(b["val_id"][q] < 0), "a query carries no value field"


def test_the_input_contract_is_the_identity_on_valid_episodes():
    """R3. `sanitize_episode` masks the value field to writes. On valid
    episodes it changes nothing, for every arm - so it cannot alter the pinned
    literature recurrence."""
    cfg = CAL.configurations()["slots"]
    b = TK.generate_batch(4404, 1)
    ep = _ep(b, 0)
    san = AM.sanitize_episode(ep)
    assert onp.array_equal(onp.asarray(san["val_id"]),
                           onp.asarray(ep["val_id"])), "identity on valid data"
    for rule in AD.RULES:
        p = AM.init_params(rule, 60, init_coeffs=cfg.get(f"{rule}/A", {}))
        a = onp.asarray(AM.rollout(rule, p, ep)["logits"])
        c = onp.asarray(AM.rollout(rule, p, san)["logits"])
        assert onp.array_equal(a, c), rule


def test_an_invented_query_value_cannot_reach_any_arm():
    """R3. With the contract applied at the common boundary, an invented query
    value is invariant for ALL seven arms - including the two literature arms,
    whose forgetting/momentum gates read the value id and are NOT masked by
    event inside their own pinned code. The earlier revision asserted this
    invariance WITHOUT the contract, which the literature rule cannot satisfy
    and which does not follow from `read only Wq`. The literature recurrence
    was not changed to satisfy the test; the input contract was specified.
    """
    cfg = CAL.configurations()["slots"]
    b = TK.generate_batch(4404, 1)
    ep = _ep(b, 0)
    qpos = onp.where(onp.asarray(ep["event"]) == TK.QUERY)[0]
    assert qpos.size > 0
    for invented in (0, 3, 7):
        alt = dict(ep)
        vv = onp.asarray(ep["val_id"]).copy()
        vv[qpos] = invented
        alt["val_id"] = jnp.asarray(vv)
        for rule in AD.RULES:
            p = AM.init_params(rule, 60, init_coeffs=cfg.get(f"{rule}/A", {}))
            a = onp.asarray(AM.rollout(rule, p, ep)["logits"])
            c = onp.asarray(AM.rollout(rule, p, alt)["logits"])
            assert onp.abs(a - c).max() < 1e-7, (rule, invented)


def test_no_arm_reads_the_label_or_any_oracle_field():
    """The rollout must be invariant to the separately stored oracle fields.
    Scrambling `label`, `category` and `age` cannot move a logit."""
    cfg = CAL.configurations()["slots"]
    b = TK.generate_batch(4410, 1)
    ep = _ep(b, 0)
    rs = onp.random.RandomState(0)
    poisoned = dict(ep)
    for fld in ("label", "category", "age"):
        poisoned[fld] = jnp.asarray(rs.randint(0, 8, size=b[fld][0].shape))
    for rule in AD.RULES:
        p = AM.init_params(rule, 60, init_coeffs=cfg.get(f"{rule}/A", {}))
        a = onp.asarray(AM.rollout(rule, p, ep)["logits"])
        c = onp.asarray(AM.rollout(rule, p, poisoned)["logits"])
        assert onp.array_equal(a, c), rule


# ------------------------------- gate and counts ---------------------------
def test_the_contrast_matrices_are_orthonormal_and_zero_sum():
    for H, n in ((AM.H32, TK.N_KEYS), (AM.H8, TK.N_VALUES)):
        r = AG.check_contrast(H)
        assert r["shape"] == [n, n - 1] and r["passed"], r
    # the declared deterministic convention, column by column
    for n in (8, 32):
        H = AG.contrast_matrix(n)
        for j in range(n - 1):
            c = 1.0 / onp.sqrt((j + 1.0) * (j + 2.0))
            assert onp.allclose(H[:j + 1, j], c)
            assert abs(H[j + 1, j] + (j + 1.0) * c) < 1e-12
            assert onp.allclose(H[j + 2:, j], 0.0)


def test_the_gate_starts_at_exactly_one_and_stays_strictly_positive():
    cfg = CAL.configurations()["slots"]
    for rule in AM.GATED_RULES:
        p = AM.init_params(rule, 61, init_coeffs=cfg[f"{rule}/A"],
                           dtype=onp.float64)
        a = onp.asarray(AG.source_weight_table(p, jnp.asarray(AM.H32),
                                               jnp.asarray(AM.H8)))
        assert a.shape == (TK.N_KEYS, TK.N_VALUES)
        assert onp.abs(a - 1.0).max() < 1e-12
        p2 = dict(p, gate_u=jnp.asarray(onp.random.RandomState(1).randn(31)),
                  gate_b=jnp.asarray(onp.random.RandomState(2).randn(7)))
        a2 = onp.asarray(AG.source_weight_table(p2, jnp.asarray(AM.H32),
                                                jnp.asarray(AM.H8)))
        assert a2.min() > 0.0 and a2.max() < 2.0


def test_parameter_and_state_counts_are_the_declared_ones():
    expected = {"adaptive_prospective": 433, "adaptive_inertial": 432,
                "adaptive_delta": 431, "tss_prospective": 432,
                "ideal_projection": 392, "gated_delta": 480,
                "momentum_delta": 569}
    cfg = CAL.configurations()["slots"]
    for rule in AD.RULES:
        p = AM.init_params(rule, 62, init_coeffs=cfg.get(f"{rule}/A", {}))
        c = AM.parameter_counts(rule, p)
        assert c["common"] == 392, (rule, c["common"])
        assert c["total"] == expected[rule], (rule, c["total"],
                                              expected[rule])
        assert c["carry_real_numbers"] == AD.CARRY[rule]
    assert AD.CARRY["adaptive_prospective"] == AD.CARRY["momentum_delta"] == 128


def test_the_common_tensors_are_bit_identical_across_arms_at_a_seed():
    cfg = CAL.configurations()["slots"]
    ref = AM.init_params("adaptive_prospective", 63,
                         init_coeffs=cfg["adaptive_prospective/A"])
    for rule in AD.RULES:
        p = AM.init_params(rule, 63, init_coeffs=cfg.get(f"{rule}/A", {}))
        for k in ("key_raw", "value_table", "readout_W", "readout_b"):
            assert onp.array_equal(onp.asarray(p[k]), onp.asarray(ref[k])), \
                (rule, k)


# --------------------------------- calibration -----------------------------
def test_calibration_matches_the_declared_observable_and_recovers_nu():
    cfg = CAL.configurations()
    bs = cfg["beta_star"]
    assert 0.0 < bs < 1.0
    assert cfg["reference_recovery"]["passed"], cfg["reference_recovery"]
    assert cfg["reference_recovery"]["absolute_error"] < NU_TOL
    for rec in cfg["records"]:
        assert rec["observable_error"] <= CAL_TOL, rec
        assert rec["bracket"][0] <= rec["rate"] <= rec["bracket"][1] or \
            rec["exact_grid_point"], rec
    for key, s in cfg["slots"].items():
        if s["rule"] == "adaptive_prospective":
            a = CAL.amplitude(CAL.prospective_G(s["nu"], s["tau"], s["rho"]),
                              s["tau"])
        elif s["rule"] == "adaptive_inertial":
            a = CAL.amplitude(CAL.inertial_G(s["eta"], s["tau"]), s["tau"])
        elif s["rule"] == "adaptive_delta":
            a = 1.0 - onp.exp(-s["eta"])
        else:
            continue
        assert abs(a - bs) <= 1e-9, (key, a, bs)
    assert len(cfg["slots"]) == 14


def test_the_executed_generator_reproduces_the_calibration_observable():
    """The calibration solves on a host float64 routine; the EXECUTED
    differentiable path must give the same single-write query amplitude."""
    cfg = CAL.configurations()
    for key, s in cfg["slots"].items():
        if s["rule"] == "adaptive_prospective":
            G = AD.prospective_generator(jnp.float64(s["nu"]),
                                         jnp.float64(s["tau"]),
                                         jnp.float64(s["rho"]),
                                         jnp.float64(1.0))
        elif s["rule"] == "adaptive_inertial":
            G = AD.inertial_generator(jnp.float64(s["eta"]),
                                      jnp.float64(s["tau"]), jnp.float64(1.0))
        else:
            continue
        a = float(AD.single_write_query_amplitude(AD.expm2(G),
                                                  jnp.float64(s["tau"])))
        assert abs(a - cfg["beta_star"]) < 1e-9, (key, a)


def test_the_raw_initializers_decode_to_the_calibrated_coefficients():
    cfg = CAL.configurations()["slots"]
    for tag in ("A", "B"):
        s = cfg[f"adaptive_prospective/{tag}"]
        p = AM.init_params("adaptive_prospective", 64, init_coeffs=s,
                           dtype=onp.float64)
        c = AD.response(p)
        for n in ("nu", "tau", "rho"):
            assert abs(float(c[n][0]) - s[n]) < 1e-12, (tag, n)
        M, gam, T = float(c["M"][0]), float(c["gamma"][0]), float(c["T"][0])
        assert abs(M - s["tau"] / (s["nu"] * s["rho"])) < 1e-12
        assert abs(gam - 1.0 / (s["nu"] * s["rho"])) < 1e-12
        assert abs(T - s["tau"] / s["rho"]) < 1e-12
        assert 0.0 < M < gam * T, (tag, M, gam * T)   # strict admissibility
        s2 = cfg[f"adaptive_inertial/{tag}"]
        p2 = AM.init_params("adaptive_inertial", 64, init_coeffs=s2,
                            dtype=onp.float64)
        c2 = AD.inertial_response(p2)
        assert abs(float(c2["eta"][0]) - s2["eta"]) < 1e-12
        assert abs(float(c2["tau"][0]) - s2["tau"]) < 1e-12


def test_the_declared_streams_are_pairwise_disjoint():
    from experiments.adaptive_memory import study as ST
    seen = {}
    for seed in (ST.DEV_SEED, *ST.FINAL_SEEDS):
        for u in range(ST.UPDATES):
            s = ST.train_stream_seed(seed, u)
            assert s not in seen, (s, seen.get(s), (seed, u))
            seen[s] = (seed, u)
    for name, base in ST.STREAM.items():
        if name == "train":
            continue
        assert base not in seen, (name, base)
    # the completed study's logged entry points
    for prior in (0, 1_000_000, 7_000_000, 9_000_000):
        assert prior not in seen and prior not in ST.STREAM.values()


# -------------------------------- checkpointing ----------------------------
@pytest.mark.parametrize("rule", list(AD.RULES))
def test_checkpoint_restore_reproduces_the_logits(rule, tmp_path):
    from flax import serialization
    cfg = CAL.configurations()["slots"]
    p = AM.init_params(rule, 65, init_coeffs=cfg.get(f"{rule}/A", {}))
    ep = _ep(TK.generate_batch(4405, 1), 0)
    ref = onp.asarray(AM.rollout(rule, p, ep)["logits"])
    path = tmp_path / "p.msgpack"
    path.write_bytes(serialization.to_bytes(
        jax.tree_util.tree_map(lambda v: onp.asarray(v), p)))
    back = serialization.from_bytes(
        jax.tree_util.tree_map(lambda v: onp.asarray(v), p),
        path.read_bytes())
    back = {k: jnp.asarray(v) for k, v in back.items()}
    got = onp.asarray(AM.rollout(rule, back, ep)["logits"])
    assert onp.array_equal(got, ref), rule


def test_evaluation_parity_across_chunk_sizes():
    """The reported metric must not depend on the evaluation chunking."""
    from experiments.adaptive_memory import study as ST
    cfg = CAL.configurations()["slots"]
    b = TK.generate_batch(4406, 8)
    rule = "adaptive_prospective"
    p = AM.init_params(rule, 66, init_coeffs=cfg[f"{rule}/A"])
    a = ST.evaluate(rule, p, b, chunk=16)
    c = ST.evaluate(rule, p, b, chunk=4)
    for key in ("primary", "recall_overall", "retention_revision_untouched",
                "revision_ce"):
        assert abs(a[key] - c[key]) < 1e-6, (key, a[key], c[key])


def test_the_literature_arms_are_the_completed_study_unchanged():
    """Gated and Momentum DeltaNet must be the source-pinned implementations,
    not re-derived here: identical parameters and identical logits."""
    for rule in ("gated_delta", "momentum_delta"):
        a = AM.init_params(rule, 67)
        b = NM.init_params(rule, 67)
        assert set(a) == set(b)
        for k in a:
            assert onp.array_equal(onp.asarray(a[k]), onp.asarray(b[k])), (rule, k)
        ep = _ep(TK.generate_batch(4407, 1), 0)
        ga = onp.asarray(AM.rollout(rule, a, ep)["logits"])
        gb = onp.asarray(NM.rollout(rule, b, ep, NM.constants_for(rule))["logits"])
        assert onp.array_equal(ga, gb), rule


# =========================================================================
#  Ordinary-prospective references (16 September 2026 amendment)
# =========================================================================
TSS_COEFFS = [dict(tau_m=1.0, epsilon=0.1), dict(tau_m=1.0, epsilon=0.5),
              dict(tau_m=4.0, epsilon=0.4), dict(tau_m=0.3, epsilon=0.03)]


def _tss_dense_WA(W, A, k, v, w, tau_m, eps, h=1.0):
    """Integrate the ORIGINAL (W, A) TSS equations, Eqs. (6)-(7), directly.

        eps Adot   = -A + f,   f = W - R
        tau_m Wdot = -W + (1 + tau_m/eps) f - (tau_m/eps) A
        R          = w (W k - v) k^T

    i.e.  Wdot = (W - A)/eps - (1/tau_m + 1/eps) R,
          Adot = (W - A - R)/eps.
    Assembled and exponentiated independently of the eliminated (W, P) step.
    """
    dv, dk = W.shape
    n = dv * dk
    K = onp.outer(k, k)
    WK = onp.kron(K.T, onp.eye(dv))
    vk = onp.reshape(onp.outer(v, k), -1, order="F")
    I = onp.eye(n)
    c = 1.0 / tau_m + 1.0 / eps
    M = onp.zeros((2 * n + 1, 2 * n + 1))
    M[:n, :n] = I / eps - c * w * WK
    M[:n, n:2 * n] = -I / eps
    M[:n, 2 * n] = c * w * vk
    M[n:2 * n, :n] = I / eps - (w / eps) * WK
    M[n:2 * n, n:2 * n] = -I / eps
    M[n:2 * n, 2 * n] = (w / eps) * vk
    z = onp.concatenate([onp.reshape(W, -1, order="F"),
                         onp.reshape(A, -1, order="F"), [1.0]])
    out = sp_expm(M * h) @ z
    return (onp.reshape(out[:n], (dv, dk), order="F"),
            onp.reshape(out[n:2 * n], (dv, dk), order="F"))


def _tss_step(c, W, P, k, v, w, dt=onp.float64):
    """R5: coefficients, generator, exponential, b0 and carries all in `dt`."""
    tau_m = jnp.asarray(c["tau_m"], dt); eps = jnp.asarray(c["epsilon"], dt)
    M, T = tau_m * eps, tau_m + eps
    G = AD.tss_generator(M, T, jnp.asarray(float(w), dt))
    F = AD.expm2(G)
    b0 = jnp.asarray(AD.H, dt) / M
    assert G.dtype == dt and F.dtype == dt and b0.dtype == dt, (dt, F.dtype)
    W2, P2 = AD.tss_two_state_step(jnp.asarray(W, dt), jnp.asarray(P, dt),
                                   jnp.asarray(k, dt), jnp.asarray(v, dt),
                                   F, b0)
    assert W2.dtype == dt and P2.dtype == dt, (dt, W2.dtype)
    return onp.asarray(W2), onp.asarray(P2)


@pytest.mark.parametrize("case", range(8))
def test_the_TSS_step_matches_direct_integration_of_the_original_equations(case):
    """The eliminated (W, P) token step against a dense affine exponential of
    the ORIGINAL (W, A) equations, over multi-token sequences with changing
    gates, keys, values and masks, from NONZERO initial states.

    The initial mapping P0 = tau_m (W0 - A0) is applied explicitly; it is the
    same mapping that must be differentiated when comparing parameter
    gradients across these two coordinate systems.
    """
    c = TSS_COEFFS[case % len(TSS_COEFFS)]
    tau_m, eps = c["tau_m"], c["epsilon"]
    rs = onp.random.RandomState(500 + case)
    dv, dk = (3, 4) if case % 2 else (4, 3)
    W = rs.randn(dv, dk); A = rs.randn(dv, dk)
    P = tau_m * (W - A)
    worst = 0.0
    for t in range(6):
        k = rs.randn(dk); k /= onp.linalg.norm(k)
        v = rs.randn(dv)
        w = 0.0 if t == 3 else float(2.0 * rs.rand())   # one query interval
        Wr, Ar = _tss_dense_WA(W, A, k, v, w, tau_m, eps)
        Wg, Pg = _tss_step(c, W, P, k, v, w)
        Pr = tau_m * (Wr - Ar)
        scale = max(1.0, onp.abs(Wr).max(), onp.abs(Pr).max())
        worst = max(worst, onp.abs(Wg - Wr).max() / scale,
                    onp.abs(Pg - Pr).max() / scale)
        W, A, P = Wr, Ar, Pr
    assert worst < EXACT64, (case, c, worst)


@pytest.mark.parametrize("case", range(len(TSS_COEFFS)))
def test_the_TSS_step_holds_over_a_64_token_float32_trajectory(case):
    c = TSS_COEFFS[case]
    tau_m, eps = c["tau_m"], c["epsilon"]
    rs = onp.random.RandomState(600 + case)
    dv = dk = 8
    W64 = onp.zeros((dv, dk)); A64 = onp.zeros((dv, dk))
    W32 = onp.zeros((dv, dk), onp.float32); P32 = onp.zeros((dv, dk), onp.float32)
    worst = 0.0
    for t in range(64):
        k = rs.randn(dk); k /= onp.linalg.norm(k)
        v = rs.randn(dv)
        w = 0.0 if t % 4 == 3 else float(2.0 * rs.rand())
        W64, A64 = _tss_dense_WA(W64, A64, k, v, w, tau_m, eps)
        W32, P32 = _tss_step(c, W32, P32, k, v, w, dt=onp.float32)
        assert W32.dtype == onp.float32 and P32.dtype == onp.float32
        worst = max(worst, onp.abs(W32 - W64).max()
                    / max(1.0, onp.abs(W64).max()))
    print(f"  TSS {c} 64-token float32: {worst:.3e}")
    assert worst < TRAJ32, (c, worst)


@pytest.mark.parametrize("case", range(len(TSS_COEFFS)))
def test_the_TSS_no_write_interval_is_undamped_drift_not_decay(case):
    """P is EXACTLY preserved and W advances by h P/M. The reference must not
    be described as memoryless, and the auxiliary must not decay here."""
    c = TSS_COEFFS[case]
    M = c["tau_m"] * c["epsilon"]
    rs = onp.random.RandomState(700 + case)
    W = rs.randn(5, 5); P = rs.randn(5, 5)
    k = rs.randn(5); k /= onp.linalg.norm(k)
    W2, P2 = _tss_step(c, W, P, k, rs.randn(5), 0.0)
    assert onp.abs(P2 - P).max() < IDENT64, (c, onp.abs(P2 - P).max())
    assert onp.abs(W2 - (W + AD.H * P / M)).max() < IDENT64, c
    assert onp.abs(P).max() > 1e-3, "an all-zero fixture proves nothing"


def test_the_TSS_reference_is_outside_the_generalized_sector():
    """gamma = 0 exactly, and no code path divides by it or uses the
    generalized (C, Z) storage formula."""
    import ast as _ast
    import inspect
    # no executed statement in the TSS token step or generator mentions gamma,
    # so nothing there can divide by it or reach the (C, Z) storage formula
    for fn in (AD.tss_two_state_step, AD.tss_generator):
        tree = _ast.parse(inspect.getsource(fn).lstrip())
        names = {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
        names |= {n.attr for n in _ast.walk(tree)
                  if isinstance(n, _ast.Attribute)}
        assert "gamma" not in names, (fn.__name__, names & {"gamma"})
    cfg = CAL.configurations()["slots"]
    for tag in ("A", "B"):
        s = cfg[f"tss_prospective/{tag}"]
        p = AM.init_params("tss_prospective", 80, init_coeffs=s,
                           dtype=onp.float64)
        c = AD.tss_response(p)
        assert float(c["gamma"][0]) == 0.0
        assert abs(float(c["tau_m"][0]) - s["tau_m"]) < 1e-10, tag
        assert abs(float(c["epsilon"][0]) - s["epsilon"]) < 1e-10, tag
        assert float(c["epsilon"][0]) < float(c["tau_m"][0]), tag
        assert abs(float(c["M"][0]) - s["tau_m"] * s["epsilon"]) < 1e-10
        assert abs(float(c["T"][0]) - (s["tau_m"] + s["epsilon"])) < 1e-10
        # M <= gamma T is FALSE here, at M > 0. Recorded, not repaired.
        assert float(c["M"][0]) > 0.0
        assert not (float(c["M"][0]) <= float(c["gamma"][0]) * float(c["T"][0]))


def test_the_TSS_slots_are_calibrated_to_the_same_beta_star():
    cfg = CAL.configurations()
    bs = cfg["beta_star"]
    for tag, ratio in (("A", 0.1), ("B", 0.5)):
        s = cfg["slots"][f"tss_prospective/{tag}"]
        assert abs(s["ratio"] - ratio) < 1e-12
        assert abs(CAL.tss_amplitude(s["q"], ratio) - bs) <= 1e-9, tag
        # the EXECUTED differentiable path reproduces the same observable
        M = jnp.float64(s["M"]); T = jnp.float64(s["T"])
        F = AD.expm2(AD.tss_generator(M, T, jnp.float64(1.0)))
        a = float(AD.tss_single_write_query_amplitude(F, M))
        assert abs(a - bs) < 1e-9, (tag, a, bs)
        assert float(F[1, 0]) < 0.0, "F21 is negative in (W, P) coordinates"
    for rec in cfg["records"]:
        if rec["kind"] == "tss":
            assert rec["observable_error"] <= CAL_TOL, rec


def _tss_probe(tag, dt):
    """A nondegenerate differentiable scalar of a short TSS trajectory, with
    every coefficient in `dt`. Fixed, parameter-independent initial carry."""
    s = CAL.configurations()["slots"][f"tss_prospective/{tag}"]
    r0 = onp.array([onp.log(s["tau_m"]),
                    onp.log(s["ratio"] / (1 - s["ratio"])), 0.0], dtype=dt)
    rs2 = onp.random.RandomState(21)
    dv = dk = 4
    ks = []
    for _ in range(6):
        x = rs2.randn(dk); ks.append(x / onp.linalg.norm(x))
    vs = [rs2.randn(dv) for _ in range(6)]
    ws = [0.0 if t == 2 else float(0.3 + 1.4 * rs2.rand()) for t in range(6)]
    qq = rs2.randn(dk); q = jnp.asarray(qq / onp.linalg.norm(qq), dt)
    proj = jnp.asarray(rs2.randn(dv), dt)

    def f(r):
        tau_m = jnp.exp(r[0])
        eps = tau_m * jax.nn.sigmoid(r[1])
        M, T = tau_m * eps, tau_m + eps
        W = jnp.zeros((dv, dk), dt); P = jnp.zeros((dv, dk), dt)
        acc = jnp.zeros((), dt)
        for t in range(6):
            wt = jnp.asarray(ws[t], dt) * (1.0 + 0.1 * r[2])
            F = AD.expm2(AD.tss_generator(M, T, wt))
            W, P = AD.tss_two_state_step(W, P, jnp.asarray(ks[t], dt),
                                         jnp.asarray(vs[t], dt), F,
                                         jnp.asarray(AD.H, dt) / M)
            acc = acc + jnp.dot(proj, W @ q)
        return acc
    return f, r0


@pytest.mark.parametrize("tag", ["A", "B"])
@pytest.mark.parametrize("di", range(3))
def test_TSS_float64_jvp_matches_central_differences(tag, di):
    """Exact parameter derivatives for BOTH declared timescale configurations,
    at both declared float64 steps. The initial carry is fixed and
    parameter-independent, as declared in the module docstring."""
    f, r0 = _tss_probe(tag, onp.float64)
    rs = onp.random.RandomState(800 + di)
    d = rs.randn(3); d /= onp.linalg.norm(d)
    jv = float(jax.jvp(f, (jnp.asarray(r0),), (jnp.asarray(d),))[1])
    assert onp.isfinite(jv)
    for h in FD64:
        fd = float((f(jnp.asarray(r0 + h * d)) - f(jnp.asarray(r0 - h * d)))
                   / (2 * h))
        if abs(fd) < 1e-6:
            assert abs(jv - fd) < NEAR64, (tag, di, h, jv, fd)
        else:
            assert abs(jv - fd) / abs(fd) < GRAD64, (tag, di, h, jv, fd)


@pytest.mark.parametrize("tag", ["A", "B"])
def test_TSS_float32_jvp_at_both_declared_perturbations(tag):
    """R5. The declared production-dtype counterpart, for BOTH calibrated
    configurations, enforced separately at both declared float32 steps."""
    f, r0 = _tss_probe(tag, onp.float32)
    rs = onp.random.RandomState(810)
    d = rs.randn(3).astype(onp.float32); d /= onp.linalg.norm(d)
    jv = float(jax.jvp(f, (jnp.asarray(r0, onp.float32),),
                       (jnp.asarray(d, onp.float32),))[1])
    assert onp.isfinite(jv)
    for h in FD32:
        fd = float((f(jnp.asarray(r0 + onp.float32(h) * d, onp.float32))
                    - f(jnp.asarray(r0 - onp.float32(h) * d, onp.float32)))
                   / (2 * h))
        a = abs(jv - fd)
        rel = a / abs(fd) if abs(fd) > 1e-4 else None
        print(f"  TSS f32 {tag} h={h:g} jvp={jv:.6g} fd={fd:.6g} abs={a:.3g}")
        assert a < NEAR32 or (rel is not None and rel < GRAD32), \
            (tag, h, jv, fd, a, rel)


@pytest.mark.parametrize("rule", list(AM.GATED_RULES))
@pytest.mark.parametrize("dt", [onp.float32, onp.float64])
def test_directional_derivatives_through_the_real_gate_and_embeddings(rule, dt):
    """R5. Through the ACTUAL source-gate and key-embedding leaves of a real
    episode, not a surrogate scalar source multiplier. Nonzero gradients alone
    do not establish that they are right, so the automatic directional
    derivative is compared with central differences of the same loss.
    """
    import optax
    slot = CAL.configurations()["slots"][f"{rule}/A"]
    p0 = AM.init_params(rule, 91, init_coeffs=slot, dtype=dt)
    b = TK.generate_batch(4321, 2)
    ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event")}
    lab = jnp.maximum(jnp.asarray(b["label"][0]), 0)
    qm = jnp.asarray(b["event"][0] == TK.QUERY)
    # a NONZERO gate is the interesting point: at u = b = 0 the sigmoid sits
    # at its symmetric point and the check would be weaker than it looks
    rs = onp.random.RandomState(92)
    p0 = dict(p0, gate_u=jnp.asarray(0.4 * rs.randn(31), dt),
              gate_b=jnp.asarray(0.4 * rs.randn(7), dt))
    leaves = ("gate_u", "gate_b", "key_raw")
    direction = {k: jnp.asarray(rs.randn(*onp.shape(p0[k])), dt)
                 for k in leaves}
    nrm = onp.sqrt(sum(float(jnp.sum(v ** 2)) for v in direction.values()))
    direction = {k: v / nrm for k, v in direction.items()}

    def loss(q):
        o = AM.rollout(rule, q, ep)
        ce = optax.softmax_cross_entropy(
            o["logits"],
            jax.nn.one_hot(lab, TK.N_VALUES, dtype=o["logits"].dtype))
        return jnp.sum(ce * qm) / jnp.sum(qm)

    def shifted(t):
        return loss({k: (v + t * direction[k] if k in direction else v)
                     for k, v in p0.items()})

    jv = float(jax.jvp(shifted, (jnp.asarray(0.0, dt),),
                       (jnp.asarray(1.0, dt),))[1])
    assert onp.isfinite(jv)
    steps = FD64 if dt is onp.float64 else FD32
    tol = GRAD64 if dt is onp.float64 else GRAD32
    near = NEAR64 if dt is onp.float64 else NEAR32
    for h in steps:
        fd = float((shifted(jnp.asarray(h, dt))
                    - shifted(jnp.asarray(-h, dt))) / (2 * h))
        a = abs(jv - fd)
        rel = a / abs(fd) if abs(fd) > (1e-6 if dt is onp.float64 else 1e-4) \
            else None
        print(f"  gate/emb {rule} {onp.dtype(dt).name} h={h:g} "
              f"jvp={jv:.6g} fd={fd:.6g} abs={a:.3g}")
        assert a < near or (rel is not None and rel < tol), \
            (rule, dt, h, jv, fd, a, rel)


# ----------------------- reference 2: ideal equilibrium --------------------
def test_the_projection_satisfies_its_constraint_and_preserves_orthogonals():
    rs = onp.random.RandomState(900)
    dv = dk = 6
    W = rs.randn(dv, dk)
    k = rs.randn(dk); k /= onp.linalg.norm(k)
    v = rs.randn(dv)
    W2 = onp.asarray(AD.projection_step(jnp.asarray(W), jnp.asarray(k),
                                        jnp.asarray(v), jnp.asarray(1.0)))
    assert onp.abs(W2 @ k - v).max() < IDENT64, "W+ k = v"
    # exactly a delta step at beta = 1
    Wd = onp.asarray(AD.delta_step(jnp.asarray(W), jnp.asarray(k),
                                   jnp.asarray(v), jnp.asarray(1.0)))
    assert onp.abs(W2 - Wd).max() < IDENT64
    # minimum change: the update is rank one along k
    assert onp.linalg.matrix_rank(W2 - W, tol=1e-10) == 1
    # an orthogonal test key is preserved exactly
    u = rs.randn(dk); u -= k * (k @ u); u /= onp.linalg.norm(u)
    assert onp.abs(W2 @ u - W @ u).max() < IDENT64
    # a query / idle / invalid-key interval holds W unchanged
    W3 = onp.asarray(AD.projection_step(jnp.asarray(W), jnp.asarray(k),
                                        jnp.asarray(v), jnp.asarray(0.0)))
    assert onp.array_equal(W3, W)


def test_the_projection_interferes_on_nonorthogonal_keys():
    """Recorded honestly: this reference is NOT a perfect memory. A second
    nonorthogonal write disturbs the first association."""
    rs = onp.random.RandomState(901)
    dv = dk = 6
    k1 = rs.randn(dk); k1 /= onp.linalg.norm(k1)
    k2 = k1 + 0.6 * rs.randn(dk); k2 /= onp.linalg.norm(k2)
    assert abs(k1 @ k2) > 0.3, "the fixture must be genuinely nonorthogonal"
    v1, v2 = rs.randn(dv), rs.randn(dv)
    W = jnp.zeros((dv, dk))
    W = AD.projection_step(W, jnp.asarray(k1), jnp.asarray(v1), jnp.asarray(1.))
    W = AD.projection_step(W, jnp.asarray(k2), jnp.asarray(v2), jnp.asarray(1.))
    assert onp.abs(onp.asarray(W) @ k2 - v2).max() < IDENT64   # last write exact
    assert onp.abs(onp.asarray(W) @ k1 - v1).max() > 1e-3      # first disturbed


def test_the_projection_autodiff_matches_the_direct_formula():
    """Autodiff through the projection, against the direct expression, at both
    declared float64 steps."""
    rs = onp.random.RandomState(902)
    dv = dk = 5
    W0 = rs.randn(dv, dk); v = rs.randn(dv); d = rs.randn(dk)
    d /= onp.linalg.norm(d)
    proj = jnp.asarray(rs.randn(dv))
    qq = rs.randn(dk); q = jnp.asarray(qq / onp.linalg.norm(qq))

    def f(x):
        kk = jnp.asarray(W0[0]) * 0.0 + x
        kk = kk / jnp.sqrt(jnp.sum(kk ** 2))
        W = AD.projection_step(jnp.asarray(W0), kk, jnp.asarray(v),
                               jnp.asarray(1.0))
        return jnp.dot(proj, W @ q)

    k0 = rs.randn(dk)
    jv = float(jax.jvp(f, (jnp.asarray(k0),), (jnp.asarray(d),))[1])
    for h in FD64:
        fd = float((f(jnp.asarray(k0 + h * d)) - f(jnp.asarray(k0 - h * d)))
                   / (2 * h))
        assert abs(jv - fd) / max(abs(fd), 1e-6) < GRAD64, (h, jv, fd)


def test_the_ideal_reference_has_no_gate_and_no_response_leaf():
    p = AM.init_params("ideal_projection", 903)
    assert set(p) == {"key_raw", "value_table", "readout_W", "readout_b"}
    assert AM.parameter_counts("ideal_projection", p)["total"] == 392
    assert AD.CARRY["ideal_projection"] == 64
    slots = CAL.configurations()["slots"]
    for tag, lr in (("A", 0.003), ("B", 0.01)):
        s = slots[f"ideal_projection/{tag}"]
        assert s["lr"] == lr and s["calibrated"] is False and s["beta"] == 1.0


def test_the_ideal_arm_rolls_out_as_a_full_strength_delta_memory():
    """The shell's rollout for this arm equals an independent literal
    reimplementation of the projection over a real episode."""
    p = AM.init_params("ideal_projection", 904, dtype=onp.float64)
    b = TK.generate_batch(4500, 1)
    ep = _ep(b, 0)
    got = onp.asarray(AM.rollout("ideal_projection", p, ep)["logits"])
    k_all, k_valid = AD.safe_normalize(p["key_raw"])
    k_all = onp.asarray(k_all); k_valid = onp.asarray(k_valid)
    kid = onp.asarray(ep["key_id"]); vid = onp.asarray(ep["val_id"])
    evt = onp.asarray(ep["event"])
    W = onp.zeros((8, 8))
    ref = []
    for t in range(kid.shape[0]):
        k = k_all[kid[t]]
        m = float(evt[t] == TK.WRITE) * float(k_valid[kid[t]])
        v = (onp.asarray(p["value_table"])[max(int(vid[t]), 0)]
             * float(vid[t] >= 0) * m)
        W = W + m * onp.outer(v - W @ k, k)
        ref.append(onp.asarray(p["readout_W"]) @ (W @ k)
                   + onp.asarray(p["readout_b"]))
    ref = onp.stack(ref)
    assert onp.abs(got - ref).max() < 1e-10, onp.abs(got - ref).max()


def _screen_rows(primary, ret=None, rec=None):
    """Synthetic held-out rows for exercising the declared screen rules."""
    from experiments.adaptive_memory import study as ST
    ret = ret or {}
    rec = rec or {}
    rows = []
    for sd in ST.FINAL_SEEDS:
        for rule, pr in primary.items():
            rows.append(dict(rule=rule, seed=sd, heldout=dict(
                primary=pr, retention_revision_untouched=ret.get(rule, 0.9),
                recall_overall=rec.get(rule, 0.9), revision_ce=0.5)))
    return rows


BASE_PRIMARY = {"adaptive_prospective": 0.60, "momentum_delta": 0.70,
                "gated_delta": 0.50, "tss_prospective": 0.40,
                "ideal_projection": 0.45, "adaptive_inertial": 0.50,
                "adaptive_delta": 0.50}


def test_the_two_screens_are_reported_separately():
    """A win against ordinary prospectivity must not be able to substitute for
    a loss against the strongest memory comparator, or the reverse."""
    from experiments.adaptive_memory import study as ST
    res = ST.screen(_screen_rows(BASE_PRIMARY), {})
    assert res["literature_screen_passed"] is False     # loses to momentum
    assert res["ordinary_prospectivity_screen_passed"] is True
    assert [c["against"] for c in res["ordinary_prospectivity"]] == \
        list(AD.ORDINARY_PROSPECTIVE)
    assert set(res["per_arm_means"]) == set(AD.RULES)


def test_attribution_is_reported_independently_of_the_literature_screen():
    """R2. The candidate LOSES to momentum but beats the equally gated
    inertial control in every seed. The ablation comparison must still be
    credited: it does not imply competitive performance, and gating it on the
    literature screen would silently discard an interpretable result."""
    from experiments.adaptive_memory import study as ST
    res = ST.screen(_screen_rows(BASE_PRIMARY), {})
    assert res["literature_screen_passed"] is False
    assert res["attribution"][0]["against"] == "adaptive_inertial"
    assert res["attribution"][0]["positive_in_all_seeds"] is True
    assert res["prospective_term_credited"] is True
    assert res["attribution_is_independent_of_the_literature_screen"] is True


def test_a_large_loss_on_one_retention_metric_fails_the_screen():
    """R2. The safeguard requires BOTH metric differences >= -1 point.

    Three fixtures: retention-only failure, recall-only failure, and both
    within the bound. The previous `not (A and B)` form passed the first two.
    """
    from experiments.adaptive_memory import study as ST
    win = dict(BASE_PRIMARY, adaptive_prospective=0.90)   # +20 on primary

    # (a) retention collapses by 20 points, recall unchanged -> MUST FAIL
    res = ST.screen(_screen_rows(
        win, ret={"adaptive_prospective": 0.70, "momentum_delta": 0.90}), {})
    c = res["comparisons"][0]
    assert c["meets_plus_one_point"] and c["positive_in_all_seeds"]
    assert c["retention_within_one_point"] is False
    assert c["recall_within_one_point"] is True
    assert c["retention_safeguard_met"] is False and c["passed"] is False
    assert res["literature_screen_passed"] is False

    # (b) recall collapses, retention unchanged -> MUST FAIL
    res = ST.screen(_screen_rows(
        win, rec={"adaptive_prospective": 0.70, "momentum_delta": 0.90}), {})
    c = res["comparisons"][0]
    assert c["retention_within_one_point"] is True
    assert c["recall_within_one_point"] is False
    assert c["passed"] is False

    # (c) both within the one-point bound -> passes
    res = ST.screen(_screen_rows(
        win, ret={"adaptive_prospective": 0.895, "momentum_delta": 0.90},
        rec={"adaptive_prospective": 0.895, "momentum_delta": 0.90}), {})
    assert all(c["passed"] for c in res["comparisons"])
    assert res["literature_screen_passed"] is True


def test_a_missing_paired_seed_cannot_produce_a_verdict():
    """R2. An incomplete set of pairs must not slip through `all()`."""
    from experiments.adaptive_memory import study as ST
    win = dict(BASE_PRIMARY, adaptive_prospective=0.90)
    rows = [r for r in _screen_rows(win)
            if not (r["rule"] == "adaptive_prospective"
                    and r["seed"] == ST.FINAL_SEEDS[-1])]
    res = ST.screen(rows, {})
    for c in res["comparisons"] + res["ordinary_prospectivity"]:
        assert c["complete_paired_seeds"] is False
        assert c["passed"] is False
    assert res["literature_screen_passed"] is False
    assert res["ordinary_prospectivity_screen_passed"] is False
    assert res["prospective_term_credited"] is False


# --------------------- R1: the calibration grid is safe --------------------
def test_the_calibration_grid_tail_is_finite_and_not_evaluated_early():
    """R1. The declared grid's upper end drove the previous host exponential
    to overflow in `cosh`, so `configurations()` never reached the bracket
    loop. Two independent guarantees are checked:

    1. the host observable is FINITE across the whole declared grid, so an
       arithmetic failure can never be mistaken for a physical obstruction;
    2. the scan is incremental and stops at the first bracket, so a slot that
       brackets early never pays for the hazardous tail at all.
    """
    grid = onp.exp(onp.linspace(CAL.GRID_LO, CAL.GRID_HI, CAL.GRID_N))
    assert grid[-1] > 1.6e5, "the declared grid really does reach exp(12)"
    for rate in (grid[0], grid[CAL.GRID_N // 2], grid[-1]):
        a = CAL.amplitude(CAL.prospective_G(rate, 0.75, 0.75), 0.75)
        assert onp.isfinite(a), ("prospective", rate, a)
        a = CAL.amplitude(CAL.inertial_G(rate, 0.75), 0.75)
        assert onp.isfinite(a), ("inertial", rate, a)
        for ratio in (0.1, 0.5):
            assert onp.isfinite(CAL.tss_amplitude(rate, ratio)), (rate, ratio)
    for rec in CAL.configurations()["records"]:
        assert rec["grid_points_evaluated"] <= CAL.GRID_N, rec
        # the bracket really is the FIRST one: nothing beyond it was touched
        hi = rec["bracket"][1]
        idx = int(onp.argmin(onp.abs(grid - hi)))
        assert rec["grid_points_evaluated"] == idx + 1, (rec, idx)


def test_a_calibration_obstruction_is_an_explicit_record_not_a_crash():
    """An impossible target must raise the declared obstruction, and must not
    be reported as evidence that a physical initialization is impossible."""
    with pytest.raises(FloatingPointError) as exc:
        CAL.solve_rate("prospective", 0.75, rho=0.75, target=5.0)
    assert "obstruction" in str(exc.value)


# ------------- derived physical coefficients, not just raw leaves ----------
@pytest.mark.parametrize("rule", list(AM.NEW_RULES))
def test_derived_physical_coefficients_are_finite_and_positive(rule):
    """A finite raw logarithm does not guarantee a finite exponential. The
    EXECUTED constraint is checked on the derived coefficients themselves."""
    cfg = CAL.configurations()["slots"]
    for tag in ("A", "B"):
        p = AM.init_params(rule, 93, init_coeffs=cfg[f"{rule}/{tag}"],
                           dtype=onp.float64)
        if rule == "adaptive_prospective":
            c = AD.response(p)
            vals = {k: float(onp.asarray(c[k]).ravel()[0]) for k in c}
            assert all(onp.isfinite(v) for v in vals.values()), (rule, vals)
            assert vals["M"] > 0 and vals["gamma"] > 0 and vals["T"] > 0
            assert vals["eta"] > 0 and vals["kappa"] > 0
            assert 0.0 < vals["M"] < vals["gamma"] * vals["T"], vals
        elif rule == "adaptive_inertial":
            c = AD.inertial_response(p)
            assert float(c["eta"][0]) > 0 and float(c["tau"][0]) > 0
        elif rule == "adaptive_delta":
            assert float(onp.exp(onp.asarray(p["raw_eta"])[0])) > 0
        elif rule == "tss_prospective":
            c = AD.tss_response(p)
            vals = {k: float(onp.asarray(c[k]).ravel()[0]) for k in c}
            assert all(onp.isfinite(v) for v in vals.values()), (rule, vals)
            assert vals["tau_m"] > 0 and 0 < vals["epsilon"] < vals["tau_m"]
            assert vals["M"] > 0 and vals["T"] > 0 and vals["gamma"] == 0.0
        else:
            assert not any(k.startswith("raw_") for k in p)


@pytest.mark.parametrize("rule", list(AD.RULES))
def test_the_checkpoint_validator_accepts_every_declared_initialization(rule):
    """F2. The gate must not reject a legitimate arm - including TSS, whose
    gamma = 0 is intentional and must never be tested against the generalized
    candidate's sector."""
    from experiments.adaptive_memory import study as ST
    cfg = CAL.configurations()["slots"]
    for tag in ("A", "B"):
        for dt in (onp.float32, onp.float64):
            p = AM.init_params(rule, 95, init_coeffs=cfg.get(f"{rule}/{tag}"),
                               dtype=dt)
            assert ST.validate_coefficients(rule, p) is None, (rule, tag, dt)


def test_the_checkpoint_validator_rejects_a_drifted_learned_coefficient():
    """F2. The learned coefficients are what matter: a checkpoint whose raw
    scalars have drifted to non-finite or out-of-sector values must produce a
    RECORDED FAILED RUN, not a silent PASS with a report flag saying
    otherwise. These fixtures edit a trained-parameter dict directly; they do
    not train and do not change any declared configuration.
    """
    from experiments.adaptive_memory import study as ST
    cfg = CAL.configurations()["slots"]

    def poisoned(rule, **kw):
        p = AM.init_params(rule, 96, init_coeffs=cfg[f"{rule}/A"],
                           dtype=onp.float64)
        return dict(p, **{k: jnp.asarray([v], onp.float64)
                          for k, v in kw.items()})

    # a raw logarithm that overflows its executed exponential
    for rule, leaf in (("adaptive_prospective", "raw_nu"),
                       ("adaptive_inertial", "raw_eta"),
                       ("adaptive_delta", "raw_eta"),
                       ("tss_prospective", "raw_tau_m")):
        msg = ST.validate_coefficients(rule, poisoned(rule, **{leaf: 1e6}))
        assert msg and "finite" in msg, (rule, leaf, msg)
        msg = ST.validate_coefficients(rule, poisoned(rule, **{leaf: onp.nan}))
        assert msg, (rule, leaf, "NaN must be rejected")

    # rho -> 1 collapses kappa and drives M onto gamma*T: the admissibility
    # boundary must be caught rather than accepted as "just inside"
    msg = ST.validate_coefficients(
        "adaptive_prospective", poisoned("adaptive_prospective", raw_rho=1e6))
    assert msg is not None, "rho = 1 must not be accepted"
    assert any(w in msg for w in ("kappa", "rho", "admissibility")), msg

    # TSS: adaptation must stay strictly faster than the membrane
    msg = ST.validate_coefficients(
        "tss_prospective", poisoned("tss_prospective", raw_ratio=1e6))
    assert msg and "faster" in msg, msg

    # and a healthy checkpoint still passes, so the gate is not vacuous
    for rule in AM.NEW_RULES:
        p = AM.init_params(rule, 96, init_coeffs=cfg[f"{rule}/A"],
                           dtype=onp.float64)
        assert ST.validate_coefficients(rule, p) is None, rule


def test_the_study_reports_the_executed_constraint_for_every_arm():
    from experiments.adaptive_memory import study as ST
    cfg = CAL.configurations()["slots"]
    for rule in AD.RULES:
        p = AM.init_params(rule, 94, init_coeffs=cfg.get(f"{rule}/A", {}))
        rep = ST.coefficient_report(rule, p, None)
        assert rep["rule"] == rule
        if rule == "adaptive_prospective":
            assert rep["admissible_0_lt_M_lt_gamma_T"] is True
        if rule == "tss_prospective":
            assert rep["gamma"] == 0.0
            assert rep["outside_the_generalized_sector"] is True
            assert rep["adaptation_faster_than_membrane"] is True
        if rule in AM.GATED_RULES:
            assert 0.0 < rep["source_weight"]["min"] <= \
                rep["source_weight"]["max"] < 2.0
