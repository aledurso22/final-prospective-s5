"""Unit tests for the closure-prospectivity branch (specification items 1-6)."""
import os
import sys

import jax
import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jax.config.update("jax_enable_x64", True)

from prospective.closure_models import (TwoModeClosure, affine_tss,
                                        nonlinear_tss, two_dim_affine_tss)
from prospective.identification import (compare_structures, evaluate_held_out,
                                        fit_closure, fit_matrix_closure,
                                        rollout, select_order)

GAMMA, A_HID, Z, B0 = 1.0, 0.4, 0.5, 1.0


# ---- 1. one-pole TSS residual decays at the prescribed rate ---------------

@pytest.mark.parametrize("factory", [affine_tss, nonlinear_tss, two_dim_affine_tss])
def test_1_ideal_tss_residual_decays_at_prescribed_rate(factory):
    sysm = factory()
    tau, dt, n = 0.35, 0.005, 3000
    traj = sysm.integrate(np.full(sysm.dim, 0.8), dt, n, lambda r: -r / tau)
    mag = np.linalg.norm(traj["r"], axis=1)
    keep = mag > 1e-12
    idx = np.arange(len(mag))[keep][:1200]
    slope = np.polyfit(idx * dt, np.log(mag[keep][:1200]), 1)[0]
    assert abs(slope + 1.0 / tau) * tau < 0.02, (slope, -1.0 / tau)


def test_1_ideal_tss_selects_one_mode():
    rng = np.random.default_rng(0)
    tau, dt, n = 0.35, 0.005, 3000
    a1 = np.exp(-dt / tau)
    trajs = []
    for _ in range(6):
        r = np.zeros(n); r[0] = rng.normal()
        for k in range(1, n):
            r[k] = a1 * r[k - 1] + 0.02 * rng.normal()
        trajs.append(r)
    best, _ = select_order(trajs, dt, orders=(1, 2, 3), criterion="bic")
    assert best.order == 1


# ---- 2. two-mode system with r0=0, b0!=0 gives rdot(0) = z b0 -------------

def test_2_hidden_mode_drives_residual_from_zero():
    sysm = TwoModeClosure(gamma=GAMMA, a=A_HID, z=Z)
    assert sysm.rdot_at_zero_residual(B0) == pytest.approx(Z * B0)
    dt, n = 1e-4, 200
    xs = sysm.simulate(0.0, B0, dt, n)
    assert xs[0, 0] == pytest.approx(0.0, abs=1e-15)
    numeric = (xs[1, 0] - xs[0, 0]) / dt
    assert numeric == pytest.approx(Z * B0, rel=2e-3)
    # and the residual genuinely leaves zero
    xs_long = sysm.simulate(0.0, B0, 0.002, 6000)
    assert np.max(np.abs(xs_long[:, 0])) > 0.25


def test_2_stability_constraint_is_enforced():
    with pytest.raises(ValueError, match="unstable"):
        TwoModeClosure(gamma=0.2, a=0.2, z=0.9)      # a*gamma < z^2


# ---- 3. oracle control keeps r ~ 0 ----------------------------------------

def test_3_oracle_control_holds_residual_at_zero():
    sysm = TwoModeClosure(gamma=GAMMA, a=A_HID, z=Z)
    dt, T = 0.001, 15.0
    n = int(T / dt)
    t = np.arange(n) * dt
    xs = sysm.simulate(0.0, B0, dt, n, u_c=sysm.oracle_control(B0, t))
    assert np.max(np.abs(xs[:, 0])) < 1e-3
    # hidden state and control are emphatically NOT zero
    assert np.max(np.abs(xs[:, 1])) > 0.9
    assert np.max(np.abs(sysm.oracle_control(B0, t))) == pytest.approx(Z * B0)


def test_3_zoh_residual_error_is_first_order_in_dt():
    sysm = TwoModeClosure(gamma=GAMMA, a=A_HID, z=Z)
    ratios = []
    for dt in (0.008, 0.004, 0.002, 0.001):
        n = int(15.0 / dt)
        t = np.arange(n) * dt
        xs = sysm.simulate(0.0, B0, dt, n, u_c=sysm.oracle_control(B0, t))
        ratios.append(np.max(np.abs(xs[:, 0])) / dt)
    assert max(ratios) / min(ratios) < 1.3, ratios


# ---- 4. analytic oracle-control effort matches numerical integration ------

def test_4_control_effort_matches_analytic():
    sysm = TwoModeClosure(gamma=GAMMA, a=A_HID, z=Z)
    dt, T = 0.001, 15.0
    t = np.arange(int(T / dt)) * dt
    u = sysm.oracle_control(B0, t)
    numeric = float(np.trapezoid(0.5 * u ** 2, t))
    assert numeric == pytest.approx(sysm.oracle_control_effort(B0, T), rel=1e-5)


# ---- 5. finite-difference full gradient matches autodiff ------------------

def _V(z, dt=0.001, T=15.0, a=A_HID, b0=B0):
    import jax.numpy as jnp
    k = jnp.arange(int(T / dt))
    u = -z * b0 * jnp.exp(-a * k * dt)
    sq = u ** 2
    return 0.5 * dt * (jnp.sum(sq) - 0.5 * (sq[0] + sq[-1]))


def test_5_autodiff_matches_finite_difference_and_closed_form():
    z0, eps = 0.5, 1e-6
    g_auto = float(jax.grad(_V)(z0))
    g_fd = float((_V(z0 + eps) - _V(z0 - eps)) / (2 * eps))
    g_ana = z0 * B0 ** 2 * (1 - np.exp(-2 * A_HID * 15.0)) / (2 * A_HID)
    assert g_auto == pytest.approx(g_fd, rel=1e-6)
    assert g_auto == pytest.approx(g_ana, rel=1e-5)


# ---- 6. reduced one-pole gradient FAILS in the sensitive example ----------

def test_6_reduced_closure_gradient_is_wrong():
    """Least-Control: r(t) == 0 yet the reduced model reports zero gradient."""
    z0 = 0.5
    g_full = float(jax.grad(_V)(z0))
    g_reduced = float(jax.grad(lambda z: 0.0 * z)(z0))   # no hidden state
    assert abs(g_full) > 0.5
    assert g_reduced == 0.0
    assert abs(g_reduced - g_full) > 0.5


def test_6_reduced_model_matches_response_but_not_tangent():
    """H_theta(p) = 1/(p+1) + theta/(p+3): forward fine, tangent wrong."""
    dt, eps = 0.01, 0.05
    t = np.arange(int(8.0 / dt)) * dt
    h = lambda th: np.exp(-t) + th * np.exp(-3 * t)

    def fit_roll(sig, order):
        f = fit_closure(sig, dt, order=order, holdout_frac=0.2)
        return np.concatenate([sig[:order],
                               rollout(f, sig[:order], len(sig) - order)])

    for order, fwd_tol, expect_tangent_ok in ((1, 1e-10, False), (2, 1e-10, True)):
        fwd = fit_roll(h(0.0), order)
        assert np.sqrt(np.mean((fwd - h(0.0)) ** 2)) < fwd_tol
        tangent = (fit_roll(h(eps), order) - fit_roll(h(-eps), order)) / (2 * eps)
        err = np.sqrt(np.mean((tangent - np.exp(-3 * t)) ** 2))
        if expect_tangent_ok:
            assert err < 1e-6, err
        else:
            assert err > 1e-2, err


# ---- collective first-order: the four model classes ----------------------

def test_collective_first_order_needs_full_matrix_not_hidden_state():
    rng = np.random.default_rng(1)
    dt, n = 0.01, 1200
    Gamma = np.array([[1.0, 0.8], [0.8, 3.0]])
    Ad = expm(-Gamma * dt)

    def make(m):
        out = []
        for _ in range(m):
            r = np.zeros((n, 2)); r[0] = rng.normal(size=2)
            for k in range(1, n):
                r[k] = Ad @ r[k - 1] + 2e-3 * rng.normal(size=2)
            out.append(r)
        return out

    train, test = make(12), make(6)
    best, fits = compare_structures(train, dt, holdout_frac=0.25)
    assert best.structure == "full", best.structure

    true_poles = np.sort(np.real(-np.linalg.eigvals(Gamma)))
    got = np.sort(np.real(fits["full"].continuous_poles))
    assert np.max(np.abs(got - true_poles)) < 0.05

    # scalar and diagonal cannot represent the coupling
    for s in ("scalar", "diagonal"):
        p = np.sort(np.real(fits[s].continuous_poles))[:2]
        assert np.max(np.abs(p - true_poles)) > 0.3, s

    # the augmented (hidden-state) control must not be materially better
    mse_full, _ = evaluate_held_out(fits["full"], test)
    mse_aug, _ = evaluate_held_out(fits["augmented"], test)
    assert (mse_full - mse_aug) / mse_full < 0.05


def test_rollout_state_ordering_is_chronological():
    """Guards the double-reversal bug: order>=2 rollouts must be exact."""
    dt = 0.01
    t = np.arange(400) * dt
    sig = np.exp(-t) + 0.4 * np.exp(-3 * t)
    f = fit_closure(sig, dt, order=2, holdout_frac=0.2)
    pred = rollout(f, sig[:2], len(sig) - 2)
    assert np.max(np.abs(pred - sig[2:])) < 1e-8
