"""Unit tests for the learning-access branch (specification items 1-10).

The older closure tests in tests/test_tss_closure.py are untouched and remain
the regression suite for this branch.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jax.config.update("jax_enable_x64", True)

from prospective.learning_access import (check_port_access, corrected_tangent,
                                         minimal_port)
from prospective.three_state import (ThreeStateConfig, corrected_estimate,
                                     dw_dtheta_analytic, dz_dtheta_analytic,
                                     numpy_rollout, prospective_inverse,
                                     rollout, w_of)

T = 30


# ---- 1. prospective inverse identity w = z + theta b ----------------------

@pytest.mark.parametrize("theta", [-0.3, -0.05, 0.0, 0.11, 0.4])
def test_1_prospective_inverse_identity(theta):
    rng = np.random.default_rng(1)
    cfg = ThreeStateConfig()
    x, v = rng.normal(size=T), rng.normal(size=T)
    out = rollout(cfg, theta, x, v)
    w = np.asarray(prospective_inverse(cfg, out["s"]))
    expect = np.asarray(out["z"])[1:] + theta * np.asarray(out["b"])[1:]
    np.testing.assert_allclose(w, expect, rtol=0, atol=1e-11)


def test_1_matches_independent_numpy_reference():
    """JAX is never checked only against itself."""
    rng = np.random.default_rng(2)
    cfg = ThreeStateConfig()
    x, v = rng.normal(size=T), rng.normal(size=T)
    j = rollout(cfg, 0.17, x, v)
    n = numpy_rollout(cfg, 0.17, x, v)
    for k in ("z", "b", "s"):
        np.testing.assert_allclose(np.asarray(j[k]), n[k], rtol=0, atol=1e-11)


# ---- 2. w = z at theta = 0 over randomized histories ----------------------

def test_2_inference_perfect_at_theta_zero():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(120):
        cfg = ThreeStateConfig(z0=float(rng.normal()), b0=float(rng.normal() * 5))
        x, v = rng.normal(size=T), rng.normal(size=T)
        out = rollout(cfg, 0.0, x, v)
        w = np.asarray(prospective_inverse(cfg, out["s"]))
        worst = max(worst, float(np.max(np.abs(w - np.asarray(out["z"])[1:]))))
    assert worst < 1e-11, worst


# ---- 3. analytic / autodiff / finite-difference agreement ----------------

def test_3_tangent_three_ways_agree():
    rng = np.random.default_rng(4)
    cfg = ThreeStateConfig()
    x, v = rng.normal(size=T), rng.normal(size=T)
    ana = dw_dtheta_analytic(cfg, x, v)
    auto = np.asarray(jax.jacobian(lambda t: w_of(cfg, t, x, v))(0.0))
    eps = 1e-6
    fd = (np.asarray(w_of(cfg, eps, x, v))
          - np.asarray(w_of(cfg, -eps, x, v))) / (2 * eps)
    np.testing.assert_allclose(auto, ana, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fd, ana, rtol=0, atol=1e-6)


def test_3_tangent_gap_is_exactly_b():
    rng = np.random.default_rng(5)
    cfg = ThreeStateConfig()
    x, v = rng.normal(size=T), rng.normal(size=T)
    ana = dw_dtheta_analytic(cfg, x, v)
    dz = dz_dtheta_analytic(cfg, x)[1:]
    b = numpy_rollout(cfg, 0.0, x, v)["b"][1:]
    np.testing.assert_allclose(ana - dz, b, rtol=0, atol=1e-11)


# ---- 4. same visible history, different hidden tangent -------------------

def test_4_impossibility_witness():
    rng = np.random.default_rng(6)
    x = rng.normal(size=T)
    A = ThreeStateConfig(b0=-4.0)
    B = ThreeStateConfig(b0=+2.5)
    vA, vB = rng.normal(size=T), rng.normal(size=T) * 0.5 - 0.3
    wA = np.asarray(w_of(A, 0.0, x, vA))
    wB = np.asarray(w_of(B, 0.0, x, vB))
    np.testing.assert_allclose(wA, wB, rtol=0, atol=1e-12)
    bA = numpy_rollout(A, 0.0, x, vA)["b"][1:]
    bB = numpy_rollout(B, 0.0, x, vB)["b"][1:]
    assert np.max(np.abs(bA - bB)) > 1.0          # corrections differ...
    assert np.max(np.abs(wA - wB)) < 1e-12        # ...on identical evidence


# ---- 5. noiseless side channel reproduces the FULL parameterized task ----

@pytest.mark.parametrize("theta", [-0.25, 0.0, 0.13, 0.31])
@pytest.mark.parametrize("delta", [0.05, 1.0, 6.0])
def test_5_noiseless_correction_exact_for_all_theta(theta, delta):
    rng = np.random.default_rng(7)
    cfg = ThreeStateConfig()
    x, v = rng.normal(size=T), rng.normal(size=T)
    zhat = np.asarray(corrected_estimate(cfg, theta, x, v, delta))
    z = numpy_rollout(cfg, theta, x, v)["z"][1:]
    np.testing.assert_allclose(zhat, z, rtol=0, atol=1e-10)


def test_5_corrected_gradient_equals_intended_gradient():
    rng = np.random.default_rng(8)
    cfg = ThreeStateConfig()
    x, v = rng.normal(size=T), rng.normal(size=T)
    y_star = 0.5
    lo = lambda y: 0.5 * (y - y_star) ** 2
    g_hat = float(jax.grad(lambda t: lo(corrected_estimate(
        cfg, t, x, v, 1.0)[-1]))(0.0))
    z = numpy_rollout(cfg, 0.0, x, v)["z"]
    g_int = (z[T] - y_star) * dz_dtheta_analytic(cfg, x)[T]
    assert g_hat == pytest.approx(g_int, rel=1e-9)


# ---- 6. gradient-variance slope approximately -2 -------------------------

def test_6_gradient_variance_scales_as_delta_to_the_minus_two():
    rng = np.random.default_rng(9)
    cfg = ThreeStateConfig()
    n, sigma = 20, 0.1
    x, v = rng.normal(size=n), rng.normal(size=n)
    y_star = 0.5

    def one(nz, dl):
        return jax.grad(lambda t: 0.5 * (corrected_estimate(
            cfg, t, x, v, dl, noise=nz)[-1] - y_star) ** 2)(0.0)

    batch = jax.jit(jax.vmap(one, in_axes=(0, None)))
    deltas = np.array([0.05, 0.2, 1.0, 5.0])
    var = []
    for dl in deltas:
        nz = jnp.asarray(rng.normal(scale=sigma, size=(4000, n)))
        var.append(float(np.var(np.asarray(batch(nz, float(dl))))))
    slope = float(np.polyfit(np.log(deltas), np.log(var), 1)[0])
    assert abs(slope + 2.0) < 0.1, slope


# ---- 7. port-access condition succeeds / fails on constructed cases ------

def test_7_port_access_holds_and_builds_a_valid_corrector():
    rng = np.random.default_rng(10)
    U = [np.array([[1.0, -0.5, 2.0]])]
    R = np.eye(3)
    res = check_port_access(U, R)
    assert res.holds and res.p_learning == 1
    for _ in range(50):
        b = rng.normal(size=3)
        resid = corrected_tangent(U, R, res.L, [0.3], b)
        assert np.max(np.abs(resid)) < 1e-11


def test_7_port_access_fails_with_explicit_witness():
    U = [np.array([[1.0, -0.5, 2.0]])]
    R = np.array([[1.0, 0, 0], [0, 1.0, 0]])       # cannot see coordinate 3
    res = check_port_access(U, R)
    assert not res.holds
    assert np.max(np.abs(R @ res.witness_b)) < 1e-12     # invisible at the port
    assert np.max(np.abs(res.witness_Ub)) > 1e-6         # but needed for learning


def test_7_rank_formula_and_minimal_port():
    U = [np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])]
    assert check_port_access(U, np.eye(3)).p_learning == 2
    assert not check_port_access(U, np.array([[1.0, 0.0, 0.0]])).holds
    Rm, rank = minimal_port(U)
    assert rank == 2 and check_port_access(U, Rm).holds


# ---- 8/9/10. BPTT: gradients, predicted signs, nuisance detection --------

def _bptt_mods():
    from experiments.learning_access.bptt_wrong_memory import (
        make_task, y_local, y_collective, loss_of, train, diagnostics, RHO)
    return make_task, y_local, y_collective, loss_of, train, diagnostics, RHO


def test_8_bptt_matches_finite_differences_for_both_models():
    make_task, y_loc, y_col, loss_of, _, _, _ = _bptt_mods()
    cfg, x, v, y_star = make_task()
    eps = 1e-6
    for model in (y_loc, y_col):
        for th in (0.0, 0.05, -0.1):
            g = float(jax.grad(lambda t: loss_of(model, cfg, t, x, v, y_star))(th))
            fd = float((loss_of(model, cfg, th + eps, x, v, y_star)
                        - loss_of(model, cfg, th - eps, x, v, y_star)) / (2 * eps))
            assert g == pytest.approx(fd, abs=1e-6)


def test_9_predicted_initial_gradient_difference():
    make_task, y_loc, y_col, loss_of, _, _, RHO = _bptt_mods()
    cfg, x, v, y_star = make_task()
    g_loc = float(jax.grad(lambda t: loss_of(y_loc, cfg, t, x, v, y_star))(0.0))
    g_col = float(jax.grad(lambda t: loss_of(y_col, cfg, t, x, v, y_star))(0.0))
    err0 = 0.81 - 0.9025
    assert g_loc == pytest.approx(err0 * (2 * RHO - 2.56), rel=1e-9)
    assert g_col == pytest.approx(err0 * (2 * RHO), rel=1e-9)
    assert g_loc > 0 > g_col          # opposite directions


def test_10_disturbance_removal_detects_nuisance_assisted_learning():
    make_task, y_loc, y_col, _, train, diagnostics, RHO = _bptt_mods()
    cfg, x, v, y_star = make_task()
    th_loc, _ = train(y_loc, cfg, x, v, y_star, record=False)
    th_col, _ = train(y_col, cfg, x, v, y_star, record=False)
    D_loc = diagnostics(y_loc, cfg, th_loc, x, v, y_star)
    D_col = diagnostics(y_col, cfg, th_col, x, v, y_star)

    # both fit the training objective
    assert D_loc["train_loss"] < 1e-10 and D_col["train_loss"] < 1e-10
    # only the collective model learned the intended memory
    assert D_col["pole_error"] < 1e-6
    assert D_loc["pole_error"] > 0.1
    # the analytic local root
    assert th_loc == pytest.approx(
        (0.76 - np.sqrt(0.76 ** 2 + 4 * 0.0925)) / 2, rel=1e-6)
    # the decisive diagnostic
    assert D_loc["loss_disturbance_removed"] > 1e-3
    assert D_col["loss_disturbance_removed"] < 1e-12
