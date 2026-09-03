"""Tests for the professor-mechanism synthetic study.

These check the DISCRETIZATIONS against closed-form solutions, so the
experiment's conclusions rest on verified numerics rather than on the
implementation agreeing with itself.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanism import metrics
from mechanism.full_state_pc import FullStatePC
from mechanism.plain_ssm import PlainSSM
from mechanism.projected_state_pc import ProjectedStatePC
from mechanism.readout_lead_control import (ReadoutLeadPerMode, apply_lead,
                                            matched_alpha)
from mechanism.static_equilibrium_bypass import StaticEquilibriumBypass
from mechanism.system import DiagonalSystem, two_timescale_system


@pytest.fixture
def sysm():
    return two_timescale_system()


def test_projector_partition_is_identity(sysm):
    np.testing.assert_array_equal(sysm.P_M + sysm.P_T, np.ones(sysm.n_modes))


def test_plain_zoh_matches_analytic_step_response(sysm):
    """Plain ZOH must reproduce the closed-form first-order step response."""
    L, t_on = 400, 0
    x = np.ones(L)
    s = PlainSSM(sysm).run(x)
    k = np.arange(1, L + 1)
    for i in range(sysm.n_modes):
        analytic = sysm.K[i] * (1.0 - sysm.lam[i] ** k)
        np.testing.assert_allclose(s[:, i], analytic, rtol=1e-10, atol=1e-12)


def test_full_state_pc_forced_response_is_exactly_Kx(sysm):
    """Idea 1 destroys the forced dynamics: s = K x exactly, zero-state."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    s = FullStatePC(sysm).run(x)
    np.testing.assert_allclose(s, np.outer(x, sysm.K), rtol=0, atol=1e-12)


def test_full_state_pc_transient_decays_at_rho(sysm):
    """From a non-equilibrium start, the residual decays exactly as rho^k."""
    s0 = np.array([1.0, 1.0])
    s = FullStatePC(sysm).run(np.zeros(200), s0=s0)
    k = np.arange(1, 201)
    for i in range(sysm.n_modes):
        np.testing.assert_allclose(s[:, i], s0[i] * sysm.rho ** k,
                                   rtol=1e-10, atol=1e-14)


def test_projected_pc_preserves_memory_pole_and_changes_tracking_pole(sysm):
    poles = ProjectedStatePC(sysm).discrete_poles()
    np.testing.assert_allclose(poles[0], sysm.lam[0])       # memory unchanged
    np.testing.assert_allclose(poles[1], sysm.rho)          # tracking -> rho


def test_projected_pc_memory_mode_equals_plain_exactly(sysm):
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    np.testing.assert_allclose(ProjectedStatePC(sysm).run(x)[:, 0],
                               PlainSSM(sysm).run(x)[:, 0], rtol=0, atol=0)


def test_projected_pc_tracking_equals_static_bypass(sysm):
    """HOSTILE CONTROL: projected PC tracking is the static equilibrium."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    np.testing.assert_allclose(ProjectedStatePC(sysm).run(x)[:, 1],
                               StaticEquilibriumBypass(sysm).run(x)[:, 1],
                               rtol=0, atol=1e-12)


def test_matched_readout_lead_equals_state_pc_on_tracking_mode(sysm):
    """HOSTILE CONTROL: an output-side lead reproduces state-level PC.

    For a first-order mode the lead's zero cancels the pole exactly when
    alpha = lam / (1 - lam).
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    lead = ReadoutLeadPerMode(sysm).run(x)[:, 1]
    pc = ProjectedStatePC(sysm).run(x)[:, 1]
    np.testing.assert_allclose(lead[1:], pc[1:], rtol=0, atol=1e-12)


def test_matched_alpha_formula(sysm):
    for lam in (0.1, 0.5, 0.9, sysm.lam[0], sysm.lam[1]):
        a = matched_alpha(lam)
        np.testing.assert_allclose(a / (1.0 + a), lam, rtol=1e-12)


def test_lead_is_causal_and_zero_at_first_sample():
    x = np.arange(10.0)
    out = apply_lead(x, 3.0)
    assert out[0] == x[0]
    changed = x.copy(); changed[6:] = -99.0
    np.testing.assert_array_equal(apply_lead(changed, 3.0)[:6], out[:6])


def test_no_parasitic_second_order_state(sysm):
    """The tracking recursion is first order: exactly one state per mode.

    Verified structurally - the output at step k is reproduced from
    (s_{k-1}, x_k, x_{k-1}) alone, with no dependence on x_{k-2}.
    """
    rng = np.random.default_rng(4)
    x = rng.normal(size=50)
    ref = FullStatePC(sysm).run(x)
    perturbed = x.copy(); perturbed[:20] = rng.normal(size=20)
    alt = FullStatePC(sysm).run(perturbed)
    # from k = 22 on, only x_k and x_{k-1} matter, plus a rho^k transient
    resid = np.abs(ref[22:, 1] - alt[22:, 1])
    assert resid.max() < 1e-6, resid.max()


def test_impulse_support_shows_full_pc_has_no_forced_memory(sysm):
    imp = metrics.impulse_response(FullStatePC(sysm), length=200)
    for i in range(sysm.n_modes):
        assert metrics.forced_impulse_support(imp[:, i]) == 1
    imp_plain = metrics.impulse_response(PlainSSM(sysm), length=200)
    assert metrics.forced_impulse_support(imp_plain[:, 0]) > 100


def test_all_mechanisms_are_affine_and_scan_compatible(sysm):
    """Every mechanism must be affine in (state, input) to stay scan-friendly."""
    rng = np.random.default_rng(5)
    x1, x2 = rng.normal(size=300), rng.normal(size=300)
    for cls in (PlainSSM, FullStatePC, ProjectedStatePC, StaticEquilibriumBypass):
        m = cls(sysm)
        lhs = m.run(2.0 * x1 + 3.0 * x2)
        rhs = 2.0 * m.run(x1) + 3.0 * m.run(x2)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-10,
                                   err_msg=cls.__name__)
