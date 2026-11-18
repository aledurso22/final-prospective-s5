"""Independent equations and finite-response diagnostics, without training."""
import numpy as np
import pytest

from experiments.s5_proximal_memory_mix.diagnostic import (
    finite_responses, modal_residues, positive_real_margin, reference,
    sensitivity_variants)


def test_same_regularization_budget_for_every_predeclared_mix():
    for _, kwargs in sensitivity_variants():
        if kwargs["variant"] != "mix":
            continue
        fast = kwargs.get("fast_ratio", .1)
        slow = kwargs.get("slow_ratio", 1.)
        weight = kwargs.get("slow_weight", 4/9)
        assert (1-weight)*fast+weight*slow == pytest.approx(.5)
        assert np.min(positive_real_margin(np.geomspace(1e-8, 1e8, 1001),
                                          fast, slow, weight, .5)) >= -1e-15


@pytest.mark.parametrize("variant", ["native", "proximal", "mix"])
def test_reference_preserves_equilibrium_and_modal_transfer(variant):
    k = .003+.4j
    A, b, Ad, bd, h = reference(k, variant)
    assert (-np.linalg.solve(A, b)[0])*k == pytest.approx(1)
    assert (np.linalg.solve(np.eye(len(A))-Ad, bd)[0])*k == pytest.approx(1)
    poles, residue, _, _ = modal_residues(A, b)
    for p in (.2+.1j, 0., -.2j):
        direct = np.linalg.solve(p*np.eye(len(A))-A, b)[0]
        assert np.sum(residue/(p-poles)) == pytest.approx(direct, rel=1e-10)


def test_mix_transfer_and_dc_delay_include_the_same_proximal_force():
    k = .01+.4j
    A, b, _, _, h = reference(k)
    fast, slow, weight, mu = .1*h, h, 4/9, .5*h
    for p in (0., .1j, -.4j, 2j):
        K = (1-weight)/(1+fast*p)+weight/(1+slow*p)
        expected = K/((1+mu*k)*p+k*K)
        assert np.linalg.solve(p*np.eye(3)-A, b)[0] == pytest.approx(expected)
    # Delay of the closed state transfer is native+mu. K's own delay cancels
    # between input and residual feedback at first order; it is not added twice.
    dc = -np.linalg.solve(A, b)[0]
    delay = np.linalg.solve(A, np.linalg.solve(A, b))[0]/dc
    assert delay == pytest.approx(1/k+mu)


def test_finite_native_metrics_are_exact_and_include_all_tokens():
    ks = np.asarray([.01+.2j, .03-.4j])
    refs = [reference(k, "native") for k in ks]
    Ad, bd = np.stack([r[2] for r in refs]), np.stack([r[3] for r in refs])
    report = finite_responses(Ad, bd, ks, 50)
    np.testing.assert_allclose(report["impulse_relative_l2_vs_native"], 0, atol=2e-14)
    np.testing.assert_allclose(report["impulse_energy_over_native"], 1, atol=2e-14)
    np.testing.assert_allclose(report["step_final_relative_error"], abs(np.exp(-50*ks)))


def test_below_uniform_bound_has_rhp_operator_with_imaginary_pole():
    fast, slow, weight, mu = .1, 1., 4/9, .49
    p = .01j
    K = (1-weight)/(1+fast*p)+weight/(1+slow*p)
    kappa = -1/(mu+K/p)
    assert kappa.real > 0
    assert abs((1+mu*kappa)*p+kappa*K) < 1e-14


@pytest.mark.parametrize("k", [.001+.8j, .003-1.2j, 2.+3j])
def test_declared_mix_stable_in_both_domains(k):
    A, _, Ad, _, _ = reference(k)
    assert max(np.linalg.eigvals(A).real) < 0
    assert max(abs(np.linalg.eigvals(Ad))) < 1
