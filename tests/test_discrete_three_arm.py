import jax
import jax.numpy as jnp
import numpy as np

from s5.discrete_recurrence import (companion_radius, generalized_coefficients,
                                    scan_companion, scan_companion_sequential,
                                    zucchet_coefficients)
from s5.train_helpers import no_bc_decay_group


def test_ordinary_coefficients_match_hand_derivation():
    a = jnp.asarray([0.8 + 0.2j, 0.6 - 0.1j])
    b = jnp.asarray([[0.3 + 0.1j], [0.2 - 0.2j]])
    t = jnp.asarray([0.05, 0.2])
    a1, a2, c1, c2 = zucchet_coefficients(a, b, t)
    k = t
    np.testing.assert_allclose(a1, (1-k) + (1+k)*a)
    np.testing.assert_allclose(a2, (k-1) - k*a)
    np.testing.assert_allclose(c1, (1+k)[:, None]*b)
    np.testing.assert_allclose(c2, -k[:, None]*b)


def test_generalized_zero_mass_zero_gamma_is_exact_ordinary_boundary():
    a = jnp.asarray([0.8 + 0.2j])
    b = jnp.asarray([[0.3 + 0.1j]])
    t = jnp.asarray([0.05])
    ordinary = zucchet_coefficients(a, b, t)
    generalized = generalized_coefficients(a, b, t, jnp.zeros(1), jnp.zeros(1))
    for left, right in zip(ordinary, generalized):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)


def test_associative_scan_matches_sequential_rollout():
    a1 = jnp.asarray([0.7 + 0.1j])
    a2 = jnp.asarray([-0.1 + 0.02j])
    c1 = jnp.asarray([[0.2 + 0.1j]])
    c2 = jnp.asarray([[-0.05 + 0.02j]])
    inputs = jnp.asarray([[1.0], [0.2], [-0.4], [0.7]], dtype=jnp.float32)
    scanned = scan_companion(a1, a2, c1, c2, inputs)
    s, previous_state, previous_input = 0j, 0j, 0j
    expected = []
    for value in inputs[:, 0]:
        next_state = (a1[0]*s + a2[0]*previous_state + c1[0, 0]*value
                      + c2[0, 0]*previous_input)
        expected.append(next_state)
        previous_state, previous_input, s = s, value, next_state
    np.testing.assert_allclose(scanned[:, 0], jnp.asarray(expected), rtol=2e-6, atol=2e-6)


def test_rematerialized_scan_matches_associative_scan_and_gradients():
    a1 = jnp.asarray([0.7 + 0.1j])
    a2 = jnp.asarray([-0.1 + 0.02j])
    c1 = jnp.asarray([[0.2 + 0.1j]])
    c2 = jnp.asarray([[-0.05 + 0.02j]])
    inputs = jnp.asarray([[1.0], [0.2], [-0.4], [0.7]], dtype=jnp.float32)
    expected = scan_companion(a1, a2, c1, c2, inputs)
    actual = scan_companion_sequential(a1, a2, c1, c2, inputs)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def objective(first):
        return jnp.real(scan_companion_sequential(
            first, a2, c1, c2, inputs)).sum()

    def reference(first):
        return jnp.real(scan_companion(first, a2, c1, c2, inputs)).sum()

    np.testing.assert_allclose(jax.grad(objective)(a1), jax.grad(reference)(a1),
                               rtol=2e-6, atol=2e-6)


def test_companion_radius_and_parameter_group_contract():
    radius = companion_radius(jnp.asarray([0.5 + 0.1j]), jnp.asarray([-0.1 + 0.02j]))
    assert bool(jnp.isfinite(radius).all())
    for name in ("prospective_T_raw", "generalized_T_raw",
                 "generalized_rho_raw", "generalized_gamma_raw"):
        assert no_bc_decay_group(name) == "ssm"


def test_exactly_three_scientific_arms():
    from experiments.s5_three_arm_full.runner import ARM_ORDER, SCIENTIFIC_NAMES
    assert len(ARM_ORDER) == 3
    assert set(SCIENTIFIC_NAMES.values()) == {
        "Native S5",
        "Zucchet prospective dynamics — finite-difference realization",
        "generalized prospective dynamics (M,γ,T) — finite-difference realization",
    }
