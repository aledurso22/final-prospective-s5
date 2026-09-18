import os
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5.gp_coefficients import gp_response_coefficients
from s5.gp_second_order import second_order_generator, second_order_zoh
from s5.ssm import discretize_zoh
from s5.ssm import apply_ssm
from s5.three_arm_recurrences import (native_matched_s5_transition,
                                      zucchet_prospective_s5_transition)


def test_ordinary_prospective_equation_matches_declared_coefficients():
    a = jnp.asarray([-0.3 + 0.2j, -0.7 + 0.1j])
    b = jnp.asarray([[0.4 + 0.1j], [0.2 - 0.3j]])
    t = jnp.asarray([0.6, 1.2])
    got = gp_response_coefficients(a, b, t)
    m = 1 - t * a
    np.testing.assert_allclose(got["a_eff"], a / m, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["b_hist"], b / (m ** 2)[:, None], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["d_x"], (t / m)[:, None] * b, rtol=0, atol=1e-12)


def test_native_core_transition_matches_unmodified_native_zoh():
    Lambda = jnp.asarray([-0.3 + 0.2j, -0.7 + 0.1j])
    B = jnp.asarray([[0.4 + 0.1j], [0.2 - 0.3j]])
    Delta = jnp.asarray([0.8, 0.4])
    state = jnp.asarray([0.1 + 0.2j, -0.2 + 0.1j])
    x = jnp.asarray([0.7])
    bar, bbar = discretize_zoh(Lambda, B, Delta)
    expected = bar * state + bbar @ x
    got = native_matched_s5_transition(state, x, bar, bbar)
    np.testing.assert_array_equal(got, expected)


def test_native_scan_matches_unmodified_native_output():
    Lambda_bar = jnp.asarray([0.8 + 0.1j, 0.6 - 0.2j])
    B_bar = jnp.asarray([[0.2 + 0.1j], [0.3 - 0.1j]])
    inputs = jnp.asarray([[0.7], [-0.2], [0.4]])
    C = jnp.asarray([[0.5 + 0.2j, -0.1 + 0.3j]])
    native = apply_ssm(Lambda_bar, B_bar, C, inputs, True, False)
    state = jnp.zeros(2, dtype=Lambda_bar.dtype)
    outputs = []
    for value in inputs:
        state = native_matched_s5_transition(state, value, Lambda_bar, B_bar)
        outputs.append(2 * (C @ state).real)
    np.testing.assert_allclose(jnp.stack(outputs), native, rtol=0, atol=1e-15)


def test_prospective_transition_uses_its_declared_zoh_state():
    a = jnp.asarray([-0.3 + 0.2j])
    b = jnp.asarray([[0.4 + 0.1j]])
    state = jnp.asarray([0.1 + 0.2j])
    value = jnp.asarray([0.7])
    T = jnp.asarray([0.6])
    next_state, observed = zucchet_prospective_s5_transition(
        state, value, a, b, T)
    coefficients = gp_response_coefficients(a, b, T)
    a_eff, b_hist, d_x = (coefficients["a_eff"], coefficients["b_hist"],
                          coefficients["d_x"])
    expected_state = jnp.exp(a_eff) * state + (
        (jnp.expm1(a_eff) / a_eff)[:, None] * b_hist) @ value
    np.testing.assert_allclose(next_state, expected_state, rtol=0, atol=1e-12)
    np.testing.assert_allclose(observed, expected_state + d_x[:, 0] * value[0],
                               rtol=0, atol=1e-12)


def test_generalized_equation_has_declared_M_gamma_T_generator():
    a = jnp.asarray([-0.4 + 0.15j])
    b = jnp.asarray([[0.25 - 0.2j]])
    t, mass, gamma = jnp.asarray([0.8]), jnp.asarray([0.3]), jnp.asarray([1.0])
    got_f, got_b = second_order_generator(a, b, t, mass)
    expected_f = jnp.asarray([[[t[0] * a[0] / mass[0], 1 / mass[0]],
                               [(1 - t[0] / mass[0]) * a[0],
                                -gamma[0] / mass[0]]]])
    expected_b = jnp.asarray([[[t[0] * b[0, 0] / mass[0]],
                               [(1 - t[0] / mass[0]) * b[0, 0]]]])
    np.testing.assert_allclose(got_f, expected_f, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got_b, expected_b, rtol=0, atol=1e-12)


def test_generalized_M_zero_boundary_converges_to_ordinary_prospective():
    a = jnp.asarray([-0.2 + 0.1j])
    b = jnp.asarray([[0.3 + 0.2j]])
    t = jnp.asarray([0.7])
    ordinary = gp_response_coefficients(a, b, t)
    sequence = jnp.asarray([[0.7], [-0.2], [0.4]])
    ordinary_state = jnp.zeros(1, dtype=a.dtype)
    ordinary_outputs = []
    for value in sequence:
        ordinary_state = ordinary["a_bar"] * ordinary_state + ordinary["b_bar"][:, 0] * value[0]
        ordinary_outputs.append(ordinary_state + ordinary["d_x"][:, 0] * value[0])
    ordinary_outputs = jnp.stack(ordinary_outputs)
    for ratio in (1e-2, 1e-3):
        generalized = second_order_zoh(a, b, t, ratio * t)
        A_bar, B_bar = generalized["A_bar"], generalized["B_bar"]
        state = jnp.zeros((1, 2), dtype=a.dtype)
        outputs = []
        for value in sequence:
            state = (A_bar @ state[..., None])[..., 0] + B_bar @ value
            outputs.append(state[:, 0])
        np.testing.assert_allclose(jnp.stack(outputs), ordinary_outputs,
                                   rtol=2e-2, atol=2e-3)


def test_three_arm_names_are_exactly_three():
    from experiments.s5_three_arm_full.runner import ARM_ORDER, SCIENTIFIC_NAMES
    assert len(ARM_ORDER) == 3
    assert list(SCIENTIFIC_NAMES.values()) == [
        "Native S5 recurrence under the shared stability constraint",
        "Zucchet prospective S5 recurrence",
        "Generalized prospective S5 recurrence (M,γ,T)",
    ]
