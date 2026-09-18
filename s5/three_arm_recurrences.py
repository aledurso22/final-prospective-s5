"""The three recurrence transitions used by the full-training experiment.

`s5/ssm.py` remains the native S5 reference. This module contains only the
prospective transition mathematics; it deliberately does not contain model
construction, optimizer, dataset, or experiment policy.
"""

import jax
import jax.numpy as np
from jax.scipy.linalg import expm


def native_matched_s5_transition(state, input_value, Lambda_bar, B_bar):
    """Native matched S5: ``s_next = Lambda_bar*s + B_bar*x``."""
    return Lambda_bar * state + B_bar @ input_value


def zucchet_prospective_s5_coefficients(a, b, T):
    """Zucchet prospective S5 recurrence, the exact M=0, gamma=1 boundary.

    (I - T*a) s_dot = a*s + b*x + T*b*x_dot, with s = u + d*x.
    """
    denominator = 1.0 - T.astype(a.dtype) * a
    a_eff = a / denominator
    b_hist = b / denominator[:, None] ** 2
    d_x = (T.astype(a.dtype) / denominator)[:, None] * b
    return a_eff, b_hist, d_x


def zucchet_prospective_s5_transition(state, input_value, a, b, T):
    """One held-token Zucchet transition and its observed state ``s``."""
    a_eff, b_hist, d_x = zucchet_prospective_s5_coefficients(a, b, T)
    next_state = np.exp(a_eff) * state + (
        (np.expm1(a_eff) / a_eff)[:, None] * b_hist) @ input_value
    observed = next_state + d_x @ input_value
    return next_state, observed


def generalized_prospective_s5_generator(a, b, M, gamma, T):
    """Generalized prospective recurrence generator and held-input column.

    M*s_ddot + (gamma - T*a)*s_dot - a*s = b*x + T*b*x_dot,
    w = M*s_dot - T*(a*s + b*x), with output state s.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    M = np.asarray(M).astype(a.real.dtype)
    gamma = np.asarray(gamma).astype(a.real.dtype)
    T = np.asarray(T).astype(a.real.dtype)
    A = np.stack([
        np.stack([T * a / M, 1.0 / M], axis=-1),
        np.stack([(1.0 - T / M) * a, -gamma / M], axis=-1),
    ], axis=-2)
    B = np.stack([T[:, None] * b / M[:, None],
                  (1.0 - T / M)[:, None] * b], axis=-2)
    return A, B


def generalized_prospective_s5_zoh(a, b, M, gamma, T):
    """Exact unit-interval ZOH for the generalized recurrence."""
    A, B = generalized_prospective_s5_generator(a, b, M, gamma, T)
    augmented = np.zeros((A.shape[0], 2 + B.shape[-1], 2 + B.shape[-1]),
                         dtype=A.dtype)
    augmented = augmented.at[:, :2, :2].set(A)
    augmented = augmented.at[:, :2, 2:].set(B)
    exponential = jax.vmap(expm)(augmented)
    return exponential[:, :2, :2], exponential[:, :2, 2:]


def generalized_prospective_s5_transition(state, input_value, a, b, M,
                                          gamma, T):
    """One held-token generalized transition; output is the first state."""
    A_bar, B_bar = generalized_prospective_s5_zoh(a, b, M, gamma, T)
    next_state = A_bar @ state[..., None]
    next_state = next_state[..., 0] + B_bar @ input_value
    return next_state, next_state[:, 0]
