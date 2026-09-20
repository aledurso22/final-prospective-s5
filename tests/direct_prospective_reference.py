"""FROZEN sequential oracles for the direct prospective recurrences. TESTS ONLY.

Nothing in `s5/` and no launcher may import this. Every function is an
independent transcription of the derivation, written as an obvious loop.
"""

import jax.numpy as np


def matched_scalar_coefficients(tau, mass, h=1.0):
    q = mass + h * tau
    return ((2.0 * mass + h * tau - h * h) / q, mass / q,
            (mass + h * tau + h * h) / q, -(2.0 * mass + h * tau) / q,
            mass / q)


def partially_matched_scalar_coefficients(response, mass, gamma, h=1.0):
    big_gamma = gamma + response
    q = mass + h * big_gamma
    return ((2.0 * mass + h * big_gamma - h * h) / q, mass / q,
            h * (h + response) / q, -h * response / q)


def _targets(lambda_bar, b_bar, tau, construction, h):
    if construction == "professor":
        return lambda_bar, b_bar
    k = (tau / h).astype(lambda_bar.dtype)
    return 1.0 + k * (lambda_bar - 1.0), k[..., None] * b_bar


def matched_sequential(lambda_bar, b_bar, inputs, tau, mass,
                       construction="professor", reverse=False, h=1.0):
    """s_{t+1} = a s_t - b s_{t-1} + c0 f_t + c1 f_{t-1} + c2 f_{t-2},
    forming f each step, with zero state and input prehistory."""
    sequence = inputs[::-1] if reverse else inputs
    a, b, c0, c1, c2 = matched_scalar_coefficients(tau, mass, h)
    F, G = _targets(lambda_bar, b_bar, tau, construction, h)
    zero_state = np.zeros_like(lambda_bar)
    zero_input = np.zeros_like(sequence[0])
    s0 = s1 = zero_state
    x1 = x2 = zero_input
    f1 = f2 = zero_state
    states = []
    for x0 in sequence:
        f0 = F * s0 + G @ x0
        nxt = (a.astype(f0.dtype) * s0 - b.astype(f0.dtype) * s1
               + c0.astype(f0.dtype) * f0 + c1.astype(f0.dtype) * f1
               + c2.astype(f0.dtype) * f2)
        states.append(nxt)
        s0, s1 = nxt, s0
        f1, f2 = f0, f1
        x1, x2 = x0, x1
    stacked = np.stack(states, axis=0)
    return stacked[::-1] if reverse else stacked


def partially_matched_sequential(lambda_bar, b_bar, inputs, response, mass,
                                 gamma, construction="professor",
                                 reverse=False, h=1.0):
    """s_{t+1} = a s_t - b s_{t-1} + c0 f_t + c1 f_{t-1}."""
    sequence = inputs[::-1] if reverse else inputs
    a, b, c0, c1 = partially_matched_scalar_coefficients(response, mass,
                                                         gamma, h)
    F, G = _targets(lambda_bar, b_bar, response, construction, h)
    zero_state = np.zeros_like(lambda_bar)
    s0 = s1 = zero_state
    f1 = zero_state
    states = []
    for x0 in sequence:
        f0 = F * s0 + G @ x0
        nxt = (a.astype(f0.dtype) * s0 - b.astype(f0.dtype) * s1
               + c0.astype(f0.dtype) * f0 + c1.astype(f0.dtype) * f1)
        states.append(nxt)
        s0, s1 = nxt, s0
        f1 = f0
    stacked = np.stack(states, axis=0)
    return stacked[::-1] if reverse else stacked


def native_sequential(lambda_bar, b_bar, inputs):
    """Native S5, for the impulse-response comparison."""
    state = np.zeros_like(lambda_bar)
    states = []
    for x in inputs:
        state = lambda_bar * state + b_bar @ x
        states.append(state)
    return np.stack(states, axis=0)


def exact_companion_radius(coefficients_A):
    """max |root| on the HOST, from numpy, for the CPU cross-check."""
    import numpy

    order = len(coefficients_A)
    arrays = [numpy.asarray(value) for value in coefficients_A]
    radii = []
    for mode in range(arrays[0].shape[0]):
        poly = [1.0] + [-complex(array[mode]) for array in arrays]
        radii.append(float(numpy.max(numpy.abs(numpy.roots(poly)))))
    return numpy.asarray(radii)
