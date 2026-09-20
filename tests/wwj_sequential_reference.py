"""FROZEN sequential oracle for the WWJ recurrence. TESTS ONLY.

Nothing in `s5/` and no launcher may import it: the model never runs this
code. Its only users are the tests and the pre-training performance gate,
which checks the scan against it before any training is authorized. It is an independent transcription of the derivation in
`docs/S5_WWJ_PROSPECTIVE.md`, not a call into the production coefficients.
"""

import jax
import jax.numpy as np


def scalar_coefficients(tau, mass, h=1.0):
    """(a, b, c0, c1, c2), written out again from the derivation."""
    q = mass + h * tau
    a = (2.0 * mass + h * tau - h * h) / q
    b = mass / q
    c0 = (mass + h * tau + h * h) / q
    c1 = -(2.0 * mass + h * tau) / q
    c2 = mass / q
    return a, b, c0, c1, c2


def sequential_states(lambda_bar, b_bar, tau, mass, inputs, reverse=False,
                      h=1.0):
    """s_{t+1} for t = 0..L-1 by an explicit loop, zero prehistory.

    The target is formed each step as f_t = F_tau s_t + G_tau x_t, exactly as
    in the derivation, rather than through the substituted A/C coefficients,
    so agreement with the production path also checks the substitution.
    """
    sequence = inputs[::-1] if reverse else inputs
    a, b, c0, c1, c2 = scalar_coefficients(tau, mass, h)
    k = (tau / h).astype(lambda_bar.dtype)
    F = 1.0 + k * (lambda_bar - 1.0)
    G = k[..., None] * b_bar
    zero_state = np.zeros_like(lambda_bar)
    zero_input = np.zeros_like(sequence[0])
    s0 = s1 = s2 = zero_state
    x1 = x2 = zero_input
    states = []
    for x0 in sequence:
        f0 = F * s0 + G @ x0
        f1 = F * s1 + G @ x1
        f2 = F * s2 + G @ x2
        s_next = (a.astype(f0.dtype) * s0 - b.astype(f0.dtype) * s1
                  + c0.astype(f0.dtype) * f0 + c1.astype(f0.dtype) * f1
                  + c2.astype(f0.dtype) * f2)
        states.append(s_next)
        s0, s1, s2 = s_next, s0, s1
        x1, x2 = x0, x1
    stacked = np.stack(states, axis=0)
    return stacked[::-1] if reverse else stacked


def sequential_scan(A0, A1, A2, drive):
    """The order-3 recurrence alone, given its coefficients and drive."""
    s0 = s1 = s2 = np.zeros_like(A0)
    states = []
    for d in drive:
        s_next = A0 * s0 + A1 * s1 + A2 * s2 + d
        states.append(s_next)
        s0, s1, s2 = s_next, s0, s1
    return np.stack(states, axis=0)


def exact_companion_radius(A0, A1, A2):
    """max |root| of z^3 - A0 z^2 - A1 z - A2 on the HOST, for comparison."""
    import numpy

    radii = []
    for a0, a1, a2 in zip(numpy.asarray(A0), numpy.asarray(A1),
                          numpy.asarray(A2)):
        roots = numpy.roots([1.0, -a0, -a1, -a2])
        radii.append(float(numpy.max(numpy.abs(roots))))
    return numpy.asarray(radii)


__all__ = ["scalar_coefficients", "sequential_states", "sequential_scan",
           "exact_companion_radius", "jax"]
