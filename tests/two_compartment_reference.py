"""Frozen pure-Python reference for the two-compartment recurrence.

Written from the recurrence itself, not from `s5/factored_recurrence.py`,
so the JAX implementation has something independent to be compared with.
No JAX: this runs on a laptop.
"""

import cmath

ROOT_FLOOR = 1e-30


def sequential(a1, a2, drive):
    """s_t = a1 s_{t-1} + a2 s_{t-2} + drive_t, zero prehistory.

    This is the definition. Everything else has to reproduce it.
    """
    states, previous, older = [], 0j, 0j
    for value in drive:
        current = a1 * previous + a2 * older + value
        states.append(current)
        previous, older = current, previous
    return states


def naive_roots(a1, a2):
    """(a1 +- sqrt(a1^2 + 4 a2))/2, the form that loses the small root."""
    root = cmath.sqrt(a1 * a1 + 4.0 * a2)
    return (a1 + root) / 2.0, (a1 - root) / 2.0


def companion_roots(a1, a2):
    """The stable form: add constructively, then divide by the product."""
    root = cmath.sqrt(a1 * a1 + 4.0 * a2)
    sign = 1.0 if (a1.conjugate() * root).real >= 0 else -1.0
    major = (a1 + sign * root) / 2.0
    if abs(major) < ROOT_FLOOR:
        return 0j, 0j
    return major, -a2 / major


def factored(a1, a2, drive):
    """v_t = r1 v_{t-1} + d_t ; s_t = r2 s_{t-1} + v_t."""
    first, second = companion_roots(a1, a2)

    def scan(pole, values):
        out, carry = [], 0j
        for value in values:
            carry = pole * carry + value
            out.append(carry)
        return out

    return scan(second, scan(first, drive))


def drive_from_inputs(c1, c2, inputs):
    """c1 x_t + c2 x_{t-1}, zero prehistory, scalar mode."""
    return [c1 * value + (c2 * inputs[index - 1] if index else 0j)
            for index, value in enumerate(inputs)]


def transfer(a1, a2, z):
    """1 / (1 - a1 z^-1 - a2 z^-2)."""
    return 1.0 / (1.0 - a1 / z - a2 / (z * z))


def factored_transfer(a1, a2, z):
    """1 / ((1 - r1 z^-1)(1 - r2 z^-1))."""
    first, second = companion_roots(a1, a2)
    return 1.0 / ((1.0 - first / z) * (1.0 - second / z))


def companion_radius(a1, a2):
    """The existing `companion_radius`, reproduced in plain Python."""
    root = cmath.sqrt(a1 * a1 + 4.0 * a2)
    return max(abs((a1 + root) / 2.0), abs((a1 - root) / 2.0))
