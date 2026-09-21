"""A frozen sequential reference for the professor's prospective cell.

Pure Python complex arithmetic, no JAX, written straight from the paper's
equations rather than from `s5/prospective_euler.py`. It exists so the
mathematics can be checked on a laptop and so the JAX implementation has
something independent to be compared against.
"""

import cmath

DT = 1.0


def companion(a, epsilon):
    """(A1, A2) exactly as the note writes them."""
    return (1 - epsilon) + (1 + epsilon) * a, -a


def roots(a, epsilon):
    first, _ = companion(a, epsilon)
    spread = cmath.sqrt(first * first - 4 * a)
    return (first + spread) / 2, (first - spread) / 2


def spectral_radius(a, epsilon):
    plus, minus = roots(a, epsilon)
    return max(abs(plus), abs(minus))


def epsilon_max(a):
    """2(1-|a|^2)|1-a|^2 / |2a - 1 - |a|^2|^2."""
    magnitude = abs(a) ** 2
    curvature = 2 * a - 1 - magnitude
    return 2 * (1 - magnitude) * abs(1 - a) ** 2 / abs(curvature) ** 2


def sequential(a, b, inputs, epsilon):
    """s_t = A1 s_{t-1} + A2 s_{t-2} + (1+e) b x_{t-1} - b x_{t-2}.

    Straight from the note, with zero prehistory. This is the definition
    every other route has to reproduce.
    """
    first, second = companion(a, epsilon)
    states, previous, older = [], 0j, 0j
    for index in range(len(inputs)):
        back_one = b * inputs[index - 1] if index >= 1 else 0j
        back_two = b * inputs[index - 2] if index >= 2 else 0j
        value = (first * previous + second * older
                 + (1 + epsilon) * back_one - back_two)
        states.append(value)
        previous, older = value, previous
    return states


def native_sequential(a, b, inputs):
    """s_t = a s_{t-1} + b x_t, the untouched S5 recurrence."""
    states, carry = [], 0j
    for value in inputs:
        carry = a * carry + b * value
        states.append(carry)
    return states


def factored(a, b, inputs, epsilon):
    """The same cell run as two first-order scans over its own roots."""
    plus, minus = roots(a, epsilon)
    projected = [b * value for value in inputs]
    excitation = []
    for index in range(len(inputs)):
        back_one = projected[index - 1] if index >= 1 else 0j
        back_two = projected[index - 2] if index >= 2 else 0j
        excitation.append((1 + epsilon) * back_one - back_two)

    def scan(pole, sequence):
        out, carry = [], 0j
        for value in sequence:
            carry = pole * carry + value
            out.append(carry)
        return out

    return scan(minus, scan(plus, excitation))
