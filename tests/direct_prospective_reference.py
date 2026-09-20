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


def _targets(lambda_bar, b_bar, tau, target_construction, h):
    if target_construction == "professor_linear_target":
        return lambda_bar, b_bar
    if target_construction == "native_matched_target":
        k = (tau / h).astype(lambda_bar.dtype)
        return 1.0 + k * (lambda_bar - 1.0), k[..., None] * b_bar
    raise ValueError(f"unknown target construction: {target_construction!r}")


def matched_sequential(lambda_bar, b_bar, inputs, tau, mass,
                       target_construction="professor_linear_target",
                       reverse=False, h=1.0):
    """s_{t+1} = a s_t - b s_{t-1} + c0 f_t + c1 f_{t-1} + c2 f_{t-2},
    forming f each step, with zero state and input prehistory."""
    sequence = inputs[::-1] if reverse else inputs
    a, b, c0, c1, c2 = matched_scalar_coefficients(tau, mass, h)
    F, G = _targets(lambda_bar, b_bar, tau, target_construction, h)
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
                                 gamma,
                                 target_construction="professor_linear_target",
                                 reverse=False, h=1.0):
    """s_{t+1} = a s_t - b s_{t-1} + c0 f_t + c1 f_{t-1}."""
    sequence = inputs[::-1] if reverse else inputs
    a, b, c0, c1 = partially_matched_scalar_coefficients(response, mass,
                                                         gamma, h)
    F, G = _targets(lambda_bar, b_bar, response, target_construction, h)
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


def professor_tss_sequential(lambda_bar, b_bar, inputs, tau,
                             target_construction="professor_linear_target",
                             reverse=False, h=1.0):
    """The M = 0 PROFESSOR_TSS two-state recurrence, written out directly:
    s_{t+1} = (1 - h/tau) s_t + (1 + h/tau) f_t - f_{t-1}."""
    sequence = inputs[::-1] if reverse else inputs
    a, c0, c1 = 1.0 - h / tau, 1.0 + h / tau, -np.ones_like(tau)
    F, G = _targets(lambda_bar, b_bar, tau, target_construction, h)
    s0 = np.zeros_like(lambda_bar)
    f1 = np.zeros_like(lambda_bar)
    states = []
    for x0 in sequence:
        f0 = F * s0 + G @ x0
        nxt = (a.astype(f0.dtype) * s0 + c0.astype(f0.dtype) * f0
               + c1.astype(f0.dtype) * f1)
        states.append(nxt)
        s0, f1 = nxt, f0
    stacked = np.stack(states, axis=0)
    return stacked[::-1] if reverse else stacked


#: measured calibration of the doubling scan's rounding, from
#: docs/analysis/direct_prospective_conditioning.txt: the observed relative
#: error never exceeded 300 * eps * peak^2, so 1000 leaves ~3x of margin
DOUBLING_SCAN_SAFETY = 1000.0


def doubling_scan_peak_norm(coefficients_A, length):
    """max_k ||H^(2^k)||_F over the levels the scan actually performs.

    This is the quantity that governs the scan's rounding: each level squares
    the operator, and ||H^(2^k)|| can be far larger than rho^(2^k) for a
    NON-NORMAL companion, so the squaring step cancels catastrophically even
    when the recurrence is stable.
    """
    import numpy

    order = len(coefficients_A)
    arrays = [numpy.asarray(value) for value in coefficients_A]
    modes = arrays[0].shape[0]
    operator = numpy.zeros((modes, order, order), dtype=numpy.complex128)
    for index, array in enumerate(arrays):
        operator[:, 0, index] = array
    for index in range(order - 1):
        operator[:, index + 1, index] = 1.0
    peak = float(numpy.max(numpy.linalg.norm(operator, axis=(1, 2))))
    distance = 1
    while distance < length:
        operator = operator @ operator
        peak = max(peak, float(numpy.max(numpy.linalg.norm(operator,
                                                           axis=(1, 2)))))
        distance *= 2
    return peak


def doubling_scan_tolerance(coefficients_A, length, epsilon=2.220446049250313e-16,
                            safety=DOUBLING_SCAN_SAFETY):
    """A tolerance DERIVED from measured conditioning, not chosen to pass.

    The tree scan and the sequential oracle associate the same sum
    differently, so bitwise equality is not expected. The scan's relative
    error is governed by the squaring step, whose condition number is
    ||H^(2^k)||^2 / ||H^(2^(k+1))||; with the powers decaying this is
    dominated by the peak transient norm squared. Measured against EXACT
    rational ground truth at the tau = 1000 cell the constant stayed below
    300 (docs/analysis/), hence

        tolerance = safety * epsilon * peak^2,

    floored at 64 * epsilon so a well-conditioned fixture is still held to
    machine precision.
    """
    peak = doubling_scan_peak_norm(coefficients_A, length)
    return max(safety * epsilon * peak * peak, 64.0 * epsilon)


def compare_finite_prefix(fast, slow):
    """The diagnostic a divergent fixture needs, reported not hidden.

    Returns the finite masks, the first nonfinite token and mode on each
    side, the length of the COMMON FINITE PREFIX and the maximum absolute
    and relative error over it. Matching NaNs prove nothing; agreement over
    the finite prefix plus an identical first-nonfinite token is what
    distinguishes an unstable MODEL from a broken SCAN.
    """
    import numpy

    fast_array = numpy.asarray(fast)
    slow_array = numpy.asarray(slow)
    fast_finite = numpy.isfinite(fast_array)
    slow_finite = numpy.isfinite(slow_array)
    length = fast_array.shape[0]

    def first_bad(mask):
        rows = numpy.where(~mask.all(axis=1))[0]
        if rows.size == 0:
            return None, None
        token = int(rows[0])
        mode = int(numpy.where(~mask[token])[0][0])
        return token, mode

    fast_token, fast_mode = first_bad(fast_finite)
    slow_token, slow_mode = first_bad(slow_finite)
    cut = min(fast_token if fast_token is not None else length,
              slow_token if slow_token is not None else length)
    prefix_fast, prefix_slow = fast_array[:cut], slow_array[:cut]
    scale = float(numpy.max(numpy.abs(prefix_slow))) if cut else 0.0
    absolute = (float(numpy.max(numpy.abs(prefix_fast - prefix_slow)))
                if cut else 0.0)
    return {
        "length": length,
        "finite_masks_identical": bool(numpy.array_equal(fast_finite,
                                                         slow_finite)),
        "first_nonfinite_fast": {"token": fast_token, "mode": fast_mode},
        "first_nonfinite_slow": {"token": slow_token, "mode": slow_mode},
        "same_first_nonfinite": (fast_token, fast_mode) == (slow_token,
                                                            slow_mode),
        "common_finite_prefix": cut,
        "max_absolute_error_on_prefix": absolute,
        "max_relative_error_on_prefix": (absolute / scale if scale else 0.0),
        "max_abs_value_on_prefix": scale,
    }


def sequential_scan(coefficients_A, drive):
    """s_t = sum_i A_i s_{t-i} + d_t, zero prehistory: the obvious loop."""
    order = len(coefficients_A)
    history = [np.zeros_like(drive[0])] * order
    states = []
    for d in drive:
        value = sum(a * h for a, h in zip(coefficients_A, history)) + d
        states.append(value)
        history = [value] + history[:-1]
    return np.stack(states, axis=0)


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
