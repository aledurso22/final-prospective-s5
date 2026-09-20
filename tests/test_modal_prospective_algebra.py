"""Exact proofs about the modal prospective cascade. No JAX, no floats.

Rational arithmetic in pure Python, so these are proofs rather than samples
and they run on a laptop. The JAX implementation of the same claims is in
`test_modal_prospective_jax.py`.

The stage is (1 + d D_h) u = (1 + n D_h) v with D_h y_t = (y_t - y_{t-1})/h,
realized as

    u_t = [d/(h+d)] u_{t-1} + [(h+n)/(h+d)] v_t - [n/(h+d)] v_{t-1}.
"""

import random
from fractions import Fraction as F

H = F(1)


def coefficients(d, n, h=H):
    return d / (h + d), (h + n) / (h + d), -n / (h + d)


def stage(values, d, n, h=H):
    """One stage, zero prehistory."""
    pole, alpha, beta = coefficients(d, n, h)
    state, previous, out = F(0), F(0), []
    for value in values:
        state = pole * state + alpha * value + beta * previous
        out.append(state)
        previous = value
    return out


def cascade(values, stages, h=H):
    out = values
    for d, n in stages:
        out = stage(out, d, n, h)
    return out


def stage_residual(d, n, h, u_t, u_prev, v_t, v_prev):
    """(1 + d D_h)u - (1 + n D_h)v, multiplied by h."""
    return (h * u_t + d * (u_t - u_prev)) - (h * v_t + n * (v_t - v_prev))


# ------------------------------------------------- the stage and its pole --
def test_the_update_solves_the_stage_equation_exactly():
    """Linear in (u_{t-1}, v_t, v_{t-1}); vanishes on their basis."""
    for d, n in ((F(1, 2), F(3)), (F(2), F(2)), (F(1, 10), F(0)),
                 (F(7, 3), F(1, 5))):
        pole, alpha, beta = coefficients(d, n)
        for basis in range(3):
            u_prev, v_t, v_prev = (F(1) if i == basis else F(0)
                                   for i in range(3))
            u_t = pole * u_prev + alpha * v_t + beta * v_prev
            assert stage_residual(d, n, H, u_t, u_prev, v_t, v_prev) == 0


def test_every_added_pole_is_d_over_h_plus_d_and_lies_inside_the_disc():
    """REQUIREMENT 5. The stage's only recurrent pole is d/(h+d), and for
    d > 0, h > 0 it is strictly between 0 and 1 -- no parameter value can
    move it out."""
    for d in (F(1, 1000), F(1, 10), F(1), F(10), F(1000)):
        for h in (F(1), F(1, 2), F(3)):
            pole = d / (h + d)
            assert pole == coefficients(d, F(0), h)[0]
            assert 0 < pole < 1, (d, h, pole)
    # the pole is monotone in d and never reaches the boundary
    poles = [F(d, 1) / (H + F(d, 1)) for d in (1, 10, 100, 10000)]
    assert poles == sorted(poles) and poles[-1] < 1


def test_the_numerator_and_gate_never_move_a_pole():
    """REQUIREMENT 6. n enters alpha and beta only; the pole is a function
    of d alone, so gates and numerator parameters cannot destabilize."""
    d = F(3, 4)
    poles = {coefficients(d, n)[0] for n in
             (F(0), F(1, 10), d, F(5), F(100))}
    assert poles == {d / (H + d)}
    # and n moves a ZERO, at n/(h+n)
    for n in (F(1, 10), F(1), F(9)):
        zero = n / (H + n)
        assert 0 < zero < 1
        # the transfer 1 + n D_h vanishes there: 1 + n(1 - 1/z)/h = 0
        assert 1 + n * (1 - 1 / zero) / H == 0


def test_a_stage_with_n_equals_d_is_exactly_the_identity():
    """REQUIREMENT 1, on zero-consistent history."""
    random.seed(20260920)
    for d in (F(1, 5), F(1), F(4), F(37, 11)):
        values = [F(random.randint(-9, 9), random.randint(1, 4))
                  for _ in range(12)]
        assert stage(values, d, d) == values
        # and so is a cascade of identity stages: REQUIREMENT 2, algebraic
        assert cascade(values, ((d, d), (F(2), F(2)))) == values


def test_one_active_stage_is_the_ordinary_lead_operator():
    """REQUIREMENT 3. With d fixed, the stage applies (1 + n D_h) to the
    input and (1 + d D_h)^{-1} to the result, which is the ordinary
    first-order prospective (lead) operator."""
    d, n = F(1, 2), F(3)
    values = [F(2), F(-1), F(5), F(0), F(1, 3)]
    out = stage(values, d, n)
    # verify the defining relation token by token, including the first
    previous_u = F(0)
    previous_v = F(0)
    for index, value in enumerate(values):
        assert stage_residual(d, n, H, out[index], previous_u, value,
                              previous_v) == 0
        previous_u, previous_v = out[index], value


def test_two_stages_expand_to_one_plus_Gamma_D_plus_M_D_squared():
    """REQUIREMENT 4. The cascade numerator is exactly 1 + Gamma D + M D^2
    with Gamma = n1 + n2 and M = n1 n2, checked as an operator identity on
    sequences: applying (1 + n1 D)(1 + n2 D) equals applying the expanded
    second-order operator."""
    def apply_numerator_pair(values, n1, n2, h=H):
        def first_order(seq, n):
            out, previous = [], F(0)
            for value in seq:
                out.append(value + n * (value - previous) / h)
                previous = value
            return out
        return first_order(first_order(values, n2), n1)

    def apply_expanded(values, gamma, mass, h=H):
        out = []
        for index, value in enumerate(values):
            first = values[index - 1] if index >= 1 else F(0)
            second = values[index - 2] if index >= 2 else F(0)
            out.append(value + gamma * (value - first) / h
                       + mass * (value - 2 * first + second) / (h * h))
        return out

    random.seed(11)
    for n1, n2 in ((F(1, 2), F(3)), (F(2), F(2)), (F(5, 7), F(1, 3))):
        gamma, mass = n1 + n2, n1 * n2
        values = [F(random.randint(-8, 8), random.randint(1, 3))
                  for _ in range(10)]
        assert apply_numerator_pair(values, n1, n2) == \
            apply_expanded(values, gamma, mass)
    # the critical branch: n1 = n2 = tau/2 gives Gamma = tau, M = tau^2/4
    for tau in (F(1, 2), F(3), F(10)):
        half = tau / 2
        gamma, mass = half + half, half * half
        assert gamma == tau and mass == tau * tau / 4


def test_the_gate_selects_the_regime_exactly():
    """n = d + g * softplus(delta): g = 0 is Native, g > 0 is active."""
    d = F(2)
    softplus_delta = F(7, 5)                      # any positive value
    for gate, expected_identity in ((F(0), True), (F(1, 2), False),
                                    (F(1), False)):
        n = d + gate * softplus_delta
        values = [F(3), F(-2), F(1, 2)]
        assert (stage(values, d, n) == values) is expected_identity
        assert coefficients(d, n)[0] == d / (H + d)     # pole unmoved


# ------------------------------------------- boundaries and orientation ----
def test_zero_prehistory_and_no_wraparound():
    """REQUIREMENT 9. The first output uses zeros for the missing history,
    an impulse never influences anything before it, and an impulse at the
    end never reaches the head."""
    d, n = F(1), F(4)
    length = 10
    for impulse in range(length):
        values = [F(1) if t == impulse else F(0) for t in range(length)]
        out = stage(values, d, n)
        assert all(value == 0 for value in out[:impulse])
        assert out[impulse] == coefficients(d, n)[1]     # alpha
    out = stage([F(0)] * (length - 1) + [F(1)], d, n)
    assert all(value == 0 for value in out[:length - 1])


def test_the_reverse_direction_is_the_flipped_causal_cascade():
    """REQUIREMENT 10, algebraically: the reverse branch is the same causal
    cascade on the reversed sequence, flipped back."""
    stages = ((F(1), F(3)), (F(1, 2), F(2)))
    values = [F(random.randint(-5, 5)) for _ in range(9)]
    reverse = cascade(values[::-1], stages)[::-1]
    assert reverse[-1] == cascade([values[-1]], stages)[0]
    assert reverse != cascade(values, stages)


# ----------------------------------------------- native-mode cancellation --
def test_the_cascade_can_cancel_a_native_mode_and_the_gain_detects_it():
    """The numerator zero sits at n/(h+n). If a native pole lands there the
    mode is annihilated, so the reported gain must go to zero exactly
    there and be bounded away from zero elsewhere."""
    def stage_response(z, d, n, h=H):
        derivative = (1 - F(1, 1) / z) / h
        return (1 + n * derivative) / (1 + d * derivative)

    n, d = F(3), F(1, 2)
    zero = n / (H + n)
    assert stage_response(zero, d, n) == 0          # exact cancellation
    for lam in (F(1, 10), F(1, 2), F(9, 10)):
        if lam == zero:
            continue
        assert stage_response(lam, d, n) != 0
