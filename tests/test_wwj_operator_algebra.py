"""Exact proofs of the PRINCIPAL WWJ architecture. No JAX, no floating point.

Everything is rational arithmetic in pure Python, so these are proofs, not
samples, and they run anywhere. The JAX implementation of the same claims is
checked in `test_wwj_ssm.py` on the cluster.

The architecture: the Native S5 recurrence s_{t+1} = Abar s_t + Bbar x_t is
untouched, and the WWJ coordinate is the EXACT discrete operator applied to
its trajectory with one consistent backward derivative,

    z_t = P(D_h) s_t = (1 + k + m) s_t - (k + 2m) s_{t-1} + m s_{t-2},
    k = tau/h,  m = M/h^2 = eps k^2.
"""

import random
from fractions import Fraction as F


def three_tap(states, k, m):
    """The claimed formula, with zero prehistory."""
    out = []
    for t, s in enumerate(states):
        s1 = states[t - 1] if t >= 1 else F(0)
        s2 = states[t - 2] if t >= 2 else F(0)
        out.append((1 + k + m) * s - (k + 2 * m) * s1 + m * s2)
    return out


def operator_from_derivatives(states, tau, mass, h):
    """P(D_h)s built from the DERIVATIVES themselves: s + tau D_h s + M D_h^2 s.

    Written independently of the three-tap formula, so agreement between the
    two is a statement about the algebra and not a restatement.
    """
    out = []
    for t, s in enumerate(states):
        s1 = states[t - 1] if t >= 1 else F(0)
        s2 = states[t - 2] if t >= 2 else F(0)
        first = (s - s1) / h
        second = (s - 2 * s1 + s2) / (h * h)
        out.append(s + tau * first + mass * second)
    return out


def test_the_three_tap_is_exactly_P_of_D_h():
    """(1+k+m, -(k+2m), m) is what s + tau D_h s + M D_h^2 s expands to."""
    for tau, eps, h in ((F(1, 4), F(1, 4), F(1)), (F(3), F(1, 16), F(1)),
                        (F(1, 10), F(0), F(1, 2)), (F(2), F(1, 4), F(1, 4))):
        mass = eps * tau * tau
        k, m = tau / h, mass / (h * h)
        assert m == eps * k * k                    # m = eps k^2, as declared
        states = [F(random.randint(-9, 9), random.randint(1, 5))
                  for _ in range(12)]
        assert three_tap(states, k, m) == \
            operator_from_derivatives(states, tau, mass, h)


def test_the_second_difference_is_the_derivative_applied_twice():
    """D_h(D_h s) = D_h^2 s exactly -- the property the mixed-stencil
    realization did NOT have, and the reason the factorization below is
    exact here."""
    h = F(1, 3)
    states = [F(random.randint(-9, 9), random.randint(1, 4))
              for _ in range(10)]

    def difference(values):
        return [(values[t] - (values[t - 1] if t >= 1 else F(0))) / h
                for t in range(len(values))]

    twice = difference(difference(states))
    direct = [(states[t] - 2 * (states[t - 1] if t >= 1 else F(0))
               + (states[t - 2] if t >= 2 else F(0))) / (h * h)
              for t in range(len(states))]
    assert twice == direct


def test_the_passive_two_stage_factorization_equals_the_three_tap():
    """(1 + t_+ D_h)(1 + t_- D_h) applied as two stages, zero prehistory in
    each, is the direct three-tap operator."""
    def stage(values, factor, h):
        a = factor / h
        return [(1 + a) * values[t] - a * (values[t - 1] if t >= 1 else F(0))
                for t in range(len(values))]

    for tau, eps, root, h in ((F(1), F(0), F(1), F(1)),
                              (F(2), F(3, 16), F(1, 2), F(1)),
                              (F(1, 2), F(4, 25), F(3, 5), F(1)),
                              (F(3), F(1, 4), F(0), F(1, 2))):
        assert root * root == 1 - 4 * eps
        t_plus = tau / 2 * (1 + root)
        t_minus = tau / 2 * (1 - root)
        assert t_plus + t_minus == tau and t_plus * t_minus == eps * tau * tau
        states = [F(random.randint(-7, 7), random.randint(1, 3))
                  for _ in range(9)]
        k, m = tau / h, eps * tau * tau / (h * h)
        assert stage(stage(states, t_minus, h), t_plus, h) == \
            three_tap(states, k, m)


def native_scan(lam, b, inputs):
    """s_{t+1} = lam s_t + b x_t, s_{-1} = 0; returns s_t for t = 0..L-1."""
    out, state = [], F(0)
    for x in inputs:
        state = lam * state + b * x
        out.append(state)
    return out


def augmented_realization(lam, b, inputs, k, m):
    """The augmented 3-block recurrent state space, run literally.

    q_{t+1} = [[lam,0,0],[1,0,0],[0,1,0]] q_t + [b x_t; 0; 0]
    z_t     = [(1+k+m), -(k+2m), m] q_t
    """
    q = [F(0), F(0), F(0)]
    out = []
    for x in inputs:
        q = [lam * q[0] + b * x, q[0], q[1]]       # q_{t+1}
        out.append((1 + k + m) * q[0] - (k + 2 * m) * q[1] + m * q[2])
    return out


def test_the_augmented_realization_equals_native_scan_plus_three_tap():
    """The optimized implementation -- Native scan, then an O(LP) three-tap --
    is not an approximation of the augmented recurrent realization: it is the
    same sequence, exactly."""
    random.seed(20260920)
    for lam, b in ((F(1, 2), F(3)), (F(-2, 3), F(1)), (F(9, 10), F(-5, 4))):
        for tau, eps in ((F(1, 20), F(1, 4)), (F(1), F(1, 16)), (F(3), F(0))):
            k, m = tau, eps * tau * tau            # h = 1
            for length in (1, 2, 3, 5, 8, 17, 40):
                inputs = [F(random.randint(-9, 9), random.randint(1, 4))
                          for _ in range(length)]
                assert augmented_realization(lam, b, inputs, k, m) == \
                    three_tap(native_scan(lam, b, inputs), k, m), length


def _determinant3(matrix):
    """Exact 3x3 determinant by cofactor expansion."""
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def test_the_recurrent_poles_are_native_plus_two_zeros():
    """The characteristic polynomial of the augmented transition, COMPUTED
    from the matrix, is (z - lam) z^2: the WWJ coordinate adds ZEROS, never
    a pole, so it cannot destabilize what Native S5 does not already do."""
    for lam in (F(1, 2), F(-3, 4), F(9, 10), F(2)):
        for z in (F(0), F(1), F(-1), F(5, 3), F(7)):
            # zI - H with H = [[lam,0,0],[1,0,0],[0,1,0]]
            characteristic = _determinant3(((z - lam, F(0), F(0)),
                                            (F(-1), z, F(0)),
                                            (F(0), F(-1), z)))
            assert characteristic == (z - lam) * z * z
        # the roots are therefore lam and a double zero, and nothing else
        for root in (lam, F(0)):
            assert _determinant3(((root - lam, F(0), F(0)),
                                  (F(-1), root, F(0)),
                                  (F(0), F(-1), root))) == 0
        # a value that is neither is not a root
        other = lam + 1 if lam != 0 else F(3)
        assert _determinant3(((other - lam, F(0), F(0)),
                              (F(-1), other, F(0)),
                              (F(0), F(-1), other))) != 0


def test_tau_zero_recovers_native_s5_exactly():
    """k = m = 0 makes the operator the identity, so the layer is Native."""
    random.seed(7)
    for lam, b in ((F(1, 2), F(3)), (F(-2, 3), F(1))):
        inputs = [F(random.randint(-9, 9), random.randint(1, 4))
                  for _ in range(15)]
        states = native_scan(lam, b, inputs)
        assert three_tap(states, F(0), F(0)) == states
        # and eps has no effect when tau = 0, since m = eps k^2
        for eps in (F(0), F(1, 16), F(1, 4)):
            assert three_tap(states, F(0), eps * 0) == states


def test_zero_prehistory_and_no_wraparound_at_both_boundaries():
    """The first two outputs use zeros for the missing history, and the end
    of the sequence never influences the beginning."""
    k, m = F(1, 4), F(1, 64)
    states = [F(1), F(0), F(0), F(0), F(0)]
    out = three_tap(states, k, m)
    assert out[0] == (1 + k + m) * 1               # no s_{-1}, s_{-2}
    assert out[1] == -(k + 2 * m) * 1
    assert out[2] == m * 1
    assert out[3] == 0 and out[4] == 0             # finite support: 3 taps
    # an impulse at the END touches only the end
    states = [F(0), F(0), F(0), F(0), F(1)]
    out = three_tap(states, k, m)
    assert out[:4] == [F(0)] * 4
    assert out[4] == (1 + k + m) * 1


def test_the_reverse_direction_is_the_flipped_causal_operator():
    """Bidirectional: the reverse branch applies the SAME causal operator to
    the reversed trajectory and is flipped back -- its history is the later
    tokens, never the earlier ones."""
    k, m = F(1, 3), F(1, 36)
    states = [F(random.randint(-5, 5)) for _ in range(9)]
    reverse = three_tap(states[::-1], k, m)[::-1]
    # the last token of the reverse branch has no history at all
    assert reverse[-1] == (1 + k + m) * states[-1]
    # and it differs from the forward branch, so orientation is not ignored
    assert reverse != three_tap(states, k, m)


def test_the_gated_readout_is_a_separate_intervention():
    """s~ = s + g(z - s) equals the direct arm only at g = 1 and Native only
    at g = 0; at intermediate g it is neither."""
    k, m = F(1, 2), F(1, 16)
    states = [F(random.randint(-5, 5)) for _ in range(8)]
    direct = three_tap(states, k, m)
    for gate in (F(0), F(1, 2), F(1)):
        gated = [s + gate * (z - s) for s, z in zip(states, direct)]
        if gate == 0:
            assert gated == states
        elif gate == 1:
            assert gated == direct
        else:
            assert gated != states and gated != direct
