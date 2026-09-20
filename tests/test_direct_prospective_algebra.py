"""Exact proofs about the direct prospective recurrences. No JAX, no floats.

Rational arithmetic in pure Python, so these are proofs rather than samples,
and they run without JAX. What is established here:

  * the matched coefficients (a, b, c0, c1, c2) solve the mixed-stencil
    matched equation, and the partially matched (a, b, c0, c1) solve the
    two-compartment equation, both on a basis of the free variables;
  * the M f'' term is MANDATORY for the matched equation: dropping it breaks
    the identity, and the two coefficient sets differ;
  * the M = 0, Gamma = T = tau boundary of both is the professor/Zucchet
    update, exactly;
  * substituting f = Abar s + Bbar x gives the stated A and C coefficients;
  * the EXACT common-stencil matched operator annihilates the residual, so
    s = f and the driven response collapses to (I - Abar) s = Bbar x --
    memoryless, no S5 memory left;
  * z = 1 is ALWAYS an exact root of both causal recurrences at a mode
    Abar = 1, for every M, Gamma, T -- a structural marginality, not a
    tuning problem;
  * the alignment of x_t, x_{t-1}, x_{t-2} and the absence of wraparound.

The JAX implementation of the same claims is in
`test_direct_prospective_scan.py`, which needs a cluster.
"""

import random
from fractions import Fraction as F

H = F(1)


# --------------------------------------------------------- coefficients ----
def matched(tau, mass, h=H):
    q = mass + h * tau
    return ((2 * mass + h * tau - h * h) / q, mass / q,
            (mass + h * tau + h * h) / q, -(2 * mass + h * tau) / q, mass / q)


def partially_matched(response, mass, gamma, h=H):
    big_gamma = gamma + response
    q = mass + h * big_gamma
    return ((2 * mass + h * big_gamma - h * h) / q, mass / q,
            h * (h + response) / q, -h * response / q)


def matched_residual(tau, mass, h, s_next, s0, s1, f0, f1, f2):
    """M s'' + tau s' + s - (f + tau f' + M f''), times h^2, mixed stencils."""
    left = mass * (s_next - 2 * s0 + s1) + h * tau * (s_next - s0) + h * h * s0
    right = mass * (f0 - 2 * f1 + f2) + h * tau * (f0 - f1) + h * h * f0
    return left - right


def partially_matched_residual(response, mass, gamma, h, s_next, s0, s1, f0,
                               f1):
    """M s'' + Gamma s' + s - (f + T f'), times h^2. NO M f'' term."""
    big_gamma = gamma + response
    left = (mass * (s_next - 2 * s0 + s1) + h * big_gamma * (s_next - s0)
            + h * h * s0)
    right = h * h * f0 + h * response * (f0 - f1)
    return left - right


def test_the_matched_coefficients_solve_the_matched_equation():
    for tau, mass in ((F(1, 2), F(1, 16)), (F(3), F(9, 4)), (F(2), F(0))):
        a, b, c0, c1, c2 = matched(tau, mass)
        for basis in range(5):
            s0, s1, f0, f1, f2 = (F(1) if i == basis else F(0)
                                  for i in range(5))
            s_next = a * s0 - b * s1 + c0 * f0 + c1 * f1 + c2 * f2
            assert matched_residual(tau, mass, H, s_next, s0, s1, f0, f1,
                                    f2) == 0


def test_the_partially_matched_coefficients_solve_their_equation():
    for response, mass, gamma in ((F(1, 2), F(1, 16), F(0)),
                                  (F(3), F(1), F(1, 2)), (F(2), F(0), F(0))):
        a, b, c0, c1 = partially_matched(response, mass, gamma)
        for basis in range(4):
            s0, s1, f0, f1 = (F(1) if i == basis else F(0) for i in range(4))
            s_next = a * s0 - b * s1 + c0 * f0 + c1 * f1
            assert partially_matched_residual(response, mass, gamma, H,
                                              s_next, s0, s1, f0, f1) == 0


def test_the_M_f_double_dot_term_is_mandatory_and_distinguishes_the_models():
    """Dropping M f'' from the matched equation breaks the identity, and the
    two coefficient sets are genuinely different whenever M > 0."""
    tau, mass = F(1), F(1, 4)
    a, b, c0, c1, c2 = matched(tau, mass)
    # the matched solution does NOT satisfy the no-M-f'' equation
    s0, s1, f0, f1, f2 = F(2), F(-1), F(3), F(1), F(5)
    s_next = a * s0 - b * s1 + c0 * f0 + c1 * f1 + c2 * f2
    assert partially_matched_residual(tau, mass, F(0), H, s_next, s0, s1, f0,
                                      f1) != 0
    # and conversely
    pa, pb, pc0, pc1 = partially_matched(tau, mass, F(0))
    other = pa * s0 - pb * s1 + pc0 * f0 + pc1 * f1
    assert matched_residual(tau, mass, H, other, s0, s1, f0, f1, f2) != 0
    assert (pa, pb) == (a, b)                   # same state side...
    assert (pc0, pc1) != (c0, c1)               # ...different target side
    assert c2 == mass / (mass + tau) != 0       # the M f'' tap itself


def test_both_reduce_to_the_professor_update_at_the_ordinary_boundary():
    """M = 0, Gamma = T = tau:
    s_{t+1} = (1 - h/tau) s_t + (1 + h/tau) f_t - f_{t-1}."""
    for tau in (F(1), F(1, 20), F(7, 3)):
        a, b, c0, c1, c2 = matched(tau, F(0))
        assert (a, b, c0, c1, c2) == (1 - H / tau, F(0), 1 + H / tau, F(-1),
                                      F(0))
        pa, pb, pc0, pc1 = partially_matched(tau, F(0), F(0))
        assert (pa, pb, pc0, pc1) == (1 - H / tau, F(0), 1 + H / tau, F(-1))


def test_the_substitution_gives_the_stated_A_and_C_coefficients():
    """f = Abar s + Bbar x, checked on a basis of six free variables."""
    for tau, mass, lam, g in ((F(1, 2), F(1, 16), F(3, 5), F(7, 4)),
                              (F(2), F(1), F(-1, 3), F(2)),
                              (F(1, 10), F(1, 400), F(9, 10), F(-1))):
        a, b, c0, c1, c2 = matched(tau, mass)
        A0, A1, A2 = a + c0 * lam, -b + c1 * lam, c2 * lam
        C0, C1, C2 = c0 * g, c1 * g, c2 * g
        for basis in range(6):
            s0, s1, s2, x0, x1, x2 = (F(1) if i == basis else F(0)
                                      for i in range(6))
            f0, f1, f2 = (lam * s0 + g * x0, lam * s1 + g * x1,
                          lam * s2 + g * x2)
            assert (a * s0 - b * s1 + c0 * f0 + c1 * f1 + c2 * f2
                    == A0 * s0 + A1 * s1 + A2 * s2 + C0 * x0 + C1 * x1
                    + C2 * x2)


def test_the_two_compartment_substitution_gives_A1_and_A2():
    for response, mass, gamma, lam, g in ((F(1, 2), F(1, 16), F(0), F(3, 5),
                                           F(2)),
                                          (F(3), F(1), F(1, 4), F(-1, 2),
                                           F(1))):
        a, b, c0, c1 = partially_matched(response, mass, gamma)
        A1t, A2t = a + c0 * lam, -b + c1 * lam
        for basis in range(4):
            s0, s1, x0, x1 = (F(1) if i == basis else F(0) for i in range(4))
            f0, f1 = lam * s0 + g * x0, lam * s1 + g * x1
            assert (a * s0 - b * s1 + c0 * f0 + c1 * f1
                    == A1t * s0 + A2t * s1 + c0 * g * x0 + c1 * g * x1)


# ------------------------------------- the exact-matching consequence ------
def test_the_exact_common_stencil_keeps_a_zero_residual_zero():
    """(1+k+m) r_t - (k+2m) r_{t-1} + m r_{t-2} = 0 with r_{-1} = r_{-2} = 0
    forces r_t = 0 for every t: exact matching means s = f."""
    for tau, mass in ((F(1), F(1, 4)), (F(1, 5), F(1, 100)), (F(3), F(0))):
        k, m = tau / H, mass / (H * H)
        r1 = r2 = F(0)
        for _ in range(25):
            r0 = ((k + 2 * m) * r1 - m * r2) / (1 + k + m)
            assert r0 == 0
            r1, r2 = r0, r1


def test_exact_matching_collapses_the_driven_S5_response_to_a_memoryless_map():
    """s = f with f = Abar s + Bbar x gives (1 - Abar) s = Bbar x, so the
    state is an instantaneous function of the CURRENT input: every bit of
    S5's input-driven memory is gone. The impulse response is a delta, not a
    decaying exponential.

    This is the scientific reason the exactly matched model cannot simply be
    adopted as an improved S5 recurrence.
    """
    for lam, g in ((F(1, 2), F(3)), (F(9, 10), F(1)), (F(-1, 3), F(2))):
        gain = g / (1 - lam)
        inputs = [F(1)] + [F(0)] * 9                  # an impulse at t = 0
        collapsed = [gain * x for x in inputs]
        assert collapsed[0] == gain
        assert all(value == 0 for value in collapsed[1:])   # no memory
        # Native S5, for contrast, remembers: s_t = lam^t * g
        native, state = [], F(0)
        for x in inputs:
            state = lam * state + g * x
            native.append(state)
        assert native[0] == g
        assert all(native[t] == lam ** t * g for t in range(10))
        assert any(value != 0 for value in native[1:])
        assert native[1:] != collapsed[1:]


# ------------------------------------------------ the structural margin ----
def test_z_equals_one_is_always_a_root_at_a_unit_mode():
    """At Abar = 1 the characteristic polynomial of BOTH causal recurrences
    vanishes at z = 1, for every M, Gamma, T. The recurrence is therefore
    marginal at best on a mode at 1: no parameter choice moves that root."""
    for tau, mass in ((F(1, 20), F(1, 1600)), (F(3), F(9, 16)), (F(1), F(0))):
        a, b, c0, c1, c2 = matched(tau, mass)
        A0, A1, A2 = a + c0, -b + c1, c2              # lam = 1
        assert 1 - A0 - A1 - A2 == 0                  # char(1) = 0
    for response, mass, gamma in ((F(1, 20), F(1, 400), F(0)),
                                  (F(2), F(1), F(1, 3))):
        a, b, c0, c1 = partially_matched(response, mass, gamma)
        A1t, A2t = a + c0, -b + c1
        assert 1 - A1t - A2t == 0


def test_the_companion_radius_exceeds_one_for_ordinary_modes():
    """A worked counterexample, exactly: a real mode at Abar = -1 with a
    small timescale gives |root| far above 1. This is the analytic content
    of the cluster's rejection, independent of the target construction."""
    tau = F(1, 20)                                    # h/tau = 20
    mass = F(1, 4) * tau * tau
    a, b, c0, c1, c2 = matched(tau, mass)
    lam = F(-1)
    A0, A1, A2 = a + c0 * lam, -b + c1 * lam, c2 * lam
    # a root of large magnitude exists: char(z) changes sign far outside the
    # unit circle, so |z| > 10 for some root
    def char(z):
        return z ** 3 - A0 * z ** 2 - A1 * z - A2
    assert char(F(-10)) * char(F(-100)) < 0           # a root beyond -10
    assert abs(A0) > 20                               # the 2h/tau blow-up


# ---------------------------------------------------------- the scan -------
def doubling_scan(coefficients, drive):
    """Exact rational mirror of `companion_doubling_scan`, order n."""
    order = len(coefficients)
    length = len(drive)
    state = [[d] + [F(0)] * (order - 1) for d in drive]
    operator = [list(coefficients)]
    for index in range(order - 1):
        operator.append([F(1) if column == index else F(0)
                         for column in range(order)])
    distance = 1
    while distance < length:
        shifted = [[F(0)] * order] * distance + state[:-distance]
        state = [[sum(operator[i][j] * shifted[t][j] for j in range(order))
                  + state[t][i] for i in range(order)]
                 for t in range(length)]
        operator = [[sum(operator[i][k] * operator[k][j]
                         for k in range(order))
                     for j in range(order)] for i in range(order)]
        distance *= 2
    return [row[0] for row in state]


def sequential(coefficients, drive):
    order = len(coefficients)
    history = [F(0)] * order
    out = []
    for d in drive:
        value = sum(c * h for c, h in zip(coefficients, history)) + d
        out.append(value)
        history = [value] + history[:-1]
    return out


def test_the_doubling_scan_equals_the_sequential_recurrence_exactly():
    random.seed(20260920)
    for order in (2, 3):
        for length in (1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 31, 33, 64, 100):
            coefficients = [F(random.randint(-6, 6), random.randint(1, 7))
                            for _ in range(order)]
            drive = [F(random.randint(-9, 9), random.randint(1, 5))
                     for _ in range(length)]
            assert doubling_scan(coefficients, drive) == \
                sequential(coefficients, drive), (order, length)


def test_the_input_alignment_and_the_absence_of_wraparound():
    """The state emitted at token t has seen x_t through C0, x_{t-1} through
    C1 and x_{t-2} through C2, and nothing from later tokens."""
    tau, mass, lam, g = F(1, 2), F(1, 16), F(3, 5), F(2)
    a, b, c0, c1, c2 = matched(tau, mass)
    A0, A1, A2 = a + c0 * lam, -b + c1 * lam, c2 * lam
    C = (c0 * g, c1 * g, c2 * g)
    length = 8
    for impulse in (0, 1, 2, 7):
        x = [F(1) if t == impulse else F(0) for t in range(length)]
        drive = [C[0] * x[t]
                 + (C[1] * x[t - 1] if t >= 1 else F(0))
                 + (C[2] * x[t - 2] if t >= 2 else F(0))
                 for t in range(length)]
        states = doubling_scan([A0, A1, A2], drive)
        assert all(value == 0 for value in states[:impulse])   # causal
        assert states[impulse] == C[0]                          # sees x_t
        if impulse + 1 < length:
            assert states[impulse + 1] == A0 * C[0] + C[1]      # then x_{t-1}
