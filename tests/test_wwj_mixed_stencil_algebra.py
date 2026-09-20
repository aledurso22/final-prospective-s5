"""Exact verification of the REJECTED mixed-stencil algebra and scan.

The realization these tests describe was rejected on the cluster at commit
bfe53fe: its discretization-induced poles are unstable for the actual S5
modes (max companion radius about 1.705 at every declared grid cell, float32
NaNs, float64 states to about 1e81). The ALGEBRA below is still correct --
the discretization is a valid discretization -- so these proofs are kept with
the failed ablation. The principal architecture is tested in
`test_wwj_operator_algebra.py` and `test_wwj_ssm.py`.

Original description follows.

Exact verification of the WWJ recurrence algebra and of the scan algorithm.

Everything here is exact rational arithmetic in pure Python: no JAX, no
NumPy, no floating point. The identities are LINEAR in the free variables, so
evaluating them on a basis is a complete symbolic proof, not sampling.

The JAX implementation is checked separately in `test_wwj_recurrence.py`
(which needs JAX and therefore runs on the cluster).
"""

import math
import random
from fractions import Fraction as F

# ------------------------------------------------------------ the algebra --
def scalar_coefficients(tau, mass, h):
    """Rational mirror of `s5.wwj_recurrence.scalar_coefficients`."""
    q = mass + h * tau
    return ((2 * mass + h * tau - h * h) / q,
            mass / q,
            (mass + h * tau + h * h) / q,
            -(2 * mass + h * tau) / q,
            mass / q)


def residual(tau, mass, h, s_next, s0, s1, f0, f1, f2):
    """M(s-f)'' + tau(s-f)' + (s-f) under the MIXED stencils, times h^2.

    State: forward stencils s' ~ (s_{t+1}-s_t)/h, s'' ~ (s_{t+1}-2s_t+s_{t-1})/h^2.
    Target: backward stencils f' ~ (f_t-f_{t-1})/h, f'' ~ (f_t-2f_{t-1}+f_{t-2})/h^2.
    """
    left = (mass * (s_next - 2 * s0 + s1) + h * tau * (s_next - s0)
            + h * h * s0)
    right = (mass * (f0 - 2 * f1 + f2) + h * tau * (f0 - f1) + h * h * f0)
    return left - right


def test_the_update_solves_the_mixed_stencil_equation_exactly():
    """The residual is linear in (s_t, s_{t-1}, f_t, f_{t-1}, f_{t-2}); it
    vanishes on a basis, hence identically."""
    for tau, mass, h in ((F(1, 2), F(1, 16), F(1)), (F(3), F(9, 4), F(1)),
                         (F(1, 20), F(1, 1600), F(1, 4)), (F(2), F(0), F(1))):
        a, b, c0, c1, c2 = scalar_coefficients(tau, mass, h)
        for basis in range(5):
            s0, s1, f0, f1, f2 = (F(1) if index == basis else F(0)
                                  for index in range(5))
            s_next = a * s0 - b * s1 + c0 * f0 + c1 * f1 + c2 * f2
            assert residual(tau, mass, h, s_next, s0, s1, f0, f1, f2) == 0


def test_the_M_zero_boundary_is_exact():
    """M = 0  =>  s_{t+1} = (1 - h/tau)s_t + (1 + h/tau)f_t - f_{t-1}."""
    for tau, h in ((F(1), F(1)), (F(1, 20), F(1)), (F(7, 3), F(1, 2))):
        a, b, c0, c1, c2 = scalar_coefficients(tau, F(0), h)
        assert a == 1 - h / tau
        assert b == 0
        assert c0 == 1 + h / tau
        assert c1 == -1
        assert c2 == 0


def test_the_critical_mass_coefficients():
    """M = tau^2/4 with h = 1, written out independently."""
    for tau in (F(1, 4), F(1), F(5, 2)):
        mass = tau * tau / 4
        a, b, c0, c1, c2 = scalar_coefficients(tau, mass, F(1))
        q = tau * tau / 4 + tau
        assert q > 0
        assert a == (tau * tau / 2 + tau - 1) / q
        assert b == tau * tau / 4 / q
        assert c0 == (tau * tau / 4 + tau + 1) / q
        assert c1 == -(tau * tau / 2 + tau) / q
        assert c2 == b
        # b and c2 are the same number: both are M/q
        assert b == c2


def test_the_passive_bound_and_factorization():
    """P(D) = (1 + t_+ D)(1 + t_- D) continuously, iff t_+ + t_- = tau and
    t_+ t_- = M; the factors are real exactly on 0 <= eps <= 1/4."""
    # eps values whose discriminant has an exact rational square root, so
    # the identity can be checked with no floating point at all
    for tau in (F(1, 2), F(2)):
        for eps, root in ((F(0), F(1)), (F(3, 16), F(1, 2)),
                          (F(4, 25), F(3, 5)), (F(1, 4), F(0))):
            mass = eps * tau * tau
            t_plus = tau / 2 * (1 + root)
            t_minus = tau / 2 * (1 - root)
            assert root * root == 1 - 4 * eps        # sqrt is exact here
            assert t_plus + t_minus == tau
            assert t_plus * t_minus == mass
            assert t_minus >= 0                       # passive: both >= 0
    # the declared grid value eps = 1/16 has an irrational root; check it in
    # floating point instead, to the same identities
    root = math.sqrt(1 - 4 * (1 / 16))
    t_plus, t_minus = 0.5 * (1 + root), 0.5 * (1 - root)
    assert abs(t_plus + t_minus - 1.0) < 1e-15
    assert abs(t_plus * t_minus - 1 / 16) < 1e-15
    # just outside the bound the discriminant is negative: no real factors
    assert 1 - 4 * F(1, 3) < 0


def test_the_native_matched_substitution_reproduces_A_and_C():
    """s_{t+1} in terms of (s, x) equals the f-form, coefficient by
    coefficient: linear in six free variables, checked on their basis."""
    for tau, mass, h, lam, g in ((F(1, 2), F(1, 16), F(1), F(3, 5), F(7, 4)),
                                 (F(2), F(1), F(1), F(-1, 3), F(2)),
                                 (F(1, 10), F(1, 400), F(1, 2), F(9, 10),
                                  F(-1))):
        a, b, c0, c1, c2 = scalar_coefficients(tau, mass, h)
        k = tau / h
        Fm = 1 + k * (lam - 1)                       # F_tau
        Gm = k * g                                   # G_tau
        A0, A1, A2 = a + c0 * Fm, -b + c1 * Fm, c2 * Fm
        C0, C1, C2 = c0 * Gm, c1 * Gm, c2 * Gm
        for basis in range(6):
            s0, s1, s2, x0, x1, x2 = (F(1) if index == basis else F(0)
                                      for index in range(6))
            f0 = Fm * s0 + Gm * x0
            f1 = Fm * s1 + Gm * x1
            f2 = Fm * s2 + Gm * x2
            from_f = a * s0 - b * s1 + c0 * f0 + c1 * f1 + c2 * f2
            from_AC = (A0 * s0 + A1 * s1 + A2 * s2
                       + C0 * x0 + C1 * x1 + C2 * x2)
            assert from_f == from_AC


# -------------------------------------------------------- the scan itself --
def _matmul(P, Q):
    return [[sum(P[i][k] * Q[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _matvec(P, v):
    return [sum(P[i][k] * v[k] for k in range(3)) for i in range(3)]


def doubling_scan(A0, A1, A2, drive):
    """Exact rational mirror of `wwj_recurrence.wwj_scan`.

    Hillis-Steele over a CONSTANT transition: v <- v + H^(2^k) shift(v, 2^k),
    H <- H^2, with zero padding. Returns s_{t+1} for t = 0..L-1.
    """
    length = len(drive)
    state = [[d, F(0), F(0)] for d in drive]
    operator = [[A0, A1, A2], [F(1), F(0), F(0)], [F(0), F(1), F(0)]]
    distance = 1
    while distance < length:
        shifted = ([[F(0)] * 3] * distance + state[:-distance]
                   if distance else state)
        state = [[state[t][i] + _matvec(operator, shifted[t])[i]
                  for i in range(3)] for t in range(length)]
        operator = _matmul(operator, operator)
        distance *= 2
    return [row[0] for row in state]


def sequential_reference(A0, A1, A2, drive):
    """The obvious loop, with zero state prehistory."""
    out = []
    s0 = s1 = s2 = F(0)
    for d in drive:
        s_next = A0 * s0 + A1 * s1 + A2 * s2 + d
        out.append(s_next)
        s0, s1, s2 = s_next, s0, s1
    return out


def test_the_parallel_scan_equals_the_sequential_recurrence_exactly():
    """Every length, including 1, 2, 3 and non-powers of two, and with no
    wraparound from the zero prehistory."""
    random.seed(20260920)
    for length in (1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 17, 31, 33, 64, 100):
        A0, A1, A2 = (F(random.randint(-6, 6), random.randint(1, 7))
                      for _ in range(3))
        drive = [F(random.randint(-9, 9), random.randint(1, 5))
                 for _ in range(length)]
        assert doubling_scan(A0, A1, A2, drive) == \
            sequential_reference(A0, A1, A2, drive), length


def test_the_scan_has_zero_prehistory_and_no_wraparound():
    """A single impulse at t influences t..L-1 and nothing before it."""
    length = 12
    A0, A1, A2 = F(1, 2), F(1, 3), F(1, 5)
    for impulse in (0, 1, 2, 5, 11):
        drive = [F(1) if index == impulse else F(0) for index in range(length)]
        states = doubling_scan(A0, A1, A2, drive)
        assert all(state == 0 for state in states[:impulse])
        assert states[impulse] == 1                  # d_t reaches s_{t+1}
        assert any(state != 0 for state in states[impulse:])


def test_the_first_state_sees_only_the_first_input():
    """s_1 = d_0 exactly: no prehistory leaks in."""
    drive = [F(3, 7), F(1), F(-2)]
    assert doubling_scan(F(2), F(3), F(5), drive)[0] == F(3, 7)


# --------------------------------- the whole pipeline, indices included ----
def pipeline_via_scan(lam, g, tau, mass, xs, h):
    """Exactly what the production path does: build the drive from shifted
    inputs, then run the doubling scan."""
    a, b, c0, c1, c2 = scalar_coefficients(tau, mass, h)
    k = tau / h
    Fm, Gm = 1 + k * (lam - 1), k * g
    A0, A1, A2 = a + c0 * Fm, -b + c1 * Fm, c2 * Fm
    C0, C1, C2 = c0 * Gm, c1 * Gm, c2 * Gm
    drive = [C0 * xs[t]
             + (C1 * xs[t - 1] if t >= 1 else F(0))
             + (C2 * xs[t - 2] if t >= 2 else F(0))
             for t in range(len(xs))]
    return doubling_scan(A0, A1, A2, drive)


def pipeline_via_target(lam, g, tau, mass, xs, h):
    """The same thing written the other way: form the target f each step and
    apply the mixed-stencil update, with zero state AND input prehistory."""
    a, b, c0, c1, c2 = scalar_coefficients(tau, mass, h)
    k = tau / h
    Fm, Gm = 1 + k * (lam - 1), k * g
    s0 = s1 = s2 = F(0)
    x1 = x2 = F(0)
    out = []
    for x0 in xs:
        f0, f1, f2 = Fm * s0 + Gm * x0, Fm * s1 + Gm * x1, Fm * s2 + Gm * x2
        s_next = a * s0 - b * s1 + c0 * f0 + c1 * f1 + c2 * f2
        out.append(s_next)
        s0, s1, s2 = s_next, s0, s1
        x1, x2 = x0, x1
    return out


def test_the_drive_shifts_and_the_scan_agree_with_the_target_form():
    """This is the index alignment: an off-by-one in the input history or in
    the scan would show up here, exactly, at every length."""
    random.seed(20260920)
    for tau, mass, h in ((F(1, 4), F(1, 64), F(1)), (F(1), F(1, 4), F(1)),
                         (F(2), F(0), F(1))):
        for lam, g in ((F(3, 5), F(7, 4)), (F(-1, 3), F(1))):
            for length in (1, 2, 3, 4, 7, 16, 17, 40):
                xs = [F(random.randint(-9, 9), random.randint(1, 4))
                      for _ in range(length)]
                assert pipeline_via_scan(lam, g, tau, mass, xs, h) == \
                    pipeline_via_target(lam, g, tau, mass, xs, h), length


def test_the_first_output_has_seen_only_the_first_input():
    """s_1 = c0 * G * x_0 exactly: no input prehistory leaks in."""
    tau, mass, h, lam, g = F(1, 2), F(1, 16), F(1), F(3, 5), F(2)
    _, _, c0, _, _ = scalar_coefficients(tau, mass, h)
    xs = [F(5, 3), F(-1), F(2)]
    first = pipeline_via_scan(lam, g, tau, mass, xs, h)[0]
    assert first == c0 * (tau / h) * g * xs[0]
