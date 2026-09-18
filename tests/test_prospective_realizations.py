"""EXACT audit of the prospective REALIZATIONS on the MDN write path
(docs/PROSPECTIVE_REALIZATION_AUDIT.md). Pure Python `Fraction`: no JAX,
no NumPy, no floating point.

Placement (the executed one, `filtered.py`): the processing state s = y is
driven by the masked MDN residual f = R_t and feeds the native momentum; the
law's residual is r = s - f, so the law reads

    M s'' + gamma s' + (s - f) + T (s' - f') = 0.

Index convention (same-token output): the state produced by token t is
s_(t+1) = y_t, driven by f_t = R_t; s_t = y_(t-1), f_(t-1) = R_(t-1).
"""

import random
from fractions import Fraction as F

from experiments.prospective_momentum import optimizer_forms as OF

RNG = random.Random(20260919)
H = F(1)


def q(lo=-40, hi=40, den=8):
    return F(RNG.randint(lo, hi), den)


def pos(lo=1, hi=40, den=8):
    return F(RNG.randint(lo, hi), den)


# ------------------------------------------------ scalar reference forms --
def fd_step(s, s_prev, f, f_prev, M, gamma, T, h=H):
    """DIRECT finite-difference realization of the law, derived term by
    term with Zucchet et al.'s Eq. (17) difference choices (forward
    difference for s', backward difference for f', the zeroth-order residual
    at (s_t, f_t)) and a centred second difference for s'':

        M (s1 - 2 s + s_prev)/h^2 + gamma (s1 - s)/h + (s - f)
          + T [(s1 - s) - (f - f_prev)]/h = 0,

    solved for s1 WITHOUT using the executed coefficients a, b, c, d."""
    lead = M / (h * h) + (gamma + T) / h
    rest = (M * (-2 * s + s_prev) / (h * h) - gamma * s / h + (s - f)
            + T * (-s - (f - f_prev)) / h)
    return -rest / lead


def executed_step(y, y_prev, R, R_prev, M, gamma, T, h=H):
    """The executed coefficient form (filtered.py), scalar."""
    a, b, c, d = OF.filter_coefficients(M, gamma, T, h)
    return a * y - b * y_prev + c * R - d * R_prev


def run_scalar(step, fs, *args):
    s, s_prev, f_prev, out = F(0), F(0), F(0), []
    for f in fs:
        s1 = step(s, s_prev, f, f_prev, *args)
        out.append(s1)
        s, s_prev, f_prev = s1, s, f
    return out


def admissible(M, gamma, T, h=H):
    A = M + h * (gamma + T)
    return M >= 0 and T >= 0 and A > 0 and gamma + T > 0 \
        and 4 * M + 2 * h * (gamma + T) > h * h


def random_point():
    while True:
        M, gamma, T = F(RNG.randint(0, 24), 8), q(-16, 24), F(RNG.randint(0, 24), 8)
        if admissible(M, gamma, T):
            return M, gamma, T


# ------------------------- 1-3. elimination, direct FD, transfer function --
def test_direct_finite_difference_realization_is_the_executed_recurrence():
    """The law's direct FD realization, solved from its own terms, equals the
    executed generalized recurrence token by token, for every admissible
    (M, gamma, T), gamma of either sign."""
    for _ in range(200):
        M, gamma, T = random_point()
        fs = [q() for _ in range(9)]
        assert run_scalar(fd_step, fs, M, gamma, T) == \
            run_scalar(executed_step, fs, M, gamma, T)


def impulse_response(M, gamma, T, n, h=H):
    """h_j from H(z) = (c - d z^-1)/(1 - a z^-1 + b z^-2) by exact long
    division: the executed recurrence with its auxiliary states ELIMINATED."""
    a, b, c, d = OF.filter_coefficients(M, gamma, T, h)
    num = [c, -d] + [F(0)] * n
    hs = []
    for j in range(n):
        v = num[j] + (a * hs[j - 1] if j >= 1 else 0) \
            - (b * hs[j - 2] if j >= 2 else 0)
        hs.append(v)
    return hs


def test_eliminating_the_auxiliary_state_gives_the_same_transfer_function():
    """y_t = sum_j h_j R_(t-j) with h the exact impulse response of
    H(z) = (c - d z^-1)/(1 - a z^-1 + b z^-2): the state-free (convolution)
    form reproduces both the executed and the direct FD recurrences."""
    for _ in range(60):
        M, gamma, T = random_point()
        fs = [q() for _ in range(8)]
        hs = impulse_response(M, gamma, T, len(fs))
        conv = [sum(hs[j] * fs[t - j] for j in range(t + 1))
                for t in range(len(fs))]
        assert conv == run_scalar(executed_step, fs, M, gamma, T)
        assert conv == run_scalar(fd_step, fs, M, gamma, T)


# ------------------------------------------------- 5. ordinary boundaries --
def test_ordinary_finite_difference_boundary_is_gamma_plus_T_equal_h():
    """DERIVED, not guessed: a = 0 and b = 0 (no auxiliary state) iff M = 0
    and gamma + T = h; there c = 1 + T/h, d = T/h, i.e. the executed
    ordinary finite-difference arm Rpros = R + kappa (R - R_prev) with
    kappa = T/h, for EVERY kappa >= 0 (kappa > 1 needs gamma < 0)."""
    for T in (F(1, 4), F(1, 2), F(1), F(3, 2), F(2), F(17, 8)):
        gamma = H - T
        assert OF.filter_coefficients(F(0), gamma, T) == (0, 0, 1 + T, T)
    for _ in range(100):
        M, gamma, T = random_point()
        a, b, _, _ = OF.filter_coefficients(M, gamma, T)
        assert (a == 0 and b == 0) == (M == 0 and gamma + T == H)


def test_generalized_recurrence_recovers_the_ordinary_fd_arm_closed_loop():
    """Full MDN write path (matrices, token gates, masks): the generalized
    recurrence at (0, h - kappa h, kappa h) equals the executed ordinary
    finite-difference arm exactly, including kappa > 1."""
    keys = ((F(3, 5), F(4, 5)), (F(1), F(0)), (F(4, 5), F(-3, 5)))
    for kappa in (F(1, 2), F(1), F(3, 2), F(2)):
        toks = [(list(RNG.choice(keys)), [q(), q()], F(RNG.randint(0, 1)))
                for _ in range(7)]
        gs = [(F(RNG.randint(1, 99), 100), F(RNG.randint(1, 99), 100),
               F(RNG.randint(1, 99), 100), F(RNG.randint(1, 199), 100))
              for _ in range(7)]
        z = OF.zeros(2, 2, F(0))
        coeff = OF.filter_coefficients(F(0), H - kappa * H, kappa * H)
        gen = OF.run(OF.filtered_step, (z,) * 5, toks, gs, coeff)
        fd = OF.run(OF.two_tap_step, (z,) * 3, toks, gs, kappa)
        assert [c[0] for c in gen] == [c[0] for c in fd]


def test_the_ordinary_adaptive_state_boundary_is_M_gamma_zero():
    """M = gamma = 0 gives Zucchet et al.'s Eq. (17) (the executed
    ordinary adaptive-state arm); it coincides with the finite-difference
    boundary only at T = h."""
    for _ in range(50):
        T = pos()
        a, b, c, d = OF.filter_coefficients(F(0), F(0), T)
        assert (a, b, c, d) == (1 - H / T, 0, 1 + H / T, 1)
        fs = [q() for _ in range(6)]
        tss = run_scalar(executed_step, fs, F(0), F(0), T)
        fdb = run_scalar(executed_step, fs, F(0), H - T, T)
        if T != H and any(fs):
            assert tss != fdb


# -------------------------------------- 6. Zucchet's adaptive current (Eq. 7) --
def adaptive_current_run(fs, tau, tau_a, M=F(0), gamma=F(0), h=H):
    """Zucchet et al. Eq. (7), generalized to the law, discretized with the
    SAME Eq. (17) choices for s and FORWARD EULER for the adaptive current:

        M s'' + (gamma + tau) s' + s = (1 + tau/tau_a) f - (tau/tau_a) a,
        tau_a a' = f - a,   a_(t+1) = a_t + (h/tau_a)(f_t - a_t),  a_0 = 0.

    At M = gamma = 0 this is exactly Eq. (7)."""
    s, s_prev, a, out = F(0), F(0), F(0), []
    lead = M / (h * h) + (gamma + tau) / h
    for f in fs:
        drive = (1 + tau / tau_a) * f - (tau / tau_a) * a
        rest = M * (-2 * s + s_prev) / (h * h) - (gamma + tau) * s / h \
            + s - drive
        s1 = -rest / lead
        out.append(s1)
        a = a + (h / tau_a) * (f - a)
        s, s_prev = s1, s
    return out


def test_adaptive_current_with_tau_a_equal_step_is_exactly_eq17():
    """EXACT mapping: forward Euler with tau_a = h gives a_t = f_(t-1), so
    Eq. (7) becomes Eq. (17) term by term (tau -> T, (1 + tau/tau_a) ->
    1 + T/h, tau/tau_a -> T/h): the executed ordinary adaptive-state arm.
    The same holds for the generalized law: the executed generalized
    recurrence IS the adaptive-current realization at tau_a = h."""
    for _ in range(100):
        M, gamma, T = random_point()
        if T == 0:
            continue
        fs = [q() for _ in range(8)]
        assert adaptive_current_run(fs, T, H, M, gamma) == \
            run_scalar(executed_step, fs, M, gamma, T)
        assert adaptive_current_run(fs, T, H) == \
            run_scalar(executed_step, fs, F(0), F(0), T)


def test_adaptive_current_with_tau_a_off_the_step_is_a_distinct_realization():
    """MINIMAL COUNTEREXAMPLE, one token: at tau_a = 2h the first write's
    amplitude is 1 + T/tau_a = 3/2 against Eq. (17)'s 1 + T/h = 2 (T = h).
    Only such an off-step realization would be a distinct adaptive-state
    realization; none is executed."""
    fs = [F(1)]
    T = F(1)
    ac = adaptive_current_run(fs, T, 2 * H)
    ex = run_scalar(executed_step, fs, F(0), F(0), T)
    assert ac == [F(3, 2)] and ex == [F(2)]
    # tau_a -> 0 is Zucchet's exact-prospectivity limit; no finite step
    # realizes it: at h fixed, tau_a < h/2 makes the Euler map of a unstable
    a_map = 1 - H / (F(1, 4) * H)
    assert abs(a_map) > 1


# ------------------------------- the two ordinary arms: one scheme, two points --
def test_the_two_ordinary_arms_differ_by_gamma_not_by_realization():
    """Both ordinary arms are M = 0 points of the SAME recurrence: the
    finite-difference arm at gamma = h - T, the adaptive-state arm at
    gamma = 0. Their difference is the parameter boundary, not a change of
    realization; they coincide at T = h."""
    fs = [F(1), F(-2), F(3)]
    for T in (F(1, 2), F(1), F(2)):
        fd = run_scalar(executed_step, fs, F(0), H - T, T)
        tt = [(1 + T) * f - T * p for f, p in zip(fs, [F(0)] + fs[:-1])]
        assert fd == tt                                  # the two-tap
        tss = run_scalar(executed_step, fs, F(0), F(0), T)
        assert (fd == tss) == (T == H)
