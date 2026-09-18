"""Exact-arithmetic checks for docs/SELECTIVE_GP_S5_AUDIT.md.

Pure Python `fractions.Fraction`: no JAX, no model, no training, no floating
point. Every identity below is polynomial or rational in its parameters, so
agreement at many random rational points is evidence for the identity, and
the explicit counterexample is exact.

Notation (repository convention, `s5/gp_fixed.py`, `s5/gp_second_order.py`):

    M s'' + gamma s' + r + T r' = 0,     r = G s - b

with G, b held constant within a token. The proposal makes G = G_t and
b_t = G_t s_hat_t token dependent; M, gamma, T stay fixed. The jump-free
coordinate is P = M s' + T r.
"""

import random
from fractions import Fraction as F

RNG = random.Random(20260918)


def rq(lo=1, hi=40, den=16):
    """A random rational in [lo, hi]/den (strictly positive for lo >= 1)."""
    return F(RNG.randint(lo, hi), RNG.randint(1, den))


# ------------------------------------------------------------ 2x2 algebra --
def mm(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0],
             A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0],
             A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def add(A, B, s=F(1)):
    return [[A[i][j] + s * B[i][j] for j in range(2)] for i in range(2)]


def det(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def tr(A):
    return A[0][0] + A[1][1]


def inv(A):
    d = det(A)
    return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]


def mv(A, x):
    return [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]


I2 = [[F(1), F(0)], [F(0), F(1)]]


def gen(g, M, gam, T):
    """Continuous generator in (s, P): d/dt (s, P) = A(g) (s, P) + f."""
    c = gam * T - M
    return [[-T * g / M, 1 / M], [c * g / M, -gam / M]]


def forcing(b, M, gam, T):
    c = gam * T - M
    return [T * b / M, -c * b / M]


def bilinear(A):
    """Cayley map (I - A/2)^-1 (I + A/2): rational, Hurwitz -> Schur."""
    return mm(inv(add(I2, A, F(-1, 2))), add(I2, A, F(1, 2)))


def schur_stable(P):
    """Exact Schur-Cohn test for a real 2x2 matrix."""
    d, t = det(P), tr(P)
    return abs(d) < 1 and abs(t) < 1 + d


# ------------------------------------------------ 1. the (s, P) realization --
def test_sP_realization_satisfies_the_generalized_law():
    """Within a token (G, b held): M s'' + gamma s' + r + T r' = 0 exactly."""
    for _ in range(200):
        M, gam, T, g, b = rq(), rq(), rq(0), rq(), rq(-40)
        s, P = rq(-40), rq(-40)
        A, f = gen(g, M, gam, T), forcing(b, M, gam, T)
        sd, Pd = [a + ff for a, ff in zip(mv(A, [s, P]), f)]
        r = g * s - b
        rd = g * sd                      # b held within the token
        sdd = (Pd - T * rd) / M          # from P = M s' + T r
        assert sd == (P - T * r) / M
        assert Pd == -(gam / M) * P + ((gam * T - M) / M) * r
        assert M * sdd + gam * sd + r + T * rd == 0


def test_P_is_the_conserved_form_so_the_state_is_continuous_at_switches():
    """d/dt P = -gamma s' - r contains no derivative of G or b: P (and s)
    stay continuous when G_t, b_t jump at a token boundary, while s' jumps by
    -T (r+ - r-)/M. That is why (s, P), not (s, s'), carries a common
    Lyapunov function across tokens."""
    for _ in range(100):
        M, gam, T = rq(), rq(), rq()
        s, P = rq(-40), rq(-40)
        g1, b1, g2, b2 = rq(), rq(-40), rq(), rq(-40)
        sd1 = (P - T * (g1 * s - b1)) / M
        sd2 = (P - T * (g2 * s - b2)) / M
        assert sd2 - sd1 == -T * ((g2 * s - b2) - (g1 * s - b1)) / M
        # and Pdot = -gamma s' - r in both tokens
        for g, b, sd in ((g1, b1, sd1), (g2, b2, sd2)):
            Pd = -(gam / M) * P + ((gam * T - M) / M) * (g * s - b)
            assert Pd == -gam * sd - (g * s - b)


def test_rank_one_structure_and_invariants():
    """A(g) = A0 + g u e1^T, forcing = -u b, det = g/M, tr = -(Tg+gamma)/M."""
    for _ in range(100):
        M, gam, T, g, b = rq(), rq(), rq(0), rq(), rq(-40)
        u = [-T / M, (gam * T - M) / M]
        A0 = [[F(0), 1 / M], [F(0), -gam / M]]
        A = gen(g, M, gam, T)
        assert A == add(A0, [[g * u[0], F(0)], [g * u[1], F(0)]])
        assert forcing(b, M, gam, T) == [-u[0] * b, -u[1] * b]
        assert det(A) == g / M
        assert tr(A) == -(T * g + gam) / M


def test_transitions_at_different_gains_do_not_commute():
    """[A(g1), A(g2)] = (g2 - g1) [A0, u e1^T], and
    [A0, u e1^T] = [[u2, -u1], [-gamma u2, -u2]]/M vanishes only if
    u1 = -T/M = 0 and u2 = (gamma T - M)/M = 0, i.e. M = 0. So for every
    admissible M > 0 the per-token transitions never commute when g1 != g2:
    the cell is NOT a diagonal (commuting) selective SSM in any fixed basis."""
    for _ in range(100):
        M, gam, T, g1, g2 = rq(), rq(), rq(), rq(), rq()
        if g1 == g2:
            continue
        A1, A2 = gen(g1, M, gam, T), gen(g2, M, gam, T)
        comm = add(mm(A1, A2), mm(A2, A1), F(-1))
        assert comm != [[0, 0], [0, 0]]


# --------------------------------------- 2. boundaries of the proposed cell --
def test_cancellation_line_is_a_first_order_selective_update():
    """M = gamma T: the P row loses its forcing, so P(0) = 0 keeps P = 0 for
    ANY gain sequence, and s' = -(G_t/gamma)(s - s_hat_t): the input-gated
    first-order innovation (minGRU / Mamba-type diagonal selective SSM)."""
    for _ in range(100):
        gam, T, g, sh, s = rq(), rq(), rq(), rq(-40), rq(-40)
        M = gam * T
        A, f = gen(g, M, gam, T), forcing(g * sh, M, gam, T)
        assert A[1][0] == 0 and f[1] == 0
        sd = mv(A, [s, F(0)])[0] + f[0]
        assert sd == -(g / gam) * (s - sh)


def test_executed_gp_rho_arm_is_the_constant_gain_member():
    """gamma = 1, M = rho T, G = J constant: the (s, P) generator is the
    executed `mass_block_generator` (s, v) block of s5/gp_fixed.py under the
    constant rescaling P = -T (1 - rho) v."""
    for _ in range(100):
        T, J, bx = rq(), rq(), rq(-40)
        rho = F(RNG.randint(1, 99), 100)
        M, gam = rho * T, F(1)
        # repository block, s5/gp_fixed.py::mass_block_generator (gamma = 1)
        Av = [[-J / rho, -(1 - rho) / rho], [-J / (rho * T), -1 / (rho * T)]]
        Bv = [bx / rho, bx / (rho * T)]
        k = -T * (1 - rho)                          # P = k v
        S, Sinv = [[F(1), F(0)], [F(0), k]], [[F(1), F(0)], [F(0), 1 / k]]
        assert gen(J, M, gam, T) == mm(mm(S, Av), Sinv)
        assert forcing(bx, M, gam, T) == mv(S, Bv)


def test_m0_boundary_needs_its_own_jump_free_coordinate():
    """M = 0 is singular in (s, P). There Q = gamma s + T r is the conserved
    form: Q' = -r, and with G, b held,
        Q' = -(G/(gamma + T G)) Q + (gamma/(gamma + T G)) b,
        s  = (Q + T b)/(gamma + T G)."""
    for _ in range(100):
        gam, T, g, b, s = rq(), rq(), rq(), rq(-40), rq(-40)
        r = g * s - b
        Q = gam * s + T * r
        assert s == (Q + T * b) / (gam + T * g)
        assert -r == -(g / (gam + T * g)) * Q + (gam / (gam + T * g)) * b


def test_rawat_is_the_m0_member_only_for_a_real_constant_gain():
    """(1 + T p)/((gamma + T G) p + G) equals Rawat's (1 + tau p)/(gR p + G)
    iff T = tau and gamma = gR - tau G. For a complex S5 pole G that gamma is
    complex, i.e. outside a real-coefficient family; for token-varying G_t it
    would have to vary per token."""
    for _ in range(100):
        tau, gR, G, p = rq(), rq(), rq(), rq()
        gam = gR - tau * G
        assert (1 + tau * p) / ((gam + tau * G) * p + G) == \
            (1 + tau * p) / (gR * p + G)


# ----------------------------------------------- 3. switching stability -----
def cqlf_margin(g1, g2, M, gam, T, sqrt_g1, sqrt_g2):
    """(gamma + T g1)(gamma + T g2) - M (sqrt g2 - sqrt g1)^2."""
    return (gam + T * g1) * (gam + T * g2) - M * (sqrt_g2 - sqrt_g1) ** 2


def detK_positive_for_all_lambda(g1, g2, M, gam, T):
    """det(A(g1) + nu A(g2)^-1) > 0 for all nu > 0, decided exactly.

    With lambda = nu / M the determinant is the quadratic
        M g1 + lambda [T^2 g1 + c + (gamma^2 + c g1)/g2] + lambda^2 M/g2."""
    c = gam * T - M
    a0, a2 = M * g1, M / g2
    a1 = T * T * g1 + c + (gam * gam + c * g1) / g2
    return a1 >= 0 or a1 * a1 < 4 * a0 * a2


def test_detK_quadratic_matches_the_matrix_it_summarizes():
    for _ in range(100):
        M, gam, T, g1, g2, lam = rq(), rq(), rq(0), rq(), rq(), rq()
        N1 = [[-T * g1, F(1)], [(gam * T - M) * g1, -gam]]
        N2 = [[-T * g2, F(1)], [(gam * T - M) * g2, -gam]]
        K = add(N1, inv(N2), lam * M)
        c = gam * T - M
        quad = (M * g1 + lam * (T * T * g1 + c + (gam * gam + c * g1) / g2)
                + lam * lam * M / g2)
        assert det(K) == quad


def test_king_nathanson_condition_equals_the_factored_margin():
    """A(g1) - A(g2) is rank one, so (King & Nathanson, after Shorten &
    Narendra) a CQLF exists iff A(g1)A(g2) has no negative real eigenvalue,
    i.e. det(A(g1) + nu A(g2)^-1) != 0 for nu > 0. That is exactly
    (gamma + T g1)(gamma + T g2) > M (sqrt g2 - sqrt g1)^2."""
    checked = 0
    for _ in range(400):
        M, gam, T = rq(), rq(), rq(0)
        r1, r2 = rq(), rq()
        g1, g2 = r1 * r1, r2 * r2                 # rational square roots
        exact = detK_positive_for_all_lambda(g1, g2, M, gam, T)
        margin = cqlf_margin(g1, g2, M, gam, T, r1, r2)
        if margin == 0:
            continue
        assert exact == (margin > 0), (M, gam, T, g1, g2)
        checked += 1
    assert checked > 300


def test_rho_at_most_one_gives_a_cqlf_for_every_gain_range():
    """gamma > 0, 0 < M <= gamma T: margin > 0 for all 0 < g1 <= g2."""
    for _ in range(400):
        gam, T = rq(), rq()
        M = gam * T * F(RNG.randint(1, 100), 100)
        r1, r2 = rq(), rq()
        assert cqlf_margin(r1 * r1, r2 * r2, M, gam, T, r1, r2) > 0


def test_per_token_stable_blocks_can_diverge_under_switching():
    """EXACT counterexample outside rho <= 1. gamma = 1, T = 1/2, M = 1
    (rho = 2), gain alternating between 1/64 and 16 each token, bilinear
    (Cayley) discretization. Each block is Schur stable; the two-token
    product has spectral radius > 1, so the state grows without bound."""
    gam, T, M = F(1), F(1, 2), F(1)
    B1 = bilinear(gen(F(1, 64), M, gam, T))
    B2 = bilinear(gen(F(16), M, gam, T))
    assert schur_stable(B1) and schur_stable(B2)
    P = mm(B2, B1)
    assert abs(det(P)) < 1 and abs(tr(P)) > 1 + det(P)   # a real root < -1
    assert cqlf_margin(F(1, 64), F(16), M, gam, T, F(1, 8), F(4)) < 0


def test_no_divergent_switching_pattern_when_rho_at_most_one():
    """Consistency with the theorem on an exhaustive exact grid."""
    gam = F(1)
    for T in (F(1, 2), F(1), F(5)):
        for rho in (F(1, 100), F(1, 4), F(3, 4), F(1)):
            M = rho * gam * T
            for g1 in (F(1, 64), F(1, 4)):
                for g2 in (F(4), F(64), F(256)):
                    B1 = bilinear(gen(g1, M, gam, T))
                    B2 = bilinear(gen(g2, M, gam, T))
                    for a in range(1, 4):
                        for b in range(1, 4):
                            P = I2
                            for _ in range(a):
                                P = mm(B1, P)
                            for _ in range(b):
                                P = mm(B2, P)
                            assert schur_stable(P), (T, rho, g1, g2, a, b)
