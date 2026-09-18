"""EXACT identities and counterexamples of docs/MDN_NESTEROV_QHM_AUDIT.md.

Pure Python `fractions.Fraction` on the reference forms of
`experiments/prospective_momentum/optimizer_forms.py`: no JAX, no NumPy, no
floating point. An identity is asserted as an exact equality; a counterexample
is an exact inequality.
"""

import random
from fractions import Fraction as F

from experiments.prospective_momentum import optimizer_forms as OF

RNG = random.Random(20260918)
#: rational unit keys (Pythagorean), so key normalization is exact
UNIT_KEYS = ((F(1), F(0)), (F(0), F(1)), (F(3, 5), F(4, 5)),
             (F(4, 5), F(-3, 5)), (F(5, 13), F(12, 13)), (F(-8, 17), F(15, 17)))


def unit(lo=1, hi=99):
    """A rational in (0, 1)."""
    return F(RNG.randint(lo, hi), 100)


def gates():
    """alpha, beta, mu in (0, 1); eta in (0, 2): the pinned gate ranges."""
    return (unit(), unit(), unit(), F(RNG.randint(1, 199), 100))


def tokens(n, d_v=2, masks=None):
    toks = []
    for t in range(n):
        k = list(RNG.choice(UNIT_KEYS))
        v = [F(RNG.randint(-9, 9), 7) for _ in range(d_v)]
        m = F(1) if masks is None else F(masks[t])
        toks.append((k, v, m))
    return toks


def z(d_v=2, d_k=2):
    return OF.zeros(d_v, d_k, F(0))


# ------------------------------------------------ 1. hand-computed tokens --
def test_hand_computed_one_two_three_token_updates():
    """d_v = d_k = 1, k = v = m = 1, alpha = beta = mu = 1/2, eta = 1."""
    tok = ([F(1)], [F(1)], F(1))
    g = (F(1, 2), F(1, 2), F(1, 2), F(1))
    c0 = ([[F(0)]], [[F(0)]])
    nag = OF.run(OF.nesterov_step, c0, [tok] * 3, [g] * 3)
    assert [(c[0][0][0], c[1][0][0]) for c in nag] == [
        (F(1, 2), F(-1)), (F(3, 4), F(-1)), (F(13, 16), F(-7, 8))]
    nat = OF.run(OF.native_step, c0, [tok] * 3, [g] * 3)
    assert [(c[0][0][0], c[1][0][0]) for c in nat] == [
        (F(1, 2), F(-1)), (F(7, 8), F(-5, 4)), (F(33, 32), F(-19, 16))]
    qhm = OF.run(OF.qhm_step, c0, [tok] * 3, [g] * 3, F(1, 2))
    assert [(c[0][0][0], c[1][0][0]) for c in qhm] == [
        (F(1, 2), F(-1)), (F(3, 4), F(-5, 4)), (F(27, 32), F(-5, 4))]


# ------------------------------------------- 2. literal Nesterov, exactly --
def test_nesterov_two_step_form_shares_the_extrapolation_with_native():
    """With token gates: L_t = alpha_t X_(t-1) + (beta_t mu_t/beta_(t-1))
    (X_(t-1) - alpha_(t-1) X_(t-2)); Nesterov X_t = L_t - beta_t eta_t
    G_t(L_t), native X_t = L_t - beta_t eta_t G_t(alpha_t X_(t-1))."""
    for _ in range(30):
        n = 6
        toks, gs = tokens(n, masks=[RNG.randint(0, 1) for _ in range(n)]), \
            [gates() for _ in range(n)]
        for step, at_lookahead in ((OF.nesterov_step, True),
                                   (OF.native_step, False)):
            X = [c[0] for c in OF.run(step, (z(), z()), toks, gs)]
            for t in range(2, n):
                a, b, mu, eta = gs[t]
                ap, bp = gs[t - 1][0], gs[t - 1][1]
                Xm1, Xm2 = X[t - 1], X[t - 2]
                L = OF.lin((a, Xm1), (b * mu / bp, Xm1), (-(b * mu / bp) * ap,
                                                          Xm2))
                P = L if at_lookahead else OF.lin((a, Xm1))
                G = OF.residual(P, *toks[t])
                assert X[t] == OF.lin((1, L), (-b * eta, G))


def test_nesterov_momentum_is_native_momentum_damped_along_the_key():
    """R(L_t) = R(Wbar_t) - m beta mu (U k) k^T, so
    U_t = mu U + eta R(Wbar) - eta m beta mu (U k) k^T: the key-subspace
    component of the carried momentum is also tested against the loss."""
    for _ in range(50):
        W, U = [[F(RNG.randint(-9, 9), 5) for _ in range(2)] for _ in range(2)], \
            [[F(RNG.randint(-9, 9), 5) for _ in range(2)] for _ in range(2)]
        (tok,), (g,) = tokens(1, masks=[RNG.randint(0, 1)]), (gates(),)
        a, b, mu, eta = g
        k, v, m = tok
        W2, U2 = OF.nesterov_step((W, U), tok, g)
        Wb = OF.lin((a, W))
        corr = [[m * x for x in row] for row in OF.outer(OF.matvec(U, k), k)]
        assert U2 == OF.lin((mu, U), (eta, OF.residual(Wb, *tok)),
                            (-eta * b * mu, corr))
        assert W2 == OF.lin((1, Wb), (-b, U2))


def test_zero_momentum_and_nonwrite_boundaries():
    """mu = 0: Nesterov = native = gated delta rule. m = 0 (no write): the
    residual vanishes, so Nesterov = native exactly on that token."""
    for _ in range(30):
        n = 5
        toks = tokens(n)
        gs = [(g[0], g[1], F(0), g[3]) for g in (gates() for _ in range(n))]
        assert OF.run(OF.nesterov_step, (z(), z()), toks, gs) == \
            OF.run(OF.native_step, (z(), z()), toks, gs)
        W, U = OF.run(OF.native_step, (z(), z()), tokens(3), [gates()] * 3)[-1]
        (tok,) = tokens(1, masks=[0])
        g = gates()
        assert OF.nesterov_step((W, U), tok, g) == OF.native_step((W, U), tok,
                                                                   g)


def test_declared_convention_vs_previous_velocity_convention():
    """c_t = beta_t mu_t (the momentum part of THIS step) and the Sutskever
    previous-velocity reading c'_t = mu_t beta_(t-1) coincide iff beta is
    constant; the audit declares c_t, the point this token's step reaches."""
    b_prev, b_now, mu = F(1, 4), F(3, 4), F(1, 2)
    assert b_now * mu != mu * b_prev
    assert F(1, 2) * mu == mu * F(1, 2)


# ---------------------------------------- 3. two-tap and Nesterov bridge --
def test_fixed_gates_alpha_one_two_tap_at_kappa_mu_is_nesterov_read_ahead():
    """alpha = 1, fixed gates, kappa = mu: the two-tap's stored W_t equals
    Nesterov's NEXT lookahead L_(t+1); its residuals are Nesterov's. The two
    read DIFFERENT points: already after the first write W_1 != X_1."""
    for _ in range(30):
        n = 6
        b, mu, eta = unit(), unit(), F(RNG.randint(1, 199), 100)
        g = (F(1), b, mu, eta)
        toks = tokens(n, masks=[RNG.randint(0, 1) for _ in range(n)])
        toks[0] = (toks[0][0], toks[0][1], F(1))
        nag = OF.run(OF.nesterov_step, (z(), z()), toks, [g] * n)
        tt = OF.run(OF.two_tap_step, (z(), z(), z()), toks, [g] * n, mu)
        for t in range(n):
            assert tt[t][0] == OF.lookahead(nag[t], g)
        if any(x != 0 for x in toks[0][1]):
            assert tt[0][0] != nag[0][0]


def test_fixed_gates_general_alpha_needs_a_rescaled_step():
    """Fixed alpha != 1: Nesterov's lookahead is alpha W of a two-tap with
    kappa* = alpha mu/(alpha + mu - alpha mu) and beta' = beta (alpha + mu -
    alpha mu)/alpha. With the executed beta (no step freedom) no kappa works."""
    for _ in range(30):
        n = 6
        a, b, mu, eta = gates()
        s = a + mu - a * mu
        kap, bp = a * mu / s, b * s / a
        toks = tokens(n, masks=[RNG.randint(0, 1) for _ in range(n)])
        nag = OF.run(OF.nesterov_step, (z(), z()), toks, [(a, b, mu, eta)] * n)
        tt = OF.run(OF.two_tap_step, (z(), z(), z()), toks,
                    [(a, bp, mu, eta)] * n, kap)
        for t in range(n):
            assert OF.lin((a, tt[t][0])) == OF.lookahead(nag[t], (a, b, mu,
                                                                  eta))


def test_token_gates_make_the_two_tap_bridge_noncausal():
    """SMALLEST COUNTEREXAMPLE, two tokens. Sequences A and B share token 1
    and its gates and differ only in token 2's (beta, mu). Nesterov's token-2
    evaluation point L_2 differs between them, while every causal state after
    token 1 - in particular the two-tap's W_1, for any kappa - is identical.
    No causal two-tap (or any causal state map) can equal L_2 in both."""
    k, v = [F(1), F(0)], [F(1), F(-1)]
    tok1 = (k, v, F(1))
    g1 = (F(1), F(1, 2), F(1, 2), F(1))
    gA, gB = (F(1), F(1, 2), F(1, 2), F(1)), (F(1), F(1, 4), F(3, 4), F(1))
    nag1 = OF.nesterov_step((z(), z()), tok1, g1)
    LA, LB = OF.lookahead(nag1, gA), OF.lookahead(nag1, gB)
    assert LA != LB
    # the fixed-gate identity (kappa = mu = 1/2, alpha = 1) holds for A, whose
    # token-2 gates equal token 1's, and fails for B
    tt1 = OF.two_tap_step((z(), z(), z()), tok1, g1, F(1, 2))
    assert tt1[0] == LA and tt1[0] != LB
    # and no other constant kappa rescues B without breaking A
    for kappa in (F(0), F(1, 4), F(1), F(3, 2)):
        w1 = OF.two_tap_step((z(), z(), z()), tok1, g1, kappa)[0]
        assert not (w1 == LA and w1 == LB)


def test_two_tap_is_qhm_exactly_when_mu_and_eta_are_constant():
    """For constant mu, eta and ARBITRARY alpha_t, beta_t, keys, values and
    masks (closed loop): two-tap(kappa) = QHM with
    nu = 1 - kappa/(mu (1 + kappa)) and step scale (1 + kappa)."""
    for _ in range(30):
        n = 7
        mu, eta = unit(), F(RNG.randint(1, 199), 100)
        kappa = F(RNG.randint(0, 300), 100)
        gs = [(unit(), unit(), mu, eta) for _ in range(n)]
        toks = tokens(n, masks=[RNG.randint(0, 1) for _ in range(n)])
        nu = 1 - kappa / (mu * (1 + kappa))
        tt = OF.run(OF.two_tap_step, (z(), z(), z()), toks, gs, kappa)
        qh = OF.run(OF.qhm_raw_step, (z(), z()), toks, gs, nu, 1 + kappa)
        assert [c[0] for c in tt] == [c[0] for c in qh]


def test_token_varying_eta_breaks_the_qhm_bridge():
    """SMALLEST COUNTEREXAMPLE, two writes: eta changes between them."""
    k, v = [F(1), F(0)], [F(1), F(0)]
    toks = [(k, v, F(1)), (k, v, F(1))]
    mu, kappa = F(1, 2), F(1, 2)
    gs = [(F(1), F(1, 2), mu, F(1)), (F(1), F(1, 2), mu, F(3, 2))]
    nu = 1 - kappa / (mu * (1 + kappa))
    tt = OF.run(OF.two_tap_step, (z(), z(), z()), toks, gs, kappa)
    qh = OF.run(OF.qhm_raw_step, (z(), z()), toks, gs, nu, 1 + kappa)
    assert tt[0][0] == qh[0][0]
    assert tt[1][0] != qh[1][0]


def test_qhm_weights_of_the_two_tap():
    for _ in range(100):
        mu = unit()
        nu = lambda kap: 1 - kap / (mu * (1 + kap))            # noqa: E731
        assert nu(mu) == mu / (1 + mu)
        assert nu(mu / (1 - mu)) == 0
        kap = F(RNG.randint(0, 500), 100)
        assert (0 <= nu(kap) <= 1) == (kap <= mu / (1 - mu))


# ------------------------------------------ 4. literal TSS and the family --
def test_tss_at_T_equal_h_is_the_two_tap_at_kappa_one():
    for _ in range(20):
        n = 6
        toks = tokens(n, masks=[RNG.randint(0, 1) for _ in range(n)])
        gs = [gates() for _ in range(n)]
        coeff = OF.filter_coefficients(F(0), F(0), F(1))
        assert coeff == (0, 0, 2, 1)
        fl = OF.run(OF.filtered_step, (z(),) * 5, toks, gs, coeff)
        tt = OF.run(OF.two_tap_step, (z(),) * 3, toks, gs, F(1))
        assert [c[0] for c in fl] == [c[0] for c in tt]


def test_literal_tss_is_never_nesterov():
    """(i) first write: TSS writes -beta eta (1 + h/T) g_1, Nesterov
    -beta eta g_1, for every finite T > 0; (ii) for T != h the processing
    filter's pole a = 1 - h/T is not cancelled (zero d/c = T/(T+h) != a), so
    TSS carries an extra pole that Nesterov's recurrence does not have; at
    T = h it is the two-tap at kappa = 1, which matches Nesterov only if
    kappa* = alpha mu/(alpha + mu - alpha mu) = 1, i.e. alpha = mu = 1,
    outside the pinned gates (alpha, mu < 1)."""
    k, v = [F(1), F(0)], [F(1), F(0)]
    tok = (k, v, F(1))
    for _ in range(50):
        g = gates()
        T = F(RNG.randint(1, 800), 100)
        coeff = OF.filter_coefficients(F(0), F(0), T)
        fl = OF.filtered_step((z(),) * 5, tok, g, coeff)
        nag = OF.nesterov_step((z(), z()), tok, g)
        assert fl[0] != nag[0]
        a_pole, c, d = coeff[0], coeff[2], coeff[3]
        if T != 1:
            assert d / c != a_pole
        a, _, mu, _ = g
        assert a * mu / (a + mu - a * mu) != 1
        assert a + mu - 2 * a * mu > 0


def test_generalized_family_contains_the_two_tap_up_to_kappa_one():
    for kap in (F(0), F(1, 4), F(1, 2), F(1)):
        assert OF.filter_coefficients(F(0), 1 - kap, kap) == (0, 0, 1 + kap,
                                                              kap)


# --------------------------------------------- 5. frozen-token stability --
def test_key_aligned_transitions_match_the_steps():
    """Unit key, v = 0, m = 1, one coordinate: stepping the reference forms
    on basis carries reproduces the declared 2x2 matrices exactly."""
    for _ in range(50):
        g = gates()
        nu = unit()
        tok = ([F(1)], [F(0)], F(1))
        for rule, step, extra in (("native", OF.native_step, ()),
                                  ("nesterov", OF.nesterov_step, ()),
                                  ("qhm", OF.qhm_step, (nu,))):
            e, o = ([[F(1)]], [[F(0)]]), ([[F(0)]], [[F(1)]])
            c1, c2 = step(e, tok, g, *extra), step(o, tok, g, *extra)
            A = [[c1[0][0][0], c2[0][0][0]], [c1[1][0][0], c2[1][0][0]]]
            assert A == OF.key_aligned_2x2(rule, *g, nu=nu)


def test_nesterov_closed_forms_and_its_stability_condition():
    """det = alpha mu (1-q), tr = (alpha+mu)(1-q), and
    1 + tr + det = 1 + (1-q)(alpha + mu + alpha mu): Nesterov is frozen-token
    stable iff (q - 1)(alpha + mu + alpha mu) < 1; the other two Jury
    expressions are positive throughout the pinned ranges."""
    for _ in range(300):
        a, b, mu, eta = gates()
        q = b * eta
        A = OF.key_aligned_2x2("nesterov", a, b, mu, eta)
        J1, J2, J3 = OF.jury(A)
        assert A[0][0] * A[1][1] - A[0][1] * A[1][0] == a * mu * (1 - q)
        assert J2 == 1 + (1 - q) * (a + mu + a * mu)
        assert J1 > 0 and J3 > 0
        assert (J2 > 0) == ((q - 1) * (a + mu + a * mu) < 1)
        assert all(j > 0 for j in OF.jury(OF.key_aligned_2x2("native", a, b,
                                                             mu, eta)))


def test_nesterov_can_be_frozen_token_unstable_where_native_is_stable():
    a, b, mu, eta = F(9, 10), F(1), F(9, 10), F(19, 10)
    assert min(OF.jury(OF.key_aligned_2x2("nesterov", a, b, mu, eta))) < 0
    assert min(OF.jury(OF.key_aligned_2x2("native", a, b, mu, eta))) > 0


def test_qhm_is_frozen_token_stable_on_the_whole_declared_domain():
    for _ in range(300):
        a, b, mu, eta = gates()
        nu = F(RNG.randint(0, 100), 100)
        assert min(OF.jury(OF.key_aligned_2x2("qhm", a, b, mu, eta,
                                              nu=nu))) > 0


def test_qhm_boundaries():
    """nu = 1 is native exactly; nu = 0 is the delta rule on Wbar."""
    for _ in range(20):
        n = 5
        toks = tokens(n, masks=[RNG.randint(0, 1) for _ in range(n)])
        gs = [gates() for _ in range(n)]
        assert OF.run(OF.qhm_step, (z(), z()), toks, gs, F(1)) == \
            OF.run(OF.native_step, (z(), z()), toks, gs)
        W = z()
        for tok, g in zip(toks, gs):
            a, b, _, eta = g
            Wb = OF.lin((a, W))
            W = OF.lin((1, Wb), (-b * eta, OF.residual(Wb, *tok)))
        assert OF.run(OF.qhm_step, (z(), z()), toks, gs, F(0))[-1][0] == W


def test_every_form_is_causal():
    """Changing tokens t+1.. never changes the carries at tokens <= t."""
    n, cut = 6, 3
    toks, gs = tokens(n), [gates() for _ in range(n)]
    toks2 = toks[:cut] + tokens(n - cut)
    gs2 = gs[:cut] + [gates() for _ in range(n - cut)]
    for step, c0, extra in ((OF.native_step, (z(), z()), ()),
                            (OF.nesterov_step, (z(), z()), ()),
                            (OF.qhm_step, (z(), z()), (F(1, 3),)),
                            (OF.two_tap_step, (z(),) * 3, (F(1, 2),)),
                            (OF.filtered_step, (z(),) * 5,
                             (OF.filter_coefficients(F(1, 4), F(1, 2),
                                                     F(3, 2)),))):
        a = OF.run(step, c0, toks, gs, *extra)
        b = OF.run(step, c0, toks2, gs2, *extra)
        assert a[:cut] == b[:cut]


# ---------------------------------- 6. the two-tap arm versus the QHM arm --
def test_hand_computed_two_tap_tokens():
    """d_v = d_k = 1, k = v = m = 1, alpha = beta = mu = 1/2, eta = 1,
    kappa = 1/2: W = 3/4, 31/32, 259/256."""
    tok = ([F(1)], [F(1)], F(1))
    g = (F(1, 2), F(1, 2), F(1, 2), F(1))
    c0 = ([[F(0)]], [[F(0)]], [[F(0)]])
    tt = OF.run(OF.two_tap_step, c0, [tok] * 3, [g] * 3, F(1, 2))
    assert [(c[0][0][0], c[1][0][0]) for c in tt] == [
        (F(3, 4), F(-3, 2)), (F(31, 32), F(-19, 16)),
        (F(259, 256), F(-135, 128))]


def test_the_ladder_qhm_arm_equals_the_two_tap_arm_only_at_kappa_zero():
    """The ladder's QHM arm has step weights nu + (1 - nu) = 1; the two-tap's
    QHM form has weights summing to 1 + kappa. So even with CONSTANT mu and
    eta the two ARMS agree token by token only at kappa = 0 (nu = 1, both
    native); for kappa > 0 the two-tap needs QHM with a step (1 + kappa)
    beta, which the QHM arm cannot express."""
    for _ in range(20):
        n = 5
        mu, eta = unit(), F(RNG.randint(1, 199), 100)
        gs = [(unit(), unit(), mu, eta) for _ in range(n)]
        toks = tokens(n)
        tt0 = OF.run(OF.two_tap_step, (z(),) * 3, toks, gs, F(0))
        q1 = OF.run(OF.qhm_step, (z(), z()), toks, gs, F(1))
        assert [c[0] for c in tt0] == [c[0] for c in q1]
        kappa = F(RNG.randint(1, 100), 100)
        nu = 1 - kappa / (mu * (1 + kappa))
        tt = OF.run(OF.two_tap_step, (z(),) * 3, toks, gs, kappa)
        qa = OF.run(OF.qhm_step, (z(), z()), toks, gs, nu)
        assert tt[0][0] != qa[0][0]           # first write already differs
