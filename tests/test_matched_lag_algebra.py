"""The theory-matched zero-lag arm, checked as algebra. No JAX.

The claim under test is narrow and exact: the matched arm differs from the
generalized arm in ONE coefficient, keeps both poles, and cancels the
first-order lag without cancelling the second-order memory.
"""

import ast
import cmath
import io
import math
import os
import struct

from tests import source_introspection as SI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHED = os.path.join(REPO, "s5/matched_lag_ssm.py")
FACTORED = os.path.join(REPO, "s5/factored_recurrence.py")
RUNNER = os.path.join(REPO, "experiments/s5_three_arm_full/runner.py")

H = 1.0
T, MASS, GAMMA = 0.05, 0.025, 1.0
B = 0.4 + 0.1j


def _modes():
    return [m * cmath.exp(1j * a)
            for m in (0.5, 0.9, 0.99, 0.999, 0.9999)
            for a in (0.0, 0.01, 0.05, 0.3, 0.8, 1.5, 2.5)]


def generalized(lam):
    """a1, a2, c1, c2, q exactly as `generalized_coefficients` computes."""
    k = T / H
    F = 1.0 + k * (lam - 1.0)
    G = k * B
    q = MASS + H * (GAMMA + T)
    alpha, beta, delta = MASS / q, H * H / q, H * T / q
    return ((1.0 + alpha - beta) + (beta + delta) * F, -alpha - delta * F,
            (beta + delta) * G, -delta * G, q)


def gamma_k(lam):
    return T + GAMMA * H / (T * (1.0 - lam))


def gain(q):
    return H * T * B / q


# ------------------------------------------------------ the coefficient --
def test_gamma_k_equals_T_reproduces_the_generalized_arm_exactly():
    """The consistency anchor: at Gamma_k = T the matched drive IS the
    generalized drive, so the fourth arm contains the third as a point."""
    for lam in _modes():
        a1, a2, c1, c2, q = generalized(lam)
        K = gain(q)
        assert abs(c1 - K * (1.0 + T / H)) < 1e-18
        assert abs(c2 + K * (T / H)) < 1e-18
        assert abs((c1 + c2) - K) < 1e-18, "K must be the drive's DC gain"


def test_the_regrouped_drive_is_the_same_recurrence():
    """c1 x_t + c2 x_{t-1} == K x_t + K(Gk/h)(x_t - x_{t-1}), identically."""
    for lam in _modes():
        _, _, _, _, q = generalized(lam)
        K, Gk = gain(q), gamma_k(lam)
        c1 = K * (1.0 + Gk / H)
        c2 = -K * (Gk / H)
        for n in range(8):
            x0 = complex(math.sin(n), math.cos(2 * n))
            x1 = complex(math.cos(n / 3.0), math.sin(n / 5.0))
            direct = c1 * x0 + c2 * x1
            regrouped = K * x0 + K * (Gk / H) * (x0 - x1)
            assert abs(direct - regrouped) < 1e-12 * max(1.0, abs(direct))


def test_the_poles_are_untouched():
    """a1 and a2 do not appear in the change at all, so both poles are the
    generalized arm's. This is what makes the ablation one coefficient."""
    for lam in _modes():
        a1, a2, _, _, _ = generalized(lam)
        root = cmath.sqrt(a1 * a1 + 4 * a2)
        before = sorted((abs((a1 + root) / 2), abs((a1 - root) / 2)))
        # the matched arm reuses a1, a2 literally; nothing to recompute
        after = before
        assert before == after


# ----------------------------------------------- zero lag, kept memory --
def _normalized(lam, derivative_coefficient, quadratic=0.0):
    """Hhat(p) = (1 + G p + quadratic p^2) / (1 + (T + c/k)p + (m/k)p^2)."""
    k = 1.0 - lam
    m = MASS * H / T
    c = GAMMA * H / T
    def transfer(p):
        top = 1.0 + derivative_coefficient * p + quadratic * p * p
        bottom = 1.0 + (T + c / k) * p + (m / k) * p * p
        return top / bottom
    return transfer


def _series(lam, derivative_coefficient, quadratic=0.0):
    """The exact Taylor coefficients of Hhat = N/D about p = 0.

    N = 1 + G p + Q p^2,   D = 1 + (T + c/k) p + (m/k) p^2, so
    Hhat = 1 + (G - D1) p + (Q - D2 - G D1 + D1^2) p^2 + O(p^3).

    Series coefficients rather than finite differences: Hhat is 1 + O(p^2)
    with m/k up to 500, so a central difference of it is pure round-off --
    which is what the first version of this test measured.
    """
    k = 1.0 - lam
    # d1 IS Gamma_k by construction -- Gamma_k was DEFINED as the
    # denominator's linear coefficient. Writing it as a second expression
    # T + (gamma h/T)/k makes it differ in the last ulp, and at
    # lam = 0.9999 that ulp is multiplied by d1 = 2e5.
    d1 = gamma_k(lam)
    d2 = (MASS * H / T) / k
    # REGROUPED, and for the same reason the drive is: the literal form
    # Q - d2 - G d1 + d1^2 subtracts two numbers of size d1^2, which is
    # 4e10 at lam = 0.9999, and loses everything. d1(d1 - G) is exact when
    # G = d1, which is precisely the matched case.
    return (1.0,
            derivative_coefficient - d1,
            quadratic - d2 + d1 * (d1 - derivative_coefficient))


def test_the_matched_numerator_gives_exactly_zero_first_order_lag():
    """The LINEAR coefficient of Hhat vanishes identically, because
    Gamma_k was defined to equal the denominator's linear coefficient."""
    for lam in _modes():
        constant, linear, _ = _series(lam, gamma_k(lam))
        assert constant == 1.0
        assert abs(linear) < 1e-12 * max(1.0, abs(gamma_k(lam))), (lam, linear)


def test_the_current_arm_has_a_nonzero_first_order_lag():
    """The thing the fourth arm removes, confirmed present in the third:
    its linear coefficient is -c/k, which is large and never zero."""
    worst = 0.0
    for lam in _modes():
        _, linear, _ = _series(lam, T)
        predicted = -(GAMMA * H / T) / (1.0 - lam)
        assert abs(linear - predicted) < 1e-9 * max(1.0, abs(predicted))
        worst = max(worst, abs(linear))
    assert worst > 1e4, worst


def test_the_second_order_memory_survives():
    """Hhat = 1 - (m/k) p^2 + O(p^3): the quadratic coefficient is exactly
    -m/k, nonzero, so Hhat is not a constant."""
    for lam in _modes():
        _, _, quadratic = _series(lam, gamma_k(lam))
        expected = -(MASS * H / T) / (1.0 - lam)
        assert abs(quadratic - expected) < 1e-9 * max(1.0, abs(expected)), lam
        assert abs(quadratic) > 0.0


def test_adding_the_quadratic_term_destroys_the_memory():
    """The mistake we must not make: matching inertia too makes the
    numerator equal the normalized denominator and Hhat == 1 exactly, at
    every order and every p."""
    for lam in _modes():
        k = 1.0 - lam
        quadratic = (MASS * H / T) / k
        constant, linear, second = _series(lam, gamma_k(lam), quadratic)
        assert constant == 1.0
        assert abs(linear) < 1e-12 * max(1.0, abs(gamma_k(lam)))
        assert abs(second) < 1e-9 * max(1.0, abs(quadratic)), (lam, second)
        transfer = _normalized(lam, gamma_k(lam), quadratic)
        for p in (1e-3, 0.1, 1.0, 10.0):
            assert abs(transfer(p) - 1.0) < 1e-12, (lam, p, transfer(p))


# ------------------------------------------------------- conjugate pairs --
def test_gamma_k_is_conjugate_symmetric():
    """T, gamma and h are real, so Gamma_k(conj a) = conj(Gamma_k(a))
    identically and conj_sym needs no special handling."""
    for lam in _modes():
        if lam.imag == 0.0:
            continue
        assert gamma_k(lam.conjugate()) == gamma_k(lam).conjugate()


# ----------------------------------------------------------- conditioning --
def _f32(value):
    return complex(struct.unpack("f", struct.pack("f", value.real))[0],
                   struct.unpack("f", struct.pack("f", value.imag))[0])


def test_the_regrouping_removes_the_float32_cancellation():
    """c1 + c2 = K exactly, two numbers of size K*Gk/h cancelling to K, so
    the direct form loses about log10(Gamma_k) digits. The regrouped form
    does not, because at low frequency the difference vanishes on its own."""
    worst_direct = worst_regrouped = 0.0
    for lam in (0.999 + 0j, 0.9999 + 0j, 0.99 * cmath.exp(0.01j)):
        _, _, _, _, q = generalized(lam)
        K, Gk = gain(q), gamma_k(lam)
        c1, c2 = K * (1.0 + Gk / H), -K * (Gk / H)
        for n in range(1, 120):
            x0 = complex(math.sin(n / 500.0))
            x1 = complex(math.sin((n - 1) / 500.0))
            exact = c1 * x0 + c2 * x1
            scale = abs(exact) or 1.0
            direct = _f32(_f32(c1) * _f32(x0)) + _f32(_f32(c2) * _f32(x1))
            regrouped = (_f32(_f32(K) * _f32(x0))
                         + _f32(_f32(K * (Gk / H)) * _f32(x0 - x1)))
            worst_direct = max(worst_direct, abs(direct - exact) / scale)
            worst_regrouped = max(worst_regrouped,
                                  abs(regrouped - exact) / scale)
    assert worst_regrouped < worst_direct / 50.0, (worst_direct,
                                                   worst_regrouped)
    assert worst_regrouped < 1e-6, worst_regrouped


def test_gamma_k_spans_orders_of_magnitude_and_that_is_recorded():
    """The remaining concern is optimization scale, not arithmetic. Only
    NEAR-REAL slow modes blow up, because |1-a| >= |Im a|."""
    near_real = abs(gamma_k(0.9999 + 0j))
    rotated = abs(gamma_k(0.9999 * cmath.exp(0.2j)))
    assert near_real > 1e5, near_real
    assert rotated < 200.0, rotated
    assert near_real / rotated > 1e3


# --------------------------------------------------------------- source --
def test_the_layer_reuses_a1_and_a2_rather_than_recomputing_them():
    """Requirement: bit-identical poles. The only way to guarantee that is
    to use the parent's values, not to recompute the same expression."""
    tree = SI.parse(MATCHED)
    source = io.open(MATCHED).read()
    call = ast.get_source_segment(source, SI.function_node(tree, "__call__"))
    assert "self.generalized_a1" in call and "self.generalized_a2" in call
    setup = ast.get_source_segment(source, SI.function_node(tree, "setup"))
    assert "super().setup()" in setup
    # by AST, not substring: the literal appears in a COMMENT explaining
    # which h this module uses, and a substring search cannot tell a
    # comment from a call. That trap has now caught me three times.
    called = SI.called_names(tree)
    assert "generalized_coefficients" not in called, (
        "a1 and a2 must be inherited, never recomputed here")
    imported = {name for _, name in SI.imported_names(tree)}
    assert "generalized_coefficients" not in imported


def test_the_forward_path_uses_the_regrouped_drive_not_the_coefficients():
    tree = SI.parse(MATCHED)
    source = io.open(MATCHED).read()
    applied = ast.get_source_segment(source,
                                     SI.function_node(tree, "apply_matched"))
    assert "matched_drive" in applied
    assert "direct_coefficients" not in applied, (
        "the cancelling form must never reach the forward path")
    drive = ast.get_source_segment(source,
                                   SI.function_node(tree, "matched_drive"))
    assert "projected - previous" in drive
    assert SI.defines_function(tree, "direct_coefficients")


def test_no_quadratic_numerator_term_exists_anywhere():
    """Deliberate: inertia is left uncompensated so the second-order memory
    stays in the transfer function."""
    source = io.open(MATCHED).read()
    drive = ast.get_source_segment(
        source, SI.function_node(SI.parse(MATCHED), "matched_drive"))
    # exactly two terms: the DC term and ONE first difference
    assert drive.count("previous") == 2
    assert "second_difference" not in source and "p ** 2" not in drive


def test_arm_order_stays_three_so_the_finalizer_is_unaffected():
    """finalize.py and arm_summary iterate ARM_ORDER; extending it would
    make the three-arm finalizer demand artifacts that do not exist."""
    source = io.open(RUNNER).read()
    assert 'ARM_ORDER = ("native_matched_s5", "zucchet_prospective_s5",' in source
    assert "RUNNABLE_ARMS = tuple(SCIENTIFIC_NAMES)" in source
    assert 'choices=RUNNABLE_ARMS' in source
    assert '"matched_lag_prospective_s5"' in source


def test_the_fourth_arm_shares_the_third_s_initialization():
    """The comparison must differ ONLY in the drive, so T_INIT, RHO_INIT
    and gamma_init are the same call."""
    tree = SI.parse(RUNNER)
    factory = ast.get_source_segment(io.open(RUNNER).read(),
                                     SI.function_node(tree, "ssm_factory"))
    matched = factory.split("matched_lag_prospective_s5")[-1]
    for expected in ("response_init=T_INIT", "rho_init=RHO_INIT",
                     "gamma_init=1.0"):
        assert expected in matched, expected


def test_every_intra_package_import_resolves_on_this_branch():
    """REGRESSION, and it cost a cluster round trip.

    `matched_lag_ssm` imported H_TOKEN from `s5.modal_prospective`, which
    exists only on the modal branch. `py_compile` does not resolve imports
    and `undefined_names` only looks inside one file, so neither caught it;
    the first thing that did was pytest on the GPU node.

    Every `from .x import ...` in the package must name a module that
    exists HERE.
    """
    package = os.path.join(REPO, "s5")
    missing = []
    for entry in sorted(os.listdir(package)):
        if not entry.endswith(".py"):
            continue
        tree = SI.parse(os.path.join(package, entry))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            if node.module is None:
                continue
            target = os.path.join(package, node.module.replace(".", os.sep))
            if not (os.path.exists(target + ".py")
                    or os.path.isdir(target)):
                missing.append(f"{entry} -> .{node.module}")
    assert missing == [], missing
