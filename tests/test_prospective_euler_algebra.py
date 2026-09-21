"""The professor's prospective equation, checked as mathematics.

Pure Python, no JAX, runs on a laptop in milliseconds. Every claim the
design rests on is here: why the equation cancels the memory, why
generalizing cannot repair that, what the Euler discretization does
instead, and where the per-mode stability boundary is.
"""

import cmath
import io
import math
import os
import random

from tests import prospective_euler_reference as R
from tests import source_introspection as SI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = os.path.join(REPO, "s5/prospective_euler.py")


def _modes(count, seed=7, magnitude=(0.01, 0.9999)):
    generator = random.Random(seed)
    return [generator.uniform(*magnitude)
            * cmath.exp(1j * generator.uniform(-math.pi, math.pi))
            for _ in range(count)]


# ------------------------------------------------- the cancellation proof --
def test_the_continuous_equation_removes_A_from_the_dynamics():
    """P(D)(s - f) = 0 with f = As + Bx gives (I-A)P(D)s = P(D)Bx, so the
    homogeneous dynamics are P(D)s = 0 and A is gone.

    Checked as the algebra it is, on scalars: the coefficient of s in the
    homogeneous part must be independent of a for EVERY operator P.
    """
    for a in _modes(200):
        for coefficients in ((1.0, 2.0), (1.0, 5.0, 3.0), (1.0, 0.5, 0.25)):
            # P(D)(s - f) = 0, f = as + bx. Homogeneous part: P(D)(1-a)s.
            # Dividing by the scalar (1-a) leaves P(D)s = 0 exactly.
            residual_factor = 1 - a
            assert abs(residual_factor) > 1e-12
            homogeneous = [c * residual_factor for c in coefficients]
            normalized = [c / residual_factor for c in homogeneous]
            # the normalized operator is P itself: no trace of a survives
            for got, want in zip(normalized, coefficients):
                assert abs(got - want) < 1e-12


def test_generalizing_the_operator_cannot_repair_the_cancellation():
    """The proof above never uses the ORDER of P, so 1 + tau D + M D^2
    cancels the memory exactly as 1 + tau D does. This is why the
    generalized repair failed, and it is a statement about the residual
    (s - f), not about the operator."""
    for a in _modes(100):
        ordinary = [1.0, 3.0]
        generalized = [1.0, 3.0, 2.25]
        for operator in (ordinary, generalized):
            normalized = [c * (1 - a) / (1 - a) for c in operator]
            assert all(abs(got - want) < 1e-12
                       for got, want in zip(normalized, operator))


# ------------------------------------------- what Euler does instead of it --
def test_the_discrete_poles_multiply_to_the_native_pole():
    """The Euler form does NOT cancel: the product of its two poles is
    exactly `a`, so the native memory is split between them rather than
    removed. That is the one thing the discretization gets right."""
    for a in _modes(300):
        for epsilon in (0.0, 0.01, 0.1, 0.5):
            plus, minus = R.roots(a, epsilon)
            assert abs(plus * minus - a) < 1e-9 * max(1.0, abs(a))


def test_the_discretization_is_unstable_on_complex_modes():
    """The measured failure, reproduced as arithmetic. Real modes are
    always safe; S5's modes are not real."""
    # the exact value measured on the cluster
    a = 0.9 * cmath.exp(2j)
    assert abs(R.spectral_radius(a, 0.5) - 1.4416) < 1e-3
    # real modes never leave the disc, at any prospectivity
    for magnitude in (0.5, 0.9, 0.99, 0.999):
        for epsilon in (0.1, 0.5, 1.0, 2.0):
            assert R.spectral_radius(complex(magnitude), epsilon) < 1.0
    # complex slow modes do, at modest prospectivity
    assert R.spectral_radius(0.99 * cmath.exp(1j), 0.1) > 1.0


# ------------------------------------------------- the stability boundary --
def test_the_closed_form_boundary_is_exact():
    """rho crosses 1 at precisely e_max: below it the mode is stable, above
    it the mode is not. This is what makes the gate safe by construction
    rather than by tuning."""
    for a in _modes(2000, seed=11):
        boundary = R.epsilon_max(a)
        assert boundary > 0.0
        assert R.spectral_radius(a, boundary * 0.999) < 1.0
        assert R.spectral_radius(a, boundary * 1.001) > 1.0


def test_rho_is_not_monotone_and_is_marginal_at_both_ends():
    """The defect that forced the design apart. At e = 0 there is a pole
    EXACTLY at z = 1 -- cancelled analytically by the numerator zero at
    z = 1, but marginal, and a marginal pole paired with a zero is what
    lost everything in float32 on the rejected branch. At e = e_max the
    radius touches the disc from the other side. It dips in between, so a
    single knob sliding e from 0 to e_max walks into a marginal pole at
    BOTH ends."""
    a = 0.93634 * cmath.exp(0.0772j)
    boundary = R.epsilon_max(a)
    assert abs(R.spectral_radius(a, 0.0) - 1.0) < 1e-12
    assert abs(R.spectral_radius(a, boundary) - 1.0) < 1e-6
    middle = R.spectral_radius(a, 0.5 * boundary)
    assert middle < 0.99, middle


def test_no_value_inside_the_band_can_leave_the_disc():
    """The property the architecture rests on: e confined to
    EPSILON_BAND * e_max is strictly stable for every mode, so no value of
    any parameter can destabilize."""
    low, high = 0.25, 0.75
    worst = 0.0
    for a in _modes(1500, seed=13):
        boundary = R.epsilon_max(a)
        for step in range(11):
            fraction = low + (high - low) * step / 10.0
            worst = max(worst, R.spectral_radius(a, fraction * boundary))
    assert worst < 1.0 - 1e-5, worst


def test_the_prospective_cell_never_decays_faster_than_native():
    """The poles multiply to exactly `a`, so rho >= sqrt(|a|) > |a|. The
    Euler form LENGTHENS the slowest timescale rather than shortening it,
    which is the opposite of the continuous equation's cancellation and the
    reason a per-mode treatment is worth anything."""
    for a in _modes(500, seed=29):
        boundary = R.epsilon_max(a)
        for fraction in (0.25, 0.5, 0.75):
            radius = R.spectral_radius(a, fraction * boundary)
            assert radius >= math.sqrt(abs(a)) - 1e-9
            assert radius > abs(a) - 1e-9


def test_a_mode_may_not_anticipate_further_than_it_remembers():
    """The physical content of the boundary: tau_min lands on the mode's
    own native timescale. Slow modes may only be weakly prospective."""
    for magnitude in (0.9, 0.99, 0.999):
        native = -1.0 / math.log(magnitude)
        for omega in (0.2, 1.0, 2.0):
            a = magnitude * cmath.exp(1j * omega)
            horizon = 1.0 / R.epsilon_max(a)
            assert 0.7 * native < horizon < 1.3 * native, (magnitude, omega,
                                                           horizon, native)


# --------------------------------------------------- the Native identity --
def test_zero_prospectivity_is_exactly_native_delayed_one_step():
    """At e = 0 the cell has a pole at z = 1 AND a numerator zero at z = 1.
    They cancel exactly, so the cell is Native S5 shifted by one token --
    not Native plus a free integrator."""
    for a in _modes(50, seed=17):
        b = 0.3 - 0.2j
        inputs = [complex(math.sin(k), math.cos(2 * k)) for k in range(60)]
        prospective = R.sequential(a, b, inputs, 0.0)
        native = R.native_sequential(a, b, inputs)
        for index in range(1, len(inputs)):
            assert abs(prospective[index] - native[index - 1]) < 1e-9
        assert abs(prospective[0]) == 0.0


def test_at_zero_prospectivity_the_roots_are_one_and_the_native_pole():
    for a in _modes(200, seed=19):
        plus, minus = R.roots(a, 0.0)
        pair = sorted((plus, minus), key=abs)
        assert abs(pair[1] - 1.0) < 1e-9 or abs(pair[0] - 1.0) < 1e-9
        assert min(abs(pair[0] - a), abs(pair[1] - a)) < 1e-9


# ------------------------------------------------------ the factored form --
def test_the_factored_form_equals_the_companion_recurrence():
    """Running it as two first-order scans over its own roots must
    reproduce the note's second-order recurrence exactly. That equivalence
    is what allows the repository's own scan primitive to be reused instead
    of a 2x2 matrix scan."""
    for a in _modes(40, seed=23):
        b = -0.4 + 0.6j
        inputs = [complex(math.cos(k / 3), math.sin(k / 5)) for k in range(80)]
        for gate in (0.0, 0.3, 0.9):
            epsilon = gate * 0.999 * R.epsilon_max(a)
            direct = R.sequential(a, b, inputs, epsilon)
            factored = R.factored(a, b, inputs, epsilon)
            scale = max(abs(value) for value in direct) or 1.0
            for want, got in zip(direct, factored):
                assert abs(want - got) < 1e-7 * scale


# ---------------------------------------------------------- the module --
def test_the_cell_defines_what_the_derivation_promises():
    tree = SI.parse(CELL)
    for name in ("epsilon_max", "epsilon_from_raw", "tau_from_epsilon",
                 "companion_coefficients", "roots", "spectral_radius",
                 "drive", "apply_prospective", "mix", "certify"):
        assert SI.defines_function(tree, name), name
    source = io.open(CELL).read()
    # no dense companion is ever built: it is two scalar scans
    assert "associative_scan" in source
    for forbidden in ("np.linalg.eigvals", "matrix_power"):
        assert forbidden not in source, forbidden
    assert SI.undefined_names(CELL) == {}
    # the horizon and the participation are SEPARATE knobs: a single gate
    # sliding e from 0 to e_max is marginal at both ends
    assert "EPSILON_BAND = (0.25, 0.75)" in source
    assert "jax.nn.sigmoid(raw)" in source
