"""GPU/JAX verification for the parallel realizations of the two-compartment
recurrence. Run on the cluster.

The equation is not changed here, so these tests ask exactly one thing in
many ways: does a parallel scan compute what `scan_companion_sequential`
computes? The sequential scan is the oracle throughout and is never
adjusted to meet a parallel result.
"""

import math

import jax
import jax.numpy as np
import numpy as onp

from s5 import factored_recurrence as FR
from s5.discrete_recurrence import (companion_radius, generalized_coefficients,
                                    scan_companion, scan_companion_sequential)
from s5.generalized_prospective_ssm import response_mass_gamma

#: awkward lengths, then the production length
LENGTHS = (1, 2, 3, 5, 17, 63, 257, 1000, 16000)
MODES, FEATURES = 8, 4

#: ALGEBRAIC tolerance, in float64. This is the correctness gate: the three
#: routes compute the same recurrence, so in double precision they must
#: agree to round-off and nothing else.
FLOAT64_TOLERANCE = 1e-10
#: float32 ACCUMULATION tolerance. Not a loosened correctness gate -- the
#: float64 test above is the correctness gate. This one records how much
#: single-precision round-off accumulates over the production length, and
#: it is set from measurement: the whole production inventory (1152 modes,
#: seeds 301-303, length 16,000) came in at 2.3e-4 to 3.5e-4 relative
#: against the sequential oracle, on modes whose spectral radius is 0.99997
#: so nothing decays away. 1e-3 leaves headroom without hiding a real
#: discrepancy, because a real discrepancy would survive into float64.
FLOAT32_TOLERANCE = 1e-3


def _coefficients(key, modes=MODES, largest=0.999, real=False):
    """Coefficients drawn as ROOTS INSIDE THE UNIT DISC, then mapped to
    (a1, a2) with `roots_to_coefficients`.

    REGRESSION. The first version drew a1 and a2 uniformly on [-0.6, 0.6]^2,
    which puts 18 percent of modes at spectral radius >= 1 and reaches
    1.368. At the production length that is 1.368^16000, about 10^2180: the
    companion scan's matrix products overflow to inf, inf*0 gives nan, and
    the test then reported a "failure" that was nothing but an unstable
    test fixture. Production modes are certified at radius <= 0.99998, so
    stable draws are also the faithful ones.
    """
    keys = jax.random.split(key, 4)
    magnitude = jax.random.uniform(keys[0], (2, modes), minval=0.05,
                                   maxval=largest)
    if real:
        roots = (magnitude * jax.random.choice(
            keys[1], np.array([-1.0, 1.0]), (2, modes))).astype(np.complex64)
    else:
        angle = jax.random.uniform(keys[1], (2, modes), minval=-math.pi,
                                   maxval=math.pi)
        roots = (magnitude * np.exp(1j * angle)).astype(np.complex64)
    return FR.roots_to_coefficients(roots[0], roots[1])


def _projections(key, modes=MODES, features=FEATURES):
    keys = jax.random.split(key, 4)
    def make(a, b):
        return (jax.random.normal(keys[a], (modes, features))
                + 1j * jax.random.normal(keys[b], (modes, features))
                ).astype(np.complex64) * 0.3
    return make(0, 1), make(2, 3)


def _inputs(key, length, features=FEATURES):
    return jax.random.normal(key, (length, features), dtype=np.float32)


def _relative(reference, candidate):
    scale = float(np.max(np.abs(reference)))
    if scale == 0.0:
        return float(np.max(np.abs(candidate)))
    return float(np.max(np.abs(reference - candidate))) / scale


# ------------------------------------------------- coefficient algebra --
def test_the_roots_reproduce_the_coefficients_in_jax():
    a1, a2 = _coefficients(jax.random.PRNGKey(0), modes=512, scale=1.5)
    first, second = FR.companion_roots(a1, a2)
    assert _relative(a1, first + second) < 1e-4
    assert _relative(a2, -(first * second)) < 1e-4


def test_the_factored_radius_agrees_with_companion_radius():
    """`companion_radius` is untouched production code; the factored route
    must agree with it, not replace it."""
    a1, a2 = _coefficients(jax.random.PRNGKey(1), modes=512, scale=1.5)
    assert _relative(companion_radius(a1, a2),
                     FR.spectral_radius_from_roots(a1, a2)) < 1e-4


# ------------------------------------------------- sequence equivalence --
def test_both_parallel_scans_match_the_oracle_at_every_length():
    """Awkward lengths included, and the production length 16,000."""
    worst = {}
    for length in LENGTHS:
        key = jax.random.PRNGKey(length)
        a1, a2 = _coefficients(key)
        c1, c2 = _projections(jax.random.split(key)[0])
        values = _inputs(jax.random.split(key)[1], length)
        oracle = scan_companion_sequential(a1, a2, c1, c2, values)
        for name, scan in (("companion", scan_companion),
                           ("factored", FR.scan_factored)):
            error = _relative(oracle, scan(a1, a2, c1, c2, values))
            worst[(length, name)] = error
            assert error < FLOAT32_TOLERANCE, (length, name, error)
    print("worst relative error:", max(worst.values()), worst)


def test_real_modes_are_handled_as_well_as_complex_ones():
    for length in (5, 257, 16000):
        key = jax.random.PRNGKey(length + 7)
        a1, a2 = _coefficients(key, real=True)
        c1, c2 = _projections(jax.random.split(key)[0])
        values = _inputs(jax.random.split(key)[1], length)
        oracle = scan_companion_sequential(a1, a2, c1, c2, values)
        assert _relative(oracle, FR.scan_factored(a1, a2, c1, c2,
                                                  values)) < FLOAT32_TOLERANCE


def test_the_reverse_direction_matches_and_is_not_a_flipped_forward_pass():
    for length in (17, 1000, 16000):
        key = jax.random.PRNGKey(length + 11)
        a1, a2 = _coefficients(key)
        c1, c2 = _projections(jax.random.split(key)[0])
        values = _inputs(jax.random.split(key)[1], length)
        oracle = scan_companion_sequential(a1, a2, c1, c2, values,
                                           reverse=True)
        for scan in (scan_companion, FR.scan_factored):
            assert _relative(oracle, scan(a1, a2, c1, c2, values,
                                          reverse=True)) < FLOAT32_TOLERANCE
        forward = scan_companion_sequential(a1, a2, c1, c2, values)
        assert _relative(oracle, forward[::-1]) > 1e-3, (
            "reverse must use the reversed DRIVE, not a flipped output")


def test_zero_prehistory_and_no_shifted_input_wraparound():
    """An impulse at token `start` may not move any earlier state, in
    either direction. The shifted input must not wrap around the ends."""
    length, start = 400, 137
    key = jax.random.PRNGKey(3)
    a1, a2 = _coefficients(key)
    c1, c2 = _projections(jax.random.split(key)[0])
    values = np.zeros((length, FEATURES), dtype=np.float32).at[start].set(1.0)
    for scan in (scan_companion_sequential, scan_companion, FR.scan_factored):
        states = scan(a1, a2, c1, c2, values)
        assert bool(np.all(states[:start] == 0)), scan.__name__
        assert float(np.max(np.abs(states[start]))) > 0.0
        backward = scan(a1, a2, c1, c2, values, reverse=True)
        assert bool(np.all(backward[start + 1:] == 0)), scan.__name__


# ------------------------------------------------------- repeated roots --
def test_repeated_roots_match_the_oracle_and_the_analytic_response():
    """a2 = -a1^2/4 is a double root. The cascade is the Jordan
    realization; a partial-fraction route would divide by (r1 - r2)."""
    value = np.array([0.3, -0.7, 0.95, 0.5], dtype=np.complex64)
    a1, a2 = 2.0 * value, -(2.0 * value) ** 2 / 4.0
    first, second = FR.companion_roots(a1, a2)
    assert _relative(first, second) < 1e-4
    length = 64
    # REGRESSION: this was np.eye(4, 1), which is [[1],[0],[0],[0]] -- only
    # mode 0 was driven, modes 1..3 were identically zero, and the analytic
    # comparison was against an all-zero oracle.
    c1 = np.ones((4, 1), dtype=np.complex64)
    c2 = np.zeros((4, 1), dtype=np.complex64)
    values = np.zeros((length, 1), dtype=np.float32).at[0].set(1.0)
    oracle = scan_companion_sequential(a1, a2, c1, c2, values)
    assert _relative(oracle, FR.scan_factored(a1, a2, c1, c2,
                                              values)) < FLOAT32_TOLERANCE
    index = onp.arange(length)
    for mode in range(4):
        analytic = (index + 1) * onp.asarray(value)[mode] ** index
        got = onp.asarray(oracle[:, mode])
        assert onp.max(onp.abs(analytic - got)) < 1e-3 * max(
            1.0, float(onp.max(onp.abs(analytic))))


def test_near_repeated_roots_do_not_degrade():
    for epsilon in (1e-3, 1e-5, 1e-7, 0.0):
        a1 = np.full((4,), 1.2, dtype=np.complex64)
        a2 = -a1 * a1 / 4.0 + np.asarray(epsilon, dtype=np.complex64)
        key = jax.random.PRNGKey(19)
        c1, c2 = _projections(key, modes=4)
        values = _inputs(jax.random.split(key)[1], 512)
        oracle = scan_companion_sequential(a1, a2, c1, c2, values)
        error = _relative(oracle, FR.scan_factored(a1, a2, c1, c2, values))
        assert error < FLOAT32_TOLERANCE, (epsilon, error)


# ------------------------------------------------------------ gradients --
def test_gradients_with_respect_to_the_coefficients_and_the_drive():
    key = jax.random.PRNGKey(23)
    a1, a2 = _coefficients(key)
    c1, c2 = _projections(jax.random.split(key)[0])
    values = _inputs(jax.random.split(key)[1], 500)

    def loss(scan):
        def inner(a1, a2, c1, c2, values):
            return np.sum(np.abs(scan(a1, a2, c1, c2, values)) ** 2)
        return inner

    grad_oracle = jax.grad(loss(scan_companion_sequential),
                           argnums=(0, 1, 2, 3, 4))(a1, a2, c1, c2, values)
    for index, value in enumerate(grad_oracle):
        assert bool(np.all(np.isfinite(np.abs(value)))), (
            "the ORACLE's gradient is not finite, so this test would be "
            "vacuous", index)
    for scan in (scan_companion, FR.scan_factored):
        grads = jax.grad(loss(scan), argnums=(0, 1, 2, 3, 4))(
            a1, a2, c1, c2, values)
        for index, (want, got) in enumerate(zip(grad_oracle, grads)):
            assert bool(np.all(np.isfinite(np.abs(got)))), (scan, index)
            assert _relative(want, got) < 1e-2, (scan.__name__, index,
                                                 _relative(want, got))


def test_gradients_through_the_original_response_mass_gamma_parameters():
    """The learned parameterization is unchanged, so the gradient must flow
    through T, rho and gamma to the same values as through the oracle."""
    key = jax.random.PRNGKey(29)
    modes = MODES
    lambda_bar = (0.9 * jax.random.uniform(key, (modes,), minval=0.5,
                                           maxval=0.99)
                  * np.exp(1j * jax.random.uniform(jax.random.split(key)[0],
                                                   (modes,), minval=-1.0,
                                                   maxval=1.0))
                  ).astype(np.complex64)
    b_bar = _projections(jax.random.split(key)[1])[0]
    values = _inputs(jax.random.split(key, 3)[2], 400)
    raw = (np.full((modes,), math.log(math.expm1(0.05)), dtype=np.float32),
           np.full((modes,), math.log(0.5 / 0.5), dtype=np.float32),
           np.full((modes,), math.log(math.expm1(1.0)), dtype=np.float32))

    def loss(scan):
        def inner(t_raw, rho_raw, gamma_raw):
            T, mass, gamma, _ = response_mass_gamma(t_raw, rho_raw, gamma_raw)
            a1, a2, c1, c2 = generalized_coefficients(lambda_bar, b_bar, T,
                                                      mass, gamma)
            return np.sum(np.abs(scan(a1, a2, c1, c2, values)) ** 2)
        return inner

    want = jax.grad(loss(scan_companion_sequential),
                    argnums=(0, 1, 2))(*raw)
    for scan in (scan_companion, FR.scan_factored):
        got = jax.grad(loss(scan), argnums=(0, 1, 2))(*raw)
        for index, (reference, candidate) in enumerate(zip(want, got)):
            assert bool(np.all(np.isfinite(candidate))), (scan, index)
            assert _relative(reference, candidate) < 1e-2, (
                scan.__name__, index, _relative(reference, candidate))


# --------------------------------------------------------- float32 path --
def test_production_length_float32_values_and_gradients_are_finite():
    key = jax.random.PRNGKey(31)
    a1, a2 = _coefficients(key, modes=64)
    c1, c2 = _projections(jax.random.split(key)[0], modes=64, features=8)
    values = _inputs(jax.random.split(key)[1], 16000, features=8)
    oracle = scan_companion_sequential(a1, a2, c1, c2, values)
    assert bool(np.all(np.isfinite(np.abs(oracle)))), (
        "the ORACLE is not finite at production length, so this test would "
        "be vacuous")
    for scan in (scan_companion, FR.scan_factored):
        states = scan(a1, a2, c1, c2, values)
        assert states.dtype == np.complex64, (scan.__name__, states.dtype)
        assert bool(np.all(np.isfinite(np.abs(states)))), scan.__name__
        assert _relative(oracle, states) < FLOAT32_TOLERANCE, scan.__name__
        grad = jax.grad(lambda v: np.sum(np.abs(scan(a1, a2, c1, c2, v)) ** 2)
                        )(values)
        assert bool(np.all(np.isfinite(grad))), scan.__name__


def test_the_registry_refuses_an_unknown_implementation():
    assert FR.DEFAULT_IMPLEMENTATION == "sequential"
    assert FR.scan_for("sequential") is scan_companion_sequential
    try:
        FR.scan_for("parallel-ish")
    except ValueError as error:
        assert "unknown scan implementation" in str(error)
    else:
        raise AssertionError("an unknown implementation was accepted")
