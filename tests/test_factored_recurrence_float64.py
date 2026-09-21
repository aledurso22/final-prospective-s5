"""The ALGEBRAIC correctness gate, in double precision.

RUN IT IN ITS OWN PROCESS WITH x64 ON:

    JAX_ENABLE_X64=1 python -m pytest tests/test_factored_recurrence_float64.py -q

Why it is separate. The first certification run claimed a float64 column
and reported numbers identical to the float32 one, to every digit, while
JAX printed

    Explicitly requested dtype complex128 ... will be truncated to complex64

x64 was never enabled, so `astype(complex128)` silently did nothing and the
float64 gate measured float32. A gate that cannot detect its own absence is
worse than no gate, so this file REFUSES TO RUN unless x64 is really on.

What it decides. The three routes evaluate the same recurrence, so in
double precision they must agree to round-off. If they do, the ~3e-4 seen
in float32 at length 16,000 is accumulation and nothing else. If they do
not, there is an algebraic discrepancy and no float32 tolerance may be
used to cover it.
"""

import math

import jax
import jax.numpy as np

from s5 import factored_recurrence as FR
from s5.discrete_recurrence import scan_companion, scan_companion_sequential

TOLERANCE = 1e-10
LENGTHS = (5, 257, 4000, 16000)


def _require_x64():
    assert jax.config.read("jax_enable_x64"), (
        "x64 is OFF. This file is the float64 gate and cannot run without "
        "it. Re-run as: JAX_ENABLE_X64=1 python -m pytest "
        "tests/test_factored_recurrence_float64.py -q")


def _stable(key, modes=16):
    keys = jax.random.split(key, 2)
    magnitude = jax.random.uniform(keys[0], (2, modes), minval=0.05,
                                   maxval=0.9999, dtype=np.float64)
    angle = jax.random.uniform(keys[1], (2, modes), minval=-math.pi,
                               maxval=math.pi, dtype=np.float64)
    roots = (magnitude * np.exp(1j * angle)).astype(np.complex128)
    return FR.roots_to_coefficients(roots[0], roots[1])


def _projections(key, modes=16, features=4):
    keys = jax.random.split(key, 2)
    return tuple(
        (jax.random.normal(keys[i], (modes, features), dtype=np.float64)
         + 1j * jax.random.normal(keys[1 - i], (modes, features),
                                  dtype=np.float64)).astype(np.complex128)
        * 0.3 for i in (0, 1))


def _relative(reference, candidate):
    scale = float(np.max(np.abs(reference))) or 1.0
    return float(np.max(np.abs(reference - candidate))) / scale


def test_x64_is_actually_enabled():
    """The check the certification silently failed."""
    _require_x64()
    assert np.zeros(1, dtype=np.complex128).dtype == np.complex128


def test_both_routes_are_algebraically_exact_in_double_precision():
    _require_x64()
    worst = {}
    for length in LENGTHS:
        key = jax.random.PRNGKey(length)
        a1, a2 = _stable(key)
        c1, c2 = _projections(jax.random.split(key)[0])
        values = jax.random.normal(jax.random.split(key)[1], (length, 4),
                                   dtype=np.float64)
        oracle = scan_companion_sequential(a1, a2, c1, c2, values)
        assert oracle.dtype == np.complex128, oracle.dtype
        for name, scan in (("companion", scan_companion),
                           ("factored", FR.scan_factored)):
            error = _relative(oracle, scan(a1, a2, c1, c2, values))
            worst[f"{name}@{length}"] = error
            assert error < TOLERANCE, (name, length, error)
    print("worst float64 relative error:", max(worst.values()), worst)


def test_the_reverse_direction_is_exact_too():
    _require_x64()
    key = jax.random.PRNGKey(7)
    a1, a2 = _stable(key)
    c1, c2 = _projections(jax.random.split(key)[0])
    values = jax.random.normal(jax.random.split(key)[1], (16000, 4),
                               dtype=np.float64)
    oracle = scan_companion_sequential(a1, a2, c1, c2, values, reverse=True)
    for scan in (scan_companion, FR.scan_factored):
        assert _relative(oracle, scan(a1, a2, c1, c2, values,
                                      reverse=True)) < TOLERANCE


def test_repeated_and_near_repeated_roots_are_exact():
    """Where a partial-fraction route would divide by (r1 - r2)."""
    _require_x64()
    key = jax.random.PRNGKey(11)
    c1, c2 = _projections(key, modes=4)
    values = jax.random.normal(jax.random.split(key)[1], (2000, 4),
                               dtype=np.float64)
    for epsilon in (0.0, 1e-14, 1e-10, 1e-6):
        a1 = np.full((4,), 1.2, dtype=np.complex128)
        a2 = -a1 * a1 / 4.0 + np.asarray(epsilon, dtype=np.complex128)
        oracle = scan_companion_sequential(a1, a2, c1, c2, values)
        error = _relative(oracle, FR.scan_factored(a1, a2, c1, c2, values))
        assert error < TOLERANCE, (epsilon, error)


def test_gradients_agree_in_double_precision():
    _require_x64()
    key = jax.random.PRNGKey(13)
    a1, a2 = _stable(key)
    c1, c2 = _projections(jax.random.split(key)[0])
    values = jax.random.normal(jax.random.split(key)[1], (2000, 4),
                               dtype=np.float64)

    def loss(scan):
        return lambda a1, a2, c1, c2: np.sum(
            np.abs(scan(a1, a2, c1, c2, values)) ** 2)

    want = jax.grad(loss(scan_companion_sequential),
                    argnums=(0, 1, 2, 3))(a1, a2, c1, c2)
    for scan in (scan_companion, FR.scan_factored):
        got = jax.grad(loss(scan), argnums=(0, 1, 2, 3))(a1, a2, c1, c2)
        for index, (reference, candidate) in enumerate(zip(want, got)):
            assert _relative(reference, candidate) < 1e-8, (
                scan.__name__, index, _relative(reference, candidate))
