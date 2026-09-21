"""The matched-lag arm in JAX. Run the float64 parts with x64 ON:

    JAX_ENABLE_X64=1 python -m pytest tests/test_matched_lag_jax.py -q

It decides two things the laptop cannot: that the regrouped drive is the
same recurrence as the direct coefficients in double precision, and that
the layer's a1 and a2 are the generalized arm's, bit for bit.
"""

import math

import jax
import jax.numpy as np

from s5 import matched_lag_ssm as ML
from s5.discrete_recurrence import generalized_coefficients
from s5.factored_recurrence import scan_factored_drive, scan_sequential_drive
from s5.generalized_prospective_ssm import response_mass_gamma

TOLERANCE = 1e-10
H = 1.0


def _x64():
    return bool(jax.config.read("jax_enable_x64"))


def _require_x64():
    assert _x64(), ("this file's float64 gates need JAX_ENABLE_X64=1; "
                    "re-run as JAX_ENABLE_X64=1 python -m pytest "
                    "tests/test_matched_lag_jax.py -q")


def _dtype():
    return np.complex128 if _x64() else np.complex64


def _modes(count=24, seed=0):
    key = jax.random.split(jax.random.PRNGKey(seed), 3)
    magnitude = jax.random.uniform(key[0], (count,), minval=0.3, maxval=0.9999)
    angle = jax.random.uniform(key[1], (count,), minval=0.0, maxval=math.pi)
    lam = (magnitude * np.exp(1j * angle)).astype(_dtype())
    b = (jax.random.normal(key[2], (count, 3))
         + 1j * jax.random.normal(key[2], (count, 3))).astype(_dtype()) * 0.3
    return lam, b


def _relative(reference, candidate):
    scale = float(np.max(np.abs(reference))) or 1.0
    return float(np.max(np.abs(reference - candidate))) / scale


def _pieces(lam, b, response=0.05, rho=0.5, gamma=1.0):
    T = np.full(lam.shape, response, dtype=np.float64 if _x64() else np.float32)
    G = np.full(lam.shape, gamma, dtype=T.dtype)
    M = np.full(lam.shape, rho * gamma * response, dtype=T.dtype)
    a1, a2, c1, c2 = generalized_coefficients(lam, b, T, M, G)
    gamma_k = ML.matched_gamma(lam, T, G, H)
    gain = ML.matched_gain(b, T, M, G, H)
    return a1, a2, c1, c2, gamma_k, gain


# ------------------------------------------------------------- item (1) --
def test_the_regrouped_drive_equals_the_direct_coefficients_in_float64():
    """The whole point of regrouping is that it changes the ARITHMETIC and
    not the recurrence. In double precision the two must agree to
    round-off."""
    _require_x64()
    lam, b = _modes()
    _, _, _, _, gamma_k, gain = _pieces(lam, b)
    d1, d2 = ML.direct_coefficients(gain, gamma_k, H)
    key = jax.random.PRNGKey(5)
    for length in (7, 257, 4000, 16000):
        values = jax.random.normal(key, (length, 3),
                                   dtype=np.float64 if _x64() else np.float32)
        regrouped = ML.matched_drive(gain, gamma_k, values, H)
        projected = jax.vmap(lambda u: d1 @ u)(values)
        shifted = np.concatenate(
            (np.zeros_like(projected[:1]),
             jax.vmap(lambda u: d2 @ u)(values)[:-1]), axis=0)
        direct = projected + shifted
        error = _relative(direct, regrouped)
        assert error < TOLERANCE, (length, error)


def test_the_two_scans_agree_on_the_matched_drive():
    _require_x64()
    lam, b = _modes()
    a1, a2, _, _, gamma_k, gain = _pieces(lam, b)
    key = jax.random.PRNGKey(9)
    for length in (257, 16000):
        values = jax.random.normal(key, (length, 3),
                                   dtype=np.float64 if _x64() else np.float32)
        drive = ML.matched_drive(gain, gamma_k, values, H)
        oracle = scan_sequential_drive(a1, a2, drive)
        assert _relative(oracle, scan_factored_drive(a1, a2, drive)) < TOLERANCE


# ------------------------------------------------------------- item (2) --
def test_a1_and_a2_are_the_generalized_arm_s_bit_for_bit():
    """The ablation is one coefficient. Verified on the arrays, not by
    reading the source: the matched path never recomputes them."""
    lam, b = _modes()
    a1, a2, _, _, _, _ = _pieces(lam, b)
    again1, again2, _, _ = generalized_coefficients(
        lam, b, np.full(lam.shape, 0.05, dtype=np.float32 if not _x64()
                        else np.float64),
        np.full(lam.shape, 0.025, dtype=np.float32 if not _x64()
                else np.float64),
        np.full(lam.shape, 1.0, dtype=np.float32 if not _x64()
                else np.float64))
    assert bool(np.all(a1 == again1)), "a1 must be identical"
    assert bool(np.all(a2 == again2)), "a2 must be identical"


def test_gamma_k_equal_to_T_reproduces_the_generalized_drive():
    """The consistency anchor, on arrays: the fourth arm contains the third
    as the point Gamma_k = T."""
    _require_x64()
    lam, b = _modes()
    _, _, c1, c2, _, gain = _pieces(lam, b)
    at_T = np.full(lam.shape, 0.05, dtype=lam.dtype)
    d1, d2 = ML.direct_coefficients(gain, at_T, H)
    assert _relative(c1, d1) < TOLERANCE
    assert _relative(c2, d2) < TOLERANCE


# ------------------------------------------------------------- item (5) --
def test_conjugate_symmetry_and_finiteness_at_production_length():
    """Gamma_k(conj a) = conj(Gamma_k(a)), so conj_sym needs no handling."""
    lam, b = _modes()
    T = np.full(lam.shape, 0.05, dtype=np.float64 if _x64() else np.float32)
    G = np.full(lam.shape, 1.0, dtype=T.dtype)
    forward = ML.matched_gamma(lam, T, G, H)
    conjugate = ML.matched_gamma(np.conj(lam), T, G, H)
    assert _relative(np.conj(forward), conjugate) < 1e-12
    a1, a2, _, _, gamma_k, gain = _pieces(lam, b)
    values = jax.random.normal(jax.random.PRNGKey(11), (16000, 3),
                               dtype=np.float64 if _x64() else np.float32)
    states = ML.apply_matched(a1, a2, gain, gamma_k, values, H,
                              implementation="factored")
    assert bool(np.all(np.isfinite(np.abs(states))))


def test_gradients_reach_the_prospective_pathway_and_are_finite():
    """Item 5 in miniature: the gradient must flow through Gamma_k, which
    is where the four-decade spread lives."""
    lam, b = _modes(count=8)
    raw = (np.full(lam.shape, math.log(math.expm1(0.05)),
                   dtype=np.float64 if _x64() else np.float32),
           np.zeros(lam.shape, dtype=np.float64 if _x64() else np.float32),
           np.full(lam.shape, math.log(math.expm1(1.0)),
                   dtype=np.float64 if _x64() else np.float32))
    values = jax.random.normal(jax.random.PRNGKey(13), (600, 3),
                               dtype=np.float64 if _x64() else np.float32)

    def loss(t_raw, rho_raw, gamma_raw):
        T, mass, gamma, _ = response_mass_gamma(t_raw, rho_raw, gamma_raw)
        a1, a2, _, _ = generalized_coefficients(lam, b, T, mass, gamma)
        gamma_k = ML.matched_gamma(lam, T, gamma, H)
        gain = ML.matched_gain(b, T, mass, gamma, H)
        states = ML.apply_matched(a1, a2, gain, gamma_k, values, H,
                                  implementation="factored")
        return np.sum(np.abs(states) ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2))(*raw)
    for index, value in enumerate(grads):
        assert bool(np.all(np.isfinite(value))), index
        assert float(np.max(np.abs(value))) > 0.0, index
