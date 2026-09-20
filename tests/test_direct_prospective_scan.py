"""The JAX implementation of the direct prospective recurrences (cluster).

The algebra is proved without JAX in `test_direct_prospective_algebra.py`;
this file checks the implementation, its gradients, its orientation, its
alignment and its behaviour at production shape and precision.

    JAX_ENABLE_X64=1 $PY -m pytest tests/test_direct_prospective_scan.py
"""

import os
import subprocess

import jax
import jax.numpy as np

from s5 import direct_prospective as DP
from tests import direct_prospective_reference as ORACLE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE_BASE = "6d55765a36f5985f3a062a8f3b6854316b26b1bd"
X64 = jax.config.read("jax_enable_x64")
TOL64, TOL32 = 1e-10, 2e-4


def _modes(seed, P=6, H=3, complex_modes=True, dtype=None):
    dtype = dtype or (np.complex128 if X64 else np.complex64)
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    radius = 0.2 + 0.7 * jax.random.uniform(keys[0], (P,))
    if complex_modes:
        angle = jax.random.uniform(keys[1], (P,), minval=-2.0, maxval=2.0)
        lambda_bar = (radius * np.exp(1j * angle)).astype(dtype)
    else:
        lambda_bar = radius.astype(dtype)
    b_bar = (jax.random.normal(keys[2], (P, H))
             + 1j * jax.random.normal(keys[3], (P, H))).astype(dtype)
    return lambda_bar, b_bar


def _inputs(seed, length, H=3):
    real = np.float64 if X64 else np.float32
    return jax.random.normal(jax.random.PRNGKey(seed), (length, H)).astype(real)


def _params(P, tau=0.5, eps=0.25, gamma=0.0):
    real = np.float64 if X64 else np.float32
    tau_vector = np.full((P,), tau, dtype=real)
    gamma_vector = np.full((P,), gamma, dtype=real)
    return tau_vector, gamma_vector


def _close(left, right, tolerance):
    scale = float(np.maximum(np.max(np.abs(right)), 1.0))
    return float(np.max(np.abs(left - right))) <= tolerance * scale


# ------------------------------------------------ scan versus the oracle ---
def test_the_matched_scan_matches_the_oracle_at_many_lengths():
    tolerance = TOL64 if X64 else TOL32
    for construction in ("professor", "native_matched"):
        for complex_modes in (True, False):
            lambda_bar, b_bar = _modes(1, complex_modes=complex_modes)
            tau, _ = _params(lambda_bar.shape[0])
            mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=tau.dtype))
            for length in (1, 2, 3, 5, 7, 8, 9, 15, 17, 100, 257, 1000):
                inputs = _inputs(length, length)
                fast = DP.matched_states(lambda_bar, b_bar, inputs, tau, mass,
                                         construction)
                slow = ORACLE.matched_sequential(lambda_bar, b_bar, inputs,
                                                 tau, mass, construction)
                assert fast.shape == (length, lambda_bar.shape[0])
                assert _close(fast, slow, tolerance), (construction, length)


def test_the_partially_matched_scan_matches_the_oracle():
    tolerance = TOL64 if X64 else TOL32
    lambda_bar, b_bar = _modes(2)
    tau, gamma = _params(lambda_bar.shape[0], gamma=1.0)
    mass = DP.mass_from_eps(gamma + tau, np.asarray(0.25, dtype=tau.dtype))
    for length in (1, 2, 3, 6, 17, 64, 333):
        inputs = _inputs(length + 1, length)
        fast = DP.partially_matched_states(lambda_bar, b_bar, inputs, tau,
                                           mass, gamma)
        slow = ORACLE.partially_matched_sequential(lambda_bar, b_bar, inputs,
                                                   tau, mass, gamma)
        assert _close(fast, slow, tolerance), length


def test_every_rematerialization_choice_agrees():
    lambda_bar, b_bar = _modes(3)
    tau, _ = _params(lambda_bar.shape[0])
    mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=tau.dtype))
    inputs = _inputs(4, 64)
    values = {mode: DP.matched_states(lambda_bar, b_bar, inputs, tau, mass,
                                      remat=mode)
              for mode in ("level", "whole", None)}
    for mode, value in values.items():
        assert _close(value, values[None], TOL64 if X64 else TOL32), mode


def test_zero_prehistory_alignment_and_no_wraparound():
    lambda_bar, b_bar = _modes(5)
    tau, _ = _params(lambda_bar.shape[0])
    mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=tau.dtype))
    length, H = 20, b_bar.shape[1]
    for impulse in (0, 1, 2, 19):
        inputs = np.zeros((length, H)).at[impulse].set(1.0)
        states = DP.matched_states(lambda_bar, b_bar, inputs, tau, mass)
        assert bool(np.all(states[:impulse] == 0)), impulse
        assert bool(np.any(states[impulse] != 0)), impulse
    # the state at token t has seen x_t through C0, exactly like Native S5
    _, C = DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass)
    impulse_input = np.zeros((length, H)).at[0, 0].set(1.0)
    states = DP.matched_states(lambda_bar, b_bar, impulse_input, tau, mass)
    assert _close(states[0], C[0][:, 0], TOL64 if X64 else TOL32)


def test_the_reverse_branch_is_a_flipped_causal_scan():
    lambda_bar, b_bar = _modes(6)
    tau, _ = _params(lambda_bar.shape[0])
    mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=tau.dtype))
    inputs = _inputs(7, 48)
    reverse = DP.matched_states(lambda_bar, b_bar, inputs, tau, mass,
                                reverse=True)
    oracle = ORACLE.matched_sequential(lambda_bar, b_bar, inputs, tau, mass,
                                       reverse=True)
    assert _close(reverse, oracle, TOL64 if X64 else TOL32)
    assert not _close(reverse, DP.matched_states(lambda_bar, b_bar, inputs,
                                                 tau, mass), 1e-3)


# ------------------------------------------------------------ gradients ----
def test_gradients_agree_with_the_oracle():
    lambda_bar, b_bar = _modes(8, P=4, H=2)
    tau, _ = _params(4)
    mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=tau.dtype))
    inputs = _inputs(9, 33, H=2)
    tolerance = 1e-7 if X64 else 1e-2

    def loss(lam, b, tau_value, mass_value, implementation):
        return np.sum(np.abs(implementation(lam, b, inputs, tau_value,
                                            mass_value)) ** 2).real

    for argnums in (0, 1, 2, 3):
        fast = jax.grad(loss, argnums=argnums)(lambda_bar, b_bar, tau, mass,
                                               DP.matched_states)
        slow = jax.grad(loss, argnums=argnums)(
            lambda_bar, b_bar, tau, mass,
            lambda lam, b, x, t, m: ORACLE.matched_sequential(lam, b, x, t, m))
        assert _close(fast, slow, tolerance), argnums


def test_results_are_deterministic_for_a_fixed_seed():
    lambda_bar, b_bar = _modes(10)
    tau, _ = _params(lambda_bar.shape[0])
    mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=tau.dtype))
    inputs = _inputs(11, 129)
    first = DP.matched_states(lambda_bar, b_bar, inputs, tau, mass)
    second = DP.matched_states(lambda_bar, b_bar, inputs, tau, mass)
    assert bool(np.all(first == second))


# ---------------------------------------------- stability, as a finding ----
def test_the_companion_radius_estimate_agrees_with_exact_roots():
    lambda_bar, b_bar = _modes(12, P=8)
    tau, _ = _params(8, tau=0.5)
    mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=tau.dtype))
    A, _ = DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass)
    estimate = DP.companion_spectral_radius(A)
    exact = ORACLE.exact_companion_radius(A)
    for index in range(8):
        assert float(estimate[index]) >= exact[index] - 1e-6      # never under
        assert float(estimate[index]) <= exact[index] * 1.02 + 1e-6


def test_a_unit_mode_gives_an_exact_unit_root():
    """The structural marginality, in JAX: at Abar = 1 the characteristic
    polynomial vanishes at z = 1 for every parameter choice."""
    real = np.float64 if X64 else np.float32
    ones = np.ones((3,), dtype=np.complex128 if X64 else np.complex64)
    b_bar = np.ones((3, 2), dtype=ones.dtype)
    for tau_value, eps in ((0.05, 0.25), (2.0, 0.0625), (50.0, 0.0)):
        tau = np.full((3,), tau_value, dtype=real)
        mass = DP.mass_from_eps(tau, np.asarray(eps, dtype=real))
        (A0, A1, A2), _ = DP.matched_state_coefficients(ones, b_bar, tau, mass)
        characteristic = 1.0 - A0 - A1 - A2            # char(1)
        assert float(np.max(np.abs(characteristic))) < 1e-6, tau_value


def test_the_exact_matching_collapse_is_reproduced_numerically():
    """s = f with f = Abar s + Bbar x gives a memoryless map: the impulse
    response is a single nonzero token, unlike Native S5's decay."""
    lambda_bar, b_bar = _modes(13, P=4, H=2)
    inputs = np.zeros((16, 2)).at[0, 0].set(1.0)
    collapsed = DP.common_stencil_collapsed_state(lambda_bar, b_bar, inputs)
    native = ORACLE.native_sequential(lambda_bar, b_bar, inputs)
    assert float(np.max(np.abs(collapsed[1:]))) == 0.0
    assert float(np.max(np.abs(native[1:]))) > 0.0


def test_production_length_float32_finiteness_is_reported_not_assumed():
    """At production length the causal recurrence is expected to diverge on
    S5-like modes. The test records which it is, and fails only if the two
    disagree with the companion radius -- the diagnosis must be consistent.
    """
    lambda_bar, b_bar = _modes(14, P=16, H=4, dtype=np.complex64)
    tau = np.full((16,), 0.05, dtype=np.float32)
    mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=np.float32))
    inputs = jax.random.normal(jax.random.PRNGKey(15),
                               (4096, 4)).astype(np.float32)
    A, _ = DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass)
    radius = float(np.max(DP.companion_spectral_radius(A)))
    states = DP.matched_states(lambda_bar, b_bar, inputs, tau, mass)
    finite = bool(np.all(np.isfinite(states)))
    if radius > 1.001:
        assert not finite or float(np.max(np.abs(states))) > 1e6, radius
    else:
        assert finite, radius


# ------------------------------------------------- what must not change ----
def _blob(relative):
    return subprocess.run(["git", "-C", REPO, "rev-parse",
                           f"{NATIVE_BASE}:{relative}"], capture_output=True,
                          text=True).stdout.strip()


def _worktree(relative):
    return subprocess.run(["git", "-C", REPO, "hash-object",
                           os.path.join(REPO, relative)],
                          capture_output=True, text=True).stdout.strip()


def test_native_and_every_existing_arm_are_byte_identical():
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py", "s5/prospective_ssm.py",
                     "s5/generalized_prospective_ssm.py",
                     "s5/wwj_operator.py", "s5/wwj_ssm.py",
                     "experiments/s5_three_arm_full/runner.py"):
        assert _blob(relative) == _worktree(relative) != "", relative
