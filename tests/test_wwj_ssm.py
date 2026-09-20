"""The JAX implementation of the PRINCIPAL WWJ architecture.

Needs JAX, so it runs on the cluster. The mathematics it implements is
proved exactly, without JAX, in `test_wwj_operator_algebra.py`; what is
checked here is the implementation, its gradients, its orientation and its
behaviour at production shapes and precisions.

    JAX_ENABLE_X64=1 $PY -m pytest tests/test_wwj_ssm.py
"""

import os
import subprocess

import jax
import jax.numpy as np

from s5 import wwj_operator as OP
from s5 import wwj_ssm as ARMS
from tests import wwj_operator_reference as ORACLE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Native S5 must be byte-identical to this commit
NATIVE_BASE = "ef004cda025b4e098041cfe5970c217fa0015bba"
X64 = jax.config.read("jax_enable_x64")
TOL64, TOL32 = 1e-11, 2e-5


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


def _close(left, right, tolerance):
    scale = float(np.maximum(np.max(np.abs(right)), 1.0))
    return float(np.max(np.abs(left - right))) <= tolerance * scale


def _km(tau=0.05, eps=0.25):
    real = np.float64 if X64 else np.float32
    return OP.k_and_m(np.asarray(tau, dtype=real), np.asarray(eps, dtype=real))


# ------------------------------------------------- the operator itself -----
def test_the_three_tap_matches_the_sequential_oracle():
    lambda_bar, b_bar = _modes(1)
    k, m = _km()
    tolerance = TOL64 if X64 else TOL32
    for length in (1, 2, 3, 5, 7, 16, 17, 100, 257, 1024, 1500):
        inputs = _inputs(length, length)
        fast = OP.wwj_states(lambda_bar, b_bar, inputs, k, m)
        slow = ORACLE.wwj_sequential(lambda_bar, b_bar, inputs, k, m)
        assert fast.shape == (length, lambda_bar.shape[0])
        assert _close(fast, slow, tolerance), length


def test_the_optimized_path_equals_the_augmented_realization():
    """Native parallel scan plus three-tap IS the augmented recurrent state
    space, not an approximation of it."""
    tolerance = TOL64 if X64 else TOL32
    for complex_modes in (True, False):
        lambda_bar, b_bar = _modes(2, complex_modes=complex_modes)
        k, m = _km()
        for length in (1, 2, 3, 9, 64, 333):
            inputs = _inputs(length + 1, length)
            optimized = OP.wwj_states(lambda_bar, b_bar, inputs, k, m)
            augmented = ORACLE.augmented_sequential(lambda_bar, b_bar, inputs,
                                                    k, m)
            assert _close(optimized, augmented, tolerance), length


def test_the_native_scan_is_the_repository_scan():
    """The states come from `s5/ssm.py`'s own binary_operator, so Native
    memory is reused rather than reimplemented."""
    import inspect

    source = inspect.getsource(OP.native_states)
    assert "binary_operator" in source and "associative_scan" in source
    lambda_bar, b_bar = _modes(3)
    inputs = _inputs(4, 128)
    assert _close(OP.native_states(lambda_bar, b_bar, inputs),
                  ORACLE.native_states_sequential(lambda_bar, b_bar, inputs),
                  TOL64 if X64 else TOL32)


def test_the_two_stage_factorization_equals_the_three_tap_in_jax():
    real = np.float64 if X64 else np.float32
    lambda_bar, b_bar = _modes(5)
    inputs = _inputs(6, 200)
    states = OP.native_states(lambda_bar, b_bar, inputs)
    for eps in (0.0, 0.0625, 0.1875, 0.25):
        tau = np.asarray(0.3, dtype=real)
        k, m = OP.k_and_m(tau, np.asarray(eps, dtype=real))
        plus, minus = OP.passive_factors(tau, np.asarray(eps, dtype=real))
        assert float(minus) >= 0.0                       # passive: both >= 0
        assert _close(OP.two_stage(states, plus, minus),
                      OP.three_tap(states, k, m), TOL64 if X64 else TOL32), eps


def test_tau_zero_recovers_native_s5_exactly_in_jax():
    real = np.float64 if X64 else np.float32
    lambda_bar, b_bar = _modes(7)
    inputs = _inputs(8, 64)
    native = OP.native_states(lambda_bar, b_bar, inputs)
    for eps in (0.0, 0.0625, 0.25):
        k, m = OP.k_and_m(np.asarray(0.0, dtype=real),
                          np.asarray(eps, dtype=real))
        assert bool(np.all(OP.three_tap(native, k, m) == native)), eps


def test_zero_prehistory_and_no_wraparound_at_both_boundaries():
    lambda_bar, b_bar = _modes(9)
    k, m = _km()
    length, H = 24, b_bar.shape[1]
    for impulse in (0, 1, 2, 11, 23):
        inputs = np.zeros((length, H)).at[impulse].set(1.0)
        states = OP.wwj_states(lambda_bar, b_bar, inputs, k, m)
        assert bool(np.all(states[:impulse] == 0)), impulse
        assert bool(np.any(states[impulse] != 0)), impulse
    # an impulse on the last token never reaches the head
    last = np.zeros((length, H)).at[length - 1].set(1.0)
    states = OP.wwj_states(lambda_bar, b_bar, last, k, m)
    assert bool(np.all(states[:length - 1] == 0))
    assert bool(np.any(states[length - 1] != 0))


def test_the_reverse_direction_is_a_flipped_causal_operator():
    lambda_bar, b_bar = _modes(10)
    k, m = _km()
    inputs = _inputs(11, 48)
    reverse = OP.wwj_states(lambda_bar, b_bar, inputs, k, m, reverse=True)
    oracle = ORACLE.wwj_sequential(lambda_bar, b_bar, inputs, k, m,
                                   reverse=True)
    assert _close(reverse, oracle, TOL64 if X64 else TOL32)
    forward = OP.wwj_states(lambda_bar, b_bar, inputs, k, m)
    assert not _close(reverse, forward, 1e-3)
    # the reverse branch's last token has no history in its own direction
    states = OP.native_states(lambda_bar, b_bar, inputs, reverse=True)
    assert _close(reverse[-1], (1.0 + k + m) * states[-1],
                  TOL64 if X64 else TOL32)


# ------------------------------------------------------------ gradients ----
def _loss(lambda_bar, b_bar, tau, eps, inputs, implementation):
    k, m = OP.k_and_m(tau, eps)
    return np.sum(np.abs(
        implementation(lambda_bar, b_bar, inputs, k, m)) ** 2).real


def test_gradients_match_the_oracle_not_only_outputs():
    lambda_bar, b_bar = _modes(12, P=4, H=2)
    inputs = _inputs(13, 37, H=2)
    real = np.float64 if X64 else np.float32
    tau, eps = np.asarray(0.05, dtype=real), np.asarray(0.25, dtype=real)
    tolerance = 1e-8 if X64 else 1e-3
    for argnums in (0, 1, 2, 3):
        fast = jax.grad(_loss, argnums=argnums)(
            lambda_bar, b_bar, tau, eps, inputs, OP.wwj_states)
        slow = jax.grad(_loss, argnums=argnums)(
            lambda_bar, b_bar, tau, eps, inputs, ORACLE.wwj_sequential)
        assert _close(fast, slow, tolerance), argnums


def test_critical_and_passive_parameters_receive_finite_gradients():
    lambda_bar, b_bar = _modes(14, P=4, H=2)
    inputs = _inputs(15, 65, H=2)
    real = np.float64 if X64 else np.float32

    def loss(tau_raw, eps_raw):
        tau = ARMS.tau_from_raw(tau_raw)
        eps = ARMS.eps_passive_from_raw(eps_raw)
        k, m = OP.k_and_m(tau, eps)
        return np.sum(np.abs(
            OP.wwj_states(lambda_bar, b_bar, inputs, k, m)) ** 2).real

    value, grads = jax.value_and_grad(loss, argnums=(0, 1))(
        np.asarray(-2.0, dtype=real), np.asarray(-1.0, dtype=real))
    assert bool(np.isfinite(value))
    assert all(bool(np.isfinite(g)) for g in grads)
    # the prospective gradients must not vanish at the initialization
    assert all(abs(float(g)) > 1e-12 for g in grads), grads


def test_production_length_float32_forward_backward_and_update_are_finite():
    """The failure that killed the mixed-stencil path, asked of this one at
    the production sequence length and width, in float32."""
    lambda_bar, b_bar = _modes(16, P=64, H=96, dtype=np.complex64)
    inputs = jax.random.normal(jax.random.PRNGKey(17),
                               (16000, 96)).astype(np.float32)
    k, m = OP.k_and_m(np.asarray(0.05, dtype=np.float32),
                      np.asarray(0.25, dtype=np.float32))

    def loss(b):
        return np.sum(np.abs(
            OP.wwj_states(lambda_bar, b, inputs, k, m)) ** 2).real

    value, grad = jax.value_and_grad(loss)(b_bar)
    assert bool(np.isfinite(value)), value
    assert bool(np.all(np.isfinite(grad)))
    states = OP.wwj_states(lambda_bar, b_bar, inputs, k, m)
    assert bool(np.all(np.isfinite(states)))
    # and the magnitudes stay ordinary: no 1e81
    assert float(np.max(np.abs(states))) < 1e6, float(np.max(np.abs(states)))


def test_float32_stays_within_tolerance_of_the_float64_reference():
    if not X64:
        return
    lambda_bar, b_bar = _modes(18, dtype=np.complex128)
    inputs = _inputs(19, 2048)
    k, m = _km()
    exact = OP.wwj_states(lambda_bar, b_bar, inputs, k, m)
    single = OP.wwj_states(lambda_bar.astype(np.complex64),
                           b_bar.astype(np.complex64),
                           inputs.astype(np.float32),
                           k.astype(np.float32), m.astype(np.float32))
    assert _close(single.astype(np.complex128), exact, TOL32)


def test_results_are_deterministic_for_a_fixed_seed():
    lambda_bar, b_bar = _modes(20)
    k, m = _km()
    inputs = _inputs(21, 129)
    first = OP.wwj_states(lambda_bar, b_bar, inputs, k, m)
    second = OP.wwj_states(lambda_bar, b_bar, inputs, k, m)
    assert bool(np.all(first == second))


# ---------------------------------------------------------- diagnostics ----
def test_the_fir_gain_and_native_radii_are_the_reported_diagnostics():
    real = np.float64 if X64 else np.float32
    lambda_bar, _ = _modes(22)
    for tau, eps in ((0.0, 0.25), (0.05, 0.25), (1.0, 0.0625)):
        k, m = OP.k_and_m(np.asarray(tau, dtype=real),
                          np.asarray(eps, dtype=real))
        gain = float(OP.max_fir_gain(k, m))
        assert gain >= 1.0 - 1e-6                 # |P_h(1)| = 1 at omega = 0
        assert np.isfinite(gain)
        if tau == 0.0:
            assert abs(gain - 1.0) < 1e-6         # identity operator
    # the operator has no poles: the layer's poles are Native plus two zeros
    poles = OP.recurrent_poles(lambda_bar)
    assert poles.shape[0] == lambda_bar.shape[0] + 2
    assert float(np.max(np.abs(poles[-2:]))) == 0.0
    assert bool(np.all(np.abs(poles[:-2]) == np.abs(lambda_bar)))
    assert bool(np.all(OP.native_radii(lambda_bar) == np.abs(lambda_bar)))


# ------------------------------------------------------------- the arms ----
def _arm_kwargs(P=8, H=4):
    import numpy

    return dict(Lambda_re_init=-0.5 * numpy.ones(P),
                Lambda_im_init=numpy.linspace(0.1, 1.0, P),
                V=numpy.eye(P, dtype=numpy.complex64),
                Vinv=numpy.eye(P, dtype=numpy.complex64),
                H=H, P=P, C_init="lecun_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=False, clip_eigs=True)


def _apply_arm(constructor, length=32, **kwargs):
    model = constructor(**_arm_kwargs(), **kwargs)()
    inputs = _inputs(23, length, H=4)
    variables = model.init(jax.random.PRNGKey(0), inputs)
    return model.apply(variables, inputs), variables, model


def test_both_principal_arms_run_and_carry_their_own_parameters():
    for constructor, expected in (
            (ARMS.init_wwj_critical_S5SSM, {"wwj_tau_raw"}),
            (ARMS.init_wwj_passive_S5SSM, {"wwj_tau_raw", "wwj_eps_raw"})):
        out, variables, _ = _apply_arm(constructor)
        assert bool(np.all(np.isfinite(out)))
        names = set(variables["params"])
        assert expected <= names
        for name in expected:
            assert variables["params"][name].shape == ()   # per layer
    # the critical arm does not learn eps
    _, variables, _ = _apply_arm(ARMS.init_wwj_critical_S5SSM)
    assert "wwj_eps_raw" not in variables["params"]


def test_a_bidirectional_arm_runs_both_orientations():
    forward, _, _ = _apply_arm(ARMS.init_wwj_critical_S5SSM)
    both, _, _ = _apply_arm(ARMS.init_wwj_critical_S5SSM, bidirectional=True)
    assert both.shape == forward.shape
    assert bool(np.all(np.isfinite(both)))
    assert float(np.max(np.abs(both - forward))) > 0.0


def test_the_gated_arm_is_named_and_behaves_as_a_separate_intervention():
    out, variables, _ = _apply_arm(ARMS.init_wwj_gated_S5SSM)
    assert bool(np.all(np.isfinite(out)))
    assert "wwj_gate_raw" in variables["params"]
    assert "wwj_gated_recoverable_s5_diagnostic" in ARMS.WWJ_DIAGNOSTIC_ARMS
    assert "wwj_gated_recoverable_s5_diagnostic" not in ARMS.WWJ_PRINCIPAL_ARMS
    assert "DIAGNOSTIC" in ARMS.WWJ_SCIENTIFIC_NAMES[
        "wwj_gated_recoverable_s5_diagnostic"]


def test_the_arm_diagnostics_report_fir_gain_not_companion_radii():
    model = ARMS.init_wwj_critical_S5SSM(**_arm_kwargs())()
    inputs = _inputs(24, 8, H=4)
    variables = model.init(jax.random.PRNGKey(0), inputs)
    diagnostics = model.apply(variables, method=lambda m: m.diagnostics())
    for key in ("tau", "eps", "mass", "k", "m", "max_fir_gain",
                "native_radius_max", "native_radius_min"):
        assert key in diagnostics, key
    assert "companion_spectral_radius" not in diagnostics
    assert float(diagnostics["eps"]) == 0.25
    assert bool(np.isfinite(diagnostics["max_fir_gain"]))


# ------------------------------------------------- what must not change ----
def _blob(relative):
    return subprocess.run(["git", "-C", REPO, "rev-parse",
                           f"{NATIVE_BASE}:{relative}"], capture_output=True,
                          text=True).stdout.strip()


def _worktree(relative):
    return subprocess.run(["git", "-C", REPO, "hash-object",
                           os.path.join(REPO, relative)],
                          capture_output=True, text=True).stdout.strip()


def test_native_s5_and_the_frozen_arms_are_byte_identical():
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py",
                     "s5/generalized_prospective_ssm.py",
                     "s5/prospective_ssm.py",
                     "experiments/s5_three_arm_full/data.py",
                     "experiments/s5_three_arm_full/runner.py"):
        assert _blob(relative) == _worktree(relative) != "", relative


def test_the_rejected_realization_is_not_reachable_from_the_principal_arms():
    for module in (OP, ARMS):
        source = open(module.__file__.replace(".pyc", ".py")).read()
        assert "wwj_mixed_stencil" not in source.replace(
            "s5/wwj_mixed_stencil.py", "").replace(
            "s5/wwj_mixed_stencil_ssm.py", "")
