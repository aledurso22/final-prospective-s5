"""The JAX implementation of the WWJ recurrence: scan, gradients, arms.

These tests need JAX and therefore run on the cluster, not on a laptop
without it. The coefficient ALGEBRA and the scan ALGORITHM are proved
exactly, in rational arithmetic and without JAX, in `test_wwj_algebra.py`;
what is checked here is the implementation of that algebra in JAX.

Run with 64-bit enabled:

    JAX_ENABLE_X64=1 $PY -m pytest tests/test_wwj_recurrence.py
"""

import os
import subprocess

import jax
import jax.numpy as np

from s5 import wwj_prospective_ssm as ARMS
from s5 import wwj_recurrence as WWJ
from tests import wwj_sequential_reference as ORACLE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the commit whose Native S5 must remain byte-identical
NATIVE_BASE = "ef004cda025b4e098041cfe5970c217fa0015bba"
X64 = jax.config.read("jax_enable_x64")
#: float64 reference tolerance and float32 production tolerance
TOL64, TOL32 = 1e-11, 2e-4


def _modes(seed, P=6, H=3, complex_modes=True, dtype=None):
    """A random discretized mode set, deliberately inside the unit disc."""
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


def _params(P, tau_value=0.25, eps_value=0.125):
    real = np.float64 if X64 else np.float32
    tau = np.full((P,), tau_value, dtype=real)
    return tau, WWJ.mass_from_eps(tau, np.asarray(eps_value, dtype=real))


def _close(left, right, tolerance):
    scale = float(np.maximum(np.max(np.abs(right)), 1.0))
    return float(np.max(np.abs(left - right))) <= tolerance * scale


# ------------------------------------------------ coefficients and bounds --
def test_the_coefficients_match_the_written_derivation():
    tau, mass = _params(4, 0.4, 0.1)
    a, b, c0, c1, c2 = WWJ.scalar_coefficients(tau, mass)
    q = mass + WWJ.H_TOKEN * tau
    assert _close(a, (2 * mass + tau - 1) / q, TOL64 if X64 else TOL32)
    assert _close(b, mass / q, TOL64 if X64 else TOL32)
    assert _close(c0, (mass + tau + 1) / q, TOL64 if X64 else TOL32)
    assert _close(c1, -(2 * mass + tau) / q, TOL64 if X64 else TOL32)
    assert _close(c2, mass / q, TOL64 if X64 else TOL32)


def test_the_M_zero_boundary_in_jax():
    """M = 0  =>  (1 - h/tau, 0, 1 + h/tau, -1, 0)."""
    real = np.float64 if X64 else np.float32
    tau = np.asarray([0.05, 0.5, 2.0], dtype=real)
    zero = np.zeros_like(tau)
    a, b, c0, c1, c2 = WWJ.scalar_coefficients(tau, zero)
    tolerance = TOL64 if X64 else TOL32
    assert _close(a, 1.0 - 1.0 / tau, tolerance)
    assert _close(b, zero, tolerance)
    assert _close(c0, 1.0 + 1.0 / tau, tolerance)
    assert _close(c1, -np.ones_like(tau), tolerance)
    assert _close(c2, zero, tolerance)


def test_the_critical_and_passive_parameterizations():
    """eps = 1/4 exactly for the critical arm; eps in (0, 1/4) for the
    passive one at every raw value, including extreme ones."""
    assert float(WWJ.EPS_CRITICAL) == 0.25
    raws = np.asarray([-40.0, -5.0, 0.0, 5.0, 40.0])
    eps = ARMS.eps_passive_from_raw(raws)
    assert float(np.min(eps)) > 0.0
    assert float(np.max(eps)) <= WWJ.EPS_MAX
    assert float(eps[2]) == 0.125                      # sigmoid(0) = 1/2
    # tau stays strictly positive, so q = M + h*tau never approaches zero
    tau = ARMS.tau_from_raw(raws)
    assert float(np.min(tau)) >= WWJ.TAU_MIN
    mass = WWJ.mass_from_eps(tau, eps)
    assert float(np.min(mass + WWJ.H_TOKEN * tau)) >= WWJ.TAU_MIN


def test_the_passive_factors_are_real_exactly_inside_the_bound():
    real = np.float64 if X64 else np.float32
    tau = np.asarray([1.0, 2.0], dtype=real)
    for eps in (0.0, 0.0625, 0.25):
        plus, minus = WWJ.passive_factors(tau, np.asarray(eps, dtype=real))
        assert _close(plus + minus, tau, 1e-6)
        assert _close(plus * minus, WWJ.mass_from_eps(tau, eps), 1e-6)
        assert float(np.min(minus)) >= 0.0


# ---------------------------------------------- scan versus the oracle -----
def test_the_scan_matches_the_sequential_oracle_for_many_lengths():
    tolerance = TOL64 if X64 else TOL32
    for complex_modes in (True, False):
        lambda_bar, b_bar = _modes(1, complex_modes=complex_modes)
        tau, mass = _params(lambda_bar.shape[0])
        for length in (1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 100, 257, 1024, 1500):
            inputs = _inputs(length, length)
            fast = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs)
            slow = ORACLE.sequential_states(lambda_bar, b_bar, tau, mass,
                                            inputs)
            assert fast.shape == (length, lambda_bar.shape[0])
            assert _close(fast, slow, tolerance), (complex_modes, length)


def test_the_scan_matches_the_oracle_with_and_without_rematerialization():
    lambda_bar, b_bar = _modes(2)
    tau, mass = _params(lambda_bar.shape[0])
    inputs = _inputs(3, 64)
    remat = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs, remat=True)
    plain = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs, remat=False)
    assert _close(remat, plain, TOL64 if X64 else TOL32)


def test_zero_prehistory_and_no_wraparound_in_jax():
    """The first state sees only the first input, and an impulse never
    influences anything before it."""
    lambda_bar, b_bar = _modes(3)
    tau, mass = _params(lambda_bar.shape[0])
    length, H = 24, b_bar.shape[1]
    for impulse in (0, 1, 2, 11, 23):
        inputs = np.zeros((length, H)).at[impulse].set(1.0)
        states = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs)
        assert float(np.max(np.abs(states[:impulse]))) == 0.0
        assert float(np.max(np.abs(states[impulse:]))) > 0.0


def test_the_reverse_direction_is_a_flipped_causal_scan():
    """Bidirectional orientation: the reverse branch is the causal scan of
    the reversed sequence, flipped back -- not a shifted forward state."""
    lambda_bar, b_bar = _modes(4)
    tau, mass = _params(lambda_bar.shape[0])
    inputs = _inputs(5, 40)
    reverse = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs,
                             reverse=True)
    manual = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs[::-1])[::-1]
    assert _close(reverse, manual, TOL64 if X64 else TOL32)
    oracle = ORACLE.sequential_states(lambda_bar, b_bar, tau, mass, inputs,
                                      reverse=True)
    assert _close(reverse, oracle, TOL64 if X64 else TOL32)
    # the last token of the reversed direction has seen only the last input
    assert not _close(reverse, WWJ.wwj_states(lambda_bar, b_bar, tau, mass,
                                              inputs), 1e-3)


def test_float32_production_path_stays_within_tolerance_of_float64():
    """The float32 path is checked against the float64 reference, not
    against itself."""
    if not X64:
        return
    lambda_bar, b_bar = _modes(6, dtype=np.complex128)
    tau, mass = _params(lambda_bar.shape[0])
    inputs = _inputs(7, 512)
    exact = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs)
    single = WWJ.wwj_states(lambda_bar.astype(np.complex64),
                            b_bar.astype(np.complex64),
                            tau.astype(np.float32), mass.astype(np.float32),
                            inputs.astype(np.float32))
    assert _close(single.astype(np.complex128), exact, TOL32)


# ------------------------------------------------------------ gradients ----
def _loss(lambda_bar, b_bar, tau, mass, inputs, implementation):
    states = implementation(lambda_bar, b_bar, tau, mass, inputs)
    return np.sum(np.abs(states) ** 2).real


def test_gradients_match_the_oracle_not_only_outputs():
    lambda_bar, b_bar = _modes(8, P=4, H=2)
    tau, mass = _params(4)
    inputs = _inputs(9, 33, H=2)
    tolerance = 1e-8 if X64 else 1e-3
    for argnums in (0, 1, 2, 3):
        fast = jax.grad(_loss, argnums=argnums, holomorphic=False)(
            lambda_bar, b_bar, tau, mass, inputs, WWJ.wwj_states)
        slow = jax.grad(_loss, argnums=argnums, holomorphic=False)(
            lambda_bar, b_bar, tau, mass, inputs, ORACLE.sequential_states)
        assert _close(fast, slow, tolerance), argnums


def test_gradients_flow_to_tau_and_eps_and_stay_finite():
    lambda_bar, b_bar = _modes(10, P=4, H=2)
    inputs = _inputs(11, 65, H=2)
    real = np.float64 if X64 else np.float32

    def loss(tau_raw, eps_raw):
        tau = ARMS.tau_from_raw(tau_raw) * np.ones(4, dtype=real)
        eps = ARMS.eps_passive_from_raw(eps_raw)
        states = WWJ.wwj_states(lambda_bar, b_bar, tau,
                                WWJ.mass_from_eps(tau, eps), inputs)
        return np.sum(np.abs(states) ** 2).real

    value, grads = jax.value_and_grad(loss, argnums=(0, 1))(
        np.asarray(0.5, dtype=real), np.asarray(-1.0, dtype=real))
    assert np.isfinite(value)
    assert all(bool(np.isfinite(g)) for g in grads)
    assert any(abs(float(g)) > 0 for g in grads)


def test_forward_and_backward_are_finite_at_production_shape():
    """A production-shaped sequence, forward and backward, all finite."""
    lambda_bar, b_bar = _modes(12, P=64, H=96)
    tau, mass = _params(64)
    inputs = _inputs(13, 4096, H=96)

    def loss(b):
        return np.sum(np.abs(
            WWJ.wwj_states(lambda_bar, b, tau, mass, inputs)) ** 2).real

    value, grad = jax.value_and_grad(loss)(b_bar)
    assert bool(np.isfinite(value))
    assert bool(np.all(np.isfinite(grad)))


def test_results_are_deterministic_for_a_fixed_seed():
    lambda_bar, b_bar = _modes(14)
    tau, mass = _params(lambda_bar.shape[0])
    inputs = _inputs(15, 129)
    first = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs)
    second = WWJ.wwj_states(lambda_bar, b_bar, tau, mass, inputs)
    assert bool(np.all(first == second))


# ---------------------------------------------------------- diagnostics ----
def test_the_companion_radius_matches_exactly_known_roots():
    """Built from chosen roots, so the answer is known in closed form. The
    estimate never understates the radius."""
    import numpy

    for roots in ([0.5, 0.25, 0.1], [0.99, 0.5, 0.5], [1.2, 0.3, 0.2],
                  [0.7, 0.7, 0.7], [0.6 + 0.3j, 0.6 - 0.3j, 0.2]):
        poly = numpy.poly(roots)                  # z^3 + p1 z^2 + p2 z + p3
        A0, A1, A2 = (np.asarray([-poly[1]]), np.asarray([-poly[2]]),
                      np.asarray([-poly[3]]))
        estimate = float(WWJ.companion_spectral_radius(A0, A1, A2)[0])
        exact = float(numpy.max(numpy.abs(roots)))
        assert estimate >= exact - 1e-6
        assert estimate <= exact * 1.01 + 1e-6, (roots, estimate, exact)


def test_the_response_diagnostics_are_labelled_and_finite():
    lambda_bar, b_bar = _modes(16)
    tau, mass = _params(lambda_bar.shape[0])
    continuous = WWJ.continuous_response(np.log(lambda_bar), tau, mass)
    assert bool(np.all(np.isfinite(continuous)))
    (A0, A1, A2), _ = WWJ.state_coefficients(lambda_bar, b_bar, tau, mass)
    discrete = WWJ.discrete_response(tau, mass, A0, A1, A2,
                                     np.exp(1j * np.asarray(0.3)))
    assert bool(np.all(np.isfinite(discrete)))
    # they are different objects: the continuous polynomial is not the
    # discrete transfer function
    assert not _close(continuous, discrete, 1e-3)


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
    inputs = _inputs(17, length, H=4)
    variables = model.init(jax.random.PRNGKey(0), inputs)
    return model.apply(variables, inputs), variables


def test_both_new_arms_run_and_carry_their_own_parameters():
    for constructor, expected in ((ARMS.init_wwj_critical_S5SSM,
                                   {"wwj_tau_raw"}),
                                  (ARMS.init_wwj_passive_S5SSM,
                                   {"wwj_tau_raw", "wwj_eps_raw"})):
        out, variables = _apply_arm(constructor)
        assert bool(np.all(np.isfinite(out)))
        names = set(variables["params"])
        assert expected <= names
        # one scalar per layer, not one per mode
        for name in expected:
            assert variables["params"][name].shape == ()


def test_the_bidirectional_arm_runs_both_directions():
    forward, _ = _apply_arm(ARMS.init_wwj_critical_S5SSM)
    both, variables = _apply_arm(ARMS.init_wwj_critical_S5SSM,
                                 bidirectional=True)
    assert both.shape == forward.shape          # C_tilde absorbs the width
    assert bool(np.all(np.isfinite(both)))
    # the reverse branch really contributes: the outputs differ
    assert float(np.max(np.abs(both - forward))) > 0.0


def test_the_critical_arm_fixes_eps_at_one_quarter():
    model = ARMS.init_wwj_critical_S5SSM(**_arm_kwargs())()
    inputs = _inputs(18, 8, H=4)
    variables = model.init(jax.random.PRNGKey(0), inputs)
    assert "wwj_eps_raw" not in variables["params"]
    diagnostics = model.apply(variables, method=lambda m: m.diagnostics())
    assert _close(diagnostics["eps"],
                  0.25 * np.ones_like(diagnostics["eps"]), 1e-6)
    assert _close(diagnostics["mass"],
                  0.25 * diagnostics["tau"] ** 2, 1e-5)
    assert bool(np.all(np.isfinite(
        diagnostics["companion_spectral_radius"])))


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
    """Native S5 is the control and must not move, and neither may the old
    two-compartment generalized arm or the production factory."""
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py",
                     "s5/generalized_prospective_ssm.py",
                     "s5/prospective_ssm.py",
                     "experiments/s5_three_arm_full/data.py"):
        assert _blob(relative) == _worktree(relative) != "", relative


def test_the_new_code_is_not_reachable_from_native_s5():
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py"):
        source = open(os.path.join(REPO, relative)).read()
        assert "wwj" not in source.lower(), relative
    # and the oracle is a test file, imported by tests only
    production = subprocess.run(
        ["grep", "-rl", "wwj_sequential_reference", "s5", "experiments",
         "bin"], capture_output=True, text=True, cwd=REPO)
    assert production.stdout.strip() == ""


def test_no_test_split_is_loaded_by_any_wwj_training_path():
    for relative in ("experiments/s5_wwj/benchmark.py",
                     "experiments/s5_wwj/dev_gate.py",
                     "experiments/s5_wwj/init_grid.py"):
        source = open(os.path.join(REPO, relative)).read()
        loads = [line for line in source.splitlines()
                 if "load_official_raw" in line]
        assert all('"test"' not in line for line in loads), relative
