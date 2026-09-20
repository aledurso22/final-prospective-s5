"""The JAX implementation of the modal prospective cascade (cluster).

The algebra is proved without JAX in `test_modal_prospective_algebra.py`;
this checks the implementation, its gradients, its orientation, its
behaviour at production shape and precision, and that Native S5 is
untouched.

    JAX_ENABLE_X64=1 $PY -m pytest -x -s tests/test_modal_prospective_jax.py
"""

import os
import subprocess

import jax
import jax.numpy as np

from s5 import modal_prospective as MP
from s5 import modal_prospective_ssm as ARMS
from tests import modal_prospective_reference as ORACLE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the clean production commit this branch starts from
NATIVE_BASE = "ef004cda025b4e098041cfe5970c217fa0015bba"
X64 = jax.config.read("jax_enable_x64")
TOL64, TOL32 = 1e-11, 2e-5
#: awkward lengths, non-powers of two, and the production length
LENGTHS = (1, 2, 3, 5, 7, 17, 100, 257, 1000, 1023, 4096, 16000)


def _complex():
    return np.complex128 if X64 else np.complex64


def _real():
    return np.float64 if X64 else np.float32


def _modes(seed, P=6, H=3, complex_modes=True):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    magnitude = 0.2 + 0.75 * jax.random.uniform(keys[0], (P,))
    if complex_modes:
        angle = jax.random.uniform(keys[1], (P,), minval=-3.1, maxval=3.1)
        lambda_bar = (magnitude * np.exp(1j * angle)).astype(_complex())
    else:
        lambda_bar = magnitude.astype(_complex())
    b_bar = (jax.random.normal(keys[2], (P, H))
             + 1j * jax.random.normal(keys[3], (P, H))).astype(_complex())
    return lambda_bar, b_bar


def _inputs(seed, length, H=3):
    return jax.random.normal(jax.random.PRNGKey(seed),
                             (length, H)).astype(_real())


def _stages(P, pairs):
    """[(d, n)] as complex arrays of the working dtype."""
    return [(np.full((P,), d, dtype=_complex()),
             np.full((P,), n, dtype=_complex())) for d, n in pairs]


def _close(left, right, tolerance):
    if not (bool(np.all(np.isfinite(left)))
            and bool(np.all(np.isfinite(right)))):
        return False
    scale = float(np.maximum(np.max(np.abs(right)), 1.0))
    return float(np.max(np.abs(left - right))) <= tolerance * scale


# --------------------------------------------- scan versus the oracle -----
def test_the_stage_scan_matches_the_sequential_oracle():
    """REQUIREMENT 7: real and complex modes, awkward lengths, and 16000."""
    tolerance = TOL64 if X64 else TOL32
    for complex_modes in (True, False):
        lambda_bar, b_bar = _modes(1, complex_modes=complex_modes)
        P = lambda_bar.shape[0]
        (d, n), = _stages(P, ((1.0, 3.0),))
        for length in LENGTHS:
            values = ORACLE.native_sequential(lambda_bar, b_bar,
                                              _inputs(length, length))
            fast = MP.apply_stage(values, d, n)
            slow = ORACLE.stage_sequential(values, d, n)
            assert fast.shape == values.shape
            assert _close(fast, slow, tolerance), (complex_modes, length)


def test_the_two_stage_cascade_matches_the_oracle():
    lambda_bar, b_bar = _modes(2)
    stages = _stages(lambda_bar.shape[0], ((1.0, 3.0), (0.5, 2.0)))
    for length in (3, 17, 257, 1023, 4096):
        values = ORACLE.native_sequential(lambda_bar, b_bar,
                                          _inputs(length + 1, length))
        assert _close(MP.apply_cascade(values, stages),
                      ORACLE.cascade_sequential(values, stages),
                      TOL64 if X64 else TOL32), length


def test_identity_stages_reproduce_native_sequences_and_gradients():
    """REQUIREMENT 2. With n = d every stage is the identity, so the layer
    is Native S5 -- in value AND in gradient."""
    lambda_bar, b_bar = _modes(3, P=4, H=2)
    inputs = _inputs(4, 257, H=2)
    P = lambda_bar.shape[0]
    stages = _stages(P, ((1.0, 1.0), (0.37, 0.37)))
    native = MP.native_states(lambda_bar, b_bar, inputs)
    cascaded = MP.apply_cascade(native, stages)
    assert _close(cascaded, native, TOL64 if X64 else TOL32)

    def loss(b, use_cascade):
        states = MP.native_states(lambda_bar, b, inputs)
        if use_cascade:
            states = MP.apply_cascade(states, stages)
        return np.sum(np.abs(states) ** 2).real

    assert _close(jax.grad(loss)(b_bar, True), jax.grad(loss)(b_bar, False),
                  1e-8 if X64 else 1e-3)


def test_gradients_match_the_oracle_for_every_parameter():
    lambda_bar, b_bar = _modes(5, P=4, H=2)
    inputs = _inputs(6, 129, H=2)
    P = lambda_bar.shape[0]
    d = np.full((P,), 1.0, dtype=_complex())
    n = np.full((P,), 2.5, dtype=_complex())
    tolerance = 1e-8 if X64 else 1e-3

    def loss(lam, b, d_value, n_value, implementation):
        values = MP.native_states(lam, b, inputs)
        return np.sum(np.abs(
            implementation(values, d_value, n_value)) ** 2).real

    for argnums in (0, 1, 2, 3):
        fast = jax.grad(loss, argnums=argnums)(lambda_bar, b_bar, d, n,
                                               MP.apply_stage)
        slow = jax.grad(loss, argnums=argnums)(lambda_bar, b_bar, d, n,
                                               ORACLE.stage_sequential)
        assert _close(fast, slow, tolerance), argnums


def test_zero_prehistory_and_no_wraparound_in_jax():
    """REQUIREMENT 9."""
    P = 4
    (d, n), = _stages(P, ((1.0, 3.0),))
    length = 24
    for impulse in (0, 1, 2, 23):
        values = np.zeros((length, P), dtype=_complex()).at[impulse].set(1.0)
        out = MP.apply_stage(values, d, n)
        assert bool(np.all(out[:impulse] == 0)), impulse
        assert bool(np.any(out[impulse] != 0)), impulse
    values = np.zeros((length, P), dtype=_complex()).at[length - 1].set(1.0)
    assert bool(np.all(MP.apply_stage(values, d, n)[:length - 1] == 0))


def test_the_reverse_direction_is_a_flipped_causal_cascade():
    """REQUIREMENT 10."""
    lambda_bar, b_bar = _modes(7)
    stages = _stages(lambda_bar.shape[0], ((1.0, 3.0), (0.5, 2.0)))
    inputs = _inputs(8, 64)
    values = MP.native_states(lambda_bar, b_bar, inputs, reverse=True)
    reverse = MP.apply_cascade(values, stages, reverse=True)
    oracle = ORACLE.cascade_sequential(values, stages, reverse=True)
    assert _close(reverse, oracle, TOL64 if X64 else TOL32)
    forward = MP.apply_cascade(MP.native_states(lambda_bar, b_bar, inputs),
                               stages)
    assert not _close(reverse, forward, 1e-3)


# --------------------------------------------------- poles and stability --
def test_every_added_pole_is_inside_the_unit_circle_in_jax():
    """REQUIREMENT 5, and REQUIREMENT 6: the gate cannot move it."""
    for d_value in (1e-3, 0.1, 1.0, 10.0, 1e4):
        d = np.full((3,), d_value, dtype=_real())
        pole = MP.stage_pole(d)
        assert float(np.max(pole)) < 1.0 and float(np.min(pole)) > 0.0
        for gate in (0.0, 0.5, 1.0):
            n = MP.numerator_from_gate(d, np.asarray(gate, dtype=_real()),
                                       np.asarray(0.7, dtype=_real()))
            assert _close(MP.stage_pole(d), pole, 1e-12)
            assert float(np.max(MP.stage_coefficients(d, n)[0])) == \
                float(np.max(pole))


def test_production_length_float32_values_and_gradients_are_finite():
    """REQUIREMENT 8, at the production shape."""
    lambda_bar, b_bar = _modes(9, P=64, H=96)
    single = lambda_bar.astype(np.complex64), b_bar.astype(np.complex64)
    inputs = jax.random.normal(jax.random.PRNGKey(10),
                               (16000, 96)).astype(np.float32)
    stages = [(np.full((64,), value, dtype=np.complex64),
               np.full((64,), other, dtype=np.complex64))
              for value, other in ((1.0, 3.0), (0.5, 2.0))]

    def loss(b):
        states = MP.native_states(single[0], b, inputs)
        return np.sum(np.abs(MP.apply_cascade(states, stages)) ** 2).real

    value, grad = jax.value_and_grad(loss)(single[1])
    assert bool(np.isfinite(value)), value
    assert bool(np.all(np.isfinite(grad)))
    states = MP.apply_cascade(MP.native_states(single[0], single[1], inputs),
                              stages)
    assert bool(np.all(np.isfinite(states)))
    print({"max_abs_state": float(np.max(np.abs(states))), "loss": float(value)})


def test_the_native_mode_gain_reports_cancellation():
    """A cascade whose numerator zero sits on a native pole annihilates it,
    and the reported gain must show that rather than hide it."""
    d_value = 0.5
    n_value = 3.0
    zero = n_value / (1.0 + n_value)            # the numerator zero
    lambda_bar = np.asarray([zero, 0.2, 0.9], dtype=_complex())
    stages = _stages(3, ((d_value, n_value),))
    gain = MP.native_mode_gain(lambda_bar, stages)
    assert float(gain[0]) < 1e-6, gain          # cancelled
    assert float(np.min(gain[1:])) > 1e-3, gain # the others survive


# -------------------------------------------------------------- the arm ---
def _arm_kwargs(P=8, H=4):
    import numpy

    return dict(Lambda_re_init=-0.5 * numpy.ones(P),
                Lambda_im_init=numpy.linspace(0.1, 1.0, P),
                V=numpy.eye(P, dtype=numpy.complex64),
                Vinv=numpy.eye(P, dtype=numpy.complex64),
                H=H, P=P, C_init="lecun_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=False, clip_eigs=True)


def test_the_layer_runs_and_starts_at_native():
    """gate_init is strongly negative, so every mode starts Native."""
    model = ARMS.init_modal_prospective_S5SSM(**_arm_kwargs())()
    inputs = _inputs(11, 64, H=4)
    variables = model.init(jax.random.PRNGKey(0), inputs)
    out = model.apply(variables, inputs)
    assert bool(np.all(np.isfinite(out)))
    for index in (0, 1):
        for suffix in ("d_raw", "delta_raw", "gate_raw"):
            assert f"stage{index}_{suffix}" in variables["params"]
    diagnostics = model.apply(variables, method=lambda m: m.diagnostics())
    print(diagnostics)
    assert diagnostics["max_stage_pole"] < 1.0
    assert all(value < 0.05 for value in diagnostics["mean_gate"])
    assert diagnostics["regime_per_mode"][MP.NATIVE] == 8
    assert diagnostics["native_mode_gain_min"] > 0.0


def test_the_penalty_grows_with_the_gates():
    kwargs = _arm_kwargs()
    inputs = _inputs(12, 32, H=4)
    closed = ARMS.init_modal_prospective_S5SSM(gate_init=-8.0, **kwargs)()
    open_gates = ARMS.init_modal_prospective_S5SSM(gate_init=2.0, **kwargs)()
    values = []
    for model in (closed, open_gates):
        variables = model.init(jax.random.PRNGKey(0), inputs)
        values.append(float(model.apply(variables,
                                        method=lambda m: m.penalty())))
    assert values[1] > values[0], values


def test_a_bidirectional_layer_runs_both_orientations():
    forward = ARMS.init_modal_prospective_S5SSM(**_arm_kwargs())()
    both = ARMS.init_modal_prospective_S5SSM(bidirectional=True,
                                             **_arm_kwargs())()
    inputs = _inputs(13, 48, H=4)
    first = forward.apply(forward.init(jax.random.PRNGKey(0), inputs), inputs)
    second = both.apply(both.init(jax.random.PRNGKey(0), inputs), inputs)
    assert bool(np.all(np.isfinite(second)))
    assert second.shape == first.shape


# ------------------------------------------------- what must not change ---
def _blob(relative):
    return subprocess.run(["git", "-C", REPO, "rev-parse",
                           f"{NATIVE_BASE}:{relative}"], capture_output=True,
                          text=True).stdout.strip()


def _worktree(relative):
    return subprocess.run(["git", "-C", REPO, "hash-object",
                           os.path.join(REPO, relative)],
                          capture_output=True, text=True).stdout.strip()


def test_native_s5_and_every_existing_arm_are_byte_identical():
    """REQUIREMENT 11."""
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py", "s5/prospective_ssm.py",
                     "s5/generalized_prospective_ssm.py",
                     "s5/seq_model.py", "s5/train_helpers.py",
                     "experiments/s5_three_arm_full/runner.py",
                     "experiments/s5_three_arm_full/data.py"):
        assert _blob(relative) == _worktree(relative) != "", relative
