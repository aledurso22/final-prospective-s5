"""Heterogeneous modal prospective cascade: stable first-order stages.

WHAT THIS IS, STATED ACCURATELY. A heterogeneous many-body, WWJ-INSPIRED
modal action/cascade. The local first-order factors are variationally
motivated by the WWJ operator; the particular coupling topology -- one
independent cascade per native S5 mode, with learned gates selecting the
regime -- is an EXPLICIT MODEL CONSTRUCTION, not a consequence of the
variational principle. It is not the matched residual law, and it does not
claim to be derived from one.

WHY A CASCADE RATHER THAN A REPLACEMENT RECURRENCE. The direct matched
route (branch `s5-direct-generalized-prospective`) replaced the S5
recurrence with an order-2/3 companion whose poles are produced by the
discretization. Those poles left the unit disc on the real S5 modes -- the
mixed-stencil form reached 1.705-1.877, the professor-target form grew like
2h/tau, tau = 2 gave 1.4416 and diverged at token 242 in float32 -- and the
exactly matched form collapses the driven response to (I - Abar)s = Bbar x,
a memoryless map. That work is preserved there as a documented negative
result. Here the S5 recurrence is UNTOUCHED and the prospective operator is
a cascade of first-order stages whose poles are known in closed form and
cannot leave (0, 1).

THE STAGE. With the backward derivative D_h y_t = (y_t - y_{t-1})/h, the
stage (1 + d D_h) u = (1 + n D_h) v is, multiplying by h and solving,

    u_t = [d/(h+d)] u_{t-1} + [(h+n)/(h+d)] v_t - [n/(h+d)] v_{t-1},

so its ONLY recurrent pole is p = d/(h+d), which lies strictly in (0, 1)
for every d > 0, h > 0. The numerator parameter n moves a ZERO, at
n/(h+n), and never a pole. At n = d the stage is the exact identity for
zero-consistent history.

THE REGIMES, selected by two learned gates g in [0, 1] through
n = d + g * softplus(delta):

    g1 = g2 = 0   both stages are identities   -> exactly Native S5
    one gate on  one active first-order stage  -> ordinary prospectivity
    both gates on two cascaded stages          -> generalized WWJ,

the latter having transfer
(1 + n1 D)(1 + n2 D) / [(1 + d1 D)(1 + d2 D)], whose numerator expands to
1 + Gamma_n D + M_n D^2 with Gamma_n = n1 + n2 and M_n = n1 n2. The
critical numerator branch is n1 = n2 = tau/2, giving M_n = tau^2/4 -- the
WWJ critical mass, as a NUMERATOR, where it cannot destabilize anything.

CANCELLATION IS MONITORED, NOT ASSUMED AWAY. Each stage's numerator zero
sits at n/(h+n). If that lands on a native pole the mode is annihilated, so
`native_mode_gain` reports |L_j(lambda_bar_j)| and the layer penalizes it as
it approaches zero. Exact cancellation must never pass silently.

IMPLEMENTATION. Every stage is a SCALAR (per-mode diagonal) affine
recurrence, run with the repository's own `binary_operator` through
`jax.lax.associative_scan` -- the same primitive `s5/ssm.py` uses. No dense
per-mode 3x3 or 4x4 companion is ever built; the generalized branch is two
sequential scalar scans.

h is the TOKEN step, 1.0, and is never identified with S5's learned Delta.
"""

import jax
import jax.numpy as np

from .ssm import binary_operator

H_TOKEN = 1.0
#: d is kept strictly positive, so the pole d/(h+d) stays inside (0, 1)
D_MIN = 1e-3
#: regime labels, decided by the gates
NATIVE, ORDINARY, GENERALIZED = "native", "ordinary", "generalized"


def stage_coefficients(d, n, h=H_TOKEN):
    """(pole, alpha, beta) of (1 + d D_h) u = (1 + n D_h) v.

    u_t = pole * u_{t-1} + alpha * v_t + beta * v_{t-1}
    pole = d/(h+d)    alpha = (h+n)/(h+d)    beta = -n/(h+d)
    """
    denominator = h + d
    return d / denominator, (h + n) / denominator, -n / denominator


def stage_pole(d, h=H_TOKEN):
    """The stage's only recurrent pole, in closed form."""
    return d / (h + d)


def numerator_zero(n, h=H_TOKEN):
    """Where the stage puts its zero. Moving n never moves a pole."""
    return n / (h + n)


def numerator_from_gate(d, gate, delta):
    """n = d + g * softplus(delta), so g = 0 gives n = d: the identity."""
    return d + gate * jax.nn.softplus(delta)


def gamma_and_mass(n1, n2):
    """(Gamma_n, M_n) of the expanded numerator 1 + Gamma_n D + M_n D^2."""
    return n1 + n2, n1 * n2


def critical_numerator(tau):
    """n1 = n2 = tau/2, so Gamma_n = tau and M_n = tau^2/4."""
    half = tau / 2.0
    return half, half


def regime_of(gate_1, gate_2, threshold=1e-6):
    """Which regime a pair of gates selects."""
    first = float(np.max(np.abs(np.asarray(gate_1)))) > threshold
    second = float(np.max(np.abs(np.asarray(gate_2)))) > threshold
    if not first and not second:
        return NATIVE
    if first and second:
        return GENERALIZED
    return ORDINARY


# ----------------------------------------------------------------- scans ---
def _shift(values, distance=1):
    """Shift forward in time by `distance`, ZERO prehistory."""
    zeros = np.zeros_like(values[:distance])
    return np.concatenate((zeros, values[:-distance]), axis=0)


def apply_stage(values, d, n, h=H_TOKEN, reverse=False):
    """One first-order stage, as a diagonal affine associative scan.

    The drive is alpha * v_t + beta * v_{t-1} with zero prehistory, and the
    recurrence u_t = pole * u_{t-1} + drive_t is exactly the shape
    `s5/ssm.py`'s `binary_operator` composes, so the repository's own scan
    primitive is reused rather than a new one written.
    """
    sequence = values[::-1] if reverse else values
    pole, alpha, beta = stage_coefficients(d, n, h)
    # every operand is pinned to the sequence's dtype: see `native_states`
    pole = pole.astype(sequence.dtype)
    drive = (alpha.astype(sequence.dtype) * sequence
             + beta.astype(sequence.dtype) * _shift(sequence)
             ).astype(sequence.dtype)
    poles = pole * np.ones((sequence.shape[0], pole.shape[0]),
                           dtype=sequence.dtype)
    _, states = jax.lax.associative_scan(binary_operator, (poles, drive))
    return states[::-1] if reverse else states


def apply_cascade(values, stages, h=H_TOKEN, reverse=False):
    """Two (or more) stages in sequence: q -> u -> w.

    Two sequential scalar scans, never a widened state. `stages` is a
    sequence of (d, n) pairs; an empty sequence returns the input, which is
    the Native regime.
    """
    out = values
    for d, n in stages:
        out = apply_stage(out, d, n, h, reverse)
    return out


def native_states(lambda_bar, b_bar, input_sequence, reverse=False):
    """q_{j,t+1} = lambda_bar_j q_{j,t} + b_bar_j x_t, from the S5 scan.

    DTYPE DISCIPLINE. The broadcast array carries lambda_bar's dtype
    explicitly. Without it, `np.ones(...)` is float64 whenever x64 is
    enabled, so a complex64 lambda_bar promotes to complex128 while the
    complex64 drive does not, and the scan's concatenate fails with a dtype
    mismatch. Both operands of the scan are pinned to the SAME dtype here.
    """
    elements = lambda_bar * np.ones((input_sequence.shape[0],
                                     lambda_bar.shape[0]),
                                    dtype=lambda_bar.dtype)
    drive = jax.vmap(lambda u: b_bar @ u)(input_sequence).astype(
        lambda_bar.dtype)
    _, states = jax.lax.associative_scan(binary_operator, (elements, drive),
                                         reverse=reverse)
    return states


# ----------------------------------------------------------- diagnostics ---
def stage_response(z, d, n, h=H_TOKEN):
    """(1 + n D_h)/(1 + d D_h) evaluated at z, with D_h -> (1 - 1/z)/h."""
    derivative = (1.0 - 1.0 / z) / h
    return (1.0 + n * derivative) / (1.0 + d * derivative)


def effective_filter(z, stages, h=H_TOKEN):
    """L(z) = product of the stage responses."""
    out = 1.0
    for d, n in stages:
        out = out * stage_response(z, d, n, h)
    return out


def native_mode_gain(lambda_bar, stages, h=H_TOKEN):
    """|L_j(lambda_bar_j)|: how much of each native mode survives.

    A value approaching zero means the cascade is CANCELLING that native S5
    mode. The layer penalizes it; exact cancellation must never pass
    silently.
    """
    return np.abs(effective_filter(lambda_bar, stages, h))


def added_poles(stages, h=H_TOKEN):
    """Every pole the cascade adds: d/(h+d) per stage. Nothing else."""
    if not stages:
        return np.zeros((0,), dtype=np.float32)
    return np.stack([stage_pole(d, h) for d, _ in stages], axis=0)
