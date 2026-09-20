"""The EXACT discrete WWJ prospective operator on a Native S5 trajectory.

WHY THIS REPLACED THE MIXED-STENCIL RECURRENCE. The continuous residual law
P(D)(s - f) = 0 is P(D)s = P(D)f, so for matched initial conditions its
transfer function is S/F = 1: it controls homogeneous residual transients and
creates NO persistent prospective transformation of f. The earlier
realization escaped that cancellation only by applying DIFFERENT discrete
derivatives to s and to f, which means its extra poles were artefacts of the
discretization -- and on the cluster, at commit bfe53fe, those poles were
unstable for the actual production-initialized S5 modes (max companion radius
about 1.705 at every declared grid cell, float32 NaNs, float64 states to
about 1e81). It is kept only as a labelled failed diagnostic in
`s5/wwj_mixed_stencil.py`; see docs/S5_WWJ_PROSPECTIVE.md.

WHAT THIS MODULE DOES INSTEAD. The Native S5 memory is untouched,

    s_{t+1} = Abar s_t + Bbar x_t,

and the WWJ operator is applied to that trajectory with ONE consistent
backward derivative,

    D_h s_t = (s_t - s_{t-1})/h,   D_h^2 s_t = (s_t - 2 s_{t-1} + s_{t-2})/h^2

(and D_h(D_h s) is exactly D_h^2 s, which is what makes the passive
factorization exact here, unlike the mixed-stencil case). With k = tau/h and
m = M/h^2 = eps*k^2,

    z_t = P(D_h) s_t = (1 + k + m) s_t - (k + 2m) s_{t-1} + m s_{t-2}.

This is an FIR (three-tap) operator: it adds ZEROS, not poles. The exact
recurrent state-space realization is

    q_t = [s_t; s_{t-1}; s_{t-2}]
    q_{t+1} = [[Abar,0,0],[I,0,0],[0,I,0]] q_t + [Bbar x_t; 0; 0]
    z_t = [(1+k+m)I, -(k+2m)I, m I] q_t

whose poles are exactly the Native poles plus two zero history-shift poles,
because the transition is block lower triangular with diagonal blocks Abar,
0, 0. So this is a genuine recurrent S5 layer carrying the WWJ
generalized-prospective coordinate, and it cannot destabilize what Native S5
does not already do.

IMPLEMENTATION. The augmented matrix is never scanned. The ordinary,
optimized Native S5 parallel scan runs first -- literally
`jax.lax.associative_scan(s5.ssm.binary_operator, ...)`, the same primitive
`s5/ssm.py` uses -- and the three-tap is then applied to its states, an
O(L*P) elementwise pass. Equality with the augmented realization is proved
algebraically in `tests/test_wwj_operator_algebra.py` and numerically in
`tests/test_wwj_ssm.py`.

h is the TOKEN step, 1.0, and is never identified with S5's learned Delta.
"""

import jax
import jax.numpy as np

from .ssm import binary_operator

#: the token step of the discrete operator. Not S5's learned Delta.
H_TOKEN = 1.0
#: P(D) has real (passive, non-oscillatory) roots iff eps <= 1/4
EPS_MAX = 0.25
EPS_CRITICAL = 0.25
#: tau > 0 strictly; a floor keeps the prospective gradients from vanishing
TAU_MIN = 1e-3


def k_and_m(tau, eps, h=H_TOKEN):
    """(k, m) = (tau/h, eps*k^2); m = M/h^2 with M = eps*tau^2."""
    k = tau / h
    return k, eps * k * k


def _shift(values, distance):
    """Shift forward in time by `distance`, with ZERO prehistory."""
    zeros = np.zeros_like(values[:distance])
    return np.concatenate((zeros, values[:-distance]), axis=0)


def three_tap(states, k, m):
    """z_t = (1+k+m) s_t - (k+2m) s_{t-1} + m s_{t-2}, zero prehistory.

    O(L*P), no scan, no recurrence: the WWJ contribution is FIR.
    """
    k = k.astype(states.dtype) if hasattr(k, "astype") else k
    m = m.astype(states.dtype) if hasattr(m, "astype") else m
    return ((1.0 + k + m) * states
            - (k + 2.0 * m) * _shift(states, 1)
            + m * _shift(states, 2))


def two_stage(states, t_plus, t_minus, h=H_TOKEN):
    """Apply (1 + t_+ D_h)(1 + t_- D_h) as two first-order stages.

    Exact here, because both stages use the SAME discrete derivative; with
    zero prehistory in each stage it equals `three_tap` exactly, which
    `tests/test_wwj_operator_algebra.py` proves in rational arithmetic.
    """
    def stage(values, factor):
        a = factor / h
        return (1.0 + a) * values - a * _shift(values, 1)

    return stage(stage(states, t_minus), t_plus)


def passive_factors(tau, eps):
    """t_pm = (tau/2)(1 +- sqrt(1-4 eps)); real exactly on 0 <= eps <= 1/4."""
    root = np.sqrt(np.maximum(1.0 - 4.0 * eps, 0.0))
    half = tau / 2.0
    return half * (1.0 + root), half * (1.0 - root)


def gated_states(states, k, m, gate):
    """s~_t = s_t + g[k(s_t - s_{t-1}) + m(s_t - 2 s_{t-1} + s_{t-2})].

    A SEPARATE intervention, equal to the direct operator only at g = 1 and
    to Native S5 only at g = 0. It is never described as the direct arm.
    """
    return states + gate * (three_tap(states, k, m) - states)


def native_states(lambda_bar, b_bar, input_sequence, reverse=False):
    """Native S5 states from the EXISTING optimized parallel scan.

    This is `s5/ssm.py`'s own `binary_operator` and the same
    `jax.lax.associative_scan` call that `apply_ssm` makes; nothing about the
    Native recurrence is reimplemented here.
    """
    lambda_elements = lambda_bar * np.ones(
        (input_sequence.shape[0], lambda_bar.shape[0]))
    bu_elements = jax.vmap(lambda u: b_bar @ u)(input_sequence)
    _, states = jax.lax.associative_scan(binary_operator,
                                         (lambda_elements, bu_elements),
                                         reverse=reverse)
    return states


def wwj_states(lambda_bar, b_bar, input_sequence, k, m, reverse=False):
    """Native S5 states with the WWJ operator applied causally.

    For `reverse=True` the Native suffix scan is used and the three-tap is
    applied in THAT direction's own causal order (its history is the later
    tokens), by flipping, applying and flipping back. No forward-history
    shift ever touches a concatenated bidirectional state.
    """
    states = native_states(lambda_bar, b_bar, input_sequence, reverse=reverse)
    if reverse:
        return three_tap(states[::-1], k, m)[::-1]
    return three_tap(states, k, m)


# ----------------------------------------------------------- diagnostics ---
def frequency_response(k, m, omega):
    """P_h(e^{i omega}) = 1 + k(1 - e^{-i omega}) + m(1 - e^{-i omega})^2."""
    delta = 1.0 - np.exp(-1j * omega)
    return 1.0 + k * delta + m * delta * delta


def max_fir_gain(k, m, points=2049):
    """max |P_h(e^{i omega})| over a dense grid of omega in [0, pi].

    The FIR replacement for the companion spectral radius: this operator has
    no poles, so what matters is how much it amplifies, not whether it is
    stable.
    """
    omega = np.linspace(0.0, np.pi, points)
    return np.max(np.abs(frequency_response(k, m, omega)))


def recurrent_poles(lambda_bar):
    """The layer's recurrent poles: the Native ones, plus two zeros.

    Stated as a function so the claim is checkable rather than asserted in
    prose; `tests/test_wwj_ssm.py` verifies it against the eigenvalues of
    the augmented transition matrix.
    """
    zeros = np.zeros((2,), dtype=lambda_bar.dtype)
    return np.concatenate((lambda_bar, zeros))


def native_radii(lambda_bar):
    """|Abar_j| per mode: the only radii that matter now."""
    return np.abs(lambda_bar)
