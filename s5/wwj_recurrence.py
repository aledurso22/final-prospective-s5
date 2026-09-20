"""WWJ prospective recurrence: coefficients, parallel scan and diagnostics.

THE CONTINUOUS LAW. The Euclidean WWJ/Bregman residual equation applies one
second-order operator to the COMPLETE residual,

    P(D)(s - f) = 0,      P(D) = 1 + tau*D + M*D^2,      M = eps * tau^2,

that is

    M*s'' + tau*s' + s  =  f + tau*f' + M*f''.

The `M*f''` term on the right is the correction relative to the older
two-compartment generalized arm in `s5/generalized_prospective_ssm.py`, which
is a DIFFERENT equation and is kept frozen for comparison.

THE DISCRETE REALIZATION IS NOT THAT EQUATION. The update implemented here is
an explicit, causal, MIXED-STENCIL (IMEX-type) discretization: the state uses
forward stencils and the target uses backward stencils,

    s'_t  ~ (s_{t+1} - s_t)/h           f'_t  ~ (f_t - f_{t-1})/h
    s''_t ~ (s_{t+1} - 2 s_t + s_{t-1})/h^2
                                        f''_t ~ (f_t - 2 f_{t-1} + f_{t-2})/h^2

It is consistent with the continuous law, and it is NOT the exact finite-step
identity P(D_h)(s - f) = 0, because the two sides use different stencils. It
must never be described as such.

DERIVATION (rederived here, not copied). Multiplying through by h^2,

    LHS*h^2 = M(s_{t+1} - 2 s_t + s_{t-1}) + h*tau*(s_{t+1} - s_t) + h^2 s_t
            = (M + h tau) s_{t+1} + (h^2 - 2M - h tau) s_t + M s_{t-1}
    RHS*h^2 = (M + h tau + h^2) f_t - (2M + h tau) f_{t-1} + M f_{t-2}

so with q = M + h*tau,

    s_{t+1} = a s_t - b s_{t-1} + c0 f_t + c1 f_{t-1} + c2 f_{t-2}
    a  = (2M + h tau - h^2)/q      b  = M/q
    c0 = (M + h tau + h^2)/q       c1 = -(2M + h tau)/q      c2 = M/q

At M = 0 this is exactly

    s_{t+1} = (1 - h/tau) s_t + (1 + h/tau) f_t - f_{t-1}.

With the native-matched target f_t = F_tau s_t + G_tau x_t,
F_tau = I + (tau/h)(Abar - I), G_tau = (tau/h) Bbar, substitution gives the
third-order recurrence

    s_{t+1} = A0 s_t + A1 s_{t-1} + A2 s_{t-2}
              + C0 x_t + C1 x_{t-1} + C2 x_{t-2}
    A0 = a + c0 F_tau     A1 = -b + c1 F_tau     A2 = c2 F_tau
    C0 = c0 G_tau         C1 = c1 G_tau          C2 = c2 G_tau

h is the TOKEN step and is 1.0 here. It is deliberately NOT identified with
S5's learned continuous-time step Delta, which enters only through Abar, Bbar.

WHY THE SCAN IS CHEAP IN TIME. F_tau is diagonal in the S5 mode basis, so
A0, A1, A2 are diagonal: per mode the recurrence is a scalar order-3 linear
recurrence whose companion matrix H is 3x3 and, crucially, CONSTANT IN TIME.
The scan therefore never materializes a per-token transition matrix: a
Hillis-Steele doubling carries a single 3x3 matrix per mode per LEVEL,
ceil(log2 L) levels. No sequential loop over tokens, no rematerialized
rollout per step.

WHAT IS AND IS NOT KNOWN ABOUT MEMORY. Three different quantities:

  * FORWARD LIVE STORAGE is O(L*P*3): the lifted state, plus one shifted
    copy while a level is being applied.
  * BACKWARD (AUTODIFF RESIDUAL) STORAGE IS NOT O(L*P*3). With
    `remat="level"` each level recomputes its own internals, but reverse
    mode still needs each level BOUNDARY, so the residuals are bounded by
    about ceil(log2 L) arrays of size L*P*3 -- O(L*P*log L) -- unless XLA
    elides some of them. Per-level checkpointing improves the constant, not
    the log factor. `remat="whole"` instead keeps only the scan's inputs and
    recomputes every level in the backward pass, trading roughly 2x the
    forward flops for O(L*P*3) residuals.
  * MEASURED PEAK DEVICE MEMORY is what actually decides, and only the GPU
    benchmark can report it.

No memory bound is claimed here beyond the forward one. If the benchmark
shows the residuals are too large, the fix is `remat="whole"`, a coarser
rematerialization region, or a custom VJP for this structured recurrence --
never a change to the mathematics above.
"""

import jax
import jax.numpy as np

#: the token step of the discretization. Not S5's learned Delta.
H_TOKEN = 1.0
#: the passive (non-oscillatory) bound: P(D) has real roots iff eps <= 1/4
EPS_MAX = 0.25
#: eps of the critically damped arm
EPS_CRITICAL = 0.25
#: keeps q = M + h*tau away from zero
TAU_MIN = 1e-3


def scalar_coefficients(tau, mass, h=H_TOKEN):
    """(a, b, c0, c1, c2) of the mixed-stencil update, from tau and M."""
    q = mass + h * tau
    return ((2.0 * mass + h * tau - h * h) / q,
            mass / q,
            (mass + h * tau + h * h) / q,
            -(2.0 * mass + h * tau) / q,
            mass / q)


def target_map(lambda_bar, b_bar, tau, h=H_TOKEN):
    """The native-matched target f_t = F_tau s_t + G_tau x_t."""
    k = (tau / h).astype(lambda_bar.dtype)
    return 1.0 + k * (lambda_bar - 1.0), k[..., None] * b_bar


def state_coefficients(lambda_bar, b_bar, tau, mass, h=H_TOKEN):
    """((A0, A1, A2), (C0, C1, C2)) for the native-matched target."""
    F, G = target_map(lambda_bar, b_bar, tau, h)
    a, b, c0, c1, c2 = (value.astype(lambda_bar.dtype) for value in
                        scalar_coefficients(tau, mass, h))
    return ((a + c0 * F, -b + c1 * F, c2 * F),
            (c0[..., None] * G, c1[..., None] * G, c2[..., None] * G))


def mass_from_eps(tau, eps):
    """M = eps * tau^2."""
    return eps * tau * tau


def passive_factors(tau, eps):
    """t_pm = (tau/2)(1 +- sqrt(1 - 4 eps)), real iff eps <= 1/4.

    CONTINUOUS identity only: P(D) = (1 + t_+ D)(1 + t_- D). It does NOT
    factor the mixed-stencil update below, because the two discrete factors
    would have to use the same discrete derivative, and the state and target
    stencils here differ. Never used as an implementation shortcut.
    """
    root = np.sqrt(np.maximum(1.0 - 4.0 * eps, 0.0))
    half = tau / 2.0
    return half * (1.0 + root), half * (1.0 - root)


# ------------------------------------------------------------------ scan ---
def _shift(values, distance):
    """Shift forward in time by `distance`, with ZERO prehistory."""
    zeros = np.zeros_like(values[:distance])
    return np.concatenate((zeros, values[:-distance]), axis=0)


def input_drive(coefficients_C, input_sequence):
    """d_t = C0 x_t + C1 x_{t-1} + C2 x_{t-2}, zero input prehistory."""
    C0, C1, C2 = coefficients_C
    u = jax.vmap(lambda x: C0 @ x)(input_sequence)
    u1 = jax.vmap(lambda x: C1 @ x)(input_sequence)
    u2 = jax.vmap(lambda x: C2 @ x)(input_sequence)
    return u + _shift(u1, 1) + _shift(u2, 2)


def companion_matrix(A0, A1, A2):
    """H = [[A0, A1, A2], [1, 0, 0], [0, 1, 0]] per mode, shape (P, 3, 3)."""
    one = np.ones_like(A0)
    zero = np.zeros_like(A0)
    return np.stack((np.stack((A0, A1, A2), axis=-1),
                     np.stack((one, zero, zero), axis=-1),
                     np.stack((zero, one, zero), axis=-1)), axis=-2)


def _level(operator, state, distance):
    """One Hillis-Steele doubling step of a CONSTANT-transition recurrence.

    v^(k)_t = sum_{j=t-2^k+1}^{t} H^{t-j} d_j, so
    v^(k+1)_t = v^(k)_t + H^{2^k} v^(k)_{t-2^k}, and the operator squares.
    """
    return state + np.einsum("pij,lpj->lpi", operator, _shift(state, distance))


def wwj_scan(A0, A1, A2, drive, remat=True):
    """Parallel prefix scan of s_{t+1} = A0 s_t + A1 s_{t-1} + A2 s_{t-2} + d_t.

    Zero state prehistory (s_0 = s_{-1} = s_{-2} = 0); returns s_{t+1} for
    t = 0..L-1, i.e. the state that has already seen x_t. Depth
    ceil(log2 L), with no per-token transition matrix and no sequential loop.

    `remat` selects the rematerialization region, which changes BACKWARD
    memory only, never the result:

      True / "level"  each doubling level is rematerialized; the level
                      boundaries are still retained for the backward pass
                      (see the module docstring: O(L*P*log L) residuals);
      "whole"         the entire scan is rematerialized: O(L*P*3) residuals,
                      at roughly twice the forward flops;
      False           no rematerialization; the largest backward footprint.
    """
    if remat == "whole":
        return jax.checkpoint(
            lambda a0, a1, a2, d: wwj_scan(a0, a1, a2, d, remat=False))(
                A0, A1, A2, drive)
    length = drive.shape[0]
    zero = np.zeros_like(drive)
    state = np.stack((drive, zero, zero), axis=-1)
    operator = companion_matrix(A0, A1, A2)
    distance = 1
    while distance < length:
        level = (jax.checkpoint(lambda o, v, d=distance: _level(o, v, d))
                 if remat else
                 (lambda o, v, d=distance: _level(o, v, d)))
        state = level(operator, state)
        operator = operator @ operator
        distance *= 2
    return state[..., 0]


def wwj_states(lambda_bar, b_bar, tau, mass, input_sequence, reverse=False,
               h=H_TOKEN, remat=True):
    """States of one causal direction. `reverse` reverses IN and OUT.

    The reverse branch of a bidirectional layer must see the reversed input
    sequence and be flipped back afterwards, so its prehistory is the end of
    the sequence. No forward-history shift is ever applied to a concatenated
    bidirectional state.
    """
    sequence = input_sequence[::-1] if reverse else input_sequence
    (A0, A1, A2), C = state_coefficients(lambda_bar, b_bar, tau, mass, h)
    states = wwj_scan(A0, A1, A2, input_drive(C, sequence), remat=remat)
    return states[::-1] if reverse else states


# ----------------------------------------------------------- diagnostics ---
def companion_spectral_radius(A0, A1, A2, squarings=10):
    """Spectral radius of the companion matrix, per mode. DIAGNOSTIC ONLY.

    Computed as ||H^n||_F^(1/n) with n = 2^squarings, evaluated by repeated
    squaring with the magnitude carried in the log, so nothing overflows for
    unstable modes. The Frobenius norm is submultiplicative, so the estimate
    NEVER UNDERSTATES the radius, and it converges to it geometrically; with
    n = 1024 the overshoot on a 3x3 is below a fraction of a percent, which
    `tests/test_wwj_recurrence.py` checks against exactly known roots.

    No eigendecomposition is used: `jnp.linalg.eigvals` is unavailable on the
    GPU backend, and eigenvalues are in any case never used to evaluate or
    differentiate the recurrence — the scan above is.
    """
    matrix = companion_matrix(A0, A1, A2)
    norm = np.sqrt(np.sum(np.abs(matrix) ** 2, axis=(-2, -1)))
    log_scale = np.log(norm)
    matrix = matrix / norm[..., None, None]
    for _ in range(squarings):
        matrix = matrix @ matrix
        norm = np.sqrt(np.sum(np.abs(matrix) ** 2, axis=(-2, -1)))
        matrix = matrix / norm[..., None, None]
        log_scale = 2.0 * log_scale + np.log(norm)
    return np.exp(log_scale / float(2 ** squarings))


def continuous_response(lambda_continuous, tau, mass):
    """P(lambda) = 1 + tau*lambda + M*lambda^2, per continuous mode.

    P(lambda) = 0 is NOT a target: exact cancellation would delete that
    mode's memory. The hypothesis is partial temporal compensation with
    long-delay information preserved.
    """
    return 1.0 + tau * lambda_continuous + mass * lambda_continuous ** 2


def discrete_response(tau, mass, A0, A1, A2, z, h=H_TOKEN):
    """Transfer function of the IMPLEMENTED mixed-stencil recurrence.

    From s_{t+1} = A0 s_t + A1 s_{t-1} + A2 s_{t-2} + (c0 + c1 z^-1 +
    c2 z^-2) u_t, the state response to the mode drive u is

        S(z)/U(z) = (c0 + c1 z^-1 + c2 z^-2)
                    / (z - A0 - A1 z^-1 - A2 z^-2).

    This is a DISCRETE-TIME diagnostic of the realization, deliberately
    distinct from the continuous P(lambda) above.
    """
    _, _, c0, c1, c2 = scalar_coefficients(tau, mass, h)
    numerator = c0 + c1 / z + c2 / z ** 2
    denominator = z - A0 - A1 / z - A2 / z ** 2
    return numerator / denominator
