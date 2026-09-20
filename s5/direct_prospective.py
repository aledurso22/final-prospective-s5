"""Direct prospective S5 recurrences: coefficients, scan, and diagnostics.

THREE DIFFERENT EQUATIONS, kept apart on purpose.

  MATCHED WWJ            (1 + tau D + M D^2)(s - f) = 0, i.e.
                         M s'' + tau s' + s = f + tau f' + M f''
                         The M f'' term is what makes it matched.

  PARTIALLY MATCHED      M s'' + Gamma s' + s = f + T f',  Gamma = gamma + T
  (two-compartment)      No M f''. Transfer function (1 + T p)/(1 + Gamma p +
                         M p^2): a first-order numerator cannot cancel two
                         second-order poles, so this is "prospective
                         correction plus retained inertial memory".

  ORDINARY (Zucchet)     (1 + tau D)(s - f) = 0, the M = 0, Gamma = T = tau
                         boundary of both of the above.

TWO INDEPENDENT AXES, never conflated. The MODEL says which equation; the
TARGET CONSTRUCTION says how f is built. "The WWJ model on the professor
linear target" is not "the Professor model".

  models                 PROFESSOR_TSS        M = 0, two-state recurrence
                         WWJ_GENERALIZED_TSS  M > 0, three-state recurrence

  target constructions   PROFESSOR_LINEAR_TARGET  f_t = Abar s_t + Bbar x_t
                         NATIVE_MATCHED_TARGET    f_t = F_tau s_t + G_tau x_t,
                           F_tau = I + (tau/h)(Abar - I), G_tau = (tau/h) Bbar

CAUSAL DISCRETIZATION (state forward, target backward), h = 1 TOKEN, never
S5's learned Delta:

  s'  ~ (s_{t+1} - s_t)/h        f'  ~ (f_t - f_{t-1})/h
  s'' ~ (s_{t+1} - 2 s_t + s_{t-1})/h^2
                                 f'' ~ (f_t - 2 f_{t-1} + f_{t-2})/h^2

This is a consistent causal approximation of the continuous equation. It is
NOT the exact finite-step identity P(D_h)(s - f) = 0, because the state and
target stencils differ; the exact common-stencil operator is
`common_stencil_residual` below, and what it does to a driven S5 layer is the
subject of `docs/S5_DIRECT_GENERALIZED_PROSPECTIVE.md`.

STABILITY IS THE OPEN QUESTION, NOT AN ASSUMPTION. See that document: over
the S5 mode disc every one of these causal recurrences has companion radius
>= 1, and z = 1 is an exact root whenever Abar has a mode at 1. Nothing here
"stabilizes" a recurrence by altering its coefficients.
"""

import jax
import jax.numpy as np

H_TOKEN = 1.0
TAU_MIN = 1e-3
EPS_MAX = 0.25

#: the two MODELS (which equation), distinct from the target construction
PROFESSOR_TSS = "professor_tss"                 # M = 0,  two-state
WWJ_GENERALIZED_TSS = "wwj_generalized_tss"     # M > 0,  three-state
MODELS = (PROFESSOR_TSS, WWJ_GENERALIZED_TSS)

#: the two TARGET CONSTRUCTIONS (how f is built), distinct from the model
PROFESSOR_LINEAR_TARGET = "professor_linear_target"
NATIVE_MATCHED_TARGET = "native_matched_target"
TARGET_CONSTRUCTIONS = (PROFESSOR_LINEAR_TARGET, NATIVE_MATCHED_TARGET)


def model_of(mass):
    """Which MODEL a mass value selects. M = 0 is Professor/TSS, not WWJ."""
    return PROFESSOR_TSS if float(np.max(np.abs(np.asarray(mass)))) == 0.0 \
        else WWJ_GENERALIZED_TSS


# ------------------------------------------------------------ coefficients -
def matched_scalar_coefficients(tau, mass, h=H_TOKEN):
    """(a, b, c0, c1, c2) of the MATCHED equation (with M f'').

    q = M + h tau
    a  = (2M + h tau - h^2)/q      b  = M/q
    c0 = (M + h tau + h^2)/q       c1 = -(2M + h tau)/q      c2 = M/q
    """
    q = mass + h * tau
    return ((2.0 * mass + h * tau - h * h) / q,
            mass / q,
            (mass + h * tau + h * h) / q,
            -(2.0 * mass + h * tau) / q,
            mass / q)


def partially_matched_scalar_coefficients(response, mass, gamma, h=H_TOKEN):
    """(a, b, c0, c1) of the PARTIALLY MATCHED equation (no M f'').

    Gamma = gamma + T,  q = M + h Gamma
    a  = (2M + h Gamma - h^2)/q    b  = M/q
    c0 = h(h + T)/q                c1 = -h T/q
    """
    big_gamma = gamma + response
    q = mass + h * big_gamma
    return ((2.0 * mass + h * big_gamma - h * h) / q,
            mass / q,
            h * (h + response) / q,
            -h * response / q)


def target_maps(lambda_bar, b_bar, tau,
                target_construction=PROFESSOR_LINEAR_TARGET, h=H_TOKEN):
    """(F, G) of the chosen TARGET CONSTRUCTION (not the model)."""
    if target_construction == PROFESSOR_LINEAR_TARGET:
        return lambda_bar, b_bar
    if target_construction == NATIVE_MATCHED_TARGET:
        k = (tau / h).astype(lambda_bar.dtype)
        return 1.0 + k * (lambda_bar - 1.0), k[..., None] * b_bar
    raise ValueError(f"unknown target construction: {target_construction!r}; "
                     f"expected one of {TARGET_CONSTRUCTIONS}")


def matched_state_coefficients(lambda_bar, b_bar, tau, mass,
                               target_construction=PROFESSOR_LINEAR_TARGET,
                               h=H_TOKEN):
    """((A0, A1, A2), (C0, C1, C2)) of the WWJ_GENERALIZED_TSS model."""
    F, G = target_maps(lambda_bar, b_bar, tau, target_construction, h)
    a, b, c0, c1, c2 = (value.astype(lambda_bar.dtype) for value in
                        matched_scalar_coefficients(tau, mass, h))
    return ((a + c0 * F, -b + c1 * F, c2 * F),
            (c0[..., None] * G, c1[..., None] * G, c2[..., None] * G))


def partially_matched_state_coefficients(lambda_bar, b_bar, response, mass,
                                         gamma,
                                         target_construction=PROFESSOR_LINEAR_TARGET,
                                         h=H_TOKEN):
    """((A1t, A2t), (C0, C1)) for the PARTIALLY MATCHED equation."""
    F, G = target_maps(lambda_bar, b_bar, response, target_construction, h)
    a, b, c0, c1 = (value.astype(lambda_bar.dtype) for value in
                    partially_matched_scalar_coefficients(response, mass,
                                                          gamma, h))
    return ((a + c0 * F, -b + c1 * F),
            (c0[..., None] * G, c1[..., None] * G))


def mass_from_eps(scale, eps):
    """M = eps * scale^2, with `scale` = tau (matched) or Gamma (partial)."""
    return eps * scale * scale


# --------------------------------------------- the exact common stencil ----
def common_stencil_residual_coefficients(tau, mass, h=H_TOKEN):
    """(1+k+m, -(k+2m), m) of the EXACT operator P(D_h) with k=tau/h,
    m=M/h^2, applied to the residual r = s - f.

    P(D_h) r = 0 with zero residual history keeps r identically zero, so the
    exactly matched model gives s = f. With f = Abar s + Bbar x that is the
    algebraic relation (I - Abar) s = Bbar x: the driven response becomes
    memoryless. That is a result, not a warning, and it is what
    `tests/test_direct_prospective_algebra.py` demonstrates.
    """
    k, m = tau / h, mass / (h * h)
    return 1.0 + k + m, -(k + 2.0 * m), m


def common_stencil_collapsed_state(lambda_bar, b_bar, input_sequence):
    """s_t = (I - Abar)^{-1} Bbar x_t, the exact-matching consequence."""
    gain = 1.0 / (1.0 - lambda_bar)
    return jax.vmap(lambda x: gain * (b_bar @ x))(input_sequence)


# ------------------------------------------------------------------ scan ---
def _shift(values, distance):
    """Shift forward in time by `distance`, ZERO prehistory."""
    zeros = np.zeros_like(values[:distance])
    return np.concatenate((zeros, values[:-distance]), axis=0)


def input_drive(coefficients_C, input_sequence):
    """d_t = sum_i C_i x_{t-i}, zero input prehistory.

    The alignment is deliberate and tested: the state emitted at token t has
    seen x_t through C_0, exactly as Native S5's does.
    """
    drives = [jax.vmap(lambda x, C=C: C @ x)(input_sequence)
              for C in coefficients_C]
    total = drives[0]
    for lag, drive in enumerate(drives[1:], start=1):
        total = total + _shift(drive, lag)
    return total


def companion_matrix(coefficients_A):
    """Companion of s_t = sum_i A_i s_{t-i}, shape (P, n, n)."""
    order = len(coefficients_A)
    zero = np.zeros_like(coefficients_A[0])
    one = np.ones_like(coefficients_A[0])
    rows = [np.stack(coefficients_A, axis=-1)]
    for index in range(order - 1):
        row = [one if column == index else zero for column in range(order)]
        rows.append(np.stack(row, axis=-1))
    return np.stack(rows, axis=-2)


def _level(operator, state, distance):
    """v <- v + H^{2^k} shift(v, 2^k) for a CONSTANT transition."""
    return state + np.einsum("pij,lpj->lpi", operator, _shift(state, distance))


def companion_doubling_scan(coefficients_A, drive, remat="level"):
    """Constant-operator doubling scan, O(log L) depth, no token loop.

    Because every A_i is diagonal in the S5 mode basis and constant in time,
    the companion H is one small matrix PER MODE PER LEVEL: no L x P x n x n
    tensor is ever materialized, and the only O(L) array is the (L, P, n)
    lifted state.

    BACKWARD MEMORY, stated honestly: with remat="level" each level
    recomputes its internals but reverse mode still retains each level
    BOUNDARY, so residuals are about ceil(log2 L) arrays of size L*P*n --
    O(L*P*log L), not O(L*P). remat="whole" retains only the inputs and
    recomputes every level, O(L*P*n) residuals at roughly twice the forward
    flops. remat=None retains everything. Measured peak device memory is the
    authority; the stability grid reports it.
    """
    if remat == "whole":
        return jax.checkpoint(
            lambda A, d: companion_doubling_scan(A, d, remat=None))(
                tuple(coefficients_A), drive)
    order = len(coefficients_A)
    length = drive.shape[0]
    zero = np.zeros_like(drive)
    state = np.stack([drive] + [zero] * (order - 1), axis=-1)
    operator = companion_matrix(list(coefficients_A))
    distance = 1
    while distance < length:
        step = (lambda o, v, d=distance: _level(o, v, d))
        state = jax.checkpoint(step)(operator, state) if remat == "level" \
            else step(operator, state)
        operator = operator @ operator
        distance *= 2
    return state[..., 0]


def matched_states(lambda_bar, b_bar, input_sequence, tau, mass,
                   target_construction=PROFESSOR_LINEAR_TARGET, reverse=False,
                   h=H_TOKEN, remat="level"):
    """States of the WWJ_GENERALIZED_TSS causal recurrence, one direction.

    M = 0 makes this the PROFESSOR_TSS model, and `professor_tss_states`
    computes the same sequence from the two-state recurrence directly.
    """
    sequence = input_sequence[::-1] if reverse else input_sequence
    A, C = matched_state_coefficients(lambda_bar, b_bar, tau, mass,
                                      target_construction, h)
    states = companion_doubling_scan(A, input_drive(C, sequence), remat)
    return states[::-1] if reverse else states


def partially_matched_states(lambda_bar, b_bar, input_sequence, response,
                             mass, gamma,
                             target_construction=PROFESSOR_LINEAR_TARGET,
                             reverse=False, h=H_TOKEN, remat="level"):
    """States of the PARTIALLY MATCHED (two-compartment) recurrence."""
    sequence = input_sequence[::-1] if reverse else input_sequence
    A, C = partially_matched_state_coefficients(lambda_bar, b_bar, response,
                                                mass, gamma,
                                                target_construction, h)
    states = companion_doubling_scan(A, input_drive(C, sequence), remat)
    return states[::-1] if reverse else states


def professor_tss_scalar_coefficients(tau, h=H_TOKEN):
    """(a, c0, c1) of the PROFESSOR_TSS model, M = 0, written out directly.

        s_{t+1} = (1 - h/tau) s_t + (1 + h/tau) f_t - f_{t-1}

    It is the M = 0 boundary of the generalized coefficients, and the tests
    prove the two agree rather than assuming it.
    """
    return 1.0 - h / tau, 1.0 + h / tau, -np.ones_like(tau)


def professor_tss_state_coefficients(lambda_bar, b_bar, tau,
                                     target_construction=PROFESSOR_LINEAR_TARGET,
                                     h=H_TOKEN):
    """((A0, A1), (C0, C1)) of the TWO-STATE Professor/TSS recurrence."""
    F, G = target_maps(lambda_bar, b_bar, tau, target_construction, h)
    a, c0, c1 = (value.astype(lambda_bar.dtype) for value in
                 professor_tss_scalar_coefficients(tau, h))
    return ((a + c0 * F, c1 * F), (c0[..., None] * G, c1[..., None] * G))


def professor_tss_states(lambda_bar, b_bar, input_sequence, tau,
                         target_construction=PROFESSOR_LINEAR_TARGET,
                         reverse=False, h=H_TOKEN, remat="level"):
    """States of the PROFESSOR_TSS (M = 0) two-state recurrence."""
    sequence = input_sequence[::-1] if reverse else input_sequence
    A, C = professor_tss_state_coefficients(lambda_bar, b_bar, tau,
                                            target_construction, h)
    states = companion_doubling_scan(A, input_drive(C, sequence), remat)
    return states[::-1] if reverse else states


# ----------------------------------------------------------- diagnostics ---
def companion_spectral_radius(coefficients_A, squarings=10):
    """||H^n||_F^(1/n) by repeated squaring, magnitude carried in the log.

    Submultiplicativity means this NEVER understates the radius, it cannot
    overflow for unstable modes, and it avoids `eigvals`, which the GPU
    backend does not provide. The stability grid cross-checks it against
    exact CPU roots.
    """
    matrix = companion_matrix(list(coefficients_A))
    norm = np.sqrt(np.sum(np.abs(matrix) ** 2, axis=(-2, -1)))
    log_scale = np.log(norm)
    matrix = matrix / norm[..., None, None]
    for _ in range(squarings):
        matrix = matrix @ matrix
        norm = np.sqrt(np.sum(np.abs(matrix) ** 2, axis=(-2, -1)))
        matrix = matrix / norm[..., None, None]
        log_scale = 2.0 * log_scale + np.log(norm)
    return np.exp(log_scale / float(2 ** squarings))


def continuous_transfer(p, tau, mass, response=None, gamma=None):
    """S(p)/F(p) for whichever equation the arguments describe.

    matched:            (1 + tau p + M p^2)/(1 + tau p + M p^2) = 1
    partially matched:  (1 + T p)/(1 + Gamma p + M p^2)
    """
    if response is None:
        numerator = 1.0 + tau * p + mass * p * p
        denominator = numerator
    else:
        big_gamma = gamma + response
        numerator = 1.0 + response * p
        denominator = 1.0 + big_gamma * p + mass * p * p
    return numerator / denominator
