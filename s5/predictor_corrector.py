"""Generalized S5 in predictor-corrector form: target scale and correction point.

RESEARCH MODULE. Production constructors (`GeneralizedProspectiveS5SSM`),
runner defaults and the completed results are untouched; this module only
re-expresses the SAME recurrence and adds two separately-labelled changes.

Per complex mode, at the token step h = 1 (the production value), with the
native ZOH pair (lambda_bar, B_bar) of `discretize_zoh` kept distinct from the
continuous generator Lambda:

    K        = 1 - lambda_bar                    (discrete, not Lambda)
    r_t(s)   = K s - B_bar x_t                   (affine, complex residual)
    Q        = M + gamma + T
    target   F = 1 - k K,  G = k B_bar           (k: target scale)
    mu = (M + T)/Q,  eta = k/Q,  zeta = k T/Q

    d_t = mu (s_{t-1} - s_{t-2}) - zeta (r_t(s_{t-1}) - r_{t-1}(s_{t-2}))
    y_t = s_{t-1} + d_t                                        (predictor)
    old : s_t = y_t - eta r_t(s_{t-1})     (correction at the previous state)
    new : s_t = y_t - eta r_t(y_t)         (correction at the prediction)

The existing generalized arm is k = T with the OLD corrector (variant A).
The residual difference crosses tokens (r_t at s_{t-1} minus r_{t-1} at
s_{t-2}), exactly as the existing recurrence has it; this module does not
replace it with a same-token difference.

Expanded, with P = 1 - eta K and L = mu - zeta K:

    old: a1 = P + L,      a2 = -L,    c1 = (eta + zeta) B_bar,   c2 = -zeta B_bar
    new: a1 = P (1 + L),  a2 = -P L,  c1 = (eta + zeta P) B_bar, c2 = -zeta P B_bar

so new = ordinary correction + P d_t, old = ordinary correction + d_t.
Initial conditions everywhere: s_{-1} = s_{-2} = 0 and x_{-1} = 0, hence
r_{-1}(s_{-2}) = 0 -- identical to `scan_companion*`'s zero prehistory.

Target scales:
    "legacy"     k = T       (existing).  eta/(1 - mu) = T/gamma.
    "calibrated" k = gamma   (control).   eta/(1 - mu) = 1, i.e. the leading
                                          slow pole matches native to O(K).

The calibrated scale introduces a REDUNDANCY: with k = gamma, eta = 1 - mu
identically, so the three raw parameters (T, rho, gamma) reach only the
two-dimensional coefficient set (mu, zeta) [eta is determined]. The
parameterization and parameter count are deliberately NOT changed here.
"""

from functools import partial

import jax
import jax.numpy as np
import numpy as onp

from .factored_recurrence import DEFAULT_IMPLEMENTATION, scan_for
from .generalized_prospective_ssm import response_mass_gamma
from .ssm import S5SSM, discretize_zoh

TARGET_SCALES = ("legacy", "calibrated")
CORRECTORS = ("old", "prediction")
VARIANTS = {"A": ("legacy", "old"), "B": ("legacy", "prediction"),
            "C": ("calibrated", "old"), "D": ("calibrated", "prediction")}


def target_scale(T, gamma, scale):
    if scale == "legacy":
        return T
    if scale == "calibrated":
        return gamma
    raise ValueError(f"unknown target scale {scale!r}; known {TARGET_SCALES}")


def residual_coefficients(T, M, gamma, k):
    """(mu, eta, zeta) of the residual form, h = 1. Works for jax or numpy."""
    Q = M + gamma + T
    return (M + T) / Q, k / Q, k * T / Q


def pc_coefficients(lambda_bar, b_bar, T, M, gamma, scale="legacy",
                    corrector="old"):
    """(a1, a2, c1, c2) of s_t = a1 s1 + a2 s2 + c1 x_t + c2 x_{t-1}.

    Shapes follow `generalized_coefficients`: lambda_bar (P,), b_bar (P, H),
    T/M/gamma (P,). Pure jax, differentiable; also accepts numpy arrays.
    """
    xp = np if isinstance(lambda_bar, jax.Array) else onp
    k = target_scale(T, gamma, scale)
    mu, eta, zeta = residual_coefficients(T, M, gamma, k)
    K = 1.0 - lambda_bar
    P = 1.0 - eta * K
    L = mu - zeta * K
    if corrector == "old":
        a1, a2, g1, g2 = P + L, -L, eta + zeta, -zeta * xp.ones_like(P)
    elif corrector == "prediction":
        a1, a2, g1, g2 = P * (1.0 + L), -P * L, eta + zeta * P, -zeta * P
    else:
        raise ValueError(f"unknown corrector {corrector!r}; known {CORRECTORS}")
    return a1, a2, g1[..., None] * b_bar, g2[..., None] * b_bar


# --------------------------------------------------------------- module ---

class PredictorCorrectorS5SSM(S5SSM):
    """Generalized S5 with an explicit target scale and correction point.

    Same backbone, the same (T, rho, gamma) parameterization with
    M = rho gamma T (`response_mass_gamma`), the same raw initial values and
    the same bidirectional readout as `GeneralizedProspectiveS5SSM`. With
    scale="legacy", corrector="old" it is that arm (variant A).
    """
    response_init: float = 0.05
    rho_init: float = 0.5
    gamma_init: float = 1.0
    scale: str = "legacy"
    corrector: str = "old"
    implementation: str = DEFAULT_IMPLEMENTATION

    def setup(self):
        super().setup()
        if self.scale not in TARGET_SCALES or self.corrector not in CORRECTORS:
            raise ValueError(f"bad variant {self.scale!r}/{self.corrector!r}")
        t_raw = np.log(np.expm1(self.response_init)).astype(np.float32)
        rho_raw = np.log(self.rho_init / (1.0 - self.rho_init)).astype(np.float32)
        gamma_raw = np.log(np.expm1(self.gamma_init)).astype(np.float32)
        # parameter NAMES match the generalized arm so a matched comparison
        # can share (or transplant) the very same initial values
        self.generalized_T_raw = self.param(
            "generalized_T_raw", lambda rng, shape: np.full(shape, t_raw), (self.P,))
        self.generalized_rho_raw = self.param(
            "generalized_rho_raw", lambda rng, shape: np.full(shape, rho_raw), (self.P,))
        self.generalized_gamma_raw = self.param(
            "generalized_gamma_raw", lambda rng, shape: np.full(shape, gamma_raw), (self.P,))
        T, mass, gamma, _ = response_mass_gamma(
            self.generalized_T_raw, self.generalized_rho_raw,
            self.generalized_gamma_raw)
        b_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        lambda_bar, b_bar = discretize_zoh(self.Lambda, b_tilde, step)
        self.pc_a1, self.pc_a2, self.pc_c1, self.pc_c2 = pc_coefficients(
            lambda_bar, b_bar, T, mass, gamma, self.scale, self.corrector)

    def __call__(self, input_sequence):
        scan = scan_for(self.implementation)
        coeffs = (self.pc_a1, self.pc_a2, self.pc_c1, self.pc_c2)
        states = scan(*coeffs, input_sequence)
        if self.bidirectional:
            states = np.concatenate(
                (states, scan(*coeffs, input_sequence, reverse=True)), axis=-1)
        ys = jax.vmap(lambda state: 2 * (self.C_tilde @ state).real)(states)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


def init_predictor_corrector_S5SSM(variant="A", response_init=0.05,
                                   rho_init=0.5, gamma_init=1.0,
                                   implementation=DEFAULT_IMPLEMENTATION,
                                   **s5_kwargs):
    scale, corrector = VARIANTS[variant]
    return partial(PredictorCorrectorS5SSM, response_init=response_init,
                   rho_init=rho_init, gamma_init=gamma_init, scale=scale,
                   corrector=corrector, implementation=implementation,
                   **s5_kwargs)


# --------------------------------------- direct (unexpanded) evaluation ---

def direct_rollout(lambda_bar, b_bar, T, M, gamma, inputs, scale="legacy",
                   corrector="old", s_init=(0.0, 0.0), x_prev=None):
    """Evaluate the predictor-corrector equations literally, numpy float64.

    lambda_bar (P,), b_bar (P,H), inputs (L,H). Prehistory s_{-1}, s_{-2}
    and x_{-1} may be set; zero by default (the scans' convention).
    """
    k = target_scale(T, gamma, scale)
    mu, eta, zeta = residual_coefficients(T, M, gamma, k)
    K = 1.0 - lambda_bar
    s1 = onp.broadcast_to(onp.asarray(s_init[0], complex), lambda_bar.shape).copy()
    s2 = onp.broadcast_to(onp.asarray(s_init[1], complex), lambda_bar.shape).copy()
    u_prev = b_bar @ (onp.zeros(inputs.shape[1]) if x_prev is None else x_prev)
    r = lambda s, u: K * s - u
    out = []
    for x in inputs:
        u = b_bar @ x
        d = mu * (s1 - s2) - zeta * (r(s1, u) - r(s2, u_prev))
        y = s1 + d
        s = y - eta * (r(s1, u) if corrector == "old" else r(y, u))
        out.append(s)
        s2, s1, u_prev = s1, s, u
    return onp.stack(out)


# ------------------------------------------------- numpy analysis (f64) ---

def scalar_coefficients(lambda_bar, T, M, gamma, scale, corrector):
    """(a1, a2, g1, g2) per mode with B_bar factored out (c = g B_bar)."""
    lb = onp.asarray(lambda_bar, complex)
    T, M, gamma = (onp.broadcast_to(onp.asarray(v, float), lb.shape)
                   for v in (T, M, gamma))
    a1, a2, c1, c2 = pc_coefficients(lb, onp.ones(lb.shape + (1,)), T, M,
                                     gamma, scale, corrector)
    return a1, a2, c1[..., 0], c2[..., 0]


def roots(a1, a2):
    """Both roots of z^2 - a1 z - a2, stable product form; (major, minor)."""
    a1, a2 = onp.asarray(a1, complex), onp.asarray(a2, complex)
    root = onp.sqrt(a1 * a1 + 4.0 * a2)
    sign = onp.where((onp.conj(a1) * root).real >= 0, 1.0, -1.0)
    major = (a1 + sign * root) / 2.0
    safe = onp.where(onp.abs(major) > 0, major, 1.0)
    return major, onp.where(onp.abs(major) > 0, -a2 / safe, 0.0)


def schur_cohn(a1, a2):
    """Both roots of z^2 - a1 z - a2 strictly inside |z|<1 (complex coeffs).

    Monic z^2 + b1 z + b0 is Schur stable iff |b0| < 1 and
    |b1 - conj(b1) b0| < 1 - |b0|^2. Here b1 = -a1, b0 = -a2.
    """
    b1, b0 = -onp.asarray(a1, complex), -onp.asarray(a2, complex)
    return (onp.abs(b0) < 1.0) & (
        onp.abs(b1 - onp.conj(b1) * b0) < 1.0 - onp.abs(b0) ** 2)


def transfer(a1, a2, g1, g2, z):
    """H(z) = (g1 + g2 z^-1) / (1 - a1 z^-1 - a2 z^-2) per unit B_bar."""
    w = 1.0 / z
    return (g1 + g2 * w) / (1.0 - a1 * w - a2 * w * w)


def dc_gain(a1, a2, g1, g2):
    return (g1 + g2) / (1.0 - a1 - a2)


def group_delay_dc(a1, a2, g1, g2):
    """-d arg H(e^{iw})/dw at w=0 (zero-state convention), analytic."""
    return (g2 / (g1 + g2) + (a1 + 2.0 * a2) / (1.0 - a1 - a2)).real


def predicted_delay(lambda_bar, T, M, gamma, scale, corrector):
    """Closed forms: old Re[(1-mu)/(eta K)] - 1, new ... - (1-mu)."""
    k = target_scale(T, gamma, scale)
    mu, eta, _ = residual_coefficients(T, M, gamma, k)
    base = ((1.0 - mu) / (eta * (1.0 - lambda_bar))).real
    return base - 1.0 if corrector == "old" else base - (1.0 - mu)


def residues(a1, a2, g1, g2):
    """Partial fractions H(w) = R1/(1 - z1 w) + R2/(1 - z2 w), w = z^-1.

    Returns (z1, z2, R1, R2) with z1 the slow (larger |.|) pole. The impulse
    response is h_t = R1 z1^t + R2 z2^t. Undefined for exactly repeated roots
    (returned as inf), which the caller must report rather than hide.
    """
    z1, z2 = roots(a1, a2)
    diff = z1 - z2
    with onp.errstate(divide="ignore", invalid="ignore"):
        R1 = (g1 * z1 + g2) / diff
        R2 = -(g1 * z2 + g2) / diff
    return z1, z2, R1, R2


def native_coefficients(lambda_bar):
    """Native S5 as a degenerate (a2 = c2 = 0) second-order recurrence."""
    lb = onp.asarray(lambda_bar, complex)
    return lb, onp.zeros_like(lb), onp.ones_like(lb), onp.zeros_like(lb)


def simulate(a1, a2, g1, g2, x):
    """Scalar recurrence per mode over a scalar input sequence x (n,)."""
    a1, a2, g1, g2 = (onp.asarray(v, complex) for v in (a1, a2, g1, g2))
    s1 = onp.zeros_like(a1); s2 = onp.zeros_like(a1); xp = 0.0
    out = onp.empty((len(x),) + a1.shape, complex)
    for t, xt in enumerate(x):
        s = a1 * s1 + a2 * s2 + g1 * xt + g2 * xp
        out[t] = s
        s2, s1, xp = s1, s, xt
    return out


__all__ = ["TARGET_SCALES", "CORRECTORS", "VARIANTS", "target_scale",
           "residual_coefficients", "pc_coefficients",
           "PredictorCorrectorS5SSM", "init_predictor_corrector_S5SSM",
           "direct_rollout", "scalar_coefficients", "roots", "schur_cohn",
           "transfer", "dc_gain", "group_delay_dc", "predicted_delay",
           "residues", "native_coefficients", "simulate"]
