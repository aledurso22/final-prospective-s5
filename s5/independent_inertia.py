"""Research-only: the generalized prospective recurrence with INDEPENDENT mass
and damping, including exactly zero ordinary damping at strictly positive mass.

NOTHING HERE CHANGES THE COMPLETED EXPERIMENT. `generalized_coefficients`,
`response_mass_gamma`, `GeneralizedProspectiveS5SSM` and the runner defaults
are untouched. This module reuses `generalized_coefficients` verbatim -- that
function already accepts `mass` and `gamma` as independent arrays; only the
trained parameterization M = rho*gamma*T ties them. So on the constrained
family the recurrence here is the production recurrence by construction, and
`tests/test_independent_inertia.py` checks it bit for bit.

THE DISCRETE OPERATOR (h = 1 token, as at generalized_prospective_ssm.py:52).
With kappa = (T/h)(1 - lambda_bar) and G = (T/h) b_bar, the production
recurrence s_t = a1 s_{t-1} + a2 s_{t-2} + c1 x_t + c2 x_{t-1} is exactly

    (M + T) D^2 s_t + gamma D s_t + kappa (1 + T D) s_{t-1} = G (1 + T D) x_t,

D = 1 - z^-1 the backward difference. Equivalently the characteristic
polynomial, scaled by q = M + h(gamma + T), is

    q p(z) = (M + T)(z - 1)^2 + gamma z (z - 1) + kappa ((1 + T) z - T),

and the transfer function is H(z) = z G ((1 + T) z - T) / (q p(z)).

THE CONTINUOUS LAW IT DISCRETIZES (token time) is

    M s'' + (gamma + T kappa) s' + kappa s = G (x + T x'),

so the discrete operator carries an extra mass T that the continuous law does
not: T D Y_t is formed from s_{t-1}, and cancels the (gamma + T) D s_t term only
up to one step, leaving T D^2 s_t.
"""

from functools import partial

import jax
import jax.numpy as np
import numpy as onp

from .discrete_recurrence import generalized_coefficients
from .factored_recurrence import DEFAULT_IMPLEMENTATION, scan_for
from .ssm import S5SSM, discretize_zoh

H_TOKEN = 1.0


# ------------------------------------------------------- parameterization ---
def independent_mass_damping(T_raw, M_raw, gamma_raw=None):
    """(T, M, gamma) with M and gamma INDEPENDENT.

    gamma_raw=None means ordinary damping is EXACTLY zero -- a literal 0.0,
    not a softplus of a very negative number -- while M = softplus(M_raw) > 0.
    """
    T = jax.nn.softplus(T_raw)
    M = jax.nn.softplus(M_raw)
    gamma = (np.zeros_like(T) if gamma_raw is None
             else jax.nn.softplus(gamma_raw))
    return T, M, gamma


def _inv_softplus(value):
    return onp.log(onp.expm1(value)).astype(onp.float32)


class IndependentInertiaS5SSM(S5SSM):
    """The production recurrence with M and gamma as separate parameters.

    zero_damping=True removes the gamma parameter entirely: gamma is the
    constant 0.0 and cannot learn its way back. Nothing else differs from
    GeneralizedProspectiveS5SSM -- same target_map, same coefficients, same
    readout, same scans.
    """
    response_init: float = 0.05
    mass_init: float = 0.025
    gamma_init: float = 1.0
    zero_damping: bool = False
    implementation: str = DEFAULT_IMPLEMENTATION

    def setup(self):
        super().setup()
        full = lambda value: (lambda rng, shape: np.full(shape, value))
        self.inertia_T_raw = self.param(
            "inertia_T_raw", full(_inv_softplus(self.response_init)), (self.P,))
        self.inertia_M_raw = self.param(
            "inertia_M_raw", full(_inv_softplus(self.mass_init)), (self.P,))
        gamma_raw = None
        if not self.zero_damping:
            gamma_raw = self.param(
                "inertia_gamma_raw", full(_inv_softplus(self.gamma_init)),
                (self.P,))
        T, M, gamma = independent_mass_damping(
            self.inertia_T_raw, self.inertia_M_raw, gamma_raw)
        b_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        lambda_bar, b_bar = discretize_zoh(self.Lambda, b_tilde, step)
        self.inertia_a1, self.inertia_a2, self.inertia_c1, self.inertia_c2 = (
            generalized_coefficients(lambda_bar, b_bar, T, M, gamma))

    def __call__(self, input_sequence):
        scan = scan_for(self.implementation)
        coeffs = (self.inertia_a1, self.inertia_a2,
                  self.inertia_c1, self.inertia_c2)
        states = scan(*coeffs, input_sequence)
        if self.bidirectional:
            states = np.concatenate(
                (states, scan(*coeffs, input_sequence, reverse=True)), axis=-1)
        ys = jax.vmap(lambda state: 2 * (self.C_tilde @ state).real)(states)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


def init_independent_inertia_S5SSM(response_init=0.05, mass_init=0.025,
                                   gamma_init=1.0, zero_damping=False,
                                   implementation=DEFAULT_IMPLEMENTATION,
                                   **s5_kwargs):
    return partial(IndependentInertiaS5SSM, response_init=response_init,
                   mass_init=mass_init, gamma_init=gamma_init,
                   zero_damping=zero_damping, implementation=implementation,
                   **s5_kwargs)


# ------------------------------------------------ analysis (numpy, float64) ---
# Everything below is per complex mode, so the coefficients are COMPLEX and
# real-coefficient Jury conditions do not apply. Discrete stability uses the
# Schur-Cohn test for complex polynomials; continuous stability uses the
# complex-coefficient Routh-Hurwitz condition. Both are cross-checked against
# explicit roots in the tests.

def stiffness(lambda_bar, T, h=H_TOKEN):
    """kappa = (T/h)(1 - lambda_bar): the restoring coefficient of r on s."""
    return (onp.asarray(T) / h) * (1.0 - onp.asarray(lambda_bar))


def operator_polynomial(lambda_bar, T, M, gamma, h=H_TOKEN):
    """Coefficients (z^2, z^1, z^0) of q p(z), from the closed form.

    q p(z) = (M+T)(z-1)^2 + gamma z(z-1) + kappa((1+T)z - T)   [h = 1]
    Written for general h so it can be checked against generalized_coefficients.
    """
    lam = onp.asarray(lambda_bar, dtype=onp.complex128)
    T, M, gamma = (onp.asarray(v, dtype=onp.float64) for v in (T, M, gamma))
    k = T / h
    F = 1.0 + k * (lam - 1.0)
    # q y_t = (2M + h(g+T) - h^2) y_{t-1} - M y_{t-2} + (h^2+hT) u_t - hT u_{t-1}
    # with u_t = F s_{t-1} + G x_t
    c2 = (M + h * (gamma + T)) * onp.ones_like(lam)
    c1 = -(2 * M + h * (gamma + T) - h * h + (h * h + h * T) * F)
    c0 = M + h * T * F
    return c2, c1, c0


def discrete_roots(lambda_bar, T, M, gamma, h=H_TOKEN):
    """Both roots of p(z), per mode, float64, by the stable product formula."""
    c2, c1, c0 = operator_polynomial(lambda_bar, T, M, gamma, h)
    a1, a2 = -c1 / c2, -c0 / c2
    root = onp.sqrt(a1 * a1 + 4.0 * a2)
    sign = onp.where((onp.conj(a1) * root).real >= 0, 1.0, -1.0)
    major = (a1 + sign * root) / 2.0
    minor = onp.where(onp.abs(major) > 0, -a2 / onp.where(
        onp.abs(major) > 0, major, 1.0), 0.0)
    return major, minor


def schur_cohn_stable(lambda_bar, T, M, gamma, h=H_TOKEN):
    """Both roots strictly inside the unit disc, for COMPLEX coefficients.

    Monic z^2 + b1 z + b0 is Schur stable iff |b0| < 1 and
    |b1 - conj(b1) b0| < 1 - |b0|^2.
    """
    c2, c1, c0 = operator_polynomial(lambda_bar, T, M, gamma, h)
    b1, b0 = c1 / c2, c0 / c2
    return (onp.abs(b0) < 1.0) & (
        onp.abs(b1 - onp.conj(b1) * b0) < 1.0 - onp.abs(b0) ** 2)


def continuous_polynomial(lambda_bar, T, M, gamma, h=H_TOKEN):
    """(sigma^2, sigma, 1) coefficients of M s^2 + (gamma + T kappa) s + kappa."""
    kap = stiffness(lambda_bar, T, h)
    M = onp.asarray(M, dtype=onp.float64)
    return M * onp.ones_like(kap), onp.asarray(gamma) + onp.asarray(T) * kap, kap


def hurwitz_stable_complex(a, b):
    """sigma^2 + a sigma + b has both roots in Re < 0, for COMPLEX a, b.

    Routh-Hurwitz for complex coefficients: Re a > 0 and
    Re a (Re a Re b + Im a Im b) > (Im b)^2.
    """
    a, b = onp.asarray(a), onp.asarray(b)
    return (a.real > 0) & (a.real * (a.real * b.real + a.imag * b.imag)
                           > b.imag ** 2)


def continuous_stable(lambda_bar, T, M, gamma, h=H_TOKEN):
    m2, m1, m0 = continuous_polynomial(lambda_bar, T, M, gamma, h)
    return hurwitz_stable_complex(m1 / m2, m0 / m2)


def zero_drag_critical_mass(lambda_bar, T, h=H_TOKEN):
    """Largest M for which the CONTINUOUS gamma = 0 law is stable.

    At gamma = 0 the Hurwitz condition reduces to
        M < T^2 Re(kappa) |kappa|^2 / Im(kappa)^2,
    infinite for a real (non-oscillatory) mode.
    """
    kap = stiffness(lambda_bar, T, h)
    T = onp.asarray(T, dtype=onp.float64)
    with onp.errstate(divide="ignore"):
        return onp.where(kap.imag == 0, onp.inf,
                         T * T * kap.real * onp.abs(kap) ** 2 / kap.imag ** 2)


def transfer(lambda_bar, T, M, gamma, z, h=H_TOKEN):
    """H(z)/G: input-to-state transfer per unit G = (T/h) b_bar.

    H(z) = z ((1+T)z - T) h^2/q  /  p(z)   (exact; q cancels between c and a)
    returned divided by G so it is independent of B.
    """
    c2, c1, c0 = operator_polynomial(lambda_bar, T, M, gamma, h)
    z = onp.asarray(z, dtype=onp.complex128)
    T = onp.asarray(T, dtype=onp.float64)
    num = z * ((h * h + h * T) * z - h * T)
    return num / (c2 * z * z + c1 * z + c0)


def native_transfer(lambda_bar, z):
    """Native H(z)/b_bar = z / (z - lambda_bar)."""
    return z / (z - lambda_bar)


def transfer_zero(T, h=H_TOKEN):
    """The single finite nonzero zero, z0 = T/(h + T), independent of M, gamma."""
    return onp.asarray(T) / (h + onp.asarray(T))


def cancellation_residual(lambda_bar, T, M, gamma, h=H_TOKEN):
    """q p(z0) / scale: zero iff the pole at z0 cancels the zero.

    Closed form at h = 1: (M + T - gamma T) / (1 + T)^2, so cancellation
    happens exactly on M = T (gamma - 1), independent of lambda_bar.
    """
    c2, c1, c0 = operator_polynomial(lambda_bar, T, M, gamma, h)
    z0 = transfer_zero(T, h)
    return c2 * z0 * z0 + c1 * z0 + c0


def low_frequency_delay(lambda_bar, T, M, gamma, h=H_TOKEN):
    """Group delay of H at omega -> 0, in tokens: Re(gamma/kappa) - 1 (h=1).

    Derived from H(e^s) = (G/kappa)(1 + (1 - gamma/kappa) s + O(s^2)). It does
    not depend on M. Native S5 is Re(1/(1 - lambda_bar)) - 1.
    """
    kap = stiffness(lambda_bar, T, h)
    return (onp.asarray(gamma) / kap).real * h - h


def recurrence_impulse(a1, a2, c1, c2, n, x=None):
    """Sequential float64 oracle for scalar per-mode coefficients."""
    a1, a2, c1, c2 = (onp.asarray(v, dtype=onp.complex128)
                      for v in (a1, a2, c1, c2))
    if x is None:
        x = onp.zeros(n)
        x[0] = 1.0
    x = onp.asarray(x, dtype=onp.float64)
    s = onp.zeros((n,) + a1.shape, dtype=onp.complex128)
    s1 = s2 = onp.zeros_like(a1)
    xp = 0.0
    for t in range(n):
        st = a1 * s1 + a2 * s2 + c1 * x[t] + c2 * xp
        s[t], s2, s1, xp = st, s1, st, x[t]
    return s


def coefficients_numpy(lambda_bar, T, M, gamma, h=H_TOKEN):
    """(a1, a2, c1/G-free, c2/G-free) in float64 numpy, b_bar = 1 per mode."""
    lam = onp.asarray(lambda_bar, dtype=onp.complex128)
    T, M, gamma = (onp.broadcast_to(onp.asarray(v, dtype=onp.float64),
                                    lam.shape) for v in (T, M, gamma))
    k = T / h
    F, G = 1.0 + k * (lam - 1.0), k
    q = M + h * (gamma + T)
    alpha, beta, delta = M / q, h * h / q, h * T / q
    return ((1.0 + alpha - beta) + (beta + delta) * F, -alpha - delta * F,
            (beta + delta) * G, -delta * G)


# ------------------------------------------- ALTERNATIVE SCHEME, SEPARATE ---
# Not the production discretization. Reported separately because it changes
# the integration scheme. It realizes the SAME continuous token-time law
#
#     M s'' + (gamma + T kappa) s' + kappa s = G (x + T x'),   M > 0,
#
# exactly under S5's own zero-order-hold input convention, through
#     M xi'' + (gamma + T kappa) xi' + kappa xi = G x,    s = xi + T xi',
# so the held input's jumps never have to be differentiated. Its poles are
# exp(sigma) for the continuous roots sigma, so its stability region IS the
# continuous Hurwitz region -- no parasitic mass T, no explicit restoring step.
# The output keeps the production four-coefficient form, so the same scans run.

def exact_zoh_coefficients(lambda_bar, b_bar, T, M, gamma, h=H_TOKEN):
    """(a1, a2, c1, c2) of the exact ZOH realization; requires M > 0."""
    kap = (T / h).astype(lambda_bar.dtype) * (1.0 - lambda_bar)
    M = M.astype(lambda_bar.dtype)
    damp = gamma.astype(lambda_bar.dtype) + T.astype(lambda_bar.dtype) * kap
    zero = np.zeros_like(kap)
    one = np.ones_like(kap)
    # augmented [[A, B], [0, 0]] with A = [[0, 1], [-kap/M, -damp/M]], B = [0, 1/M]
    aug = np.stack([
        np.stack([zero, one, zero], -1),
        np.stack([-kap / M, -damp / M, one / M], -1),
        np.stack([zero, zero, zero], -1)], -2) * h
    E = jax.vmap(lambda m: jax.scipy.linalg.expm(m, max_squarings=64))(aug)
    Phi, Gam = E[:, :2, :2], E[:, :2, 2]
    trace = Phi[:, 0, 0] + Phi[:, 1, 1]
    det = Phi[:, 0, 0] * Phi[:, 1, 1] - Phi[:, 0, 1] * Phi[:, 1, 0]
    out = np.stack([one, T.astype(lambda_bar.dtype)], -1)       # s = xi + T xi'
    first = np.sum(out * Gam, -1)
    shifted = np.einsum("pi,pij,pj->p", out,
                        Phi - trace[:, None, None] * np.eye(2, dtype=Phi.dtype),
                        Gam)
    G = (T / h).astype(lambda_bar.dtype)[:, None] * b_bar
    return trace, -det, first[:, None] * G, shifted[:, None] * G


def continuous_roots(lambda_bar, T, M, gamma, h=H_TOKEN):
    """Roots sigma of M s^2 + (gamma + T kappa) s + kappa, float64, stable form."""
    m2, m1, m0 = continuous_polynomial(lambda_bar, T, M, gamma, h)
    b, c = m1 / m2, m0 / m2
    root = onp.sqrt(b * b - 4 * c)
    sign = onp.where((onp.conj(-b) * root).real >= 0, 1.0, -1.0)
    major = (-b + sign * root) / 2
    return major, c / major
