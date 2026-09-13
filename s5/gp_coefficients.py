"""Pure coefficient builders for generalized prospective dynamics in S5.

No Flax, no module state: everything here is a function of arrays, so the
algebra can be tested against an independent dense reference.

CLOCK MAPPING (absorbed once, never twice)
------------------------------------------
S5 stores complex native poles ``Lambda_i``, learned positive steps
``Delta_i = exp(log_step_i)`` and continuous input rows ``B_tilde_i``. Define

    a_i = Delta_i * Lambda_i          F0 = diag(a_i)
    b_i = Delta_i * B_tilde_i         B0 = (b_i)_i

Native S5 is then exactly the UNIT-interval ZOH of ``sdot = F0 s + B0 x``:

    exp(a_i)            = exp(Lambda_i Delta_i)              = Lambda_bar_i
    phi1(a_i) b_i       = (Lambda_bar_i - 1)/Lambda_i B_tilde_i = B_bar_i

so `absorb_clock` reproduces the stock discretization with no second factor of
Delta. Both `a_i` and `b_i` depend on `log_step_i`, so autodiff carries the
step gradient through both.

GENERALIZED EQUATION (gamma = 1, unit sample clock)
--------------------------------------------------
With residual ``r = J s - B x``, ``J = -F0``, ``B = B0``:

    (I - T F0) sdot = F0 s + B0 x + T B0 xdot

For one nonnegative real ``t_i`` per stored complex mode:

    m_i      = 1 - t_i a_i
    a_eff_i  = a_i / m_i
    b_hist_i = b_i / m_i**2
    d_x_i    = t_i b_i / m_i

giving the exact derivative-free realization

    hdot_i = a_eff_i h_i + b_hist_i x,     s_i = h_i + d_x_i x

whose unit-interval ZOH coefficients are ``a_bar = exp(a_eff)`` and
``b_bar = phi1(a_eff) b_hist``.

STABILITY
---------
Writing ``a = sigma + i omega``, an exact identity gives

    Re(a_eff) = (sigma - t |a|^2) / |m|^2

so for ``sigma < 0`` and ``t >= 0`` the effective pole is strictly stable for
EVERY admissible t. No extra constraint beyond a stable native pole is needed.
"""

import jax
import jax.numpy as np

# Series switch for phi1. Chosen so the truncated series and the closed form
# agree to well under float64 resolution at the crossover.
_PHI1_SMALL = 1e-4


def phi1(z):
    """phi1(z) = (exp(z) - 1)/z, with phi1(0) = 1. Autodiff-safe.

    The inactive branch of the `where` must not contain a literal 0/0: that
    produces NaN in the cotangent even though the value is discarded. The
    denominator is therefore replaced by 1 where the series is used, so both
    branches are finite in value AND derivative.
    """
    z = np.asarray(z)
    small = np.abs(z) < _PHI1_SMALL
    safe = np.where(small, np.ones_like(z), z)
    series = 1.0 + z / 2.0 + z ** 2 / 6.0 + z ** 3 / 24.0
    exact = (np.exp(z) - 1.0) / safe
    return np.where(small, series, exact)


def absorb_clock(Lambda, B_tilde, step):
    """(Lambda, B_tilde, Delta) -> clocked (F0 diagonal, B0). Applied ONCE."""
    a = step * Lambda
    b = step[:, None] * B_tilde
    return a, b


def softplus_response(raw):
    """Smooth positive map for the response coefficient t >= 0.

    A squared parameterization initialized at exactly zero has zero tangent and
    cannot learn away from zero; softplus does not.
    """
    return jax.nn.softplus(raw)


def inverse_softplus(value):
    """Initial raw value giving softplus(raw) == value, for value > 0.

    Computed HOST-SIDE with the math module on purpose. Under `jax.jit` every
    jnp operation - even on a Python constant - is staged into the jaxpr and
    returns a tracer, so a jnp implementation cannot be converted with float()
    inside a traced `setup()`. This is an initializer, so it belongs on the
    host anyway.
    """
    import math
    v = float(value)
    if v <= 0.0:
        raise ValueError("inverse_softplus requires value > 0")
    return math.log(math.expm1(v))


def gp_response_coefficients(a, b, t):
    """Generalized prospective response, scalar/diagonal T.

    Args:
        a: (P,) complex clocked poles.
        b: (P, H) complex clocked input rows.
        t: (P,) real, nonnegative response coefficients (sample-clock units).
    Returns:
        dict with m, a_eff, b_hist, d_x, a_bar, b_bar.
    """
    t_c = t.astype(a.dtype)
    m = 1.0 - t_c * a
    a_eff = a / m
    b_hist = b / (m ** 2)[:, None]
    d_x = (t_c / m)[:, None] * b
    return dict(m=m, a_eff=a_eff, b_hist=b_hist, d_x=d_x,
                a_bar=np.exp(a_eff), b_bar=phi1(a_eff)[:, None] * b_hist)


def prospective_input_coefficients(a, b, tau_p):
    """2026-style input-side law: sdot = F0 s + B0 x + tau_p B0 xdot.

    The GENERATOR IS UNCHANGED (a_eff = a); only input coupling and feedthrough
    move. Derivative-free form: d_x = tau_p b, b_hist = b + a d_x.
    """
    tau_c = np.asarray(tau_p).astype(a.dtype)
    d_x = tau_c * b if np.ndim(tau_p) == 0 else (tau_c)[:, None] * b
    b_hist = b + a[:, None] * d_x
    return dict(m=np.ones_like(a), a_eff=a, b_hist=b_hist, d_x=d_x,
                a_bar=np.exp(a), b_bar=phi1(a)[:, None] * b_hist)


def matched_tss_coefficients(a, b, tau):
    """Fully matched TSS negative control: tau * rdot = -r, gamma = 0.

    Multiplying the gamma=0 law by J^-1 gives tau sdot = -s + K x + tau K xdot
    with K = J^-1 B = -b/a. Its derivative-free form has

        a_eff = -1/tau,     b_hist = 0,     d_x = K

    so the DRIVEN HISTORY IS EXACTLY ZERO: the transfer is static plus a
    decaying initial transient. This is the predicted cancellation, kept as an
    explicit control rather than as a claim about all TSS constructions.
    """
    tau_c = np.asarray(tau).astype(a.dtype)
    K = -b / a[:, None]
    a_eff = -np.ones_like(a) / tau_c
    b_hist = np.zeros_like(b)
    return dict(m=np.ones_like(a), a_eff=a_eff, b_hist=b_hist, d_x=K,
                a_bar=np.exp(a_eff), b_bar=np.zeros_like(b))


def effective_pole_real_part(a, t):
    """Closed form Re(a_eff) = (sigma - t|a|^2)/|m|^2, for the stability check."""
    sigma = a.real
    m = 1.0 - t.astype(a.dtype) * a
    return (sigma - t * (np.abs(a) ** 2)) / (np.abs(m) ** 2)


def is_admissible(a, t):
    """Documented admissibility: native pole strictly stable and t >= 0."""
    return np.logical_and(a.real < 0.0, t >= 0.0)
