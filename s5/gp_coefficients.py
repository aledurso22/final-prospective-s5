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

import math as _math

import jax
import jax.numpy as np

# phi1 switch thresholds, chosen from MEASURED error, per dtype.
#
# Two errors fight each other:
#   * series truncation grows with |z|;
#   * cancellation in (exp(z)-1)/z, and far more severely in its DERIVATIVE,
#     grows as |z| shrinks - roughly eps/|z|^2 for phi1'.
#
# Measured in complex64, derivative absolute error of the direct branch:
#     |z|=1e-4 -> 1.0e+03      |z|=1e-3 -> 1.3e+01
#     |z|=1e-2 -> 5.2e-02      |z|=0.1  -> 1.4e-03
# so a 1e-4 cutoff (the previous value) leaves the derivative useless, and
# even wrong-signed, just outside it. The series with 14 terms is accurate to
# ~1e-12 at |z|=1, so the 32-bit switch is placed there instead.
_PHI1_SWITCH_32 = 1.0
_PHI1_SWITCH_64 = 1e-2
_PHI1_TERMS = 14
# 1/(n+1)! for n = 0..13, the phi1 Taylor coefficients
_PHI1_COEFFS = tuple(1.0 / _math.factorial(n + 1) for n in range(_PHI1_TERMS))


def _switch_threshold(dtype):
    """32-bit and 64-bit need very different crossover points."""
    return _PHI1_SWITCH_32 if np.finfo(
        np.zeros((), dtype).real.dtype).eps > 1e-10 else _PHI1_SWITCH_64


def safe_expm1(z):
    """exp(z) - 1 for complex z, without the catastrophic cancellation.

    Writing z = x + iy,

        Re = expm1(x) cos y - 2 sin^2(y/2)
        Im = exp(x) sin y

    Both pieces are computed from primitives that are themselves accurate near
    zero, so the small-|z| relative error stays at eps rather than eps/|z|.
    """
    z = np.asarray(z)
    if not np.issubdtype(z.dtype, np.complexfloating):
        return np.expm1(z)
    x, y = z.real, z.imag
    half = np.sin(0.5 * y)
    return (np.expm1(x) * np.cos(y) - 2.0 * half * half) + 1j * (np.exp(x) * np.sin(y))


def phi1(z):
    """phi1(z) = (exp(z) - 1)/z, with phi1(0) = 1. Accurate in VALUE AND
    DERIVATIVE, in float32/complex64 as well as float64/complex128.

    The inactive branch of the `where` never contains a literal 0/0: the
    denominator is replaced by 1 where the series is used, so both branches are
    finite in value and in cotangent.
    """
    z = np.asarray(z)
    cutoff = _switch_threshold(z.dtype)
    small = np.abs(z) < cutoff
    safe = np.where(small, np.ones_like(z), z)
    series = np.zeros_like(z)
    for c in reversed(_PHI1_COEFFS):          # Horner, most accurate ordering
        series = series * z + np.asarray(c, dtype=z.dtype)
    exact = safe_expm1(z) / safe
    return np.where(small, series, exact)


def absorb_clock(Lambda, B_tilde, step):
    """(Lambda, B_tilde, Delta) -> clocked (F0 diagonal, B0). Applied ONCE."""
    a = step * Lambda
    b = step[:, None] * B_tilde
    return a, b


def softplus_response(raw):
    """Smooth positive map for the response coefficient, t = softplus(raw) > 0.

    softplus is a bijection onto (0, inf): no finite raw value gives t = 0
    exactly, and inverse_softplus(0) is not finite. That is why a zero target
    scale is rejected.

    Note softplus'(0) = 1/2, so raw = 0 is a perfectly usable starting point -
    the "zero tangent at zero" problem belongs to a SQUARED parameterization,
    not to this one. (Corrected per coordinator review R7.4.)
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
