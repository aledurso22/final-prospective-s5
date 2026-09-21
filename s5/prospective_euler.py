"""The professor's prospective equation, per mode, with the stability
boundary the equation itself imposes.

THE EQUATION.  tau s' = -s + f(s,t) + tau d/dt f(s,t),  f(s,t) = A s + B x.

WHY IT CANCELS THE MEMORY, AND WHY GENERALIZING CANNOT FIX THAT. The
equation is P(D)(s - f) = 0 with P(D) = 1 + tau D. Since f = As + Bx,

    s - f = (I - A) s - B x,
    P(D)[(I - A)s - Bx] = 0   =>   (I - A) P(D) s = P(D) B x
                              =>   P(D) s = (I - A)^{-1} P(D) B x.

The homogeneous dynamics are P(D)s = 0: A has left the dynamics entirely
and survives only as a static input gain. Every mode then relaxes with the
single timescale tau, whatever A was. THE PROOF NEVER USES THE FORM OF P,
so the generalized operator 1 + tau D + M D^2 cancels the memory in exactly
the same way. The order of the operator was never the problem. The problem
is that the residual (s - f) carries (I - A) as a constant left factor,
which commutes with any P(D) and divides out.

WHAT THE EULER DISCRETIZATION DOES INSTEAD. The discrete form,

    s_t = A1 s_{t-1} + A2 s_{t-2} + xt,
    A1 = (1 - e) I + (1 + e) A,   A2 = -A,   e = dt / tau,
    xt = (1 + e) B x_{t-1} - B x_{t-2},

does NOT cancel, because its stencils are inconsistent: the poles are the
roots of L^2 - A1 L + a, whose PRODUCT IS EXACTLY a. The memory is not
destroyed, it is split between two poles. The price is stability -- for
COMPLEX modes the radius leaves the unit disc (1.4416 at |a| = 0.9,
omega = 2, tau = 2, which is the value measured on the cluster). Real
modes are always safe; S5's modes are not real.

THE MANY-BODY REPAIR. The instability is PER MODE and its boundary is
exact. Schur-Cohn for L^2 + c1 L + c0 with complex coefficients requires
|c0| < 1 and |c1 - conj(c1) c0| < 1 - |c0|^2. Substituting c1 = -A1,
c0 = a and A1 = (1-e) + (1+e)a, the condition collapses to |R + eC| < R
with R = 1 - |a|^2 and C = 2a - 1 - |a|^2, whose real part is -|1-a|^2.
Hence, in closed form,

    e_max(a) = 2 (1 - |a|^2) |1 - a|^2 / |2a - 1 - |a|^2|^2,

and rho crosses 1 at exactly this value. Because tau = dt / e, the
condition reads: A MODE MAY NOT ANTICIPATE FURTHER AHEAD THAN IT CAN
REMEMBER -- tau_min comes out at the mode's own native timescale. Slow
modes may only be weakly prospective, fast modes strongly so. That is the
phase boundary, and the equation imposes it rather than the modeller.

TWO SEPARATE KNOBS, AND WHY THEY MUST BE SEPARATE. rho is NOT monotone in
e. At e = 0 the cell has a pole exactly at z = 1 -- cancelled analytically
by the numerator zero at z = 1, but marginal, and a marginal pole paired
with a zero is precisely what lost everything in float32 on the rejected
branch. At e = e_max it touches the disc from the other side. rho dips in
between. So a single gate sliding e from 0 to e_max walks into a marginal
pole at BOTH ends.

The two questions are therefore asked separately, which is also the more
faithful reading of the many-body idea:

    HOW FAR AHEAD does mode j look?   e_j = e_max(a_j) * band(raw_j),
                                      band confined to EPSILON_BAND, so the
                                      cell is strictly inside the disc and
                                      away from both edges;
    DOES mode j participate AT ALL?   g_j in (0, 1), mixing the native and
                                      prospective states. g = 0 is Native
                                      S5 exactly and bit-identically, with
                                      no marginal pole computed anywhere.

Stability is structural: no value of any parameter can put a pole outside
the disc, because e never leaves the band and the band is inside (0, e_max)
by construction.

WHAT THE TWO POLES DO TO MEMORY. Their product is exactly a, so
rho >= sqrt(|a|) > |a| for every |a| < 1: the prospective cell's slowest
pole is always SLOWER than the native one. The Euler form does not shorten
memory, it lengthens it -- which is the opposite of the continuous
equation's cancellation, and the reason a per-mode treatment is worth
anything at all.

RUNNING IT. The second-order scalar recurrence factors over its own roots,

    (1 - L+ z^-1)(1 - L- z^-1) s = xt,

so it is two first-order scans, not a 2x2 matrix scan -- the same shape
`s5/ssm.py`'s `binary_operator` composes, and better conditioned than the
companion form.
"""

import jax
import jax.numpy as np

from .ssm import binary_operator

#: the token step of the Euler integration
DT = 1.0
#: rho is certified at or below 1 - MARGIN. The margin is necessarily
#: small: the two poles multiply to a, so rho >= sqrt(|a|), and a native
#: mode at |a| = 0.9999 already forces rho >= 0.99995. The bound is a
#: guard against the DISCRETIZATION leaving the disc, not a claim that the
#: cell decays quickly.
MARGIN = 1e-5
#: e is confined to this fraction of e_max, away from the marginal pole at
#: e = 0 and the stability boundary at e = e_max. Measured over 4000 random
#: modes the worst radius in this band is 0.99986, against 1.00000 at the
#: edges.
EPSILON_BAND = (0.25, 0.75)


def epsilon_max(lambda_bar):
    """The largest e = dt/tau this mode can carry and stay Schur-stable.

    Closed form, exact: rho crosses 1 at precisely this value. Positive for
    every |a| < 1 with a != 1, since Re(2a - 1 - |a|^2) = -|1 - a|^2 <= 0.
    """
    magnitude = np.abs(lambda_bar) ** 2
    curvature = 2 * lambda_bar - 1 - magnitude
    return (2 * (1 - magnitude) * np.abs(1 - lambda_bar) ** 2
            / np.maximum(np.abs(curvature) ** 2, 1e-30))


def epsilon_from_raw(lambda_bar, raw, band=EPSILON_BAND):
    """e inside the band, as a fraction of this mode's own e_max.

    The horizon is learned, but only within the interval where the cell is
    strictly inside the unit disc. No value of `raw` can leave it, and the
    band excludes both e = 0 (pole at z = 1) and e = e_max (the boundary).
    """
    low, high = band
    fraction = low + (high - low) * jax.nn.sigmoid(raw)
    return fraction * epsilon_max(lambda_bar)


def mix(native, prospective, gate):
    """Per-mode participation: g = 0 is Native EXACTLY, g = 1 is fully
    prospective. This is the 'some modes use it, some do not' of the
    many-body reading, made differentiable."""
    return (1.0 - gate) * native + gate * prospective


def tau_from_epsilon(epsilon, dt=DT):
    """The prospective horizon in tokens. e = 0 is tau = infinity, which is
    Native: the cell stops anticipating rather than anticipating nothing."""
    return dt / np.maximum(epsilon, 1e-30)


def companion_coefficients(lambda_bar, epsilon):
    """(A1, A2) of s_t = A1 s_{t-1} + A2 s_{t-2} + xt, per mode."""
    return (1 - epsilon) + (1 + epsilon) * lambda_bar, -lambda_bar


def roots(lambda_bar, epsilon):
    """The two poles, in closed form.

    Their PRODUCT is exactly lambda_bar -- the native memory is split
    between them rather than cancelled, which is the one thing the Euler
    discretization gets right and the continuous equation does not.
    """
    first, _ = companion_coefficients(lambda_bar, epsilon)
    spread = np.sqrt(first * first - 4 * lambda_bar)
    return (first + spread) / 2, (first - spread) / 2


def spectral_radius(lambda_bar, epsilon):
    """max |root|, the quantity the certification bounds."""
    plus, minus = roots(lambda_bar, epsilon)
    return np.maximum(np.abs(plus), np.abs(minus))


def drive(b_bar, input_sequence, epsilon):
    """xt = (1 + e) B x_{t-1} - B x_{t-2}, with ZERO prehistory.

    At e = 0 this is B(x_{t-1} - x_{t-2}), whose zero at z = 1 cancels the
    pole the cell then has at z = 1 -- which is why e = 0 is exactly Native
    and not Native plus a free integrator.
    """
    projected = jax.vmap(lambda u: b_bar @ u)(input_sequence)
    first = _shift(projected, 1)
    second = _shift(projected, 2)
    return (1 + epsilon) * first - second


def _shift(values, distance):
    zeros = np.zeros_like(values[:distance])
    return np.concatenate((zeros, values[:-distance]), axis=0)


def _first_order(pole, sequence):
    """u_t = pole * u_{t-1} + sequence_t, as the repository's own scan."""
    poles = pole * np.ones((sequence.shape[0],) + pole.shape,
                           dtype=sequence.dtype)
    _, states = jax.lax.associative_scan(binary_operator,
                                         (poles, sequence.astype(pole.dtype)))
    return states


def apply_prospective(lambda_bar, b_bar, input_sequence, epsilon):
    """The professor's cell, per mode, as two first-order scans.

    The second-order recurrence factors over its own roots, so no 2x2
    companion is ever built and the scan primitive is the repository's.
    """
    plus, minus = roots(lambda_bar, epsilon)
    excitation = drive(b_bar, input_sequence, epsilon)
    return _first_order(minus, _first_order(plus, excitation))


def certify(lambda_bar, epsilon, bound=1.0 - MARGIN):
    """Whole-inventory certification, as a dict of plain numbers.

    `radius_over_native` is reported because rho >= sqrt(|a|) is forced by
    the poles' product: a value above 1 is expected and is not a defect.
    What would be a defect is rho >= 1.
    """
    radius = spectral_radius(lambda_bar, epsilon)
    native = np.abs(lambda_bar)
    return {
        "max_spectral_radius": float(np.max(radius)),
        "max_native_pole": float(np.max(native)),
        "max_radius_over_sqrt_native": float(
            np.max(radius / np.sqrt(np.maximum(native, 1e-30)))),
        "modes_at_or_above_bound": int(np.sum(radius >= bound)),
        "bound": float(bound),
        "certified": bool(np.max(radius) < bound),
    }
