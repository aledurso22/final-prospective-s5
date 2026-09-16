"""Adaptive generalized prospective memory: learned response, learned source gate.

The dynamical law is UNCHANGED from the completed fixed-coefficient study:

    E_t(W) = (w_t/2)||W k - v||^2 ,  w_t = m_t a_psi(k, v) ,  R = w_t (Wk - v)k^T
    M Wddot + gamma Wdot + R + T Rdot = 0 ,  M, gamma, T > 0 ,  M <= gamma T

What is new is that the three response coefficients are LEARNED between
optimizer updates by outer BPTT, and the objective carries a learned positive
source weight `a_psi`. `Rdot` is the total derivative, so a change in the
source weight enters the generator, not merely its forcing. Coefficients are
constant WITHIN an episode: no token-dependent mass or damping is substituted
into a constant-coefficient derivation.

IDENTIFIABLE PARAMETERIZATION. Three trainable scalars

    nu = T/M ,  tau = M/gamma ,  rho = M/(gamma T)
    gamma = 1/(nu rho) ,  M = tau/(nu rho) ,  T = tau/rho

with `nu = exp(raw_nu)`, `tau = exp(raw_tau)`, `rho = sigmoid(raw_rho)`. The
transforms impose `M, gamma, T > 0` and `M <= gamma T` BY CONSTRUCTION, so
there is no clipping, projection or weight decay anywhere on them, and no
boundary at which a gradient can be lost. The original fixed reference is
`nu = 4/3`, `tau = 3/4`, `rho = 3/4`.

PERSISTENT/TRANSIENT COORDINATES are the executed ones:

    W = C + Z ,  Cdot = -eta R ,  tau Zdot = -Z - kappa R
    eta = nu rho ,  kappa = tau nu (1 - rho)

so `nu` is the initial correction rate when `Z = 0`, `rho` the fraction
allocated to the accumulated update, and `tau` the no-write decay time of `Z`.
`C` is constant only while `R = 0`; it is NOT protected from later interfering
writes. The readout is `W q` throughout - never `C q`, never a learned mixture
of `W` with the auxiliary, and never the auxiliary alone.

Carrying `(W, Z)` keeps both continuous through key, value, mask and GATE
jumps, which is the property the equation needs; `(W, P)` with `P = -gamma Z`
is the same thing in the earlier study's coordinates.
"""

import jax
import jax.numpy as jnp

RULES = ("adaptive_prospective", "adaptive_inertial", "adaptive_delta",
         "tss_prospective", "ideal_projection",
         "gated_delta", "momentum_delta")
#: the two ordinary-prospective references added by the 16 September 2026
#: amendment. They are reported on their OWN screen, separately from the
#: Momentum/Gated DeltaNet screen, and neither is assumed memoryless or
#: assumed to score worse.
ORDINARY_PROSPECTIVE = ("tss_prospective", "ideal_projection")
LITERATURE = ("gated_delta", "momentum_delta")
DISPLAY = {
    "adaptive_prospective": "Adaptive generalized prospective memory",
    "adaptive_inertial": "Adaptive inertial memory (no prospective correction)",
    "adaptive_delta": "Adaptive first-order delta memory",
    "tss_prospective": "TSS finite-adaptation prospective dynamics",
    "ideal_projection": "Ideal prospective equilibrium (minimum-change)",
    "gated_delta": "Gated DeltaNet rule",
    "momentum_delta": "Momentum DeltaNet rule",
}
CARRY = {"adaptive_prospective": 128, "adaptive_inertial": 128,
         "adaptive_delta": 64, "tss_prospective": 128,
         "ideal_projection": 64, "gated_delta": 64, "momentum_delta": 128}
#: the fixed reference of the completed study, in the new coordinates
NU_REF, TAU_REF, RHO_REF = 4.0 / 3.0, 0.75, 0.75
H = 1.0


# ------------------------------------------- differentiable 2x2 expm -------
def _expm2_scalars(mu):
    """`cosh(sqrt(mu))` and `sinh(sqrt(mu))/sqrt(mu)`, both ENTIRE in `mu`.

    Written so the derivative is finite everywhere, including at a confluent
    root `mu = 0` and in the inactive branch. Two traps are avoided
    deliberately:

    * `sqrt` at zero has an infinite derivative, so the small-|mu| branch
      substitutes a CONSTANT into the closed form rather than zero - a
      `where(small, 0, mu)` would give `inf * 0 = NaN` on the backward pass;
    * negative `mu` (complex poles, which the inertial control can have) is
      handled with the real trigonometric form rather than complex arithmetic.

    Both functions are analytic in `mu`, so the series branch is exact to the
    order kept and the two branches agree at the switch.
    """
    small = jnp.abs(mu) < 1e-6
    safe = jnp.where(small, jnp.ones_like(mu), mu)      # never 0 in this branch
    a = jnp.sqrt(jnp.abs(safe))
    ch = jnp.where(safe >= 0, jnp.cosh(a), jnp.cos(a))
    sh = jnp.where(safe >= 0, jnp.sinh(a) / a, jnp.sin(a) / a)
    ch_series = 1.0 + mu / 2.0 + mu * mu / 24.0
    sh_series = 1.0 + mu / 6.0 + mu * mu / 120.0
    return jnp.where(small, ch_series, ch), jnp.where(small, sh_series, sh)


def expm2(G, h=H):
    """`exp(h G)` for a 2x2 `G`, differentiable in every entry.

    Splits off the trace: with `N = G - (tr/2) I` traceless, `N^2 = mu I` where
    `mu = disc/4 = ((g11-g22)^2 + 4 g12 g21)/4`, hence

        exp(hG) = e^{h tr/2} [ cosh(h sqrt(mu)) I + h sinhc(h sqrt(mu)) N ].

    No eigendecomposition, no branch cut, no host constant: every coefficient
    stays inside autodiff, which the learned response requires.
    """
    g11, g12 = G[..., 0, 0], G[..., 0, 1]
    g21, g22 = G[..., 1, 0], G[..., 1, 1]
    tr = g11 + g22
    mu = ((g11 - g22) ** 2 + 4.0 * g12 * g21) / 4.0
    ch, sh = _expm2_scalars(mu * h * h)
    pre = jnp.exp(h * tr / 2.0)
    half = tr / 2.0
    n11, n12, n21, n22 = g11 - half, g12, g21, g22 - half
    f11 = pre * (ch + h * sh * n11)
    f12 = pre * (h * sh * n12)
    f21 = pre * (h * sh * n21)
    f22 = pre * (ch + h * sh * n22)
    return jnp.stack([jnp.stack([f11, f12], axis=-1),
                      jnp.stack([f21, f22], axis=-1)], axis=-2)


# ------------------------------------------------- response coefficients ---
def response(raw):
    """`(nu, tau, rho)` and every derived quantity, from the raw scalars."""
    nu = jnp.exp(raw["raw_nu"])
    tau = jnp.exp(raw["raw_tau"])
    rho = jax.nn.sigmoid(raw["raw_rho"])
    return dict(nu=nu, tau=tau, rho=rho,
                eta=nu * rho, kappa=tau * nu * (1.0 - rho),
                gamma=1.0 / (nu * rho), M=tau / (nu * rho), T=tau / rho)


def inertial_response(raw):
    """`(eta, tau)` for the ablation; `eta = 1/gamma > 0`, `tau = M/gamma > 0`."""
    return dict(eta=jnp.exp(raw["raw_eta"]), tau=jnp.exp(raw["raw_tau"]))


def prospective_generator(nu, tau, rho, w):
    """`G(w) = [[-nu w, -1/tau], [-nu(1-rho) w, -1/tau]]` in `(e, z)`.

    Derived, not asserted: with `R = w e k^T` for a held unit key,
    `edot = Wdot k = -(eta + kappa/tau) w e - z/tau` and
    `eta + kappa/tau = nu rho + nu(1-rho) = nu`, while
    `zdot = -z/tau - (kappa/tau) w e` with `kappa/tau = nu(1-rho)`.
    """
    z = jnp.zeros_like(w)
    return jnp.stack([
        jnp.stack([-nu * w, z - 1.0 / tau], axis=-1),
        jnp.stack([-nu * (1.0 - rho) * w, z - 1.0 / tau], axis=-1)], axis=-2)


def inertial_generator(eta, tau, w):
    """`G_I(w) = [[0, -1/tau], [eta w, -1/tau]]`, with `Z = -tau Wdot`."""
    z = jnp.zeros_like(w)
    return jnp.stack([
        jnp.stack([z, z - 1.0 / tau], axis=-1),
        jnp.stack([eta * w, z - 1.0 / tau], axis=-1)], axis=-2)


def single_write_query_amplitude(F, tau, h=H):
    """`1 - F11 + (1 - exp(-h/tau)) F21`: the calibration observable.

    One isolated unit write from `W = Z = 0`, read one IDLE interval later.
    The write leaves `W = (1-F11) v k^T` and `Z = -F21 v k^T`; the idle
    interval adds `(a0 - 1) Z`, giving the amplitude above. It matches ONE
    scalar observable at zero initial carry - not the transfer function, and
    not the response under interfering writes.
    """
    a0 = jnp.exp(-h / tau)
    return 1.0 - F[..., 0, 0] + (1.0 - a0) * F[..., 1, 0]


# ------------------------------------------------------------ token step ---
def two_state_step(W, Z, k, v, F, a0):
    """One exact interval in `(W, Z)`, for a held input and source weight.

    `F` already encodes the source weight THROUGH THE GENERATOR, which is the
    point: a gate must not be applied as a multiplier on a fixed discrete
    increment, because `w` changes `G(w)` and not only its forcing.

    There is no mask branch here, and none is needed. The idle generator
    `G(0) = [[0, -1/tau], [0, -1/tau]]` gives `F11 = 1`, `F21 = 0` and
    `F12 = a0 - 1` exactly, so on a no-write interval the rank-one term
    cancels identically and this reduces to `W' = W + (a0-1)Z`, `Z' = a0 Z` -
    the autonomous motion a query interval also undergoes. An invalid key is
    passed as `k = 0`, which zeroes the outer products for the same reason.
    """
    e = W @ k - v
    z = Z @ k
    e_new = F[0, 0] * e + F[0, 1] * z
    z_new = F[1, 0] * e + F[1, 1] * z
    W = W + (a0 - 1.0) * Z + jnp.outer((e_new - e) - (a0 - 1.0) * z, k)
    Z = a0 * Z + jnp.outer(z_new - a0 * z, k)
    return W, Z


def delta_step(W, k, v, beta):
    """Exact gradient-flow step of the first-order control.

    `Cdot = -eta R` alone gives `e(h) = e exp(-h eta w)`, hence
    `W' = W + beta(w)(v - Wk)k^T` with `beta(w) = -expm1(-h eta w)`. A no-write
    interval leaves `W` fixed, which `beta(0) = 0` delivers without a branch.
    """
    return W + beta * jnp.outer(v - W @ k, k)


def safe_normalize(x, floor=1e-6):
    """Unit key plus a validity flag; below the floor the key is NOT unit."""
    n2 = jnp.sum(x * x, axis=-1, keepdims=True)
    n = jnp.sqrt(jnp.maximum(n2, floor * floor))
    valid = (n2 >= floor * floor).astype(x.dtype)
    return jnp.where(valid > 0, x / n, jnp.zeros_like(x)), valid[..., 0]


# ------------------------------- ordinary prospective reference 1: TSS -----
def tss_response(raw):
    """`tau_m = exp(raw_tau_m)`, `epsilon = tau_m * sigmoid(raw_ratio)`.

    The sigmoid keeps `epsilon < tau_m`, i.e. finite adaptation strictly faster
    than the membrane, and keeps the prospective horizon matched to `tau_m`.
    Learning these two positive scalars is OUR budget-matched adaptation of the
    published forward law to this shell; it is not a claim about the paper's
    own coefficient-training policy.

    Eliminating `A` from

        eps Adot = -A + f ,  tau_m Wdot = -W + (1 + tau_m/eps) f - (tau_m/eps)A

    with `f = W - R` gives EXACTLY `tau_m eps Wddot + R + (tau_m + eps) Rdot = 0`,
    so `M = tau_m eps`, `T = tau_m + eps` and `gamma = 0`. `gamma = 0` puts this
    reference OUTSIDE the generalized candidate's `M <= gamma T` sector at
    `M > 0`; no damping is added to force it in, and nothing here may divide by
    `gamma` or use the `(C, Z)` storage formula.
    """
    tau_m = jnp.exp(raw["raw_tau_m"])
    eps = tau_m * jax.nn.sigmoid(raw["raw_ratio"])
    return dict(tau_m=tau_m, epsilon=eps, ratio=eps / tau_m,
                M=tau_m * eps, T=tau_m + eps, gamma=jnp.zeros_like(tau_m))


def tss_generator(M, T, w):
    """`[[-T w/M, 1/M], [-w, 0]]` in `(e, p)`, with `P = M Wdot + T R`.

    From `Wdot = (P - T R)/M` and `Pdot = -R`: for a held unit key,
    `edot = Wdot k = -(T w/M) e + p/M` and `pdot = Pdot k = -w e`.
    """
    z = jnp.zeros_like(w)
    return jnp.stack([
        jnp.stack([-T * w / M, z + 1.0 / M], axis=-1),
        jnp.stack([-w, z], axis=-1)], axis=-2)


def tss_single_write_query_amplitude(F, M, h=H):
    """`1 - F11 - (h/M) F21`: the SAME calibration observable, in `(W, P)`.

    One isolated unit write from `W = P = 0`, read one idle interval later.
    The write leaves `W = (1-F11) v k^T` and `P = -F21 v k^T`; the idle
    interval adds `b0 P` with `b0 = h/M`, giving the amplitude above. Note the
    MINUS sign: `(W, P)` here is a different auxiliary coordinate with
    different idle dynamics from the generalized candidate's `(W, Z)`, whose
    formula is `1 - F11 + (1 - a0) F21`. The two are not interchangeable.
    """
    return 1.0 - F[..., 0, 0] - (h / M) * F[..., 1, 0]


def tss_two_state_step(W, P, k, v, F, b0):
    """One exact interval in `(W, P)` for the TSS reference.

    No mask branch and none needed: at `w = 0` the generator is
    `[[0, 1/M], [0, 0]]`, so `F11 = 1`, `F21 = 0`, `F12 = h/M = b0`, the
    rank-one terms cancel identically and this reduces to the UNDAMPED
    no-write motion `P' = P`, `W' = W + h P/M`. That motion follows from the
    published matching law in this rank-deficient placement: it is not frozen,
    and the reference is not memoryless. `A` and `P` are never reset at key
    changes or queries, only at episode boundaries.
    """
    e = W @ k - v
    p = P @ k
    e_new = F[0, 0] * e + F[0, 1] * p
    p_new = F[1, 0] * e + F[1, 1] * p
    W = W + b0 * P + jnp.outer((e_new - e) - b0 * p, k)
    P = P + jnp.outer(p_new - p, k)
    return W, P


# --------------------- ordinary prospective reference 2: ideal equilibrium --
def projection_step(W, k, v, m):
    """Minimum-Frobenius-change completion of the zero-residual branch.

        W+ = argmin_U ||U - W-||_F^2 / 2   s.t.  U k = v   ==>
        W+ = W- + (v - W- k) k^T     (unit k)

    A full-strength delta projection: it preserves associations orthogonal to
    `k` and can interfere with nonorthogonal ones. It is NOT a full-rank
    memoryless control, NOT the unique consequence of TSS Eq. (5), and NOT the
    paper's finite-difference scheme - the zero-residual branch, the
    instantaneous update at input jumps and the minimum-change completion are
    additional specifications made here explicitly.

    A strictly positive scalar source weight changes neither the constraint nor
    this update, so this arm carries NO gate leaves rather than parameters with
    identically zero gradients. No relaxation factor is learned. Queries, idle
    intervals and invalid keys hold `W` unchanged, so there is no autonomous
    query motion for this particular completion.
    """
    return W + m * jnp.outer(v - W @ k, k)
