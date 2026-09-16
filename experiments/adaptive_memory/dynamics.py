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
def _switch(dtype):
    """Dtype-appropriate series/closed-form switch on `m = mu h^2`.

    The series below keeps terms through `m^3`, so its truncation error is
    about `m^4/8!`. Setting that equal to the machine epsilon gives
    `|m| <= (8! eps)^(1/4)`: roughly 0.26 in float32 and 1.7e-3 in float64.
    On the OTHER side of the switch the closed form subtracts two exponentials
    and divides by `2a` with `a = sqrt(m)`, whose relative cancellation error
    is about `eps/(2a)`: at the switch that is ~1e-7 in float32 and ~1e-15 in
    float64. Both sides are therefore well conditioned AT the switch, which is
    what makes a single threshold legitimate rather than merely convenient.
    """
    return float((jnp.finfo(dtype).eps * 40320.0) ** 0.25)


def expm2(G, h=H):
    """`exp(h G)` for a 2x2 `G`, differentiable in every entry, overflow-safe.

    With `t = tr G`, `N = G - (t/2) I` traceless and `N^2 = mu I` where
    `mu = ((g11-g22)^2 + 4 g12 g21)/4`, the exponential is

        exp(hG) = e^{s}[ ch(m) I + h shc(m) N ],
        s = h t/2 ,  m = mu h^2 ,

    with `ch` and `shc` entire in `m`. Written that way it OVERFLOWS for a
    perfectly finite answer: `G = diag(-200, -1)` in float32 has true entries
    `e^{-200}` and `e^{-1}`, but `cosh(99.5)` exceeds the float32 range and
    multiplying afterwards by the tiny `e^{s}` cannot repair the intermediate
    infinity. Admissible learned rates reach this regime and no coefficient
    bound excludes it - so the fix is arithmetic, not a clamp.

    The trace factor is therefore folded INSIDE the hyperbolic functions,
    giving the eigenvalues `s +- a` of `hG` directly:

        m >  switch:  C = (e^{L+} + e^{L-})/2 ,  S = (e^{L+} - e^{L-})/(2a)
        m < -switch:  C = e^{s} cos(b) ,           S = e^{s} sin(b)/b
        |m| <= switch: the series in `m`, times `e^{s}`

    and `exp(hG) = C I + h S N`, where `L+- = s +- a` are the eigenvalues of
    `hG`. For a stable generator `s + a <= 0`, so every exponent above is
    non-positive and nothing overflows. The near-zero eigenvalue is obtained
    from `L+ L- = det(hG)` rather than from `s + a`, which avoids the
    cancellation in that sum when the eigenvalues are widely separated. Every one of our
    generators has `t <= 0`: prospective `-nu w - 1/tau`, inertial `-1/tau`,
    TSS `-T w/M`, and the idle limits of each.

    Each branch is evaluated on `where`-guarded SAFE inputs, so the inactive
    branches produce neither NaN nor Inf and cannot poison the backward pass
    through `where`'s zero-weighted cotangent.

    No eigendecomposition, no branch cut, no host constant: every coefficient
    stays inside autodiff, which the learned response requires.
    """
    g11, g12 = G[..., 0, 0], G[..., 0, 1]
    g21, g22 = G[..., 1, 0], G[..., 1, 1]
    dtype = jnp.result_type(g11)
    sw = _switch(dtype)
    t = g11 + g22
    s = h * t / 2.0
    m = (((g11 - g22) ** 2 + 4.0 * g12 * g21) / 4.0) * h * h

    big_pos = m > sw
    big_neg = m < -sw
    small = jnp.logical_not(jnp.logical_or(big_pos, big_neg))

    one = jnp.ones_like(m)
    # real, well-separated eigenvalues: fold e^{s} into each exponential
    a = jnp.sqrt(jnp.where(big_pos, m, one))
    # the trace is guarded too, so a huge `s` in an INACTIVE branch cannot
    # produce an Inf that `where` would later multiply by a zero cotangent
    s_pos = jnp.where(big_pos, s, jnp.zeros_like(s))
    # `s + a` is a CANCELLATION when the eigenvalues are widely separated: it
    # forms the near-zero eigenvalue as a sum of two large opposite-sign
    # quantities. For the prospective generator at nu = e^9 that cancels
    # ~4052 down to ~1, discarding 12 of float32's 24 bits and leaving a
    # relative error of ~2.4e-4 in e^{s+a} - which the cluster measured as
    # 1.05e-4 on the whole matrix.
    #
    # Same remedy as the stable quadratic formula: form the LARGE-magnitude
    # eigenvalue by adding same-sign terms, then recover the near-zero one
    # from the exact product relation `lam_+ lam_- = det(hG)`, which avoids
    # the cancellation in `s + a`. (The generic determinant still subtracts
    # two products; for our generators that subtraction is well conditioned.)
    # `|lam_far| = |s| + a >= a > 0` in this branch, so the division is safe.
    det_h = (h * h) * (g11 * g22 - g12 * g21)
    # The determinant is guarded BEFORE forming the near root. Otherwise an
    # inactive real branch still computes it: for the oscillatory
    # [[-1, 1], [-1e6, -1]] it gives lam_near = 1e6+1 and exp overflows, and
    # an infinite derivative times the zero cotangent of `where` is NaN.
    # Masking the exponential afterwards would be too late. With the mask the
    # inactive branch has roots 1 and 0: finite dummies, zero derivative.
    det_h = jnp.where(big_pos, det_h, jnp.zeros_like(det_h))
    sgn = jnp.where(s_pos >= 0, one, -one)
    lam_far = s_pos + sgn * a
    lam_near = det_h / lam_far
    lam_hi = jnp.where(sgn > 0, lam_far, lam_near)          # = s + a
    lam_lo = jnp.where(sgn > 0, lam_near, lam_far)          # = s - a
    ep, em = jnp.exp(lam_hi), jnp.exp(lam_lo)
    C_real, S_real = 0.5 * (ep + em), 0.5 * (ep - em) / a

    # complex-conjugate poles: cos and sin are bounded, so e^{s} is safe
    b = jnp.sqrt(jnp.where(big_neg, -m, one))
    s_osc = jnp.where(big_pos, jnp.zeros_like(s), s)
    es = jnp.exp(s_osc)
    C_osc, S_osc = es * jnp.cos(b), es * jnp.sin(b) / b

    # confluent / near-confluent: entire series in m, through m^3
    ms = jnp.where(small, m, jnp.zeros_like(m))
    C_ser = es * (1.0 + ms / 2.0 + ms * ms / 24.0 + ms * ms * ms / 720.0)
    S_ser = es * (1.0 + ms / 6.0 + ms * ms / 120.0 + ms * ms * ms / 5040.0)

    C = jnp.where(big_pos, C_real, jnp.where(big_neg, C_osc, C_ser))
    S = jnp.where(big_pos, S_real, jnp.where(big_neg, S_osc, S_ser))

    half = t / 2.0
    n11, n12, n21, n22 = g11 - half, g12, g21, g22 - half
    f11 = C + h * S * n11
    f12 = h * S * n12
    f21 = h * S * n21
    f22 = C + h * S * n22
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
