"""Generalized prospective memory on BOTH sides of the delta boundary.

Brief: META_DELTA_NEXT_AGENT_BRIEF_2026_09_16.md. Analytical basis, audited
independently in docs/META_DELTA_PROOF_AUDIT.md:
GENERAL_PROSPECTIVE_META_DELTA_AUDIT_2026_09_16.md.

The inner law is UNCHANGED from the completed adaptive-memory study:

    M Wddot + gamma Wdot + R + T Rdot = 0 ,  R = w (W k - v) k^T ,
    coefficients constant within an episode, W and P = M Wdot + T R (hence
    Z = -P/gamma) continuous at token jumps.

What changes is the RESPONSE PARAMETERIZATION and its domain.

Coordinates (all trainable, direct logs, no sigmoid, no square)
----------------------------------------------------------------
    eta = exp(raw_eta) = 1/gamma   the first-order delta rate
    tau = exp(raw_tau) = M/gamma   the relaxation time of Z
    rho = exp(raw_r)   = M/(gamma T)

hence gamma = 1/eta, M = tau/eta, T = tau/rho, and for the executed (e, z)
generator nu = T/M = eta/rho. At raw_r = 0 (rho = 1 EXACTLY, a finite
trainable coordinate) M = gamma T: with Z0 = 0 the auxiliary stays zero and
the token step is exactly the existing first-order delta step with the same
eta. Moving raw_r at fixed (raw_eta, raw_tau) varies T at FIXED gamma and M,
so the new direction is the strength of the key-directed prospective term
T Rdot, not a re-scaling of the delta rate (which raw_eta already carries).
d rho / d raw_r = 1 at the start, on both sides.

Domain
------
    0 < rho < 1   (M < gamma T):  the previously certified sector, no extra
                                  condition.
    rho = 1       (M = gamma T):  exact delta.
    rho > 1       (M > gamma T):  the audit's sufficient switching bound
                                  d L < gamma^2, d = M - gamma T, L = 2 the
                                  executed gate bound, i.e. with x = eta tau L
                                     x <= 1 : every rho > 1
                                     x >  1 : rho < x / (x - 1)

THE WIDER SECTOR rho > 1 IS A COMPUTATIONAL APPLICATION OF THE SAME MECHANICAL
EQUATION. IT IS NOT THE PASSIVE TWO-COMPARTMENT CIRCUIT, which gives M <=
gamma T only. The bound is sufficient, not claimed maximal.

Enforced by POST-UPDATE projection of raw_r from the UPDATED eta and tau, with
a representation-aware margin on the full log ratio and rho = 1 as the exact
fallback:

    x' = x (1 + 16 eps)                      (rounding-conservative)
    U  = log(x'/(x'-1)) = -log1p(-1/x')      (x' > 1)
    raw_r <= max(0, U - 32 eps (1 + |U|))

No forward clip. Coefficients do not change within an episode.
"""

import math

import jax
import jax.numpy as jnp
import numpy as onp

from experiments.adaptive_memory import dynamics as AD

H = AD.H
#: executed source-gate bound: a = 2 sigmoid(.), times a {0,1} mask
GATE_BOUND_L = 2.0
EPS_NUM_FACTOR = 32.0
#: declared initial relaxation time tau0 = T0 at rho = 1 (see protocol s3)
TAU0 = 1.0

RULES = ("gp_two_sided", "adaptive_delta", "heavy_ball_same_mass",
         "tss_eq17", "gated_delta", "momentum_delta")
LITERATURE = ("gated_delta", "momentum_delta")
ORDINARY_PROSPECTIVE = ("tss_eq17",)
DISPLAY = {
    "gp_two_sided": "Generalized prospective memory, both sides of M = gamma T",
    "adaptive_delta": "Adaptive first-order delta memory (existing control)",
    "heavy_ball_same_mass": ("Heavy ball, T = 0: separately trained family, "
                             "gamma and M matched at initialization only"),
    "tss_eq17": ("TSS Eq. (17) applied directly to the fast weight; "
                 "applicability-limited"),
    "gated_delta": "Gated DeltaNet rule",
    "momentum_delta": "Momentum DeltaNet rule",
}
CARRY = {"gp_two_sided": 128, "adaptive_delta": 64,
         "heavy_ball_same_mass": 128, "tss_eq17": 128,
         "gated_delta": 64, "momentum_delta": 128}


def eps_num(dtype):
    return EPS_NUM_FACTOR * float(onp.finfo(onp.dtype(dtype)).eps)


# ------------------------------------------------------------ response -----
def two_sided_response(p):
    """Executed coefficients of the candidate, in the leaves' dtype."""
    eta = jnp.exp(p["raw_eta"])
    tau = jnp.exp(p["raw_tau"])
    rho = jnp.exp(p["raw_r"])
    return dict(eta=eta, tau=tau, rho=rho, nu=eta / rho,
                gamma=1.0 / eta, M=tau / eta, T=tau / rho,
                d=(tau / eta) * (1.0 - 1.0 / rho))


def two_sided_generator(eta, tau, rho, w):
    """(e, z) generator of the candidate, via the existing prospective
    generator with nu = eta / rho. Algebraically valid for every rho > 0; at
    rho = 1 its (2,1) entry is EXACTLY zero, so Z0 = 0 stays zero."""
    return AD.prospective_generator(eta / rho, tau, rho, w)


def log_rho_upper(eta, tau, dtype=None, L=GATE_BOUND_L):
    """Projection bound on raw_r from the executed eta and tau (log arithmetic).

    x = eta tau L. For x <= 1 the sufficient bound admits every rho > 1 and
    the result is +inf; that branch is masked BEFORE the logarithm.

    Precision. U = -log1p(-1/x) is ill-conditioned as x -> 1+: a relative
    rounding error e in x (the float product, 1/x, or a unit key whose squared
    norm rounds slightly above one, i.e. an effective L a few eps above 2)
    moves U by about e/(x - 1), which the margin 32 eps (1 + |U|) does NOT
    cover near x = 1. U decreases in x, so x is first INFLATED by the exactly
    representable factor (1 + 16 eps): the bound is then evaluated at a
    larger x than any rounding of the exact one, hence is not larger than the
    exact U. The 32 eps (1 + |U|) margin then covers the rounding of U itself
    and of rho = exp(raw_r). rho = 1 (raw_r = 0) remains the exact fallback.
    """
    dtype = dtype or eta.dtype
    e = float(onp.finfo(onp.dtype(dtype)).eps)
    x = eta * tau * L * jnp.asarray(1.0 + 16.0 * e, dtype=eta.dtype)
    active = x > 1.0
    x_safe = jnp.where(active, x, jnp.full_like(x, 2.0))
    U = -jnp.log1p(-1.0 / x_safe)
    margin = jnp.asarray(eps_num(dtype), dtype=U.dtype) * (1.0 + jnp.abs(U))
    bound = jnp.maximum(jnp.zeros_like(U), U - margin)
    return jnp.where(active, bound, jnp.full_like(bound, jnp.inf))


def project_two_sided(p):
    """Post-update projection of raw_r ONLY, from the updated eta and tau.
    Returns (params, telemetry). A no-op for every other rule's tree."""
    if "raw_r" not in p:
        z = jnp.asarray(0.0)
        return p, dict(n_projected=jnp.asarray(0), max_overshoot=z,
                       log_margin=jnp.asarray(jnp.inf))
    r = p["raw_r"]
    b = log_rho_upper(jnp.exp(p["raw_eta"]), jnp.exp(p["raw_tau"]),
                      r.dtype).astype(r.dtype)
    r_new = jnp.minimum(r, b)
    out = dict(p, raw_r=r_new)
    return out, dict(n_projected=jnp.sum(r > b),
                     max_overshoot=jnp.max(jnp.maximum(r - b, 0.0)),
                     log_margin=jnp.min(b - r_new))


def domain_report(p):
    """Validate EXECUTED arithmetic first, then the analytical certificate.

    Review R2 (535fb02):
      * executed-dtype response scalars are kept under `executed_*` names; the
        float64 reconstruction used ONLY for the certificate is kept under
        `certificate_f64_*` names, never overwriting executed values;
      * if any executed scalar is zero or non-finite the report FAILS before
        any division, instead of raising while diagnosing it;
      * the PRODUCTION `two_sided_generator` is formed in the executed dtype at
        the permitted gate endpoints w = 0 and w = L, and its entries must be
        finite; the executed idle coefficient a0 = exp(-h/tau) must be finite
        and in [0, 1] (zero is legitimate rounded rapid relaxation, F2);
      * the side of M = gamma T and the sufficient switching certificate are
        then evaluated in float64. Arithmetic checks and certificate are
        reported separately. Nothing is clamped.
    """
    c = two_sided_response(p)
    dtype = p["raw_r"].dtype
    executed = {k: float(onp.asarray(v).ravel()[0]) for k, v in c.items()}
    names = ("eta", "tau", "rho", "nu", "gamma", "M", "T")
    finite = all(onp.isfinite(executed[k]) for k in names + ("d",))
    positive = all(executed[k] > 0 for k in names)
    rep = dict({f"executed_{k}": v for k, v in executed.items()},
               executed_dtype=str(dtype), scalars_finite=bool(finite),
               scalars_positive=bool(positive),
               note=("rho > 1 is NOT the passive two-compartment circuit; "
                     "the certificate there is the sufficient switching "
                     "bound d L < gamma^2"))
    if not (finite and positive):
        return dict(rep, passed=False, failed="executed response scalar "
                    "zero, negative or non-finite", generator_finite=None,
                    certified=None, side=None)
    eta, tau, rho = c["eta"][0], c["tau"][0], c["rho"][0]
    gen_ok = True
    for w in (0.0, GATE_BOUND_L):
        G = onp.asarray(two_sided_generator(eta, tau, rho,
                                            jnp.asarray(w, dtype=dtype)))
        if G.dtype != onp.dtype(dtype) or not onp.all(onp.isfinite(G)):
            gen_ok = False
    a0 = onp.asarray(jnp.exp(-H / tau))
    # review F2 (f13295c): exp(-h/tau) may correctly round to ZERO for a
    # finite positive tau (rapid relaxation, e.g. tau = 1e-3). That is not an
    # invalid coefficient, so the executed idle factor needs only to be finite,
    # of the executed dtype, and in [0, 1]. Nothing is clamped.
    a0_ok = bool(onp.isfinite(a0) and 0.0 <= float(a0) <= 1.0
                 and a0.dtype == onp.dtype(dtype))
    rep.update(generator_finite=bool(gen_ok), idle_coefficient_a0=float(a0),
               idle_coefficient_finite=a0_ok)
    if not (gen_ok and a0_ok):
        return dict(rep, passed=False, certified=None, side=None,
                    failed="production generator or idle coefficient "
                           "non-finite in the executed dtype")
    e64 = {k: float(onp.float64(executed[k])) for k in ("eta", "tau", "rho")}
    g64, M64 = 1.0 / e64["eta"], e64["tau"] / e64["eta"]
    T64 = e64["tau"] / e64["rho"]
    d64 = M64 - g64 * T64
    if e64["rho"] < 1.0:
        side, cert = "passive_sector_M_lt_gammaT", True
    elif e64["rho"] == 1.0:
        side, cert = "exact_delta_boundary", True
    else:
        side = "wider_computational_sector_M_gt_gammaT"
        cert = bool(d64 * GATE_BOUND_L < g64 * g64)
    return dict(rep, side=side, certified=cert, passed=bool(cert),
                certificate_f64_gamma=g64, certificate_f64_M=M64,
                certificate_f64_T=T64, certificate_f64_d=d64,
                certificate_f64_dL_over_gamma2=d64 * GATE_BOUND_L / (g64 * g64))


def storage_V(X, Y, gamma, M, T):
    """Audit eq. (10) for d = M - gamma T > 0; singular at d = 0 (not used)."""
    d = M - gamma * T
    return (0.5 * gamma * onp.sum((X + Y / gamma) ** 2)
            + M / (2.0 * gamma * d) * onp.sum(Y ** 2))


# --------------------------------------------------------- TSS Eq. (17) ----
def eq17_step(W, f_prev, k, v, w, eta, T, h=H):
    """ACTUAL TSS Eq. (17) on the fast weight W, with f = W - eta R:

        R_k     = w_k (W_k k_k - v_k) k_k^T
        f_k     = W_k - eta R_k
        W_{k+1} = W_k + (h/T)(-W_k + f_k) + f_k - f_{k-1}

    Carry (W, f_prev). Clock h = 1 token interval. Cache initialization at an
    episode start: W_0 = 0 and f_{-1} = W_{-1} - eta R_{-1} = 0 (no observation
    before the episode). This is TSS's finite-step rule, NOT the earlier
    finite-adaptation ZOH arm and NOT the minimum-change projection.

    Applicability (protocol s6; review D1). With f(W) = W - eta R(W) for a
    single association, (I - Df)[X] = eta w (X k) k^T: singular in every
    direction orthogonal to the current key and ZERO on idle intervals, so
    the inverse used in TSS Eq. (15) does not exist here. Applying Eq. (17) is
    still a well-defined discrete experiment, labelled "TSS Eq. (17) applied
    directly to the fast weight; applicability-limited". It is NOT a
    reproduction of TSS's teaching-synchronization experiments. Its increment
    obeys

        Delta W_{k+1} = Delta W_k - eta (1 + h/T) R_k + eta R_{k-1},

    so the increment is preserved only after TWO consecutive zero-residual
    inputs; the first idle interval after a write still changes it. In the
    key direction its roots are stable iff 0 < eta w < 2 and
    eta w (2 + h/T) < 4.
    """
    R = w * jnp.outer(W @ k - v, k)
    f = W - eta * R
    W_next = W + (h / T) * (-W + f) + f - f_prev
    return W_next, f


def tss_eq17_response(p):
    return dict(eta=jnp.exp(p["raw_eta"]), T=jnp.exp(p["raw_T"]))
