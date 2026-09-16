"""Stable generalized prospective recurrence with prospective input.

Authoritative brief: NEXT_PROSPECTIVE_S5_CODING_BRIEF_2026_09_16.md. Binding
references: PROFESSOR_TO_GENERAL_PROSPECTIVITY_CONTRACT_2026_09_16.md and
GENERALIZED_RESPONSE_DOMAIN_AUDIT_2026_09_16.md. Nothing in this module trains
or reads data; it holds the mathematics the new arms and their checks share.

The candidate, per stored conjugate mode, with gamma normalized to one
----------------------------------------------------------------------

    rho T s'' + s' + R + T R' = 0 ,   R = j s - b (x + T_in x')
    j = -Delta lambda = a + i omega  (a > 0 after pole clipping),  b = Delta B_c
    T_in = 5 exp(q) ,  rho = exp(r) ,  T = 5 exp(t) ,  M = rho T  (derived)

DIRECT log coordinates, all initialized at zero. At r = 0 (rho = 1) the
denominator factors as (1 + T p)(p + j), so the model is the learned-input-
horizon control for EVERY q and t, and at q = r = 0 it is fixed-horizon
Rawat. No straight-through derivative and no static bypass: that identity is
a property of the executed algebra, checked numerically on the cluster.

Stability domain (reference 2, Section 2), NOT the passive sector
----------------------------------------------------------------

With c = 1 + T a, both roots of rho T p^2 + (1 + T j) p + j lie strictly in the
left half-plane iff

    S = a c^2 + omega^2 (T c - rho T) > 0 .

For omega = 0 every rho > 0 is allowed. For omega != 0 this is

    rho < rho_max = 1 + z ,   z = T a + a c^2 / (T omega^2) > 0 .

Since z > 0, rho = 1 is always strictly inside. The passive reciprocal
two-compartment identification holds only on rho <= 1 (M <= gamma T); the
wider domain is a stable NONCONSERVATIVE mechanical realization of the same
response equation (positive inertia and damping, a gyroscopic velocity force
and a circulatory positional force). It is never described as the passive
circuit, and arbitrary positive M is NOT claimed stable for a complex mode.

Declared numerical interior, enforced by POST-UPDATE projection of r only:

    r <= log1p((1 - epsilon_num) z) ,  epsilon_num = 32 * eps(executed dtype)

using the UPDATED poles, clock and T of the complete layer. Optimizer state is
left untouched. This excludes a rounding strip at the exact boundary; it is not
an additional biological or circuit sector.
"""

import math

import jax
import jax.numpy as jnp
import numpy as onp

#: raw leaf names, one real per stored complex mode, shared with its partner
LEAF_T_IN = "log_T_in"        # q : T_in = 5 exp(q)
LEAF_RHO = "log_rho_rec"      # r : rho  = exp(r)
LEAF_T = "log_T_rec"          # t : T    = 5 exp(t)
ADDED_LEAVES = (LEAF_T_IN, LEAF_RHO, LEAF_T)

#: Rawat's literature horizons, in native-clock intervals
INPUT_T_REFERENCE = 5.0
RECURRENT_T_REFERENCE = 5.0
#: the paper's pole clip, Appendix E.3; identical to S5SSM's forward clip
POLE_CLIP = -1e-4
#: declared numerical interior factor
EPS_NUM_FACTOR = 32.0


def eps_num(dtype):
    """32 * machine epsilon of the EXECUTED real dtype."""
    return EPS_NUM_FACTOR * float(onp.finfo(onp.dtype(dtype)).eps)


# ------------------------------------------------------------ modal j ------
def modal_j(Lambda_re, Lambda_im, log_step, clip=True):
    """Per-mode (a, omega) of j = -Delta lambda, from the RAW layer leaves.

    Reproduces exactly what the forward pass executes: S5SSM clips
    Re(lambda) <= -1e-4 before use, and SubstrateSSM uses step_rescale = 1 and
    Delta = exp(log_step[:, 0]). Reading the initializer fields or the unclipped
    pole would validate a different model.
    """
    lam_re = jnp.minimum(Lambda_re, POLE_CLIP) if clip else Lambda_re
    Delta = jnp.exp(log_step[:, 0])
    return -lam_re * Delta, -Lambda_im * Delta


def stability_S(a, omega, T, rho):
    """S = a c^2 + omega^2 (T c - rho T), c = 1 + T a. Positive iff stable."""
    c = 1.0 + T * a
    return a * c ** 2 + omega ** 2 * (T * c - rho * T)


def log_rho_upper(a, omega, T, dtype=None):
    """The projection bound on r, evaluated safely in log arithmetic.

        bound = log(1 + (1 - eps_num) z)
              = logaddexp(0, log1p(-eps_num) + log z)
        log z = logaddexp(log(T a), log a + 2 log c - log T - log omega^2)

    Real modes (omega^2 == 0 in the executed dtype) have no bound: +inf.
    Their division is MASKED before it is evaluated, so no inf/nan is ever
    formed, and the masked value never reaches the result.
    """
    dtype = dtype or a.dtype
    w2 = omega * omega
    complex_mode = w2 > 0
    w2_safe = jnp.where(complex_mode, w2, jnp.ones_like(w2))
    c = 1.0 + T * a
    log_z = jnp.logaddexp(jnp.log(T * a),
                          jnp.log(a) + 2.0 * jnp.log(c) - jnp.log(T)
                          - jnp.log(w2_safe))
    shift = jnp.asarray(math.log1p(-eps_num(dtype)), dtype=log_z.dtype)
    bound = jnp.logaddexp(jnp.zeros_like(log_z), shift + log_z)
    return jnp.where(complex_mode, bound, jnp.full_like(bound, jnp.inf))


def rho_max_float64(a, omega, T):
    """Host float64 rho_max = 1 + z (inf for real modes). For reporting only."""
    a = onp.asarray(a, dtype=onp.float64)
    omega = onp.asarray(omega, dtype=onp.float64)
    T = onp.asarray(T, dtype=onp.float64)
    c = 1.0 + T * a
    w2 = omega * omega
    with onp.errstate(divide="ignore", invalid="ignore"):
        z = T * a + onp.where(w2 > 0, a * c ** 2 / (T * onp.where(w2 > 0, w2,
                                                                   1.0)),
                              onp.inf)
    return onp.where(w2 > 0, 1.0 + z, onp.inf)


# ------------------------------------------------ coupled projection ------
def _group_layers(params):
    """{layer prefix: {leaf name: array}} for every layer carrying LEAF_RHO."""
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    layers = {}
    for k in flat:
        if k[-1] == LEAF_RHO:
            layers[k[:-1]] = {n: flat[k[:-1] + (n,)] for n in
                              ("Lambda_re", "Lambda_im", "log_step", LEAF_T,
                               LEAF_RHO)}
    return flat, layers


def project_stable_domain(params):
    """Post-update projection of r onto the declared stable interior.

    Operates on the COMPLETE updated layer: the bound is computed from the
    updated Lambda_re, Lambda_im, log_step and log_T_rec, because changing the
    common poles or T changes the feasible set. Only r is moved. Leaves of any
    arm without LEAF_RHO are returned unchanged.

    Returns (projected params, telemetry). Telemetry counts (entry, update)
    EVENTS, the largest PROPOSED overshoot beyond the bound, and the smallest
    post-projection log margin over complex modes.
    """
    from flax.traverse_util import unflatten_dict
    flat, layers = _group_layers(params)
    if not layers:
        z = jnp.asarray(0.0)
        return params, dict(n_projected=jnp.asarray(0), max_overshoot=z,
                            min_log_margin=jnp.asarray(jnp.inf),
                            n_passive=jnp.asarray(0), n_modes=jnp.asarray(0),
                            n_real_modes=jnp.asarray(0))
    out = dict(flat)
    moved, over, margin, passive, n_modes, n_real = [], [], [], [], [], []
    for prefix, lv in layers.items():
        r = lv[LEAF_RHO]
        a, omega = modal_j(lv["Lambda_re"], lv["Lambda_im"], lv["log_step"])
        T = RECURRENT_T_REFERENCE * jnp.exp(lv[LEAF_T])
        bound = log_rho_upper(a.astype(r.dtype), omega.astype(r.dtype),
                              T.astype(r.dtype), r.dtype).astype(r.dtype)
        r_new = jnp.minimum(r, bound)
        out[prefix + (LEAF_RHO,)] = r_new
        moved.append(jnp.sum(r > bound))
        over.append(jnp.max(jnp.maximum(r - bound, 0.0)))
        finite = jnp.isfinite(bound)
        margin.append(jnp.min(jnp.where(finite, bound - r_new, jnp.inf)))
        passive.append(jnp.sum(r_new <= 0.0))
        n_modes.append(r.size)
        n_real.append(jnp.sum(~finite))
    tel = dict(n_projected=jnp.sum(jnp.stack(moved)),
               max_overshoot=jnp.max(jnp.stack(over)),
               min_log_margin=jnp.min(jnp.stack(margin)),
               n_passive=jnp.sum(jnp.stack(passive)),
               n_modes=jnp.asarray(sum(n_modes)),
               n_real_modes=jnp.sum(jnp.stack(n_real)))
    return unflatten_dict(out), tel


def executed_domain_report(params):
    """Host-side validation of the EXECUTED coefficients, per layer.

    rho, T and M are formed in the executed dtype exactly as the forward pass
    forms them; stability is then evaluated in float64 from those executed
    values, so a float32 rounding across the boundary would be DETECTED rather
    than reproduced. Non-finite or underflowed mass/horizon fails. Nothing is
    clamped.
    """
    _, layers = _group_layers(params)
    rows, ok = [], True
    for prefix, lv in layers.items():
        r = lv[LEAF_RHO]
        rho = onp.asarray(jnp.exp(r))
        T = onp.asarray(RECURRENT_T_REFERENCE * jnp.exp(lv[LEAF_T]))
        a, omega = modal_j(lv["Lambda_re"], lv["Lambda_im"], lv["log_step"])
        a = onp.asarray(a, dtype=onp.float64)
        omega = onp.asarray(omega, dtype=onp.float64)
        M = rho.astype(onp.float64) * T.astype(onp.float64)
        rmax = rho_max_float64(a, omega, T)
        S = stability_S(a, omega, T.astype(onp.float64),
                        rho.astype(onp.float64))
        finite = bool(onp.all(onp.isfinite(rho)) and onp.all(onp.isfinite(T))
                      and onp.all(onp.isfinite(M)))
        positive = bool(onp.all(rho > 0) and onp.all(T > 0) and onp.all(M > 0))
        stable = bool(onp.all(S > 0))
        inside = bool(onp.all(rho < rmax))
        cplx = onp.isfinite(rmax)
        rel = onp.where(cplx, (rmax - rho) / onp.where(cplx, rmax, 1.0),
                        onp.inf)
        ok_layer = finite and positive and stable and inside
        ok = ok and ok_layer
        rows.append(dict(
            layer="/".join(prefix), n_modes=int(rho.size),
            n_real_modes=int(onp.sum(~cplx)),
            rho=_stats(rho), T=_stats(T), M=_stats(M),
            n_passive_rho_le_1=int(onp.sum(rho <= 1.0)),
            min_S=float(onp.min(S)),
            min_relative_margin_complex=(float(onp.min(rel[cplx]))
                                         if cplx.any() else None),
            finite=finite, positive=positive, stable_S_positive=stable,
            strictly_inside_rho_max=inside, passed=ok_layer))
    return dict(layers=rows, passed=ok,
                scope=("float64 evaluation of the EXECUTED float32 rho, T and "
                       "poles; passive subdomain is rho <= 1"))


def _stats(v):
    v = onp.asarray(v, dtype=onp.float64).ravel()
    return dict(min=float(v.min()), median=float(onp.median(v)),
                max=float(v.max()), mean=float(v.mean()))


# ------------------------------------------ continuous transfer (checks) ---
def transfer(p, j, b, T, rho, T_in):
    """G(p) = b (1 + T p)(1 + T_in p) / [rho T p^2 + (1 + T j) p + j]."""
    return b * (1 + T * p) * (1 + T_in * p) / (rho * T * p * p
                                                 + (1 + T * j) * p + j)


# ------------------------------------------------------ TSS Eq. (17) -------
def tss_eq17_step(s, f, f_prev, h, T):
    """ACTUAL TSS Eq. (17): s_{k+1} = s_k + (h/T)(-s_k + f_k) + f_k - f_{k-1}.

    This is the professor's discrete rule. It is NOT the exact continuous
    memoryless reduction r + T r' = 0 (the `prospective_recurrence` arm), which
    has no history by construction; Eq. (17) can retain discrete history.
    """
    return s + (h / T) * (-s + f) + f - f_prev


def generalized_discrete_step(s, s_prev, f, f_prev, h, M, gamma, T):
    """The generalized finite-difference descendant, d = s_{k+1} - s_k:

        [M + h(gamma + T)] d = M d_{k-1} + h^2 (f_k - s_k) + h T (f_k - f_{k-1})

    At M = gamma = 0 it is Eq. (17) exactly, including the previous-drive
    convention. This is an EQUATION-LEVEL reference; the production held-token
    ZOH law is a different discretization and is not claimed identical to it.
    """
    d = (M * (s - s_prev) + h * h * (f - s) + h * T * (f - f_prev)) \
        / (M + h * (gamma + T))
    return s + d
