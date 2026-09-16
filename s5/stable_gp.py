"""Stable generalized prospective recurrence with prospective input.

Authoritative brief: NEXT_PROSPECTIVE_S5_CODING_BRIEF_2026_09_16.md. Binding
references: PROFESSOR_TO_GENERAL_PROSPECTIVITY_CONTRACT_2026_09_16.md and
GENERALIZED_RESPONSE_DOMAIN_AUDIT_2026_09_16.md. Nothing in this module trains
or reads data; it holds the mathematics the new arms and their checks share.

The candidate, per stored conjugate mode, with gamma normalized to one
----------------------------------------------------------------------

    rho T s'' + s' + R + T R' = 0 ,   R = j s - b (x + T_in x')
    j = -Delta lambda = a + i omega  (a > 0 after pole clipping),  b = Delta B_c
    T_in = 5 exp(q) ,  rho = exp(r) ,  T = 10 exp(t) ,  M = rho T  (derived)

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

Declared numerical interior, enforced by POST-UPDATE projection of r only
(AMENDED before execution, review R1 - see the protocol):

    L = log(rho_max) = log1p(z)
    r <= max(0, L - 32 eps (1 + |L|))      complex modes
    r unbounded                            exactly real modes (omega == 0)

using the UPDATED poles, clock and T of the complete layer. The margin is on the
scale of the FULL log ratio, and rho = 1 is kept exactly as the known stable
fallback. The coordinator's original prescription, log1p((1 - 32 eps) z),
shrank only the EXCESS z; for small z its safety distance is below one float32
step of rho around one, so a correctly rounded exp could land beyond the exact
boundary. Optimizer state is left untouched. This is a conservative numerical
interior, not an additional biological or circuit sector.

RECURRENT REFERENCE HORIZON (AMENDED before execution, review R0): T = 10 exp(t)
while T_in = 5 exp(q). The original brief set both references to 5. At r = 0
and T = T_in the tangent

    dG/dr = -b T p^2 (1 + T_in p) / [(1 + T p)(p + j)^2]

loses the auxiliary pole -1/T, so the new temporal response would enter the
first-order learning direction only at second order. With T != T_in its
residue there is -b (1 - T_in/T) / [T^2 (j - 1/T)^2], nonzero for nonzero drive.
Ten is a transparent nondegenerate initialization, not a measured or predicted
optimum. C still starts at EXACTLY B's function, since rho = 1 removes T.
"""

import math

import jax
import jax.numpy as jnp
import numpy as onp

#: raw leaf names, one real per stored complex mode, shared with its partner
LEAF_T_IN = "log_T_in"        # q : T_in = 5 exp(q)
LEAF_RHO = "log_rho_rec"      # r : rho  = exp(r)
LEAF_T = "log_T_rec"          # t : T    = 10 exp(t)  (review R0)
ADDED_LEAVES = (LEAF_T_IN, LEAF_RHO, LEAF_T)

#: Rawat's literature horizons, in native-clock intervals
INPUT_T_REFERENCE = 5.0
#: review R0: distinct from the input reference so the first-order r tangent
#: carries the auxiliary pole; a declared initialization, not an optimum
RECURRENT_T_REFERENCE = 10.0
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
    """The projection bound on r (review R1), in log arithmetic.

        log z = logaddexp(log(T a), log a + 2 log c - log T - 2 log|omega|)
        L     = log(rho_max) = log1p(z) = logaddexp(0, log z)
        bound = max(0, L - 32 eps (1 + |L|))

    `2 log|omega|` replaces `log(omega^2)`, so a genuinely complex mode whose
    omega^2 would underflow is NOT misread as real. Only omega == 0 exactly is
    unbounded. The inactive division is masked BEFORE evaluation, and
    max(0, .) keeps rho = 1 as the exact stable fallback.
    """
    dtype = dtype or a.dtype
    complex_mode = omega != 0
    abs_w = jnp.where(complex_mode, jnp.abs(omega), jnp.ones_like(omega))
    c = 1.0 + T * a
    log_z = jnp.logaddexp(jnp.log(T * a),
                          jnp.log(a) + 2.0 * jnp.log(c) - jnp.log(T)
                          - 2.0 * jnp.log(abs_w))
    L = jnp.logaddexp(jnp.zeros_like(log_z), log_z)
    margin = jnp.asarray(eps_num(dtype), dtype=L.dtype) * (1.0 + jnp.abs(L))
    bound = jnp.maximum(jnp.zeros_like(L), L - margin)
    return jnp.where(complex_mode, bound, jnp.full_like(bound, jnp.inf))


def rho_max_float64(a, omega, T):
    """Host float64 rho_max = 1 + z (inf for real modes). For reporting only."""
    a = onp.asarray(a, dtype=onp.float64)
    omega = onp.asarray(omega, dtype=onp.float64)
    T = onp.asarray(T, dtype=onp.float64)
    c = 1.0 + T * a
    cplx = omega != 0                     # same classification as the bound
    w2 = onp.where(cplx, omega * omega, 1.0)
    with onp.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = T * a + a * c ** 2 / (T * w2)
    return onp.where(cplx, 1.0 + z, onp.inf)


# ------------------------------------------------ coupled projection ------
def _group_layers(params, marker=LEAF_RHO):
    """{layer prefix: {leaf name: array}} for every layer carrying `marker`."""
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    names = {LEAF_RHO: ("Lambda_re", "Lambda_im", "log_step", LEAF_T, LEAF_RHO,
                        LEAF_T_IN),
             LEAF_T_IN: (LEAF_T_IN,)}[marker]
    layers = {}
    for k in flat:
        if k[-1] == marker:
            layers[k[:-1]] = {n: flat[k[:-1] + (n,)] for n in names
                              if k[:-1] + (n,) in flat}
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
    """Validate the EXECUTED arithmetic, then assess it in float64 (review R4).

    Every quantity is first formed in the executed dtype exactly as the forward
    pass forms it - T_in = 5 exp(q), rho = exp(r), T = 10 exp(t), M = rho T,
    and the block generator entries of `mass_block_generator` - and those
    ROUNDED values are what is checked:

      * finite and strictly positive a, T, rho, M and T_in; finite omega;
      * finite generator entries;
      * the FORMULA diagnostic S > 0 and rho < rho_max, evaluated in float64
        from the executed a, omega, T and rho;
      * separately, the EIGENVALUES of the executed generator, promoted to
        complex128: max Re < 0.

    Covers B (T_in only) and C. A layer passes only if all apply and hold. An
    underflowed horizon or an overflowed mass fails even if a float64
    reconstruction of the product would not. Nothing is clamped.
    """
    rows, ok = [], True
    _, c_layers = _group_layers(params, LEAF_RHO)
    _, q_layers = _group_layers(params, LEAF_T_IN)
    for prefix, lv in q_layers.items():
        if prefix in c_layers:
            continue
        T_in = onp.asarray(INPUT_T_REFERENCE * jnp.exp(lv[LEAF_T_IN]))
        good = bool(onp.all(onp.isfinite(T_in)) and onp.all(T_in > 0))
        ok = ok and good
        rows.append(dict(layer="/".join(prefix), kind="input_horizon_only",
                         T_in=_stats(T_in), T_in_finite_positive=good,
                         passed=good))
    for prefix, lv in c_layers.items():
        dt = lv[LEAF_RHO].dtype
        rho_e = jnp.exp(lv[LEAF_RHO])
        T_e = RECURRENT_T_REFERENCE * jnp.exp(lv[LEAF_T])
        M_e = rho_e * T_e
        T_in_e = INPUT_T_REFERENCE * jnp.exp(lv[LEAF_T_IN])
        a_e, w_e = modal_j(lv["Lambda_re"], lv["Lambda_im"], lv["log_step"])
        j_e = a_e + 1j * w_e
        A_e = onp.asarray(jnp.stack([
            jnp.stack([-j_e / rho_e, -((1.0 - rho_e) / rho_e) + 0j], -1),
            jnp.stack([-j_e / (rho_e * T_e), -(1.0 / (rho_e * T_e)) + 0j], -1),
        ], -2))
        rho, T, M, T_in = (onp.asarray(v) for v in (rho_e, T_e, M_e, T_in_e))
        a, omega = onp.asarray(a_e), onp.asarray(w_e)

        def fp(v):
            return bool(onp.all(onp.isfinite(v)) and onp.all(v > 0))
        executed_ok = (fp(a) and fp(T) and fp(rho) and fp(M) and fp(T_in)
                       and bool(onp.all(onp.isfinite(omega)))
                       and bool(onp.all(onp.isfinite(A_e))))
        a64, w64 = a.astype(onp.float64), omega.astype(onp.float64)
        T64, rho64 = T.astype(onp.float64), rho.astype(onp.float64)
        with onp.errstate(all="ignore"):
            S = stability_S(a64, w64, T64, rho64)
            rmax = rho_max_float64(a64, w64, T64)
            ev = (onp.linalg.eigvals(A_e.astype(onp.complex128))
                  if executed_ok else onp.full((rho.size, 2), onp.nan))
        formula_ok = bool(executed_ok and onp.all(S > 0)
                          and onp.all(rho64 < rmax))
        max_re = onp.max(ev.real, axis=-1)
        eig_ok = bool(executed_ok and onp.all(onp.isfinite(max_re))
                      and onp.all(max_re < 0))
        cplx = onp.isfinite(rmax)
        rel = onp.where(cplx, (rmax - rho64) / onp.where(cplx, rmax, 1.0),
                        onp.inf)
        ok_layer = executed_ok and formula_ok and eig_ok
        ok = ok and ok_layer
        rows.append(dict(
            layer="/".join(prefix), kind="stable_generalized",
            executed_dtype=str(dt), n_modes=int(rho.size),
            n_real_modes=int(onp.sum(~cplx)),
            rho=_stats(rho), T=_stats(T), M=_stats(M), T_in=_stats(T_in),
            n_passive_rho_le_1=int(onp.sum(rho <= 1.0)),
            min_S=float(onp.min(S)) if executed_ok else None,
            min_relative_margin_complex=(float(onp.min(rel[cplx]))
                                         if (executed_ok and cplx.any())
                                         else None),
            max_real_part_executed_generator=(float(onp.max(max_re))
                                              if eig_ok or executed_ok
                                              else None),
            executed_finite_positive=executed_ok,
            formula_stable=formula_ok, executed_generator_stable=eig_ok,
            passed=ok_layer))
    return dict(layers=rows, passed=ok,
                scope=("executed-dtype arithmetic checked first; the formula "
                       "diagnostic and the executed-generator eigenvalues are "
                       "assessed separately in float64; passive subdomain is "
                       "rho <= 1"))


def executed_coefficients(params):
    """Executed-dtype T_in, rho, T and M per layer, for summaries (review R4)."""
    out = {}
    _, c_layers = _group_layers(params, LEAF_RHO)
    _, q_layers = _group_layers(params, LEAF_T_IN)
    for prefix, lv in q_layers.items():
        rec = dict(T_in=onp.asarray(INPUT_T_REFERENCE
                                    * jnp.exp(lv[LEAF_T_IN])).tolist())
        if prefix in c_layers:
            cl = c_layers[prefix]
            rho = jnp.exp(cl[LEAF_RHO])
            T = RECURRENT_T_REFERENCE * jnp.exp(cl[LEAF_T])
            rec.update(rho=onp.asarray(rho).tolist(),
                       T=onp.asarray(T).tolist(),
                       M=onp.asarray(rho * T).tolist())
        out["/".join(prefix)] = rec
    return out


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
