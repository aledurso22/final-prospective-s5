"""Residual-processing placement of the master law, with its exact literal
TSS boundary.

Specification: docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md. Frozen protocol:
docs/PROSPECTIVE_TSS_CONTAINMENT_PROTOCOL.md. Clearance:
PROSPECTIVE_TSS_IMPLEMENTATION_CLEARANCE_d95266d_2026_09_17.md; implementation
review: PROSPECTIVE_TSS_IMPLEMENTATION_REVIEW_7613c86_2026_09_17.md (R2).

A processing state y is driven by the masked associative residual, and its
NEWLY COMPUTED output enters the UNCHANGED Momentum update:

    R_t     = m_t (alpha_t W_prev k_t - v_t) k_t^T
    A (y_next - y) = M (y - y_prev) + h^2 (R_t - y) + h T (R_t - R_prev)
    A       = M + h (gamma + T)
    U_next  = mu_t U_prev + eta_t y_next
    W_next  = alpha_t W_prev - beta_t U_next

h = 1 token; every right-hand side uses the OLD states.

EXECUTED FORM (review R2). The same equation is executed in its explicit,
algebraically equivalent coefficient form

    a = [2M + h(gamma+T) - h^2] / A     b = M / A
    c = [h^2 + hT] / A                  d = hT / A
    y_next = a y - b y_prev + c R_t - d R_prev,

whose normalized coefficients are formed ONCE per compiled program by
`coefficients`, inside the rollout, with no parameter-dependent branch; the
derivatives with respect to M, gamma and T flow through them everywhere,
including on the boundaries. The rollout returns exactly those rounded values,
and acceptance classifies exactly those values (not basis responses of the
step, which floating-point non-linearity does not turn into coefficients).
The original-law form survives only in separately coded references
(`filtered_reference` here, the sensitivity recursions in the checks).

Exact points (hand-proved in the specification, checked on the cluster):

  * M = gamma = 0, T > 0: a = 1 - h/T, b = 0, c = 1 + h/T, d = 1, i.e.
    y_next = y + (h/T)(R_t - y) + R_t - R_prev, literal TSS Eq. (17) driven by R;
  * (M, gamma, T) = (0, h, 0): a = b = d = 0, c = 1, so y_next = R_t and the
    Momentum update is native. The executed form avoids the y + (R - y)
    cancellation of the original form;
  * M = 0, gamma + T = h: the two-tap operator with kappa = T/h and
    gamma = h(1 - kappa); kappa > 1 needs gamma < 0 and is OUTSIDE this
    nonnegative-gamma family.

Carry (W, U, y, y_prev, R_prev) = 320 real numbers as IMPLEMENTED, in both
processing arms. 256 is only the theoretical minimum of a law with M fixed at
zero; it is not the implemented cost.

Acceptance. The derivation's domain is M, gamma, T >= 0, gamma + T > 0, A > 0,
4M + 2h(gamma+T) > h^2. The declared gaps G_MIN and DELTA_FILTER are a
numerical robustness policy enforced by the repair, without any universal
rounding guarantee. The GATE is: every executed quantity (M, gamma, T, A, a,
b, c, d) finite, and the three strict Jury conditions of the rounded
homogeneous polynomial z^2 - a z + b,

    1 - a + b > 0,   1 + a + b > 0,   1 - b > 0,

in the executed dtype after every update and by exact rational classification
at every checkpoint. This is a result about those rounded coefficients only:
not a theorem about every floating-point trajectory, the closed-loop memory,
switching across tokens, or the Momentum block below.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as onp

from experiments.nested_memory import model as NM

from . import dynamics as PD

FILTERED = "filtered_processing"
DISPLAY = ("Master-law residual processing (M, gamma, T) feeding the "
           "unchanged Momentum update")
#: token interval
H = 1.0
#: declared numerical gaps (policy, not derivation); see the specification s2.1
G_MIN = 2.0 ** -10          # in units of h: gamma + T >= G_MIN * h
DELTA_FILTER = 1e-3         # 4M + 2h(gamma+T) >= h^2 (1 + DELTA_FILTER)
#: the three stored coefficient leaves, in this order
LEAVES = ("fil_M", "fil_gamma", "fil_T")
#: W, U, y, y_prev, R_prev as executed
CARRY_EXECUTED = 5 * NM.D_V * NM.D_K
#: what a law with M fixed at zero would need (y_prev dormant)
CARRY_MINIMAL_M_ZERO = 4 * NM.D_V * NM.D_K
#: declared start: literal TSS at T0 = h (also the two-tap point kappa = 1)
T0 = H


# additive registry entries so any importer (model, study, tests) sees this
# law's display name and executed carry; no existing entry is modified
PD.DISPLAY.setdefault(FILTERED, DISPLAY)
PD.CARRY.setdefault(FILTERED, CARRY_EXECUTED)


def initial_leaves(dtype, T_value=T0):
    """The declared start point: M = gamma = 0 exactly, T = T_value."""
    return {"fil_M": jnp.zeros((1,), dtype=dtype),
            "fil_gamma": jnp.zeros((1,), dtype=dtype),
            "fil_T": jnp.full((1,), T_value, dtype=dtype)}


#: the executed quantities, in the order `coefficients` returns them
COEFF_NAMES = ("M", "gamma", "T", "A", "a", "b", "c", "d")


def coefficients(p):
    """The executed coefficients, formed ONCE from the stored leaves in their
    dtype, with no branch on their values. Returned as a tuple in COEFF_NAMES
    order; the rollout multiplies by exactly these values and returns them."""
    M, gam, T = (p[k][0] for k in LEAVES)
    h = jnp.asarray(H, dtype=M.dtype)
    hh = h * h
    A = M + h * (gam + T)
    a = (2.0 * M + h * (gam + T) - hh) / A
    b = M / A
    c = (hh + h * T) / A
    d = (h * T) / A
    return M, gam, T, A, a, b, c, d


# ------------------------------------------------------------- updates -----
def filtered_update(Wb, U, y, y_prev, Rprev, beta, mu, eta, coeff, R):
    """One step given Wbar and the masked residual R, in the executed
    coefficient form."""
    _, _, _, _, a, b, c, d = coeff
    y_new = a * y - b * y_prev + c * R - d * Rprev
    U_new = mu * U + eta * y_new
    return Wb - beta * U_new, U_new, y_new, y, R


def filtered_step(carry, k, v, m, alpha, beta, mu, eta, coeff):
    W, U, y, y_prev, Rprev = carry
    Wb = alpha * W
    R = m * jnp.outer(Wb @ k - v, k)
    return filtered_update(Wb, U, y, y_prev, Rprev, beta, mu, eta, coeff, R)


# ------------------------------------------------- executed filter gate ----
def jury_slacks(a, b):
    """(1 - a + b, 1 + a + b, 1 - b) of z^2 - a z + b, in the dtype of a, b.
    All three strictly positive is Schur stability of the quadratic
    (b > -1 follows from the sum of the first two)."""
    return 1.0 - a + b, 1.0 + a + b, 1.0 - b


def in_loop_guard(coeff):
    """Executed-dtype guard for coefficients a compiled program actually
    used. `ok` is False if ANY executed quantity is non-finite or any slack is
    not strictly positive."""
    M, gam, T, A, a, b, c, d = coeff
    s1, s2, s3 = jury_slacks(a, b)
    smin = jnp.minimum(jnp.minimum(s1, s2), s3)
    finite = jnp.all(jnp.isfinite(jnp.stack(
        [jnp.asarray(x) for x in coeff] + [s1, s2, s3])))
    return dict(zip(COEFF_NAMES, coeff), jury_1=s1, jury_2=s2, jury_3=s3,
                jury_min=smin, finite=finite, ok=finite & (smin > 0))


def classify_quadratic(a, b):
    """Exact rational classification of the ROUNDED coefficients a, b.
    stable: all three slacks > 0; neutral: none negative, one zero;
    unstable: any negative; nonfinite: a or b non-finite."""
    if not (onp.isfinite(a) and onp.isfinite(b)):
        return "nonfinite", None
    fa, fb = Fraction(float(a)), Fraction(float(b))
    J = (1 - fa + fb, 1 + fa + fb, 1 - fb)
    if min(J) < 0:
        return "unstable", tuple(float(j) for j in J)
    if min(J) == 0:
        return "neutral", tuple(float(j) for j in J)
    return "stable", tuple(float(j) for j in J)


def coefficient_values(coeff):
    """Host floats of an executed coefficient tuple or mapping, WITHOUT any
    recomputation (a scalar per name; batched rollout copies are checked equal
    by the caller)."""
    if isinstance(coeff, dict):
        return {k: float(onp.asarray(coeff[k]).ravel()[0])
                for k in COEFF_NAMES}
    return {k: float(onp.asarray(v).ravel()[0])
            for k, v in zip(COEFF_NAMES, coeff)}


def filter_report(executed, source="executed by the evaluation rollout"):
    """Acceptance report for EXECUTED coefficient values (a mapping from
    COEFF_NAMES to host floats taken from a compiled program). The declared
    gaps are reported alongside; they are never the certificate."""
    v = dict(executed)
    label, slacks = classify_quadratic(v["a"], v["b"])
    M, gam, T = v["M"], v["gamma"], v["T"]
    h = H
    finite = all(onp.isfinite(v[k]) for k in COEFF_NAMES)
    return dict(rule=FILTERED, source=source, executed=v,
                all_finite=bool(finite),
                M=M, gamma=gam, T=T, A=v["A"], a=v["a"], b=v["b"],
                c=v["c"], d=v["d"], classification=label,
                jury_slacks_exact=(list(slacks) if slacks is not None
                                   else None),
                min_jury_expression=(min(slacks) if slacks is not None
                                     else None),
                gap_gamma_plus_T=gam + T - G_MIN * h,
                gap_filter=4.0 * M + 2.0 * h * (gam + T) - h * h
                * (1.0 + DELTA_FILTER),
                nonnegative=bool(M >= 0.0 and gam >= 0.0 and T >= 0.0),
                at_tss_boundary=bool(M == 0.0 and gam == 0.0),
                at_native_point=bool(M == 0.0 and T == 0.0 and gam == h),
                note=("acceptance classifies the rounded coefficients a, b "
                      "that the named compiled program executed; the gaps "
                      "G_MIN and DELTA_FILTER are a robustness policy, "
                      "reported, not certified. A coefficient-polynomial "
                      "result for the isolated filter only."))


def report_from_tree(p):
    """DIAGNOSTIC/TEST helper: recompute the coefficients from a tree in THIS
    context and report them. Not the study's acceptance record, which always
    uses values returned by the compiled evaluation rollout."""
    return filter_report(coefficient_values(coefficients(p)),
                         source="recomputed from the stored leaves "
                                "(diagnostic, not the executed record)")


def filter_failure(rep):
    """None if acceptable; any non-finite executed quantity, a negative
    coefficient, or a rounded polynomial that is not strictly stable fails."""
    if not rep["all_finite"]:
        return f"non-finite executed filter quantities {rep['executed']}"
    if not rep["nonnegative"]:
        return (f"negative coefficient: M={rep['M']} gamma={rep['gamma']} "
                f"T={rep['T']} (the family is nonnegative; kappa > 1 of the "
                "two-tap operator is outside it)")
    if rep["classification"] != "stable":
        return (f"executed filter polynomial not strictly stable: "
                f"{rep['classification']} slacks {rep['jury_slacks_exact']} "
                f"a={rep['a']} b={rep['b']} A={rep['A']}")
    return None


# ------------------------------------------------- feasibility repair ------
def repair(p, frozen=()):
    """Deterministic post-update feasibility repair (specification s2.2).

    Component clamping, then a MINIMAL increase of T to restore the declared
    gaps, in this order. It is NOT an orthogonal or Euclidean projection and
    does not return the nearest admissible point; repairing through T is the
    declared asymmetric choice, because raising T always restores both gaps
    and never leaves the nonnegative orthant. Leaves named in `frozen` are
    restored to their incoming values, bitwise.

    The optimizer state is untouched. Telemetry records the pre-repair
    proposals, the post values, the number of changed coordinates and the
    achieved gaps."""
    dt = p["fil_M"].dtype
    h = jnp.asarray(H, dtype=dt)
    gmin = jnp.asarray(G_MIN, dtype=dt) * h
    delta = jnp.asarray(DELTA_FILTER, dtype=dt)
    pre = {k: p[k][0] for k in LEAVES}
    M = jnp.maximum(pre["fil_M"], 0.0)
    gam = jnp.maximum(pre["fil_gamma"], 0.0)
    T = jnp.maximum(pre["fil_T"], 0.0)
    T = jnp.maximum(T, gmin - gam)                                   # (N1)
    T = jnp.maximum(T, ((h * h) * (1.0 + delta) - 4.0 * M)
                    / (2.0 * h) - gam)                               # (N2)
    post = {"fil_M": M, "fil_gamma": gam, "fil_T": T}
    for name in frozen:
        if name in post:
            post[name] = pre[name]
    out = dict(p, **{k: jnp.asarray([v], dtype=dt) for k, v in post.items()})
    changed = sum(jnp.asarray(post[k] != pre[k], dtype=jnp.int32)
                  for k in LEAVES)
    tel = dict(n_repaired=changed,
               overshoot=jnp.max(jnp.stack([jnp.abs(post[k] - pre[k])
                                            for k in LEAVES])),
               gap_gamma_plus_T=gam + T - gmin,
               gap_filter=4.0 * M + 2.0 * h * (gam + T)
               - (h * h) * (1.0 + delta))
    for k in LEAVES:
        tel[k + "_pre"] = pre[k]
        tel[k + "_post"] = post[k]
    return out, tel


# ------------------------------------------------- reference recursions ----
def filtered_reference(residuals, M, gamma, T, h=H):
    """CHECK-ONLY independent float64 host recursion of the processing state,
    with the specified episode initialization y = y_prev = R_prev = 0 and the
    same-token output convention (entry t is produced by token t)."""
    R = [onp.asarray(x, onp.float64) for x in residuals]
    A = M + h * (gamma + T)
    y = onp.zeros_like(R[0])
    y_prev = onp.zeros_like(R[0])
    prev = onp.zeros_like(R[0])
    out = []
    for r in R:
        y_new = y + (M * (y - y_prev) + h * h * (r - y)
                     + h * T * (r - prev)) / A
        y_prev, y, prev = y, y_new, r
        out.append(y.copy())
    return out


def two_tap_mapping(kappa, h=H):
    """(M, gamma, T) of the ordinary two-tap operator with horizon kappa.
    gamma < 0 for kappa > 1: outside this family, by construction."""
    return 0.0, h * (1.0 - kappa), kappa * h


def mapping_admissible(kappa):
    """Whether the two-tap horizon maps into the nonnegative-gamma family."""
    return bool(0.0 <= kappa <= 1.0)


def native_point(h=H):
    """(M, gamma, T) at which the placement is exactly native Momentum."""
    return 0.0, h, 0.0


def tss_boundary(T_value=T0):
    """(M, gamma, T) on the literal TSS boundary."""
    return 0.0, 0.0, T_value
