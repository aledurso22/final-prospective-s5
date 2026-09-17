"""Residual-processing placement of the master law, with its exact literal
TSS boundary.

Specification, frozen for review before any execution:
docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md (commit d95266d, corrected for the
static review of bdc1c19) and docs/PROSPECTIVE_TSS_CONTAINMENT_PROTOCOL.md.
Clearance: PROSPECTIVE_TSS_IMPLEMENTATION_CLEARANCE_d95266d_2026_09_17.md.

A processing state y is driven by the masked associative residual, and its
NEWLY COMPUTED output enters the UNCHANGED Momentum update:

    R_t     = m_t (alpha_t W_prev k_t - v_t) k_t^T
    A (y_next - y) = M (y - y_prev) + h^2 (R_t - y) + h T (R_t - R_prev)
    A       = M + h (gamma + T)
    U_next  = mu_t U_prev + eta_t y_next
    W_next  = alpha_t W_prev - beta_t U_next

h = 1 token. Every right-hand side uses the OLD states. Carry:
(W, U, y, y_prev, R_prev) = 320 real numbers as executed; a law with M fixed
at zero needs only 256, and both numbers are reported (`CARRY_EXECUTED`,
`CARRY_MINIMAL_M_ZERO`).

Exact points, proved by hand in the specification and checked numerically on
the cluster:

  * M = gamma = 0, T > 0:  y_next = y + (h/T)(R_t - y) + R_t - R_prev, which
    is literal TSS Eq. (17) driven by R at the same clock, initialization,
    same-token output convention and downstream gates;
  * M = 0, T = 0, gamma = h:  y_next = R_t, exactly the native Momentum
    update in real arithmetic (the executed float form is y + (R - y), so
    this recovery is checked at trajectory tolerances, never bitwise);
  * M = 0, gamma + T = h:  y_next = R_t + (T/h)(R_t - R_prev), the ordinary
    two-tap operator with kappa = T/h, so gamma = h(1 - kappa). kappa > 1
    needs gamma < 0 and is therefore OUTSIDE this nonnegative-gamma family:
    the learned two-tap operator stays a separate comparator.

Domain. The derivation requires M >= 0, gamma >= 0, T >= 0, gamma + T > 0,
A > 0 and 4M + 2h(gamma+T) > h^2 (the strict Jury conditions of the isolated
filter). The executed arithmetic uses declared positive gaps G_MIN and
DELTA_FILTER instead of strict comparisons; these are a NUMERICAL ROBUSTNESS
POLICY, not part of the physical derivation, and they do not by themselves
certify the rounded recurrence. Acceptance is the executed gate below.

Executed acceptance gate (clearance s1). The rounded normalized recurrence
coefficients are obtained by applying the PRODUCTION step to basis carries
with no residual, so they are exactly the coefficients the rollout executes:

    executed 2x2 = [[c1, -c0], [1, 0]],  polynomial z^2 - c1 z + c0.

Finiteness and all three strict Jury conditions of THAT rounded polynomial
are required, in the loop (executed dtype) and at every validation point
(exact rational classification). A violating update or run is refused; no
algebraically equivalent pre-rounding expression is substituted. This
certifies the isolated executed filter only - not the closed-loop associative
memory, not switching across tokens, and not the downstream Momentum block.
"""

from fractions import Fraction

import jax
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


def coefficients(p):
    """(M, gamma, T, A) in the leaves' dtype, with A formed ONCE here and
    used by the step, the telemetry and the acceptance gate."""
    M, gam, T = (p[k][0] for k in LEAVES)
    h = jnp.asarray(H, dtype=M.dtype)
    return M, gam, T, M + h * (gam + T)


# ------------------------------------------------------------- updates -----
def filtered_update(Wb, U, y, y_prev, Rprev, beta, mu, eta, coeff, R):
    """One step given Wbar and the masked residual R (open-loop form)."""
    M, gam, T, A = coeff
    h = jnp.asarray(H, dtype=jnp.asarray(A).dtype)
    y_new = y + (M * (y - y_prev) + (h * h) * (R - y)
                 + (h * T) * (R - Rprev)) / A
    U_new = mu * U + eta * y_new
    return Wb - beta * U_new, U_new, y_new, y, R


def filtered_step(carry, k, v, m, alpha, beta, mu, eta, coeff):
    W, U, y, y_prev, Rprev = carry
    Wb = alpha * W
    R = m * jnp.outer(Wb @ k - v, k)
    return filtered_update(Wb, U, y, y_prev, Rprev, beta, mu, eta, coeff, R)


# ------------------------------------------------- executed filter gate ----
def executed_filter_transition(coeff, dtype):
    """The rounded 2x2 (y, y_prev) transition of the PRODUCTION step, taken
    with an inactive write (m = 0, so R = 0) from basis carries. These are
    exactly the coefficients the rollout executes; nothing is recomputed from
    an algebraically equivalent expression."""
    d_k, d_v = NM.D_K, NM.D_V
    k = jnp.zeros((d_k,), dtype=dtype).at[0].set(1)
    v = jnp.zeros((d_v,), dtype=dtype)
    e = jnp.zeros((d_v, d_k), dtype=dtype).at[0, 0].set(1)
    z = jnp.zeros((d_v, d_k), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)

    def col(y, y_prev):
        _, _, y_new, y_prev_new, _ = filtered_step(
            (z, z, y, y_prev, z), k, v, zero, one, zero, one, zero, coeff)
        return y_new[0, 0], y_prev_new[0, 0]

    (a11, a21), (a12, a22) = col(e, z), col(z, e)
    return jnp.stack([jnp.stack([a11, a12]), jnp.stack([a21, a22])])


def jury_slacks(A2):
    """(1 - tr + det, 1 + tr + det, 1 - det) of a rounded 2x2, in ITS dtype.
    All three strictly positive is Schur stability for a real 2x2."""
    tr = A2[0, 0] + A2[1, 1]
    det = A2[0, 0] * A2[1, 1] - A2[0, 1] * A2[1, 0]
    return 1.0 - tr + det, 1.0 + tr + det, 1.0 - det


def in_loop_guard(coeff, dtype):
    """Executed-dtype telemetry and guard for ONE update: the rounded
    transition's entries and its three Jury slacks. `ok` is False if any
    quantity is non-finite or any slack is not strictly positive."""
    A2 = executed_filter_transition(coeff, dtype)
    s1, s2, s3 = jury_slacks(A2)
    smin = jnp.minimum(jnp.minimum(s1, s2), s3)
    finite = jnp.all(jnp.isfinite(A2)) & jnp.isfinite(smin)
    M, gam, T, A = coeff
    return dict(c1=A2[0, 0], c0=-A2[0, 1], A=A, M=M, gamma=gam, T=T,
                jury_1=s1, jury_2=s2, jury_3=s3, jury_min=smin,
                ok=finite & (smin > 0))


def filter_report(p):
    """Host-side acceptance report for a parameter tree: the SAME executed
    rounded transition, classified exactly over the rationals, plus the
    declared numerical gaps' achieved slacks (reported, never substituted for
    the executed gate)."""
    dtype = p["fil_M"].dtype
    coeff = coefficients(p)
    A2 = onp.asarray(executed_filter_transition(coeff, dtype))
    label, slacks = PD.classify(A2)
    M, gam, T, A = (float(onp.asarray(x)) for x in coeff)
    h = H
    return dict(rule=FILTERED, executed_dtype=str(A2.dtype),
                M=M, gamma=gam, T=T, A=A,
                executed_c1=float(A2[0, 0]), executed_c0=float(-A2[0, 1]),
                executed_transition=[[float(x) for x in row] for row in A2],
                classification=label,
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
                two_tap_kappa=(T / h if (M == 0.0 and abs(gam + T - h)
                                         <= 0.0) else None),
                note=("acceptance is the exact classification of the rounded "
                      "executed transition; the declared gaps G_MIN and "
                      "DELTA_FILTER are a numerical policy and are reported, "
                      "not used as the certificate. Isolated filter only."))


def filter_failure(rep):
    """None if acceptable; anything but a strictly stable, finite executed
    filter with nonnegative coefficients is refused."""
    if not all(onp.isfinite([rep["M"], rep["gamma"], rep["T"], rep["A"]])):
        return f"non-finite filter coefficients {rep}"
    if not rep["nonnegative"]:
        return (f"negative coefficient: M={rep['M']} gamma={rep['gamma']} "
                f"T={rep['T']} (the family is nonnegative; kappa > 1 of the "
                "two-tap operator is outside it)")
    if rep["classification"] != "stable":
        return (f"executed filter polynomial not strictly stable: "
                f"{rep['classification']} slacks {rep['jury_slacks_exact']} "
                f"c1={rep['executed_c1']} c0={rep['executed_c0']} A={rep['A']}")
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


def exact_rational_polynomial(p):
    """The executed polynomial's exact rational coefficients, for reports."""
    A2 = onp.asarray(executed_filter_transition(coefficients(p),
                                                p["fil_M"].dtype),
                     onp.float64)
    c1 = Fraction(float(A2[0, 0]))
    c0 = Fraction(float(-A2[0, 1]))
    return c1, c0
