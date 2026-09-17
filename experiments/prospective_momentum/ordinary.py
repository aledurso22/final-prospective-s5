"""Ordinary first-order prospective residual processing on the same memory.

Audit: docs/PROSPECTIVE_SAME_BACKBONE_AUDIT.md. Brief:
PROSPECTIVE_SAME_BACKBONE_AND_RETENTION_BRIEF_2026_09_17.md.

The comparator applies the causal first-order prospective operator to the
write residual and feeds the NATIVE Momentum update:

    Rpros_t = R_t + kappa (R_t - R_(t-1))
    U_t     = mu_t U_(t-1) + eta_t Rpros_t
    W_t     = Wbar_t - beta_t U_t

h = 1; kappa is the prospective horizon in token units. The write mask is
applied to R BEFORE the difference and Rpros is never re-masked, so the first
idle step after a write carries -kappa R_(t-1). The previous residual is real
streaming state: carried across chunk boundaries, zero at episode start.
Carry: W, U and R_(t-1) = 192 real numbers (the candidate's is 128).

Relations established in the audit and checked on the cluster:

* constant eta and mu: `U_t = Q_t + kappa (eta/mu) R_t` maps the candidate
  onto this recurrence EXACTLY, closed loop, for arbitrary alpha_t, beta_t,
  masks, keys and values. Same model class, not two classes.
* token-varying gates: the candidate obeys `transported_step` below, which
  differs from this operator by
  `kappa [eta_t - mu_t eta_(t-1)/mu_(t-1)] R_(t-1)`; they agree iff the RATIO
  eta/mu is unchanged between consecutive tokens.
* frozen-token stability: the executed 3x3 transition's characteristic
  polynomial factorizes as z times the candidate's quadratic, so the
  candidate's Jury conditions and kappa bound apply unchanged (derived, not
  copied), with a third, zero eigenvalue.
* literal TSS Eq. (17) as a processing stage reduces to this operator at
  kappa = 1 only when tau = h, under the declared same-token output timing
  (`tss_processing_reference`, checks only).
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as onp

from experiments.nested_memory import model as NM

from . import dynamics as PD

ORDINARY = "ordinary_prospective"
#: W, U and the previous residual
CARRY_REALS = 3 * NM.D_V * NM.D_K
DISPLAY = ("Ordinary first-order prospective residual operator feeding the "
           "native Momentum update")


# ------------------------------------------------------------- updates -----
def ordinary_update(Wb, U, Rprev, beta, mu, eta, kappa, R):
    """One step given Wbar and the masked residual R (open-loop form)."""
    Rpros = R + kappa * (R - Rprev)
    U_new = mu * U + eta * Rpros
    return Wb - beta * U_new, U_new, R


def ordinary_step(carry, k, v, m, alpha, beta, mu, eta, kappa):
    W, U, Rprev = carry
    Wb = alpha * W
    R = m * jnp.outer(Wb @ k - v, k)
    return ordinary_update(Wb, U, Rprev, beta, mu, eta, kappa, R)


def transported_step(carry, k, v, m, alpha, beta, mu, eta, kappa):
    """CHECK-ONLY third realization: the candidate, written in the operator's
    mapped state with the gate-transported previous residual (audit eq. 2).
    Carry: (W, U, R_(t-1), eta_(t-1), mu_(t-1))."""
    W, U, Rprev, eta_prev, mu_prev = carry
    Wb = alpha * W
    R = m * jnp.outer(Wb @ k - v, k)
    U_new = (mu * U + eta * (1.0 + kappa) * R
             - kappa * mu * (eta_prev / mu_prev) * Rprev)
    return Wb - beta * U_new, U_new, R, eta, mu


def tss_processing_reference(residuals, tau, h=1.0):
    """CHECK-ONLY literal TSS Eq. (17) as a processing stage:

        y_(t+1) = y_t + (h/tau)(R_t - y_t) + R_t - R_(t-1)

    Returns the sequence whose t-th entry is the state produced by the token-t
    update, i.e. the DECLARED same-token output convention. Read with Eq.
    (17)'s own index that entry is y_(t+1); driving a write with it one token
    later would delay the correction by one token. At tau = h it equals
    `2 R_t - R_(t-1)`, the operator's Rpros at kappa = 1."""
    R = [onp.asarray(x, onp.float64) for x in residuals]
    y = onp.zeros_like(R[0])
    prev = onp.zeros_like(R[0])
    out = []
    for r in R:
        y = y + (h / tau) * (r - y) + r - prev
        prev = r
        out.append(y.copy())
    return out


# ------------------------------------------------- frozen-token transition --
def analytic_transition(alpha, beta, mu, eta, kappa):
    """A_ord of the audit, s4 (float64 host arithmetic)."""
    q = beta * eta
    return onp.array([
        [alpha * (1.0 - q * (1.0 + kappa)), -beta * mu, q * kappa],
        [alpha * eta * (1.0 + kappa), mu, -eta * kappa],
        [alpha, 0.0, 0.0]])


def executed_transitions(gates, kappa, dtype):
    """The key-aligned 3x3 transition of the PRODUCTION operator step, from
    basis carries with the exactly representable unit key e_0, v = 0, m = 1.
    Certifies these rounded matrices only."""
    a, b, mu, eta = gates
    d_k, d_v = NM.D_K, 1
    k = jnp.zeros((d_k,), dtype=dtype).at[0].set(1)
    v = jnp.zeros((d_v,), dtype=dtype)
    e = jnp.zeros((d_v, d_k), dtype=dtype).at[0, 0].set(1)
    z = jnp.zeros((d_v, d_k), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    kap = jnp.asarray(kappa, dtype=dtype)

    def col(W, U, Rp, ai, bi, mi, ei):
        W2, U2, R2 = ordinary_step((W, U, Rp), k, v, one, ai, bi, mi, ei, kap)
        return W2[0, 0], U2[0, 0], R2[0, 0]

    f = jax.vmap(lambda ai, bi, mi, ei: (col(e, z, z, ai, bi, mi, ei),
                                         col(z, e, z, ai, bi, mi, ei),
                                         col(z, z, e, ai, bi, mi, ei)))
    (w1, u1, r1), (w2, u2, r2), (w3, u3, r3) = f(a, b, mu, eta)
    rows = (jnp.stack([w1, w2, w3], -1), jnp.stack([u1, u2, u3], -1),
            jnp.stack([r1, r2, r3], -1))
    return jnp.stack(rows, -2)


def classify_cubic(A):
    """Exact rational Jury classification of a rounded executed 3x3.

    For p(z) = z^3 + a2 z^2 + a1 z + a0 the conditions are
    p(1) > 0, -p(-1) > 0, |a0| < 1 and |a0^2 - 1| > |a0 a2 - a1|.
    stable: all strict; neutral: none violated, at least one equality;
    unstable: any violated; nonfinite: any entry non-finite."""
    A = onp.asarray(A, onp.float64)
    if not onp.all(onp.isfinite(A)):
        return "nonfinite", None
    f = [[Fraction(float(x)) for x in row] for row in A]
    tr = f[0][0] + f[1][1] + f[2][2]
    m2 = (f[0][0] * f[1][1] - f[0][1] * f[1][0]
          + f[0][0] * f[2][2] - f[0][2] * f[2][0]
          + f[1][1] * f[2][2] - f[1][2] * f[2][1])
    det = (f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
           - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
           + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]))
    a2, a1, a0 = -tr, m2, -det
    p1 = 1 + a2 + a1 + a0
    pm1 = -(-1 + a2 - a1 + a0)
    c3 = 1 - abs(a0)
    c4 = abs(a0 * a0 - 1) - abs(a0 * a2 - a1)
    tests = (p1, pm1, c3, c4)
    if min(tests) < 0:
        lab = "unstable"
    elif min(tests) == 0:
        lab = "neutral"
    else:
        lab = "stable"
    return lab, tuple(float(t) for t in tests)


def transition_report(p):
    """TABLE coverage for the operator: exact classification of the rounded
    executed 3x3 transitions over the write table, and the float64 kappa
    bound. The bound is the CANDIDATE's because the audit derives that the
    operator's characteristic polynomial is z times the candidate's
    quadratic; it is not assumed."""
    dtype = p["key_raw"].dtype
    gates = PD.table_gates(p, PD.write_table())
    kappa = float(onp.asarray(p["kappa"]).ravel()[0])
    A = onp.asarray(executed_transitions(gates, kappa, dtype))
    labels, worst = {}, None
    for i in range(A.shape[0]):
        lab, t = classify_cubic(A[i])
        labels[lab] = labels.get(lab, 0) + 1
        if t is not None and (worst is None or min(t) < worst):
            worst = min(t)
    g64 = [onp.asarray(x, onp.float64) for x in gates]
    aq = g64[0] * g64[1] * g64[3]
    kb = float(onp.min(onp.where(aq > 0,
                                 ((1 + g64[0]) * (1 + g64[2]) - aq)
                                 / (2 * onp.maximum(aq, 1e-300)), onp.inf)))
    return dict(rule=ORDINARY, executed_dtype=str(A.dtype),
                n_write_settings=int(A.shape[0]), classification=labels,
                min_jury_expression=worst, kappa=kappa,
                kappa_bound_f64=kb,
                kappa_over_bound=(kappa / kb if onp.isfinite(kb) else 0.0),
                gates_finite=bool(all(onp.all(onp.isfinite(x)) for x in g64)),
                bound_note=("the candidate's frozen-token bound, derived for "
                            "this realization in the audit (third eigenvalue "
                            "is zero); a frozen-token diagnostic only"))


def transition_failure(rep):
    """None if acceptable; unstable or non-finite executed transitions fail.
    Neutral rounded cases are counted and reported, not failed."""
    if not rep["gates_finite"]:
        return "non-finite gates on the write table"
    bad = {k: v for k, v in rep["classification"].items()
           if k in ("unstable", "nonfinite")}
    if bad:
        return f"executed frozen-token transitions {bad}"
    k = rep["kappa"]
    if not (onp.isfinite(k) and 0.0 <= k < rep["kappa_bound_f64"]):
        return (f"kappa {k} outside [0, {rep['kappa_bound_f64']}) "
                "(float64 bound from executed gates)")
    return None
