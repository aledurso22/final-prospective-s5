"""Prospective correction of the Momentum DeltaNet recurrence.

Brief: PROSPECTIVE_MOMENTUM_NEXT_STEP_2026_09_16.md. Independent audit of its
equations (2)-(12): docs/PROSPECTIVE_MOMENTUM_PROOF_AUDIT.md.

Native Momentum DeltaNet (pinned gate family, this repository's orientation):

    Wbar_t    = alpha_t W_(t-1)
    R_t       = m_t (Wbar_t k_t - v_t) k_t^T
    Qnative_t = mu_t Q_(t-1) + eta_t R_t
    W_t       = Wbar_t - beta_t Qnative_t

Candidate, the declared semi-implicit step of M Wddot + gamma Wdot + s R +
T s Rdot = 0 under M = h/beta, gamma = (1-mu)/(beta mu), s = eta/(mu h),
kappa = T/h (audit s2):

    Q_t = Qnative_t - kappa eta_t ((1-mu_t)/mu_t) R_t
    W_t = Wbar_t - beta_t Qnative_t - kappa beta_t eta_t R_t

BOTH lines use Qnative from the OLD carry. At kappa = 0 this is the native
recurrence: every native operation is evaluated first, in the native order,
and the correction is subtracted afterwards (a subtraction of an exact zero).
`m` is a {0, 1} mask, so `eta * (m E)` and the native `(eta m) E` are the same
floating-point value.

Gain-only control (no residual-derivative term):

    Q_t = mu_t Q_(t-1) + g eta_t R_t ,  W_t = Wbar_t - beta_t Q_t ,  g = exp(log_g)

Frozen-token scope. For one fixed unit key, fixed gates and an active write
the key-aligned (W, Q) transition is A_kappa (audit s5). Its three Jury
expressions are a FROZEN-TOKEN diagnostic, NOT a common Lyapunov proof under
key or gate switching; nothing here claims global stability.

kappa is a directly stored scalar, initialized to exactly 0, never clipped in
the forward pass. After each optimizer update it is projected to
[0, (1 - PROJ_REL_MARGIN) Kmax], Kmax the frozen-token bound (12) minimized
over EVERY gate setting at which a write can occur, from the UPDATED gate
parameters. PROJ_REL_MARGIN is a DECLARED numerical safety margin, supported
by the arithmetic estimates of audit s6 and by the measured checks; it is not
a universal certificate for arbitrary trained gate parameters, reduction
orders or rollout states (review R4, a1f0439).

Coverage labels used throughout:
  * TABLE coverage: gates recomputed by `table_gates` on the finite gate-input
    table; the rounded executed 2x2 transitions of these table gates (unit key
    e_0) are classified exactly. This certifies those rounded matrices only.
  * OBSERVED-ROLLOUT coverage: gates RETURNED by actual evaluation rollouts
    (`observed_gate_report`). The same gate function in a different
    compilation context is not proven bitwise identical to the table.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as onp

from experiments.nested_memory import dynamics as NMD
from experiments.nested_memory import model as NM
from experiments.nested_memory.task import N_KEYS, N_VALUES, WRITE, QUERY, IDLE

#: relative margin below the frozen-token bound; derivation: audit s6
PROJ_REL_MARGIN = 1e-3

RULES = ("prospective_momentum", "momentum_delta", "gain_momentum",
         "gated_delta", "gp_two_sided", "tss_eq17")
MOMENTUM_FAMILY = ("prospective_momentum", "momentum_delta", "gain_momentum")
LITERATURE = ("momentum_delta", "gated_delta")
DISPLAY = {
    "prospective_momentum": ("Momentum DeltaNet with generalized prospective "
                             "correction (kappa)"),
    "momentum_delta": "Momentum DeltaNet rule, native continuation",
    "gain_momentum": "Momentum DeltaNet with gain-only control (g)",
    "gated_delta": "Gated DeltaNet rule, continuation",
    "gp_two_sided": ("Generalized prospective memory (completed study's rule, "
                     "unchanged), continuation"),
    "tss_eq17": ("TSS Eq. (17) applied directly to the fast weight; "
                 "applicability-limited; continuation"),
}
CARRY = {"prospective_momentum": 128, "momentum_delta": 128,
         "gain_momentum": 128, "gated_delta": 64, "gp_two_sided": 128,
         "tss_eq17": 128}
#: the scalar each extension adds to the native tree
EXTRA_LEAF = {"prospective_momentum": "kappa", "gain_momentum": "log_g"}
# The same-backbone study registers its ordinary-operator rule here
# (`experiments.prospective_momentum.ordinary`). These are ADDITIVE dictionary
# entries only: RULES, MOMENTUM_FAMILY and LITERATURE are unchanged, so no
# completed study's arm list, projection or report changes.


# ------------------------------------------------------------ updates ------
def prospective_update(Wb, Q, R, beta, mu, eta, kappa):
    """Equation (6) given Wbar and the residual R (open-loop form)."""
    Qn = mu * Q + eta * R
    Q_new = Qn - kappa * eta * ((1.0 - mu) / mu) * R
    W_new = Wb - beta * Qn - kappa * beta * eta * R
    return W_new, Q_new


def prospective_step(carry, k, v, m, alpha, beta, mu, eta, kappa):
    W, Q = carry
    Wb = alpha * W
    R = m * jnp.outer(Wb @ k - v, k)
    return prospective_update(Wb, Q, R, beta, mu, eta, kappa)


def gain_update(Wb, Q, R, beta, mu, eta, g):
    Q_new = mu * Q + (g * eta) * R
    return Wb - beta * Q_new, Q_new


def gain_step(carry, k, v, m, alpha, beta, mu, eta, g):
    W, Q = carry
    Wb = alpha * W
    R = m * jnp.outer(Wb @ k - v, k)
    return gain_update(Wb, Q, R, beta, mu, eta, g)


# ----------------------------------------------------- gate-setting table --
def write_table():
    """Every input at which a write CAN occur: event WRITE, all 32 keys, all
    eight values and the absent value (a superset of the task's writes, which
    always carry a value). Gates depend only on these features."""
    key = onp.repeat(onp.arange(N_KEYS), N_VALUES + 1)
    val = onp.tile(onp.arange(-1, N_VALUES), N_KEYS)
    ev = onp.full_like(key, WRITE)
    return key.astype(onp.int32), val.astype(onp.int32), ev.astype(onp.int32)


def full_table():
    """Every representable token input (all events), for gate-range checks."""
    ks, vs, es = [], [], []
    for e in (WRITE, QUERY, IDLE):
        k, v, _ = write_table()
        ks.append(k); vs.append(v); es.append(onp.full_like(k, e))
    return (onp.concatenate(ks), onp.concatenate(vs), onp.concatenate(es))


def table_gates(p, table):
    """The pinned gate function on a table, in the leaves' dtype.

    TABLE coverage. This is the same function the rollout calls, but in a
    different array shape and compilation context; bitwise equality with the
    gates computed inside a rollout is NOT assumed or claimed."""
    key, val, ev = table
    x = NM.gate_features(jnp.asarray(key), jnp.asarray(val),
                         jnp.asarray(ev)).astype(p["key_raw"].dtype)
    return NM._momentum_gates(p, x)


def executed_gain(p):
    """g exactly as the production rollout forms it."""
    return jnp.exp(p["log_g"][0])


# ------------------------------------------------------------- bounds ------
def kappa_bound(alpha, beta, mu, eta):
    """Frozen-token bound (12) per setting; +inf where alpha q = 0 (the
    unsimplified 1 + tr + det = (1+alpha)(1+mu) > 0 is then independent of
    kappa). No division by zero is executed."""
    aq = alpha * (beta * eta)
    num = (1.0 + alpha) * (1.0 + mu) - aq
    pos = aq > 0
    return jnp.where(pos, num / (2.0 * jnp.where(pos, aq, 1.0)), jnp.inf)


def gain_bound(alpha, beta, mu, eta):
    """alpha beta eta g < (1+alpha)(1+mu) per setting; +inf where alpha q = 0."""
    aq = alpha * (beta * eta)
    pos = aq > 0
    return jnp.where(pos, (1.0 + alpha) * (1.0 + mu) / jnp.where(pos, aq, 1.0),
                     jnp.inf)


def project(p):
    """Post-update projection of the ONE extension scalar, from the UPDATED
    gates. Optimizer state untouched. A no-op for trees without an extension
    scalar (the meta-delta raw_r projection is applied separately).

    Telemetry: n_projected, pre (proposal), post, cap (the PROJECTION cap,
    margin included), bound (the un-margined frozen-token bound), overshoot.
    kappa entries are values of kappa; log_g entries are LOGARITHMS (cap and
    bound are log caps). For a tree without an extension scalar every entry
    is NaN: unavailable, not an observation."""
    WT = write_table()
    if "kappa" in p:
        a, b, mu, eta = table_gates(p, WT)
        kmax = jnp.min(kappa_bound(a, b, mu, eta))
        cap = jnp.maximum(kmax * (1.0 - PROJ_REL_MARGIN), 0.0)
        pre = p["kappa"]
        post = jnp.minimum(jnp.maximum(pre, 0.0), cap)
        tel = dict(n_projected=jnp.sum(post != pre), pre=pre[0], post=post[0],
                   cap=cap, bound=kmax, overshoot=jnp.max(jnp.abs(pre - post)))
        return dict(p, kappa=post), tel
    if "log_g" in p:
        a, b, mu, eta = table_gates(p, WT)
        gmax = jnp.min(gain_bound(a, b, mu, eta))
        cap = jnp.log(gmax) + jnp.log1p(-PROJ_REL_MARGIN)
        pre = p["log_g"]
        post = jnp.minimum(pre, cap)
        tel = dict(n_projected=jnp.sum(post != pre), pre=pre[0], post=post[0],
                   cap=cap, bound=jnp.log(gmax),
                   overshoot=jnp.max(jnp.maximum(pre - post, 0.0)))
        return dict(p, log_g=post), tel
    nan = jnp.full((), jnp.nan, dtype=p["key_raw"].dtype)
    return p, dict(n_projected=jnp.asarray(0), pre=nan, post=nan, cap=nan,
                   bound=nan, overshoot=nan)


# ------------------------------------------------ executed transition ------
def analytic_transition(alpha, beta, mu, eta, kappa):
    """A_kappa of the brief, s6 (float64 host arithmetic)."""
    q = beta * eta
    c = 1.0 - kappa * (1.0 - mu) / mu
    return onp.array([[alpha * (1.0 - q * (1.0 + kappa)), -beta * mu],
                      [alpha * eta * c, mu]])


def analytic_gain_transition(alpha, beta, mu, eta, g):
    return onp.array([[alpha * (1.0 - beta * eta * g), -beta * mu],
                      [alpha * eta * g, mu]])


def jury(alpha, beta, mu, eta, kappa):
    """Equation (11), unsimplified in q (no division)."""
    q = beta * eta
    return ((1 - alpha) * (1 - mu) + alpha * q,
            (1 + alpha) * (1 + mu) - alpha * q * (1 + 2 * kappa),
            1 - alpha * mu + alpha * q * kappa)


def executed_transitions(step, gates, scalar, dtype):
    """The key-aligned transition of the PRODUCTION step, extracted by
    applying it to basis carries with the exactly representable unit key e_0,
    v = 0 and m = 1, for the SUPPLIED gates. Returns (n, 2, 2) in the executed
    dtype. Classifying these matrices certifies exactly these rounded
    frozen-token matrices; it is not a proof for arbitrary rollout states,
    non-basis keys, other reduction orders or switching trajectories."""
    a, b, mu, eta = gates
    d_k = NM.D_K
    k = jnp.zeros((d_k,), dtype=dtype).at[0].set(1)
    v = jnp.zeros((1,), dtype=dtype)
    e = jnp.zeros((1, d_k), dtype=dtype).at[0, 0].set(1)
    z = jnp.zeros((1, d_k), dtype=dtype)
    one = jnp.ones((), dtype=dtype)

    def col(W, Q, ai, bi, mi, ei):
        W2, Q2 = step((W, Q), k, v, one, ai, bi, mi, ei, scalar)
        return W2[0, 0], Q2[0, 0]

    f = jax.vmap(lambda ai, bi, mi, ei: (col(e, z, ai, bi, mi, ei),
                                         col(z, e, ai, bi, mi, ei)))
    (w1, q1), (w2, q2) = f(a, b, mu, eta)
    return jnp.stack([jnp.stack([w1, w2], -1), jnp.stack([q1, q2], -1)], -2)


def classify(A):
    """Exact rational Jury classification of rounded executed entries.

    stable: 1 - tr + det > 0, 1 + tr + det > 0 and 1 - det > 0 (equivalent to
    both eigenvalues strictly inside the unit circle for a real 2x2);
    neutral: none negative, at least one exactly zero (a root on the circle,
    e.g. rounded alpha = 1 with q = 0); unstable: any negative;
    nonfinite: any entry non-finite."""
    A = onp.asarray(A, onp.float64)
    if not onp.all(onp.isfinite(A)):
        return "nonfinite", None
    a11, a12, a21, a22 = (Fraction(float(x)) for x in A.ravel())
    tr, det = a11 + a22, a11 * a22 - a12 * a21
    J = (1 - tr + det, 1 + tr + det, 1 - det)
    if min(J) < 0:
        lab = "unstable"
    elif min(J) == 0:
        lab = "neutral"
    else:
        lab = "stable"
    return lab, tuple(float(j) for j in J)


def transition_report(p, rule):
    """TABLE coverage: exact classification of the rounded executed
    frozen-token transitions over the write table, plus float64 bounds
    evaluated on the rounded table gates, for a momentum-family tree.
    Host-side; used for source, preflight and final validation.

    Review R3: the gain is the PRODUCTION transform `jnp.exp(log_g)` in the
    leaves' dtype (`executed_g`); positivity and the bound are validated on
    that value. The host float64 exponential is reported separately as
    `reference_g_f64` and is never used for acceptance."""
    dtype = p["key_raw"].dtype
    a, b, mu, eta = table_gates(p, write_table())
    ga = [onp.asarray(x) for x in (a, b, mu, eta)]
    if rule == "prospective_momentum":
        step, scalar = prospective_step, p["kappa"][0]
    elif rule == "gain_momentum":
        step, scalar = gain_step, executed_gain(p)
    else:                                   # the executed native step itself
        step = (lambda c, k, v, m, a_, b_, mu_, e_, _:
                NMD.momentum_delta_step(c, k, v, m, a_, b_, mu_, e_))
        scalar = jnp.zeros((), dtype=dtype)
    A = onp.asarray(executed_transitions(step, (a, b, mu, eta), scalar, dtype))
    labels, worst = {}, None
    for i in range(A.shape[0]):
        lab, J = classify(A[i])
        labels[lab] = labels.get(lab, 0) + 1
        if J is not None and (worst is None or min(J) < worst):
            worst = min(J)
    g64 = [x.astype(onp.float64) for x in ga]
    kb = onp.min(onp.where(g64[0] * g64[1] * g64[3] > 0,
                           ((1 + g64[0]) * (1 + g64[2])
                            - g64[0] * g64[1] * g64[3])
                           / (2 * onp.maximum(g64[0] * g64[1] * g64[3],
                                              1e-300)), onp.inf))
    gb = onp.min(onp.where(g64[0] * g64[1] * g64[3] > 0,
                           (1 + g64[0]) * (1 + g64[2])
                           / onp.maximum(g64[0] * g64[1] * g64[3], 1e-300),
                           onp.inf))
    rep = dict(rule=rule, executed_dtype=str(A.dtype),
               n_write_settings=int(A.shape[0]), classification=labels,
               min_jury_expression=worst,
               kappa_bound_f64=float(kb), gain_bound_f64=float(gb),
               gates_finite=bool(all(onp.all(onp.isfinite(x)) for x in ga)))
    if rule == "prospective_momentum":
        k = float(onp.asarray(p["kappa"]).ravel()[0])
        rep.update(kappa=k, kappa_over_bound=(k / kb if onp.isfinite(kb)
                                              else 0.0))
    if rule == "gain_momentum":
        lg = float(onp.asarray(p["log_g"]).ravel()[0])
        g_exec = float(onp.asarray(scalar))
        rep.update(raw_log_g=lg, executed_g=g_exec,
                   executed_g_dtype=str(onp.asarray(scalar).dtype),
                   reference_g_f64=float(onp.exp(onp.float64(lg))),
                   executed_g_over_bound=(g_exec / gb if onp.isfinite(gb)
                                          else 0.0))
    return rep


def transition_failure(rep):
    """None if acceptable. Unstable or non-finite executed transitions fail;
    neutral rounded cases are counted and reported, not failed."""
    if not rep["gates_finite"]:
        return "non-finite gates on the write table"
    bad = {k: v for k, v in rep["classification"].items()
           if k in ("unstable", "nonfinite")}
    if bad:
        return f"executed frozen-token transitions {bad}"
    if rep["rule"] == "prospective_momentum":
        k = rep["kappa"]
        if not (onp.isfinite(k) and 0.0 <= k and k < rep["kappa_bound_f64"]):
            return (f"kappa {k} outside [0, {rep['kappa_bound_f64']}) "
                    "(float64 bound from executed gates)")
    if rep["rule"] == "gain_momentum":
        g = rep["executed_g"]                     # production value (R3)
        if not (onp.isfinite(g) and 0.0 < g < rep["gain_bound_f64"]):
            return (f"executed g {g} outside (0, {rep['gain_bound_f64']}) "
                    f"(reference float64 exp {rep['reference_g_f64']})")
    return None


def gate_range_report(p):
    """Finiteness and ranges of the pinned gates on EVERY token input, and
    the passive sector (9) occupancy kappa >= mu/(1-mu)."""
    a, b, mu, eta = (onp.asarray(x, onp.float64)
                     for x in table_gates(p, full_table()))
    ok = bool(all(onp.all(onp.isfinite(x)) for x in (a, b, mu, eta))
              and onp.all((a >= 0) & (a <= 1)) and onp.all((b >= 0) & (b <= 1))
              and onp.all((mu > 0) & (mu <= 1))
              and onp.all((eta >= 0) & (eta <= 2)))

    def st(v):
        return dict(min=float(v.min()), median=float(onp.median(v)),
                    max=float(v.max()), mean=float(v.mean()))
    rep = dict(gates_valid=ok, alpha=st(a), beta=st(b), mu=st(mu), eta=st(eta),
               q=st(b * eta), alpha_rounded_to_one=int(onp.sum(a == 1)),
               mu_rounded_to_one=int(onp.sum(mu == 1)),
               beta_rounded_to_zero=int(onp.sum(b == 0)),
               mu_at_clamp=int(onp.sum(onp.abs(onp.log(mu) - NM.MIN_LOG_MU)
                                       < 1e-6)))
    if "kappa" in p:
        k = float(onp.asarray(p["kappa"], onp.float64).ravel()[0])
        wmu = mu[: N_KEYS * (N_VALUES + 1)]            # write settings first
        thr = onp.where(wmu < 1, wmu / onp.maximum(1 - wmu, 1e-300), onp.inf)
        nu = 1.0 - k * (1.0 - wmu) / wmu
        rep["sector_9"] = dict(
            kappa=k, passive_threshold_mu_over_1_minus_mu=st(thr),
            fraction_of_write_settings_in_passive_sector=float(
                onp.mean(k >= thr)),
            qhm_nu=st(nu),
            note=("kappa >= mu/(1-mu) is the passive two-compartment sector; "
                  "kappa = 0 (native) is outside it"))
    return rep


def observed_gate_report(rule, gates, event, kappa=None, executed_g=None):
    """OBSERVED-ROLLOUT coverage: statistics of the gates RETURNED by actual
    evaluation rollouts (arrays (n_episodes, L)), with float64 frozen-token
    quantities evaluated analytically on those rounded observed gates at
    WRITE tokens. This is not an executed-entry classification and not a
    switching-stability statement; it is labelled separately from TABLE
    coverage."""
    a, b, mu, eta = (onp.asarray(x, onp.float64) for x in gates)
    w = onp.asarray(event) == WRITE

    def st(v):
        if v.size == 0:
            return None
        return dict(min=float(v.min()), median=float(onp.median(v)),
                    max=float(v.max()), mean=float(v.mean()))
    out = dict(coverage="observed rollout gates (returned by evaluation)",
               finite=bool(all(onp.all(onp.isfinite(x))
                               for x in (a, b, mu, eta))),
               all_tokens={k: st(v) for k, v in
                           dict(alpha=a, beta=b, mu=mu, eta=eta).items()},
               write_tokens={k: st(v[w]) for k, v in
                             dict(alpha=a, beta=b, mu=mu, eta=eta,
                                  q=b * eta).items()},
               n_tokens=int(a.size), n_write_tokens=int(w.sum()))
    if not w.any():
        return out
    aw, bw, mw, ew = a[w], b[w], mu[w], eta[w]
    aq = aw * bw * ew
    pos = aq > 0
    kb = onp.where(pos, ((1 + aw) * (1 + mw) - aq)
                   / (2 * onp.where(pos, aq, 1.0)), onp.inf)
    gb = onp.where(pos, (1 + aw) * (1 + mw) / onp.where(pos, aq, 1.0),
                   onp.inf)
    out.update(kappa_bound_f64_min=float(kb.min()),
               gain_bound_f64_min=float(gb.min()))
    if rule == "prospective_momentum" and kappa is not None:
        J = jury(aw, bw, mw, ew, kappa)
        out.update(kappa=float(kappa),
                   kappa_over_min_bound=(float(kappa / kb.min())
                                         if onp.isfinite(kb.min()) else 0.0),
                   analytic_jury_min=[float(onp.min(j)) for j in J],
                   write_tokens_with_negative_analytic_jury=int(
                       onp.sum(onp.minimum(onp.minimum(J[0], J[1]), J[2])
                               < 0)))
    if rule == "gain_momentum" and executed_g is not None:
        out.update(executed_g=float(executed_g),
                   executed_g_over_min_bound=(float(executed_g / gb.min())
                                              if onp.isfinite(gb.min())
                                              else 0.0))
    return out
