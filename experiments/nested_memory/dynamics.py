"""Associative-memory update rules: the proposed law, its controls, and two
literature rules.

Orientation, fixed for every rule here: `W` is `(d_v, d_k)`, keys and queries
are `(d_k,)` columns, the readout sees `W @ q`. Writes carry a unit-norm key, a
value `v` of shape `(d_v,)` and a mask `m in {0, 1}`; a query or filler has
`m = 0` and **no value source at all**.

THE PROPOSED LAW (see the coordinator's EQUATION_CONTRACT):

    E_t(W) = (m_t/2) ||W k_t - v_t||^2 ,   R_t = m_t (W k_t - v_t) k_t^T
    M Wddot + gamma Wdot + R + T Rdot = 0 ,  M, gamma, T > 0 ,  M <= gamma T

carried in the two-state coordinates `P = M Wdot + T R`, which is what makes
the token step exact:

    Wdot = (P - T R)/M ,   Pdot = -gamma P/M + (gamma T/M - 1) R

Carrying `P` rather than `Wdot` is not a convenience. At a key, value or mask
jump the velocity jumps by `-T dR/M`, so a scheme that carried `Wdot` unchanged
across a change in `R` would be solving a different equation. `P` is continuous
there, and nothing in this file resets it at a query, a token boundary or an
episode-internal key change; only a new episode resets `W = P = 0`.

The token step is EXACT for a held input, not a discretization. Splitting `W`
into its `k` and `k`-perpendicular parts for a held unit key gives a 2x2 system
in `(e, p) = (W k - v, P k)` with generator `A_1`, and a scalar decay on the
perpendicular part, so one matrix exponential of a 2x2 - computed once on the
host, never inside the scan - describes the whole interval.

NOTE ON THE QUERY INTERVAL. `W` moves during a query because its velocity has
not vanished. Freezing it would define a different method and could hide an
interference cost, so it is not frozen.
"""

import numpy as onp

# ---- the declared first reference. FIXED for this study: no source-strength
#      or horizon sweep is permitted in this batch.
M_REF, GAMMA_REF, T_REF, H_REF = 0.75, 1.0, 1.0, 1.0
#: below this raw key norm the normalized key is undefined and the write is
#: suppressed. A vector of tiny norm is never treated as exactly unit.
KEY_NORM_FLOOR = 1e-6


def _expm2(A):
    """Exact 2x2 matrix exponential in closed form, float64, host-side.

    `expm(A) = c0 I + c1 A` with `c0, c1` from the eigenvalues, using the
    confluent limit when they coincide. Complex intermediate values are real in
    the result, which the caller asserts. This is deliberately dependency-free
    and evaluated ONCE: the contract forbids calling a matrix exponential
    inside the scan.
    """
    A = onp.asarray(A, dtype=onp.float64)
    tr, det = A[0, 0] + A[1, 1], A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    disc = complex(tr * tr - 4.0 * det)
    r = onp.sqrt(disc)
    lp, lm = (tr + r) / 2.0, (tr - r) / 2.0
    if abs(lp - lm) < 1e-12:
        c0, c1 = onp.exp(lp) * (1.0 - lp), onp.exp(lp)
    else:
        c0 = (lp * onp.exp(lm) - lm * onp.exp(lp)) / (lp - lm)
        c1 = (onp.exp(lp) - onp.exp(lm)) / (lp - lm)
    out = c0 * onp.eye(2, dtype=complex) + c1 * A
    assert onp.max(onp.abs(out.imag)) < 1e-12, "complex residue in a real expm"
    return onp.real(out)


def prospective_constants(M=M_REF, gamma=GAMMA_REF, T=T_REF, h=H_REF):
    """`F`, `a0`, `b0` for the proposed law. Host-side, float64, computed once.

    `a0 = exp(-gamma h / M)` is the decay of the perpendicular auxiliary part
    and `b0 = (1 - a0)/gamma` its integrated effect on `W`; `-expm1` is used so
    `1 - a0` keeps its precision when the exponent is small.
    """
    if not (M > 0 and gamma > 0 and T > 0):
        raise ValueError("M, gamma, T must be positive")
    if M > gamma * T + 1e-12:
        raise ValueError(f"M = {M} exceeds gamma*T = {gamma * T}: outside the "
                         f"declared admissible sector M <= gamma*T")
    A1 = onp.array([[-T / M, 1.0 / M],
                    [gamma * T / M - 1.0, -gamma / M]], dtype=onp.float64)
    a0 = float(onp.exp(-gamma * h / M))
    b0 = float(-onp.expm1(-gamma * h / M) / gamma)
    return dict(A1=A1, F=_expm2(h * A1), a0=a0, b0=b0,
                M=float(M), gamma=float(gamma), T=float(T), h=float(h))


def inertial_constants(M=M_REF, gamma=GAMMA_REF, h=H_REF):
    """`F`, `a0`, `b0` for the same-state ablation `M Wddot + gamma Wdot + R = 0`.

    Its auxiliary is `P = M Wdot` with no `T R` term, so its generator is
    `[[0, 1/M], [-1, -gamma/M]]`. The perpendicular constants are identical to
    the proposed law's, which is what makes it an equal-state ablation of the
    prospective residual derivative and nothing else.

    It is a passive heavy-ball comparison. Setting `T = 0` puts it OUTSIDE the
    `M <= gamma T` sector, so it is not another admissible point of the same
    circuit family and is never described as one.
    """
    A1 = onp.array([[0.0, 1.0 / M], [-1.0, -gamma / M]], dtype=onp.float64)
    a0 = float(onp.exp(-gamma * h / M))
    b0 = float(-onp.expm1(-gamma * h / M) / gamma)
    return dict(A1=A1, F=_expm2(h * A1), a0=a0, b0=b0,
                M=float(M), gamma=float(gamma), T=0.0, h=float(h),
                note="heavy-ball ablation; T = 0 is outside M <= gamma*T")


def beta_match(const=None):
    """`beta_match = 1 - F11`: the first-write response of the proposed law.

    From `W = P = 0` with one unit key the proposed update gives
    `W' = (1 - F11) v k^T`, so the fixed-delta comparator uses exactly this
    `beta`. It matches the FIRST write, not the full history response, and it
    does not equalize state capacity.
    """
    c = const or prospective_constants()
    return float(1.0 - c["F"][0, 0])


def ordinary_gradient_beta(gamma=GAMMA_REF, h=H_REF):
    """`beta0 = 1 - exp(-h/gamma)`, the delta write of the `M = gamma T` limit.

    At `M = gamma T` with `P0 = 0` the generator is upper triangular, so `P`
    stays zero for any history of changing keys and masks and `Wdot = -R/gamma`
    exactly. Tested as an identity; NOT substituted for the declared `M = 3/4`.
    """
    return float(-onp.expm1(-h / gamma))


# =========================================================== token rules ====
import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402

RULES = ("prospective_memory", "inertial_memory", "delta_matched_write",
         "gated_delta", "momentum_delta")
DISPLAY = {
    "prospective_memory": "Generalized prospective memory",
    "inertial_memory": "Inertial memory without prospective correction",
    "delta_matched_write": "Delta memory with matched first write",
    "gated_delta": "Gated DeltaNet rule",
    "momentum_delta": "Momentum DeltaNet rule",
}
#: real numbers carried per rule at d_k = d_v = 8
CARRY = {"prospective_memory": 128, "inertial_memory": 128,
         "delta_matched_write": 64, "gated_delta": 64,
         "momentum_delta": 128}


def safe_normalize(x, floor=KEY_NORM_FLOOR):
    """Unit key plus a validity flag. Below the floor the key is NOT unit.

    The exact rank-one step assumes `||k|| = 1`. A raw vector of tiny norm is
    therefore not silently rescaled into a pretend unit vector: it returns a
    zero key and `valid = 0`, and the caller suppresses the write. The gradient
    is taken through the safe branch, so no NaN reaches the outer optimizer.
    """
    n2 = jnp.sum(x * x, axis=-1, keepdims=True)
    n = jnp.sqrt(jnp.maximum(n2, floor * floor))
    valid = (n2 >= floor * floor).astype(x.dtype)
    return jnp.where(valid > 0, x / n, jnp.zeros_like(x)), valid[..., 0]


def two_state_step(carry, k, v, m, F, a0, b0):
    """One exact token interval of a two-state memory law. `carry = (W, P)`.

    For a held unit key the interval splits exactly:

        W' = W + b0 P + [(e' - e) - b0 p] k^T ,   P' = a0 P + [p' - a0 p] k^T
        (e', p') = F (e, p) ,  e = W k - v ,  p = P k

    and at `m = 0` the rank-one terms vanish, leaving `W' = W + b0 P`,
    `P' = a0 P` - the autonomous motion that a query interval also undergoes.
    """
    W, P = carry
    e = W @ k - v
    p = P @ k
    e_new = F[0, 0] * e + F[0, 1] * p
    p_new = F[1, 0] * e + F[1, 1] * p
    W = W + b0 * P + m * jnp.outer((e_new - e) - b0 * p, k)
    P = a0 * P + m * jnp.outer(p_new - a0 * p, k)
    return (W, P)


def delta_step(carry, k, v, m, beta):
    """Fixed-beta delta write, no auxiliary state, no decay."""
    (W,) = carry
    return (W + m * beta * jnp.outer(v - W @ k, k),)


def gated_delta_step(carry, k, v, m, alpha, beta):
    """Gated DeltaNet rule: `Wbar = alpha W`, `W' = Wbar + beta m (v - Wbar k) k^T`."""
    (W,) = carry
    Wb = alpha * W
    return (Wb + m * beta * jnp.outer(v - Wb @ k, k),)


def momentum_delta_step(carry, k, v, m, alpha, beta, mu, eta):
    """Momentum DeltaNet rule, transposed into this file's orientation.

        Wbar = alpha W ,  Q' = mu Q + eta m (Wbar k - v) k^T ,  W' = Wbar - beta Q'

    The mask suppresses the residual SOURCE on a query, not the momentum carry
    or its effect on `W`: at `m = 0` the momentum still decays by `mu` and
    still moves `W`. With `mu = 0, eta = 1` this reduces to the gated delta
    recurrence, which is checked.
    """
    W, Q = carry
    Wb = alpha * W
    Q = mu * Q + eta * m * jnp.outer(Wb @ k - v, k)
    return (Wb - beta * Q, Q)


def init_carry(rule, d_v, d_k, dtype=jnp.float32):
    z = jnp.zeros((d_v, d_k), dtype=dtype)
    if rule in ("prospective_memory", "inertial_memory", "momentum_delta"):
        return (z, z)
    return (z,)


def carry_size(rule, d_v, d_k):
    return len(init_carry(rule, d_v, d_k)) * d_v * d_k


# ===================== analytical-supplement quantities (16 Sep 2026) =======
# From ANALYTICAL_AUDIT.md. These are ANALYTICAL identities and diagnostics.
# They add no architecture, coefficient, training configuration or success
# criterion, and they do not change the declared study.

def response_coefficients(const=None, gamma=GAMMA_REF):
    """The candidate's effective scalar response at three observation times.

    `beta_match = 1 - F11` is the response at the END OF THE WRITE INTERVAL. A
    query in this protocol occupies a FURTHER no-write interval and reads `W`
    afterwards, and starting from `W = P = 0` a write leaves `P = -F21 v k^T`,
    so the idle map changes the effective coefficient:

        beta_write   = 1 - F11
        beta_query   = 1 - F11 - b0 F21      (one idle interval later)
        beta_settled = 1 - F11 - F21/gamma   (fully settled)

    With `F21 > 0` at the declared reference, the first-query effect is SMALLER
    than the write-end effect. The fixed-delta comparator keeps
    `beta = beta_write` exactly as declared: a first-write magnitude match was
    never a full transfer-function match, the comparator is not relabelled
    "query-matched", and beta is not changed after seeing outcomes.
    """
    c = const or prospective_constants()
    F, b0 = c["F"], c["b0"]
    return dict(beta_write=float(1.0 - F[0, 0]),
                beta_query=float(1.0 - F[0, 0] - b0 * F[1, 0]),
                beta_settled=float(1.0 - F[0, 0] - F[1, 0] / gamma),
                F11=float(F[0, 0]), F21=float(F[1, 0]))


def persistent_transient(W, P, gamma=GAMMA_REF):
    """The exact `(C, Z)` coordinates: `C = W + P/gamma`, `Z = -P/gamma`.

    `C` accumulates residual-based updates and `Z` is an exponentially filtered
    correction driven by the same residual, evaluated at `W = C + Z`. These are
    coordinates of the EXISTING two-state system, not added states or learned
    gates, and `C` is not an independent delta memory with an output filter.

    "Persistent" refers only to no-write intervals, where `R = 0` leaves `C`
    constant while `Z` decays with time constant `M/gamma`. It is not immunity
    to later interfering writes. The readout stays `W q`; reading `C q` would
    be a different architecture and is not used anywhere here.
    """
    return W + P / gamma, -P / gamma


def lyapunov_V(X, Y, M=M_REF, gamma=GAMMA_REF, T=T_REF):
    """`V(X, Y)` for the difference of two trajectories. Positive definite for
    `0 < M < gamma T`.

        V = gamma/2 ||X||^2 + <X, Y> + T/(2 delta) ||Y||^2
          = gamma/2 ||X + Y/gamma||^2 + M/(2 gamma delta) ||Y||^2

    Under identical key/value/mask sequences the audit gives
    `Vdot = -m ||X k||^2 - ||Y||^2/delta <= 0`, so initial-state differences do
    not grow in this fixed weighted norm. That is a statement about two
    trajectories on the SAME inputs: it does not compare different inputs or
    parameters, does not guarantee retrieval, and does not establish
    bounded-input bounded-state behaviour. The objective's nullspace permits
    non-decaying differences.
    """
    delta = gamma * T - M
    if delta <= 0:
        raise ValueError(f"V is singular at delta = gamma*T - M = {delta}; the "
                         f"proof needs 0 < M < gamma*T")
    import numpy as _n
    X = _n.asarray(X, dtype=_n.float64); Y = _n.asarray(Y, dtype=_n.float64)
    return float(0.5 * gamma * _n.sum((X + Y / gamma) ** 2)
                 + M / (2.0 * gamma * delta) * _n.sum(Y ** 2))
