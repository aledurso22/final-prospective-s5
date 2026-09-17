# Audit: the prospective correction versus ordinary prospective residual
# processing on the same Momentum memory

17 September 2026. Brief audited:
`PROSPECTIVE_SAME_BACKBONE_AND_RETENTION_BRIEF_2026_09_17.md`.

**Method.** Every identity below was re-derived by hand from the executed
update equations. No numerical computation was run locally. The cluster
checks test the same identities with independent implementations; none has
been executed yet.

**Verdict.** The brief's algebra is correct. For **constant eta and mu** the
candidate and the ordinary residual operator are the *same* law under an
invertible state map — closed loop, not merely an open-loop transfer
argument. They differ only when `eta/mu` changes between consecutive tokens.
The audit adds: the exact necessary-and-sufficient condition for equality,
the frozen-token stability of the operator realization (derived, not copied),
and the resolution of the literal TSS Eq. (17) comparator.

## 0. Notation and the executed laws

Shared shell, unchanged (`prospective_momentum/model.py`):

    Wbar_t = alpha_t W_(t-1),   R_t = m_t (Wbar_t k_t - v_t) k_t^T,
    d_t = (1 - mu_t)/mu_t,      q = beta eta

**Candidate** (`prospective_momentum`, executed today):

    Qn_t = mu_t Q_(t-1) + eta_t R_t                      (native intermediate)
    Q_t  = Qn_t - kappa eta_t d_t R_t
    W_t  = Wbar_t - beta_t Qn_t - kappa beta_t eta_t R_t

Equivalently, as the brief writes it:

    Q_t = mu_t Q_(t-1) + eta_t (1 - kappa d_t) R_t
    W_t = Wbar_t - beta_t [ Q_t + kappa (eta_t/mu_t) R_t ]

These two forms agree: substituting Q_t into the second line gives
`W_t = Wbar_t - beta_t [ mu_t Q_(t-1) + eta_t (1 + kappa) R_t ]`, which is the
executed form, because `1 - kappa d_t + kappa/mu_t = 1 + kappa`. Checked.

**Ordinary prospective operator** (proposed comparator,
`ordinary_prospective`):

    Rpros_t = R_t + kappa (R_t - R_(t-1)) = (1+kappa) R_t - kappa R_(t-1)
    U_t = mu_t U_(t-1) + eta_t Rpros_t
    W_t = Wbar_t - beta_t U_t

with h = 1, `R_(-1) = 0` at an episode start, the write mask applied to R
**before** the difference, and `Rpros` never re-masked (so the first idle step
after a write carries `-kappa R_(t-1)`). Old carries on every right-hand
side.

## 1. Fixed-gate equivalence is exact, and closed loop

Assume `eta_t = eta` and `mu_t = mu` constant within the episode; `alpha_t`,
`beta_t`, `m_t`, `k_t`, `v_t` may vary arbitrarily. Define

    U_t := Q_t + kappa (eta/mu) R_t.                                   (1)

**W agrees by construction.** The candidate's second form is exactly
`W_t = Wbar_t - beta_t U_t`, the operator's output equation.

**U obeys the operator recurrence.** From (1) at t-1,
`mu Q_(t-1) = mu U_(t-1) - kappa eta R_(t-1)`, so

    U_t = mu Q_(t-1) + eta (1 - kappa d) R_t + kappa (eta/mu) R_t
        = mu U_(t-1) - kappa eta R_(t-1) + eta (1 + kappa) R_t
        = mu U_(t-1) + eta Rpros_t.                                    ✓

**Initial states match.** At an episode start `Q_(-1) = 0` and `R_(-1) = 0`,
hence `U_(-1) = 0`: the two realizations start at the same mapped state with
no extra initialization convention.

**Closed loop.** `R_t` is a function of `W_(t-1)` and the token only. Since
the W equations coincide and the mapped states coincide, the identity holds
step by step with the state-dependent residual, not only for a prescribed
residual sequence.

**Scope of "invertible".** Given the cached residual, (1) is an invertible
affine relation between `Q_t` and `U_t`, so along a shared trajectory the two
realizations carry the same information. It is NOT a bijection between an
arbitrary three-matrix state `(W, U, R_(t-1))` and a two-matrix state
`(W, Q)`: the operator's reachable states are exactly those whose cached
residual is the one its own trajectory generated. The equivalence is a
statement about matched, reachable states.

**Consequences, to be respected in all labels.**

- For constant eta and mu these are the **same model class**, not two
  expressive classes. The candidate is a two-matrix realization of ordinary
  first-order prospective residual processing.
- Ordinary prospectivity already uses past information; the candidate does
  not, by itself, "make past information available" that the operator lacks.
- At `kappa = 0` both reduce exactly to native Momentum DeltaNet. The
  operator's previous-residual cache is then dormant but still stored and
  reported.

## 2. Where token-varying gates separate them

Now let eta and mu vary per token, with kappa fixed within the episode, and
use the gate-indexed map `U_t := Q_t + kappa (eta_t/mu_t) R_t`. The same
substitution gives

    U_t = mu_t U_(t-1) + eta_t (1+kappa) R_t
          - kappa mu_t (eta_(t-1)/mu_(t-1)) R_(t-1).                   (2)

The operator instead computes

    U_t = mu_t U_(t-1) + eta_t (1+kappa) R_t - kappa eta_t R_(t-1).    (3)

Signs and indices checked independently. Conditional on the same incoming
mapped state and the same residuals,

    (candidate - ordinary) in U_t
        = kappa [ eta_t - mu_t eta_(t-1)/mu_(t-1) ] R_(t-1).           (4)

**Exact condition for equality.** For `kappa != 0` and `R_(t-1) != 0`, the two
agree at token t if and only if

    eta_t / mu_t = eta_(t-1) / mu_(t-1),

i.e. the **ratio** eta/mu is constant across consecutive tokens. This is
weaker than requiring both constant: only the ratio matters. Constant eta and
mu (§1) is the special case.

**Interpretation.** The candidate transports the previous residual through
the current momentum gate (`mu_t eta_(t-1)/mu_(t-1)`); the operator scales it
by the current input gate (`eta_t`). The measured runs have
`mu ≈ 0.99997` on write tokens with `eta` varying over roughly 1.1–1.4, so
the difference is approximately `kappa (eta_t - eta_(t-1)) R_(t-1)`: **nonzero
in general; its magnitude is to be measured, not asserted.** Larger eta
variation or larger kappa can make it material.

Two further cautions. (4) compares the two laws at the SAME incoming mapped
state, residuals and parameters — it is a one-step statement. Separately
trained endpoints also differ because the two realizations follow different
optimization paths, so an endpoint difference between the trained arms is not
a direct numerical measurement of this one-step term.

**Gate-transported reference.** Equation (2) is implemented as a
check-only third realization (carry `W, U, R_(t-1), eta_(t-1), mu_(t-1)`). It
must match the candidate on arbitrary valid token schedules. That is a
correctness identity, not a trained competitor; a failure is an
implementation defect to fix before training.

## 3. Literal TSS Eq. (17) as a processing stage: resolved

For a processing state y driven by an externally supplied residual R, literal
Eq. (17) is

    y_(t+1) = y_t + (h/tau)(R_t - y_t) + R_t - R_(t-1).

**Special case tau = h.** The lag term cancels the state exactly:
`y_(t+1) = 2 R_t - R_(t-1)`, which is `Rpros_t` at `kappa = 1`. The
equivalence holds **only under the output-timing convention** that the state
produced by the token-t update drives the token-t write — the same convention
the candidate and the operator use: feeding the state just computed from the
current token to that token's memory update is causal. Eq. (17)'s index alone
does not mandate an extra input-token delay; an explicit pipeline that instead
reads the OLD processing state is a different timing choice, and would delay
the correction by one token. Whichever is used must be stated whenever this
equality is quoted.

**General tau.** Writing `lam = 1 - h/tau`,

    y_(t+1) = lam y_t + (1 + h/tau) R_t - R_(t-1),

a first-order lag (pole at lam) in series with a lead. The operator is a pure
two-tap FIR lead, `(1+kappa) - kappa z^-1`, with no pole. The pole vanishes
only at `tau = h`. So a learnable-kappa operator is **not** an exact
reproduction of Eq. (17) for arbitrary tau; it reproduces exactly the
`tau = h` member.

**Decision (to be confirmed before the arm list is frozen).** No literal TSS
processing-stage arm is added to this batch:

1. At `tau = h` it is algebraically the operator at `kappa = 1`, a special
   case of the trained-kappa operator arm; adding it would be an extra
   trained arm for an algebraically identical point.
2. For `tau != h` it is a different mechanism (extra state and pole), which
   would need its own recurrence, initialization, placement, coefficient
   policy, carry and cost declaration — none of which the two questions in
   the brief require.
3. The completed direct-fast-weight Eq. (17) arm remains valid **only** within
   its stated applicability limits and does not settle the same-backbone
   question. Nothing here relabels it.

Instead the batch carries a **free algebraic check** (no arm, no training):
literal Eq. (17) at `tau = h`, under the declared timing, must equal the
operator at `kappa = 1` on the same residual sequence. If the coordinator
wants the `tau`-learning processing stage as a trained comparator, it is a
separate specification; §6 of the protocol records what it would need.

## 4. Frozen-token stability of the operator realization (derived)

The operator keeps a third matrix, so its frozen-token analysis is not the
candidate's and is derived here rather than copied. One fixed unit key,
fixed gates, active write (m = 1), homogeneous (v = 0), key-aligned
components `x = W k`, `u = U k`, `r = R_(t-1) k`:

    R k = alpha x,   Rpros k = (1+kappa) alpha x - kappa r
    u' = mu u + eta[(1+kappa) alpha x - kappa r]
    x' = alpha(1 - q(1+kappa)) x - beta mu u + q kappa r
    r' = alpha x

so

    A_ord = [[ alpha(1 - q(1+kappa)), -beta mu,  q kappa ],
             [ alpha eta (1+kappa),    mu,      -eta kappa],
             [ alpha,                  0,        0        ]].

Expanding and collecting terms gives the monic characteristic polynomial
(note the determinant order: in odd dimension `det(A - zI) = -det(zI - A)`,
so the monic form is the second one)

    det(z I - A_ord) = z [ z^2 - (alpha + mu - alpha q(1+kappa)) z
                           + (alpha mu - alpha q kappa) ],
    det(A_ord - z I) = - det(z I - A_ord).

The bracket is **exactly the candidate's frozen-token characteristic
polynomial** (trace `alpha + mu - alpha q(1+kappa)`, determinant
`alpha mu - alpha q kappa`, audit of 16 September, §5). Therefore:

- the operator realization has the candidate's two eigenvalues plus a **zero**
  eigenvalue (the previous-residual cache is a pure feed-through state that
  does not feed back into the mapped (x, u) dynamics);
- the three Jury conditions and the bound

      kappa < [ (1+alpha)(1+mu) - alpha q ] / (2 alpha q)

  apply **unchanged** to the operator. This is a derivation, not an
  assumption, and it justifies reusing the existing projection and its
  declared numerical margin for the operator arm.

At `kappa = 0` the third column of `A_ord` vanishes: the cache is dormant, and
both realizations reduce to native Momentum. As before this is a frozen-token
statement for one fixed key and gate, **not** a switching-stability proof.

## 5. Carry and cost, counted honestly

| Rule | Carry matrices | Real numbers | Extra trained scalars |
|---|---|---:|---|
| Native Momentum | W, Q | 128 | 0 |
| Candidate | W, Q | 128 | 1 (kappa) |
| Ordinary operator | W, U, R_(t-1) | **192** | 1 (kappa) |
| Gate-transported reference (checks only) | W, U, R_(t-1), eta_(t-1), mu_(t-1) | 192 + 2 scalars | — |

The operator's previous-residual matrix is genuine streaming state: it is
carried across chunk boundaries and reset only at episode start. No claim of
equal carry is made. A reduced-state realization exists for constant gates
(the candidate itself, §1); no reduced-state realization is claimed for
token-varying gates, since (2) and (3) differ.

## 6. What this audit does and does not establish

- It establishes an exact fixed-gate equivalence and the exact condition
  under which token-varying gates break it.
- It establishes that the operator's frozen-token stability region is the
  candidate's.
- It does **not** establish which realization trains better, nor any
  superiority over TSS, nor any retention outcome. The retention intervention
  (coefficient-only continuation) is an experimental hypothesis, not a
  guarantee.
- Frozen-token conditions remain distinct from switching stability, and the
  fixed-coefficient QHM relationship continues to bound novelty claims.
