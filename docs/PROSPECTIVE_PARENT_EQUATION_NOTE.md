# Can the parent generalized equation recover the ordinary residual
# operator exactly under token-varying gates?

17 September 2026. Written for
`PROSPECTIVE_MATCHED_RETENTION_BRIEF_2026_09_17.md`, §2. Derived by hand; no
numerical work. Nothing here changes an executed equation: the matched-
retention study uses the existing laws unchanged.

**Status (17 September 2026):** the matched-retention launcher is **deferred**
by `PROSPECTIVE_EXACT_CONTAINMENT_PRIORITY_2026_09_17.md`. No study code,
protocol or launcher was written for it. The exact-containment question this
note leaves open at the application level is taken up, for a different and
explicitly specified placement, in
`docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md`.

**Two different questions, kept apart.**

1. **At the level of the broad master equation: containment is ESTABLISHED,
   not an open derivation** (§1). Its specified TSS-compatible discrete
   descendant reduces exactly to TSS Eq. (17) at `M = gamma = 0`, `T > 0`.
   That boundary is a property of the broad family and is recorded, not
   re-derived here.
2. **At the application level, between the two Momentum corrections this
   study compares: NOT established** (§2–§4). With the native gates fixed and
   the executed discretization, the generalized candidate reproduces the
   ordinary residual operator exactly **iff** `eta_t/mu_t` is constant
   between consecutive tokens. Letting the prospective horizon vary per token
   (`kappa_t`) does **not** remove that condition: it can match the
   previous-residual coefficient or the current-residual coefficient, not
   both. Exact recovery for arbitrary gate schedules needs a **second** free
   coefficient rescaling the native input drive — a different model.

**Consequently the matched-retention study is labelled as a comparison of the
two specified Momentum corrections, not of the entire generalized family
against the entire TSS family. Containment is not assumed anywhere in the
protocol, and no blend or router is implemented.**

## 1. The broad master equation and its TSS boundary (established)

Per `PROFESSOR_TO_GENERAL_PROSPECTIVITY_CONTRACT_2026_09_16.md` and the
brief's clarification, the broad master equation and its specified
TSS-compatible discrete descendant are

    M s'' + gamma s' + (1 + T D)(s - f) = 0,
    [M + h(gamma + T)](s_next - s) = M (s - s_prev)
                                     + h^2 (f - s) + h T (f - f_prev).

At `M = gamma = 0` with `T > 0` the left side is `h T (s_next - s)` and the
descendant reduces exactly to

    s_next = s + (h/T)(f - s) + f - f_prev,

which is literal TSS Eq. (17). This containment is recorded as established;
it is **not** the open question. Its scope caveat is also recorded: a strict
positive-component circuit parameterization need not admit `M = gamma = 0` as
an executed point, so this is a boundary of the broad family rather than a
claim about any particular circuit realization.

**What is open is the APPLICATION level.** The Momentum augmentation executed
here uses a different discretization (§2) with `M = h/beta > 0` and
`gamma = (1-mu)/(beta mu)`, under the pinned gate ranges, and therefore does
not expose that TSS boundary as an executed point. The ordinary
residual-processing comparator is additionally a different *placement*: it
filters the write residual and feeds the native Momentum update, rather than
applying the master law directly to W. Sections 2–4 settle the remaining
question for exactly these two executed laws.

## 2. The candidate's parent equation, assumptions and discretization

The parent is the generalized prospective law with a positive source scale,

    M Wddot + gamma Wdot + s R + T s Rdot = 0,
    R(W, t) = m (W k - v) k^T,                                        (P)

with, for the executed discretization (audit of 16 September, §1):

- coefficients **piecewise frozen within a token interval** of length h = 1,
  and mapped from the native Momentum gates by

      M = h/beta,   gamma = (1-mu)/(beta mu),   s = eta/(mu h),
      kappa = T/h,   Q = -P,   P = M Wdot + T s R;

- a **semi-implicit step**: the residual is explicit and evaluated at
  `Wbar = alpha W` (after the native forgetting), the linear friction is
  implicit, and the position update uses the new velocity;
- forgetting multiplies W only, never Q; Q is carried across tokens;
- `beta > 0`, `0 < mu < 1`, so M, gamma, s are positive and finite.

Solving the frozen step gives the executed candidate

    Q_t = mu_t Q_(t-1) + eta_t (1 - kappa d_t) R_t,  d_t = (1-mu_t)/mu_t,
    W_t = Wbar_t - beta_t [ Q_t + kappa (eta_t/mu_t) R_t ].           (C)

This is the only place the parent enters: **(C) is what the parent yields
under these assumptions**, and it is what the study executes. Using
token-dependent coefficients inside (P) is a modelling choice, not the ODE
with time-varying coefficients — the derivatives of the coefficients would
otherwise appear.

## 3. The two mapped recurrences

In the mapped coordinate `U_t = Q_t + kappa (eta_t/mu_t) R_t` the candidate
obeys

    U_t = mu_t U_(t-1) + eta_t (1+kappa) R_t
          - kappa mu_t (eta_(t-1)/mu_(t-1)) R_(t-1),                  (1)

while the ordinary residual operator obeys

    U_t = mu_t U_(t-1) + eta_t (1+kappa) R_t - kappa eta_t R_(t-1).   (2)

Both use the same W equation `W_t = Wbar_t - beta_t U_t`, the same gates and
the same episode-start state `U_(-1) = 0`, `R_(-1) = 0`. Their conditional
one-step difference, at the same incoming mapped state and residuals, is

    kappa [ eta_t - mu_t eta_(t-1)/mu_(t-1) ] R_(t-1).                (3)

## 4. Allowing a token-varying horizon does not give containment

Let the parent's horizon vary per token, `T_t = kappa_t h`, keeping the native
gates and the same discretization. Repeating the elimination with the
gate-and-horizon indexed map `U_t = Q_t + kappa_t (eta_t/mu_t) R_t`:

    U_t = mu_t U_(t-1) + eta_t (1 + kappa_t) R_t
          - kappa_(t-1) mu_t (eta_(t-1)/mu_(t-1)) R_(t-1).            (4)

Matching (4) to the operator with a constant horizon kappa requires **both**

    (a) previous residual:  kappa_(t-1) mu_t eta_(t-1)/mu_(t-1) = kappa eta_t,
    (b) current residual:   eta_t (1 + kappa_t) = eta_t (1 + kappa).

Condition (a) alone is solvable: it is the recursion
`kappa_t = kappa_(t-1) (mu_t/mu_(t-1)) (eta_(t-1)/eta_t)`, whose solution is

    kappa_t = c mu_t / eta_t,   c constant,

i.e. the horizon must be scheduled inversely to the input gate. But (b) forces
`kappa_t = kappa` constant, and the two are compatible only when
`mu_t/eta_t` is constant. Hence:

> **Within the declared discretization, with the native gates fixed and one
> prospective horizon (constant or gate-scheduled), the candidate reproduces
> the ordinary operator exactly if and only if `eta_t/mu_t` is constant
> across consecutive tokens.** The condition is the same one found in the
> same-backbone audit; a token-varying horizon does not weaken it.

Equivalently: matching the previous-residual coefficient by scheduling
`kappa_t = c mu_t/eta_t` changes the current-residual coefficient from
`eta_t(1+kappa)` to `eta_t + c mu_t`. One can match the timing or the gain,
not both.

## 5. What exact recovery would require

Allow, in addition, an independent drive scale `s_t` (equivalently an
effective input gate `eta~_t = mu_t h s_t` decoupled from the native
`eta_t`). Matching (2) then needs

    eta~_t (1 + kappa_t) = eta_t (1 + kappa),
    kappa_(t-1) mu_t eta~_(t-1)/mu_(t-1) = kappa eta_t,

which can be propagated forward from an initial condition — but only by
setting `eta~_t != eta_t`, i.e. **by changing the native input gate**. That is
a second free coefficient per token and a different model: no longer "the same
pretrained backbone plus one scalar". It is not derived from (P) with the
declared assumptions, and it is not implemented here.

Two further cautions:

- A convex blend `lambda (candidate) + (1-lambda)(operator)` trivially
  contains both updates, but it is **not** derived from the neuronal
  equation, and a global model-selection fallback is not an adaptive
  recurrence. Neither is implemented in this batch.
- Because (3) is a one-step statement at matched states and parameters,
  differences between **separately trained endpoints** cannot be attributed
  to it: gates, keys and trajectories have also changed. The completed
  same-backbone run is read this way.

## 6. Consequences for the matched-retention study

- Neither law is a special case of the other for arbitrary token-varying
  gates; neither has a blanket information advantage (both use one previous
  residual). The protocol therefore treats them as **two sibling laws**, and
  the "which extension would the same ordering pick" record is labelled
  **selection between two algorithms**, never a learned generalized
  recurrence.
- The only exact nesting used operationally is the uncontested one: at
  `kappa = 0` both laws are exactly native Momentum, with the operator's
  previous-residual cache dormant. That is what the native-fallback mapping
  relies on, and it is verified numerically per selection.
- At fixed gates both executed laws satisfy the same second-order W
  recursion, `W_t - (alpha+mu) W_(t-1) + alpha mu W_(t-2) = -beta eta [R_t +
  kappa (R_t - R_(t-1))]`, which is structurally the descendant's form with
  the identification above; under varying gates they separate exactly as (3)
  says. That structural parallel is not an executed-point containment claim.
- Scope limits already established are unchanged: the ordinary residual
  operator is not arbitrary-timescale TSS Eq. (17); the candidate's learned
  sector need not be a passive two-compartment circuit; frozen-token spectral
  checks do not establish switching stability. A future full-family
  implementation must preserve the known `M = gamma = 0` boundary rather than
  invent a blend to replace it.
