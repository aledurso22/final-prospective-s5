# Specification: a common placement whose generalized equation recovers
# literal TSS exactly

17 September 2026. Written for
`PROSPECTIVE_EXACT_CONTAINMENT_PRIORITY_2026_09_17.md`, which supersedes
launch preparation for the matched-retention brief. Corrected on
17 September 2026 for the static review of `bdc1c19`
(`PROSPECTIVE_TSS_CONTAINMENT_REVIEW_bdc1c19_2026_09_17.md`, R1-R3 and the
reporting clarifications); the amendment record is §9.

**Static specification only: nothing is implemented, no numerical work was
run, and no launch is requested.** All completed studies, artifacts and
verdicts are preserved unchanged; the matched-retention comparison of the two
existing Momentum corrections remains a deferred question.

Companion: `docs/PROSPECTIVE_PARENT_EQUATION_NOTE.md` (why the two *existing*
corrections do not contain one another under token-varying gates).

## 1. Placement, parameter map and the exact boundary identities

### 1.1 The proposed placement

A processing state `y` is driven by the masked associative residual and its
newly computed output enters the **unchanged** Momentum update:

    R_t      = m_t (alpha_t W_prev k_t - v_t) k_t^T                     (R)
    A (y_next - y) = M (y - y_prev) + h^2 (R_t - y) + h T (R_t - R_prev) (Y)
    A        = M + h (gamma + T)
    U_next   = mu_t U_prev + eta_t y_next                                (U)
    W_next   = alpha_t W_prev - beta_t U_next                            (W)

Every right-hand side uses the **old** states: `W_prev`, `y`, `y_prev`,
`R_prev`, `U_prev`. `alpha, beta, mu, eta` are the pinned native Momentum
gates, unchanged, and `h = 1` token interval. This is the same master law
applied to a residual-processing state. It is **not** a claim that the
completed two-carry Momentum correction already implements this family.

### 1.2 Exact literal-TSS boundary

At `M = gamma = 0`, `T > 0`: `A = h T`, and (Y) becomes

    h T (y_next - y) = h^2 (R_t - y) + h T (R_t - R_prev)
    =>  y_next = y + (h/T)(R_t - y) + (R_t - R_prev).                    (TSS)

This is literal TSS Eq. (17) driven by `f = R_t`, at the **same** clock `h`,
the same state initialization (§1.4), the same same-token output convention
and the same downstream gates. The identity is exact and holds token by token
for arbitrary sequences, including writes, queries, idle tokens and episode
boundaries, because (Y) is a pointwise algebraic identity in the old states.

### 1.3 Exact native-Momentum point, and the two-tap correspondence

- **Native Momentum:** `M = 0`, `T = 0`, `gamma = h` gives `A = h^2` and
  `y_next = R_t`, so (U)-(W) are exactly the native update and `y_prev`,
  `R_prev` are dormant. This point is inside the admissible domain (§2), but
  it lies at `gamma = h != 0`, so it is **outside the literal-TSS boundary**
  `M = gamma = 0`. It is a point of the generalized family only; it is never a
  setting of the TSS-only arm (§5, §6).
- **Two-tap operator:** `M = 0` and `gamma + T = h` give
  `y_next = R_t + (T/h)(R_t - R_prev)`, i.e. the learned operator with
  `kappa = T/h`, `gamma = h(1 - kappa)`, `T = kappa h`.
  **Therefore `kappa > 1` requires `gamma < 0`.** The completed run learned
  `kappa ~ 1.87-2.14`, so that operator is **not** inside this
  nonnegative-`gamma` family. It is kept as a separate strong comparator,
  explicitly outside the containment claim. No alternative admissible mapping
  is claimed; if one is proposed later it needs its own proof.
- At `T = h` (so `kappa = 1`, `gamma = 0`, `M = 0`) the filter is
  `y_next = 2 R_t - R_prev`: literal TSS **and** the two-tap operator at
  `kappa = 1`. Only that point may be called both.

### 1.4 Initialization and streaming

- Episode start: `W = U = y = y_prev = R_prev = 0`. The first token then gives
  `y_next = (h^2 + hT) R_0 / A`, which at the TSS boundary is
  `(1 + h/T) R_0`, and at `T = h` is `2 R_0` - consistent with (TSS) read with
  `y = 0`, `R_prev = 0`.
- Streaming: all five carries `(W, U, y, y_prev, R_prev)` cross chunk
  boundaries; a chunked rollout must agree with the unchunked one at the
  declared trajectory tolerances (§4.3), not bitwise.
- Output convention: the `y_next` computed from token `t` drives token `t`'s
  Momentum update (causal, same-token). This is the convention already used by
  the existing laws; it must be stated whenever the TSS identity is quoted.
- Masking: the mask enters `R_t` only, before any difference; `R_prev` is the
  previous **masked** residual and is never re-masked. Full BPTT; no query
  label, category, age, future input or family identity enters the recurrence.

## 2. Coefficient domain, learnable coordinates and the feasibility repair

### 2.1 Physical domain and numerical policy, kept apart

**Physical/derivational domain** (what the derivation and §4 require):

    M >= 0,  gamma >= 0,  T >= 0,
    gamma + T > 0,                                                      (D1)
    A = M + h (gamma + T) > 0,                                          (D2)
    4M + 2h (gamma + T) > h^2.                                          (D3)

(D1) was derived in §4 but was **missing** from the earlier declared set and
from the repair. It is not implied by (D2)-(D3): with `h = 1`, `M = 1`,
`gamma = T = 0` one has `A = 1 > 0` and `4M + 2h(gamma+T) = 4 > 1 = h^2`, yet
the homogeneous polynomial is `z^2 - z + 1`, whose roots have modulus one -
not a strictly stable driven filter. That point is now excluded, and it is a
required fixture (§4.3).

**Numerical policy** (executed arithmetic, *not* part of the derivation).
A float comparison against zero is perfectly possible; the executed set
nevertheless uses declared positive gaps, as a **numerical robustness
policy**, so that a repaired point is not placed exactly on a boundary
(clearance of d95266d, s1):

    gamma + T >= g_min,                                                 (N1)
    4M + 2h (gamma + T) >= h^2 (1 + delta_filter),                      (N2)

with `g_min > 0` and `delta_filter > 0` declared in advance as numerical
constants (`g_min = 2^-10 h`, `delta_filter = 1e-3`, the value already used
for the existing coefficient repair). (N1) implies (D2) because
`A >= h g_min > 0`. These margins are a numerical policy: they are never
presented as physical requirements, and the reported coefficients state both
the executed values and the achieved slacks.

**Scope of the gaps, and the executed acceptance gate** (clearance s1). The
gaps do **not** certify strict stability of the *rounded* recurrence: for
unbounded `M`, `A = M + h(gamma+T)` can round to `M`, `M/A` can round to one,
and a slack can vanish when the actual coefficients are formed - e.g.
`M = 1e18, gamma = 0, T = g_min` meets both gaps yet executes `b = M/A = 1`.
What the gaps do give is a positive gap *in exact arithmetic* for every
repaired point; no universal rounding guarantee is claimed. Acceptance is
therefore a **gate on the executed coefficients** (as corrected for the
review of 7613c86): the law is executed in its algebraically equivalent
coefficient form `y_next = a y - b y_prev + c R - d R_prev`, whose rounded
coefficients each compiled program forms once and returns; every executed
`M, gamma, T, A, a, b, c, d` must be finite and `1 - a + b`, `1 + a + b`,
`1 - b` strictly positive for the rounded `a, b` of `z^2 - a z + b` - for each
update's own forward pass (executed dtype) and at every checkpoint (exact
rational classification). A violation refuses the update and fails the run.
This is a result about those rounded coefficients of the isolated filter, not
a theorem about every floating-point trajectory or the closed-loop memory.

Both exact points survive the executed set: literal TSS `M = gamma = 0`,
`T = h` has `gamma + T = h`, slack `2h^2 - h^2(1+delta) > 0`; native
`M = T = 0`, `gamma = h` has `gamma + T = h` and slack `2h^2 - h^2(1+delta)`.

### 2.2 Learnable coordinates and the boundary

`M`, `gamma`, `T` are stored **directly** as three real scalars, never through
`exp` or `softplus`: a positive log-parameterization cannot represent
`M = gamma = 0` as an executed point, which is exactly the boundary this study
exists to test. The policy mirrors the existing `kappa`: initialized at the
declared point, no forward clipping, no branch on their values, and a
**post-update feasibility repair** that leaves the optimizer state untouched,
counts events and records the achieved slacks.

**Feasibility repair** (deterministic, applied in this order after each
optimizer update):

    1. M <- max(M, 0);  gamma <- max(gamma, 0);  T <- max(T, 0)
    2. T <- max(T, g_min - gamma)                                  [(N1)]
    3. T <- max(T, (h^2 (1 + delta_filter) - 4M) / (2h) - gamma)   [(N2)]

This is a **deterministic feasibility repair, not an orthogonal or Euclidean
projection**: it is component clamping plus a minimal increase of one chosen
coordinate, and it does not return the nearest admissible point. The
asymmetric choice to repair through `T` is declared in advance, because
raising `T` always restores (N1) and (N2) and never leaves the nonnegative
orthant. Step 3 is idempotent given step 2, and the whole repair is idempotent;
that is a required fixture. Repairs are counted, logged with the pre-repair
coordinates, and reported per run.

**Admissible departure directions from the TSS boundary.** From
`M = gamma = 0` the admissible departures are `M` increasing, `gamma`
increasing, and `T` moving subject to (N1)-(N2). The direction `gamma < 0` -
which is what the learned two-tap operator with `kappa > 1` needs - is **not**
admissible. Departures are one-sided at the boundary, so a negative proposal
is clamped and recorded as a repair event, not as a learned value.

**Physical interpretation and the additional assumptions.** `M`, `gamma`, `T`
keep their mechanical meaning (mass, damping, prospective horizon) for the
processing state. What is **additional** to the derivation, and must be
labelled as such: (i) coefficients are episode-fixed scalars, piecewise frozen
per token, so coefficient derivatives never appear; (ii) the placement - 
filtering the residual and feeding the unchanged Momentum update - is a
modelling choice, not a consequence of the master law; (iii) the broad
equation, its `M = gamma = 0` boundary and the passive two-compartment circuit
sector are three different things. **The admissible coefficient domain above
is broader than the passive-compartment domain**: nonnegative `M, gamma, T`
does not imply that a strict positive-component circuit realizes the point, and
no physical realizability is inferred from feasibility here.

## 3. Carry and parameter costs

| Rule | Carry matrices | Carry reals | Stored params | Trainable params |
|---|---|---:|---:|---:|
| Native Momentum | W, U | 128 | 569 | 569 |
| Existing generalized correction | W, Q | 128 | 570 | 570 |
| Learned two-tap operator | W, U, R_prev | 192 | 570 | 570 |
| **`generalized_processing`** (M, gamma, T trainable) | W, U, y, y_prev, R_prev | **320** | **572** | **572** |
| **`tss_processing`** (M = gamma = 0 stored constants, T trainable) | W, U, y, R_prev | **256** | **572** | **570** |
| Generic `M`-fixed-zero family (gamma and T trainable) | W, U, y, R_prev | 256 | 572 | 571 |

**The old 128-value carry claim does not apply to this realization.** The
`y_prev` matrix is required only by the `M` term, so an arm with `M` fixed at
zero does not carry it. The last row is a *generic* restricted family that
still trains `gamma`; it is **not** the literal-TSS arm, which additionally
fixes `gamma = 0` as a stored constant and therefore has one trainable
coefficient. Stored constants are counted in the stored column and excluded
from the trainable column, and both columns are reported per arm. Per-token
compute adds a few element-wise operations on 64-value matrices and no new
matrix product; preflight measures the actual step cost rather than assuming
it.

## 4. Numerical risk and the checks that are actually valid

### 4.1 Isolated filter

With episode-fixed coefficients the homogeneous part of (Y) is

    y_next = [1 + M/A - h^2/A] y - (M/A) y_prev,
    characteristic polynomial  z^2 - [1 + M/A - h^2/A] z + M/A.

Its strict Jury conditions are

    p(1)  = h^2/A > 0            <=>  A > 0,                        (J1)
    p(-1) = 2 + 2M/A - h^2/A > 0 <=>  4M + 2h(gamma + T) > h^2,     (J2)
    |det| = M/A < 1              <=>  gamma + T > 0,                (J3)

which is the condition quoted in the priority note, and at the TSS boundary
requires `T > h/2`. (J3) is the condition restored to the domain in §2.1. This
concerns the **isolated driven processing filter only**.

**What must not be claimed.** The existing `kappa` bound of the Momentum
correction is a different filter and **must not be transferred**. This
polynomial says nothing about the closed-loop associative memory (where `R`
depends on `W`), nothing about switching across tokens with varying gates, and
nothing about float32 rounding. No inherited stability certificate is used.

### 4.2 Numerical-domain policy for checks

All numerical work runs on the cluster. Two comparison classes, never mixed:

- **Bitwise equality** is asserted only for (i) stored parameter leaves that
  must not change (frozen constants `M`, `gamma` in `tss_processing`, frozen
  backbone leaves) and (ii) genuinely shared executed operations - the same
  compiled value copied or routed to two consumers.
- **Tolerance comparison, finite first** is used for everything computed by
  two separately compiled recurrences: the streaming/chunking check, the
  native-recovery check, the TSS-recovery check, carries, logits, losses and
  gradients. Each comparison first requires all compared quantities finite,
  then applies the already declared identity/trajectory/gradient tolerances
  and near-zero policy for the dtype in use, and prints absolute and relative
  discrepancies. Mathematical identity of two separately jitted graphs does
  **not** establish bitwise equality, and a rounding difference must not fail
  an otherwise correct implementation.

**Gradient checks.** Nonzero derivatives are not guaranteed for every loss or
sequence, so *nonzero* is not the criterion. The criterion is **agreement with
an independently coded sequential float64 sensitivity**, including legitimate
zeros, at the declared gradient tolerance. For each new response direction
(`M`, `gamma`, `T`) a fixture is constructed whose analytic sensitivity is
nondegenerate **at the true start point**, and it is justified analytically
before it is coded. A reported zero must be classified: an exact analytic zero
(reference also zero) versus a magnitude below the float32 resolvable
threshold (reported as unresolvable, not as a correct zero). At the boundary
`M = gamma = 0` the derivative is taken **inward only** - a one-sided inward
directional derivative or the analytic sensitivity - and never by
differentiating through the feasibility repair and never by evaluating
finite-difference points with negative mass or damping, which are not
admissible models.

### 4.3 Checks to specify before coding

1. boundary identity: `M = gamma = 0` matches an independently coded literal
   TSS processing recursion for arbitrary token sequences, same clock,
   initialization, output convention and downstream gates, at trajectory
   tolerances;
2. native point: `M = 0, T = 0, gamma = h` matches native Momentum at
   trajectory tolerances, with dormant `y_prev`/`R_prev` and matched incoming
   carries, and with the frozen stored leaves compared bitwise;
3. two-tap correspondence for `kappa <= 1` (`M = 0`, `gamma = h(1-kappa)`,
   `T = kappa h`), and an explicit fixture showing `kappa > 1` maps to
   `gamma < 0` and is refused by the repair;
4. streaming: five-carry chunk-boundary agreement at trajectory tolerances,
   and the episode-start values of §1.4;
5. gradients: the §4.2 policy for `M`, `gamma`, `T` through full BPTT at the
   boundary (inward) and at an interior point;
6. feasibility repair: nonnegativity clamping, (N1) and (N2) restoration,
   idempotence, event counting, updated-parameter provenance, **and the
   `h = 1, M = 1, gamma = T = 0` counterexample of §2.1, which must be
   repaired rather than accepted**; plus a fixture that both exact points are
   fixed points of the repair;
7. executed-recurrence diagnostics: per run, the executed normalized
   coefficients `M/A`, `h^2/A` and all three Jury slacks (J1)-(J3), computed
   finite-first from the executed values, together with W/U/y norms - reported
   as diagnostics, never as a stability certificate;
8. production float32 probe in an x64-disabled process, reusing the existing
   dtype, finiteness and tolerance policy, and rejecting a non-finite updated
   parameter or optimizer tree before any movement, equality or preservation
   decision.

## 5. Minimal comparison protocol (proposed, for review)

Reuses without change: the task and input contract, the read-only source
restore and checksum verifier from replication `20260917-011842` (source 500
development, 501-503 final), paired batches per seed, the unweighted
query-mean cross-entropy, the optimizer, full BPTT, evaluation, supervisor,
terminal verdict, finalizer and digest.

**Arms** (all full continuation of the pretrained backbone; no coefficient-only
arms, no blend, no router):

| Arm | Coefficients | Trains | Role |
|---|---|---|---|
| `tss_processing` | `M = gamma = 0` **stored constants**, never written; `T` trainable | backbone + T | literal TSS at every executed point |
| `generalized_processing` | `M, gamma, T` trainable, started at `M = gamma = 0`, `T = T0` | backbone + 3 | the actual extension; **starts function-matched to `tss_processing`** |
| `native_full` | - | backbone | active baseline |
| `operator_full` | `kappa` trainable from 0 | backbone + kappa | strong learned two-tap comparator, outside containment |
| `native_frozen` | - | nothing | descriptive anchor only |

`T0` is declared in advance, not searched: **`T0 = h = 1`**, which satisfies
(N1)-(N2) and is simultaneously literal TSS and the two-tap point `kappa = 1`.
The update-zero agreement of `generalized_processing` and `tss_processing` is
verified per source seed before training, at identity tolerances.

**`tss_processing` stays literal TSS.** Its `M` and `gamma` are stored
constants at zero, excluded from the gradient and from the repair's writable
coordinates, and asserted bitwise unchanged after every update. The native
point (`gamma = h, T = 0`) is a point of the generalized family and is **never
written into this arm**: doing so would change the family while keeping the
label.

**Work:** two learning rates per family (0.003, 0.01), 200 updates,
development checkpoints saved at updates 0, 25, 50, 100, 200 - the same
opportunities for every family. 4 trained families x 2 rates = 8 development
trajectories (1,600 updates); 4 x 3 seeds = 12 final trajectories, each
stopping at its selected update (<= 2,400 updates); <= **4,000 updates**, plus
4 frozen-source evaluations. Streams are fresh, and their full ranges are
declared and asserted disjoint from every previous study and fixture before
execution. One 600-second cap covers checks, preflight, training, checkpoint
persistence, evaluation and finalization; preflight measures the actual paths
and refuses as INCOMPLETE if they do not fit. No retries, no trimming.

## 6. Selection and assessment (equal opportunities, frozen in advance)

1. **Native first.** Over its development checkpoints (both rates x the five
   update counts, the identical update-zero checkpoint deduplicated), select by
   revision macro accuracy, then lower revision cross-entropy, then fewer
   updates, then lower learning rate. Record its development retention
   `R_native` and recall `C_native`.
2. **Feasibility.** For each extension family separately, a development
   checkpoint is *feasible* if **retention >= R_native AND recall >=
   C_native** - no minus-one-point allowance. Among feasible checkpoints the
   ordering is: revision macro accuracy, then lower revision cross-entropy,
   then fewer updates, then lower learning rate. This ordering is the single
   development ordering used everywhere below.
3. **`tss_processing`.** Select the best feasible TSS checkpoint. If none is
   feasible, record **TSS constrained selection: INFEASIBLE**. It is *not*
   mapped to native and not relabelled: native may be used as an explicit
   external deployment fallback, but such a selection is reported as **native**,
   never as TSS. **Deterministic final endpoint** (clearance s3): the pure-TSS
   final endpoint is *always* trained - the best feasible development
   checkpoint if one exists, otherwise the best unconstrained one by the same
   ordering, flagged **diagnostic**, constraint-failing and not a feasible
   contender. The same feasible-then-diagnostic rule fixes the trained
   generalized and operator endpoints. These use the existing final slots
   (four trained families x three sources, <= 4,000 updates in total), and
   all of them are frozen before any final run.
4. **`generalized_processing`.** Its candidate set is: (a) feasible trained
   generalized checkpoints; (b) the **native fallback** - the selected native
   checkpoint mapped into the generalized family at its valid native point
   `M = 0, T = 0, gamma = h`, with correct episode-start carries and verified
   value equivalence; (c) the **TSS fallback** - a *genuinely selected,
   feasible* TSS checkpoint placed at the exact boundary `M = gamma = 0`. If
   TSS was infeasible, or if TSS itself fell back externally to native, there
   is **no** TSS fallback: a native choice is never wrapped in a supposed
   `M = gamma = 0` map. Rank the available fallbacks (b), (c) by the same
   development ordering; a trained generalized checkpoint is preferred only if
   its development revision is **strictly higher** than the best available
   fallback's. Otherwise the best fallback is selected and flagged.
5. **`operator_full`.** Best feasible operator checkpoint, with the native
   fallback at `kappa = 0`; same strict-improvement rule; infeasible recorded
   as such.
6. **Selection record.** Every selection stores: executed family, the exact
   executed parameter map (`M, gamma, T` or `kappa`), the source checkpoint it
   came from, the full fallback chain, the feasibility flag, the learning rate
   and the update count. Family, rate, update count and flags are frozen
   before any final run. Final trajectories train from each seed's own source
   with the selected recipe; a fallback uses the corresponding final native (or
   genuine TSS) checkpoint mapped without further optimization. Nothing is
   selected on final validation or held-out data; a deep-copied selection is
   re-checked after evaluation.
7. **Held-out**: one fresh common set, generated only after all final models
   and selections are frozen; only the selected endpoints (and any predeclared
   diagnostic TSS endpoint) are evaluated. Development checkpoint curves may be
   plotted; held-out checkpoints are never searched.

**Verdicts, each reported separately**, with paired per-seed differences and
all four query categories by full name:

- `generalized_processing` - `tss_processing`: does departing from the exact
  TSS boundary help, starting from the same function? **A genuine extension
  comparison needs a trained generalized endpoint against a trained
  literal-TSS endpoint** (clearance s2). Scientific model comparisons are
  kept apart from deployment selections. If literal TSS has no development
  checkpoint meeting the retention constraints, the *constrained* screen is
  reported as infeasible/unavailable; the comparison of the two genuine
  trained endpoints is still reported, and keeps its constraint-failing
  label. A native model is never called TSS.
- `generalized_processing` - `native_full`, and `tss_processing` -
  `native_full`: matched-retention screens against the active baseline.
- `generalized_processing` - `operator_full`: reported separately and
  explicitly **outside** the containment claim (the operator's learned
  `kappa > 1` is not in this family).

A promising matched-retention signal requires mean held-out revision >= +1 pp,
all three paired revision differences positive, and mean retention **and**
recall differences >= 0. Per-seed retention or recall decreases are reported
even when the means satisfy it. This is a finite-sample screen, not
statistical noninferiority and not a no-loss guarantee. A trade-off (one wins
revision, the other retention) is reported as a trade-off.

**Fallbacks are reported as fallbacks.** A selected fallback is a real,
feasible deployment choice, and that is all it is: it does **not** show that
joint optimization learned to revert to a subfamily, and it is never counted as
an improvement. The trained generalized endpoint's own scores are reported
separately from the fact that model selection could reuse a baseline.

## 7. Scope limits carried forward

- Literal TSS containment by this placement is established (§1.2). Containment
  of the strong learned two-tap operator is **not**: its learned `kappa > 1`
  needs `gamma < 0`, outside the nonnegative-`gamma` family.
- The admissible coefficient domain is broader than the passive-compartment
  domain; nonnegative `M, gamma, T` does not imply passive physical
  realizability.
- No Gated DeltaNet arm: no joint literature win and no SOTA claim follows.
- Frozen-token and isolated-filter spectral conditions do not establish
  closed-loop or switching stability.
- Three seeds on this small associative task are not significance, a benchmark
  or SOTA, and this specification promises no optimization or held-out gain.

## 8. What this document does not do

It implements nothing, launches nothing and runs no numerical work. Coding
begins only after the equation and comparator contract above is cleared, and
then reuses the existing task, source restore, evaluation and orchestration
code. Cluster execution remains a separate reviewed step.

## 9. Amendment record (review of bdc1c19)

Implementation clearance of d95266d, incorporated while coding (no separate
specification round): (s1) the executed filter's acceptance is a gate on the
rounded coefficients the production step executes, and the gaps are
re-scoped as a robustness policy without a universal rounding guarantee; (s2)
verdict wording separates the scientific generalized-versus-literal-TSS
comparison from deployment selections; (s3) every family's final endpoint is
fixed now by the feasible-then-diagnostic rule, inside the existing slots.
Implementation note: both processing arms execute the same five-carry step, so
the **implemented** carry is 320 reals in both; 256 is only the theoretical
minimum of a law with `M` fixed at zero, not the implemented cost.

Implementation review of 7613c86: the law is executed in its algebraically
equivalent coefficient form `y_next = a y - b y_prev + c R - d R_prev`
(`a = [2M + h(gamma+T) - h^2]/A`, `b = M/A`, `c = [h^2 + hT]/A`, `d = hT/A`),
and the acceptance gate classifies the rounded `a, b` each compiled program
returns (`z^2 - a z + b`; slacks `1 - a + b`, `1 + a + b`, `1 - b`), which
replaces the basis-response extraction; the native-point recovery is decided
by direct trajectory agreement at TRAJ32, not by the withdrawn 1e-3 metric
tolerance. Dispositions: protocol s11.


- **R1** - `gamma + T > 0` was derived in §4 but missing from the declared
  domain and the repair; added as (D1) with the executed gap (N1), with the
  `h = 1, M = 1, gamma = T = 0` counterexample recorded and made a required
  fixture. Numerical margins are now stated as numerical policy, separate from
  the derivation, and the executed normalized recurrence and all three Jury
  slacks are checked finite-first. The coordinate map is renamed a
  deterministic feasibility repair (its asymmetric repair through `T` remains
  declared), not an orthogonal projection. Both exact points are unchanged.
- **R2** - `tss_processing` is held strictly at `M = gamma = 0` with stored
  constants; the native point is no longer mapped into it. Infeasible TSS
  selection is recorded as infeasible, an external native fallback is labelled
  native, an optional pure-TSS diagnostic endpoint is predeclared as
  constraint-failing, the generalized arm's fallback set and strict-improvement
  rule are deterministic, a native choice is never wrapped as a TSS map, and
  the selection record and verdict labelling are specified.
- **R3** - bitwise assertions are restricted to unchanged stored leaves and
  genuinely shared executed operations; streaming, native recovery and TSS
  recovery use finite-first tolerance comparisons. Gradient checks require
  agreement with an independent sensitivity including legitimate zeros, use
  analytically justified nondegenerate fixtures at the true start point,
  distinguish exact zeros from float32-unresolvable ones, and at the boundary
  use inward or analytic derivatives rather than differentiating through the
  repair or evaluating negative-mass points.
- **Reporting** - per-arm stored versus trainable counts with a separate
  `tss_processing` row (569 backbone + `T` trainable, `M`, `gamma` stored
  constants, 256-real carry), distinct from the generic `M`-fixed-zero family;
  the domain/passive-sector distinction; and the fallback reporting rule.

Budget, loss, equations, cap and all completed verdicts are unchanged by these
corrections.
