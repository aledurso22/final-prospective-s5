# Specification: a common placement whose generalized equation recovers
# literal TSS exactly

17 September 2026. Written for
`PROSPECTIVE_EXACT_CONTAINMENT_PRIORITY_2026_09_17.md`, which supersedes
launch preparation for the matched-retention brief. **Static specification
only: nothing is implemented, no numerical work was run, and no launch is
requested.** All completed studies, artifacts and verdicts are preserved
unchanged; the matched-retention comparison of the two existing Momentum
corrections remains a deferred question.

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
  `y_next = R_t`, so (U)–(W) are exactly the native update and `y_prev`,
  `R_prev` are dormant. This is the fallback point, and it is inside the
  admissible domain (§2).
- **Two-tap operator:** `M = 0` and `gamma + T = h` give
  `y_next = R_t + (T/h)(R_t - R_prev)`, i.e. the learned operator with
  `kappa = T/h`, `gamma = h(1 - kappa)`, `T = kappa h`.
  **Therefore `kappa > 1` requires `gamma < 0`.** The completed run learned
  `kappa ≈ 1.87–2.14`, so that operator is **not** inside this
  nonnegative-`gamma` family. It is kept as a separate strong comparator,
  explicitly outside the containment claim. No alternative admissible mapping
  is claimed; if one is proposed later it needs its own proof.
- At `T = h` (so `kappa = 1`, `gamma = 0`, `M = 0`) the filter is
  `y_next = 2 R_t - R_prev`: literal TSS **and** the two-tap operator at
  `kappa = 1`. Only that point may be called both.

### 1.4 Initialization and streaming

- Episode start: `W = U = y = y_prev = R_prev = 0`. The first token then gives
  `y_next = (h^2 + hT) R_0 / A`, which at the TSS boundary is
  `(1 + h/T) R_0`, and at `T = h` is `2 R_0` — consistent with (TSS) read with
  `y = 0`, `R_prev = 0`.
- Streaming: all five carries `(W, U, y, y_prev, R_prev)` cross chunk
  boundaries; a chunked rollout must equal the unchunked one exactly. Carries
  reset only between episodes.
- Output convention: the `y_next` computed from token `t` drives token `t`'s
  Momentum update (causal, same-token). This is the convention already used by
  the existing laws; it must be stated whenever the TSS identity is quoted.
- Masking: the mask enters `R_t` only, before any difference; `R_prev` is the
  previous **masked** residual and is never re-masked. Full BPTT; no query
  label, category, age, future input or family identity enters the recurrence.

## 2. Coefficient domain, learnable coordinates and optimizer policy

**Admissible set** (episode-fixed scalars, declared in advance):

    M >= 0,  gamma >= 0,  T >= 0,
    A = M + h(gamma + T) > 0,
    4M + 2h(gamma + T) >= h^2 (1 + delta_filter).                        (D)

The last line is the strict stability margin of the isolated driven filter
(§4), with a declared relative margin `delta_filter` (proposed 1e-3, the value
already used for the existing projection; its rounding budget must be derived
before use, as was done for kappa). At the TSS boundary (D) reduces to
`T >= (h/2)(1 + delta_filter)`.

**Learnable coordinates.** `M`, `gamma`, `T` are stored **directly** as three
real scalars, never through `exp` or `softplus`: a positive log-
parameterization cannot represent `M = gamma = 0` as an executed point, which
is exactly the boundary this study exists to test. The policy mirrors the
existing `kappa`: initialized at the declared point, no forward clipping, no
branch on their values, and a **post-update projection** onto (D) that leaves
the optimizer state untouched, counts events and records the effective margin.

**Projection rule** (explicit, deterministic): clamp `M <- max(M, 0)`,
`gamma <- max(gamma, 0)`, `T <- max(T, 0)`; then, if (D) is violated, raise `T`
minimally to `T <- max(T, [h^2(1 + delta_filter) - 4M] / (2h) - gamma)`.
Raising `T` is chosen because it is the coordinate whose increase always
restores (D) and never leaves the admissible set.

**Admissible departure directions from the TSS boundary.** From
`M = gamma = 0` the admissible departures are `M` increasing, `gamma`
increasing, and `T` moving subject to (D). The direction `gamma < 0` — which
is what the learned two-tap operator with `kappa > 1` needs — is **not**
admissible. Departures are one-sided at the boundary, so a negative proposal
is clamped and recorded as a projection event, not as a learned value.

**Physical interpretation and the additional assumptions.** `M`, `gamma`, `T`
keep their mechanical meaning (mass, damping, prospective horizon) for the
processing state. What is **additional** to the derivation, and must be
labelled as such: (i) coefficients are episode-fixed scalars, piecewise frozen
per token, so coefficient derivatives never appear; (ii) the placement —
filtering the residual and feeding the unchanged Momentum update — is a
modelling choice, not a consequence of the master law; (iii) the broad
equation, its `M = gamma = 0` boundary and the passive two-compartment circuit
sector are three different things, and attaining the boundary is **not** a
claim that a strict positive-component circuit can realize it.

## 3. Carry and parameter costs

| Rule | Carry matrices | Carry reals | Stored params |
|---|---|---:|---:|
| Native Momentum | W, U | 128 | 569 |
| Existing generalized correction | W, Q | 128 | 570 |
| Learned two-tap operator | W, U, R_prev | 192 | 570 |
| **This placement** | **W, U, y, y_prev, R_prev** | **320** | **572** |
| This placement with `M` fixed at 0 | W, U, y, R_prev | 256 | 571 |

**The old 128-value carry claim does not apply to this realization.** The
`y_prev` matrix is required only by the `M` term; a boundary-only variant that
fixes `M = 0` drops it, and that variant must then be labelled as a restricted
family that cannot depart in the `M` direction. Per-token compute adds a few
element-wise operations on 64-value matrices and no new matrix product;
preflight measures the actual step cost rather than assuming it.

## 4. Numerical risk and the checks that are actually valid

**Isolated filter.** With episode-fixed coefficients the homogeneous part of
(Y) is

    y_next = [1 + M/A - h^2/A] y - (M/A) y_prev,
    characteristic polynomial  z^2 - [1 + M/A - h^2/A] z + M/A.

Its strict Jury conditions are

    p(1)  = h^2/A > 0            <=>  A > 0,
    p(-1) = 2 + 2M/A - h^2/A > 0 <=>  4M + 2h(gamma + T) > h^2,
    |det| = M/A < 1              <=>  gamma + T > 0,

which is the condition quoted in the priority note, and at the TSS boundary
requires `T > h/2`. This concerns the **isolated driven processing filter
only**.

**What must not be claimed.** The existing `kappa` bound of the Momentum
correction is a different filter and **must not be transferred**. This
polynomial says nothing about the closed-loop associative memory (where `R`
depends on `W`), nothing about switching across tokens with varying gates, and
nothing about float32 rounding. No inherited stability certificate is used.

**Checks to specify before coding** (all numerical work on the cluster):

1. boundary identity: `M = gamma = 0` matches an independently coded literal
   TSS processing recursion for arbitrary token sequences, same clock,
   initialization, output convention and downstream gates;
2. native point: `M = 0, T = 0, gamma = h` matches native Momentum exactly,
   including dormant `y_prev`/`R_prev`, with matched incoming carries;
3. two-tap correspondence for `kappa <= 1` (`M = 0`, `gamma = h(1-kappa)`,
   `T = kappa h`), and an explicit fixture showing `kappa > 1` maps to
   `gamma < 0` and is refused by the projection;
4. streaming: five-carry chunk-boundary equality, and episode-start values;
5. gradients: finite, non-vanishing `dL/dM`, `dL/dgamma`, `dL/dT` through full
   BPTT at the boundary and at an interior point, against an independently
   coded float64 sensitivity, with the declared finite-first and
   resolvability policy;
6. projection: boundary clamping, the (D) restoration formula, idempotence,
   event counting, and updated-parameter provenance;
7. closed-loop diagnostics: per-run reporting of the executed coefficients, the
   filter conditions and W/U/y norms — reported as diagnostics, never as a
   stability certificate;
8. production float32 probe in an x64-disabled process, reusing the existing
   dtype, finiteness and tolerance policy.

## 5. Minimal comparison protocol (proposed, for review)

Reuses without change: the task and input contract, the read-only source
restore and checksum verifier from replication `20260917-011842` (source 500
development, 501–503 final), paired batches per seed, the unweighted
query-mean cross-entropy, the optimizer, full BPTT, evaluation, supervisor,
terminal verdict, finalizer and digest.

**Arms** (all full continuation of the pretrained backbone; no coefficient-only
arms, no blend, no router):

| Arm | Coefficients | Trains | Role |
|---|---|---|---|
| `tss_processing` | `M = gamma = 0` fixed; `T` trainable | backbone + T | literal TSS reference at every point |
| `generalized_processing` | `M, gamma, T` trainable, started at `M = gamma = 0`, `T = T0` | backbone + 3 | the actual extension; **starts function-matched to `tss_processing`** |
| `native_full` | — | backbone | active baseline |
| `operator_full` | `kappa` trainable from 0 | backbone + kappa | strong learned two-tap comparator, outside containment |
| `native_frozen` | — | nothing | descriptive anchor only |

`T0` is declared in advance, not searched: **`T0 = h = 1`**, which is inside
(D) and is simultaneously literal TSS and the two-tap point `kappa = 1`. The
update-zero equality of `generalized_processing` and `tss_processing` is
verified numerically per source seed before training.

**Work:** two learning rates per family (0.003, 0.01), 200 updates,
development checkpoints saved at updates 0, 25, 50, 100, 200 — the same
opportunities for every family. 4 trained families × 2 rates = 8 development
trajectories (1,600 updates); 4 × 3 seeds = 12 final trajectories, each
stopping at its selected update (≤ 2,400 updates); ≤ **4,000 updates**, plus
4 frozen-source evaluations. Streams are fresh, and their full ranges are
declared and asserted disjoint from every previous study and fixture before
execution. One 600-second cap covers checks, preflight, training, checkpoint
persistence, evaluation and finalization; preflight measures the actual paths
and refuses as INCOMPLETE if they do not fit. No retries, no trimming.

## 6. Selection and assessment (equal opportunities, frozen in advance)

1. **Native first.** Over its development checkpoints (both rates × the five
   update counts, the identical update-zero checkpoint deduplicated), select by
   revision macro accuracy, then lower revision cross-entropy, then fewer
   updates, then lower learning rate. Record its development retention
   `R_native` and recall `C_native`.
2. **Each extension separately** (`tss_processing`, `generalized_processing`,
   `operator_full`): eligible checkpoints are those with **retention ≥
   R_native AND recall ≥ C_native** — no minus-one-point allowance. Among
   eligible ones, the same ordering: revision, then cross-entropy, then fewer
   updates, then lower learning rate.
3. **Explicit fallbacks, never counted as improvements.**
   - Every extension has a **native fallback**: the selected native checkpoint
     mapped into that family at its native point (`M = 0, T = 0, gamma = h`
     for the processing arms; `kappa = 0` for the operator), with correct
     episode-start carries and verified value equivalence.
   - `generalized_processing` additionally has a **TSS fallback**: the selected
     `tss_processing` checkpoint, which is the same function at
     `M = gamma = 0`.
   - If no eligible checkpoint strictly improves development revision over the
     applicable fallback, the fallback is selected and **flagged**; a
     fallback is reported as a fallback, never as a learned correction.
4. Family, learning rate, update count and fallback flags are frozen before
   any final run; final trajectories train from each seed's own source with
   the selected recipe, or, for a fallback, use the corresponding final native
   (or TSS) checkpoint mapped without further optimization. Nothing is
   selected on final validation or held-out data; a deep-copied selection is
   re-checked after evaluation.
5. **Held-out**: one fresh common set, generated only after all final models
   and selections are frozen; only the selected endpoints are evaluated.
   Development checkpoint curves may be plotted; held-out checkpoints are
   never searched.

**Verdicts, each reported separately**, with paired per-seed differences and
all four query categories by full name:

- `generalized_processing` − `tss_processing`: does departing from the exact
  TSS boundary help, starting from the same function?
- `generalized_processing` − `native_full`, and `tss_processing` −
  `native_full`: matched-retention screens against the active baseline.
- `generalized_processing` − `operator_full`: reported separately and
  explicitly **outside** the containment claim (the operator's learned
  `kappa > 1` is not in this family).

A promising matched-retention signal requires mean held-out revision ≥ +1 pp,
all three paired revision differences positive, and mean retention **and**
recall differences ≥ 0. Per-seed retention or recall decreases are reported
even when the means satisfy it. This is a finite-sample screen, not
statistical noninferiority and not a no-loss guarantee. A trade-off (one wins
revision, the other retention) is reported as a trade-off. If a selection
collapses to a fallback, that is stated plainly.

## 7. Scope limits carried forward

- No Gated DeltaNet arm: no joint literature win and no SOTA claim follows.
- The learned two-tap operator is **not** arbitrary-timescale TSS Eq. (17),
  and this placement's `kappa > 1` region is not admissible.
- The broad equation, its `M = gamma = 0` boundary and the passive
  two-compartment sector are distinct; attaining the boundary is not a claim
  that a positive-component circuit realizes it.
- Frozen-token and isolated-filter spectral conditions do not establish
  closed-loop or switching stability.
- Three seeds on this small associative task are not significance, a benchmark
  or SOTA, and this specification promises no optimization or held-out gain.

## 8. What this document does not do

It implements nothing, launches nothing and runs no numerical work. Coding
begins only after the equation and comparator contract above is cleared, and
then reuses the existing task, source restore, evaluation and orchestration
code.
