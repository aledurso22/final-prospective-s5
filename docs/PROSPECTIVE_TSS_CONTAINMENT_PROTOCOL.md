# Frozen protocol: exact TSS containment on the same Momentum backbone

17 September 2026. **Frozen for static review before any execution. NOT
AUTHORIZED TO RUN until the corrected implementation is cleared.** Corrected
for the static review of 7613c86
(`PROSPECTIVE_TSS_IMPLEMENTATION_REVIEW_7613c86_2026_09_17.md`); dispositions
in s11. No numerical
work has been run locally or on the cluster.

- Specification: `docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md` (d95266d, with the
  clearance's three clarifications folded in; its §9 records them).
- Clearance: `PROSPECTIVE_TSS_IMPLEMENTATION_CLEARANCE_d95266d_2026_09_17.md`.
- Code: `experiments/prospective_momentum/filtered.py` (the law, gate, repair,
  references), `tss_containment.py` (the study), `tss_containment_summary.py`
  (digest), and an additive dispatch in `model.py`.
- Checks: `tests/test_prospective_tss_containment.py` (float64) and
  `tests/prospective_tss_containment_float32_probe.py` (production float32,
  x64 off).
- Launcher: `bin/run_experiments/cluster_prospective_tss_containment.sh`.

All completed studies, artifacts and verdicts are unchanged. The
matched-retention comparison of the two *existing* Momentum corrections
remains deferred.

## 1. Question

Starting from the **same function** - literal TSS residual processing at
`T0 = h` on a pretrained Momentum backbone - does letting the master law depart
from the exact boundary `M = gamma = 0` improve revision on fresh held-out
episodes at matched retention and recall? Native Momentum and the learned
two-tap operator are kept as separate references; the operator's learned
`kappa > 1` is outside this nonnegative-`gamma` family, so it is **not** part
of the containment claim.

## 2. The executed law (unchanged from the specification)

    R_t     = m_t (alpha_t W_prev k_t - v_t) k_t^T
    A (y_next - y) = M (y - y_prev) + h^2 (R_t - y) + h T (R_t - R_prev)
    A       = M + h (gamma + T),   h = 1
    U_next  = mu_t U_prev + eta_t y_next
    W_next  = alpha_t W_prev - beta_t U_next

Old states on every right-hand side; carries `(W, U, y, y_prev, R_prev)` zero
at episode start and passed across chunk boundaries; mask inside `R` only;
same-token output convention; full BPTT; pinned native gates.

**Executed form** (review R2; the same equation, not a new model):

    a = [2M + h(gamma+T) - h^2]/A,  b = M/A,  c = [h^2 + hT]/A,  d = hT/A
    y_next = a y - b y_prev + c R_t - d R_prev

The coefficients are formed once per compiled program from the stored
leaves, with no parameter-dependent branch, and are returned by that program.
Derivatives with respect to `M, gamma, T` flow through them everywhere. The
original form survives only in separately coded references.

Exact points: `M = gamma = 0` is literal TSS Eq. (17) driven by `R`
(`a = 1 - h/T, b = 0, c = 1 + h/T, d = 1`); `(M, gamma, T) = (0, h, 0)` is
native Momentum (`a = b = d = 0, c = 1`, exact in IEEE arithmetic, so the
`y + (R - y)` cancellation is gone); `M = 0, gamma + T = h` is the two-tap
operator with `kappa = T/h <= 1`.

## 3. Arms

| Arm | Law | Stored constants | Trains | Stored / trainable params | Implemented carry |
|---|---|---|---|---:|---:|
| `tss_processing` | filtered | `M = gamma = 0` | backbone + `T` | 572 / 570 | 320 |
| `generalized_processing` | filtered | - | backbone + `M, gamma, T` | 572 / 572 | 320 |
| `native_full` | native Momentum | - | backbone | 569 / 569 | 128 |
| `operator_full` | two-tap operator | - | backbone + `kappa` | 570 / 570 | 192 |
| `native_frozen` | native Momentum | all | nothing (anchor) | 569 / 0 | 128 |

The implemented carry of both processing arms is 320 reals; 256 is only the
theoretical minimum of a law with `M` fixed at zero, not the implemented cost.

Both processing arms start at `M = gamma = 0, T = T0 = h`; their start trees
are checked bitwise identical per source (a storage-identity check under the
same law - it does not by itself test literal TSS; the independent literal-TSS
reference comparison is in the focused checks). That start is **not** native: the
native point `(0, h, 0)` is checked against the native rule separately.
Stored constants are removed from the optimizer input (gradient zeroed before
the transformation, update masked again), restored bitwise by the repair, and
asserted unchanged after every run; the TSS arm's validation also refuses any
tree with `M` or `gamma` different from zero.

## 4. Numerical policy

- **Feasibility repair** after every update (not a projection): clamp
  `M, gamma, T >= 0`; `T <- max(T, g_min - gamma)`;
  `T <- max(T, (h^2 (1 + delta) - 4M)/(2h) - gamma)`, with `g_min = 2^-10 h`,
  `delta = 1e-3`. Declared robustness gaps, not a certificate; repairs are
  counted and logged with the pre-repair proposals.
- **Executed-coefficient gate** (clearance s1, review R2): every executed
  `M, gamma, T, A, a, b, c, d` finite and `1 - a + b > 0`, `1 + a + b > 0`,
  `1 - b > 0` for the rounded coefficients.
  - *Every update*: the coefficients the update's own forward pass executed
    (returned by that compiled program) are gated in the executed dtype; a
    failure refuses the update and fails the run. The repaired tree's
    coefficients are executed, and gated, by the next forward pass or by the
    endpoint's checkpoint evaluation.
  - *Every checkpoint*: each distinct coefficient set the evaluation program
    executed is classified exactly over the rationals and must be `stable`.
  - Scope: a result about those rounded coefficients of the isolated filter;
    not a theorem about every floating-point trajectory, the closed-loop
    memory, switching, or the Momentum block (which keeps the native arm's own
    frozen-token diagnostic: unstable or non-finite fails).
- **Checkpoint acceptance and persistence** (review R3): at every selectable
  checkpoint (updates 0, 25, 50, 100, 200 in development; every checkpoint up
  to and including each final endpoint) the runner requires finite parameters
  and optimizer state, finite metrics and state norms, a finite
  processing-state diagnostic (max |entry| of `y, y_prev, R_prev` over every
  evaluated token), executed-coefficient acceptance, unchanged stored
  constants and the arm's domain; then saves the parameter and optimizer trees
  and writes the checkpoint record to `status.json` before the next update. An
  invalid checkpoint fails the run immediately, so it can neither enter
  selection nor be hidden by a valid endpoint; selection additionally refuses
  any checkpoint not recorded as accepted. Every measured step is checked -
  non-finite scalars, a failed gate or non-finite coefficient telemetry fail
  at that step.
- **Zero-update endpoints** (review R1): a selected update of 0 gives a valid
  zero-update NAMED-FAMILY endpoint (initialized optimizer state, full
  checkpoint acceptance, no step scalars required), distinct from the
  frozen-source anchor in records and counts.
- **Recovery** (review R2): DECISIVE = finite-first logits and `W, U` carries
  of the processing law at `(0, h, 0)` against the native rule on
  representative episodes (the first two of each task family) of every
  restored source, at the existing trajectory tolerance TRAJ32 = 2e-5.
  Aggregate differences are RECORDED ONLY, from integer correct counts with
  equal denominators and a documented consistency check (review R4), against
  the previously declared identity tolerance (zero count difference,
  cross-entropy relative 2e-5); malformed records fail, magnitudes never
  decide. The proposed 1e-3 tolerance is withdrawn.
- Bitwise equality is asserted only for stored constants, genuinely shared
  executed operations, and coefficients whose IEEE arithmetic is exact at the
  named points.

## 5. Sources, streams and work

- Sources: `/Users/durso/s5-runs/prospective-momentum-replication/20260917-011842`,
  read-only and sha256 verified before, during restore and after; seed 500
  development, 501-503 final.
- Fresh streams, asserted disjoint from every range consumed by all previous
  studies (including the same-backbone study) before execution:

| Stream | Identifier(s) |
|---|---|
| continuation training | 425,000,000 - 425,030,199 (`420,000,000 + seed x 10,000 + update`, seeds 500-503) |
| development validation | 450,000,000 |
| final-run validation | 451,000,000 |
| held-out (opened after every final run and selection is frozen) | 460,000,000 |

- Work: two learning rates (A 0.003, B 0.01) x 4 trained families = **8
  development trajectories** of 200 updates, checkpoints at updates
  0, 25, 50, 100, 200 (9 distinct per family after deduplicating update 0);
  4 families x 3 sources = **12 named-family final endpoints**, each stopping
  at its selected update, which may be 0; **at most 4,000 updates** in total;
  4 frozen-source anchor evaluations. No other trajectories exist: diagnostic
  and fallback endpoints reuse these slots. Checkpoint validation and
  persistence are part of this work and are timed in preflight (one full
  checkpoint per arm, projected at every checkpoint of every run).
- One 600-second cap covers GPU startup, the float64 checks, the float32
  probe, start-point checks, measured preflight (40 s host allowance recorded
  as an allowance), training, checkpoint persistence, evaluations, source
  verification, digest and terminal verdict. Preflight refuses as INCOMPLETE
  if the projected work does not fit and FAILS on invalid numerical state. No
  retries, no trimming, no larger cap, no loosened tolerance.

## 6. Selection (development only, frozen before finals)

1. **Native** over its 9 checkpoints: highest revision macro accuracy, then
   lower revision cross-entropy, then fewer updates, then lower learning rate.
   Its development retention and recall are `R_native`, `C_native`.
2. **Each extension** (`tss_processing`, `generalized_processing`,
   `operator_full`): feasible checkpoints satisfy retention >= `R_native` AND
   recall >= `C_native`, no allowance; best feasible by the same ordering. If
   none is feasible, the best unconstrained checkpoint is the final endpoint,
   flagged **diagnostic** and constraint-failing (clearance s3).
3. A non-finite checkpoint metric fails selection (and the run).
4. **Deployment plan**, separate: `generalized_processing` may fall back to
   the native model placed at its exact native point `(0, h, 0)` or to a
   *genuinely selected, feasible* literal-TSS checkpoint; `operator_full` to
   the native model at `kappa = 0`; `tss_processing` only to the native model
   as an **external** deployment selection, which is not a point of the
   literal-TSS family. Fallbacks are ranked by the frozen full ordering
   (primary, cross-entropy, updates, learning rate), then the declared tie
   rule native before TSS. A named-family endpoint is deployed only if
   feasible and **strictly** higher in development revision than the best
   available fallback. With no feasible TSS checkpoint there is no TSS
   fallback, and a native model is never labelled TSS. Every choice records
   the executed family, the checkpoint identity (arm, configuration, learning
   rate, update, saved development tree) and the parameter map.
5. Selection and deployment plan are deep-copied, written to
   `selection.json` and re-checked after the finals and after evaluation.
   Nothing is selected on final validation or held-out data.

## 7. Verdicts (each reported separately; declared before execution)

Held-out, with paired per-seed primary, retention and recall differences and
all four query categories by full name:

| Comparison | Kind |
|---|---|
| `generalized_processing` - `tss_processing` | scientific: departure from the exact boundary |
| `generalized_processing` - `native_full` | scientific |
| `tss_processing` - `native_full` | scientific |
| `generalized_processing` - `operator_full` | outside the containment claim |
| `operator_full` - `native_full` | outside the containment claim |
| `generalized_processing` - `native_frozen` | descriptive anchor |

**Promising matched-retention signal**: mean revision difference >= +1 pp, all
three paired revision differences positive, mean retention AND mean recall
differences >= 0, and neither endpoint flagged diagnostic. Where an endpoint is
diagnostic, the constrained screen is reported **infeasible/unavailable** and
the trained-endpoint comparison is still shown with its constraint-failing
label. The historical -1 pp safeguard is printed beside it, never as this
study's criterion. Per-seed retention or recall decreases are reported even
when the means meet the condition.

The **deployment outcome** (held-out mean of the frozen deployment choice,
taken from the already trained final runs) is reported in its own table,
`counted_as_improvement = False` for every fallback.

All comparisons are between named-family final endpoints; each records its
endpoint kind (trained or zero-update) so a zero-update endpoint is never
described as trained. Work counts separate development runs, named-family
final endpoints (with the zero-update ones counted), frozen-source anchor
evaluations and validated checkpoints.

Operational PASS/INCOMPLETE/FAILED is reported apart from all performance
verdicts.

## 8. Focused checks (cluster, inside the cap)

Float64 module (`tests/test_prospective_tss_containment.py`):
- executed form vs the separately coded original form on closed-loop
  residuals, and the four coefficients vs their formulas; literal TSS boundary
  vs the independent literal-TSS recursion (T = 0.6, 1, 4); exact coefficients
  at the native and T = h points; native point vs the native step; two-tap
  mapping for kappa in {0, 0.5, 1}; kappa in {1.87, 2.14, 2.61} refused and
  changed by the repair; episode-start values;
- five-carry streaming, returned coefficients, implemented counts; other rules'
  rollout outputs unchanged; the study loss equals the shared loss; the TSS
  start differs from native while the native point matches it;
- integer count differences: zero, exactly one count (the review's 0.5 vs
  0.51 case), two counts, non-finite, mismatched denominators, non-integer
  counts;
- gate: classification equals the strict conditions; the missing-damping
  counterexample; rounded coefficients refused although the gaps hold; any
  non-finite executed quantity refused by report and guard; guard and report
  agree on a compiled rollout's returned values; repair clamping, gaps,
  idempotence, fixed exact points, frozen leaves; validation requires executed
  records and the TSS boundary;
- gradients: M, gamma inward and T at the TSS boundary, M inward at the
  native point, and all three at an interior point, against the independent
  original-law sensitivity (SENS64) with its primal carries and loss checked,
  finite-first decisions (injected-NaN regressions), inward forward
  differences only; T inward at the native point is the one DECLARED analytic
  zero (s12), with its own regressions;
- runner: masked step freezes constants and gates the forward pass (an
  unstable tree is refused); step failures at any step; zero-update endpoints
  of all four named families are valid, persisted and not anchors, and a
  non-finite zero-update state fails; every checkpoint is validated, persisted
  and logged, and a saved update-1 tree reproduces its recorded metrics; an
  invalid intermediate checkpoint fails the run and cannot be hidden;
- wiring: streams, work, selection order and flags, an unaccepted
  intermediate checkpoint blocks selection, update-0 deduplication, non-finite
  refusal, deployment ordering by the full key and declared tie rule, strict
  improvement, identity/map records, no relabelling, screen labels with
  endpoint kinds; the decisive direct recovery passes on a restored source and
  fails on non-finite state.

Float32 probe (x64 off, own launcher stage): dtypes and finiteness of the
five-carry rollout and streaming (TRAJ32); native-point recovery and the
T = h / kappa = 1 correspondence (TRAJ32); gate verdicts on coefficients the
compiled rollout returned, including two masses refused although the gaps
hold, and exact native-point coefficients; repair; derivatives against the
independent float64 original-law reference with primal loss and carries at
TRAJ32 and derivatives at REF32 (the declared analytic zero of s12 against
dtype-specific absolute floors), finite-first with injected-NaN and
declared-zero regressions;
the masked optimizer path of both processing arms followed by full checkpoint
acceptance; NaN refusal.

## 9. Launch and digest (for the reviewed launch only)

    bash bin/run_experiments/cluster_prospective_tss_containment.sh

    python -m experiments.prospective_momentum.tss_containment_summary \
        <run_dir>

## 10. Scope limits

Literal TSS containment by this placement is established; containment of the
learned `kappa > 1` operator is not. The coefficient domain is broader than the
passive-compartment domain. The executed gate certifies the isolated filter,
not closed-loop or switching stability. No Gated DeltaNet arm: no joint
literature win. Reused sources: an intervention study, not another
independent-source replication. Three seeds on this small task are not
significance, a benchmark or SOTA, and nothing here promises an optimization
or held-out gain.

## 11. Dispositions for the review of 7613c86

| Item | Disposition |
|---|---|
| R1 update-zero endpoint fails | **Fixed.** `run_one` treats `updates = 0` as a zero-update named-family endpoint: initialized optimizer state, full checkpoint acceptance, step scalars required only when a step occurred; non-finite zero-update state still fails. Records carry `endpoint_kind`; work counts and screen wording no longer call every endpoint trained. Runner fixtures cover all four named families and a non-finite zero-update state. |
| R2 basis responses are not coefficients | **Fixed.** Executed coefficient form `y_next = a y - b y_prev + c R - d R_prev`, formed once per compiled program with no boundary branch and returned by the program; the gate classifies exactly those rounded values (in-loop for the update's own forward pass, exact rationals at checkpoints) and rejects any non-finite executed quantity. Basis-response extraction removed. Native point is now exact (`a = b = d = 0, c = 1`). Original form kept only in separately coded references. |
| R2 recovery tolerance 1e-3 | **Withdrawn.** Decisive recovery is direct finite-first logits and W, U carries at TRAJ32 on representative episodes of every restored source; aggregate count/CE differences are recorded only, against the previously declared identity tolerance. |
| R3 checkpoints neither validated nor saved | **Fixed.** Every checkpoint is fully accepted (parameters, optimizer, metrics, processing state, executed coefficients, stored constants, domain), persisted and logged before the next update; an invalid one fails the run; selection refuses unaccepted checkpoints and records the saved tree. Every step and every measured preflight step is checked. Preflight times a full checkpoint. Runner fixtures: reproduction of a saved update-1 tree's metrics; an invalid intermediate checkpoint cannot be hidden; an unaccepted intermediate checkpoint blocks selection. |
| R4 count rounding in the fixture | **Fixed.** Integer counts reconstructed with a consistency check and equal denominators; fixtures for zero, one and two counts, non-finite, mismatched and non-integer inputs; aligned to the recorded-only policy. No model tolerance loosened. |
| R5 NaN evidence in the float32 sensitivity check | **Fixed.** A single finite-first decision over production loss/JVP, reference loss/derivative/carries, perturbed loss, FD and errors; independent primal loss and W, U, y carries checked at TRAJ32; injected-NaN regressions for reference derivative, reference primal loss and carries, and perturbed loss. The float64 module applies the same finite-first rule to its diagnostic FDs. |
| Fallback ranking by primary alone | **Fixed.** Frozen full ordering, then a declared tie rule (native before TSS). |
| TSS native-fallback metadata | **Fixed.** Labelled an external deployment selection, not a point of the literal-TSS family. |
| Deployment identity | **Fixed.** Executed family, checkpoint identity (arm, configuration, learning rate, update, saved tree) and parameter map persisted per choice and in the deployment outcome. |
| Start-check scope | **Fixed.** Each start check is labelled with its actual scope; the storage-identity check is not claimed to test literal TSS; the independent literal-TSS comparisons stay in the focused checks. |
| Count reporting and carry | **Fixed.** Zero-update endpoints counted as named-family endpoints, not anchors; 320 reported as the implemented carry of both processing arms and 256 as a theoretical minimum only. |

Unchanged: the equation (the coefficient form is algebraically equivalent),
arms, sources, streams, loss, learning-rate slots, checkpoint opportunities,
search budget, selection criterion, performance criteria and the 600-second
cap. The shared `study.evaluate` gains two optional arguments whose defaults
leave every completed study's evaluation unchanged.

## 12. Failed dispatch 20260917-152713 and test-method amendment

**Dispatch (preserved).** Commit `e38b61a`, run stamp `20260917-152713`, logs
`/Users/durso/s5-runs/prospective-tss-containment/logs/20260917-152713`.
Operational verdict **FAILED (exit 4)** at the focused float64 checks after
76 s of 600: 62 passed, 1 failed. The float32 probe, start-point checks,
preflight, training and held-out evaluation never started; no status file or
digest exists. Source re-verification: all six files unchanged. No retry.

**Failure.** `test_coefficient_derivatives_match_the_independent_sensitivity
[T inward at the native point]`: production JVP `0.0`, independent reference
`-1.112707e-17`, forward difference `5.55e-11`. The fixture required a
nondegenerate reference (`|ref| > 1e-12`) and relative agreement, and so
failed as a declared FIXTURE DEFECT. All other derivative fixtures agreed with
the reference to about 1e-15.

**Cause: an analytically degenerate fixture, not a model defect.** For
`M = 0, gamma = h` the executed coefficients are `a = d = T/(h+T)`, `b = 0`,
`c = 1`, i.e.

    y_next = R_t + T/(h+T) (y - R_prev).

With matched (zero) initialization `y = R_prev` holds on every token by
induction, so the entire trajectory is native for EVERY fixed `T` on this
line, and `dL/dT = 0` exactly for any data. That direction was added in the
correction of 7613c86 without checking it; neither processing arm starts or is
evaluated at the native point.

**Amendment (tests and documentation only; approved before this patch).**
- The fixture is marked `expected_zero` by declaration from the identity
  above, never classified from its measured magnitude.
- For it, the production derivative and the independent reference are checked
  SEPARATELY against the absolute floor of the dtype that computed each:
  `1e3 x eps x max(|loss|, 1)`, with eps32 for the float32 production JVP and
  eps64 for the float64 production JVP and for the reference. Finite-first
  checks, primal agreement (ID64 in float64, TRAJ32 in float32) and the finite
  forward-difference diagnostic are unchanged. A pass is reported as
  **"analytic zero verified within tolerance"**, not as unresolvable.
- Regressions in both precision paths reject a production or reference
  derivative above its floor and a non-finite value in either derivative; the
  float32 path also rejects a reference that is below the float32 floor but
  above the float64 floor.
- Every other derivative fixture, including every direction at the literal-TSS
  start, keeps its nondegeneracy requirement and relative tolerance unchanged.

Unchanged: model and study code, equation, arms, sources, streams, loss,
selection, performance criteria, all other tolerances and the 600-second cap.
The relaunch is a single normal launch after static confirmation.

