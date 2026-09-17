# Frozen protocol: exact TSS containment on the same Momentum backbone

17 September 2026. **Frozen for static review before any execution. NOT
AUTHORIZED TO RUN until the implementation review clears it.** No numerical
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

Exact points: `M = gamma = 0` is literal TSS Eq. (17) driven by `R`;
`(M, gamma, T) = (0, h, 0)` is native Momentum; `M = 0, gamma + T = h` is the
two-tap operator with `kappa = T/h <= 1`.

## 3. Arms

| Arm | Law | Stored constants | Trains | Stored / trainable params | Carry executed (minimal) |
|---|---|---|---|---:|---:|
| `tss_processing` | filtered | `M = gamma = 0` | backbone + `T` | 572 / 570 | 320 (256) |
| `generalized_processing` | filtered | - | backbone + `M, gamma, T` | 572 / 572 | 320 |
| `native_full` | native Momentum | - | backbone | 569 / 569 | 128 |
| `operator_full` | two-tap operator | - | backbone + `kappa` | 570 / 570 | 192 |
| `native_frozen` | native Momentum | all | nothing (anchor) | 569 / 0 | 128 |

Both processing arms start at `M = gamma = 0, T = T0 = h`, and are checked
bitwise identical at update 0 per source. That start is **not** native: the
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
- **Executed filter gate** (clearance s1): after every update the production
  step is applied to basis carries with no residual inside the compiled step,
  yielding the executed `[[c1, -c0], [1, 0]]`; a non-finite entry or any Jury
  slack `<= 0` refuses the update and fails the run. At every validation
  point the same executed matrix is classified exactly over the rationals and
  must be `stable`. Isolated filter only; the Momentum block below it gets the
  native arm's own frozen-token diagnostic (unstable or non-finite fails).
- **Recovery comparisons** are finite-first and tolerance-based:
  per category at most one query of accuracy difference and relative
  cross-entropy <= 1e-3 for the native-point recovery on every source; the
  stricter completed-study identity check is recorded as a diagnostic only.
  Bitwise equality is asserted only for stored constants and genuinely shared
  executed operations.

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
  4 families x 3 sources = **12 final trajectories**, each stopping at its
  selected update; **at most 4,000 updates** in total; 4 frozen-source
  evaluations. No other trajectories exist: diagnostic and fallback endpoints
  reuse these slots.
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
4. **Deployment plan**, separate: each extension's fallbacks are native at
   its exact native point, plus - for `generalized_processing` only - a
   *genuinely selected, feasible* literal-TSS checkpoint. A trained endpoint
   is deployed only if feasible and **strictly** higher in development revision
   than the best available fallback. With no feasible TSS checkpoint there is
   no TSS fallback, and a native choice is never labelled TSS.
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

Operational PASS/INCOMPLETE/FAILED is reported apart from all performance
verdicts.

## 8. Focused checks (cluster, inside the cap)

Float64 module - exact points: TSS boundary vs an independently coded
literal TSS recursion on the same closed-loop residuals (T = 0.6, 1, 4); native
point vs the native step; two-tap mapping for kappa in {0, 0.5, 1}; kappa in
{1.87, 2.14, 2.61} maps to gamma < 0, is refused by the gate and changed by the
repair; T = h is both; episode-start values. Rollout: five-carry streaming,
counts, the TSS start differs from native while the native point matches it.
Gate: executed transition equals the declared polynomial and its
classification matches the strict conditions; the `M = 1, gamma = T = 0`
counterexample is refused and repaired; repair clamping, gaps, idempotence,
fixed exact points, frozen leaves; a mass that rounds `c0` to one is refused
although the gaps hold; the in-loop guard reads the same executed values as
the report; validation refuses an unstable arm and a moved stored constant.
Gradients: `M`, `gamma` inward and `T` at the boundary and all three at an
interior point against an independent sequential float64 sensitivity
(SENS64 = 1e-6), with a degenerate fixture reported as a fixture defect,
inward forward differences only and a resolvability classification; finite
backbone gradients. Training step: stored constants bitwise fixed,
trainable coefficients and backbone move finitely, mask routing, gate accepted;
NaN rejection. Wiring: stream disjointness, work <= 4,000, selection order,
feasibility and diagnostic flag, update-0 deduplication, non-finite refusal,
deployment fallbacks (no TSS fallback when TSS is infeasible, strict
improvement, no relabelling), screen labels and criterion.

Float32 probe (x64 off, own process): dtypes and finiteness of the five-carry
rollout and streaming (TRAJ32 = 2e-5); native-point recovery and the
T = h / kappa = 1 correspondence at trajectory tolerance; executed gate
verdicts including two large masses refused while the gaps hold; repair in
float32; float32 JVPs against the independent float64 sensitivity
(REF32 = 2e-2) with resolvability; the masked optimizer path of both processing
arms with finite-first decisions; NaN refusal.

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
