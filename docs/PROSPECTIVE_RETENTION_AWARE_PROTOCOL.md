# Frozen protocol: retention-aware continuation

17 September 2026. **Frozen for static review. NOT AUTHORIZED TO RUN.** No
numerical work has been run. This is a new study: the completed
TSS-containment and temporal-response results, and their failed screens, are
unchanged and are not reinterpreted.

**Question.** Can generalized processing achieve more revision than literal
TSS when training explicitly discourages damage to untouched associations
and recall? This tests whether the added response-shaping freedom helps meet
a preservation requirement. It does not assume that mass protects memory,
and no positive result is promised.

## 1. Unchanged

- **Model:** the Momentum backbone, residual-processing placement and
  coefficient form, full BPTT, the `gamma >= 0` domain and repair, and the
  executed-coefficient gate. There is no new architecture, gate or readout,
  and no negative damping.
- **Four trained families:** generalized processing, literal TSS
  (`M = gamma = 0` constants), native Momentum and the learned operator.
  Each starts from the same source mapping as in the temporal-response study.
  The frozen source is the anchor.
- **Task and measurement:** the temporal-response task, its equal-cell
  metrics, and validation that every checkpoint is accepted, persisted and
  logged before the next update.
- **Integrity and reporting:** read-only source verification (500
  development, 501–503 final), the supervisor, finalizer and terminal verdict.

## 2. The one change: the training objective

    L = L_existing
        + lambda * max(0, CE_untouched - CE_untouched_reference)
        + lambda * max(0, CE_recall    - CE_recall_reference)

- **`L_existing`:** the unchanged unweighted query cross-entropy.
- **Category weighting:** the evaluation's own. Each category CE is computed
  on the training batch, with `CE = 0` outside queries.
  - `CE_untouched` = mean(revision-family untouched-probe CE, the
    equal-weight mean over the 10 delay × condition cells; revision-family
    late-untouched CE, the mean over the 2 conditions).
  - `CE_recall` = the recall family's equal-weight macro CE (revised probes,
    untouched probes, late selected, late untouched).
- **Reference terms:** the same quantities from a **fixed, read-only native
  Momentum endpoint** (s3), on the **same training episodes**, with
  `stop_gradient` on its parameters and outputs.
- **Hinge:** `max(0, x)` is implemented with a zero derivative for `x <= 0`.
- **Metadata:** `kind`, `delay`, `condition` and `family` form the loss only.
  They are passed separately from the model inputs (`key_id`, `val_id`,
  `event`, `label`) and never enter the model.
- **Scope:** these CE penalties are training proxies. They do not guarantee
  accuracy preservation, which is measured.

**Slots.** `lambda` ∈ {0, 1} for every family, at the previously selected
learning rate 0.01 and the existing 200-update schedule. The two loss slots
replace the two learning-rate slots, so the grid is not multiplied.
Checkpoints stay at 0, 25, 50, 100 and 200 (9 distinct per family), with
three final source seeds.

**Per-step acceptance** (finite-first, at every step): the existing step
checks, plus finite `base`, both category CEs, both reference terms and both
penalties, and a non-empty smallest loss cell (4 queries at 8 episodes per
family).

## 3. Fixed native references (verified from saved artifacts)

**Declared reference run:**
`/Users/durso/s5-runs/prospective-temporal-response/20260917-163003`
(complete, PASS). Its frozen native selection is slot B (lr 0.01), update
200, and its final plan follows that selection.

| Source seed | Reference endpoint (in `params/`) | Recorded metrics compared on |
|---|---|---|
| 500 (development) | `dev_native_full_B_seed500_u200.msgpack` | that run's development validation (550,000,000) |
| 501 | `final_native_full_B_seed501_u200.msgpack` | that run's final validation (551,000,000) |
| 502 | `final_native_full_B_seed502_u200.msgpack` | same |
| 503 | `final_native_full_B_seed503_u200.msgpack` | same |

**Verification before any training.** Any failure refuses the run with no
substitution:
1. **Mapping from saved status only.** The mapping is re-derived solely from
   the run's saved `status.json`: the frozen native selection must equal the
   final selection and the declared recipe, and each final seed must have
   exactly one native row that follows it. Every recorded file path must equal
   the declared file.
2. **Restore and hash.** Each reference is restored against its source's
   native leaf set, shapes and dtype, and checked finite. `status.json` and
   the four files are sha256-hashed.
3. **Reproduction.** Each reference must reproduce its recorded revision,
   retention, recall, immediate and later metrics at TRAJ32 on the completed
   study's own validation streams. Those streams are used for this
   verification only.
4. **Integrity at finish.** The hashes are re-verified at finish by the
   finalizer and by the launcher's extra verification.

References are never chosen by score and never swapped between seeds.

## 4. Selection (development only, frozen before held-out)

1. **Continued-native reference first.** The native family's development
   checkpoint with the highest revision, then lower revision CE, then fewer
   updates, then lower `lambda`. Its retention and recall are `R_native` and
   `C_native`.
2. **For each extension:**
   - **constrained endpoint:** the same ordering over accepted checkpoints
     with retention ≥ `R_native` **and** recall ≥ `C_native`. If none
     qualifies, it is recorded **INFEASIBLE**;
   - **unconstrained endpoint** (descriptive): the same ordering over all
     accepted checkpoints;
   - **`lambda = 0` control** (descriptive): the same ordering within the
     `lambda = 0` slot.

   The shared update-0 checkpoint is counted once, in the `lambda = 0` slot.
   Any unaccepted or non-finite checkpoint fails selection.
3. **Deployment plan** (unchanged rule, reported separately): an infeasible
   family falls back to a feasible baseline. **A fallback is never an
   improvement.**
4. Endpoints, deployment plan and final plan are frozen before any final
   trajectory; held-out data is generated only after finals.

## 5. Verdicts

**Primary** (held-out, constrained endpoints; native uses its reference
endpoint):

| Comparison | Role |
|---|---|
| **generalized − literal TSS** | **central**: both received the same retention-aware opportunity |
| generalized − native Momentum | reported separately |
| generalized − learned operator | reported separately |

**Pass condition, unchanged:**
- mean revision difference ≥ +1 pp;
- positive paired revision differences in all three seeds;
- mean retention and recall differences ≥ 0.

If either endpoint is infeasible, the verdict is **unavailable** and the
report says so.

**Descriptive only** (these cannot rescue a primary verdict):
- the same three comparisons between unconstrained endpoints and between
  `lambda = 0` controls;
- each extension's constrained endpoint against its own `lambda = 0` control;
- TSS and the operator against the native reference;
- the generalized unconstrained endpoint against the frozen source;
- delay and condition cells for all of the above.

A generalized gain over its own `lambda = 0` control is insufficient: it must
also beat equally trained literal TSS.

## 6. Work, streams and cap

| Item | Count |
|---|---:|
| Development | 4 families × 2 slots = 8 trajectories (1,600 updates) |
| Finals | one trajectory per (family, slot, seed) needed by any endpoint, run to the largest needed update; at most 4 × 2 × 3 = 24 trajectories |
| **Maximum updates** | **6,400** |
| Reference forward | one per training step |
| Frozen-source evaluations | 4 |

Up to 24 final trajectories are needed because the constrained endpoint, the
unconstrained endpoint and the `lambda = 0` control may lie in different
slots.

**Streams** (fresh, asserted disjoint from every earlier range):

| Stream | Identifier(s) |
|---|---|
| training | 625,000,000 – 625,030,199 |
| development validation | 650,000,000 |
| final validation | 651,000,000 |
| held-out | 660,000,000 |

**Cap.** One 600-second cap covers, in order:
1. GPU check;
2. the existing checks (TSS-containment float64 and float32,
   temporal-response);
3. this study's targeted checks;
4. task, source and reference verification;
5. start points;
6. measured preflight;
7. training, held-out evaluation, source and reference integrity, digest and
   verdict.

**Measured preflight** (review of 73022f1, R1) exercises **both lambda slots
of all four families**: eight executables, on disposable production float32
state, before any training. For each (family, slot) it runs the actual host
step (reference forward, host synchronization and per-step acceptance at
every measured step) and records:
- its initial-call time;
- its steady step time;
- whether it retraced;
- one full checkpoint (evaluation, acceptance, persistence);
- the acceptance result, with every timing required finite and non-negative.

A failure or retrace in the `lambda = 0` slot is recorded under that slot and
cannot be hidden by a passing `lambda = 1` slot.

The projection uses those **slot-specific** measurements at the worst case:
- both slots in development and for every final seed, each with all five
  checkpoints (the 6,400-update maximum);
- held-out evaluation of every endpoint role at the slower slot's evaluation
  cost;
- the anchor;
- a recorded 40 s host allowance.

Both compilations are incurred inside preflight and are **not charged again**.
Preflight refuses as FAILED on any acceptance failure and as INCOMPLETE on a
retrace or if the projection does not fit. There are no retries and no
trimming.

**Budget estimate (not a measurement; preflight now also compiles the
`lambda = 0` executables before deciding).** The earlier runs measured law checks
at 76 s, the float32 probe at 42 s and temporal checks at 24 s. The targeted
checks and study are estimated from those runs' step and checkpoint times
(about 13–17 ms per step without the reference forward). Together they put
this run near 430 s of the 600 s cap. The retention-aware step is estimated
at about 20 ms. Whether it fits is decided by measured preflight, not by this
estimate.

## 7. Checks

**Existing suites, unchanged:** `test_prospective_tss_containment.py`,
`prospective_tss_containment_float32_probe.py`,
`test_prospective_temporal_response.py`.

**Targeted additions** (`tests/test_prospective_retention_aware.py`):
- the loss categories equal the evaluation's category CE weighting;
- the reference receives exactly zero gradient; the penalties are hinges; the
  candidate's gradient equals the gradient with reference terms held as
  constants (GRAD64);
- `lambda = 0` recovers the existing training steps (generalized and native)
  within ID64;
- finite-first rejection of non-finite or missing loss terms and empty cells,
  including a non-finite reference;
- constrained-selection wiring: native first, feasibility, INFEASIBLE,
  unconstrained and `lambda = 0` endpoints, update-0 sharing, unaccepted
  checkpoints, the final plan, and primary-verdict unavailability;
- the reference mapping from saved status: derivation, a missing seed, a
  swapped file, a changed frozen selection, a changed recipe, a failed run;
- preflight orchestration with stubbed numerics (R1): all eight (family,
  slot) combinations are exercised with seven checked steps each; a
  `lambda = 0` failure and a `lambda = 0` retrace surface under that slot and
  are not hidden by `lambda = 1`;
- digest pairing (D1): seed-keyed revision, retention and recall differences
  come from the saved full-precision values, with a missing pair reported.

## 8. Reuse and shared-code change

`temporal_response.run_one` gains two optional hooks (`step_fn`,
`step_failure_fn`). Their defaults leave that study's behaviour unchanged.

## 9. Launch (after static review only)

    bash bin/run_experiments/cluster_prospective_retention_aware.sh

## 10. Amendment record (review of 73022f1)

- **R1:** production preflight now measures and validates both lambda slots
  for every family, and projects remaining work from the slot-specific costs
  without re-charging completed compilations. A stubbed orchestration fixture
  shows that all eight combinations are exercised and that a `lambda = 0`
  failure or retrace cannot be hidden.
- **D1:** the digest prints seed-keyed revision, retention and recall
  differences for every comparison, computed from saved full-precision
  held-out metrics before formatting, with missing pairs reported explicitly.
  This is read-only reporting, not a new metric or criterion.

Unchanged: loss, equations, arms, reference files, slots, learning rate,
checkpoint schedule, streams, numerical tolerances, performance criteria, the
6,400-update maximum and the single 600-second cap.

