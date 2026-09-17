# Frozen protocol: temporal response of generalized prospective processing

17 September 2026. **Frozen for static review. NOT AUTHORIZED TO RUN.** No
numerical work has been run, locally or on the cluster. This is a new,
hypothesis-driven study, not a retry of the failed matched-retention screen.
The completed TSS-containment study (run `20260917-154035`, report
`docs/PROSPECTIVE_TSS_CONTAINMENT_REPORT.md`) and all of its results are
unchanged.

**Question.** Can generalized prospective processing keep the
immediate-revision benefit of literal TSS while improving later recall and
reducing interference with untouched associations? Whether the added freedom
helps is the open question; no positive result is promised.

## 1. What is reused unchanged

- **Placement and equations:** `R_t -> y -> native Momentum update -> W`, in
  the executed coefficient form
  `y_next = a y - b y_prev + c R_t - d R_prev` with
  `A = M + h(gamma+T)`, `a = [2M + h(gamma+T) - h^2]/A`, `b = M/A`,
  `c = [h^2 + hT]/A`, `d = hT/A`, and `h = 1`.
- **Training machinery:** full BPTT, the unweighted query cross-entropy, the
  executed-coefficient gate, and the feasibility repair (`gamma >= 0`). There
  is no negative `gamma`, no new gate, no router and no BPTT replacement.
- **Arms:** the same four trained families and the frozen anchor, with the
  same training steps (`tss_containment.train_step_filtered`, `study.train_step`):

| Arm | Trains | Start |
|---|---|---|
| `tss_processing` | backbone + `T`; `M = gamma = 0` stored constants | literal TSS, `T0 = 1` |
| `generalized_processing` | backbone + `M, gamma, T` | the same TSS function |
| `native_full` | backbone | source |
| `operator_full` | backbone + `kappa` (outside the family) | `kappa = 0` |
| `native_frozen` | nothing (evaluation-only anchor) | source |

- **Sources:** the independent Momentum sources of replication
  `20260917-011842`, read-only and sha256-verified (500 development, 501–503
  final).
- **Infrastructure:** source restore, checkpoint acceptance and persistence,
  step checks, the supervisor, terminal verdict, finalizer and digest
  pattern.

What changes: the task (s2), the metric weighting (s3), the selection rules
(s4), and a mechanism diagnostic added to the same batch (s5). The shared
rollout gains an opt-in `trace` flag, used only by the diagnostic, whose
default changes no output.

## 2. Task schedule (`experiments/prospective_momentum/temporal_task.py`)

Same vocabulary (32 keys, 8 values), event types, input contract and
latest-write oracle. Each episode is 64 tokens (indices zero-based):

| Tokens | Content |
|---|---|
| 0–9 | write 10 target associations, random order |
| 10–12 | 3 writes to distinct non-target keys |
| 13–53 | 5 probe blocks, one per delay `d` in {1, 2, 4, 8, 16}, in random block order |
| 54–63 | query all 10 targets once, random order (late probes) |

**A block starting at token r:**
- `r`: rewrite one of 5 selected targets;
- `r+1 .. r+d-1`: `d−1` fill tokens;
- `r+d`, `r+d+1`: a probe pair that queries the revised key and one of 5
  untouched targets. Every block uses a distinct untouched target.

The two queries sit at `d` and `d+1` tokens after the revision. Their order
is balanced, so revised and untouched probes are measured at the same
distribution of times after that revision.

**Families:** as before, the model receives no family identifier.
- `revision`: the rewrite uses a different value.
- `recall`: the rewrite repeats the original value.

**Interference conditions:** one per episode.
- `idle_gap`: every fill token is IDLE (key 0 with no value, the generator's
  existing idle default).
- `intervening_writes`: every fill token writes a uniformly drawn non-target
  key with an independent value.

**Nominal delay.** `d` is a nominal block delay. The revised query sits at
actual offset `d` or `d+1`, balanced across episodes. So immediate revision
`I` pools revised queries at actual offsets 1 and 2; it is not exclusively
one-token accuracy, and tables and plots label delays as nominal.

**IDLE means no new residual write, not frozen memory.** Momentum and the
processing state keep evolving through IDLE tokens, and the existing key-0
gate features apply to them. The pretrained sources never saw IDLE tokens;
every arm shares that shift.

**Two distinct times are recorded per query:**
- `since_revision`: tokens since the block's revision, or since the key's own
  revision for late selected probes;
- `age`: tokens since the queried key's own last write.

For revised probes the two are equal. For untouched probes `age` is always
larger (checked).

**Exact balance.** The batch size per family must be a multiple of 4; all
declared batches are (8, 256, 512, 32). Episodes alternate conditions, and
probe order flips every second episode of a condition. So in each family:
- every block cell (kind × delay × condition) has n/2 queries, split exactly
  half and half between the two probe offsets;
- each (late kind × condition) cell has 5n/2 queries.

These properties are checked on every generated set at runtime.

**Counts and documented changes from the original task.**

| | Original task | This task, idle | This task, intervening |
|---|---:|---:|---:|
| Length | 64 | 64 | 64 |
| Queries | 16 | 20 | 20 |
| Non-fill writes | 48 | 18 | 18 |
| Fill tokens | — | 26 idle | 26 writes |
| Total writes | 48 | 18 | 44 |
| Targets / selected | 8 / 4 | 10 / 5 | 10 / 5 |

- **Write counts differ by condition.** Total writes differ by exactly the
  26 fill tokens, which are the manipulated variable. Length, query count,
  target writes and the per-cell query counts are identical across
  conditions.
- **Distribution shift:** the pretrained sources never saw IDLE tokens. This
  shift is shared by every arm.
- **One frozen mixture** (both families, both conditions) is used for
  training, development, final validation, held-out evaluation and the
  diagnostic, each from its own fresh stream.
- **No selective use of delays:** no delay or condition is dropped,
  reweighted or chosen after results.

**Model inputs** are only `key_id`, `val_id`, `event` and the loss `label`.
Delay, kind, offset, condition, family, the two times and the oracle never
enter the model; a check intercepts what evaluation passes.

## 3. Metrics and weighting (`temporal_task.aggregate`)

**Per family:**
- accuracy and cross-entropy per block cell (kind × delay × condition), and
  per late cell (late kind × condition);
- `revised_probe`, `untouched_probe`: equal-weight means over their 10 cells;
- `late_selected`, `late_untouched`: equal-weight means over the two
  conditions;
- `macro`: the equal-weight mean of those four;
- delay curves (mean over conditions) and condition splits.

**Study aggregates:**

| Name | Definition |
|---|---|
| primary (revision) | revision-family macro accuracy |
| revision CE | revision-family macro cross-entropy, same weights |
| retention | mean(revision `untouched_probe`, revision `late_untouched`) |
| recall | recall-family macro accuracy |
| immediate revision `I` | revision-family `revised_probe` at delay 1 |
| later `L` | mean of: revision `revised_probe` at delays 4, 8, 16; retention; recall later = mean(recall block probes of both kinds at delays 4, 8, 16, recall late macro) |

Every cell must be non-empty and finite at every checkpoint, together with
the aggregates and state norms. Otherwise the checkpoint fails.

## 4. Selection, screens and the timing analysis (development only; frozen)

### 4.1 Checkpoint selection (main endpoints)

Every trained family gets the same opportunities: learning rates 0.003 and
0.01, 200 updates, and checkpoints at 0, 25, 50, 100 and 200 (9 distinct).

**Selection rule, identical for every family:** highest development revision
macro accuracy, then lower revision CE, then fewer updates, then lower
learning rate.
- Only accepted, finite checkpoints are eligible. Any unaccepted or
  non-finite checkpoint fails selection.
- **No retention or recall constraint enters selection.** Those are
  conditions of the held-out screen.

**Deployment feasibility** is recorded separately and never used for
selection. A trained endpoint is deployable only if, on development data, it
has revision ≥ native + 1 pp, retention ≥ native and recall ≥ native.
Deployment then follows the existing plan (`tss_containment.deployment_plan`):
strict improvement over explicit fallbacks, and a native model is never
labelled TSS. Fallbacks are deployment choices and never count as scientific
improvements.

### 4.2 Primary aggregate screen

Held-out, reported separately for each comparison:

| Comparison | Kind |
|---|---|
| generalized − literal TSS | primary |
| generalized − native Momentum | primary |
| generalized − learned operator | primary (the operator is outside the nonnegative-`gamma` family) |
| literal TSS − native | reference |
| operator − native | reference |
| generalized − frozen source | descriptive |

**Pass condition:**
- mean revision difference ≥ +1 pp;
- positive paired revision differences in all three final seeds;
- mean retention difference ≥ 0 **and** mean recall difference ≥ 0.

The historical −1 pp safeguard is printed alongside for reference only.

**Secondary:** paired differences per cell and by delay and condition, plus
`I` and `L` and its three parts. Secondary results **cannot rescue** a failed
aggregate verdict.

### 4.3 Timing versus stronger writing (secondary, frozen)

**(a) Main endpoints.** Report generalized − TSS differences in `I` and `L`,
per seed, together with each endpoint's executed immediate amplitude `c` (the
first-write gain). The descriptive flag `later_improved_at_no_worse_immediate`
means **"later improved in every seed with no mean immediate decrease"**. It
requires both:
- mean held-out `I` difference ≥ 0;
- `L` difference > 0 in all three seeds.

It can be true when immediate accuracy rises by ten points, or when one seed
loses immediate accuracy. It is therefore **not** a held-out match and does
not separate timing from stronger writing.

**(b) Matched operating point.**
- **Reference:** the selected literal-TSS development checkpoint's `I_ref`.
- **Feasible:** generalized development checkpoints with
  `I_ref <= I <= I_ref + 0.01`.
- **Choice:** highest development `L`, then `I` closest to `I_ref`, then
  fewer updates, then lower learning rate.
- **No feasible checkpoint:** the analysis is reported as unavailable. The
  band is never widened.

When a matched checkpoint exists, its recipe is carried to the final seeds
(s6). The endpoint is labelled **development-matched**.

**Held-out match, reported separately** (review of 36b8e57, R2), for both
the main and the development-matched endpoints:
- for each seed, whether the held-out `I` difference against the selected
  TSS endpoint lies in the already declared band `[0, 0.01]`;
- whether that holds in all seeds;
- separately, whether the mean difference lies in the band.

If the development match does not hold on held-out data, the report says so.
Nothing is reselected, the band is not widened, and no result is suppressed.
The descriptive flag is reported beside this record, not merged into it.

**Attribution.** Even a held-out accuracy match does not match `c`, the
gates, the learned backbone or the actual write matrices. The trained
comparison can support an operating trade-off; it cannot by itself attribute
that trade-off solely to response timing. The frozen-backbone,
equal-first-write diagnostic (s5) is the controlled mechanism comparison, and
it is separate from evidence of learned performance. The matched checkpoint
is chosen against the primary-selected TSS endpoint, not against an
exhaustively optimized TSS retention frontier.

## 5. Mechanism diagnostic (`temporal_diagnostic.py`, same batch, before training)

**Settings.** For `h = 1` with zero processing histories and a unit residual
impulse, checked exactly with rationals:

| Setting | `(M, gamma, T)` | `a, b, c, d` | Impulse response |
|---|---|---|---|
| literal TSS | (0, 0, 1) | 0, 0, 2, 1 | (2, −1, 0, 0, …) |
| generalized | (1/4, 0, 1/2) | 0, 1/3, 2, 2/3 | (2, −2/3, −2/3, 2/9, 2/9, −2/27, …) |
| native point (reference) | (0, 1, 0) | 0, 0, 1, 0 | (1, 0, …) |

Both contrasted settings have immediate amplitude `c = 2`, unit DC gain and
strict stability, but different subsequent dynamics. They are **diagnostic
settings only**: never imposed on the trained arms, and not claimed as
winning coefficients.

1. **Open loop (prescribed residual).** The production coefficient formation
   and the production update run on a prescribed impulse. The result must
   agree with the exact rational response at TRAJ32 (2e-5, relative with a
   unit floor), and the stated first values must equal the exact recursion.
2. **Closed loop (actual memory).** Later residuals depend on W.
   - **Backbone:** the frozen development source, 64 diagnostic episodes
     (32 per family, balanced conditions).
   - **Incoming state:** each episode's delay-16 block starts from the
     common incoming W and U after the preceding token. Those come from the
     same episode under the processing law at its exact native point, whose
     logits are verified against the native rule at TRAJ32 in the same
     diagnostic.
   - **Common inputs:** identical events and gates, and explicitly zero
     processing state for every setting.
   - **Precondition:** the first-write change in W (a full matrix, not a norm
     or coefficient) must agree between TSS and the generalized setting at
     TRAJ32 in every episode, and be nonzero. Only then are later readouts
     compared.
   - **Readouts:** the revised key's and the block's untouched key's
     label probability and accuracy at every token offset 0–17, per family
     and condition, as differences between settings.

**Finite-first contract** (review of 36b8e57, R1). Before any use or
reduction, the diagnostic checks each of the following and returns a named
failure at the first failing stage:
1. **Returned rollout arrays**, from both the full and the suffix rollouts:
   logits, W and U traces, the processing-state summary (max |entry| of
   `y, y_prev, R_prev`) and every returned coefficient copy. Each is checked
   finite and in the production dtype. Only after this are production outputs
   converted to host float64 for analysis.
2. **Native reference:** the native rule's logits.
3. **Inputs to the readout:** the incoming W and U, the normalized keys, and
   the readout weights and bias.
4. **Derived readouts:** logits before softmax and argmax, probabilities
   before aggregation, a non-empty cell before every mean, and every
   reported curve and difference before acceptance.

Every distinct returned coefficient set must pass the executed-coefficient
gate.

**Failure policy:** any open-loop disagreement, native-point disagreement,
first-write disagreement, non-finite or wrong-dtype value, empty cell or
unaccepted executed filter **fails the run** before training.

**Scope:** neither diagnostic alone establishes a task improvement, a
novelty claim or a SOTA result. An isolated-filter stability check is not a
switching closed-loop stability theorem.

## 6. Work, streams and cap

| Item | Count |
|---|---:|
| Development trajectories | 4 families × 2 learning rates = 8 (1,600 updates) |
| Main final trajectories | 4 families × 3 seeds = 12 (≤ 2,400 updates) |
| Matched-recipe trajectories | 0, or 3 if the matched checkpoint's learning rate differs from the main selection (≤ 600 updates) |
| **Maximum updates** | **4,600** |
| Frozen-source evaluations | 4 |
| Diagnostic episodes | 64 |

**Final plan.** There is one trajectory per (family, learning rate, seed),
run to the largest update any endpoint needs at that learning rate. Each
endpoint is that trajectory's validated, persisted checkpoint at its
selected update. Training is deterministic, so this equals stopping there. A
selected update of 0 gives a valid zero-update named-family endpoint.

**Streams** (fresh, asserted disjoint from every earlier range):

| Stream | Identifier(s) |
|---|---|
| continuation training | 525,000,000 – 525,030,199 |
| development validation | 550,000,000 |
| final validation | 551,000,000 |
| held-out (opened after every selection and endpoint is frozen) | 560,000,000 |
| diagnostic | 570,000,000 |

**Cap.** One 600-second cap covers, in order:
1. GPU check;
2. the TSS-containment float64 checks and float32 probe (unchanged; they cover
   the law and gate);
3. this study's checks;
4. task structure checks;
5. start-point checks (storage identity, native-point recovery, literal-TSS
   start acceptance);
6. the diagnostic;
7. measured preflight;
8. training, held-out evaluation, source verification, digest and verdict.

**Measured preflight** times a real step (7 steps, each checked), one full
checkpoint (evaluation, acceptance, persistence) and one evaluation per arm.
It projects the worst case:
- every development and main final trajectory at 200 updates with 5
  checkpoints;
- the matched trajectories as if required;
- every held-out evaluation, scaled by the declared 2× held-out size;
- the anchor;
- a recorded 40 s host allowance.

The diagnostic and start checks have already consumed their time. Preflight
refuses as INCOMPLETE if the projection does not fit, and FAILS on invalid
numerical state. There are no retries, no trimming, no separate exploratory
dispatch and no automatic relaunch.

## 7. Focused checks (cluster, inside the cap)

`tests/test_prospective_temporal_response.py` (float64). The review of
36b8e57 added two groups of regressions:
- diagnostic payload and readout guards: an otherwise valid payload fails on
  a non-finite U trace, processing summary, logit, W trace or coefficient
  copy, on a missing array, or on a non-production dtype; readout checks
  reject non-finite, empty or empty-cell values;
- a reporting fixture: an immediate difference of +0.10 makes the
  descriptive flag true and the held-out match false, and mixed per-seed
  cases are reported separately from the mean.

The existing checks:

- **Task:** exact balance and oracle; declared block shapes and fill;
  `since_revision` versus `age`; offset balance in every cell; family
  rewrites; batch-size refusal; an intercept showing only model inputs reach
  evaluation.
- **Metrics:** equal-cell weighting and the declared aggregates; rejection of
  NaN and empty cells.
- **Diagnostic:** exact impulse responses and coefficients (`c = 2`, unit DC
  gain); open-loop production agreement; closed-loop first-write agreement
  with identical readouts at offset 0 and differing later readouts.
- **Rollout trace:** opt-in, and shares the scan.
- **Selection:** unconstrained main selection with separate deployment
  feasibility; the matched rule (band, ordering, unavailable without
  relaxation); unaccepted checkpoints blocking selection; learning-rate
  merging and the work bound in the final plan.
- **Screen:** the aggregate criterion and the secondary cells.
- **Runner:** kept zero-update endpoints; a kept intermediate endpoint that
  matches its saved checkpoint and recorded metrics; `keep_at` restricted to
  checkpoints.

The TSS-containment modules re-verify the law, gate, repair, gradients,
recovery and runner. The float32 production paths of this study, including
the diagnostic, are checked at runtime by the study's own hard checks.

## 8. Reporting

For each comparison the report gives:
- the aggregate verdict first;
- then per-seed revision, retention and recall;
- per-delay and per-condition curves for revised and untouched probes in
  both families;
- late probes;
- `I` and `L` with its three parts;
- executed coefficients, including the immediate amplitude `c`;
- selected checkpoints;
- the matched-operating-point record;
- the diagnostic, labelled diagnostic;
- costs.

Deployment fallbacks appear in their own table. The screen is a finite-sample
screen on three seeds of a small synthetic task, not significance, a
benchmark or SOTA. No Gated DeltaNet arm is included.

## 9. Launch (after static review only)

    bash bin/run_experiments/cluster_prospective_temporal_response.sh

## 10. Amendment record (review of 36b8e57)

- **R1:** the diagnostic enforces its finite-first contract on every
  returned state, logit, processing summary and coefficient copy (in the
  production dtype), on the incoming states and readout inputs, and on
  derived readout logits, probabilities, cells, curves and differences. Each
  stage returns a named failure. Guard regressions were added.
- **R2:** held-out immediate matching against the declared `[0, 0.01]` band
  is reported per seed, for all seeds and for the mean, separately from the
  unchanged descriptive flag. The flag is now worded "later improved in every
  seed with no mean immediate decrease". The matched endpoint is labelled
  development-matched, and an attribution note separates accuracy matching
  from timing. A reporting fixture was added.
- **Clarifications recorded:** `d` is a nominal delay (`I` pools offsets 1
  and 2); IDLE means no new write, not frozen memory; isolated-filter
  stability is not a switching theorem; the matched checkpoint is chosen
  against the primary-selected TSS endpoint; the diagnostic supports no win,
  novelty or SOTA claim.

Unchanged: equations, coefficients, arms, streams, loss, schedule, training,
selection rules, the match band, numerical tolerances, performance criteria
and the single 600-second cap.

