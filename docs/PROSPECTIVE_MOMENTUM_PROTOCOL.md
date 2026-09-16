# Protocol: prospective correction of a trained Momentum DeltaNet memory

**Status: frozen for static review. NOT AUTHORIZED TO RUN.** Nothing in this
study has executed, locally or on the cluster.

- Brief: `PROSPECTIVE_MOMENTUM_NEXT_STEP_2026_09_16.md`.
- Audit: `docs/PROSPECTIVE_MOMENTUM_PROOF_AUDIT.md`.
- Branch: `prospective-momentum-delta`, from `generalized-meta-delta` at
  e3139f4. All completed branches, runs and reports are untouched.

## 1. Question

Does adding the generalized prospective correction, a discrete residual
derivative, to a working Momentum DeltaNet memory improve revision accuracy
without losing its retention and recall? The comparison is against:

- the literature rules;
- a gain-only control that has no derivative term.

A successful screen would be a development signal only. It would not be a
benchmark result, significance, SOTA, or optimizer novelty: the
fixed-coefficient update is QHM-equivalent at alpha = 1 (audit §4).

## 2. Update laws (audited)

Native Momentum DeltaNet, source-pinned gates (`nested_memory.model._momentum_gates`):

    Wbar = alpha W,  R = m (Wbar k - v) k^T,  Qn = mu Q + eta R,  W' = Wbar - beta Qn

Candidate `prospective_momentum`, equation (6):

    Q' = Qn - kappa eta ((1-mu)/mu) R
    W' = Wbar - beta Qn - kappa beta eta R

- Both lines use Qn from the old carry.
- kappa is stored directly, initialized to exactly 0, and never clipped in
  the forward pass.
- There is no branch on kappa.

Gain control `gain_momentum`:

    Q' = mu Q + g eta R,  W' = Wbar - beta Q',  g = exp(log_g), log_g initialized to 0

All three momentum arms share the same contract. They reuse the native
shell line for line:

- `sanitize_episode` input contract;
- key normalization (the same helper);
- value masking, gate features, the gate block and the mu clamp;
- readout `W k` and output timing;
- carry (W, Q): 128 real numbers;
- full BPTT through `lax.scan`.

There is no stop-gradient, extra matrix, future input or query value.

## 3. Arms and sources

Source run, **read only**: `/Users/durso/s5-runs/meta-delta/20260916-222310`
(commit 5dc4b07).

| # | Arm | Starts from | Added parameters |
|---|---|---|---|
| 1 | `momentum_delta`, native continuation | `params/dev_momentum_delta_B_seed300.msgpack` | none (569 total) |
| 2 | `prospective_momentum` | the same file, + `kappa = 0` | 1 (570) |
| 3 | `gain_momentum` | the same file, + `log_g = 0` | 1 (570) |
| 4 | `gated_delta` | `params/dev_gated_delta_B_seed300.msgpack` | none |
| 5 | `gp_two_sided`, old generalized rule, unchanged | `params/dev_gp_two_sided_B_seed300.msgpack` | none |
| 6 | `tss_eq17`, direct and applicability-limited | `params/dev_tss_eq17_B_seed300.msgpack` | none |

Source verification. Any failure below is a refusal (FAILED), never a
substitution.

- **Metadata.** `status.json` must have `run_id = 20260916-222310`,
  `dev_seed = 300` and `complete = true`. It must contain exactly one
  development row with `tag = dev`, the family, `config = B` and
  `seed = 300`, whose stem equals the file name. `selection.json` must select
  B for that family. Slot B is designated by the brief. It is not re-chosen
  here, and no final-seed checkpoint is considered.
- **File.** The leaf set, shapes and float32 dtype must equal the family's
  template, and every value must be finite.
- **Hashes.** sha256 of `status.json`, `selection.json` and the four
  checkpoints is taken before anything else. It is re-checked at every exit,
  both by the launcher (`sha256sum -c`) and by the study (`source_unchanged`).
  The re-check affects the verdict (§8): a changed or missing source forces
  FAILED, and an unavailable verification can never give PASS or claim
  invariance.
- **Reproduction.** Each restored source is re-evaluated on the completed
  study's development validation stream (60,000,000, 256 per family) and
  compared with the saved `final_validation`.
  - Tolerance, new and explained: every family × category accuracy within
    **1 query**, and cross-entropy within **1e-4 relative**. These are the
    same code, data, chunking and dtype. An allowed one-query difference is
    recorded as a discrepancy; no cause (such as an argmax tie) is
    attributed without observing it.
  - All differences are recorded, including primary, retention and recall.
- **Update-zero identity of arms 1–3.** Evaluated on the new development
  validation stream: per-category accuracy counts must be **identical** (0
  queries), and cross-entropy within **2e-5 relative** (the float32
  trajectory tolerance). The starting functions of arms 4–6 differ and are
  reported through each run's `start_validation`.
- **Source transitions.** The source momentum gates at kappa = 0 and g = 1
  must give no unstable or non-finite executed frozen-token transition
  (§6). Otherwise the result is FAILED with a diagnosis before any training,
  and no tolerance is changed.

## 4. Schedule, data, selection

- **Optimizer.** The outer optimizer is reset identically for every run: the
  unchanged `clip_by_global_norm(1.0)` + Adam(0.9, 0.999, 1e-8) object. All
  existing parameters are trained. The learning rate is applied per call.
- **Updates and batches.** 200 updates per run. Batch 8 episodes per family.
  Validation 256 per family at updates 0, 100 and 200. Held-out 512 per
  family.
- **Slots.** A (lr 0.003) and B (lr 0.01) for every family.
- **Development seed 400.** 6 × 2 = 12 runs. Selection by highest revision
  macro accuracy, then lower revision cross-entropy, then A.
- **Final seeds 401, 402, 403.** 6 × 3 = 18 runs. Each restarts from its
  designated source, not from a development endpoint. They measure
  training-stream variation conditional on one source per family, not
  independent pretrained models.
- **Streams.** New named streams:
  - train: 80,000,000 + seed·10,000 + update, i.e. 84,000,000–84,030,199;
  - development validation: 90,000,000;
  - final validation: 91,000,000;
  - held-out: 100,000,000, generated and hashed only after all final runs.

  The study asserts at start that these ranges are disjoint from each other
  and from every earlier generator stream: nested, adaptive, and meta-delta
  (whose train range 50,000,000–59,990,999 is included in full). It refuses
  otherwise.
- **Production dtype.** float32 with x64 disabled, asserted in the study
  process.
- **Artifacts.** Parameters and optimizer states for every run are saved
  under `out/params/{dev|final}_{rule}_{A|B}_seed{seed}[_opt].msgpack`.

## 5. Projection

After every optimizer update, and before the next forward pass, from the
**updated** gate parameters:

    kappa ← min(max(kappa, 0), (1 - 1e-3) · min_s B(s))
    log_g ← min(log_g, log(min_s G(s)) + log1p(-1e-3))

- `B(s)` is bound (12) and `G(s) = (1+alpha)(1+mu)/(alpha q)`. Both are +inf
  where `alpha q = 0`.
- s ranges over **every write setting**: event WRITE × 32 keys × {8 values,
  absent}, i.e. 288 inputs. This is a superset of the task's writes.
  Non-write tokens have `m = 0` and triangular transitions independent of
  kappa.
- The margin 1e-3 is a declared numerical safety margin, supported by the
  arithmetic estimates and measured checks in audit §6. It is not a
  universal certificate for trained gates, and it is not copied from rho
  (amendment R4).
- Zero is the exact native fallback. The optimizer state is untouched.
- The meta-delta `raw_r` projection is applied unchanged to arm 5.
- **Telemetry, every update, arms 2, 3 and 5.** Proposal, post value,
  projection cap (margin included), un-margined bound, overshoot,
  projection flag and the extension scalar's gradient.
  - kappa is a direct value, so its summary is the relative margin
    `1 - post/cap`.
  - log_g and raw_r are logarithms, so their summaries are the log slack
    `cap - post`; the gain additionally reports its relative margin
    `1 - exp(post - cap)`.
  - Arm 5 exports its real proposal, cap (`log_rho_upper`) and un-margined
    log bound. Arms without an extension scalar report NaN, meaning
    unavailable, never zero placeholders.

## 6. Validity (checked at source, preflight and every run end)

Nothing is clamped by validation. FAILED stops the batch.

- **All arms.** Finite parameters, optimizer state, training scalars and
  metrics. Metrics include the computed W/Q state norms and, for the
  momentum family, the finiteness of the observed-rollout gates (amendment
  R3.4).
- **Momentum family.**
  - Gates on all 864 token inputs must be finite, with alpha ∈ [0,1],
    beta ∈ [0,1], mu ∈ (0,1] and eta ∈ [0,2].
  - Executed frozen-token transitions of the production step, on the 288
    write settings, are formed by applying the step to basis carries with
    the exactly representable key e_0, v = 0 and m = 1.
  - Classification is exact rational arithmetic on the rounded float32
    entries: stable, **neutral** (a Jury expression exactly 0, only possible
    from rounded native gates; counted and reported, not failed; a spectral
    label for that rounded matrix, not a theorem that such trajectories stay
    bounded),
    **unstable** or **non-finite**. The last two are FAILED, including for
    the native arm, since exact arithmetic excludes them (audit §5).
  - Candidate: `0 ≤ kappa < B_f64`, the float64 bound on the executed float32
    gates.
  - Gain: `0 < executed_g < G_f64`. Here `executed_g = jnp.exp(log_g)` in the
    parameter dtype, the value the rollout uses. The host float64
    exponential is reported only as `reference_g_f64` (amendment R3).
  - Coverage labels (amendment R4). TABLE coverage is the exact
    classification of rounded executed transitions of table-evaluated gates.
    OBSERVED-ROLLOUT coverage is the gates returned by the actual
    evaluation rollouts (`observed_rollout_gates` in every evaluation):
    distributions, and at write tokens the float64 bound, kappa/bound (or
    executed g/bound) and analytic Jury minima on those rounded gates. They
    are reported separately; neither is a proof for arbitrary rollout
    states, reduction orders or switching.
- **Arm 5 and arm 6.** The completed study's `validate_coefficients`,
  unchanged.
- **Scope.** This is a frozen-token diagnostic, not a stability proof under
  switching.

## 7. Focused checks (cluster, inside the cap)

`tests/test_prospective_momentum.py` runs in float64. Tolerances:

| Tolerance | Value |
|---|---|
| ID64 | 1e-9 |
| GRAD64 | 1e-8, with floor 1e3 eps64 G |
| FD64 | 1e-6 at h = 1e-5, 1e-6 |
| EXACT64 | 1e-12 |

1. The semi-implicit canonical step (3), solved independently as a linear
   system with map (4), equals (6), at h ∈ {1, 0.5} and kappa ∈ {0, 0.37, 2.5}.
2. Identity (7), and the transfer (8) via `scipy.signal.lfilter`, on a
   prescribed residual.
3. Pulse checks at alpha = 1: immediate factor (1+kappa), unchanged total
   `-q/(1-mu)`, and the 3/2 example.
4. The gain control changes both responses; kappa changes only the immediate
   one.
5. QHM equivalence in closed loop: state-dependent residual, key switches,
   idle tokens, nonzero incoming W and Q, and the state map
   `g = (1-mu)Q/(eta nu)`. Both nu > 0 and nu < 0 are covered.
6. kappa = 0 and g = 1 against native Momentum DeltaNet, on a stress fixture
   **and on the restored development checkpoint** (float64 cast):
   - logits and both carries;
   - a chunk boundary (streaming);
   - nonzero incoming carries;
   - all shared parameter gradients;
   - incoming-carry (input) gradients;
   - a finite gradient of the extension scalar.
7. Idle-token (m = 0) identity with native for the same state and gates, at
   any kappa and g.
8. The kappa JVP against central differences at the restored start
   (kappa = 0) and at the interior point 0.5·Kmax. The start derivative must
   not be identically zero.
9. Executed 2×2 transitions against A_kappa and against (11), for random
   gates including alpha = 1, at kappa ∈ {0, 0.5B, (1−1e-3)B, (1+1e-3)B}.
   Eigenvalue moduli must agree with the classification, and above the
   bound the classification must be unstable. The gain transition is
   checked too.
10. At q = 0, kappa is unconstrained and the Q source remains. Neutral,
    unstable and non-finite classifications are checked. The native point is
    strictly inside the bound in exact arithmetic.
11. Projection:
    - it uses the updated gates;
    - zero is the fallback;
    - an outward proposal is projected, and a subsequent MANUAL inward move
      is not blocked (projection behaviour only; not optimizer or
      task-gradient recovery);
    - gain cap;
    - trees without an extension scalar are untouched;
    - in a real compiled training step, the cap equals the bound of the
      post-update gates and differs from the pre-update bound.
12. Input contract: query value and label edits, and future tokens, do not
    change outputs. Counts are 569/570, carry 128. Conversion refuses stray
    leaves.
13. Stream disjointness. Screens: joint literature screen, separate gain,
    old-generalized and TSS comparisons, incomplete pairs, retention
    safeguard. Source refusals and the reproduction rule.
14. The restored sources verify, and their gates are valid.
15. Finalization and terminal paths, with no model:
    - the study finalizer: a changed checksum, a re-hash exception, a missing
      baseline and a persistence failure; a post-restore runtime exception
      and termination through `guarded`, with the original reason kept;
    - terminal verdict rules, including the explicit not-started state
      versus supervisor failure;
    - merging the saved study verdict (amendment F2, JSON fixtures):
      - saved FAILED + watchdog stays FAILED, with the original reason kept;
      - saved FAILED + stage exit 0 stays FAILED;
      - saved PASS + watchdog is INCOMPLETE;
      - saved INCOMPLETE is kept;
      - an unfinalized status is INCOMPLETE;
      - an inconsistent or unreadable status is FAILED;
      - no saved field is erased;
    - the supervisor (amendment F1), with dummy children:
      - leader and descendant both ignore TERM → KILL;
      - the leader exits on TERM while a descendant ignores it → the
        descendant is still KILLed at TERM + grace;
      - a completed leader leaves a TERM-ignoring member → cleaned up and
        recorded;
      - ordinary completion keeps its exit code;
      - an unstartable command gives supervisor_failure.

      Survival is checked as a non-zombie process (`ps` state). Fixture pid
      files are KILLed in `finally`;
    - the launcher library with the supervisor: not_started versus a stub
      supervisor failure; source verification unchanged, changed, no time
      and verifier failure; digest omitted with no time; a saved FAILED study
      with a watchdog stage outcome stays FAILED in the merged
      `status.json`.

    Also checked: metrics acceptance with state norms and observed gates;
    the observed-rollout gate report; extension summary units; NaN
    telemetry for non-extension trees.
16. `tests/prospective_momentum_float32_probe.py` in its own process, x64 off:
    - guard regressions for finite-before-resolution;
    - on the restored checkpoint: native nesting of logits, carries and
      streaming at TRAJ32 = 2e-5, with the mixed parameter and incoming-carry
      gradient criterion;
    - kappa tangents at the start and in the interior, with FD32 (1e-2, 3e-3),
      REL32 2e-2 and the resolvability rule. A finite unresolvable derivative
      is a reported limitation. If 0.5·Kmax ± h does not fit strictly inside
      the cap, the interior float32 check is a reported limitation and the
      float64 check is the evidence;
    - float32 projection caps against float64 bounds, with effective margin
      ≥ 0.5·1e-3;
    - executed float32 transitions (native, candidate at cap, gain at cap):
      classification, and entries against A in float64 at ENTRY32 = 2e-5
      relative;
    - a real float32 training step from a valid interior kappa: cap equals
      the updated-gate cap; projection behaviour (outward proposal, manual
      inward move) on the trained gates;
    - executed-gain underflow (`log_g = -110` executes as 0) is rejected, and
      an ordinary positive executed gain is accepted;
    - dtype assertions.

## 8. Budget, finalization and statuses

- **Cap.** One absolute deadline, DEADLINE = start + 600 s, covers the
  backend probe, focused checks, source restoration and reproduction,
  update-zero identity, preflight, 30 runs, held-out evaluation, the kill
  grace, source verification, the digest and the terminal verdict. The
  30-second reserve is allocated explicitly
  (`bin/run_experiments/prospective_momentum_terminal.sh`, amendment R2):
  - every stage's process group receives TERM at DEADLINE−30, and KILL to
    any remaining member 5 s later;
  - source verification (`sha256sum -c`) is bounded to finish by
    DEADLINE−20;
  - the digest is bounded to DEADLINE−8, at most 12 s;
  - the terminal verdict is bounded to DEADLINE−3, at most 5 s.

  A step with no time left is not started and is recorded as the explicit
  outcome `not_started`, not an exit-code sentinel. Partial logs are kept.
- **Supervisor (amendment F1, 8a09586).** GNU `timeout` stops supervising
  once its direct child is reaped, so a TERM-ignoring descendant of a leader
  that exits on TERM could outlive the KILL. It is replaced by the
  stdlib-only `experiments/prospective_momentum/supervise.py`:
  - it runs each step as the leader of a new session and process group, and
    keeps responsibility for the whole group until it is empty;
  - at TERM time it sends TERM to the group; after the grace it sends KILL
    to any remaining member, even if the leader has already exited;
  - members left behind by a leader that completed are sent TERM, then KILL
    after the grace (never later than TERM time + grace);
  - on Linux it registers as a child subreaper, so orphaned descendants are
    reaped rather than left as zombies;
  - it cleans up the same way if it is itself sent TERM.

  Outcomes are `completed` (with the exit code), `watchdog_term`,
  `watchdog_kill` and `supervisor_failure` (unstartable command, members
  surviving KILL, or no outcome file). A descendant that deliberately leaves
  the process group is outside this guarantee.
- **Preflight.** It measures the actual continuation path from each restored
  source, for all six arms:
  - one shared host-step function (batch generation, compiled step with
    projection, every host synchronization) is used by both preflight and
    training;
  - evaluation and validation/report cost are measured too;
  - retraces are detected;
  - a 40 s host allowance is recorded as such.

  Its updates are discarded.
- **Decisions.** Invalid state or non-finite measured scalars give FAILED
  (exit 4). A retrace, or a projection larger than the remaining time, gives
  INCOMPLETE (exit 3), and nothing starts. There are no trimmed arms, fewer
  updates, retries or extended cap.
- **Study finalization (amendment R1).** One finalizer handles success,
  ordinary failure, runtime exceptions and SIGTERM, which is converted to an
  exception so the finalizer runs within the kill grace.
  - It keeps `computation_status`.
  - It re-hashes the source. Changed or missing gives FAILED. An
    unavailable verification (no baseline, or the re-hash raised) gives at
    best INCOMPLETE and sets `source_unchanged = null` and
    `integrity_verified = false`.
  - It preserves the original `failed` reason and appends
    `integrity_failures` and `runtime_failures`.
  - It persists `study_status`. A persistence failure gives FAILED.
  - The held-out opening (`heldout_opened`, `heldout_opened_at`) is persisted
    BEFORE held-out generation and evaluation;
    `heldout_evaluation_complete` marks the end.
- **Terminal verdict (amendments R1–R2).** It is computed by
  `experiments.prospective_momentum.terminal` and merged into `status.json`
  under `terminal`, and it is always written to `logs/<stamp>/terminal.json`.
  The saved `status.json` is read and validated BEFORE the verdict is
  computed (amendment F2). Worst wins:
  - the saved study verdict: `study_status` with a consistent `study_exit`
    enters with its own severity, and its `failed`/`incomplete` reason is
    kept in `terminal.saved_study` and in the reasons. An unfinalized status
    is INCOMPLETE; an unreadable or inconsistent one is FAILED. Neither a
    watchdog outcome nor an outer exit 0 can erase a saved failure;
  - the stage outcome: completed with exit 0 is PASS; completed with 3,
    `watchdog_term`, `watchdog_kill` or `not_started` is INCOMPLETE;
    completed with 4 or any other exit, or `supervisor_failure`, is FAILED;
  - a verifier or supervisor failure during source verification is FAILED;
  - a changed or missing source checksum gives FAILED;
  - an uncompleted verification or no baseline is never PASS;
  - when a `status.json` exists, a digest that is not `complete` is never
    an unqualified PASS (INCOMPLETE). The saved results stay on disk.

  If the verdict cannot be computed or persisted, the launcher reports
  FAILED.
- **Reference cost.** The completed meta-delta dispatch took 273 s of 600,
  with checks 140 s. This study's cost is unmeasured until preflight.
- **Status fields.** `PROSPECTIVE_MOMENTUM_STATUS=PASS|INCOMPLETE|FAILED` is
  the terminal verdict. PASS is operational only. Partial `status.json` is
  persisted after every run.

## 9. Screens (development)

Rule for every comparison: mean primary (revision macro) difference ≥ +1 pp;
all three paired primary differences > 0; mean retention (revision untouched)
difference ≥ −1 pp **and** mean recall difference ≥ −1 pp.

- **Literature screen, joint.** The candidate must pass against Momentum
  DeltaNet **and** Gated DeltaNet.
- **Gain control.** The same rule, reported independently. If the gain
  control matches the candidate, the simpler gain explanation survives.
- **Old generalized rule** and **TSS Eq. (17)**, the latter direct and
  applicability-limited. Each is reported separately; neither substitutes
  for the literature screen.
- No favourable category substitutes for the primary metric. Every per-seed
  difference is kept whether or not a comparison passes.

## 10. Reporting (digest written before the run)

`python -m experiments.prospective_momentum.summary <run_dir>` runs
automatically at the end of the launcher. It reads JSON only. It prints:

- source hashes, reproduction tables and update-zero identity;
- preflight costs, parameters, carry and runtime;
- every development and final run: start and end validation, gains, curves;
- held-out per seed per arm, with all eight category accuracies, means,
  W/Q norms and observed-rollout gates;
- the full kappa and log_g history (sampled) with projection summary and
  gradients;
- executed transition classification, bounds and ratios;
- gate-table distributions and rounding counts;
- validation-token gate distributions;
- sector (9) occupancy and QHM nu;
- every screen condition.

Raw logarithmic leaves (`raw_log_g`, `raw_log_leaves`) are printed separately
from exponentiated coefficients (`executed_g` as executed in float32,
`reference_g_f64`, `exponentiated`). The earlier
mislabelling of exponentiated values as `raw_eta` is not reproduced.

## 11. Not claimed

- Frozen-token Jury conditions are not global or switching stability.
- Model containment and stability do not imply a performance gain.
- A continuation screen on this custom task is not MQAR and not a benchmark.
- kappa = 0 is outside the passive sector (9). Arms 2–3 use the
  computational generalized equation, and the occupied sector is reported.

## 12. Amendments after static review of a1f0439 (before execution)

These record corrections; the equations, arms, sources, initialization,
tolerances, data, schedule, criteria and cap are unchanged.

- **R1.** Source integrity now determines the terminal verdict. There is
  one finalizer for all exit paths, the held-out opening is persisted
  before use, and control-flow fixtures cover these paths.
- **R2.** All finalization runs inside the absolute deadline, with a bounded
  grace, process-group cleanup and explicit digest status. Stub-child
  fixtures cover it.
- **R3.** Acceptance uses the executed gain. Computed state norms and
  observed gates are included in finiteness acceptance.
- **R4.** The projection margin is described as a declared safety margin
  with its scope stated. TABLE and OBSERVED-ROLLOUT coverage are separated,
  and the observed coverage uses gates returned by the rollouts.
- **Reporting.** The telemetry reports real quantities or NaN, and each
  margin in its parameterization's units. The manual inward move is labelled
  as projection behaviour. The reproduction-discrepancy wording no longer
  attributes a cause.

### Amendments after static review of 8a09586 (before execution)

The model, equations, gates, arms, checkpoints, numerical tolerances,
training protocol and cap are unchanged.

- **F1.** GNU `timeout` is replaced by the process-group supervisor, which
  KILLs descendants even after the leader exits. Dummy-child regressions
  cover it.
- **F2.** The terminal verdict now merges the saved study verdict and keeps
  its reason. "Not started" is an explicit outcome, distinct from a
  supervisor failure. JSON fixtures cover this.
- **F3.** The float32 gain fixture now requires exact equality only with the
  same executed value being reported. The independent float64 exponential
  is compared at the declared production float32 tolerance, 2e-5 relative,
  and the discrepancy is printed.

## 13. Open items for the reviewer (as at a1f0439)

1. The two new tolerances: reproduction (≤ 1 query per category, CE 1e-4
   relative) and update-zero identity (0 queries, CE 2e-5 relative).
2. A native-arm unstable or non-finite executed transition is FAILED rather
   than recorded. Exact arithmetic excludes it; it could only arise from
   rounding.
3. The write table is a superset: it includes the absent value on WRITE.
4. The candidate's final validation uses the float64 bound on executed
   float32 gates. The compiled projection uses float32 with the derived
   margin.
5. The budget is unmeasured. Source reproduction adds four evaluations and
   the float64 suite includes restored-checkpoint gradients and one compiled
   float64 training step.
6. The float32 interior kappa FD may be reported as a limitation if the
   source cap is below about 0.04.
