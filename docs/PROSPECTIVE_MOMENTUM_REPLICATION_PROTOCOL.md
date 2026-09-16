# Protocol: independent-source replication of prospective Momentum DeltaNet

**Status: frozen for static review. NOT AUTHORIZED TO RUN.** Nothing in this
replication has executed, locally or on the cluster.

- Brief: `PROSPECTIVE_MOMENTUM_INDEPENDENT_REPLICATION_2026_09_17.md`.
- Reused protocol and audit: `docs/PROSPECTIVE_MOMENTUM_PROTOCOL.md`,
  `docs/PROSPECTIVE_MOMENTUM_PROOF_AUDIT.md` (equations, projection, frozen-
  token scope, passive-sector labels — all unchanged).
- Branch: `prospective-momentum-replication`, from
  `prospective-momentum-delta` at 2ab420d. Earlier branches, runs and reports
  are untouched.

## 1. What is replicated, and what is preserved

The completed run `/Users/durso/s5-runs/prospective-momentum/20260917-000431`
measured, on held-out data over three continuation seeds that shared ONE
pretrained source per family:

| Rule | Revision % | Retention % | Recall % |
|---|---:|---:|---:|
| Candidate (prospective correction) | 67.51 | 66.96 | 76.69 |
| Native Momentum DeltaNet | 55.61 | 68.04 | 77.20 |
| Gain-only control | 55.62 | 68.11 | 77.19 |

The joint literature screen FAILED: retention fell 1.08 pp against native
Momentum (allowed −1.00), and 1.15 pp against the gain control. **Both facts
are preserved.** This replication asks whether the revision benefit and the
retention trade-off recur across independently pretrained models. It does not
revise that verdict.

**No new mechanism.** There is no new architecture, loss, retention
regularizer, teacher, gate, coefficient bound or response parameterization.
The coordinator's interim retention-weighting and checkpoint-selection
proposal is withdrawn and is NOT implemented. The loss stays the query-mean
cross-entropy; selection stays the update-200 rule; the success criteria and
thresholds are unchanged.

## 2. Six arms, unchanged

1. `momentum_delta` — native continuation.
2. `prospective_momentum` — same Momentum checkpoint, kappa = 0.
3. `gain_momentum` — same Momentum checkpoint, log_g = 0.
4. `gated_delta` — its own source.
5. `gp_two_sided` — previous generalized rule, unchanged, its own source.
6. `tss_eq17` — TSS Eq. (17) direct, applicability-limited, its own source.

Production rollouts, gate initializers, projection, dtype, task, carry sizes,
optimizer, BPTT, batch size and output timing are the reused, passing
implementation. Arms 2, 3 and 1 start from identical copies of the same new
Momentum checkpoint, and that identity is checked per source seed before any
continuation.

## 3. Independent sources (new)

Source seeds: **500** (development) and **501, 502, 503** (final). For each
seed all four source families are freshly initialized and pretrained.

- **Initialization:** `experiments.meta_delta.model.init_params` with the
  original slot-B coefficients from `meta_delta.calibrate.configurations()`.
  Common tensors come from the same seed across families; the four seeds are
  independent.
- **Recipe, fixed:** 200 updates, lr 0.01, original unweighted loss, batch 8
  episodes per family, original optimizer and projection. This reproduces the
  prior designated slot-B source recipe; it is not a hyperparameter search.
- **Artifacts:** each source saves its parameter tree and optimizer state in
  `<run>/sources/`, with `manifest.json` (initializer metadata, recipe,
  stream identity, start/end metrics, curve, parameter counts, sha256) and
  `SHA256SUMS` in `sha256sum -c` format.
- **Save/restore check:** every source is restored immediately and must be
  bitwise equal and reproduce its metrics under the completed study's
  reproduction rule.
- **Read-only afterwards:** every continuation restores its source from disk
  and refuses on a checksum mismatch. The launcher re-verifies `SHA256SUMS`
  at finalization, and the study finalizer re-hashes every manifest file; a
  changed source forces FAILED.
- **Validity:** the existing numerical requirements apply to sources
  (finite scalars, parameters, optimizer state and metrics; the arm's own
  validator, including the frozen-token transition classification).
- **A low source score is an observation**, never a refusal and never a
  reason to replace a seed. Sources are never ranked or discarded.
- The completed study's read-only checkpoints may still be READ by the
  unchanged restored-checkpoint checks. They supply no parameter and no
  selection measurement here; a focused check asserts the replication code
  never calls that restorer.

## 4. Continuation, selection, evaluation

- **Development:** source 500 only, six arms × slots A (lr 0.003) and B
  (lr 0.01), 200 updates each (12 runs).
- **Selection:** the original rule — update-200 development revision macro
  accuracy, then lower revision cross-entropy, then slot A. Updates 0 and 100
  stay diagnostics. No checkpoint selection is added. The choice is deep-copied
  and frozen; the run FAILS if it differs before held-out or after the screens.
- **Finals:** seeds 501–503, each from its OWN source, selected slot, 200
  updates, outer optimizer reset identically (18 runs). Continuation batches
  are paired across all six arms at a seed; the momentum trio shares that
  seed's Momentum source.
- **Held-out:** one common set, generated and hashed only after all final runs;
  the opening is persisted before use; never used for selection.
- Validation 256 episodes per family, held-out 512 per family, original
  chunking and evaluation conventions.
- **Work:** 16 + 12 + 18 = **46 runs, 9,200 updates**, all inside the one
  600-second cap and all reported. This is not "30 continuation runs".

## 5. Streams (frozen, verified disjoint before execution)

| Purpose | Generator seed | Range used |
|---|---|---|
| Source training | 200,000,000 + seed·10,000 + update | 205,000,000–205,030,199 |
| Continuation training | 220,000,000 + seed·10,000 + update | 225,000,000–225,030,199 |
| Source validation / diagnostics | 240,000,000 | single |
| Development continuation validation | 250,000,000 | single |
| Final continuation validation (diagnostic) | 251,000,000 | single |
| Held-out evaluation | 260,000,000 | single |

Disjointness is asserted at start against nested, adaptive, meta-delta, the
completed prospective-momentum study (including its full train range for all
seeds < 1000) and the small fixed fixture seeds (0–9,999) of the focused
checks. A collision refuses the run; nothing is substituted silently. Data
digests and all ranges are saved.

## 6. Verdicts, unchanged

The existing screen implementation and thresholds, evaluated on seeds
501–503:

- mean paired revision improvement ≥ +1 pp;
- revision improvement positive in every final seed;
- mean retention AND mean recall differences each ≥ −1 pp.

The screen implementation, its comparisons and thresholds are reused
unchanged; only its scope note is replaced by the replication's own (R3).
The joint literature screen (native Momentum AND Gated DeltaNet) and the
gain-only comparison are reported together; the previous generalized rule and
the applicability-limited TSS comparison are reported separately. The digest
prints, for every comparison, whether mean retention and recall **increased or
decreased**, with the explicit note that passing the −1 pp safeguard does not
mean zero measured loss. No threshold may be amended after seeing this
replication.

## 7. Focused checks (added; existing coverage reused)

`tests/test_prospective_momentum.py` is unchanged and runs in the same
invocation (model, loss, projection, analytic kappa sensitivity, supervisor,
terminal paths). `tests/test_prospective_momentum_replication.py` adds only:

1. frozen stream values and disjointness, including source vs continuation
   streams at the same seed;
2. the arm-to-source plan: each final arm receives its own seed's source, the
   momentum trio shares one source, development uses only source 500, and no
   completed-study checkpoint appears;
3. source independence across seeds, and shared common tensors within a seed;
4. planned work: 46 runs and 9,200 updates;
5. manifest round trip, `sha256sum -c` format, and refusals for a missing
   file, a wrong checksum, a tampered file and an ambiguous entry;
6. a changed or removed source file fails the finalizer;
7. the source recipe is the original slot-B definition, and parameter counts
   are 569/570;
8. the selection is frozen, uses only source-500 development rows, and a later
   mutation is detected;
9. the screens run on seeds 501–503 with unchanged thresholds, state
   directions, and are incomplete under the old seeds;
10. the launcher's extra verification of the new sources: unchanged, changed,
    missing baseline, and that it can only worsen the integrity code.
11. finalization before any source exists reports invariance as unavailable,
    preserves a FAILED verdict and keeps a preflight refusal INCOMPLETE; a
    partial manifest is verified against its own baseline (R2);
12. the replication's screen scope note replaces the reused default, which is
    kept alongside and is unchanged for the completed study's own runs (R3).

## 8. Budget, preflight and finalization

One absolute 600-second deadline covers the backend probe, focused checks,
preflight, all 46 runs, evaluation, the kill grace, both source verifications,
the digest and the terminal verdict. The reserve allocation, process-group
supervisor, terminal verdict merging and finalizer are the reused,
passing implementation.

Preflight measures the ACTUAL paths on disposable state: source-training steps
on the source stream for all four families, continuation steps on the
continuation stream for all six arms, evaluation, validation and reporting,
and one save + checksum + restore. It projects all remaining work — 16 source
runs, per-seed identity checks, 30 continuations, held-out evaluation and a
40-second host allowance — without re-counting compilation already incurred.
If it does not fit, the run stops INCOMPLETE with the measured shortfall.
Nothing is trimmed: no fewer sources, arms, steps or checks.

Initializers are pure functions of the seed, so preflight leaves no state
behind; its disposable trees are never used later.

## 9. Reporting

The digest (`experiments.prospective_momentum.replication_summary`, written
before the run) prints the source table with metrics, checksums, initializer
metadata, recipe and restore checks; the per-source-seed identity results;
preflight costs and the projection breakdown; every development and final run;
held-out results per source seed with the four query categories, revision CE,
initial metrics, state norms and observed-rollout gates; per-seed paired
differences; learned kappa and gain with projection activity; gates;
parameters and cost; and every screen with its direction statements.

## 10. Not claimed

Three independently pretrained seeds on this small task do not establish a
benchmark or SOTA result, and do not erase the other scope limits:

- frozen-token stability is not switching stability;
- the **completed study** (20260917-000431) observed learned operating points
  outside the passive two-compartment sector under the declared mapping. That
  is a record of that run. These new sources' occupied sectors are not
  asserted in advance: each is reported from its own learned coefficients in
  the digest (`sector (9)`, QHM nu, kappa and the bound ratios);
- the fixed-gate QHM relationship remains relevant to novelty.

## 11. Amendments after static review of 7344e32 (before execution)

Fixture, orchestration and reporting only. The six arms, equations, loss,
source seeds, streams, schedule, criteria and the 600-second cap are
unchanged.

- **R1.** The launcher-verification fixture now sources the shared terminal
  library first and the replication helper second, exactly as the launcher
  does, supplies its own `PY`, `LOG_DIR` and deadline instead of relying on
  inherited shell state, and asserts the subprocess exit code (with stdout
  and stderr) before parsing its output. The four integrity assertions and
  the production verifier are unchanged.
- **R2.** Before the first source exists the baseline is `None`, meaning
  source invariance is **unavailable**, never "verified". The re-hash
  callback tolerates that state without raising. An existing FAILED verdict
  is preserved, and a preflight refusal stays INCOMPLETE. Once a baseline
  exists, every recorded file is verified as before. New non-numerical
  fixtures cover the pre-source phase and a partial manifest.
- **R3.** The replication sets its own screen scope note: the final seeds
  vary initialization, source pretraining and continuation, while the
  held-out set and the development-selected recipe are common by design; the
  small-task, three-seed, non-SOTA, QHM and frozen-token limits are retained,
  and each source's sector is reported from its own coefficients. The reused
  screen's comparisons, thresholds and original default note are untouched;
  the default is kept alongside as `reused_screen_default_note`.
