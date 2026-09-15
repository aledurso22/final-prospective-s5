# Recall study: correction and cluster verification plan

Narrowly scoped. **No training has been run for this plan, and none is
requested by it.** The completed run at
`b5d7211b2d631e104935a231aa41bbd00debf48d` and its artifacts at
`/Users/durso/s5-runs/recall/20260915-220252/` are preserved unchanged.

The physical equation, the coefficient policy, the bounds, the initialization,
the task generator and the data streams are **unchanged**.

## What was repaired

| item | defect | repair |
|---|---|---|
| **R1** | raw `log_response_rho_only` never projected after an optimizer update; above the bound `d rho/d eta = 0`, so zero task gradient, and AdamW decay shrinks `eta` toward zero, away from a negative bound | `project_response_leaves` applied inside `train_step` after `apply_updates`, optimizer state untouched; telemetry records projection events, max raw overshoot, and `rho` gradient/update norms |
| **R2** | gate ran for seed 100 only; used the impulse criterion alone; zero-reference layers dropped | gate runs before **every** seed's continuations; **both** impulse and frequency criteria enforced; non-finite fails; zero-reference layers get a declared absolute criterion (`1e-9`) instead of being skipped |
| **R3** | reported `rho` reconstructed with hard-coded bounds `[-9.21, -1e-4]` instead of `LOG_RHO_BOUNDS`; no parameter trees saved; stored and trainable counts conflated | reporting uses the shared constant and records **raw** values alongside executed `rho` and boundary occupancy; warm-up and final parameter trees saved per seed and arm; `trainable_count` separates stored from trainable |
| **R4** | protocol reversed the mass direction and mis-stated the bound as one-sided and physical | corrected in protocol and report: `mu = T rho`, so lowering `rho` **lowers** mass; `rho` can rise as well as fall; `rho = 1` is the physical boundary and `0.9999` a chosen numerical margin; the `+0.3` pp threshold is withdrawn as never declared for this study |
| **R5** | task described as equal-delay and class-balanced by construction | realized mix recorded as **37.5 / 31.25 / 31.25** per cent and classes as balanced in expectation; `realized_delay_counts` reports it and the runner logs it. **The generator is not changed** |

## Regression coverage added

* an outward crossing is projected back onto the bound, in both directions;
* an **unprojected** outward value has **exactly zero** task gradient — the
  failure mode itself is pinned, so a future regression is detectable;
* after projection the inward task gradient is available again;
* the production `train_step` projects, counts and reports the overshoot;
* the frozen arm's `rho` is untouched while its ordinary weights still train;
* stored versus trainable counts differ for the frozen arm only;
* the realized delay mix is reported and is **not** equal thirds;
* the gate enforces both criteria and implements the zero-reference branch.

## Second correction pass (after the review of `43142d7`)

| item | defect | repair |
|---|---|---|
| stale interface | a focused test still unpacked the five-value `train_step`, which now returns telemetry as a sixth | all callers updated; every caller audited |
| summary reader | `recall_summary` assumed the old `rho` list schema and would fail on new artifacts | reads **both** schemas; old JSON is never rewritten; prints per-seed gates, stored-vs-trainable counts and the projection telemetry |
| bound comparisons | tests compared float32 leaves against double-precision bounds at 1e-12, which fails on representation alone | bounds cast to the **leaf's** dtype, dtype asserted, float32 coverage added to the production probe |
| weak regressions | `n_projected >= 0` and `max_overshoot >= 0` pass when nothing happens | a **deterministic** outward start forces a crossing: the event count must equal the leaf size and the overshoot must be positive; then a controlled inward objective, **carrying optimizer state forward**, must move the leaf **strictly** off the boundary |
| gate test | searched source text for field names | replaced by executable cases against `evaluate_gate`: both-pass, frequency-only failure, impulse-only failure, non-finite, zero-reference pass and zero-reference fail, missing record |

`evaluate_gate` is now a pure function, so the decision is exercised directly
rather than inferred from the runner.

## Cluster verification plan, when a run is authorized

Same launcher, same 1,200 s cap, same seeds, data streams, equation and
hyperparameters. Nothing is tuned.

1. Focused checks, including the new projection regressions. A failure stops the
   batch, as before.
2. Gate before every seed; both criteria; report per-seed impulse and frequency
   figures rather than one seed's.
3. The same six arms, same paired continuation.
4. Report, per arm and seed: accuracies by delay; raw `log_response_rho_only`
   values; executed `rho`; boundary occupancy; projection-event counts and max
   overshoot; `rho` gradient and update norms; stored and trainable counts.

**The outcome is to be reported whichever way it goes.** The projection defect
explains why the previous run's `rho` direction cannot be interpreted; it is
**not** a reason to expect different accuracies. If the corrected run reproduces
the same ordering, that is the result.

## What cannot be recovered

The completed run wrote only clipped summaries — no parameter trees, optimizer
states or raw trajectories. Its raw overshoot is therefore **unrecoverable**,
and no retrospective diagnosis of that run will be offered. Saving those
artifacts is part of the repair, not a claim about the past.
