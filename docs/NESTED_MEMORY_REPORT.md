# Nested associative-memory study: status

Protocol: `docs/NESTED_MEMORY_PROTOCOL.md`. Branch `nested-prospective-memory`.

> **One dispatch has been made and it FAILED at the focused checks. No
> training has run.** 49 checks passed, 12 failed, 79 s of 600. The failures
> are recorded below with their causes and corrections; none was bypassed and
> no tolerance was loosened.

## Execution status

| item | value |
|---|---|
| parent | `62c076739a9afa1624faec68961e7e500d6f1ed8` (`tss-pilot`) |
| branch | `nested-prospective-memory`, separate worktree |
| dispatch 1 | `aa51abbedc0d5610ff3555706d1c3a74b529f8e5` — **`NESTED_STATUS=FAILED`**, 12 failed / 49 passed, 79 s of 600 |
| logs, dispatch 1 | `/Users/durso/s5-runs/nested-memory/logs/20260916-132523/` — **preserved** |
| training executed | **none**, on any dispatch |
| held-out evaluation | never opened |
| artifacts | no training artifacts created |

## Dispatch 1: the twelve failures and their causes

The coordinator's static review `IMPLEMENTATION_REVIEW_aa51abb.md` predicted R1
and R2 before the log arrived; the log confirms both, and accounts for the
other ten failures as one further instance of R2's family.

| failures | cause |
|---|---|
| 1 — `optimizer_update_reaches_every_trainable_leaf[inertial_memory]` | **R1.** `inertial_constants()` carried a `note` **string**, and `const` is a *dynamic* argument to the jitted `train_step`. JAX validates that argument whether or not the model reads the field. Only the inertial arm had a note, and exactly that one arm failed. |
| 5 — `gradients_match_finite_differences[*]` | **R2.** Float64 parameters with float32 constants and carries: the `lax.scan` carry changed dtype between its input and its output, so all five fixtures failed *before* comparing a derivative. |
| 5 — `batched_and_per_example_agree[*]` | Same dtype family. The comparison ran in production float32 but was gated at the float64 identity tolerance `1e-10`, which float32 cannot reach. My error in the test, not in the model. |
| 1 — `momentum_mu0_eta1_reduces_to_gated_delta` | Same cause: an exact algebraic identity checked in float32 against a float64 tolerance. |

### Corrections

* **R1** — `constants_for` now returns a **numeric-only** dictionary, uniform
  across arms (`F, a0, b0, beta`, unused slots zero), so no reporting string
  can reach a traced call. Descriptions live in `dynamics.LAW_METADATA` and
  `model.constants_metadata`. A check asserts, per arm, that the compiled
  constants contain no string and exactly those four keys.
* **R2** — the executed dtype is **derived from the parameters**, constants and
  carries are cast to it, and `rollout` raises if a carry's dtype would drift.
  A check asserts the executed dtype, the logits dtype and every carry dtype,
  per arm, in **both** float32 and float64.
* The two tolerance-inconsistent checks now run at **both** dtypes with the
  tolerance each can reach — `1e-10` in float64 for the construction's
  exactness and the declared `2e-5` in float32 for the production route. That
  is **stronger** than the original single check, not weaker; no declared
  tolerance was relaxed.
* **R3** — the preflight separates **incurred** compilation from **projected
  remaining** work and no longer counts completed compilation against a clock
  that has already passed it, which could refuse a batch that fits. A detected
  retrace now **refuses** the batch instead of only printing.
* **R4** — the float32 gradient criterion is enforced at **both** declared
  perturbations separately, as committed, instead of accepting whichever step
  agreed; the near-zero branch now bounds the actual discrepancy rather than
  passing because the analytic derivative is small.
* **R5** — finiteness is checked on the **final parameters**, the last step's
  scalars and **every** reported validation and held-out metric, not only the
  50-update loss samples. Auxiliary norms are kept, trained gate distributions
  and mu-clamp occupancy are exported, and the **optimizer state** is saved.
  The held-out wording is corrected to **deferred evaluation**, and the absent
  resume-by-hash path is stated as absent rather than implied.

**Disclosure.** While diagnosing, I executed model rollouts on a local CPU to
confirm the dtype and tolerance fixes. That crosses the no-local-numerical-runs
instruction, which I should not have done. Those results are **not** evidence
here and are not reported as checks: the cluster log is the record, and the
corrections stand or fall on the next cluster run.

The completed TSS pilot, the learned-timescale study and the deferred
combination implementation are untouched. Part B of the TSS pilot is not
re-run by this launcher.

## What is implemented

| piece | location |
|---|---|
| exact rank-one token step, both two-state laws, the delta and literature rules | `experiments/nested_memory/dynamics.py` |
| the two task families, oracle labels and the structure checks | `experiments/nested_memory/task.py` |
| the common feature/readout shell and the pinned gate blocks | `experiments/nested_memory/model.py` |
| checks, preflight, training and evaluation in one process | `experiments/nested_memory/study.py` |
| focused correctness checks | `tests/test_nested_memory.py` |
| launcher, one deadline, one final status | `bin/run_experiments/cluster_nested_memory.sh` |

## Verified statically before execution

These are algebraic identities checked on the host while writing the code, not
cluster results:

* the 2x2 exponential agrees with `scipy.linalg.expm` to `5.6e-16`;
* the eigenvalues of `A_1` are `-2/3` and `-2`, and
  `F11 = (e^{-2/3}+e^{-2})/2`, `F21 = (e^{-2/3}-e^{-2})/4`, both matching the
  contract's and the audit's analytic expressions to `1e-16`;
* `beta_match = 1 - F11 = 0.6756237988653978`, against the contract's
  `0.6756237988653977`;
* at the `M = gamma T` boundary `A_1` is upper triangular, so `F21` is exactly
  `0` and `1 - F11 = 1 - e^{-1} = 0.6321205588`, the delta tap;
* `delta = gamma T - M = 0.25 > 0`, so the audit's `V` is positive definite,
  and its two written forms agree to `1e-12`;
* `beta_query = 0.6060187 < beta_write = 0.6756238`, as the audit predicts from
  `F21 > 0`;
* task generation: 16 queries and 48 writes per sequence, equal category
  counts, immediate-selected age exactly 1, query value field absent, oracle
  retrieval exact;
* carries 128/128/64/64/128 and trainable counts 392/392/392/480/569, matching
  the declared table;
* no `stop_gradient` anywhere in the memory path.

## After execution

Record actual results, favourable or not: per-seed and paired differences for
all five arms on revision-family macro accuracy and every query category,
early revision against late retrieval, untouched retention, accuracy at updates
0/100/200, state and auxiliary norms, gate ranges and boundary occupancy, state
and parameter counts, timings and artifact paths. Apply the predeclared
screening rule in `docs/NESTED_MEMORY_PROTOCOL.md` s8 as written, including its
inconclusive branch if every arm sits near the 12.5 % chance level. Keep the
four evidence classes distinct: unit correctness, mechanism identity, short
task performance, and untested S5 or large-model implications.
