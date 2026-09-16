# Nested associative-memory study: status

Protocol: `docs/NESTED_MEMORY_PROTOCOL.md`. Branch `nested-prospective-memory`.

> **No result is reported here, because nothing has been executed.** The
> implementation, the frozen protocol and the focused checks are committed; the
> checks are cluster-only and have not been run. This file records results only
> after a run.

## Execution status

| item | value |
|---|---|
| parent | `62c076739a9afa1624faec68961e7e500d6f1ed8` (`tss-pilot`) |
| branch | `nested-prospective-memory`, separate worktree |
| cluster runs dispatched | **none** |
| focused checks executed | **none** (written, cluster-only) |
| budget consumed | **0 s** |
| artifacts | none created |

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
