# TSS-based prospective pilot: status

Protocol: `docs/TSS_PILOT_PROTOCOL.md`.

> **No result is reported here, because nothing has been executed.** The
> implementation and the protocol are committed; the focused checks are written
> and are cluster-only. This file records results only after a run.

## Execution status

| item | value |
|---|---|
| preparation parent | `cce550b5cc16d5c5d20857359f1ad8a30a4f7741` |
| branch | `tss-pilot` (prepared in a separate worktree) |
| cluster runs dispatched | **none** |
| focused checks executed | **none** |
| budget consumed | **0 s** |
| artifacts | none created |

No earlier run directory, branch or report was touched. The learned-timescale
study and the deferred combination implementation are unchanged.

## What the two parts will answer, and what they will not

**Part A** holds the optimizer fixed (exact BPTT through each arm's own
discrete updates) and varies the temporal forward law. It can support a claim
about forward-model usefulness. It says nothing about credit assignment.

**Part B** holds the forward model, its parameters, the data and the loss
fixed, and varies only how error is propagated in time. It is the part that
could support a credit-assignment claim. Exact agreement between BPTT and
forward sensitivities in it is a correctness check, not an advantage.

Beating the memoryless cancellation control is **insufficient**: it is expected
to fail a balanced history-only query. The memory-capable TSS comparator is the
one that matters, and it may equal or beat our candidate.

## The equivalence is a check, not a competitor

Our law and the finite-adaptation realization are the same family under the map
`M = tau_m eps`, `gamma + T = tau_m + eps`, `T = eps + tau_p`. Two facts follow
and are enforced in code:

* our declared circuit point has an **exact adaptation twin**
  (`tau_m = 12, eps = 4, tau_p = 4`), used as an implementation check on both
  the forward trajectory and its gradients;
* TSS's own matching prescription `tau_p = tau_m` maps to **`gamma = 0`**, so
  `M <= gamma T` fails for any `M > 0`.

So arms 2 and 4 are two declared **coefficient sectors of one family** — not
equivalent realizations of each other, and not different model classes. Whether
the circuit inequality that excludes the TSS-matched sector is a *meaningful
constraint* or merely a restriction is not settled by this pilot, and the
protocol says so.

## After execution

Record: checkout, branch, full commit, environment, exact commands, the task
structure and leakage probe, per-seed scores and paired differences for Part A
with recall reported by delay, the filter audit and per-bandwidth gradient
metrics for Part B, timing, and every declared limitation in
`docs/TSS_PILOT_PROTOCOL.md` s8. An unfavourable or inconclusive outcome is a
result and is reported as it comes out; a completed failed comparison is not a
software failure.
