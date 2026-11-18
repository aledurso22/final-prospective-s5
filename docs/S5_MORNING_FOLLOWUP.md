# History recovery and paired replication

The completed proximal-history pilot at `a4c12b2` has a selected-validation
mean of 96.9754% on seeds 301/302, versus 96.6379% for native S5 and 96.8809%
for calibrated prediction correction (D). These exploratory differences do
not establish general superiority. The original history pilot, job 76866,
was killed with signal 9 after saving epoch 9 for both seeds. Its termination
coincided with the completion of job 76867; the cause is unknown.

This follow-up changes no equation or hyperparameter. Its purpose is to finish
the interrupted comparison and add paired random seeds.

| GPU worker | First task | Following task |
|---|---|---|
| 0 | History 301: resume after epoch 9 | Proximal alpha=1, seed 303 |
| 1 | History 302: resume after epoch 9 | Proximal alpha=0.1, seed 302 |
| 2 | Native, seed 304 | — |
| 3 | Calibrated prediction correction D, seed 304 | — |
| 4 | Proximal alpha=1, seed 304 | — |
| 5 | Native, seed 305 | — |
| 6 | Calibrated prediction correction D, seed 305 | — |
| 7 | Proximal alpha=1, seed 305 | — |

One Slurm allocation holds all eight workers. It does not depend on another
training allocation remaining alive. This arrangement avoids the previous
coincident ending of separate allocations, but is not a diagnosis or proven
fix for that unexplained termination.

Every fresh task uses 15 epochs, batch 16, float32 initialization and training,
the existing dataset split, optimizer, warm-up/cosine schedule, and selection
rule. N/D use the original constructors (D uses the factored scan). Proximal
history uses its original exact-ZOH state recurrence, stationary block scan,
and fixed proximal ratio 0.5. GPU smoke checks precede training.

History recovery restores the complete serialized TrainState, including
optimizer and BatchNorm state. The epoch and global step continue at the
epoch-9 boundary, preserving the original 15-epoch schedule, deterministic
batch ordering and dropout keys. Earlier validation epochs remain eligible
for selection. Source results are read-only; continuation writes to a new
directory with provenance. Discarded partial epoch-10 updates are recomputed.

Analysis must keep the comparisons paired:

- N/D/P alpha=1: seeds 301–305 once all corresponding tasks finish.
- H versus proximal alpha=0.1: seeds 301/302, with a matched history horizon.
- Proximal alpha=0.1 versus alpha=1: seeds 301/302, to isolate horizon choice.
- C and A retain their existing three-seed results. Do not compare their
  three-seed means with five-seed means as if all seeds were matched.

Use each task's original selected-validation rule, report failures as
failures, and inspect selected-checkpoint poles. No test split is read and no
new accuracy claim is made by this implementation.

## Verification and execution

The focused local float32 suite passed 51 tests: existing baseline/history
integration, exact checkpoint continuation, fresh N/D/P initialization and
smokes, and the eight-worker launcher's queue/failure behavior. The real tiny
JAX continuation test produced byte-identical epoch checkpoints and identical
validation/step records to uninterrupted training. These CPU checks do not
establish full-size GPU execution; restored-state and fresh GPU gates execute
inside the cluster job before their training tasks.

From a clean checkout pinned to the published recovery commit, submit with
`EXPECTED_COMMIT=<full-sha> bash bin/run_experiments/submit_s5_morning_followup.sh`.
The job requests eight RTX4090 GPUs, 32 CPUs, 200 GB and an eight-hour limit on
`pgi14-gpu1` in partition `pgi14`. It makes its own node-local clone and uses
the existing environment without installing or changing packages.

Output is `/Users/durso/s5-runs/s5-morning-followup/job-<id>/`:

- `history/gradient_history_s5/{301,302}`
- `fresh/native_matched_s5/{304,305}`
- `fresh/calibrated_prediction_s5/{304,305}`
- `fresh/proximal_history_s5/alpha1/{303,304,305}`
- `fresh/proximal_history_s5/alpha0p1/302`

The allocation log is `/Users/durso/s5-runs/s5-morning-followup-<id>.log`.
An unsuccessful history restore does not block the independent proximal task
queued on that worker; the allocation still reports incomplete. A failed
fresh smoke never authorizes its training. No source checkpoint is modified.
Resumed pole records include a duplicate boundary epoch marked `at: resume`;
readers should distinguish this from a newly completed epoch.
