# Recover the interrupted adaptive P comparison

This is continuation of job 77021, not a new model or hyperparameter trial.
The candidate, frozen control, seed pairing, saved BatchNorm protocol, original
15-epoch learning-rate schedule, and validation selection rule are retained.
The separate BatchNorm diagnostic is not substituted into the training scores.

## Evidence for isolation

The user supplied these accounting records from pgi14-gpu1 on 2026-11-18:

| Interrupted job | Exit | End | Other completed job | End |
|---|---|---|---|---|
| 76866, original history | 0:9 | 01:38:17 | 76867, seed-303 training | 01:38:17 |
| 77021, adaptive P | 0:9 | 19:13:04 | 77044, frozen BN diagnostic | 19:13:04 |

Both coincidences are exact to the timestamp precision shown. This motivates
isolating the recovery from other job completions. It does not identify the
sender of SIGKILL, prove an epilog bug, or establish an out-of-memory event.
MaxRSS is accounting telemetry, not a complete exclusion of every memory issue.

For a cluster administrator, the concrete request is to inspect slurmctld,
slurmd/slurmstepd, job epilog, cgroup and kernel records around the two times,
particularly why the batch process of a still-running allocation received
SIGKILL when another allocation completed. No administrator message is sent by
this repository.

Slurm documents plain --exclusive as node exclusivity, allocating all CPUs and
GRES, with only the requested memory. Partition configuration can override it:
https://slurm.schedmd.com/sbatch.html#OPT_exclusive

## Source and continuation

Pinned source commit: 4372acd76c9f53e8d83d4f929d5bbd4745ace3ad.

Source root:

    /Users/durso/s5-runs/s5-adaptive-proximal-history/job-77021

Tasks are adaptive_proximal_history_s5 and adaptive_proximal_history_control_s5,
each at seeds 301 and 302. All four have reported validation through epoch 3
and checkpoint_epoch_03.msgpack files. File presence alone is not validation:
the recovery checks provenance, complete state and reproduced validation on GPU.

Continuation restores parameters (including log_horizon_scale), optimizer
moments/counts/rates, BatchNorm statistics and TrainState.step. Epoch 3 implies
3 * floor(30769 / 16) = 5769 completed updates. Partial epoch-4 updates are
discarded because they are not contained in the epoch-end checkpoint. Epoch 4
is replayed with the original deterministic batch order and dropout keys.

Each task first reproduces saved epoch-3 validation and executes two scheduled
training updates as a disposable check. All four checks must pass before any
task continues training. Continuation restores the checkpoint again after its
checks, because checking an update can mutate learning-rate state. GPU memory
and arithmetic for the restored states are measured by execution, not inferred
from successful CPU tests.

Source epochs 1-3, their checkpoints, and the earlier best-checkpoint record are
carried into the new task directories. Selection includes all epochs 1-15.
Original source artifacts are hashed and remain read-only. Timing in recovery
results covers only resumed training; it must not be reported as total time
for a fresh 15-epoch run. A repeated epoch-3 pole record marked at=resume is a
continuation boundary record, not a repeated validation epoch.

The existing three-state model, custom adjoint, ordinary training runner and
optimizer implementation are not changed. The original 384 trainable horizon
parameters remain trainable only in the adaptive arm; the control keeps zero
horizon parameters frozen.

## Cluster execution

Submission entry point:

    bin/run_experiments/submit_s5_adaptive_history_resume.sh

The user runs this on a head node. Preparing or pushing code does not submit a
cluster job. The script requests pgi14/pgi14-gpu1, plain --exclusive, all eight
RTX 4090 GPUs, 32 CPUs, 160 GiB memory and 10 hours. Four workers use four GPUs;
the other GPUs remain reserved to prevent another GPU allocation from ending
on this node during continuation. The wrapper verifies scheduler isolation
metadata and fails before training if exclusivity is not demonstrated.

All work uses a fresh node-local checkout and a new output root:

    /Users/durso/s5-runs/s5-adaptive-history-resume/job-<id>

The post-training adaptive pole/impulse audit runs inside the same allocation,
after all four continuations. No separate diagnostic job is launched during
recovery. Audit failure preserves completed training results and marks the
audit separately; it does not authorize retraining successful tasks.

No dataset, checkpoint, environment, or original run is modified. Only the
training and validation splits are read. Exclusive scheduling mitigates the
observed co-location pattern; it is not a guarantee against unrelated node or
scheduler failures.
