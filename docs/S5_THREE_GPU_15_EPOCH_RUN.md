# Three-GPU, 15-epoch S5 run inside one allocation

19 September 2026. Branch `s5-three-arm-discrete-prospective`, on top of
`831bd49`. Execution topology and training horizon only. No recurrence, no
dataset, no split, no architecture, no optimizer setting, no batch size and
no seed is changed.

Arms and reporting order, unchanged:

1. Native S5
2. Zucchet prospective dynamics — finite-difference realization
3. generalized prospective dynamics (M,γ,T) — finite-difference realization

## 1. Why this topology

The diagnosis (see `S5_EXECUTION_TOPOLOGY_DIAGNOSIS.md` §5) ended with:
concurrent execution of two experimental **Slurm jobs** on one node is the
implicated condition — array membership and GPU sharing were both excluded,
and the generalized arm passed the identical full path alone. What every
failing configuration had in common was a *sibling Slurm job or array element
reaching a terminal state* while another task trained.

This run removes that entirely. One allocation is held (job `66760` on
`pgi15-gpu2`, four RTX 3090s), and every task is a **direct child process of
the launcher's shell**. There is no `sbatch`, no `salloc`, no `srun`, so no
child job or step exists that could terminate while another task trains. The
allocation itself ends only after the launcher returns.

## 2. The horizon: 15 epochs, explicitly

The 12-hour allocation cannot hold nine 40-epoch trainings. The runner
therefore takes `--epochs`, **defaulting to the existing 40** so no previous
command changes meaning, and the new launcher passes exactly 15.

The schedule is recomputed from the requested horizon: one warm-up epoch
(`WARMUP_END = 1`, linear) and cosine decay over the remaining epochs —
14 here, 39 at the default. Nothing else about the optimizer moves: `lr`
0.008, `ssm_lr` 0.002, `weight_decay` 0.04, `lr_final` 1e-6, batch 16.
Validation and a checkpoint still happen after **every** epoch, and the best
validation checkpoint is still selected by accuracy then cross-entropy.

`epochs_requested` is recorded in `task_result.json`, `production_check.json`,
`smoke_result.json`, `failure.json` and the run's `run_metadata.txt`; the
finalizer reads it per row for its undertrained heuristic and reports the set
of horizons it saw in `results.json`.

## 3. The run

`bin/run_experiments/allocation_s5_three_arm_three_gpu.sh`, executed inside
the allocation. Guards first: exact commit, clean worktree, executable
interpreter, existing data cache, numeric `SLURM_JOB_ID`, and at least four
visible GPU tokens, of which the first three are used and the fourth is left
unused. Each child gets its own task directory, `TMPDIR`, JAX compilation
cache, log pair and a `CUDA_VISIBLE_DEVICES` holding exactly one token.

**Stage 1, mandatory concurrent topology smoke.** Native S5 seed 301 on token
0, generalized seed 301 on token 1, and (by default) a second Native smoke,
seed 302, on token 2 — all concurrent, each preceded by the pre-JAX
telemetry. The gate is the artifacts, not the exit status:
`topology_report.py --require-pass --require-distinct-gpus` demands
`SMOKE_PASS`, two training steps, a restored checkpoint and no `failure.json`
for every child, and a **resolved** process-GPU binding whose uuid no other
child shares. If it fails, the launcher aborts and full training never
starts.

**Stage 2, three sequential waves**, in scientific order, each with seeds
301, 302 and 303 concurrent on tokens 0, 1 and 2. Every child of a wave is
awaited before the next wave begins.

| wave | arm | failure policy |
|---|---|---|
| 1 | Native S5 | any missing `task_result.json` is **unexpected**: abort |
| 2 | Zucchet FD | `failure.json` is the **declared negative control**: recorded, run continues |
| 3 | generalized FD | any missing `task_result.json` is **unexpected**: abort |

An abort preserves every artifact and skips the finalizer.

**Stage 3, the finalizer**, run once, directly, after every arm and seed has
reached its expected terminal state. It remains the **only** reader of the
test split, and it opens it once, after checkpoint selection.

## 4. What is unchanged

`s5/ssm.py`, `s5/discrete_recurrence.py`, `s5/prospective_ssm.py`,
`s5/generalized_prospective_ssm.py`, `s5/three_arm_factory.py` and
`experiments/s5_three_arm_full/data.py` are byte-identical to `831bd49`. In
the runner, `SharedS5Config`, `ARM_CONFIGS`, `SCIENTIFIC_NAMES` and the seeds
are textually identical to that commit; the only change is the `--epochs`
horizon and the `epochs_requested` provenance field. The serialized `sbatch`
production path (`--array=0-8%1`, its two guards and the
submitting-allocation dependency) is untouched and remains available.
