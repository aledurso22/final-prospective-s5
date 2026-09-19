# S5 three-arm execution-topology diagnosis

19 September 2026. Branch `s5-three-arm-discrete-prospective`, from
`c8f91bf`. **Infrastructure only.** No recurrence, dataset, split,
architecture, optimizer, schedule, seed or protocol is changed, and nothing
is launched here.

Scientific arms and reporting order, unchanged:

1. Native S5
2. Zucchet prospective dynamics — finite-difference realization
3. generalized prospective dynamics (M,γ,T) — finite-difference realization

## 1. What has to be explained

Job 66583 (`/Users/durso/s5-runs/s5-three-arm-full-smoke/20260919-030902`)
ran two sibling tasks of one array on `pgi15-gpu3`:

- task 0, Native S5, seed 301: `SMOKE_PASS` in 1:13, two training steps,
  validation, checkpoint saved and restored, peak GPU ≈ 7.42 GB, MaxRSS
  ≈ 2.04 GB;
- task 6, generalized prospective dynamics, seed 301: finite production
  update (loss ≈ 2.49438, gradient norm ≈ 1.54456, finite gradients and
  state), peak GPU ≈ 8.96 GB, MaxRSS ≈ 1.53 GB, then **SIGKILL `0:9` at
  exactly 1:13**, the instant the sibling finished. Empty stderr. It never
  reached the two smoke steps.

An empty stderr with `0:9` at a sibling's exit instant is an external kill,
not a Python exception and not a JAX abort. Both tasks reported logical
`CUDA_VISIBLE_DEVICES=0`, which does not establish that they shared a
physical device, because Slurm renumbers an assigned GPU to logical 0.

## 2. Audit of in-repository kill paths

Read at `c8f91bf`: `bin/slurm/s5_three_arm_full_array.sbatch`,
`bin/slurm/s5_three_arm_full_finalize.sbatch`,
`bin/slurm/s5_three_arm_preflight.sbatch`,
`bin/run_experiments/cluster_s5_three_arm_full.sh`,
`bin/run_experiments/cluster_env.sh`,
`experiments/s5_three_arm_full/{runner,finalize,preflight,data}.py`.

| Candidate | Found? | Note |
|---|---|---|
| `trap` handlers | **none** | no script installs one |
| `kill`, `pkill`, `killall`, negative-PID signals | **none** | no process-group signalling anywhere |
| Python `signal` handlers, `atexit`, `os.kill` | **none** | the runner installs nothing |
| Temporary-directory cleanup | **none** | `TMPDIR` is per task, `$TASK_ROOT/tmp`, never deleted |
| Shared cache eviction | **none** | `JAX_COMPILATION_CACHE_DIR` is per task, `$TASK_ROOT/jax-cache` |
| Shared output path collisions | **none** | artifacts are per `arm/seed` |
| `subprocess` use | one | `preflight.py` spawns a child of itself and waits; not on the array path |
| Node-level pressure | not excluded | `--mem=50G` per task with `%4` concurrency; MaxRSS was small, but the node limit was never recorded |

**Nothing in this repository can terminate a sibling task.** The remaining
candidates are site-level and outside the repository: the job-array cgroup
and epilog behaviour when one member exits, the node OOM killer, or a shared
physical GPU. The telemetry below distinguishes them; the independent
submission removes the first.

## 3. Change 1: pre-JAX execution telemetry

`experiments/s5_three_arm_full/gpu_telemetry.py` (new; standard library
only, no JAX import) records, **before** the runner starts:

- `SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`,
  `SLURM_JOB_GPUS`, `SLURM_STEP_GPUS`, and the node, partition, memory and
  CPU allocation;
- `CUDA_VISIBLE_DEVICES` and `GPU_DEVICE_ORDINAL`;
- the **physical** GPU `uuid` and `pci.bus_id` from `nvidia-smi`, plus name,
  total and used memory and compute mode;
- pid, ppid, **pgid**, hostname, UTC timestamp;
- `/proc/meminfo` and the cgroup memory limit, for the node-pressure
  hypothesis.

It is invoked from both `s5_three_arm_full_array.sbatch` and the new
`s5_three_arm_one_task.sbatch` before the runner, writing
`$TASK_ROOT/gpu_telemetry.json`. Because it runs pre-JAX in its own process,
the record survives a later external kill.

The runner attaches the same identity to its artifacts: every
`lifecycle.jsonl` line now carries `slurm`, the first line of each process
carries the full `pre_jax_telemetry` and the extracted `physical_gpus`, and
`production_check.json`, `smoke_result.json` and `failure.json` carry
`slurm` (and `physical_gpus` for the smoke result).

**Two tasks shared a physical GPU iff their `gpu_telemetry.json` records show
the same `uuid`.** That question is now answerable from the artifacts.

## 4. Change 2: independent concurrent smoke

`bin/slurm/s5_three_arm_one_task.sbatch` (new) runs one arm and one seed as
an ordinary job: no `--array`, no `SLURM_ARRAY_TASK_ID`, arm and seed from
the environment and validated against the declared sets, the same commit and
clean-worktree guards, the same per-task `TMPDIR` and JAX cache.

`bin/run_experiments/cluster_s5_three_arm_independent_smoke.sh` (new)
submits **two separate `sbatch` jobs with different base job ids** — Native
S5 seed 301 and generalized prospective dynamics seed 301 — each with
`--smoke`, so each exercises the production check, two real training
updates, validation, checkpoint save, checkpoint reload and lifecycle
telemetry. `DRY_RUN=1` prints the exact submissions without contacting
Slurm.

## 5. How the result is read

A pass requires, for **both** jobs, from the JSON artifacts:

- `smoke_result.json` with `status: SMOKE_PASS`, `training_steps: 2`,
  `checkpoint_restored: true`;
- no `failure.json`;
- `lifecycle.jsonl` reaching `after_smoke_training_steps`.

Slurm `COMPLETED` alone is **not** a pass. The launcher prints the exact
inspection commands, including the physical-GPU comparison.

- **Both pass concurrently** → the array topology is implicated. The
  production launcher then submits nine independent one-GPU jobs and makes
  the finalizer depend `afterany` on all nine job ids.
- **Interference persists** → the node is implicated. The production
  launcher then serializes (`%1`).

Neither production change is made yet: the decision waits for the artifacts.
