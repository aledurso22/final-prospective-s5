# S5 three-arm execution-topology diagnosis

19 September 2026. Branch `s5-three-arm-discrete-prospective`, from
`c8f91bf`. **Infrastructure only.** No recurrence, dataset, split,
architecture, optimizer, schedule, seed or protocol is changed, and nothing
is launched here. The diagnostic artifacts are *not* untouched: their schema
is intentionally extended, and the new fields are listed in §3.

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

**No explicit in-repository kill path was found** in the files listed above:
no signal is sent, no handler is installed, no shared resource is reclaimed.
An external signal is therefore the likely proximate cause. That is not the
same as excluding this repository: a workload-triggered kernel, driver,
scheduler or cgroup response — the OOM killer, a GPU fault, a step or job
cgroup limit, a device-reset path — remains possible, and such a response is
provoked by what the workload does. The audit narrows the search; it does not
close it.

## 3. Change 1: pre-JAX execution telemetry

`experiments/s5_three_arm_full/gpu_telemetry.py` (new; standard library
only, no JAX import) records, **before** the runner starts:

- `SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`,
  `SLURM_JOB_GPUS`, `SLURM_STEP_GPUS`, and the node, partition, memory and
  CPU allocation;
- `CUDA_VISIBLE_DEVICES` and `GPU_DEVICE_ORDINAL`;
- a **node/NVML GPU inventory** from `nvidia-smi --query-gpu`, under
  `gpu_inventory` with its scope written into the record. This query is *not*
  filtered to the assigned device and is *not* evidence about which GPU any
  process used; it is background about the node;
- pid, ppid, **pgid**, hostname, UTC timestamp — of the telemetry process,
  which exits before the runner starts and therefore does **not** identify
  the runner's pid. The record says so in `pid_scope`;
- the memory limit of **this process's own cgroup**: the path is read from
  `/proc/self/cgroup`, joined to the hierarchy mounted in `/proc/mounts`
  (v2 `memory.max`/`memory.current`/`memory.peak`, v1
  `memory.limit_in_bytes`/`memory.usage_in_bytes`/`memory.max_usage_in_bytes`),
  taking the nearest ancestor that publishes a limit and recording which path
  it came from. A value found only at the hierarchy root is reported as
  `hierarchy_root` with `is_job_specific: false`, never as the job's limit;
- `/proc/meminfo` separately, as node-wide memory.

It is invoked from both `s5_three_arm_full_array.sbatch` and the new
`s5_three_arm_one_task.sbatch` before the runner, writing
`$TASK_ROOT/gpu_telemetry.json`. Because it runs pre-JAX in its own process,
the record survives a later external kill.

### 3b. The process-specific GPU binding

The pre-JAX record cannot say which GPU the runner used, for two reasons:
its pid is the telemetry process's, and its GPU list is an inventory. The
binding is therefore made separately, in the runner:

after JAX/CUDA has initialized and the production update has completed — so
the runner is an active compute application — and **before** the two smoke
training steps begin — so the artifact survives a kill during them —
`bind_process_gpu` queries
`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name`,
matches the rows against the runner's own `os.getpid()`, and writes
`$TASK_ROOT/gpu_process_binding.json` with one of three statuses:

| status | meaning |
|---|---|
| `resolved` | exactly one GPU uuid for this pid; that uuid is the device |
| `multiple` | the pid maps to more than one uuid; every uuid is recorded |
| `unresolved` | the query failed, or the pid is absent from its output; the failure or the full row set is recorded |

No identity is ever inferred from the inventory when the binding does not
resolve. Reasons it may not: the query is unavailable or permission-denied,
the pid namespace differs from the one NVML reports, or the process is not
holding a context at the moment of the query.

Artifact schema changes: every `lifecycle.jsonl` line carries `slurm`; the
first line of each process carries `pre_jax_telemetry`; every line after the
binding carries a `process_gpu` summary; `production_check.json` and
`failure.json` carry `slurm` (and `failure.json` the `process_gpu` summary);
`smoke_result.json` carries `slurm`, `process_gpu`, and, kept separate and
labelled, `gpu_inventory_uuids`.

**Two tasks shared a physical GPU if and only if both `gpu_process_binding.json`
records are `resolved` and carry the same `gpu_uuid`. If either is not
resolved, the shared-GPU question is inconclusive** — and
`experiments/s5_three_arm_full/topology_report.py` prints exactly that.

## 4. Change 2: independent concurrent smoke

`bin/slurm/s5_three_arm_one_task.sbatch` (new) runs one arm and one seed as
an ordinary job: no `--array`, no `SLURM_ARRAY_TASK_ID`, arm and seed from
the environment and validated against the declared sets, the same commit and
clean-worktree guards, the same per-task `TMPDIR` and JAX cache.

`bin/run_experiments/cluster_s5_three_arm_independent_smoke.sh` (new)
submits **two separate `sbatch` jobs**, normalizes the `--parsable` output
(`<jobid>` or `<jobid>;<cluster>`) and **fails, without printing an
inspection recipe, unless the two submissions carry two distinct base job
ids** — the topology under test does not exist otherwise. The arms are Native
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
inspection command, `topology_report.py`, which reports the pass from the
smoke artifacts and the shared-GPU question from the resolved process
bindings only, or `inconclusive`.

- **Both pass concurrently** → the array topology is implicated as the
  setting in which the kill occurs. It is not thereby proven to be the unique
  cause: passing independent jobs are consistent with an array-specific
  mechanism and also with an intermittent or load-dependent one. The
  production launcher then submits nine independent one-GPU jobs and makes
  the finalizer depend `afterany` on all nine job ids.
- **Interference persists** → the node is implicated. The production
  launcher then serializes (`%1`).

Neither production change is made yet: the decision waits for the artifacts.
