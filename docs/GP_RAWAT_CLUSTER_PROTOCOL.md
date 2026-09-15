# Cluster protocol: fixed-coefficient GP versus Rawat alpha-P-S5

**Committed before any score-driven selection.** Governed by
`docs/handoff_2026_09_15/DERIVATION_CONTRACT.md` and
`CLUSTER_CODING_BRIEF.md`. Coefficient policy is RESOLVED and is not revisited
here: the added physical coefficients are fixed declared hyperparameters;
ordinary S5 parameters, including the native learned steps, train with full
BPTT.

Branch `cluster-gp-rawat`, from `a4c5b12599c247923ee0425c25a09d92e8556ce3`.

## 0. Execution rules

* **GPU only.** `experiments/gp/rawat_benchmark.py` refuses to run on a CPU
  backend unless `--allow_cpu` is passed, and that flag is documented as
  structural-smoke-only whose output may never be reported as evidence.
* No local training. Nothing in this protocol was executed locally.
* `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, highest matmul precision, no
  preallocation, offline logging — set in `bin/run_experiments/cluster_env.sh`
  and recorded per run.
* Historical smokes and the September-14 cue/recall protocol are untouched.
  That protocol remains **predeclared and unexecuted**; this batch does not run
  it. Only a short integration check of its runner is used.

## 1. Arms

All five share ONE substrate (`s5/rawat_s5.py`) and have **identical trainable
parameter counts** (35,050 at depth 4, width 32), verified by test.

| arm | input gain | pole clipping | recurrent response | role |
|---|---|---|---|---|
| `native_s5` | none (`B_tilde`) | off | one tap | published reference recipe |
| `alpha_p_s5` | `diag(alpha) B_tilde` | on | two tap | reproduced published baseline |
| `gain_clip_s5` | `diag(alpha) B_tilde` | on | one tap | **attribution control** |
| `gp_fixed_m0` | `diag(alpha) B_tilde` | on | M = 0 | reduced-model ablation |
| `gp_fixed_mass` | `diag(alpha) B_tilde` | on | M = 3.75 | **primary physical candidate** |

`gain_clip_s5` exists because the paper states its own Table 6 does not isolate
the second tap from the input-gain and clipping changes.

`gp_fixed_m0` is an explicitly labelled **reduced/constitutive ablation**, not
the fast-dendrite circuit limit (which sends gamma and M to zero together).

Fixed coefficients on both GP arms: `T = 5I`, normalized `gamma = 1`,
`M = 0` or normalized `M = 3.75I` (`rho = 0.75`). **Not swept, not optimized,
not learned.**

## 2. Task and model

Speech Commands v0.02, 10-word subset, MFCC, stratified 70/15/15 split with
split seed 0, feature-wise standardization from the training split. Depth 4,
width 32, eight HiPPO blocks, conjugate symmetry, ZOH, unidirectional, batch
pre-normalization, half-GLU, readout 64 -> mean-pool -> 64->256->10 GELU MLP
with dropout 0.1. Full detail and every declared difference:
`docs/RAWAT_BASELINE_MAP.md`.

## 3. Stage 1 — integration and cost (no accuracy claim)

One initialization plus two updates per arm, then a throughput and peak-memory
measurement, recording compile-plus-first-step and steady-state step times
separately. `bin/run_experiments/cluster_stage1.sh`. This establishes
execution, not superiority, and its numbers are labelled accordingly.

## 4. Stage 2 — bounded validation-only development batch

`bin/run_experiments/cluster_stage2.sh`.

* development seed **100**; at most **two** candidate configurations per
  candidate family; at most **10 epochs** each; **validation only**.
* candidate families and their learning rates: `gain_clip_s5`, `alpha_p_s5`,
  `gp_fixed_m0`, `gp_fixed_mass`, each at **1e-3 and 3e-4**, same schedule
  definition. `native_s5` gets **one unswept reference configuration** at 1e-3.
* every other optimizer setting follows the published recipe identically on
  every arm.
* **Hard cap: two GPU-hours total**, including compile and validation, checked
  between runs against a wall clock. If the cap is reached, remaining runs are
  recorded as `not_started_budget` — **the currently losing arm is never the
  one stopped**, because the order is fixed in advance and the manifest records
  what did not start.
* If the estimate from stage 1 says the batch will exceed the cap, the epoch
  budget is reduced **for all families equally** and that decision is recorded
  before development.

**Selection order, fixed now:** validation accuracy, then validation cross
entropy, then the declared candidate order above. Test data is untouched; the
runner refuses `--split test` without `--confirm_test`.

## 5. Stage 3 — confirmation, only if stage 2 warrants it

Primary physical candidate is **`gp_fixed_mass`**. If it improves on
validation, freeze its setting and compare against `native_s5`, `alpha_p_s5`
and `gain_clip_s5` on fresh paired seeds **0, 1, 2** with the source's full
training and early-stopping schedule. Maximum **eight additional GPU-hours**.

`gp_fixed_m0` succeeding on its own triggers a **separately labelled
reduced-model confirmation**; it never replaces the physical candidate's
recorded result.

Each selected checkpoint is scored on test **once**, after every training and
configuration choice is fixed.

If the balanced comparison does not fit the cap, runs are kept resumable and
reported as **incomplete**. A shortened run is not called a full reproduction.

## 6. Predeclared positive outcome

Same-task **mean accuracy improvement over both** the reproduced `alpha_p_s5`
**and** the matched `gain_clip_s5` control, with a **positive paired difference
in every confirmation seed**, and parameter, state and runtime costs disclosed.
Practical screening target: **at least 0.3 percentage points** in mean
accuracy.

This is a **screening target, not statistical significance**. Three seeds are
preliminary and differ from the paper's five-seed aggregate.

If only faster learning or only some objectives improve, that narrower result
is what gets reported. No objective and no arm is dropped from the report.

## 7. Costs that must be disclosed with any result

Measured locally as a structural indication only (CPU, synthetic data, not
evidence): `gp_fixed_mass` costs roughly **2.6x the per-step time** of the
one-tap arms, and doubles the recurrent carry.

| arm | trainable params | physical real state | auxiliary real state | previous-input buffer |
|---|---|---|---|---|
| `native_s5` | 35,050 | 128 | 0 | 0 |
| `alpha_p_s5` | 35,050 | 128 | 0 | 128 |
| `gain_clip_s5` | 35,050 | 128 | 0 | 0 |
| `gp_fixed_m0` | 35,050 | 128 | 0 | 0 |
| `gp_fixed_mass` | 35,050 | 128 | **128** | 0 |

(depth 4, width 32; 32 physical real coordinates per layer.)

**A finite-mass win would not by itself establish a benefit beyond added
dynamic capacity.** An ordinary SSM with the same internal state count is
required for that claim; it has a different trainable parameter count, which
must be reported. That comparator is a **subsequent bounded study**, designed
and recorded before its outcomes. The initial claim is restricted to
same-external-width, equal-added-parameter comparison, and parameter count
alone is not asserted to establish equal representation or compute.

## 8. Reporting rules

* Historical local positives, the unsuccessful fixed-coefficient confirmation,
  numerical verification, GPU integration, validation screening and final
  benchmark comparison are reported as **separate evidence classes**.
* A failed predeclared criterion is reported, not loosened.
* No claim of execution without a run identity and artifact path.
* If the full physical candidate loses while `M = 0` wins, only the
  reduced-model outcome is claimed.

## 9. Verified cluster environment

Recorded from `cluster_status.sh` on 15 September 2026, before any run:

| item | value |
|---|---|
| host | `pgi15-gpu3.iff.kfa-juelich.de` |
| SLURM job | 65870 |
| visible GPUs | `0` (single-device allocation) |
| JAX backend | **gpu**, `CudaDevice(id=0)` |
| jax / flax / optax / scipy | 0.11.0 / 0.12.8 / 0.2.8 / 1.18.1 |
| torch / torchaudio | 2.14.0+cpu / 2.11.0+cpu (MFCC extraction only) |
| interpreter | `/Local/durso/prospective_ssm_project/.venv/bin/python` |

**Version drift from the development machine is recorded, not corrected:**
local jax 0.11.1 and flax 0.12.9 versus cluster jax 0.11.0 and flax 0.12.8.
The shared environment is used as-is. Every run's `provenance()` stores the
versions it actually ran under.

**Single GPU is the declared configuration.** At 35,050 parameters, width 32,
sequence 161 and batch 32, full-BPTT activations are on the order of megabytes;
the run is launch-overhead bound, not memory or FLOP bound. Sharding batch 32
across devices would also change per-device batch-normalization statistics,
which are part of the published recipe. If several GPUs are free, the supported
use is **one whole run per GPU** (different arm or seed), which changes no
recipe — noting that the GPU-hour caps in s4 and s5 are summed across devices.

## 10. Cluster commands

Short, one line each. Run in order. Nothing below has been executed.

```bash
git -C /Local/durso/final-prospective-s5 fetch origin && git -C /Local/durso/final-prospective-s5 checkout cluster-gp-rawat && git -C /Local/durso/final-prospective-s5 pull
```
```bash
bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_status.sh
```
```bash
bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_fetch_data.sh
```
```bash
DATA_ROOT=/Local/durso/speech_commands_v0.02 bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_prepare_data.sh
```
```bash
bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_gpu_checks.sh
```
```bash
bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_stage1.sh
```
```bash
nohup bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_stage2.sh > /Users/durso/s5-runs/stage2.out 2>&1 &
```

Resume or status at any time:

```bash
bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_status.sh
```

Re-issuing a training command with the **same `--run_dir`** continues from the
last checkpoint; `cluster_stage2.sh` does this automatically per run.

Paths default to the handoff's values and are **verified, not assumed** —
`cluster_env.sh` exits non-zero if the interpreter, repository or run directory
is missing. Override with `PROSPECTIVE_VENV`, `PROSPECTIVE_REPO`,
`PROSPECTIVE_RUNS`, `PROSPECTIVE_DATA`. The scripts never install, upgrade or
replace anything in the shared environment.
