# Experiment registry

One row per executed run. A run is only citable if it has a commit SHA, an
environment record, hardware, seed, the exact command, an artifact path and an
evidence label. Runs without all seven are not recorded here.

Evidence labels are assigned by the project owner, not inferred from the
result. **E2 = smoke evidence only**: the run demonstrates that the pipeline
executes and produces plausible numbers. It is not a benchmark, not a
comparison, and must not be cited as a performance claim.

| ID | Date | Branch | Commit | Evidence | Summary |
|---|---|---|---|---|---|
| `E2-001` | 2026-09-01 | `main` | `6fcbca7` | E2 (smoke, CPU) | plain S5, 1 epoch sMNIST, CPU host — test acc 0.8995 |
| `E2-002` | 2026-09-02 | `main` | `6fcbca798a93e7b511d8862cae4cce4afa43c34f` | E2 (smoke, GPU) | plain S5, 1 epoch sMNIST, RTX 3090 — test acc 0.8998 |
| `E1-001` | 2026-09-02 | `prospective-lead` | `a93b845bf8760e7693c6ec2ef4e04d17e182ac9e` | E1 (correctness, GPU) | **G2a-core PASSED** — 63/63 exact-identity tests on RTX 3090 |
| `E1-002` | 2026-09-02 | `prospective-lead` | `0316e3c3d102d230ed3660b0b6f5084a66bc01ea` | E1 (correctness, GPU) | **G2a-core re-PASSED post-hardening; F-001 GPU-VERIFIED/CLOSED** — 87/87 |
| `E2-004` | 2026-09-03 | `main` `3c17a9a` + `prospective-lead` `0316e3c` | see below | E1 (correctness, GPU) | **G2b PASSED under deterministic XLA** — all four runs bit-identical |
| `E2-003` | 2026-09-03 | `main` `3c17a9a` + `prospective-lead` `0316e3c` | see below | E2 (diagnostic, GPU) | **G2b = INVESTIGATE** — alpha=0 differs from plain S5 in full training; the training loop is itself nondeterministic |

---

## `E2-002` — plain-S5 GPU baseline smoke

**Evidence label: E2 — smoke evidence only.** This run establishes that plain
S5 executes end to end on the target GPU and produces plausible numbers. It is
explicitly *not* a benchmark and *not* a comparison against anything.

### Provenance

| Field | Value |
|---|---|
| Date | 2026-09-02T15:44:42+02:00 |
| Branch | `main` |
| Commit | `6fcbca798a93e7b511d8862cae4cce4afa43c34f` |
| Working tree | clean (`git status --porcelain` empty) |
| Host | `pgi15-gpu3.iff.kfa-juelich.de` |
| `SLURM_JOB_ID` | `63277` - **VERIFIED** against cluster `sacct` records |
| `SLURM_JOB_NODELIST` | `pgi15-gpu3` |
| Artifact path | `/Users/durso/s5-runs/20260902-154442-main-6fcbca7/` (NFS `pgi-15.iff.kfa-juelich.de:/Users/pgi-15/durso`) |
| Artifacts | `provenance.txt`, `requirements-frozen.txt`, `train.log`, `gpu_mem.csv` |

> **Provenance VERIFIED - this record is frozen.**
>
> `SLURM_JOB_ID=63277` was confirmed against cluster `sacct` records, not
> inferred:
>
> | Evidence | Value |
> |---|---|
> | `provenance.txt` | `SLURM_JOB_ID = 63277`, `SLURM_JOB_NODELIST = pgi15-gpu3` |
> | `sacct` job 63277 | started `2026-09-02 15:05:52`, node `pgi15-gpu3`, RUNNING throughout |
> | provenance capture | `15:44:42` - inside 63277 |
> | training run start | `15:52:07` (wandb dir `offline-run-20260902_155207`) - inside 63277 |
> | job 63249 | separate earlier allocation, `12:02`-`14:20`; ended before both timestamps, **not** associated with this run |
>
> This also resolves the earlier concern that provenance capture and the run
> were separate shell invocations ~7.5 minutes apart: a single allocation
> (63277) demonstrably spans both timestamps, so the capture attests to the
> run's allocation. The concern is closed by evidence, not waived.
>
> Artifact path verified independently: `df -h "$HOME"` reported filesystem
> `pgi-15.iff.kfa-juelich.de:/Users/pgi-15/durso` mounted at `/Users/durso`,
> and the artifacts are owned `durso:pgi-15`. It is a cluster NFS path.
>
> **Standing practice going forward** (process improvement, not a defect in
> this record): capture provenance inside the same invocation as the run, so
> allocation identity never depends on a separate `sacct` reconciliation.

### Hardware

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, compute capability 8.6 |
| Driver | 570.86.10 (CUDA 12.8) |
| `CUDA_VISIBLE_DEVICES` | `0` (single GPU) |
| PyTorch device | CPU-only by design (`torch.cuda.is_available() == False`); JAX is the compute backend |

### Environment

Python 3.12.3, venv `/Local/durso/prospective_ssm_project/.venv` (intentionally
shared with the `prospective_ssm_project` checkout). Authoritative record:
`requirements-frozen.txt` (82 packages) in the artifact directory.

```
jax==0.11.0              flax==0.12.8       torch==2.14.0+cpu
jaxlib==0.11.0           optax==0.2.8       torchvision==0.29.0+cpu
jax-cuda12-pjrt==0.11.0  numpy==2.5.2       torchaudio==2.11.0+cpu
jax-cuda12-plugin==0.11.0 scipy==1.18.1     einops==0.8.2  datasets==5.0.1
```

`python -m pip check`: no broken requirements.

Environment variables set for the run:

```
CUDA_VISIBLE_DEVICES=0
XLA_PYTHON_CLIENT_PREALLOCATE=false     # so peak memory is measurable
WANDB_MODE=offline
```

### Command

```bash
./bin/run_experiments/run_baseline_mnist_smoke.sh
```

`sha256(bin/run_experiments/run_baseline_mnist_smoke.sh) =
55fab5cbcc5095a75e80112c1ac876617fa641652b440decd22400aa6c4ebcbf`

Resolving, with the script's defaults `EPOCHS=1`, `SEED=1919`, to:

```bash
python run_train.py \
  --dataset=mnist-classification --epochs=1 --bsz=64 \
  --n_layers=2 --d_model=64 --ssm_size_base=64 --blocks=2 \
  --batchnorm=False --bidirectional=False --p_dropout=0.0 \
  --jax_seed=1919 --USE_WANDB=False
```

### Result

| Metric | Value |
|---|---|
| Baseline unit tests | 13/13 passed on this GPU |
| Trainable parameters | 26,058 |
| Epochs / seed | 1 / 1919 |
| Train loss | 1.40519 |
| Val loss | 0.36402 |
| Val accuracy | 0.8952 |
| Test loss | 0.33766 |
| **Test accuracy** | **0.8998** |
| Wall clock | 56.431 s (`real`) |
| Peak GPU memory | **837 MiB** (of 24576 MiB available) |
| Peak sampled GPU utilization | 42% |
| Mean GPU utilization | **invalid - not reported** (see below) |

Peak memory of 837 MiB confirms `XLA_PYTHON_CLIENT_PREALLOCATE=false` took
effect; without it the figure would read as roughly 75% of the card and mean
nothing. It also shows this configuration uses ~3.4% of the RTX 3090.

**Mean GPU utilization is not reported.** The `nvidia-smi` sampler was started
before the run and stopped manually afterwards, so it continued sampling after
the training script exited. The post-run zero-utilization samples contaminate
any average. The computed 0.8% mean is an artifact of the measurement window
and is deliberately excluded rather than recorded.

No conclusion about whether this run is GPU-bound or dataloader-bound is drawn
from this measurement. A 1 Hz sampler spanning a ~10 s training phase cannot
support such a conclusion in either direction. Steady-state training can be
profiled separately if it becomes relevant.

Throughput observed in the log: 843 train steps at ~80 it/s; validation ~9 it/s
and test ~20 it/s. The lower eval rates are consistent with one-off XLA
compilation of the evaluation graphs (the val loader uses `drop_last=False`, so
the final partial batch forces an extra shape specialization), but this was not
directly verified.

### Relationship to `E2-001` (CPU)

| Metric | `E2-001` CPU | `E2-002` RTX 3090 |
|---|---|---|
| Trainable parameters | 26,058 | 26,058 |
| Train loss | 1.40866 | 1.40519 |
| Val loss | 0.36526 | 0.36402 |
| Val accuracy | 0.8928 | 0.8952 |
| Test loss | 0.34703 | 0.33766 |
| Test accuracy | 0.8995 | 0.8998 |
| Wall clock | 1m57.32s | 0m56.43s |

The parameter count matches exactly, confirming the configuration did not
drift. The metrics agree to roughly three decimal places but are not identical.

**The cause of the residual numerical difference has not been established.**
Different backends can differ through reduced-precision matmul modes, differing
kernel and reduction orders, or differing fusion decisions, and this run does
not discriminate between them. No attribution is claimed here. A direct test is
described in `docs/GATES.md`; it has not been run.

Two runs on different hardware are in any case not a controlled comparison.
Nothing about the CPU/GPU delta should be read as a result.


---

## `E1-001` — G2a-core alpha=0 exact identity, PASSED on GPU

**Evidence label: E1 — correctness evidence (owner-assigned).** Deterministic
exact-equality tests on the target hardware. Unlike E2 smoke runs this is a
correctness result, not a plausibility check.

### Verdict

**G2a-core PASSED.** The claim

```
F_PC-S5,alpha=0(u; theta) == F_S5(u; theta)
```

holds exactly — `assert_array_equal`, zero tolerance — for finite valid model
tensors on the RTX 3090, under identical parameters, input, masks, RNG keys,
optimizer state and precision.

Notably, exact equality held on GPU. The fusion/rounding divergence I flagged
as a possible INVESTIGATE/BLOCKED outcome did **not** occur.

### Provenance

| Field | Value |
|---|---|
| Branch | `prospective-lead` |
| Commit | `a93b845bf8760e7693c6ec2ef4e04d17e182ac9e` |
| Working tree | clean |
| Host / node | `pgi15-gpu3` |
| `SLURM_JOB_ID` | `63277` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU | NVIDIA GeForce RTX 3090 |
| Result | **63 passed in 107.75s**, `exit=0` |
| Artifact | `g2a.log` (exact directory to be confirmed — see below) |

Provenance for this run was captured **inside the same invocation** as the
test run, per the standing practice adopted after `E2-002`.

> **Artifact path pending.** The run directory was generated as
> `$HOME/s5-runs/<timestamp>-G2a-a93b845/`. The exact timestamp has not been
> supplied and is **not inferred here**. To be filled in from the cluster.

### What passed

All 63 tests, comprising the 44-test prospective suite plus the 19-test
G2a-core suite (`tests/test_g2a_identity.py`) covering the specified checks:
parameter-tree structure/shapes/dtypes, parameter count exactly 26,058,
no alpha parameter allocated at fixed zero, operator and parallel-path
identity, model logits (eager and JIT), fixed-batch loss, gradients for every
ordinary parameter, one optimizer update including optimizer state, streaming
output identity, and the boundary cases (t=0, length-one, reset/restart,
packed segment, stale cache).

### Explicitly still open

`F-001` (G2a-robustness) remains **OPEN**. This PASS is scoped to finite
tensors per the theory specification. Non-finite behaviour is unchanged by
this result. See [FINDINGS.md](FINDINGS.md).


---

## `E1-002` — post-hardening G2a verification, PASSED on GPU

**Evidence label: E1 — correctness evidence.**

### Verdict

- **G2a-core = PASSED** on the hardened implementation.
- **F-001 = GPU-VERIFIED / CLOSED.**

### Provenance

| Field | Value |
|---|---|
| Branch | `prospective-lead` |
| Commit | `0316e3c3d102d230ed3660b0b6f5084a66bc01ea` |
| Hardening commit contained | `d60252f9b08f4ba020a829a38de9a186a69d4332` |
| Host / node | `pgi15-gpu3` |
| `SLURM_JOB_ID` | `63311` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| GPU | NVIDIA GeForce RTX 3090 |
| Result | **87 passed in 111.15s**, `exit=0` |
| Artifact | `g2a.log` — *exact directory pending, not inferred* |

### Coverage

87 tests = the 63 of `E1-001` plus 24 F-001 hardening tests. The static-zero
bypass is confirmed on GPU to: preserve finite exact identity (eager, JIT, with
reset mask); leave Inf/NaN values unchanged **and** the following timestep
clean; treat any stale cache (including NaN/Inf caches) as observationally
irrelevant; and leave `alpha > 0` byte-identical to the un-bypassed free
functions, with learned alpha never taking the bypass.

### Residual note

Module-level `alpha=0` no longer exercises the correction arithmetic, so
G2a-core is a weaker *arithmetic* control than before the patch. The arithmetic
control is retained through the free functions, still tested at `alpha=0` for
finite inputs and still asserted to exhibit the Inf/NaN behaviour. See `F-001`.


---

## `E2-003` — G2b paired one-epoch alpha=0 training diagnostic

**Verdict: INVESTIGATE. G2b does NOT pass.**

Applied verbatim from the criterion frozen in [GATES.md](GATES.md) before
execution: *any* C2-vs-T discrepancy is INVESTIGATE. There is a discrepancy on
all five full-precision metrics.

**This does not retract G2a-core.** Exact architectural identity — logits,
loss, gradients and a full optimizer step, all `assert_array_equal` — is
established on this same GPU by `E1-001` and `E1-002`. G2b probes whether that
survives 843 optimizer steps through the training loop.

### Provenance

| Field | Value |
|---|---|
| Artifact path | `/Users/durso/s5-runs/20260903-150747-G2b/` |
| Host / node | `pgi15-gpu3.iff.kfa-juelich.de` |
| GPU | RTX 3090, `CUDA_VISIBLE_DEVICES=0` |
| `C1` commit | `3c17a9af9725cfb8c06123a5ebd7add4cd819bfa` (`main`) |
| `C2`/`C2R`/`T` commit | `0316e3c3d102d230ed3660b0b6f5084a66bc01ea` (`prospective-lead`) |
| Clean trees | confirmed before each checkout |
| Exit codes | all four `0` |
| Env | `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `WANDB_MODE=offline`, seed 1919 |

Commits and argument lists were verified from provenance embedded **inside each
W&B run artifact**, not from directory timestamps. `tools/wandb_extract.py`
reads them from the binary `run-*.wandb` datastore.

### Run identification

Three attempts exist under `wb/T`; two are excluded on the evidence of their
recorded argument lists:

| Dir | Recorded args | Disposition |
|---|---|---|
| `151801-6ryb7x3d` | `--prospective_alpha=0.0`, **no `--prospective_layers`** | invalid — `layers` defaulted to `last`; **excluded** |
| `152011-wtvvhp49` | correct flags | interrupted, no metrics; **excluded** |
| `152030-2vn4mtb0` | correct flags, complete | **the valid `T`** |

### Hard precondition

Trainable parameters = **26,058 in all four runs**. Satisfied; no STOP.

### Full-precision metrics

| Metric | `C1` (main) | `C2` (off) | `C2R` (off repeat) | `T` (alpha=0) |
|---|---|---|---|---|
| Training Loss | 1.405187726020813 | 1.4051856994628906 | 1.405187726020813 | 1.4051895141601562 |
| Val loss | 0.3641023337841034 | 0.36359065771102905 | 0.36366006731987 | 0.3639678657054901 |
| Val Accuracy | 0.8951666355133057 | 0.8951666355133057 | 0.8951666355133057 | 0.8949999809265137 |
| Test Loss | 0.33774080872535706 | 0.33723869919776917 | 0.3372980058193207 | 0.33759671449661255 |
| Test Accuracy | 0.899899959564209 | 0.8999999761581421 | 0.8999999761581421 | 0.899899959564209 |

### Deltas

| Metric | `d_noise` (C2,C2R) | `d_ident` (C2,T) | `d_branch` (C1,C2) |
|---|---|---|---|
| Training Loss | 2.027e-06 | 3.815e-06 | 2.027e-06 |
| Val loss | 6.941e-05 | 3.772e-04 | 5.117e-04 |
| Val Accuracy | 0 | 1.667e-04 | 0 |
| Test Loss | 5.931e-05 | 3.580e-04 | 5.021e-04 |
| Test Accuracy | 0 | 1.000e-04 | 1.000e-04 |

### Findings

1. **The training loop is nondeterministic.** `C2` and `C2R` are the same
   commit, flags and seed and still differ (2e-6 train loss, 7e-5 val loss).
   Bit-reproducibility does not hold for full training runs on this GPU.
   Recorded separately as `F-002`.
2. **Every accuracy difference is exactly one sample**: `d_ident` on val
   accuracy is 0.99993 x (1/6000), on test accuracy 1.00017 x (1/10000).
   Accuracy is too coarse to function as an identity test at this scale.
3. **`C1` vs `C2` also exceeds the repeat discrepancy** on three metrics, which
   triggers the branch-trustworthiness rule. Both runs are plain S5 with no
   prospective code executing, which points at the training loop rather than
   the `alpha=0` path.

### Methodological caveat

`d_noise` is a **sample of size one**. A single repeat pair is a weak estimator
of the noise distribution, so `d_ident > d_noise` is not strong evidence that
`T` is anomalous. The criterion was predeclared and has been applied as
written rather than reinterpreted after seeing the data; the caveat is recorded
so the investigation does not begin by assuming `T` is at fault.

### Recommended investigation

**Primary.** Re-run `C2`, `C2R` and `T` with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`. If `C2 == C2R` and `C2 == T`
exactly, G2b passes cleanly. If `C2 == C2R` but `T` differs, that is a genuine
finding and work stops. Four runs, ~4 minutes.

**Fallback**, if the flag is unsupported on this XLA build: 5x `C2` and 5x `T`,
then test whether `T` lies inside the `C2` spread — replacing a size-1 estimate
with an interpretable distribution.

### Status

`alpha > 0` remains blocked until G2b is closed.


---

## `E2-004` — G2b re-run under deterministic XLA: **PASSED**

**Verdict: PASS.** Supersedes the INVESTIGATE verdict of `E2-003`, whose
divergence is now explained by `F-002` (cross-process backward-pass
nondeterminism) and eliminated by `XLA_FLAGS=--xla_gpu_deterministic_ops=true`.

Evidence label **E1 — correctness**: with the training loop bit-reproducible,
this is an exact result, not a smoke comparison.

### Provenance

| Field | Value |
|---|---|
| Artifact path | `/Users/durso/s5-runs/20260903-170803-G2b-deterministic/` |
| Host | `pgi15-gpu3` |
| `XLA_FLAGS` | `--xla_gpu_deterministic_ops=true` |
| `C1` commit | `3c17a9af9725cfb8c06123a5ebd7add4cd819bfa` (`main`) |
| `C2`/`C2R`/`T` commit | `0316e3c3d102d230ed3660b0b6f5084a66bc01ea` (`prospective-lead`) |
| Clean trees | confirmed before each detached checkout |
| Exit codes | all four `0` |
| Env | `CUDA_VISIBLE_DEVICES=0`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `WANDB_MODE=offline`, seed 1919 |

Flags verified from the argument list embedded in each W&B artifact; `T` carries
`--prospective_mode=lead --prospective_alpha=0.0
--prospective_alpha_learned=False --prospective_layers=all`.

### Result — all four runs bit-identical

| Metric | C1 | C2 | C2R | T |
|---|---|---|---|---|
| Training Loss | 1.4052175283432007 | *same* | *same* | *same* |
| Val loss | 0.36393970251083374 | *same* | *same* | *same* |
| Val Accuracy | 0.8949999809265137 | *same* | *same* | *same* |
| Test Loss | 0.3375144600868225 | *same* | *same* | *same* |
| Test Accuracy | 0.9000999927520752 | *same* | *same* | *same* |

```
d_identity (C2 vs T)   = 0    -> PASS
d_noise    (C2 vs C2R) = 0    -> training loop is bit-reproducible
d_branch   (C1 vs C2)  = 0    -> off-mode is a faithful plain-S5 control
```

### What this closes

1. **G2b PASSED.** Fixed `alpha=0` is a genuine no-op end to end, through the
   live code path, not merely at tensor level (which G2a had already shown).
2. **The branch-trustworthiness gate passes.** `prospective-lead` with
   `--prospective_mode=off` is bit-identical to `main`, so it is a valid
   control for every future paired comparison.
3. **`F-002`'s remedy is validated on a real training run**, not just the probe.

### Cost of determinism

| | wall clock |
|---|---|
| ordinary (`E2-002`) | 56.431 s |
| deterministic (this run, 4 runs) | 67.402 / 66.833 / 67.192 / 67.287 s |

Mean 67.2 s, **+19.1%**. Accepted as the standing cost for paired comparisons.
