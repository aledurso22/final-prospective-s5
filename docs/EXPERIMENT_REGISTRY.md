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
