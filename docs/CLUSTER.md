# GPU cluster installation

The GPU JAX wheel is intentionally **not** pinned in any requirements file: it
must match the cluster's CUDA runtime and driver. Install the CPU-independent
dependencies from `requirements_dev.txt`, then override JAX.

## Verified cluster environment

Confirmed working on `pgi15-gpu3` (RTX 3090, cc 8.6, driver 570.86.10 /
CUDA 12.8), Python 3.12.3: 13/13 baseline tests and a full one-epoch sMNIST
training run. See `E2-002` in [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md).

```
jax==0.11.0               flax==0.12.8        torch==2.14.0+cpu
jaxlib==0.11.0            optax==0.2.8        torchvision==0.29.0+cpu
jax-cuda12-pjrt==0.11.0   numpy==2.5.2        torchaudio==2.11.0+cpu
jax-cuda12-plugin==0.11.0 scipy==1.18.1       einops==0.8.2  datasets==5.0.1
```

Notes from that install:

- `torchaudio` is **required even for MNIST** - `s5/dataloaders/__init__.py`
  imports the audio module eagerly. Its absence does not show up in `pytest` or
  in `run_train.py --help`; it fails a minute into a run, at dataset creation.
  Verify with `python -c "from s5.dataloaders.basic import MNIST"` before any run.
- The PyTorch stack is deliberately CPU-only (`+cpu` wheels); JAX is the GPU
  backend. Install torch/torchvision/torchaudio from
  `--index-url https://download.pytorch.org/whl/cpu` to keep it that way.
- These versions differ slightly from `requirements_dev.txt` (jax 0.11.0 vs
  0.11.1, flax 0.12.8 vs 0.12.9, torch 2.14.0 vs 2.13.0). Both combinations
  pass. The authoritative record for any given run is its own
  `requirements-frozen.txt` artifact, not the requirements file.

## Recommended run-time environment variables

```bash
export CUDA_VISIBLE_DEVICES=0            # one GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # else JAX grabs ~75% and peak memory is unmeasurable
export WANDB_MODE=offline
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` changes allocator behaviour, so treat
wall-clock from such runs as indicative. Accuracy and loss are unaffected.

## CUDA 12

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_dev.txt
python -m pip install --upgrade --force-reinstall "jax[cuda12]" "jaxlib"
python -c 'import jax; print(jax.__version__, jax.default_backend(), jax.devices())'
python -m pytest tests/ -q
```

## CUDA 13

Identical, with:

```bash
python -m pip install --upgrade --force-reinstall "jax[cuda13]" "jaxlib"
```

## Notes

- Do not use the upstream `requirements_gpu.txt`. Its `jax[cuda]>=version` line
  is a literal placeholder, and `requirements_cpu.txt` pins `jax==0.3.5` /
  `torch==1.11.0`, which are not installable on a modern Python.
- Install `torch` from the CPU index if the cluster's CUDA torch build
  conflicts with the JAX CUDA runtime; torch is only used for dataloading:
  `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- Re-run `python -m pytest tests/ -q` after installing the GPU wheel. All tests
  are device-agnostic and must still pass.
- Record `nvidia-smi`, `jax.devices()` and peak memory in any result you report.

## Slurm

`bin/slurm/s5_job.sbatch` is a generic single-GPU job that forwards its
arguments to `run_train.py`. Edit the `module load` block and the
partition/account directives for your site before first use.

```bash
sbatch bin/slurm/s5_job.sbatch --dataset=mnist-classification --epochs=1
sbatch --array=0-38 bin/slurm/s5_job.sbatch --cmdfile=sweep.txt
```
