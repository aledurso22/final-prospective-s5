# GPU cluster installation

The GPU JAX wheel is intentionally **not** pinned in any requirements file: it
must match the cluster's CUDA runtime and driver. Install the CPU-independent
dependencies from `requirements_dev.txt`, then override JAX.

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
