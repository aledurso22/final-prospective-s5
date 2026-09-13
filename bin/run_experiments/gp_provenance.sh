#!/usr/bin/env bash
# Capture provenance BEFORE a generalized-prospectivity GPU run.
# Usage:  ./bin/run_experiments/gp_provenance.sh "$GP"
set -euo pipefail
OUT="${1:?usage: gp_provenance.sh <run_dir>}"
mkdir -p "$OUT"
{
  date -Is
  hostname
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "XLA_FLAGS=${XLA_FLAGS:-unset}"
  echo "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-unset}"
  echo "WANDB_MODE=${WANDB_MODE:-unset}"
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  echo "--- porcelain (empty = clean) ---"
  git status --porcelain
  echo "--- end porcelain ---"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  python -c "import sys,jax;print(sys.prefix);print(jax.__version__,jax.default_backend(),jax.devices())"
} > "$OUT/provenance.txt" 2>&1
python -m pip freeze > "$OUT/requirements-frozen.txt"
echo "wrote $OUT/provenance.txt and $OUT/requirements-frozen.txt"
