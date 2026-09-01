#!/usr/bin/env bash
#SBATCH --job-name=s5
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#
# Generic single-GPU S5 job.
#
# Deliberately dataset- and branch-agnostic: pass the full run_train.py
# argument list through, so the same script serves plain S5 and any
# prospective branch.
#
#   sbatch bin/slurm/s5_job.sbatch --dataset=mnist-classification --epochs=1
#
# For a sweep, use --array and index into a command file:
#   sbatch --array=0-38 bin/slurm/s5_job.sbatch --cmdfile=sweep.txt
#
# ADJUST FOR YOUR SITE: partition/account, module names and the CUDA-matched
# JAX wheel. Nothing here is JUWELS/JURECA-specific.

set -euo pipefail

mkdir -p slurm_logs

# --- site modules (edit to match the cluster) -------------------------------
# module purge
# module load Stages/2024 GCC CUDA/12 cuDNN

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source .venv/bin/activate

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi || true
python -c 'import jax; print("jax", jax.__version__, jax.default_backend(), jax.devices())'
git rev-parse HEAD
git status --short

# --- resolve arguments ------------------------------------------------------
# Either a literal argument list, or --cmdfile=<file> plus $SLURM_ARRAY_TASK_ID
# selecting one line from that file.
if [[ "${1:-}" == --cmdfile=* ]]; then
  CMDFILE="${1#--cmdfile=}"
  LINE=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
  CMD=$(sed -n "${LINE}p" "$CMDFILE")
  echo "=== [$LINE] $CMD ==="
  eval "$CMD"
else
  python run_train.py "$@"
fi
