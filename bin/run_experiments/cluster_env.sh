#!/usr/bin/env bash
# Shared cluster environment. SOURCE this; do not execute it.
#
# Paths come from the handoff and are VERIFIED here rather than assumed. The
# shared environment is used as-is: nothing is upgraded, replaced or installed
# by these scripts.
set -uo pipefail

: "${PROSPECTIVE_VENV:=/Local/durso/prospective_ssm_project/.venv}"
: "${PROSPECTIVE_REPO:=/Local/durso/final-prospective-s5}"
: "${PROSPECTIVE_RUNS:=/Users/durso/s5-runs}"
: "${PROSPECTIVE_DATA:=$PROSPECTIVE_RUNS/sc10_cache}"

export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export WANDB_DIR="$PROSPECTIVE_RUNS/wandb"
export TOKENIZERS_PARALLELISM=false

PY="$PROSPECTIVE_VENV/bin/python"

cluster_check_env() {
  local ok=0
  echo "host        : $(hostname)"
  echo "date        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm job   : ${SLURM_JOB_ID:-<none>}"
  echo "visible gpus: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  [ -x "$PY" ] || { echo "MISSING interpreter: $PY"; ok=1; }
  [ -d "$PROSPECTIVE_REPO" ] || { echo "MISSING repo: $PROSPECTIVE_REPO"; ok=1; }
  [ -d "$PROSPECTIVE_RUNS" ] || { echo "MISSING run dir: $PROSPECTIVE_RUNS"; ok=1; }
  echo "interpreter : $PY"
  echo "repo        : $PROSPECTIVE_REPO"
  echo "runs        : $PROSPECTIVE_RUNS"
  echo "data cache  : $PROSPECTIVE_DATA"
  return $ok
}
