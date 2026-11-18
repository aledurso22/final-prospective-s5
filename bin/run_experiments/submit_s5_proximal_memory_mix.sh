#!/usr/bin/env bash
# Submit only a published, pinned revision; this script never pushes code.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned full commit}"
SHA="$(git -C "$REPO" rev-parse HEAD)"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ && "$SHA" == "$EXPECTED_COMMIT" ]] || {
  echo "FAIL: wrong submission commit" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "FAIL: submission checkout must be clean, including untracked files" >&2; exit 1; }
STAGE="${STAGE:-train}"; DRY_RUN="${DRY_RUN:-0}"
[[ "$STAGE" == smoke || "$STAGE" == train ]] || { echo "FAIL: invalid STAGE" >&2; exit 1; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: one allocation on pgi14-gpu1, partition pgi14; 2 RTX4090, 16 CPUs, 100G, 8h"
  echo "DRY_RUN: commit $SHA; stage $STAGE; seeds 301/302 on distinct assigned GPU tokens"
  echo "DRY_RUN: alpha=1 fast_ratio=0.1 slow_ratio=1 slow_weight=4/9 proximal_ratio=0.5"
  echo "DRY_RUN: profile and both full-size GPU smokes gate fresh batch16/15-epoch training"
  exit 0
fi
sbatch --partition=pgi14 --nodelist=pgi14-gpu1 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=100G \
  --gres=gpu:nvidia_geforce_rtx_4090:2 --gpu-bind=none \
  --time=08:00:00 --job-name=s5-prox-memory-mix \
  --output='/Users/durso/s5-runs/s5-proximal-memory-mix-%j.log' \
  --export="ALL,S5_MEMORY_MIX_COMMIT=$SHA,S5_MEMORY_MIX_STAGE=$STAGE" <<'S5_MEMORY_MIX_JOB'
#!/bin/bash
set -euo pipefail
TASK_REPO="/Local/durso/s5-proximal-memory-mix-${SLURM_JOB_ID}"
[[ ! -e "$TASK_REPO" ]] || { echo "FAIL: isolated clone destination exists" >&2; exit 1; }
git clone --no-checkout \
  "$(git -C /Local/durso/final-prospective-s5 remote get-url origin)" "$TASK_REPO"
cd "$TASK_REPO"
git checkout --detach "$S5_MEMORY_MIX_COMMIT"
export PROSPECTIVE_REPO="$PWD"
export PROSPECTIVE_VENV=/Local/durso/final-prospective-s5/.venv
export PROSPECTIVE_RUNS=/Users/durso/s5-runs
export S5_THREE_ARM_DATA="$PROSPECTIVE_RUNS/sc10_official_cache"
export EXPECTED_COMMIT="$S5_MEMORY_MIX_COMMIT"
export STAGE="$S5_MEMORY_MIX_STAGE" DRY_RUN=0 JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
exec bash bin/run_experiments/cluster_s5_proximal_memory_mix.sh
S5_MEMORY_MIX_JOB
