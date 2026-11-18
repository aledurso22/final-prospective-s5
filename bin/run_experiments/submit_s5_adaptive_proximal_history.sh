#!/usr/bin/env bash
# Submit one adaptive-P/frozen-P comparison, seeds 301/302, on pgi14-gpu1.
# Submission does not modify any active checkout, environment, or old run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
SHA="$(git -C "$REPO" rev-parse HEAD)"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned full commit}"
[[ "$SHA" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: wrong submission commit" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || { echo "FAIL: modified submission checkout" >&2; exit 1; }
STAGE="${STAGE:-train}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$STAGE" == train || "$STAGE" == smoke ]] || { echo "FAIL: invalid STAGE" >&2; exit 1; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: 4 RTX4090 on pgi14-gpu1; 32 CPUs 160G 10h; stage=$STAGE; commit=$SHA"
  echo "tasks: adaptive_proximal_history_s5/{301,302} adaptive_proximal_history_control_s5/{301,302}"
  echo "All four GPU smokes must pass before any training starts."
  exit 0
fi
sbatch --partition=pgi14 --nodelist=pgi14-gpu1 \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=160G \
  --gres=gpu:nvidia_geforce_rtx_4090:4 --gpu-bind=none \
  --time=10:00:00 --job-name=s5-adaptive-prox \
  --output='/Users/durso/s5-runs/s5-adaptive-proximal-history-%j.log' \
  --export="ALL,S5_AP_COMMIT=$SHA,S5_AP_STAGE=$STAGE" <<'S5_AP_JOB'
#!/bin/bash
set -euo pipefail
REPO="/Local/durso/s5-adaptive-proximal-history-${SLURM_JOB_ID}"
[[ ! -e "$REPO" ]] || { echo "FAIL: clone destination exists" >&2; exit 1; }
git clone --no-checkout "$(git -C /Local/durso/final-prospective-s5 remote get-url origin)" "$REPO"
cd "$REPO"
git checkout --detach "$S5_AP_COMMIT"
export PROSPECTIVE_VENV=/Local/durso/final-prospective-s5/.venv
export PROSPECTIVE_RUNS=/Users/durso/s5-runs
export S5_THREE_ARM_DATA="$PROSPECTIVE_RUNS/sc10_official_cache"
export EXPECTED_COMMIT="$S5_AP_COMMIT" STAGE="$S5_AP_STAGE" JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec bash bin/run_experiments/cluster_s5_adaptive_proximal_history.sh
S5_AP_JOB
