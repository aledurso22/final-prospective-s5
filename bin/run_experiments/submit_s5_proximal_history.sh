#!/usr/bin/env bash
# Submit an isolated one-GPU sequence; no existing checkout or run is changed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
SHA="$(git -C "$REPO" rev-parse HEAD)"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned new commit}"
[[ "$SHA" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: wrong submission commit" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || { echo "FAIL: modified submission checkout" >&2; exit 1; }
STAGE="${STAGE:-train}"
[[ "$STAGE" == train || "$STAGE" == smoke ]] || { echo "FAIL: invalid STAGE" >&2; exit 1; }
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN: 1 RTX4090 on pgi14-gpu1; 8CPUs 48G 12h; stage=$STAGE; commit=$SHA"
  echo "sequence: alpha1/301 alpha1/302 alpha0p1/301; matching full-size GPU smoke before each training"
  exit 0
fi
sbatch --partition=pgi14 --nodelist=pgi14-gpu1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G \
  --gres=gpu:nvidia_geforce_rtx_4090:1 --gpu-bind=none \
  --time=12:00:00 --job-name=s5-proximal-history \
  --output='/Users/durso/s5-runs/s5-proximal-history-%j.log' \
  --export="ALL,S5_PROXIMAL_COMMIT=$SHA,S5_PROXIMAL_STAGE=$STAGE" <<'S5_PROXIMAL_JOB'
#!/bin/bash
set -euo pipefail
PILOT_REPO="/Local/durso/s5-proximal-history-${SLURM_JOB_ID}"
[[ ! -e "$PILOT_REPO" ]] || { echo "FAIL: clone destination exists" >&2; exit 1; }
git clone --no-checkout \
  "$(git -C /Local/durso/final-prospective-s5 remote get-url origin)" "$PILOT_REPO"
cd "$PILOT_REPO"
git checkout --detach "$S5_PROXIMAL_COMMIT"
export PROSPECTIVE_REPO="$PWD"
export PROSPECTIVE_VENV=/Local/durso/final-prospective-s5/.venv
export PROSPECTIVE_RUNS=/Users/durso/s5-runs
export S5_THREE_ARM_DATA="$PROSPECTIVE_RUNS/sc10_official_cache"
export EXPECTED_COMMIT="$S5_PROXIMAL_COMMIT"
export STAGE="$S5_PROXIMAL_STAGE" JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
exec bash bin/run_experiments/cluster_s5_proximal_history.sh
S5_PROXIMAL_JOB
