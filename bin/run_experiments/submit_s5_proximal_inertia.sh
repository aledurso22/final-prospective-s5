#!/usr/bin/env bash
# Submit candidate/control x seeds 301/302 as ONE four-GPU allocation on
# pgi14-gpu1. The job clones the pinned commit into its own node-local
# directory; no existing checkout, environment or run is changed.
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
  echo "DRY_RUN: 4 RTX4090 on pgi14-gpu1; 32 CPUs 160G 10h; stage=$STAGE; commit=$SHA"
  echo "tasks: proximal_inertial_s5/301 proximal_inertial_s5/302 proximal_inertial_control_s5/301 proximal_inertial_control_s5/302 (candidate/control)"
  exit 0
fi
sbatch --partition=pgi14 --nodelist=pgi14-gpu1 \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=160G \
  --gres=gpu:nvidia_geforce_rtx_4090:4 --gpu-bind=none \
  --time=10:00:00 --job-name=s5-prox-inertia \
  --output='/Users/durso/s5-runs/s5-proximal-inertia-%j.log' \
  --export="ALL,S5_PI_COMMIT=$SHA,S5_PI_STAGE=$STAGE" <<'S5_PI_JOB'
#!/bin/bash
set -euo pipefail
REPO="/Local/durso/s5-proximal-inertia-${SLURM_JOB_ID}"
[[ ! -e "$REPO" ]] || { echo "FAIL: clone destination exists" >&2; exit 1; }
git clone --no-checkout "$(git -C /Local/durso/final-prospective-s5 remote get-url origin)" "$REPO"
cd "$REPO"
git checkout --detach "$S5_PI_COMMIT"
export PROSPECTIVE_VENV=/Local/durso/final-prospective-s5/.venv
export PROSPECTIVE_RUNS=/Users/durso/s5-runs
export S5_THREE_ARM_DATA="$PROSPECTIVE_RUNS/sc10_official_cache"
export EXPECTED_COMMIT="$S5_PI_COMMIT" STAGE="$S5_PI_STAGE" JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec bash bin/run_experiments/cluster_s5_proximal_inertia.sh
S5_PI_JOB
