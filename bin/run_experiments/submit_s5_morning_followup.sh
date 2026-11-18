#!/usr/bin/env bash
# One allocation owns the whole recovery/replication wave.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
SHA="$(git -C "$REPO" rev-parse HEAD)"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned recovery commit}"
[[ "$SHA" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: wrong submission commit" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || { echo "FAIL: modified submission checkout" >&2; exit 1; }
SOURCE_ROOT="${HISTORY_RESUME_SOURCE_ROOT:-/Users/durso/s5-runs/s5-history-pilot/job-76866/gradient_history_s5}"
[[ "$SOURCE_ROOT" != *','* ]] || { echo "FAIL: source path cannot contain a comma" >&2; exit 1; }
for seed in 301 302; do
  for file in checkpoint_epoch_09.msgpack metrics.jsonl history_protocol.json; do
    [[ -s "$SOURCE_ROOT/$seed/$file" ]] || { echo "FAIL: missing source $SOURCE_ROOT/$seed/$file" >&2; exit 1; }
  done
done
[[ "${DRY_RUN:-0}" == 0 || "${DRY_RUN:-0}" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN: one allocation, 8 RTX4090 on pgi14-gpu1; 32 CPUs, 200G, 8h; commit $SHA"
  echo "GPU0: H resume301 then P alpha1/303; GPU1: H resume302 then P alpha0.1/302"
  echo "GPU2..7: N304 D304 P304 N305 D305 P305; fresh runs use 15 epochs and batch16"
  exit 0
fi
sbatch --partition=pgi14 --nodelist=pgi14-gpu1 \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=200G \
  --gres=gpu:nvidia_geforce_rtx_4090:8 --gpu-bind=none \
  --time=08:00:00 --job-name=s5-morning-followup \
  --output='/Users/durso/s5-runs/s5-morning-followup-%j.log' \
  --export="ALL,S5_MORNING_COMMIT=$SHA,S5_MORNING_SOURCE=$SOURCE_ROOT" <<'S5_MORNING_JOB'
#!/bin/bash
set -euo pipefail
TASK_REPO="/Local/durso/s5-morning-followup-${SLURM_JOB_ID}"
[[ ! -e "$TASK_REPO" ]] || { echo "FAIL: clone destination exists" >&2; exit 1; }
git clone --no-checkout \
  "$(git -C /Local/durso/final-prospective-s5 remote get-url origin)" "$TASK_REPO"
cd "$TASK_REPO"
git checkout --detach "$S5_MORNING_COMMIT"
export PROSPECTIVE_REPO="$PWD"
export PROSPECTIVE_VENV=/Local/durso/final-prospective-s5/.venv
export PROSPECTIVE_RUNS=/Users/durso/s5-runs
export S5_THREE_ARM_DATA="$PROSPECTIVE_RUNS/sc10_official_cache"
export EXPECTED_COMMIT="$S5_MORNING_COMMIT"
export HISTORY_RESUME_SOURCE_ROOT="$S5_MORNING_SOURCE"
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
exec bash bin/run_experiments/allocation_s5_morning_followup.sh
S5_MORNING_JOB
