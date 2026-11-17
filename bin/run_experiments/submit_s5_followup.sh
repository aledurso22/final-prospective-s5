#!/usr/bin/env bash
# Submit only the authorized seed303 comparison and selected-checkpoint audit.
# Run on pgi14-head from a pinned checkout. Each job uses its own node-local
# clone; the existing training checkout, environment and results are preserved.
# No gradient-history training is submitted by this helper.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
SHA="$(git -C "$REPO" rev-parse HEAD)"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned follow-up commit}"
[[ "$SHA" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: checkout differs from EXPECTED_COMMIT" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || {
  echo "FAIL: refusing modified submission checkout" >&2; exit 1; }
for task in seed303 selected-audit; do
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN: $task on pgi14-gpu1, partition pgi14, 4 RTX4090 GPUs, 16 CPUs, 100G, 4h; commit $SHA"
    continue
  fi
  sbatch --partition=pgi14 --nodelist=pgi14-gpu1 \
    --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=100G \
    --gres=gpu:nvidia_geforce_rtx_4090:4 --gpu-bind=none \
    --time=04:00:00 --job-name="s5-followup-$task" \
    --output='/Users/durso/s5-runs/s5-followup-%x-%j.log' \
    --export="ALL,S5_FOLLOWUP_COMMIT=$SHA,S5_FOLLOWUP_TASK=$task" <<'S5_FOLLOWUP_JOB'
#!/bin/bash
set -euo pipefail
case "$S5_FOLLOWUP_TASK" in
  seed303) launcher=allocation_s5_seed303_four_gpu.sh ;;
  selected-audit) launcher=allocation_s5_selected_audit_four_gpu.sh ;;
  *) echo "FAIL: unknown follow-up task" >&2; exit 1 ;;
esac
# Per-job checkout avoids checkout races even if another task is submitted.
FOLLOWUP_REPO="/Local/durso/s5-followup-${S5_FOLLOWUP_TASK}-${SLURM_JOB_ID}"
[[ ! -e "$FOLLOWUP_REPO" ]] || { echo "FAIL: clone destination exists" >&2; exit 1; }
git clone --no-checkout \
  "$(git -C /Local/durso/final-prospective-s5 remote get-url origin)" "$FOLLOWUP_REPO"
cd "$FOLLOWUP_REPO"
git checkout --detach "$S5_FOLLOWUP_COMMIT"
export PROSPECTIVE_REPO="$PWD"
export PROSPECTIVE_VENV=/Local/durso/final-prospective-s5/.venv
export PROSPECTIVE_RUNS=/Users/durso/s5-runs
export S5_THREE_ARM_DATA="$PROSPECTIVE_RUNS/sc10_official_cache"
export EXPECTED_COMMIT="$S5_FOLLOWUP_COMMIT"
export STAGE=train JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
"$PROSPECTIVE_VENV/bin/python" -u - <<'PY'
import os
tokens = [t.strip() for t in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if t.strip()]
assert len(tokens) == len(set(tokens)) == 4, f"expected four distinct allocated GPUs, got {tokens}"
assert len(os.sched_getaffinity(0)) >= 16, "allocation exposes fewer than 16 CPUs"
import jax
devices = jax.devices()
print("Allocated devices:", devices, flush=True)
assert len(devices) == 4 and all(d.platform == "gpu" for d in devices)
PY
exec bash "bin/run_experiments/$launcher"
S5_FOLLOWUP_JOB
done
