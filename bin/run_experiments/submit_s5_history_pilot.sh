#!/usr/bin/env bash
# Queue only the two-seed history pilot; never resubmit the N/A/C/D jobs.
# The selected-checkpoint audit must finish successfully first. Each pilot
# seed passes a full-shape GPU smoke before either training process starts.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
SHA="$(git -C "$REPO" rev-parse HEAD)"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned pilot commit}"
: "${HISTORY_AFTER_JOB:?set HISTORY_AFTER_JOB to the selected-checkpoint audit job id}"
[[ "$SHA" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: checkout differs from EXPECTED_COMMIT" >&2; exit 1; }
[[ "$HISTORY_AFTER_JOB" =~ ^[0-9]+$ ]] || { echo "FAIL: audit job id must be numeric" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || {
  echo "FAIL: refusing modified submission checkout" >&2; exit 1; }
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN: history301/302 afterok:$HISTORY_AFTER_JOB on pgi14-gpu1, 2 RTX4090 GPUs, 16 CPUs, 100G, 6h; commit $SHA"
  exit 0
fi
AUDIT_ROOT="/Users/durso/s5-runs/s5-selected-audit/job-$HISTORY_AFTER_JOB"
DEPENDENCY=("--dependency=afterok:$HISTORY_AFTER_JOB")
# Slurm can forget completed jobs after MinJobAge. A complete final audit
# report is an alternative completion witness, so a later submission need
# not refer to an expired job id. An incomplete/invalid report fails closed.
if [[ -f "$AUDIT_ROOT/bn/selected_bn_summary.json" ]]; then
  python3 - "$AUDIT_ROOT/bn/selected_bn_summary.json" "$AUDIT_ROOT/checkpoints/manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
manifest_bytes = Path(sys.argv[2]).read_bytes()
assert report["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
arms = ("native_matched_s5", "generalized_prospective_s5", "calibrated_old_s5", "calibrated_prediction_s5")
expected = {(a, s) for a in arms for s in (301, 302)}
tasks = report["tasks"]
assert len(tasks) == 8 and {(t["arm"], t["seed"]) for t in tasks} == expected
assert all(t["reproduction"]["within_atol"] for t in tasks)
assert report["epoch_reselection"] is False and report["test_split_read"] is False
print("Selected-checkpoint audit already completed successfully; final report verified.")
PY
  DEPENDENCY=()
fi
sbatch --partition=pgi14 --nodelist=pgi14-gpu1 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=100G \
  --gres=gpu:nvidia_geforce_rtx_4090:2 --gpu-bind=none \
  "${DEPENDENCY[@]}" \
  --time=06:00:00 --job-name=s5-history-pilot \
  --output='/Users/durso/s5-runs/s5-history-pilot-%j.log' \
  --export="ALL,S5_HISTORY_COMMIT=$SHA,S5_HISTORY_AFTER_JOB=$HISTORY_AFTER_JOB" <<'S5_HISTORY_JOB'
#!/bin/bash
set -euo pipefail
HISTORY_REPO="/Local/durso/s5-history-pilot-${SLURM_JOB_ID}"
[[ ! -e "$HISTORY_REPO" ]] || { echo "FAIL: clone destination exists" >&2; exit 1; }
git clone --no-checkout \
  "$(git -C /Local/durso/final-prospective-s5 remote get-url origin)" "$HISTORY_REPO"
cd "$HISTORY_REPO"
git checkout --detach "$S5_HISTORY_COMMIT"
export PROSPECTIVE_REPO="$PWD"
export PROSPECTIVE_VENV=/Local/durso/final-prospective-s5/.venv
export PROSPECTIVE_RUNS=/Users/durso/s5-runs
export S5_THREE_ARM_DATA="$PROSPECTIVE_RUNS/sc10_official_cache"
export EXPECTED_COMMIT="$S5_HISTORY_COMMIT"
export HISTORY_AFTER_JOB="$S5_HISTORY_AFTER_JOB"
export STAGE=train JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
"$PROSPECTIVE_VENV/bin/python" -u - <<'PY'
import os
tokens = [t.strip() for t in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if t.strip()]
assert len(tokens) == len(set(tokens)) == 2, f"expected two distinct allocated GPUs, got {tokens}"
assert len(os.sched_getaffinity(0)) >= 16, "allocation exposes fewer than 16 CPUs"
import jax
devices = jax.devices()
print("Allocated history devices:", devices, flush=True)
assert len(devices) == 2 and all(d.platform == "gpu" for d in devices)
PY
exec bash bin/run_experiments/allocation_s5_history_two_gpu.sh
S5_HISTORY_JOB
