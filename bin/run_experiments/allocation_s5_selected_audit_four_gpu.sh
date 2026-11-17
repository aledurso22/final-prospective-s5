#!/usr/bin/env bash
# Frozen selected-checkpoint audit of N/A/C/D x 301/302, on FOUR GPUs.
# First record original selections and actual SSM modes on CPU. Then each GPU
# handles two selected checkpoints sequentially, with the same training-only
# BatchNorm recalibration procedure. There is no training, epoch reselection,
# finalizer, or test-set access. The original run is read-only.
#
# Start inside a four-GPU allocation in this separate checkout. No nested
# Slurm commands are used. DRY_RUN=1 prints all eight jobs without data access.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$HERE/cluster_env.sh"
cd "$REPO_ROOT"

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the full authoritative sha}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "FAIL: HEAD $(git rev-parse HEAD) is not $EXPECTED_COMMIT" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  git status --short >&2; echo "FAIL: refusing a modified worktree" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FAIL: interpreter is not executable: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in
  0|false|False|FALSE|"") ;;
  *) echo "FAIL: production replay requires JAX_ENABLE_X64=0" >&2; exit 1 ;;
esac
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || {
  echo "FAIL: SLURM_JOB_ID='$ALLOCATION' is not numeric" >&2; exit 1; }
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
  echo "FAIL: DRY_RUN must be 0 or 1" >&2; exit 1; }

IFS=',' read -r -a RAW_TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
TOKENS=()
for token in ${RAW_TOKENS[@]+"${RAW_TOKENS[@]}"}; do
  token="${token//[[:space:]]/}"
  [[ -n "$token" ]] || continue
  duplicate=0
  for seen in ${TOKENS[@]+"${TOKENS[@]}"}; do [[ "$seen" == "$token" ]] && duplicate=1; done
  [[ "$duplicate" == 0 ]] && TOKENS+=("$token")
done
[[ "${#TOKENS[@]}" -ge 4 ]] || {
  echo "FAIL: need 4 distinct GPU tokens, have ${#TOKENS[@]}" >&2; exit 1; }

SOURCE_RUN="${SOURCE_RUN:-$PROSPECTIVE_RUNS/s5-timescale-2x2/20261117-195818}"
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
AUDIT_ROOT="${AUDIT_ROOT:-$PROSPECTIVE_RUNS/s5-selected-audit/job-$ALLOCATION}"
MANIFEST="$AUDIT_ROOT/checkpoints/manifest.json"
BN_ROOT="$AUDIT_ROOT/bn"
ARMS=(native_matched_s5 generalized_prospective_s5 calibrated_old_s5 calibrated_prediction_s5)

echo "selected-checkpoint audit: job $ALLOCATION, commit $EXPECTED_COMMIT"
echo "  source run: $SOURCE_RUN"
echo "  output:     $AUDIT_ROOT"
echo "  GPU tokens: ${TOKENS[*]:0:4} (two selected checkpoints per GPU)"
echo "  fixed tasks: N/A/C/D x seeds 301 302; original selected epochs"
echo "  CPU: $PY -u -m experiments.s5_predictor_corrector.selected_checkpoint_audit --run-root $SOURCE_RUN --out $AUDIT_ROOT/checkpoints"
for slot in 0 1 2 3; do
  for index in "$slot" "$((slot + 4))"; do
    arm="${ARMS[$((index / 2))]}"; seed="$((301 + index % 2))"
    echo "  GPU ${TOKENS[$slot]}: task-index $index $arm/$seed -> $BN_ROOT/$arm/$seed"
  done
done
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: no checkpoints, dataset, or GPU inspected; nothing started."
  exit 0
fi

[[ -d "$SOURCE_RUN" ]] || { echo "FAIL: missing source run: $SOURCE_RUN" >&2; exit 1; }
[[ -d "$DATA_CACHE" ]] || { echo "FAIL: missing data cache: $DATA_CACHE" >&2; exit 1; }
# Resolve symlinks before creating any output. Neither the source nor anything
# inside it may be used as this job's destination.
"$PY" - "$SOURCE_RUN" "$AUDIT_ROOT" <<'PYEOF'
from pathlib import Path
import sys
source, out = map(lambda p: Path(p).resolve(), sys.argv[1:])
if source == out or source in out.parents:
    raise SystemExit("FAIL: audit output must be outside the original run")
if out.exists():
    raise SystemExit(f"FAIL: refusing existing output: {out}")
PYEOF
mkdir -p "$AUDIT_ROOT/logs" "$AUDIT_ROOT/jax-cache" "$BN_ROOT"
printf 'audit_commit=%s\nsource_run=%s\ndata_cache=%s\nallocation=%s\ngpu_tokens=%s\nseeds=301 302\nepoch_selection=original task_result.json selection, frozen\n' \
  "$EXPECTED_COMMIT" "$SOURCE_RUN" "$DATA_CACHE" "$ALLOCATION" "${TOKENS[*]:0:4}" \
  > "$AUDIT_ROOT/audit_metadata.txt"

# This reads parameters and result records; it never initializes a model or
# opens train/validation/test arrays. Float64 NumPy mode analysis is separate
# from the float32 JAX checkpoint replay below.
JAX_PLATFORMS=cpu "$PY" -u -m experiments.s5_predictor_corrector.selected_checkpoint_audit \
  --run-root "$SOURCE_RUN" --out "$AUDIT_ROOT/checkpoints" \
  > "$AUDIT_ROOT/logs/checkpoint_audit.log" 2>&1
cat "$AUDIT_ROOT/logs/checkpoint_audit.log"
"$PY" - "$MANIFEST" <<'PYEOF'
import json, sys
manifest = json.load(open(sys.argv[1]))
arms = ("native_matched_s5", "generalized_prospective_s5",
        "calibrated_old_s5", "calibrated_prediction_s5")
expected = [(a, s) for a in arms for s in (301, 302)]
actual = [(t["arm"], int(t["seed"])) for t in manifest["tasks"]]
if actual != expected:
    raise SystemExit(f"FAIL: unexpected manifest tasks/order: {actual}")
PYEOF

ACTIVE_PIDS=(); WORKER_PIDS=()
cleanup_workers() {
  local pid
  for pid in ${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in ${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}; do wait "$pid" 2>/dev/null || true; done
  ACTIVE_PIDS=()
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_worker() {
  local slot="$1" token="${TOKENS[$1]}"
  (
    export CUDA_VISIBLE_DEVICES="$token" JAX_PLATFORMS=cuda
    export JAX_COMPILATION_CACHE_DIR="$AUDIT_ROOT/jax-cache/worker-$slot"
    mkdir -p "$JAX_COMPILATION_CACHE_DIR"
    worker_child=""
    cleanup_child() {
      if [[ -n "$worker_child" ]]; then
        kill -TERM "$worker_child" 2>/dev/null || true
        wait "$worker_child" 2>/dev/null || true
      fi
    }
    trap cleanup_child EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    "$PY" - <<'PYEOF' > "$AUDIT_ROOT/logs/worker-$slot-device.log" 2>&1 &
import os
import jax
assert not jax.config.read("jax_enable_x64"), "production replay must be float32"
devices = jax.devices()
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"], "JAX:", devices)
assert len(devices) == 1 and devices[0].platform == "gpu", "worker must see exactly one GPU"
PYEOF
    worker_child="$!"; wait "$worker_child"; worker_child=""
    for index in "$slot" "$((slot + 4))"; do
      arm="${ARMS[$((index / 2))]}"; seed="$((301 + index % 2))"
      echo "worker $slot GPU $token: starting $arm/$seed (manifest index $index)"
      "$PY" -u -m experiments.s5_predictor_corrector.selected_bn_audit \
        --manifest "$MANIFEST" --task-index "$index" --data-cache "$DATA_CACHE" \
        --out "$BN_ROOT/$arm/$seed" > "$AUDIT_ROOT/logs/${arm}_${seed}.log" 2>&1 &
      worker_child="$!"; wait "$worker_child"; worker_child=""
      echo "worker $slot GPU $token: finished $arm/$seed"
    done
  ) &
  ACTIVE_PIDS+=("$!"); WORKER_PIDS+=("$!")
}
for slot in 0 1 2 3; do start_worker "$slot"; done
failed=0
for slot in 0 1 2 3; do
  status=0; wait "${WORKER_PIDS[$slot]}" || status=$?
  unset 'ACTIVE_PIDS[slot]'
  echo "worker $slot exited $status"
  [[ "$status" == 0 ]] || failed=1
done
if [[ "$failed" != 0 ]]; then
  echo "FAIL: one or more BN audits failed; inspect $AUDIT_ROOT/logs. No aggregate produced." >&2
  exit 1
fi
# Aggregation rechecks checkpoint and selection hashes, original-score replay,
# fixed epochs, all eight reports, and identical recalibration sample counts.
JAX_PLATFORMS=cpu "$PY" -u -m experiments.s5_predictor_corrector.selected_bn_audit \
  --manifest "$MANIFEST" --aggregate "$BN_ROOT" \
  > "$AUDIT_ROOT/logs/aggregate.log" 2>&1
cat "$AUDIT_ROOT/logs/aggregate.log"
echo "Completed selected-checkpoint audit: $AUDIT_ROOT"
echo "Primary results unchanged. Separate BN sensitivity report: $BN_ROOT/selected_bn_summary.json"
