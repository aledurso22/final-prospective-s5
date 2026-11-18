#!/usr/bin/env bash
# Two exact continuations plus eight fresh tasks, in eight GPU worker queues.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO"
source "$HERE/cluster_env.sh"
cd "$REPO"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned recovery commit}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: wrong commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "FAIL: modified checkout" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FAIL: missing Python: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in 0|false|False|FALSE|"") ;; *) echo "FAIL: x64 must be off" >&2; exit 1;; esac
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric Slurm job required" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#TOKENS[@]}" == 8 ]] || { echo "FAIL: need exactly eight distinct GPU tokens" >&2; exit 1; }
for i in 0 1 2 3 4 5 6 7; do
  TOKENS[$i]="${TOKENS[$i]//[[:space:]]/}"
  [[ -n "${TOKENS[$i]}" ]] || { echo "FAIL: empty GPU token" >&2; exit 1; }
  for ((j=0; j<i; j++)); do
    [[ "${TOKENS[$i]}" != "${TOKENS[$j]}" ]] || { echo "FAIL: need exactly eight distinct GPU tokens" >&2; exit 1; }
  done
done
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
SOURCE_ROOT="${HISTORY_RESUME_SOURCE_ROOT:-$PROSPECTIVE_RUNS/s5-history-pilot/job-76866/gradient_history_s5}"
RUN_ROOT="${MORNING_RUN_ROOT:-$PROSPECTIVE_RUNS/s5-morning-followup/job-$ALLOCATION}"
[[ -d "$DATA_CACHE" ]] || { echo "FAIL: missing data cache" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: output already exists: $RUN_ROOT" >&2; exit 1; }
for seed in 301 302; do
  for file in checkpoint_epoch_09.msgpack metrics.jsonl history_protocol.json; do
    [[ -s "$SOURCE_ROOT/$seed/$file" ]] || { echo "FAIL: missing source $SOURCE_ROOT/$seed/$file" >&2; exit 1; }
  done
done
python3 - "$SOURCE_ROOT" "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys
source, out = (Path(p).resolve() for p in sys.argv[1:])
source_run = source.parent
if out == source_run or source_run in out.parents or out in source_run.parents:
    raise SystemExit("FAIL: output must not overlap the source run")
PY
echo "morning follow-up: allocation $ALLOCATION, commit $EXPECTED_COMMIT"
echo "  GPU tokens: ${TOKENS[*]}"
echo "  source H checkpoints: $SOURCE_ROOT (epoch9, read-only)"
echo "  output: $RUN_ROOT"
echo "  GPU0: resume H301 -> P alpha1/303; GPU1: resume H302 -> P alpha0.1/302"
echo "  GPU2..7: N304 D304 P304 N305 D305 P305; batch16, original 15-epoch schedule"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: ten tasks in eight queues; each fresh task profiles where applicable, then smokes, then trains"
  exit 0
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/inventories" "$RUN_ROOT/memory-profiles"
"$PY" -u - <<'PY'
import os, jax
devices = jax.devices()
assert len(devices) == 8 and all(d.platform == "gpu" for d in devices)
assert len(os.sched_getaffinity(0)) >= 32
assert not jax.config.read("jax_enable_x64")
print("Allocated devices:", devices, flush=True)
PY
JAX_PLATFORMS=cpu "$PY" -u - "$DATA_CACHE" "$RUN_ROOT/preflight.json" <<'PY'
import json, sys
import jax, numpy as np
from flax.traverse_util import flatten_dict
from experiments.s5_three_arm_full import runner as R
from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_history import runner as P
from experiments.s5_three_arm_full.data import validate_official_raw_cache
assert not jax.config.read("jax_enable_x64")
assert R.BATCH_SIZE == 16 and R.SEQ_LEN == 16000 and R.N_LAYERS == 6 and H.EPOCHS == 15
validate_official_raw_cache(sys.argv[1], ("train", "val"))
rows = []
for seed, alpha in ((302, .1), (303, 1.), (304, 1.), (305, 1.)):
    native = R.init_state("native_matched_s5", seed)
    nparts = flatten_dict(native.params)
    with P.configured(alpha):
        with H.registered_history(): proximal = R.init_state(P.ARM, seed)
    pparts = flatten_dict(proximal.params)
    assert nparts.keys() == pparts.keys()
    assert all(np.array_equal(np.asarray(v), np.asarray(pparts[k])) for k, v in nparts.items())
    nopt, popt = jax.tree_util.tree_flatten(native.opt_state), jax.tree_util.tree_flatten(proximal.opt_state)
    assert nopt[1] == popt[1]
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(nopt[0], popt[0]))
    if seed in (304, 305):
        dparts = flatten_dict(R.init_state("calibrated_prediction_s5", seed).params)
        assert all(np.array_equal(np.asarray(v), np.asarray(dparts[k])) for k, v in nparts.items())
    rows.append({"seed": seed, "alpha": alpha, "native_backbone_identical": True})
with open(sys.argv[2], "w") as f:
    json.dump({"tasks": rows, "batch": 16, "epochs": 15, "test_split_read": False}, f, indent=2)
print("preflight passed: data, protocol, native/P optimizer and native/D/P backbone", flush=True)
PY
printf 'commit=%s\njob=%s\nsource=%s\nresume_epoch=9\nbatch=16\nepochs=15\nGPU_tokens=%s\n' \
  "$EXPECTED_COMMIT" "$ALLOCATION" "$SOURCE_ROOT" "${TOKENS[*]}" > "$RUN_ROOT/run_metadata.txt"

run_logged() {
  local log="$1"; shift
  "$@" <&0 > "$log" 2>&1 &
  worker_child="$!"
  local status=0
  wait "$worker_child" || status=$?
  worker_child=""
  [[ "$status" == 0 ]] || { echo "FAIL: exit $status; inspect $log" >&2; return "$status"; }
}
prepare_task() {
  local label="$1"
  export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache/$label"
  export TMPDIR="$RUN_ROOT/tmp/$label"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
}
check_result() {
  local task="$1"
  [[ -f "$task/task_result.json" ]] || { echo "FAIL: missing result $task" >&2; return 1; }
  "$PY" - "$task/task_result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["epochs_requested"] == 15 and 1 <= r["selected_epoch"] <= 15
v = r["validation"]
print(f"FINISHED {r['code_identifier']}/{r['seed']}: selected epoch {r['selected_epoch']}; "
      f"accuracy {100*v['accuracy']:.4f}%; CE {v['cross_entropy']:.6f}", flush=True)
PY
}
resume_history() {
  local seed="$1" label="history-$1" out="$RUN_ROOT/history/gradient_history_s5/$1"
  prepare_task "$label" || return $?
  echo "RESUME H/$seed epochs10..15 on GPU $CUDA_VISIBLE_DEVICES"
  run_logged "$RUN_ROOT/logs/$label-resume.log" "$PY" -u -m experiments.s5_gradient_history.resume \
    --source-task "$SOURCE_ROOT/$seed" --out "$out" --epoch 9 --data-cache "$DATA_CACHE" || return $?
  check_result "$out" || return $?
}
fresh_task() {
  local arm="$1" seed="$2" alpha="$3" tag="$4"
  local label="$tag-$seed" relative="$arm/$seed"
  if [[ "$arm" == proximal_history_s5 ]]; then
    if [[ "$alpha" == 1 ]]; then relative="$arm/alpha1/$seed"; else relative="$arm/alpha0p1/$seed"; fi
  fi
  local out="$RUN_ROOT/fresh/$relative" smoke="$RUN_ROOT/smoke/$relative"
  prepare_task "$label"
  local common=(--arm "$arm" --seed "$seed" --alpha "$alpha" --proximal-ratio 0.5)
  if [[ "$arm" == proximal_history_s5 ]]; then
    echo "PROFILE $tag/$seed on GPU $CUDA_VISIBLE_DEVICES"
    run_logged "$RUN_ROOT/logs/$label-profile.log" "$PY" -u -m experiments.s5_morning_followup.runner \
      "${common[@]}" --profile --out "$RUN_ROOT/memory-profiles/$label.json"
  fi
  echo "SMOKE $tag/$seed on GPU $CUDA_VISIBLE_DEVICES"
  run_logged "$RUN_ROOT/logs/$label-smoke.log" "$PY" -u -m experiments.s5_morning_followup.runner \
    "${common[@]}" --data-cache "$DATA_CACHE" --out "$smoke" --smoke
  echo "TRAIN $tag/$seed (fresh initialization after matching GPU smoke)"
  run_logged "$RUN_ROOT/logs/$label-train.log" "$PY" -u -m experiments.s5_morning_followup.runner \
    "${common[@]}" --data-cache "$DATA_CACHE" --out "$out" --train --smoke-report "$smoke/smoke_result.json"
  check_result "$out"
}
worker() {
  local slot="$1"
  export CUDA_VISIBLE_DEVICES="${TOKENS[$slot]}"
  worker_child=""
  cleanup_worker() {
    if [[ -n "$worker_child" ]]; then kill -TERM "$worker_child" 2>/dev/null || true; wait "$worker_child" 2>/dev/null || true; fi
  }
  trap cleanup_worker EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  run_logged "$RUN_ROOT/logs/worker-$slot-device.log" "$PY" -u - <<'PY'
import jax
d = jax.devices()
assert len(d) == 1 and d[0].platform == "gpu"
print("Worker device:", d, flush=True)
PY
  run_logged "$RUN_ROOT/logs/worker-$slot-inventory.log" "$PY" -u -m experiments.s5_three_arm_full.gpu_telemetry \
    --out "$RUN_ROOT/inventories/worker-$slot.json" --label "morning_worker_$slot"
  case "$slot" in
    0)
      local history_status=0
      resume_history 301 || history_status=$?
      fresh_task proximal_history_s5 303 1 proximal_history_alpha1
      return "$history_status"
      ;;
    1)
      local history_status=0
      resume_history 302 || history_status=$?
      fresh_task proximal_history_s5 302 0.1 proximal_history_alpha0p1
      return "$history_status"
      ;;
    2) fresh_task native_matched_s5 304 1 native_matched_s5 ;;
    3) fresh_task calibrated_prediction_s5 304 1 calibrated_prediction_s5 ;;
    4) fresh_task proximal_history_s5 304 1 proximal_history_alpha1 ;;
    5) fresh_task native_matched_s5 305 1 native_matched_s5 ;;
    6) fresh_task calibrated_prediction_s5 305 1 calibrated_prediction_s5 ;;
    7) fresh_task proximal_history_s5 305 1 proximal_history_alpha1 ;;
  esac
}
PIDS=()
cleanup_workers() {
  local pid
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for slot in 0 1 2 3 4 5 6 7; do
  (worker "$slot") &
  PIDS[$slot]="$!"
  echo "started worker$slot GPU=${TOKENS[$slot]} pid=${PIDS[$slot]}"
done
failed=0
for slot in 0 1 2 3 4 5 6 7; do
  status=0; wait "${PIDS[$slot]}" || status=$?
  unset 'PIDS[slot]'
  echo "worker$slot exited $status"
  [[ "$status" == 0 ]] || failed=1
done
[[ "$failed" == 0 ]] || { echo "INCOMPLETE: one or more worker queues failed; artifacts preserved" >&2; exit 1; }
echo "Completed all ten morning tasks; selected-validation outputs: $RUN_ROOT"
echo "No finalizer ran; the test split was not read."
