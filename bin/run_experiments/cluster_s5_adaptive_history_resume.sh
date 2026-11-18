#!/usr/bin/env bash
# Restore full epoch-3 states, preserving the original 15-epoch experiment.
# A single exclusive allocation contains all four workers and the final audit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$HERE/cluster_env.sh"
cd "$REPO_ROOT"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned recovery commit}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: wrong commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "FAIL: modified checkout" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FAIL: missing Python: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in 0|false|False|FALSE|"") ;; *) echo "FAIL: x64 must be off" >&2; exit 1;; esac
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
STAGE="${STAGE:-train}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$STAGE" == train || "$STAGE" == preflight ]] || { echo "FAIL: invalid STAGE" >&2; exit 1; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric Slurm job required" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
DISTINCT=()
for token in "${TOKENS[@]:-}"; do
  token="${token//[[:space:]]/}"
  [[ -n "$token" ]] || continue
  duplicate=0
  for previous in "${DISTINCT[@]:-}"; do [[ "$previous" == "$token" ]] && duplicate=1; done
  [[ "$duplicate" != 0 ]] || DISTINCT+=("$token")
done
[[ "${#DISTINCT[@]}" == 8 ]] || { echo "FAIL: need exactly eight allocated GPU tokens" >&2; exit 1; }
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
SOURCE_ROOT="${AP_RESUME_SOURCE_ROOT:-$PROSPECTIVE_RUNS/s5-adaptive-proximal-history/job-77021}"
RUN_ROOT="$PROSPECTIVE_RUNS/s5-adaptive-history-resume/job-$ALLOCATION"
[[ -d "$DATA_CACHE" && -d "$SOURCE_ROOT" ]] || { echo "FAIL: source run or data cache missing" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: run root already exists: $RUN_ROOT" >&2; exit 1; }
ARMS=(adaptive_proximal_history_s5 adaptive_proximal_history_s5 adaptive_proximal_history_control_s5 adaptive_proximal_history_control_s5)
SEEDS=(301 302 301 302)
for i in 0 1 2 3; do
  [[ -s "$SOURCE_ROOT/${ARMS[$i]}/${SEEDS[$i]}/checkpoint_epoch_03.msgpack" ]] \
    || { echo "FAIL: missing source epoch-3 checkpoint for ${ARMS[$i]}/${SEEDS[$i]}" >&2; exit 1; }
done
echo "adaptive history recovery: allocation $ALLOCATION, stage $STAGE"
echo "  source $SOURCE_ROOT; epoch 3 -> 15; original state, schedule and selection"
echo "  output $RUN_ROOT; recovery commit $EXPECTED_COMMIT"
echo "  all eight GPUs reserved exclusively; four workers use ${DISTINCT[*]:0:4}"
for i in 0 1 2 3; do echo "  GPU ${DISTINCT[$i]}: ${ARMS[$i]} seed ${SEEDS[$i]}"; done
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: verify node exclusivity; four restored GPU preflights gate all continuations"
  echo "DRY_RUN: final CPU audit stays inside this same allocation"
  exit 0
fi
[[ "$(hostname -s)" == pgi14-gpu1 ]] || { echo "FAIL: recovery must run on pgi14-gpu1" >&2; exit 1; }
# Retain scheduler evidence and fail closed if the exclusive request was overridden.
JOB_INFO="$(scontrol show job -o "$ALLOCATION")"
ISOLATION_CONFIRMED=0
for field in $JOB_INFO; do
  case "$field" in
    OverSubscribe=NO|Shared=0|Exclusive=NODE) ISOLATION_CONFIRMED=1 ;;
    OverSubscribe=*|Shared=*|Exclusive=*)
      echo "FAIL: scheduler reports nonexclusive allocation ($field)" >&2
      printf '%s\n' "$JOB_INFO" >&2
      exit 1 ;;
  esac
done
if [[ "$ISOLATION_CONFIRMED" != 1 ]]; then
  echo "FAIL: scheduler did not confirm node-exclusive allocation; inspect scontrol show job $ALLOCATION" >&2
  printf '%s\n' "$JOB_INFO" >&2
  exit 1
fi
OTHER_JOBS="$(squeue -h -w pgi14-gpu1 -t RUNNING,COMPLETING,CONFIGURING,SUSPENDED -o '%i')"
while IFS= read -r job; do
  job="${job//[[:space:]]/}"
  [[ -z "$job" || "$job" == "$ALLOCATION" ]] || {
    echo "FAIL: another active job $job is on the node" >&2
    printf '%s\n' "$OTHER_JOBS" >&2
    exit 1
  }
done <<< "$OTHER_JOBS"
mkdir -p "$RUN_ROOT"/{logs,status,preflight}
printf '%s\n' "$JOB_INFO" > "$RUN_ROOT/slurm_job.txt"
printf '%s\n' "$OTHER_JOBS" > "$RUN_ROOT/slurm_node_jobs_at_start.txt"
JAX_PLATFORMS=cuda "$PY" -u - <<'PY'
import os
import jax
devices = jax.devices()
print("reserved devices:", devices, flush=True)
assert len(devices) == 8 and all(d.platform == "gpu" for d in devices)
assert not jax.config.read("jax_enable_x64")
assert len(os.sched_getaffinity(0)) >= 32
PY
printf 'recovery_commit=%s\nsource_commit=4372acd76c9f53e8d83d4f929d5bbd4745ace3ad\nsource_root=%s\nsource_epoch=3\nfinal_epoch=15\njob=%s\nnode=pgi14-gpu1\nexclusive=true\nreserved_gpu_tokens=%s\nworker_count=4\nstage=%s\ntest_split_read=false\n' \
  "$EXPECTED_COMMIT" "$SOURCE_ROOT" "$ALLOCATION" "${DISTINCT[*]}" "$STAGE" > "$RUN_ROOT/run_metadata.txt"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
prepare_worker() {
  local i="$1"
  export CUDA_VISIBLE_DEVICES="${DISTINCT[$i]}" JAX_PLATFORMS=cuda
  export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache/${ARMS[$i]}/${SEEDS[$i]}"
  export TMPDIR="$RUN_ROOT/tmp/${ARMS[$i]}/${SEEDS[$i]}"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
}
preflight_worker() {
  local i="$1" arm="${ARMS[$1]}" seed="${SEEDS[$1]}"
  local out="$RUN_ROOT/preflight/$arm/$seed" status="$RUN_ROOT/status/$arm-$seed"
  prepare_worker "$i"
  echo "[$arm/$seed] restored epoch-3 GPU preflight"
  "$PY" -u -m experiments.s5_adaptive_proximal_history.resume \
    --arm "$arm" --source-task "$SOURCE_ROOT/$arm/$seed" --out "$out" \
    --data-cache "$DATA_CACHE" --epoch 3 --preflight-only \
    > "$RUN_ROOT/logs/$arm-$seed-preflight.log" 2>&1 \
    || { echo FAIL:preflight > "$status"; return 1; }
  [[ -s "$out/resume_preflight.json" ]] || { echo FAIL:no_preflight_report > "$status"; return 1; }
  "$PY" - "$out/resume_preflight.json" "$SOURCE_ROOT/$arm/$seed" "$arm" "$seed" <<'PY' \
    || { echo FAIL:preflight_binding > "$status"; return 1; }
import hashlib, json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
source = Path(sys.argv[2]).resolve()
assert report["status"] == "RESUME_PREFLIGHT_PASS" and report["backend"] == "gpu"
assert report["arm"] == sys.argv[3] and report["seed"] == int(sys.argv[4])
assert Path(report["source_task"]).resolve() == source and report["source_epoch"] == 3
assert report["source_unchanged"] is True
digest = hashlib.sha256((source / "checkpoint_epoch_03.msgpack").read_bytes()).hexdigest()
assert report["source_checkpoint_sha256"] == digest
PY
  echo PREFLIGHT_PASS > "$status"
}
echo "STAGE 1: all four restored GPU preflights"
for i in 0 1 2 3; do preflight_worker "$i" & PIDS+=("$!"); done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
PIDS=()
for i in 0 1 2 3; do
  [[ "$(cat "$RUN_ROOT/status/${ARMS[$i]}-${SEEDS[$i]}" 2>/dev/null || true)" == PREFLIGHT_PASS ]] || FAILED=1
done
if [[ "$FAILED" != 0 ]]; then echo "FAIL: restored preflight failed; NO continuation started" >&2; exit 1; fi
echo "All four restored GPU preflights passed."
if [[ "$STAGE" == preflight ]]; then echo "Completed preflight-only stage: $RUN_ROOT"; exit 0; fi
resume_worker() {
  local i="$1" arm="${ARMS[$1]}" seed="${SEEDS[$1]}"
  local out="$RUN_ROOT/$arm/$seed" status="$RUN_ROOT/status/$arm-$seed"
  prepare_worker "$i"
  echo "[$arm/$seed] resuming epochs 4..15 from the complete epoch-3 state"
  "$PY" -u -m experiments.s5_adaptive_proximal_history.resume \
    --arm "$arm" --source-task "$SOURCE_ROOT/$arm/$seed" --out "$out" \
    --data-cache "$DATA_CACHE" --epoch 3 \
    > "$RUN_ROOT/logs/$arm-$seed-resume.log" 2>&1 \
    || { echo FAIL:resume > "$status"; return 1; }
  [[ -s "$out/task_result.json" ]] || { echo FAIL:no_task_result > "$status"; return 1; }
  echo TRAINED > "$status"
  echo "[$arm/$seed] completed"
}
echo "STAGE 2: four full-state continuations"
for i in 0 1 2 3; do resume_worker "$i" & PIDS+=("$!"); done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
PIDS=()
for i in 0 1 2 3; do
  [[ "$(cat "$RUN_ROOT/status/${ARMS[$i]}-${SEEDS[$i]}" 2>/dev/null || true)" == TRAINED ]] || FAILED=1
done
if [[ "$FAILED" == 0 ]]; then
  audit_args=()
  for i in 0 1 2 3; do audit_args+=(--task "$RUN_ROOT/${ARMS[$i]}/${SEEDS[$i]}"); done
  echo "All four trainings completed; auditing inside the same allocation."
  if JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PY" -u -m experiments.s5_adaptive_proximal_history.audit \
      "${audit_args[@]}" --out "$RUN_ROOT/audit" > "$RUN_ROOT/logs/audit.log" 2>&1 \
      && [[ -s "$RUN_ROOT/audit/block_audit.json" && -s "$RUN_ROOT/audit/summary.txt" ]]; then
    echo PASS > "$RUN_ROOT/status/audit"
  else
    echo FAIL > "$RUN_ROOT/status/audit"
    echo "AUDIT FAILED after all four trainings completed; saved results remain. Rerun only the audit." >&2
    FAILED=1
  fi
else
  echo SKIPPED:incomplete_training > "$RUN_ROOT/status/audit"
fi
# This reporting step cannot change the training or audit exit status.
"$PY" - "$RUN_ROOT" <<'PY' || echo "WARNING: summary formatting failed; inspect task_result.json files" >&2
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print("arm seed status selected_epoch validation_accuracy cross_entropy")
for arm in ("adaptive_proximal_history_s5", "adaptive_proximal_history_control_s5"):
    for seed in (301, 302):
        status = root / "status" / f"{arm}-{seed}"
        label = status.read_text().strip() if status.exists() else "MISSING"
        path = root / arm / str(seed) / "task_result.json"
        if not path.exists():
            print(arm, seed, label, "n/a n/a n/a")
            continue
        try:
            r = json.loads(path.read_text())
            v = r.get("validation") or {}
            print(arm, seed, label, r.get("selected_epoch", "n/a"),
                  v.get("accuracy", "n/a"), v.get("cross_entropy", "n/a"))
        except (ValueError, OSError, AttributeError):
            print(arm, seed, label, "unreadable")
audit = root / "status/audit"
print("AUDIT:", audit.read_text().strip() if audit.exists() else "MISSING")
PY
if [[ "$FAILED" != 0 ]]; then echo "INCOMPLETE: inspect $RUN_ROOT/status and logs" >&2; exit 1; fi
echo "Completed adaptive history recovery: $RUN_ROOT"
