#!/usr/bin/env bash
# THREE-GPU, 15-EPOCH S5 RUN INSIDE ONE EXISTING SLURM ALLOCATION.
#
# Run this from inside the interactive allocation that already holds the node
# (for example job 66760 on pgi15-gpu2, four RTX 3090s). Every task is a
# DIRECT CHILD PROCESS of this shell: no sbatch, no salloc, no srun, hence no
# child job or step can terminate while another task is training — the
# condition that killed job 66684.
#
# Topology: seeds 301, 302 and 303 run concurrently on the first three visible
# GPU tokens, one seed per GPU; the fourth token is left unused. The arms run
# in three sequential waves, in scientific order, and every child of a wave is
# waited for before the next wave starts.
#
#   Native S5  ->  Zucchet prospective dynamics — finite-difference
#   realization  ->  generalized prospective dynamics (M,gamma,T) —
#   finite-difference realization
#
# A mandatory concurrent topology smoke runs first and gates the training.
#
#   DRY_RUN=1 bash bin/run_experiments/allocation_s5_three_arm_three_gpu.sh
# prints the plan and every child command without starting anything.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"
REPO_ROOT="$(pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"

#: this run trains for FIFTEEN epochs; the runner still defaults to 40
EPOCHS=15
#: the three arms, in scientific reporting order, and their failure policy
WAVE_ARMS=(native_matched_s5 zucchet_prospective_s5 generalized_prospective_s5)
#: Zucchet is the declared negative control: its finite-difference recurrence
#: is intentionally unrepaired and a numerical failure is an EXPECTED result,
#: recorded as failure.json and carried into the finalizer. A Native or
#: generalized failure is unexpected and stops the run.
EXPECTED_FAILURE_ARM="zucchet_prospective_s5"
RUN_SEEDS=(301 302 303)

# ---------------------------------------------------------------- guards ---
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the authoritative commit}"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "FAIL: HEAD $(git rev-parse HEAD) is not $EXPECTED_COMMIT" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short >&2
  echo "FAIL: refusing dirty worktree: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "FAIL: interpreter is not executable: $PY" >&2
  exit 1
fi
DATA_CACHE="${S5_THREE_ARM_DATA:-/Users/durso/s5-runs/sc10_official_cache}"
if [[ ! -d "$DATA_CACHE" ]]; then
  echo "FAIL: data cache is not a directory: $DATA_CACHE" >&2
  exit 1
fi
# this launcher must run INSIDE the allocation, never outside it
ALLOCATION="${SLURM_JOB_ID:-}"
if [[ ! "$ALLOCATION" =~ ^[0-9]+$ ]]; then
  echo "FAIL: SLURM_JOB_ID='$ALLOCATION' is not a numeric job id;" >&2
  echo "      run this inside the existing interactive allocation." >&2
  exit 1
fi
# the visible GPU tokens of THIS allocation; we use the first three
IFS=',' read -r -a GPU_TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ "${#GPU_TOKENS[@]}" -lt 4 ]]; then
  echo "FAIL: expected at least 4 visible GPU tokens, got" \
       "'${CUDA_VISIBLE_DEVICES:-<unset>}'" >&2
  exit 1
fi
USE_TOKENS=("${GPU_TOKENS[0]}" "${GPU_TOKENS[1]}" "${GPU_TOKENS[2]}")
if [[ "${USE_TOKENS[0]}" == "${USE_TOKENS[1]}" || \
      "${USE_TOKENS[0]}" == "${USE_TOKENS[2]}" || \
      "${USE_TOKENS[1]}" == "${USE_TOKENS[2]}" ]]; then
  echo "FAIL: the three assigned GPU tokens are not distinct:" \
       "${USE_TOKENS[*]}" >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-three-arm-15epoch}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$OUT_ROOT/$STAMP"
SMOKE_ROOT="$RUN_ROOT/topology_smoke"

if [[ "$DRY_RUN" != "1" ]]; then
  "$PY" - "$DATA_CACHE" <<'PYEOF'
import sys
from experiments.s5_three_arm_full.data import validate_official_raw_cache
validate_official_raw_cache(sys.argv[1], ("train", "val"))
print(f"validated official raw cache: {sys.argv[1]}")
PYEOF
  mkdir -p "$RUN_ROOT"
  printf 'authoritative_commit=%s\nbranch=%s\ndata_cache=%s\nepochs_requested=%s\nallocation=%s\ngpu_tokens=%s\nunused_gpu_tokens=%s\n' \
    "$EXPECTED_COMMIT" "$(git rev-parse --abbrev-ref HEAD)" "$DATA_CACHE" \
    "$EPOCHS" "$ALLOCATION" "${USE_TOKENS[*]}" \
    "${GPU_TOKENS[*]:3}" > "$RUN_ROOT/run_metadata.txt"
fi

echo "three-GPU 15-epoch S5 run, direct children of allocation $ALLOCATION"
echo "  1. Native S5"
echo "  2. Zucchet prospective dynamics — finite-difference realization"
echo "  3. generalized prospective dynamics (M,gamma,T) — finite-difference realization"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse HEAD)"
echo "data cache: $DATA_CACHE"
echo "epochs_requested: $EPOCHS (one warm-up epoch, cosine over the other $((EPOCHS - 1)))"
echo "gpu tokens in use: ${USE_TOKENS[*]}   unused: ${GPU_TOKENS[*]:3}"
echo "run root: $RUN_ROOT"
echo "monitor:  tail -n 3 -f $RUN_ROOT/*/*/logs/train.log   # and, per epoch:"
echo "          tail -n 2 $RUN_ROOT/*/*/metrics.jsonl; nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv"

# ------------------------------------------------------------- children ----
CHILD_PIDS=()
CHILD_LABELS=()
CHILD_DIRS=()

start_child() {   # arm seed gpu_token root [--smoke]
  local arm="$1" seed="$2" token="$3" root="$4" smoke="${5:-}"
  local task_root="$root/$arm/$seed"
  local label="$arm/$seed@gpu$token"
  local runner_args=(--data-cache "$DATA_CACHE" --out "$task_root"
                     --arm "$arm" --seed "$seed" --epochs "$EPOCHS")
  if [[ -n "$smoke" ]]; then
    runner_args+=("$smoke")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN child $label"
    echo "    CUDA_VISIBLE_DEVICES=$token"
    echo "    TMPDIR=$task_root/tmp"
    echo "    JAX_COMPILATION_CACHE_DIR=$task_root/jax-cache"
    echo "    log=$task_root/logs/train.log"
    echo "    $PY -u -m experiments.s5_three_arm_full.gpu_telemetry --out $task_root/gpu_telemetry.json --label pre_jax_gpu$token"
    echo "    $PY -u -m experiments.s5_three_arm_full.runner ${runner_args[*]}"
    return 0
  fi
  mkdir -p "$task_root/tmp" "$task_root/jax-cache" "$task_root/logs"
  (
    # one GPU per child, and private scratch and compilation cache so no two
    # children share a path
    export CUDA_VISIBLE_DEVICES="$token"
    export TMPDIR="$task_root/tmp"
    export JAX_COMPILATION_CACHE_DIR="$task_root/jax-cache"
    # PRE-JAX telemetry, written before any JAX import so it survives a kill
    "$PY" -u -m experiments.s5_three_arm_full.gpu_telemetry \
      --out "$task_root/gpu_telemetry.json" \
      --label "pre_jax_gpu$token" \
      > "$task_root/logs/gpu_telemetry.log" 2>&1
    exec "$PY" -u -m experiments.s5_three_arm_full.runner \
      "${runner_args[@]}" > "$task_root/logs/train.log" 2>&1
  ) &
  CHILD_PIDS+=("$!")
  CHILD_LABELS+=("$label")
  CHILD_DIRS+=("$task_root")
  echo "started $label as pid $! (log $task_root/logs/train.log)"
}

#: exit status of each child of the wave just awaited, by index
WAVE_STATUS=()

wait_for_wave() {
  WAVE_STATUS=()
  local index=0 status
  for pid in "${CHILD_PIDS[@]}"; do
    status=0
    wait "$pid" || status=$?
    WAVE_STATUS+=("$status")
    echo "finished ${CHILD_LABELS[$index]} pid $pid exit $status"
    index=$((index + 1))
  done
}

reset_wave() {
  CHILD_PIDS=()
  CHILD_LABELS=()
  CHILD_DIRS=()
}

# ------------------------------------------------- 1. concurrent smoke -----
echo
echo "STAGE 1: concurrent topology smoke on ${USE_TOKENS[*]} (gates training)"
reset_wave
start_child native_matched_s5 301 "${USE_TOKENS[0]}" "$SMOKE_ROOT" --smoke
start_child generalized_prospective_s5 301 "${USE_TOKENS[1]}" "$SMOKE_ROOT" --smoke
if [[ "${SMOKE_THIRD:-1}" == "1" ]]; then
  start_child native_matched_s5 302 "${USE_TOKENS[2]}" "$SMOKE_ROOT" --smoke
fi

SMOKE_DIRS=("${CHILD_DIRS[@]}")
if [[ "$DRY_RUN" != "1" ]]; then
  wait_for_wave
  for status in "${WAVE_STATUS[@]}"; do
    if [[ "$status" -ne 0 ]]; then
      echo "ABORT: a smoke child exited nonzero; artifacts preserved under" >&2
      echo "       $SMOKE_ROOT. Full training was NOT started." >&2
      exit 1
    fi
  done
  # artifact-level gate: SMOKE_PASS with two steps and a restored checkpoint,
  # no failure.json, and a RESOLVED, DISTINCT physical GPU per child
  if ! "$PY" -u -m experiments.s5_three_arm_full.topology_report \
        --require-pass --require-distinct-gpus "${SMOKE_DIRS[@]}" \
        | tee "$SMOKE_ROOT/topology_report.json"; then
    echo "ABORT: the concurrent topology smoke did not pass its artifact" >&2
    echo "       gate; artifacts preserved under $SMOKE_ROOT. Full training" >&2
    echo "       was NOT started." >&2
    exit 1
  fi
  echo "topology smoke passed: concurrent, distinct physical GPUs"
fi

# ------------------------------------------------- 2. the three waves ------
UNEXPECTED_FAILURES=()
EXPECTED_FAILURES=()
wave_number=1
for arm in "${WAVE_ARMS[@]}"; do
  echo
  echo "STAGE 2.$wave_number: $arm, seeds ${RUN_SEEDS[*]} concurrently"
  reset_wave
  index=0
  for seed in "${RUN_SEEDS[@]}"; do
    start_child "$arm" "$seed" "${USE_TOKENS[$index]}" "$RUN_ROOT"
    index=$((index + 1))
  done
  if [[ "$DRY_RUN" == "1" ]]; then
    wave_number=$((wave_number + 1))
    continue
  fi
  # every child of this wave is awaited before the next wave starts
  wait_for_wave
  index=0
  for status in "${WAVE_STATUS[@]}"; do
    task_root="${CHILD_DIRS[$index]}"
    label="${CHILD_LABELS[$index]}"
    if [[ -f "$task_root/task_result.json" ]]; then
      :
    elif [[ "$arm" == "$EXPECTED_FAILURE_ARM" && -f "$task_root/failure.json" ]]; then
      # the declared negative control failed numerically, as it may: this is a
      # scientific result, it is preserved, and it does not stop the run
      EXPECTED_FAILURES+=("$label")
      echo "expected numerical failure recorded: $label"
    else
      UNEXPECTED_FAILURES+=("$label (exit $status)")
    fi
    index=$((index + 1))
  done
  if [[ "${#UNEXPECTED_FAILURES[@]}" -gt 0 ]]; then
    echo "ABORT: unexpected failure in wave $arm: ${UNEXPECTED_FAILURES[*]}" >&2
    echo "       All artifacts are preserved under $RUN_ROOT." >&2
    echo "       Later waves and the finalizer were NOT run." >&2
    exit 1
  fi
  wave_number=$((wave_number + 1))
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN finalizer (after every arm and seed is terminal):"
  echo "    $PY -u -m experiments.s5_three_arm_full.finalize --task-root $RUN_ROOT --data-cache $DATA_CACHE --out $RUN_ROOT/final"
  echo "DRY_RUN: nothing started"
  exit 0
fi

# -------------------------------------------------- 3. finalize once -------
echo
echo "STAGE 3: finalizer (the ONLY reader of the test split)"
mkdir -p "$RUN_ROOT/final/tmp" "$RUN_ROOT/final/jax-cache"
(
  export CUDA_VISIBLE_DEVICES="${USE_TOKENS[0]}"
  export TMPDIR="$RUN_ROOT/final/tmp"
  export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/final/jax-cache"
  "$PY" -u -m experiments.s5_three_arm_full.finalize \
    --task-root "$RUN_ROOT" \
    --data-cache "$DATA_CACHE" \
    --out "$RUN_ROOT/final"
) 2>&1 | tee "$RUN_ROOT/final/finalize.log"

echo
echo "run root: $RUN_ROOT"
if [[ "${#EXPECTED_FAILURES[@]}" -gt 0 ]]; then
  echo "expected numerical failures (Zucchet negative control): ${EXPECTED_FAILURES[*]}"
fi
echo "results: $RUN_ROOT/final/results.json"
