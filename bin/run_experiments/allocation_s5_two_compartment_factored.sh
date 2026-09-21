#!/usr/bin/env bash
# THE TWO-COMPARTMENT GENERALIZED ARM, 15 EPOCHS, INSIDE ONE EXISTING
# SLURM ALLOCATION, EVALUATED BY THE FACTORED PARALLEL SCAN.
#
# Run from inside the interactive allocation that already holds the node.
# Every task is a DIRECT CHILD PROCESS of this shell: no sbatch, no salloc,
# no srun, so no child job or step can terminate while another is training.
#
# WHAT IS AND IS NOT CHANGED. The equation, the coefficients
# (generalized_coefficients), the parameterization (response_mass_gamma),
# the initialization, the output semantics and the arm name are exactly as
# they were. Only the SCAN changes, from scan_companion_sequential to the
# factored two-scalar-scan form, which was verified against the sequential
# oracle to 1.83e-13 relative in float64 over the whole production
# inventory (1152 modes, seeds 301-303, length 16,000).
#
# WHY. The sequential scan measured 17.565 s/step, 117.7 hours for a
# three-seed wave. The factored scan measures 0.218 s/step, 80.5x faster.
#
# ONE SEED PER GPU, NEVER TWO. The measured peak is 12.885 GiB against the
# RTX 3090's 24.0 GiB, so two concurrent seeds on one card would need 25.8
# GiB and fail. Seeds are therefore scheduled in waves no larger than the
# number of DISTINCT visible GPU tokens.
#
#   DRY_RUN=1 EXPECTED_COMMIT=<sha> bash bin/run_experiments/allocation_s5_two_compartment_factored.sh
# prints the plan and every child command without starting anything.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"
REPO_ROOT="$(pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"

ARM="generalized_prospective_s5"
IMPLEMENTATION="${IMPLEMENTATION:-factored}"
EPOCHS="${EPOCHS:-15}"
RUN_SEEDS=(301 302 303)
#: measured on an RTX 3090 at batch 16, length 16,000, full step
PEAK_GIB_MEASURED=12.885
CARD_GIB=24.0

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
ALLOCATION="${SLURM_JOB_ID:-}"
if [[ ! "$ALLOCATION" =~ ^[0-9]+$ ]]; then
  echo "FAIL: SLURM_JOB_ID='$ALLOCATION' is not a numeric job id;" >&2
  echo "      run this inside the existing interactive allocation." >&2
  exit 1
fi
if [[ "$IMPLEMENTATION" == "sequential" ]]; then
  echo "NOTE: IMPLEMENTATION=sequential reproduces the 117-hour run." >&2
fi

# the DISTINCT visible GPU tokens of this allocation
IFS=',' read -r -a RAW_TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
TOKENS=()
for token in ${RAW_TOKENS[@]+"${RAW_TOKENS[@]}"}; do
  [[ -z "$token" ]] && continue
  duplicate=0
  for seen in ${TOKENS[@]+"${TOKENS[@]}"}; do
    [[ "$seen" == "$token" ]] && duplicate=1
  done
  [[ "$duplicate" == "0" ]] && TOKENS+=("$token")
done
if [[ "${#TOKENS[@]}" -lt 1 ]]; then
  echo "FAIL: no visible GPU tokens: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'" >&2
  exit 1
fi
WAVE_SIZE="${#TOKENS[@]}"
if [[ "$WAVE_SIZE" -gt "${#RUN_SEEDS[@]}" ]]; then
  WAVE_SIZE="${#RUN_SEEDS[@]}"
fi

DRY_RUN="${DRY_RUN:-0}"
OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-two-compartment-factored}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$OUT_ROOT/$STAMP"
SMOKE_ROOT="$RUN_ROOT/smoke"

echo "two-compartment generalized arm, direct children of allocation $ALLOCATION"
echo "  arm:            $ARM"
echo "  scan:           $IMPLEMENTATION"
echo "  epochs:         $EPOCHS (one warm-up, cosine over the other $((EPOCHS - 1)))"
echo "  seeds:          ${RUN_SEEDS[*]}"
echo "  gpu tokens:     ${TOKENS[*]}   (wave size $WAVE_SIZE, one seed per GPU)"
echo "  measured peak:  $PEAK_GIB_MEASURED GiB of $CARD_GIB GiB per card"
echo "  branch:         $(git rev-parse --abbrev-ref HEAD)"
echo "  commit:         $(git rev-parse HEAD)"
echo "  data cache:     $DATA_CACHE"
echo "  run root:       $RUN_ROOT"
echo "  monitor:        tail -n 3 -f $RUN_ROOT/*/logs/train.log"

if [[ "$DRY_RUN" != "1" ]]; then
  "$PY" - "$DATA_CACHE" <<'PYEOF'
import sys
from experiments.s5_three_arm_full.data import validate_official_raw_cache
validate_official_raw_cache(sys.argv[1], ("train", "val"))
print(f"validated official raw cache: {sys.argv[1]}")
PYEOF
  mkdir -p "$RUN_ROOT"
  printf 'authoritative_commit=%s\nbranch=%s\narm=%s\nscan_implementation=%s\nepochs_requested=%s\nseeds=%s\nallocation=%s\ngpu_tokens=%s\nwave_size=%s\ndata_cache=%s\n' \
    "$EXPECTED_COMMIT" "$(git rev-parse --abbrev-ref HEAD)" "$ARM" \
    "$IMPLEMENTATION" "$EPOCHS" "${RUN_SEEDS[*]}" "$ALLOCATION" \
    "${TOKENS[*]}" "$WAVE_SIZE" "$DATA_CACHE" > "$RUN_ROOT/run_metadata.txt"
fi

# ------------------------------------------------------------- children ----
CHILD_PIDS=(); CHILD_LABELS=(); CHILD_DIRS=(); CHILD_SEEDS=()

start_child() {   # seed gpu_token root [--smoke]
  local seed="$1" token="$2" root="$3" smoke="${4:-}"
  local task_root="$root/$seed"
  local label="$ARM/$seed@gpu$token"
  local runner_args=(--data-cache "$DATA_CACHE" --out "$task_root"
                     --arm "$ARM" --seed "$seed" --epochs "$EPOCHS"
                     --implementation "$IMPLEMENTATION")
  [[ -n "$smoke" ]] && runner_args+=("$smoke")
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN child $label"
    echo "    CUDA_VISIBLE_DEVICES=$token  TMPDIR=$task_root/tmp"
    echo "    $PY -u -m experiments.s5_three_arm_full.runner ${runner_args[*]}"
    return 0
  fi
  mkdir -p "$task_root/tmp" "$task_root/jax-cache" "$task_root/logs"
  (
    export CUDA_VISIBLE_DEVICES="$token"
    export TMPDIR="$task_root/tmp"
    export JAX_COMPILATION_CACHE_DIR="$task_root/jax-cache"
    "$PY" -u -m experiments.s5_three_arm_full.gpu_telemetry \
      --out "$task_root/gpu_telemetry.json" --label "pre_jax_gpu$token" \
      > "$task_root/logs/gpu_telemetry.log" 2>&1
    exec "$PY" -u -m experiments.s5_three_arm_full.runner \
      "${runner_args[@]}" > "$task_root/logs/train.log" 2>&1
  ) &
  CHILD_PIDS+=("$!"); CHILD_LABELS+=("$label")
  CHILD_DIRS+=("$task_root"); CHILD_SEEDS+=("$seed")
  echo "started $label as pid $! (log $task_root/logs/train.log)"
}

WAVE_STATUS=()
wait_for_wave() {
  WAVE_STATUS=(); local index=0 status
  for pid in ${CHILD_PIDS[@]+"${CHILD_PIDS[@]}"}; do
    status=0; wait "$pid" || status=$?
    WAVE_STATUS+=("$status")
    echo "finished ${CHILD_LABELS[$index]} pid $pid exit $status"
    index=$((index + 1))
  done
}
reset_wave() { CHILD_PIDS=(); CHILD_LABELS=(); CHILD_DIRS=(); CHILD_SEEDS=(); }

# ------------------------------------------------------ 1. smoke gate ------
echo
echo "STAGE 1: smoke on ${TOKENS[0]} (gates training)"
reset_wave
start_child "${RUN_SEEDS[0]}" "${TOKENS[0]}" "$SMOKE_ROOT" --smoke
if [[ "$DRY_RUN" != "1" ]]; then
  wait_for_wave
  for status in ${WAVE_STATUS[@]+"${WAVE_STATUS[@]}"}; do
    if [[ "$status" -ne 0 ]]; then
      echo "ABORT: the smoke child exited nonzero; artifacts under" >&2
      echo "       $SMOKE_ROOT. Training was NOT started." >&2
      exit 1
    fi
  done
  # the smoke must have used the scan we asked for, not the default
  "$PY" - "$SMOKE_ROOT/${RUN_SEEDS[0]}/production_check.json" "$IMPLEMENTATION" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as handle:
    record = json.load(handle)
actual, wanted = record.get("scan_implementation"), sys.argv[2]
if actual != wanted:
    raise SystemExit(f"FAIL: smoke ran scan {actual!r}, expected {wanted!r}")
print(f"smoke used the requested scan: {actual}")
PYEOF
  echo "smoke passed"
fi

# ------------------------------------------------------ 2. the seeds -------
UNEXPECTED=()
index=0
while [[ "$index" -lt "${#RUN_SEEDS[@]}" ]]; do
  echo
  echo "STAGE 2: seeds $((index + 1))..$((index + WAVE_SIZE)) of ${#RUN_SEEDS[@]}, one per GPU"
  reset_wave
  slot=0
  while [[ "$slot" -lt "$WAVE_SIZE" && $((index + slot)) -lt "${#RUN_SEEDS[@]}" ]]; do
    start_child "${RUN_SEEDS[$((index + slot))]}" "${TOKENS[$slot]}" "$RUN_ROOT"
    slot=$((slot + 1))
  done
  if [[ "$DRY_RUN" != "1" ]]; then
    wait_for_wave
    position=0
    for status in ${WAVE_STATUS[@]+"${WAVE_STATUS[@]}"}; do
      if [[ ! -f "${CHILD_DIRS[$position]}/task_result.json" ]]; then
        UNEXPECTED+=("${CHILD_LABELS[$position]} (exit $status)")
      fi
      position=$((position + 1))
    done
    if [[ "${#UNEXPECTED[@]}" -gt 0 ]]; then
      echo "ABORT: ${UNEXPECTED[*]}" >&2
      echo "       Artifacts preserved under $RUN_ROOT." >&2
      exit 1
    fi
  fi
  index=$((index + WAVE_SIZE))
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN: nothing started"
  exit 0
fi

echo
echo "run root: $RUN_ROOT"
echo "per-seed results: $RUN_ROOT/*/task_result.json"
echo
echo "NO FINALIZER WAS RUN. The finalizer is the only reader of the test"
echo "split and it compares all three arms; running it on a single arm is a"
echo "separate, explicit decision."
