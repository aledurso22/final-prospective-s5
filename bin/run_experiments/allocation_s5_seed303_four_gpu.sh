#!/usr/bin/env bash
# TIMESCALE STUDY: N / A / C / D x SEED 303 -- FOUR TASKS, FOUR GPUs.
#
#   N  native_matched_s5            native S5
#   A  generalized_prospective_s5   the ORIGINAL generalized constructor (k=T)
#   C  calibrated_old_s5            k=gamma, correction at the previous state
#   D  calibrated_prediction_s5     k=gamma, correction at the prediction y_t
#
# Shared protocol, unchanged: runner.train_arm, batch 16, 15 epochs (one
# warm-up epoch, cosine decay over the other 14), the same data cache, the
# same split, and the same selection rule (best validation accuracy, then
# lower validation cross-entropy, then the earlier epoch). The test split is
# NEVER read here, and no finalizer runs. Every arm starts fresh from
# runner.init_state(arm, seed) in production float32. A, C and D share one
# initialization exactly, and N shares its backbone; this is checked before
# anything trains.
#
# Run it from inside an allocation that holds >= 4 GPUs: every task is a
# DIRECT CHILD of this shell. There is no srun and no nested sbatch.
#
#   STAGE=smoke  pre-flight + the 4-task smoke gate only
#   STAGE=train  pre-flight + smoke gate + training (the default)
#   DRY_RUN=1    print the plan and every child command; start nothing
#
# OPTIMIZATIONS, none of which changes a training computation:
#   * all four smokes run at once, then all four trainings, one per GPU;
#   * each task's persistent JAX compilation cache is shared between its own
#     smoke and its own training run, so training reuses the smoke's compiles;
#   * per-step telemetry is fused into one jitted call and one device->host
#     transfer, and logs are buffered. Every smoke re-measures the fused path
#     against the original eager path on its GPU and gates on equal values
#     and lower cost (smoke_result.json: telemetry_check).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"
REPO_ROOT="$(pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"

ARMS=(native_matched_s5 generalized_prospective_s5 calibrated_old_s5 calibrated_prediction_s5)
RUN_SEEDS=(303)
EPOCHS=15
BATCH=16
N_TASKS=$(( ${#ARMS[@]} * ${#RUN_SEEDS[@]} ))
IMPLEMENTATION="${IMPLEMENTATION:-factored}"
STAGE="${STAGE:-train}"
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------- guards ---
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the full authoritative sha}"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "FAIL: HEAD $(git rev-parse HEAD) is not $EXPECTED_COMMIT" >&2; exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  git status --short >&2
  echo "FAIL: refusing a modified worktree: $REPO_ROOT" >&2; exit 1
fi
if [[ "$STAGE" != "smoke" && "$STAGE" != "train" ]]; then
  echo "FAIL: STAGE must be smoke or train, got '$STAGE'" >&2; exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "FAIL: interpreter is not executable: $PY" >&2; exit 1
fi
case "${JAX_ENABLE_X64:-0}" in
  0|false|False|FALSE|"") ;;
  *) echo "FAIL: JAX_ENABLE_X64=$JAX_ENABLE_X64; production training is float32" >&2; exit 1 ;;
esac
export JAX_ENABLE_X64=0
DATA_CACHE="${S5_THREE_ARM_DATA:-/Users/durso/s5-runs/sc10_official_cache}"
if [[ ! -d "$DATA_CACHE" ]]; then
  echo "FAIL: data cache is not a directory: $DATA_CACHE" >&2; exit 1
fi
ALLOCATION="${SLURM_JOB_ID:-}"
if [[ ! "$ALLOCATION" =~ ^[0-9]+$ ]]; then
  echo "FAIL: SLURM_JOB_ID='$ALLOCATION' is not a numeric job id" >&2; exit 1
fi

# exactly one task per DISTINCT visible GPU token
IFS=',' read -r -a RAW_TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
TOKENS=()
for token in ${RAW_TOKENS[@]+"${RAW_TOKENS[@]}"}; do
  [[ -z "$token" ]] && continue
  duplicate=0
  for seen in ${TOKENS[@]+"${TOKENS[@]}"}; do [[ "$seen" == "$token" ]] && duplicate=1; done
  [[ "$duplicate" == "0" ]] && TOKENS+=("$token")
done
if [[ "${#TOKENS[@]}" -lt "$N_TASKS" ]]; then
  echo "FAIL: need $N_TASKS distinct GPU tokens, have ${#TOKENS[@]}:" \
       "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'" >&2; exit 1
fi

OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-timescale-seed303}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$OUT_ROOT/$STAMP"
SMOKE_ROOT="$RUN_ROOT/smoke"
if [[ "$DRY_RUN" != "1" && -e "$RUN_ROOT" ]]; then
  echo "FAIL: run root exists, refusing to reuse it: $RUN_ROOT" >&2; exit 1
fi

echo "timescale confirmation: N/A/C/D x seeds ${RUN_SEEDS[*]}, allocation $ALLOCATION"
echo "  stage:          $STAGE"
echo "  arms:           ${ARMS[*]}"
echo "  epochs / batch: $EPOCHS / $BATCH (protocol batch asserted below)"
echo "  scan:           $IMPLEMENTATION (second-order arms A, C, D)"
echo "  gpu tokens:     ${TOKENS[*]:0:$N_TASKS}  (one task per GPU, one wave)"
echo "  commit:         $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "  interpreter:    $PY"
echo "  data cache:     $DATA_CACHE"
echo "  run root:       $RUN_ROOT"
echo "  monitor:        tail -n 2 $RUN_ROOT/*/*/logs/train.log; tail -n 1 $RUN_ROOT/*/*/pole_diagnostics.jsonl"

# ------------------------------------------------------------ pre-flight ---
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RUN_ROOT"
  JAX_PLATFORMS=cpu "$PY" - "$DATA_CACHE" "$BATCH" "$EPOCHS" "$RUN_ROOT/preflight.json" \
      "${RUN_SEEDS[@]}" <<'PYEOF'
import json, sys
import jax, numpy as np
from flax.traverse_util import flatten_dict
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.data import validate_official_raw_cache
cache, batch, epochs, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
seeds = [int(s) for s in sys.argv[5:]]
assert not jax.config.read("jax_enable_x64"), "x64 must be off"
assert R.BATCH_SIZE == batch, (R.BATCH_SIZE, batch)
assert epochs > R.WARMUP_END
validate_official_raw_cache(cache, ("train", "val"))
report = {"batch_size": R.BATCH_SIZE, "epochs": epochs, "seeds": seeds,
          "selection_rule": "max val accuracy, then min val cross-entropy, then earliest epoch",
          "matched_initialization": {}}
for seed in seeds:
    p = {a: flatten_dict(R.init_state(a, seed).params) for a in
         ("native_matched_s5", "generalized_prospective_s5",
          "calibrated_old_s5", "calibrated_prediction_s5")}
    A = p["generalized_prospective_s5"]
    for arm in ("calibrated_old_s5", "calibrated_prediction_s5"):
        assert sorted(p[arm]) == sorted(A), arm
        for k in A:
            assert np.array_equal(np.asarray(A[k]), np.asarray(p[arm][k])), (arm, k)
    for k, v in p["native_matched_s5"].items():
        assert np.array_equal(np.asarray(v), np.asarray(A[k])), ("native", k)
    report["matched_initialization"][str(seed)] = {
        "A_C_D_all_leaves_identical": True, "native_leaves_identical_to_A": True,
        "n_leaves_A": len(A), "n_leaves_native": len(p["native_matched_s5"]),
        "dtype": str(np.asarray(A[next(iter(A))]).dtype)}
json.dump(report, open(out, "w"), indent=2)
print(f"pre-flight passed: batch {batch}, cache valid, matched init for seeds {seeds}")
PYEOF
  printf 'authoritative_commit=%s\nbranch=%s\narms=%s\nscan_implementation=%s\nepochs_requested=%s\nbatch_size=%s\nseeds=%s\nallocation=%s\ngpu_tokens=%s\ndata_cache=%s\nselection_rule=%s\nstage=%s\n' \
    "$EXPECTED_COMMIT" "$(git rev-parse --abbrev-ref HEAD)" "${ARMS[*]}" \
    "$IMPLEMENTATION" "$EPOCHS" "$BATCH" "${RUN_SEEDS[*]}" "$ALLOCATION" \
    "${TOKENS[*]:0:$N_TASKS}" "$DATA_CACHE" \
    "max val accuracy, then min val cross-entropy, then earliest epoch" \
    "$STAGE" > "$RUN_ROOT/run_metadata.txt"
fi

# ------------------------------------------------------------- children ----
CHILD_PIDS=(); CHILD_LABELS=(); CHILD_DIRS=()
ACTIVE_PIDS=()
stop_children() {
  # Runner processes exec in the background subshell, so these are their PIDs.
  # Only the current, not-yet-waited wave is eligible for cleanup.
  local pid
  for pid in ${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in ${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}; do
    wait "$pid" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
}
trap stop_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_child() {   # arm seed gpu_token root [--smoke]
  local arm="$1" seed="$2" token="$3" root="$4" smoke="${5:-}"
  local task_root="$root/$arm/$seed"
  local label="$arm/$seed@gpu$token${smoke:+ (smoke)}"
  # one cache per (arm, seed), shared by that task's smoke and training only
  local cache_dir="$RUN_ROOT/jax-cache/$arm/$seed"
  local runner_args=(--data-cache "$DATA_CACHE" --out "$task_root"
                     --arm "$arm" --seed "$seed" --epochs "$EPOCHS"
                     --implementation "$IMPLEMENTATION")
  [[ -n "$smoke" ]] && runner_args+=("$smoke")
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN child $label"
    echo "    CUDA_VISIBLE_DEVICES=$token JAX_ENABLE_X64=0 JAX_COMPILATION_CACHE_DIR=$cache_dir"
    echo "    $PY -u -m experiments.s5_three_arm_full.runner ${runner_args[*]}"
    return 0
  fi
  mkdir -p "$task_root/tmp" "$task_root/logs" "$cache_dir"
  (
    export CUDA_VISIBLE_DEVICES="$token"
    export TMPDIR="$task_root/tmp"
    export JAX_COMPILATION_CACHE_DIR="$cache_dir"
    "$PY" -u -m experiments.s5_three_arm_full.gpu_telemetry \
      --out "$task_root/gpu_telemetry.json" --label "pre_jax_gpu$token" \
      > "$task_root/logs/gpu_telemetry.log" 2>&1
    exec "$PY" -u -m experiments.s5_three_arm_full.runner \
      "${runner_args[@]}" > "$task_root/logs/train.log" 2>&1
  ) &
  ACTIVE_PIDS+=("$!")
  CHILD_PIDS+=("$!"); CHILD_LABELS+=("$label"); CHILD_DIRS+=("$task_root")
  echo "started $label as pid $! (log $task_root/logs/train.log)"
}

WAVE_STATUS=()
wait_for_wave() {
  WAVE_STATUS=(); local index=0 status
  for pid in ${CHILD_PIDS[@]+"${CHILD_PIDS[@]}"}; do
    status=0; wait "$pid" || status=$?
    WAVE_STATUS+=("$status")
    unset 'ACTIVE_PIDS[index]'
    echo "finished ${CHILD_LABELS[$index]} pid $pid exit $status"
    index=$((index + 1))
  done
}
reset_wave() { CHILD_PIDS=(); CHILD_LABELS=(); CHILD_DIRS=(); ACTIVE_PIDS=(); }

launch_all() {   # root [--smoke]
  reset_wave
  local slot=0
  for arm in "${ARMS[@]}"; do
    for seed in "${RUN_SEEDS[@]}"; do
      start_child "$arm" "$seed" "${TOKENS[$slot]}" "$1" "${2:-}"
      slot=$((slot + 1))
    done
  done
}

# ------------------------------------------------------ 1. smoke gate ------
echo
echo "STAGE 1: smoke, all $N_TASKS tasks at once (gates training)"
launch_all "$SMOKE_ROOT" --smoke
if [[ "$DRY_RUN" != "1" ]]; then
  wait_for_wave
  for i in "${!WAVE_STATUS[@]}"; do
    if [[ "${WAVE_STATUS[$i]}" -ne 0 || ! -f "${CHILD_DIRS[$i]}/smoke_result.json" ]]; then
      echo "ABORT: smoke failed for ${CHILD_LABELS[$i]}; see ${CHILD_DIRS[$i]}/logs/train.log" >&2
      echo "       Training was NOT started." >&2
      exit 1
    fi
  done
  "$PY" - "$SMOKE_ROOT" "$IMPLEMENTATION" "$RUN_ROOT/smoke_gate.json" "${ARMS[@]}" -- "${RUN_SEEDS[@]}" <<'PYEOF'
import json, os, sys
root, impl, out = sys.argv[1:4]
rest = sys.argv[4:]; cut = rest.index("--")
arms, seeds = rest[:cut], rest[cut + 1:]
scan_arms = {"generalized_prospective_s5", "calibrated_old_s5", "calibrated_prediction_s5"}
rows, failures = [], []
for arm in arms:
    for seed in seeds:
        d = os.path.join(root, arm, seed)
        smoke = json.load(open(os.path.join(d, "smoke_result.json")))
        prod = json.load(open(os.path.join(d, "production_check.json")))
        tc = smoke["telemetry_check"]
        poles = prod["initial_poles"].get("production_precision") or {}
        row = {"arm": arm, "seed": seed, "status": smoke["status"],
               "scan": prod["scan_implementation"], "x64": prod["x64_enabled"],
               "telemetry_legacy_ms": 1e3 * tc["legacy_seconds"],
               "telemetry_fused_ms": 1e3 * tc["fused_seconds"],
               "telemetry_agree": tc["values_agree"],
               "init_stable": poles.get("stable"),
               "init_boundary": poles.get("numerical_boundary"),
               "init_unstable": poles.get("unstable"),
               "init_min_margin_f32": poles.get("min_margin_production")}
        rows.append(row)
        want_scan = impl if arm in scan_arms else None
        if smoke["status"] != "SMOKE_PASS": failures.append((arm, seed, "status"))
        if prod["scan_implementation"] != want_scan: failures.append((arm, seed, "scan"))
        if prod["x64_enabled"]: failures.append((arm, seed, "x64"))
        if not tc["values_agree"]: failures.append((arm, seed, "telemetry values"))
        if not tc["fused_seconds"] < tc["legacy_seconds"]:
            failures.append((arm, seed, "fused telemetry not cheaper"))
        if poles.get("unstable"): failures.append((arm, seed, "unstable at init"))
for r in rows:
    print("{arm:28s} {seed}  {status}  scan={scan}  telemetry {telemetry_legacy_ms:7.2f} ms -> "
          "{telemetry_fused_ms:6.2f} ms  init f32 stable/boundary/unstable "
          "{init_stable}/{init_boundary}/{init_unstable}  min margin {init_min_margin_f32}".format(**r))
json.dump({"rows": rows, "failures": failures}, open(out, "w"), indent=2)
if failures:
    raise SystemExit(f"FAIL: smoke gate: {failures}")
print("smoke gate passed for every task")
PYEOF
fi

if [[ "$STAGE" == "smoke" ]]; then
  echo
  echo "STAGE=smoke: stopping after the smoke gate. Nothing trained."
  exit 0
fi

# ------------------------------------------------------ 2. training --------
echo
echo "STAGE 2: training, all $N_TASKS tasks at once, one per GPU"
launch_all "$RUN_ROOT"
if [[ "$DRY_RUN" == "1" ]]; then
  echo; echo "DRY_RUN: nothing started"; exit 0
fi
wait_for_wave
MISSING=()
for i in "${!WAVE_STATUS[@]}"; do
  if [[ "${WAVE_STATUS[$i]}" -ne 0 || ! -f "${CHILD_DIRS[$i]}/task_result.json" ]]; then
    reason="exit ${WAVE_STATUS[$i]}"
    [[ -f "${CHILD_DIRS[$i]}/failure.json" ]] && reason="$reason, failure.json recorded"
    MISSING+=("${CHILD_LABELS[$i]} ($reason)")
  fi
done
echo
echo "run root: $RUN_ROOT"
echo "per-task results: $RUN_ROOT/*/*/task_result.json"
echo "per-epoch poles:  $RUN_ROOT/*/*/pole_diagnostics.jsonl"
echo "NO FINALIZER WAS RUN and the test split was not read. Any accuracy"
echo "comparison uses these runs' selected-validation results, all under one"
echo "split and one selection rule; never mix them with held-out-test scores."
if [[ "${#MISSING[@]}" -gt 0 ]]; then
  echo "INCOMPLETE: ${MISSING[*]}" >&2
  echo "Artifacts preserved; failures are recorded, nothing was retried." >&2
  exit 1
fi
