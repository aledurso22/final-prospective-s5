#!/usr/bin/env bash
# Fixed-alpha history pilot: seeds 301/302, one GPU per seed, fresh training.
# Both full-size GPU smokes must pass before either 15-epoch task starts.
# Task outputs remain absent until the history runner creates them; logs,
# temporary files and pre-JAX inventories live outside those guarded outputs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$HERE/cluster_env.sh"
cd "$REPO_ROOT"

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the full authoritative sha}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "FAIL: checkout is not EXPECTED_COMMIT" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "FAIL: refusing modified worktree" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FAIL: interpreter is not executable: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in
  0|false|False|FALSE|"") ;;
  *) echo "FAIL: history production requires JAX_ENABLE_X64=0" >&2; exit 1 ;;
esac
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
STAGE="${STAGE:-train}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$STAGE" == smoke || "$STAGE" == train ]] || {
  echo "FAIL: STAGE must be smoke or train" >&2; exit 1; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
  echo "FAIL: DRY_RUN must be 0 or 1" >&2; exit 1; }
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric SLURM_JOB_ID required" >&2; exit 1; }
IFS=',' read -r -a RAW_TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
TOKENS=()
for token in ${RAW_TOKENS[@]+"${RAW_TOKENS[@]}"}; do
  token="${token//[[:space:]]/}"
  [[ -n "$token" ]] || continue
  duplicate=0
  for seen in ${TOKENS[@]+"${TOKENS[@]}"}; do [[ "$seen" == "$token" ]] && duplicate=1; done
  [[ "$duplicate" == 0 ]] && TOKENS+=("$token")
done
[[ "${#TOKENS[@]}" -eq 2 ]] || {
  echo "FAIL: need exactly 2 distinct GPU tokens, got ${#TOKENS[@]}" >&2; exit 1; }

ARM=gradient_history_s5
SEEDS=(301 302)
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
RUN_ROOT="${HISTORY_RUN_ROOT:-$PROSPECTIVE_RUNS/s5-history-pilot/job-$ALLOCATION}"
SMOKE_ROOT="$RUN_ROOT/smoke"
[[ -d "$DATA_CACHE" ]] || { echo "FAIL: missing data cache: $DATA_CACHE" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: run root exists: $RUN_ROOT" >&2; exit 1; }

echo "history pilot: seeds 301 302, allocation $ALLOCATION, stage $STAGE"
echo "  commit: $EXPECTED_COMMIT"
echo "  GPUs: ${TOKENS[*]} (one seed per GPU)"
echo "  alpha: 0.10; tau=ell=alpha/abs(-Lambda*Delta); exact ZOH"
echo "  epochs / batch: 15 / 16; scan: history_stationary_block"
echo "  data: $DATA_CACHE"
echo "  output: $RUN_ROOT"

if [[ "$DRY_RUN" == 0 ]]; then
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/inventories"
  JAX_PLATFORMS=cpu "$PY" - "$DATA_CACHE" "$RUN_ROOT/preflight.json" <<'PYEOF'
import json, sys
import jax, numpy as np
from flax.traverse_util import flatten_dict
from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.data import validate_official_raw_cache
assert not jax.config.read("jax_enable_x64")
assert R.BATCH_SIZE == 16 and R.SEQ_LEN == 16000 and R.N_LAYERS == 6
assert H.EPOCHS == 15 and H.ALPHA == .10 and H.SCAN == "parallel"
validate_official_raw_cache(sys.argv[1], ("train", "val"))
matches = {}
for seed in (301, 302):
    native = R.init_state("native_matched_s5", seed)
    with H.registered_history():
        history = R.init_state(H.ARM, seed)
    pn, ph = flatten_dict(native.params), flatten_dict(history.params)
    assert set(pn) == set(ph), "history changes the parameter tree"
    for key in pn:
        assert np.array_equal(np.asarray(pn[key]), np.asarray(ph[key])), (seed, key)
    tn, th = jax.tree_util.tree_flatten(native.opt_state), jax.tree_util.tree_flatten(history.opt_state)
    assert tn[1] == th[1]
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(tn[0], th[0]))
    matches[str(seed)] = {"native_parameters_identical": True, "optimizer_identical": True}
json.dump({"batch": 16, "epochs": 15, "seq_len": 16000, "layers": 6,
           "alpha": .10, "matched_initialization": matches, "test_split_read": False},
          open(sys.argv[2], "w"), indent=2)
print("preflight passed: native backbone/optimizer, protocol, train/val cache")
PYEOF
  printf 'authoritative_commit=%s\narm=%s\nseeds=301 302\nepochs_requested=15\nbatch_size=16\nhistory_alpha=0.10\nhistory_horizons=tau=ell=0.10/abs(-Lambda*Delta)\nscan_implementation=history_stationary_block\nallocation=%s\ngpu_tokens=%s\ndata_cache=%s\nstage=%s\nselection_rule=max validation accuracy, then min cross-entropy, then earliest epoch\n' \
    "$EXPECTED_COMMIT" "$ARM" "$ALLOCATION" "${TOKENS[*]}" "$DATA_CACHE" "$STAGE" \
    > "$RUN_ROOT/run_metadata.txt"
fi

PIDS=(); ACTIVE_PIDS=()
cleanup_children() {
  local pid
  for pid in ${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in ${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}; do wait "$pid" 2>/dev/null || true; done
  ACTIVE_PIDS=()
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_task() {
  local slot="$1" stage="$2" seed="${SEEDS[$1]}" token="${TOKENS[$1]}"
  local task_root="$RUN_ROOT/$ARM/$seed"
  local cache="$RUN_ROOT/jax-cache/$ARM/$seed"
  local args=(--data-cache "$DATA_CACHE" --seed "$seed")
  if [[ "$stage" == smoke ]]; then
    task_root="$SMOKE_ROOT/$ARM/$seed"
    args+=(--smoke)
  else
    args+=(--train --smoke-report "$SMOKE_ROOT/$ARM/$seed/smoke_result.json")
  fi
  args+=(--out "$task_root")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY_RUN $stage seed $seed GPU $token cache $cache"
    echo "  $PY -u -m experiments.s5_gradient_history.runner ${args[*]}"
    return
  fi
  mkdir -p "$cache" "$RUN_ROOT/tmp/$stage-$seed"
  (
    export CUDA_VISIBLE_DEVICES="$token" JAX_PLATFORMS=cuda
    export JAX_COMPILATION_CACHE_DIR="$cache" TMPDIR="$RUN_ROOT/tmp/$stage-$seed"
    child=""
    cleanup_precheck() {
      if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
    }
    trap cleanup_precheck EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    "$PY" - <<'PYEOF' > "$RUN_ROOT/logs/$stage-$seed-device.log" 2>&1 &
import jax
devices = jax.devices()
print("Devices:", devices, flush=True)
assert len(devices) == 1 and devices[0].platform == "gpu", "expected exactly one GPU"
assert not jax.config.read("jax_enable_x64")
PYEOF
    child="$!"; wait "$child"; child=""
    "$PY" -u -m experiments.s5_three_arm_full.gpu_telemetry \
      --out "$RUN_ROOT/inventories/$stage-$seed.json" --label "pre_jax_$stage-$seed" \
      > "$RUN_ROOT/logs/$stage-$seed-inventory.log" 2>&1 &
    child="$!"; wait "$child"; child=""
    exec "$PY" -u -m experiments.s5_gradient_history.runner "${args[@]}" \
      > "$RUN_ROOT/logs/$stage-$seed.log" 2>&1
  ) &
  PIDS+=("$!"); ACTIVE_PIDS+=("$!")
  echo "started $stage seed $seed GPU $token; log $RUN_ROOT/logs/$stage-$seed.log"
}

wait_wave() {
  local failed=0 slot status
  for slot in 0 1; do
    status=0; wait "${PIDS[$slot]}" || status=$?
    unset 'ACTIVE_PIDS[slot]'
    echo "seed ${SEEDS[$slot]} exited $status"
    [[ "$status" == 0 ]] || failed=1
  done
  PIDS=(); ACTIVE_PIDS=()
  [[ "$failed" == 0 ]] || { echo "FAIL: a $1 task failed; inspect task logs" >&2; return 1; }
}

echo "STAGE 1: both full-size GPU smokes"
for slot in 0 1; do start_task "$slot" smoke; done
if [[ "$DRY_RUN" == 0 ]]; then
  wait_wave smoke
  JAX_PLATFORMS=cpu "$PY" - "$SMOKE_ROOT" "$DATA_CACHE" "$RUN_ROOT/smoke_gate.json" <<'PYEOF'
import json, math, sys
from pathlib import Path
from experiments.s5_gradient_history import runner as H
rows = []
for seed in (301, 302):
    task = Path(sys.argv[1]) / H.ARM / str(seed)
    report = json.loads((task / "smoke_result.json").read_text())
    prod = json.loads((task / "production_check.json").read_text())
    H.check_smoke_gate(report, H.protocol_identity(sys.argv[2], seed))
    assert report["scan_implementation"] == "history_stationary_block"
    assert prod["scan_implementation"] == "history_stationary_block" and not prod["x64_enabled"]
    assert prod["gradients_finite"] and prod["state_finite"]
    assert all(math.isfinite(prod[k]) for k in ("loss", "accuracy", "gradient_norm"))
    assert prod["loss"] >= 0 and prod["gradient_norm"] >= 0 and 0 <= prod["accuracy"] <= 1
    for poles in (prod["initial_poles"], report["poles_after_smoke_steps"]):
        assert poles["n_modes"] == 384 and poles["nonfinite_modes"] == 0
        pp = poles["production_precision"]
        assert pp["unstable"] == 0 and pp["nonfinite"] == 0
    rows.append({"seed": seed, "status": "SMOKE_PASS", "report": str(task / "smoke_result.json")})
json.dump({"tasks": rows, "all_passed": True}, open(sys.argv[3], "w"), indent=2)
print("both GPU smoke gates passed; exact-shape training may start")
PYEOF
fi
if [[ "$STAGE" == smoke ]]; then
  echo "Smoke-only stage complete. Nothing trained."
  exit 0
fi

echo "STAGE 2: fresh 15-epoch training, seeds 301/302"
for slot in 0 1; do start_task "$slot" train; done
if [[ "$DRY_RUN" == 0 ]]; then
  wait_wave training
  for seed in "${SEEDS[@]}"; do
    [[ -f "$RUN_ROOT/$ARM/$seed/task_result.json" ]] || {
      echo "FAIL: missing task_result.json for seed $seed" >&2; exit 1; }
  done
  echo "Completed history pilot: $RUN_ROOT"
fi
