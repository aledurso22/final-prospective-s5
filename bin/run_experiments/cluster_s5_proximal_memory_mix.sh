#!/usr/bin/env bash
# Two isolated seed tasks; the whole allocation waits for both GPU smoke gates.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$HERE/cluster_env.sh"
cd "$REPO_ROOT"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned full commit}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ && "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "FAIL: wrong commit" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "FAIL: checkout must be clean, including untracked files" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FAIL: missing Python: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in 0|false|False|FALSE|"") ;; *) echo "FAIL: x64 must be off" >&2; exit 1;; esac
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda
STAGE="${STAGE:-train}"; DRY_RUN="${DRY_RUN:-0}"
[[ "$STAGE" == smoke || "$STAGE" == train ]] || { echo "FAIL: invalid STAGE" >&2; exit 1; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric Slurm allocation required" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#TOKENS[@]}" == 2 && -n "${TOKENS[0]}" && -n "${TOKENS[1]}" && "${TOKENS[0]}" != "${TOKENS[1]}" ]] || {
  echo "FAIL: exactly two distinct allocated GPU tokens required" >&2; exit 1; }
for token in "${TOKENS[@]}"; do
  [[ "$token" != *[[:space:]]* ]] || { echo "FAIL: whitespace in GPU token" >&2; exit 1; }
done
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
RUN_ROOT="$PROSPECTIVE_RUNS/s5-proximal-memory-mix/job-$ALLOCATION"
[[ -d "$DATA_CACHE" ]] || { echo "FAIL: data cache missing" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: run root already exists" >&2; exit 1; }
SEEDS=(301 302)
COMMON=(--alpha 1 --fast-ratio 0.1 --slow-ratio 1 --slow-weight 0.4444444444444444 --proximal-ratio 0.5)
echo "proximal memory mix: allocation $ALLOCATION, stage $STAGE"
echo "  assigned GPU tokens: ${TOKENS[*]}; one seed per GPU"
echo "  alpha=1; fast_ratio=0.1; slow_ratio=1; slow_weight=4/9; proximal_ratio=0.5"
echo "  batch=16; epochs=15; optimizer/split/selection unchanged; test split never read"
echo "  commit: $EXPECTED_COMMIT"
echo "  output: $RUN_ROOT"
if [[ "$DRY_RUN" == 1 ]]; then
  for i in 0 1; do
    echo "DRY_RUN profile/smoke seed=${SEEDS[$i]} GPU=${TOKENS[$i]}"
    [[ "$STAGE" == smoke ]] || echo "DRY_RUN fresh train seed=${SEEDS[$i]} after BOTH GPU gates pass"
  done
  exit 0
fi
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs" "$RUN_ROOT/memory-profiles" "$RUN_ROOT/devices" "$RUN_ROOT/gates"
"$PY" -u - <<'PY'
import os,jax
devices=jax.devices()
print("Allocated devices:",devices,flush=True)
assert len(devices)==2 and all(d.platform=="gpu" for d in devices)
assert len(os.sched_getaffinity(0))>=16, "allocation must expose 16 CPUs"
assert not jax.config.read("jax_enable_x64")
PY
JAX_PLATFORMS=cpu "$PY" -u - "$DATA_CACHE" "$RUN_ROOT/preflight.json" <<'PY'
import json,sys
import jax,numpy as np
from flax.traverse_util import flatten_dict
from experiments.s5_proximal_memory_mix import runner as M
from experiments.s5_proximal_history import runner as P
from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.data import validate_official_raw_cache
assert not jax.config.read("jax_enable_x64")
assert R.BATCH_SIZE==16 and R.SEQ_LEN==16000 and R.N_LAYERS==6 and H.EPOCHS==15
validate_official_raw_cache(sys.argv[1],("train","val"))
rows=[]
for seed in (301,302):
    native=R.init_state("native_matched_s5",seed)
    with P.configured(1.,.5):
        with H.registered_history(): proximal=R.init_state(P.ARM,seed)
    with M.configured(alpha=1.,fast_ratio=.1,slow_ratio=1.,slow_weight=4/9,proximal_ratio=.5):
        with H.registered_history(): mixed=R.init_state(M.ARM,seed)
    for label,state in (("P",proximal),("mix",mixed)):
        a,b=flatten_dict(native.params),flatten_dict(state.params)
        assert set(a)==set(b), label
        assert all(np.array_equal(np.asarray(a[k]),np.asarray(b[k])) for k in a), label
        ta,tb=jax.tree_util.tree_flatten(native.opt_state),jax.tree_util.tree_flatten(state.opt_state)
        assert ta[1]==tb[1]
        assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(ta[0],tb[0])), label
    rows.append({"seed":seed,"native_P_mix_parameters_identical":True,"optimizer_identical":True})
with open(sys.argv[2],"x") as f:
    json.dump({"tasks":rows,"batch":16,"epochs":15,"test_split_read":False},f,indent=2)
print("preflight passed: native/P/mix backbone and optimizer, protocol and train/val cache",flush=True)
PY
printf 'commit=%s\njob=%s\nGPU_tokens=%s\nalpha=1\nfast_ratio=0.1\nslow_ratio=1\nslow_weight=4/9\nproximal_ratio=0.5\nseeds=301 302\nbatch=16\nepochs=15\nouter_optimizer=unchanged\n' \
  "$EXPECTED_COMMIT" "$ALLOCATION" "${TOKENS[*]}" > "$RUN_ROOT/run_metadata.txt"

# Every cleanup is confined to PIDs created by this shell. Never kill a process
# group: another Slurm job can share a session/process group on this cluster.
PIDS=()
cleanup_parent() {
  local pid
  [[ -n "${PIDS[*]:-}" ]] || return 0
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup_parent EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

worker() (
  set -euo pipefail
  seed="$1"; token="$2"; phase="$3"
  child=""
  cleanup_worker() {
    if [[ -n "$child" ]]; then
      kill -TERM "$child" 2>/dev/null || true
      wait "$child" 2>/dev/null || true
    fi
  }
  trap cleanup_worker EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  export CUDA_VISIBLE_DEVICES="$token"
  export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache/$seed"
  export TMPDIR="$RUN_ROOT/tmp/$seed"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
  run_logged() {
    local log="$1"; shift
    "$@" > "$log" 2>&1 &
    child="$!"
    local status=0
    wait "$child" || status=$?
    child=""
    [[ "$status" == 0 ]] || { echo "FAIL: seed $seed exit $status; inspect $log" >&2; return "$status"; }
  }
  if [[ "$phase" == smoke ]]; then
    "$PY" -u - "$RUN_ROOT/devices/$seed.json" <<'PY'
import json,os,sys,jax
devices=jax.devices()
assert len(devices)==1 and devices[0].platform=="gpu"
assert not jax.config.read("jax_enable_x64")
stats=devices[0].memory_stats() or {}
result={"backend":"gpu","devices":[str(d) for d in devices],"device_count":1,
        "x64":False,"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
        "allocator_limit_bytes":stats.get("bytes_limit"),
        "memory_scope":"JAX allocator limit for this assigned device, when provided"}
with open(sys.argv[1],"x") as f: json.dump(result,f,indent=2)
PY
    echo "PROFILE seed $seed on assigned GPU token $token"
    run_logged "$RUN_ROOT/logs/profile-$seed.log" "$PY" -u -m experiments.s5_proximal_memory_mix.runner \
      "${COMMON[@]}" --seed "$seed" --profile --out "$RUN_ROOT/memory-profiles/$seed.json"
    "$PY" -u - "$RUN_ROOT/memory-profiles/$seed.json" "$RUN_ROOT/devices/$seed.json" "$RUN_ROOT/gates/memory-$seed.json" "$seed" <<'PY'
import json,math,os,sys
profile=json.load(open(sys.argv[1])); device=json.load(open(sys.argv[2])); seed=int(sys.argv[4])
assert profile["arm"]=="proximal_memory_mix_s5"
assert profile["seed_for_shape_construction"]==seed
assert profile["git_commit"]==os.environ["EXPECTED_COMMIT"] and not profile["git_worktree_dirty"]
for key,value in {"history_alpha":1.,"fast_ratio":.1,"slow_ratio":1.,"slow_weight":4/9,"proximal_ratio":.5}.items():
    assert math.isclose(profile[key],value,rel_tol=0,abs_tol=1e-15), key
assert profile["backend"]=="gpu" and len(profile["devices"])==1
assert profile["x64"] is False and profile["batch_size"]==16
assert profile["sequence_length"]==16000 and profile["layers"]==6
assert device["device_count"]==1 and device["x64"] is False
assert {r["kind"] for r in profile["updates"]}=={"observable","telemetry"}
limit=device.get("allocator_limit_bytes")
known_limit=isinstance(limit,(int,float)) and not isinstance(limit,bool) and math.isfinite(limit) and limit>0
rows=[]
for row in profile["updates"]:
    total=(row.get("memory") or {}).get("buffer_accounting_total_bytes")
    known_total=isinstance(total,(int,float)) and not isinstance(total,bool) and math.isfinite(total) and total>0
    checked=known_limit and known_total
    if checked:
        assert total<limit, f"compiler buffers {total} exceed assigned allocator limit {limit}"
    rows.append({"kind":row["kind"],"compiler_bytes":total,"allocator_limit_bytes":limit,
                 "fit_check_available":checked,"fits_compiler_lower_bound":True if checked else None})
with open(sys.argv[3],"x") as f:
    json.dump({"seed":seed,"rows":rows,
      "scope":"Compiler accounting is not actual peak use; the full-size runtime smoke remains mandatory."},f,indent=2)
PY
    smoke="$RUN_ROOT/smoke/proximal_memory_mix_s5/$seed"
    echo "SMOKE seed $seed on assigned GPU token $token"
    run_logged "$RUN_ROOT/logs/smoke-$seed.log" "$PY" -u -m experiments.s5_proximal_memory_mix.runner \
      "${COMMON[@]}" --seed "$seed" --data-cache "$DATA_CACHE" --out "$smoke" --smoke
    "$PY" -u - "$smoke" "$DATA_CACHE" "$seed" "$RUN_ROOT/gates/smoke-$seed.json" <<'PY'
import json,math,sys
from pathlib import Path
from experiments.s5_proximal_memory_mix import runner as M
from experiments.s5_gradient_history import runner as H
root=Path(sys.argv[1]); seed=int(sys.argv[3])
report=json.loads((root/"smoke_result.json").read_text())
production=json.loads((root/"production_check.json").read_text())
with M.configured(alpha=1.,fast_ratio=.1,slow_ratio=1.,slow_weight=4/9,proximal_ratio=.5):
    M.check_smoke_gate(report,production,H.protocol_identity(sys.argv[2],seed))
assert production["code_identifier"]==M.ARM and production["seed"]==seed
assert production["x64_enabled"] is False
assert production["gradients_finite"] and production["state_finite"]
assert production["scan_implementation"]=="history_stationary_block"
pp=production["initial_poles"]["production_precision"]
assert pp["unstable"]==0 and pp["nonfinite"]==0
assert math.isfinite(pp["max_radius_float64"])
with open(sys.argv[4],"x") as f:
    json.dump({"seed":seed,"status":"SMOKE_PASS","production_finite":True,
               "init_production_precision":pp,"smoke_report":str(root/"smoke_result.json")},f,indent=2)
print(f"seed {seed}: GPU smoke and checkpoint/source/data binding passed",flush=True)
PY
  else
    echo "TRAIN seed $seed on assigned GPU token $token (fresh initialization after both GPU gates)"
    task="$RUN_ROOT/proximal_memory_mix_s5/$seed"
    run_logged "$RUN_ROOT/logs/train-$seed.log" "$PY" -u -m experiments.s5_proximal_memory_mix.runner \
      "${COMMON[@]}" --seed "$seed" --data-cache "$DATA_CACHE" --out "$task" \
      --train --smoke-report "$RUN_ROOT/smoke/proximal_memory_mix_s5/$seed/smoke_result.json"
    [[ -s "$task/task_result.json" ]] || { echo "FAIL: missing result for seed $seed" >&2; exit 1; }
    "$PY" -u - "$task/task_result.json" "$seed" <<'PY'
import json,math,sys
r=json.load(open(sys.argv[1])); seed=int(sys.argv[2]); v=r["validation"]
assert r["code_identifier"]=="proximal_memory_mix_s5" and r["seed"]==seed
assert r["epochs_requested"]==15 and 1<=r["selected_epoch"]<=15
assert math.isfinite(v["accuracy"]) and 0<=v["accuracy"]<=1
assert math.isfinite(v["cross_entropy"]) and v["cross_entropy"]>=0
print(f"FINISHED mix/{seed}: selected epoch {r['selected_epoch']}; accuracy {100*v['accuracy']:.4f}%; CE {v['cross_entropy']:.6f}",flush=True)
PY
  fi
)

run_phase() {
  local phase="$1" i pid failed=0
  PIDS=()
  for i in 0 1; do
    worker "${SEEDS[$i]}" "${TOKENS[$i]}" "$phase" &
    PIDS+=("$!")
  done
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if ! wait "$pid"; then failed=1; fi
    unset 'PIDS[i]'
  done
  PIDS=()
  [[ "$failed" == 0 ]] || { echo "FAIL: $phase phase failed; inspect per-seed logs" >&2; return 1; }
}
run_phase smoke
"$PY" -u - "$RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[json.loads((root/f"gates/smoke-{s}.json").read_text()) for s in (301,302)]
assert all(r["status"]=="SMOKE_PASS" for r in rows)
with (root/"smoke_gate.json").open("x") as f: json.dump({"status":"SMOKE_PASS","tasks":rows},f,indent=2)
PY
echo "Both full-size GPU smoke gates passed."
if [[ "$STAGE" == train ]]; then run_phase train; fi
echo "Completed proximal memory mix stage $STAGE: $RUN_ROOT"
