#!/usr/bin/env bash
# One allocated GPU; three fresh state-dynamics tasks, run sequentially.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$HERE/cluster_env.sh"
cd "$REPO_ROOT"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned full sha}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: wrong commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "FAIL: modified checkout" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FAIL: missing Python: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in 0|false|False|FALSE|"") ;; *) echo "FAIL: x64 must be off" >&2; exit 1;; esac
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda
STAGE="${STAGE:-train}"; DRY_RUN="${DRY_RUN:-0}"
[[ "$STAGE" == smoke || "$STAGE" == train ]] || { echo "FAIL: invalid STAGE" >&2; exit 1; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric Slurm job required" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#TOKENS[@]}" == 1 && -n "${TOKENS[0]//[[:space:]]/}" ]] || {
  echo "FAIL: need exactly one allocated GPU token" >&2; exit 1; }
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
RUN_ROOT="${PROXIMAL_RUN_ROOT:-$PROSPECTIVE_RUNS/s5-proximal-history/job-$ALLOCATION}"
[[ -d "$DATA_CACHE" ]] || { echo "FAIL: data cache missing" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: run root already exists" >&2; exit 1; }
ALPHAS=(1 1 0.1); SEEDS=(301 302 301); TAGS=(alpha1 alpha1 alpha0p1)
echo "proximal history: one GPU token ${TOKENS[0]}, allocation $ALLOCATION, stage $STAGE"
echo "  sequence: alpha=1 seed301; alpha=1 seed302; alpha=0.1 seed301"
echo "  proximal ratio=0.5; unchanged batch16/15epochs/optimizer/selection"
echo "  commit: $EXPECTED_COMMIT"
echo "  output: $RUN_ROOT"
if [[ "$DRY_RUN" == 1 ]]; then
  for i in 0 1 2; do
    echo "DRY_RUN profile/smoke alpha=${ALPHAS[$i]} seed=${SEEDS[$i]} GPU=${TOKENS[0]}"
    [[ "$STAGE" == smoke ]] || echo "DRY_RUN train alpha=${ALPHAS[$i]} seed=${SEEDS[$i]} after matching smoke"
  done
  exit 0
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/memory-profiles" "$RUN_ROOT/inventories"
"$PY" -u - <<'PY'
import os,jax
devices=jax.devices()
print("Allocated devices:",devices,flush=True)
assert len(devices)==1 and devices[0].platform=="gpu"
assert len(os.sched_getaffinity(0))>=8
assert not jax.config.read("jax_enable_x64")
PY
JAX_PLATFORMS=cpu "$PY" -u - "$DATA_CACHE" "$RUN_ROOT/preflight.json" <<'PY'
import json,sys
import jax,numpy as np
from flax.traverse_util import flatten_dict
from experiments.s5_proximal_history import runner as P
from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.data import validate_official_raw_cache
assert not jax.config.read("jax_enable_x64")
assert R.BATCH_SIZE==16 and R.SEQ_LEN==16000 and R.N_LAYERS==6 and H.EPOCHS==15
validate_official_raw_cache(sys.argv[1],("train","val"))
rows=[]
for alpha,seed in ((1.,301),(1.,302),(.1,301)):
    native=R.init_state("native_matched_s5",seed)
    with P.configured(alpha):
        with H.registered_history(): state=R.init_state(P.ARM,seed)
    pn,pp=flatten_dict(native.params),flatten_dict(state.params)
    assert set(pn)==set(pp)
    assert all(np.array_equal(np.asarray(pn[k]),np.asarray(pp[k])) for k in pn)
    tn,tp=jax.tree_util.tree_flatten(native.opt_state),jax.tree_util.tree_flatten(state.opt_state)
    assert tn[1]==tp[1]
    assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(tn[0],tp[0]))
    rows.append({"alpha":alpha,"seed":seed,"native_parameters_identical":True,"optimizer_identical":True})
json.dump({"tasks":rows,"batch":16,"epochs":15,"test_split_read":False},open(sys.argv[2],"w"),indent=2)
print("preflight passed",flush=True)
PY
printf 'commit=%s\njob=%s\nGPU_token=%s\nproximal_ratio=0.5\nsequence=alpha1/301 alpha1/302 alpha0p1/301\nbatch=16\nepochs=15\nouter_optimizer=unchanged\n' \
 "$EXPECTED_COMMIT" "$ALLOCATION" "${TOKENS[0]}" > "$RUN_ROOT/run_metadata.txt"
child=""
cleanup() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_logged() {
  local log="$1"; shift
  "$@" > "$log" 2>&1 &
  child="$!"
  local status=0
  wait "$child" || status=$?
  child=""
  [[ "$status" == 0 ]] || { echo "FAIL: exit $status; inspect $log" >&2; return "$status"; }
}
for i in 0 1 2; do
  alpha="${ALPHAS[$i]}"; seed="${SEEDS[$i]}"; tag="${TAGS[$i]}"
  export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache/$tag/$seed"
  export TMPDIR="$RUN_ROOT/tmp/$tag/$seed"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
  common=(--alpha "$alpha" --proximal-ratio 0.5 --seed "$seed")
  echo "PROFILE $tag/$seed"
  run_logged "$RUN_ROOT/logs/$tag-$seed-profile.log" "$PY" -u -m experiments.s5_proximal_history.runner \
    "${common[@]}" --profile --out "$RUN_ROOT/memory-profiles/$tag-$seed.json"
  run_logged "$RUN_ROOT/logs/$tag-$seed-inventory.log" "$PY" -u -m experiments.s5_three_arm_full.gpu_telemetry \
    --out "$RUN_ROOT/inventories/$tag-$seed.json" --label "pre_smoke_$tag-$seed"
  smoke="$RUN_ROOT/smoke/$tag/$seed"
  echo "SMOKE $tag/$seed"
  run_logged "$RUN_ROOT/logs/$tag-$seed-smoke.log" "$PY" -u -m experiments.s5_proximal_history.runner \
    "${common[@]}" --data-cache "$DATA_CACHE" --out "$smoke" --smoke
  if [[ "$STAGE" == train ]]; then
    echo "TRAIN $tag/$seed (fresh initialization; its GPU smoke passed)"
    run_logged "$RUN_ROOT/logs/$tag-$seed-train.log" "$PY" -u -m experiments.s5_proximal_history.runner \
      "${common[@]}" --data-cache "$DATA_CACHE" --out "$RUN_ROOT/$tag/$seed" \
      --train --smoke-report "$smoke/smoke_result.json"
    [[ -f "$RUN_ROOT/$tag/$seed/task_result.json" ]] || { echo "FAIL: missing task result" >&2; exit 1; }
    echo "FINISHED $tag/$seed"
    "$PY" - "$RUN_ROOT/$tag/$seed/task_result.json" <<'PY'
import json,sys
with open(sys.argv[1]) as handle: result=json.load(handle)
v=result["validation"]
print(f"selected epoch {result['selected_epoch']}; validation accuracy {100*v['accuracy']:.4f}%; CE {v['cross_entropy']:.6f}",flush=True)
PY
  fi
done
echo "Completed proximal history stage $STAGE: $RUN_ROOT"
