#!/usr/bin/env bash
# WWJ GATES, inside an existing interactive Slurm allocation. NO TRAINING RUN.
#
# Three bounded stages, each a DIRECT CHILD of this shell -- no sbatch, no
# salloc, no srun, so no child job or step can terminate while another task
# runs, which is the condition diagnosed in S5_EXECUTION_TOPOLOGY_DIAGNOSIS.md:
#
#   1. initialization grid: sweep k = tau/h and eps over the ACTUAL
#      initialized S5 modes and select tau by the declared rule in
#      experiments/s5_wwj/init_grid.py. No validation result is consulted.
#   2. performance gate: Native S5 versus the WWJ arm, production-shaped, on
#      one GPU. Authorizes nothing unless every declared condition holds,
#      including WWJ throughput at least half of Native's.
#   3. developmental gate: two real updates, a fixed subset, checkpoint save
#      and reload, validation. Never the test split.
#
# A full 15-epoch experiment is NOT launched here and is not authorized until
# stages 2 and 3 pass and their artifacts have been read.
#
#   DRY_RUN=1 bash bin/run_experiments/allocation_s5_wwj_gates.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"
REPO_ROOT="$(pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"

ARM="${S5_WWJ_ARM:-wwj_critical_s5}"
SEED="${S5_WWJ_SEED:-301}"
ALLOCATION_HOURS="${S5_WWJ_ALLOCATION_HOURS:-18}"
BENCH_STEPS="${S5_WWJ_BENCH_STEPS:-10}"
SUBSET_STEPS="${S5_WWJ_SUBSET_STEPS:-20}"
DRY_RUN="${DRY_RUN:-0}"

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
IFS=',' read -r -a GPU_TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ "${#GPU_TOKENS[@]}" -lt 1 ]]; then
  echo "FAIL: no visible GPU token in CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi
TOKEN="${GPU_TOKENS[0]}"

OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-wwj-gates}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$OUT_ROOT/$STAMP"

echo "WWJ gates, direct children of allocation $ALLOCATION"
echo "arm: $ARM (seed $SEED)   gpu token: $TOKEN   allocation: ${ALLOCATION_HOURS}h"
echo "commit: $(git rev-parse HEAD)"
echo "data cache: $DATA_CACHE"
echo "run root: $RUN_ROOT"
echo "monitor:  tail -n 20 -f $RUN_ROOT/logs/*.log"

run_child() {   # name -- command...
  local name="$1"; shift
  local log="$RUN_ROOT/logs/$name.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN stage $name"
    echo "    CUDA_VISIBLE_DEVICES=$TOKEN"
    echo "    TMPDIR=$RUN_ROOT/$name/tmp"
    echo "    JAX_COMPILATION_CACHE_DIR=$RUN_ROOT/$name/jax-cache"
    echo "    log=$log"
    echo "    $*"
    return 0
  fi
  mkdir -p "$RUN_ROOT/$name/tmp" "$RUN_ROOT/$name/jax-cache" "$RUN_ROOT/logs"
  (
    export CUDA_VISIBLE_DEVICES="$TOKEN"
    export TMPDIR="$RUN_ROOT/$name/tmp"
    export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/$name/jax-cache"
    "$@"
  ) 2>&1 | tee "$log"
}

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RUN_ROOT/logs"
  printf 'commit=%s\narm=%s\nseed=%s\nallocation=%s\ngpu_token=%s\nallocation_hours=%s\n' \
    "$EXPECTED_COMMIT" "$ARM" "$SEED" "$ALLOCATION" "$TOKEN" \
    "$ALLOCATION_HOURS" > "$RUN_ROOT/run_metadata.txt"
fi

echo
echo "STAGE 1: initialization grid (declared rule, no validation consulted)"
run_child init_grid "$PY" -u -m experiments.s5_wwj.init_grid \
  --out "$RUN_ROOT/init_grid.json" --seed "$SEED"

TAU_INIT="0.25"
EPS_INIT="0.0625"
if [[ "$DRY_RUN" != "1" ]]; then
  TAU_INIT="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected"]["tau_init"])' "$RUN_ROOT/init_grid.json")"
  EPS_INIT="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected"]["eps_init_passive"])' "$RUN_ROOT/init_grid.json")"
  echo "selected by the declared rule: tau_init=$TAU_INIT eps_init=$EPS_INIT"
fi

echo
echo "STAGE 2: performance gate, Native S5 versus $ARM"
run_child benchmark "$PY" -u -m experiments.s5_wwj.benchmark \
  --out "$RUN_ROOT/benchmark.json" --arm "$ARM" --seed "$SEED" \
  --steps "$BENCH_STEPS" --tau-init "$TAU_INIT" --eps-init "$EPS_INIT" \
  --allocation-hours "$ALLOCATION_HOURS"

echo
echo "STAGE 3: developmental learning gate (validation only, never test)"
run_child dev_gate "$PY" -u -m experiments.s5_wwj.dev_gate \
  --data-cache "$DATA_CACHE" --out "$RUN_ROOT/dev_gate" --arm "$ARM" \
  --seed "$SEED" --subset-steps "$SUBSET_STEPS" \
  --tau-init "$TAU_INIT" --eps-init "$EPS_INIT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN: nothing started; no training run and no Slurm job"
  exit 0
fi

echo
echo "artifacts:"
echo "  $RUN_ROOT/init_grid.json"
echo "  $RUN_ROOT/benchmark.json"
echo "  $RUN_ROOT/dev_gate/dev_gate.json"
echo "Read them before authorizing any 15-epoch experiment; a passing gate"
echo "authorizes nothing by itself."
