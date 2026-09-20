#!/usr/bin/env bash
# WWJ GATES, inside an existing interactive Slurm allocation. NO TRAINING RUN.
#
# Three bounded stages, each a DIRECT CHILD of this shell -- no sbatch, no
# salloc, no srun, so no child job or step can terminate while another task
# runs, which is the condition diagnosed in S5_EXECUTION_TOPOLOGY_DIAGNOSIS.md:
#
#   1. initialization grid: sweep k = tau/h and eps over the ACTUAL
#      initialized S5 modes and select tau by the declared FIR rule in
#      experiments/s5_wwj/init_grid.py -- maximum FIR gain and non-vanishing
#      WWJ gradients, NOT companion radii, because the principal operator is
#      FIR and adds no recurrent poles. No validation result is consulted.
#   2. performance gate: Native S5 versus each WWJ arm, production-shaped,
#      on one GPU. Authorizes nothing unless every declared condition holds,
#      including WWJ throughput at least half of Native's.
#   3. developmental gate: two real updates, a fixed subset, checkpoint save
#      and reload, validation. Never the test split.
#
# Stages 2 and 3 run for EVERY principal arm -- wwj_critical_s5 and
# wwj_passive_s5 -- into separate artifact directories. A critical-arm pass
# authorizes nothing about the passive arm.
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

# BOTH principal arms are gated, each into its own artifact directory. A
# critical-arm pass authorizes nothing about the passive arm. Override with
# S5_WWJ_ARMS="wwj_critical_s5" to gate one of them alone.
read -r -a WWJ_ARMS <<< "${S5_WWJ_ARMS:-wwj_critical_s5 wwj_passive_s5}"
# The REJECTED mixed-stencil realization is unstable on the real S5 modes
# (max companion radius ~1.705, float32 NaN, float64 states ~1e81) and is a
# failed ablation: it can never be gated or trained from here.
for arm in "${WWJ_ARMS[@]}"; do
  case "$arm" in
    wwj_critical_s5|wwj_passive_s5|wwj_gated_recoverable_s5_diagnostic) ;;
    *mixed_stencil*)
      echo "FAIL: $arm is the REJECTED mixed-stencil realization; it is a" >&2
      echo "      failed ablation and must not be gated or trained." >&2
      exit 1 ;;
    *)
      echo "FAIL: $arm is not a WWJ arm this launcher will run" >&2
      exit 1 ;;
  esac
done
#: how many 15-epoch waves the projection must cover: one per gated arm
WAVES="${S5_WWJ_WAVES:-${#WWJ_ARMS[@]}}"
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
echo "arms: ${WWJ_ARMS[*]} (seed $SEED)   gpu token: $TOKEN"
echo "allocation: ${ALLOCATION_HOURS}h   projection covers $WAVES wave(s)"
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
  printf 'commit=%s\narms=%s\nseed=%s\nallocation=%s\ngpu_token=%s\nallocation_hours=%s\nwaves_projected=%s\n' \
    "$EXPECTED_COMMIT" "${WWJ_ARMS[*]}" "$SEED" "$ALLOCATION" "$TOKEN" \
    "$ALLOCATION_HOURS" "$WAVES" > "$RUN_ROOT/run_metadata.txt"
fi

echo
echo "STAGE 1: initialization grid (declared rule, no validation consulted)"
run_child init_grid "$PY" -u -m experiments.s5_wwj.init_grid \
  --out "$RUN_ROOT/init_grid.json" --seed "$SEED"

# placeholders for DRY_RUN only; the real path reads both from the grid
TAU_INIT="0.05"
EPS_INIT="0.0625"
if [[ "$DRY_RUN" != "1" ]]; then
  TAU_INIT="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected"]["tau_init"])' "$RUN_ROOT/init_grid.json")"
  EPS_INIT="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected"]["eps_init_passive"])' "$RUN_ROOT/init_grid.json")"
  echo "selected by the declared rule: tau_init=$TAU_INIT eps_init=$EPS_INIT"
fi

# Each arm is gated on its own, in its own directory, and a failure in one
# arm stops the script: a gate that has not run has not passed.
stage=2
for ARM in "${WWJ_ARMS[@]}"; do
  echo
  echo "STAGE $stage.1: performance gate, Native S5 versus $ARM"
  run_child "$ARM-benchmark" "$PY" -u -m experiments.s5_wwj.benchmark \
    --out "$RUN_ROOT/$ARM/benchmark.json" --arm "$ARM" --seed "$SEED" \
    --steps "$BENCH_STEPS" --tau-init "$TAU_INIT" --eps-init "$EPS_INIT" \
    --allocation-hours "$ALLOCATION_HOURS" --waves "$WAVES"

  echo
  echo "STAGE $stage.2: developmental learning gate for $ARM (never the test split)"
  run_child "$ARM-dev_gate" "$PY" -u -m experiments.s5_wwj.dev_gate \
    --data-cache "$DATA_CACHE" --out "$RUN_ROOT/$ARM/dev_gate" --arm "$ARM" \
    --seed "$SEED" --subset-steps "$SUBSET_STEPS" \
    --tau-init "$TAU_INIT" --eps-init "$EPS_INIT"
  stage=$((stage + 1))
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN: nothing started; no training run and no Slurm job"
  exit 0
fi

echo
echo "artifacts:"
echo "  $RUN_ROOT/init_grid.json"
for ARM in "${WWJ_ARMS[@]}"; do
  echo "  $RUN_ROOT/$ARM/benchmark.json"
  echo "  $RUN_ROOT/$ARM/dev_gate/dev_gate.json"
done
echo "Read them before authorizing any 15-epoch experiment. Each arm is"
echo "authorized only by its OWN artifacts, and a passing gate authorizes"
echo "nothing by itself."
