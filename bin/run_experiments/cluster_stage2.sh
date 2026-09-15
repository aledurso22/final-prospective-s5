#!/usr/bin/env bash
# Stage 2: the bounded validation-only development batch.
#
#   depth 4, width 32, MFCC, development seed 100, at most TWO candidate
#   learning rates per candidate family, at most 10 epochs each, VALIDATION
#   ONLY. The reference arm native_s5 gets ONE unswept configuration.
#   Hard cap: two GPU-hours total, enforced by a wall-clock budget check
#   BETWEEN runs. Nothing here touches the test split.
#
# Resumable: each run keeps its own run_dir; re-running the script continues an
# interrupted run from its last checkpoint instead of restarting it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cluster_check_env || { echo "ENVIRONMENT CHECK FAILED"; exit 2; }
cd "$PROSPECTIVE_REPO" || exit 2

BUDGET_S=${BUDGET_S:-7200}          # two GPU-hours, the predeclared cap
EPOCHS=${EPOCHS:-10}
SEED=${SEED:-100}
OUT=${OUT:-$PROSPECTIVE_RUNS/stage2}
mkdir -p "$OUT"
START=$(date +%s)
MANIFEST="$OUT/manifest.jsonl"
echo "stage2 output: $OUT   budget ${BUDGET_S}s   epochs $EPOCHS   seed $SEED"

run_one() {
  local arm=$1 lr=$2
  local tag="${arm}__lr${lr}__seed${SEED}"
  local rd="$OUT/$tag"
  local elapsed=$(( $(date +%s) - START ))
  if [ "$elapsed" -ge "$BUDGET_S" ]; then
    echo "BUDGET REACHED before $tag (${elapsed}s >= ${BUDGET_S}s). NOT STARTED."
    echo "{\"run\":\"$tag\",\"status\":\"not_started_budget\"}" >> "$MANIFEST"
    return 0
  fi
  echo "=== $tag  (elapsed ${elapsed}s) ==="
  local rc=0
  "$PY" -m experiments.gp.rawat_benchmark --mode train --arm "$arm" \
      --data_cache "$PROSPECTIVE_DATA" --run_dir "$rd" \
      --epochs "$EPOCHS" --lr "$lr" --seed "$SEED" \
      --matmul_precision highest >> "$rd.log" 2>&1 || rc=$?
  tail -3 "$rd.log"
  echo "exit: $rc"
  echo "{\"run\":\"$tag\",\"arm\":\"$arm\",\"lr\":$lr,\"seed\":$SEED,\"exit\":$rc,\"run_dir\":\"$rd\"}" >> "$MANIFEST"
}

# four candidate families x two learning rates, plus one reference config
for arm in gain_clip_s5 alpha_p_s5 gp_fixed_m0 gp_fixed_mass; do
  for lr in 1e-3 3e-4; do
    run_one "$arm" "$lr"
  done
done
run_one native_s5 1e-3

echo "STAGE2 done. elapsed $(( $(date +%s) - START ))s   manifest=$MANIFEST"
