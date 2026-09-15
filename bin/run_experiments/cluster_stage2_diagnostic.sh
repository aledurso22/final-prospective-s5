#!/usr/bin/env bash
# Stage 2 diagnostic: analysis of SAVED models. GPU only, 20-minute cap.
#
# READ-ONLY on /Users/durso/s5-runs/stage2. No training, no optimizer update,
# no test scoring. Output goes to a NEW directory. Source file hashes are
# compared before and after and the run reports whether they changed.
#
# Restartable: each invocation creates its own output directory and repeats the
# whole bounded diagnostic; nothing is resumed into the source runs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cluster_check_env || { echo "ENVIRONMENT CHECK FAILED"; exit 2; }
cd "$PROSPECTIVE_REPO" || exit 2

STAGE2_DIR=${STAGE2_DIR:-$PROSPECTIVE_RUNS/stage2}
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/stage2-diagnostics}
BUDGET_S=${BUDGET_S:-1200}
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR" || { echo "cannot create $LOG_DIR"; exit 2; }
LOG="$LOG_DIR/diagnostic-$(date +%Y%m%d-%H%M%S).log"

[ -d "$STAGE2_DIR" ] || { echo "MISSING Stage 2 runs: $STAGE2_DIR"; exit 2; }
echo "stage2 source (read-only): $STAGE2_DIR"
echo "diagnostic output        : $OUT_ROOT"
echo "GPU budget               : ${BUDGET_S}s"
echo "log                      : $LOG"

# focused pre-checks only: the adapter tests, NOT the historical suite
rc_t=0
"$PY" -u -m pytest tests/test_stage2_diagnostic.py -q \
    > "$LOG_DIR/adapter_tests.log" 2>&1 || rc_t=$?
echo "--- adapter tests ---"; tail -4 "$LOG_DIR/adapter_tests.log"
echo "adapter tests exit: $rc_t"
if [ "$rc_t" -ne 0 ]; then
  echo "REFUSING to run the diagnostic: the adapter is not verified."
  exit 2
fi

rc=0
"$PY" -u -m experiments.gp.stage2_diagnostic \
    --stage2_dir "$STAGE2_DIR" --out_root "$OUT_ROOT" \
    --data_cache "$PROSPECTIVE_DATA" --budget_s "$BUDGET_S" \
    --matmul_precision highest > "$LOG" 2>&1 || rc=$?
echo "--- diagnostic tail ---"; tail -30 "$LOG"
echo "DIAGNOSTIC_EXIT=$rc  log=$LOG"
grep -E "source hashes unchanged|DIAGNOSTIC_DONE" "$LOG" || true
exit $rc
