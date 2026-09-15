#!/usr/bin/env bash
# Memory-recall study: focused checks, then the bounded comparison.
#
# ONE hard 1200 s budget covers backend startup, checks, compilation, warm-ups,
# every continuation, evaluation and cleanup. GPU only, no CPU fallback.
#
#   RECALL_STATUS=PASS|INCOMPLETE|FAILED   RECALL_EXIT=0|3|4
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"

TOTAL_S=${TOTAL_S:-1200}
RESERVE_S=${RESERVE_S:-40}
START=$(date +%s)
DEADLINE=$(( START + TOTAL_S ))
STAMP=$(date +%Y%m%d-%H%M%S)

LOG_DIR=""
finish() {
  echo "RECALL_STATUS=$1"; echo "RECALL_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/recall}
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"

echo "commit : $(git rev-parse HEAD)"
echo "branch : $(git rev-parse --abbrev-ref HEAD)"
echo "budget : ${TOTAL_S}s total (checks INCLUDED), reserve ${RESERVE_S}s"
echo "logs   : $LOG_DIR"

LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 20 ] || finish INCOMPLETE 3 "no time for the backend probe"
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -c "import jax,sys; print('backend',jax.default_backend()); \
print('devices',jax.devices()); print('jax',jax.__version__); \
sys.exit(0 if jax.default_backend()=='gpu' else 3)" \
  > "$LOG_DIR/backend.txt" 2>&1
rc=$?; cat "$LOG_DIR/backend.txt"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "backend probe timed out"
fi
if [ "$rc" -ne 0 ]; then finish FAILED 4 "backend is not gpu (raw exit $rc)"; fi

LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 30 ] || finish INCOMPLETE 3 "no time for the focused checks"
rc=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_recall_study.py -q \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -10 "$LOG_DIR/checks.log"
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "focused checks hit the deadline"
fi
if [ "$rc" -ne 0 ]; then finish FAILED 4 "focused checks did not pass"; fi

LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 60 ] || finish INCOMPLETE 3 "no time for the study"
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.gp.recall_study \
    --out_root "$OUT_ROOT" --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    --matmul_precision highest > "$LOG_DIR/study.log" 2>&1 || rc=$?
echo "--- study tail ---"; tail -40 "$LOG_DIR/study.log"
grep -E "RECALL_STATUS|\[gate\]|\[STOP\]|\[INCOMPLETE\]" "$LOG_DIR/study.log" || true
echo "study raw exit: $rc"
case "$rc" in
  0) finish PASS 0 "" ;;
  3) finish INCOMPLETE 3 "study reported INCOMPLETE or stopped at the gate" ;;
  124|137) finish INCOMPLETE 3 "watchdog stopped the study at the wall clock" ;;
  *) finish FAILED 4 "study failed with exit $rc" ;;
esac
