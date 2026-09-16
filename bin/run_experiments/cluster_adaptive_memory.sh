#!/usr/bin/env bash
# Adaptive associative-memory comparison: checks, calibration, selection, batch.
#
# ONE hard 600-second budget covers GPU startup, focused checks, calibration,
# compilation, preflight, 14 development runs, 21 final runs, evaluation and a
# 30-second serialization reserve.
#
# SEVEN arms. 35 runs x 200 updates = 7,000 optimizer steps. The preflight
# MEASURES all seven arms and projects the WHOLE batch - both stages, every
# validation pass and all 21 held-out evaluations. If it does not fit, the
# batch is not started and the measured obstruction is reported: no arm, slot,
# seed or update count is reduced and the cap is never raised.
#
#   ADAPTIVE_STATUS=PASS|INCOMPLETE|FAILED   ADAPTIVE_EXIT=0|3|4
# A completed unfavourable comparison is execution PASS with the performance
# screen FAILED. A numerical refusal is FAILED; deadline exhaustion is
# INCOMPLETE.
#
# The completed nested-memory study is NOT re-run and its artifacts are NOT
# touched: this launcher writes under its own output root.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"

TOTAL_S=${TOTAL_S:-600}
RESERVE_S=${RESERVE_S:-30}
START=$(date +%s)
DEADLINE=$(( START + TOTAL_S ))
STAMP=$(date +%Y%m%d-%H%M%S)

LOG_DIR=""
finish() {
  echo "ADAPTIVE_STATUS=$1"; echo "ADAPTIVE_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/adaptive-memory}
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"

echo "commit  : $(git rev-parse HEAD)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD)"
echo "budget  : ${TOTAL_S}s total (checks AND calibration INCLUDED),"
echo "          reserve ${RESERVE_S}s"
echo "out     : $OUT_ROOT/$STAMP"
echo "logs    : $LOG_DIR"
echo "note    : the completed nested-memory artifacts are not read or written"

# ---- 1. GPU first, bounded by the deadline
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 15 ] || finish INCOMPLETE 3 "no time for the backend probe"
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

# ---- 2. focused checks, inside the same cap
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 30 ] || finish INCOMPLETE 3 "no time for the focused checks"
rc=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_adaptive_memory.py -q --durations=10 \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -30 "$LOG_DIR/checks.log"
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "focused checks hit the deadline"
fi
if [ "$rc" -ne 0 ]; then finish FAILED 4 "focused checks did not pass"; fi

# ---- 3. the study: calibration, preflight, then the batch only if it fits
LEFT=$(( $(remaining) - RESERVE_S ))
echo "remaining for the study: ${LEFT}s (calibration + preflight + 35 runs)"
[ "$LEFT" -gt 60 ] || finish INCOMPLETE 3 "no time for the study"
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.adaptive_memory.study \
    --out_root "$OUT_ROOT" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    > "$LOG_DIR/study.log" 2>&1 || rc=$?
echo "--- study tail ---"; tail -48 "$LOG_DIR/study.log"
grep -E "ADAPTIVE_STATUS|PREFLIGHT_|\[preflight\]|\[selection\]|\[!\]|SCREEN" \
  "$LOG_DIR/study.log" || true
echo "study raw exit: $rc"
echo "output=$OUT_ROOT/$STAMP"

case "$rc" in
  0) finish PASS 0 "" ;;
  3) finish INCOMPLETE 3 "the study reported INCOMPLETE" ;;
  4) finish FAILED 4 "a correctness, calibration or runtime failure" ;;
  124|137) finish INCOMPLETE 3 "watchdog stopped the study at the wall clock" ;;
  *) finish FAILED 4 "the study failed with exit $rc" ;;
esac
