#!/usr/bin/env bash
# Learned response timescale: focused checks, preflight, five-arm screen.
#
# ONE hard 1200 s budget covers backend startup, checks, preflight,
# compilation, all five runs, validation and cleanup. GPU only, no CPU
# fallback. The test split is never opened.
#
#   TIMESCALE_STATUS=PASS|INCOMPLETE|FAILED   TIMESCALE_EXIT=0|3|4
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
  echo "TIMESCALE_STATUS=$1"; echo "TIMESCALE_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/timescale}
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"

echo "commit : $(git rev-parse HEAD)"
echo "branch : $(git rev-parse --abbrev-ref HEAD)"
echo "budget : ${TOTAL_S}s total (checks INCLUDED), reserve ${RESERVE_S}s"
echo "logs   : $LOG_DIR"

# ---- 1. GPU verified first, itself bounded by the deadline
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

# ---- 2. focused numerical checks, inside the same cap
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 30 ] || finish INCOMPLETE 3 "no time for the focused checks"
rc=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_learned_timescale.py -q \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -20 "$LOG_DIR/checks.log"
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "focused checks hit the deadline"
fi
if [ "$rc" -ne 0 ]; then finish FAILED 4 "focused checks did not pass"; fi

# ---- 3. the screen: preflight, initialization check, five arms
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 60 ] || finish INCOMPLETE 3 "no time for the screen"
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.gp.timescale_study \
    --data_cache "$PROSPECTIVE_DATA" --out_root "$OUT_ROOT" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    ${STAGE2_CONFIG:+--stage2_config "$STAGE2_CONFIG"} \
    --matmul_precision highest > "$LOG_DIR/study.log" 2>&1 || rc=$?
echo "--- screen tail ---"; tail -45 "$LOG_DIR/study.log"
grep -E "TIMESCALE_STATUS|PREFLIGHT_PROJECTED|\[gate\]|\[isolating\]|\[STOP\]|\[INCOMPLETE\]" \
  "$LOG_DIR/study.log" || true
echo "screen raw exit: $rc"
case "$rc" in
  0) finish PASS 0 "" ;;
  3) finish INCOMPLETE 3 "screen reported INCOMPLETE or stopped at the initialization check" ;;
  124|137) finish INCOMPLETE 3 "watchdog stopped the screen at the wall clock" ;;
  *) finish FAILED 4 "screen failed with exit $rc" ;;
esac
