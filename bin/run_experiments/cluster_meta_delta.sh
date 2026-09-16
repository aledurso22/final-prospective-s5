#!/usr/bin/env bash
# Generalized prospective memory around the delta boundary: checks,
# preflight, six-arm batch. NOT AUTHORIZED TO RUN until the protocol
# docs/META_DELTA_PROTOCOL.md is cleared.
#
# ONE hard 600-second budget covers GPU startup, focused checks, preflight,
# 12 development runs, 18 final runs, evaluation and a 30-second reserve.
# Preflight refuses the batch (INCOMPLETE) if it cannot fit and FAILS it on
# invalid numerical state; nothing is trimmed and the cap is never raised.
#
#   META_DELTA_STATUS=PASS|INCOMPLETE|FAILED   META_DELTA_EXIT=0|3|4
# A completed unfavourable comparison is execution PASS; the three screens
# (literature, ordinary prospectivity, T Rdot attribution) are separate.
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
  echo "META_DELTA_STATUS=$1"; echo "META_DELTA_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/meta-delta}
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
# -rP prints the captured output of PASSING tests too, so measured
# magnitudes (not only "below tolerance") are preserved in checks.log
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_meta_delta.py -q -rP --durations=10 \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -30 "$LOG_DIR/checks.log"
echo "--- measured magnitudes (passing and failing) ---"
grep -nE "^  [a-zA-Z]" "$LOG_DIR/checks.log" | head -60 || true
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "focused checks hit the deadline"
fi
if [ "$rc" -ne 0 ]; then finish FAILED 4 "focused checks did not pass"; fi

# ---- 3. the study: calibration, preflight, then the batch only if it fits
LEFT=$(( $(remaining) - RESERVE_S ))
echo "remaining for the study: ${LEFT}s (calibration + preflight + 30 runs)"
[ "$LEFT" -gt 60 ] || finish INCOMPLETE 3 "no time for the study"
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.meta_delta.study \
    --out_root "$OUT_ROOT" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    > "$LOG_DIR/study.log" 2>&1 || rc=$?
echo "--- study tail ---"; tail -48 "$LOG_DIR/study.log"
grep -E "META_DELTA_STATUS|PREFLIGHT_|\[preflight\]|\[selection\]|\[!\]|SCREEN" \
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
