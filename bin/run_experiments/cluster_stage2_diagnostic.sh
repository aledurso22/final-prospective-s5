#!/usr/bin/env bash
# Stage 2 diagnostic: analysis of SAVED models. GPU only.
#
# ONE 20-minute budget covers EVERYTHING: backend verification, the focused
# numerical tests and the diagnostic. Every numerical subprocess, the backend
# probe included, is bounded by the remaining deadline.
#
# READ-ONLY on the Stage 2 runs. No training, no optimizer update, no test
# scoring. Output goes to a NEW directory; source hashes are compared before
# and after and reported.
#
# Status contract, printed on EVERY terminal path:
#   DIAGNOSTIC_STATUS=PASS|INCOMPLETE|FAILED   DIAGNOSTIC_EXIT=0|3|4
#   PASS       0  all required checks passed
#   INCOMPLETE 3  timeout, exhausted budget, or intentionally unexecuted work
#   FAILED     4  a required check failed, or a runtime/environment failure
# Raw subprocess statuses (124/137 from the watchdog, pytest codes) are kept in
# the logs and echoed, but the launcher's own exit follows the contract above.
#
# Each invocation is a FRESH PARTIAL DIAGNOSTIC, not a resume. To run selected
# phases use ONLY_PHASES; prerequisites are auto-included by the runner, so
#   ONLY_PHASES=D1,D2   actually runs A,B,B2,D1,D2
# because checkpoint loading and its restore checks live in B2. The restore
# checks therefore REPEAT on every partial run; the analytical phases you omit
# are simply not produced.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"

TOTAL_S=${TOTAL_S:-1200}
RESERVE_S=${RESERVE_S:-25}
START=$(date +%s)
DEADLINE=$(( START + TOTAL_S ))
STAMP=$(date +%Y%m%d-%H%M%S)

finish() {   # status, exit code, message
  echo "DIAGNOSTIC_STATUS=$1"
  echo "DIAGNOSTIC_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "${LOG_DIR:-}" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

LOG_DIR=""
cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"

STAGE2_DIR=${STAGE2_DIR:-$PROSPECTIVE_RUNS/stage2}
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/stage2-diagnostics}
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"
[ -d "$STAGE2_DIR" ] || finish FAILED 4 "missing Stage 2 runs: $STAGE2_DIR"

echo "commit                   : $(git rev-parse HEAD)"
echo "branch                   : $(git rev-parse --abbrev-ref HEAD)"
echo "stage2 source (read-only): $STAGE2_DIR"
echo "diagnostic output        : $OUT_ROOT"
echo "total budget             : ${TOTAL_S}s (checks INCLUDED), reserve ${RESERVE_S}s"
echo "phases                   : ${ONLY_PHASES:-all}"
echo "logs                     : $LOG_DIR"

# ---- 1. GPU verified before any numerical fixture, ITSELF under a timeout
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 20 ] || finish INCOMPLETE 3 "no time for the backend probe"
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u - > "$LOG_DIR/backend.txt" 2>&1 <<'PYEOF'
import jax, sys
print("backend", jax.default_backend())
print("devices", jax.devices())
print("jax", jax.__version__)
sys.exit(0 if jax.default_backend() == "gpu" else 3)
PYEOF
rc_backend=$?
cat "$LOG_DIR/backend.txt"
echo "backend probe raw exit: $rc_backend"
if [ "$rc_backend" -eq 124 ] || [ "$rc_backend" -eq 137 ]; then
  finish INCOMPLETE 3 "backend initialization exceeded the deadline"
fi
[ "$rc_backend" -eq 0 ] || finish FAILED 4 \
  "backend is not gpu (raw exit $rc_backend); no CPU fallback"

# ---- 2. focused numerical validation, same deadline
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 30 ] || finish INCOMPLETE 3 "no time for the focused tests"
rc_t=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_stage2_diagnostic.py -q \
  > "$LOG_DIR/adapter_tests.log" 2>&1 || rc_t=$?
echo "--- focused tests ---"; tail -8 "$LOG_DIR/adapter_tests.log"
echo "focused tests raw exit: $rc_t  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc_t" -eq 124 ] || [ "$rc_t" -eq 137 ]; then
  finish INCOMPLETE 3 "focused tests hit the deadline"
fi
[ "$rc_t" -eq 0 ] || finish FAILED 4 "focused validation did not pass"

# ---- 3. the diagnostic, sharing the deadline, with a watchdog
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 20 ] || finish INCOMPLETE 3 "budget exhausted by the checks"
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.gp.stage2_diagnostic \
    --stage2_dir "$STAGE2_DIR" --out_root "$OUT_ROOT" \
    --data_cache "$PROSPECTIVE_DATA" --deadline "$DEADLINE" \
    --reserve_s "$RESERVE_S" \
    ${ONLY_PHASES:+--only_phases "$ONLY_PHASES"} \
    --matmul_precision highest > "$LOG_DIR/diagnostic.log" 2>&1 || rc=$?
echo "--- diagnostic tail ---"; tail -35 "$LOG_DIR/diagnostic.log"
grep -E "source hashes unchanged|DIAGNOSTIC_STATUS|\[FAIL\]|RUNTIME ERROR" \
     "$LOG_DIR/diagnostic.log" || true
echo "diagnostic raw exit: $rc"

# normalize the raw status to the documented contract
case "$rc" in
  0) finish PASS 0 "" ;;
  3) finish INCOMPLETE 3 "runner reported INCOMPLETE" ;;
  4) finish FAILED 4 "runner reported FAILED" ;;
  124|137) finish INCOMPLETE 3 "watchdog stopped the diagnostic at the wall clock" ;;
  *) finish FAILED 4 "unexpected runner exit $rc" ;;
esac
