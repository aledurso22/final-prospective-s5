#!/usr/bin/env bash
# Stage 2 diagnostic: analysis of SAVED models. GPU only.
#
# ONE 20-minute budget covers EVERYTHING: backend verification, the focused
# numerical tests and the diagnostic itself (R4). The deadline is computed
# first and passed down, and an outer watchdog enforces it even if a single
# phase overruns.
#
# READ-ONLY on the Stage 2 runs. No training, no optimizer update, no test
# scoring. Output goes to a NEW directory; source hashes are compared before
# and after and reported.
#
# Each invocation is a FRESH RERUN into a new timestamped directory, not a
# resumption. To run only genuinely missing phases, pass e.g.
#   ONLY_PHASES=D1,D2 bash cluster_stage2_diagnostic.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"

TOTAL_S=${TOTAL_S:-1200}          # the whole cap, checks included
RESERVE_S=${RESERVE_S:-25}        # cleanup reserve kept inside the cap
START=$(date +%s)
DEADLINE=$(( START + TOTAL_S ))
STAMP=$(date +%Y%m%d-%H%M%S)

cluster_check_env || { echo "ENVIRONMENT CHECK FAILED"; exit 2; }
cd "$PROSPECTIVE_REPO" || exit 2

STAGE2_DIR=${STAGE2_DIR:-$PROSPECTIVE_RUNS/stage2}
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/stage2-diagnostics}
LOG_DIR="$OUT_ROOT/logs/$STAMP"          # unique per invocation (R4)
mkdir -p "$LOG_DIR" || { echo "cannot create $LOG_DIR"; exit 2; }

[ -d "$STAGE2_DIR" ] || { echo "MISSING Stage 2 runs: $STAGE2_DIR"; exit 2; }
echo "commit                   : $(git rev-parse HEAD)"
echo "branch                   : $(git rev-parse --abbrev-ref HEAD)"
echo "stage2 source (read-only): $STAGE2_DIR"
echo "diagnostic output        : $OUT_ROOT"
echo "total budget             : ${TOTAL_S}s (checks INCLUDED), reserve ${RESERVE_S}s"
echo "logs                     : $LOG_DIR"

# ---- 1. GPU verified BEFORE any numerical fixture (R4), inside the budget
"$PY" -u - > "$LOG_DIR/backend.txt" 2>&1 <<'PYEOF'
import jax, sys
print("backend", jax.default_backend())
print("devices", jax.devices())
print("jax", jax.__version__)
sys.exit(0 if jax.default_backend() == "gpu" else 3)
PYEOF
rc_backend=$?
cat "$LOG_DIR/backend.txt"
if [ "$rc_backend" -ne 0 ]; then
  echo "REFUSING: backend is not gpu (exit $rc_backend). No CPU fallback."
  exit 2
fi

remaining() { echo $(( DEADLINE - $(date +%s) )); }

# ---- 2. focused numerical validation, bounded by the SAME deadline
LEFT=$(( $(remaining) - RESERVE_S ))
if [ "$LEFT" -le 30 ]; then
  echo "NO TIME for focused tests within the cap; refusing to start."
  exit 3
fi
rc_t=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_stage2_diagnostic.py -q \
  > "$LOG_DIR/adapter_tests.log" 2>&1 || rc_t=$?
echo "--- focused tests ---"; tail -6 "$LOG_DIR/adapter_tests.log"
echo "focused tests exit: $rc_t  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc_t" -ne 0 ]; then
  echo "REFUSING to run the diagnostic: focused validation did not pass."
  exit 4
fi

# ---- 3. the diagnostic, sharing the same deadline, with an outer watchdog
LEFT=$(( $(remaining) - RESERVE_S ))
if [ "$LEFT" -le 20 ]; then
  echo "BUDGET EXHAUSTED by the checks; diagnostic not started."
  echo "DIAGNOSTIC_STATUS=INCOMPLETE"
  exit 3
fi
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.gp.stage2_diagnostic \
    --stage2_dir "$STAGE2_DIR" --out_root "$OUT_ROOT" \
    --data_cache "$PROSPECTIVE_DATA" --deadline "$DEADLINE" \
    --reserve_s "$RESERVE_S" \
    ${ONLY_PHASES:+--only_phases "$ONLY_PHASES"} \
    --matmul_precision highest > "$LOG_DIR/diagnostic.log" 2>&1 || rc=$?
echo "--- diagnostic tail ---"; tail -35 "$LOG_DIR/diagnostic.log"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "WATCHDOG: the diagnostic hit the wall clock and was stopped."
  echo "DIAGNOSTIC_STATUS=INCOMPLETE (timeout)"
fi
grep -E "source hashes unchanged|DIAGNOSTIC_STATUS|\[FAIL\]" \
     "$LOG_DIR/diagnostic.log" || true
echo "DIAGNOSTIC_EXIT=$rc   total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
echo "logs=$LOG_DIR"
exit $rc
