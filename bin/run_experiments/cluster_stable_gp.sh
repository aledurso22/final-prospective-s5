#!/usr/bin/env bash
# Stable generalized prospective continuation: checks, identity gate,
# preflight, nine paired runs from ONE saved Rawat checkpoint.
#
# ONE hard 1200 s budget covers backend startup, the focused checks, source
# restoration, the pre-training identity gate, compilation, preflight, all
# nine runs, validation and cleanup. GPU only. The test split is never opened.
#
#   STABLE_GP_STATUS=PASS|INCOMPLETE|FAILED   STABLE_GP_EXIT=0|3|4
# Numerical PASS is not performance success: the development screen verdict is
# printed separately as "[screen] DEVELOPMENT SUCCESS: True|False".
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
  echo "STABLE_GP_STATUS=$1"; echo "STABLE_GP_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
STAGE2_DIR=${STAGE2_DIR:-$PROSPECTIVE_RUNS/stage2}
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/stable-gp}
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"
[ -f "$STAGE2_DIR/manifest.jsonl" ] || \
  finish FAILED 4 "source blocker: no Stage 2 manifest at $STAGE2_DIR"

echo "commit : $(git rev-parse HEAD)"
echo "branch : $(git rev-parse --abbrev-ref HEAD)"
echo "budget : ${TOTAL_S}s total (checks INCLUDED), reserve ${RESERVE_S}s"
echo "source : $STAGE2_DIR (read only)"
echo "out    : $OUT_ROOT/$STAMP"
echo "logs   : $LOG_DIR"

# ---- 1. GPU first, bounded by the deadline
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

# ---- 2. focused checks, inside the same cap
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 30 ] || finish INCOMPLETE 3 "no time for the focused checks"
rc=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_stable_gp.py -q --durations=8 \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -30 "$LOG_DIR/checks.log"
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "focused checks hit the deadline"
fi
if [ "$rc" -ne 0 ]; then finish FAILED 4 "focused checks did not pass"; fi

# ---- 3. the screen: source, identity gate, preflight, nine runs
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 60 ] || finish INCOMPLETE 3 "no time for the screen"
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.gp.stable_gp_study \
    --data_cache "$PROSPECTIVE_DATA" --stage2_dir "$STAGE2_DIR" \
    --out_root "$OUT_ROOT" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    --matmul_precision highest > "$LOG_DIR/study.log" 2>&1 || rc=$?
echo "--- screen tail ---"; tail -60 "$LOG_DIR/study.log"
grep -E "STABLE_GP_STATUS|PREFLIGHT_|\[source\]|\[gate\]|\[epoch0\]|\[preflight\]|\[screen\]|\[!\]|BLOCKER" \
  "$LOG_DIR/study.log" || true
echo "screen raw exit: $rc"
echo "output=$OUT_ROOT/$STAMP"
case "$rc" in
  0) finish PASS 0 "" ;;
  3) finish INCOMPLETE 3 "screen reported INCOMPLETE" ;;
  4) finish FAILED 4 "a source, identity, domain or numerical failure" ;;
  124|137) finish INCOMPLETE 3 "watchdog stopped the screen at the wall clock" ;;
  *) finish FAILED 4 "screen failed with exit $rc (includes a BLOCKER refusal)" ;;
esac
