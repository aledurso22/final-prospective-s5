#!/usr/bin/env bash
# TSS pilot, PART A ONLY: checks, then the four-arm forward-model comparison.
#
# Part B is already COMPLETE at commit 6000b26 with artifacts under
# /Users/durso/s5-runs/tss_pilot/20260916-020800/part_b. It is a
# frozen-parameter probe with fixed inputs, seeds and filters that reproduced
# identically on five dispatches, so it is NOT re-run here; its provenance is
# recorded in Part A's status file instead. Nothing about the experiment
# changes: same four arms, three seeds, 300 updates, equations, data and
# tolerances.
#
# BUDGET, from measured costs at 6000b26 (logs 20260916-020800):
#
#   environment + GPU probe                        ~4 s
#   focused checks, 50 tests                      ~174 s
#   Part A setup: task, leakage probe, preflight   ~30 s   (profiling OFF)
#   Part A training, projected                    ~188 s   (68.5 arm + 120 host)
#   cleanup reserve                                 30 s
#   --------------------------------------------------- 
#   total                                         ~426 s of 600 s
#
# The projected training figure is what the preflight itself will recompute and
# gate on; it is not assumed here. Part A now receives the WHOLE deadline
# because Part B does not run.
#
#   TSS_A_STATUS=PASS|INCOMPLETE|FAILED   TSS_A_EXIT=0|3|4
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"

TOTAL_S=${TOTAL_S:-600}
RESERVE_S=${RESERVE_S:-30}
START=$(date +%s)
DEADLINE=$(( START + TOTAL_S ))
STAMP=$(date +%Y%m%d-%H%M%S)

#: the COMPLETED Part B, recorded and not re-run
PART_B_RUN=${PART_B_RUN:-$PROSPECTIVE_RUNS/tss_pilot/20260916-020800/part_b}
PART_B_COMMIT=${PART_B_COMMIT:-6000b26cd8c982ca75e3ffe2e57c8932f3e56ace}

LOG_DIR=""
finish() {
  echo "TSS_A_STATUS=$1"; echo "TSS_A_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  echo "part B (preserved, not re-run): $PART_B_RUN @ $PART_B_COMMIT"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/tss_pilot}
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"

echo "commit  : $(git rev-parse HEAD)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD)"
echo "budget  : ${TOTAL_S}s total (checks INCLUDED), reserve ${RESERVE_S}s"
echo "logs    : $LOG_DIR"
echo "part B  : PRESERVED, not re-run -> $PART_B_RUN @ $PART_B_COMMIT"
[ -d "$PART_B_RUN" ] && echo "          (artifact directory present)" \
  || echo "          (WARNING: artifact directory not found at that path)"

# ---- 1. GPU verified first, itself bounded by the deadline
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
  "$PY" -u -m pytest tests/test_tss_pilot.py -q --durations=10 \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -24 "$LOG_DIR/checks.log"
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "focused checks hit the deadline"
fi
if [ "$rc" -ne 0 ]; then finish FAILED 4 "focused checks did not pass"; fi

# ---- 3. Part A, with the WHOLE remaining deadline
LEFT=$(( $(remaining) - RESERVE_S ))
echo "remaining for Part A: ${LEFT}s (setup + preflight + training)"
[ "$LEFT" -gt 60 ] || finish INCOMPLETE 3 "no time for Part A"
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.tss.pilot_a \
    --out_root "$OUT_ROOT" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    --part_b_run "$PART_B_RUN" --part_b_commit "$PART_B_COMMIT" \
    > "$LOG_DIR/part_a.log" 2>&1 || rc=$?
echo "--- Part A tail ---"; tail -40 "$LOG_DIR/part_a.log"
grep -E "TSS_A_STATUS|PREFLIGHT_A_PROJECTED|leakage|\[gate\]|\[FAIL\]|\[INCOMPLETE\]" \
  "$LOG_DIR/part_a.log" || true
echo "Part A raw exit: $rc"
echo "output=$OUT_ROOT/$STAMP/part_a"

case "$rc" in
  0) finish PASS 0 "" ;;
  3) finish INCOMPLETE 3 "Part A reported INCOMPLETE" ;;
  4) finish FAILED 4 "Part A failed a correctness check or produced non-finite output" ;;
  124|137) finish INCOMPLETE 3 "watchdog stopped Part A at the wall clock" ;;
  *) finish FAILED 4 "Part A failed with exit $rc" ;;
esac
