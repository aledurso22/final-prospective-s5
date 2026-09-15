#!/usr/bin/env bash
# TSS-based prospective pilot: checks, then Part A, then Part B.
#
# ONE hard 600 s budget covers backend startup, focused checks, compilation,
# preflight, both parts and cleanup. GPU only, no CPU fallback.
#
# Part A tests a temporal FORWARD model under exact BPTT. Part B tests CREDIT
# ASSIGNMENT on a frozen forward model. They answer different questions and a
# result from one is not evidence for the other.
#
#   TSS_STATUS=PASS|INCOMPLETE|FAILED   TSS_EXIT=0|3|4
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
  echo "TSS_STATUS=$1"; echo "TSS_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/tss_pilot}
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"

echo "commit : $(git rev-parse HEAD)"
echo "branch : $(git rev-parse --abbrev-ref HEAD)"
echo "budget : ${TOTAL_S}s total (checks INCLUDED), reserve ${RESERVE_S}s"
echo "logs   : $LOG_DIR"

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
  "$PY" -u -m pytest tests/test_tss_pilot.py -q \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -20 "$LOG_DIR/checks.log"
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "focused checks hit the deadline"
fi
if [ "$rc" -ne 0 ]; then finish FAILED 4 "focused checks did not pass"; fi

RUN_ID="$STAMP"
rc_a=0; rc_b=0

# R9: an explicit sub-deadline per part. Part A is capped before the end of the
# budget so that Part B cannot be squeezed into a partial comparison, and each
# runner enforces its own deadline in its inner loop rather than relying on the
# outer watchdog. PART_B_RESERVE_S is measured against Part B's own preflight.
PART_B_RESERVE_S=${PART_B_RESERVE_S:-200}
DEADLINE_A=$(( DEADLINE - PART_B_RESERVE_S ))

# ---- 3. Part A: forward-model comparison under exact BPTT
LEFT=$(( $(remaining) - RESERVE_S ))
if [ "$LEFT" -le 40 ]; then
  echo "BUDGET: Part A NOT STARTED (recorded, not shortened)."
  rc_a=3
else
  timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
    "$PY" -u -m experiments.tss.pilot_a \
      --out_root "$OUT_ROOT" --run_id "$RUN_ID" \
      --deadline "$DEADLINE_A" --reserve_s "$RESERVE_S" \
      > "$LOG_DIR/part_a.log" 2>&1 || rc_a=$?
  echo "--- Part A tail ---"; tail -30 "$LOG_DIR/part_a.log"
  grep -E "TSS_A_STATUS|PREFLIGHT_A_PROJECTED|leakage|\[INCOMPLETE\]" \
    "$LOG_DIR/part_a.log" || true
  echo "Part A raw exit: $rc_a"
fi

# R9: fail FAST. A correctness or runtime failure in Part A must not be
# followed by Part B producing a misleading partial comparison beside it.
if [ "$rc_a" -eq 4 ]; then
  finish FAILED 4 "Part A failed a correctness check or produced non-finite \
output; Part B was NOT started"
fi

# ---- 4. Part B: credit-assignment probe on a frozen forward model
LEFT=$(( $(remaining) - RESERVE_S ))
if [ "$LEFT" -le 20 ]; then
  echo "BUDGET: Part B NOT STARTED (recorded, not shortened)."
  rc_b=3
else
  timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
    "$PY" -u -m experiments.tss.pilot_b \
      --out_root "$OUT_ROOT" --run_id "$RUN_ID" \
      --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
      > "$LOG_DIR/part_b.log" 2>&1 || rc_b=$?
  echo "--- Part B tail ---"; tail -30 "$LOG_DIR/part_b.log"
  grep -E "TSS_B_STATUS|\[audit\]|\[INCOMPLETE\]" "$LOG_DIR/part_b.log" || true
  echo "Part B raw exit: $rc_b"
fi

echo "output=$OUT_ROOT/$RUN_ID"
for rc in "$rc_a" "$rc_b"; do
  case "$rc" in
    0|3|124|137) ;;
    *) finish FAILED 4 "a part failed with exit $rc" ;;
  esac
done
# An unfavourable SCIENTIFIC verdict is a result and leaves the status PASS;
# only execution faults change it. The two are reported separately.
if [ "$rc_a" -eq 0 ] && [ "$rc_b" -eq 0 ]; then
  finish PASS 0 ""
fi
finish INCOMPLETE 3 "one or both parts reported INCOMPLETE or hit the wall clock"
