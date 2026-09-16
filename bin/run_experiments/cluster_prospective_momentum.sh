#!/usr/bin/env bash
# Prospective correction of a trained Momentum DeltaNet: checks, source
# restoration, preflight, six-arm continuation. NOT AUTHORIZED TO RUN until
# docs/PROSPECTIVE_MOMENTUM_PROTOCOL.md is cleared.
#
# ONE hard 600-second budget covers GPU startup, focused checks, checkpoint
# restoration and reproduction, preflight, 12 development runs, 18 final runs,
# evaluation, the digest and a 30-second reserve. Preflight refuses the batch
# (INCOMPLETE) if it cannot fit and FAILS it on invalid numerical state;
# nothing is trimmed, retried or extended.
#
#   PROSPECTIVE_MOMENTUM_STATUS=PASS|INCOMPLETE|FAILED  ..._EXIT=0|3|4
# Execution PASS is not a performance verdict; the screens are separate.
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
  echo "PROSPECTIVE_MOMENTUM_STATUS=$1"; echo "PROSPECTIVE_MOMENTUM_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/prospective-momentum}
SOURCE_DIR=${SOURCE_DIR:-$PROSPECTIVE_RUNS/meta-delta/20260916-222310}
export PM_SOURCE_DIR="$SOURCE_DIR"
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"

echo "commit  : $(git rev-parse HEAD)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD)"
echo "budget  : ${TOTAL_S}s total (checks INCLUDED), reserve ${RESERVE_S}s"
echo "source  : $SOURCE_DIR (READ ONLY)"
echo "out     : $OUT_ROOT/$STAMP"
echo "logs    : $LOG_DIR"
[ -d "$SOURCE_DIR" ] || finish FAILED 4 "designated source directory missing"
( cd "$SOURCE_DIR" && sha256sum status.json selection.json \
    params/dev_momentum_delta_B_seed300.msgpack \
    params/dev_gated_delta_B_seed300.msgpack \
    params/dev_gp_two_sided_B_seed300.msgpack \
    params/dev_tss_eq17_B_seed300.msgpack ) > "$LOG_DIR/source_sha256_before.txt" 2>&1 \
  || { cat "$LOG_DIR/source_sha256_before.txt"; \
       finish FAILED 4 "designated source file missing; no substitute"; }
cat "$LOG_DIR/source_sha256_before.txt"

source_after() {
  ( cd "$SOURCE_DIR" && sha256sum -c "$LOG_DIR/source_sha256_before.txt" ) \
    > "$LOG_DIR/source_sha256_after.txt" 2>&1
  echo "source unchanged check (exit $?):"; cat "$LOG_DIR/source_sha256_after.txt"
}

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
  source_after; finish INCOMPLETE 3 "backend probe timed out"
fi
if [ "$rc" -ne 0 ]; then source_after; finish FAILED 4 "backend is not gpu (raw exit $rc)"; fi

# ---- 2. focused checks, inside the same cap
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 30 ] || { source_after; finish INCOMPLETE 3 "no time for the focused checks"; }
rc=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_prospective_momentum.py -q -rP --durations=10 \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -30 "$LOG_DIR/checks.log"
echo "--- measured magnitudes (passing and failing) ---"
grep -nE "^  [a-zA-Z]" "$LOG_DIR/checks.log" | head -80 || true
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  source_after; finish INCOMPLETE 3 "focused checks hit the deadline"
fi
if [ "$rc" -ne 0 ]; then source_after; finish FAILED 4 "focused checks did not pass"; fi

# ---- 3. the study: restore/reproduce, preflight, batch only if it fits
LEFT=$(( $(remaining) - RESERVE_S ))
echo "remaining for the study: ${LEFT}s"
[ "$LEFT" -gt 60 ] || { source_after; finish INCOMPLETE 3 "no time for the study"; }
rc=0
timeout --kill-after="${RESERVE_S}s" --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.prospective_momentum.study \
    --out_root "$OUT_ROOT" --source_dir "$SOURCE_DIR" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    > "$LOG_DIR/study.log" 2>&1 || rc=$?
echo "--- study tail ---"; tail -40 "$LOG_DIR/study.log"
grep -E "PROSPECTIVE_MOMENTUM_STATUS|SOURCE_UNCHANGED|PREFLIGHT_|\[source\]|\[preflight\]|\[screen\]|\[!\]" \
  "$LOG_DIR/study.log" || true
echo "study raw exit: $rc"
echo "output=$OUT_ROOT/$STAMP"
source_after

# ---- 4. the pre-written digest, whatever the outcome (read-only JSON)
if [ -f "$OUT_ROOT/$STAMP/status.json" ]; then
  timeout 20s "$PY" -u -m experiments.prospective_momentum.summary \
    "$OUT_ROOT/$STAMP" > "$LOG_DIR/digest.txt" 2>&1 || true
  echo "--- digest ($LOG_DIR/digest.txt) ---"; cat "$LOG_DIR/digest.txt"
fi

case "$rc" in
  0) finish PASS 0 "" ;;
  3) finish INCOMPLETE 3 "the study reported INCOMPLETE" ;;
  4) finish FAILED 4 "a correctness, source or runtime failure" ;;
  124|137) finish INCOMPLETE 3 "watchdog stopped the study at the wall clock" ;;
  *) finish FAILED 4 "the study failed with exit $rc" ;;
esac
