#!/usr/bin/env bash
# STAGE B: frozen-checkpoint predictive diagnostic for the prospective
# readout. NO TRAINING. NOT AUTHORIZED TO RUN until
# docs/PROSPECTIVE_READOUT_STAGE_B_PROTOCOL.md is cleared.
#
# ONE hard 600-second budget covers GPU startup, the existing law checks, this
# stage's checks, checkpoint and source verification, the diagnostic itself,
# integrity re-verification, the digest and the terminal verdict. Sources and
# the declared native checkpoints are READ ONLY. No retries, no trimming.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
source "$HERE/prospective_momentum_terminal.sh"
source "$HERE/prospective_momentum_replication_verify.sh"   # pm_worst_integrity
export PM_DIGEST_MODULE=experiments.prospective_momentum.readout_probe_summary

TOTAL_S=${TOTAL_S:-600}
RESERVE_S=${RESERVE_S:-30}
START=$(date +%s)
DEADLINE=$(( START + TOTAL_S ))
STAMP=$(date +%Y%m%d-%H%M%S)

early_exit() {
  echo "PROSPECTIVE_MOMENTUM_STATUS=FAILED"; echo "PROSPECTIVE_MOMENTUM_EXIT=4"
  echo "reason: $1"; exit 4
}

cluster_check_env || early_exit "environment check failed"
cd "$PROSPECTIVE_REPO" || early_exit "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/prospective-readout-probe}
SOURCE_RUN=${SOURCE_RUN:-$PROSPECTIVE_RUNS/prospective-momentum-replication/20260917-011842}
SOURCE_DIR="$SOURCE_RUN/sources"
CHECKPOINT_RUN="$PROSPECTIVE_RUNS/prospective-temporal-response/20260917-163003"
export PM_SOURCE_RUN="$SOURCE_RUN"
RUN_DIR="$OUT_ROOT/$STAMP"
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || early_exit "cannot create $LOG_DIR"

echo "commit  : $(git rev-parse HEAD)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD)"
echo "budget  : ${TOTAL_S}s total (checks and finalization INCLUDED), reserve ${RESERVE_S}s"
echo "source  : $SOURCE_DIR (READ ONLY)"
echo "ckpts   : $CHECKPOINT_RUN (READ ONLY; declared native endpoints)"
echo "out     : $RUN_DIR"
echo "logs    : $LOG_DIR"
[ -d "$SOURCE_DIR" ] || early_exit "designated source directory missing"
( cd "$SOURCE_DIR" && sha256sum manifest.json SHA256SUMS \
    src500_momentum_delta.msgpack src501_momentum_delta.msgpack \
    src502_momentum_delta.msgpack src503_momentum_delta.msgpack \
    ) > "$LOG_DIR/source_sha256_before.txt.partial" 2>&1 \
  || { cat "$LOG_DIR/source_sha256_before.txt.partial"; early_exit "source file missing; no substitute"; }
mv "$LOG_DIR/source_sha256_before.txt.partial" "$LOG_DIR/source_sha256_before.txt"
cat "$LOG_DIR/source_sha256_before.txt"
( cd "$CHECKPOINT_RUN" && sha256sum status.json \
    params/dev_native_full_B_seed500_u200.msgpack \
    params/final_native_full_B_seed501_u200.msgpack \
    params/final_native_full_B_seed502_u200.msgpack \
    params/final_native_full_B_seed503_u200.msgpack \
    ) > "$LOG_DIR/checkpoint_sha256_before.txt.partial" 2>&1 \
  || { cat "$LOG_DIR/checkpoint_sha256_before.txt.partial"; early_exit "declared checkpoint missing; no substitute"; }
mv "$LOG_DIR/checkpoint_sha256_before.txt.partial" "$LOG_DIR/checkpoint_sha256_before.txt"
cat "$LOG_DIR/checkpoint_sha256_before.txt"

pm_extra_verify() {
  local base="$LOG_DIR/checkpoint_sha256_before.txt" rc
  if [ ! -s "$base" ]; then
    PM_INTEGRITY_RC=$(pm_worst_integrity "$PM_INTEGRITY_RC" 3); return 0
  fi
  pm_bounded $(( DEADLINE - PM_VERIFY_END )) 1 \
    bash -c 'cd "$1" && sha256sum -c "$2"' _ "$CHECKPOINT_RUN" "$base" \
    > "$LOG_DIR/checkpoint_sha256_after.txt" 2>&1
  case "$PM_OUTCOME" in
    completed) if [ "$PM_RC" = "0" ]; then rc=0; else rc=1; fi ;;
    watchdog_term|watchdog_kill|not_started) rc=2 ;;
    *) rc=4 ;;
  esac
  echo "checkpoint verification: outcome=$PM_OUTCOME rc=$PM_RC integrity=$rc"
  cat "$LOG_DIR/checkpoint_sha256_after.txt" 2>/dev/null || true
  PM_INTEGRITY_RC=$(pm_worst_integrity "$PM_INTEGRITY_RC" "$rc")
}

STAGE_TERM_AT=$(( DEADLINE - RESERVE_S ))
stage_end() {
  if [ "$PM_OUTCOME" != completed ] || [ "$PM_RC" != "0" ]; then
    pm_finish "$RUN_DIR" "$1" "$PM_OUTCOME" "$PM_RC"
  fi
}

pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -c "import jax,sys; print('backend',jax.default_backend()); \
print('devices',jax.devices()); sys.exit(0 if jax.default_backend()=='gpu' else 4)" \
  > "$LOG_DIR/backend.txt" 2>&1
cat "$LOG_DIR/backend.txt"; echo "backend: outcome=$PM_OUTCOME rc=$PM_RC"
stage_end backend

pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m pytest tests/test_prospective_tss_containment.py -q \
  > "$LOG_DIR/law_checks.log" 2>&1
echo "--- law checks ---"; tail -6 "$LOG_DIR/law_checks.log"
echo "law checks: outcome=$PM_OUTCOME rc=$PM_RC (elapsed $(( $(date +%s) - START ))s)"
stage_end law_checks

pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m pytest tests/test_prospective_readout_probe.py -q -rP \
  --durations=10 > "$LOG_DIR/checks.log" 2>&1
echo "--- stage checks ---"; tail -20 "$LOG_DIR/checks.log"
echo "checks: outcome=$PM_OUTCOME rc=$PM_RC (elapsed $(( $(date +%s) - START ))s)"
stage_end checks

echo "remaining until the probe's TERM: $(( STAGE_TERM_AT - $(date +%s) ))s"
pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m experiments.prospective_momentum.readout_probe \
    --out_root "$OUT_ROOT" --source_run "$SOURCE_RUN" \
    --checkpoint_run "$CHECKPOINT_RUN" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    > "$LOG_DIR/probe.log" 2>&1
echo "--- probe tail ---"; tail -30 "$LOG_DIR/probe.log"
grep -E "PROSPECTIVE_MOMENTUM_STATUS|SOURCE_UNCHANGED|PROBE_PROJECTED|\[checkpoints\]|\[seed|\[stopping rule\]|\[!\]" \
  "$LOG_DIR/probe.log" || true
echo "probe: outcome=$PM_OUTCOME rc=$PM_RC"
pm_finish "$RUN_DIR" probe "$PM_OUTCOME" "$PM_RC"
