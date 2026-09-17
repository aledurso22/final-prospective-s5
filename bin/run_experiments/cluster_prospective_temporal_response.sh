#!/usr/bin/env bash
# TEMPORAL RESPONSE: can generalized prospective processing keep literal TSS's
# immediate revision benefit while improving later recall and reducing
# interference with untouched associations? NOT AUTHORIZED TO RUN until
# docs/PROSPECTIVE_TEMPORAL_RESPONSE_PROTOCOL.md is cleared.
#
# ONE hard 600-second budget covers GPU startup, the TSS-containment focused
# checks (float64 module and float32 probe, unchanged), this study's focused
# checks, task and start-point checks, the mechanism diagnostic, measured
# preflight, all training (<= 4,600 updates), held-out evaluation, kill grace,
# source verification, the digest and the terminal verdict. Sources of run
# 20260917-011842 are REUSED READ-ONLY. No retries, no trimming.
#
#   PROSPECTIVE_MOMENTUM_STATUS=PASS|INCOMPLETE|FAILED  ..._EXIT=0|3|4
# PASS is operational only; screens are separate performance verdicts.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
source "$HERE/prospective_momentum_terminal.sh"
export PM_DIGEST_MODULE=experiments.prospective_momentum.temporal_response_summary

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
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/prospective-temporal-response}
SOURCE_RUN=${SOURCE_RUN:-$PROSPECTIVE_RUNS/prospective-momentum-replication/20260917-011842}
SOURCE_DIR="$SOURCE_RUN/sources"
export PM_SOURCE_RUN="$SOURCE_RUN"
RUN_DIR="$OUT_ROOT/$STAMP"
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || early_exit "cannot create $LOG_DIR"

echo "commit  : $(git rev-parse HEAD)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD)"
echo "budget  : ${TOTAL_S}s total (checks, diagnostic and finalization INCLUDED), reserve ${RESERVE_S}s"
echo "source  : $SOURCE_DIR (READ ONLY)"
echo "out     : $RUN_DIR"
echo "logs    : $LOG_DIR"
[ -d "$SOURCE_DIR" ] || early_exit "designated source directory missing"
( cd "$SOURCE_DIR" && sha256sum manifest.json SHA256SUMS \
    src500_momentum_delta.msgpack src501_momentum_delta.msgpack \
    src502_momentum_delta.msgpack src503_momentum_delta.msgpack \
    ) > "$LOG_DIR/source_sha256_before.txt.partial" 2>&1
rc=$?
cat "$LOG_DIR/source_sha256_before.txt.partial"
if [ "$rc" -ne 0 ]; then
  early_exit "designated source file missing; no substitute"
fi
mv "$LOG_DIR/source_sha256_before.txt.partial" "$LOG_DIR/source_sha256_before.txt"

STAGE_TERM_AT=$(( DEADLINE - RESERVE_S ))

stage_end() {
  if [ "$PM_OUTCOME" != completed ] || [ "$PM_RC" != "0" ]; then
    pm_finish "$RUN_DIR" "$1" "$PM_OUTCOME" "$PM_RC"
  fi
}

# ---- 1. GPU first
pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -c "import jax,sys; print('backend',jax.default_backend()); \
print('devices',jax.devices()); print('jax',jax.__version__); \
sys.exit(0 if jax.default_backend()=='gpu' else 4)" \
  > "$LOG_DIR/backend.txt" 2>&1
cat "$LOG_DIR/backend.txt"
echo "backend: outcome=$PM_OUTCOME rc=$PM_RC"
stage_end backend

# ---- 2. the law's focused checks (unchanged), inside the same cap
pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m pytest tests/test_prospective_tss_containment.py \
  -q -rP --durations=5 > "$LOG_DIR/law_checks.log" 2>&1
echo "--- law checks ---"; tail -12 "$LOG_DIR/law_checks.log"
echo "law checks: outcome=$PM_OUTCOME rc=$PM_RC (elapsed $(( $(date +%s) - START ))s)"
stage_end law_checks

pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  env JAX_ENABLE_X64=0 "$PY" -u tests/prospective_tss_containment_float32_probe.py \
  > "$LOG_DIR/law_float32_probe.log" 2>&1
echo "--- law float32 probe ---"; tail -6 "$LOG_DIR/law_float32_probe.log"
echo "law float32 probe: outcome=$PM_OUTCOME rc=$PM_RC (elapsed $(( $(date +%s) - START ))s)"
stage_end law_float32_probe

# ---- 3. this study's focused checks
pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m pytest tests/test_prospective_temporal_response.py \
  -q -rP --durations=10 > "$LOG_DIR/checks.log" 2>&1
echo "--- temporal-response checks ---"; tail -30 "$LOG_DIR/checks.log"
echo "--- measured magnitudes ---"
grep -nE "^  [a-zA-Z]" "$LOG_DIR/checks.log" | head -40 || true
echo "checks: outcome=$PM_OUTCOME rc=$PM_RC (elapsed $(( $(date +%s) - START ))s)"
stage_end checks

# ---- 4. the study: task, start points, diagnostic, preflight, batch
echo "remaining until the study's TERM: $(( STAGE_TERM_AT - $(date +%s) ))s"
pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m experiments.prospective_momentum.temporal_response \
    --out_root "$OUT_ROOT" --source_run "$SOURCE_RUN" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    > "$LOG_DIR/study.log" 2>&1
echo "--- study tail ---"; tail -30 "$LOG_DIR/study.log"
grep -E "PROSPECTIVE_MOMENTUM_STATUS|SOURCE_UNCHANGED|PREFLIGHT_|\[source\]|\[diagnostic\]|\[preflight\]|\[selection\]|\[matched\]|\[screen\]|\[!\]" \
  "$LOG_DIR/study.log" || true
echo "study: outcome=$PM_OUTCOME rc=$PM_RC"
pm_finish "$RUN_DIR" study "$PM_OUTCOME" "$PM_RC"
