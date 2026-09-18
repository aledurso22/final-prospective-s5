#!/usr/bin/env bash
# MDN NESTEROV/QHM LADDER: native, the frozen two-tap operator
# (ordinary_prospective), literal TSS, literal Nesterov, QHM and generalized
# prospectivity on the completed temporal-response protocol
# (same task, sources, streams, learning rates, checkpoints, selection and
# metrics). NOT AUTHORIZED TO RUN until docs/MDN_NESTEROV_QHM_AUDIT.md s8 is
# cleared.
#
# ONE hard 600-second budget covers GPU startup, the FAIL-CLOSED gate (exact
# identities and float64 production checks: any failure stops the dispatch
# before the experiment), the completed study's own checks, start points,
# measured preflight, all training (<= 6,000 updates),
# held-out evaluation, the paired analysis, kill grace, source verification,
# the digest and the terminal verdict. Sources of run 20260917-011842 are
# REUSED READ-ONLY. No retries, no trimming.
#
#   PROSPECTIVE_MOMENTUM_STATUS=PASS|INCOMPLETE|FAILED  ..._EXIT=0|3|4
# PASS is operational only; the recommendation is a separate verdict.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
source "$HERE/prospective_momentum_terminal.sh"
export PM_DIGEST_MODULE=experiments.prospective_momentum.nesterov_ladder_summary

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
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/prospective-nesterov-ladder}
SOURCE_RUN=${SOURCE_RUN:-$PROSPECTIVE_RUNS/prospective-momentum-replication/20260917-011842}
SOURCE_DIR="$SOURCE_RUN/sources"
export PM_SOURCE_RUN="$SOURCE_RUN"
RUN_DIR="$OUT_ROOT/$STAMP"
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || early_exit "cannot create $LOG_DIR"

echo "commit  : $(git rev-parse HEAD)"
echo "branch  : $(git rev-parse --abbrev-ref HEAD)"
echo "budget  : ${TOTAL_S}s total (checks and finalization INCLUDED), reserve ${RESERVE_S}s"
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

# ---- 2. FAIL-CLOSED gate: exact identities and the float64 production
#         checks (new rules, the unchanged two-tap arm, the analysis). A
#         non-zero exit ends the dispatch here: no experiment runs.
pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m pytest tests/test_mdn_nesterov_qhm_identities.py \
  tests/test_prospective_realizations.py \
  tests/test_mdn_nesterov_qhm_production.py \
  -q -rP --durations=10 > "$LOG_DIR/production_checks.log" 2>&1
echo "--- exact and float64 production checks ---"
tail -30 "$LOG_DIR/production_checks.log"
echo "production checks: outcome=$PM_OUTCOME rc=$PM_RC (elapsed $(( $(date +%s) - START ))s)"
stage_end production_checks

# ---- 3. the completed temporal-response study's own checks (unchanged)
pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m pytest tests/test_prospective_temporal_response.py \
  -q -rP --durations=5 > "$LOG_DIR/checks.log" 2>&1
echo "--- completed-study checks ---"; tail -12 "$LOG_DIR/checks.log"
echo "checks: outcome=$PM_OUTCOME rc=$PM_RC (elapsed $(( $(date +%s) - START ))s)"
stage_end checks

# ---- 4. the study
echo "remaining until the study's TERM: $(( STAGE_TERM_AT - $(date +%s) ))s"
pm_bounded "$STAGE_TERM_AT" "$PM_GRACE_S" \
  "$PY" -u -m experiments.prospective_momentum.nesterov_ladder \
    --out_root "$OUT_ROOT" --source_run "$SOURCE_RUN" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    > "$LOG_DIR/study.log" 2>&1
echo "--- study tail ---"; tail -30 "$LOG_DIR/study.log"
grep -E "PROSPECTIVE_MOMENTUM_STATUS|SOURCE_UNCHANGED|PREFLIGHT_|\[source\]|\[preflight\]|\[selection\]|\[nesterov\]|\[paired\]|\[immediate claim\]|\[recommendation\]|\[!\]" \
  "$LOG_DIR/study.log" || true
echo "study: outcome=$PM_OUTCOME rc=$PM_RC"
pm_finish "$RUN_DIR" study "$PM_OUTCOME" "$PM_RC"
