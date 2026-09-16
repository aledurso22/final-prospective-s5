#!/usr/bin/env bash
# Prospective correction of a trained Momentum DeltaNet: checks, source
# restoration, preflight, six-arm continuation. NOT AUTHORIZED TO RUN until
# docs/PROSPECTIVE_MOMENTUM_PROTOCOL.md is cleared.
#
# ONE hard 600-second budget covers GPU startup, focused checks, checkpoint
# restoration and reproduction, preflight, 12 development runs, 18 final runs,
# evaluation, kill grace, source verification, the digest and the terminal
# verdict. Every step is bounded by the same absolute DEADLINE
# (prospective_momentum_terminal.sh). Preflight refuses the batch
# (INCOMPLETE) if it cannot fit and FAILS it on invalid numerical state;
# nothing is trimmed, retried or extended.
#
#   PROSPECTIVE_MOMENTUM_STATUS=PASS|INCOMPLETE|FAILED  ..._EXIT=0|3|4
# The terminal verdict combines the stage outcome, the source checksum
# re-verification and the declared digest; it is also merged into
# status.json under `terminal`. PASS is operational only; the screens are
# separate performance verdicts.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
source "$HERE/prospective_momentum_terminal.sh"

TOTAL_S=${TOTAL_S:-600}
RESERVE_S=${RESERVE_S:-30}
START=$(date +%s)
DEADLINE=$(( START + TOTAL_S ))
STAMP=$(date +%Y%m%d-%H%M%S)

early_exit() {        # before a log directory exists: nothing to verify
  echo "PROSPECTIVE_MOMENTUM_STATUS=FAILED"; echo "PROSPECTIVE_MOMENTUM_EXIT=4"
  echo "reason: $1"; exit 4
}

cluster_check_env || early_exit "environment check failed"
cd "$PROSPECTIVE_REPO" || early_exit "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/prospective-momentum}
SOURCE_DIR=${SOURCE_DIR:-$PROSPECTIVE_RUNS/meta-delta/20260916-222310}
export PM_SOURCE_DIR="$SOURCE_DIR"
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
( cd "$SOURCE_DIR" && sha256sum status.json selection.json \
    params/dev_momentum_delta_B_seed300.msgpack \
    params/dev_gated_delta_B_seed300.msgpack \
    params/dev_gp_two_sided_B_seed300.msgpack \
    params/dev_tss_eq17_B_seed300.msgpack ) > "$LOG_DIR/source_sha256_before.txt.partial" 2>&1
rc=$?
cat "$LOG_DIR/source_sha256_before.txt.partial"
if [ "$rc" -ne 0 ]; then
  early_exit "designated source file missing; no substitute"
fi
mv "$LOG_DIR/source_sha256_before.txt.partial" "$LOG_DIR/source_sha256_before.txt"

# stage TERM time: DEADLINE - RESERVE; KILL after PM_GRACE_S
stage_limit() { echo $(( $(pm_left) - RESERVE_S )); }

# ---- 1. GPU first
rc=0
pm_bounded "$(stage_limit)" "$PM_GRACE_S" \
  "$PY" -u -c "import jax,sys; print('backend',jax.default_backend()); \
print('devices',jax.devices()); print('jax',jax.__version__); \
sys.exit(0 if jax.default_backend()=='gpu' else 4)" \
  > "$LOG_DIR/backend.txt" 2>&1 || rc=$?
cat "$LOG_DIR/backend.txt"
[ "$rc" -eq 0 ] || pm_finish "$RUN_DIR" backend "$rc"

# ---- 2. focused checks, inside the same cap
rc=0
pm_bounded "$(stage_limit)" "$PM_GRACE_S" \
  "$PY" -u -m pytest tests/test_prospective_momentum.py -q -rP --durations=10 \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -30 "$LOG_DIR/checks.log"
echo "--- measured magnitudes (passing and failing) ---"
grep -nE "^  [a-zA-Z]" "$LOG_DIR/checks.log" | head -80 || true
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -ne 0 ]; then
  # pytest failure codes are not the study's codes: map to FAILED unless
  # the watchdog or lack of time ended it
  case "$rc" in 124|125|137) ;; *) rc=4 ;; esac
  pm_finish "$RUN_DIR" checks "$rc"
fi

# ---- 3. the study: restore/reproduce, preflight, batch only if it fits
echo "remaining for the study: $(stage_limit)s"
rc=0
pm_bounded "$(stage_limit)" "$PM_GRACE_S" \
  "$PY" -u -m experiments.prospective_momentum.study \
    --out_root "$OUT_ROOT" --source_dir "$SOURCE_DIR" --run_id "$STAMP" \
    --deadline "$DEADLINE" --reserve_s "$RESERVE_S" \
    > "$LOG_DIR/study.log" 2>&1 || rc=$?
echo "--- study tail ---"; tail -40 "$LOG_DIR/study.log"
grep -E "PROSPECTIVE_MOMENTUM_STATUS|SOURCE_UNCHANGED|PREFLIGHT_|\[source\]|\[preflight\]|\[screen\]|\[!\]" \
  "$LOG_DIR/study.log" || true
echo "study raw exit: $rc"
case "$rc" in 0|3|4|124|125|137) ;; *) rc=4 ;; esac
pm_finish "$RUN_DIR" study "$rc"
