#!/usr/bin/env bash
# Constrained learned response: preflight, then the bounded batch ONLY if
# the checks pass and the MEASURED projection fits the remaining budget.
#
# ONE hard 20-minute budget covers focused checks, compilation, preflight and
# training. GPU only, no CPU fallback. Test split is never opened.
#
# Status contract, printed on EVERY terminal path:
#   CONSTRAINED_STATUS=PASS|INCOMPLETE|FAILED   CONSTRAINED_EXIT=0|3|4
#     PASS       0  checks passed and the batch ran to completion
#     INCOMPLETE 3  timeout, exhausted budget, or the projection did not fit
#     FAILED     4  a required check failed, or an environment/runtime failure
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"

TOTAL_S=${TOTAL_S:-1200}
RESERVE_S=${RESERVE_S:-30}
START=$(date +%s)
DEADLINE=$(( START + TOTAL_S ))
STAMP=$(date +%Y%m%d-%H%M%S)
ARMS=${ARMS:-gp_learned_response,prospective_recurrence}

LOG_DIR=""
finish() {
  echo "CONSTRAINED_STATUS=$1"; echo "CONSTRAINED_EXIT=$2"
  [ -n "${3:-}" ] && echo "reason: $3"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  [ -n "$LOG_DIR" ] && echo "logs=$LOG_DIR"
  exit "$2"
}
remaining() { echo $(( DEADLINE - $(date +%s) )); }

cluster_check_env || finish FAILED 4 "environment check failed"
cd "$PROSPECTIVE_REPO" || finish FAILED 4 "cannot enter $PROSPECTIVE_REPO"
OUT_ROOT=${OUT_ROOT:-$PROSPECTIVE_RUNS/constrained}
LOG_DIR="$OUT_ROOT/logs/$STAMP"
mkdir -p "$LOG_DIR" || finish FAILED 4 "cannot create $LOG_DIR"

echo "commit   : $(git rev-parse HEAD)"
echo "branch   : $(git rev-parse --abbrev-ref HEAD)"
echo "arms     : $ARMS"
echo "budget   : ${TOTAL_S}s total (checks INCLUDED), reserve ${RESERVE_S}s"
echo "logs     : $LOG_DIR"

# ---- 1. GPU verified first, itself bounded by the deadline
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
if [ "$rc" -ne 0 ]; then
  finish FAILED 4 "backend is not gpu (raw exit $rc)"
fi

# ---- 2. focused numerical checks
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 30 ] || finish INCOMPLETE 3 "no time for the focused checks"
rc=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m pytest tests/test_constrained_response.py -q \
  > "$LOG_DIR/checks.log" 2>&1 || rc=$?
echo "--- focused checks ---"; tail -8 "$LOG_DIR/checks.log"
echo "checks raw exit: $rc  (elapsed $(( $(date +%s) - START ))s)"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "focused checks hit the deadline"
fi
[ "$rc" -eq 0 ] || finish FAILED 4 "focused numerical checks did not pass"

# ---- 3. preflight: two updates per new arm, measured cost and projection
LEFT=$(( $(remaining) - RESERVE_S ))
[ "$LEFT" -gt 30 ] || finish INCOMPLETE 3 "no time for the preflight"
rc=0
timeout --kill-after=10 --signal=TERM "${LEFT}s" \
  "$PY" -u -m experiments.gp.arm_preflight \
    --data_cache "$PROSPECTIVE_DATA" --arms "$ARMS" \
    --out "$LOG_DIR/preflight.json" \
  > "$LOG_DIR/preflight.log" 2>&1 || rc=$?
echo "--- preflight ---"; cat "$LOG_DIR/preflight.log"
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  finish INCOMPLETE 3 "preflight hit the deadline"
fi
[ "$rc" -eq 0 ] || finish FAILED 4 "preflight failed (raw exit $rc)"

PROJ=$(grep -o 'PREFLIGHT_PROJECTED_TOTAL_S=[0-9.]*' "$LOG_DIR/preflight.log" \
       | tail -1 | cut -d= -f2)
[ -n "$PROJ" ] || finish FAILED 4 "no projection produced"
LEFT=$(( $(remaining) - RESERVE_S ))
echo "projection ${PROJ}s vs remaining ${LEFT}s"

# ---- 4. the gate: train ONLY if the projection fits
FITS=$("$PY" -c "import sys; print(1 if float('$PROJ') <= float('$LEFT') else 0)")
if [ "$FITS" -ne 1 ]; then
  echo "The measured projection does NOT fit the remaining budget."
  echo "No arm is shortened and the budget is not widened. Correctness work is"
  echo "complete; training was not started."
  finish INCOMPLETE 3 "projection ${PROJ}s exceeds remaining ${LEFT}s"
fi

# ---- 5. bounded training batch, fixed arm order
OUT="$OUT_ROOT/$STAMP"; mkdir -p "$OUT"
rc_total=0
IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
  LEFT=$(( $(remaining) - RESERVE_S ))
  if [ "$LEFT" -le 20 ]; then
    echo "BUDGET REACHED before $arm; NOT STARTED (recorded, not shortened)."
    echo "{\"arm\":\"$arm\",\"status\":\"not_started_budget\"}" >> "$OUT/manifest.jsonl"
    rc_total=3
    continue
  fi
  echo "=== $arm  (remaining ${LEFT}s) ==="
  rc=0
  timeout --kill-after=10 --signal=TERM "${LEFT}s" \
    "$PY" -u -m experiments.gp.rawat_benchmark --mode train --arm "$arm" \
      --data_cache "$PROSPECTIVE_DATA" --run_dir "$OUT/$arm" \
      --epochs 10 --lr 1e-3 --seed 100 --matmul_precision highest \
      >> "$OUT/$arm.log" 2>&1 || rc=$?
  tail -3 "$OUT/$arm.log"
  echo "{\"arm\":\"$arm\",\"exit\":$rc,\"run_dir\":\"$OUT/$arm\"}" >> "$OUT/manifest.jsonl"
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then rc_total=3; else
    [ "$rc" -eq 0 ] || rc_total=4
  fi
done
echo "output=$OUT  manifest=$OUT/manifest.jsonl"
case "$rc_total" in
  0) finish PASS 0 "" ;;
  3) finish INCOMPLETE 3 "one or more arms did not complete within the budget" ;;
  *) finish FAILED 4 "an arm failed" ;;
esac
