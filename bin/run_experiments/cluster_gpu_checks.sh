#!/usr/bin/env bash
# GPU correctness gate. Runs the numerical suites on the CLUSTER GPU.
#
# Never pipes a test process into `tail` and reports tail's status: the log is
# saved, the real exit code is captured, and only then is the tail displayed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cluster_check_env || { echo "ENVIRONMENT CHECK FAILED"; exit 2; }
cd "$PROSPECTIVE_REPO" || exit 2

LOG_DIR="$PROSPECTIVE_RUNS/gpu_checks/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
echo "logs: $LOG_DIR"

"$PY" - <<'PYEOF' | tee "$LOG_DIR/backend.txt"
import jax
print("backend", jax.default_backend())
print("devices", jax.devices())
print("jax", jax.__version__)
raise SystemExit(0 if jax.default_backend() == "gpu" else 3)
PYEOF
rc_backend=${PIPESTATUS[0]}
if [ "$rc_backend" -ne 0 ]; then
  echo "REFUSING: backend is not gpu (exit $rc_backend). This gate is GPU-only."
  exit 2
fi

# -u and -v so the log shows WHICH test is running, line by line, instead of
# buffering dots for minutes. A silent log made a slow run indistinguishable
# from a hung one. --durations exposes what is actually costing the time.
rc=0
"$PY" -u -m pytest tests/ -v --durations=15 \
    > "$LOG_DIR/pytest.log" 2>&1 || rc=$?
echo "--- pytest tail ---"; tail -25 "$LOG_DIR/pytest.log"
echo "--- failures, if any ---"
grep -E "^(FAILED|ERROR)" "$LOG_DIR/pytest.log" | head -20 || true
echo "pytest exit: $rc"

rc_p1=0
"$PY" -u tests/cluster_float32_probe.py > "$LOG_DIR/cluster_float32.log" 2>&1 || rc_p1=$?
echo "--- cluster float32 probe ---"; tail -15 "$LOG_DIR/cluster_float32.log"
echo "probe exit: $rc_p1"

rc_p2=0
"$PY" -u tests/gp_float32_probe.py > "$LOG_DIR/gp_float32.log" 2>&1 || rc_p2=$?
echo "--- gp float32 probe ---"; tail -10 "$LOG_DIR/gp_float32.log"
echo "probe exit: $rc_p2"

rc_p3=0
"$PY" -u tests/so_float32_probe.py > "$LOG_DIR/so_float32.log" 2>&1 || rc_p3=$?
echo "--- second-order probe ---"; tail -6 "$LOG_DIR/so_float32.log"
echo "probe exit: $rc_p3"

total=$(( rc + rc_p1 + rc_p2 + rc_p3 ))
echo "GPU_CHECKS_EXIT=$total   logs=$LOG_DIR"
exit $total
