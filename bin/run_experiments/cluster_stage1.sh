#!/usr/bin/env bash
# Stage 1 of the bounded protocol: one initialization plus two updates per
# executable arm, then a throughput and memory measurement. GPU only.
# Establishes EXECUTION, not accuracy.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cluster_check_env || { echo "ENVIRONMENT CHECK FAILED"; exit 2; }
cd "$PROSPECTIVE_REPO" || exit 2

OUT="$PROSPECTIVE_RUNS/stage1/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "stage1 output: $OUT"
rc_total=0
for arm in native_s5 alpha_p_s5 gain_clip_s5 gp_fixed_m0 gp_fixed_mass; do
  echo "=== $arm ==="
  rc=0
  "$PY" -m experiments.gp.rawat_benchmark --mode integration --arm "$arm" \
      --data_cache "$PROSPECTIVE_DATA" --outdir "$OUT" \
      --matmul_precision highest > "$OUT/$arm.log" 2>&1 || rc=$?
  tail -4 "$OUT/$arm.log"
  echo "exit: $rc"
  rc_total=$(( rc_total + rc ))
done
echo "STAGE1_EXIT=$rc_total  output=$OUT"
exit $rc_total
