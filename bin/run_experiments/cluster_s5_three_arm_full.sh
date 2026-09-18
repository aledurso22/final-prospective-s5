#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"

OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-three-arm-full-training}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="$OUT_ROOT/$STAMP"
mkdir -p "$OUT"

echo "scientific arms: Native matched S5 | Zucchet prospective S5 recurrence | Generalized prospective S5 recurrence (M,gamma,T)"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse HEAD)"
echo "gpu: $(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -c 'import jax; print(jax.devices())')"
echo "data cache: ${PROSPECTIVE_DATA}"
echo "output: $OUT"

exec "$PY" -u -m experiments.s5_three_arm_full.runner \
  --data-cache "$PROSPECTIVE_DATA" \
  --out "$OUT" \
  --protocol "$PROSPECTIVE_REPO/docs/S5_THREE_ARM_FULL_TRAINING_PROTOCOL.md" \
  --commit "$(git rev-parse HEAD)" \
  2>&1 | tee "$OUT/console.log"
