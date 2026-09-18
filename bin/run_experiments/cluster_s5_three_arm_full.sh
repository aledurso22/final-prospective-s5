#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"

DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-three-arm-full-training}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="$OUT_ROOT/$STAMP"
mkdir -p "$OUT"

echo "scientific arms: Native matched S5 | Zucchet prospective S5 recurrence | Generalized prospective S5 recurrence (M,gamma,T)"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse HEAD)"
echo "data cache: ${DATA_CACHE} (official validation/testing lists)"
echo "output: $OUT"

ARRAY_JOB="$(sbatch --parsable \
  --array=0-8%4 \
  --export=ALL,S5_THREE_ARM_DATA="$DATA_CACHE",S5_THREE_ARM_RUN_ROOT="$OUT" \
  "$PROSPECTIVE_REPO/bin/slurm/s5_three_arm_full_array.sbatch")"
FINALIZER_JOB="$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB}" \
  --export=ALL,S5_THREE_ARM_DATA="$DATA_CACHE",S5_THREE_ARM_RUN_ROOT="$OUT" \
  "$PROSPECTIVE_REPO/bin/slurm/s5_three_arm_full_finalize.sbatch")"

printf 'array job: %s\nfinalizer job: %s\n' "$ARRAY_JOB" "$FINALIZER_JOB"
