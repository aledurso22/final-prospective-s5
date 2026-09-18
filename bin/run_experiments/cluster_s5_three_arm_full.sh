#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the final authoritative commit}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-three-arm-full-training}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="$OUT_ROOT/$STAMP"
mkdir -p "$OUT"

echo "scientific arms: Native S5 | Zucchet prospective dynamics — finite-difference realization | generalized prospective dynamics (M,gamma,T) — finite-difference realization"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse HEAD)"
echo "data cache: ${DATA_CACHE} (official validation/testing lists)"
echo "output: $OUT"
printf 'authoritative_commit=%s\nbranch=%s\ndata_cache=%s\n' \
  "$EXPECTED_COMMIT" "$(git rev-parse --abbrev-ref HEAD)" "$DATA_CACHE" \
  > "$OUT/run_metadata.txt"

ARRAY_JOB="$(sbatch --parsable \
  --array=0-8%4 \
  --export=ALL,EXPECTED_COMMIT="$EXPECTED_COMMIT",S5_THREE_ARM_DATA="$DATA_CACHE",S5_THREE_ARM_RUN_ROOT="$OUT" \
  "$PROSPECTIVE_REPO/bin/slurm/s5_three_arm_full_array.sbatch")"
FINALIZER_JOB="$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB}" \
  --export=ALL,EXPECTED_COMMIT="$EXPECTED_COMMIT",S5_THREE_ARM_DATA="$DATA_CACHE",S5_THREE_ARM_RUN_ROOT="$OUT" \
  "$PROSPECTIVE_REPO/bin/slurm/s5_three_arm_full_finalize.sbatch")"

printf 'array job: %s\nfinalizer job: %s\n' "$ARRAY_JOB" "$FINALIZER_JOB"
