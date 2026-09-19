#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"
REPO_ROOT="$(pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the final authoritative commit}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short >&2
  echo "Refusing dirty worktree: $PROSPECTIVE_REPO" >&2
  exit 1
fi

DATA_CACHE="${S5_THREE_ARM_DATA:-/Users/durso/s5-runs/sc10_official_cache}"
"$PY" - "$DATA_CACHE" <<'PYEOF'
import sys
from experiments.s5_three_arm_full.data import validate_official_raw_cache
validate_official_raw_cache(sys.argv[1], ("train", "val"))
print(f"validated official raw cache: {sys.argv[1]}")
PYEOF
OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-three-arm-full-training}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="$OUT_ROOT/$STAMP"
mkdir -p "$OUT"
OUTPUT_DIR="$OUT"
mkdir -p "$OUTPUT_DIR/slurm"

echo "scientific arms: Native S5 | Zucchet prospective dynamics — finite-difference realization | generalized prospective dynamics (M,gamma,T) — finite-difference realization"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse HEAD)"
echo "data cache: ${DATA_CACHE} (official validation/testing lists)"
echo "output: $OUT"
printf 'authoritative_commit=%s\nbranch=%s\ndata_cache=%s\n' \
  "$EXPECTED_COMMIT" "$(git rev-parse --abbrev-ref HEAD)" "$DATA_CACHE" \
  > "$OUT/run_metadata.txt"

ARRAY_JOB="$(sbatch --parsable \
  --chdir="$REPO_ROOT" \
  --array=0-8%4 \
  --output="$OUTPUT_DIR/slurm/array-%A_%a.out" \
  --error="$OUTPUT_DIR/slurm/array-%A_%a.err" \
  --export=ALL,PROSPECTIVE_REPO="$REPO_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",S5_THREE_ARM_DATA="$DATA_CACHE",S5_THREE_ARM_RUN_ROOT="$OUT" \
  "$REPO_ROOT/bin/slurm/s5_three_arm_full_array.sbatch")"
FINALIZER_JOB="$(sbatch --parsable \
  --dependency="afterany:${ARRAY_JOB}" \
  --chdir="$REPO_ROOT" \
  --output="$OUTPUT_DIR/slurm/finalizer-%j.out" \
  --error="$OUTPUT_DIR/slurm/finalizer-%j.err" \
  --export=ALL,PROSPECTIVE_REPO="$REPO_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",S5_THREE_ARM_DATA="$DATA_CACHE",S5_THREE_ARM_RUN_ROOT="$OUT" \
  "$REPO_ROOT/bin/slurm/s5_three_arm_full_finalize.sbatch")"

printf 'array job: %s\nfinalizer job: %s\n' "$ARRAY_JOB" "$FINALIZER_JOB"
