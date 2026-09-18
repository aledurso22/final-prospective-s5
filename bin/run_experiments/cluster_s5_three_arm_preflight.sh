#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the final authoritative commit}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
OUT="${S5_THREE_ARM_PREFLIGHT_OUT:-$PROSPECTIVE_RUNS/s5-three-arm-preflight/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
exec sbatch --parsable \
  --export=ALL,EXPECTED_COMMIT="$EXPECTED_COMMIT",S5_THREE_ARM_PREFLIGHT_OUT="$OUT" \
  "$PROSPECTIVE_REPO/bin/slurm/s5_three_arm_preflight.sbatch"
