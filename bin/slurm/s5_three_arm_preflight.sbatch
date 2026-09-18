#!/usr/bin/env bash
#SBATCH --job-name=s5-three-preflight
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../run_experiments/cluster_env.sh"
cd "$PROSPECTIVE_REPO"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the final authoritative commit}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
: "${S5_THREE_ARM_PREFLIGHT_OUT:?set preflight output directory}"
mkdir -p "$S5_THREE_ARM_PREFLIGHT_OUT/tmp" "$S5_THREE_ARM_PREFLIGHT_OUT/jax-cache"
export TMPDIR="$S5_THREE_ARM_PREFLIGHT_OUT/tmp"
export JAX_COMPILATION_CACHE_DIR="$S5_THREE_ARM_PREFLIGHT_OUT/jax-cache"
preflight_args=(--out "$S5_THREE_ARM_PREFLIGHT_OUT")
if [[ -n "${PREFLIGHT_ARM:-}" ]]; then
  preflight_args+=(--arm "$PREFLIGHT_ARM")
fi
exec "$PY" -u -m experiments.s5_three_arm_full.preflight "${preflight_args[@]}"
