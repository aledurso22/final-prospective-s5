#!/usr/bin/env bash
#SBATCH --job-name=s5-three-final
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${PROSPECTIVE_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset}}"
test -d "$REPO_ROOT"
test -f "$REPO_ROOT/bin/run_experiments/cluster_env.sh"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$REPO_ROOT/bin/run_experiments/cluster_env.sh"
cd "$REPO_ROOT"

: "${S5_THREE_ARM_DATA:?set S5_THREE_ARM_DATA to the prepared official raw cache}"
: "${S5_THREE_ARM_RUN_ROOT:?set S5_THREE_ARM_RUN_ROOT to the run directory}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the final authoritative commit}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

mkdir -p "$S5_THREE_ARM_RUN_ROOT/final/tmp" "$S5_THREE_ARM_RUN_ROOT/final/jax-cache"
export TMPDIR="$S5_THREE_ARM_RUN_ROOT/final/tmp"
export JAX_COMPILATION_CACHE_DIR="$S5_THREE_ARM_RUN_ROOT/final/jax-cache"
exec "$PY" -u -m experiments.s5_three_arm_full.finalize \
  --task-root "$S5_THREE_ARM_RUN_ROOT" \
  --data-cache "$S5_THREE_ARM_DATA" \
  --out "$S5_THREE_ARM_RUN_ROOT/final" \
  2>&1 | tee "$S5_THREE_ARM_RUN_ROOT/final/finalize.log"
