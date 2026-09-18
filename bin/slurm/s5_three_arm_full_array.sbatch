#!/usr/bin/env bash
#SBATCH --job-name=s5-three-arm
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --array=0-8%4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../run_experiments/cluster_env.sh"
cd "$PROSPECTIVE_REPO"

: "${S5_THREE_ARM_DATA:?set S5_THREE_ARM_DATA to the prepared official raw cache}"
: "${S5_THREE_ARM_RUN_ROOT:?set S5_THREE_ARM_RUN_ROOT to the run directory}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the final authoritative commit}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

# Dispatch longest-running arms first; reporting order remains runner.ARM_ORDER.
DISPATCH_ARMS=(generalized_prospective_s5 zucchet_prospective_s5 native_matched_s5)
SEEDS=(301 302 303)
ARM="${DISPATCH_ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
TASK_ROOT="$S5_THREE_ARM_RUN_ROOT/$ARM/$SEED"
mkdir -p "$TASK_ROOT/tmp" "$TASK_ROOT/jax-cache" "$TASK_ROOT/logs"
export TMPDIR="$TASK_ROOT/tmp"
export JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/jax-cache"

printf 'arm=%s\nseed=%s\ncommit=%s\n' "$ARM" "$SEED" "$(git rev-parse HEAD)" \
  > "$TASK_ROOT/task_metadata.txt"
exec "$PY" -u -m experiments.s5_three_arm_full.runner \
  --data-cache "$S5_THREE_ARM_DATA" \
  --out "$TASK_ROOT" \
  --arm "$ARM" \
  --seed "$SEED" \
  2>&1 | tee "$TASK_ROOT/logs/train.log"
