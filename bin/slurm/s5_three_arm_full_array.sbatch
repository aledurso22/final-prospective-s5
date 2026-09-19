#!/usr/bin/env bash
#SBATCH --job-name=s5-three-arm
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --array=0-8%1

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
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short >&2
  echo "Refusing dirty worktree: $PROSPECTIVE_REPO" >&2
  exit 1
fi

# Dispatch in fixed scientific order; no prior timing result is reused.
DISPATCH_ARMS=(native_matched_s5 zucchet_prospective_s5 generalized_prospective_s5)
SEEDS=(301 302 303)
ARM="${DISPATCH_ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
# HARD SERIALIZATION GUARD, enforced here and not merely requested at
# submission. The diagnosis (jobs 66683 / 66684: separate non-array jobs,
# separate cgroups, two DIFFERENT physical GPUs, both ending at 00:01:13, one
# SIGKILLed with empty stderr; and job 66688, the same generalized arm passing
# alone) implicates concurrent execution on this node. A task therefore
# refuses to start while any sibling element of this array is running, so the
# throttle cannot be silently widened by a resubmission or a scheduler
# setting. This costs one squeue call per task and fails closed.
if ! RUNNING_SIBLINGS="$(squeue -h -j "${SLURM_ARRAY_JOB_ID:?}" -t RUNNING -o '%i' | wc -l | tr -d '[:space:]')"; then
  echo "Refusing to run: cannot verify serialization (squeue failed)" >&2
  exit 1
fi
if [[ "$RUNNING_SIBLINGS" -ne 1 ]]; then
  echo "Refusing to run: $RUNNING_SIBLINGS elements of array ${SLURM_ARRAY_JOB_ID} are RUNNING;" >&2
  echo "this production run requires exactly one (--array=0-8%1)." >&2
  exit 1
fi

TASK_ROOT="$S5_THREE_ARM_RUN_ROOT/$ARM/$SEED"
mkdir -p "$TASK_ROOT/tmp" "$TASK_ROOT/jax-cache" "$TASK_ROOT/logs"
export TMPDIR="$TASK_ROOT/tmp"
export JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/jax-cache"

printf 'arm=%s\nseed=%s\ncommit=%s\n' "$ARM" "$SEED" "$(git rev-parse HEAD)" \
  > "$TASK_ROOT/task_metadata.txt"

# PRE-JAX execution telemetry: job identity and the PHYSICAL GPU (UUID and PCI
# bus id), written before any JAX import so it survives an external kill.
"$PY" -u -m experiments.s5_three_arm_full.gpu_telemetry \
  --out "$TASK_ROOT/gpu_telemetry.json" --label pre_jax_array_task \
  | tee "$TASK_ROOT/logs/gpu_telemetry.log"

exec "$PY" -u -m experiments.s5_three_arm_full.runner \
  --data-cache "$S5_THREE_ARM_DATA" \
  --out "$TASK_ROOT" \
  --arm "$ARM" \
  --seed "$SEED" \
  ${S5_THREE_ARM_SMOKE:+--smoke} \
  2>&1 | tee "$TASK_ROOT/logs/train.log"
