#!/usr/bin/env bash
# ONE arm, ONE seed, ONE independent Slurm job (NO job array).
#
# Execution-topology diagnosis: sibling array tasks of job 66583 died together
# (a longer-running task received SIGKILL 0:9 at the instant its sibling
# finished, with empty stderr). Independent submissions have different base
# job ids, separate job cgroups and separate epilog runs, which is the control
# this script provides. The scientific recurrences, data, protocol and
# artifacts are untouched.
#
#SBATCH --job-name=s5-one-task
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G

set -euo pipefail

REPO_ROOT="${PROSPECTIVE_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset}}"
test -d "$REPO_ROOT"
test -f "$REPO_ROOT/bin/run_experiments/cluster_env.sh"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$REPO_ROOT/bin/run_experiments/cluster_env.sh"
cd "$REPO_ROOT"

: "${S5_THREE_ARM_DATA:?set S5_THREE_ARM_DATA to the prepared official raw cache}"
: "${S5_THREE_ARM_RUN_ROOT:?set S5_THREE_ARM_RUN_ROOT to the run directory}"
: "${S5_THREE_ARM_ARM:?set S5_THREE_ARM_ARM to the arm identifier}"
: "${S5_THREE_ARM_SEED:?set S5_THREE_ARM_SEED to the seed}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the final authoritative commit}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short >&2
  echo "Refusing dirty worktree: $PROSPECTIVE_REPO" >&2
  exit 1
fi

case "$S5_THREE_ARM_ARM" in
  native_matched_s5|zucchet_prospective_s5|generalized_prospective_s5) ;;
  *) echo "unknown arm: $S5_THREE_ARM_ARM" >&2; exit 1 ;;
esac
case "$S5_THREE_ARM_SEED" in
  301|302|303) ;;
  *) echo "unknown seed: $S5_THREE_ARM_SEED" >&2; exit 1 ;;
esac

ARM="$S5_THREE_ARM_ARM"
SEED="$S5_THREE_ARM_SEED"
TASK_ROOT="$S5_THREE_ARM_RUN_ROOT/$ARM/$SEED"
mkdir -p "$TASK_ROOT/tmp" "$TASK_ROOT/jax-cache" "$TASK_ROOT/logs"
export TMPDIR="$TASK_ROOT/tmp"
export JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/jax-cache"

printf 'arm=%s\nseed=%s\ncommit=%s\njob=%s\n' \
  "$ARM" "$SEED" "$(git rev-parse HEAD)" "${SLURM_JOB_ID:-<none>}" \
  > "$TASK_ROOT/task_metadata.txt"

# PRE-JAX execution telemetry: job identity and the PHYSICAL GPU (UUID and PCI
# bus id), written before any JAX import so it survives an external kill.
"$PY" -u -m experiments.s5_three_arm_full.gpu_telemetry \
  --out "$TASK_ROOT/gpu_telemetry.json" --label pre_jax_independent_job \
  | tee "$TASK_ROOT/logs/gpu_telemetry.log"

exec "$PY" -u -m experiments.s5_three_arm_full.runner \
  --data-cache "$S5_THREE_ARM_DATA" \
  --out "$TASK_ROOT" \
  --arm "$ARM" \
  --seed "$SEED" \
  ${S5_THREE_ARM_SMOKE:+--smoke} \
  2>&1 | tee "$TASK_ROOT/logs/train.log"
