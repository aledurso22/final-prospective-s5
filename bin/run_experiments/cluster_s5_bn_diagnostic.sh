#!/usr/bin/env bash
# FROZEN-WEIGHT BATCHNORM DIAGNOSTIC on saved checkpoints of ONE task. ONE GPU.
#
# Read-only with respect to the run: experiments/s5_predictor_corrector/
# bn_collapse_diagnostic.py hashes every checkpoint before and after, never
# opens the run directory for writing, and refuses an output inside it.
# Results go to a new directory per job:
#
#   $PROSPECTIVE_RUNS/bn-diagnostic/<run stamp>_<arm>_<seed>/job-<id>/
#       result/bn_diagnostic.json      the report
#       result/progress.jsonl          one line per finished measurement
#       result/recalibrated_stats/     the recalibrated BatchNorm statistics
#       diagnostic.log                 stdout/stderr
#       jax-cache/                     this job's own compilation cache
#
# Run it from its OWN checkout (never the worktree of an active run).
#   TASK=<run root>/<arm>/<seed>   default: A302 of the 2x2 run 20261117-195818
#   EPOCHS="1 2 3"  REPRODUCE_EPOCH=2
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$HERE/cluster_env.sh"
cd "$REPO_ROOT"

TASK="${TASK:-/Users/durso/s5-runs/s5-timescale-2x2/20261117-195818/generalized_prospective_s5/302}"
EPOCHS="${EPOCHS:-1 2 3}"
REPRODUCE_EPOCH="${REPRODUCE_EPOCH:-2}"
DATA_CACHE="${S5_THREE_ARM_DATA:-/Users/durso/s5-runs/sc10_official_cache}"

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the full sha of this checkout}"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "FAIL: HEAD $(git rev-parse HEAD) is not $EXPECTED_COMMIT" >&2; exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  git status --short >&2; echo "FAIL: modified worktree: $REPO_ROOT" >&2; exit 1
fi
[[ -x "$PY" ]] || { echo "FAIL: interpreter is not executable: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in
  0|false|False|FALSE|"") ;;
  *) echo "FAIL: JAX_ENABLE_X64=$JAX_ENABLE_X64; training ran in float32" >&2; exit 1 ;;
esac
export JAX_ENABLE_X64=0
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
  echo "FAIL: SLURM_JOB_ID='${SLURM_JOB_ID:-}' is not a numeric job id" >&2; exit 1; }
[[ -d "$TASK" ]] || { echo "FAIL: no task directory $TASK" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ "${#TOKENS[@]}" -ne 1 || -z "${TOKENS[0]}" ]]; then
  echo "FAIL: expected exactly one visible GPU, got '${CUDA_VISIBLE_DEVICES:-}'" >&2; exit 1
fi

ARM="$(basename "$(dirname "$TASK")")"
SEED="$(basename "$TASK")"
STAMP="$(basename "$(dirname "$(dirname "$TASK")")")"
JOB_ROOT="$PROSPECTIVE_RUNS/bn-diagnostic/${STAMP}_${ARM}_${SEED}/job-$SLURM_JOB_ID"
[[ ! -e "$JOB_ROOT" ]] || { echo "FAIL: $JOB_ROOT exists" >&2; exit 1; }
mkdir -p "$JOB_ROOT/jax-cache"
export JAX_COMPILATION_CACHE_DIR="$JOB_ROOT/jax-cache"

echo "bn diagnostic: host $(hostname) job $SLURM_JOB_ID gpu ${CUDA_VISIBLE_DEVICES}"
echo "  commit $EXPECTED_COMMIT   task $TASK   epochs $EPOCHS (reproduce $REPRODUCE_EPOCH)"
echo "  output $JOB_ROOT"
set +e
# shellcheck disable=SC2086
"$PY" -u -m experiments.s5_predictor_corrector.bn_collapse_diagnostic \
  --task "$TASK" --epochs $EPOCHS --reproduce-epoch "$REPRODUCE_EPOCH" \
  --data-cache "$DATA_CACHE" --out "$JOB_ROOT/result" \
  2>&1 | tee "$JOB_ROOT/diagnostic.log"
status="${PIPESTATUS[0]}"
set -e
echo "bn diagnostic exit $status; report: $JOB_ROOT/result/bn_diagnostic.json"
exit "$status"
