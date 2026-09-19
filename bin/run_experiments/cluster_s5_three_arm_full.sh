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
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != "1" ]]; then
  "$PY" - "$DATA_CACHE" <<'PYEOF'
import sys
from experiments.s5_three_arm_full.data import validate_official_raw_cache
validate_official_raw_cache(sys.argv[1], ("train", "val"))
print(f"validated official raw cache: {sys.argv[1]}")
PYEOF
fi
OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-three-arm-full-training}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="$OUT_ROOT/$STAMP"
OUTPUT_DIR="$OUT"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUTPUT_DIR/slurm"
fi

# SERIALIZED TOPOLOGY, forced by the diagnosis: jobs 66683 (Native S5,
# COMPLETED) and 66684 (generalized, SIGKILL 0:9 at the same 00:01:13) were
# separate non-array jobs, with separate job cgroups, on two DIFFERENT
# physical GPUs, and the generalized arm passed the identical path alone
# (job 66688). Concurrent execution on this node is the implicated condition,
# so the nine tasks run ONE AT A TIME. The throttle is part of the run's
# definition: it is a literal here, the sbatch default must agree with it, and
# the array task itself refuses to start if a sibling is already running.
ARRAY_SPEC="0-8%1"
ARRAY_SBATCH="$REPO_ROOT/bin/slurm/s5_three_arm_full_array.sbatch"
SBATCH_DEFAULT_ARRAY="$(sed -n 's/^#SBATCH --array=//p' "$ARRAY_SBATCH")"
if [[ "$SBATCH_DEFAULT_ARRAY" != "$ARRAY_SPEC" ]]; then
  echo "FAIL: the sbatch default --array=$SBATCH_DEFAULT_ARRAY disagrees with" >&2
  echo "      the required serialized topology $ARRAY_SPEC; refusing to submit." >&2
  exit 1
fi

# The repository is node-local (/Local/durso/final-prospective-s5), so this
# launcher is run from an INTERACTIVE allocation on pgi15-gpu3. That
# allocation is itself a concurrent job on the node: if an array element
# starts while it is alive and it then ends, that is precisely the
# concurrent-completion condition under which job 66684 was SIGKILLed. The
# array therefore waits for the submitting allocation to terminate. Only the
# array carries this dependency; the finalizer still waits on the array.
PARENT_DEPENDENCY=()
SUBMIT_ALLOCATION="${SLURM_JOB_ID:-}"
if [[ -n "$SUBMIT_ALLOCATION" ]]; then
  if [[ ! "$SUBMIT_ALLOCATION" =~ ^[0-9]+$ ]]; then
    echo "FAIL: SLURM_JOB_ID='$SUBMIT_ALLOCATION' is not a numeric job id;" >&2
    echo "      refusing to build an sbatch dependency from it." >&2
    exit 1
  fi
  PARENT_DEPENDENCY=(--dependency="afterany:$SUBMIT_ALLOCATION")
fi

echo "scientific arms: Native S5 | Zucchet prospective dynamics — finite-difference realization | generalized prospective dynamics (M,gamma,T) — finite-difference realization"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse HEAD)"
echo "data cache: ${DATA_CACHE} (official validation/testing lists)"
echo "output: $OUT"
echo "topology: ONE task at a time (--array=$ARRAY_SPEC)"
if [[ -n "$SUBMIT_ALLOCATION" ]]; then
  echo "submission allocation: $SUBMIT_ALLOCATION"
  echo "array waits for it: --dependency=afterany:$SUBMIT_ALLOCATION"
else
  echo "submission allocation: none (no SLURM_JOB_ID); no parent dependency"
fi
if [[ "$DRY_RUN" != "1" ]]; then
  printf 'authoritative_commit=%s\nbranch=%s\ndata_cache=%s\narray=%s\n' \
    "$EXPECTED_COMMIT" "$(git rev-parse --abbrev-ref HEAD)" "$DATA_CACHE" \
    "$ARRAY_SPEC" > "$OUT/run_metadata.txt"
  printf 'submit_allocation=%s\narray_dependency=%s\n' \
    "${SUBMIT_ALLOCATION:-none}" \
    "${SUBMIT_ALLOCATION:+afterany:$SUBMIT_ALLOCATION}" \
    >> "$OUT/run_metadata.txt"
fi

array_submit=(sbatch --parsable
  ${PARENT_DEPENDENCY[@]+"${PARENT_DEPENDENCY[@]}"}
  --chdir="$REPO_ROOT"
  --array="$ARRAY_SPEC"
  --output="$OUTPUT_DIR/slurm/array-%A_%a.out"
  --error="$OUTPUT_DIR/slurm/array-%A_%a.err"
  --export=ALL,PROSPECTIVE_REPO="$REPO_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",S5_THREE_ARM_DATA="$DATA_CACHE",S5_THREE_ARM_RUN_ROOT="$OUT"
  "$ARRAY_SBATCH")

if [[ "$DRY_RUN" == "1" ]]; then
  ARRAY_JOB="<ARRAY_JOB_ID>"
else
  ARRAY_JOB="$("${array_submit[@]}")"
  ARRAY_JOB="${ARRAY_JOB%%;*}"
fi

# the finalizer waits for EVERY element: afterany on the ARRAY JOB ID is
# satisfied only when all nine elements have reached a terminal state
finalizer_submit=(sbatch --parsable
  --dependency="afterany:${ARRAY_JOB}"
  --chdir="$REPO_ROOT"
  --output="$OUTPUT_DIR/slurm/finalizer-%j.out"
  --error="$OUTPUT_DIR/slurm/finalizer-%j.err"
  --export=ALL,PROSPECTIVE_REPO="$REPO_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",S5_THREE_ARM_DATA="$DATA_CACHE",S5_THREE_ARM_RUN_ROOT="$OUT"
  "$REPO_ROOT/bin/slurm/s5_three_arm_full_finalize.sbatch")

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN array submission (nine tasks, one at a time):"
  printf '    %s\n' "${array_submit[@]}"
  echo "DRY_RUN finalizer submission:"
  printf '    %s\n' "${finalizer_submit[@]}"
  echo "DRY_RUN: nothing submitted"
  exit 0
fi

FINALIZER_JOB="$("${finalizer_submit[@]}")"
FINALIZER_JOB="${FINALIZER_JOB%%;*}"

printf 'array job: %s (%s)\nfinalizer job: %s\n' \
  "$ARRAY_JOB" "$ARRAY_SPEC" "$FINALIZER_JOB"
