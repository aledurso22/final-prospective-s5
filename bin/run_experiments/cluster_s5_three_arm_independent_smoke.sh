#!/usr/bin/env bash
# CONCURRENT FULL-PATH SMOKE, two INDEPENDENT Slurm submissions.
#
# Diagnosis, not science. Job 66583 ran Native S5 and generalized prospective
# dynamics as two sibling tasks of ONE array; the generalized task received
# SIGKILL 0:9 with empty stderr at the exact instant the Native sibling
# completed, after a finite production update. This script runs the same two
# tasks as two SEPARATE jobs with DIFFERENT base job ids, so each has its own
# job cgroup and its own epilog. If both pass concurrently, the array topology
# is implicated; if they still interfere, the node is.
#
# Each job exercises the full path: production check, two real training
# updates, validation, checkpoint save, checkpoint reload, lifecycle
# telemetry, the PRE-JAX job identity, and the process-specific binding of the
# runner pid to a physical GPU UUID.
#
# Neither recurrence, the data, the protocol nor the artifacts change. This
# launches NO full training.
#
#   DRY_RUN=1 bash bin/run_experiments/cluster_s5_three_arm_independent_smoke.sh
# prints the exact submissions without contacting Slurm.
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
OUT_ROOT="${OUT_ROOT:-$PROSPECTIVE_RUNS/s5-three-arm-independent-smoke}"
STAMP="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="$OUT_ROOT/$STAMP"
DRY_RUN="${DRY_RUN:-0}"

#: the two concurrent tasks, in scientific reporting order
SMOKE_TASKS=("native_matched_s5:301" "generalized_prospective_s5:301")

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUT/slurm"
  "$PY" - "$DATA_CACHE" <<'PYEOF'
import sys
from experiments.s5_three_arm_full.data import validate_official_raw_cache
validate_official_raw_cache(sys.argv[1], ("train", "val"))
print(f"validated official raw cache: {sys.argv[1]}")
PYEOF
fi

echo "concurrent full-path smoke, INDEPENDENT submissions (no job array)"
echo "  1. Native S5"
echo "  2. generalized prospective dynamics (M,gamma,T) — finite-difference realization"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse HEAD)"
echo "data cache: $DATA_CACHE"
echo "output: $OUT"

JOB_IDS=()
for task in "${SMOKE_TASKS[@]}"; do
  ARM="${task%%:*}"
  SEED="${task##*:}"
  submit=(sbatch --parsable
    --job-name="s5-smoke-${ARM}-${SEED}"
    --chdir="$REPO_ROOT"
    --output="$OUT/slurm/${ARM}-${SEED}-%j.out"
    --error="$OUT/slurm/${ARM}-${SEED}-%j.err"
    --export="ALL,PROSPECTIVE_REPO=$REPO_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,S5_THREE_ARM_DATA=$DATA_CACHE,S5_THREE_ARM_RUN_ROOT=$OUT,S5_THREE_ARM_ARM=$ARM,S5_THREE_ARM_SEED=$SEED,S5_THREE_ARM_SMOKE=1"
    "$REPO_ROOT/bin/slurm/s5_three_arm_one_task.sbatch")
  if [[ "$DRY_RUN" == "1" ]]; then
    # one argument per line: the array is passed to sbatch verbatim, so no
    # shell quoting is applied to it in the real path either
    echo "DRY_RUN submission:"
    printf '    %s\n' "${submit[@]}"
    continue
  fi
  # --parsable prints "<jobid>" or "<jobid>;<cluster>"; normalize both
  job_id="$("${submit[@]}")"
  job_id="${job_id%%;*}"
  job_id="${job_id//[[:space:]]/}"
  if [[ ! "$job_id" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
    echo "FAIL: unparsable sbatch --parsable output: '$job_id'" >&2
    exit 1
  fi
  JOB_IDS+=("$job_id")
  printf 'submitted %-32s seed %s as INDEPENDENT job %s\n' "$ARM" "$SEED" "$job_id"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN: nothing submitted"
  exit 0
fi

printf 'independent job ids: %s\n' "${JOB_IDS[*]}"

# HARD REQUIREMENT, not a printout: the whole point of this smoke is that the
# two tasks are not members of one job. Different base job ids mean different
# job cgroups and different epilogs.
BASE_IDS=()
for id in "${JOB_IDS[@]}"; do BASE_IDS+=("${id%%_*}"); done
DISTINCT="$(printf '%s\n' "${BASE_IDS[@]}" | sort -u | wc -l | tr -d '[:space:]')"
printf 'base job ids: %s (distinct: %s)\n' "${BASE_IDS[*]}" "$DISTINCT"
if [[ "${#JOB_IDS[@]}" -ne 2 || "$DISTINCT" -ne 2 ]]; then
  echo "FAIL: the two submissions must have two distinct base job ids;" >&2
  echo "      got ${#JOB_IDS[@]} job(s), $DISTINCT distinct base id(s):" \
       "${JOB_IDS[*]}" >&2
  echo "      the execution topology under test was NOT created." >&2
  exit 1
fi
printf 'run_root=%s\njobs=%s\ncommit=%s\n' \
  "$OUT" "${JOB_IDS[*]}" "$EXPECTED_COMMIT" > "$OUT/smoke_metadata.txt"

cat <<EOF

Inspect the ARTIFACTS, not the Slurm state, when both jobs end:

  $PY -m experiments.s5_three_arm_full.topology_report \\
    $OUT/native_matched_s5/301 $OUT/generalized_prospective_s5/301
  tail -3 $OUT/native_matched_s5/301/lifecycle.jsonl
  tail -3 $OUT/generalized_prospective_s5/301/lifecycle.jsonl
  sacct -j ${JOB_IDS[0]},${JOB_IDS[1]} --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS,NodeList

A pass requires, for BOTH jobs: smoke_result.json with status SMOKE_PASS,
training_steps 2 and checkpoint_restored true. Slurm COMPLETED alone is not a
pass.

The shared-GPU question is answered ONLY from gpu_process_binding.json, which
maps the runner's own pid to a physical GPU UUID via
nvidia-smi --query-compute-apps. If either binding is not "resolved", the
report prints classification "inconclusive" and the report must say so: the
node inventory in gpu_telemetry.json lists every device on the node and can
neither establish nor exclude sharing.
EOF
