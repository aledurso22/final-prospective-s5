#!/usr/bin/env bash
# Read-only checkpoint analysis on CPU; never trains or pushes code.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the published pinned full commit}"
SHA="$(git -C "$REPO" rev-parse HEAD)"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ && "$SHA" == "$EXPECTED_COMMIT" ]] || {
  echo "FAIL: wrong submission commit" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "FAIL: submission checkout must be clean, including untracked files" >&2; exit 1; }
[[ "$#" == 0 ]] || { echo "Usage: $0" >&2; exit 1; }
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: CPU-only checkpoint audit on pgi14-gpu1, partition pgi14; 4 CPUs, 24G, 2h; no GPUs"
  echo "DRY_RUN: commit $SHA; N/A/C x seeds301/302/303; selected checkpoints, reconstructed init, all15 saved epochs"
  echo "DRY_RUN: fresh per-job clone/output; no training, dataset access or test evaluation"
  echo "DRY_RUN: plotting checked before analysis; install missing plotting packages only in a new isolated environment"
  echo "DRY_RUN: numerical reports saved before separate rendering; shared training environment is not modified"
  exit 0
fi
# A caller's SBATCH GPU defaults must not turn this into a GPU allocation.
unset SBATCH_GRES SBATCH_GPUS SBATCH_GPUS_PER_NODE SBATCH_GPUS_PER_TASK SBATCH_GPUS_PER_SOCKET
sbatch --partition=pgi14 --nodelist=pgi14-gpu1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G \
  --time=02:00:00 --job-name=s5-memory-audit \
  --output='/Users/durso/s5-runs/s5-memory-audit-%j.log' \
  --export="ALL,S5_MEMORY_AUDIT_COMMIT=$SHA" <<'S5_MEMORY_AUDIT_JOB'
#!/bin/bash
set -euo pipefail
TASK_REPO="/Local/durso/s5-memory-audit-${SLURM_JOB_ID}"
[[ ! -e "$TASK_REPO" ]] || { echo "FAIL: isolated clone destination exists" >&2; exit 1; }
git clone --no-checkout \
  "$(git -C /Local/durso/final-prospective-s5 remote get-url origin)" "$TASK_REPO"
cd "$TASK_REPO"
git checkout --detach "$S5_MEMORY_AUDIT_COMMIT"
[[ "$(git rev-parse HEAD)" == "$S5_MEMORY_AUDIT_COMMIT" ]] || exit 1
[[ -z "$(git status --porcelain)" ]] || { echo "FAIL: analysis checkout is dirty" >&2; exit 1; }
export PROSPECTIVE_VENV=/Local/durso/final-prospective-s5/.venv
export PYTHONPATH="$PWD"
unset JAX_PLATFORM_NAME XLA_FLAGS
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/s5-memory-audit-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
RUNS=/Users/durso/s5-runs
BASE="$RUNS/s5-timescale-2x2/20261117-195818"
THIRD="$RUNS/s5-timescale-seed303/20261117-230628"
OUT="$RUNS/s5-memory-audit/job-${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "FAIL: output already exists" >&2; exit 1; }
echo "CPU memory audit: commit $S5_MEMORY_AUDIT_COMMIT; job $SLURM_JOB_ID"
echo "Output: $OUT"
# Check dependencies BEFORE numerical work. Missing plotting packages go in a
# separate per-job environment; pip never modifies the shared training venv.
PLOT_PYTHON="$PROSPECTIVE_VENV/bin/python"
PLOT_IMPORTS='from matplotlib.backends.backend_agg import FigureCanvasAgg; from matplotlib.figure import Figure; import scipy'
if "$PLOT_PYTHON" -c "$PLOT_IMPORTS" > "$MPLCONFIGDIR/dependency-check.log" 2>&1; then
  echo "Plotting imports passed in the existing environment."
else
  PLOT_ENV="/Local/durso/s5-memory-audit-plots-${SLURM_JOB_ID}"
  [[ ! -e "$PLOT_ENV" ]] || { echo "FAIL: isolated plotting environment already exists" >&2; exit 1; }
  echo "Preparing isolated plotting environment: $PLOT_ENV"
  "$PROSPECTIVE_VENV/bin/python" -m venv "$PLOT_ENV"
  PLOT_PYTHON="$PLOT_ENV/bin/python"
  "$PLOT_PYTHON" -m pip --disable-pip-version-check install --no-input matplotlib scipy
  "$PLOT_PYTHON" -c "$PLOT_IMPORTS"
fi
"$PLOT_PYTHON" -m pip --disable-pip-version-check freeze > "$MPLCONFIGDIR/plot-requirements-frozen.txt"
echo "Plotting dependencies ready; starting numerical checkpoint analysis."
"$PROSPECTIVE_VENV/bin/python" -u -m experiments.s5_predictor_corrector.memory_poles \
  --native \
    "$BASE/native_matched_s5/301" \
    "$BASE/native_matched_s5/302" \
    "$THIRD/native_matched_s5/303" \
  --original \
    "$BASE/generalized_prospective_s5/301" \
    "$BASE/generalized_prospective_s5/302" \
    "$THIRD/generalized_prospective_s5/303" \
  --calibrated \
    "$BASE/calibrated_old_s5/301" \
    "$BASE/calibrated_old_s5/302" \
    "$THIRD/calibrated_old_s5/303" \
  --plots none --all-epochs --out "$OUT"
cp "$MPLCONFIGDIR/plot-requirements-frozen.txt" "$OUT/plot-requirements-frozen.txt"
echo "Numerical reports saved. Rendering figures from the saved JSON."
"$PLOT_PYTHON" -u -m experiments.s5_predictor_corrector.memory_pole_render \
  --report "$OUT/memory_poles.json" --out "$OUT/figures"
echo "Completed checkpoint audit and figure rendering: $OUT"
S5_MEMORY_AUDIT_JOB
