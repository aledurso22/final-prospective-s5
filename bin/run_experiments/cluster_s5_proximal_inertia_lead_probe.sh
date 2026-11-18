#!/usr/bin/env bash
# INPUT-LEAD ABLATION, VALIDATION BEFORE TRAINING. ONE RTX 4090. Trains nothing.
#   GPU (sequential, separate processes):
#     1. sigma=1 at this commit vs the trained baseline commit acb4d85, one
#        production-width layer at 16,000 tokens: outputs, every gradient, Ad, bd
#        (exactness checked and reported, not assumed);
#     2. measured full-shape forward/backward memory and synchronized step time
#        for sigma=0 (proximal_inertial_nolead_s5) and sigma=1 (proximal_inertial_s5).
#   CPU, alongside: read-only audit of the SELECTED sigma=1 checkpoints of
#     $BASELINE_ROOT seeds 301/302 with the diagnostic input policy sigma=0 at the
#     same frozen weights (B/C-coupled kernel differences, auxiliary residues,
#     interference). Checkpoints are hashed before/after; nothing is written there.
# Status per step in $RUN_ROOT/status: PASS or FAIL; any FAIL exits 1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$HERE/cluster_env.sh"
cd "$REPO_ROOT"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned full sha}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: wrong commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "FAIL: modified checkout" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FAIL: missing Python: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in 0|false|False|FALSE|"") ;; *) echo "FAIL: x64 must be off" >&2; exit 1;; esac
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
DRY_RUN="${DRY_RUN:-0}"
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric Slurm job required" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#TOKENS[@]}" == 1 && -n "${TOKENS[0]//[[:space:]]/}" ]] || {
  echo "FAIL: need exactly one allocated GPU token, got '${CUDA_VISIBLE_DEVICES:-}'" >&2; exit 1; }
BASELINE_ROOT="${BASELINE_ROOT:-$PROSPECTIVE_RUNS/s5-proximal-inertia/job-76984}"
RUN_ROOT="$PROSPECTIVE_RUNS/s5-proximal-inertia-lead/probe-job-$ALLOCATION"
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: run root exists: $RUN_ROOT" >&2; exit 1; }
echo "input-lead probe: allocation $ALLOCATION, GPU ${TOKENS[0]}, commit $EXPECTED_COMMIT"
echo "  baseline (read only): $BASELINE_ROOT/proximal_inertial_s5/{301,302}"
echo "  output: $RUN_ROOT"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN GPU: lead_identity (sigma=1 vs acb4d85); gpu_profile proximal_inertial_nolead_s5; gpu_profile proximal_inertial_s5"
  echo "DRY_RUN CPU: block_audit --compare-input-lead 0 on the selected baseline checkpoints"
  exit 0
fi
mkdir -p "$RUN_ROOT"/{logs,status}
printf 'commit=%s\njob=%s\ngpu_token=%s\nbaseline_root=%s\npurpose=input-lead ablation validation; no training\n' \
  "$EXPECTED_COMMIT" "$ALLOCATION" "${TOKENS[0]}" "$BASELINE_ROOT" > "$RUN_ROOT/run_metadata.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$RUN_ROOT/gpu.csv" 2>&1 || true

AUDIT_PID=""
cleanup() { [[ -n "$AUDIT_PID" ]] && kill -TERM "$AUDIT_PID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

( export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  if "$PY" -u -m experiments.s5_proximal_inertia.block_audit \
       --task "$BASELINE_ROOT/proximal_inertial_s5/301" --task "$BASELINE_ROOT/proximal_inertial_s5/302" \
       --compare-input-lead 0 --out "$RUN_ROOT/checkpoint-audit" > "$RUN_ROOT/logs/checkpoint-audit.log" 2>&1; then
    echo PASS > "$RUN_ROOT/status/checkpoint-audit"
  else
    echo "FAIL exit=$?" > "$RUN_ROOT/status/checkpoint-audit"
  fi ) &
AUDIT_PID="$!"

export JAX_PLATFORMS=cuda JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache" TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
step() {   # step NAME COMMAND...  -> status/NAME = PASS | FAIL exit=N
  local name="$1"; shift
  echo "[$name]"
  if "$@" > "$RUN_ROOT/logs/$name.log" 2>&1; then echo PASS > "$RUN_ROOT/status/$name"
  else echo "FAIL exit=$?" > "$RUN_ROOT/status/$name"; fi
}
step sigma1-identity "$PY" -u -m experiments.s5_proximal_inertia.lead_identity --require-gpu \
  --out "$RUN_ROOT/sigma1_identity_gpu.json"
step gpu-profile-sigma0 "$PY" -u -m experiments.s5_proximal_inertia.gpu_profile \
  --arm proximal_inertial_nolead_s5 --seed 301 --require-gpu --out "$RUN_ROOT/gpu_profile_sigma0.json"
step gpu-profile-sigma1 "$PY" -u -m experiments.s5_proximal_inertia.gpu_profile \
  --arm proximal_inertial_s5 --seed 301 --require-gpu --out "$RUN_ROOT/gpu_profile_sigma1.json"

wait "$AUDIT_PID" || true
AUDIT_PID=""
echo "==== probe summary ($RUN_ROOT)"
FAILED=0
for s in "$RUN_ROOT"/status/*; do
  printf '%-24s %s\n' "$(basename "$s")" "$(cat "$s")"
  [[ "$(cat "$s")" == PASS ]] || FAILED=1
done
for n in checkpoint-audit sigma1-identity gpu-profile-sigma0 gpu-profile-sigma1; do
  [[ -f "$RUN_ROOT/status/$n" ]] || { echo "FAIL: no status for $n" >&2; FAILED=1; }
done
"$PY" - "$RUN_ROOT" <<'PY' || echo "WARNING: probe summary formatting failed" >&2
import json, sys
from pathlib import Path
from experiments.s5_proximal_inertia.job_report import num
root = Path(sys.argv[1])
def load(name):
    try: return json.loads((root / name).read_text())
    except (OSError, ValueError): return None
i = load("sigma1_identity_gpu.json")
if i:
    s1 = i["sigma1_vs_legacy"]
    print(f"sigma=1 vs acb4d85 on {i['backend']}: bitwise identical = {s1['all_bitwise_identical']}; "
          f"output max abs diff {s1['output']['max_abs_difference']:.3g}")
for tag in ("sigma0", "sigma1"):
    g = load(f"gpu_profile_{tag}.json")
    if g:
        st = g.get("steady_step_seconds") or {}
        print(f"{tag}: peak {num(g.get('peak_gib'), '.2f')} GiB, synchronized step median "
              f"{num(st.get('median'), '.3f')} s (max {num(st.get('max'), '.3f')}), device {g.get('device')}")
s = root / "checkpoint-audit" / "summary.txt"
if s.exists():
    print(s.read_text())
PY
if [[ "$FAILED" != 0 ]]; then echo "INCOMPLETE: see $RUN_ROOT/status and logs" >&2; exit 1; fi
echo "Probe passed: $RUN_ROOT"
