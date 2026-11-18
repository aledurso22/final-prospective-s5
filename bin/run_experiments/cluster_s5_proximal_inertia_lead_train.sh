#!/usr/bin/env bash
# INPUT-LEAD ABLATION TRAINING: proximal_inertial_nolead_s5 (sigma=0, m=1.5 tau)
# x seeds 301, 302. TWO RTX 4090s, one fresh task each, in parallel.
# The matched baseline is the COMPLETED proximal_inertial_s5 (sigma=1, m=1.5 tau)
# run in $BASELINE_ROOT -- not the m=tau control -- and it is reused only if:
#   * baseline_check.py passes (completed, rule-consistent selection, matched
#     configuration, identical recorded protocol: batch, epochs, scan, ratios,
#     data manifest digest, JAX version, float32, trained at acb4d85), and
#   * sigma=1 at this commit is bitwise identical to acb4d85 on this GPU type.
# Otherwise the job stops BEFORE any training (fail closed); the baseline would
# then have to be rerun alongside instead of compared across protocols.
# Every task: compile profile -> measured memory/step time -> GPU smoke -> train
# (only after its own smoke passed). Unchanged: native init/parameters, outer
# optimizer, batch 16, 15 epochs, official split, selection rule. The test split
# is never read. Output goes to a new directory; the baseline is only read.
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
STAGE="${STAGE:-train}"; DRY_RUN="${DRY_RUN:-0}"
[[ "$STAGE" == smoke || "$STAGE" == train ]] || { echo "FAIL: invalid STAGE" >&2; exit 1; }
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric Slurm job required" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
DISTINCT=()
for t in "${TOKENS[@]}"; do
  t="${t//[[:space:]]/}"; [[ -n "$t" ]] || continue
  dup=0; for u in "${DISTINCT[@]:-}"; do [[ "$u" == "$t" ]] && dup=1; done
  [[ "$dup" == 1 ]] || DISTINCT+=("$t")
done
[[ "${#DISTINCT[@]}" == 2 ]] || { echo "FAIL: need exactly two allocated GPU tokens, got '${CUDA_VISIBLE_DEVICES:-}'" >&2; exit 1; }
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
BASELINE_ROOT="${BASELINE_ROOT:-$PROSPECTIVE_RUNS/s5-proximal-inertia/job-76984}"
RUN_ROOT="$PROSPECTIVE_RUNS/s5-proximal-inertia-lead/train-job-$ALLOCATION"
[[ -d "$DATA_CACHE" ]] || { echo "FAIL: data cache missing: $DATA_CACHE" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: run root exists: $RUN_ROOT" >&2; exit 1; }
ARM=proximal_inertial_nolead_s5
SEEDS=(301 302)
echo "input-lead ablation training: allocation $ALLOCATION, stage $STAGE, GPUs ${DISTINCT[*]}"
echo "  commit $EXPECTED_COMMIT; $ARM (sigma=0, m=1.5 tau, mu=tau/2, alpha=0.1)"
echo "  matched baseline (read only, reused if checks pass): $BASELINE_ROOT/proximal_inertial_s5"
echo "  output $RUN_ROOT"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN preflight: baseline_check, native identity, sigma=1 GPU identity vs acb4d85"
  for i in 0 1; do echo "DRY_RUN GPU ${DISTINCT[$i]}: $ARM/${SEEDS[$i]} profile, gpu_profile, smoke$([[ $STAGE == train ]] && echo ', train')"; done
  exit 0
fi
mkdir -p "$RUN_ROOT"/{logs,profiles,status,smoke}
printf 'commit=%s\njob=%s\ngpu_tokens=%s\nstage=%s\narm=%s\ninput_lead=0.0\nmass_ratio=1.5\nproximal_ratio=0.5\nalpha=0.1\nbaseline_root=%s\nbatch=16\nepochs=15\ndata_cache=%s\nselection_rule=max val accuracy, then min CE, then earliest epoch\ntest_split_read=false\n' \
  "$EXPECTED_COMMIT" "$ALLOCATION" "${DISTINCT[*]}" "$STAGE" "$ARM" "$BASELINE_ROOT" "$DATA_CACHE" > "$RUN_ROOT/run_metadata.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$RUN_ROOT/gpu.csv" 2>&1 || true

# ---------------------------------------------------------- preflight ---
JAX_PLATFORMS=cpu "$PY" -u -m experiments.s5_proximal_inertia.baseline_check \
  --baseline-root "$BASELINE_ROOT" --data-cache "$DATA_CACHE" --out "$RUN_ROOT/baseline_check.json" \
  || { echo "FAIL: baseline reuse not supported; see $RUN_ROOT/baseline_check.json. No training started." >&2; exit 1; }
JAX_PLATFORMS=cpu "$PY" -u - "$DATA_CACHE" "$RUN_ROOT/preflight.json" <<'PY'
import json, sys
import jax, numpy as np
from flax.traverse_util import flatten_dict
from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_inertia import runner as PIR
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.data import validate_official_raw_cache
arm = "proximal_inertial_nolead_s5"
assert not jax.config.read("jax_enable_x64")
assert R.BATCH_SIZE == 16 and R.SEQ_LEN == 16000 and R.N_LAYERS == 6 and H.EPOCHS == 15
assert PIR.ARMS[arm] == 1.5 and PIR.INPUT_LEAD[arm] == 0.0
validate_official_raw_cache(sys.argv[1], ("train", "val"))
rows = []
for seed in (301, 302):
    native = R.init_state("native_matched_s5", seed)
    with PIR.configured(arm), H.registered_history():
        state = R.init_state(arm, seed)
        poles = R.pole_summary(state.params, arm)["production_precision"]
    pn, pp = flatten_dict(native.params), flatten_dict(state.params)
    assert set(pn) == set(pp) and all(np.array_equal(np.asarray(pn[k]), np.asarray(pp[k])) for k in pn)
    tn, tp = jax.tree_util.tree_flatten(native.opt_state), jax.tree_util.tree_flatten(state.opt_state)
    assert tn[1] == tp[1] and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(tn[0], tp[0]))
    assert poles["unstable"] == 0 and poles["nonfinite"] == 0
    rows.append({"seed": seed, "native_parameters_identical": True, "optimizer_identical": True,
                 "initial_poles": poles})
json.dump({"arm": arm, "tasks": rows, "test_split_read": False}, open(sys.argv[2], "w"), indent=2)
print("preflight passed", flush=True)
PY
CUDA_VISIBLE_DEVICES="${DISTINCT[0]}" JAX_PLATFORMS=cuda "$PY" -u -m experiments.s5_proximal_inertia.lead_identity \
  --require-gpu --out "$RUN_ROOT/sigma1_identity_gpu.json" \
  || { echo "FAIL: sigma=1 at this commit is not bitwise the baseline's computation on this GPU. No training started." >&2; exit 1; }

# ------------------------------------------------------------ workers ---
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
worker() {
  local seed="${SEEDS[$1]}" token="${DISTINCT[$1]}"
  local log="$RUN_ROOT/logs/$ARM-$seed" st="$RUN_ROOT/status/$ARM-$seed" smoke="$RUN_ROOT/smoke/$ARM/$seed"
  export CUDA_VISIBLE_DEVICES="$token" JAX_PLATFORMS=cuda
  export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache/$seed" TMPDIR="$RUN_ROOT/tmp/$seed"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
  local common=(--arm "$ARM" --seed "$seed")
  "$PY" -u -m experiments.s5_proximal_inertia.runner "${common[@]}" --profile \
    --out "$RUN_ROOT/profiles/$ARM-$seed-compile.json" > "$log-profile.log" 2>&1 || { echo "FAIL:compile_profile" > "$st"; return 1; }
  "$PY" -u -m experiments.s5_proximal_inertia.gpu_profile "${common[@]}" --require-gpu \
    --out "$RUN_ROOT/profiles/$ARM-$seed-gpu.json" > "$log-gpu.log" 2>&1 || { echo "FAIL:gpu_profile" > "$st"; return 1; }
  "$PY" -u -m experiments.s5_proximal_inertia.runner "${common[@]}" --data-cache "$DATA_CACHE" \
    --out "$smoke" --smoke > "$log-smoke.log" 2>&1 || { echo "FAIL:smoke" > "$st"; return 1; }
  if [[ "$STAGE" == smoke ]]; then echo SMOKE_PASS > "$st"; return 0; fi
  "$PY" -u -m experiments.s5_proximal_inertia.runner "${common[@]}" --data-cache "$DATA_CACHE" \
    --out "$RUN_ROOT/$ARM/$seed" --train --smoke-report "$smoke/smoke_result.json" \
    > "$log-train.log" 2>&1 || { echo "FAIL:train" > "$st"; return 1; }
  [[ -f "$RUN_ROOT/$ARM/$seed/task_result.json" ]] || { echo "FAIL:no_task_result" > "$st"; return 1; }
  echo TRAINED > "$st"
}
for i in 0 1; do worker "$i" & PIDS+=("$!"); done
FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done
PIDS=()

if [[ "$STAGE" == train && "$FAILED" == 0 ]]; then
  # trained sigma=0 poles and kernels, plus the sigma=1 policy as an untrained counterfactual
  if JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PY" -u -m experiments.s5_proximal_inertia.block_audit \
       --task "$RUN_ROOT/$ARM/301" --task "$RUN_ROOT/$ARM/302" --compare-input-lead 1 \
       --out "$RUN_ROOT/audit" > "$RUN_ROOT/logs/audit.log" 2>&1; then
    echo PASS > "$RUN_ROOT/status/audit"
  else
    echo "FAIL exit=$?" > "$RUN_ROOT/status/audit"; FAILED=1
  fi
fi

echo "==== summary ($RUN_ROOT)"
"$PY" -m experiments.s5_proximal_inertia.job_report summary "$RUN_ROOT" \
  || echo "WARNING: summary formatting failed" >&2
if [[ "$STAGE" == train ]]; then
  echo "==== paired with the matched sigma=1 baseline (reused; checks in baseline_check.json)"
  "$PY" -m experiments.s5_proximal_inertia.job_report paired "$RUN_ROOT" "$BASELINE_ROOT" \
    || echo "WARNING: paired comparison formatting failed" >&2
fi
if [[ "$FAILED" != 0 ]]; then echo "INCOMPLETE: see $RUN_ROOT/status and logs" >&2; exit 1; fi
echo "Completed stage $STAGE: $RUN_ROOT"
