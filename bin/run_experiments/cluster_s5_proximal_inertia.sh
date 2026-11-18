#!/usr/bin/env bash
# PROXIMAL INERTIAL S5: candidate (m=1.5 tau) / control (m=tau) x seeds 301, 302.
# One allocation, FOUR GPUs, one task per GPU, all four in parallel; the
# trained-P block audit runs on CPU alongside. Every task, on its own GPU:
#   compile-time memory profile -> measured full-shape memory/step timing ->
#   full-path GPU smoke (2 updates, validation, checkpoint restore) ->
#   training (only if that smoke passed; enforced again by the runner).
# Shared protocol unchanged: native init/parameters, outer optimizer, batch 16,
# 15 epochs, official train/validation split, selection rule. The test split
# is never read. Existing runs are only read (P audit), never written.
#   STAGE=smoke|train (default train)   DRY_RUN=1 prints the plan only
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
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric Slurm job required" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
DISTINCT=()
for t in "${TOKENS[@]}"; do
  t="${t//[[:space:]]/}"; [[ -n "$t" ]] || continue
  dup=0; for u in "${DISTINCT[@]:-}"; do [[ "$u" == "$t" ]] && dup=1; done
  [[ "$dup" == 1 ]] || DISTINCT+=("$t")
done
[[ "${#DISTINCT[@]}" == 4 ]] || { echo "FAIL: need exactly four allocated GPU tokens, got '${CUDA_VISIBLE_DEVICES:-}'" >&2; exit 1; }
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
RUN_ROOT="$PROSPECTIVE_RUNS/s5-proximal-inertia/job-$ALLOCATION"
[[ -d "$DATA_CACHE" ]] || { echo "FAIL: data cache missing: $DATA_CACHE" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: run root already exists: $RUN_ROOT" >&2; exit 1; }

ARMS=(proximal_inertial_s5 proximal_inertial_s5 proximal_inertial_control_s5 proximal_inertial_control_s5)
SEEDS=(301 302 301 302)
PH="$PROSPECTIVE_RUNS/s5-proximal-history/job-76878"
PM="$PROSPECTIVE_RUNS/s5-morning-followup/job-76913/fresh/proximal_history_s5"
P_TASKS=("$PH/alpha1/301" "$PH/alpha1/302" "$PM/alpha1/303" "$PM/alpha1/304" "$PM/alpha1/305"
         "$PH/alpha0p1/301" "$PM/alpha0p1/302")

echo "proximal inertial S5: allocation $ALLOCATION, stage $STAGE, GPUs ${DISTINCT[*]}"
echo "  commit $EXPECTED_COMMIT"
echo "  alpha=0.1 proximal_ratio=0.5; candidate mass_ratio=1.5, control mass_ratio=1.0"
echo "  batch 16, 15 epochs, native init/optimizer/split/selection unchanged; test split unread"
echo "  output $RUN_ROOT"
for i in 0 1 2 3; do echo "  GPU ${DISTINCT[$i]}: ${ARMS[$i]} seed ${SEEDS[$i]}"; done
echo "  CPU alongside: trained-P block audit of ${#P_TASKS[@]} tasks (missing ones recorded, not fatal)"
if [[ "$DRY_RUN" == 1 ]]; then
  for i in 0 1 2 3; do
    echo "DRY_RUN ${ARMS[$i]}/${SEEDS[$i]}: profile, gpu_profile, smoke$([[ $STAGE == train ]] && echo ', train after its smoke passes')"
  done
  exit 0
fi

mkdir -p "$RUN_ROOT"/{logs,profiles,status,smoke}
JAX_PLATFORMS=cuda "$PY" -u - <<'PY'
import jax
d = jax.devices()
print("visible devices:", d, flush=True)
assert len(d) == 4 and all(x.platform == "gpu" for x in d)
assert not jax.config.read("jax_enable_x64")
PY
JAX_PLATFORMS=cpu "$PY" -u - "$DATA_CACHE" "$RUN_ROOT/preflight.json" <<'PY'
import json, sys
import jax, numpy as np
from flax.traverse_util import flatten_dict
from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_inertia import runner as PIR
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.data import validate_official_raw_cache
assert not jax.config.read("jax_enable_x64")
assert R.BATCH_SIZE == 16 and R.SEQ_LEN == 16000 and R.N_LAYERS == 6 and H.EPOCHS == 15
validate_official_raw_cache(sys.argv[1], ("train", "val"))
rows = []
for seed in (301, 302):
    native = R.init_state("native_matched_s5", seed)
    pn = flatten_dict(native.params)
    tn = jax.tree_util.tree_flatten(native.opt_state)
    for arm in PIR.ARMS:
        with PIR.configured(arm), H.registered_history():
            state = R.init_state(arm, seed)
            poles = R.pole_summary(state.params, arm)["production_precision"]
        pp = flatten_dict(state.params)
        assert set(pn) == set(pp) and all(np.array_equal(np.asarray(pn[k]), np.asarray(pp[k])) for k in pn)
        tp = jax.tree_util.tree_flatten(state.opt_state)
        assert tn[1] == tp[1] and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(tn[0], tp[0]))
        assert poles["unstable"] == 0 and poles["nonfinite"] == 0
        rows.append({"arm": arm, "seed": seed, "native_parameters_identical": True,
                     "optimizer_identical": True, "initial_poles": poles})
json.dump({"tasks": rows, "batch": 16, "epochs": 15, "test_split_read": False},
          open(sys.argv[2], "w"), indent=2)
print("preflight passed: native parameters/optimizer identical, initial poles stable", flush=True)
PY
printf 'commit=%s\njob=%s\ngpu_tokens=%s\nstage=%s\nalpha=0.1\nproximal_ratio=0.5\ncandidate_mass_ratio=1.5\ncontrol_mass_ratio=1.0\nbatch=16\nepochs=15\ndata_cache=%s\nselection_rule=max val accuracy, then min CE, then earliest epoch\ntest_split_read=false\n' \
  "$EXPECTED_COMMIT" "$ALLOCATION" "${DISTINCT[*]}" "$STAGE" "$DATA_CACHE" > "$RUN_ROOT/run_metadata.txt"

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# trained-P audit, CPU only, alongside the GPU workers
( export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  args=(); for t in "${P_TASKS[@]}"; do args+=(--task "$t"); done
  # status: PASS (all listed P tasks audited), PARTIAL (audit succeeded, some
  # listed tasks unfinished/unavailable; reported, not fatal) or FAIL (fatal)
  if "$PY" -u -m experiments.s5_proximal_inertia.block_audit "${args[@]}" --allow-missing \
       --out "$RUN_ROOT/p-audit" > "$RUN_ROOT/logs/p-audit.log" 2>&1; then
    "$PY" -m experiments.s5_proximal_inertia.job_report p-audit-status \
      "$RUN_ROOT/p-audit/block_audit.json" > "$RUN_ROOT/status/p-audit" 2>>"$RUN_ROOT/logs/p-audit.log" || true
  else
    echo "FAIL block_audit_exit=$?" > "$RUN_ROOT/status/p-audit"
  fi ) &
PIDS+=("$!")

worker() {
  local i="$1" arm="${ARMS[$1]}" seed="${SEEDS[$1]}" token="${DISTINCT[$1]}"
  local tag="$arm/$seed" log="$RUN_ROOT/logs/${arm}-${seed}" st="$RUN_ROOT/status/${arm}-${seed}"
  export CUDA_VISIBLE_DEVICES="$token" JAX_PLATFORMS=cuda
  export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache/$arm/$seed" TMPDIR="$RUN_ROOT/tmp/$arm/$seed"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
  local common=(--arm "$arm" --seed "$seed")
  echo "[$tag] compile profile"
  "$PY" -u -m experiments.s5_proximal_inertia.runner "${common[@]}" --profile \
    --out "$RUN_ROOT/profiles/$arm-$seed-compile.json" > "$log-profile.log" 2>&1 || { echo "FAIL:compile_profile" > "$st"; return 1; }
  echo "[$tag] measured full-shape memory/timing"
  "$PY" -u -m experiments.s5_proximal_inertia.gpu_profile "${common[@]}" --require-gpu \
    --out "$RUN_ROOT/profiles/$arm-$seed-gpu.json" > "$log-gpu.log" 2>&1 || { echo "FAIL:gpu_profile" > "$st"; return 1; }
  local smoke="$RUN_ROOT/smoke/$arm/$seed"
  echo "[$tag] GPU smoke"
  "$PY" -u -m experiments.s5_proximal_inertia.runner "${common[@]}" --data-cache "$DATA_CACHE" \
    --out "$smoke" --smoke > "$log-smoke.log" 2>&1 || { echo "FAIL:smoke" > "$st"; return 1; }
  if [[ "$STAGE" == smoke ]]; then echo "SMOKE_PASS" > "$st"; return 0; fi
  echo "[$tag] training (fresh initialization; GPU smoke passed)"
  "$PY" -u -m experiments.s5_proximal_inertia.runner "${common[@]}" --data-cache "$DATA_CACHE" \
    --out "$RUN_ROOT/$arm/$seed" --train --smoke-report "$smoke/smoke_result.json" \
    > "$log-train.log" 2>&1 || { echo "FAIL:train" > "$st"; return 1; }
  [[ -f "$RUN_ROOT/$arm/$seed/task_result.json" ]] || { echo "FAIL:no_task_result" > "$st"; return 1; }
  echo "TRAINED" > "$st"
  echo "[$tag] finished"
}

for i in 0 1 2 3; do worker "$i" & PIDS+=("$!"); done
FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done
PIDS=()
P_AUDIT_STATUS="$(cat "$RUN_ROOT/status/p-audit" 2>/dev/null || echo "FAIL no_status")"
case "$P_AUDIT_STATUS" in
  PASS*) ;;
  PARTIAL*) echo "NOTE: trained-P audit $P_AUDIT_STATUS (unfinished P tasks listed in $RUN_ROOT/p-audit/summary.txt)" ;;
  *) echo "FAIL: trained-P audit $P_AUDIT_STATUS; see $RUN_ROOT/logs/p-audit.log" >&2
     echo "$P_AUDIT_STATUS" > "$RUN_ROOT/status/p-audit"; FAILED=1 ;;
esac

if [[ "$STAGE" == train ]]; then
  trained=()
  for i in 0 1 2 3; do
    [[ "$(cat "$RUN_ROOT/status/${ARMS[$i]}-${SEEDS[$i]}" 2>/dev/null)" == TRAINED ]] && trained+=(--task "$RUN_ROOT/${ARMS[$i]}/${SEEDS[$i]}")
  done
  if [[ "${#trained[@]}" -gt 0 ]]; then
    echo "trained-pole and B/C-coupled impulse audit of the new tasks"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PY" -u -m experiments.s5_proximal_inertia.block_audit \
      "${trained[@]}" --out "$RUN_ROOT/inertia-audit" > "$RUN_ROOT/logs/inertia-audit.log" 2>&1 \
      && echo PASS > "$RUN_ROOT/status/inertia-audit" || { echo FAIL > "$RUN_ROOT/status/inertia-audit"; FAILED=1; }
  fi
fi

echo "==== summary ($RUN_ROOT)"
# reporting only: never changes the job outcome; null telemetry prints as n/a
"$PY" -m experiments.s5_proximal_inertia.job_report summary "$RUN_ROOT" \
  || echo "WARNING: summary formatting failed; results are in $RUN_ROOT" >&2
if [[ "$FAILED" != 0 ]]; then echo "INCOMPLETE: see $RUN_ROOT/status and logs" >&2; exit 1; fi
echo "Completed stage $STAGE: $RUN_ROOT"
