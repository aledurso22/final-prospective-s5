#!/usr/bin/env bash
# Adaptive proximal history P versus the same P with horizon multipliers frozen.
# Four tasks, one GPU each; ALL full-shape GPU smokes gate ALL training.
# A compile profile estimates buffers; it is not measured peak GPU memory.
# Real-data smoke executes the full-size update, validation, telemetry and
# checkpoint restore. Training starts fresh only after every smoke succeeds.
# The existing batch16/15-epoch/train-val/selection protocol is unchanged.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
export PROSPECTIVE_REPO="$REPO_ROOT"
source "$HERE/cluster_env.sh"
cd "$REPO_ROOT"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pinned full commit}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "FAIL: wrong commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "FAIL: modified checkout" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FAIL: missing Python: $PY" >&2; exit 1; }
case "${JAX_ENABLE_X64:-0}" in 0|false|False|FALSE|"") ;; *) echo "FAIL: x64 must be off" >&2; exit 1;; esac
export JAX_ENABLE_X64=0
unset JAX_PLATFORM_NAME
STAGE="${STAGE:-train}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$STAGE" == smoke || "$STAGE" == train ]] || { echo "FAIL: invalid STAGE" >&2; exit 1; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "FAIL: invalid DRY_RUN" >&2; exit 1; }
ALLOCATION="${SLURM_JOB_ID:-}"
[[ "$ALLOCATION" =~ ^[0-9]+$ ]] || { echo "FAIL: numeric Slurm job required" >&2; exit 1; }
IFS=',' read -r -a TOKENS <<< "${CUDA_VISIBLE_DEVICES:-}"
DISTINCT=()
GPU_COUNT=0
for t in "${TOKENS[@]:-}"; do
  t="${t//[[:space:]]/}"
  [[ -n "$t" ]] || continue
  dup=0
  for u in "${DISTINCT[@]:-}"; do [[ "$u" == "$t" ]] && dup=1; done
  if [[ "$dup" == 0 ]]; then
    DISTINCT+=("$t")
    GPU_COUNT=$((GPU_COUNT + 1))
  fi
done
[[ "$GPU_COUNT" == 4 ]] || { echo "FAIL: need exactly four allocated GPU tokens, got '${CUDA_VISIBLE_DEVICES:-}'" >&2; exit 1; }
DATA_CACHE="${S5_THREE_ARM_DATA:-$PROSPECTIVE_RUNS/sc10_official_cache}"
RUN_ROOT="$PROSPECTIVE_RUNS/s5-adaptive-proximal-history/job-$ALLOCATION"
[[ -d "$DATA_CACHE" ]] || { echo "FAIL: data cache missing: $DATA_CACHE" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "FAIL: run root already exists: $RUN_ROOT" >&2; exit 1; }
ARMS=(adaptive_proximal_history_s5 adaptive_proximal_history_s5 adaptive_proximal_history_control_s5 adaptive_proximal_history_control_s5)
SEEDS=(301 302 301 302)
echo "adaptive proximal history: allocation $ALLOCATION, stage $STAGE, GPUs ${DISTINCT[*]}"
echo "  commit $EXPECTED_COMMIT"
echo "  alpha_init=1; tau=exp(xi)/abs(kappa); m=ell=tau; mu=tau/2"
echo "  candidate: 384 learned xi; control: identical xi leaves frozen at zero"
echo "  batch16, 15epochs, same shared-parameter optimizer/split/selection; test split unread"
echo "  output $RUN_ROOT"
for i in 0 1 2 3; do echo "  GPU ${DISTINCT[$i]}: ${ARMS[$i]} seed ${SEEDS[$i]}"; done
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: CPU preflight; four GPU compile profiles and full-shape smokes; global smoke gate"
  [[ "$STAGE" == smoke ]] || echo "DRY_RUN: fresh training only after all four gates pass; CPU selected-checkpoint audit"
  exit 0
fi
mkdir -p "$RUN_ROOT"/{logs,profiles,status,smoke}
JAX_PLATFORMS=cuda "$PY" -u - <<'PY'
import os
import jax
devices = jax.devices()
print("visible devices:", devices, flush=True)
assert len(devices) == 4 and all(d.platform == "gpu" for d in devices)
assert not jax.config.read("jax_enable_x64")
cpus = len(os.sched_getaffinity(0))
print("CPU affinity:", cpus, flush=True)
assert cpus >= 32, "Slurm did not expose the requested 32 CPUs"
PY
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PY" -u - "$DATA_CACHE" "$RUN_ROOT/preflight.json" <<'PY'
import json, sys
import jax
import numpy as np
from flax.traverse_util import flatten_dict
from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_history import runner as P
from experiments.s5_adaptive_proximal_history import runner as AP
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.data import validate_official_raw_cache
assert not jax.config.read("jax_enable_x64")
assert R.BATCH_SIZE == 16 and R.SEQ_LEN == 16000 and R.N_LAYERS == 6 and H.EPOCHS == 15
validate_official_raw_cache(sys.argv[1], ("train", "val"))
rows = []
for seed in (301, 302):
    with P.configured(1), H.registered_history():
        baseline = R.init_state(P.ARM, seed)
    reference = flatten_dict(baseline.params)
    ref_blocks = np.asarray(P.transitions(baseline.params, 1, .5))
    paired = []
    for arm in AP.ARMS:
        with AP.configured(arm), H.registered_history():
            state = R.init_state(arm, seed)
            poles = R.pole_summary(state.params, arm)["production_precision"]
            blocks = np.asarray(AP.transitions(state.params))
        params = flatten_dict(state.params)
        extras = set(params) - set(reference)
        assert extras and all(k[-1] == "log_horizon_scale" for k in extras)
        assert set(reference) <= set(params)
        assert all(np.array_equal(np.asarray(reference[k]), np.asarray(params[k])) for k in reference)
        assert sum(np.asarray(params[k]).size for k in extras) == 384
        assert all(np.count_nonzero(np.asarray(params[k])) == 0 for k in extras)
        assert np.array_equal(blocks, ref_blocks), "zero multipliers must reproduce P's blocks exactly"
        assert poles["unstable"] == 0 and poles["nonfinite"] == 0
        paired.append(params)
        rows.append({"arm": arm, "seed": seed, "shared_parameters_match_P": True,
                     "zero_horizon_multipliers": True, "initial_blocks_match_P": True,
                     "additional_parameter_entries": 384,
                     "trainable_horizon": arm == AP.ARM, "initial_poles": poles})
    assert set(paired[0]) == set(paired[1])
    assert all(np.array_equal(np.asarray(paired[0][k]), np.asarray(paired[1][k])) for k in paired[0])
with open(sys.argv[2], "x") as f:
    json.dump({"tasks": rows, "batch": 16, "epochs": 15, "test_split_read": False}, f, indent=2)
print("preflight passed: shared P parameters, paired initial parameters, exact P blocks, stable poles", flush=True)
PY
printf 'commit=%s\njob=%s\ngpu_tokens=%s\nstage=%s\nalpha_init=1\nproximal_ratio=0.5\ncandidate_learned_horizon_parameters=384\ncontrol_horizon_parameters=frozen_zero\nbatch=16\nepochs=15\ndata_cache=%s\nselection_rule=max val accuracy, then min CE, then earliest epoch\ntest_split_read=false\nprofile_scope=compiler buffer estimate; no measured peak claim\n' \
  "$EXPECTED_COMMIT" "$ALLOCATION" "${DISTINCT[*]}" "$STAGE" "$DATA_CACHE" > "$RUN_ROOT/run_metadata.txt"

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

prepare_worker() {
  local i="$1"
  export CUDA_VISIBLE_DEVICES="${DISTINCT[$i]}" JAX_PLATFORMS=cuda
  export JAX_COMPILATION_CACHE_DIR="$RUN_ROOT/jax-cache/${ARMS[$i]}/${SEEDS[$i]}"
  export TMPDIR="$RUN_ROOT/tmp/${ARMS[$i]}/${SEEDS[$i]}"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"
}

smoke_worker() {
  local i="$1" arm="${ARMS[$1]}" seed="${SEEDS[$1]}"
  local log="$RUN_ROOT/logs/${arm}-${seed}" st="$RUN_ROOT/status/${arm}-${seed}"
  local profile="$RUN_ROOT/profiles/$arm-$seed.json" smoke="$RUN_ROOT/smoke/$arm/$seed"
  prepare_worker "$i"
  echo "[$arm/$seed] GPU compile profile"
  "$PY" -u -m experiments.s5_adaptive_proximal_history.runner --arm "$arm" --seed "$seed" \
    --profile --out "$profile" > "$log-profile.log" 2>&1 || { echo FAIL:profile > "$st"; return 1; }
  "$PY" - "$profile" "$arm" "$seed" "$EXPECTED_COMMIT" <<'PY' || { echo FAIL:profile_binding > "$st"; return 1; }
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert r["arm"] == sys.argv[2] and r["seed_for_shape_construction"] == int(sys.argv[3])
assert r["git_commit"] == sys.argv[4] and not r["git_worktree_dirty"]
assert r["backend"] == "gpu" and r["x64"] is False
assert r["training_executed"] is False and r["data_read"] is False
assert (r["batch_size"], r["sequence_length"], r["layers"]) == (16, 16000, 6)
assert {x["kind"] for x in r["updates"]} == {"observable", "telemetry"}
for x in r["updates"]:
    n = (x.get("memory") or {}).get("buffer_accounting_total_bytes")
    assert isinstance(n, int) and n > 0, "missing compiler memory estimate"
print("GPU compile profile verified; estimates are not measured peak memory", flush=True)
PY
  echo "[$arm/$seed] full-shape GPU smoke"
  "$PY" -u -m experiments.s5_adaptive_proximal_history.runner --arm "$arm" --seed "$seed" \
    --data-cache "$DATA_CACHE" --out "$smoke" --smoke > "$log-smoke.log" 2>&1 \
    || { echo FAIL:smoke > "$st"; return 1; }
  [[ -f "$smoke/smoke_result.json" ]] || { echo FAIL:no_smoke_result > "$st"; return 1; }
  echo SMOKE_PASS > "$st"
  echo "[$arm/$seed] smoke completed"
}

echo "STAGE 1: four profiles and full-shape GPU smokes"
for i in 0 1 2 3; do smoke_worker "$i" & PIDS+=("$!"); done
FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done
PIDS=()
for i in 0 1 2 3; do
  [[ "$(cat "$RUN_ROOT/status/${ARMS[$i]}-${SEEDS[$i]}" 2>/dev/null || true)" == SMOKE_PASS ]] || FAILED=1
done
if [[ "$FAILED" != 0 ]]; then
  echo "FAIL: at least one GPU profile/smoke failed; NO training started. See $RUN_ROOT/status and logs." >&2
  exit 1
fi
echo "All four full-shape GPU smoke gates passed."
if [[ "$STAGE" == smoke ]]; then echo "Completed smoke-only stage: $RUN_ROOT"; exit 0; fi

train_worker() {
  local i="$1" arm="${ARMS[$1]}" seed="${SEEDS[$1]}"
  local log="$RUN_ROOT/logs/${arm}-${seed}" st="$RUN_ROOT/status/${arm}-${seed}"
  prepare_worker "$i"
  echo "[$arm/$seed] training (fresh initialization after all four GPU smokes passed)"
  "$PY" -u -m experiments.s5_adaptive_proximal_history.runner --arm "$arm" --seed "$seed" \
    --data-cache "$DATA_CACHE" --out "$RUN_ROOT/$arm/$seed" --train \
    --smoke-report "$RUN_ROOT/smoke/$arm/$seed/smoke_result.json" > "$log-train.log" 2>&1 \
    || { echo FAIL:train > "$st"; return 1; }
  [[ -f "$RUN_ROOT/$arm/$seed/task_result.json" ]] || { echo FAIL:no_task_result > "$st"; return 1; }
  echo TRAINED > "$st"
  echo "[$arm/$seed] finished"
}
echo "STAGE 2: four fresh 15-epoch trainings"
for i in 0 1 2 3; do train_worker "$i" & PIDS+=("$!"); done
FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done
PIDS=()
for i in 0 1 2 3; do
  [[ "$(cat "$RUN_ROOT/status/${ARMS[$i]}-${SEEDS[$i]}" 2>/dev/null || true)" == TRAINED ]] || FAILED=1
done
if [[ "$FAILED" == 0 ]]; then
  echo "All four trainings completed; starting CPU checkpoint/impulse audit."
  audit_args=()
  for i in 0 1 2 3; do audit_args+=(--task "$RUN_ROOT/${ARMS[$i]}/${SEEDS[$i]}"); done
  if JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PY" -u -m experiments.s5_adaptive_proximal_history.audit \
      "${audit_args[@]}" --out "$RUN_ROOT/audit" > "$RUN_ROOT/logs/audit.log" 2>&1 \
      && [[ -s "$RUN_ROOT/audit/block_audit.json" && -s "$RUN_ROOT/audit/summary.txt" ]]; then
    echo PASS > "$RUN_ROOT/status/audit"
  else
    echo FAIL > "$RUN_ROOT/status/audit"
    echo "AUDIT FAILED after all four trainings completed. Saved training results remain in $RUN_ROOT; rerun only the audit." >&2
    FAILED=1
  fi
else
  echo SKIPPED:incomplete_training > "$RUN_ROOT/status/audit"
fi

# Reporting only: a formatting failure cannot invalidate a completed run.
"$PY" - "$RUN_ROOT" <<'PY' || echo "WARNING: summary formatting failed; inspect task_result.json files" >&2
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print("arm seed status selected_epoch validation_accuracy cross_entropy")
for arm in ("adaptive_proximal_history_s5", "adaptive_proximal_history_control_s5"):
    for seed in (301, 302):
        status_path = root / "status" / f"{arm}-{seed}"
        status = status_path.read_text().strip() if status_path.exists() else "MISSING"
        path = root / arm / str(seed) / "task_result.json"
        if not path.exists():
            print(arm, seed, status, "n/a n/a n/a")
            continue
        try:
            r = json.loads(path.read_text())
            v = r.get("validation") or {}
            print(arm, seed, status, r.get("selected_epoch", "n/a"),
                  v.get("accuracy", "n/a"), v.get("cross_entropy", "n/a"))
        except (ValueError, OSError, AttributeError):
            print(arm, seed, status, "unreadable")
audit = root / "status/audit"
print("AUDIT:", audit.read_text().strip() if audit.exists() else "MISSING")
PY
if [[ "$FAILED" != 0 ]]; then echo "INCOMPLETE: inspect $RUN_ROOT/status and logs" >&2; exit 1; fi
echo "Completed adaptive/frozen P comparison: $RUN_ROOT"
echo "No finalizer ran; the test split was not read."
