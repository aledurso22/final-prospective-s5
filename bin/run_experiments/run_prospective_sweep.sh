#!/usr/bin/env bash
# Prospective-S5 stage-1 sweep scaffold (sequential MNIST smoke configuration).
#
# This script only PRINTS the commands by default so it can be reviewed before
# any GPU time is spent.  Set EXECUTE=1 to actually run them, or pipe the
# printed lines into a cluster job array.
#
#   ./bin/run_experiments/run_prospective_sweep.sh            # dry run
#   EXECUTE=1 ./bin/run_experiments/run_prospective_sweep.sh  # run serially
#
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
EPOCHS="${EPOCHS:-10}"
SEEDS="${SEEDS:-1919 1920 1921}"
ALPHAS="${ALPHAS:-0 0.125 0.25 0.5 1.0}"
PLACEMENTS="${PLACEMENTS:-last all}"
RESULTS_DIR="${RESULTS_DIR:-./results/prospective_mnist}"

mkdir -p "${RESULTS_DIR}"

# Shared model/optimizer configuration.  Deliberately small: this stage is a
# plumbing and lag smoke test, not a benchmark run.  bidirectional MUST stay
# False -- the prospective operator is causal and rejects bidirectional S5.
COMMON=(
  --dataset=mnist-classification
  --epochs="${EPOCHS}"
  --bsz=64
  --n_layers=3
  --d_model=128
  --ssm_size_base=128
  --blocks=4
  --batchnorm=False
  --bidirectional=False
  --prospective_alpha_max=1.0
  --USE_WANDB=False
)

emit () {  # emit <run_name> <extra args...>
  local name="$1"; shift
  local log="${RESULTS_DIR}/${name}.log"
  if [[ "${EXECUTE}" == "1" ]]; then
    echo "### ${name}"
    python run_train.py "${COMMON[@]}" "$@" 2>&1 | tee "${log}"
  else
    echo "python run_train.py ${COMMON[*]} $* 2>&1 | tee ${log}"
  fi
}

for SEED in ${SEEDS}; do
  # (1) Ordinary S5 baseline.
  emit "baseline_seed${SEED}" \
    --prospective_mode=off --jax_seed="${SEED}"

  for PLACEMENT in ${PLACEMENTS}; do
    # (2) Fixed-alpha prospective S5 over the diagnostic alpha grid.
    #     alpha=0 is included on purpose: it must match the baseline exactly
    #     and therefore acts as an end-to-end identity check.
    for ALPHA in ${ALPHAS}; do
      emit "fixed_a${ALPHA}_${PLACEMENT}_seed${SEED}" \
        --prospective_mode=lead \
        --prospective_alpha="${ALPHA}" \
        --prospective_alpha_learned=False \
        --prospective_layers="${PLACEMENT}" \
        --jax_seed="${SEED}"
    done

    # (3) Learned-alpha prospective S5, initialized at 0.25.
    emit "learned_${PLACEMENT}_seed${SEED}" \
      --prospective_mode=lead \
      --prospective_alpha=0.25 \
      --prospective_alpha_learned=True \
      --prospective_layers="${PLACEMENT}" \
      --jax_seed="${SEED}"
  done
done
