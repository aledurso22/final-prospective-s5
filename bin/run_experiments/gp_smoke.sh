#!/usr/bin/env bash
# The FROZEN historical sMNIST smoke.
#
# Deliberately NO --checkpoint_dir: the E2-002/E2-004 records were produced
# without checkpointing, so adding it here would make the "frozen" run not
# actually frozen. Matched treatment/control runs use gp_matched_pair.sh,
# which does enable it.
#
# Validates integration only. It is NOT evidence for the research claim and
# must not be reported as a benchmark result.
#
#   ./bin/run_experiments/gp_smoke.sh <run_dir> <mechanism> [extra args...]
set -euo pipefail
OUT="${1:?usage: gp_smoke.sh <run_dir> <mechanism>}"; shift
MECH="${1:?mechanism}"; shift || true
mkdir -p "$OUT"
exec python run_train.py \
  --dataset=mnist-classification \
  --epochs=1 \
  --bsz=64 \
  --n_layers=2 \
  --d_model=64 \
  --ssm_size_base=64 \
  --blocks=2 \
  --batchnorm=False \
  --bidirectional=False \
  --p_dropout=0.0 \
  --jax_seed=1919 \
  --USE_WANDB=False \
  --ssm_mechanism="${MECH}" \
  "$@"
