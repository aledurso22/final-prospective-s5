#!/usr/bin/env bash
# A MATCHED treatment/control pair with IDENTICAL explicit stability settings.
#
# This is NOT the frozen historical smoke. The historical plain smoke keeps its
# original unclipped configuration and is run separately by gp_smoke.sh; mixing
# the two would compare a newly clipped GP run against an unclipped plain run
# and attribute the difference to prospectivity.
#
#   ./bin/run_experiments/gp_matched_pair.sh <run_dir> <mechanism> [extra...]
set -euo pipefail
OUT="${1:?usage: gp_matched_pair.sh <run_dir> <mechanism>}"; shift
MECH="${1:?mechanism}"; shift || true
mkdir -p "$OUT"
exec python run_train.py \
  --dataset=mnist-classification \
  --epochs=1 --bsz=64 --n_layers=2 --d_model=64 --ssm_size_base=64 --blocks=2 \
  --batchnorm=False --bidirectional=False --p_dropout=0.0 \
  --jax_seed=1919 --USE_WANDB=False \
  --clip_eigs=True \
  --ssm_mechanism="${MECH}" \
  --checkpoint_dir="${OUT}/ckpt" \
  "$@"
