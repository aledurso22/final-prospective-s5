#!/usr/bin/env bash
# Plain-S5 one-epoch sequential-MNIST smoke test.
#
# This exact configuration is the trusted baseline. It is re-used verbatim on
# the prospective-lead branch (with --prospective_* flags appended) so that
# alpha=0 can be compared against plain S5 under identical seed, batch size,
# model size, data order and hardware.
set -euo pipefail

EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-1919}"

exec python run_train.py \
  --dataset=mnist-classification \
  --epochs="${EPOCHS}" \
  --bsz=64 \
  --n_layers=2 \
  --d_model=64 \
  --ssm_size_base=64 \
  --blocks=2 \
  --batchnorm=False \
  --bidirectional=False \
  --p_dropout=0.0 \
  --jax_seed="${SEED}" \
  --USE_WANDB=False \
  "$@"
