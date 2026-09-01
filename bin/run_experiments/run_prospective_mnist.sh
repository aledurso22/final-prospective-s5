#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-lead}"
ALPHA="${ALPHA:-0.25}"
LEARNED="${LEARNED:-False}"
PLACEMENT="${PLACEMENT:-last}"
SEED="${SEED:-1919}"
EPOCHS="${EPOCHS:-10}"

python run_train.py \
  --dataset=mnist-classification \
  --epochs="${EPOCHS}" \
  --bsz=64 \
  --n_layers=3 \
  --d_model=128 \
  --ssm_size_base=128 \
  --blocks=4 \
  --batchnorm=False \
  --bidirectional=False \
  --prospective_mode="${MODE}" \
  --prospective_alpha="${ALPHA}" \
  --prospective_alpha_learned="${LEARNED}" \
  --prospective_layers="${PLACEMENT}" \
  --prospective_alpha_max=1.0 \
  --jax_seed="${SEED}" \
  --USE_WANDB=False
