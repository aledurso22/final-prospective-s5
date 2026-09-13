#!/usr/bin/env bash
# Milestone D, step 1: run the CPU-verified correctness gates ON THE GPU.
# Nothing here trains; it re-runs the predeclared numerical gates on the
# target hardware, where dtype and kernel selection differ.
set -euo pipefail
OUT="${1:?usage: gp_gpu_checks.sh <run_dir>}"
mkdir -p "$OUT"
python -m pytest tests/test_gp_prospective.py tests/test_gp_infrastructure.py \
       tests/test_s5_baseline.py -q > "$OUT/gpu_checks.log" 2>&1
echo "exit=$?"
tail -5 "$OUT/gpu_checks.log"
