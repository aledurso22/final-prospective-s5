#!/usr/bin/env bash
# Milestone D step 1: correctness gates ON THE GPU.
#
# Fails loudly on a CPU-only JAX backend. The previous version would pass
# silently on CPU and be reported as GPU evidence.
#
# Note: the x64 reference gates and the NumPy/SciPy references are CPU
# computations even inside a GPU run. The genuine complex64 production check is
# tests/gp_float32_probe.py, run separately in an x64-disabled process.
set -euo pipefail
OUT="${1:?usage: gp_gpu_checks.sh <run_dir>}"
mkdir -p "$OUT"

python - <<'PY'
import sys, jax
backend = jax.default_backend()
devs = jax.devices()
print(f"backend={backend} devices={devs}")
if backend == "cpu" or not any(d.platform in ("gpu", "cuda") for d in devs):
    print("FATAL: no JAX GPU backend visible; this is not GPU evidence.")
    sys.exit(2)
print("GPU_BACKEND_OK")
PY

python -m pytest tests/test_gp_prospective.py tests/test_gp_infrastructure.py \
       tests/test_gp_review_fixes.py tests/test_s5_baseline.py \
       -q > "$OUT/gpu_checks.log" 2>&1
rc=$?
echo "pytest exit=$rc"; tail -5 "$OUT/gpu_checks.log"

echo "--- genuine complex64 production probe (separate x64-disabled process) ---"
python tests/gp_float32_probe.py | tee "$OUT/float32_probe.log"
exit $rc
