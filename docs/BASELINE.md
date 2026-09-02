# Trusted plain-S5 baseline

This file records the verified state of the `main` branch: plain S5, modernized
only enough to run on a current JAX, with no prospective code of any kind.

## What was modernized

Three changes, all API-level. **No equation, discretization, parameter,
associative operator or parallel scan was touched.**

| File | Change | Reason |
|---|---|---|
| `s5/ssm.py` | `np.DeviceArray` -> `jax.Array` on 4 `S5SSM` dataclass field annotations | `jax.numpy.DeviceArray` was removed after JAX 0.4.13 |
| `s5/train_helpers.py` | `variables["params"].unfreeze()` -> `flax.core.unfreeze(variables["params"])` | Flax `.init()` returns a plain dict since Flax 0.7.1; `flax.core.unfreeze` accepts both |
| `s5/train_helpers.py` | `jax.tree_leaves` -> `jax.tree_util.tree_leaves` | the top-level `jax.tree_*` aliases were removed |

`s5/ssm.py` is otherwise byte-identical to upstream `3c18fdb` apart from those
four annotations; `binary_operator`, both `jax.lax.associative_scan` calls,
`discretize_zoh`, `discretize_bilinear` and the state recurrence are unchanged.

## Verified environment

macOS arm64, Python 3.14.4, CPU backend.

```
jax==0.11.1        flax==0.12.9      torch==2.13.0        einops==0.8.2
jaxlib==0.11.1     optax==0.2.8      torchvision==0.28.0  datasets==5.0.1
numpy==2.5.2       scipy==1.18.1     torchaudio==2.11.0   wandb==0.29.0
```

Pinned in `requirements_dev.txt` (full) and `requirements_test.txt` (unit
tests only). `torchaudio` is required even for the MNIST-only path because
`s5/dataloaders/__init__.py` imports `audio` eagerly.

## Verification

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_dev.txt
python -m pytest tests/ -q
```

Result: **13 passed**. The suite covers, on the real S5SSM (no stubs):

1. real model import and construction (`test_real_model_imports_and_constructs`)
2. real S5 forward pass, plus `apply_ssm` shape/realness
   (`test_real_s5_forward_pass`, `test_ssm_layer_output_is_real_and_correct_shape`)
   and a check that the parallel scan reproduces the sequential recurrence
   `x_t = Lambda_bar x_{t-1} + B_bar u_t` (`test_associative_scan_matches_sequential_recurrence`),
   plus associativity of `binary_operator`
3. JIT compilation matches eager (`test_jit_compiles_and_matches_eager`)
4. gradients through the associative scan reach `Lambda_re`, `Lambda_im`,
   `log_step`, `C`, `D` and are finite and non-zero
   (`test_gradients_flow_through_associative_scan`, `test_gradient_wrt_inputs_is_finite`)
5. one optimizer update strictly decreases the loss, and the repository's own
   `create_train_state` + `train_step` plumbing runs
   (`test_single_optimizer_update_decreases_loss`, `test_create_train_state_and_train_step`)
6. bidirectional S5 still runs, and both `zoh` and `bilinear` discretizations run

## One-epoch sequential-MNIST training smoke test

Exact command (also wrapped by `bin/run_experiments/run_baseline_mnist_smoke.sh`):

```bash
python run_train.py \
  --dataset=mnist-classification --epochs=1 --bsz=64 \
  --n_layers=2 --d_model=64 --ssm_size_base=64 --blocks=2 \
  --batchnorm=False --bidirectional=False --p_dropout=0.0 \
  --jax_seed=1919 --USE_WANDB=False
```

| Metric | Value |
|---|---|
| Trainable parameters | 26,058 |
| Train loss | 1.40866 |
| Val loss | 0.36526 |
| Val accuracy | 0.8928 |
| Test loss | 0.34703 |
| **Test accuracy** | **0.8995** |
| Wall clock | 1m57.32s (528s user, 461% CPU) |
| Peak GPU memory | n/a - CPU-only host, no GPU present |

Registry ID `E2-001`. Evidence label **E2 - smoke evidence only**.

`--p_dropout=0.0` is deliberate: it removes the dropout RNG stream so this run
can be compared exactly against the prospective branch at `alpha=0`.

## GPU baseline (`E2-002`)

The same command and seed on the target cluster GPU. Full provenance - commit,
environment, hardware, seed, command, artifact path and evidence label - is in
[EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md#e2-002--plain-s5-gpu-baseline-smoke).

Host `pgi15-gpu3`, NVIDIA RTX 3090 (24576 MiB, cc 8.6), driver 570.86.10 /
CUDA 12.8, `SLURM_JOB_ID=63277`, commit
`6fcbca798a93e7b511d8862cae4cce4afa43c34f`, clean tree, seed 1919, 1 epoch.

| Metric | Value |
|---|---|
| Baseline unit tests | 13/13 passed on the RTX 3090 |
| Trainable parameters | 26,058 |
| Train loss | 1.40519 |
| Val loss | 0.36402 |
| Val accuracy | 0.8952 |
| Test loss | 0.33766 |
| **Test accuracy** | **0.8998** |
| Wall clock | 56.431 s |
| Peak GPU memory | 837 MiB (of 24576 MiB) |
| Peak sampled GPU utilization | 42% |
| Mean GPU utilization | invalid - not reported (sampler ran past job end) |

Artifacts: `/Users/durso/s5-runs/20260902-154442-main-6fcbca7/`.

Evidence label **E2 - smoke evidence only**: this shows the pipeline runs and
produces plausible numbers on the target hardware. It is not a benchmark and
not a comparison.

The parameter count matches the CPU run exactly; the metrics agree to about
three decimal places without being identical. **The cause of that residual
difference is not established** - reduced-precision matmul modes, kernel and
reduction order, and fusion decisions can all differ across backends, and this
run does not discriminate between them. See `docs/GATES.md` for a test that
would.
