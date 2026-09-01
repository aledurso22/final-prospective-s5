# prospective-lead: verification runs

Same host, same environment and the same command as the plain-S5 baseline in
[BASELINE.md](BASELINE.md), with `--prospective_*` flags appended. Seed, batch
size, model size, data order and hardware are identical across all three rows.

```bash
python run_train.py \
  --dataset=mnist-classification --epochs=1 --bsz=64 \
  --n_layers=2 --d_model=64 --ssm_size_base=64 --blocks=2 \
  --batchnorm=False --bidirectional=False --p_dropout=0.0 \
  --jax_seed=1919 --USE_WANDB=False \
  [--prospective_mode=lead --prospective_alpha=A --prospective_layers=all]
```

| Run | Params | Train loss | Val loss | Val acc | Test loss | Test acc | Wall clock |
|---|---|---|---|---|---|---|---|
| plain S5 (`main`) | 26,058 | 1.40866 | 0.36526 | 0.8928 | 0.34703 | 0.8995 | 1m57.32s |
| lead, fixed `alpha=0`, `all` | 26,058 | 1.40866 | 0.36526 | 0.8928 | 0.34703 | 0.8995 | 1m40.60s |
| lead, fixed `alpha=0.25`, `all` | 26,058 | 1.40177 | 0.34619 | 0.8993 | 0.32091 | 0.9078 | 1m38.16s |

Peak GPU memory: **not measured** - this host is CPU-only (macOS arm64, JAX CPU
backend). It must be re-measured on the cluster.

## alpha = 0 identity

The `alpha=0` row is **bit-identical** to plain S5 on every reported metric, not
merely close. This is by construction, not luck:

- with `prospective_alpha_learned=False` the operator calls no `self.param`, so
  the parameter tree and the PRNG consumption are unchanged (both runs report
  26,058 trainable parameters);
- `x + 0.0 * (x - previous)` is exact in floating point for finite `x`;
- `--p_dropout=0.0` removes the dropout RNG stream.

No nondeterminism had to be excused. Wall-clock differs (1m57s vs 1m41s) because
of ordinary machine load, not computation - the first run also downloaded MNIST.

## alpha = 0.25

Test accuracy 0.9078 vs 0.8995. **This is not a result.** It is one epoch, one
seed, on a deliberately tiny model, and the difference is well inside seed
noise for sMNIST at this scale. It is reported only as evidence that the
operator trains and does not destabilize S5. Any claim about lag reduction
needs the full sweep, multiple seeds and the lag-specific metrics
(prefix accuracy, time-to-correct), none of which have been run.
