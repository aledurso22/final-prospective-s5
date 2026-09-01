# Normal Prospective S5: first engineering stage

This branch tests the smallest causal prospective modification that preserves
S5's memory dynamics and parallel training implementation.

For an S5 block preactivation `b` produced by the existing associative scan,
the optional operator is

```text
b_pc[t] = b[t] + alpha_l * (b[t] - b[t-1]).
```

At a sequence or packed-segment boundary, `b[t-1]` is defined as `b[t]`, so
the first correction is zero.  The corresponding causal filter is
`P(z) = 1 + alpha * (1 - z^-1)`.  It is applied after the SSM readout and
feedthrough and before the block's activation/gating, dropout, and residual.

## Scope and hypothesis

The hypothesis is that a small first-order lead can reduce representation lag
and make a correct online decision available earlier.  It cannot predict an
unobservable future event, change or improve the S5 poles, or eliminate BPTT
during training.  Training remains normal GPU backpropagation through S5's
parallel scan.  Streaming inference adds one cached `H`-vector and one
subtract/multiply per enabled block.

The S5 recurrence, discretization, `binary_operator`, and
`jax.lax.associative_scan` in `s5/ssm.py` are intentionally unchanged.  The
standalone interface in `s5/prospective.py` provides both:

- `apply_parallel(sequence, alpha, reset_mask=None)`
- `step(token, cache, alpha, reset=False)`

This boundary is deliberately swappable so a future optimized least-action or
boundary-conditioned operator can replace the lead without changing S5's
scan.  Such an extension is not part of this stage.

## Relationship to `main`

This branch is a **pure superset of `main`** (plain, runnable, modern-JAX S5).
Everything about the environment, the modern-JAX compatibility fixes, the
real-S5 tests and the cluster infrastructure lives on `main` and is documented
in [docs/BASELINE.md](docs/BASELINE.md) and [docs/CLUSTER.md](docs/CLUSTER.md).

```bash
git diff main...prospective-lead --stat
```

must show only the prospective feature, its tests, its documentation and its
experiment scripts -- no dependency or compatibility differences.  A future
alternative prospective architecture should normally branch from `main`, so it
is directly comparable with plain S5; only a method that explicitly extends
this lead operator should branch from here.

## Environment

Identical to `main`; no extra dependency is introduced.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_dev.txt
python -m pytest tests/ -q
```

For CUDA 12 / CUDA 13 clusters see [docs/CLUSTER.md](docs/CLUSTER.md).

## Configuration

The training entry point accepts:

```text
--prospective_mode={off,lead}
--prospective_alpha=FLOAT
--prospective_alpha_learned={True,False}
--prospective_layers={all,last}
--prospective_alpha_max=FLOAT
```

`--prospective_mode=off` is the upstream path.  A fixed
`--prospective_alpha=0` is an exact identity.  Learned alpha is one scalar per
enabled block, initialized from `prospective_alpha` and constrained by a
sigmoid to `(0, prospective_alpha_max)`.  Prospective mode rejects
bidirectional S5 because the deployment target is causal online inference.

## First runs

Cheap CPU/GPU plumbing smoke test (downloads MNIST on first use):

```bash
python run_train.py --dataset=mnist-classification --epochs=1 --bsz=32 \
  --n_layers=1 --d_model=32 --ssm_size_base=32 --blocks=1 \
  --batchnorm=False --bidirectional=False \
  --prospective_mode=lead --prospective_alpha=0.25 \
  --prospective_layers=last --USE_WANDB=False
```

The prospective runs reuse `main`'s exact baseline configuration (see
`bin/run_experiments/run_baseline_mnist_smoke.sh`) with `--prospective_*`
flags appended, so `alpha=0` is directly comparable with plain S5 under an
identical seed, batch size, model size, data order and host.

`bin/run_experiments/run_prospective_sweep.sh` prints (dry run) or executes
(`EXECUTE=1`) the full 39-run stage-1 grid: baseline + 5 fixed alphas + learned
alpha, over placements `{last, all}` and 3 seeds.  The single-run scaffold
`bin/run_experiments/run_prospective_mnist.sh`
accepts `MODE`, `ALPHA`, `LEARNED`, `PLACEMENT`, and `SEED` environment
variables.  A cluster job array should cover:

- baseline S5;
- fixed-alpha PC-S5;
- learned-alpha PC-S5;
- diagnostic alpha grid `{0, .125, .25, .5, 1}`;
- placements `last` and `all`, with three seeds.

Track final and prefix accuracy, time-to-correct/prediction lag, throughput,
peak memory, and robustness to input noise.  No GPU result is claimed by this
bootstrap.

## Scientific limitations

The operator is exactly the causal first-order lead filter

```text
P(z) = 1 + alpha * (1 - z^-1).
```

It can only compensate for *predictable* representation lag -- the phase lag a
low-pass SSM readout introduces on signals already present in the state.  It
cannot:

- predict an unknowable future event;
- change or improve the S5 poles (`Lambda_bar` is untouched);
- create longer memory;
- eliminate BPTT during training.

It also amplifies high-frequency content, including input noise, by up to
`1 + 2*alpha`.  Noise robustness is therefore an explicit metric in the sweep,
not an afterthought.

Training remains ordinary GPU backpropagation through S5's parallel scan.
Inference remains causal and recurrent, adding one subtraction, one
multiplication and one cached `H`-vector per enabled block.

## Exact insertion points

- `s5/layers.py::SequenceLayer.__call__`: immediately after `self.seq(x)` and
  before the activation branches.
- `s5/seq_model.py::StackedEncoderModel.setup`: selects all or last block.
- `s5/train.py` and `run_train.py`: model and CLI configuration plumbing.
- `s5/train.py`: `--bidirectional` + `lead` guard, and `prospective_*`
  plumbing into both `model_cls` partials.
- `run_train.py`: the five `--prospective_*` CLI flags.
- `s5/ssm.py`: unchanged relative to `main`, guarded by a source-digest test
  and a `git diff main -- s5/ssm.py` assertion.
- `s5/train_helpers.py`: unchanged relative to `main`.
