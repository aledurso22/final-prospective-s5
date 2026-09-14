# Predeclared protocol: controlled cue/recall comparison

**Committed before any GPU comparison.** Fixed here so it cannot be adjusted
after seeing results. Every element below — generator, model shapes, metrics,
candidates, seeds, evaluation sets, stopping rule — is independent of any
final-test observation.

Checkout `/Users/alessandrodurso/Documents/final-prospective-s5`, branch
`generalized-prospective-s5`.

## 1. Task (frozen)

`tasks/cue_recall.py`. Length 256; 12 channels (8 payload ±1 fresh at every
token, 2 current ±1, cue flag, query flag); cue ~ U{8..31}; delay bucket
uniform over `[16,31] [32,63] [64,95] [96,128]`, delay uniform in bucket;
query = cue + delay (max 159 < 256). Recall target = the 8 cue bits, scored
**only at the query token**. Current target = XOR of the two current bits,
scored at **every** token. Labels and masks are never inputs.

```
L = L_recall + L_current          (separately averaged BCEs)
```

RNG domains `TRAIN=0, DEV=1, TEST=2, EXTRAP=3` folded into the root key.
Evaluation batches are materialized once and frozen before model selection.

**Extrapolation split** (length 512, delays 192/256/384) is **never** used to
select settings or to decide which modes to intervene on.

## 2. Model shape (frozen)

Two layers, `d_model = 32`, `ssm_size_base = 32`, HiPPO `blocks = 2`
(initialization blocks, **not** response blocks), conjugate symmetry,
unidirectional ZOH, unit sample clock, **`clip_eigs=True` on every comparison
arm**, LayerNorm, no dropout. `StackedEncoderModel` plus two token-wise Dense
heads (8 recall logits, 1 current logit). No pooling, no temporal batchnorm, no
future tokens, no teacher forcing.

Resulting shapes: 16 stored complex modes per layer (32 real state
coordinates), `local_P = 32`.

| arm | total params | temporal params |
|---|---|---|
| `plain` | 7,209 | 48 (3P) |
| `gp_diagonal` | 7,241 | 64 (4P) |
| `modal_ssm` | 7,241 | 64 (4P) |

`gp_diagonal` and `modal_ssm` are **exactly parameter-matched**.

## 3. Arms

**Primary** (full development budget): `plain`, `gp_diagonal`, `modal_ssm`.

**Supporting** (one configuration each, using the development-selected learning
rate of the nearest primary arm): `gp_scalar`; scalar `prospective_input`;
`gp_diagonal` with the response **frozen** at its selected initial value while
all ordinary parameters train.

**Sanity / negative**: `full_state_pc`, checked structurally, not swept — its
zero-history token-wise response is memoryless, so payload recall must be at
chance. Verified without pooling across time
(`test_full_state_pc_is_memoryless_so_recall_must_be_at_chance`). A same-token
control for the absence of a static shortcut is
`test_no_history_shortcut_from_the_query_token`.

Scalar `prospective_input` has the **same response-parameter count** as
`gp_scalar` and is compared against it. It is **not** capacity-matched to
`gp_diagonal`, and will not be presented as if it were. `prospective_input`
remains an ingredient/placement control, **not** a reproduction of the
preprint's full alpha-P-S5 recipe.

## 4. Development budget (predeclared)

**At most four configurations per trainable primary arm, one development seed
(0), 750 updates each.** Selection by **development joint BCE**. Equal budget
for every primary arm; no arm receives extra trials after results are seen.

| arm | candidate 1 | candidate 2 | candidate 3 | candidate 4 |
|---|---|---|---|---|
| `plain` | lr/ssm 1e-3/1e-3 | 3e-3/1e-3 | 3e-3/3e-3 | 1e-2/3e-3 |
| `modal_ssm` | 1e-3/1e-3 | 3e-3/1e-3 | 3e-3/3e-3 | 1e-2/3e-3 |
| `gp_diagonal` | 1e-3/1e-3, t0 0.05 | 3e-3/1e-3, t0 0.05 | 1e-3/1e-3, t0 0.20 | 3e-3/1e-3, t0 0.20 |

`gp_diagonal` spends half its budget on the response initialization, because
the brief requires response initialization to be compared **within this task**
rather than via a reassuring sMNIST sweep. That is a deliberate trade: it
explores two learning rates where the other arms explore four. Stated here in
advance, not rationalized afterwards.

## 5. Confirmation runs (predeclared)

Five fresh paired seeds (1..5), **2,000 updates**, batch 64, validation every
100 updates. Select the checkpoint with the lowest joint **validation** BCE.
Identical data schedules and evaluation sets across corresponding seeds for
every arm. Five seeds are **screening evidence, not a universal claim**.

## 6. Metrics — all recorded, not only the flattering ones

Primary declared measure: **held-out joint BCE**. Also recorded: recall BCE,
recall bit accuracy, exact-8-bit accuracy, current BCE, current accuracy, and
**recall bit accuracy per delay bucket**.

Also logged per run: initial and final response values, gradient/update norms,
effective poles, `t_i |a_i|`, stability/clipping statistics, parameter and
state counts, wall time, peak memory, backend, dtypes, matmul precision and
XLA flags.

## 7. Numerical protocol

**Highest matmul precision on both arms of this experiment**
(`--matmul_precision highest`), with the existing deterministic XLA protocol
(`XLA_FLAGS=--xla_gpu_deterministic_ops=true`). The historical smoke keeps its
original backend-default precision, unchanged.

## 8. Interpretation rules, fixed in advance

- Slower poles or a larger learned `t` are **not** a memory measure. Any claim
  about learned memory allocation uses validation-defined interventions and
  actual delayed input-output sensitivity.
- The frozen-response control is stronger evidence about whether `t` is
  learned than a post-training permutation alone.
- A difference smaller than the recorded precision deviation is not
  attributable to the mathematics.
- Unfavourable outcomes are preserved with the same care as favourable ones.
- **Clipping is logged, never inferred.** Every run records the number and
  fraction of raw poles the `clip_eigs` guard moves, per layer, initially and
  finally. A pre-registered observation: in a two-update local check
  `modal_ssm` already clipped 1 of 16 modes in layer 1 while the GP arms
  clipped none. If that asymmetry persists, it is reported **with** any loss
  difference, not discovered afterwards.
- Diagnostics must read the **executed** coefficients (post-clip), not raw
  parameters. Enforced by
  `test_spectral_stats_describe_the_system_that_actually_ran`.
