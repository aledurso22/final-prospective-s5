# Rawat alpha-P-S5: reference map

Source: **Rawat, Morello, Morone, Heeger, "Prospective Coding Improves
Learning in Deep Continuous-Time Recurrent Networks", arXiv:2609.04134v1
[cs.LG], 3 September 2026.** Appendices E.3 and E.4, and Table 6.
Retrieved 15 September 2026 from `https://arxiv.org/html/2609.04134v1`.

## 1. Reference code status: UNAVAILABLE

| item | status |
|---|---|
| `https://github.com/Sequel-Institute/prospective-rqf` | **404, repository not found** (`git ls-remote`, 15 Sep 2026) |
| exact reference commit | **none — cannot be pinned** |
| paper HTML | retrieved, 669,476 bytes, appendix text quoted verbatim below |

**This port is therefore PAPER-BASED.** Every setting below is either quoted
from the paper or marked as a resolved local decision. No claim of code-level
equality with the authors' implementation is made anywhere.

## 2. The construction, quoted

Appendix E.3, verbatim:

> "We define alpha_p = -Re(lambda_p) > 0 and set **B_c = diag(alpha) B_tilde**
> in the native clock. The corresponding physical input row is
> B_eff,p: = (Delta_p/h) B_c,p: = -Re(lambda_eff,p) B_tilde_p:, so the alpha
> gain remains tied to the physical pole decay rate and is not learned
> separately."

> "Prospective input uses the same shared physical horizon as the RQF
> experiments, **tau = 5h**. ... the factor h cancels when the update is
> expressed in S5's native clock"

Equations (E.11):

```
A_bar   = diag(exp(lambda * Delta))
B_bar   = Lambda^-1 (A_bar - I) B_c
B_plus  = B_bar + 5 diag(Delta) A_bar B_c
B_minus =       - 5 diag(Delta) A_bar B_c
h_{t+1} = A_bar h_t + B_plus x_{t+1} + B_minus x_t
```

> "The delayed input starts at zero, and the direct D (*) x_t feedthrough
> remains instantaneous. Within alpha-P-S5, the two-tap operation adds no
> trainable parameters and leaves the recurrent transition, residual path,
> output projection, and scan structure unchanged."

> "In the reference alpha-P-S5 configuration, **real pole parts are clipped to
> at most -10^-4 before forming alpha**, whereas the native one-tap S5
> configuration does not apply this clipping and uses the unscaled learned
> input matrix. Tables 2 and 6 therefore compare the complete alpha-P-S5
> construction with native S5; **they do not isolate the second tap while
> holding the continuous-time input map and pole clipping fixed.**"

That last sentence is why `gain_clip_s5` exists as a separate arm: the paper
itself states the published comparison confounds three changes.

## 3. Settings: paper value, local code location, verified or declared

| setting | paper | local | status |
|---|---|---|---|
| dataset | SC v0.02, 10-word subset | `dataloaders/speech_commands10.py:WORDS` | matches |
| split | stratified 70/15/15, split seed 0 | `build_splits` | value matches; **algorithm declared** (see 4) |
| standardization | feature-wise, from training split | `prepare` | matches |
| MFCC | 20 coefficients, 200-sample FFT window, 64 mel bands, 100-sample hop, 161 frames | `N_MFCC/N_FFT/N_MELS/HOP` | matches; frame count asserted per file |
| augmentation | none | — | matches |
| input projection | real, d_in=20 -> n=32 | `RawatClassifier.encoder` | matches |
| depth | L = 4 (Table 6 row used) | `--n_layers 4` | matches |
| width | n = 32 | `--d_model 32 --ssm_size 32` | matches |
| HiPPO blocks | eight | `HIPPO_BLOCKS = 8` | matches |
| conjugate symmetry | yes | `conj_sym=True` | matches |
| discretization | ZOH | `discretization="zoh"` | matches |
| direction | unidirectional | `bidirectional=False`, enforced by a raise | matches |
| normalization | batch **pre**-normalization | `prenorm=True, batchnorm=True` | matches |
| pointwise map | half-GLU | `activation="half_glu1"` | **declared** (see 4) |
| readout | norm -> 64 -> mean-pool -> 64->256->10 GELU MLP, dropout 0.1 | `RawatClassifier` | matches; dropout placement **declared** (see 4) |
| init B_tilde, C_tilde | LeCun normal, std 1/sqrt(P_local) | upstream `trunc_standard_normal` + `init_CV` | **declared** (see 4) |
| init D | N(0, 1) | upstream `normal(stddev=1.0)` | matches |
| init log steps | Delta in [1e-3, 1e-1] | `dt_min=0.001, dt_max=0.1` | matches |
| optimizer | AdamW | `optax.adamw` | matches |
| batch size | 32 | `--batch 32` | matches |
| learning rate | 1e-3, all S5 params incl. Lambda, B_tilde, C_tilde, log steps | `--lr 1e-3`, single group | matches |
| weight decay | 1e-4, **all** parameters | `--weight_decay 1e-4`, single group | matches; **differs from upstream S5** (see 5) |
| gradient clipping | global norm 1.0 | `--grad_clip 1.0` | matches |
| loss | cross entropy, label smoothing 0.1, mean-pooled logits | `--label_smoothing 0.1` | matches |
| schedule | cosine annealing to 1e-6, max 300 epochs | `--lr_final 1e-6 --epochs 300` | matches |
| early stopping | 20 epochs without improved validation accuracy | `--patience 20` | matches |
| checkpointing | one per epoch; test read from highest-validation-accuracy checkpoint | atomic `best`/`last`, `--mode evaluate` | matches |
| seeds | 0-4 (five) | protocol uses 0,1,2 | **fewer, stated as preliminary** |

## 4. Declared local decisions (paper does not specify)

1. **half-GLU variant.** `half_glu1` vs `half_glu2` is not stated. We use
   `half_glu1`, the repository default. Identical on every arm.
2. **MLP dropout placement.** "64 -> 256 -> 10 GELU MLP with dropout 0.1" does
   not fix where dropout sits. We apply GELU then dropout then the output
   layer, and use no block dropout. Identical on every arm.
3. **Stratification algorithm.** The paper gives the 70/15/15 proportions and
   split seed 0, not the procedure. We shuffle each class's sorted file list
   with `numpy.random.RandomState(0)` and cut. The manifest stores a SHA-256 of
   the resulting file list per split, so the exact split is auditable.
4. **B_tilde / C_tilde initializer.** The paper says LeCun normal with standard
   deviation 1/sqrt(P_local); upstream S5's `trunc_standard_normal` is a
   truncated variant at the same scale. We keep upstream's, unchanged on every
   arm.

Each of these affects how closely the ABSOLUTE published number can be
reproduced. None of them differs between arms, so none affects the comparison
the experiment is actually built to make.

## 5. Two findings recorded because they affect interpretation

**Weight decay on SSM parameters.** The paper applies weight decay 1e-4 to all
S5 parameters including `Lambda` and the log steps. Upstream S5 deliberately
excludes SSM parameters from weight decay. We follow the paper, because this is
a reproduction, and expose `--ssm_weight_decay` for a separately labelled run
under the upstream convention. The predeclared protocol does not use it.

**The published MFCC configuration leaves empty mel filters.** With
`n_fft = 200` the spectrum has 101 bins, and 64 mel bands over 101 bins is
degenerate: measured with torchaudio, **62 of 64 filters have any nonzero
weight; 2 are empty**. Output is finite and the frame count is exactly 161 as
published. This is a property of the published configuration, reproduced rather
than silently corrected.

**The split is not speaker-disjoint.** Speech Commands ships official
`validation_list.txt` / `testing_list.txt` that are speaker-disjoint. A
stratified random file split is not: the same speaker can appear in train and
test. We reproduce the published protocol as stated, but the absolute accuracy
is therefore not comparable with official-split literature. Every arm shares
the split, so the between-arm comparison is unaffected.

## 6. Published reference numbers (Table 6, depth 4, MFCC, width 32)

| arm | accuracy | params |
|---|---|---|
| alpha-P-S5 | **96.31 +/- 0.32** | 35.1k |
| native S5 | **95.84 +/- 0.32** | 35.1k |

Mean +/- sample standard deviation over five seeds, full BPTT.

**Our reconstruction gives 35,050 parameters** for the same configuration,
consistent with the reported 35.1k. That is an independent check on the
architecture reconstruction, not on training.

These are **reference measurements, not an acceptance threshold** for the short
bounded run in `docs/GP_RAWAT_CLUSTER_PROTOCOL.md`. A 10-epoch development run
is not a 300-epoch five-seed result and will not be presented as one.
