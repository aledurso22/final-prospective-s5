# Stage 2 diagnostic protocol

**Frozen before any data-dependent diagnostic is run.** Governed by
`docs/handoff_2026_09_15_diagnostic/CODING_AGENT_BRIEF.md`.

This is analysis of saved models. **No training, no optimizer update, no
resume, no new seed, no hyperparameter or physical-coefficient change, no test
scoring, no architecture or readout change.** Derivative evaluation without a
parameter update is authorized and used.

The Stage 2 verdict stands: the screen FAILED. Nothing here revises it.

Base: branch `stage2-diagnostic` from
`227648e6d64a9b786b59e4bbfb64bfa8cda8d6ff`.

## 1. Read-only discipline

* Source runs under `/Users/durso/s5-runs/stage2/` are **never written**. All
  output goes to `/Users/durso/s5-runs/stage2-diagnostics/<run-id>/`.
* SHA-256 of every source metrics file, checkpoint, checkpoint metadata and
  config is recorded **before and after** the diagnostic and compared. Any
  difference fails the run.
* `experiments/gp/rawat_benchmark.py` is **NOT invoked on original run
  directories**: `write_config` opens `config.json` with mode `"w"` and would
  truncate the original provenance, and `append_metrics` would append
  evaluation records to the original metrics file. Verified by reading the
  source, not assumed.
* `s5/gp_diagnostics.py:core_from_module` is **not used**. It dispatches on a
  `mechanism` attribute that `SubstrateSSM` does not define, and a dispatch
  miss there silently returns inherited native coefficients — it would report
  the wrong dynamics for four of five arms. `s5/substrate_diagnostics.py` is
  the dedicated adapter, and every quantity it produces is checked against an
  actual forward impulse call before interpretation.
* Only the **train and validation** splits are opened, through
  `SC.load_splits`. The test arrays are never opened by this diagnostic. That
  is a stronger statement than "test not scored"; both are kept distinct, and
  merely opening a file would not constitute training on it.

## 2. Frozen defaults

| item | value |
|---|---|
| primary models | the five Stage 2 `best` checkpoints selected at lr 1e-3 |
| layers analysed | all four, per arm |
| impulse lags | 0..511 |
| lag bands | `0`; `1-4`; `5-16`; `17-64`; `65-160`; `161-511` |
| frequency grid | 129 points, 0 to Nyquist inclusive |
| gradient subset | 32 TRAINING examples, permutation seed **20260915** |
| VJP directions | 4 unit-normalized Rademacher, seed **20260916** |
| anchor frames | 80 and 160 |
| primary mode | inference: saved batch statistics fixed, dropout disabled |
| GPU budget | **20 minutes wall, including compilation**, one worker |

The gradient subset is chosen by a fixed permutation of the training split,
independent of losses and labels. It is a **post-screen diagnostic sample**,
not an independent confirmation sample.

## 3. Declared tolerances

Scoped to the recorded dtype (float32 execution, complex64 coefficients), and
consistent with the production tolerances already declared in
`tests/cluster_float32_probe.py`.

| check | tolerance |
|---|---|
| restored validation **correct count** vs saved | **exact integer equality** |
| restored validation cross entropy vs saved | 1e-5 absolute |
| adapter impulse vs executed forward (float64 fixture) | 1e-10 |
| adapter impulse vs executed forward (float32 checkpoint) | 1e-4 relative |
| frequency response vs impulse DFT | max(1e-6, 10x geometric tail bound) |
| JVP/VJP vs central finite difference | 1e-3 relative, **gated at step 1e-2**, step 1e-3 also reported |
| future-input sensitivity, inference mode | 0 exactly |
| source file hashes before vs after | byte-identical |

A failed check is reported. If restored counts differ from saved, the
discrepancy is resolved as a loading/dtype/data problem **before** any
performance explanation is offered.

**Amendment made before execution, with its reason.** The finite-difference
check was first written with a unit-L2 direction and a single step of 1e-3. On
a `(32, 161, 20)` input that moves each element by only ~2.5e-6, about 20x
float32 epsilon, so the central difference measured ROUNDING rather than the
derivative and disagreed with the JVP by 0.4 to 2.9 relative. The direction is
now **RMS-1** and two steps are evaluated. Measured on the local fixture after
the fix: rel error 2.6e-3 to 2.3e-2 at step 1e-3, and **4.8e-5 to 7.7e-4 at
step 1e-2**, for all five arms. The gate therefore uses step 1e-2, where
rounding no longer dominates, and both steps are reported. The tolerance itself
is unchanged at 1e-3; what changed is a defective measurement, and it was
changed before any cluster execution, not after seeing a diagnostic result.

## 4. Truncation is declared, never assumed

At the production clock `Delta` can be as small as 1e-3, giving decay times
over 2000 frames, so the 512-lag window is **not** the whole response. Every
windowed statistic is reported with the geometric tail bound
`||K_511|| * rho/(1 - rho)`, `rho = max |discrete pole|`. Band fractions are
reported **alongside absolute energies**, because fractions alone can hide a
collapse in total response.

No monotonic-Hankel claim is made, and "longer poles" is never treated as
"better useful memory".

## 5. Frequency response: the specific trap

For a conjugate-pair realization with real input,
`H(w) = T(e^{-iw}) + conj(T(e^{+iw}))`, **not** `2 Re(T(e^{-iw}))`. Taking
`2 Re` of a complex-frequency response would be wrong, and a regression test
(`test_two_re_of_the_complex_transfer_is_NOT_the_real_response`) pins the
difference so a refactor cannot reintroduce it.

## 6. Within-checkpoint counterfactual

For each alpha-P and GP `best` checkpoint, hold `Lambda`, `Delta`, `B_tilde`,
`C_tilde`, `D`, the input gain and the clipping exactly as trained and evaluate
the **one-tap** law's linear response beside the executed response. This is an
**untrained counterfactual response calculation**: not a new accuracy score,
not a trained baseline. It separates the direct effect of the equation from
coadaptation of the learned weights.

## 7. Pre-pooling probe

The probe target is the **final encoder output before temporal pooling**, shape
`(L, d_model)`. Pooled logits depend directly on every earlier output, so
sensitivity of pooled logits to early inputs would **not** by itself
demonstrate recurrent memory; the pre-pooling probe is what makes the
distinction.

In inference mode the batched model and an unbatched model under `vmap` are
verified to agree exactly, so the probe uses the unbatched module without
changing what is measured. These probes measure **local sensitivity**, not
delayed-recall accuracy or semantic usefulness.

One additional gradient pass per best checkpoint may use training-mode
normalization with a fixed dropout RNG, discarding all mutable updates. It is
labelled separately: training BatchNorm couples samples and times, so it is
not a purely causal recurrence-gradient diagnostic.

## 8. Schedule description, corrected

Stage 2 passed `epochs=10`, and `make_optimizer` uses
`steps_per_epoch * epochs` as the cosine decay duration. Stage 2 is therefore a
**ten-epoch cosine schedule**, NOT a ten-epoch prefix of the paper's 300-epoch
schedule. The two definitions may be plotted analytically for illustration. The
alternative schedule is **not run**, and no claim is made that the schedule
caused the ranking.

Also corrected in reporting: the logged `train_loss` is **label-smoothed** and
validation CE is **unsmoothed**, so their difference is not a comparable
generalization gap; and the logged `grad_norm` is the **last minibatch** value,
not an epoch average and not a record of clipping frequency.

`epoch_s` is captured **before** the checkpoint writes, so it excludes
checkpoint I/O.

## 9. Cost statements, kept separate

The ~1.93x figure is a **stage-1 two-step microbenchmark**. The ~2.5x figure is
the **measured end-to-end epoch ratio**. They are different measurements and
are never merged into one number.

## 10. Priority under the 20-minute cap

Phases run in this order, with a deadline check between each. If time runs
short, work is dropped from the END of this list, and whatever is incomplete is
recorded as incomplete rather than silently omitted:

1. **A** inventory, hashes, read-only restore
2. **B** screen reconstruction and restoration corroboration
3. **C** core extraction, poles, impulse, bands, frequency, counterfactual
4. **D1** gradients at the five best checkpoints
5. **D2** pre-pooling sensitivity probes
6. **D3** optional training-mode gradient pass
7. **D4** reconstructed-initialization gradient probes — **first to drop**

The budget is not extended automatically and completed phases are not
restarted.

## 11. Hypotheses to examine, none assumed

The brief's candidate explanations are examined against saved-model evidence:
excessive change to learned temporal filtering; insufficient current-path
benefit over the common `D`/residual substrate; learning curves consistent with
optimization differences (not proof of a schedule cause); normalization or
readout effects, or a concrete implementation defect; or **insufficient
evidence to attribute the gap**, which is a permitted and possibly correct
conclusion.

Continuous-time facts to be examined, not asserted as causes: at `rho = 0.75`
the positive-mass core's high-frequency gain relative to the same ordinary mode
tends to `1/rho = 4/3` while still decaying as `1/p`; and for `M = 0` the pole
map `a_eff = a/(1 - T a)` sends the stable half-plane into the disk of centre
`-1/(2T)` and radius `1/(2T)`, which at `T = 5` bounds effective continuous
imaginary magnitude by 0.1 per frame and the real part within `(-0.2, 0)`.
Whether relevant mode frequencies or residues are actually compressed is a
measurement, not an inference from the bound.

Dense learned-`T` local gains are a **different model family**. No switch to
that family, to arbitrary off-diagonal coefficients, to learned physical
constants or to another readout is made here.

## 12. Deliverable

A diagnosis. An improved score is neither required nor sought, and no
experiment is authorized by this protocol.
