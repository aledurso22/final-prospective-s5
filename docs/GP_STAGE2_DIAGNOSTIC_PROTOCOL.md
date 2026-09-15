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
| JVP/VJP vs central finite difference | 1e-3 relative, **gated at step 1e-2**, step 1e-3 also reported |
| frequency response vs finite DFT **plus exact remainder** | 1e-8 (identity, not an estimate) |
| future-input sensitivity, inference mode | 0 exactly |
| source file hashes before vs after | byte-identical |

A failed check is reported. If restored counts differ from saved, the
discrepancy is resolved as a loading/dtype/data problem **before** any
performance explanation is offered.

**Amendment made after local-fixture inspection and BEFORE any Stage 2
checkpoint analysis, with its reason.** The finite-difference
check was first written with a unit-L2 direction and a single step of 1e-3. On
a `(32, 161, 20)` input that moves each element by only ~2.5e-6, about 20x
float32 epsilon, so the central difference measured ROUNDING rather than the
derivative and disagreed with the JVP by 0.4 to 2.9 relative. The direction is
now **RMS-1** and two steps are evaluated. Measured on the local fixture after
the fix: rel error 2.6e-3 to 2.3e-2 at step 1e-3, and **4.8e-5 to 7.7e-4 at
step 1e-2**, for all five arms. The gate therefore uses step 1e-2, where
rounding no longer dominates, and both steps are reported. The tolerance itself
is unchanged at 1e-3; what changed is a defective measurement, and it was
changed before any Stage 2 checkpoint was analysed, not after seeing a
diagnostic result. The measurements quoted above come from a LOCAL synthetic
fixture (see section 15), not from the cluster.

## 4. Truncation is declared, never assumed

At the production clock `Delta` can be as small as 1e-3, giving decay times
over 2000 frames, so the 512-lag window is **not** the whole response.

**AMENDED after coordinator review (R2).** An earlier version of this protocol
reported a geometric tail bound `||K_last|| * rho/(1 - rho)`. **That bound is
invalid and has been removed, not loosened.** It is not a general bound on a
multimode output tail: output modes can cancel exactly at the last measured
sample and not at the next, and a nonnormal block need not contract in
Euclidean norm at its spectral radius. The analytical counterexample, which
needs no experiment, is the scalar two-mode response

    K_l = (1/2)^l - 2 (1/4)^l ,   K_1 = 0 but K_2 = 1/8

for which a two-sample window reports a zero tail while the true tail is
nonzero. It is kept as an executable test.

What replaces it:

* **Lag-band energies are explicitly windowed at 0..511 and the remainder is
  marked `unknown (not bounded)`.** No windowed statistic is used to call
  memory negligible.
* **The frequency comparison uses the EXACT finite-window remainder** computed
  from the resolvent, so `H(w) = finite DFT + remainder` is an identity rather
  than an approximation. For a one-tap realization `K_l = C A^l B` the
  remainder after `l = 0..N-1` is `C (uA)^N (I - uA)^-1 B`, `u = e^{-iw}`;
  native `D` has no tail; the two-tap lag-one drive `A B_plus + B_minus` and
  its index shift are handled explicitly; complex pairs are realified exactly
  as in the frequency adapter. Verified on the nonnormal mass blocks too.
* `spectral_radius` is retained as a **descriptive** quantity only, and is
  never used as a tolerance in the frequency gate.

Band fractions are reported **alongside absolute energies**, because fractions
alone can hide a collapse in total response. No monotonic-Hankel claim is made,
and "longer poles" is never treated as "better useful memory".

## 4a. Poles: trained, executed, continuous and discrete (R1)

* The **trained raw pole** is the parameter `Lambda_re + i Lambda_im`.
  `Lambda_re_init` / `Lambda_im_init` are the static **initializer fields**;
  an earlier version read those, so trained clipping counts and
  raw-versus-clipped shifts could be wrong even where the executed impulse
  response was right. Fixed, with a test that moves both raw parts away from
  initialization, including a real part above the clipping boundary.
* **Continuous poles are exported directly, never as `log` of a discrete
  eigenvalue**, which returns a principal-branch value and therefore ALIASES
  any continuous frequency above pi per unit interval. Sources: `a` for the
  one/two-tap arms, `a_eff` for `M = 0`, and eigenvalues of the executed
  continuous block generator for the mass arm. Discrete magnitude and angle are
  reported separately and the angle is labelled an aliased frequency; a test
  uses a continuous frequency beyond pi to prevent relabelling.

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

## 12. Enforcement: a failed check fails the run (R3)

Every check is persisted the moment it is taken. A failed **required** check —
restoration counts or CE, the adapter-versus-executed-core comparison, the
derivative check, nonzero future sensitivity, or a changed source file —
**blocks dependent interpretation for that arm and forces a nonzero exit**. An
unavailable optional phase is recorded as `not_executed` and is distinguished
from a failure.

Final status and exit codes: `PASS` 0, `INCOMPLETE` 3, `FAILED` 4. A
budget-incomplete run is never reported as a pass. On an error, interrupt or
timeout, a `finally` path preserves results and re-hashes the sources; if that
cannot complete, source invariance is reported **unknown**, never true.

The adapter comparison covers **every reported layer and every input
coordinate**, batched; the checked scope is recorded in the result. An earlier
version checked layer 0 and six coordinates while reporting four layers.

## 13. Budget enforcement (R4)

**One deadline covers backend verification, the focused numerical tests and the
diagnostic.** The launcher computes it first, verifies the GPU before any
numerical fixture, and passes the absolute deadline down. An outer
process-group watchdog stops an overrunning phase, with a cleanup reserve kept
inside the cap; the result is reported as `INCOMPLETE (timeout)`, not as a
pass. There is no CPU fallback.

Each invocation is a **fresh rerun** into a new timestamped directory. It is
NOT a resumption, and the earlier claim that "completed phases are not
restarted" has been withdrawn. To run only genuinely missing phases, pass
`ONLY_PHASES=...`. Test logs are unique per invocation.

## 14. Implemented versus absent (R5)

| item | status |
|---|---|
| reconstructed-initialization **cores** | implemented (phase `C_init`), labelled as reconstructed, not a saved checkpoint |
| initialization **gradient** probes (D4) | **not implemented**; deferred by the priority order and recorded as such in the run summary |
| per-layer activation RMS | implemented |
| gradients at recurrent-core **inputs** and at the pre-pooling activation | implemented, by differentiating additive zero offsets injected through the bound submodules |
| last-vs-best checkpoint comparison | implemented, lightweight pole summary only, and only when best is not last |
| dtype record | implemented; actual parameter and batch-statistic dtypes, with `cast_applied: false` |
| schedule metadata | implemented; steps-per-epoch read from **saved run metadata**, not hardcoded |

Parameter-gradient norms alone do **not** establish depth attenuation or
gradient quality, and no conclusion resting on the unimplemented item is drawn.
"Core" is used throughout for the linear recurrent core; the superposition
check validates that core, not the whole nonlinear residual `SequenceLayer`.

## 15. Provenance of the finite-difference amendment

The FD direction/step amendment in section 3 was made **after inspecting a
local synthetic fixture and before any Stage 2 checkpoint analysis**. It was
not made after seeing a Stage 2 diagnostic result. The 36-second local fixture
run that informed it was a **local fixture result**, outside the brief's
no-local-numerical-execution scope; it is recorded as such and is not relabelled
as a cluster check. All numerical validation from this revision onward runs on
the cluster, inside the launcher.

## 16. Deliverable

A diagnosis. An improved score is neither required nor sought, and no
experiment is authorized by this protocol.
