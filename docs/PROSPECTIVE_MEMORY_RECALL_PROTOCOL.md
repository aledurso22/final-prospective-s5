# Memory-recall study: protocol

**Committed before numerical execution.** New mechanistic experiment. The
completed Speech Commands studies and their unsuccessful screens are preserved
and unchanged; this does not replace them.

Branch `adaptive-recurrence`, parent `d6c4a222568dcdc93974164c1a8d39015e08aa2e`.

## 1. Permitted claim

A positive result could support: *in a controlled recall task, an S5 using the
physically derived generalized prospective recurrence improves recall under
matched training conditions.* It would **not** establish a general benchmark
advantage, a literal biological implementation of complex S5 feedback, or that
a physical interpretation creates computational capacity. A fixed linear
realization remains an augmented SSM.

The earlier 84.85 % from prospective coding in the recurrence was a mean-pooled
speech classifier and does **not** decompose accuracy into static and memory
parts. This task removes pooling and makes the query input independent of the
answer.

## 2. The equation, with the redundant coordinate removed

The post-run algebra review established that `(Delta, gamma_n, rho)` is
input-output equivalent to `(Delta/gamma_n, 1, rho)` for every admissible
`rho`. `gamma_n` is therefore dropped: with `hat_Delta = Delta/gamma_n` carried
by the existing learned `log_step`,

```
rho T s'' + s' + r + T r' = 0 ,   r = -hat_Delta lambda s - hat_Delta B_c x
```

`gamma_n = 1`, `T = 5` input intervals fixed, **one learned `rho` per stored
complex mode** shared with its conjugate, `rho` bounds `[0.01, 0.9999]`
unchanged, and the mass **derived**: `mu = T rho`. Existing pole clipping and
the common alpha-scaled input map are retained for the matched arms.

Realization unchanged: `rho s' = -r - (1-rho) v`, `T v' = s' - v`, with the
existing block ZOH, associative scan, physical-state readout and native `D`.
Coefficients are constant within a sequence; updates happen between batches.

The positive component map remains available at `gamma_n = 1`:
`kappa_0 = 1.5`, `c_s = c_d = 7.5`, `G_s = G_d = kappa_0/rho`,
`h = G_s sqrt(1-rho)`, `g_L = kappa_0/(1+sqrt(1-rho))`. Applying this response
to learned S5 feedback remains the declared **computational extension**, not an
independent microscopic derivation.

## 3. Task

`tasks/recall.py`. 8 symbols, length 128, query at index 127 carrying **no
symbol payload**. Exactly two marked cues; every other earlier token is an
unmarked distractor from the same alphabet. **Target = the symbol of the latest
marked cue.** Training delays 8, 32, 64 cycled within each batch, which for a
16-pair batch realizes **6/5/5 pairs = 37.5 / 31.25 / 31.25 per cent**, close
to but not exactly equal representation; the runner records the realized mix.
Cue gap from 8/16/24. Sequences are generated in **pairs** sharing distractors and marker
positions with the two marked symbols swapped, so the pair has the same token
multiset and different targets and no bag-of-symbols statistic separates them.

Evaluation: 1,024 examples per delay at 8, 32, 64 and the **held-out** delay
96, from an independent stream shared across arms. **Delay 96 is never trained
on and never used for selection.**

Objective: cross entropy at the final query only. No auxiliary head, no
memory-preservation penalty, no hand-assigned memory neurons.

## 4. Model, identical across arms

Two layers, width 32, base SSM size 32, four HiPPO blocks, conjugate symmetry,
ZOH, no dropout. **Tokenwise LayerNorm in every arm** — no batch or temporal
normalization, no pooling, no bidirectionality, no attention, no readout access
to earlier activations. Shared ordinary nonlinear blocks and residual
connections. State counts are read from the executed modules, not inferred.

## 5. Arms

| arm | what it is |
|---|---|
| `ordinary` | matched ordinary S5, the warm-up model continued |
| `rawat` | the implemented input-side two-tap law, horizon 5 |
| `professor` | prospective coding in the recurrence, `s_k = J^-1 b x_k` |
| `gp_rho` | the generalized recurrence, `rho` learned |
| `gp_rho_frozen` | identical initial function and carry, `rho` frozen |
| `ordinary_2x` | ordinary S5 with **twice** the stored modes |

`professor` forms `-B_c/lambda` **directly**, so `Delta` cancels structurally
and its exact data-loss gradient with respect to `log_step` is **ZERO** — a
property of the model, tested as such. Its failure to optimize is not a
correctness condition.

`ordinary_2x` embeds the warm-up model function-preservingly: the original
modes are copied, the added modes get stable poles and **nonzero input
coupling** with **zero readout coupling**, so the initial function is unchanged
while the added modes retain a trainable readout path. This matches recurrent
state, not every resource; its extra parameters and cost are reported.

## 6. Paired continuation

For each seed 100, 101, 102: train `ordinary` for **1,024 warm-up updates**,
clone its **common** parameters into every arm, continue each for **1,024**
further updates on **identical ordered minibatches**, batch 32, full BPTT,
fixed lr 1e-3, existing AdamW policy, **a fresh optimizer for every arm
including `ordinary`**. Weight decay acts in log coordinates on the `rho` leaf —
an optimizer choice, recorded, not a derived rule.

The shared warm-up cost is reported. **This does not answer which model trains
best from scratch**, and Rawat's mechanism is transplanted here, not reproduced
at its published schedule.

The raw log-response leaf is **projected back into the declared interval after
every optimizer update**, not merely clipped in the forward pass. A forward clip
alone leaves a raw value above the upper bound with `d(rho)/d(eta) = 0` and
therefore **zero task gradient**, and AdamW's decoupled decay shrinks `eta`
toward zero, which is *away* from a negative upper bound — so nothing brings it
back. Optimizer state is preserved; only parameters are projected. Raw values,
executed `rho`, boundary occupancy and projection counts are logged.

`rho` is initialized at **0.9998** — a declared initialization choice, not a
physiological measurement.

**A property of that choice — CORRECTED after the coordinator's review.** In
log space `log(0.9998) = -2.000e-4` sits **1.0e-4 BELOW** the declared upper
bound `log(1 - 1e-4)`. `rho` can therefore **rise as well as fall**, and in the
executed run it mostly rose.

Two statements in an earlier version of this section were wrong and are
withdrawn:

* *"it can decrease, adding mass"* — **reversed**. With `gamma_n = 1` the mass
  is `mu = T rho`, so **decreasing `rho` DECREASES the mass**. What decreasing
  `rho` does is break the `rho = 1` pole-zero cancellation and change the
  observable contribution of the auxiliary dynamics; it is not the same thing
  as increasing memory importance.
* *"`rho` moved can only mean `rho` fell"* — false, for the reason above.

`rho = 1` is the **physical** family boundary; `0.9999` is a **chosen numerical
margin** below it, not a physical limit.

It also makes a finite-difference gradient check at the initialization point
invalid: a usable float32 step of 1e-2 is 100x the available headroom, so the
`+h` evaluation saturates and a central difference returns about half the true
directional derivative. The float32 gradient check is therefore evaluated at an
**interior** `rho = 0.75`, and saturation at the ceiling is checked separately
as its own property.

## 7. Initialization-response gate, before any comparative score

Near-one `rho` does not guarantee functional closeness for high-Q modes, so it
is **measured**, not assumed. Predeclared norm, per layer, on the **executed**
modules:

* **signal-only** core impulse response over lags 0..127, with the native `D`
  contribution **removed**, relative Frobenius
  `||K_arm - K_ord||_F / ||K_ord||_F`;
* the gate runs **before every seed's continuations**, since each warm-up has
  different learned poles and readouts, and **both** the impulse and frequency
  criteria are enforced, with non-finite values failing;
* the same on a 65-point frequency grid;
* query-logit change on unlabeled initialization probes.

**Zero-reference handling:** if `||K_ord||_F <= 1e-12` the relative figure is
reported as `None` and an **absolute** criterion applies instead, with a
declared tolerance of `1e-9`: a layer with no ordinary response must also have
no arm response to count as matched. Such layers are **not** dropped from the
decision.

**If the generalized arms' worst signal-only core relative change exceeds 1 %,
the run STOPS and reports before reading any comparative score**, and before
changing `rho`. These arms are **not** described as function-matched on the
strength of `rho`'s numerical value; Rawat's arm reports its own initial change
and is likewise not called function-matched.

## 8. Measurements

Primary: **equally weighted mean accuracy at the two trained longer delays, 32
and 64**. Reported per arm and per seed, with paired differences against **both**
`ordinary` and `rawat`, every seed shown. Delay 8 reported separately to expose
a retention/responsiveness tradeoff; delay 96 as held-out extrapolation.
**No checkpoint or seed is selected by the comparison.**

**No accuracy threshold is predeclared for this study.** The `+0.3` percentage
point figure used in the Speech Commands screens is **not** a criterion here and
is not imported. Differences are reported per seed with their signs and
magnitudes, and interpreted as such.

Also reported: parameter counts, physical and auxiliary carry, Rawat's input
buffer, wall time, learned `rho` and effective clock distributions. Saved
coefficients are not a memory-importance score.

Paired interventions on one fixed held-out probe batch: change the relevant
marked cue versus change an unmarked distractor, everything else held.
**Both are reported even if they do not support selectivity.**

## 9. What the comparisons permit

* `gp_rho` vs `ordinary`/`rawat`: whether the full construction improves this
  recall experiment over the literature mechanisms.
* `gp_rho` vs `gp_rho_frozen`: whether learning `rho` helps relative to the same
  initial fixed response. That is task-trained response shaping — **not**
  meta-learning and not within-sequence adaptation.
* `gp_rho` vs `ordinary_2x`: whether a larger ordinary recurrent state is an
  adequate alternative under these training conditions. This alone does not
  isolate every benefit of the exact coefficient ties.
* `professor`: the fully matched prospective recurrence loses driven linear
  memory. This is **not** a claim that TSS's memory architecture cannot learn
  memory; TSS explicitly retains non-prospective memory neurons.

Asserting that the **particular physical constraints** beat more general filters
would need a further matched response-parameterization control, and is **not**
claimed from this batch. Three seeds do not establish population variance.

## 10. Budget

**Hard total 1,200 s** including backend startup, focused checks, compilation,
warm-ups, all continuations, evaluation and cleanup. A deadline is passed to the
runner, a projection is made from the measured warm-up rate **before** any
comparative training, and compilations are reused across seeds by running
everything in one process. If the declared batch does not fit, the run reports
**INCOMPLETE** with the projection; seeds and controls are **not** silently
dropped and the cap is not widened. This study does not trigger a larger run.

## 11. Focused checks

`tests/test_recall_study.py`: the gamma/clock reparameterization preserving
blocks, drive, carry and forward response; the `rho = 1` algebraic limit against
ordinary S5 (a boundary check, not the training configuration); task structure,
pairing, latest-cue targeting and held-out delay; query causality; the professor
arm's invariance to the earlier cue and its **structurally zero** `log_step`
gradient; bit-identical cloning; the function-preserving embedding with a
trainable readout path for added modes and a genuinely doubled carry; `rho`
updating when learned and **not** when frozen while ordinary weights still move;
finite full-BPTT gradients for every arm; tokenwise normalization verified by
batch-splitting; and a production float32 probe at `highest` precision.

Artifacts go to a fresh directory. No old Speech Commands data, checkpoint,
score or protocol is modified.
