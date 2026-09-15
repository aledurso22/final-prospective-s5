# Stage 2 diagnostic report

Analysis of the SAVED Stage 2 models. No training, no optimizer update, no
coefficient change, no test scoring. **The Stage 2 verdict is unchanged: the
predeclared screen FAILED** (`docs/GP_IMPLEMENTATION_REPORT.md` s12.12).

Protocol, frozen before execution: `docs/GP_STAGE2_DIAGNOSTIC_PROTOCOL.md`.

## 1. Execution identity

| item | value |
|---|---|
| branch / commit executed | `stage2-diagnostic` / `cc4c75293c8ded2b69f46bed6c1cfd13688a2531` |
| host | `pgi15-gpu3.iff.kfa-juelich.de`, RTX 3090 |
| SLURM job | 65870, `CUDA_VISIBLE_DEVICES=0`, backend `gpu`, jax 0.11.0 |
| source analysed (read-only) | `/Users/durso/s5-runs/stage2/` |
| output | `/Users/durso/s5-runs/stage2-diagnostics/20260915-172533/` |
| logs | `/Users/durso/s5-runs/stage2-diagnostics/logs/20260915-172417/` |
| focused tests | **49 passed, 1 skipped, 71.5 s**, exit 0 |
| wall time | **202 s of a 1124 s** remaining budget; 281 s total including tests |
| status / exit | **INCOMPLETE / 3** |
| digest tool | `experiments/gp/stage2_digest.py` at `8760b7f` |

**Why INCOMPLETE, precisely.** One optional artifact did not run: plots, because
`matplotlib` is not installed in the shared environment. **No required check
failed, no phase was skipped for budget, and no analysis was truncated.** The
status rule treats any declared-but-unexecuted item as short of PASS. That rule
was written before the run and has not been adjusted after seeing the result;
the plots can be regenerated from the saved JSON at any time.

## 2. Verification before interpretation

Nothing below is interpreted until these passed.

| check | result |
|---|---|
| restored validation **correct counts** vs saved, all five arms | **exact integer match** |
| restored validation cross entropy vs saved | **0.00e+00** difference, all five |
| adapter vs executed core, **4 layers x 32 inputs per arm** | worst relative **3.8e-08 to 5.0e-08** |
| JVP vs central finite difference (gate step 1e-2) | 2.0e-05 to 1.3e-04, all pass |
| pre-pooling future-input sensitivity | **0.0 exactly**, every arm and anchor |
| Stage 2 source file hashes, before vs after | **unchanged** |
| restored dtypes | `float32`, `cast_applied: false` |

Restored accuracies reproduce the screen exactly: `native_s5` 94.17 %,
`alpha_p_s5` 95.00 %, `gain_clip_s5` 94.60 %, `gp_fixed_m0` 93.60 %,
`gp_fixed_mass` 94.26 %; all from epoch-9 checkpoints, n = 5783.

## 3. What the saved models actually compute

### 3.1 The current tap, separated from the shared feedthrough

`|K0|` is dominated by the native `D` in every arm, so the mechanism-specific
quantity is `|K0 - diag(D)|`, the **dynamical** current tap. Layer 0, with the
matched ordinary control as the unit:

| arm | `|K0 - D|` | relative to `gain_clip_s5` | validation |
|---|---|---|---|
| `gain_clip_s5` | 0.486 | 1.00x | 94.60 % |
| `native_s5` | 0.610 | 1.26x | 94.17 % |
| **`gp_fixed_mass`** | **0.613** | **1.26x** | 94.26 % |
| `gp_fixed_m0` | 1.842 | 3.79x | 93.60 % |
| **`alpha_p_s5`** | **2.616** | **5.38x** | **95.00 %** |

The same ordering holds in all four layers (alpha-P 2.6-3.8, M=0 1.8-2.8,
mass 0.61-0.79, ordinary 0.49-0.80).

### 3.2 How far each law moves the response at its own learned weights

The within-checkpoint counterfactual holds `Lambda`, `Delta`, `B_tilde`,
`C_tilde`, `D`, gain and clipping exactly as trained and swaps only the
recurrent law. Relative change of the executed response versus the one-tap law:

| arm | layer 0 | range over layers |
|---|---|---|
| `alpha_p_s5` | 0.382 | 0.38 - 0.60 |
| `gp_fixed_m0` | 0.294 | 0.29 - 0.40 |
| **`gp_fixed_mass`** | **0.100** | **0.10 - 0.16** |

### 3.3 End-to-end sensitivity by lag, pre-pooling

Mean over four fixed Rademacher directions, anchor frame 80. Fractions of the
sensitivity within the available history:

| arm | lag 0 | lags 1-64 (history) | total magnitude |
|---|---|---|---|
| `native_s5` | 0.666 | **32.6 %** | 96.3 |
| `alpha_p_s5` | 0.719 | 27.9 % | 169.0 |
| `gain_clip_s5` | 0.721 | 27.5 % | 100.0 |
| `gp_fixed_mass` | 0.765 | 23.4 % | 110.4 |
| **`gp_fixed_m0`** | **0.963** | **3.7 %** | 208.2 |

Anchor 160 gives the same ordering. `gp_fixed_m0` retains **0.13x** the history
share of its matched control while carrying the **largest** total sensitivity:
its response is large and almost entirely instantaneous.

### 3.4 The M=0 pole map, predicted and measured

The contract predicts that `a_eff = a/(1 - T a)` maps the stable half-plane
into the disk of centre `-1/(2T)` and radius `1/(2T)`; at `T = 5` that is
`Re in (-0.2, 0)` and `|Im| <= 0.1` per frame. Measured continuous poles,
layer 0:

| arm | min Re | max abs Im | inside the M=0 disk |
|---|---|---|---|
| `native_s5` | -0.0639 | 0.2364 | no |
| `alpha_p_s5` | -0.0764 | 0.2191 | no |
| `gain_clip_s5` | -0.0784 | 0.1982 | no |
| **`gp_fixed_m0`** | **-0.1046** | **0.0820** | **yes** |
| `gp_fixed_mass` | -0.2689 | 0.1856 | no |

**The prediction is confirmed on the trained model**: the M=0 arm's modal
frequencies are compressed by **2.88x** relative to native and sit inside the
predicted disk, while every other arm lies outside it. Positive mass changes
the map and is not confined by that bound.

No pole was clipped in any arm (`n_raw_poles_clipped = 0`, raw-minus-clipped
`0.0e+00`), so clipping plays no part in these results. Trained poles moved
substantially from initialization (`trained_vs_init` 0.36-0.56), so the
trained-parameter fix of R1 was necessary for these numbers to mean anything.

### 3.5 Reconstructed initialization

Before any training, layer 0 median decay was 279.5 frames for the three
ordinary/two-tap arms, **159.5** for `gp_fixed_m0` and **9.7** for
`gp_fixed_mass`, at nearly equal total impulse energy (38.8-39.7). The M=0 law
therefore halves the median decay **at initialization**, before optimization
has any say.

## 4. Assessment against the candidate explanations

**`gp_fixed_m0` — excessive change to learned temporal filtering. Supported.**
Three independent measurements agree: frequencies compressed into the disk the
algebra predicts (2.88x), core impulse energy 99.1 % at lag 0, and end-to-end
history share cut to 3.7 % against 27.5 % for its matched control. It is also
the worst arm (93.60 %). The mechanism is predicted a priori by the contract's
own bound and confirmed on the trained model, which is stronger than a
post-hoc correlation.

**`gp_fixed_mass` — insufficient current-path benefit over the common
substrate. Supported, and the more interesting finding.** The positive-mass arm
changes its response *least* of the three interventions: counterfactual change
0.10-0.16 versus 0.38-0.60 for alpha-P, and a dynamical current tap
indistinguishable in size from the ordinary control (1.26x, the same as plain
native S5). It pays 1.93x per-step compute, ~2.5x measured epoch time and
double the recurrent carry for a realized response close to the ordinary one.
On this evidence it did not fail by distorting the computation; it failed by
**not changing it enough to pay for itself**.

**Optimization or gradient pathology. Not supported.** Total parameter-gradient
norms span 1.13-2.09 across arms with no depth-attenuation signature; per-layer
norms are the same order in every arm, and per-example activation gradients
decay gently with depth (roughly 4.0e-3 to 2.0e-3 at block inputs) almost
identically across arms. Nothing here distinguishes the arms in the way their
accuracies differ.

**Normalization, readout, or an implementation defect. Not supported.** The
adapter reproduces the executed core to 5e-08 over every layer and input, the
restoration is exact, causality is exact, and no pole was clipped.

**A schedule cause. Not testable here and not claimed.** Stage 2 used a
**ten-epoch cosine schedule** (`steps_per_epoch * epochs` = 843 x 10 = 8430
decay steps), not a ten-epoch prefix of a 300-epoch schedule. The alternative
was not run.

**The positive relationship that frames all of it.** Across these five points,
the size of the dynamical current tap tracks accuracy in the right direction —
alpha-P 5.38x and best, M=0 3.79x but with its history destroyed, mass 1.26x
and no better than the control it matches. That is five points, one seed, one
checkpoint each: a suggestive structure-function pattern, not an established
relationship.

## 5. Limitations, including one in my own reporting

* **One seed, one checkpoint per arm, ten epochs.** The accuracy gaps being
  explained are 0.35-1.00 pp. These diagnostics describe what the trained
  models compute; they do not establish why the optimizer arrived there, and
  they cannot resolve whether a 0.35 pp gap is real.
* **The reported `decay_frames_median` is misleading for `gp_fixed_mass`** and
  should not be quoted. For the mass arm the discrete spectrum contains two
  eigenvalues per mode — the physical branch and the auxiliary velocity branch
  — and the median falls in the fast auxiliary branch (5.6-7.5 frames) even
  though the spectral radius is 0.9991, i.e. slow modes are present. This is a
  defect in a statistic I chose, not in the model. The band energies and the
  pre-pooling sensitivity are unaffected, because both read the physical state
  only; the conclusions above rest on those.
* **The 512-lag window is not the whole response.** Its remainder is reported
  as `unknown (not bounded)`. The earlier geometric bound was invalid and was
  removed, not loosened; the frequency comparison instead uses an exact
  closed-form window remainder.
* **Local sensitivity is not memory.** These probes measure derivative
  magnitude at a point, not delayed-recall accuracy or semantic usefulness.
* **Lag-0 dominance is partly substrate.** `|K0|` is dominated by the shared
  native `D` in every arm, which is why the analysis uses `|K0 - D|`.
* **Plots were not produced** (`matplotlib` absent); all underlying JSON is
  saved and the figures can be regenerated without rerunning anything.
* **Initialization gradient probes (D4) were not implemented**, by the
  protocol's priority order. Reconstructed-initialization cores were computed.
* A measured aside on the R3 batch fix: per-example gradients here turn out to
  be positively aligned (shared-offset norm about 10x the per-example mean,
  against 5.7x for random signs), so in this particular run the old
  shared-offset code would have reported a differently-scaled quantity rather
  than a near-zero one. The correction still matters — the cancellation failure
  mode is real and data-dependent — but it did not, on these data, hide a
  collapse.

## 6. Conclusion and the next supported question

The evidence distinguishes the two GP arms rather than lumping them together,
and points at different problems.

For **M = 0**, the diagnosis is concrete and predicted: at `T = 5` the pole map
confines every effective mode to a disk of radius `1/(2T)`, the measured
frequencies are compressed 2.88x into exactly that disk, and the network's
end-to-end response collapses onto the current frame. The supported next
question is **whether the horizon `T` is simply too long for a 161-frame MFCC
sequence at this clock** — `T` is a declared model-time convention, not a
fitted value, and the bound it imposes is a direct function of it. That is a
question about the convention, answerable analytically before any training, and
it is not authorized by this brief.

For **finite mass**, the diagnosis is the opposite and is the more useful
result: at its learned weights the law produces a response close to the
ordinary one (10-16 % change) with a current tap no larger than the matched
control, while costing roughly 2.5x the epoch time and double the state. The
supported next question is **what would have to be true for the mass law to
change the computation materially at these poles** — since on this evidence the
finite-mass member as configured is close to an expensive re-parameterization
of the ordinary response, rather than a different computation that underperforms.

Neither question is a rescue sweep, and neither is started here. The failed
Stage 2 screen stands.
