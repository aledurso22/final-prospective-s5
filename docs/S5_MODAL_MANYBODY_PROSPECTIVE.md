# Heterogeneous modal prospective S5: a stable cascade

20 September 2026. Branch `s5-modal-manybody-prospective`, from the clean
production commit `ef004cda025b4e098041cfe5970c217fa0015bba`. Native S5 is
byte-identical and every file here is new.

## 0. What this is, stated accurately

A **heterogeneous many-body, WWJ-inspired modal action/cascade**. The local
first-order factors are **variationally motivated** by the WWJ operator; the
particular **coupling topology** — one independent cascade per native S5
mode — and the **learned gates** that select each mode's regime are an
**explicit model construction**, not a consequence of the variational
principle. This is not the matched residual law and does not claim to be
derived from one.

## 1. The negative result this replaces

The direct matched route is stopped and preserved on branches
`s5-direct-generalized-prospective` (tip `a08c858`) and
`s5-three-arm-discrete-prospective`. What it established, and why no more
time is being spent on it:

* the **exactly matched** law `P(D)(s−f)=0` has `S/F = 1`, and with
  `f = Ā s + B̄ x` it collapses the driven response to `(I−Ā)s = B̄x` — a
  **memoryless map**, proved exactly;
* the **mixed-stencil** realization escapes that cancellation only by using
  inconsistent stencils, so its poles are discretization artefacts. They
  left the unit disc on the real S5 modes: 1.705–1.877 with the
  native-matched target, growing like `2h/τ` with the professor-consistent
  one, **1.4416 at τ = 2** with float32 divergence at token 242 — the
  *sequential* path too, so no scan could repair it;
* the surviving cells were **nearly marginal** (ρ = 0.998 at τ = 1000) and
  numerically fragile: the full-sequence doubling scan lost everything in
  float32 (measured 4.18e7 relative on the cluster), and even a chunked scan
  injects `ε·|H^C|` per boundary;
* three separate certification errors came from measuring a **mode subset**
  rather than the production inventory.

The lesson carried into this design: **do not let the discretization choose
the poles.** Here it cannot.

## 2. The stage, and why it is stable by construction

With the backward derivative `D_h y_t = (y_t − y_{t−1})/h`, the first-order
mechanical stage `(1 + d D_h) u = (1 + n D_h) v` is, multiplying by `h` and
solving for `u_t`,

$$u_t=\frac{d}{h+d}\,u_{t-1}+\frac{h+n}{h+d}\,v_t-\frac{n}{h+d}\,v_{t-1}.$$

Its **only** recurrent pole is

$$p=\frac{d}{h+d}\in(0,1)\quad\text{for every }d>0,\;h>0,$$

with no parameter able to move it out. The numerator parameter `n` moves a
**zero**, at `n/(h+n)`, and never a pole — so gates and numerator learning
cannot destabilize anything. At `n = d` the stage is the **exact identity**
for zero-consistent history.

## 3. The three regimes, selected by gates

Per mode `j` and stage `ℓ`:

$$d_{\ell j}=\text{D\_MIN}+\mathrm{softplus}(\cdot)>0,\qquad
g_{\ell j}=\sigma(\cdot)\in(0,1),\qquad
n_{\ell j}=d_{\ell j}+g_{\ell j}\,\mathrm{softplus}(\delta_{\ell j}).$$

| gates | cascade | regime |
|---|---|---|
| `g₁ = g₂ = 0` | both stages identities | **exactly Native S5** |
| one gate on | `q → u` | ordinary prospectivity |
| both gates on | `q → u → w` | generalized WWJ |

The two-stage transfer is
`(1+n₁D)(1+n₂D) / [(1+d₁D)(1+d₂D)]`, whose numerator expands to
`1 + Γⁿ D + Mⁿ D²` with

$$\Gamma^n_j=n_{1j}+n_{2j},\qquad M^n_j=n_{1j}n_{2j},$$

and the **critical numerator branch** `n₁ = n₂ = τ/2` gives `Γⁿ = τ`,
`Mⁿ = τ²/4` — the WWJ critical mass, appearing as a **numerator**, where it
shapes the response without owning a pole.

Gates are **per mode**, deliberately: the construction exists so that
different modes may choose different regimes. A gate penalty (mean gate)
keeps the model from switching mechanics on everywhere for free, and gate
values are reported per layer and per mode.

## 4. Cancellation is monitored, not assumed away

Each stage's numerator zero sits at `n/(h+n)`. If that lands on a native
pole the mode is **annihilated**. The layer therefore reports
`|L_j(λ̄_j)|` — the effective filter's gain at each native pole — penalizes
it quadratically below a declared floor, and counts the modes beneath that
floor. Exact cancellation cannot pass silently; a test constructs a
cancelled mode and checks the gain detects it.

## 5. Implementation

Each stage is a **scalar (per-mode diagonal) affine associative scan**, run
with `s5/ssm.py`'s own `binary_operator` through
`jax.lax.associative_scan` — the repository's scan primitive, reused. The
generalized branch is **two sequential scalar scans**. **No dense per-mode
3×3 or 4×4 companion is ever constructed.** The S5 recurrence itself is
untouched; `h = 1` token, never identified with S5's learned `Δ`.

## 6. What is proved, and where

`tests/test_modal_prospective_algebra.py` — exact rational arithmetic, no
JAX, runs anywhere (10 tests): the update solves the stage equation on a
basis; every added pole is `d/(h+d)` and lies strictly inside the disc; the
numerator and gate never move a pole; `n = d` is the exact identity and a
cascade of identity stages is too; one active stage is the ordinary lead
operator; two stages expand exactly to `1 + ΓD + MD²`, with the critical
branch giving `Γ = τ`, `M = τ²/4`; the gate selects the regime exactly; zero
prehistory with no wraparound at either boundary; the reverse branch is the
flipped causal cascade; and a cancelled native mode is detected by the gain.

`tests/test_modal_prospective_jax.py` — the cluster suite (13 tests): scan
versus sequential oracle for real and complex modes at awkward lengths
**through 16 000**; the two-stage cascade against the oracle; identity
stages reproducing Native sequences **and gradients**; gradients against the
oracle for every parameter; zero prehistory and no wraparound; bidirectional
orientation; poles inside the disc with gates unable to move them;
production-length **float32** values and gradients finite; the cancellation
report; the layer starting at Native with its penalty increasing in the
gates; and Native S5 byte-identity against `ef004cd`.

## 6a. First cluster results, and what they corrected

**Correctness: 24 passed** (10 exact + 14 JAX) at `5550f9d`, including the
production-shape float32 case (`max|state| = 686.7`, finite gradients) and
the layer starting Native — `regime_per_mode {native: 8}`, every added pole
`0.50025`, `native_mode_gain_min = 0.99967`, zero modes below the floor.

**Throughput, measured:** one stage `0.58×` Native, two stages `0.46×`, at
16 000 tokens. Each stage costs about one more scan of the same size, which
is what three scans against one should cost. **My "near-Native throughput"
prediction was wrong and is withdrawn.** Two caveats that made the first
numbers less useful than they looked, now fixed:

* peak memory was **identical for all three arms** because
  `peak_bytes_in_use` is a process-wide high-water mark — the `1.0` memory
  ratio was an artifact, not a measurement. Each arm now runs in its own
  process (`--arm`), and a layer-level comparison is measured too, since
  the scan-level ratio overstates the model-level cost.

**Synthetic gate check: `NOT_DEMONSTRATED`.** The numbers, and why:

| component | all-Native | gated | change |
|---|---|---|---|
| total MSE | 0.003024 | 0.002989 | −1.2% |
| long-delay memory | 0.002155 | 0.002240 | **+3.9% (degraded)** |
| lead | 0.000869 | 0.000749 | −13.8% |

So the gated model **traded memory for lead** rather than getting both. The
old criterion asked for heterogeneous gates and a lower *total*, and would
have called that a win; the criterion now requires **reduced lead error AND
long-delay memory not degraded**, because the scientific claim needs both.

Two structural findings from the same run:

* **the two stages were tied to six decimals** (`|n₁ − n₂| = 6.4e-7`).
  Identical initialization gives identical gradients, so `n₁ = n₂` for ever,
  which pins `Mⁿ = (Γⁿ/2)²` — the cascade was locked to the **critical**
  branch and could never reach the general passive branch it claims to
  span. Initialization is now asymmetric across stages and modes, and
  `n₁ − n₂` is reported so the tie cannot recur unnoticed;
* the gates **did** differentiate across modes in relative terms (24.7×
  between the largest and smallest) but stayed small in absolute terms
  (spread 0.075 against a 0.1 threshold), so they neither passed the
  heterogeneity test nor changed the model much.

None of this is tuned toward a positive result: the measurement defects are
fixed and the criterion is made stricter, which is the opposite direction.

## 6b. Second cluster run: memory is free, the trade is not

**Peak memory in the scan-only benchmark:** native 72 663 808 B, one stage
72 664 832 B, two stages 72 665 856 B — about 1 KB per stage. **That "the
cascade is free in memory" reading was wrong**, and §6d corrects it: the
measurement was forward-only, batch 1, one layer, so the 1 KB is just the
parameters. Under real training the stages' intermediates are retained for
the backward pass, and the true cost is +4.59 GiB. Throughput
is unchanged at the scan level: **0.58× / 0.46×**. The layer-level number is
still missing because `layer_comparison` omitted `bidirectional`, which
`init_S5SSM` requires and which has no default — a TypeError after four
minutes of GPU time. Fixed, and a **local** AST test now asserts the
benchmark supplies every argument `init_S5SSM` requires, so this class of
mistake fails on a laptop instead.

**Layer-level cost, measured:** native layer 0.000273 s, modal layer
0.000460 s at 16 000 tokens — **0.593× Native, i.e. about 1.69× slower per
layer**. That is better than the scan-level 0.46×, because the layer also
does the readout, the norm and the GLU, but only modestly: the scan is a
large share of the layer, so there is less amortization than predicted.
Memory remains free (~1 KB per stage) and everything stays finite.

Whether 1.69× per layer is affordable is an arithmetic question about the
allocation, not a judgement: it needs Native's own measured 15-epoch wall
time, which the completed Native wave already recorded in its
`task_result.json` (`training_seconds`, `examples_per_second`). No GPU is
needed to answer it.

**Synthetic, second run: still `NOT_DEMONSTRATED`, and the trade is
unchanged.**

| | first run | second run |
|---|---|---|
| lead error | −13.8% | **−13.2%** |
| long-delay memory | +3.9% | **+3.6% (degraded)** |
| `stages_are_tied` | **true** (6.4e-7) | **false** (\|n₁−n₂\| = 0.745) |
| gate ratio across modes | 24.7× | **39.1× / 16.0×** |
| gate spread (threshold 0.1) | 0.075 | 0.066 / 0.085 |

The asymmetric initialization did what it was meant to: the stages are no
longer tied, so `Γⁿ = 1.204` and `Mⁿ = 0.473` are now independent rather
than locked to `Mⁿ = (Γⁿ/2)²`, and the general passive branch is reachable.
**It did not change the result**, exactly as predicted before the run: the
gated model still buys ~13% on lead by giving up ~3.6% on long-delay
memory.

What this is evidence for, stated carefully: the gates **do** specialize
across modes — a 39× ratio between the most and least open mode is not
noise — but they stay small in absolute terms (0.02–0.09), so the
prospective mechanics are only weakly engaged, and where they are engaged
they cost memory. On this probe the construction trades rather than wins.

Two honest possibilities, not yet distinguished: the **gate penalty**
(1e-3 × mean gate, against an MSE of 3e-3) may be suppressing the gates, or
the **task** may genuinely not reward prospectivity without a memory cost.
A `--gate-penalty 0` run separates them, and is a diagnostic rather than a
tuning step, because its outcome is reported either way.

## 6c. The gate-penalty diagnostic settles the question

Running the synthetic task with `--gate-penalty 0` separated the two
explanations, and the answer is unambiguous:

| | penalty 1e-3 | **penalty 0** |
|---|---|---|
| gates heterogeneous | false | **true** |
| max gate spread | 0.085 | **0.502** |
| lead error | −13.2% | **−17.2%** |
| long-delay memory | +3.6% | **+6.1% (worse)** |
| total MSE | 0.002989 | 0.003001 |

The penalty **was** suppressing the gates — with it removed they open wide
(up to 0.59, spreads 0.24 and 0.50, ratios 15× and 6.7× across modes) and
the heterogeneity criterion passes. But opening them buys **more** lead and
costs **more** memory, monotonically. The penalty was never what stopped the
model getting both.

**On this probe the trade is intrinsic**, and that is the honest reading:
engaging the prospective mechanics reduces lag and degrades long-delay
memory, in proportion. `NOT_DEMONSTRATED` stands, and no tuning of the
penalty will change it.

**What IS demonstrated**, and was a stated deliverable: *different modes
learn different gates*. With the penalty off, gates range from 0.017 to
0.591 across the eight modes, a 15× ratio within a stage, and the two
stages are genuinely independent (`|n₁ − n₂| = 2.30`, `Γⁿ = 1.913`,
`Mⁿ = 0.930`, so `Mⁿ ≠ (Γⁿ/2)² = 0.915` — the general passive branch, not
the critical one). Every added pole stayed inside the disc
(`max_stage_pole = 0.687`).

**What is not demonstrated** is scientific benefit, and one structural
reason is visible in the probe itself: both demands are summed into a
**single scalar target**, so the shared linear readout must serve memory and
lead at once and per-mode specialization cannot be exploited. A two-channel
target — long-delay memory on one output, the switch edge on the other —
would let different modes serve different channels and is the minimal
honest redesign. If that *also* trades, the construction does not deliver
both on synthetic data, and that conclusion should be recorded rather than
engineered away.

## 6d. The production-shaped measurement, and a correction

The real model — `BatchClassificationModel`, batch 16, 16 000 tokens, width
96, six bidirectional layers, genuine optimizer updates:

| | Native | modal | ratio |
|---|---|---|---|
| seconds per step | 0.1894 | 0.2957 | **1.56× slower** |
| steps per minute | 317 | 203 | 0.640× |
| **peak GPU memory** | 6.91 GiB | **11.49 GiB** | **1.66× (+4.59 GiB)** |
| parameters | 281 098 | 283 402 | +2 304 |
| loss at init | 1.855573 | 1.855557 | — |
| gradient norm at init | 1.13601 | 1.13519 | — |

**The memory claim in §6b was wrong and is corrected here.** "About 1 KB per
stage" came from a forward-only, batch-1, single-layer measurement, where
1 KB is simply the parameter arrays. Under real training the cascade's
intermediates are retained for the backward pass across six layers and a
batch of sixteen, and the actual cost is **+4.59 GiB, a 1.66× increase**.
This is the third time a toy measurement has misled; the production-shaped
number is the one that counts, and 11.49 GiB is 48% of a 24 GiB RTX 3090,
so it fits with headroom.

Two sanity checks that the construction behaves as designed: the parameter
count rises by exactly 6 layers × 2 stages × 64 modes × 3 arrays = 2 304,
and the loss and gradient norm at initialization match Native to five
decimals — because the gates start closed, so the layer *is* Native there.

**Wall-clock projection.** Native's recorded 15-epoch wave is 5.51 h, of
which only 1.52 h is training steps; the other 3.99 h is validation,
checkpointing, data handling and the per-step host synchronizations in the
runner, none of which is arm-specific. So:

* if those overheads are arm-independent: **6.36 h**;
* if everything scaled by 1.56×: **8.60 h**.

The truth is between. Both ends **fit a 12 h allocation with a 25% margin**
(budget 9.0 h) and fit 18 h comfortably. **The cost question is closed: the
cascade is affordable.**

What remains open is the science, and it is not a speed problem.

## 6e. The two-channel probe was degenerate — no verdict was available

Splitting the target into two channels did not rescue the experiment, and
the reason is that **the probe could not measure the question at all**. Both
arms, penalty off, against a per-channel target variance of 1/256:

| channel | Native error | gated error | **Native R²** | **gated R²** |
|---|---|---|---|---|
| long-delay memory | 0.003472 | 0.003907 | **0.111** | **−0.000** |
| lead | 1.14e-5 | 3.69e-6 | **0.997** | 0.999 |

**Lead is solved by everyone; memory is learned by nobody.** The reported
"−67.5% lead, +12.5% memory" is a move from 99.7% to 99.9% variance
explained on one channel, and from 11% to 0% on the other. That is not
evidence that the construction trades memory for lead — it is a failed
measurement, and declaring a negative result from it would have been the
error.

Two structural faults, both mine:

* **the mode initialization could not span the task.** `lambda_re` was
  drawn from −(0.05 + 0.4·U), so the slowest representable mode had a
  timescale of 20 tokens — for a 32-token recall. The probe asked the model
  to hold something its own parameterization could not hold. The range is
  now −(0.002 + 0.3·U), i.e. 3.3 to 500 tokens;
* **the lead target was the first difference of an input channel**, which a
  two-tap readout solves exactly — hence Native's R² of 0.997. The lead
  input is now a **blurred** switch (exponential smoothing, timescale 8) and
  the target is the sharp one, so sharpening it is what a lead filter is
  for;
* the memory target was a delayed **delta**, which a bank of exponential
  modes cannot represent at all. It is now an exponential **trace** of
  timescale 32 — something a slow mode holds naturally and a lead filter
  attenuates, which is exactly the retention claim.

A **power check** now runs before any verdict: if the baseline explains less
than 20% or more than 99% of a channel, the status is `PROBE_UNDERPOWERED`
and no conclusion is drawn. The first two probes would both have been caught
by it.

This is the third probe, and the justification is not that the previous
answer was unwelcome: it is that the previous measurement provably had no
power. If the powered probe shows a trade, that is the answer and it gets
written up as the negative result.

## 6f. The powered probe shows both components improving — and what that
still does not settle

With the probe fixed, the gates unpenalised, 16 modes and 1200 steps:

| channel | Native | gated | change | Native R² |
|---|---|---|---|---|
| long-delay memory | 7.53e-5 | **2.48e-5** | **−67.1%** | 0.9984 |
| lead | 2.76e-4 | **4.10e-6** | **−98.5%** | 0.9989 |
| total MSE | 1.76e-4 | **1.44e-5** | **12.2× better** | |

**Both components improved**, the gates are strongly heterogeneous (spread
0.87, a 71× ratio across modes), the stages are independent
(\|n₁−n₂\| = 5.56), every added pole stayed inside the disc (max 0.846),
and the specialization correlation — a mode's gate against its share of the
**lead** channel — is **0.678**, against 0.22 on the degenerate probe. That
is the construction doing what it was built to do.

**The run was nonetheless reported `PROBE_UNDERPOWERED`, by my own guard,
and the guard was wrong.** It rejected a baseline at R² = 0.9984 as
"already solved" while the gated arm was cutting that residual twelvefold.
R² near 1 is the wrong way to ask whether headroom exists. The ceiling is
removed; the floor (0.20) stays, because a channel nobody learns really does
carry no information.

**What replaces it, because a single-seed win is not a finding:**

* **three seeds**, and an improvement must exceed the baseline's
  seed-to-seed half-range to count;
* **a capacity-matched Native control.** With the gates off, `d_raw`,
  `delta_raw` and `gate_raw` are inert, so the all-Native arm had 160
  effective parameters against the gated arm's 256 — the improvement could
  be capacity rather than mechanism. A second Native arm now runs at **26
  modes** (260 parameters) to remove that confound;
* the verdict `DEMONSTRATED_AGAINST_BOTH_CONTROLS` requires heterogeneous
  gates **and** both components improved beyond seed spread against **both**
  Native arms.

Until that runs, the honest statement is: **a promising single-seed result
with the capacity confound uncontrolled.** Not a demonstration yet.

## 6g. The controlled result: lead is the mechanism, memory was capacity

Three seeds, three arms — gated (16 modes, 256 effective parameters),
Native (16 modes, 160), and **Native capacity-matched (26 modes, 260)**.
Paired seed by seed, because the same seed is the same data and the same
initialization draw:

| component | vs Native (16) | **vs capacity-matched (26)** | verdict |
|---|---|---|---|
| **lead** | 3/3, 31–655× | **3/3, 28–85×** | **mechanism** |
| long-delay memory | 3/3, 5–178× | **2/3, and the control's mean is lower** | **capacity** |

Means: lead — gated 5.25e-6, Native 1.32e-3, capacity-matched 2.75e-4.
Memory — gated 1.88e-5, Native 2.58e-3, capacity-matched **1.25e-5**.

**The honest split.** The cascade delivers a large, robust improvement in
lead that extra modes do **not** buy: against a Native arm with *more*
parameters it is still 28–85× better on every seed. The apparent memory
gain, by contrast, is matched and slightly beaten by simply adding modes —
the capacity-matched control has the lower mean and wins one seed outright.
So the earlier "12× better on both" was mechanism on one channel and
capacity on the other, and only the control separated them.

**A statistical correction this exposed.** The mean-versus-spread test
vetoed the memory comparison against plain Native even at a −99.3%
difference, because one Native seed was 180× worse than its siblings and its
half-range alone exceeded the effect. A paired seed-by-seed test is the
right instrument when the baseline is that unstable, and it is now reported
alongside, with per-seed factors and a per-component verdict.

**Where this leaves the construction.** It does what it was built to do —
gates specialize (71× across modes, correlation 0.679 between a mode's gate
and its share of the lead channel), stages stay independent, poles stay
inside the disc, Native is recovered exactly at closed gates — and it buys a
real, capacity-controlled improvement in *temporal lead*. What it does not
yet show is capacity-controlled **memory retention**, which the original
claim requires alongside the lead. On this probe, that half is not there.

## 6h. The frontier experiment, predeclared before it is run

The §6g result is a point, not a curve, and three seeds cannot establish
either half of it: an exact sign test on three pairs cannot reach p < 0.05
even when the treatment wins all three. The next experiment measures the
whole memory-versus-lag frontier and runs the ablation that §6g never did.
Everything below was fixed **before** the run, and is asserted by
`tests/test_frontier_design.py` so it cannot drift afterwards.

**Delays** 16, 32, 64, 128, 256, at a fixed sequence length of 1024, so the
only thing that changes along the frontier is the delay.

**Seeds** 15, paired. A seed fixes the data stream and, where the shapes
allow it, the initialization draw. At 15 pairs the sign test needs 12 wins
for p < 0.05.

**Arms** — five, so both equal-mode and equal-parameter comparisons exist
for every question:

| arm | gates | stages | modes | effective parameters |
|---|---|---|---|---|
| `native` | off | – | 16 | 160 |
| `native_capacity_matched` | off | – | **26** | **260** |
| `one_stage` | on | 1 | 16 | 208 |
| `one_stage_capacity_matched` | on | 1 | **20** | **260** |
| `two_stage` | on | 2 | 16 | 256 |

Both controls carry *more* parameters than the construction (260 against
256), which is the conservative direction. `one_stage` and `two_stage`
share their mode count, and Λ, B and the readout do not depend on the stage
count, so for one seed those two arms start from exactly the same
recurrence and the same readout: the equal-mode ablation differs in the
second stage and in nothing else.

**Normalized error.** Every number is MSE divided by that channel's own
variance. The memory target's variance moves by orders of magnitude across
this sweep, so absolute MSEs are not comparable along the frontier; 1.0
means "no better than predicting the mean".

**Tests.** The exact paired **sign test** is primary — it assumes only that
the seeds are independent, and it is what every verdict reads. The
**Wilcoxon signed-rank** test is reported alongside for its extra power,
with a normal approximation that refuses to return a p-value below ten
pairs. Both can be read; they are allowed to disagree, and neither is the
only one shown.

**Memory non-inferiority margin: 10%**, with 5% reported alongside as a
secondary. α = 0.05. Non-inferiority is a *separate* question from
improvement and is tested as one — H₀ is that the paired ratio is at or
above 1 + margin. Failing to reject is **not** evidence of non-inferiority,
and the report says so instead of reading a null result as a pass.

**Lead improvement** must be statistically paired and significant, not a
mean comparison — the correction §6g forced.

**Gate selectivity.** Per-mode gates are recorded against per-mode
timescales. The construction predicts slow modes near g = 0 and faster
modes opening theirs, so the gate-versus-decay-rate correlation is
predicted **positive**, and the slow and fast halves are reported
separately so the correlation is not the only evidence.

### The decision rule, written down in advance

`experiments/s5_modal/frontier_design.py:decide` is branching over booleans
and nothing else, tested exhaustively over all 64 combinations:

- **`GATES_NOT_SELECTIVE`** — neither gated arm is memory non-inferior. The
  gates are not doing what they were designed to do, and a large lead gain
  does not change that.
- **`ORDINARY_SUFFICIENT`** — one stage matches two on both components. The
  second stage is unjustified and generalized WWJ mechanics are not needed.
- **`GENERALIZED_SUPPORTED`** — the two-stage arm retains memory, improves
  the lead, *and* beats the one-stage filter it contains at equal modes.
- `TWO_STAGE_WORKS_ONE_STAGE_DOES_NOT`, `NO_LEAD_IMPROVEMENT`,
  `UNDERPOWERED`, `INCONCLUSIVE` — so that a run which answers none of the
  three cannot be forced into one that it did not.

### One correction carried into this run

The previous probe drew mode decay rates **uniformly** on [0.002, 0.302].
The minimum of 16 such draws sits near 0.02 — a 51-token timescale — so at
a delay of 256 the model could not have held what the task asked it to
recall, and the comparison would have measured nothing. Rates are now drawn
**log-uniformly** across the same range, so slow modes are reliably present
at every delay. This is identical in all five arms and so favours none of
them.

### The smoke run, and the power check it made necessary

Job 67211 on pgi15-gpu5 at `5d38a2b`, four seeds and thirty steps over
delays 16 and 256. The pipeline runs end to end on GPU and the power guard
fired at both delays, correctly: after thirty steps the **normalized memory
error is above 1.0 in every arm** — worse than predicting the mean — while
the lead channel is already near 0.05.

That is what thirty steps should look like, and it is also why the smoke
cannot license the sweep. Whether the memory channel is learnable *at all*
at each delay is a precondition for every comparison in this experiment,
and it is not free to assume: this task is sparser than the §6g probe was.
A trace of timescale 16 occupies roughly 50 of 1024 tokens, five percent of
the sequence, where the earlier probe's occupied thirty-seven percent of a
length-256 sequence.

So a **power check** runs first: `native_capacity_matched` alone, at the
full step count, across all five delays. One arm instead of five, for a
fifth of the sweep's cost, answering the one question that could invalidate
all of it. `--arms` selects the subset, an unknown name is refused rather
than silently dropped, and a subset that lacks the comparisons the decision
rule reads returns `PARTIAL_ARM_SUBSET` instead of a verdict assembled from
tests that were never run.

Per-arm wall time is now printed and recorded, which the smoke should have
reported and did not.

### The power check, and the optimizer bug it found

`native_capacity_matched` alone, 15 seeds, 1200 steps, all five delays
(`7df3006`, job 67211). Normalized memory error in the **reference** arm:

| delay | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| memory | 0.008 | 0.004 | **0.77** | **1.11** | **1.44** |
| lead | 0.002 | 0.002 | 0.004 | 0.007 | 0.010 |

The reference arm fails at exactly the long delays where the construction
was expected to have its best chance, and it fails *monotonically* in the
delay. No comparison survives a reference that cannot do the task, so this
would have wasted the whole sweep — which is what running one arm first
was for. Cost: ~40 s per arm-delay, so the full sweep is about 17 minutes.

**The cause was my probe's optimizer geometry, not the task.** Adam moves
every parameter by roughly the learning rate each step, whatever the
gradient's size. The decay rate was stored ADDITIVELY, so a 3e-2 step means
something different depending on where the rate already is: at a delay of
256 the needed rate is 1/256 = 0.0039 and one step is an **eightfold
overshoot in timescale**, destroying slow modes as fast as they are found;
at a delay of 16 the rate is 0.0625 and the same step is a survivable 50%.
That is precisely the observed pattern — clean at 16 and 32, degrading
through 64, gone by 128.

The rate is now stored and learned in the **log**, which makes a 3e-2 step
a 3% change in timescale at every scale. Native S5 parameterizes its own
timescale as `log_step` for this same reason, so this is the established
practice rather than a thumb on the scale. The initialization
*distribution* is unchanged — rates were already drawn log-uniformly — so
only the geometry moves, and it moves identically in all five arms.

The frontier is only worth measuring where the reference arm can do the
task, so the power check is repeated before the sweep.

### Three power checks, and what each one falsified

The reference arm's normalized memory error, 15 seeds, 1200 steps:

| delay | 16 | 32 | 64 | 128 | 256 | lead (all delays) |
|---|---|---|---|---|---|---|
| **v1** additive rate, uniform ω, no bias | 0.008 | 0.004 | 0.77 | 1.11 | 1.44 | 0.002–0.010 |
| **v2** log rate | 0.006 | 0.004 | **0.011** | **0.030** | 1.31 | 0.002–0.006 |
| **v3** + bias, log-uniform \|ω\| | 0.236 | 0.233 | 0.417 | 0.540 | **0.756** | 0.013 |

Five runs each, ~3.5 minutes, instead of twenty-five. Every one of these
would have silently wasted the sweep.

**v1 → v2: the rate was stored additively.** Adam moves every parameter by
about the learning rate per step whatever the gradient's size, so a 3e-2
step meant something different depending on where the rate already was: at
delay 256 the needed rate is 1/256 = 0.0039 and one step is an eightfold
overshoot in timescale, destroying slow modes as fast as they are found; at
delay 16 the rate is 0.0625 and the same step is a survivable 50%. Exactly
the observed pattern — clean at 16 and 32, degrading through 64, gone by
128. Learning the rate in the **log** improved 64 by 73× and 128 by 37× and
removed the monotone degradation. Native S5 parameterizes its own timescale
as `log_step` for this reason.

**The readout had no bias.** `einsum(features, readout)` cannot emit a
constant, so "predict the channel mean" — the fallback that *defines* a
normalized error of 1.0 — was unavailable, and scores above 1.0 carried no
information about how much of the channel was captured. Counted in the
parameter budget, identically in every arm.

**v3: ω has no absolute scale to be drawn on.** Uniform on [−2, 2], the
smallest of 26 magnitudes is ≈0.077 — a period of 82 tokens, nine
oscillations across a 768-token trace — and 256 was unreachable at any
training length (1200 → 1.31, 4000 → 1.28, 12000 → **1.42**, worse).
Drawing \|ω\| log-uniformly instead put the median magnitude at 0.045, so
nearly every mode became near-DC, the basis collapsed into near-duplicates,
and the short end regressed 39–60× while the lead regressed 8× everywhere.
Both drawings are wrong in the same way.

**What matters is ω relative to the mode's own decay rate** — the
dimensionless **quality factor** `Q = ω/r`, with `λ̄ = exp(r(−1 + iQ))`. A
mode is slow only if it decays slowly *and* oscillates slowly; otherwise it
is a fast oscillation inside a slow envelope, useless for a trace. At the
fast end (`r = 0.3`) `Q_MAX = 8` restores `|ω| ≤ 2.4`, the range that
worked; at the slow end (`r = 1/256`) it gives a period ≥ 201 tokens
against a 256-token decay — 2.5× better than the uniform draw's *best*
case, as its *worst* case.

It also repairs the same optimizer geometry a second time. An additive ω
takes a 3e-2 step that is a 100% overshoot at ω = 0.03 and a rounding error
at ω = 2. `Q` is O(1) at every timescale, so the step means the same thing
everywhere.

**The lesson, stated once.** Every one of these three defects is the same
mistake: a parameter carrying physical units, tuned by an optimizer whose
step size is unit-free. The fix each time is a dimensionless coordinate —
`log_rate` for the timescale, `Q` for the frequency. All three changes
apply identically to all five arms and none favours a channel; they change
what the mode basis can express, not who gets to express it.

`slowest_timescale`, `omega_of_slowest_mode`, `quality_of_slowest_mode` and
`min_abs_omega` are now reported per arm, so the next failure of this kind
is read off rather than guessed at.

## 6i. The frontier result

Five delays, five arms, 15 paired seeds, 1200 steps, length 1024
(`9432669`, job 67211). **Predeclared verdict: `GATES_NOT_SELECTIVE` at
every delay.** That verdict stands. What follows is what the run contains,
including why that verdict is not the whole of it.

### What is established

**Gating improves lead, overwhelmingly.** `two_stage` against equal-mode
Native: **15/15 seeds at every delay**, median 2.44–3.30×, sign p < 0.0001,
Wilcoxon p = 0.0004. Against the capacity-matched Native, also 15/15 at
every delay, 2.76–3.34× — a gated arm with *fewer* parameters beating a
wider Native on every seed. This is far stronger than §6g's single point.

**The second stage earns its place — the ablation you asked for.**
`two_stage` against `one_stage` at equal modes and an *identical* Λ, B and
readout draw, so the second stage is the only difference:

| delay | lead wins | median | p | memory non-inferior |
|---|---|---|---|---|
| 16 | 13/15 | 1.63× | 0.0037 | 13/15 ✓ (also at 5%) |
| 32 | 12/15 | 1.43× | 0.0176 | 15/15 ✓ (also at 5%) |
| 64 | 13/15 | 1.40× | 0.0037 | 14/15 ✓ (also at 5%) |
| 128 | 13/15 | 1.44× | 0.0037 | 15/15 ✓ (also at 5%) |
| 256 | 15/15 | 1.34× | 0.0001 | 15/15 ✓ (also at 5%) |

Significant lead gain at **every** delay while memory non-inferior at
**every** delay. At equal *parameters* (against
`one_stage_capacity_matched`) the lead gain is larger still, 1.67–2.03×,
14–15/15. By the criterion declared in §6h — "if two stages improve the
lead–memory frontier, that specifically supports the generalized
mechanics" — this is that outcome. Ordinary prospectivity is **not**
sufficient.

**Memory against equal-mode Native** is non-inferior at 32, 64, 128 and 256
(12, 14, 13, 13 of 15; p ≤ 0.018), and misses at 16 with 11/15, p = 0.0592
— one seed short of the 12 the sign test needs.

### Why the verdict says otherwise: the capacity-matched control is not paired

The decision rule tests memory against `native_capacity_matched`, and that
arm failed at every delay. It is not because the gated arm lost memory. Its
median ratios are 0.718, 0.793, 0.799, 1.045, 0.899 — the gated arm mostly
*better* — while only 8–11 of 15 seeds fall inside the margin. Low median
with high scatter and no significance is the signature of a broken
comparison, not of a trade.

The cause is structural and it is mine. A capacity-matched arm has 26 modes
instead of 16, so it draws a **different initialization**. A seed then
shares only the data stream, not the recurrence it starts from. Measured at
delay 16:

| pair | draw | correlation across seeds | spread of paired ratios |
|---|---|---|---|
| native vs two_stage | **same** | **0.987** | 0.60 |
| native vs capacity_matched | different | 0.187 | 4.10 |

Against a decoupled arm the "paired" test degenerates into an unpaired one
at n = 15, which has almost no power. The 16-mode arms, by contrast, track
each other seed for seed — delay 16, seed 0: 0.1322 / 0.1489 / 0.1336;
seed 14: 0.2560 / 0.2529 / 0.2352 — which is exactly why the tests among
them resolve so cleanly.

The same rank-sensitivity explains a discrepancy worth recording: per-seed
memory spans 0.006 to 0.256, a 40× range, so a median-of-ratios (0.989)
and a ratio-of-medians (1.098) disagree. The paired statistic is the
correct one, which is why it was declared primary before the run.

**The control was also worse than the thing it controls.** Adding 10 modes
degraded Native's memory at every delay — 0.095 vs 0.037, 0.053 vs 0.026,
0.035 vs 0.023, 0.038 vs 0.031, 0.055 vs 0.051. A control that gets worse
when given more parameters is not isolating capacity; it is measuring
optimization difficulty at 26 modes. So §6g's conclusion that memory was
"explained by capacity" rested on the same defective instrument.

### What is falsified: the selectivity premise

The construction predicts slow modes holding at g ≈ 0 while faster modes
open. Measured in `two_stage`:

| delay | corr(gate, rate) | slow half | fast half |
|---|---|---|---|
| 16 | +0.026 | 0.085 | 0.067 (**inverted**) |
| 32 | +0.025 | 0.085 | 0.078 |
| 64 | +0.022 | 0.083 | 0.077 |
| 128 | +0.027 | 0.081 | 0.110 |
| 256 | +0.044 | 0.069 | 0.106 |

Essentially no specialization, and at short delay the wrong way round.
`one_stage` does better (+0.19 to +0.20, halves 0.078 against 0.176) but is
the arm that loses the ablation. Mean gates sit at 0.09–0.13 throughout:
mostly closed, near-uniformly.

**So the second stage's advantage does not come from per-mode selectivity.
It comes from the second-order numerator 1 + Γ_n D + M_n D² itself.** The
cascade works; the "heterogeneous many-body" reading of *why* it works does
not survive this run, and the framing should follow the evidence rather
than the other way round.

### Where this leaves the claim

The standing bar is retained long-delay memory **and** reduced response
lag. Against equal-mode Native that holds at four of five delays. Against a
capacity control it is **not established**, and cannot be until the control
shares its initialization — which is a new experiment, not a re-run.

Two results are solid and do not depend on that repair: gating improves
lead on every seed at every delay, and **two stages beat one** on the
lead–memory frontier at every delay. The third — that gates specialize by
timescale — is falsified.

## 7. First deliverables, and what is deliberately absent

Delivered: the derivation above, the implementation, both test suites, a
Native-versus-one-stage-versus-two-stage benchmark
(`experiments/s5_modal/benchmark.py`, 16 000 tokens, compile timed
separately, peak memory, forward and backward), and a tiny synthetic
memory-plus-switch experiment (`experiments/s5_modal/synthetic_gates.py`).

**Not** delivered, deliberately: the 15-epoch experiment is not wired and not
launched, Native S5 is not rerun, and **no scientific benefit is claimed**.
The synthetic experiment reports
`DIFFERENTIATION_DEMONSTRATED` only when the fitted gates are heterogeneous
across modes **and** the gated model beats the all-Native baseline on the
same data and budget; it separately reports the long-delay-memory error and
the lead error, because the scientific claim requires **both** retained
long-delay memory and reduced response lag, and neither is assumed.

Nothing in this commit launches anything.
