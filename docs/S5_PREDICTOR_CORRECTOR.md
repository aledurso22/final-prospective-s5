# Generalized S5: effective timescale vs correction at the prediction

Local research diagnostic, 2026-11-17. No training was run. Production
constructors, runner defaults, the training protocol and historical results are
unchanged.

> **Revision (2026-11-17, before transfer to the cluster).**
> 1. **Initialization draw corrected.** The first version of this diagnostic
>    was launched with `JAX_ENABLE_X64=1`, and under x64 `runner.init_state`
>    draws different `log_step` and `B` values (`Lambda` is deterministic).
>    It therefore analysed a statistically equivalent sample, not the float32
>    production initialization.
>    * Every initialization is now drawn by
>      `experiments/s5_predictor_corrector/production_init.py`, in an
>      x64-off subprocess.
>    * All §4 numbers were regenerated. They moved slightly; no conclusion
>      changed.
>    * The zero-drag diagnostic was unaffected. It drew the init before
>      enabling x64, and its rerun output is identical.
> 2. **Conclusions narrowed.**
>    * The auxiliary (fast) component's contribution is small at
>      initialization. Whether it becomes useful after learning is **not
>      established**.
>    * C preserves DC gain and DC delay exactly, but matches native's slow-pole
>      expansion **only to first order** in K.
>    * The cross-token residual difference contains the changing input. It is
>      an **adaptation** of the paper's fixed-objective gradient difference,
>      not the same term.
>    * The fast-component "energy share" omits slow/fast interference. It is a
>      **component-energy proxy**; the exact-total version is reported beside
>      it.
> 3. **Trained-checkpoint comparison added** (§4.7). It covers native and
>    generalized, each against its matched initialization.

## 0. Provenance and files

* **Starting state.** Branch `s5-two-compartment-parallel-scan` at
  `a01d2b2738a27bda6011e11f01a9d532ac3bd9af`. The working tree was dirty:
  * modified: `docs/S5_THREE_ARM_REFERENCE.md`;
  * untracked: `docs/S5_ZERO_DRAG_DIAGNOSTIC.md`, `experiments/s5_zero_drag/`,
    `s5/independent_inertia.py`, `tests/test_independent_inertia.py`,
    `.DS_Store`.
* **Work branch.** `s5-nesterov-timescale-research`, created with `git switch -c`
  so the uncommitted work was carried over intact. Nothing was reset, stashed or
  committed.

| file | role |
|---|---|
| `s5/predictor_corrector.py` | research module: coefficients, `PredictorCorrectorS5SSM`, literal rollout, f64 analysis helpers |
| `tests/test_predictor_corrector.py` | 46 float64 tests (`JAX_ENABLE_X64=1`) |
| `tests/test_predictor_corrector_float32.py` | 16 production-float32 tests (x64 **off**) |
| `experiments/s5_predictor_corrector/diagnostic.py` | 2×2 diagnostic on the initialized modes (trained ones: `checkpoint_compare.py`) |
| `experiments/s5_predictor_corrector/production_init.py` | the float32 production initialization, drawn x64-off whatever the caller |
| `experiments/s5_predictor_corrector/checkpoint_compare.py` | trained native vs trained generalized, each against its matched init (§4.7) |
| `tests/test_checkpoint_compare.py` | 7 tests: pairing, provenance, refusal, init identity, drift on effective poles |
| `experiments/s5_predictor_corrector/benchmark_local.py` | CPU cost per layer |
| `experiments/s5_predictor_corrector/quadratic_example.py` | separate fixed-quadratic illustration |
| `results/s5_predictor_corrector/` (gitignored) | outputs; committed copies in `docs/evidence/s5_predictor_corrector/` (`diagnostic.json`, `responses.png`, `benchmark_local.txt`, `quadratic_example.json`) |
| `docs/S5_ZERO_DRAG_DIAGNOSTIC.md` | scope note and four narrowed sentences (§6) |

---

## 1. The current recurrence in predictor–corrector form

The implementation is the source of truth.

* `discretize_zoh` (`s5/ssm.py:29-42`) gives `λ̄ = exp(ΛΔ)` and
  `B̄ = Λ⁻¹(λ̄ − 1)B̃`. Λ is never used again below; only the discrete pair
  `(λ̄, B̄)` is.
* `target_map` (`s5/discrete_recurrence.py:7-9`) gives `F = 1 + k(λ̄−1)` and
  `G = kB̄`, with `k = response/h`.
* `generalized_coefficients` (`s5/discrete_recurrence.py:20-26`) sets
  `q = M + h(γ + T)`, `α = M/q`, `β = h²/q`, `δ = hT/q`, and returns:
  * `a1 = 1 + α − β + (β+δ)F`, `a2 = −α − δF`;
  * `c1 = (β+δ)G`, `c2 = −δG`.
* `response_mass_gamma` (`s5/generalized_prospective_ssm.py:16-20`) gives
  `M = ρ γ T`. The module call is at `:50-52`.

At `h = 1`, with `K = 1 − λ̄`, `r_t(s) = K s − B̄ x_t` and `Q = M + γ + T`, the
code's recurrence regroups exactly as

```
s_t = s_{t−1} + (α+δ)(s_{t−1} − s_{t−2}) + β(Y_t(s_{t−1}) − s_{t−1})
      + δ(Y_t(s_{t−1}) − Y_{t−1}(s_{t−2})),          Y_t(s) = s − k r_t(s)
```

so

```
μ = α + δ = (M + T)/Q,     η = k β = k/Q,     ζ = k δ = k T/Q
d_t = μ (s_{t−1} − s_{t−2}) − ζ (r_t(s_{t−1}) − r_{t−1}(s_{t−2}))
y_t = s_{t−1} + d_t                                  (predictor)
s_t = y_t − η r_t(s_{t−1})                           (existing corrector)
```

The production arm is `k = T`. The response T enters the code twice: as the
target scale k inside `target_map`, and as the damping-difference weight δ
inside `q`. The explicit k separates these two roles.

**Initial conditions.** The scans use `s_{−1} = s_{−2} = 0` and `x_{−1} = 0`
(`discrete_recurrence.py:49`, `:74`), hence `r_{−1}(s_{−2}) = 0`. The literal
rollout uses the same convention. Each direction of the bidirectional layer
starts from that zero prehistory independently.

**Verification.**
* Variant-A coefficients equal `generalized_coefficients` to 2e-16 on random
  modes and parameters.
* The literal predictor–corrector rollout (`direct_rollout`, which evaluates
  `d`, `y` and `r` as written) equals the expanded recurrence through the
  sequential and companion scans, for all four variants. This holds with zero
  prehistory and also with a nonzero `(s_{−1}, s_{−2}, x_{−1})`.
* The variant-A module has the same parameter tree, names and values as
  `GeneralizedProspectiveS5SSM` under the same key.
* Its outputs agree to float32 rounding (rtol 1e-6), not bitwise. The raw
  parameters are float32 in both modules, so T, M and γ are rounded at float32
  and then pass through two different groupings of the same algebra.

The cross-token residual difference `r_t(s_{t−1}) − r_{t−1}(s_{t−2})` is kept.
It plays the role of the Hessian-damping term of the paper's Eqs. (6)/(7), but
it is an **adaptation**, not the same term. The paper differences the gradient
of one fixed objective at two iterates. Here
`r_t(s_{t−1}) − r_{t−1}(s_{t−2}) = K(s_{t−1} − s_{t−2}) − B̄(x_t − x_{t−1})`
also contains the change of input between tokens.

## 2. Correction at the prediction (variants B, D)

**Source.** Hadjisavvas, Lara, Marcavillaca and Vuong, arXiv:2506.15632 v1. The
Springer page redirects to a login, so the journal version was not read. The
preprint's Eqs. (6) and (7) were read verbatim, and they share the predictor:

```
y_k     = x_k + α_k (x_k − x_{k−1}) − θ_k (∇h(x_k) − ∇h(x_{k−1}))
(6)  x_{k+1} = y_k − β_k ∇h(x_k)        heavy ball + Hessian correction
(7)  x_{k+1} = y_k − β_k ∇h(y_k)        Nesterov-type (Section 4.2)
```

With the mapping `α → μ`, `θ → ζ`, `β → η`, `∇h(·) → r_t(·)`, the existing S5
corrector has the form of Eq. (6). It is an adaptation: the objective changes
with the token, so the "gradient difference" carries the input change (§1). The only change here is the correction point:
`s_t = y_t − η r_t(y_t)`. The predictor is unchanged.

**Coefficients.** With `P = 1 − ηK` and `L = μ − ζK`:

| | a1 | a2 | c1 / B̄ | c2 / B̄ |
|---|---|---|---|---|
| old | `P + L` | `−L` | `η + ζ` | `−ζ` |
| new | `P(1 + L)` | `−PL` | `η + ζP` | `−ζP` |

These formulas were verified, not assumed. The literal rollout of each corrector
matches its expanded recurrence (tests 2.x). The algebraic forms
`old = s_{t−1} − η r_t(s_{t−1}) + d_t` and `new = s_{t−1} − η r_t(s_{t−1}) + P d_t`
are tested separately. Equivalently, `s_new − s_old = −ηK d_t` at the same state.
This is a discrete algorithm change, not an added continuous force.

**What is not imported.** The paper's convergence results assume a real,
L-smooth, strongly quasiconvex objective `h: ℝⁿ → ℝ` with restricted coefficient
ranges. The coefficient bounds of its Theorems 13 and 17 were seen only through a
summary and are unverified here. S5's residual is complex, is affine in an input
that changes every token, and has no fixed minimizer. Nothing below relies on
those theorems.

## 3. Timescale-calibrated control (variants C, D)

For any k, `η/(1−μ) = k/γ`.

| target scale | `η/(1−μ)` | at the original init |
|---|---|---|
| legacy `k = T` | `T/γ` | **0.05** |
| calibrated `k = γ` (γ > 0) | 1 | 1 |

This is a change of target scaling, labeled separately from the corrector change.
It does not preserve the complete native transfer function (§4).

**Identities checked** (numerically, against explicit roots and simulation):

* **DC gain.** Both correctors, both scales: `H(1) = B̄/K`, which is native.
  Relative error ≤ 6e-12.
* **Slow pole.** `z_slow = 1 − cK + c₂K² + O(K³)`, with `c = η/(1−μ)`:
  * old: `c₂ = −c(c − η − ζ)/(1−μ)`;
  * new: `c₂ = −c(c − η(1+μ) − ζ)/(1−μ)`.

  So the correction point moves the principal pole by **`μc²K² + O(K³)`**.
  Native has `c = 1`, `c₂ = 0`.

  **C matches native's slow-pole expansion only to first order.** It has
  `c = 1`, but its old-corrector `c₂ = −(μ − ζ)/(1 − μ)`, which vanishes only on
  `M = T(γ−1)` (below). At the original init, `c₂ = −0.025`. DC gain and DC
  delay, by contrast, are preserved exactly.
* **DC group delay** (zero-state convention, `−d arg H/dω` at ω=0, analytic and
  by finite differences of the phase):
  * old: `Re[(1−μ)/(ηK)] − 1`;
  * new: `Re[(1−μ)/(ηK)] − (1−μ)`.

  The new corrector adds exactly **μ tokens** on both scales. Under the
  calibrated scale the old delay equals native, `Re(1/K) − 1`, exactly.
* **Exact native family.** Under `k = γ`, the old corrector has `μ = ζ` if and
  only if `M = T(γ − 1)`. Then:
  * the poles are `{λ̄, μ}`;
  * the numerator is `B̄(1 − μz⁻¹)`;
  * the μ-mode cancels, so `H` is native S5 for **every** γ > 1, i.e.
    `ρ = 1 − 1/γ`.

  The brief's reference point `γ = 2, M = T` (ρ = ½) is one point on this curve.
  On the init modes it reproduces native trajectories to 6e-14 from zero state,
  which is the compatible initialization. The new corrector at the same
  parameters differs by 6.9% relative L2.

  The original init (γ = 1, ρ = ½) is **not** on the curve, since `μ − ζ = 0.023`.
  Both correctors share the same initialization in this comparison.
* **Redundancy.** Under `k = γ`, `η = 1 − μ` identically. The map
  `(T, ρ, γ) → (μ, η, ζ)` has Jacobian rank **2** (legacy: 3), so one direction
  of the three raw parameters is redundant. The parameterization and parameter
  count were deliberately left unchanged for this first comparison.
* `k = γ` was **not** applied to the zero-drag experiment, since `γ = 0` would
  also remove the drive.

## 4. Local 2×2 comparison at the original initialization

**Configuration.**
* Modes: 1152 initialized modes, seeds 301–303. They are the float32
  production draw of `runner.init_state`, via `production_init.py`, and the
  backbone is the same for every arm. All modes are complex.
* Range of the modes:
  * `|λ̄|` from 0.9515 to 0.9995;
  * `|K|` from 6.9e-4 to 1.63, median 0.035;
  * `|arg λ̄|` up to 1.97 rad/token.
* Parameters: `T = 0.05`, `ρ = 0.5`, `γ = 1`, `M = 0.025`, formed exactly as the
  module forms them from float32 raw values. This gives `μ = 0.0698` for all
  variants, with:
  * A/B: `η = 0.0465`, `ζ = 0.0023`;
  * C/D: `η = 0.930`, `ζ = 0.0465`.
* Precision: float64 analysis; frequency grid over **[−π, π]**, because the
  modes are complex.

**Variants.**

| | old corrector | prediction corrector |
|---|---|---|
| **legacy k=T** | **A** (existing arm) | **B** |
| **calibrated k=γ** | **C** | **D** |

### 4.1 Per-mode results (median [5th, 95th percentile] unless stated)

| quantity | native | A | B | C | D |
|---|---|---|---|---|---|
| stable (Schur–Cohn) | 1152 | 1152 | 1152 | 1152 | 1152 |
| spectral radius | .9949 | .9996 | .9996 | .9953 | .9944 |
| stability margin `1−ρ` | 5.1e-3 [6.5e-4, .039] | 3.5e-4 [3.4e-5, .012] | same as A | 4.7e-3 [6.4e-4, .039] | 5.6e-3 [6.5e-4, .050] |
| slow time const. / native | 1 | **18.3** [1.45, 20.0] | 18.3 | **1.004** [1.00, 1.64] | 0.991 [0.45, 1.00] |
| DC delay (tokens) | 4.31 | 105.2 | 105.3 | 4.31 (= native, to 4e-11) | 4.38 (native + μ) |
| \|fast pole\| | — | .070 | .070 | .070 | .070 |
| \|R_fast\|/\|R_slow\| | — | .023 | .023 | .023 | .023 |
| fast-component energy **proxy** (max) | — | 9.5e-5 | 8.2e-5 | 7.6e-5 | 2.3e-4 |
| fast energy / exact total (max) | — | 9.6e-5 | 8.2e-5 | 7.6e-5 | 2.3e-4 |
| interference term / exact total (max) | — | 7.6e-3 | 7.0e-3 | 5.2e-3 | 7.1e-3 |
| freq. response vs native (rel. L2) | 0 | 1.00 [.91, 1.02] | 1.00 | **.006** [.001, .63] | .008 [.001, .46] |
| H∞ / native H∞ | 1 | 0.59 [.07, .97] | 0.58 | 1.004 [1.0, 1.62] | 0.99 [.45, 1.0] |
| new vs old corrector (rel. L2) | | | **3.8e-4** [1.2e-6, 4.8e-3] | | **.010** [1.1e-4, .65] |
| `ℓ1` gain / native (16k steps) | 1 | 0.67 | 0.66 | 1.004 [1.0, 1.62] | 0.99 [.45, 1.0] |
| min disc/a1² (repeated-root proximity) | | .74 | .75 | .74 | .75 |

The **proxy** is `|R_f|²/(1−|z_f|²)` divided by the sum of both component
energies. It omits the slow/fast interference term
`2 Re[R_s R̄_f/(1 − z_s z̄_f)]`, so it is not an exact fraction of the
impulse-response energy. The "exact total" row divides by the full
`Σ_t |h_t|²`, computed in closed form. At initialization the two agree, because
the interference term is under 0.8% of the total.

* Pole predictions agree with simulated trajectories to ≤ 3e-16 in log-growth,
  for every variant.
* All partial fractions match the trajectories.
* Nothing is clipped or stabilized.

### 4.2 Where C and D differ from native (by `|K|`, medians)

| `|K|` | n | C vs native | D vs native | D vs C | τ_slow/native C | D |
|---|---|---|---|---|---|---|
| < 0.01 | 312 | .0014 | .0015 | .0004 | 1.000 | 1.000 |
| 0.01–0.1 | 503 | .0057 | .0073 | .009 | 1.005 | 0.992 |
| 0.1–0.5 | 249 | .069 | .10 | .16 | 1.10 | 0.86 |
| 0.5–1.63 | 88 | .70 | .50 | .68 | **1.77** | **0.41** |

The calibration is first-order in K: `c = 1` exactly, but `c₂ ≠ 0`. It is
therefore near-exact for slow modes and not exact for strongly rotating ones.
For those modes the second-order terms have opposite signs:
* C lengthens the time constant, by up to 2.0×, and raises the resonance peak,
  by up to 1.97×;
* D shortens the time constant and lowers the resonance peak, by as much as a
  factor of 0.26–0.45.

### 4.3 The whole layer at initialization

Setup:
* all six production SSM layers of each seed (`H = 96`, `P = 64`,
  bidirectional), each at its actual float32 initialization and evaluated in
  float64;
* parameters shared across variants;
* random input of length 2000;
* SSM path only (the `D·x` skip is removed).

| seed | A vs native | B vs A | C vs native | D vs native | D vs C | SSM RMS: native / A / C / D |
|---|---|---|---|---|---|---|
| 301 | 0.995 | 0.0015 | 0.303 | 0.247 | 0.333 | .0251 / .0038 / .0261 / .0229 |
| 302 | 0.990 | 0.0013 | 0.287 | 0.231 | 0.334 | .0255 / .0041 / .0268 / .0235 |
| 303 | 0.994 | 0.0016 | 0.281 | 0.227 | 0.329 | .0243 / .0036 / .0255 / .0225 |

The layer-level C/D differences come almost entirely from the 88 strongly
rotating modes, which carry much of the output energy. These are differences
of the operator at initialization. They say nothing about which operator
learns better representations.

### 4.4 What the legacy scale does to the initialized operator

* For slow modes, **A is approximately native S5 with its step Δ shrunk by
  `T/γ = 0.05`**. Under ZOH the DC gain `−B̃/Λ` does not depend on Δ, so the
  comparison needs no gain normalization. Per-mode relative L2 against that
  rescaled native mode has median 0.017 for `|K| < 0.1`, and 0.053 over all
  modes.
* For strongly rotating modes the correspondence breaks:
  * A's pole is `1 − 0.05K`, and it has no aliasing counterpart;
  * the fastest mode's resonance moves from ω ≈ −2.0 rad/token to near 0
    (`responses.png`, right column).
* At init, A responds about 20× more weakly than native at every non-DC
  frequency, and its SSM output RMS is 6.6× smaller.
* Whether training compensates can only be read from trained checkpoints, and
  not from `log_step` alone. Δ, Λ, T and γ jointly set the effective poles, so
  §4.7 compares those poles directly.
* The earlier report found the trained arm still at `γ/T ≈ 20–26`. That
  observation concerns T and γ only.

### 4.5 Stability beyond the initialization

`|P| < 1` does **not** imply stability of the full recurrence.

* **Counterexample** (legacy scale, near-Nyquist mode, low drag; tested):
  * mode `λ̄ = −0.9566 − 0.1446i`, with `T = 0.1406`, `ρ = 0.0184`,
    `γ = 0.00424`;
  * `|P| = 0.91`;
  * old radius 0.925, new radius **1.887**; the impulse response diverges past
    1e10 in 60 steps.
* **Random search** (2,000,000 in-disc modes, `T, γ ∈ [1e-3, 1e3]`):

  | | old stable | new stable | stable-old/unstable-new | of which \|λ̄\| > 0.9 | stable-new/unstable-old |
  |---|---|---|---|---|---|
  | legacy | 53.3% | 57.0% | **11,285** (all with \|P\| < 1) | 3,906 | 85,019 |
  | calibrated | 67.3% | 76.2% | **0** | 0 | 179,527 |

  The calibrated zero is empirical; no proof is offered.
* **Grid on the init modes** (T ∈ 0.05–50, γ ∈ 0.01–100, ρ ∈ {1e-4, .5, 1}): no
  stable-old/unstable-new mode in either scale. The new corrector stabilizes
  many modes that the old one leaves unstable. This does not mean it is "more
  stable" in general (see the counterexample).

### 4.6 Scans and precision

* **float64.** On all four variants the sequential, companion and factored scans
  agree in value (rtol 1e-10) and in gradients with respect to T, M, γ and λ̄
  (rtol 1e-7).
* **Repeated roots.** On a mode solved so that the new corrector's roots
  coincide exactly, and at relative perturbations 1e-9 and 1e-5, the sequential
  and companion scans agree in value and gradient, and both are finite. The
  factored scan was not relied on there; its known singular gradient at zero
  discriminant is recorded in `test_independent_inertia.py`. At init the
  discriminant stays ≥ 0.74·|a1|², far from repeated roots.
* **float32** (production precision), all four variants:
  * Each scan matches the float64 literal rollout to < 2e-3 relative over 2000
    steps.
  * d/d(T, M, γ) matches float64 central differences to ≤ 5.5e-3 of the largest
    entry. The worst case is the M-gradient of the legacy variants through the
    *sequential* scan.
  * At production width over **16,000 tokens**, the sequential and factored
    implementations give finite outputs and finite gradients for every
    parameter of every variant.
* **Local cost** (CPU, L = 4000, batch 2, forward + backward; noise ±20%
  between runs):

  | scan | cost vs native S5 |
  |---|---|
  | factored | 2.2–2.4× (production arm 2.3–2.8×) |
  | companion | 1.9–2.5× |
  | sequential | 23.6× |

  The corrector change costs nothing measurable; it only changes the
  coefficients.

### 4.7 Trained-checkpoint comparison

`experiments/s5_predictor_corrector/checkpoint_compare.py` reads checkpoints
only. It does no training, needs no data, and does not repeat the
2-million-sample search.

**Inputs.**
* Both arms' task directories (`<run>/<arm>/<seed>/`), paired **by seed**.
* For every checkpoint it records arm, seed, epoch, the task's own
  selected-validation epoch, sha256, and the run's `run_metadata.txt` (commit,
  scan).

**Matched initialization.** Each arm is compared against
`runner.init_state(arm, seed)`, reconstructed today in float32.

**What the init-agreement check does and does not verify.** The field
`reconstructed_ssm_init_identical_across_arms` compares only the SSM-layer
leaves (Λ, `log_step`, B, C1, C2) of the two arms' **reconstructed**
initializations.
* It does not verify that the historical runs started from that
  reconstruction. No initial checkpoint was stored, so this cannot be checked.
* It does not compare the encoder, norm or decoder leaves.

**Stable modes only.** Each variant reports its Schur–Cohn stable and unstable
counts. DC delay, decay times, residue ratios, the energy proxy, the
exact-total energy share and the component gains are computed over stable
modes only; their keys end in `_stable_modes_only`. An unstable mode has none
of these infinite-horizon quantities. Paired decay ratios use only modes that
are stable in both arms.

**Evidence is never overwritten.** If `<out>/checkpoint_compare.json` already
exists, the tool refuses before doing any work.

**Reported per arm, at both trained and matched init:**
* Λ and Δ jointly (`ΛΔ`), the actual effective poles, decay times
  `−1/log|z|`, and rotation frequencies `|arg z|`;
* for the generalized arm: T, γ, ρ, M, `T/γ`, both companion roots, and the
  auxiliary (fast) residues;
* learned B/C coupling: `‖B̄_p‖`, `‖C_p‖`, and a per-component gain proxy
  `‖C_p‖‖B̄_p‖|R_i|/√(1−|z_i|²)`, with interference omitted.

**Two questions, kept separate in the output:**
1. **actual:** trained native vs trained generalized, plus per-mode ratios
   (same seed, layer and mode index; identity defined by the shared init);
2. **counterfactual:** A (recomputed, checked against the production
   coefficients), B, C and D evaluated **at the generalized arm's learned
   parameters**. B, C and D were never trained.

**It fails loudly (exit 2) on:**
* unmatched seeds, or a seed given twice;
* a directory whose arm or seed disagrees with its path or `task_result.json`;
* a missing checkpoint;
* a native checkpoint passed as generalized, or the reverse;
* a parameter tree that does not match the matched init.

**Which epoch.** The comparison uses **epoch 15** for every task. These are
diagnostics of the epoch-15 parameters. They are not the selected-validation
epoch's parameters, unless the two coincide; the selected epoch is printed
next to each one. Nothing here is a performance comparison.

**Result of the first cluster run** (as reported, epoch 15). The JSON is
preserved on the cluster at
`/Users/durso/s5-runs/checkpoint_compare_epoch15/checkpoint_compare.json` and
is not copied into this repository.
* **Actual comparison:** trained A keeps a median paired slow-decay-time ratio
  of **17.04×, 13.99× and 15.19×** against trained native (seeds 301, 302,
  303). Training did not compensate for the legacy timescale.
* **Counterfactual comparison** at A's learned parameters:
  * C has **64, 51 and 46** unstable modes out of 384;
  * D is stable on all 384 modes for every seed.

  These counterfactuals were never trained, so they do not predict how C or D
  will train. They show that A's learned parameters lie outside C's stability
  region, which is why §10 monitors C and D's own poles every epoch.

That first run predates the stable-mode-only summaries and the renamed
init-agreement field. A rerun must use a **new** `--out`.

**Cluster command** (epoch-15 groups; the earlier native seed-301 run
`20260919-190727` is excluded by naming the paths explicitly):

```bash
# pin the commit named in the hand-off: git checkout --detach <SHA>
cd /Local/durso/final-prospective-s5 && \
JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -m \
  experiments.s5_predictor_corrector.checkpoint_compare \
  --native /Users/durso/s5-runs/s5-three-arm-15epoch/20260919-224525/native_matched_s5/301 /Users/durso/s5-runs/s5-three-arm-15epoch/20260919-224525/native_matched_s5/302 /Users/durso/s5-runs/s5-three-arm-15epoch/20260919-224525/native_matched_s5/303 \
  --generalized /Users/durso/s5-runs/s5-two-compartment-factored/20260921-175242/generalized_prospective_s5/301 /Users/durso/s5-runs/s5-two-compartment-factored/20260921-175242/generalized_prospective_s5/302 /Users/durso/s5-runs/s5-two-compartment-factored/20260921-175242/generalized_prospective_s5/303 \
  --epoch 15 --out /Users/durso/s5-runs/checkpoint_compare_epoch15_v2
```

## 5. Fixed-quadratic illustration (separate from S5)

Setup: `h = ½xᵀAx` with 50 eigenvalues in [1, 100], β = 1/L, constant
coefficients, Eqs. (6) and (7). Per eigenvalue this is the same algebra with
`K → λ` and no input.

| α | θ | Eq. (6) iterations to 1e-8 | Eq. (7) iterations |
|---|---|---|---|
| 0.5 | 0 | 287 | 290 |
| 0.8 | 0 | 87 | 93 |
| 0.9 | 0 | 167 | **115** |
| 0.95 | 0.5β | 231 | **182** |
| 0.5, β = 1.9/L | 0 | 147 | **diverges** (rate 1.63) |

Evaluating at `y_k` helps at high momentum, slightly hurts at moderate momentum,
and diverges at a large step where Eq. (6) converges. This illustrates the
mechanism only. Faster minimization of a fixed quadratic is **not** evidence
about sequence representations.

## 6. Zero-drag conclusions, narrowed

In `docs/S5_ZERO_DRAG_DIAGNOSTIC.md` the numbers are unchanged. A scope note and
four sentence-level edits now state:
1. Zero low-frequency lag does not in general mean no dynamical memory. The
   near-static behavior was observed in the tested regime only.
2. At `M = γ = 0` the law is `r + Tṙ = 0`, so `r(t) = r(0)e^{−t/T}`. The static
   map assumes a compatible initial residual, or that the transient has decayed.
3. The zero-drag negative result does not rule out positive-damping inertial S5
   variants.

## 7. Answers

1. **How much does correcting the timescale change the original behavior?**
   Almost entirely.
   * A (the existing init) is a different operator from native: leading time
     constants 18× longer, DC delay 105 vs 4.3 tokens, rotation frequencies
     compressed, and layer output 99% different from native with 6.6× lower RMS.
   * C preserves DC gain and DC delay **exactly**. It matches native's slow
     pole **only to first order** in K (`c₂ = −0.025` at this init). Per mode
     it is within 0.6% of native at the median.
   * C is not native for strongly rotating modes (`|K| > 0.5`: 70%), and at the
     layer level it differs from native by 28–30%.
2. **What does evaluating at `y_t` add?**
   * Exactly: `s_new − s_old = −ηK d_t`.
   * The DC delay grows by μ tokens.
   * The principal pole moves by `μc²K² + O(K³)`.
   * The input numerator changes from `(η+ζ, −ζ)` to `(η+ζP, −ζP)`.
   * The fast pole barely changes.

   The effect scales with `η|K|`.
3. **Is it appreciable at the original init?**
   * Legacy scale: no. `η = 0.047`, so B differs from A by 0.04% per mode
     (median; max 0.5%) and by 0.13–0.16% at the layer.
   * Calibrated scale: as an operator change, yes, but only for rotating modes.
     D differs from C by about 1% for `|K| < 0.1`, 16% for `|K|` 0.1–0.5 and
     68% above 0.5, and by 33% at the layer. The effect is extra damping of
     rotating modes: their time constant is 0.41× native, against 1.77× for C.
     Whether that damping helps or hurts learning is not established.
4. **Is there a stable, observable, useful extra mode?**
   * Not at initialization. In every variant the second pole sits at
     `|z| ≈ 0.04–0.07`, so it e-folds in under 0.4 tokens.
   * Its residue is about 2.3% of the slow one. Its component-energy proxy is
     ≤ 2.3e-4, and its share of the exact total, interference included, is also
     ≤ 2.3e-4.
   * It is observable and well separated (not near-cancelled, except exactly on
     `M = T(γ−1)` under C), and it is numerically benign. But its contribution
     at initialization is small.
   * Whether training makes it useful is **not established**. §4.7 reports the
     learned residues and B/C coupling of the trained generalized arm.
   * What D changes at initialization is the **principal** pole, not a new
     mode.
5. **Matched initialization and monitoring for a later training run:**
   * **Initialization.** Same backbone seed and same raw parameter values for
     every arm (shared names, tested): `T = 0.05`, `ρ = 0.5`, `γ = 1`, and the
     native `log_step` init unchanged.
   * **Scan.** The companion or factored scan; the discriminant is far from 0 at
     init, but this must be monitored.
   * **Monitoring,** logged per layer each epoch from the live coefficients,
     with no clipping, so that failures are recorded rather than hidden:
     * Schur–Cohn stability count, plus spectral-radius and margin quantiles;
     * the minimum `|a1² + 4a2|/|a1|²`, switching to the companion scan if it
       approaches 0;
     * quantiles of `|P|` and of `η|K|`;
     * slow time constant relative to native at the same λ̄, and the DC-delay
       quantiles;
     * the fast-component residue and energy proxy;
     * learned `T`, `ρ`, `γ`, and `log_step` drift against the native arm.

## 8. Recommendation

* **Timescale hypothesis: a small training comparison is justified locally.**
  * At initialization, the existing arm's operator is about 20× slower than
    native, with compressed rotation frequencies. The calibrated control
    removes that difference to first order, at no cost.
  * Before spending GPU time, run the §4.7 comparison on the epoch-15
    checkpoints. Training compensated for the legacy scale only if the trained
    generalized arm's **effective slow poles** (decay times and rotations)
    moved toward the trained native arm's. Do not judge this from `log_step`
    alone. If it compensated, C tests less than it appears to.
  * Smallest informative design: **C** (and optionally **D**), 3 seeds, under
    the completed protocol.
* **Protocol rule for any later accuracy comparison.**
  * Every arm must use the **same split and the same selection rule**.
  * The historical numbers are selected-validation results on the protocol's
    split. New arms may be compared with them only under that same rule.
  * If a held-out test evaluation is added, native and A must be re-evaluated
    under it as well. A new held-out-test score is never compared with a
    historical selected-validation score.
  * The protocol, the split and the historical results stay unchanged.
* **Correction-at-prediction: no separate run on its own merits.**
  * Under the legacy scale it changes nothing measurable (B ≈ A to 0.1%); skip B.
  * Under the calibrated scale it acts as extra damping on rotating modes, not
    as a new useful mode. Include D only as the second cell of the calibrated
    row, and only if budget allows.
* **No accuracy gain is predicted from the optimization paper.** Its rates
  concern a fixed real objective, and in our own quadratic example the
  Nesterov-type correction is not uniformly faster.

**Pre-existing test state (not changed here).**
* `tests/test_s5_three_arm_equations.py::test_three_arm_names_are_exactly_three`
  fails at `a01d2b2`, because the committed runner now has a fourth arm.
* Pre-existing at `15aa1a2` and unchanged by the training integration:
  * `test_s5_three_arm_protocol_corrections.py` fails 2 tests
    (`exact_three_names`, `ARM_CONFIGS == ARM_ORDER`);
  * `test_s5_three_arm_equations.py` fails 1 test (`exactly_three`);
  * `test_matched_lag_jax.py` fails 3 tests.

  All six fail identically on a clean `15aa1a2` worktree. The name tests
  predate the matched-lag arm. The training-integration suites
  (`test_timescale_training_integration.py` 14, `test_checkpoint_compare.py`
  9) and `test_s5_three_arm_production.py` (3) and
  `test_s5_execution_telemetry.py` (20) pass.
* Several existing test modules enable x64 globally at import. The float32 file
  must therefore run in its own pytest process; its guard refuses to run
  otherwise, rather than silently testing float64.

**Limitations.**
* Trained checkpoints were not available locally, so every result in §4.1–§4.6
  is at initialization. §4.7 is the tool for the trained checkpoints; it has
  been tested on synthetic run directories only.
* The Springer version of the paper was not read, and Section 4.2's theorem
  constants were not verified.
* The calibrated-scale "no stable-old/unstable-new" result is empirical.

## 10. Training integration: N / A / C / D × seeds 301, 302

**Arms.**

| arm | code identifier | constructor |
|---|---|---|
| N | `native_matched_s5` | `init_S5SSM`, unchanged |
| A | `generalized_prospective_s5` | `init_generalized_prospective_S5SSM`, the **original** constructor, unchanged |
| C | `calibrated_old_s5` | `init_predictor_corrector_S5SSM(variant="C")` |
| D | `calibrated_prediction_s5` | `init_predictor_corrector_S5SSM(variant="D")` |

* C and D reuse A's raw initial values, parameter names, optimizer groups
  (`NO_BC_DECAY_SSM_KEYS`) and scan. Only the coefficient map differs.
* `ARM_ORDER` and the finalizer remain the original three-arm study.

**Shared protocol, unchanged.**
* `runner.train_arm`; batch 16; 15 epochs (one warm-up epoch, then cosine
  decay over 14).
* The same data cache and the same official split.
* The same selection rule: best validation accuracy, then lower validation
  cross-entropy, then the earlier epoch.
* No finalizer, so the test split is never read.
* Every arm starts fresh from `runner.init_state(arm, seed)`.
* Comparisons stay inside this wave, which shares one split and one selection
  rule. These results are never mixed with held-out-test scores.

**Production float32.**
* `run_task` refuses to start if x64 is on, and the launcher forces
  `JAX_ENABLE_X64=0`.
* Before anything trains, the launcher's pre-flight confirms on CPU:
  * A, C and D have identical initial leaves;
  * N's leaves are identical to A's backbone;
  * the batch size is 16.

**Pole diagnostics** (`experiments/s5_three_arm_full/pole_diagnostics.py`).
* **Every step:** the logged `companion_spectral_radius` is computed from each
  arm's own recurrence. For native, Zucchet and A it is unchanged and
  test-checked against a verbatim copy of the original code. C and D use their
  predictor–corrector coefficients, which the old code would have computed
  incorrectly.
* **At initialization, every epoch, and at a numerical failure,**
  `pole_diagnostics.jsonl` records:
  * Schur–Cohn stability, margins, repeated-root proximity, `|P|` and `η|K|`;
  * slow and fast poles, decay times and rotations;
  * the slow time constant relative to the native pole of the same backbone;
  * residues (stable modes only), DC delay (stable modes only), and `T/γ`.
* **Production precision.** Each mode is also evaluated in the model's own
  float32, through the same coefficient functions and the same
  `companion_roots` the factored scan uses, and classified:
  * **stable**;
  * **numerical boundary:** within 1e-5 of the unit circle, or float32 and
    float64 disagree about stability;
  * **unstable:** the float64 radius exceeds 1 + 1e-5, so the equations
    themselves are unstable.
* Nothing is clipped or repaired. A finite instability is recorded and
  training continues under the unchanged protocol. A non-finite step raises
  `NumericalTrainingFailure`, which writes `failure.json` and the poles at the
  failure.
* At the production initialization, all four arms are 384/0/0 stable. A has
  the smallest margin, about 2.5e-5.

**Monitoring and logging overhead,** measured. The training computation is
unchanged.
* **Before:** after every step, one blocking device-to-host sync per
  TrainState leaf (239–293 leaves), plus eager norm and radius ops, plus an
  open, append and close of the JSONL file.
* **After:** one jitted call and one transfer per step; buffered JSONL flushed
  every 50 steps and at each epoch end; validation sums brought back in one
  transfer.
* **CPU, per step:** 8.0–11.0 ms → 0.28–0.51 ms (22–28×). JSONL append:
  23 µs → 3 µs.
* **On the cluster,** every smoke re-measures this on its own GPU and gates on
  equal values and lower cost (`smoke_result.json: telemetry_check`).
* **Trajectory identity.** On synthetic data, 2 epochs × 2 steps, the old
  runner (`15aa1a2`) and the new one produce:
  * **byte-identical** epoch checkpoints;
  * identical per-step records in every field;
  * identical validation results and the same selected epoch.

  This was checked for A and for native.

**Launcher:** `bin/run_experiments/allocation_s5_timescale_2x2_eight_gpu.sh`.
* Eight tasks, one per GPU, run as direct children of the allocation.
* The eight smokes run at once and gate training. The gate checks: scan,
  x64 off, telemetry agreement and lower cost, and no unstable mode at init.
* All eight trainings then run at once.
* Each task's JAX compilation cache is shared between its own smoke and its
  own training run.
* `STAGE=smoke` stops after the gate. `DRY_RUN=1` prints the plan.
* It refuses an existing run root.
* A task that fails is recorded and never retried.

## 9. Reproduce

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_predictor_corrector.py -q    # 46
.venv/bin/python -m pytest tests/test_predictor_corrector_float32.py -q             # 16, x64 OFF, own process
.venv/bin/python -m pytest tests/test_checkpoint_compare.py -q                      # 9
.venv/bin/python -m pytest tests/test_timescale_training_integration.py -q          # 14, x64 OFF
# the init is drawn float32 in a subprocess; x64 only affects the analysis
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -m \
    experiments.s5_predictor_corrector.diagnostic --out results/s5_predictor_corrector   # ~60 s
JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -m experiments.s5_predictor_corrector.benchmark_local
PYTHONPATH=. .venv/bin/python -m experiments.s5_predictor_corrector.quadratic_example
```
