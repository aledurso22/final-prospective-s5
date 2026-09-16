# Adaptive associative-memory comparison — executable protocol

Frozen **before** any numerical check, calibration or training. Everything
numerical runs on the cluster under one 600-second cap. This file is the
contract; the code is `experiments/adaptive_memory/`, the checks are
`tests/test_adaptive_memory.py`, the launcher is
`bin/run_experiments/cluster_adaptive_memory.sh`.

Sources:
`ADAPTIVE_MEMORY_CODING_BRIEF.md`, `ADAPTIVE_MEMORY_PROTOCOL.md`,
`ANALYTICAL_AUDIT.md` and `ORDINARY_PROSPECTIVE_BASELINES_AMENDMENT.md`
(16 September 2026).

Branch `adaptive-prospective-memory`, parent
`9b427273263e7667c43253589bf25333c975cb3d` (the completed nested-memory
study). That study, its report and its cluster artifacts are preserved
untouched; this launcher writes under its own output root
`$PROSPECTIVE_RUNS/adaptive-memory`.

## 0. Amendment status — which case applies

The ordinary-prospective amendment requires the agent to record which case
applies. **Numerical execution had NOT begun** when the amendment arrived: no
cluster batch for this study had been dispatched, and nothing had been run
locally. The two ordinary-prospective references are therefore **included in
this batch**, committed before execution. The five original arms, their
coefficients, initializations, learning rates, task, seeds, schedule and
success criteria are unchanged. The executed commit and start status are
recorded in `docs/ADAPTIVE_MEMORY_REPORT.md`.

## 1. Unchanged scope

The 64-interval, 16-query, two-family generator of the completed study,
unchanged: `d_k = d_v = 8`, 32 keys, 8 values, four query categories, the same
readout timing (the query is read **after** advancing its interval), the same
loss weighting and the same information restrictions. No easier data, no query
bypass, no noise manipulation, no position input, no changed primary metric.

Primary metric: **revision-family macro accuracy** over the four categories.

## 2. Seven arms

| # | Arm | Law | Params | Carry |
|---|---|---|---|---|
| 1 | Adaptive generalized prospective memory | `M Ẅ + γẆ + R + T Ṙ = 0`, `ν, τ, ρ` learned | 433 | 128 |
| 2 | Adaptive inertial control | `M Ẅ + γẆ + R = 0`, `η, τ` learned | 432 | 128 |
| 3 | Adaptive first-order delta | `Ċ = −ηR`, `η` learned | 431 | 64 |
| 4 | TSS finite-adaptation prospective | `τ_m ε Ẅ + R + (τ_m+ε) Ṙ = 0`, `τ_m, ε` learned | 432 | 128 |
| 5 | Ideal prospective equilibrium | minimum-change completion of `Uk = v` | 392 | 64 |
| 6 | Gated DeltaNet rule | source-pinned, unchanged | 480 | 64 |
| 7 | Momentum DeltaNet rule | source-pinned, unchanged | 569 | 128 |

Arms 1–3 and 4 carry the same 38-parameter positive source gate
`a(k,v) = 2σ((H₃₂u)_k + (H₈b)_v)`, initialised at `u = b = 0` so `a ≡ 1`.
Arm 5 carries **no** gate: a strictly positive scalar source weight changes
neither its constraint nor its update, so gate leaves there would have
identically zero gradients. Counts are **asserted in the checks**, not assumed,
and no inert parameter is added to equalise them.

Arm 4 has `γ = 0` exactly, so `M ≤ γT` is **false** at `M > 0`: it lies outside
the generalized candidate's admissible sector. No damping is added to force it
in, no code path divides by `γ`, and the `(C, Z)` incremental-storage theorem
is **not** applied to it. Its no-write motion is undamped drift
(`P' = P`, `W' = W + hP/M`), not decay — it is not memoryless.

Arm 5 is a **full-strength delta projection**, not the old full-rank
memoryless control, not the unique consequence of TSS Eq. (5) and not the
paper's finite-difference scheme. The zero-residual branch, the instantaneous
update at input jumps and the minimum-change completion are additional
specifications made here explicitly. It has no autonomous query motion.

Outer learning is full BPTT in every arm.

## 3. Calibration — declared before execution

Target, at full precision (never a rounded decimal):

```
β* = 1 − F₁₁^ref + (1 − e^{−1/(3/4)}) F₂₁^ref ,   F^ref = exp(G(ν=4/3, ρ=3/4, τ=3/4))
```

Algorithm, identical for every calibrated slot: scan `exp(linspace(−12,12,257))`
ascending; take the **first** consecutive bracket where `amplitude − β*` goes
from nonpositive to nonnegative; bisect to absolute observable error
`≤ 1e−10` in float64. An exactly matching grid point is accepted. If no bracket
exists, or the response or its derivative is non-finite, the run **stops before
training** and reports an initialization-design obstruction. The target is
never moved and the interval is never changed. This is a declared numerical
procedure, not a claim of global root uniqueness.

Observables differ by coordinate system and are **not** interchangeable:

* arms 1–2, in `(W, Z)`:  `1 − F₁₁ + (1 − e^{−h/τ}) F₂₁`
* arm 4, in `(W, P)`:     `1 − F₁₁ − (h/M) F₂₁`   (`F₂₁ < 0` here)
* arm 3: `η₀ = −log(1 − β*)` in closed form
* arm 5: **not calibrated**. Its equality constraint fixes the write strength
  at exactly 1. Reported openly rather than matched.

Slot A of arm 1 must recover `ν = 4/3` to absolute error `< 1e−8`.

### The fourteen development slots

| Family | Configuration A | Configuration B |
|---|---|---|
| 1 Generalized prospective | `τ=.75, ρ=.75`; calibrate `ν`; lr `.003` | `τ=32, ρ=.75`; calibrate `ν`; lr `.003` |
| 2 Inertial | `τ=.75`; calibrate `η`; lr `.003` | `τ=32`; calibrate `η`; lr `.003` |
| 3 First-order delta | `η=−log(1−β*)`; lr `.003` | same `η`; lr `.01` |
| 4 TSS | `ε/τ_m=0.1`; calibrate `q=1/τ_m`; lr `.003` | `ε/τ_m=0.5`; calibrate `q`; lr `.003` |
| 5 Ideal equilibrium | lr `.003` | lr `.01` |
| 6 Gated DeltaNet | official initializers; lr `.003` | same; lr `.01` |
| 7 Momentum DeltaNet | official initializers; lr `.003` | same; lr `.01` |

The axes searched differ across families. This is an **equal selection
budget**, not exhaustive or identical hyperparameter tuning, and is never
advertised as such. No extra initializations, learning rates or
best-checkpoint trials exist in this batch.

## 4. Training and the two stages

200 updates, batch 16 (8 per family), Adam `b1=.9, b2=.999, eps=1e−8`, no
decay, global gradient clipping at 1.0, production float32 on GPU. All slow
parameters in a configuration share its declared learning rate; no schedules
and no weight-group multipliers.

The learning rate is applied as a **dynamic** scalar
(`optax.adam(lr) ≡ chain(scale_by_adam, scale(−lr))`), so both declared rates
share one compilation per arm and preflight cannot silently omit a second
compilation for every configuration-B slot.

1. **Development** — train all 14 slots at initialization seed 200; evaluate
   the fixed update-200 checkpoint on 256 validation sequences per family.
   Select one configuration per family by **highest revision macro accuracy,
   then lower revision CE, then configuration A**. Held-out results and
   category-specific preferences play no part.
2. **Freeze** the seven selections in `selection.json` with all 14 development
   outcomes retained.
3. **Evaluation** — train each selected configuration from scratch at seeds
   201, 202, 203, paired streams across arms. Validation at 0/100/200 is
   descriptive only. Held-out (512/family) is evaluated **only after all 21
   final runs finish**.

35 runs, 7,000 optimizer updates. Preflight projects this entire batch plus all
remaining evaluation, compilation and serialization.

### Streams (disjoint, verified in the checks)

| Purpose | Entry point |
|---|---|
| training | `20,000,000 + seed×10,000 + update` |
| development validation | `30,000,000` |
| evaluation-seed validation | `31,000,000` |
| held-out | `40,000,000` |

The completed study used `0`, `1,000,000`, `7,000,000`, `9,000,000`. A check
asserts pairwise disjointness and no collision with those. Gate parameters are
appended after the common draw and so cannot perturb the common tables or the
data order.

The evaluation streams are fresh, but the task distribution has already guided
this proposal: **this remains a development study.**

## 5. The two screens, and attribution

Reported separately; neither substitutes for the other, in either direction.

**Literature screen** — candidate = arm 1, against Momentum DeltaNet (primary)
and Gated DeltaNet:

* mean revision macro accuracy at least **+1 point** over each;
* **positive paired primary difference in all three final seeds** against each;
* no greater than 1-point mean regression in revision-family untouched
  retention **or** overall recall-family accuracy against either.

**Ordinary-prospectivity extension screen** — the identical three conditions
against the TSS reference and the ideal equilibrium reference.

**Attribution**, separate from both: credit the prospective derivative only if
arm 1 also exceeds the equally source-gated adaptive inertial control on
primary accuracy in **all three paired final seeds** without a >1-point mean
regression on either retention check. Arm 3 is reported too; if it matches the
outcome, a simpler write-control account remains viable.

Arms 1, 2 and 4 are **independently calibrated and trained**. Their differences
compare optimized rule families; they are not causal term-removal measurements
on one trained trajectory. Every baseline here is freshly trained in this
batch — no previously reported score (including the old momentum 0.5329) is
used as a control. Neither ordinary reference is assumed memoryless or assumed
to score worse.

A failed criterion is **reported, not loosened**.

## 6. Frozen tolerances and finite-difference steps

Declared here before any numerical check ran. No step is chosen after seeing a
result; a needed repair is recorded in this file rather than silently applied.

| Name | Value | Scope |
|---|---|---|
| `EXACT64` | `1e−9` | one exact token step vs an independently assembled dense augmented-matrix exponential, float64 |
| `TRAJ32` | `2e−5` | the same over a 64-token genuine float32 trajectory |
| `GRAD64` | `1e−6` | relative, JVP vs central differences, float64, away from a near-zero reference |
| `NEAR64` | `1e−9` | **absolute** bound when the reference directional derivative is `< 1e−6` |
| `GRAD32` | `2e−2` | production float32, enforced **separately at both** declared steps |
| `NEAR32` | `3e−3` | float32 absolute fallback, which **bounds** the discrepancy rather than excusing it |
| `IDENT64` | `1e−10` | analytic limits and reductions of the law |
| `REPRO64` | `1e−10` | the completed study's trajectories at reference coefficients, gate = 1, under `Z = −P/γ` |
| `STREAM32` | `2e−5` | chunked carry vs an unsplit sequence; batched vs per-example |
| `CAL` | `1e−10` | calibration observable; `1e−8` on recovering `ν = 4/3` |
| FD steps | float64 `(1e−5, 1e−6)`; float32 `(1e−2, 3e−3)` | both enforced, each separately |

**Initial-condition convention.** Every derivative check holds the initial
carry parameter-independent. `Z₀` and `P₀ = −γZ₀` (and `P₀ = τ_m(W₀−A₀)` for
arm 4) define *different* initial-condition dependencies once the coefficients
are learned, so parameter gradients across those graphs need not agree and are
never compared. Where the TSS `(W,A) → (W,P)` mapping is used, it is applied
explicitly and differentiated.

## 7. Checks that must pass before training

Executed on the cluster, inside the same cap. A numerical failure is **FAILED**
and training does not start.

1. Exact token step vs an independent dense augmented-matrix exponential —
   prospective and inertial in `(W,Z)`; **TSS against the original `(W,A)`
   Eqs. (6)–(7)**, not the eliminated form — over multi-token sequences with
   changing keys, values, gate weights and masks, from nonzero initial states,
   at both float64 and float32.
2. TSS no-write motion is exactly `P' = P`, `W' = W + hP/M`.
3. At reference coefficients with gate = 1, reproduce the completed study's
   prospective and inertial `W` trajectories under `Z = −P/γ`, including jumps
   and idle motion.
4. The `ρ = 1` boundary with `Z₀ = 0` equals the first-order delta rule —
   an explicit boundary evaluation, not an unreachable sigmoid parameter.
5. `expm2` accuracy and **finite derivatives at the confluent root** `μ = 0`
   and on the complex-pole branch `μ < 0`.
6. JVP vs central differences for the response scalars, gate and embeddings,
   both dtypes, both steps, for both TSS timescale configurations.
7. Streaming/chunked vs unsplit, batched vs per-example, causality, genuine
   dtypes, no weakly typed leaves. Identity tests do not rest on an all-zero
   fixture.
8. Incremental storage inequality — **prospective law only**, fixed
   parameters, identical exogenous inputs. Not applied to the inertial control
   and not to TSS.
9. A real optimizer step reaches every response leaf, both gate leaves and the
   embeddings, with finite gradients and moved parameters; arm 5 is asserted to
   have no gate or response leaf at all.
10. Ideal reference: `W⁺k = v`, orthogonal-key preservation, equality to a
    delta step at `β = 1`, held query state, autodiff vs the direct projection,
    and a **nonorthogonal interference example recorded without expecting
    perfect retention**.
11. Calibration, contrast conventions (`HᵀH = I`, `1ᵀH = 0`), the
    no-query-value-source restriction, parameter/carry counts, stream
    disjointness, checkpoint restore, evaluation parity across chunk sizes, and
    that the two literature arms are bit-identical to the completed study.
12. The two screens are computed independently of each other.

## 8. Budget and status discipline

One absolute 600-second deadline covering GPU startup, checks, calibration,
compilation, preflight, all 35 runs, evaluation and a 30-second cleanup
reserve. Preflight measures all seven arms, reports **incurred** compilation
separately from **projected remaining** work, and refuses the batch if the
projection does not fit — without reducing any arm, slot, seed or update count
and without raising the cap. A retrace detected during step timing invalidates
the projection and refuses the batch.

Status is separate from the verdict:

* **FAILED** — a numerical refusal, a calibration obstruction or a non-finite
  quantity (exit 4);
* **INCOMPLETE** — the deadline was exhausted; held-out is **not** opened and
  no comparative verdict is claimed (exit 3);
* **PASS** — the declared batch completed. A completed **unfavourable**
  comparison is PASS with the screen FAILED.

No automatic retry, no longer cap, no missing-arm deletion, no shortened run,
no follow-up sweep.

## 9. Boundary of any conclusion

A passing screen would be a small-scale development signal against strong
published update rules and against two ordinary-prospective references. It
would motivate a separately declared published associative-recall benchmark and
later a full model integration. It would not establish SOTA, superior S5
accuracy, improved spatial-only credit assignment, or a biological
implementation of BPTT and the learned gate. The fixed-objective overlap with
inertial optimization under Hessian-driven damping (Alvarez–Attouch–Bolte–
Redont 2002; Attouch–Chbani–Fadili–Riahi 2019) is acknowledged: the absence of
a numerical Hessian is not itself a novelty claim.
