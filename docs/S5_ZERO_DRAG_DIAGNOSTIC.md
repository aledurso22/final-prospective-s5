# Finite inertia, zero ordinary drag in S5: equations, checks, verdict

Branch `s5-two-compartment-parallel-scan`, on top of `a01d2b2`. Local work only;
no training wave was launched. The completed experiment and its defaults are
unchanged: nothing under `s5/` that the runner imports was edited.

| new file | role |
|---|---|
| `s5/independent_inertia.py` | research constructor with independent `M`, `γ` (and `γ ≡ 0`), analysis helpers, separate exact-ZOH scheme |
| `tests/test_independent_inertia.py` | 22 tests (`JAX_ENABLE_X64=1`), all pass; existing float64 scan suites still 32/32 |
| `experiments/s5_zero_drag/diagnostic.py` | per-mode analysis on the 1152 initialized modes; `--checkpoint` for trained ones |
| `experiments/s5_zero_drag/benchmark_local.py` | CPU cost per layer |
| `results/s5_zero_drag/diagnostic.json`, `responses.png` | outputs (gitignored); `diagnostic.json` committed as `docs/evidence/s5_zero_drag/diagnostic.json` |

**Verdict: outcome 3.** This S5 formulation does not support the intended
motion-memory mechanism. The implemented discretization also obstructs zero drag,
but that is a second, independent problem. Even where the continuous law allows
zero drag, removing drag removes the mode's low-frequency lag, and in the tested
regime the stable modes became near-static. Keep the online-memory experiment
separate. See §5.

> **Scope note (added 2026-11-17).** The numerical findings below are unchanged.
> Three statements are narrowed:
> 1. **Zero low-frequency lag does not in general mean no dynamical memory.** A
>    resonance or a slow transient can remain. What was observed is narrower:
>    in the tested regime (exact scheme, initialized modes, `T = 0.05` at
>    `M = 0.5·M_c`, and `T = 5, M = 0.1`) the stable zero-drag modes behaved
>    near-statically, with `H/H(0) ≈ 1`.
> 2. **At `M = γ = 0` the continuous law is `r + T ṙ = 0`**, so
>    `r(t) = r(0) e^{−t/T}`. The "static map" `s = (g/κ)x` is the response after
>    this transient has decayed, or from a compatible initial residual
>    `r(0) = 0`. It is not the complete dynamics for arbitrary initial
>    conditions.
> 3. **The negative zero-drag result does not rule out positive-damping inertial
>    S5 variants.** `docs/S5_PREDICTOR_CORRECTOR.md` studies one such family.

---

## 1. Equation-to-code mapping

Three time bases are involved, and they are never mixed below:

1. the **S5 continuous generator** Λ, which lives in its own time units;
2. the **discrete S5 transition** λ̄, one step per token;
3. the **prospective law's continuous time**, which is *token time* with `h = 1`.

### 1.1 Native S5

| quantity | code | expression |
|---|---|---|
| generator | `s5/ssm.py:155-160` | `Λ = min(Re Λ, −1e-4) + i Im Λ` (`clip_eigs=True`) |
| step | `s5/ssm.py:219-222` | `Δ = exp(log_step)`, one per mode, log-uniform on [0.001, 0.1] at init |
| discrete transition | `s5/ssm.py:39-42` | `λ̄ = exp(ΛΔ)`, `B̄ = Λ⁻¹(λ̄ − 1)B̃` (ZOH, input held over the token) |
| recurrence | `s5/ssm.py:46-57, 73-77` | `s_t = λ̄ s_{t−1} + B̄ x_t`, first-order associative scan |

At initialization every mode has `Re Λ = −0.5` and `Im Λ ∈ {−19.86, −5.35, −1.96,
−0.43}` (HiPPO-LegS, 4 conjugate modes per block). **No mode is real.**
Across seeds 301–303 the inventory has `|λ̄| ∈ [0.9515, 0.9995]` (median 0.9949) and
`arg(1 − λ̄) ∈ [33°, 88°]` (median 75°).

### 1.2 The prospective constructor

| quantity | code | expression |
|---|---|---|
| parameters | `s5/generalized_prospective_ssm.py:16-20` | `T = softplus`, `ρ ∈ (1e-4, 1)`, `γ = softplus`, **`M = ργT`** |
| native discretization reused | `generalized_prospective_ssm.py:48-50` | the same `λ̄, B̄` as §1.1 |
| coefficients | `generalized_prospective_ssm.py:51-52` | `generalized_coefficients(λ̄, B̄, T, M, γ)`, **`h` left at its default 1.0** |
| rescaled target | `s5/discrete_recurrence.py:7-9` | `F = 1 + k(λ̄ − 1)`, `G = kB̄`, `k = T/h` |
| recurrence coefficients | `s5/discrete_recurrence.py:20-26` | below |

**The rescaled target** is `Y_t = F s_{t−1} + G x_t = s_{t−1} + k (f_t − s_{t−1})`,
where `f_t = λ̄ s_{t−1} + B̄ x_t` is the native one-step update. `Y` is a step
of size `k = T` from the current state toward the S5 target. It is neither the
S5 target itself nor the S5 generator.

**The differentiation convention.** The code does not document it, so it was
reverse-engineered from lines 22-26 and verified
(`test_backward_difference_form_with_parasitic_mass_T`). With `q = M + h(γ+T)`,
`α = M/q`, `β = h²/q` and `δ = hT/q`, the coefficients are the solution of

```
M ∇²y_t + (γ+T) ∇y_t = −y_{t−1} + u_t + T ∇u_t ,      ∇ = backward difference, h = 1
```

The derivatives are backward differences, the restoring term `−y` is **explicit**
(taken at `t−1`), and the drive is implicit at `t`. Closing the loop with
`u_t = Y_t` gives

```
a₁ = 1 + α − β + (β+δ)F     a₂ = −α − δF     c₁ = (β+δ)G     c₂ = −δG
```

**Closed form of the implemented operator** (all 1152 modes, max abs error 5.3e-15):

```
(M + T) ∇²s_t + γ ∇s_t + κ (1 + T∇) s_{t−1}  =  G (1 + T∇) x_t ,     κ = (T/h)(1 − λ̄)

q·p(z) = (M + T)(z − 1)² + γ z(z − 1) + κ((1 + T)z − T)
H(z)   = z G ((1 + T)z − T) / (q p(z))
```

### 1.3 The continuous law, and the physical mapping

The code is the discretization of the token-time law

```
M s̈ + γ ṡ = −(r + T ṙ),     r = s − Y = κ s − g x,     κ = (T/h)(1 − λ̄),   g = (T/h) B̄
⟺   M s̈ + (γ + Tκ) ṡ + κ s = g (x + T ẋ)
```

So the physical parameters are **`m = M`, `c = γ`, `τ = T`**, in units of tokens
(`h = 1` token = 1/16 000 s). They are not in S5's Λ-time. The residual is already
scaled by T: `r = (T/h)(s − f)`. If the residual is instead normalized as the
native one-step error `r̃ = s − f`, the same law reads
`(M/T) s̈ + (γ/T) ṡ = −(r̃ + T ṙ̃)`, with `m̃ = ργ` and `c̃ = γ/T`.

**The discrete and continuous operators differ by a parasitic mass `T`.** In the
continuous law, the `T ṡ` inside `T Ẏ` cancels against the `T ṡ` in `(γ+T)ṡ`. In
the code, `Y_t` is formed from `s_{t−1}`. The cancellation is therefore off by one
step and leaves `T∇²s_t`, so the effective discrete mass is `M + T`. The restoring
term is also delayed by one step.

### 1.4 The region the existing constructor can represent

`{(T, γ, M) : T > 0, γ > 0, 1e-4·γT < M < γT}`. It is open, and `M` is a
fraction of `γT`.

* **`γ → 0` forces `M → 0`** in either normalization (`M = ργT`, `m̃ = ργ`).
  Zero drag at positive mass is not representable.
* `M ≥ γT` is not representable. In float32 the sigmoid saturates, so `M = γT` is
  reachable as a boundary point.
* The tie has a physical origin (`docs/DERIVATION_TRACE.md`, NLA circuit):
  `M = Tτ_d` and `γ = G_s τ_d/κ`. Zero drag needs `τ_d = 0`, which makes `M = 0`.
  **The circuit the arm was derived from has no finite-mass, zero-drag point.**
* `s5/three_arm_recurrences.py:18-19` calls Zucchet "the exact `M=0, γ=1`
  boundary". That module is a different formulation and not the production path
  (`three_arm_factory.py` imports the three constructors above). It agrees with
  §1.5: `(M=0, γ=1)` is where the generalized law becomes first order.

### 1.5 Continuous order, discrete order, and state dimension are different things

| point | continuous law | discrete recurrence order | minimal state dimension (discrete) |
|---|---|---|---|
| general `(M, γ, T)` | 2nd-order ODE | 2 | 2 |
| `M = 0`, `γ > 0` | 1st order | 2 (parasitic `T∇²`) | 2, except where the next row holds |
| `M = T(γ − 1)` | — | 2 | **1**: the pole at `T/(1+T)` cancels the drive zero |
| `M = 0, γ = 1` | 1st order | 2 | **1**: reduces exactly to `s_t = F s_{t−1} + T x_t` (tested) |
| **Zucchet, `M = γ = 0`** | **order 0**: `(1+Tσ)κS = g(1+Tσ)X ⇒ s = (g/κ)x` | 2 | **2** (no cancellation: normalized residual 0.907) |

The massless arm has a two-step recurrence for two reasons: the parasitic `T∇²`
and the one-step-delayed restoring term. **Both come from the discretization.**
Continuous time has nothing there to integrate, since `s` is slaved to `x`. The
second root of the implemented recurrence is a numerical artefact, and it is the
one that leaves the unit disc (ρ = 1.10–1.71).

The cancellation condition is `M = T(γ − 1)`, independent of λ̄. The production
initialization sits at normalized residual 0.021, close to it, and trained `γ`
rose to about 1.1. Whether training approaches the cancellation surface is
something the trained checkpoints can answer (§2.6).

---

## 2. The independent `M`, `γ` family

`generalized_coefficients` already accepts independent `mass` and `gamma`. Only
the parameterization ties them, so the family is the existing recurrence with the
tie removed.

### 2.1 Invariants (all tested)

* **DC gain** `H(1) = G/κ = B̄/(1 − λ̄)`, the native value, for every `(M, γ, T)`.
* **One finite zero**, at `z₀ = T/(1+T)`, independent of `M` and `γ`.
* **Low-frequency group delay** `= Re(γ/κ) − 1` tokens. It is independent of `M`.
  Native S5 has `Re(1/(1−λ̄)) − 1`.
* **Continuous normalized transfer function**

  ```
  H(σ)/H(0) = (1 + Tσ) / (1 + (T + γ/κ)σ + (M/κ)σ²)
  ```

  and its slow time constant (as `M → 0`) is **`T + γ/κ`**.

So in S5, ordinary drag is the memory knob. Each mode's low-frequency lag is
`γ/κ = (γ/T)/(1 − λ̄)`, which is `γ/T` times the native lag. At initialization
`γ/T = 20`; the trained medians give 23–26.

### 2.2 Stability is computed for complex coefficients

The per-mode coefficients are complex, so real-coefficient Jury conditions do not
apply.

* **Discrete:** Schur–Cohn for complex polynomials. Monic `z² + b₁z + b₀` is stable
  iff `|b₀| < 1` and `|b₁ − b̄₁b₀| < 1 − |b₀|²`. Agrees with explicit roots on 3000
  random cases with both outcomes present.
* **Continuous:** complex Routh–Hurwitz for `σ² + aσ + b`: `Re a > 0` and
  `Re a (Re a Re b + Im a Im b) > (Im b)²`. Agrees with roots on 4000 random cases.

At `γ = 0` the continuous condition reduces exactly (tested at 0.99× and 1.01×) to

```
M < M_c = T² Re κ |κ|² / (Im κ)² = T² |κ| cos φ / sin² φ ,      φ = arg κ = arg(1 − λ̄)
```

`M_c` is infinite for a real mode and shrinks toward zero as the mode becomes
oscillatory.

### 2.3 Results on the 1152 initialized modes

**At the production horizon `T = 0.05`:**

| | value |
|---|---|
| implemented scheme, `γ = 0`, any `M ∈ [0, 1e5]` | **0 / 1152 stable** |
| implemented, `γ = 0`, `M = 0.025` (matched to production) | ρ median 1.095, max 1.522 |
| minimal `γ` the implemented scheme needs, at `M = 0 / 0.025 / 1 / 100` | median 0.023 / 0.028 / 0.10 / 0.96; max 0.087 / 0.10 / 0.32 / 2.9 |
| continuous `M_c` | median **1.2e-6**, range 6.6e-8 – 5.9e-4 (**4.7e-5 × the production M**) |
| inertial frequency at the bound, `√(\|κ\|/M_c) = sin φ /(T √cos φ)` | **12 – 106 rad/token**, all above Nyquist (π) |
| damping ratio at the bound, `sin φ √cos φ / 2` | 0.09 – 0.31 |

At production T, zero drag is stable only when the inertia is so small that its
oscillation is 4–34× faster than the token rate. **The inertia is unresolvable,
not merely small.** The exact-ZOH realization at `M = 0.5·M_c` confirms the
consequence:

* all 1152 modes are stable;
* spectral radius median **2.6e-8**;
* impulse peak / DC gain = 1.00, and step overshoot 1.00.

In this regime the mode behaves as the near-static map `s = B̄x/(1−λ̄)`, as §2.1
predicts: the slow pole sits at `−1/T = −20` per token, so the residual transient
`r(0)e^{−t/T}` dies within a fraction of a token.

**Over horizons `T` (stable modes out of 1152, `γ = 0`):**

| T | implemented, best over M | continuous, M = 1e-3 / 0.1 / 10 |
|---|---|---|
| 0.05 – 1 | **0** | ≤ 1058 / ≤ 131 / 0 |
| 5 | 59 | 1152 / 1093 / 156 |
| 50 | 391 | 1152 / 1152 / 1152 |
| 500 | 707 | all |

The continuous region grows like `T⁵` in `T` (since `κ ∝ T`). The implemented
scheme never covers more than 61% of modes at any tested `(T, M)`. Its stable
counts are nearly flat in `M` from 0 to 1, which means those modes are held up by
the parasitic mass and the explicit restoring step, not by `M`.

For the inertia to be resolvable (`ω_n ≤ 1` rad/token at the bound), a mode needs
`T ≥ sin φ/√cos φ`. That is **0.59 – 5.3 tokens**, median 1.9, so a horizon
12–100× the production T.

### 2.4 Pole–zero structure and the double unit pole

* Normalized cancellation residual `|q p(z₀)|/q`: production 0.021; zero drag and
  Zucchet 0.907; `(M=0, γ=1)` 0 (exact cancellation).
* **`κ → 0` at `γ = 0`:** `q p(z) = (M+T)(z−1)²`, a double pole at 1. The
  companion form with a repeated eigenvalue is a single Jordan block. It is
  observable from `s`, and the numerator is 1 at `z = 1`, so the pole is **not
  cancelled**. A step input grows quadratically (tested). This is a genuine
  double integrator in the transfer function, not an artefact of the
  realization.
* `clip_eigs` keeps `Re Λ ≤ −1e-4`, so `κ = 0` is never reached exactly. Near it
  the poles split to `1 ± i√ε − ε(1+T)/2` with `ε = κ/(M+T)`. Because `κ` is
  complex, one of the pair moves **outward** at order `√|ε|` while the damping
  term is only of order `|ε|`. That is the discrete instability in one line.
* Production `(γ = 1)` has distinct roots `{1, α + δ}` at `κ = 0`, so no defect.

### 2.5 Transients and gains (inventory quantiles, stable modes only)

| arm | stable | delay (tokens), median | ‖H‖∞ / DC, p95 | Nyquist / DC, median |
|---|---|---|---|---|
| Native | 1152 | 4.2 | 1.00 | 0.018 |
| Generalized init, implemented (`M=.025, γ=1`) | 1152 | 65 (analytic 105) | 1.00 | 0.0008 |
| Zero drag, implemented (`M=.025, γ=0`) | **0** | — | — | — |
| ALT exact, `T=.05, M=0.5M_c, γ=0` | 1152 | ≈ 0 | 1.00 | 1.00 (memoryless) |
| ALT exact, `T=5, M=0.1, γ=0` | 1093 | −0.08 | **4.3** (max 73) | 1.00 (max 13) |
| ALT exact, `T=5, M=0.1, γ=1` (matched control) | 1152 | 0.68 | 1.00 | 0.62 |

**Read this table carefully.** The generalized arm *adds* low-frequency lag, about
`γ/T` times native at initialization. Zero drag removes it entirely. That is a
literal memory difference, not a statement about speed.

Numerical delays come from the phase slope at ω = 7.7e-4, so they underestimate
very long lags. The analytic value is in parentheses.

### 2.6 Trained modes

They are not available locally; the checkpoints are on the cluster. Run:

```
JAX_PLATFORMS=cpu python -m experiments.s5_zero_drag.diagnostic \
  --checkpoint '/Users/durso/s5-runs/s5-two-compartment-factored/20260921-175242/generalized_prospective_s5/*/checkpoint_epoch_15.msgpack'
```

(Adjust the glob to the actual checkpoint names.) This reports, for each learned
mode: the radius, stability at zero drag with the same `T` and `M` in both
schemes, `M_c / M_learned`, `arg κ`, and the cancellation residual. On the
documented medians (`T ≈ 0.044`, `γ ≈ 1.08`, `M ≈ 0.02`), the prediction is that
zero drag is unstable on every mode in the implemented scheme, with `M_c / M`
around 1e-4 continuously, unless training moved `arg κ` close to 0.

---

## 3. The conceptual boundary

### 3.1 The S5 "Hessian" is not symmetric

The state derivative of the residual is `∂r/∂s = κ`, a complex scalar. In real
coordinates it is

```
K = [[Re κ, −Im κ], [Im κ, Re κ]]  =  Re κ · I  +  Im κ · J
```

It has a positive symmetric part and a **circulatory** (skew) part. For
`M s̈ = −(γ + Tκ)ṡ − κ s` (with `x = 0`), with the identity verified numerically to
6e-7:

```
d/dt [ ½M|ṡ|² + ½Re κ |s|² ]  =  −(γ + T Re κ) |ṡ|²  +  Im κ · Im(s ṡ̄)
```

* The convex-gradient tracking arguments rely on `r = ∇L` with a symmetric PSD
  Hessian. There the energy is non-increasing at `c = 0`, because `τ ṙ` alone
  dissipates.
* In S5 the circulatory term **injects** energy. The only dissipation at `γ = 0`
  is `T Re κ`, which is of order `T²(1 − |λ̄|)`.
* This is the flutter mechanism of undamped systems under follower forces, and it
  is exactly the `M_c` bound of §2.2.
* The skew part is `Im Λ`, the oscillation S5 itself relies on, and every
  initialized mode has it.
* **So the convex-tracking arguments do not apply.** They apply only to real
  modes, `Im κ = 0`, where zero drag is continuously stable for every `M > 0`
  (tested).

### 3.2 Zero input is not zero residual

With `x = 0`, `r = κ s ≠ 0`. S5 keeps pulling the state toward 0. In the shifted
variable `z = ṡ + (τ/m) r`, at `c = 0`:

```
ż = −r/m = −κ s / M        (with x = 0; verified numerically to 9e-7)
```

So `z` keeps changing whenever `s ≠ 0`. A persistent `z` requires `r ≡ 0` along
the trajectory, and with no input that means `s ≡ 0`. **Setting the input to zero
cannot reveal a learned persistent velocity.** It reveals the S5 restoring force
acting on `z`.

### 3.3 What removing drag changes in this system

From `ż = −(c/m)z + ((cτ − m)/m²) r`, at `c = 0` we get `ż = −r/m`. `z` integrates
the residual without leaking. Under a constant drift of the input (a ramp
`x = v t`, if stable), the steady state needs `ż = 0`, hence `r = 0`:

```
s = (g/κ) x = B̄ x / (1 − λ̄)       and       z = ṡ = B̄ v / (1 − λ̄)
```

`z` stores the drift velocity and `s` tracks with zero lag. This is the same
algebra as the online learner. **But the object being tracked is different.**

* **Online learner.** The target is an external optimum observed through a loss.
  Lag is pure error, so removing it is a gain.
* **S5.** The "target" `f = λ̄ s + B̄ x` contains the state. Its fixed point
  `B̄x/(1−λ̄)` is the static map of the current input. Tracking it with zero lag
  means the mode stops low-pass integrating the past.

The S5 mode's *low-frequency* memory is its first-order lag (§2.1). Zero drag sets
that lag to zero for every `M`, leaving `H/H(0) = 1 + O(σ²)`. Zero lag alone does
not exclude dynamical memory; the next sentences describe what remained in the
tested regime. What remains of memory is a
resonance at `ω_n = √(κ/M)`. That resonance is unresolvable at production T, and
where it is resolvable it is a lightly damped oscillation, not an integration
window. In addition, nothing in the S5 recurrence is updated by an observed
prediction error. `z` is a fixed LTI state with no adaptation, so it cannot
acquire anything the native layer stack could not already represent as a lead
filter.

---

## 4. Implementation checks

`JAX_ENABLE_X64=1 python -m pytest tests/test_independent_inertia.py -q`: **22
passed**.

| check | result |
|---|---|
| recovery on the constrained family: same backbone bits, same outputs as `GeneralizedProspectiveS5SSM` | pass (float32 rtol 1e-5) |
| `zero_damping=True`: no γ parameter exists, `γ == 0.0` exactly, `M > 0`, gradients flow to every remaining parameter | pass |
| closed form and backward-difference form vs `generalized_coefficients` (h = 1 and h = 0.37) | ≤ 1e-13 |
| factored and companion vs sequential oracle, values and gradients wrt `T, M, γ`, at `γ = 1` and `γ = 0` | rtol 1e-8 |
| near-coincident poles, relative separation 1e-14, 1e-6, 1e-3: values and gradients | rtol 1e-8 |
| pole prediction vs trajectory: partial-fraction reconstruction | < 1e-8; late-time log-growth equals log ρ to ~1e-14 (diagnostic) |
| exact-ZOH: poles = `exp(σ)`, DC gain native, equals a direct two-state simulation | ≤ 1e-10 |
| `M = 0, γ = 1` equals the first-order `F` recurrence; cancellation exactly on `M = T(γ−1)` | pass |

**Finding, recorded and not fixed** (`test_FINDING_...`): the factored scan's
forward pass is exact at repeated roots, as documented. Its **gradient is NaN when
the discriminant rounds to zero**, because `d√disc` is infinite there. Measured:
float64 at relative root separation 0 and 1e-10; float32 at 0. The oracle's
gradient is finite. This is irrelevant to the completed run, whose roots are about
0.999 and 0.05. It does matter for any arm that drives roots together, and zero
drag near `κ → 0` does. A guarded square root would fix it. That is a change to a
production module, so I have not made it.

**Local cost** (`benchmark_local.py`; CPU; one layer, `H=96, P=64`, bidirectional,
`L=4000`, batch 2, fwd+bwd):

| | ms/step | × native |
|---|---|---|
| Native | 8.4 | 1.0 |
| Generalized, sequential | 189 | 22.5 |
| Generalized, factored | 19.5 | 2.3 |
| Independent `M, γ`, factored, damped | 20.2 | 2.4 |
| Independent, factored, `γ ≡ 0` | 19.5 | 2.3 |
| exact-ZOH coefficient setup (64 × 3×3 expm, fwd+bwd) | +1.5 per forward | — |

---

## 5. Recommendation

**Outcome 3: this S5 formulation does not support the intended motion-memory
mechanism.** Keep the online-memory experiment separate.

1. **In the implemented discretization, zero drag is unstable on every initialized
   mode for any `M ∈ [0, 1e5]` at `T ≤ 1`.** Stable subsets appear only at
   `T ≥ 5`, never above 61%, and there `M` barely matters. The obstruction is the
   parasitic mass `T` plus the explicit restoring step (§1.3). An exact
   discretization removes this obstruction, but that alone is not enough.
2. **The continuous law allows zero drag only below a flutter bound**,
   `M < T²|κ|cos φ/sin²φ`, because S5 modes are complex and their restoring force is
   circulatory. At production T the bound puts the inertial frequency at 12–106
   rad/token, so the inertia is unresolvable. That is "stable only because inertia
   is negligible", quantified at about 5e-5 × the production `M`.
3. **Where zero drag is stable and resolvable** (exact scheme, `T ≳ 2–5` tokens),
   the result is still not motion memory. Zero drag sets the low-frequency lag to
   zero for every `M`, and in S5 that lag is the mode's low-frequency memory
   (§2.1, §3.3). In the tested regime the stable modes became near-static maps (`H/H(0) ≈ 1`, radius ≈ 0.82 at `T = 5`, slow
   time constant `T`). The matched damped control at the same `T` and `M` keeps
   memory (delay 0.68 vs −0.08 tokens; it low-passes where the zero-drag version
   is flat).
4. The convex-gradient tracking theory needs a symmetric positive Hessian and an
   observed error signal. S5's residual has a skew part on every mode and no
   observed error (§3.1).

**What would make a zero-drag arm meaningful** is a different experiment, not a
retune of this one:

* a residual that is a genuine **tracking error with symmetric positive Jacobian**,
  such as the drive placement, or an online learner whose state is a parameter
  estimate; or
* an S5 restricted to **real modes** (`Im Λ ≡ 0`, frozen), where zero drag is
  continuously stable for every `M`. The current discretization still needs
  checking there, and the memory-erasure argument of §3.3 still applies.

This negative result is about **zero** drag. It does not rule out
positive-damping inertial S5 variants.

If inertia is to be studied in S5 at all, the meaningful knob is **low, not zero,
drag at independent M**. The implemented scheme needs only `γ ≥ 0.02–0.3` at
`M ≤ 1` (§2.3), and `γ/T` sets each mode's memory multiplier. That question is
about memory timescales, and it should be framed that way rather than as motion
memory.

---

## 6. Corrections to the previous summary, and provenance

**Overstatements, corrected.**

1. **The numbers are selected validation results.** Each seed reports the
   validation accuracy at the epoch selected on that same validation split, and no
   test split was evaluated. They are optimistically biased, and the bias is not
   guaranteed to be equal across arms.
2. **A shared, non-speaker-disjoint split does not prove the between-arm comparison
   is unaffected.** Speaker leakage can reward architectures unequally, for example
   one that fits speaker-specific features better. The split controls for data
   differences, not for this interaction.
3. **"Damping matters, mass does not" holds only for the tested settings.** It was
   true at `T = 0.05` and `γ = 0` for the masses tried, and in the implemented
   scheme. Mass changes the damping needed (median `γ_min` goes from 0.023 to 0.96
   as `M` goes from 0 to 100). The continuous law's zero-drag stability depends on
   `M`. At large `T`, mass changes which modes are stable.
4. **The MDN result is not clean corroboration.** It has frontend confounds, so it
   cannot serve as an independent replication of the S5 conclusion.

**Scan provenance (resolved from repo evidence).** The reported generalized numbers
(`20260921-175242`, commit `d845663`) used the **factored** scan:

* `allocation_s5_two_compartment_factored.sh:40` defaults to `IMPLEMENTATION=factored`;
* `:139` passes it to the runner;
* `:196-203` refuses to start training unless the smoke run recorded that scan;
* `runner.py:546` writes `scan_implementation` into every result.

The "generalized ran `scan_companion_sequential`" statement in
`S5_THREE_ARM_REFERENCE.md` describes the earlier, aborted three-arm wave. Zucchet
ran `scan_companion` (`prospective_ssm.py:27`) and produced no number. Definitive
check on the cluster:
`grep -h scan_implementation /Users/durso/s5-runs/s5-two-compartment-factored/20260921-175242/*/*/task_result.json`.

**Runtime provenance.**

| figure | origin | status |
|---|---|---|
| 155 h | projection from the real sequential run at 2.98 steps/min | measured rate, projected hours |
| 117.7 h | sequential benchmark, 3.42 steps/min, × a **guessed** 536 steps/epoch | retracted (scan doc §8) |
| 1.46 h | factored benchmark × the same guessed 536 | **retracted**, but still printed in reference §5.5 and scan doc §8–10 |
| **≈ 4.3 h** | **measured**: ≈17 min/epoch, three seeds concurrently on three RTX 3090 | the number to use |
| 80.5× | per-step benchmark ratio | sound |
