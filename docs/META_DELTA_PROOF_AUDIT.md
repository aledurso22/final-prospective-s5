# Independent audit of the meta-delta analytical note

Audited: `GENERAL_PROSPECTIVE_META_DELTA_AUDIT_2026_09_16.md`.

Method: hand derivation only. No numerical, symbolic-software or model
execution was performed. The corresponding numerical corroborations are
declared as cluster checks in `tests/test_meta_delta.py` and have not run.

**Verdict: no discrepancy found in the algebra.** Two points about executed
precision needed implementation care; they are recorded in §8.

## 1. Coordinate identities, eqs. (2)–(3)

Start from `P = M Ẇ + T R`. Then `Ṗ = M Ẅ + T Ṙ`. Using (1),
`M Ẅ = −γẆ − R − TṘ`, so `Ṗ = −γẆ − R`.

With `Ẇ = (P − TR)/M` this gives

```
Ṗ = −γ(P − TR)/M − R = −γP/M + (γT − M)R/M.    ✓
```

For `C = W + P/γ`:

```
Ċ = (P − TR)/M − P/M + (γT − M)R/(γM)
  = R(−γT + γT − M)/(γM)
  = −R/γ.    ✓
```

For `Z = −P/γ`:

```
Ż = P/M − (γT − M)R/(γM)
  = −γZ/M − (γT − M)R/(γM)
(M/γ)Ż = −Z − (γT − M)R/γ².    ✓
```

The implementation's `(e, z)` generator uses `ν = T/M`, `τ = M/γ`,
`ρ = M/(γT)`, and `κ/τ = ν(1 − ρ) = (γT − M)/(γM)`. This matches for every
`ρ > 0`: the generator carries no hidden `ρ < 1` assumption. When `ρ > 1` its
`(2,1)` entry changes sign, and complex poles become possible. The existing
`expm2` has an oscillatory branch that is already covered by the completed
checks.

## 2. Filtered solution and signs, eq. (4)

Variation of constants on (3) with `Z(0) = 0` gives eq. (4). ✓

Suppose that, along one component, `R < 0` throughout the write. Then:
* `M < γT`: `Z > 0`. During idle `R = 0` and `Z` decays, so `W = C + Z`
  decreases toward `C`.
* `M > γT`: `Z < 0`, and `W` increases toward `C` during idle.
* `M = γT`: `Z ≡ 0`.

All three hold only under the stated residual sign. ✓

## 3. Delta boundary, eq. (5)

At `M = γT` with `P₀ = 0`, `Ṗ = −γP/M` keeps `P ≡ 0`, so `Ẇ = −R/γ`. For a
held unit key, `e = Wk − v` obeys `ė = −(w/γ) e`, so
`e(h) = e·exp(−hw/γ)`. Because `Ẇ` is proportional to `kᵀ`, `W` is unchanged
on `k⊥`. Hence `W⁺ = W + (1 − e^{−hw/γ})(v − Wk)kᵀ`. ✓

**Implementation check (static).** At `raw_r = 0`, `ρ = exp(0) = 1.0` exactly
in floating point, so `1 − ρ = 0` exactly. The generator's `(2,1)` entry is
then exactly zero, and in `expm2`, `F21 = h·S·n21 = 0`, so `Z` stays exactly
zero from `Z₀ = 0`. The `W` update reduces to
`W − (1 − F11)(Wk − v)kᵀ`. `F11` comes from the closed-form 2×2 exponential
rather than `−expm1`, so float32 identity with the delta arm is exact only up
to rounding; this is a declared production tolerance. The float64 check is at
1e-9.

## 4. Residual-velocity integral, eq. (7)

`M ë + c ė + λe = 0` with `c = γ + Tλ > 0`, `λ > 0` and `M > 0` is
asymptotically stable.

* Multiplying by `ė` and integrating gives
  `c∫ė² = ½Mv₀² + ½λe(0)² = ½Mv₀²`.
* Multiplying by `e` and using `e ë = (eė)' − ė²` gives
  `M[eė]₀^∞ − M∫ė² + (c/2)[e²]₀^∞ + λ∫e² = 0`. Every boundary term vanishes,
  because `e(0) = 0` and the solution decays. So `λ∫e² = M∫ė²`.

Therefore `∫e² = M²v₀²/(2λ(γ + Tλ))`, and the ratio to `T = 0` is
`γ/(γ + Tλ)`. ✓

Witness `(γ, T, M, λ) = (1, ½, ¾, 1)`: `ρ = 3/2` and the ratio is `2/3`. ✓

Scope, as the note states: this is one fixed mode with a zero initial error.
It is not a retrieval claim and not a comparison with the gated Momentum
DeltaNet rule.

## 5. Switching storage, eqs. (8)–(12)

**(8).** `D = wX kkᵀ`. Then `⟨X, D⟩ = w‖Xk‖² ≥ 0` because `w ≥ 0`, and
`‖D‖² = w²‖Xk‖²‖k‖² ≤ L·w‖Xk‖²` because `w ≤ L` and `‖k‖ ≤ 1`. ✓

**Difference dynamics.** `Ẋ = (Y − TD)/M` and `Ẏ = −γY/M − dD/M`, with
`d = M − γT`. ✓

**V̇.** Since `γT + d = M`, `Ẋ + Ẏ/γ = −D(γT + d)/(γM) = −D/γ`. Hence

```
V̇ = γ⟨X + Y/γ, −D/γ⟩ + (M/(γd))⟨Y, −γY/M − dD/M⟩
  = −⟨X,D⟩ − (2/γ)⟨Y,D⟩ − ‖Y‖²/d.    ✓
```

Completing the square gives
`−(2/γ)⟨Y,D⟩ − ‖Y‖²/d = −(1/d)‖Y + (d/γ)D‖² + (d/γ²)‖D‖²`. With (8), this
yields (11). ✓

**V** is positive definite when `γ > 0`, `M > 0` and `d > 0`. It is singular
at `d = 0`, so the delta boundary is handled separately by the non-expansive
exact delta step. For `d < 0` the earlier storage applies.

**Jumps.** The implementation carries `(W, Z)` with `Z = −P/γ` and `γ`
constant within an episode, so `P` is continuous. `V` depends only on
`(X, Y)`, so a gate or source jump at a token boundary cannot make it jump.
The exact held-token steps are samples of the flow, so they inherit
monotonicity. ✓

**In ρ.** `M < γT + γ²/L` is equivalent to `ρ < 1 + γ/(TL)`. ✓

**In the executed coordinates** `(η, τ, ρ)`, with `γ = 1/η`, `M = τ/η` and
`T = τ/ρ`:
`d = (τ/η)(1 − 1/ρ)`, and `dL < γ²` is equivalent to `1 − 1/ρ < 1/x` with
`x = ητL`. For `x ≤ 1` this holds for every `ρ > 1`; for `x > 1` it holds iff
`ρ < x/(x − 1)`. Derived here; not stated in this form by the note. ✓

## 6. Hypotheses against the executed code

| Hypothesis | Executed fact | Holds |
|---|---|---|
| `0 ≤ w ≤ L`, `L = 2` | `a = 2·sigmoid(·)` times a {0,1} write/valid mask. In float32, sigmoid can round to 1, giving `w = 2 = L`, which (8) permits | yes |
| `‖k‖ ≤ 1` | `safe_normalize`: `k/‖k‖`, or `k = 0` with `w = 0` below the floor. `‖k‖²` can round a few eps above 1, so the effective `L` is a few eps above 2 | yes, up to rounding; absorbed by the projection's inflation (§8) |
| Constant coefficients per episode | response leaves are parameters, fixed during a rollout | yes |
| `W`, `P` continuous at jumps | `(W, Z)` carried; `Z = −P/γ` with constant `γ` | yes |
| Same exogenous inputs | keys, values and gates are functions of observed ids only; `sanitize_episode` removes query values | yes |
| `R = w(Wk − v)kᵀ` | the rank-one step's `e = Wk − v`, `w` inside the generator | yes |

## 7. Scope points confirmed, not changed

* **Region:** `ρ > 1` is a computational use of the same mechanical equation.
  It is **not** the passive two-compartment circuit, which gives
  `M ≤ γT` only.
* **Bound:** sufficient, not maximal. It is proven for identical inputs and
  fixed coefficients only.
* **TSS Eq. (17):** the `M = γ = 0` boundary, outside the certified
  `γ > 0` sector.
* **Momentum DeltaNet:** not nested.

## 8. Precision points found while implementing (not errors in the note)

1. **Bound near `x = 1`.** The projection `raw_r ≤ U`, with
   `U = −log1p(−1/x)`, is ill-conditioned as `x → 1⁺`. A relative rounding
   error `e` in `x` moves `U` by about `e/(x−1)`, which a
   `32·eps·(1+|U|)` margin does not cover near `x = 1`. Since `U` decreases
   in `x`, the implementation first inflates `x` by the representable factor
   `(1 + 16 eps)`, then applies the margin, keeping `ρ = 1` as the fallback.
   A float32 probe exercises the edge from `x − 1 = 1e-2` down to about
   `1.5e-7`.
2. **Float32 delta identity.** It holds only up to rounding (§3). It is gated
   at a declared production tolerance, not claimed bitwise.
