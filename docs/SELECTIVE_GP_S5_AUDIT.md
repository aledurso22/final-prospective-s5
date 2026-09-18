# Selective-residual generalized-prospective S5: static audit

18 September 2026. Branch `selective-gp-s5-audit`, based on `11f58f6`
(`prospective-readout`); `docs/PROSPECTIVE_READOUT_REPORT.md` is unchanged and
its INDETERMINATE verdict stands. **No training, no production
implementation, no cluster launch.** The only execution was the exact
rational-arithmetic checks in `tests/test_selective_gp_s5_audit.py` (Python
`fractions`, no JAX, no model), all 13 passing locally; they are written to run
unchanged on the cluster.

## 0. Answers

**1. Did we already test the token-dependent residual operator?** **No.** Every
executed S5 study used a residual whose state coefficient was fixed across
tokens: `r = J s − B x` with `J = −ΔΛ`, learned globally, constant within a
sequence. A token-dependent generalized S5 **was implemented and never run**:
`s5/adaptive_circuit.py` (arms `gp_adaptive_mass`, `gp_frozen_adaptive`,
`ordinary_adaptive`). It modulated `(M, γ, T)` through one parameter-free
conductance, not the residual gain. The closest executed relative is the
Momentum-DeltaNet line, where a fixed-coefficient generalized filter processed
a token-dependent delta-rule residual. There the filter's own state
coefficient was fixed at `I` (§1.4).

**2. Does it provide a capability that a compute-matched generic selective
two-mode SSM does not?** **No.** Per mode and per token, the proposed cell is
`ż = A₀z + u·(G_t e₁ᵀz − b_t)`: a fixed 2×2 system `A₀` driven by one scalar
innovation that enters along one fixed direction `u` (§4). A generic selective
2×2 block contains it exactly. It does contain something a *diagonal*
selective SSM (two Mamba/S7-type channels) lacks: its per-token transitions
never commute (§4.3). That property belongs to block selectivity in general,
not to the prospective structure. What is specific to the structure is a
constrained parameterization with a clean stability theorem (§6), and the
theorem does not carry over to complex S5 modes.

**Classification: `DISTINCT_PARAMETERIZATION_OF_KNOWN_SELECTIVE_SSM`.**
**Recommendation: `NO-GO`** (§10).

## 1. What we already tested: the executed record

Established from code paths and arm tables, not from names. The S5 files cited
are byte-identical at this branch's base `11f58f6` and at the executed commit
`d07380c` (timescale study). The `gp_rho` generator is also byte-identical at
the recall commits `b5d7211`/`fb16614`; later changes to `s5/gp_fixed.py` and
`s5/rawat_s5.py` add and register new arms and leave the generator code unchanged.

### 1.1 The residual actually used

| Arm (executed in) | Law | Residual | State coefficient | Learned | Token-dependent |
|---|---|---|---|---|---|
| `gp_scalar`, `gp_diagonal` (integration smokes; `s5/gp_ssm.py`, `s5/gp_coefficients.py`) | `γ ṡ + r + T ṙ = 0`, `γ = 1`, `M = 0` | `r = J s − B x`, `J = −F0 = −diag(ΔΛ)`, `B = ΔB̃` | `J` | `Λ, B, Δ` and `t = softplus(raw) ≥ 0`, global | nothing but `x_t` entering linearly |
| `gp_fixed_m0`, `gp_fixed_mass` (Stage 1/2; `s5/gp_fixed.py`, `s5/rawat_s5.py:ARMS`) | `M s̈ + γ ṡ + r + T ṙ = 0`, `T = 5`, `γ`, `ρ` frozen (`s5/physical_coefficients.py`) | same, with `B_c = diag(α) B̃`, `α = −Re λ` | `J` | only native S5 parameters; the runner refuses a response parameter | none |
| `gp_rho`, `gp_rho_frozen` (recall study, `experiments/gp/recall_study.py`) | `ρT s̈ + ṡ + r + T ṙ = 0`, `T = 5`, `γ = 1` | same | `J` | `ρ_i` per mode, global | none |
| `gp_rho_T`, `gp_rho_T_fixed` (timescale study, `experiments/gp/timescale_study.py`) | same, with `T_i = 5·e^{η_i}` | same | `J` | `ρ_i`, `T_i` per mode, global | none |
| `gp_rho_prospin` (combined study) | same recurrence, drive `x + T_in ẋ` | same | `J` | `ρ_i` | none |
| `alpha_p_s5`, `gain_clip_s5`, `native_s5` | Rawat two-tap / one-tap | native S5 | `Λ` | native | none |

`docs/LEARNED_RESPONSE_TIMESCALE_PROTOCOL.md` §2 states it outright:
coefficients "are **constant within each sequence** … No activity-dependent
conductance, token gate, meta-learner or new plasticity rule."

### 1.2 Answers to the five questions

1. **Residual:** `r = J s − B x`, `J = −diag(Δ_i Λ_i)` (the clipped native S5
   pole, clock absorbed once), `B = diag(Δ) B_c`, `B_c = diag(α) B̃` on the
   gain-scaled substrate. Same sign convention as the proposal (`r = G s − b`
   with `G = J`, `b = B x`), so no convention change is needed.
2. **Fixed across tokens:** yes, in every executed arm.
3. **Input-dependent state coefficient in an executed experiment:** no. In an
   implemented but unexecuted one, yes: `gp_adaptive_mass` makes the
   generator's `s` coefficient `−γ₀J/M_t` and its damping
   `−(γ_t + T_tγ₀J)/M_t` token dependent through `q_{p,t} = f(|β_p·x_t|)`
   (`s5/adaptive_circuit.py:adaptive_generator`). `ordinary_adaptive` rescales
   the first-order pole by `G_d(q_t)/G_d0`, which makes it a parameter-free
   diagonal selective S5. `docs/CONSTRAINED_PROSPECTIVE_RESPONSE_PROTOCOL.md`
   §0: "Nothing from that direction was ever executed numerically — no cluster
   job was submitted from it."
4. **Algebraically equivalent to `G_t(s − ŝ_t)`:** no. The adaptive arm keeps
   the residual gain `γ₀J` fixed and varies `(M_t, γ_t, T_t)` jointly through
   one tied scalar. Rewriting it with a fixed `(M, γ, T)` and a varying `G_t`
   would need both `T_t = T` and `γ_t/M_t = γ/M`, which the tied coefficients
   violate whenever `q` varies.
5. **Fixed / learned globally / token-conditioned:**
   - Fixed by declaration: `γ = 1`, `T = 5` (except `gp_rho_T`), the input
     gain tie `α = −Re λ`, and pole clipping.
   - Learned globally: `Λ`, `B̃`, `C`, `D`, `Δ`, `ρ_i`, `T_i`.
   - Token-conditioned: **nothing** in any executed S5 arm. In the unexecuted
     adaptive arms, only `q_t`, with no learned parameters.

### 1.3 Why the doubled-mode control could win

With every coefficient constant, each generalized mode is LTI, with transfer
`(1 + Tp)/(M p² + (γ + TJ) p + J)`: two poles and one zero. For distinct
poles, this is a partial-fraction sum of two first-order modes, so a native
diagonal S5 with `2P` modes contains it. At exceptional points (coalescing
poles) the containment holds only as a limit. This is consistent with
`ordinary_2x` beating `gp_rho` in every recall seed by a mean of 5.03 pp, at
equal carry (`docs/PROSPECTIVE_MEMORY_RECALL_REPORT.md`).

### 1.4 The closest executed relative: the Momentum-DeltaNet processing line

`experiments/prospective_momentum/filtered.py` (branches `prospective-tss-containment`,
`prospective-temporal-response`, `prospective-retention-aware`) executes

    R_t = m_t (α_t W_prev k_t − v_t) k_tᵀ
    M ÿ + γ ẏ + (y − R_t) + T (ẏ − Ṙ_t) = 0          (discretized, h = 1)
    U = μ_t U + η_t y,   W = α_t W − β_t U

Here the generalized law's residual is `y − R_t`, so its state coefficient on
`y` is the **fixed identity**. The token-dependent gain (`α_t k_t k_tᵀ`) acts on
the *backbone* `W`, through the target `R_t`. So we have executed "a
fixed-coefficient generalized filter applied to a token-selected delta-rule
residual". None of those runs produced a clean positive: containment
generalized − literal TSS was +1.45 pp revision with −0.96 retention; temporal
response +0.09; retention-aware verdicts UNAVAILABLE; the readout study came
out INDETERMINATE, with the carry-free lookahead the best predictor. We have
not executed a selective gain on the filter's own state.

## 2. Old versus new: the equation map

| | Executed fixed GP-S5 (`gp_rho_T`) | Proposed selective-residual GP-S5 |
|---|---|---|
| Law | `M s̈ + γ ṡ + r + T ṙ = 0` | same |
| Residual | `r = J s − B x_t` | `r_t = G_t s − b_t`, `b_t = G_t ŝ_t` |
| State gain | `J = −ΔΛ`, fixed within the sequence | `G_t`, a function of `x_t` |
| Target | `J⁻¹B x_t` (implicit) | `ŝ_t`, a function of `x_t` |
| `M, γ, T` | `M = ρ_iT_i`, `γ = 1`, `T_i`, learned globally | fixed, learned globally, stably parameterized |
| Carry per stored mode | `(s, v)`, 4 real | `(s, P)`, 4 real |
| Discretization | exact ZOH, once per sequence | exact ZOH, per token |
| Scan | 2×2 block scan with constant blocks | 2×2 block scan with per-token blocks |
| Class | LTI | linear time-varying, input-selected |

S5-compatible choice, recommended if anything is built:
`G_t = g_t J`, with `g_t ∈ [g_min, g_max]` a real positive per-mode gate from
`x_t` shared with the conjugate partner. This is the "quadrature sharing" that
`s5/adaptive_circuit.py` already argued for. With `ŝ_t = J⁻¹B x_t` it gives
`b_t = g_t B x_t`, so `r_t = g_t (J s − B x_t)`: **the gate multiplies the
executed residual.** At `g_t ≡ 1` this is exactly the executed `gp_rho_T`
(§3.3).

## 3. Continuous derivation

### 3.1 The jump-free coordinate

With `M > 0` and `M, γ, T` constant, define `P = M ṡ + T r`. Differentiating,

    Ṗ = M s̈ + T ṙ = −γ ṡ − r

This contains **no derivative of `G_t` or `b_t`**. So `P`, like `s`, is
continuous when `G_t` and `b_t` jump at a token boundary, while `ṡ` jumps by
`−T(r⁺ − r⁻)/M`. Then

    ṡ = (P − T r)/M
    Ṗ = −(γ/M) P + ((γT − M)/M) r

and substituting `r = G_t s − b_t`:

    d/dt [s; P] = [[ −(T/M)G_t ,  1/M  ],        [s; P]  +  [  (T/M) b_t ]
                   [ ((γT−M)/M)G_t, −γ/M ]]                  [ −((γT−M)/M) b_t ]

Every sign agrees with the brief and with the repository's `r = J s − B x`
(checked in `test_sP_realization_satisfies_the_generalized_law` and
`test_P_is_the_conserved_form_so_the_state_is_continuous_at_switches`).

This settles the concern `s5/adaptive_circuit.py` raised against
time-dependent `(s, w)` blocks. That module varied `M` and `T`, which leaves
`−T(A′s + B′w)` terms. Here `M` and `T` are fixed and only `r` varies, so
`(s, P)` needs no impulsive correction.

### 3.2 Structure

Writing `u = [−T/M, (γT − M)/M]ᵀ` and `A₀ = [[0, 1/M], [0, −γ/M]]`:

    ż = A₀ z + u r_t,      r_t = G_t e₁ᵀ z − b_t
    A(G) = A₀ + G u e₁ᵀ,   det A(G) = G/M,   tr A(G) = −(TG + γ)/M

This is a fixed LTI pair `(A₀, u)` closed through a token-gated scalar feedback
from the position `s`. It is the continuous-time α–β (constant-velocity Kalman)
form, with a fixed gain *ratio* and a token-scheduled gain *magnitude*.

### 3.3 Exact link to the executed arm

With `γ = 1`, `M = ρT`, `G ≡ J`, `b = B x`, the `(s, P)` system equals the
executed `mass_block_generator` `(s, v)` block of `s5/gp_fixed.py` under the
constant rescaling `P = −T(1 − ρ) v`, for `ρ < 1` (checked:
`test_executed_gp_rho_arm_is_the_constant_gain_member`). At `ρ = 1` the
repository's `v` is unobservable and `P ≡ 0`; both give native S5.

## 4. Discrete update, scan and cost

### 4.1 Discretization

Every generalized module in the repository rejects non-ZOH configurations
(`GPSSM.setup`, `SubstrateSSM.setup`: "ZOH only"), so **exact ZOH is the one to
implement**. With `g_t` and `x_t` held over the unit interval:

    z_k = Ā_k z_{k−1} + Φ_k f_k,   Ā_k = e^{A_k},   Φ_k = ∫₀¹ e^{A_k τ} dτ = φ₁(A_k)

For a 2×2 matrix, with `τ = tr/2` and `δ² = τ² − det`:
`e^{A} = e^{τ}[cosh δ · I + sinhc δ · (A − τI)]`. This is entire in `δ²`, so it
stays smooth through exceptional points, and `φ₁(A)` follows by the same
Cayley–Hamilton reduction. `det A_k = G_k/M → 0` makes `A_k` singular as the
gate closes, so `φ₁` must be formed without `A⁻¹`, as the repository already
does with augmented exponentials and series switches. Per token this replaces
the executed per-mode 4×4 `expm`, which ran once per sequence, with a
closed-form 2×2 evaluation at every token.

Bilinear, for completeness: `Ā_k = (I − A_k/2)⁻¹(I + A_k/2)`,
`Γ_k = (I − A_k/2)⁻¹`, `z_k = Ā_k z_{k−1} + Γ_k f_k`. It is rational, maps
Hurwitz to Schur, preserves the Lyapunov function of §6, and is what the exact
counterexample uses. It is not what the repository runs.

### 4.2 When it remains an associative scan

`z_k = Ā_k z_{k−1} + c_k` composes as `(Ā, c)∘(Ā′, c′) = (ĀĀ′, Āc′ + c)`, the
existing `block_binary_operator`. It is associative for *any* per-token
matrices; what matters is whether they can be precomputed, and at what size.

| `G_t` | Scan? | Element size and combine cost |
|---|---|---|
| diagonal (one gate per mode), from `x_t` only | **yes**, parallel | 2×2 per mode; `block_binary_operator` unchanged |
| block-diagonal (`k` modes coupled), from `x_t` only | yes | `2k×2k` per block, `O(k³)` per combine; fine for small `k` |
| low-rank across all `P` modes, from `x_t` only | associative, but the ZOH exponential of `A₀⊗I + (ue₁ᵀ)⊗G_t` is dense | `O(P³)` per combine and `O(LP²)` memory. DeltaNet-style efficiency needs a discrete-first identity-plus-low-rank transition, which a ZOH exponential does not provide |
| arbitrary dense, from `x_t` only | associative, same obstacle | `O(P³)`, impractical |
| depends on `s_t`, `P_t` or `|r_t|` | **no**: the per-token map is nonlinear in the state | sequential `O(L)` depth, like a GRU/LSTM |

So the brief's expectation is confirmed. An input-only diagonal or small-block
`G_t` gives a time-varying affine block scan; state-conditioned `G_t` loses the
parallel scan.

### 4.3 Non-commutation

`[A(g₁), A(g₂)] = (g₂ − g₁)[A₀, ue₁ᵀ]`, and
`[A₀, ue₁ᵀ] = [[u₂, −u₁], [−γu₂, −u₂]]/M` vanishes only if `M = 0`. So for
every admissible `M > 0`, two tokens with different gains have non-commuting
transitions (`test_transitions_at_different_gains_do_not_commute`), and no
fixed basis diagonalizes the cell. It is therefore not a diagonal selective
SSM. It is a member of the per-token 2×2 block class.

### 4.4 Complex modes, conjugates, real outputs, initialization

- **Complex modes.** With `G_t = g_t J` and a real `g_t` shared by the
  conjugate partner, the implicit partner evolves with conjugated
  coefficients. `y = 2Re(C̃ s) + D x` therefore remains exact, exactly as in
  `mass_readout`. A complex per-mode gate would also preserve real outputs,
  but it takes the cell further from the stability result of §6.
- **Initialization.** `P(0) = 0`, `s(0) = 0`. Gate bias set so `g_t ≡ 1`, with
  zero input weights, starts from exactly the executed `gp_rho_T_fixed`
  function. Starting at `ρ = 1` makes `T` unidentifiable (`dG/dT = 0` there,
  as the timescale study recorded), so `ρ₀ = 0.75`.
  `g_t ∈ [g_min, g_max]` with `g_min > 0` (§6).
- **State and compute**, per layer and stored complex mode:

  | | Native S5 (LTI) | Diagonal selective (first-order) | Proposed |
  |---|---|---|---|
  | carry (real) | 2 | 2 | **4** (same as `gp_rho`) |
  | scan element (complex numbers per token) | 2 | 2 | **6** (2×2 + 2-vector) |
  | combine (real flops, counted as 6 per complex multiply, 2 per add) | 14 | 14 | **≈ 88**, about 6× |
  | per-token discretization | none | 1 complex `exp` | closed-form 2×2 `exp` and `φ₁` |
  | extra parameters | none | gate projection `P×H + P` | gate projection `P×H + P` |

## 5. Recoveries and equivalences

| Target | Exact? | How |
|---|---|---|
| Fixed GP-S5 already tested (`gp_rho`, `gp_rho_T`, `gp_fixed_mass`) | **yes** | `G_t ≡ J`, `b_t = B x_t`, `γ = 1`, `M = ρT`; coordinate change `P = −T(1−ρ)v` (§3.3) |
| `gp_fixed_m0`, `gp_diagonal` (M = 0) | **yes, with a separate realization** | `(s, P)` is singular at `M = 0`. Use `Q = γ s + T r`: `Q̇ = −r`, `Q̇ = −(G/(γ+TG)) Q + (γ/(γ+TG)) b`, `s = (Q + T b)/(γ + TG)`. `Q` is also continuous across switches. Not a smooth limit of `M → 0` |
| Native S5 | **yes** | `M = γT` (ρ = 1), `G_t ≡ J`, `P(0) = 0`: the `P` row loses its forcing, `P ≡ 0`, `ṡ = −(J s − B x)/γ` |
| Ordinary prospective / literal TSS | **yes, degenerate** | `M = 0`, `γ = 0`: `Q̇ = −Q/T`, `s = Q/(TG_t) + ŝ_t`. `Q` is autonomous, so **the gate can rescale but cannot create memory** at the TSS boundary (the matched-TSS "zero driven history" result, now for any `G_t`) |
| Rawat prospective input `ṡ = −(Js − Bx)/γ_R + (τ/γ_R) Bẋ` | **only for a real, constant gain** | `M = 0`, `T = τ`, `γ = γ_R − τG` matches the transfer function exactly (`test_rawat_is_the_m0_member_only_for_a_real_constant_gain`). For a complex S5 pole that `γ` is complex, and for a varying `G_t` it would have to vary. Not recoverable at `M > 0`: the only pole–zero cancellation, `ρ = 1`, also removes the zero. The repository composes the two explicitly instead (`gp_rho_prospin`) |
| First-order input-gated innovation | **yes, for any gate sequence** | `M = γT`, `P(0) = 0`: `ṡ = −(G_t/γ)(s − ŝ_t)` (`test_cancellation_line_is_a_first_order_selective_update`). Under ZOH, `s_t = e^{−g_t/γ}s_{t−1} + (1 − e^{−g_t/γ})ŝ_t`: minGRU with `z_t = 1 − e^{−g_t/γ}`, and Mamba's continuous law `ḣ = Δ_t(Ah + B_t x_t)` with `G_t = −Δ_tA` |
| Generic two-mode selective SSM | **contained in it, strictly** | Choose `A_t = A₀ + G_t ue₁ᵀ`, `B_t = −u b_t`. The generic block has up to 4 transition and 2 input degrees of freedom per mode per token; the cell has 1 gate and 1 target, both along the fixed `u` |

**The decisive question.** The cell is a *constrained parameterization* of a
two-state selective SSM. Relative to a compute-matched generic selective 2×2
block, it adds no function the block cannot represent. Relative to a
state-matched *diagonal* selective SSM, it adds non-commuting per-token
transitions, but so would any generic block. Any claim for it must therefore
be inductive bias, stability, optimization or interpretability, never
expressivity.

## 6. Switching stability

**Per token.** `A(g)` is Hurwitz iff `g > 0` and `γ + Tg > 0`, from
`det = g/M` and `tr = −(Tg + γ)/M`. The fixed-coefficient Jury gate of the
readout and containment studies says nothing here: it certifies single rounded
polynomials, and its own docstring excludes "switching across tokens".

**Per-token stability does not imply stability under switching.** Take
`γ = 1`, `T = ½`, `M = 1` (`ρ = 2`), the bilinear map, and a gain alternating
1/64, 16, 1/64, 16, … Each block is Schur-stable, yet the two-token product
has `|tr| > 1 + det` with `|det| < 1`, so it has a real eigenvalue below −1 and
the state grows without bound. This is exact rational arithmetic, not a float
observation (`test_per_token_stable_blocks_can_diverge_under_switching`).

**A uniform sufficient condition for real modes.** `A(g₁) − A(g₂)` is rank
one. By King & Nathanson (after Shorten & Narendra), two real Hurwitz matrices
differing by a rank-one term share a quadratic Lyapunov function iff their
product has no negative real eigenvalue. For this family that condition
reduces to

    (γ + T g₁)(γ + T g₂)  >  M (√g₂ − √g₁)²

(`test_detK_quadratic_matches_the_matrix_it_summarizes`,
`test_king_nathanson_condition_equals_the_factored_margin`). Since `A(g)` is
affine in `g`, a function common to the endpoints serves the whole interval
`[g_min, g_max]`.

**Corollary: `ρ = M/(γT) ≤ 1` gives a common Lyapunov function for every
positive gain range, however wide.** The left side exceeds
`γT(g₁ + g₂) ≥ γT(√g₂ − √g₁)² ≥ M(√g₂ − √g₁)²`
(`test_rho_at_most_one_gives_a_cqlf_for_every_gain_range`, and an exhaustive
exact grid with no divergent pattern: `test_no_divergent_switching_pattern_when_rho_at_most_one`).

Because `z = (s, P)` is continuous across switches (§3.1) and the gate is held
within each token, the continuous-time decrease carries over exactly. Under
ZOH, `V` falls by a uniform factor every token, whatever the gate sequence.
The Cayley map preserves the same `Q`. The repository's `gp_rho` domain
already sits in `ρ ∈ [0.01, 1 − 10⁻⁴]`. Uniform exponential rather than
marginal decay needs `g_min > 0`: at `g = 0`, `det = 0` and `s` becomes a pure
integrator, which is perfect retention but not a contraction.

**Blocker: complex S5 modes.** With `G_t = g_t J` and `Im J ≠ 0`, the real 4×4
realization changes by `(g₁ − g₂)(ue₁ᵀ)⊗R(J)`, a **rank-two** difference. The
theorem above does not apply, and I have no closed-form condition for
`0 < ρ < 1`. It is settled only on the cancellation line `ρ = 1`. There the
`P` row is autonomous and the cascade `ṡ = −(G_t/γ)s + P/M`, `Ṗ = −(γ/M)P` is
uniformly stable for any `Re G_t ≥ g_min Re J > 0`, but that is exactly the
degenerate first-order case. A complex-mode build would need a certified
Hermitian common Lyapunov function, or a parameterization forcing one, before
any run. Per-token eigenvalues must not stand in for it.

This guarantee is not unique to the prospective structure either. A generic
selective 2×2 block parameterized with a negative-definite symmetric part,
`A_t = −(εI + L_tL_tᴴ) + (S_t − S_tᴴ)`, is contractive in the Euclidean norm
under arbitrary switching, for real or complex blocks.

## 7. Closest existing mechanisms

| Mechanism | Relation |
|---|---|
| Native S5, doubled-mode S5 | LTI. Contain the fixed GP-S5 (§1.3); cannot represent any input-dependent gain |
| Mamba (S6) | Diagonal selective, `h_t = e^{Δ_tA}h_{t−1} + Δ_tB_t x_t`. Equals the cell's `ρ = 1` boundary (continuous law; Mamba simplifies `B̄` to `Δ_tB_t`). Its transitions commute; the cell's do not |
| S7 (Soydan et al., arXiv:2410.03464) | Input-dependent `Λ̄_k` with the stability map `Λ̄_k = I − (Λ_k² + ½I)⁻¹`, as stated in the arXiv HTML. Whether its `Λ` is diagonal in practice was not confirmed here. If diagonal, it is the same class as Mamba for this comparison |
| GRU / minGRU (Feng et al., arXiv:2410.01201) | minGRU gates depend on `x_t` only: `h_t = (1−z_t)h_{t−1} + z_t h̃_t`, which is exactly the cell's `ρ = 1` ZOH boundary. A full GRU's reset gate reads `h_{t−1}`, so it is not a scan |
| Kalman / α–β innovation filter | Same skeleton: fixed dynamics, innovation measured through `e₁`, correction along a gain vector. The cell fixes the gain direction `u(M, γ, T)` and schedules only its magnitude from the token. A Kalman gain would come from a covariance |
| DeltaNet / Gated DeltaNet | Token-selected gain **and subspace** (`β_t k_tk_tᵀ`), identity plus rank one, via a chunked algorithm. The cell's subspace is fixed per mode (axis-aligned keys). Token-dependent subspaces need `G_t` across modes, which loses the efficient scan (§4.2) |
| Momentum DeltaNet | Second order in `(W, U)`, with four token gates `(α, β, μ, η)` and a token subspace. Per mode with diagonal keys it is a per-token 2×2 block with more selective freedom than the cell. The two are not nested; both sit inside the generic block class. The two-tap readout at `κ = μ` is Nesterov (`docs/PROSPECTIVE_PRIOR_ART.md`) |
| Fixed GP-S5 (executed) | The cell's `g_t ≡ 1` member (§3.3) |
| `gp_adaptive_mass` (implemented, unexecuted) | Token-dependent `(M, γ, T)` through one parameter-free scalar with a fixed residual gain. A different one-parameter selective slice of the same 2×2 block class |

## 8. Classification

**`DISTINCT_PARAMETERIZATION_OF_KNOWN_SELECTIVE_SSM`.** It is not
`ALREADY_TESTED`, because no executed arm had a token-dependent state gain. It
is not `EXPRESSIVELY_REDUNDANT` with LTI or diagonal selective S5: its
transitions vary with the input and do not commute. It is not
`DISTINCT_AND_WORTH_TESTING` as a capability: a generic selective 2×2 block
contains it exactly, and its first-order boundary is minGRU/Mamba. What is
specific to it:

- a rank-one gating direction fixed by `(M, γ, T)`;
- the jump-free `(s, P)` coordinate;
- for real modes, arbitrary-switching stability throughout `ρ ≤ 1`.

## 9. If it were tested: required controls, task, measurements and rule

**Not authorized.** Specified so that a GO decision, if ever made, cannot drop
the decisive control.

Arms, all on one causal substrate with equal layers, `H`, clipping and input
gain:

1. **Native S5, `2P` modes (state-matched).** At this carry it coincides with
   the "doubled-mode" control, so a second native arm should be
   **parameter-matched** instead (`P` modes, width raised to equal
   parameters).
2. **Input-gated first-order S5, `2P` modes**: the cell's `ρ = 1` law, with the
   same gate projection.
3. **Generic selective 2×2 block**, the essential control. Same carry, same
   scan cost, gate parameters matched, and uniformly dissipative
   (`A_t = −(εI + L_tL_tᴴ) + (S_t − S_tᴴ)`) so that both arms have a switching
   guarantee.
4. **Fixed GP-S5**, `gp_rho_T` (`g_t ≡ 1`).
5. **Rawat prospective-input S5**, `alpha_p_s5`.
6. **The proposed cell**, real gates `G_t = g_t J`, only after the
   complex-mode stability blocker (§6) is resolved.

Task: tokens forcing retain-versus-revise decisions — **write**,
**confirmation** (the same value repeated), **contradiction** (a new value for
a stored key), **distractor** (unrelated content), **overwrite** — with queries
at matched delays after each event type.

Mechanistic measurements:
- `g_t` by event type (the prediction: high on contradiction and overwrite,
  low on distractors);
- linear decodes of `s_t` and `P_t`;
- immediate revision accuracy;
- later retention of untouched keys;
- matched-delay recall.

Decision rule, correcting the flaw recorded in
`docs/PROSPECTIVE_READOUT_REPORT.md` §5:
- compute the per-seed **paired difference**, proposed minus arm 3, and use
  the standard error or confidence interval **of that difference**;
- win: the lower 95 % bound exceeds the margin;
- loss: the upper bound falls below it;
- otherwise indeterminate.

No individual-metric SE is compared against the margin.

## 10. Recommendation: **NO-GO**

1. **It provides no capability** that a compute-matched generic selective
   two-mode SSM lacks (§5). What distinguishes it from diagonal selective
   models (non-commuting transitions) is generic to block selectivity.
2. **Its stability advantage is real but not distinctive.** It holds for real
   modes at `ρ ≤ 1`; the complex modes S5 actually uses are an open blocker
   (§6); and a dissipative generic block gets an equivalent guarantee
   trivially.
3. **The prior record points the same way.** Every generalized-prospective
   variant executed so far lost to, or failed to separate from, its matched
   ordinary control:
   - recall: `ordinary_2x` by 5.03 pp mean, every seed;
   - timescale: behind the ordinary substrate and Rawat;
   - Momentum-DeltaNet: no clean positive;
   - readout: carry-free control best, verdict INDETERMINATE.

   A remaining inductive-bias claim would need to beat arm 3 of §9 by a
   declared paired margin. Nothing in the repository makes that likely enough
   to spend a cap on.

What would change this: a certified switching-stability result for complex
modes at `ρ < 1`, *and* a concrete reason to expect the fixed rank-one gating
direction to be a better prior than a learned one.

**For the methods note**, three small exact facts from this audit are worth
recording alongside the Nesterov identification:

- the jump-free coordinate `P = Mṡ + Tr` for token-switched residuals;
- the `ρ = 1` identity between the cell and minGRU/Mamba-type first-order
  innovation;
- the rank-one switching-stability criterion
  `(γ + Tg₁)(γ + Tg₂) > M(√g₂ − √g₁)²`, with its `ρ ≤ 1` corollary and its
  exact counterexample outside it.

## 11. Evidence classes

- **Established from code and records**, and cited: §1.
- **Derived by hand and checked in exact rational arithmetic** (13 tests,
  local, Fractions only): §3, §4.3, §5 (except the Mamba/minGRU/S7 rows),
  §6's criterion, corollary and counterexample.
- **From the literature, checked against the arXiv abstract or HTML on
  18 September 2026:** the King & Nathanson rank-one theorem
  (arXiv:math/0403467); S7's stability map (arXiv:2410.03464); minGRU's
  input-only gates (arXiv:2410.01201). Mamba's recurrence is from the
  published formulation and was not re-fetched.
- **Not claimed:**
  - a ZOH counterexample (the exact one is bilinear);
  - complex-mode stability for `0 < ρ < 1`;
  - any performance expectation;
  - novelty beyond what the table in §7 supports.
