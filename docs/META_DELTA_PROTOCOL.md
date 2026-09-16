# Generalized prospective memory around the delta boundary — proposed frozen protocol

**Status: prepared for static review, amended before execution by the review
of `535fb02` (§12, which supersedes the corresponding passages). No local
numerical run. No cluster launch is authorized.** Completed memory and S5 studies, their verdicts and
their artifacts are unchanged.

| | |
|---|---|
| Brief | `META_DELTA_NEXT_AGENT_BRIEF_2026_09_16.md` |
| Analytical basis | `GENERAL_PROSPECTIVE_META_DELTA_AUDIT_2026_09_16.md`, audited independently in `docs/META_DELTA_PROOF_AUDIT.md` |
| Branch | `generalized-meta-delta` |
| Parent | `b5827b7` (`adaptive-prospective-memory`: the completed adaptive-memory study and report) |
| Code | `experiments/meta_delta/{dynamics,model,calibrate,study}.py` |
| Checks | `tests/test_meta_delta.py`, `tests/meta_delta_float32_probe.py` |
| Launcher | `bin/run_experiments/cluster_meta_delta.sh` |

## 1. Question and scope

Does generalized prospective memory — started **exactly** at the working
first-order delta model and allowed to learn on **both** sides of
`M = γT` — improve on the actual Gated DeltaNet and Momentum DeltaNet rules?

Three questions are reported separately:

* does it improve on the literature rules;
* does it improve on a precisely identified ordinary-prospective reference;
* can a difference be attributed to `T Ṙ` specifically.

**Unchanged from the completed adaptive-memory study:** the task generator (64
intervals, 16 queries, two families, four categories); `d_k = d_v = 8`, 32
keys, 8 values; the objective `R = w(Wk − v)kᵀ`; the 38-parameter positive
source gate; the embeddings; the readout `Wq` read after advancing the query
interval; the exact rank-one steps; the memory-state counts; the loss; the
primary metric (revision-family macro accuracy); and the information
restrictions. No oracle feature, alternative readout, additional memory matrix
or derivative approximation is added.

## 2. Arms

| Arm | Law | Trainable | Carry |
|---|---|---|---|
| **Candidate** `gp_two_sided` | `M Ẅ + γẆ + R + T Ṙ = 0`, both sides of `M = γT` | 433 | 128 |
| First-order delta control `adaptive_delta` | `Ẇ = −ηR`, the existing arm | 431 | 64 |
| Attribution control `heavy_ball_same_mass` | `M Ẅ + γẆ + R = 0`, the existing inertial law and tree | 432 | 128 |
| Ordinary prospective `tss_eq17` | actual TSS Eq. (17), `f = W − ηR` | 432 | 128 |
| Gated DeltaNet `gated_delta` | source-pinned, unchanged | 480 | 64 |
| Momentum DeltaNet `momentum_delta` (primary literature) | source-pinned, unchanged | 569 | 128 |

Counts are asserted by a check (the common shell is 392). No inert parameter
equalizes them.

## 3. Candidate parameterization and initialization

```
eta = exp(raw_eta) = 1/gamma      tau = exp(raw_tau) = M/gamma
rho = exp(raw_r)   = M/(gamma T)
gamma = 1/eta,  M = tau/eta,  T = tau/rho,  executed nu = T/M = eta/rho
```

* **Coordinates:** direct logs throughout. There is no sigmoid excluding
  `ρ = 1` and no square with zero tangent at the origin; `dρ/draw_r = 1` at
  the start.
* **Baseline point:** `raw_r = 0` gives `ρ = 1` exactly. With `Z₀ = 0`, the
  token step is exactly the delta step with rate `η`, for every `τ`.
* **The new direction:** moving `raw_r` at fixed `(raw_eta, raw_tau)` varies
  `T` at fixed `γ` and `M`. It adjusts the key-directed prospective term, not
  the delta rate, which `raw_eta` already carries.

**Initialization.** `raw_eta` is identical to the delta control's `η₀`. Gate
`u = b = 0`, and the common tensors are bit-identical at a seed.

**Choice of `T₀ = τ₀ = 1`** (one token interval). `T` is unidentifiable at
`ρ = 1` in the forward function, but it sets:
* the relaxation time of `Z` once `ρ ≠ 1`;
* the room on the wider side: `ρ < x/(x−1)` with `x = η₀τ₀L ≈ 1.86`, giving
  `ρ_max ≈ 2.16`;
* the residual-velocity damping `γ/(γ + Tλ)`.

`T₀ = 1` matches the task clock (one interval per token) and gives
comparable room on both sides of `ρ = 1`. It is not tuned; no `T` sweep is
proposed. A much larger `T₀` would narrow the wider side toward `ρ ≤ 1`.

**Map from a saved delta tree.** `model.delta_to_two_sided` copies every
common, gate and `raw_eta` leaf, and adds `raw_tau = log τ₀` and `raw_r = 0`.
The auxiliary carry resets to zero at every episode boundary. It is provided
and checked for identity, but **not used by this protocol** (§5).

## 4. Domain and projection

| Region | Condition | Status |
|---|---|---|
| `0 < ρ < 1` (`M < γT`) | none | the previously certified sector; the passive two-compartment circuit lies here |
| `ρ = 1` | — | exact delta |
| `ρ > 1` (`M > γT`) | `dL < γ²`, `L = 2` the executed gate bound; equivalently `ρ < x/(x−1)` for `x = ητL > 1`, any `ρ > 1` for `x ≤ 1` | sufficient switching certificate; **a computational application of the same equation, NOT the passive two-compartment circuit** |

**Post-update projection of `raw_r` only**, using the **updated** `η` and `τ`:

```
x' = eta tau L (1 + 16 eps)
U  = -log1p(-1/x')          (x' > 1)
raw_r <= max(0, U - 32 eps (1 + |U|))
```

* The branch `x' ≤ 1` is masked before the logarithm and left unbounded.
* The `(1 + 16 eps)` inflation makes the bound conservative against rounding
  of `x`, of `1/x`, and of unit key norms; see audit §8.
* `ρ = 1` is the exact fallback.
* There is no forward clip. Optimizer state is untouched. Coefficients are
  constant within an episode, so changing `γ` or `T` in an update cannot
  invalidate the certificate: the projection is recomputed from the updated
  values.

**Validation.** A per-run `domain_report` evaluates, in float64 from the
executed float32 `η`, `τ`, `ρ`, whether all coefficients are finite and
positive, which side the point is on, and whether the certificate holds. A
violation is a FAILED run.

## 5. Why cold start, not warm start

Starting the candidate from a saved delta tree would require every paired
control to receive identical common parameters, provenance, data and
optimizer policy. Gated and Momentum DeltaNet have no delta tree to warm-start
from, and warm-starting them from the delta's shared tensors alone would make
them different models.

This protocol therefore starts **every arm cold**, from identical common
initializations at each seed, with identical data streams and one optimizer
policy. The candidate still starts exactly as the delta control, which starts
at the same seed. The map in §3 is prepared, but a warm-start study is not
proposed.

## 6. Comparators — definitions

**First-order delta.** The existing `adaptive_delta` rule:
`β(w) = −expm1(−ηw)`, with the same gate.

**Attribution control for `T Ṙ`.** The existing heavy-ball law
`M Ẅ + γẆ + R = 0`, learning `(η, τ)`. It starts at the candidate's `γ` and
`M`, with `T Ṙ` removed. The candidate's generator tends to this generator as
`T → 0` at fixed `γ` and `M` (a check). It does **not** equal the delta model
at the start, because removing `T Ṙ` changes the function there, and its
initial single-write amplitude is recorded. It has one fewer response scalar
than the candidate.

**Ordinary prospective reference: actual TSS Eq. (17)** applied to the fast
weight:

```
R_k     = w_k (W_k k_k - v_k) k_k^T
f_k     = W_k - eta R_k
W_{k+1} = W_k + (h/T)(-W_k + f_k) + f_k - f_{k-1}
```

* **Clock:** `h = 1` token interval.
* **Cache initialization at each episode:** `W₀ = 0`,
  `f_{−1} = W_{−1} − ηR_{−1} = 0`, meaning no observation before the episode.
* **Carry:** `(W, f_prev)`.
* **Learned:** `η` and `T` (logs), plus the same gate.
* **Initialization:** `T₀ = 10`, and `η₀ = β*/(1 + 2h/T₀)`, so its
  single-write, one-idle-interval query amplitude equals `β*`.

This is TSS's finite-step rule. It is **not** the earlier finite-adaptation
arm and **not** the minimum-change projection.

**Applicability audit (analytical):**
* In the key direction, the error obeys
  `μ² − [2 − (1 + h/T)ηw]μ + (1 − ηw) = 0`. It is stable iff `0 < ηw < 2`
  and `ηw(2 + h/T) < 4`.
* The increment obeys
  `ΔW_{k+1} = ΔW_k − η(1+h/T)R_k + ηR_{k−1}`. It is preserved only after
  **two consecutive zero-residual inputs**; the first idle interval after a
  write still changes it. After that a write keeps drifting, linearly in the
  number of further idle intervals (§12 D1).

This reference is **not a stable memory in unforced directions**, and it is
outside the certified `γ > 0` sector. **This is recorded as an unresolved
applicability issue for the coordinator:** keep this comparator as specified,
or declare the ordinary-prospective comparison unresolved. No substitute is
used.

**Literature arms.** The completed study's source-pinned Gated DeltaNet and
Momentum DeltaNet, with their official initializers. They are asserted to give
bitwise-identical logits to that implementation. They are **not** claimed to
be nested by the candidate: Momentum DeltaNet's token-dependent gates are not
mapped.

## 7. Selection and training

| | |
|---|---|
| Selection budget | **equal and identical axis**: two slots for every family, outer lr 0.003 (A) and 0.01 (B), identical initialization otherwise |
| Development | seed **300**, 12 runs; selection by highest revision macro accuracy, then lower revision CE, then A |
| Final | seeds **301, 302, 303**, 18 runs; trained from scratch with the selected slot |
| Held-out | 512 sequences per family, evaluated only after all final runs |
| Training | 200 updates, batch 16 (8 per family), Adam (0.9, 0.999, 1e-8), global clip 1.0, float32; lr applied dynamically, so both slots share one compilation |

Every step is followed by the `raw_r` projection, which is a no-op for other
arms.

**Streams**, fresh and asserted disjoint from each other and from both earlier
memory studies:

| Purpose | Entry point |
|---|---|
| training | `50,000,000 + seed×10,000 + update` |
| development validation | `60,000,000` |
| evaluation validation | `61,000,000` |
| held-out | `70,000,000` |

The task distribution has informed this proposal: **this is a development
study.**

## 8. Verdicts — SUPERSEDED IN PART by §12 D1

The verdicts in force are: literature, and matched delta alongside it; TSS
Eq. (17) applied directly to the fast weight (applicability-limited); and the
heavy-ball family comparison (not causal attribution). The table below is the
original proposal, kept as the record.

Every comparison uses paired final seeds, and all three pairs must be present.

| Verdict | Candidate vs | Rule |
|---|---|---|
| **Literature** | Momentum DeltaNet AND Gated DeltaNet | +1 point mean primary over each; positive paired primary in all three seeds; revision-untouched retention and overall recall **each** within −1 point |
| **Ordinary prospectivity** | TSS Eq. (17) reference | same rule |
| **`T Ṙ` attribution** | heavy-ball same-mass control | same rule |

The comparison with first-order delta is **reported descriptively**. A gain
over delta, a second state, or extra parameter freedom attributes nothing to
`T Ṙ`.

None of these verdicts substitutes for another. None is a significance test,
a published-benchmark result or a SOTA claim. Switching stability and
baseline nesting do not imply better accuracy.

**Diagnostics reported:**
* the candidate's learned `η`, `τ`, `ρ`, `γ`, `M`, `T`, its side of
  `M = γT`, and its certificate margin;
* projection events and overshoot;
* gate distributions;
* per-category held-out results and training gain;
* parameters, carry and time.

## 9. Focused checks before training (cluster only)

Tolerances are frozen in the test headers.

* Exact delta nesting at `ρ = 1` over arbitrary episodes with key changes and
  idle intervals, with a nonzero gate. Logits within 1e-9; common, gate and
  `η` gradients within 1e-8 relative; the `raw_tau` gradient zero. Plus
  `Z ≡ 0` on every token, and the saved-tree map.
* A nonzero `raw_r` tangent at the actual start against central differences
  (1e-5 and 1e-6; 1e-6 relative), with finite nonzero derivatives just below
  and above `ρ = 1`.
* Mechanism, not accuracy: `Z` sign and idle motion on each side of `ρ = 1`;
  the residual-velocity identity (eq. 7) against an independent Lyapunov
  solve, including the `ρ = 3/2`, ratio-2/3 witness.
* Switching storage non-increasing for `ρ > 1` at 30 % and 95 % of the
  certified room, with arbitrary unit keys, `w ∈ [0, 2)` and idle intervals.
  The delta boundary is checked separately as a non-expansive step, not
  through the singular formula.
* Projection from updated `η` and `τ`: 200 random float32 points certified,
  and the unbounded branch for `x ≤ 1`. A real optimizer update with an
  outward `raw_r` must be projected, certified, finite, float32 and
  differentiable.
* Eq. (17) against a literal reference; its first-write and idle amplitudes;
  the preserved idle increment; its calibration.
* Attribution-control generator limit; counts; common tensors; bitwise
  literature arms; stream disjointness; separate screens with complete pairs;
  preflight decision (FAILED vs INCOMPLETE).
* **float32 probe** (smoke checks, not accuracy certificates):
  * nesting logits within 1e-4, and shared gradients within the mixed
    tolerance;
  * the `raw_r` tangent at `h = 1e-2` and `3e-3`, 2e-2 relative, with
    resolvability;
  * `raw_tau` gradient ≈ 0;
  * 400 projected points, and the ill-conditioned edge `x → 1⁺`, certified
    in float64.

The launcher runs pytest with `-rP`, so the measured magnitudes of passing
checks are kept in the log.

## 10. Budget

| | |
|---|---|
| Cap | one **600 s** cap covering GPU startup, focused checks, preflight, 30 runs (6,000 updates), evaluation and a 30 s reserve |
| Preflight | measures every arm on the executed step path (host batch generation, step, synchronization of every recorded scalar) and evaluation |
| Refusal | invalid state or a non-finite projection is **FAILED**; a retrace or an over-budget projection is **INCOMPLETE**, with nothing started or trimmed |
| Prohibited | automatic retry, larger cap, rescue sweep, threshold change |
| Reference cost | the completed seven-arm adaptive study took 283 s of 600, 164 s of it checks. This batch has fewer runs (30 vs 35), but its check suite is new and its duration is not measured. |

## 11. Unresolved issues, stated plainly

1. **Ordinary-prospective comparator applicability.** Actual TSS Eq. (17) on
   a fast weight preserves increments through idle intervals, so it is not a
   stable memory in unforced directions. Keep it as specified, or declare
   that comparison unresolved. This is the coordinator's decision.
2. **Attribution control imperfection.** Heavy ball with the same `γ` and `M`
   removes `T Ṙ` but cannot start at the delta function and has one fewer
   response scalar. A difference compares separately trained rule families;
   it is not a term-removal ablation of one trajectory.
3. **Cold start** is proposed instead of a warm-start continuation (§5). The
   saved-tree map is prepared but unused.
4. **Check-suite timing** is unmeasured; the 600 s fit is decided by
   preflight, not assumed.


## 12. Pre-execution amendments — review of `535fb02`

Source: `META_DELTA_REVIEW_535fb02_2026_09_16.md`. Recorded **before** any
execution. The six arms, equations, initial coefficients, data, schedule,
tolerances and 600 s cap are unchanged. The completed memory and S5 studies
and their conclusions are unchanged.

### Accepted by the review

* The equation.
* The `(η, τ, ρ)` parameterization, in which `raw_r` varies `T` at fixed
  `γ` and `M`.
* The exact delta start.
* The wider-sector certificate and the direction of its conservative
  inflation.
* The "computational application, not the passive circuit" labelling.
* Cold starts, which answer a fresh-training question, not a continuation
  question.

### R1 — verification of the actual start and of the float32 wider region

* **Fixture names.** The nonzero-gate fixtures are kept and renamed as stress
  fixtures.
* **Actual start.** New checks use the **unchanged initialized tree** (zero
  gate, `ρ = 1`, `τ = 1`): nesting in value and shared gradients (float64),
  the `raw_r` tangent against finite differences, and a vanishing `raw_tau`
  tangent. This runs in float64 and in the float32 probe.
* **Resolvability.** If a derivative falls below the declared resolvability
  threshold, that limitation of finite differences in that dtype is
  **reported**, not treated as a failure or a dead parameter.
* **Wider region in float32.** The float32 probe executes a sequence with key
  changes, writes and idle intervals, at `ρ` safely above 1 and at the
  projection margin. It is compared with an independent float64 dense
  augmented-ODE reference for the same rounded inputs, at the existing float32
  tolerance `2e-5`.
* **Wider-region derivative.** A `raw_r` directional derivative is checked at
  an interior wider-region point, with `r ± h` strictly inside the bound, so
  no finite difference crosses the projection.
* **Dtypes.** Assertions cover coefficients, generator, `F`, `a0`, carries and
  outputs.
* **Training process.** `study.main` refuses to run unless x64 is disabled
  and the default dtype is float32.

### R2 — executed arithmetic and measured preflight scalars

* **Report fields.** `domain_report` keeps executed values as `executed_*` and
  the float64 certificate reconstruction as `certificate_f64_*`.
* **Early failure.** It fails, without raising, when any executed scalar is
  zero, negative or non-finite, before any division.
* **Generator.** It forms the **production** `two_sided_generator` in the
  executed dtype at the gate endpoints `w = 0` and `w = L`, and checks its
  entries and the executed idle coefficient `a0` for finiteness. The
  certificate is reported separately.
* **Regression.** The scalar-finite, generator-non-finite counterexample
  (`η ≈ 1e20`, `τ = 1`, `ρ ≈ 5e-19`, `w = 2`) is rejected.
* **Preflight.** Acceptance includes the **measured** loss, accuracy,
  gradient norm, update norm and state norms, and records any failed field.
  Any non-finite measured scalar makes preflight FAILED (4) before training.
* No coefficient is clamped and the law is unchanged.

### D1 — comparator scope (coordinator decisions)

**TSS Eq. (17)** is kept, labelled **"TSS Eq. (17) applied directly to the
fast weight; applicability-limited"**.
* With `f(W) = W − ηR(W)` for one association,
  `(I − Df)[X] = ηw(Xk)kᵀ`. This is singular off the current key and zero on
  idle intervals, so TSS Eq. (15)'s inverse does not exist for this
  application.
* The comparison is a well-defined discrete experiment, **not** a reproduction
  of TSS's teaching-synchronization experiments. No unrestricted "ordinary
  prospectivity" verdict is printed.
* The idle statement is corrected to require two consecutive zero-residual
  inputs, and a check covers the first idle increment.
* No damping, reset or substitute comparator is added.

**Heavy ball** is kept as a **separately trained family comparison**, with
`γ` and `M` matched at initialization only. The verdict is renamed
`heavy_ball_family_comparison_passed` and is not causal attribution to
`T Ṙ`. The fixed-quadratic residual-velocity identity remains a separate
analytical mechanism statement.

**Matched delta** becomes an **explicit development verdict**,
`matched_delta_departure_passed`, with the same rule: +1 point mean primary,
positive in all three paired seeds, and retention and recall each within −1
point. It is reported **alongside** the literature verdict, so a literature
win cannot hide a loss to the simpler delta rule with the same source gate.
All per-seed differences are kept whether or not a verdict passes.

This supersedes §8's descriptive treatment of delta and its "attribution"
wording.

### Reporting corrections

* **Held-out data.** Held-out episodes are now **generated and hashed only at
  final evaluation**, after all final runs. The stream seed is unchanged.
* **Literature gates.** Gated and Momentum DeltaNet coefficient reports now
  save gate distributions on validation inputs, using the completed study's
  gate reporter.
