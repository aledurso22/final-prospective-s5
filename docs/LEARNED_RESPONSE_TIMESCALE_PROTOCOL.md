# Learned response timescale: protocol

Committed **before** any numerical execution. Brief: coordinator
`outputs/learned_timescale_priority_handoff_2026_09_15/CODING_AGENT_BRIEF.md`.
Implementation parent `7c13ef7` on `combined-input-recurrence` (itself from
`6a5bc7f`); this study is on branch `learned-response-timescale`.

This supersedes the experiment ORDER only. The composition implementation of
`docs/COMBINED_INPUT_RECURRENCE_PROTOCOL.md` is preserved, committed and
unrun; its batch is deferred, not cancelled.

## 1. Question

Does letting each stored mode learn its own response timescale `T_i`, in
addition to `rho_i`, change the short Speech Commands validation screen — and
specifically, does it beat the same model with `T` frozen?

**Arm 5 versus arm 4 is the isolating comparison.** It is not compared with the
superseded learned-`gamma_n`/`rho` result: `gamma_n` was removed as redundant
with the learned clock, and that optimizer trajectory was a different one.

## 2. What is learned, what stays derived

Per stored complex mode `i`, shared with its conjugate partner, with the clock
absorbed **exactly once** (`Delta = exp(log_step)`,
`B_c = diag(-Re lambda) B_tilde`, real poles clipped at most to `-1e-4`):

```
rho_i T_i s_i'' + s_i' + r_i + T_i r_i' = 0 ,  r_i = j_i s_i - b_i x
j_i = -Delta_i lambda_i ,   b_i = Delta_i B_c,i
```

| quantity | status |
|---|---|
| `rho_i` | **learned**, one per stored mode |
| `T_i` | **learned**, one per stored mode (arm 5); frozen (arm 4) |
| `gamma_n` | fixed at 1 — a learned `gamma_n` duplicates the learned clock |
| `mu_i = rho_i T_i` | **derived at every forward call**, never a parameter |
| `T_in` (input horizon) | not used by this batch; the generalized arms take **no** prospective input correction |

Parameterization `T_i = 5 exp(eta_i)`, so `eta = 0` is exactly the reference
horizon. Declared numerical guardrails for this bounded study:
`eta in [log(0.01), log(100)]`, i.e. **`T in [0.05, 500]` sample intervals**,
and the existing `rho in [0.01, 0.9999]`. **These are numerical guardrails, not
physiological bounds, and were not selected from accuracy.** A guardrail that
fails a numerical check is reported and amended with numerical justification —
never moved because a score improved.

Both raw coordinates are clipped in the forward map **and projected after every
optimizer update**, preserving optimizer state, through the single shared
projector `s5/response_projection.py`. Boundary occupancy, projection events
and pre-projection overshoot are logged for both.

Native parameters and both response coordinates use **the same optimizer
policy**: one AdamW group, no response-specific learning rate, no weight-decay
exemption. **Recorded rather than glossed:** decoupled decay shrinks `eta`
toward 0, which biases `T` toward its **reference value 5**, not toward 0.
Decay in a log coordinate is not coordinate invariant.

Coefficients are **constant within each sequence** and are updated between
minibatches by BPTT. No activity-dependent conductance, token gate,
meta-learner or new plasticity rule. They shape temporal responses; they do not
implement an online controller that decides what to remember.

## 3. Circuit realization (`s5/timescale_response.py`)

For any `T_i > 0`, `0 < rho_i < 1`, with `c_* = 7.5`:

```
c_s = c_d = c_* ,        G_s = G_d = c_*/(T rho) ,   h = G_s sqrt(1-rho)
g_L = G_s - h = c_*/(T (1 + sqrt(1-rho))) > 0
```

giving identically `kappa = c_*/T`, `tau_d = T rho`, circuit horizon `= T`,
physical `gamma = T`, physical `M = T^2 rho`, and after dividing the whole
equation by the physical `gamma_i = T_i`, the implemented normalized
`mu_i = T_i rho_i`. At `T = 5`, `rho = 3/4` this returns `G_s = G_d = 2` and
`g_L = h = 1`: the original symmetric reference.

Normalized and physical coefficients are reported **separately**; the
normalization divisor is now per-mode, which is the substantive difference from
the fixed-horizon map. The passive reciprocal plant, constant-conductance /
additive-current and prescribed-prospective-source qualifications are
unchanged. Fitting `T` and `rho` by BPTT between independent sequences is
optimization within a declared family; it is **not** a claim of biological
plasticity of these quantities and does not inherit NLA's local-learning
theorem.

## 4. Initialization, and why the old gate does not apply

`rho_0 = 0.75`, `eta_0 = 0` so `T_0 = 5` exactly — the earlier **symmetric
reference**, deliberately **not** the near-one recall/composition
initialization. The reason is identifiability:

```
G(p) = b (1 + T p) / [rho T p^2 + (1 + T j) p + j]
dG/dT = b (1 - rho) p^2 / [ ... ]^2
```

which **vanishes at `rho = 1`**. Starting at `rho_0 = 0.9998` would begin a
parameter-freedom test almost at an unidentifiable limit. This is a predeclared
model-based choice, not a search on scores, and it changes the starting
response relative to the earlier studies.

**The superseded 1 % function-match-to-ordinary gate does NOT apply here.**
These arms are meant to differ from the ordinary substrate. What is enforced
instead, before training:

* arms 4 and 5 are the **same function** at initialization — identical
  parameter trees and `max|logit difference| <= 1e-6` on a label-free probe,
  with `T_0 = 5` and `rho_0 = 0.75` verified from the executed module;
* their signal-only impulse and frequency differences **from the matched
  ordinary substrate** are **recorded, not gated**, with no label or score
  read.

If that equality check fails the study stops and reports. `rho` and `T` are not
adjusted to make it pass.

## 5. The five-arm screen

| # | arm | what it is |
|---|---|---|
| 1 | `native_s5` | native input gain, original clipping policy; literature baseline |
| 2 | `gain_clip_s5` | alpha-scaled input + clipped poles, ordinary one-tap; isolates the substrate change |
| 3 | `alpha_p_s5` | Rawat prospective-input two-tap, input horizon 5 |
| 4 | `gp_rho_T_fixed` | generalized prospective recurrence, learned `rho`, `T = 5` frozen |
| 5 | `gp_rho_T` | the same, with learned per-mode `T_i` |

**Seed 100, ten epochs, batch 32, from scratch, full BPTT.** One seed: this is
a **development screen**, and it cannot establish a robust gain. Multi-seed
confirmation would be a separately declared batch.

Architecture and recipe, identical for every arm and taken from the saved
Stage 2 configuration: 4 layers, `d_model = 32`, base SSM size 32, 8 HiPPO
blocks, conjugate symmetry, forward-only ZOH, same normalization, residuals,
nonlinearities, pooling, classifier and dropout; AdamW `lr = 1e-3` with the
ten-epoch cosine schedule to `1e-6`, weight decay `1e-4`, global gradient clip
1, label smoothing 0.1. `--stage2_config` compares this against a saved
`config.json` and **refuses to run on a discrepancy** rather than reconciling
it; if no saved config is supplied, the report records that the values were not
verified against a saved run.

Common initial parameters and batch statistics are cloned from `gain_clip_s5`
into every arm and their digests recorded; the same data order and the same
dropout stream are used. The native-versus-alpha input gain and the clipping
difference are static configuration and remain explicit. Preflight updates are
discarded and every arm re-initializes before comparative training. Fresh
artifact directories; `SC.load_splits(..., splits=("train","val"))`, so the
test arrays are never opened.

**Primary measure: validation accuracy at the fixed tenth epoch.** Secondary:
endpoint unsmoothed validation cross entropy, best-validation checkpoints,
learning curves, runtime, memory, parameter and state counts. The primary
endpoint is **not** changed after seeing the ordering, and best-validation
scores never substitute for it. Historical best-validation numbers from other
batches are context only, never a paired endpoint comparator.

Expected counts, to be **verified before training**, not assumed:

| arm | trainable | stored | carried real state / 4 layers |
|---|---|---|---|
| `native_s5`, `gain_clip_s5`, `alpha_p_s5` | 35,050 | 35,050 | 128 (+128 input buffer for `alpha_p_s5`) |
| `gp_rho_T_fixed` | **35,114** | **35,178** | 256 |
| `gp_rho_T` | **35,178** | 35,178 | 256 |

The two generalized arms carry **identical** state: more trainable
coefficients do not imply more carried state.

No imported `+0.3` pp threshold, no automatic significance claim, no seed or
checkpoint cherry-picking, no automatic rescue sweep. A positive one-seed
result is a candidate for confirmation — not a demonstrated benchmark
improvement and not proof of physical causation.

## 6. Predeclared checks and tolerances

`tests/test_learned_timescale.py` (float64) and
`tests/timescale_float32_probe.py` (production dtypes, own process). Cluster
only, inside the same cap.

| check | tolerance |
|---|---|
| per-mode `T` generator and ZOH vs an independent dense real-pair reference (scipy `expm` + Gauss-Legendre), `P != H` | `1e-10` rel, float64 |
| declared corners `T in {0.05, 500}` x `rho in {0.01, 0.9999}`: finite value **and** derivative | finite (not relaxed) |
| the same corners, coefficient agreement with that reference | `1e-8` rel, **amended from measurement**, see below |
| `T = 5` reproduces the existing fixed-horizon coefficient law | `1e-10` |
| `rho = 1` gives ordinary S5 for several `T`, with **zero** output `T` sensitivity | `1e-10` / `1e-9` |
| arms 4 and 5 identical at initialization, forward and in shared-coordinate gradients | `1e-12` |
| `d/d(log T)` vs central differences on a nonconstant probe, step ladder | best `< 1e-6`, converging |
| gradient through the derived `mu = rho T` | `1e-10` |
| scan carries, resets and chunking with a per-mode horizon | `1e-10` |
| circuit map identities, normalized vs physical kept distinct | `1e-9` |
| diagnostics vs an actual forward impulse / its DFT | `1e-6` / `1e-4` |
| a **real production update** moves `log T` in arm 5 and leaves it exactly fixed in arm 4, while `rho` and native parameters move in both | exact `0.0` for the frozen leaf |
| forced outward-then-inward projection, **both** coordinates, **both** bounds | zero gradient outside, nonzero after projection |
| production `float32`/`complex64` against the same reference, corners included | `2e-4` rel |

`rho = 1` is used as an algebraic limit; the trained parameter's numerical
margin is `0.9999`. **No nonzero-gradient assertion is made at that boundary**,
because the `T` derivative genuinely vanishes there.

### Amendment, 16 September 2026: the corner coefficient tolerance

Declared **before any training and before any validation score was read**, from
a cluster measurement on the first execution attempt (`53f9c9b`, logs
`20260916-000336`), which ended `TIMESCALE_STATUS=FAILED` at the focused checks
and started no training.

Measured float64 agreement between production `mass_block_zoh` and the
independent real-pair / Gauss-Legendre reference:

| regime | `A_bar` relative error |
|---|---|
| ordinary range, `T in [0.2, 50]` | `1.6e-15`, `2.6e-15`, `4.3e-15` |
| guardrail corner `T = 500`, `rho = 0.25` | **`1.42e-9`** — exceeded the original `1e-10` |

The corner is **stiff**: at `T = 500`, `rho = 0.25` the block's eigenvalues are
about `4a` (order 10) and order `1e-3`, a separation near `5e3`, so its
eigenvectors are nearly parallel and two *correct* matrix-exponential
implementations diverge by far more than machine epsilon. Production compounds
this by exponentiating the augmented `[[A, I],[0, 0]]` to get `A_bar` and `Phi`
together, which the reference does not. **This is a conditioning property of
the declared corner, not an error in either implementation and not a statement
about the model.**

The amendment is therefore narrow:

* the ordinary range keeps `1e-10`, with five orders of measured margin;
* the **corners only** get `1e-8`, about `7x` over the measured worst case;
* **the finiteness requirement at the corners is not relaxed at all** — value
  and derivative must be finite, which held at all sixteen corners on the
  failing run;
* the check **prints the achieved relative error and the measured eigenvector
  conditioning at every corner on every run**, so the justification is
  re-measured rather than asserted once.

No guardrail was moved: `T in [0.05, 500]` and `rho in [0.01, 0.9999]` are
unchanged. Nothing here was informed by a validation score.

### Amendment, 16 September 2026: the preflight's evaluation projection

The second execution attempt (`cce550b`, logs `20260916-002141`) passed all 58
focused checks and then **refused to start training**:

```
[preflight] native_s5    compile 13.4s  step 1.99ms  epoch 1.7s  val 43.7s  arm 467.3s
[preflight] gp_rho_T     compile 16.5s  step 7.17ms  epoch 6.0s  val 60.9s  arm 686.4s
PREFLIGHT_PROJECTED_TOTAL_S=2790.1
[!] projected 2790s > remaining 856s; comparative training NOT started.
```

The refusal behaved exactly as designed — no epoch, arm or control was reduced
and the cap was not widened — but **the projection was wrong**, and the fault
was in the preflight, not in the experiment.

The evaluation cost was measured with a single 512-sample call and scaled by
`n_val / 512`. That call includes the one-off **evaluation compile**, so the
scaling multiplied a roughly 4 s compile by about 11 and reported `val 43.7s`
*per epoch*. The measured per-epoch training cost tells the real story: 1.7 s
for the ordinary arms and 6.0 s for the block arms, against a claimed 43-61 s
of validation on about 5,700 sequences through the same model. Validation is
cheaper than an epoch here, not twenty-five times more expensive.

Two fixes, both in `experiments/gp/timescale_study.py`:

* evaluation compile and steady-state cost are now timed **separately**, the
  compile counted once per arm and only the steady cost scaled by split size;
* `evaluate_split` pads the ragged final batch to a fixed shape behind a mask,
  so exactly one evaluation program compiles per arm.

The unbatched read-only module view is also cached; it was reconstructed for
every layer of every per-epoch response snapshot.

With the corrected accounting the same measured numbers project to roughly
300 s of training against the 856 s that were available. **No timing was
assumed** — the corrected preflight re-measures on the next run and refuses
again if it does not fit.

### Amendment, 16 September 2026: two defects in the checks themselves

Also from that failing run, and also before any training:

* **The production-dtype probes were measuring float64.** Each asserted x64 was
  off and then imported reference helpers from a test module that enables x64
  at import; the assertion ran before the import. The shared reference now
  lives in `tests/response_reference.py`, which never touches `jax.config`, and
  each probe re-asserts x64 is off **after** its imports. The declared `2e-4`
  float32 tolerance is unchanged — it had simply never been exercised.
* **The diagnostics frequency check compared against a truncated DFT.** With
  `Delta` starting as low as `1e-3`, `A_bar` is close to the identity and the
  impulse has not decayed within any affordable window; the measured
  discrepancy was `0.37`, not a rounding effect. The check now compares against
  the DFT of a **forward-measured** impulse **plus the exact resolvent
  remainder** `u^(N-1) (uA)(I - uA)^-1 S_{N-1}`, tolerance `1e-8`. Truncating
  the tail and calling the difference agreement would have been the invalid
  geometric-tail-bound error the Stage 2 review rejected, in another form.

## 7. Telemetry saved

Raw and executed `T` and `rho` at initialization and after **every** epoch,
per-mode final values, derived `mu`, boundary occupancy, projection events and
pre-projection overshoot, `T`/`rho` gradient and update norms, `log_step`, and
the dimensionless per-mode product `T_i j_i` (magnitude, real and imaginary
parts). A fixed signal-only impulse/frequency probe with banded lag energies is
recorded at initialization and at the endpoint; the tail beyond the window is
labelled **unknown**.

These show whether the response band moved relative to the native modes. They
are response measurements, not semantic memory-importance measurements, and
neither a successful derivative check nor a change in `T` establishes task
improvement.

Final and best checkpoints, optimizer trees, batch statistics, RNG/provenance
and all epoch metrics are saved.

## 8. Budget

**One 1,200 s cap** covering backend startup, focused checks, compilation,
preflight, all five runs, validation and cleanup. This is the next training
budget, **not** 1,200 s on top of a simultaneously running combined batch — the
combined batch has not started and its launcher refuses to run unless
explicitly re-authorized.

The preflight measures compile and step cost for **all five arms** and projects
the whole batch including per-epoch validation and the host-side gate.
Training compilation, **evaluation compilation** and the steady-state
evaluation pass are measured separately: see the amendment below. If the
batch does not fit, the projection is reported and **comparative training does
not start**: no epoch, arm or control is reduced and the cap is not widened.
Execution status (`PASS` / `INCOMPLETE` / `FAILED`) is reported separately from
performance ordering — a completed unfavourable comparison is a result, not a
software failure.
