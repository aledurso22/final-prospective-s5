# Stable generalized prospective continuation — frozen protocol

**Frozen before any execution. Amended before any execution by the static
review of `97cedfa` — see §13, which supersedes the corresponding passages
below.** Nothing in this study has run anywhere — no
check, calibration, restore or training, on the cluster or locally. Local work
was limited to editing, `ast` syntax checks and static symbol/signature audits.

| | |
|---|---|
| Authoritative brief | `NEXT_PROSPECTIVE_S5_CODING_BRIEF_2026_09_16.md` |
| Binding references | `PROFESSOR_TO_GENERAL_PROSPECTIVITY_CONTRACT_2026_09_16.md`, `GENERALIZED_RESPONSE_DOMAIN_AUDIT_2026_09_16.md` |
| Superseded here | the implementation choices of `CORRECTED_PROSPECTIVE_SSM_APPLICATION_2026_09_16.md` (fixed T, projected delta), the deferred composition `gp_rho_prospin` and its passive `rho <= 0.9999` bound |
| Branch | `stable-generalized-prospective-s5`, parent `89a05ca` (`learned-response-timescale`, which contains the combined and timescale work) |
| Code | `s5/stable_gp.py`, `s5/rawat_s5.py`, `s5/substrate_diagnostics.py`, `experiments/gp/stable_gp_study.py` |
| Checks | `tests/test_stable_gp.py`, `tests/stable_gp_float32_probe.py` |
| Launcher | `bin/run_experiments/cluster_stable_gp.sh` |

Historical arms, runs, reports and artifacts are untouched. The deferred
`gp_rho_prospin` composition stays selectable and unchanged, and it is **not**
run here.

## 1. Objective

The objective is to improve on the **working Rawat prospective-input S5**. The
candidate starts as that exact function, keeps its prospective input, and
learns a wider stable recurrent response. A gain over a static cancellation
control, or over native S5 while still losing to Rawat, does not satisfy this
objective.

## 2. The three arms

All three continue from **one** saved, validation-selected Rawat checkpoint.
They share Rawat's substrate: the alpha input gain, pole clipping
`Re(lambda) <= -1e-4`, native `D`, readout, normalization, residual path, and
architecture (depth 4, width 32, conjugate symmetry, 8 HiPPO blocks, ZOH).

| | Arm | Response | Added leaves per layer | Carry per layer |
|---|---|---|---|---|
| A | Rawat unchanged | `alpha_p_s5` | none | 32 physical + 32 input buffer |
| B | Rawat + learned input horizon | `rawat_learned_input` | `log_T_in` (q), 16 | 32 + 32 |
| C | B + stable generalized recurrence | `sgp_learned_input` | `log_T_in` (q), `log_rho_rec` (r), `log_T_rec` (t), 48 | 32 + **32 auxiliary** + 32 |

`q` is **shared in kind** between B and C: both learn it from the same start,
under the same optimizer group. The candidate's input horizon is not held
fixed while its control learns one.

### Law, per stored conjugate mode, gamma normalized to one

```
rho T s'' + s' + R + T R' = 0 ,   R = j s - b (x + T_in x')
j = -Delta lambda = a + i omega ,  b = Delta B_c   (clock absorbed exactly once)
T_in = 5 exp(q) ,  rho = exp(r) ,  T = 10 exp(t) ,  M = rho T   (derived; §13 R0)
```

These are direct log coordinates, all initialized at **exactly zero**, which
gives `T_in = 5`, `rho = 1` and `T = 10` (amended from 5, §13 R0). The conjugate partner shares the real
coefficients. Coefficients are constant within a sequence. Gamma stays absorbed
in the clock, and the mass is derived, never learned.

### Nesting — a property of the executed algebra

```
G(p) = b (1 + T p)(1 + T_in p) / [rho T p^2 + (1 + T j) p + j]
```

At `rho = 1` the denominator factors as `(1 + T p)(p + j)`. Therefore:

* **C at r = 0 equals B for every q and t**, forward and in every shared
  gradient;
* **B at q = 0 equals A**;
* at `rho = 1` the task derivative with respect to `t` vanishes. `T` is
  unidentifiable there, which is expected; it becomes learnable once `rho`
  departs from one.

There is no static bypass, no straight-through estimator and no stop-gradient.

### Production realization (C)

This reuses `mass_block_zoh` and the two-tap scans of `s5/gp_fixed.py`.

```
A = [[-j/rho, -(1-rho)/rho], [-j/(rho T), -1/(rho T)]] ,  B = [b/rho, b/(rho T)]
A_bar = exp(A) ,  B_bar = int_0^1 exp(A t) B dt
J_in  = T_in * A_bar @ B      (CONTINUOUS B)
z_k   = A_bar z_{k-1} + (B_bar + J_in) x_k - J_in x_{k-1}
```

Tokens jump at the start of the unit interval and outputs are read at its end.
Only `s` is read through `C`, and native `D` is unchanged. Reset clears **both**
the block carry and the previous input. At `rho = 1` the `s` row of `A` has an
exactly zero `v` entry. No zero-mass feedthrough is appended.

B is Rawat's own diagonal two-tap coefficient helper with a per-mode horizon.
At `q = 0` its horizon is exactly `5.0`.

## 3. Stability domain and projection

With `c = 1 + T a`, both roots are strictly stable iff

```
S = a c^2 + omega^2 (T c - rho T) > 0
omega != 0 :  rho < rho_max = 1 + z ,  z = T a + a c^2 / (T omega^2) > 0
omega == 0 :  every rho > 0
```

`rho = 1` is always strictly inside. The **passive** reciprocal
two-compartment identification holds only on `rho <= 1`, and its occupancy is
reported. Beyond it the model is a stable **nonconservative** mechanical
realization of the same response equation, not the passive circuit. Arbitrary
positive M is not claimed stable for a complex mode. The old
`rho <= 0.9999` passive-sector restriction is **not** restored.

**Post-update projection**, `s5/stable_gp.py::project_stable_domain`, is applied
inside the jitted step after every optimizer update:

```
SUPERSEDED (§13 R1):  r <= log1p((1 - epsilon_num) z)
IN FORCE:             L = log1p(z) ,  r <= max(0, L - 32 eps (1 + |L|))   complex modes
                      r unbounded                                        omega == 0 exactly
```

* It is computed from the **updated** `Lambda_re`, `Lambda_im`, `log_step` and
  `log_T_rec` of the **complete** layer, with the forward pass's clip and
  clock. Only `r` is moved.
* It is evaluated in log arithmetic:
  `logaddexp(0, log1p(-eps) + logaddexp(log(Ta), log a + 2 log c - log T - log omega^2))`.
  The `omega^2 = 0` division is **masked before** evaluation, and real modes
  get `+inf`.
* Optimizer state is untouched. Telemetry records proposal events, maximum
  proposed overshoot and the minimum post-projection log margin.
* There is no forward clip, no fixed leaf-name interval and no bound on `q` or
  `t`.

**Executed validation**, `executed_domain_report`: `rho`, `T` and `M` are formed
in the executed dtype as the forward pass forms them. `S` and `rho < rho_max`
are then evaluated in **float64** from those executed values, so a float32
rounding across the boundary is detected rather than reproduced. A non-finite
or non-positive mass or horizon fails. This runs on C's starting point, after
every epoch of every C run, and on its final checkpoint. A failure is a
**FAILED** run and is never clamped.

## 4. Actual TSS Eq. (17) — kept separate

`s5/stable_gp.py` implements the equation-level references:

```
TSS Eq. (17):   s_{k+1} = s_k + (h/T)(-s_k + f_k) + f_k - f_{k-1}
generalized:    [M + h(gamma+T)] d_k = M d_{k-1} + h^2 (f_k - s_k) + h T (f_k - f_{k-1})
```

At `M = gamma = 0` the generalized form recovers Eq. (17) exactly, including
its previous-drive convention. Eq. (17) **retains discrete history**, and the
characteristic polynomial for `f = a s` depends on `a`. The exact continuous
memoryless reduction (`prospective_recurrence`) has none. They are never
described as interchangeable. The production ZOH law is a different
discretization and is **not** claimed identical to Eq. (17). These are
equation-level checks, not a trained TSS benchmark.

## 5. Focused checks, before training

Checks run on the cluster inside the cap. Tolerances are frozen in the test
module headers.

| Check | Dtype | Tolerance |
|---|---|---|
| Characteristic polynomial of the executed block equals the law | f64 | 1e-10 rel |
| Transfer identity, and the rho = 1 factorization equal to Rawat's input-horizon response | f64 | 1e-10 rel |
| Analytic domain vs **executed-generator eigenvalues**, inside and outside `rho_max (1 -+ 1e-3)`; real modes stable at every rho | f64 | sign agreement |
| Passive sector is a strict subset of the domain | f64 | exact |
| Log bound equals the closed form | f64 | 1e-12 rel |
| Projection recomputes from the updated complete layer, moves only r, and ignores inward points and real modes | f64 | exact / 1e-12 |
| Leaves declared after the common tree, zero, float32; common tree identical to `alpha_p_s5` | f64 | bitwise |
| Production two-tap block coefficients vs independent reference, per-mode T, rho > 1, T_in, P != H | f64 | 1e-10 rel |
| **C(r=0) = B for q in {0, random} and t in {0, random}**: layer outputs and all shared + q + input gradients; `dL/dt ~ 0` | f64 | 1e-9 / 1e-8 / 1e-9 |
| **B(q=0) = A**: outputs, parameter and input gradients | f64 | 1e-9 / 1e-8 |
| rho = 1 decouples the s row at the first token | f64 | exact / 1e-14 |
| Streaming carries and resets clear state **and** delayed input, for B and C | f64 | 1e-10 rel |
| Full network nesting A < B < C: logits, input and every shared gradient | f64 | 1e-9 / 1e-8 |
| Nonzero (q, r, t) tangent vs central differences at h = 1e-5 **and** 1e-6; T derivative nonzero away from rho = 1 | f64 | 1e-6 rel |
| First common update with added leaves **frozen** agrees; optimizer labels | f64 | 1e-8 of lr |
| Real update drives r outward, projection restores the domain, gradient still finite | f64 | domain |
| Checkpoint round trip reproduces logits | f64 | bitwise |
| Source selection applies the original rule; screen requires both baselines, every stream, lower CE, complete pairs | host | exact |
| Eq. (17) recovery on a nonlinear drive; Euler and heavy-ball rows; Eq. (17) history vs memoryless; unforced drift vs decay | f64 | 1e-12 abs |
| **float32 probe**: projection over 40 trials × 2 layers × 16 modes including nearly real and exactly real modes, far and near proposals, coupled pole/clock/T updates, all validated in float64; production A/B/C identities at the study's gate tolerances; T derivative | f32 | domain / gate |

## 6. Source checkpoint

* **Where:** `$PROSPECTIVE_RUNS/stage2/manifest.jsonl`, `alpha_p_s5` runs with
  exit 0.
* **Rule:** Stage 2's original rule, applied before any new result. Within a
  run, the saved `best` checkpoint is the first strict maximum of validation
  accuracy. Across its learning-rate slots: validation accuracy, then
  validation CE, then declared order (1e-3 before 3e-4). The expected result is
  `alpha_p_s5__lr1e-3__seed100`, best at epoch 9, 95.00 % / CE 0.28953. The
  runner **derives** this from the files and does not assume it.
* **Recorded:** run directory, source commit, SHA-256 of `best.msgpack`,
  `best.meta.json`, `config.json` and `metrics.jsonl`.
* **Reproduction:** exact match of correct counts, and |CE difference| ≤ 1e-5.
  Otherwise FAILED.
* **Config:** a mismatch with the declared screen **refuses** the run (depth,
  width, batch, weight decay, clipping, label smoothing, no SSM decay
  exemption, final learning rate).
* **Blockers:** an absent or incompatible checkpoint is reported as a
  **BLOCKER**. No substitute is used.

## 7. Warm start and the identity gate

* Common weights and batch statistics are **cloned explicitly** into all three
  arms. Only each arm's declared added leaves may be absent from the source,
  and they must be zero `(16,)` vectors; anything else refuses. Common-parameter
  and batch-statistics digests must be identical across arms.
* **Before any arm trains**, all on production float32 on the GPU:

| Pair | Logits (256 validation probes, rel) | Common gradients (per leaf, rel) | First common update, added leaves frozen (max / lr) |
|---|---|---|---|
| B(q=0) vs A | ≤ 5e-4 | ≤ 2e-3 | ≤ 2e-3 |
| C(r=0) vs B | ≤ 5e-4 | ≤ 2e-3 | ≤ 2e-3 |

**Amended by §13 R2:** gradients use a mixed absolute/relative criterion on
the pair-specific shared leaves (q included for C vs B) and on input
gradients; the update gate is a routing identity with COPIED gradients;
independently computed first updates are reported, not gated.

  Plus C's initial executed stable domain. These tolerances allow for the
  repository's declared float32 resolution of the Padé block exponential
  (`F32 = 2e-4`) propagated through four layers. The **exact** identities are
  established in float64 by the checks.

  The update metric uses `lr` as its scale because Adam's first step is about
  `-lr g/(|g| + eps)` entrywise, which would magnify float noise in near-zero
  entries under a per-leaf relative norm.

  **Any disagreement stops the screen as FAILED**, with no changed tolerance,
  clamp or baseline switch.
* Epoch-0 full validation is measured for every arm.

## 8. Training

| | |
|---|---|
| Streams | seeds **201, 202, 203**: data order `epoch_batches(n, 32, seed, epoch)` and dropout `PRNGKey(seed)` split per step, **identical across the three arms of a stream** |
| Epochs | **10** additional epochs per run; **9 runs** |
| Optimizer | **reset in all arms** — a weight warm-start, not an exact resume |
| Schedule | cosine `1e-3 -> 1e-6` over the ten epochs, both groups |
| Groups (published labels) | `common`: AdamW(0.9, 0.999, 1e-8, weight decay 1e-4) on every inherited parameter, Stage 2's policy. `response`: Adam(0.9, 0.999, 1e-8), **no weight decay**, on `log_T_in`, `log_rho_rec`, `log_T_rec` |
| Clipping | global norm 1.0 over all gradients; label smoothing 0.1; identical across arms |
| Data | train and validation only. **The test split is never opened.** |

Every epoch records: training loss and accuracy, validation accuracy and
unsmoothed CE, epoch time, projection telemetry, the executed-domain result and
passive occupancy (for C). Each final checkpoint is saved.

## 9. Performance screen

The **primary endpoint is epoch 10**. Best-validation epochs are descriptive
only and cannot replace it.

**Development success** requires all three conditions, against **both** A and B:

1. mean validation accuracy gain of **≥ 0.3 percentage points**;
2. **positive** paired accuracy difference in **every** stream;
3. **lower mean unsmoothed validation CE**.

All three pairs must be present. Every paired value, and B − A, is reported.
This is a development screen on one source checkpoint and three continuation
streams. It is not a significance test, not an independent-initialization
result and not a benchmark claim.

Selecting the original baseline as a fallback is not a win for the new method.
A strong attribution or superiority claim would need fresh independent
confirmation, comparable tuning budgets and a generic equal-carry control.

## 10. Budget and status

* **Budget:** one hard **1200 s** cap covering startup, checks, restore, the
  gate, compilation, preflight, the nine runs, validation, diagnostics and a
  40 s reserve.
* **Preflight:** measures every arm and projects all nine runs, including the
  per-epoch validation passes, the measured diagnostics and serialization.
  **Amended by §13 R3:** steps are timed through the same host path as
  training, and epoch acceptance/persistence and final
  serialization/diagnostics are timed separately. A
  retrace refuses the screen. If the projection does not fit, the screen is not
  started and the measured obstruction is reported. No seed, epoch, arm or
  schedule is trimmed.
* **Status:** `PASS` means the batch completed numerically, and is separate
  from `DEVELOPMENT SUCCESS`. `FAILED` covers a source, identity, domain,
  non-finite or check failure. `INCOMPLETE` means the deadline was reached.
* **No retry**, no expanding sweep, no rescue run.

## 11. Recorded per run

Parameters, including the added count. Carry: physical, auxiliary and input
buffer. Wall and epoch time. Final `T_in`, `rho`, `T` per layer with
distributions. Executed modal stability margins, passive-sector occupancy and
projection telemetry. **Current-input and history response changes**: per
layer, `||K_0 - diag D||` and `||K_1..K_127||` over a 128-lag window, from start
to end; the tail beyond the window is not bounded.

## 12. Guarantees: checked versus analytical

**Checked, on execution:** the domain against executed eigenvalues, the
projection, the float32 interior, the exact nesting in value and gradient, the
transfer factorization, streaming and resets, the tangent, frozen-extra
updates, the Eq. (17) recovery, the source reproduction and the identity gate.

**Analytical only:** stability for the constant diagonal core. This does **not**
extend to the nonlinear network, to coefficients that change between
minibatches, or to local error loops.

**Empirical hypothesis, the thing under test:** that the wider learned response
improves on both working prospective baselines.

**Not tested here:** spatial-only training, Delta/Momentum augmentation, and any
TSS benchmark.


## 13. Pre-execution amendments — static review of `97cedfa`

Source: `STABLE_GP_REVIEW_97cedfa_2026_09_16.md`. Recorded **before** any
check, restore or training ran. No result of this study exists.

The unchanged parts are the equation, the three arms, the shared learned input
horizon, the source checkpoint and its selection rule, the continuation
streams, epochs, schedule and optimizer groups, the success criteria, the
tolerances of §5, and the 1200 s cap.

### R0 — recurrent reference horizon 10 (coordinator correction)

**Provenance.** The equal initial horizons `T = T_in = 5` were prescribed by
the coordinator's brief and implemented faithfully at `97cedfa`. This is a
coordinator correction, not an implementation defect.

**Why.** At `r = 0`,

```
dG/dr = -b T p^2 (1 + T_in p) / [(1 + T p)(p + j)^2]
```

At `T = T_in` this reduces to `-b T p^2 / (p + j)^2`, and the auxiliary pole
`-1/T` drops out of the first-order direction. At `T != T_in` its residue is
`-b (1 - T_in/T) / [T^2 (j - 1/T)^2]`, nonzero for nonzero drive.

**Change.** `T = 10 exp(t)`; `T_in = 5 exp(q)`; `rho = exp(r)`. All raw
coordinates still start at zero, so C still starts at exactly B's function.
Ten is a transparent nondegenerate initialization. It is not a measurement or
a predicted optimum, and no horizon sweep is run. Fixtures use the single
constant `RECURRENT_T_REFERENCE`.

**New checks.** The tangent identity against exact AD of the transfer
(1e-8 relative). The equal-horizon cancellation as an analytical regression.
The distinct-horizon residue at `delta = 1e-9` (1e-5 relative), with
`j != 1/T`. The module starts at `T = 10`, `T_in = 5`, `rho = 1`, equal to B.

None of this explains any earlier, different experiment retrospectively.

### R1 — representation-aware projection margin (coordinator correction)

**Provenance.** The fractional-excess interior `log1p((1 - 32 eps) z)` was
prescribed by the coordinator's brief and implemented faithfully. This is a
coordinator correction.

**Why.** That margin shrinks only the excess `z`. For small `z` its safety
distance is below one float32 step of `rho` above one, so a correctly rounded
`exp` can land beyond the boundary.

**Witness** (analytical): `T = 1`, `omega = 1`, `a = 3 * 2^-26`. Here
`rho_max - 1 ≈ 0.75` of a float32 step, the old bound rounds `rho` to
`1 + 2^-23`, and `S < 0`.

**In force:**

```
L = log(rho_max) = log1p(z) = logaddexp(0, log z)
log z = logaddexp(log(T a), log a + 2 log c - log T - 2 log|omega|)
r_upper = max(0, L - 32 eps (1 + |L|))     complex modes
unbounded                                  omega == 0 exactly
```

* `2 log|omega|` replaces `log omega^2`, so a complex mode whose `omega^2`
  underflows is not misread as real.
* `rho = 1` is retained exactly as the stable fallback.
* It is still direct log coordinates and post-update projection of `r` from
  the updated complete layer, with optimizer state untouched, no forward clip
  and no straight-through estimator.
* It is a conservative numerical interior, not a circuit restriction.
* The analytical stability domain and every §5 comparison tolerance are
  unchanged.

**New checks.** The float64 closed form of the new bound (1e-12). The witness
in float64 (the old bound fails, the new bound returns `r = 0`, stable) and
through the production float32 bound and `exp` in the probe. The existing
projection fixtures now use production float32 leaves, because a float64
fixture projected to its own `32 eps64` interior sits at the eigensolver's
resolution. The float32 probe is unchanged in scope.

If this policy fails the existing checks, the concrete obstruction is reported
before launch. No boundary policy is chosen from task scores.

### R2 — verification of the new direction and of numerical identity

**r-only derivative at the actual start `q = r = t = 0`.**
* float64: steps `1e-5` and `1e-6`, 1e-6 relative at each, `|jvp| > 1e-6`.
* float32 probe: steps `1e-2` and `3e-3`, 2e-2 relative at each,
  resolvability `|jvp| >= 100 eps32 |f| / h_min`.
* The fixture uses clock range `[0.1, 1]` so the path is resolvable.
* `t`'s task derivative at the start: `<= 1e-9 max|dL/dr|` (float64) and
  `<= 1e-3 max|dL/dr|` (float32).
* The existing off-baseline joint tangent is kept.

**Pair-specific shared leaves.** Comparisons use the intersection of the two
trees, so C vs B includes `q`. Input derivatives are included in the
production fixture and in the gate.

**Mixed absolute/relative gradient criterion.** A leaf passes iff it is finite
and

```
||g_x - g_ref||_F <= max(GRAD_REL_TOL ||g_ref||_F,  1e3 * eps(dtype) * G_ref)
```

Here `G_ref` is the reference global norm over the compared leaves. The
absolute floor is `1.2e-4 G_ref` in float32 and `2.2e-13 G_ref` in float64.
Every leaf's absolute and relative error, magnitude, finiteness and deciding
branch is logged; no leaf is dropped.

The motivation is the training-mode null direction: the encoder Dense bias
shift is removed by batch normalization, so its exact gradient is zero.

**Fixtures:**
* rounding-level residual on an analytically zero leaf: accepted, absolute
  branch;
* a finite `1e-2 G` gradient on that leaf: rejected;
* a 5 % error on a resolved leaf: rejected;
* NaN: rejected and recorded;
* the actual training-mode null direction on a small `BatchRawatClassifier`,
  where the encoder bias gradient is `< 1e-10 G`.

The criterion was not tuned against any checkpoint.

**Optimizer checks, split in three:**
1. **Forward/gradient identity** uses independently computed gradients with
   the mixed criterion.
2. **Routing identity** uses the SAME shared gradient values copied into both
   trees, with genuinely extra leaves frozen, and keeps the 2e-3·lr gate.
3. **Independently computed first updates** are REPORTED, not gated: their
   maximum difference over lr, near-zero reference coordinates (`|g| < 1e-7`),
   and the post-update logit discrepancy. A finite-precision forward identity
   is not claimed to imply identical trajectories.

This amends the measurement of numerical equality only. The performance
success criterion is unchanged.

### R3 — preflight times the executed path; epochs persist

* `step_loop` is the single host path for training and timing: slicing,
  transfer, RNG split, the jitted step with projection, and synchronization of
  every recorded scalar.
* Preflight times 40 steps after a warm-up; compilation is incurred, not
  projected.
* It separately times a validation pass (32 batches, scaled), per-epoch
  acceptance and persistence, and final serialization and diagnostics.
* A retrace still refuses the screen.
* Every completed epoch's metrics, acceptance and telemetry are written
  immediately to `runs/seed<k>_<arm>.json`. This preserves artifacts; it is
  not a resume or retry feature.

### R4 — validation of executed arithmetic

* `executed_domain_report` forms `T_in`, `rho`, `T`, `M` and the block
  generator entries in the **executed dtype**. It requires finite, positive
  `a`, `T`, `rho`, `M`, `T_in` and finite `omega` and generator entries.
* It then assesses in float64, **separately**, the formula diagnostic
  (`S > 0`, `rho < rho_max`) and the eigenvalues of the rounded executed
  generator.
* It covers B (`T_in`) as well as C.
* **Fixture:** a real mode with `rho, T ~ 1e30` in float32, whose float64
  product is finite but whose executed mass overflows, fails; an underflowed
  `T_in` fails for B.
* Every epoch's acceptance checks parameters, optimizer state, normalization
  state and every reported scalar for finiteness, plus the executed domain for
  B and C. A failure is a FAILED run, persisted.
* Summaries record the executed-dtype coefficients, not a float64
  recomputation.

### Statement of scope

Physics and exact nesting do not imply that finite BPTT reaches a better task
optimum, or that held-out accuracy improves. No new observations or future
information enter the network. Larger general SSMs can represent such rational
responses, so an equal-capacity control remains necessary for any attribution
claim.
