# TSS-based prospective pilot: protocol

Committed **before** any numerical execution. Coordinator sources:
`TSS_SMALL_PILOT_DESIGN_2026_09_16.md`,
`PROSPECTIVE_RESEARCH_ROADMAP_2026_09_16.md`,
`prospective_theory_extension_2026_09_14/LEARNING_THEORY_AND_CAUSAL_ALGORITHM.md`
and `.../CAUSAL_LEARNING_AGENT.md` (s11).

Prepared on branch `tss-pilot`, in a worktree, from
`cce550b5cc16d5c5d20857359f1ad8a30a4f7741`. The SSM studies and the running
learned-timescale experiment are untouched: this pilot adds files and changes
none of theirs.

**Nothing here has been executed.** No result is recorded until after a run.

## 0. What the two parts can and cannot establish

| | Part A | Part B |
|---|---|---|
| holds fixed | optimizer (exact BPTT), task, budget | forward model, parameters, data, loss |
| varies | the temporal forward law | how error is propagated in **time** |
| can support | a forward-model usefulness claim | a credit-assignment claim |
| cannot support | anything about local learning rules | anything about forward-model quality |

A better Part-A score is **not** evidence about credit assignment, and exact
agreement between BPTT and forward sensitivities in Part B is a **correctness
check**, not a gradient advantage. Neither part tests biological plausibility,
and neither reproduces TSS's published training schedule.

## 1. The clock and the coefficients

**Absolute clock: one input step is one unit of model time, `dt = 1`.** All
timescales below are in steps.

**Declared horizon `T = 8` steps.** It is *not* imported from the S5 studies:
those used 5 sample intervals for a 161-frame speech task, which is neither a
biological constant nor an optimal toy setting. `T = 8` puts the slow pole
`3T/2 = 12` at the order of the *short* recall delay (8) and well below the
long one (32), so the temporal layer must use its recurrence to span the delays
rather than being handed them by one sufficiently slow leak. Declared before
execution; not tuned on any score.

**Response coefficients are FIXED for this pilot** and taken from the declared
positive circuit — the symmetric reference in unnormalized form:

```
gamma = T = 8 ,   M = 3T^2/4 = 48 ,   rho = M/(gamma T) = 3/4
M s'' + gamma s' + (1 + T D)(s - f) = 0 ,  equivalently
M s'' + (gamma+T) s' + s = f + T f'
```

No independent fit of `M`, `gamma` and `T`. Learned response parameters are a
separate later arm and are **not** part of this batch.

### The one family, two sectors point

The finite-adaptation realization eliminates **exactly** to

```
tau_m eps s'' + (tau_m+eps) s' + s = f + (eps+tau_p) f'
```

so matching term by term gives the map

```
M = tau_m eps ,  gamma + T = tau_m + eps ,  T = eps + tau_p        (MAP)
```

Two consequences are enforced in code and checked:

1. **Our point has an exact adaptation twin.** `tau_m = 3T/2 = 12`,
   `eps = T/2 = 4`, `tau_p = T - eps = 4`. `tests/` checks that the twin
   reproduces arm 4's trajectory **and its gradients**. This is an
   implementation check, never a fifth competitor — identical parameterized
   functions are not competing architectures.
2. **TSS's matching prescription `tau_p = tau_m` maps to `gamma = 0`**, so
   `M <= gamma T` fails for any `M > 0`. Arms 2 and 4 are therefore two
   declared **coefficient sectors of one family**, not equivalent realizations
   and not different model classes, and the circuit inequality excludes the
   TSS-matched sector. Measured by `tss_gamma_equivalent`, not asserted.

To keep the comparison about response **shape**, arm 2 is given the **same two
poles** as arm 4 (`tau_m = 12`, `eps = 4`). The arms then differ only in the
prospective zero: `f + T f'` against `f + 2T f'`. Arm 3's memory leak is set to
the same slow pole, `tau_mem = 12`, so no arm starts with a longer intrinsic
memory than another.

## 2. Part A — task

64 steps. Two supervised outputs share **one** temporal representation:

* reconstruct the current scalar signal `u_t` at every step;
* report which of 8 cues was written earlier, when a query marker arrives.

`u_t` is AR(1) with **unit stationary variance** and correlation time 4 steps.
The cue is written at a step drawn from `{4..8}`; the query arrives
`delay ∈ {8,16,32}` steps later, so it lands at 12..40, always inside the
window. The content channel carries an **independent** distractor one-hot at
every step except the write step. Cue class, current signal and distractors are
drawn independently. Write and query markers are given identically to every
arm. 11 input channels.

**Loss**: mean current-signal MSE **plus** mean recall cross entropy over query
positions, each averaged separately before summing, so 63 non-query steps
cannot drown out one recall decision. Both metrics are reported separately, and
recall is reported per delay.

**No** pooling, flattening, cue buffer, absolute-time feature or future input.
The readout sees **only** the temporal state — feeding the input forward would
let it reconstruct the signal without the temporal layer and erase the
trade-off the task exists to create.

**Evaluation set**: fresh, never trained on, with exactly equal counts for all
8 classes × 3 delays (8 repetitions each, 192 sequences).

**Checks, not competitors**: (a) a cue shuffle — the recall output should
follow the *new* cue; (b) distractor replacement — recall should be unchanged;
(c) a closed-form ridge classifier on the **query-step input alone**, fitted and
scored on the same data so it upper-bounds linear leakage. It must sit near the
1/8 chance level; if it does not, the task leaks and the recall numbers mean
something else.

## 3. Part A — the four temporal laws and the state budget

Primary budget: **16 real temporal coordinates**.

| # | arm | temporal states | units | why |
|---|---|---|---|---|
| 1 | `ideal_prospective` | **0** | 16 | the memory-**cancellation** control: `s = f(s,x)` solved as a fixed point |
| 2 | `tss_finite_adaptation` | 16 (8×2) | 8 | TSS Eqs. 6–7 with its own `tau_p = tau_m` matching |
| 3 | `memory_then_prospective` | 16 (16×1) + 0 | 16+16 | memory-bearing leaky layer, then prospective processing |
| 4 | `retained_compartment` | 16 (8×2) | 8 | our circuit law with its coefficient ties intact |

Arm 1 is **not** artificially enlarged to look state-matched: having no driven
temporal state after its residual transient is the property it exists to
exhibit. It is given 16 units so its parameter count is not the smallest of the
set, which makes it a *stronger* control.

**Arm 3's processing stage is the ideal (stateless) prospective map**, which
lets the comparator keep the full 16 coordinates as *memory*. This is a
declared choice in the comparator's favour; a finite-adaptation processing
stage (which would cost states and shrink the memory layer) was **not** run.

**Parameter counts differ at this state budget.** They are recorded per arm.
State matching and parameter matching are not claimed simultaneously: if a
prospective advantage appears, a parameter-matched comparison is a prerequisite
before any capacity-efficiency claim.

## 4. Part A — initialization, optimizer, integration

* Shared input encoder `11 → 16` and both output heads are drawn from one key
  and are **bit-identical across arms** at a given seed; `shared_tensor_report`
  records which tensors actually matched rather than asserting it.
* Recurrent weights: `N(0, (0.9)^2/N)`. Biases zero.
* **Contraction cap.** The two arms that *solve a fixed point* (1, and arm 3's
  processing stage) have `‖W‖₂ ≤ 0.8` re-imposed after every update, with 40
  unrolled iterations and the residual measured and reported. Uniqueness of the
  solution is part of those arms' definition; the cap is not imposed on arms
  that integrate an ODE, which need no such condition.
* **Optimizer: Adam, `lr = 3e-3`, global gradient clip 1.0, 300 updates,
  batch 16, seeds 100/101/102.** One rate for every arm. **No development
  budget was spent on any arm** — the budget is therefore equal, at zero — and
  in particular no rate was chosen for the new model.
* Every arm sees the **same ordered stream** of training batches at a seed.
* **Integration: classical RK4, 2 substeps per input step, input held (ZOH).**
  Every driven core is nonlinear in `s`, so no exact discretization applies and
  a method must be declared. The step-refinement check (`refinement_error`,
  factor 4) must be below `1e-4` relative for every ODE arm; it is a check, not
  a formality, and a coarse step that has not converged fails it rather than
  silently biasing an arm.
* The ideal control is checked to have **no memory** (perturbing an earlier
  input leaves later outputs unchanged) and the other three to **have** it. A
  control that silently retained memory — through a backward-difference
  parasitic state, say — would invalidate the comparison.

## 5. Part B — credit assignment on one frozen forward model

Two layers of four retained-compartment cells, **spatially feedforward**:

```
layer 1 drive:  f1 = W1 x + b1          (no state dependence)
layer 2 drive:  f2 = W2 tanh(s1) + b2
```

so the error coupling is strictly triangular. This is deliberate: s7 of the
causal-error note exhibits a stable forward node whose reciprocal error loop
has pole **+0.125**. Forward stability does not establish error-dynamics
stability, and this pilot is not entitled to assume it. The witness is
reproduced in the audit output. **Recurrent error dynamics are out of scope.**

32 trajectories × 64 steps, parameters frozen, initial states fixed.
Predeclared input bandwidths `Ω ∈ {0.05, 0.15, 0.40}` rad/step, band-limited
and unit-variance; targets generated the same way.

**Reference.** Each layer's per-step drive carries an additive perturbation
`d_t` held over the step exactly as the drive is; then `rho_exact(t) = dJ/d(d_t)`
is the exact discrete drive adjoint. Because `r_t` is held over the same step,
`G_W = Σ_t rho_t r_t^T` is **exact** for this cascade — checked numerically
(`1e-9`) rather than assumed. Cross-checked three ways: reverse-mode,
forward-mode sensitivities (`jacfwd`), and central finite differences,
**including the initial states** `z0_1`, `z0_2`.

**The two approximations**, both stable causal filters integrated forward,
discretized **exactly** (ZOH) so the comparison is not polluted by their own
integration error:

* **(i) reciprocal / GLE-style ordinary-prospective baseline** — `E_eps` of
  (C2)-(C3) cascaded with `R_delta` of (C5)-(C6) so the path is strictly
  proper. Same phase as the adjoint, different magnitude, `O(ω²)` error.
* **(ii) moment-matched** — `K_eps` of (M2)-(M3): three low-pass states, weights
  summing to one with `w2 < 0` supplying the extrapolation, `O(ω³)` error.

Predeclared `eps ∈ {0.25, 0.5, 1.0} × t_-` with `t_- = T/2 = 4`, and
`delta = eps`. **All** of them are reported at **all** bandwidths; neither
approximation is presented at a selected operating point.

**Metrics**: gradient cosine similarity, relative error, fraction of negative
cosines across trajectories, and the actual objective change after a small
**norm-matched** update at `‖step‖ ∈ {1e-3, 1e-2}`. Absolute error is reported
and the cosine flagged undefined whenever the reference gradient is below
`1e-12`. Assessed parameters: `W1, b1, W2, b2`.

**Audited before use, and reported**: the executed filter poles, the peak-gain
bound (M6) and the small-`eps` peak estimate `2c₂/(3√3 eps²)`, the exact
remainder coefficients `B3, B4` with `c₂ ≥ γ²` and `c₃ ≥ γ³`, and the
**measured** band error for both filters — because a cubic asymptotic order
settles nothing at a finite bandwidth.

## 6. Decision rules, predeclared

**Part A.** Beating arm 1 alone is **insufficient**: it is the memoryless
control and is expected to fail a balanced history-only query. The comparisons
that count are arm 4 against arms **3** and **2**.

| outcome | declared reading |
|---|---|
| arm 4 better on recall **and** no worse on signal MSE, in all three seeds, against arms 2 and 3 | a development signal; a parameter-matched confirmation is then required before any capacity claim |
| better recall, worse MSE | a **trade-off**, reported as such. Not an unqualified win |
| mixed or worse | no signal |
| every arm within 5 pp of the 12.5 % chance level on recall | recall is **inconclusive**; only the tracking comparison is interpretable and the pilot does not answer the memory question |

No imported threshold, no significance claim from three seeds, no seed or
checkpoint selection, no automatic enlargement or rescue sweep.

**Part B.** An approximation is *usable* only if, at **every** predeclared
bandwidth: mean cosine > 0, the negative-cosine fraction is ~0, and the
norm-matched update decreases the objective. It is *preferable to the
reciprocal baseline* only if it achieves lower relative error across the
predeclared bandwidths — not at one chosen bandwidth. **Local-rule training
does not follow from this batch under any outcome**; it follows only after the
probe is interpretable and favourable, and it would be a separate declared
experiment.

## 7. Budget and execution

**One hard 600 s cap** covering backend startup, focused checks, compilation,
preflight, Part A, Part B and cleanup, with a 30 s reserve. Part A's preflight
measures compile and steady step cost for **every** arm and projects
`seeds × updates`; if it does not fit, the projection is reported and
comparative training **does not start** — no arm, seed or update count is
reduced and the cap is not widened. Status (`PASS`/`INCOMPLETE`/`FAILED`) is
reported separately from any performance ordering.

Prepared locally with **no local numerical runs**; all execution is on the
cluster. This launcher must not run alongside another cluster batch.

## 8. Unresolved scientific issues, stated rather than smoothed over

1. **Neither causal approximation supplies an initial-state derivative.** Its
   error states start at zero while the exact adjoint is terminal-valued. The
   reference differentiates `z0`; the approximations cannot. The gradient
   comparison is therefore restricted to `W1, b1, W2, b2`, and the asymmetry is
   reported rather than hidden.
2. **Boundary prescription.** For the same reason the causal and exact
   teaching variables disagree near both sequence ends by construction. This is
   structural, not a tuning artifact.
3. **Continuous versus discrete.** The reference is the exact gradient of the
   *discretized* system; the approximations are continuous-time filters. The
   gap is *measured* (`exact_backward_filter` against `rho_exact`) but not
   eliminated.
4. **Arm 2 versus arm 4 is a coefficient-sector comparison**, not an
   architecture comparison. That TSS's matched point sits outside our
   admissible sector (`gamma = 0`) is a checkable fact; whether that exclusion
   is a *meaningful constraint* or merely a restriction is **not** settled by
   this pilot.
5. **Arm 3's processing stage** is the ideal stateless map, chosen to give the
   comparator the full memory budget. A finite-adaptation processing stage was
   not run, and could behave differently.
6. **Parameter counts differ** at matched state budget; no capacity-efficiency
   claim is available from this batch.
7. **`T = 8` is declared, not derived.** A different horizon could reorder the
   arms, and this pilot does not explore that.
8. **One learning rate, no development budget for any arm.** Equal at zero is
   equal, but it is not the same as each arm being well tuned.
9. **300 updates is very small.** Recall at or near chance for every arm is a
   live possibility and is a declared outcome, not a failure to hide.
10. **The small-nudging theorem is not tested here.** Its learning functional
    is the squared residual of the forward equation and is explicitly *not* the
    physical action; Part B's reference is the constrained derivative of the
    stated objective, which is a different object.
11. **Recurrent error-loop stability is not tested.** Part B is feedforward by
    construction, and nothing here licenses applying either filter around a
    recurrent loop.
