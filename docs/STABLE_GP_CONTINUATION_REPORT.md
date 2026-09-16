# Stable generalized prospective continuation — report

**Status: dispatch 1 at `c94ab1f` FAILED at the focused checks (2 of 38), no
training. Logs preserved. Awaiting diagnosis and authorization; no retry.**

No check, restore, calibration or training for this study has run anywhere.
Local work was limited to editing, `ast` syntax checks, and static audits of
cross-module symbols and call signatures. Every result section below is empty
until a cluster run fills it, favourable or not.

## Provenance

| | |
|---|---|
| Brief | `NEXT_PROSPECTIVE_S5_CODING_BRIEF_2026_09_16.md` |
| Protocol | `docs/STABLE_GP_CONTINUATION_PROTOCOL.md` (frozen before execution) |
| Branch | `stable-generalized-prospective-s5` |
| Parent | `89a05ca` (`learned-response-timescale`) |
| Worktree | `/private/tmp/wt/sgp` |
| Implementation commits | `97cedfa` (initial), R0–R4 correction on top (see git log) |
| Executed commit | *(recorded from the launcher)* |
| Source checkpoint | *(derived and hashed by the runner from `$PROSPECTIVE_RUNS/stage2`)* |
| Command | `bash bin/run_experiments/cluster_stable_gp.sh` |
| Artifacts | `$PROSPECTIVE_RUNS/stable-gp/<stamp>/`, logs in `$PROSPECTIVE_RUNS/stable-gp/logs/<stamp>/` |

## What was implemented

| Piece | Location |
|---|---|
| Modal `j` from raw leaves with the forward clip and clock; `S`; log-arithmetic projection bound with masked division; coupled post-update projection of the complete layer; float64 validation of executed coefficients; transfer function; **TSS Eq. (17)** and its generalized discrete descendant | `s5/stable_gp.py` (new) |
| `rawat_learned_input` (B) and `sgp_learned_input` (C): leaves declared after the common tree, coefficients, two-tap block and diagonal realizations, carries, joint resets, state counts, arms | `s5/rawat_s5.py` (additions only) |
| Diagonal and block two-tap diagnostics for B and C; the one-tap block remainder refuses the two-tap block law instead of returning a wrong tail | `s5/substrate_diagnostics.py` (additions only) |
| Source selection, restore and reproduction, whitelisted warm start, optimizer groups, identity gate, epoch-0 measurement, preflight, nine paired runs, per-epoch domain validation, telemetry, response change, screen | `experiments/gp/stable_gp_study.py` (new) |
| Focused float64 checks and float32 probe | `tests/test_stable_gp.py`, `tests/stable_gp_float32_probe.py` (new) |
| Capped launcher | `bin/run_experiments/cluster_stable_gp.sh` (new) |

No historical arm, runner, protocol or report was modified. `gp_rho_prospin`,
its `rho <= 0.9999` bound and its deferred runner are unchanged and not run.

## Deliberate choices a reviewer should check

1. **B is implemented diagonally, and C as a block.** B uses Rawat's own
   coefficient helper with a per-mode horizon, so `B(q=0) = A` holds with
   identical arithmetic. C uses the Padé block exponential. The **exact**
   `C(r=0) = B` identity is therefore checked in float64 (1e-9 logits, 1e-8
   gradients). In float32 it is gated at 5e-4 logits, 2e-3 gradients and 2e-3
   of `lr` for first updates. Those production-gate tolerances are a judgement
   call and are stated in the protocol with their justification.
2. **Update comparison metric** (superseded in scope by R2). The gated update
   check is now a routing identity on COPIED gradients, still measured as the
   maximum entrywise difference in units of `lr`. Independently computed first
   updates are reported only.
3. **Projection scope.** Only `r` is projected, and its bound is recomputed from
   the updated poles, clock and `T`. `q` and `t` are unconstrained, and their
   non-finiteness is caught by the per-epoch executed-domain validation and the
   finiteness checks.
4. **float32 probe coverage.** Nearly real modes go down to `|Im lambda| = 1.2e-4`,
   with `z` up to about 6e12, plus exactly real modes, and (R1) the small-`z`
   rounding witness. The covered range is stated explicitly. Trained checkpoints are validated per epoch regardless of
   whether they fall in that range.

## Static review of `97cedfa` — dispositions

Source: `STABLE_GP_REVIEW_97cedfa_2026_09_16.md`. Full text of each amendment:
protocol §13.

| | Finding | Provenance | Disposition |
|---|---|---|---|
| **R0** | Equal horizons `T = T_in = 5` cancel the auxiliary pole in the first-order `r` direction | **Coordinator correction**: the brief prescribed equal horizons; implemented faithfully | Recurrent reference `T = 10 exp(t)`, input `T_in = 5 exp(q)`; all raw coordinates still zero, so C still starts exactly at B. Tangent identity, equal-horizon cancellation and distinct-horizon residue checks added |
| **R1** | The fractional-excess interior can round past the boundary when `z` is small (witness `a = 3·2⁻²⁶`) | **Coordinator correction**: the brief prescribed `log1p((1-32 eps) z)`; implemented faithfully | `r ≤ max(0, L - 32 eps (1+\|L\|))`, `L = log1p(z)`, `2 log\|omega\|` for underflow safety, `rho = 1` fallback. Witness checked in float64 and through the production float32 path; projection fixtures moved to production float32 |
| **R2** | Missing `r`-only check at the start; the shared-leaf comparison omitted `q`; purely relative identity undefined on null-direction leaves; update gate conflated rounding with routing | Implementation | `r`-only JVP vs FD at the start in both dtypes, and `dt ≈ 0`. Pair-specific shared leaves plus input gradients. Mixed absolute/relative criterion with published fixtures, including the training-mode encoder-bias null direction. Routing identity with copied gradients; independent first updates reported only |
| **R3** | Preflight timed a resident batch, not the training host path; epoch records were lost on a stop | Implementation | Shared `step_loop` timed as executed; validation, acceptance/persistence and final serialization/diagnostics timed separately; every epoch persisted immediately |
| **R4** | Validation used float64 reconstructions that could hide executed overflow or underflow; B's `T_in` unchecked; acceptance checked only losses | Implementation | Executed-dtype products and generator entries checked first; formula and executed-generator eigenvalues assessed separately; B covered; per-epoch finiteness of parameters, optimizer state, normalization state and scalars; executed coefficients recorded |

### Static review of `e2c5b2f` — dispositions (protocol §14)

| | Finding | Disposition |
|---|---|---|
| **F1** | The domain validator hand-built the generator instead of using the executed one; `S` finiteness was not required; real modes were classified by `rho_max` finiteness | `executed_generator` uses the production `mass_block_generator` with the forward pass's clip, clock and coefficients; eigenvalues are taken from it; the formula diagnostic is kept separately; finite `S` is required; classification is by `omega == 0`; float32 fixture checks bitwise equality with the module's `coefficients()["A"]` |
| **F2** | Preflight ran acceptance on zeroed metrics and discarded the verdict | Actual measured metrics are accepted and persisted; timing must be finite and nonnegative; `decide_after_preflight` returns FAILED for invalid state and INCOMPLETE for retrace or over-budget; `execute_screen` enforces it before `run_one`; stub fixture proves the stop and a positive control |
| Cleanup | Stale `T = 5 exp(t)` docstring; "frozen-extra" wording | Corrected to `T = 10 exp(t)` and copied-gradient routing |

Scope recorded: the float32 FD constants are production smoke checks, and the
mixed gradient criterion is reported as "within the mixed tolerance", given its
float32 per-leaf absolute floor of about `1.19e-4 G_ref`.

The equation, three arms, source checkpoint, schedule, performance criteria and
1200 s cap are unchanged.

## Known risks, before execution

* **Budget.** Timescale-study measurements on the same GPU put a 10-epoch Rawat
  run at about 70 s and a block run at about 130 s. That projects roughly
  810 s of training, plus checks, restore, the gate and compilation, against
  1160 s usable. R2 added checks, including one small training-mode network,
  and R3 makes the projection more honest; both tighten the margin. It may not fit. Preflight will then refuse and report the
  measured obstruction, without trimming anything.
* **First execution.** This is the first execution of all new code, inside the
  cap. Static audits do not substitute for the cluster checks.

## Dispatch 1 — `c94ab1f`: FAILED at the focused checks, no training

| | |
|---|---|
| host | `pgi15-gpu3`, RTX 3090, SLURM 66044, jax 0.11.0, backend `gpu` |
| started | 2026-09-16T17:56:52Z |
| status | `STABLE_GP_STATUS=FAILED`, `STABLE_GP_EXIT=4`: focused checks did not pass |
| checks | **2 failed, 36 passed in 263.5 s**; 269 s of 1200 elapsed |
| not reached | source restore, identity gate, epoch 0, preflight, training |
| logs | `/Users/durso/s5-runs/stable-gp/logs/20260916-195652/` (preserved) |
| artifacts | `/Users/durso/s5-runs/stable-gp/20260916-195652/` |

**Operationally FAILED. There is no performance result.** No retry and no
protocol change was made.

### Failure 1 — `test_production_float32_probe_in_its_own_process`

The probe's own dtype assertion fired:

```
C leaf encoder/layers_{0,1}/seq/Lambda_{re,im} is float64, not float32
```

**Static cause.** The probe builds its small network from
`tests/response_reference.ssm_kwargs`, whose `Lambda_re_init` and
`Lambda_im_init` are NumPy float64 arrays. `S5SSM` stores them unchanged as
parameters, so the fixture's poles were float64 even with x64 off. The
production path builds them from `make_DPLR_HiPPO` through
`rawat_benchmark.ssm_kwargs`, and the saved Stage 2 checkpoint restored as
float32 in the earlier diagnostic. So this is a **fixture dtype defect, and
the assertion was right to reject it**.

**Consequence for the probe's printed numbers.** Sections [1] (projection:
1280 executed float32 modes, worst relative margin 3.73e-6) and [R1] (witness
returns the `rho = 1` fallback) used explicitly float32 arrays and are valid
float32 measurements. Sections [2], [3] and [R2] used this network, so their
numbers came from **mixed float64 poles and float32 other leaves**, and are
**not** production float32 certifications:

* identity logits 0 and 5.0e-7;
* r-only derivative relative error 4.7e-6 and 9.6e-6;
* `dL/dt` exactly 0 at the start.

### Failure 2 — `test_generalized_discrete_equation_recovers_TSS_eq17_exactly`

The traceback is not in the console tail and must be read from `checks.log`
before any diagnosis is recorded as fact.

**Static suspicion, unverified.** The test propagates the Eq. (17) and
generalized trajectories *separately* for 40 steps under an absolute
tolerance of 1e-12. Algebraically identical updates round differently, and a
second-order nonlinear recursion can amplify that. If so, the defect is that
the test compares accumulated trajectories instead of single steps from
identical states. It would not be a disagreement in the equation. This is a
hypothesis until the traceback confirms which step and what magnitude.

### Fixture corrections after dispatch 1 (test code only)

No model, equation, arm, tolerance, performance criterion or cap changed.
Nothing has been rerun.

1. **Probe dtypes.** `stable_gp_float32_probe.py` now builds every network,
   including the separate r-only fixture, from `production_kwargs`: pole
   initializers explicitly float32, `V`/`Vinv` complex64. The shared neutral
   helper is unchanged. The dtype assertion now covers the params and batch
   statistics of **all three arms and the r-only fixture**, not only C's
   params.
2. **Eq. (17) rollout test.** It shared one previous-drive buffer between the
   two independently evolving trajectories. Each now keeps its own
   `f_{k-1}`, and the generalized path its own `s_{k-1}`. The rollout check
   is **kept**, with TSS64 = 1e-12 unchanged, and now reports the worst
   discrepancy and its step.
3. **New one-step identity** `test_generalized_step_equals_eq17_on_identical_inputs`:
   200 random identical `(s_k, s_{k-1}, f_k, f_{k-1}, h, T)`, M = gamma = 0,
   same tolerance. It separates an algebraic disagreement from accumulated
   trajectory rounding.

**Diagnosis status: UNRESOLVED.** The cause of dispatch 1's Eq. (17) failure
is not established. The shared previous-drive defect may by itself explain the
threshold crossing, as may accumulated rounding, or both. No cause is recorded
until the original traceback from `checks.log` is read.

Nor will the next run settle it by itself. A failed one-step check alone does
not establish an algebraic error, and a failed rollout alone does not
establish "rounding only". Any remaining discrepancy will be reported with its
step and magnitude before a measurement amendment is proposed.

Both identity tests now assert finiteness of each step's outputs and of the
discrepancy before updating the worst value, so a NaN cannot be silently
skipped. Tolerances and fixtures are unchanged.

### Timing observed

The focused checks took 263.5 s, more than planned. That leaves about 890 s
for restore, the gate, compilation, preflight and nine runs, against roughly
810 s of estimated training. This is recorded as a budget risk only; preflight
would have measured it.

## Results

*(empty — no training has run)*

### Checks
### Source checkpoint and reproduction
### Identity gate and epoch 0
### Preflight
### Per-stream endpoints (epoch 10)
### Screen: C vs A, C vs B (paired values), B vs A (descriptive)
### T_in / rho / T movement, stability margins, passive occupancy, projection telemetry
### Current-input and history response changes
### Cost: parameters, carry, time

## Guarantees: checked versus analytical

See protocol §12. Numerical PASS is distinct from development success, and
neither by itself establishes superiority.
