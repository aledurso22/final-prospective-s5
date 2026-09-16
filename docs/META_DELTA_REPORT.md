# Generalized prospective memory around the delta boundary — report

**Status: dispatch 2 at `5dc4b07` — operational PASS (273 s of 600).
Verdicts: literature FAILED, matched delta PASSED, TSS Eq. (17) direct
fast weight (applicability-limited) FAILED, heavy-ball family comparison
PASSED. Per-seed values pending the read-only digest. Dispatch 1 below is
preserved.**

Previous status line (dispatch 1): **dispatch 1 at `2a86cc9` FAILED at the focused checks (1 of 33).
No calibration stage, preflight or training ran. Logs preserved. No retry. The
narrow probe correction was cleared; one relaunch is authorized.**

Check scope, stated precisely: the actual initialized tree is checked for
value/shared-gradient nesting AND tangents in float64; in the float32 probe it
is checked for tangents only, while float32 value/shared-gradient nesting uses
the nonzero-gate stress fixture. No local numerical run, no cluster launch.
Every result section will be filled only from cluster output.

Proof audit: `docs/META_DELTA_PROOF_AUDIT.md` (no algebraic discrepancy found;
two executed-precision points handled). Protocol: `docs/META_DELTA_PROTOCOL.md`
(proposed, for static review). Unresolved issues are listed in protocol §11.

The completed adaptive-memory and S5 results are unchanged and are not
re-interpreted here: the earlier restricted family (`rho < 1`) remains a
negative result for that family, and its trained rho values (~0.66) were not
pressing against the bound.

## Verdict structure, fixed before execution

The literature verdict (Momentum and Gated DeltaNet) and the matched-delta
verdict are reported together. Two further verdicts are reported separately:
the TSS Eq. (17) comparison, "applied directly to the fast weight;
applicability-limited", and the heavy-ball comparison, "separately trained
family; gamma and M matched at initialization only".

None substitutes for another, and none is causal attribution to `T Rdot`.

## Dispatch 1 — `2a86cc9`: FAILED at the focused checks, no training

| | |
|---|---|
| host | `pgi15-gpu3`, RTX 3090, SLURM 66044, jax 0.11.0, backend `gpu` |
| started | 2026-09-16T19:44:02Z |
| status | `META_DELTA_STATUS=FAILED`, exit 4, focused checks did not pass |
| checks | **1 failed, 32 passed in 140.0 s**; 143 s of 600 elapsed |
| logs | `/Users/durso/s5-runs/meta-delta/logs/20260916-214402/` (preserved) |

### The single failure

It is in `tests/meta_delta_float32_probe.py`, the nonzero-gate **stress
fixture** block:

```
FAILURES:
  - raw_r tangent not resolvable: jvp -3.365e-03
raw_r jvp -3.36539e-03 fd(h=0.01) -3.36170e-03 rel 1.10e-03
raw_r jvp -3.36539e-03 fd(h=0.003) -3.37760e-03 rel 3.62e-03
```

The derivative is **finite**, and its finite-difference agreement is **within
tolerance** at both declared steps: 1.1e-3 and 3.6e-3 against 2e-2. The
failure comes solely from the declared resolvability threshold
`100 eps32 max(|f|, 1)/h_min`, about 7.4e-3 here.

**Cause: an inconsistency in my implementation.** Review R1 of `535fb02` set
the rule that a *finite* derivative below the float32 resolvability threshold
is REPORTED as a limitation, not a failure. I applied that rule, through
`tangent_decision`, to the new actual-start and wider-interior blocks. I left
the pre-existing stress-fixture block with its original hard failure
(probe lines 93–94). The launch therefore enforced two different
resolvability policies in one probe.

This is not a numerical defect of the model, and no discrepancy in
equations, nesting, domain or trajectories was observed.

### Everything else passed; measured magnitudes

| Check | Measured | Tolerance |
|---|---|---|
| float32 nesting logits (stress fixture) | 1.77e-7 rel | 1e-4 |
| float32 shared gradients (stress fixture) | within mixed tolerance | — |
| `raw_tau` gradient at start (stress fixture), float32 | 1.2e-7 vs `|dL/draw_r|` 3.4e-3 | ≤ 1e-3 × |
| **actual start** `raw_r` tangent, float32 | jvp 1.514e-2 (resolvable); FD rel 2.8e-4 / 1.0e-3 | 2e-2 |
| actual start `raw_tau` gradient, float32 | −1.26e-7 | ~0 |
| **actual start**, float64: nesting, shared gradients and `raw_r` tangent | jvp 1.174e-2; `raw_tau` gradient 1.3e-16 | passed |
| **wider region**, float32 vs float64 dense, ρ = 1.2754 | 1.74e-7 rel | 2e-5 |
| **wider region at the projection margin**, ρ = 2.25 | 2.84e-7 rel | 2e-5 |
| **wider interior** `raw_r` tangent, float32 | jvp 1.481e-2; FD rel 1.4e-4 / 1.9e-3 | 2e-2 |
| projection | 400 float32 points plus 49 edge points (`x → 1⁺`) certified in float64 | — |
| float64 suite | all 32 other tests passed: nesting on three seeds, storage in the wider sector, delta non-expansion, residual-velocity identity, Eq. (17) including the first-idle increment, generator-overflow and zero-a0 regressions, preflight refusal, screens | — |

### Narrow fix — cleared and applied (commit below), before relaunch

* **Stress-fixture tangent.** It now goes through the same
  `tangent_decision` as every other block, with both perturbations, the
  unchanged tolerance and threshold, and strict rejection of non-finite
  values.
* **Helper order.** The helper and its dependencies are defined before first
  use.
* **Printing.** Measured errors are printed in every finite case, including a
  reported limitation.
* **Regression.** A lightweight regression confirms that a finite
  below-threshold derivative yields a limitation, while the existing
  NaN-JVP and non-finite-perturbed-loss regressions still fail.

### Original proposal text

Route the stress-fixture tangent through the same `tangent_decision`, so a
finite derivative below the threshold is reported as a limitation. That is
consistent with R1; no tolerance, threshold, fixture or equation changes. The
fix and any relaunch need authorization.

## Dispatch 2 — `5dc4b07`: operational PASS

| | |
|---|---|
| started | 2026-09-16T20:23:10Z, `pgi15-gpu3`, SLURM 66044 |
| checks | **33 passed** in 139.8 s |
| preflight | projected 120.0 s; no failures and no retrace |
| study | completed; held-out opened after all 18 final runs |
| total | **273 s of 600** |
| artifacts | `/Users/durso/s5-runs/meta-delta/20260916-222310/` |

**Checks.**
* The stress-fixture `raw_r` tangent is finite and below float32
  resolvability (jvp −3.365e-3, threshold 7.37e-3). It is reported as a
  limitation, with FD relative errors of 1.10e-3 and 3.62e-3.
* Actual start in float32: `raw_r` jvp 1.514e-2, FD relative errors 2.84e-4
  and 1.03e-3, `raw_tau` gradient −1.26e-7.
* Actual start in float64: `raw_r` jvp 1.174e-2, `raw_tau` gradient 1.3e-16.
* Wider region against the float64 dense reference: 1.74e-7 at `rho = 1.275`
  and 2.84e-7 at `rho = 2.250`, the projection margin.
* Wider-interior tangent: FD relative errors 1.40e-4 and 1.88e-3.
* Projection: 400 points plus 49 edge points certified.
* Float32 nesting: logits 1.77e-7.

**Verdicts, as printed by the run:**

| Verdict | Result |
|---|---|
| Literature: Momentum DeltaNet AND Gated DeltaNet | **FAILED** |
| Matched delta: departure from the exact delta boundary | **PASSED** |
| TSS Eq. (17) applied directly to the fast weight (applicability-limited) | **FAILED** |
| Heavy-ball family comparison (not causal attribution) | **PASSED** |

Operational PASS is not a performance claim. Per-seed primary, retention,
recall, categories, selection and learned coefficients are pending
`python -m experiments.meta_delta.summary <run_dir>`.

## Results

*(per-seed values pending the digest)*
