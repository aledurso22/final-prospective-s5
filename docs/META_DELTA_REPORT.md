# Generalized prospective memory around the delta boundary — report

**Status: dispatch 2 at `5dc4b07` completed — operational PASS (273 s of
600). Matched-delta screen PASSED; heavy-ball family comparison PASSED;
literature screen FAILED; TSS Eq. (17) direct-fast-weight comparison
(applicability-limited) FAILED. The candidate had the highest mean primary
accuracy of all six arms. Every failed screen failed on its retention or
recall safeguard, not on primary accuracy. The learned candidate settled at
rho ~ 0.15, on the PASSIVE side M < gamma T, not in the wider sector.
Dispatch 1 below is preserved.**

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

## Results — dispatch 2 (`5dc4b07`)

All values below were read from the saved
`/Users/durso/s5-runs/meta-delta/20260916-222310/status.json` with the
read-only digest `experiments.meta_delta.summary`. No training or numerical
rerun was done.

**Source of the digest output.** The digest that produced this output is the
version at `b3c8d50`. The later `1258e8b` version additionally prints every
arm's validation gain from update 0 to 200 and each screen condition as a
separate boolean. Every condition below is derived from the printed exact
differences. Gains from update 0 to 200 are therefore available for the
candidate only (§6).

### 1. Held-out results (512 sequences per family; seeds 301 / 302 / 303)

| Arm | Seed | Primary | Revision untouched retention | Recall | Revision CE |
|---|---|---|---|---|---|
| **Generalized, two-sided** | 301 | 0.5834 | 0.4729 | 0.6290 | 1.2282 |
| | 302 | 0.5981 | 0.4978 | 0.6422 | 1.2091 |
| | 303 | 0.5831 | 0.4700 | 0.6388 | 1.2343 |
| | **mean** | **0.5882** | **0.4802** | **0.6367** | **1.2239** |
| First-order delta | 301 | 0.5468 | 0.4822 | 0.6176 | 1.4272 |
| | 302 | 0.5548 | 0.5024 | 0.6285 | 1.4081 |
| | 303 | 0.5457 | 0.4763 | 0.6223 | 1.4361 |
| | **mean** | **0.5491** | **0.4870** | **0.6228** | **1.4238** |
| Heavy ball, same mass | 301 | 0.5439 | 0.4846 | 0.6237 | 1.4238 |
| | 302 | 0.5518 | 0.5039 | 0.6350 | 1.4071 |
| | 303 | 0.5459 | 0.4812 | 0.6251 | 1.4348 |
| | **mean** | **0.5472** | **0.4899** | **0.6279** | **1.4219** |
| TSS Eq. (17), direct fast weight | 301 | 0.5182 | 0.6467 | 0.7428 | 1.3670 |
| | 302 | 0.5248 | 0.6606 | 0.7572 | 1.3482 |
| | 303 | 0.5201 | 0.6514 | 0.7584 | 1.3751 |
| | **mean** | **0.5210** | **0.6529** | **0.7528** | **1.3635** |
| Gated DeltaNet | 301 | 0.5468 | 0.4893 | 0.6272 | 1.4143 |
| | 302 | 0.5565 | 0.5144 | 0.6338 | 1.3996 |
| | 303 | 0.5482 | 0.4836 | 0.6273 | 1.4280 |
| | **mean** | **0.5505** | **0.4958** | **0.6294** | **1.4140** |
| Momentum DeltaNet | 301 | 0.4910 | 0.7500 | 0.8038 | 1.5199 |
| | 302 | 0.5569 | 0.6558 | 0.7616 | 1.3007 |
| | 303 | 0.5422 | 0.6448 | 0.7562 | 1.3557 |
| | **mean** | **0.5300** | **0.6835** | **0.7739** | **1.3921** |

### 2. Screens: exact differences and each condition

Differences are candidate minus comparator. A screen passes only if all three
conditions hold, over all three pairs:
* **mean primary difference ≥ +0.01** (one point);
* **every paired primary difference > 0**;
* **retention difference ≥ −0.01 AND recall difference ≥ −0.01**.

| Comparison | Mean primary | Paired primary 301 / 302 / 303 | Retention | Recall | ≥ +1 pt | All > 0 | Retention ≥ −1 pt | Recall ≥ −1 pt | **Verdict** |
|---|---|---|---|---|---|---|---|---|---|
| **Matched delta** | **+0.0391** | +0.0366 / +0.0433 / +0.0375 | −0.0068 | +0.0139 | yes | yes | yes | yes | **PASSED** |
| **Heavy-ball family** | **+0.0410** | +0.0394 / +0.0464 / +0.0372 | −0.0097 | +0.0087 | yes | yes | yes (by 0.0003) | yes | **PASSED** |
| **Gated DeltaNet** (literature) | **+0.0377** | +0.0366 / +0.0416 / +0.0349 | **−0.0155** | +0.0072 | yes | yes | **no** | yes | **FAILED** |
| **Momentum DeltaNet** (literature, primary) | **+0.0582** | +0.0924 / +0.0413 / +0.0409 | **−0.2033** | **−0.1372** | yes | yes | **no** | **no** | **FAILED** |
| **TSS Eq. (17)**, direct fast weight | **+0.0672** | +0.0652 / +0.0734 / +0.0630 | **−0.1727** | **−0.1161** | yes | yes | **no** | **no** | **FAILED** |

**Literature screen: FAILED.**
* Against **Gated DeltaNet**, the candidate met both primary conditions,
  +3.8 points mean and positive in every seed. Recall held at +0.7. It failed
  only because retention was **−1.55 points**, beyond the −1-point
  safeguard.
* Against **Momentum DeltaNet**, it met both primary conditions, +5.8 points
  mean and positive in every seed. It failed on both safeguards: retention
  **−20.3** and recall **−13.7 points**.

**A failed screen here does not mean lower mean accuracy.** The candidate's
mean primary exceeded every comparator. What it gives up is retention of
untouched associations, and against Momentum and TSS also overall recall.

**TSS Eq. (17) applied directly to the fast weight: FAILED** on the same two
safeguards, despite +6.7 points primary. This remains **applicability-limited**:
with `f = W − ηR`, `(I − Df)` is singular off the current key and zero when
idle, so TSS Eq. (15) does not apply. This is a well-defined discrete
comparison, not a reproduction of TSS's experiments and not a verdict on
ordinary prospectivity.

**Matched delta: PASSED.** Leaving the exact delta boundary improved primary
accuracy by +3.9 points, positive in all three seeds, within both safeguards.
This is the pre-registered question of whether departing from delta adds
value, answered positively for this pilot. It is not attribution to the
prospective term.

**Heavy-ball family: PASSED**, by +4.1 points primary, positive in all seeds,
with retention only 0.0003 inside its safeguard. This is a comparison of
**separately trained families**, with `γ` and `M` matched at initialization
only. It is **not** causal attribution to `T Ṙ`. The residual-velocity
identity remains a separate analytical statement.

### 3. Query categories (held-out accuracy, mean over seeds)

| Arm | Recall: immediate | middle untouched | late selected | late untouched | Revision: immediate | middle untouched | late selected | late untouched |
|---|---|---|---|---|---|---|---|---|
| Generalized, two-sided | **1.0000** | 0.5907 | 0.6436 | 0.3125 | **1.0000** | 0.6436 | 0.3924 | 0.3169 |
| First-order delta | 0.9924 | 0.6237 | 0.5802 | 0.2949 | 0.8516 | 0.6738 | 0.3708 | 0.3001 |
| Heavy ball, same mass | 0.9922 | 0.6273 | 0.5902 | 0.3021 | 0.8345 | 0.6756 | 0.3745 | 0.3042 |
| TSS Eq. (17), direct | 0.9782 | 0.7607 | 0.8185 | 0.4538 | 0.3221 | 0.8114 | 0.4562 | 0.4945 |
| Gated DeltaNet | 0.9915 | 0.6300 | 0.5938 | 0.3024 | 0.8346 | 0.6839 | 0.3758 | 0.3076 |
| Momentum DeltaNet | 0.9629 | 0.7720 | **0.8634** | **0.4972** | 0.3447 | **0.8288** | 0.4084 | **0.5382** |

These are descriptive observations, not tested hypotheses:
* The candidate reaches **1.000 immediate revision**, against 0.83–0.85 for
  delta, heavy ball and Gated, and 0.32–0.34 for TSS and Momentum. It also
  leads the delta-like arms on late selected recall (0.644 vs 0.580–0.594)
  and late selected revision (0.392 vs 0.371–0.376).
* It trails the delta-like arms slightly on middle untouched associations.
* Momentum and TSS retain untouched associations far better, 0.81–0.83
  middle-revision, and give up immediate revision.
* The primary gains therefore come mainly from immediate and late selected
  queries. The safeguard failures come from untouched retention.

### 4. Development and selection (seed 300, update-200 validation)

| Arm | A (lr 0.003): primary / revCE / retention / recall | B (lr 0.01): primary / revCE / retention / recall | Selected |
|---|---|---|---|
| Generalized, two-sided | 0.4688 / 1.5336 / 0.3013 / 0.4956 | **0.5781** / 1.2440 / 0.4678 / 0.6379 | B |
| First-order delta | 0.4487 / 1.6670 / 0.2842 / 0.4680 | **0.5479** / 1.4327 / 0.4824 / 0.6216 | B |
| Heavy ball, same mass | 0.4551 / 1.6959 / 0.2964 / 0.4739 | **0.5449** / 1.4276 / 0.4878 / 0.6304 | B |
| TSS Eq. (17), direct | 0.4565 / 1.6808 / 0.4458 / 0.5610 | **0.5186** / 1.3685 / 0.6470 / 0.7480 | B |
| Gated DeltaNet | 0.4478 / 1.6623 / 0.2842 / 0.4707 | **0.5464** / 1.4240 / 0.4907 / 0.6316 | B |
| Momentum DeltaNet | 0.5195 / 1.3635 / 0.5791 / 0.7063 | **0.5427** / 1.3311 / 0.6523 / 0.7468 | B |

All six families selected the lr = 0.01 slot, on the same learning-rate axis
for every family.

### 5. Learned coefficients of the candidate, and domain

| Seed | η | τ | ρ | γ = 1/η | M | T | Side of `M = γT` | Certificate `dL/γ²` | Projection events | Minimum log margin to bound |
|---|---|---|---|---|---|---|---|---|---|---|
| 301 | 0.6079 | 1.2113 | **0.1495** | 1.6449 | 1.9924 | 8.1022 | **passive, `M < γT`** | −8.38 | 0 | 0.652 |
| 302 | 0.6315 | 1.2315 | **0.1501** | 1.5836 | 1.9501 | 8.2033 | **passive, `M < γT`** | −8.81 | 0 | 0.656 |
| 303 | 0.6377 | 1.2327 | **0.1498** | 1.5682 | 1.9331 | 8.2286 | **passive, `M < γT`** | −8.92 | 0 | 0.656 |

Initialization: `η₀ = 0.93145`, `τ₀ = 1`, `ρ₀ = 1`, so `T₀ = 1`. The initial
bound was `ρ_max = 2.1589`. Candidate source-weight gate [min / median / max]:
0.740 / 1.025 / 1.281, 0.745 / 0.996 / 1.308, and 0.722 / 0.967 / 1.273.

**The candidate moved from `ρ = 1` to `ρ ≈ 0.15` in every seed,** into the
previously certified **passive sector**. `T` rose from 1 to about 8.2. It
**never entered the wider `M > γT` sector.**
* `d = M − γT < 0`, so the wider-sector certificate is inactive.
* There were zero projection events and a log margin of about 0.65.

**The matched-delta success therefore did not use the wider-sector
extension.** This run provides no evidence either way about the value of
`ρ > 1`; it only shows that the extension was available and not selected by
training.

Its trained-coefficient seed variation is small (ρ 0.1495–0.1501). This is not
statistical evidence. Note that the completed adaptive-memory study's
sigmoid-restricted candidate, which could not reach `ρ = 1`, was trained only
at lr 0.003 and did not win. Different initialization and parameterization
make that comparison informal.

**Other arms' learned coefficients.** The digest labels these `raw_*` but
prints `exp(raw)`:
* First-order delta: `η` 0.2773 / 0.2702 / 0.2794.
* Heavy ball: `η` 0.2490 / 0.2435 / 0.2540, `τ` 0.2794 / 0.2782 / 0.2795.
* TSS Eq. (17): `η` 0.1314 / 0.1303 / 0.1326, `T` 19.80 / 19.93 / 19.67,
  starting from `η₀ = 0.5050`, `T₀ = 10`.
* Delta-like arms' gate medians: 0.97–1.00.

**Literature gates.**
* Gated DeltaNet's `α` stayed near 1 (median 0.9997–0.99998).
* Momentum DeltaNet again showed **inconsistent regimes across seeds**. Its
  `α` median was 0.988 in seed 301, which had the weakest primary (0.491) and
  strongest retention (0.750). It was 0.055 in seed 302 and 1.0e-4 in seed
  303.

### 6. Initialization-to-final validation gain

* **Candidate** (update 0 → 200, evaluation validation): primary
  **+0.1501 / +0.1404 / +0.1404**; revision CE **−0.6671 / −0.6792 / −0.6573**.
* **Other arms:** not in the executed digest version's output (see the source
  note above). They are stored in `status.json`, and `1258e8b`'s digest prints
  them; they are not reported here rather than estimated.

### 7. Cost

| Arm | Parameters | Carry (real) | Preflight step | Preflight eval (256 per family) |
|---|---|---|---|---|
| Generalized, two-sided | 433 | 128 | 13.89 ms | 14.1 ms |
| First-order delta | 431 | 64 | 11.93 ms | 13.2 ms |
| Heavy ball, same mass | 432 | 128 | 14.15 ms | 14.6 ms |
| TSS Eq. (17), direct | 432 | 128 | 12.70 ms | 15.0 ms |
| Gated DeltaNet | 480 | 64 | 12.18 ms | 13.3 ms |
| Momentum DeltaNet | 569 | 128 | 13.39 ms | 13.9 ms |

Each candidate final run took about 2.7–2.8 s. The study phase took 129 s; the
whole dispatch took **273 s of 600**.

### 8. What this run supports

* **Operationally:** 33/33 checks passed, preflight fit, the batch completed,
  and every candidate checkpoint was certified.
* **Matched delta (PASSED):** departing from the exact delta boundary improved
  held-out primary accuracy by 3.9 points in all three seeds, within the
  retention and recall safeguards. The departure went into the **passive**
  sector, not the wider one.
* **Heavy-ball family (PASSED):** +4.1 points, as a separately trained family
  comparison, not a causal attribution to `T Ṙ`.
* **Literature (FAILED):** higher mean primary accuracy than Gated (+3.8) and
  Momentum (+5.8) in every seed, but larger losses in untouched retention:
  −1.55 against Gated, −20.3 against Momentum. Against Momentum it also lost
  −13.7 in recall. This is not a win over the literature rules under the
  declared criterion.
* **TSS Eq. (17) direct fast weight (FAILED, applicability-limited):** +6.7
  primary, with −17.3 retention and −11.6 recall.
* **Scope:** one small task, a 200-update development pilot, three seeds.
  Not significance, not a published benchmark, not SOTA. No criterion,
  selection or configuration is changed in response, and no follow-up run is
  launched.
