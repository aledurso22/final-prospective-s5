# Adaptive associative-memory comparison — report

**Status: executed. `ADAPTIVE_STATUS=PASS` (dispatch 2, `f096242`), all 14
development and 21 final runs complete, held-out opened. Both performance
screens FAILED. The prospective term is credited against the inertial control
only.** Per-seed held-out values and categories are pending the read-only digest
(`experiments/adaptive_memory/summary.py`); the numbers below are exactly those
printed by the run.

## Provenance

| | |
|---|---|
| Branch | `adaptive-prospective-memory` |
| Parent | `9b427273263e7667c43253589bf25333c975cb3d` (completed nested-memory study) |
| Worktree | `/private/tmp/wt/adaptive` |
| Repository | `https://github.com/aledurso22/final-prospective-s5` |
| Protocol | `docs/ADAPTIVE_MEMORY_PROTOCOL.md` (frozen before execution) |
| Implementation | `experiments/adaptive_memory/` |
| Checks | `tests/test_adaptive_memory.py` |
| Launcher | `bin/run_experiments/cluster_adaptive_memory.sh` |
| Reviewed commit | `c12e20b` — static review `IMPLEMENTATION_REVIEW_c12e20b.md`, fix-before-launch |
| Reviewed commit | `9ca0de6` — follow-up review `IMPLEMENTATION_REVIEW_9ca0de6.md`, F1/F2 |
| Executed commit | **`f09624215eab4096d27fe4795dc7686188e49221`** (dispatch 2) |

Coordinator sources, all dated 16 September 2026:
`ADAPTIVE_MEMORY_CODING_BRIEF.md`, `ADAPTIVE_MEMORY_PROTOCOL.md`,
`ANALYTICAL_AUDIT.md`, `ORDINARY_PROSPECTIVE_BASELINES_AMENDMENT.md`.

Literature arms are the source-pinned implementations of the completed study,
reused unchanged and asserted bit-identical by a check:
`https://github.com/HuuYuLong/MomentumDeltaNet` @
`c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`, `min_log_mu = −2` (the official
constructor default). TSS reference: *Teaching signal synchronization in deep
neural networks with prospective neurons*, arXiv 2511.14917v2, Eqs. (5)–(7).

### Amendment status

The ordinary-prospective amendment arrived **before any numerical execution
began**. Its two references are therefore included in this batch, committed
before execution, with the five original arms and all their settings
unchanged. The batch is seven arms: 14 development runs on seed 200, 21 final
runs on seeds 201/202/203, 35 runs and 7,000 optimizer updates in one
600-second cap.

## Preparation without local numerical verification

The brief requires all numerical work — including calibration and the unit
tests — to run on the cluster. Local verification was therefore limited to
static inspection: `ast.parse` syntax checks, a cross-module symbol audit and a
call-signature/arity audit. **No module was imported, no rollout evaluated and
no numerical value computed locally for this study.**

That is the instruction, and it is followed here, but it should be stated
plainly: it means the first execution of this code is on the cluster, inside
the cap, and a defect that static analysis cannot see will consume a dispatch.
Three defects *were* caught statically before commit and are listed below.

Separately, and for the record: during the previous nested-memory study I ran
model rollouts on local CPU while diagnosing dtype failures, which crossed the
no-local-numerical-runs rule. That disclosure stands; nothing of the kind was
done here.

### Defects found by static audit before commit

1. `model.rollout` dropped the required `const` argument when delegating the
   two literature arms to the completed study's implementation — both would
   have raised `TypeError` on first call.
2. The learning rate was a **static** jit argument, so each configuration-B
   slot at lr `0.01` would have triggered a compilation that preflight never
   measured — the exact class of timing-accounting defect raised as R3 in the
   nested review. The rate is now dynamic.
3. `study.coefficient_report` had an unreachable/incorrect branch for the
   literature arms' gate reporting.

### Coordinator static review of `c12e20b` — R1–R5

The coordinator reviewed `c12e20b` before launch and found five further
defects, all static. Full dispositions are in
`docs/ADAPTIVE_MEMORY_PROTOCOL.md` §10; in brief:

* **R1** — the calibration grid's upper end (`ν = exp(12)`) overflowed `cosh`
  in the host exponential, so `configurations()` could never reach the bracket
  loop. The host routine now uses `scipy.linalg.expm`, and the scan is
  incremental and stops at the first bracket.
* **R2** — the retention safeguard rejected a candidate only when *both*
  metrics regressed. It now requires **both** differences `≥ −1 point`. All
  three paired seeds are required. Attribution no longer depends on the
  literature screen.
* **R3** — the query-value test demanded invariance the pinned literature
  gates cannot provide. Fixed by specifying the input contract at the common
  boundary, **not** by changing the literature rule. This was never label
  leakage in a valid episode.
* **R4** — `expm2` overflowed float32 for finite answers such as
  `diag(−200,−1)`. The trace is now folded inside the hyperbolic functions,
  with a dtype-derived series switch. No coefficient clamp was added.
* **R5** — the "float32 trajectory" tests used float64 coefficients; the TSS
  JVP was float64-only; the confluent check asserted only finiteness; the
  derivative path used a surrogate scalar. All four are corrected, and the
  pre-existing float32 coverage was preserved rather than discarded.

Two of these — R1 and R4 — would have failed the batch at the checks with no
training, exactly as the nested study's first dispatch did. They were found by
reading, not by running.

### Follow-up review of `9ca0de6` — F1, F2

* **F1** — my own inactive-branch fixture was wrong: `[[0,1],[1e6,0]]` has
  eigenvalues `±1000`, whose exponential genuinely exceeds float64 range, and
  I had asserted a finite derivative from it unconditionally. No correct
  implementation could pass. Replaced with the review's stable shifted family
  `[[−c,1],[m,−c]]`, `c = 1+√max(m,0)`, held fixed under differentiation —
  same extreme discriminants, representable answers. This would have failed
  the batch at the checks.
* **F2** — the derived-coefficient assertions covered only the *initialized*
  slots, while the report recorded a constraint boolean it did not enforce.
  `validate_coefficients` now gates every trained checkpoint on its own arm's
  constraint, with TSS's intentional `γ = 0` preserved and never tested
  against the generalized sector. The delta arm's `η` is now read through the
  executed transform rather than host NumPy.

Full dispositions in `docs/ADAPTIVE_MEMORY_PROTOCOL.md` §11.

## Declared configuration

See `docs/ADAPTIVE_MEMORY_PROTOCOL.md` for the frozen contract: seven arms,
fourteen development slots, the calibration algorithm and target, the training
schedule, the named streams, the two screens, and the frozen tolerances and
finite-difference steps.

## Dispatch 1 — `580913d`: FAILED at the checks, no training

| | |
|---|---|
| host | `pgi15-gpu3`, RTX 3090, SLURM 66010, jax 0.11.0, backend `gpu` |
| commit | `580913df691464c236353889a877f25ba026599b` |
| status | `ADAPTIVE_STATUS=FAILED`, `ADAPTIVE_EXIT=4` — focused checks did not pass |
| result | **1 failed, 189 passed in 200.6 s**; 204 s of 600 elapsed. No study-stage calibration, no preflight, **no training**. (Calibration routines did run *inside* the focused checks, which call them.) |
| logs | `/Users/durso/s5-runs/adaptive-memory/logs/20260916-150404/` (preserved) |
| artifacts | `/Users/durso/s5-runs/adaptive-memory/20260916-150404/` |

The single failure:

```
test_expm2_does_not_overflow_in_float32[stiff_prospective-G7]
  expm2 f32 stiff_prospective      rel 1.049e-04
  assert 1.049e-04 < 2e-05
```

This is the guard working as intended. It is a genuine float32 accuracy defect
in the production exponential, not a bad tolerance, and it was caught before
any training ran.

### Cause

`stiff_prospective` is the prospective generator at `ν = e⁹ ≈ 8103`,
`τ = ρ = 3/4`, `w = 1`. Its eigenvalues are `−0.99996` and `−8103.42`. The
revised `expm2` folds the trace inside the hyperbolic functions and forms the
eigenvalues as `s ± a` with `s = −4052.209` and `a = 4051.209`. The near-zero
eigenvalue is therefore a **catastrophic cancellation**: two quantities of
magnitude ~4052 producing ~1, which discards `log₂(4052) ≈ 12` of float32's 24
bits and permits a relative error of order `2⁻¹² = 2.4e−4` in `e^{s+a}`.

On the `O(0.368)` entry that gives an order-of-magnitude estimate of
**~9e−5**; the cluster measured **1.049e−4**. This is a sensitivity estimate,
not a deterministic prediction: lost bits bound how large rounding error *can*
be, not how large it *must* be. The diagnosis is arithmetic, and the R4 fix
removed the *overflow* hazard without removing this *cancellation* hazard.

### Correction

The standard stable-quadratic-root remedy, applied to the real-eigenvalue
branch only: form the **large-magnitude** eigenvalue by adding same-sign terms
(`λ_far = s + sign(s)·a`, never a cancellation), then recover the near-zero one
from the exact product relation `λ₊λ₋ = det(hG)`, which avoids the cancellation
in `s + a`. (A generic determinant still subtracts two products; for this
fixture that subtraction is well conditioned.) `|λ_far| = |s| + a ≥ a > 0` in
that branch, so the division is safe.

**Follow-up correction (`IMPLEMENTATION_REVIEW_ea80581.md`).** As first
committed at `ea80581`, the determinant was **not** masked in the inactive
real branch. For the existing oscillatory shifted fixture
`[[−1, 1], [−10⁶, −1]]` that branch computed `λ_near = 10⁶+1` and an
overflowing exponential; selecting the oscillatory result afterwards does not
protect reverse mode, where an infinite derivative times a zero cotangent
gives NaN. `det(hG)` is now masked to zero where the real branch is inactive,
**before** `λ_near` is formed, giving finite dummy roots 1 and 0 and a zero
derivative. The active branch and the model are unchanged, and the existing
shifted oscillatory gradient checks cover this regression. This would have
failed the next dispatch at the checks.

Order-of-magnitude estimate after the fix for the failing case: ~1e−6, inside the
unchanged `TRAJ32 = 2e−5`. **No tolerance was relaxed, no case removed, no
coefficient clamped and no equation, arm, seed, schedule or cap changed.** The
formulation is identical in exact arithmetic.

### `grid_top_prospective` passing is not a contradiction

I previously called this case's pass inconsistent with the diagnosis. That was
a misreading of my own estimate. `grid_top_prospective` (`ν = e¹²`) has a worse
conditioned `s + a`, which **permits** larger rounding error but does not
**require** it; a more ill-conditioned subtraction can round closer to its true
result. For this family (`τ = ρ = 3/4`, unit source) the characteristic
polynomial is `λ² + (ν + 4/3)λ + ν = 0`, so the slow root is
`λ = −1 + 1/(3ν) + O(ν⁻²)`, approaching the exactly representable −1 as `ν`
grows — favourable rounding toward it is one analytical possibility. That is
not a verified account of the old GPU intermediates, and none is needed: the
old formula has been replaced, and no additional dispatch is warranted to
reconstruct its historical rounding.

Every measured numerical error is still appended, pass or fail, to
`measured_errors.tsv` in the log directory and printed by the launcher. That
records the **new** formula's margins; it cannot and is not meant to explain
the old formula's intermediates.

## Dispatch 2 — `f096242`: PASS

| | |
|---|---|
| host | `pgi15-gpu3`, RTX 3090, SLURM 66010, jax 0.11.0, backend `gpu` |
| started | 2026-09-16T13:38:03Z |
| status | `ADAPTIVE_STATUS=PASS`, `ADAPTIVE_EXIT=0`, **283 s of 600** |
| checks | **190 passed** in 164.5 s; every measured numerical error inside tolerance |
| preflight | incurred compilation 31.8 s, projected remaining 85.8 s, no retrace reported |
| study | 114 s wall; 14/14 development, 21/21 final, held-out opened |
| artifacts | `/Users/durso/s5-runs/adaptive-memory/20260916-153803/` |
| logs | `/Users/durso/s5-runs/adaptive-memory/logs/20260916-153803/` |

After the launcher had returned exit 0 and printed its status, the user's
interactive SLURM step (`66010.1`) was terminated (`srun ... Killed`). That
followed the run and is not a run failure; the artifacts were written before
exit.

### Numerical checks (recorded errors, largest per class)

| class | worst case | error | tolerance |
|---|---|---|---|
| `expm2` float32 | oscillatory_fast | 6.37e−8 | 2e−5 |
| `expm2` float32, previously failing | stiff_prospective | 1.85e−8 | 2e−5 |
| `expm2` float32 | grid_top_prospective | 5.04e−8 | 2e−5 |
| `expm2` float64 | grid_top_prospective | 4.90e−13 | 1e−9 |
| shifted family float64 | m = −1e6 | 4.61e−13 | 1e−9 |
| series switch float32 | at switch, m = +0.263 | 8.20e−8 | 2e−5 |
| series switch float64 | above switch, m = +3.46e−3 | 1.11e−15 | 1e−9 |
| 64-token float32 trajectory | TSS case 3 | 8.35e−7 | 2e−5 |

The failing case of dispatch 1 fell from 1.049e−4 to 1.85e−8 after the
cancellation fix. The launcher printed only the first 40 sorted records, which
include only some trajectory classes; the complete record is
`measured_errors.tsv`.

### Preflight and cost

| Arm | Compile (incurred) | Step | Eval | Projected arm | Params | Carry |
|---|---|---|---|---|---|---|
| Generalized prospective | 4.9 s (+1.1) | 6.35 ms | 14.1 ms | 6.6 s | 433 | 128 |
| Inertial control | 4.7 s (+1.0) | 6.17 ms | 13.2 ms | 6.5 s | 432 | 128 |
| First-order delta | 2.9 s (+0.7) | 4.85 ms | 13.7 ms | 5.1 s | 431 | 64 |
| TSS prospective | 4.6 s (+1.1) | 6.08 ms | 13.3 ms | 6.4 s | 432 | 128 |
| Ideal equilibrium | 2.4 s (+0.6) | 4.33 ms | 11.3 ms | 4.6 s | 392 | 64 |
| Gated DeltaNet | 3.0 s (+0.8) | 5.13 ms | 12.0 ms | 5.4 s | 480 | 64 |
| Momentum DeltaNet | 3.4 s (+0.8) | 5.94 ms | 13.1 ms | 6.2 s | 569 | 128 |

All counts match the declared values.

### Selection (development seed 200)

| Family | Selected |
|---|---|
| Generalized prospective | **A** (τ = 0.75) |
| Inertial control | **A** (τ = 0.75) |
| First-order delta | **B** (lr 0.01) |
| TSS prospective | **A** (ε/τ_m = 0.1) |
| Ideal equilibrium | **B** (lr 0.01) |
| Gated DeltaNet | **B** (lr 0.01) |
| Momentum DeltaNet | **A** (lr 0.003) |

Development values for the unselected slots were not in the console tail and
are pending the digest. Both new-rule families selected the short timescale;
the declared long-τ configuration did not win selection for either.

### Final-seed validation primary (update 200, evaluation-seed validation)

These are the per-run lines printed during training, **not** the held-out
split.

| Arm | 201 | 202 | 203 |
|---|---|---|---|
| Generalized prospective | 0.4663 | 0.4622 | 0.4814 |
| Inertial control | 0.4585 | 0.4502 | 0.4702 |
| First-order delta | 0.5398 | 0.5581 | 0.5620 |
| TSS prospective | 0.4968 | 0.5012 | 0.5166 |
| Ideal equilibrium | 0.4084 | 0.3972 | 0.4143 |
| Gated DeltaNet | 0.5430 | 0.5574 | 0.5681 |
| Momentum DeltaNet | 0.5215 | 0.5232 | 0.4739 |

### Held-out, mean over three seeds (512 sequences per family)

| Arm | Primary (revision macro) | Revision untouched retention | Recall-family accuracy |
|---|---|---|---|
| Generalized prospective | 0.4671 | 0.3068 | 0.4755 |
| Inertial control | 0.4547 | 0.2918 | 0.4653 |
| First-order delta | **0.5497** | 0.4917 | 0.6253 |
| TSS prospective | 0.5017 | 0.4822 | 0.6018 |
| Ideal equilibrium | 0.4069 | 0.2174 | 0.4147 |
| Gated DeltaNet | **0.5512** | 0.5029 | 0.6333 |
| Momentum DeltaNet | 0.5015 | **0.6135** | **0.7136** |

### Verdicts

**Literature screen — FAILED.**

| vs | Δ mean primary | all seeds positive | Δ retention | Δ recall | safeguard |
|---|---|---|---|---|---|
| Momentum DeltaNet | −0.0344 | no | −0.3067 | −0.2381 | failed |
| Gated DeltaNet | −0.0841 | no | −0.1961 | −0.1578 | failed |

**Ordinary-prospectivity extension screen — FAILED.**

| vs | Δ mean primary | all seeds positive | Δ retention | Δ recall | safeguard |
|---|---|---|---|---|---|
| TSS prospective | −0.0347 | no | −0.1754 | −0.1263 | failed |
| Ideal equilibrium | +0.0602 | **yes** | +0.0894 | +0.0608 | met |

The candidate passes every condition against the ideal minimum-change
reference, but the screen needs both references, and it loses to TSS on all
three measures.

**Attribution (independent of both screens) — credited against the inertial
control.** Mean differences: primary +0.0124, retention +0.0150, recall
+0.0102, with positive paired primary differences in all three final seeds. The
per-seed held-out values are pending the digest. This is a comparison of
separately calibrated and trained rule families, not a term-removal ablation
of one trajectory. It does not imply competitive performance.

**First-order delta comparison — the simpler account wins.** The equally
source-gated adaptive first-order delta arm scores **0.5497** primary against
the generalized candidate's 0.4671, with better retention and recall. It is
essentially level with Gated DeltaNet (0.5512) on primary. The protocol said
that if the first-order delta arm matched the candidate's outcome, a simpler
write-control account would remain viable; here it *exceeds* the candidate on
every reported measure. In this pilot the second-order structure, prospective
or inertial, costs accuracy relative to the gated first-order rule.

### What the results support

* The implementation passed all 190 focused checks, including the dense
  references, the TSS original-equation integration, both derivative routes and
  every recorded numerical error, and completed the declared batch inside the
  cap.
* **No improvement over Momentum or Gated DeltaNet.** The candidate is below
  both on primary, with retention 31 and 20 points lower respectively.
* **No improvement over ordinary prospectivity as a whole.** The candidate
  beats the ideal minimum-change reference on all declared conditions but loses
  to TSS finite adaptation, which also has substantially better retention and
  recall.
* **A small, seed-consistent advantage over the inertial control** (+1.2
  points primary), in a regime where both second-order rules trail the gated
  first-order rule by about 8–9 points.
* Momentum DeltaNet has the best retention and recall by a wide margin, while
  Gated DeltaNet and the adaptive first-order delta arm lead on primary. No
  single arm dominates every measure.
* This is one bounded development screen with three seeds, one task and 200
  updates. The failed verdicts are reported as they stand; no criterion,
  selection or configuration is changed in response.

## Results pending the digest

Per-seed held-out primary/retention/recall and CE, the paired per-seed
differences behind each verdict, all four categories for both families,
development values for every slot, learned coefficients and gate
distributions, training gains and norms. None requires another run; all are in
the saved JSON. Obtain them with the read-only digest:

```
python -m experiments.adaptive_memory.summary \
  /Users/durso/s5-runs/adaptive-memory/20260916-153803 \
  /Users/durso/s5-runs/adaptive-memory/logs/20260916-153803
```

## Limitations, declared in advance

* A bounded development screen on one small synthetic task, three seeds, 200
  updates. Not statistical significance, not a benchmark, not SOTA.
* Equal selection budget across families, not equal or exhaustive
  hyperparameter tuning — the axes searched differ by family.
* The evaluation streams are fresh, but the task distribution has already
  guided this proposal.
* Arms 1, 2 and 4 are independently calibrated and trained; their differences
  compare optimized rule families and are not term-removal ablations of a
  single trained trajectory. The completed study's fixed same-coefficient
  ablation remains the separate evidence for that.
* A matched single-write response does **not** imply matched initialization
  performance; the latter is measured and reported separately.
* The TSS reference is an application of a published forward law to this shell,
  not a reproduction of its teaching-signal task or local learning rule, and
  its unique-equilibrium tracking guarantee is not invoked here. The ideal
  reference's zero-residual branch, instantaneous jump update and
  minimum-change completion are our explicit additional specifications.
* Neither ordinary reference is assumed memoryless or assumed to score worse.
* The incremental storage result is an analytical property of the prospective
  law at fixed coefficients and identical inputs. It is not a retrieval
  guarantee, does not cover learned coefficients changing within an episode,
  and is not applied to the inertial or TSS arms.
* The fixed-objective overlap with inertial optimization under Hessian-driven
  damping (Alvarez–Attouch–Bolte–Redont 2002; Attouch–Chbani–Fadili–Riahi
  2019) is acknowledged.
