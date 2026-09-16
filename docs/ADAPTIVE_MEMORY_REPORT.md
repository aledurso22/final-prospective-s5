# Adaptive associative-memory comparison — report

**Status: executed. `ADAPTIVE_STATUS=PASS` (dispatch 2, `f096242`), all 14
development and 21 final runs complete, held-out opened. Both performance
screens FAILED. The prospective term is credited against the inertial control
only.** Every number below was printed by the run or by the read-only digest
`experiments/adaptive_memory/summary.py`, which reads the saved JSON without
recomputation.

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

| Family | A primary | A rev-CE | B primary | B rev-CE | Selected |
|---|---|---|---|---|---|
| Generalized prospective | **0.4795** | 1.5933 | 0.4531 | 1.6233 | A (τ .75) |
| Inertial control | **0.4685** | 1.6687 | 0.2019 | 2.0873 | A (τ .75) |
| First-order delta | 0.4624 | 1.6511 | **0.5466** | 1.4229 | B (lr .01) |
| TSS prospective | **0.4968** | 1.6101 | 0.4136 | 1.7950 | A (ε/τ_m .1) |
| Ideal equilibrium | 0.4036 | 1.7340 | **0.4138** | 1.7108 | B (lr .01) |
| Gated DeltaNet | 0.4353 | 1.6586 | **0.5481** | 1.4147 | B (lr .01) |
| Momentum DeltaNet | **0.4280** | 1.6170 | 0.3506 | 1.5937 | A (lr .003) |

Both new second-order families selected the short timescale; at τ = 32 the
inertial control reached only 0.2019.

### Calibration

`β* = 0.606018662389`; slot 1A recovered `ν = 4/3` to 1.23e−10 (tolerance
1e−8). Every solve found its first bracket and met the 1e−10 observable
tolerance (errors 9.2e−12 to 3.4e−11).

| Slot | Solved rate | Derived |
|---|---|---|
| prospective A | ν = 1.333333333 | τ .75, ρ .75 |
| prospective B | ν = 0.9484261736 | τ 32, ρ .75 |
| inertial A | η = 0.8055350393 | τ .75 |
| inertial B | η = 14.08597438 | τ 32 |
| TSS A | q = 0.07330897071 | τ_m 13.641, ε 1.364, M 18.61, T 15.00 |
| TSS B | q = 0.2295413235 | τ_m 4.357, ε 2.178, M 9.49, T 6.53 |
| delta A/B | η = 0.931452 (closed form) | |

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

### Held-out per seed (512 sequences per family)

| Arm | Seed | Primary | Revision untouched retention | Recall | Rev CE |
|---|---|---|---|---|---|
| Generalized prospective | 201 | 0.4688 | 0.3052 | 0.4783 | 1.5946 |
| | 202 | 0.4586 | 0.2971 | 0.4692 | 1.6183 |
| | 203 | 0.4739 | 0.3181 | 0.4791 | 1.5991 |
| | **mean** | **0.4671** | **0.3068** | **0.4755** | 1.6040 |
| Inertial control | 201 | 0.4585 | 0.2952 | 0.4703 | 1.6683 |
| | 202 | 0.4468 | 0.2817 | 0.4601 | 1.6924 |
| | 203 | 0.4587 | 0.2986 | 0.4653 | 1.6717 |
| | **mean** | **0.4547** | **0.2918** | **0.4653** | 1.6775 |
| First-order delta | 201 | 0.5432 | 0.4836 | 0.6216 | 1.4202 |
| | 202 | 0.5476 | 0.4851 | 0.6188 | 1.4410 |
| | 203 | 0.5583 | 0.5063 | 0.6356 | 1.4158 |
| | **mean** | **0.5497** | **0.4917** | **0.6253** | 1.4256 |
| TSS prospective | 201 | 0.5009 | 0.4795 | 0.6035 | 1.6018 |
| | 202 | 0.4907 | 0.4673 | 0.5900 | 1.6325 |
| | 203 | 0.5137 | 0.4998 | 0.6118 | 1.6090 |
| | **mean** | **0.5017** | **0.4822** | **0.6018** | 1.6145 |
| Ideal equilibrium | 201 | 0.4108 | 0.2217 | 0.4130 | 1.7289 |
| | 202 | 0.3964 | 0.2024 | 0.4115 | 1.7426 |
| | 203 | 0.4136 | 0.2280 | 0.4196 | 1.7105 |
| | **mean** | **0.4069** | **0.2174** | **0.4147** | 1.7273 |
| Gated DeltaNet | 201 | 0.5475 | 0.4980 | 0.6300 | 1.4108 |
| | 202 | 0.5461 | 0.4937 | 0.6265 | 1.4272 |
| | 203 | 0.5599 | 0.5171 | 0.6433 | 1.4060 |
| | **mean** | **0.5512** | **0.5029** | **0.6333** | 1.4147 |
| Momentum DeltaNet | 201 | 0.5189 | 0.5681 | 0.6848 | 1.3796 |
| | 202 | 0.5217 | 0.6064 | 0.7241 | 1.3822 |
| | 203 | 0.4639 | 0.6660 | 0.7318 | 1.6475 |
| | **mean** | **0.5015** | **0.6135** | **0.7136** | 1.4698 |

### Held-out categories (accuracy, mean over seeds)

| Arm | Recall: immediate | middle untouched | late selected | late untouched | Revision: immediate | middle untouched | late selected | late untouched |
|---|---|---|---|---|---|---|---|---|
| Generalized prospective | 1.0000 | 0.4285 | 0.3006 | 0.1730 | 1.0000 | 0.4378 | 0.2547 | 0.1758 |
| Inertial control | 1.0000 | 0.4167 | 0.2752 | 0.1691 | 0.9995 | 0.4181 | 0.2355 | 0.1655 |
| First-order delta | 0.9924 | 0.6271 | 0.5898 | 0.2920 | 0.8426 | 0.6756 | 0.3729 | 0.3078 |
| TSS prospective | 0.9792 | 0.6217 | 0.5251 | 0.2811 | 0.6582 | 0.6802 | 0.3844 | 0.2842 |
| Ideal equilibrium | 1.0000 | 0.2948 | 0.2129 | 0.1510 | 1.0000 | 0.2868 | 0.1929 | 0.1479 |
| Gated DeltaNet | 0.9919 | 0.6346 | 0.6035 | 0.3031 | 0.8242 | 0.6859 | 0.3747 | 0.3200 |
| Momentum DeltaNet | 0.9518 | 0.7134 | 0.7912 | 0.3979 | 0.3688 | 0.7886 | 0.4102 | 0.4385 |

Chance is 0.125.

### Verdicts, with paired seeds

**Literature screen — FAILED.**

| vs | Δ mean primary | paired Δ 201 / 202 / 203 | Δ retention | Δ recall |
|---|---|---|---|---|
| Momentum DeltaNet | −0.0344 | −0.0502 / −0.0631 / **+0.0100** | −0.3067 | −0.2380 |
| Gated DeltaNet | −0.0841 | −0.0787 / −0.0875 / −0.0861 | −0.1961 | −0.1577 |

The single positive seed against Momentum is seed 203, where Momentum's own
primary fell to 0.4639 (see below). All other conditions fail.

**Ordinary-prospectivity extension screen — FAILED.**

| vs | Δ mean primary | paired Δ 201 / 202 / 203 | Δ retention | Δ recall | condition |
|---|---|---|---|---|---|
| TSS prospective | −0.0347 | −0.0321 / −0.0321 / −0.0398 | −0.1754 | −0.1262 | failed |
| Ideal equilibrium | +0.0602 | +0.0580 / +0.0623 / +0.0603 | +0.0894 | +0.0609 | **passed** |

**Attribution (independent of both screens).**

| vs | paired Δ primary 201 / 202 / 203 | per-seed Δ retention | per-seed Δ recall | mean Δ ret / rec | verdict |
|---|---|---|---|---|---|
| Inertial control | +0.0103 / +0.0118 / +0.0151 | +0.0100 / +0.0154 / +0.0195 | +0.0080 / +0.0091 / +0.0138 | +0.0150 / +0.0103 | **credited** |
| First-order delta | −0.0745 / −0.0890 / −0.0845 | −0.1784 / −0.1880 / −0.1882 | −0.1433 / −0.1496 / −0.1565 | −0.1849 / −0.1498 | not exceeded |

The per-seed retention and recall differences here are arithmetic on the
per-seed table above.

The inertial comparison compares separately calibrated and trained rule
families; it is not a term-removal ablation, and it does not make the
candidate competitive. The equally gated first-order delta arm exceeds the
candidate by 7.5–8.9 points in every seed, with better retention and recall.
By the protocol's own rule, the simpler write-control account remains viable
and, in this pilot, is the better one.

### Learned coefficients (final seeds)

| Arm | 201 | 202 | 203 | Initial |
|---|---|---|---|---|
| Generalized prospective | ν 1.469, τ 1.643, ρ .659, M 1.697, γ 1.033, T 2.493 | ν 1.493, τ 1.643, ρ .664 | ν 1.509, τ 1.635, ρ .675 | ν 1.333, τ .75, ρ .75 |
| Inertial control | η .666, τ .412 | η .693, τ .410 | η .719, τ .408 | η .806, τ .75 |
| First-order delta | η .2748 | η .2781 | η .2741 | η .9315 |
| TSS prospective | τ_m 21.75, ε 3.44 (ratio .158) | τ_m 21.70, ε 3.42 | τ_m 21.79, ε 3.48 | τ_m 13.64, ε 1.36 |

Every checkpoint passed `validate_coefficients`. The generalized candidate
stayed admissible (`γT − M` = 0.878 / 0.838 / 0.771) and TSS kept `γ = 0`. The
learned source weight stayed near one for the second-order arms (range about
0.89–1.18) and spread more for first-order delta (0.76–1.31).

Gated DeltaNet's α gate saturated near 1 (median ≈ 0.9997) in every seed.
Momentum DeltaNet's α median was 0.008 and 0.011 in seeds 201/202 but 0.976
in seed 203, where primary fell to 0.4639 while retention rose to 0.666.
Momentum therefore did not converge to one consistent gate regime across
seeds.

### Training gain (validation primary, update 0 → 200)

| Arm | 201 | 202 | 203 |
|---|---|---|---|
| Generalized prospective | +0.022 | +0.021 | +0.015 |
| Inertial control | +0.041 | +0.031 | +0.028 |
| First-order delta | +0.102 | +0.120 | +0.101 |
| TSS prospective | +0.062 | +0.061 | +0.057 |
| Ideal equilibrium | +0.025 | +0.013 | +0.012 |
| Gated DeltaNet | +0.174 | +0.205 | +0.137 |
| Momentum DeltaNet | +0.179 | +0.061 | +0.073 |

Each final run took 2.0–2.3 s of wall time.

### What the results support

* The implementation passed all 190 focused checks and recorded 45 numerical
  errors, none outside tolerance. It completed the declared batch in 283 s of
  600.
* **No improvement over Momentum or Gated DeltaNet.** The candidate trails both
  on mean primary, trails Gated in every seed, and is 20–31 points lower on
  retention.
* **No improvement over ordinary prospectivity as a whole.** It clears every
  condition against the ideal minimum-change reference but trails TSS finite
  adaptation in every seed and on retention and recall.
* **A small, seed-consistent advantage over the inertial control** (+1.0 to +1.5
  points primary, with retention and recall also higher in every seed).
* **The gated first-order delta rule beats both second-order rules in every
  seed.** The adaptive second-order structure did not help in this pilot.

### Descriptive observations — not tested hypotheses

These patterns were seen after the results. They were not predeclared and are
not claims:

* **Immediate revision versus retention.** The arms at ≈1.0 immediate revision
  accuracy (generalized prospective, inertial, ideal equilibrium) have the
  lowest untouched and late-category retention. The arms that give up
  immediate revision (first-order delta 0.84, Gated 0.82, TSS 0.66, Momentum
  0.37) retain far better. Across seven arms and one task this is a
  correlation, not a mechanism.
* **Softer writes.** The first-order delta arm reduced its write rate from
  η = 0.93 to about 0.275, a first-write strength of about 0.24 instead of the
  calibrated 0.61. The two second-order arms that kept ≈1.0 immediate revision
  did not move their effective response nearly as far.
* **Learning-rate budget asymmetry.** The declared selection axes differ by
  family: the generalized, inertial and TSS families searched timescales at
  lr 0.003, while first-order delta, ideal equilibrium, Gated and Momentum
  searched lr 0.003 against 0.01. Three of those four chose lr 0.01. The
  candidate learned least (+1.5 to +2.2 points) and was never offered lr 0.01.
  This was declared in advance as equal selection budget rather than equal
  tuning, and it limits interpretation. It does **not** license a rerun, and
  none is proposed.

No criterion, selection or configuration is changed in response to these
results, and no follow-up run is launched.

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
