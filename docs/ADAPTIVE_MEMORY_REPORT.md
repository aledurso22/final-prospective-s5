# Adaptive associative-memory comparison — report

**Status: prepared, not yet executed.** No numerical check, calibration or
training run for this study has been executed anywhere — not on the cluster and
not locally. Every result section below is empty and will be filled from actual
cluster output, favourable or not.

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
| Launch commit | *(this commit — R1–R5 and F1/F2 corrected)* |
| Executed commit | *(to be recorded from the launcher's `git rev-parse HEAD`)* |

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

## Results

*(empty — to be filled from cluster output)*

### Calibration

| Slot | Solved rate | Bracket | Observable error | Realized β |
|---|---|---|---|---|
| | | | | |

`ν = 4/3` recovery for slot 1A: *(pending, tolerance 1e−8)*

### Checks

*(pass/fail counts and any failures, verbatim from `checks.log`)*

### Preflight and cost

| Arm | Compile (incurred) | Step | Eval | Projected arm |
|---|---|---|---|---|
| | | | | |

Incurred compilation: *(pending)*  Projected remaining: *(pending)*
Whether the batch fit: *(pending)*

### Development stage and selection

| Family | A primary | A rev-CE | B primary | B rev-CE | Selected |
|---|---|---|---|---|---|
| | | | | | |

### Final seeds — held-out

| Arm | seed 201 | 202 | 203 | mean primary | retention | recall |
|---|---|---|---|---|---|---|
| | | | | | | |

Per-category results for all four categories and both families, initialization
and update-100/update-200 validation, training gain reported separately from
endpoint accuracy, learned response coefficients, gate distributions,
`η = νρ` and `κ = τν(1−ρ)`, state/auxiliary norms, gradient/update norms:
*(pending — see `status.json`, `development.json`, `final.json`)*

### Verdicts

* **Literature screen** (vs Momentum and Gated DeltaNet): *(pending)*
* **Ordinary-prospectivity extension screen** (vs TSS and ideal equilibrium):
  *(pending)*
* **Attribution** to the prospective derivative (vs the equally gated inertial
  control, and the first-order delta comparison): *(pending)*

These are reported separately. A win on one cannot substitute for a loss on the
other, in either direction.

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
