# Report: literal TSS versus master-law residual processing on the same Momentum backbone

17 September 2026. Run `20260917-154035`, commit `073af37a8ab6fff83200a199729a2adbb0b9541b`,
branch `prospective-tss-containment`. Specification
`docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md`; frozen protocol
`docs/PROSPECTIVE_TSS_CONTAINMENT_PROTOCOL.md` (amendment s12). Outputs
`/Users/durso/s5-runs/prospective-tss-containment/20260917-154035`, logs
`.../logs/20260917-154035`. All numbers below are copied from the saved
digest; nothing was re-run for this report.

## Summary

Two findings, both on three independently pretrained Momentum sources and
fresh held-out episodes:

1. **Literal TSS residual processing produces a substantial revision gain over
   native Momentum.** With `M = gamma = 0` fixed and only `T` learned, held-out
   revision macro accuracy is **69.39%** against **55.35%** for continued native
   Momentum: **+14.04 points**, positive in every seed (+12.37, +14.03,
   +15.72). The gain costs **2.40 points of retention** (−1.37, −3.32, −2.51)
   and 0.27 points of recall.
2. **Departing from the exact TSS boundary has not shown a consistent
   retention-preserving benefit.** The generalized law, started on the same
   function, reaches 70.84% revision: **+1.45 points** over literal TSS, but
   paired differences of −0.78, +0.45, +4.68 are not consistently positive,
   and retention is **lower in every seed** (−0.29, −1.07, −1.51; mean
   −0.96).

**The matched-retention verdicts are infeasible.** No development checkpoint
of any extension (literal TSS, generalized processing or the learned operator)
reached continued native Momentum's development retention *and* recall. Every
extension endpoint is therefore a flagged diagnostic endpoint, every
constrained screen is reported as infeasible/unavailable, and every
deployment selection falls back to the native model. None of this is an
improvement claim. The comparisons below are between trained, constraint-
failing endpoints and are reported as such.

## 1. Operational status

**PASS** (exit 0), 242 s of the 600-second cap: float64 checks 76 s (64
passed), float32 probe 42 s, study 123 s (preflight projected 125 s, no
failures or retraces). 8 development trajectories and 12 final endpoints,
4,000 updates, 104 checkpoints validated and persisted, 0 zero-update
endpoints, 4 frozen-source evaluations. Every update's executed filter
coefficients passed the gate; every checkpoint was accepted. Sources were
sha256-verified before and after; selection and the deployment plan were
unchanged through finals and evaluation. Operational PASS is not a
performance verdict.

The previous dispatch `20260917-152713` (commit `e38b61a`) FAILED at the
focused checks and is preserved; its test-method amendment is protocol s12.
In this run the declared analytic zero ("T inward at the native point") was
**verified within tolerance** in both precisions (float64: JVP 0.0, reference
−1.1e-17, floors 2.2e-13; float32: JVP 0.0, float32 floor 1.19e-4, float64
reference −2.1e-17 against 2.2e-13). All nondegenerate derivative fixtures
agreed with the independent reference (float64 relative error ≤ 4.1e-15;
float32 ≤ 3.8e-6, primal loss ≤ 6.8e-8, carries ≤ 1.6e-6).

**Start points.** On all four sources the two processing arms' start trees
were bitwise identical, the literal-TSS start executed `a = 0, b = 0, c = 2,
d = 1` and was accepted, and the processing law at the native point
`(M, gamma, T) = (0, h, 0)` reproduced native Momentum with relative error
0.0 on logits and on W, U over the representative episodes (aggregate counts
and cross-entropy identical). Neither processing arm starts training at the
native point, but these recovery checks evaluate it. The identity along the
whole line `M = 0, gamma = h` (native for every fixed `T`) is exact
mathematically; numerically, recovery is checked within tolerance.

## 2. Development selection (source 500) — retention constrained in SELECTION, not in optimization

Training used the unchanged, unweighted query cross-entropy for every arm;
**no retention term, weight or constraint entered optimization.** Retention
and recall entered only afterwards, as eligibility conditions on development
checkpoints.

Native Momentum was selected first (unconstrained ordering): lr 0.01, update
200, revision 55.76%, **retention R_native = 68.46%, recall C_native =
77.00%**. Its continued training *raised* retention from 65.72% at update 0.

| Family | Best development retention (any checkpoint) | Best development recall | Feasible checkpoints | Final endpoint |
|---|---:|---:|---:|---|
| Literal TSS processing | 66.06 | 76.68 | 0 of 9 | lr 0.01, update 200: 70.39 / 65.23 / 76.51 — **diagnostic** |
| Generalized processing | 66.46 | 76.73 | 0 of 9 | lr 0.01, update 200: 72.85 / 62.84 / 76.10 — **diagnostic** |
| Learned two-tap operator | 66.60 | 76.51 | 0 of 9 | lr 0.01, update 200: 72.71 / 62.16 / 75.66 — **diagnostic** |

(Endpoint columns: development revision / retention / recall, percent.)
Every extension checkpoint failed **both** thresholds, including the update-0
checkpoints. The processing arms' shared update-0 checkpoint had revision
56.13, retention 65.72, recall 75.42.

Per the frozen rule, each family's final endpoint is its best unconstrained
development checkpoint, flagged diagnostic and constraint-failing. The
constrained (matched-retention) screen is **infeasible** for every family.

## 3. Held-out results

Means over seeds 501–503 (percent; revision cross-entropy in the last column):

| Arm | Revision | Retention | Recall | Rev. CE |
|---|---:|---:|---:|---:|
| Generalized processing (M, gamma, T) | 70.84 | 63.22 | 77.18 | 0.936 |
| Literal TSS processing (T) | 69.39 | 64.18 | 77.19 | 0.965 |
| Learned two-tap operator (kappa) | **72.68** | 61.37 | 76.21 | 0.872 |
| Native Momentum, continued | 55.35 | **66.59** | **77.45** | 1.313 |
| Native source, frozen (anchor) | 54.52 | 65.74 | 76.20 | 1.338 |

Per seed (revision / retention / recall):

| Arm | Seed 501 | Seed 502 | Seed 503 |
|---|---|---|---|
| Generalized | 68.71 / 65.36 / 78.10 | 69.46 / 62.57 / 76.56 | 74.35 / 61.74 / 76.88 |
| Literal TSS | 69.49 / 65.65 / 77.87 | 69.01 / 63.65 / 76.23 | 69.68 / 63.26 / 77.45 |
| Operator | 73.32 / 62.26 / 76.72 | 72.12 / 60.82 / 75.18 | 72.62 / 61.04 / 76.73 |
| Native, continued | 57.13 / 67.02 / 77.45 | 54.98 / 66.97 / 76.89 | 53.96 / 65.77 / 78.02 |
| Native, frozen | 54.72 / 66.26 / 76.59 | 54.50 / 65.65 / 75.99 | 54.33 / 65.31 / 76.03 |

Query categories, means over seeds (revision task | recall task):

| Arm | immediate selected | middle untouched | late selected | late untouched |
|---|---|---|---|---|
| Generalized | 87.62 \| 99.33 | 76.74 \| 70.15 | 69.30 \| 93.61 | 49.71 \| 45.64 |
| Literal TSS | 92.56 \| 99.81 | 76.97 \| 69.84 | 56.64 \| 91.19 | 51.40 \| 47.90 |
| Operator | 98.31 \| 99.85 | 73.62 \| 66.28 | 69.69 \| 93.80 | 49.12 \| 44.92 |
| Native, continued | 49.56 \| 98.65 | 80.66 \| 75.16 | 38.69 \| 85.63 | 52.50 \| 50.37 |
| Native, frozen | 48.44 \| 98.44 | 79.82 \| 73.52 | 38.17 \| 84.44 | 51.66 \| 48.40 |

Retention is the mean of the revision task's middle-untouched and
late-untouched categories.

## 4. Comparisons, each reported separately

Criterion for a promising matched-retention signal: mean held-out revision
difference ≥ +1 pp, all three paired revision differences positive, mean
retention and recall differences ≥ 0, and no diagnostic endpoint. **No
comparison meets it, and every constrained screen is infeasible.** The
historical −1 pp safeguard is shown for reference only.

Paired differences (seeds 501 / 502 / 503; mean), percentage points:

| Comparison | Kind | Revision | Retention | Recall | No measured decrease | −1 pp safeguard |
|---|---|---|---|---|---|---|
| Literal TSS − native | scientific | +12.37 / +14.03 / +15.72; **+14.04** | −1.37 / −3.32 / −2.51; **−2.40** | +0.42 / −0.66 / −0.56; −0.27 | no | no |
| Generalized − literal TSS | scientific | −0.78 / +0.45 / +4.68; **+1.45** | −0.29 / −1.07 / −1.51; **−0.96** | +0.23 / +0.33 / −0.57; −0.00 | no | no |
| Generalized − native | scientific | +11.58 / +14.48 / +20.40; **+15.49** | −1.66 / −4.39 / −4.03; **−3.36** | +0.65 / −0.33 / −1.14; −0.27 | no | no |
| Generalized − operator | outside the claim | −4.60 / −2.66 / +1.73; **−1.84** | +3.10 / +1.76 / +0.71; **+1.86** | +1.38 / +1.38 / +0.15; +0.97 | yes | no |
| Operator − native | outside the claim | +16.19 / +17.14 / +18.66; **+17.33** | −4.76 / −6.15 / −4.74; **−5.22** | −0.73 / −1.71 / −1.28; −1.24 | no | no |
| Generalized − frozen source | descriptive | +13.99 / +14.95 / +20.02; +16.32 | −0.90 / −3.08 / −3.56; −2.51 | +1.51 / +0.57 / +0.85; +0.98 | no | no |

### 4.1 Literal TSS versus native Momentum

A **substantial revision gain**: +14.04 points, positive in all seeds, with
revision cross-entropy 0.965 against 1.313. By category it comes from the
*selected* keys that must be revised: immediate selected +43.00 and late
selected +17.95 points. The untouched categories that define retention fall
(middle untouched −3.70, late untouched −1.10). In the recall task, late
selected rises (+5.57) while middle and late untouched fall (−5.32, −2.47).
Most of the processing arms' improvement over native is already present at
the literal-TSS boundary.

### 4.2 Departing from the boundary: generalized versus literal TSS

Both arms start from the same function, so this is the extension question.
The generalized law **did not show a consistent retention-preserving
benefit**:

- revision +1.45 on average, but −0.78 in seed 501, +0.45 in 502 and +4.68
  in 503, so not consistently positive;
- retention lower in every seed (mean −0.96), recall unchanged on average;
- by category the generalized law trades immediate-selected revision (−4.95)
  for late-selected revision (+12.66), with small losses on both untouched
  categories (−0.22, −1.69) and on recall late untouched (−2.26).

Both endpoints are constraint-failing diagnostics, so this is not a
matched-retention comparison.

### 4.3 Generalized processing versus the learned two-tap operator

Outside the containment claim: the operator's learned `kappa > 1` corresponds
to `gamma < 0` and is not a point of this nonnegative-`gamma` family. The
result is a **trade-off**. The operator is higher in revision in two of three
seeds (mean +1.84 over generalized), mostly through immediate selected
(+10.69). The generalized law keeps more retention (+1.86, all seeds) and
recall (+0.97, all seeds). "No measured decrease" holds for generalized
versus operator, but with a negative mean revision difference it is not a
win. Both are constraint-failing endpoints.

### 4.4 Deployment fallbacks — separate from scientific results

With no feasible checkpoint in any extension family, the frozen deployment
plan selects the **native Momentum model** for all three families (native
lr 0.01, update 200; held-out revision 55.35%). For the literal-TSS family
this is recorded as an *external* native selection, not a point of the
literal-TSS family, and it is never labelled TSS. Fallbacks are deployment
choices only (`counted_as_improvement = False`). They do not show that joint
optimization reverted to a subfamily, and they are not part of any finding
above. The declared native-before-TSS tie rule was never applied.

## 5. Learned coefficients

Executed values at the final endpoints (lr 0.01, update 200):

| Arm | Seed 501 | Seed 502 | Seed 503 |
|---|---|---|---|
| Literal TSS `T` (M = gamma = 0) | 0.751 | 0.733 | 0.738 |
| Literal TSS executed `a, c` (b = 0, d = 1) | −0.332, 2.332 | −0.364, 2.364 | −0.354, 2.354 |
| Generalized `M, gamma, T` | 0.316, 0, 0.269 | 0.289, 0, 0.233 | 0.245, 0, 0.210 |
| Generalized executed `a, b, c, d` | −0.167, 0.540, 2.167, 0.460 | −0.361, 0.553, 2.361, 0.447 | −0.658, 0.538, 2.658, 0.462 |
| Generalized minimum Jury slack | 0.460 | 0.447 | 0.462 |
| Operator `kappa` | 2.159 | 2.070 | 1.883 |

- **Literal TSS** shortened its horizon from `T0 = 1` to about 0.74 (after a
  rise to about 1.3 early in training), staying strictly stable (minimum
  slack 0.64–0.67). Its `M` and `gamma` stayed at exactly 0.
- **Generalized processing** acquired mass (M about 0.25–0.32 at the end,
  after mid-training peaks of about 0.36, 0.40 and 0.28 in seeds 501–503) and a much shorter horizon (T fell from
  1.01 to 0.21–0.27). All executed filters stayed strictly stable.
- **Learned operator:** kappa 1.88–2.16 (kappa/bound ≤ 0.092, no projections,
  288 of 288 frozen-token transitions stable), again corresponding to
  `gamma = h(1 − kappa) < 0`.

### 5.1 Activity at the gamma boundary

The generalized arm's `gamma` spent most of training **at its lower
boundary**. In the per-update telemetry sampled every 25 updates, the
optimizer's pre-repair proposal for `gamma` was negative (as low as about
−0.013) at 7 of 9 logged updates in each seed. Each time the deterministic
feasibility repair clamped it back to 0. Every endpoint executed
`gamma = 0`. Early in training `gamma` was briefly positive (0.034 and 0.018
at update 25 in seeds 501 and 502), and seed 503 logged a positive value at
update 175. The full per-update histories are in `status.json`.

This records that the `gamma >= 0` boundary was **active**: the unweighted
revision-and-recall loss pushed toward negative damping. It does **not**
show that allowing negative `gamma` would improve retention. Nothing in
this run evaluates that region. The learned operator, which does sit in the
corresponding `kappa > 1` region, has the *lowest* retention of all arms
(61.37%). No domain change is proposed or made by this report.

### 5.2 Physical scope

The coefficient domain is broader than the passive two-compartment sector.
Under the established condition for that circuit, `M <= gamma T` with
`gamma > 0` (`docs/META_DELTA_PROOF_AUDIT.md` s7), the **learned generalized
endpoints lie outside the passive-compartment sector**: they have
`gamma = 0` and `M > 0`, so `M > gamma T`. The literal-TSS endpoints sit on
the `M = gamma = 0` boundary, not inside the `gamma > 0` sector. The learned
operator's equivalent `gamma < 0` is outside the nonnegative family
altogether. These are computational uses of the same equation, not
realizations of a passive circuit.

The executed-coefficient gate certifies the rounded coefficients of the
isolated processing filter only. It is not a statement about every
floating-point trajectory, the closed-loop associative memory, switching
across tokens, or the Momentum block below the filter.

## 6. Costs

| Arm | Stored / trainable params | Carry (reals) | Step (preflight) | Checkpoint | Final trajectory |
|---|---:|---:|---:|---:|---:|
| Literal TSS | 572 / 570 | 320 (theoretical minimum 256) | 18.64 ms | 0.09 s | 4.4–4.5 s |
| Generalized | 572 / 572 | 320 | 18.80 ms | 0.09 s | 4.5–4.6 s |
| Native, continued | 569 / 569 | 128 | 14.12 ms | 0.10 s | 3.8 s |
| Operator | 570 / 570 | 192 | 15.45 ms | 0.34 s | 4.0 s |
| Native, frozen | 569 / 0 | 128 | — | — | 0.2 s |

The processing arms cost about 33% more per step than native Momentum and
2.5 times its carry. Wall time for the whole run was 242 s: checks 76 s,
float32 probe 42 s, study 123 s.

## 7. Scope and limits

- Three seeds on a small synthetic associative task: finite-sample screens,
  not statistical significance, noninferiority, a benchmark or SOTA.
- Sources are reused from replication `20260917-011842` (read-only). This is
  an intervention study, not another independent-source replication.
- No Gated DeltaNet arm was run, so no joint literature comparison follows.
- The matched-retention selector compares against *continued* native
  Momentum, whose retention rose during continuation. The infeasibility is
  relative to that baseline and the frozen thresholds. No threshold was
  changed after the results.
- Literal TSS containment by this placement is exact. Containment of the
  learned `kappa > 1` operator is not claimed.
- Reporting only: no additional training, no domain change, and no new
  selection rule.

## 8. Files

- Digest: `experiments/prospective_momentum/tss_containment_summary.py`
  (read-only display fixes in this commit: native selection flags now read
  from the frozen selection, and native Momentum's frozen-token table is no
  longer labelled "OPERATOR"; no stored result changes).
- Protocol s12: corrected wording about native-point evaluation; s13 points
  here.
