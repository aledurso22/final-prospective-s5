# Report: independent-source replication of prospective Momentum DeltaNet

Protocol: `docs/PROSPECTIVE_MOMENTUM_REPLICATION_PROTOCOL.md`.
Equations and audit: `docs/PROSPECTIVE_MOMENTUM_PROOF_AUDIT.md`.
Completed shared-source study: `docs/PROSPECTIVE_MOMENTUM_REPORT.md`
(run `20260917-000431`, unchanged).

## Finding

Across three **independently initialized and independently pretrained**
sources, the prospective correction raised revision accuracy over native
Momentum DeltaNet by **+12.21 percentage points**, with the improvement
positive in every final seed. Untouched-association retention fell by
**1.24 points** and recall by 0.61 points.

**The declared joint literature screen therefore FAILED its retention
safeguard.** The result supports a repeatable revision benefit with a
retention trade-off. It is not unconditional superiority, significance or a
SOTA claim.

## 1. Record and verification

| | |
|---|---|
| run | `20260917-011842` |
| commit | `6b0c0b26eaf2d107d5908c31bad383c359fd57a6` |
| branch | `prospective-momentum-replication` |
| cluster | `pgi15-gpu3`, SLURM 66104, GPU, JAX 0.11.0 |
| artifacts | `/Users/durso/s5-runs/prospective-momentum-replication/20260917-011842` |
| logs and digest | `.../logs/20260917-011842` |
| operational outcome | terminal **PASS**, exit 0; **326 s of the 600 s cap** |

- **Checks:** 66 passed, including the 16 replication checks. The independent
  analytic kappa sensitivity matched the production JVP to 2.7e-16 at the
  start and 9.1e-15 at the interior point. The h = 1e-6 finite-difference
  discrepancy (1.16e-6) remains a printed diagnostic, not an acceptance
  reference; the float32 derivative resolvability limitations remain
  limitations.
- **Work:** 16 source-pretraining + 12 development + 18 final runs =
  **46 runs, 9,200 optimizer updates**, all inside the single cap.
- **Cost:** backend ≈ 8 s, focused checks 138 s, study 186 s. Preflight
  projected 247.9 s (sources 60.2, identity 1.6, continuation 146.1, host
  allowance 40.0) and refused nothing. Step cost 12.8–14.6 ms on both the
  source and continuation paths; save + checksum + restore 0.11 s.
- **Integrity:** all 32 new source files verified unchanged at finalization,
  and the completed study's read-only checkpoints likewise
  (`integrity=0`, `source_unchanged=True`, digest complete).

Evidence is the cluster console and the saved digest, cross-checked line by
line against this report. No independent inspection of the remote files and
no new numerical work were performed to write it.

## 2. Independent sources

Each source seed freshly initialized and pretrained all four families with
the original initializer, slot-B coefficients and the fixed recipe
(200 updates, lr 0.01, unweighted query cross-entropy, batch 8 per family,
original optimizer and projection). Every source restored **bitwise equal**
with no reproduction failure and no invalid source; none was ranked,
replaced or discarded.

| Source | Revision primary, start → end | sha256 (first 16) |
|---|---|---|
| `src500_momentum_delta` | 38.75 → 55.76 | `cd5fea25105af27b` |
| `src500_gated_delta` | 34.86 → 55.59 | `062a1e6e20ec6b32` |
| `src500_gp_two_sided` | 44.78 → 59.67 | `d9b267e423a4b673` |
| `src500_tss_eq17` | 41.77 → 53.34 | `9fdbd4261cf1c248` |
| `src501_momentum_delta` | 45.85 → 54.71 | `ae00ba1d582199c0` |
| `src501_gated_delta` | 34.45 → 54.79 | `f04277c32ac7b6af` |
| `src501_gp_two_sided` | 43.99 → 58.96 | `08f0b94fd9924b95` |
| `src501_tss_eq17` | 40.11 → 51.29 | `13997eca346ac3a8` |
| `src502_momentum_delta` | 29.47 → 54.69 | `d582b02f945ae77f` |
| `src502_gated_delta` | 44.87 → 56.45 | `8602bad642f91018` |
| `src502_gp_two_sided` | 43.65 → 60.03 | `c2b7ed13d8ba805d` |
| `src502_tss_eq17` | 40.67 → 53.22 | `22e8b50ebc96fe25` |
| `src503_momentum_delta` | 35.72 → 54.25 | `5ec05eda35f1643d` |
| `src503_gated_delta` | 38.21 → 55.98 | `a1502b257198aca9` |
| `src503_gp_two_sided` | 45.14 → 59.86 | `14cb41a0c6d0c059` |
| `src503_tss_eq17` | 41.85 → 52.91 | `2d7e731dda7a56a9` |

Starting primaries span 29.47–45.85 %, so the four seeds really are distinct
models. The candidate, native Momentum and learned-gain arms shared each
seed's Momentum checkpoint, and their update-zero identity check passed for
all four source seeds. Every continuation run records the sha256 of the file
it restored.

## 3. Design (unchanged from the completed study)

Development used source seed 500 only; final models used their own sources
501, 502 and 503. Continuations were 200 updates with the unchanged
learning-rate slots 0.003 and 0.01; the update-200 rule selected **slot 0.01
for every arm**, and the selection was frozen before the finals and verified
unchanged afterwards.

The unweighted query cross-entropy, full BPTT, equations, gates, projection,
carry sizes, data recipe, evaluation conventions and screen thresholds were
unchanged. There was **no retention reweighting and no endpoint checkpoint
search.** Source-training, continuation, validation and held-out streams were
declared fresh and asserted disjoint from every earlier study. One common
held-out set evaluated all final models, opened only after the finals
finished.

The three final seeds vary initialization, source pretraining and
continuation. They are not independent selections of the development recipe
and not independent held-out datasets.

## 4. Held-out means

Percentages. Revision is the revision-family macro accuracy; retention is
untouched associations within that family; recall is the recall-family
accuracy.

| Rule | Revision | Retention | Recall |
|---|---:|---:|---:|
| **Prospective-corrected Momentum** | **67.68** | 66.93 | 75.92 |
| Native Momentum DeltaNet rule | 55.47 | 68.16 | 76.54 |
| Momentum with learned gain (control) | 55.49 | 68.16 | 76.53 |
| Gated DeltaNet rule | 57.23 | 59.40 | 70.11 |
| Earlier generalized recurrence | 65.16 | 58.98 | 70.88 |
| TSS Eq. (17), directly applied to fast weights | 54.83 | 72.40 | 78.89 |

The TSS comparator keeps its applicability limitation: `(I − Df)` is singular
off the current key and zero on idle intervals. These numbers do not rank the
candidate against TSS in TSS's own setting.

## 5. Screens, reported separately

Every comparison requires a mean revision improvement of at least +1 pp,
positive paired revision differences in all three seeds, and mean retention
**and** recall differences each no worse than −1 pp.

| Candidate minus comparator | Revision, pp | Retention, pp | Recall, pp | Verdict |
|---|---:|---:|---:|---|
| Native Momentum DeltaNet | +12.21 | −1.24 | −0.61 | **FAIL: retention** |
| Gated DeltaNet | +10.46 | +7.53 | +5.81 | PASS |
| **Joint literature screen** | | | | **FAILED** |
| Learned-gain control | +12.19 | −1.24 | −0.60 | **FAIL: retention** |
| Earlier generalized recurrence | +2.52 | +7.94 | +5.05 | PASS |
| TSS Eq. (17), applicability-limited | +12.86 | −5.47 | −2.97 | FAIL: retention and recall |

All five comparisons have positive paired revision differences in every final
seed. The joint literature screen needs both literature comparisons and
fails. The gain control shows a clear revision advantage for the candidate
but its full screen fails on the same retention safeguard. **Operational PASS
does not alter any of these verdicts.**

**Safeguard pass is not zero loss.** Retention *decreased* against native
Momentum, the gain control and TSS, and *increased* against Gated DeltaNet
and the earlier generalized recurrence. Recall *decreased* against native
Momentum (−0.61), the gain control (−0.60) and TSS (−2.97); the first two
passed their safeguard while still losing accuracy.

### Paired per-seed differences (percentage points)

| Comparator | Seed 501 rev / ret / rec | Seed 502 rev / ret / rec | Seed 503 rev / ret / rec |
|---|---|---|---|
| Native Momentum | +11.72 / −1.29 / −0.32 | +12.18 / −1.59 / −0.78 | +12.73 / −0.83 / −0.74 |
| Learned-gain control | +11.62 / −1.32 / −0.31 | +12.26 / −1.56 / −0.76 | +12.71 / −0.83 / −0.74 |
| Gated DeltaNet | +10.99 / +9.13 / +6.23 | +10.22 / +6.76 / +5.58 | +10.17 / +6.69 / +5.64 |
| Earlier generalized recurrence | +3.06 / +9.28 / +5.48 | +2.27 / +7.23 / +4.92 | +2.23 / +7.32 / +4.74 |
| TSS Eq. (17), applicability-limited | +13.06 / −4.86 / −2.27 | +12.96 / −6.25 / −3.26 | +12.55 / −5.30 / −3.38 |

### Comparison with the completed shared-source study

| | Revision, pp | Retention, pp | Recall, pp |
|---|---:|---:|---:|
| Shared source (`20260917-000431`) | +11.90 | −1.08 | −0.51 |
| Independent sources (this run) | +12.21 | −1.24 | −0.61 |

Independent sources reproduced both the large revision gain and the smaller
retention cost. The earlier failed safeguard is preserved, not revised.

## 6. Where the effect is

Revision-family query categories, full names, held-out means:

| Revision query category | Prospective-corrected Momentum | Native Momentum |
|---|---:|---:|
| Immediate query of the revised association | 98.49 % | 47.01 % |
| Middle query of an untouched association | 79.15 % | 81.25 % |
| Late query of the revised association | 38.40 % | 38.56 % |
| Late query of an untouched association | 54.70 % | 55.08 % |

The improvement is concentrated in immediate access to the revised
association. Late accuracy on that same revised association is essentially
unchanged, and most of the untouched-association loss appears in the middle
query category. The data therefore support **faster revision access, not
improved long-term retention of revisions.**

Recall-family categories move the same way: immediate 100.00 % versus
98.18 %, middle untouched 70.20 % versus 73.83 %, late revised 85.19 % versus
85.38 %, late untouched 48.31 % versus 48.76 %.

## 7. Learned coefficients and stability record

- The candidate keeps the same 128-value recurrent carry as native Momentum
  and adds **one** learned scalar (570 parameters versus 569).
- Final correction coefficients: **2.625, 2.393, 2.488**, rising steadily with
  negative gradients throughout each continuation.
- **No projection events in any run.** Final kappa sat at 6–16 % of the
  table bound and at most 0.37 % of the bound computed from the gates the
  evaluation rollouts actually returned. The data therefore do not support a
  binding projection cap as the cause of the retention deficit.
- All 288 table-coverage frozen-token transitions were classified stable in
  every run, and no observed rollout write token had a negative analytic Jury
  value.
- **Sector, reported from each source's own learned coefficients:** all three
  final candidates occupy the computational extension entirely outside the
  passive two-compartment sector (passive fraction 0.0), with QHM `nu`
  ≈ 0.9999 and write-token `mu` ≈ 0.99997.
- The learned-gain control settled at g ≈ 0.990, 0.949, 0.945 with no
  projection, and performed like native Momentum (55.49 versus 55.47).
- The earlier generalized recurrence learned rho ≈ 0.060 and T ≈ 11.4–11.8 on
  all three seeds, inside the passive sector and certified, with no
  projections.

The failed gain-only control distinguishes this extension from a simple
learned write-gain adjustment. It does not isolate a fixed-parameter causal
effect: gates and the other parameters trained jointly with the correction.

## 8. Interpretation limits

- A small associative-memory experiment, not a published-scale SSM benchmark
  and not a test of statistical significance.
- Three seeds; the held-out set and the development-selected slot are common
  to all final models by design.
- The fixed-coefficient update remains QHM-equivalent at alpha = 1; no
  optimizer novelty is claimed.
- Frozen-token transition checks are not switching-stability proofs.
- The learned operating points lie outside the passive circuit sector under
  the declared mapping, so a passive biological realization is not
  established by this result.
- Operational PASS is an execution statement only.

## 9. Status

The completed shared-source study, this replication, all runs, logs, source
checkpoints and digests are preserved unchanged. No further run is requested
by this report. A next proposal should first explain how it addresses the
measured revision/retention trade-off, and must preserve these findings and
criteria.
