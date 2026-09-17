# Report: retention-aware continuation

17 September 2026. Run `20260917-171413`, commit
`84389be6a2a62f5c4ab0dff4299ba1b60c422000`, branch
`prospective-retention-aware`. Protocol
`docs/PROSPECTIVE_RETENTION_AWARE_PROTOCOL.md`. Outputs
`/Users/durso/s5-runs/prospective-retention-aware/20260917-171413`; logs in
`.../logs/20260917-171413`. Numbers are copied from the saved digest. Paired
revision, retention and recall differences come from its full-precision lines;
the secondary delay curves in s5 use its two-decimal per-seed values. Nothing
was re-run for this report. All earlier studies and their verdicts are
unchanged.

## Summary

**Question.** Can generalized processing achieve more revision than literal
TSS when training explicitly discourages damage to untouched associations
and recall?

**Answer for this design: not determinable. Every primary verdict is
UNAVAILABLE.** No extension family (literal TSS, generalized processing or
the learned operator) had a development checkpoint whose retention **and**
recall both reached the continued-native reference. Recall was the binding
condition: no extension checkpoint reached native's development recall of
80.02%. The central comparison, generalized versus equally trained literal
TSS, therefore has no constrained endpoints to compare. The infeasibility is
recorded as such, and no fallback is counted as an improvement.

**Descriptive findings** (not verdicts):
- **λ = 1 penalty:** it held the category cross-entropies at or below the
  fixed references (both penalties were zero by the final logged steps), but
  it slowed revision learning by about 4 points at update 200 without lifting
  recall to native's level. No λ = 1 checkpoint was selected for any role.
- **λ = 0 controls** (also every unconstrained endpoint): generalized and
  literal TSS were practically indistinguishable: revision +0.11, retention
  +0.10, recall +0.09 points, positive in all three seeds but far below the
  +1 pp requirement. This replicates the temporal-response comparison on
  fresh streams (+0.09 revision there).
- **Temporal pattern:** generalized processing again concentrated its
  advantage at nominal delay 1 (+5.21 points) and gave no later benefit
  (later score −0.16).

A failed or infeasible result describes this declared objective, grid
(λ ∈ {0, 1}, lr 0.01, 200 updates) and native bar. It does not show that the
equation family has no parameters meeting the preservation requirement, and
it does not show that mass protects or harms memory.

## 1. Operational status

**PASS** (exit 0), 396 s of the 600-second cap.

| Stage | Result |
|---|---|
| TSS-containment checks / float32 probe | 64 passed (76 s) / OK (42 s) |
| Temporal-response checks | 24 passed (25 s) |
| Retention-aware checks | 13 passed (66 s)¹ |
| Source verification | unchanged before and after |
| Reference verification | 4 native references plus the reference run's `status.json`: declared mapping confirmed, restored, hashed, recorded metrics reproduced **exactly**; re-verified unchanged at finish by the finalizer and the launcher |
| Start points (4 sources) | storage identity, native-point recovery and literal-TSS start acceptance all passed |
| Preflight | all 8 family × λ executables accepted, no retraces; projection 173 s |
| Study | 185 s: 8 development and 12 final trajectories, 4,000 updates, 104 checkpoints validated and persisted, 27 endpoint rows, 4 frozen-source evaluations |

¹ The `native_full lambda0 retraced True` preflight line in that log comes
from the stubbed orchestration fixture that injects a retrace, not from
production.

**Preflight (production, disposable state):**

| Family | λ0 step | λ1 step | Warm-up (λ0 / λ1) |
|---|---:|---:|---|
| Literal TSS | 19.69 ms | 20.13 ms | 10.6 / 9.9 s |
| Generalized | 20.08 ms | 20.26 ms | 10.5 / 9.9 s |
| Native | 15.63 ms | 14.79 ms | 9.8 / 9.0 s |
| Operator | 16.83 ms | 15.87 ms | 10.4 / 9.7 s |

Checkpoints took 0.06–0.08 s, except 0.30 s for operator λ0.

**Operational PASS is not a performance verdict.** Because every selected
role landed on the λ = 0 slot at update 200, only 12 of the up to 24
planned final trajectories were needed. The rules require exactly this and
nothing was trimmed.

## 2. Fixed native references

From temporal-response run `20260917-163003` (native, slot B = lr 0.01,
update 200). Recorded and reproduced values were identical:

| Source seed | File | Revision | Retention | Recall |
|---|---|---:|---:|---:|
| 500 (development) | `dev_native_full_B_seed500_u200.msgpack` | 72.13 | 71.76 | 79.37 |
| 501 | `final_native_full_B_seed501_u200.msgpack` | 71.23 | 69.57 | 80.55 |
| 502 | `final_native_full_B_seed502_u200.msgpack` | 69.94 | 70.51 | 79.16 |
| 503 | `final_native_full_B_seed503_u200.msgpack` | 70.64 | 70.31 | 78.73 |

These references supplied only the training thresholds of the penalty.
Development feasibility and the held-out comparisons use the continued
native family trained in this run.

## 3. Development and constrained selection (source 500)

**Native reference** (unconstrained ordering): λ0, update 200, with revision
70.45, **retention R = 69.61** and **recall C = 80.02**.

Development revision / retention / recall (percent) at update 200:

| Family | λ = 0 | λ = 1 |
|---|---|---|
| Literal TSS | 71.82 / 69.18 / 79.73 | 67.32 / 70.04 / 79.59 |
| Generalized | 71.72 / 69.14 / 79.79 | 67.62 / 69.96 / 79.47 |
| Native | 70.45 / 69.61 / 80.02 | 66.31 / 70.12 / 80.08 |
| Operator | 71.05 / 69.26 / 79.57 | 67.29 / 70.23 / 79.53 |

**Feasibility:**

| Family | Feasible checkpoints | Highest development recall | Constrained endpoint |
|---|---:|---:|---|
| Literal TSS | 0 of 9 | 79.73 | **INFEASIBLE** |
| Generalized | 0 of 9 | 79.79 | **INFEASIBLE** |
| Operator | 0 of 9 | 79.57 | **INFEASIBLE** |

Several λ = 1 checkpoints met the retention bar, for example literal TSS
71.41 and generalized 71.56 at update 50. None met the recall bar.

**Selected descriptive roles:** for every family, the unconstrained endpoint
and the λ = 0 control are the same checkpoint (λ0, update 200).

### 3.1 The penalty during training (development)

With λ = 1, both penalties were driven to zero quickly. At updates 50, 150
and 199 they were exactly 0 in all four families. The recall penalty was
still active at update 100, between 0.038 and 0.049. With λ = 0, the same
(inactive) terms showed the category CEs sitting above the reference for
much of training: at update 50 the untouched penalty would have been
0.08–0.14. The penalty therefore changed training as intended at the CE
level. Its accuracy consequence was mainly slower revision learning.

## 4. Primary verdicts

| Comparison | Status |
|---|---|
| **Generalized − literal TSS (central)** | **UNAVAILABLE**: constrained selection infeasible for both |
| Generalized − native reference | **UNAVAILABLE**: generalized infeasible |
| Generalized − learned operator | **UNAVAILABLE**: both infeasible |

The required +1 pp revision gain, all-positive paired seeds and no mean
retention or recall decrease could not be evaluated, because no constrained
endpoints existed.

## 5. Descriptive held-out results (λ = 0 controls = unconstrained endpoints)

**Means over seeds 501–503 (percent):**

| Arm | Revision | Retention | Recall | Immediate | Later |
|---|---:|---:|---:|---:|---:|
| Generalized | 70.52 | 67.51 | 78.30 | 84.57 | 73.69 |
| Literal TSS | 70.41 | 67.40 | 78.21 | 79.36 | 73.85 |
| Native | 69.54 | **68.55** | **78.75** | 70.70 | 73.49 |
| Operator | 69.85 | 67.17 | 77.98 | 67.77 | 73.86 |
| Frozen source | 58.87 | 70.09 | 77.84 | — | — |

**Paired differences** (percentage points, from saved full-precision metrics;
seeds 501 / 502 / 503, then the mean). None passes the aggregate screen, and
none can stand in for a primary verdict.

| Comparison | Revision | Retention | Recall |
|---|---|---|---|
| Generalized − literal TSS | +0.117 / +0.127 / +0.078; **+0.11** | +0.020 / +0.176 / +0.117; **+0.10** | +0.117 / +0.137 / +0.020; **+0.09** |
| Generalized − native | +0.957 / +1.348 / +0.615; **+0.97** | −0.762 / −0.430 / −1.934; **−1.04** | −0.225 / −0.254 / −0.889; **−0.46** |
| Generalized − operator | +0.889 / +0.859 / +0.254; **+0.67** | +0.215 / +0.430 / +0.371; **+0.34** | +0.303 / +0.381 / +0.273; **+0.32** |
| Generalized − frozen source | +13.12 / +12.63 / +9.20; +11.65 | −2.56 / −1.89 / −3.30; −2.58 | +0.60 / +0.47 / +0.30; +0.46 |

**Generalized − literal TSS, secondary.**
- Immediate revision: +7.62 / +7.23 / +0.78, mean **+5.21**.
- Later score: −0.17 / −0.35 / +0.05, mean **−0.16**.

Revision-family delay curves, as mean paired differences over seeds, computed
from two-decimal per-seed values:

| Nominal delay | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Revised probes | +5.21 | −0.26 | −0.91 | −0.85 | −0.33 |
| Untouched probes | 0.00 | +0.07 | +0.13 | +0.20 | −0.13 |

Seed 503's generalized endpoint stayed close to literal TSS (M ≈ 0.014), and
most of its differences were zero.

## 6. Learned coefficients (held-out endpoints, λ = 0)

| Arm | Seed 501 | Seed 502 | Seed 503 |
|---|---|---|---|
| Generalized `M, gamma, T` | 0.179, 0, 1.507 | 0.191, 0, 1.573 | 0.014, 0.019, 1.100 |
| Generalized executed `c` | 1.487 | 1.459 | 1.855 |
| Literal TSS `T` | 1.647 | 1.696 | 1.112 |
| Literal TSS executed `c` | 1.607 | 1.590 | 1.900 |

The pattern matches the temporal-response run:
- `T` grew above `T0 = 1`;
- the generalized arm used modest mass in two seeds;
- the generalized arm's larger immediate revision again came with a
  **smaller** immediate amplitude `c` than literal TSS.

## 7. Deployment fallbacks (separate from scientific results)

No extension was deployable, so all three deployment choices are the native
reference model, with held-out revision 69.54%.
`counted_as_improvement = False`.

## 8. Costs

| Arm | Step (λ0 / λ1) | Carry (reals) |
|---|---|---:|
| Literal TSS | 19.7 / 20.1 ms | 320 |
| Generalized | 20.1 / 20.3 ms | 320 |
| Native | 15.6 / 14.8 ms | 128 |
| Operator | 16.8 / 15.9 ms | 192 |

The reference forward pass is included in every step. Of the 396 s total,
checks took 209 s and the study 185 s, of which the eight preflight
compilations were about 80 s.

## 9. Interpretation and limits

- **Infeasible, not negative.** The constrained question was not answered.
  Under λ ∈ {0, 1}, lr 0.01 and 200 updates, neither generalized processing
  nor literal TSS reached native's development recall, so they could not be
  compared under the preservation requirement.
- **A soft CE penalty is not an accuracy constraint.** λ = 1 met its CE
  targets but cost revision and did not restore recall accuracy.
- **Unconstrained behaviour is stable across studies.** Generalized
  processing and literal TSS differ by about a tenth of a point in aggregate.
  Generalized processing's short-delay advantage is offset at longer delays.
- **Native as bar and reference.** The continued-native bar comes from this
  run's own native training, while the penalty thresholds come from the fixed
  temporal-response references. The two are intentionally distinct.
- **Scope.** Three seeds on a small synthetic associative task: finite-sample
  screens, not significance, a benchmark or SOTA. There is no Gated DeltaNet
  arm, and no claim that mass protects memory.
