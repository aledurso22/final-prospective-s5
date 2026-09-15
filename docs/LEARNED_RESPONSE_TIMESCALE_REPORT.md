# Learned response timescale: results

Protocol committed before execution:
`docs/LEARNED_RESPONSE_TIMESCALE_PROTOCOL.md`.

**Outcome: the per-mode response timescale was genuinely learned — `T` moved
away from its reference and fanned out across modes — and it changed the
endpoint by `-0.035` pp. Both generalized arms finished behind the matched
ordinary substrate and behind Rawat's prospective-input S5.**

This is a **one-seed development screen**. It cannot establish a robust effect
in either direction, and nothing here escalates to a larger batch.

## 1. Execution identity

| item | value |
|---|---|
| checkout | `/Local/durso/final-prospective-s5` |
| branch | `learned-response-timescale` |
| executed commit | **`d07380c5d48240660dc65495cd364677f608004e`** |
| host | `pgi15-gpu3.iff.kfa-juelich.de`, RTX 3090 |
| SLURM job | 65910, `CUDA_VISIBLE_DEVICES=0` |
| backend | `gpu`, `CudaDevice(id=0)`, jax 0.11.0 |
| artifacts | `/Users/durso/s5-runs/timescale/20260916-003607/` |
| logs | `/Users/durso/s5-runs/timescale/logs/20260916-003309/` |
| focused checks | **58 passed**, 172 s |
| preflight projection | 337.9 s |
| study | **5/5 arms, `complete=True`, no incomplete stages** |
| total | **657 s of the 1200 s cap**, study wall 477 s |
| status | **`TIMESCALE_STATUS=PASS`, `TIMESCALE_EXIT=0`** |

```
bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_timescale.sh
```

Seed 100, ten epochs, batch 32, 843 steps per epoch, from scratch, full BPTT.
Train and validation splits only; the test arrays were never opened.

## 2. The result

Validation accuracy after the **fixed tenth epoch**, the predeclared primary:

| arm | endpoint | endpoint CE | best | stored | trainable | carried state |
|---|---|---|---|---|---|---|
| `alpha_p_s5` (Rawat) | **0.9500** | 0.28953 | 0.9500 | 35,050 | 35,050 | 256 |
| `gain_clip_s5` (matched ordinary) | 0.9460 | 0.29908 | 0.9460 | 35,050 | 35,050 | 128 |
| `gp_rho_T_fixed` | 0.9429 | 0.29982 | 0.9429 | 35,178 | **35,114** | 256 |
| **`gp_rho_T`** | 0.9426 | 0.29946 | 0.9426 | 35,178 | 35,178 | 256 |
| `native_s5` | 0.9417 | 0.30953 | 0.9417 | 35,050 | 35,050 | 128 |

Every count matches the protocol's predeclared expectation exactly, including
the frozen arm's 64-value gap between stored and trainable. Best-validation and
endpoint coincide for every arm, so the secondary selection changes nothing
here; it is reported separately either way and did not replace the endpoint.

Paired differences in percentage points:

| comparison | pp |
|---|---|
| **`gp_rho_T` − `gp_rho_T_fixed`** (the isolating comparison) | **−0.035** |
| `gp_rho_T` − `gain_clip_s5` | −0.346 |
| `gp_rho_T` − `alpha_p_s5` | −0.744 |
| `gp_rho_T` − `native_s5` | +0.086 |
| `gp_rho_T_fixed` − `gain_clip_s5` | −0.311 |
| `gp_rho_T_fixed` − `alpha_p_s5` | −0.709 |

The three control arms reproduce the earlier batch's numbers exactly —
`native_s5` 94.17 %, `gain_clip_s5` 94.60 %, `alpha_p_s5` 95.00 % — which is a
provenance check on the shared substrate, not a new measurement.

## 3. The decisive question: was the new freedom exercised?

An endpoint of `-0.035` pp has two very different readings — *the coordinate
moved and did not help*, or *the coordinate never moved* — and they imply
different follow-ups. The saved telemetry settles it.

**`T` moved, and modes differentiated.** Learned arm, final executed `T` per
layer (min / median / max over 16 stored modes):

| layer | `T` | `rho` | `mu = rho T` | median `\|T j\|` |
|---|---|---|---|---|
| 0 | 4.526 / **5.574** / 7.180 | 0.651 / 0.758 / 0.961 | 3.708 / 4.348 / 5.581 | 0.151 |
| 1 | 3.909 / **5.495** / 7.618 | 0.658 / 0.787 / 0.954 | 3.731 / 4.267 / 6.213 | 0.183 |
| 2 | 4.009 / **4.945** / 5.777 | 0.720 / 0.861 / 1.000 | 3.368 / 4.207 / 5.054 | 0.180 |
| 3 | 4.109 / **5.096** / 6.513 | 0.664 / 0.819 / 0.990 | 2.729 / 4.114 / 4.632 | 0.194 |

Layer 0's median `T` rises monotonically from the declared `5.000` and its
**per-mode spread grows from 0 to 2.65**:

```
epoch   -1     0      1      2      3      4      5      6      7      8      9
med T  5.000  5.214  5.234  5.389  5.443  5.549  5.535  5.551  5.566  5.572  5.574
spread 0.000  2.085  2.373  2.464  2.424  2.581  2.641  2.616  2.628  2.653  2.655
```

**The frozen arm's `T` stayed at exactly `5.0000` in every layer, with spread
`0.0000`, zero projection events and a mean update norm of exactly `0`.** The
freeze is therefore exact, including under decoupled weight decay, which is why
it was implemented by zeroing `eta`'s updates rather than by dropping its
gradient.

**The gradient reached `T`, and Adam moved it at the same rate as `rho`:**

| coordinate | mean \|grad\| | mean \|update\| | projection events | max proposed overshoot |
|---|---|---|---|---|
| `rho` (learned arm) | 6.73e-02 | 1.15e-03 | 2,423 of 539,520 (0.45 %) | 9.37e-04 |
| `T` (learned arm) | 8.15e-03 | 1.08e-03 | **0** of 539,520 (0.00 %) | 0 |
| `rho` (fixed arm) | 6.68e-02 | 1.14e-03 | 2,494 of 539,520 (0.46 %) | 9.56e-04 |
| `T` (fixed arm) | 7.84e-03 | **0** | 0 | 0 |

`T`'s gradient is about eight times smaller than `rho`'s, but Adam normalizes,
so the two coordinates moved per step at essentially the same size. **`T` never
touched its guardrails** — zero projection events and zero overshoot — so
`T ∈ [0.05, 500]` was never binding and the result is **not** an artifact of
those bounds. `rho`'s bound was active on under half a percent of
entry-updates, and no raw value of either coordinate finished outside its
interval in any of the eight leaves.

So this is the stronger reading: **the extra response freedom was genuinely
exercised, and bought nothing measurable.** The learning curves say the same
thing — the two arms differ by at most 0.001 in validation accuracy at any
epoch while their response timescales diverge by a factor of nearly two across
modes.

`rho` also learned in **both** arms, moving from the declared 0.75 to layer
medians of 0.76–0.86 with per-mode ranges spanning 0.65–1.00, so the shared
coordinate was active too and the comparison is not between a learning arm and
a static one.

**One mechanistic observation, offered as such and not as an explanation.** The
dimensionless per-mode product `|T j|` sits at 0.15–0.19 at the endpoint. With
`T` near 5 that puts `|j|` around 0.03, i.e. the stored modes are slow relative
to the response horizon, and the prospective zero at `-1/T` is far from the
band where it would dominate the response. Whether the timescale freedom would
matter in a regime where `|T j|` is of order one is **not** tested here.

## 4. Initialization

The two generalized arms were **identical at initialization**:
`max|logit difference| = 0.000e+00` on a label-free validation probe, identical
parameter digests, `T0 = 5.000000` and `rho0 = 0.750000` read back from the
executed modules. Their later difference is therefore attributable to `eta`
receiving updates and to nothing else.

The superseded 1 % function-match-to-ordinary gate does **not** apply to these
arms, which start at `rho = 0.75` and are meant to differ. Their signal-only
differences from the matched ordinary substrate were **recorded, not gated**:

| measure | layer 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| impulse relative | 0.349 | 0.538 | 0.396 | 0.597 |
| frequency relative | 0.277 | 0.426 | 0.362 | 0.488 |

i.e. a 28–60 % change in the initial signal response, by design.

The identifiability argument that motivated `rho_0 = 0.75` is **supported by
the outcome**: `dG/dT` vanishes at `rho = 1`, and at 0.75 the derivative was
large enough that `T` moved substantially within ten epochs. Had the near-one
recall initialization been reused, this test would have begun close to an
unidentifiable limit and the flat result would have been uninterpretable.

## 5. What this supports, and what it does not

**Supported.** The per-mode learned timescale is implemented correctly and
trains: the two arms start as the same function, the freeze is exact, the
gradient reaches `eta`, `T` moves and differentiates across modes without
touching its guardrails, and every derived quantity (`mu = rho T`, the circuit
component ties) is computed from the executed coefficients rather than from a
static field. All 58 focused checks passed, including the `rho = 1` limit where
the `T` derivative is asserted to be **zero**, a converging finite-difference
ladder for `d/d(log T)`, and a real production update that moves `log T` in the
learned arm and leaves it exactly fixed in the frozen one.

**Not demonstrated.** Any benefit from learning the response timescale. The
isolating comparison is `-0.035` pp; both generalized arms are behind the
matched ordinary substrate by about 0.3 pp and behind Rawat by about 0.7 pp,
and ahead only of `native_s5`, by 0.09 pp. Nothing here supports a
capacity-efficiency claim either: the generalized arms carry twice the state
(256 against 128 real coordinates) and 128 more parameters than the ordinary
controls.

## 6. Limitations

* **One seed, ten epochs, one task.** This is a development screen. A
  `-0.035` pp difference is far inside what a single seed can resolve, and the
  larger gaps to the controls are not established as robust either.
* **One learning rate, no development budget for any arm.** Equal at zero is
  equal, but it is not the same as each arm being well tuned.
* **One initialization.** `rho_0 = 0.75`, `T_0 = 5`. The result is conditioned
  on starting at the symmetric reference.
* **State is matched, parameters are not.** 256 against 128 carried real
  coordinates, 35,178 against 35,050 parameters. No efficiency claim follows.
* **`|T j| ≈ 0.15–0.19` at the endpoint**, so the response operated far from
  the regime where the prospective zero dominates. A different clock or task
  could place it elsewhere; that was not tested.
* The summary's boundary-occupancy listing labels all eight leaves `seq`, so
  `rho` and `T` leaves are not distinguished by name in that printout. The
  counts themselves are unambiguous — none at a bound, none outside, in all
  eight — and the per-coordinate telemetry above is separated correctly.

## 7. Amendments made during this batch, all before any score was read

Three execution attempts were needed, and every correction is recorded in the
protocol with its measured numbers.

1. **`53f9c9b`** — `TIMESCALE_STATUS=FAILED` at the checks. Three defects, two
   in the checks themselves: the production-dtype probes were silently running
   in float64 because they imported helpers from an x64-enabling module, and
   the diagnostics frequency check compared against a truncated DFT whose
   omitted tail was not small (measured discrepancy 0.37).
2. **`cce550b`** — all 58 checks passed; the preflight then **refused to
   train**, projecting 2,790 s against 856 s available. The refusal behaved as
   designed, but the projection was wrong: evaluation cost was measured with a
   single call that included the one-off compile and then scaled per sample,
   turning a ~4 s compile into a claimed 43.7 s *per epoch*.
3. **`d07380c`** — corrected accounting projected 337.9 s, the batch ran, and
   the study completed in 477 s.

A declared tolerance was amended once, at the stiff guardrail corner
`rho = 0.25, T = 500`, from `1e-10` to `1e-8`, justified by the measured
conditioning of that corner and not by any accuracy number. The ordinary range
kept `1e-10`, and the corners' finiteness requirement was not relaxed.

## 8. Disposition

The bounded screen is complete and its outcome is reported as it came out. No
sweep, no additional seeds and no enlargement follows from this batch. A later
proposal should name a specific regime — a task or clock where `|T j|` is of
order one would be the obvious candidate, given s3 — and a discriminating
prediction, before another training budget is spent on this coordinate.
