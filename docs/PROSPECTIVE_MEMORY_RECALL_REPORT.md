# Memory-recall study: results

Protocol committed before execution:
`docs/PROSPECTIVE_MEMORY_RECALL_PROTOCOL.md`.

**Outcome: the generalized prospective recurrence did not improve recall over
the literature mechanisms, and an ordinary S5 with twice the stored modes beat
it in every seed.**

> **CORRECTIONS APPLIED 15 September 2026** after the coordinator's review of
> the executed commit. The completed run, its artifacts and every number below
> are preserved unchanged. What is corrected is interpretation, coverage
> statements, and one physics sign error. A **material implementation defect**
> found by that review is recorded in s0; it does not invalidate the observed
> accuracies but it does remove one of the conclusions I drew.

## 0. A defect in the executed optimization, and what it costs the conclusions

**The raw response leaf was never projected back into its interval after an
optimizer update.** `rho_only()` clips in the FORWARD pass, which keeps the
executed law admissible, but the raw parameter `eta` was free to leave the
interval. Above the upper bound `u = log(0.9999)`,

```
rho = exp(u) ,   d rho / d eta = 0 ,   dL_data / d eta = 0
```

so a leaf that crosses outward has **zero task gradient** and nothing brings it
back: AdamW's decoupled decay shrinks `eta` toward zero, which is *away* from a
negative upper bound. The earlier `rawat_benchmark.py` did project; this runner
did not, and the older helper would not have recognized the new leaf name in any
case.

**Consequence for the conclusions.** The observed median `rho = 0.9999` is
compatible with a genuine boundary optimum, with clipping lockout, or with a
mixture. My earlier conclusion that *"the learned direction was overwhelmingly
toward the ordinary-SSM limit"* is therefore **withdrawn**: the data cannot
distinguish a preference from a lockout.

The accuracies, controls, professor result and interventions **remain on
record** as measurements of the optimization procedure that was actually run.
The forward law stayed in bounds throughout, so they are valid measurements of
*that* procedure — but the defect's causal effect on accuracy is **unknown**,
and an earlier phrasing that they were "unaffected" claimed more than that.

**What cannot be recovered.** The completed run saved only clipped summaries.
No warm-up or final parameter trees, optimizer states or raw trajectories were
written, so the raw overshoot of *this* run cannot be reconstructed after the
fact. That limitation is stated rather than worked around; the repaired runner
saves parameter trees and projection telemetry.

The repair is committed (post-update projection, preserving optimizer state,
with regression coverage), but **no corrected run has been executed**. Missing
projection is **not** evidence that a repaired run would beat the baselines.

## 1. Execution identity

| item | value |
|---|---|
| checkout | `/Local/durso/final-prospective-s5` |
| branch | `adaptive-recurrence` |
| executed commit | **`b5d7211b2d631e104935a231aa41bbd00debf48d`** |
| host | `pgi15-gpu3.iff.kfa-juelich.de`, RTX 3090 |
| SLURM job | 65910, `CUDA_VISIBLE_DEVICES=0` |
| backend | `gpu`, `CudaDevice(id=0)`, jax 0.11.0 |
| artifacts | `/Users/durso/s5-runs/recall/20260915-220252/` |
| logs | `/Users/durso/s5-runs/recall/logs/20260915-215629/` |
| focused checks | **25 passed**, 377 s |
| study | **18/18 rows, `complete=True`, no incomplete stages**, 354 s |
| total | **739 s of the 1200 s cap** |
| status | **`RECALL_STATUS=PASS`, `RECALL_EXIT=0`** |

Command:

```
bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_recall.sh
```

Seeds 100, 101, 102; 1,024 warm-up updates on `ordinary`, then 1,024
continuation updates per arm on identical ordered minibatches with a fresh
optimizer; batch 32, lr 1e-3, full BPTT. Task structure verified at run time:
query marker only at the end, query carries no symbol, exactly two cues, label
balance deviation 0.148.

## 2. Initialization gate

Signal-only core impulse change from the warm-up model, native `D` removed,
relative Frobenius over lags 0..127, measured on the executed modules:

| arm | worst core impulse rel | query-logit rel | scope of the claim |
|---|---|---|---|
| `gp_rho` | **1.99e-03** | 1.17e-03 | approximately matched **on the seed-100 impulse probe** |
| `gp_rho_frozen` | **1.99e-03** | 1.17e-03 | approximately matched **on the seed-100 impulse probe** |
| `rawat` | **2.09e+00** | 1.48e+00 | **not matched** |
| `professor` | **7.57e+00** | 1.18e+00 | **not matched** |

No cell here says "function-matched": the executed gate probed one seed with one
of the two declared criteria, so none of these is a verified match across the
study.

Gate **PASSED** for the generalized arms at 1.99e-03 against the predeclared
1e-2.

**Coverage correction.** These probes are **seed 100 only**: the executed gate
ran once and was then skipped, although each seed's warm-up has different
learned poles and readouts. The decision also used the **impulse** criterion
alone — the frequency differences were computed and saved but not enforced —
and zero-reference layers were filtered out rather than receiving the declared
absolute criterion. So the correct statement is that the generalized arms are
**approximately matched on the reported seed-100 probes**, not exactly
function-matched across the study. The saved frequency entries may be reported,
but nothing should be inferred about them from the overall PASS. These are
finite-window and finite-grid checks, not a uniform transfer-function theorem.

The repaired runner runs the gate before every seed, enforces both criteria,
fails on non-finite values, and implements the zero-reference branch.

`rawat` starts about **209 %** away from the warm-up function. It is therefore
**not** function-matched, exactly as the protocol required be reported, and
that matters for reading its results: it is the largest perturbation applied at
the start of continuation, and it is also the arm whose seed-to-seed spread is
largest.

## 3. Primary result

Mean accuracy at the trained longer delays 32 and 64, per seed:

| arm | seed 100 | seed 101 | seed 102 | mean |
|---|---|---|---|---|
| **`ordinary_2x`** | **0.9927** | **1.0000** | **0.9951** | **0.9959** |
| `rawat` | 0.9155 | 0.9971 | 0.9722 | 0.9616 |
| `gp_rho` | 0.9429 | 0.9712 | 0.9595 | 0.9578 |
| `ordinary` | 0.9414 | 0.9707 | 0.9585 | 0.9569 |
| `gp_rho_frozen` | 0.9238 | 0.9639 | 0.9585 | 0.9487 |
| `professor` | 0.1196 | 0.1177 | 0.1235 | 0.1203 |

Paired differences in percentage points, every seed shown:

| comparison | s100 | s101 | s102 | mean | sign |
|---|---|---|---|---|---|
| `gp_rho` - `ordinary` | +0.146 | +0.049 | +0.098 | **+0.098** | all + |
| `gp_rho` - `rawat` | +2.734 | -2.588 | -1.270 | -0.374 | **mixed** |
| `gp_rho` - `gp_rho_frozen` | +1.904 | +0.732 | +0.098 | **+0.911** | all + |
| `gp_rho` - `ordinary_2x` | -4.980 | -2.881 | -3.564 | **-3.809** | all - |

**No accuracy threshold was predeclared for this study.** The `+0.3` pp figure
used in the Speech Commands screens appears in neither the recall brief nor the
committed recall protocol; an earlier version of this report presented it as a
criterion here, which was an import from a different study and is withdrawn.

The findings stand without it. Against `ordinary` the difference is
consistently positive but averages **0.098 pp** — **3, 1 and 2 additional
correct examples out of 2,048** in seeds 100, 101 and 102 — which does not
establish a practical advantage. Against `rawat` the sign flips across seeds,
and on average the generalized arm is **behind** it by 0.374 pp.

By delay, mean over seeds (96 is held out, never trained or selected on):

| arm | d8 | d32 | d64 | d96 (held out) |
|---|---|---|---|---|
| `ordinary_2x` | 0.992 | 0.995 | 0.997 | 0.371 |
| `rawat` | 0.961 | 0.969 | 0.954 | 0.346 |
| `gp_rho` | 0.972 | 0.972 | 0.943 | 0.293 |
| `ordinary` | 0.971 | 0.974 | 0.940 | 0.302 |
| `gp_rho_frozen` | 0.970 | 0.965 | 0.933 | 0.294 |
| `professor` | 0.127 | 0.115 | 0.125 | 0.122 |

No arm extrapolates to delay 96; every one collapses to 0.29-0.37 against a
0.125 chance level. `ordinary_2x` degrades least. Nothing here supports a
retention/responsiveness tradeoff claim at delay 8: the memory-bearing arms are
within 2 pp of each other there.

## 4. The control that dominates

**`ordinary_2x` beats `gp_rho` by 2.9-5.0 pp in every seed**, and does it at
**equal total recurrent carry**: `gp_rho` carries 64 physical + 64 auxiliary =
**128 real coordinates**, `ordinary_2x` carries 128 physical + 0 = **128 real
coordinates**. It is also no slower in wall time (10.1 s vs 15.3 s per
continuation).

It is **not** matched in parameters: 11,368 against 7,208, i.e. **+4,160
(+58 %)**. It is therefore best described as **an effective larger ordinary
model under this budget**, not an isolation of recurrent state independent of
parameter count. At equal carry and greater parameter count, plain extra
ordinary modes bought substantially more recall on this task than the derived
response did. The simplest capacity explanation is not ruled out here.

Reported timings are for this script and configuration; they are not universal
runtime claims.

## 5. What learning `rho` actually did — and a correction to my own protocol

| seed | `rho` init | final min | final median | modes that FELL |
|---|---|---|---|---|
| 100 | 0.9998 | 0.9497 | **0.9999** | 4 / 32 |
| 101 | 0.9998 | 0.9323 | **0.9999** | 5 / 32 |
| 102 | 0.9998 | 0.8212 | **0.9999** | 4 / 32 |

**The median final `rho` is 0.9999, the clip value.** Since the executed `rho`
cannot exceed it, a median at the clip means **at least 16 of 32 modes sit at
it**. Only 4-5 modes per seed fell below the initialization, though those fell
substantially (to 0.82-0.95).

**The exact bound occupancy was not recorded for this run**, so the count at the
margin is bounded (at least 16, at most 28) rather than known. An earlier
version of this report stated "27-28 of 32 modes" — that was inferred from the
median plus `n_fell` and is withdrawn. The repaired runner records occupancy
directly.

**Two statements I put in the protocol before the run were wrong.**

1. I wrote that `rho` *"can effectively only fall"*. False: there was 1.0e-4 of
   headroom in log space and the optimizer used it. `rho` can rise as well as
   fall, and here it mostly rose.
2. I wrote that decreasing `rho` *"adds mass"*. **Reversed.** With `gamma_n = 1`
   the mass is `mu = T rho`, so decreasing `rho` **decreases** the mass. What
   decreasing `rho` does is break the `rho = 1` pole-zero cancellation and
   change the observable contribution of the auxiliary dynamics — not the same
   thing as increasing memory importance.

Also: `rho = 1` is the **physical** family boundary; `0.9999` is a **chosen
numerical margin** below it.

`rho -> 1` is the ordinary memory-bearing SSM limit (the transfer reduces to
`b/(p+j)` with a rescaled clock). The derived mass barely moved in the median:
`mu = T rho` went from 4.9990 to 4.9995; the minority of modes that fell reached
`mu` of 4.11-4.75.

**But the direction cannot be interpreted**, for the reason in s0: without
post-update projection, a raw leaf that crossed the upper bound had zero task
gradient and could not return. A median at the bound is equally consistent with
a boundary optimum and with lockout, and this run saved no raw trajectory that
could separate them.

The `gp_rho` vs `gp_rho_frozen` gap of **+0.911 pp, positive in all three
seeds**, is the one comparison that favours the mechanism. The most it supports
is that *allowing* `rho` to move helped relative to holding it at 0.9998. It is
**not** evidence that a distinctly generalized response is what helped, and with
three seeds, most modes at a bound, and the projection defect of s0 unresolved
in this run, it is not a population claim.

`gp_rho_frozen` scored **0.82 pp below `ordinary`** on average despite being
function-matched to 2e-3 at initialization. Two arms that start within 0.2 % of
each other diverged by more than the effect being screened for, which is a
useful caution about the resolution of this setup.

Effective clocks were essentially identical across `ordinary`, `gp_rho` and
`gp_rho_frozen` **for seed 100**, the only seed whose clocks were printed
(median 0.0071, range 0.0012-0.105). That is not generalized to the other seeds
here; the repaired summary prints all of them.

## 6. The professor control, and what it says about the earlier speech result

`professor` implements `r + T r' = 0` as `s_k = J^-1 b x_k`, formed directly
from `-B_c/lambda`. Verified: zero driven history at nonzero lag, invariance to
the earlier cue, and a **structurally zero** data-loss gradient with respect to
`log_step`.

It scored **0.1203**, against a chance level of **0.125**. In the paired
interventions its logit change is **exactly 0.000** for both a changed cue and a
changed distractor: it cannot see either, by construction.

This is the designed outcome and confirms the task is a genuine recall probe.
It also puts the earlier Speech Commands number in context: the same mechanism
scored **84.85 %** on the mean-pooled speech classifier and **chance** here. The
speech figure was therefore not informative about memory, which is exactly why
the corrected constrained-response report withdrew the claim that it bounded a
memory contribution.

## 7. Causal interventions

Mean over seeds, on one fixed held-out probe batch:

| arm | base acc | acc after changing the cue | \|d logit\| cue | \|d logit\| distractor | ratio |
|---|---|---|---|---|---|
| `ordinary_2x` | 1.000 | 1.000 | 29.629 | 0.276 | 107 : 1 |
| `gp_rho` | 0.990 | 0.995 | 26.705 | 0.257 | 104 : 1 |
| `ordinary` | 0.990 | 0.990 | 26.809 | 0.255 | 105 : 1 |
| `gp_rho_frozen` | 0.984 | 0.984 | 26.704 | 0.256 | 105 : 1 |
| `rawat` | 0.958 | 0.964 | 24.212 | 0.195 | 124 : 1 |
| `professor` | 0.146 | 0.099 | 0.000 | 0.000 | — |

Every memory-bearing arm is strongly and comparably selective: changing the
relevant marked cue moves the logits about 100x more than changing an unmarked
distractor, and the prediction follows the new cue symbol. **Both interventions
are reported, and they do not distinguish the arms.** These figures establish
useful cue sensitivity in ordinary and generalized S5 alike; they do **not**
show improved selection by the generalized model.

## 8. Costs

| arm | params | physical carry | auxiliary | input buffer | s per continuation |
|---|---|---|---|---|---|
| `ordinary` | 7,176 | 64 | 0 | 0 | 10.8 |
| `rawat` | 7,176 | 64 | 0 | 64 | 9.7 |
| `professor` | 7,176 | **0** | 0 | 0 | 5.3 |
| `gp_rho` | 7,208 | 64 | **64** | 0 | 15.3 |
| `gp_rho_frozen` | 7,208 stored / **7,176 trainable** | 64 | 64 | 0 | 14.9 |
| `ordinary_2x` | 11,368 | **128** | 0 | 0 | 10.1 |

`gp_rho` adds 32 parameters (one `rho` per stored mode per layer) and doubles
the carry, and is the slowest memory-bearing arm at 1.4x `ordinary` in this
configuration.

**Stored values are not trainable degrees of freedom.** `gp_rho_frozen` stores
7,208 values but its 32 `rho` entries are excluded from learning, so it has
7,176 trainable parameters — the same as `ordinary`. The repaired runner reports
both counts separately.

## 9. Limitations

* **The post-update projection was missing in this run (s0).** Conclusions
  about what learning `rho` preferred are not supported by these artifacts, and
  the raw trajectory cannot be recovered from them.
* **The initialization gate covered seed 100 only**, and enforced the impulse
  criterion alone. The arms are approximately matched on those probes, not
  verified as matched across the study.
* **The realized training-delay mix is 37.5 / 31.25 / 31.25 per cent** at
  delays 8 / 32 / 64, not equal thirds: the generator cycles delays within each
  batch. Target classes are balanced in expectation, not by construction. All
  arms shared the same data, so the comparisons are unaffected; the generator
  was **not** regenerated, since that would be a separate declared change.
* **Three seeds.** They do not establish population variance. The
  frozen-vs-learned gap is consistent in sign across three seeds; that is a
  weak form of evidence, not a significance test.
* **Paired continuation from a shared warm-up.** This does not answer which
  model trains best from scratch, and `rawat` is a transplanted mechanism, not
  a reproduction of its published benchmark.
* **The `rho` bound was active for ~85 % of modes.** Conclusions about what
  learning `rho` does are conditioned on that clipping.
* **`ordinary_2x` matches carry, not parameters** (+58 %).
* One task, one size, one budget. No claim is made that the particular physical
  constraints outperform more general filters; that would need a matched
  response-parameterization control, which this batch did not include.
* The held-out delay 96 was never trained or selected on; no arm succeeded
  there, so it discriminates nothing here beyond showing all arms fail alike.

## 10. Conclusion

On a task that genuinely requires recall — confirmed by the memoryless control
scoring at chance — the physically derived generalized prospective recurrence
**did not** improve recall over ordinary S5 or Rawat's prospective-input S5. Its
edge over ordinary S5 was +0.098 pp, i.e. 3, 1 and 2 additional correct examples
out of 2,048; against Rawat it was inconsistent in sign and behind on average.

The most informative comparison is the one that did not involve the mechanism
at all: **doubling the ordinary recurrent state gained 3.8 pp over the
generalized recurrence in every seed, at equal total carry**. Whatever this task
rewards, extra ordinary modes supplied it more effectively than the derived
response did.

Allowing `rho` to learn helped relative to freezing it, in all three seeds. What
that movement *meant* cannot be read from this run: the missing post-update
projection (s0) leaves a median at the bound equally consistent with a boundary
optimum and with a clipping lockout. The result does not support the derived
response being the active ingredient, and it does not rule it out either.

The supported next steps are, in order: (1) a **bounded corrected rerun** on the
same data streams and equations with the projection repaired and the gate
covering every seed, whose outcome is to be reported regardless of whether
projection helps; and only then (2) the question of whether any response-shaping
in this family can compete with simply adding ordinary state, which would need
the matched response-parameterization control this batch deliberately omitted.

**Missing projection is not evidence that a repaired run will beat the
baselines.** No larger run and no rescue sweep follows from this result. The
earlier Speech Commands studies and their verdicts are unchanged.
