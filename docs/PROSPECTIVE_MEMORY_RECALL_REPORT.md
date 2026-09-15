# Memory-recall study: results

Protocol committed before execution:
`docs/PROSPECTIVE_MEMORY_RECALL_PROTOCOL.md`.

**Outcome: the generalized prospective recurrence retains useful recurrent
memory and trains with full BPTT, but it showed no recall advantage over
ordinary S5 or Rawat's prospective-input S5. An ordinary S5 with twice the
stored modes was ahead in every seed.**

**TWO RUNS ARE REPORTED.** The first, at `b5d7211`, was executed with a defect:
the raw response parameter was never projected back into its interval after an
optimizer update. The second, at `fb16614`, repairs exactly that and changes
nothing else. Both are preserved. **The corrected run is authoritative for the
learned-response arm**; every other arm reproduced **identical recorded
metrics** across the two runs, which is the control showing the repair touched
only what it should. See s3 for exactly which quantities were compared — the
first run saved no parameter trees, so this is not a claim of bit-identical
trajectories.

**The correction reversed the single result that had favoured the mechanism.**
In the defective run, learning `rho` beat freezing it in all three seeds
(+0.911 pp). With the projection repaired that comparison is mixed and slightly
negative (-0.309 pp). Those earlier figures are **results of the previous
unprojected optimization procedure**, not fabricated measurements; the strongest
supported statement is that their positive signs were **not robust to the
correction**.

> **CORRECTIONS APPLIED 15 September 2026**, in two rounds: after the
> coordinator's review of the executed commit, and after the coordinator's
> interpretation of the corrected run. Both runs, their artifacts and every
> number below are preserved unchanged. What is corrected is interpretation,
> coverage statements, claim scope, and one physics sign error. A **material
> implementation defect** found by the first review is recorded in s0; it does
> not invalidate the observed accuracies but it does remove one of the
> conclusions I drew.

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

**The crossing happens immediately from this initialization.** Measured while
repairing the regression: a single production update from the declared
`rho_0 = 0.9998` already pushed **5 of 16** entries in an untouched layer across
the upper bound. The headroom is `1.0e-4` in log space and an AdamW step is of
order `1e-3` — ten times larger — so outward-pointing entries cross on step one.
Under the defective procedure those entries would then have carried zero task
gradient for the rest of training. This makes the lockout hypothesis for the
completed run concrete rather than theoretical, and it is the main reason the
`rho` distribution from that run cannot be interpreted.

**What cannot be recovered.** The completed run saved only clipped summaries.
No warm-up or final parameter trees, optimizer states or raw trajectories were
written, so the raw overshoot of *this* run cannot be reconstructed after the
fact. That limitation is stated rather than worked around; the repaired runner
saves parameter trees and projection telemetry.

The repair is committed (post-update projection, preserving optimizer state,
with regression coverage) and the corrected run has since been executed; s3 and
s5 report it. Missing projection was **not** evidence that a repaired run would
beat the baselines, and in the event it did not.

**Limit on retrospective diagnosis.** Because no raw trajectories were saved,
this report does **not** identify which modes of the first run were stranded
outside the interval, and does not claim that any particular score difference
has been traced mechanistically to a specific mode. The lockout is established
as a mechanism by the source and by the first-update measurement above; its
incidence in the completed first run is unknown.

## 1. Execution identity

| item | value |
|---|---|
| checkout | `/Local/durso/final-prospective-s5` |
| branch | `adaptive-recurrence` |
| executed commit, run 1 (defective) | `b5d7211b2d631e104935a231aa41bbd00debf48d` |
| executed commit, run 2 (**corrected, authoritative**) | **`fb166146aa07f9b13bac44fcc5a7e3d75c09480e`** |
| host | `pgi15-gpu3.iff.kfa-juelich.de`, RTX 3090 |
| SLURM job | 65910, `CUDA_VISIBLE_DEVICES=0` |
| backend | `gpu`, `CudaDevice(id=0)`, jax 0.11.0 |
| artifacts, run 1 | `/Users/durso/s5-runs/recall/20260915-220252/` |
| artifacts, run 2 | `/Users/durso/s5-runs/recall/20260915-225516/` |
| logs, run 1 | `/Users/durso/s5-runs/recall/logs/20260915-215629/` |
| logs, run 2 | `/Users/durso/s5-runs/recall/logs/20260915-224757/` |
| focused checks | run 1: 25 passed; run 2: **38 passed**, 433 s |
| study | **18/18 rows, `complete=True`, no incomplete stages** in both |
| total | run 1: 739 s; run 2: **824 s of the 1200 s cap** |
| status | **`RECALL_STATUS=PASS`, `RECALL_EXIT=0`** |

Checkpoints and telemetry written by the corrected run (cluster-side; not
committed to this repository, per the artifact policy):

| item | path |
|---|---|
| warm-up and final parameter trees | `/Users/durso/s5-runs/recall/20260915-225516/params/` |
| per-seed, per-arm summary JSON | `/Users/durso/s5-runs/recall/20260915-225516/` |
| launcher and check logs | `/Users/durso/s5-runs/recall/logs/20260915-224757/` |
| first (defective) run, preserved | `/Users/durso/s5-runs/recall/20260915-220252/` |

The first run saved **no** parameter trees, optimizer states or raw response
trajectories; only the corrected run does. All tabulated scores below are the
values as recorded in those summary JSON files, at the precision the runner
emits.

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

**Corrected run: the gate ran before every seed and enforced both criteria.**

| seed | `gp_rho` impulse | `gp_rho` frequency | `rawat` impulse | `rawat` frequency |
|---|---|---|---|---|
| 100 | 1.99e-03 | 2.33e-03 | 2.09e+00 | 1.86e+00 |
| 101 | 3.03e-03 | 2.99e-03 | 4.88e+00 | 4.82e+00 |
| 102 | 2.55e-03 | 2.91e-03 | 2.67e+00 | 2.59e+00 |

All three seeds passed on both criteria. The frequency figures are comparable in
size to the impulse ones, so enforcing them was not cosmetic even though it did
not change the verdict here. `rawat`'s initial function change varies from
**2.09 to 4.88** across seeds — a far larger spread than seed 100 alone
suggested, and relevant to its seed-to-seed variability.

The table below is from the defective run, where the gate ran once.

**Coverage correction, for the defective run only.** Those probes were **seed
100 only** and used the **impulse** criterion alone; zero-reference layers were
filtered out rather than receiving the declared absolute criterion. For that run
the correct statement is that the generalized arms were *approximately matched
on the seed-100 impulse probe*, not function-matched across the study.

The corrected run fixes all of it: the gate runs before every seed, enforces
both criteria, fails on non-finite values, and applies the declared absolute
criterion to zero-reference layers. These remain finite-window and finite-grid
checks, not a uniform transfer-function theorem.

`rawat` starts about **209 %** away from the warm-up function. It is therefore
**not** function-matched, exactly as the protocol required be reported, and
that matters for reading its results: it is the largest perturbation applied at
the start of continuation, and it is also the arm whose seed-to-seed spread is
largest.

## 3. Primary result

Mean accuracy at the trained longer delays 32 and 64, per seed:

**Corrected run** (`fb16614`), mean accuracy at the trained longer delays:

| arm | seed 100 | seed 101 | seed 102 | mean |
|---|---|---|---|---|
| **`ordinary_2x`** | **0.9927** | **1.0000** | **0.9951** | **0.9959** |
| `rawat` | 0.9155 | 0.9971 | 0.9722 | 0.9616 |
| `ordinary` | 0.9414 | 0.9707 | 0.9585 | 0.9569 |
| `gp_rho_frozen` | 0.9238 | 0.9639 | 0.9585 | 0.9487 |
| `gp_rho` | 0.9336 | 0.9365 | 0.9668 | **0.9456** |
| `professor` | 0.1196 | 0.1177 | 0.1235 | 0.1203 |

Paired differences in percentage points, every seed shown, **defective run above
and corrected run below**:

| comparison | s100 | s101 | s102 | mean | sign |
|---|---|---|---|---|---|
| `gp_rho` - `ordinary` (defective) | +0.146 | +0.049 | +0.098 | +0.098 | all + |
| **`gp_rho` - `ordinary` (corrected)** | **-0.781** | **-3.418** | **+0.830** | **-1.123** | **mixed** |
| `gp_rho` - `rawat` (defective) | +2.734 | -2.588 | -1.270 | -0.374 | mixed |
| **`gp_rho` - `rawat` (corrected)** | **+1.807** | **-6.055** | **-0.537** | **-1.595** | **mixed** |
| `gp_rho` - `gp_rho_frozen` (defective) | +1.904 | +0.732 | +0.098 | +0.911 | **all +** |
| **`gp_rho` - `gp_rho_frozen` (corrected)** | **+0.977** | **-2.734** | **+0.830** | **-0.309** | **mixed** |
| `gp_rho` - `ordinary_2x` (defective) | -4.980 | -2.881 | -3.564 | -3.809 | all - |
| **`gp_rho` - `ordinary_2x` (corrected)** | **-5.908** | **-6.348** | **-2.832** | **-5.029** | **all -** |

**The reversal is the headline.** The frozen-versus-learned comparison was the
only one consistent in sign across all three seeds, and the only evidence that
allowing the response to learn helped at all. With the projection repaired it is
mixed and slightly negative. `gp_rho` itself fell from 0.9579 to 0.9456.

Every other arm produced **identical recorded metrics** between the two runs:
`ordinary`, `rawat`, `professor`, `gp_rho_frozen` and `ordinary_2x` reproduce
their per-seed and per-delay accuracies, their intervention figures and their
parameter counts exactly. That is the comparison actually made. It is **not** a
claim that the runs were bit-identical in their trajectories or parameter trees:
the first run saved no trees, so no such comparison is possible. Within that
scope, the change in `gp_rho` is attributable to the repair rather than to
run-to-run variation.

**No accuracy threshold was predeclared for this study.** The `+0.3` pp figure
used in the Speech Commands screens appears in neither the recall brief nor the
committed recall protocol; an earlier version of this report presented it as a
criterion here, which was an import from a different study and is withdrawn.

The findings stand without it. In the **defective** run the difference against
`ordinary` was consistently positive but averaged **0.098 pp** — **3, 1 and 2
additional correct examples out of 2,048** in seeds 100, 101 and 102 — which did
not establish a practical advantage even then. In the **corrected** run that
comparison is mixed in sign and averages **-1.123 pp**. Against `rawat` the sign
flips across seeds in both runs, and on average the generalized arm is
**behind** it: 0.374 pp defective, 1.595 pp corrected.

By delay, mean over seeds (96 is held out, never trained or selected on):

| arm | d8 | d32 | d64 | d96 (held out) |
|---|---|---|---|---|
| `ordinary_2x` | 0.992 | 0.995 | 0.997 | 0.371 |
| `rawat` | 0.961 | 0.969 | 0.954 | 0.346 |
| `gp_rho` | 0.972 | 0.972 | 0.943 | 0.293 |
| `ordinary` | 0.971 | 0.974 | 0.940 | 0.302 |
| `gp_rho_frozen` | 0.970 | 0.965 | 0.933 | 0.294 |
| `professor` | 0.127 | 0.115 | 0.125 | 0.122 |

Every memory-bearing arm degrades sharply at the held-out delay, to 0.29-0.37
against a 0.125 chance level. Those figures are **above chance** — the arms
retain some transfer to the longer delay — but they are far below the
trained-delay accuracies, and the generalized arm gains nothing here either:
0.293 against `ordinary` 0.302, `rawat` 0.346 and `ordinary_2x` 0.371.
`ordinary_2x` degrades least. Nothing here supports a retention/responsiveness
tradeoff claim at delay 8: the memory-bearing arms are within 2 pp of each other
there.

## 4. The control that dominates

**`ordinary_2x` is ahead of `gp_rho` in every seed** — by 5.908, 6.348 and
2.832 pp in seeds 100, 101 and 102, a mean of 5.029 pp — and it does so at
**equal total recurrent carry**: `gp_rho` carries 64 physical + 64 auxiliary =
**128 real coordinates**, `ordinary_2x` carries 128 physical + 0 = **128 real
coordinates**. It is also no slower in wall time (10.1 s vs 15.3 s per
continuation).

It is **not** matched in parameters: 11,368 against 7,208 stored/trainable
values for the generalized arm, i.e. **+4,160 (+58 %)**. That extra parameter
freedom is **part of** this comparison, not a reason to discount it: its
practical advantage in this experiment is real, and it is best described as **an
effective larger ordinary model under this budget** rather than an isolation of
recurrent state independent of parameter count. At equal carry and greater
parameter count, plain extra ordinary modes bought substantially more recall on
this task than the derived response did. The simplest capacity explanation is
not ruled out here.

Reported timings are for this script and configuration; they are not universal
runtime claims.

## 5. What learning `rho` actually did — the corrected picture

**Defective run.** Median `rho` 0.9999, the clip value; 4-5 of 32 modes fell.

**Corrected run.** The distribution inverts:

| seed | final min | final median | modes that FELL | at the cap | raw outside |
|---|---|---|---|---|---|
| 100 | 0.9438 | **0.9806** | **29 / 32** | 3 | 0 |
| 101 | 0.9333 | **0.9834** | **28 / 32** | 3 | 0 |
| 102 | 0.8224 | **0.9986** | **23 / 32** | 8 | 0 |

With the projection in place **most modes fall** and only 3-8 sit at the
numerical margin, against 4-5 falling and a median pinned at the margin before.
Median `mu = T rho` is 4.90 / 4.92 / 4.99 against 4.999 at initialization: the
derived mass moves **down**, i.e. *away* from the ordinary-SSM limit.

**The bound was active throughout, and that is why it mattered.** Projection
telemetry, over 1,024 updates x 32 entries = 32,768 entry-updates per seed:

| seed | projection events | share | max raw overshoot | mean \|g_rho\| | mean \|u_rho\| |
|---|---|---|---|---|---|
| 100 | 5,852 | 17.9 % | 1.05e-3 | 1.62 | 1.11e-3 |
| 101 | 6,424 | 19.6 % | 1.00e-3 | 2.13 | 1.12e-3 |
| 102 | 11,111 | 33.9 % | 1.26e-3 | 1.53 | 1.44e-3 |

Between a fifth and a third of all entry-updates required projection.

Two readings of this table have to be excluded. **The event totals count
coordinate/update events** — one per (entry, step) pair that needed projecting —
**not** distinct neurons, distinct steps, or continuous time spent at the
boundary; 5,852 events does not mean 5,852 modes or 5,852 steps at the bound.
And **the maximum overshoot is a proposed, pre-projection value**: it is the size
of the update the optimizer offered before the projection rejected it, not
evidence that the forward model ever used an inadmissible coefficient. It is
about `1e-3`, the AdamW step size, exactly as the headroom argument predicts.

`gp_rho_frozen` recorded **zero** events, as it must. No raw value finished
outside the interval in any seed. Together these observations support that
response learning was **active** and that the specific out-of-bounds gradient
lockout was addressed. They do not establish that the tuning was optimal or that
every mode finishing at the bound belongs there.

**A claim of mine is withdrawn.** The earlier report said the learned direction
was *"overwhelmingly toward the ordinary-SSM limit"*. Under the corrected
optimization the direction is the **opposite**: `rho` predominantly falls. The
pile-up at the margin in the first run is consistent with the lockout mechanism
established in s0, but no per-mode diagnosis is offered — without the raw
trajectories it is not possible to say which of that run's modes were stranded,
and none is claimed.

The corrected direction comes without benefit: `rho` genuinely moves away from
the ordinary limit, and the arm is **not** better for it — mixed against its own
frozen control, mixed against `ordinary`, and behind `ordinary_2x` in all three
seeds by **5.908 / 6.348 / 2.832 pp**, a mean of 5.029 pp.

**What these coefficients are, and are not.** The learned `rho` values are
**constant within each sequence**. They shape the temporal response of each mode;
they do **not** implement an online controller that identifies a cue and decides
whether to remember it. Learned importance or selectivity would be a task-level
claim requiring its own measurement — s7 reports that measurement, and it does
not separate the arms — and it is not conferred by what the coefficients are
named.

Effective clocks, now printed for **all three seeds**, are essentially identical
across `ordinary`, `gp_rho` and `gp_rho_frozen` (medians 0.0071 / 0.0136 /
0.0217 by seed, matching to three significant figures), so the arms did not
diverge by silently retiming.

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

* **The defective run's `rho` conclusions are void**, and its raw trajectory
  cannot be recovered from its artifacts. The corrected run supersedes it for
  the learned-response arm.
* **The bound remains active in the corrected run** — 18-34 % of entry-updates
  required projection — so `rho` is learning against a constraint, not freely.
  The constraint is now a projection rather than a trap.
* **Three seeds**, and the corrected differences are mixed in sign with a spread
  of several points, so the study does not resolve small effects.
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
* **The `rho` bound was active throughout the corrected run** — 17.9 / 19.6 /
  33.9 % of entry-updates were projected, and 3 / 3 / 8 of 32 modes finished at
  the upper margin. Conclusions about what learning `rho` does are conditioned
  on optimizing against that constraint.
* **`ordinary_2x` matches carry, not parameters** (+58 %).
* One task, one size, one budget. No claim is made that the particular physical
  constraints outperform more general filters; that would need a matched
  response-parameterization control, which this batch did not include.
* The held-out delay 96 was never trained or selected on. All arms score well
  above the 0.125 chance level there (0.29-0.37) but far below their
  trained-delay accuracies, and the ordering matches the trained delays, so it
  shows no extrapolation advantage for the generalized recurrence.
* **Scope.** This is a three-seed bounded experiment on one task, one size, one
  schedule. It is evidence against an advantage for **this particular diagonal
  response family, this initialization, this task and this training schedule**.
  It does **not** refute the mathematical circuit reduction, and it does not
  show that all generalized prospective constructions must fail. Those scopes
  are kept distinct throughout.

## 10. Conclusion

On a task that genuinely requires recall — confirmed by the memoryless control
scoring at chance — the physically derived generalized prospective recurrence
**did not** improve recall over ordinary S5 or Rawat's prospective-input S5. In
the corrected run it is **behind ordinary S5 by 1.12 pp** and **behind Rawat by
1.60 pp** on average, mixed in sign in both cases.

The most informative comparison is the one that did not involve the mechanism
at all: **doubling the ordinary recurrent state was ahead of the generalized
recurrence in every seed** — 5.908 / 6.348 / 2.832 pp, mean 5.029 — at equal
total carry and with 58 % more parameters. Whatever this task rewards, extra
ordinary modes supplied it more effectively than the derived response did.

Allowing `rho` to learn helped in all three seeds **only in the defective run**.
With the optimization repaired the comparison is mixed and slightly negative, so
the study contains no consistent evidence that response learning helps here.

The repair also settled what the defective run could not. `rho` does not drift
toward the ordinary-SSM limit; with projection it predominantly **falls**, in
23-29 of 32 modes per seed, lowering the derived mass. The pile-up at the margin
in the first run is consistent with the lockout mechanism, which is demonstrated
to occur from this initialization on the first update; which modes it caught in
that run cannot be recovered and is not asserted.

**The corrected rerun has been executed, and it did not help.** That was the
honest test of the repair: fixing a defect that could only have suppressed the
mechanism's measured performance in fact *removed* its one apparent advantage.
The earlier positive signs were produced by the previous unprojected
optimization procedure and did not survive the correction. The bounded
correction study is therefore complete, and its outcome is reported as it came
out.

**Supported by this study.** The derived generalized recurrence **can retain
useful recurrent memory**, and **it trains with full BPTT** — response learning
was demonstrably active under the corrected constrained optimization. The
professor recurrence, in its fully matched memoryless driven realization,
performs **near the eight-class chance expectation** on this query-only task.

**Not demonstrated.** Better recall than ordinary S5 or Rawat's
prospective-input S5; better delay extrapolation; more useful cue/distractor
selection — the intervention measurements remain similar to ordinary S5. The old
positive signs cannot be carried forward as evidence for the corrected
implementation.

What the two runs together establish, with three seeds on one task at one size:

* allowing the constrained response to learn does **not** consistently help,
  and the single positive result that suggested otherwise came from the
  previous unprojected optimization procedure and did not survive the repair;
* when `rho` can actually move it moves **away** from the ordinary-SSM limit,
  and the model is not better for it;
* the strongest comparator is **not** a prospective mechanism at all but
  **twice the ordinary recurrent state**, ahead in every seed at equal total
  carry — by 5.908, 6.348 and 2.832 pp, a **mean** of 5.029 pp, with 58 % more
  parameters;
* the fully matched prospective recurrence is at **chance**, which is what
  makes this task a usable recall probe and what the pooled speech task could
  not show.

The remaining question this supports is whether any response-shaping in this
family can compete with simply adding ordinary state, which would need the
matched response-parameterization control this batch deliberately omitted. **No
larger run and no rescue sweep follows**, and this bounded experiment is closed
here: any later proposal should name a specific mathematical or task-level
mechanism and a discriminating prediction before further training, because
nothing in this result promises that a different task, horizon, learning rate or
added controller would produce an advantage. The earlier Speech Commands studies
and their verdicts are unchanged.

**Scope of the negative result.** This is evidence against an advantage for
**this** diagonal response family, **this** initialization, **this** task and
**this** training schedule, at three seeds. It does **not** refute the
mathematical circuit reduction from the generalized prospective law to the
diagonal recurrence, and it does **not** show that every generalized prospective
construction must fail. Nor do the learned coefficients amount to a memory
controller: they are constant within a sequence and shape temporal responses
across modes, rather than identifying a cue and deciding whether to retain it.
