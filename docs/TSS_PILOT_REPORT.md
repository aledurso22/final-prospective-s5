# TSS-based prospective pilot: status and Part B results

Protocol: `docs/TSS_PILOT_PROTOCOL.md`. Branch `tss-pilot`.

> **BOTH PARTS ARE NOW COMPLETE.**
>
> **Part A: every arm finished at the 12.5 % chance level on recall.** That
> triggers the outcome declared in the protocol before execution: recall is
> **inconclusive** and only the tracking comparison is interpretable. On
> tracking the memoryless control wins by about twenty-fold, and our retained
> compartment is the worst or joint-worst arm.
>
> **Part B: one of six configurations is usable, and it is the GLE-inspired
> baseline, not our preferred Section 11 filter.**
>
> Neither half supports the mechanism this pilot was built to test. Both
> outcomes were declared as possibilities in advance and are reported as they
> came out.

## 1. Execution status

| item | value |
|---|---|
| host | `pgi15-gpu3.iff.kfa-juelich.de`, RTX 3090, SLURM 65910 |
| backend | `gpu`, `CudaDevice(id=0)`, jax 0.11.0 |
| focused checks | **50 passed on GPU** at `6000b26`, 173 s |
| **Part B** | **COMPLETE**, 9/9 rows, 33 s, `TSS_B_STATUS=COMPLETE`, at `6000b26` |
| **Part A** | **COMPLETE**, 12/12 rows, 113 s, `TSS_A_STATUS=PASS`, at `d7dd53e` |
| Part A artifacts | `/Users/durso/s5-runs/tss_pilot/20260916-120620/part_a/` |
| Part A logs | `/Users/durso/s5-runs/tss_pilot/logs/20260916-120620/` |
| Part B artifacts | `/Users/durso/s5-runs/tss_pilot/20260916-020800/part_b/` |
| Part A total | 292 s of the 600 s cap, SLURM 66010 |

### Dispatch log

| # | commit | outcome | cause |
|---|---|---|---|
| 1 | `163dd8b` | FAILED at checks | implementation defect: `vmap` arity, complex scan-carry dtype |
| 2 | `358db41` | FAILED at checks | finite-difference probe at its own resolution, not the reference |
| 3 | `e2432a9` | Part A refused; **Part B complete** | memory arm reported 1459 ms/update |
| 4 | `e60522d` | Part A refused; Part B complete | vectorized fixed point — real for the ideal arm (94 → 2.83 ms), not the cause here |
| 5 | `17f009e` | Part A refused; Part B complete | real block associative scan — correct, not the cause |
| 6 | `bc11b5d` | Part A refused; Part B complete | batch-native model — correct, not the cause |
| 7 | `c7d7bf2` | Part A refused; Part B complete | stage-by-stage instrumentation only |
| 8 | `6000b26` | Part A refused; **Part B complete** | **cause found**: a weakly typed leaf retraced the step. 2560.91 → **3.00 ms**. Then refused on budget overhead: 128.5 s needed against 107 s left |
| 9 | `d7dd53e` | **Part A COMPLETE** | Part-A-only continuation; projection 188.5 s against 393 s available |

Dispatches 4-6 were three consecutive incorrect diagnoses of one measurement.
The evidence that settled it — `step x 3 == compile_s`, one recompilation
amortized over a three-call timing loop — was present in dispatch 3's output
and in every dispatch after it. The three changes made in between are retained
because each is correct and separately tested, but none of them was the
reported cost, and the protocol's amendments say so.

## 2. Part B — COMPLETE: credit assignment on a frozen forward model

Two layers of four retained-compartment cells, spatially feedforward,
32 trajectories x 64 steps, parameters frozen, seeds 100/101/102, three
predeclared input bands, three predeclared `eps`. **Reproduced identically on
five consecutive dispatches.**

### The reference passed its predeclared gates

| route | measured | tolerance |
|---|---|---|
| drive-adjoint identity, `G_W = sum_t rho_t r_t^T` | `2.32e-16` | `1e-9` |
| forward-mode against reverse-mode | `5.55e-17` | `1e-9` |
| central differences, best over the declared step ladder | within `1e-5` | `1e-5` |

So the object every approximation is measured against is sound, by three
independent routes, on the data actually used.

### The audit, before any comparison

* circuit inequality `M <= gamma T`: **holds**;
* the Section 7 witness reproduces exactly — forward pole `-0.090909`,
  GLE-inspired error-loop pole **`+0.125000`**: a stable forward node with an
  unstable error loop. Part B is spatially feedforward for this reason;
* every executed filter's states are **stable** at every declared `eps`;
* peak-gain bounds `46.4 / 17.2 / 8.3` at `eps = 0.25 / 0.5 / 1.0 x t_-`.

### Result

| filter | usable | mean-cosine range | mean relative-error range |
|---|---|---|---|
| moment-matched, `eps = 0.25 t_-` | **False** | [−0.419, 0.944] | [0.326, 1392] |
| moment-matched, `eps = 0.5 t_-` | **False** | [0.106, 0.861] | [0.761, 777] |
| moment-matched, `eps = 1.0 t_-` | **False** | [−0.189, 0.833] | [1.15, 174] |
| GLE-inspired, `eps = 0.25 t_-` | **True** | [0.533, 0.941] | [0.762, 74.7] |
| GLE-inspired, `eps = 0.5 t_-` | **False** | [0.210, 0.933] | [0.861, 78.3] |
| GLE-inspired, `eps = 1.0 t_-` | **False** | [−0.442, 0.931] | [0.983, 30.6] |

Under the rule committed before execution — mean cosine `> 0`, negative-cosine
fraction exactly `0`, and a decrease in the batch objective at both step norms,
at every band and `eps` and for every seed — **one of six configurations is
usable, and it is the GLE-inspired baseline, not the Section 11 moment-matched
filter.**

**The moment-matched filter is usable at no `eps`** and is generally the worse
of the two here. At seed 102, band `(0.35, 0.60)`, `eps = 1.0`: cosine
`-0.045`, relative error `1152`, with **59 %** of trajectories at negative
cosine. Its `O(omega^3)` asymptotic advantage over the baseline's `O(omega^2)`
does not translate into better gradients at these finite bandwidths. The
causal-error note carried exactly that caveat; it is now measured rather than
hedged.

Both approximations degrade sharply with bandwidth. At `(0.03, 0.08)` both are
serviceable — cosines around 0.83-0.94, no negative cases. By `(0.35, 0.60)`
both fail. Even in the best case the relative error is `0.33`, so magnitudes
are substantially wrong where directions agree.

### What Part B does and does not establish

It is a **gradient-fidelity probe at frozen parameters**, not a training
result. It says these two causal approximations to the future-facing adjoint do
not reproduce exact gradients well on this model at these bandwidths, and that
the preferred Section 11 construction is not the better of the two here.

It does **not** test recurrent error loops — the arrangement is feedforward by
construction, precisely because the witness above shows forward stability does
not imply error-loop stability. Neither approximation supplies an initial-state
derivative, so the comparison is restricted to `W1, b1, W2, b2`. The reference
is the exact gradient of the **discretized** system while the approximations are
continuous-time filters; that gap is measured, not eliminated. **No local-rule
training follows from this batch under any outcome.**

## 3. Part A — COMPLETE: four temporal laws under exact BPTT

Seeds 100/101/102, 300 updates, batch 16, four arms, full BPTT, one shared
ordered batch stream per seed. `TSS_A_STATUS=PASS`, 12/12 rows, 113 s.

**Verified at run time:** task structure as declared (192 evaluation sequences,
one content symbol per step, exactly one write and one query, cue at the write
step, query after the cue and inside the window, no cue marker at the query,
signal variance `1.038`, delays `64/64/64`, classes `24` each); leakage probe
held-out `0.1302` against a permutation-null threshold of `0.1790`
(chance `0.1250`), **PASS**; the TSS-matched sector maps to `gamma = 0`, not
admissible under `M <= gamma T`; parameter and state counts as declared. Every
arm passed its trained-parameter correctness checks.

### Result

| arm | recall | vs chance | d8 / d16 / d32 | signal MSE | implied CE |
|---|---|---|---|---|---|
| `ideal_prospective` | 0.1250 | +0.00 pp | 0.11 / 0.14 / 0.13 | **0.0027** | 2.0931 |
| `tss_finite_adaptation` | 0.1198 | −0.52 pp | 0.15 / 0.11 / 0.10 | 0.0254 | 2.0827 |
| `tss_memory_then_prospective` | 0.1562 | +3.12 pp | 0.17 / 0.15 / 0.15 | 0.0506 | 2.0757 |
| `retained_compartment` | 0.1354 | +1.04 pp | 0.16 / 0.12 / 0.12 | 0.0553 | 2.0789 |

### Recall: INCONCLUSIVE, by the rule declared before execution

**Every arm is within 5 pp of the 12.5 % chance level.** The protocol's s6
declared that case in advance: *"every arm within 5 pp of the 12.5 % chance
level on recall → recall is **inconclusive**; only the tracking comparison is
interpretable and the pilot does not answer the memory question."* That is the
reading, and it is not adjusted after the fact.

An independent confirmation, not a restatement: the implied recall cross
entropies are `2.0757`-`2.0931` against `ln 8 = 2.0794`. Every arm's recall
head finished at — in two cases marginally worse than — a uniform predictor.
**Nothing learned the cue at all in 300 updates.** The protocol named this a
live possibility and a declared outcome rather than a failure to hide.

Two numbers must therefore **not** be read as signals, and are recorded only so
nobody else reads them that way:

* `retained_compartment` − `ideal_prospective` on recall is positive in all
  three seeds (+0.52, +1.04, +1.56 pp, mean +1.04). At chance, with CE at
  `ln 8`, this is variation between two arms that both failed;
* `tss_memory_then_prospective`'s 0.1562 is driven by one seed (0.1979 at
  seed 100 against 0.1198 and 0.1510 at 101 and 102).

### Tracking: interpretable, and unfavourable to the candidate

Current-signal MSE, lower is better, paired per seed:

| comparison | per seed | mean | sign |
|---|---|---|---|
| `retained` − `ideal` | +0.0575, +0.0509, +0.0492 | **+0.0525** | all + |
| `retained` − `tss_finite_adaptation` | +0.0306, +0.0295, +0.0294 | **+0.0298** | all + |
| `retained` − `tss_memory_then_prospective` | +0.0137, −0.0077, +0.0079 | +0.0046 | mixed |

The **memoryless cancellation control tracks about twenty times better than
every memory-bearing arm** (0.0027 against 0.025-0.055), which is exactly the
property it exists to exhibit: with no driven temporal state it carries no lag.
Our retained compartment is consistently worse at tracking than the TSS
finite-adaptation arm, in all three seeds.

**This is not a memory/reactivity trade-off.** A trade-off would require the
memory-bearing arms to have bought something with their worse tracking. At
chance recall they bought nothing measurable. The honest statement is that the
arms carrying temporal state tracked worse and did not use that state for
anything the task rewarded.

### What Part A does not establish

It does **not** show that the retained-compartment law cannot support recall.
It shows that **no** arm — including the literature-informed TSS comparator —
learned an 8-way cued recall at delays 8/16/32 within 300 updates at batch 16,
so the task did not discriminate between the temporal laws at this budget. A
longer schedule is the obvious thing that might change this; it is **not**
proposed here, and nothing in this result predicts its outcome.

## 4. Provenance

Part B was **not** re-run to accompany Part A. It is a frozen-parameter probe
with fixed inputs, seeds and filters that reproduced identically on five
dispatches, so re-running it would have added cost and no information. Part A's
status file records Part B's run directory and commit, so the two halves stay
linked.

| | value |
|---|---|
| Part B commit | **`6000b26cd8c982ca75e3ffe2e57c8932f3e56ace`** |
| Part B artifacts | `/Users/durso/s5-runs/tss_pilot/20260916-020800/part_b/` |
| Part B logs | `/Users/durso/s5-runs/tss_pilot/logs/20260916-020800/part_b.log` |
| Part B status | `TSS_B_STATUS=COMPLETE`, 9/9 rows, 33 s |

The Part-A continuation records this reference in its own status file, so the
two halves stay linked without either being re-executed.

| | value |
|---|---|
| Part A commit | **`d7dd53efa46adf85c1973c37d9ded1ba214b9415`** |
| Part A artifacts | `/Users/durso/s5-runs/tss_pilot/20260916-120620/part_a/` |

## 5. What the pilot as a whole yields

**Neither half supports the mechanism it was built to test**, and both outcomes
were declared as possibilities in advance.

* **Part A** cannot separate the four temporal laws, because none of them
  learned the memory task at this budget. Its one interpretable comparison —
  tracking — favours the memoryless control decisively and places our retained
  compartment last or joint-last.
* **Part B** finds one of six causal-approximation configurations usable, and
  it is the GLE-inspired baseline rather than our preferred Section 11
  construction, whose asymptotic advantage does not survive at finite
  bandwidth.

The roadmap's stated gate was to *"identify and demonstrate a distinct benefit
over the correct prospective baseline at the toy level, then choose the SSM
placement that addresses that measured benefit."* **No such benefit was
demonstrated**, so that gate is not met and no SSM placement follows from this
pilot.

Three declared expectations did hold, which is worth recording since they were
predictions: the ideal control failed a balanced history-only query (exactly
chance); the equivalence between our law and its adaptation twin held in
forward values and gradients; and the TSS-matched sector was confirmed to sit
outside our admissible set at `gamma = 0`.

### Limitations, unchanged from the protocol

Three seeds; 300 updates at batch 16; one task, one clock, one initialization,
one learning rate with zero development budget for any arm; a state budget
matched while parameter counts are not (1145 / 689 / 1417 / 689); Part B
feedforward by construction with no initialization derivative from either
approximation and a measured but uneliminated discrete-versus-continuous gap.
The full list is `docs/TSS_PILOT_PROTOCOL.md` s8.

### Cost

Nine cluster dispatches, of which Part A trained on one. Five were spent on a
timing measurement that was itself wrong — the evidence that settled it
(`step x 3 == compile_s`) was present in the third dispatch's output and in
every one after. The protocol's amendments record each correction with the
measurement that prompted it.
