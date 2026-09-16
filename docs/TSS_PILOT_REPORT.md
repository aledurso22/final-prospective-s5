# TSS-based prospective pilot: status and Part B results

Protocol: `docs/TSS_PILOT_PROTOCOL.md`. Branch `tss-pilot`.

> **Part B is COMPLETE and its results are below. Part A has NEVER TRAINED.**
> An earlier version of this file said nothing had been executed. That is stale
> and is corrected here: eight cluster dispatches have been made, the focused
> checks pass on GPU, Part B has completed identically five times, and Part A
> has been refused by its own preflight on every attempt.

## 1. Execution status

| item | value |
|---|---|
| host | `pgi15-gpu3.iff.kfa-juelich.de`, RTX 3090, SLURM 65910 |
| backend | `gpu`, `CudaDevice(id=0)`, jax 0.11.0 |
| focused checks | **50 passed on GPU** at `6000b26`, 173 s |
| **Part B** | **COMPLETE**, 9/9 rows, 33 s, `TSS_B_STATUS=COMPLETE` |
| **Part A** | **NEVER TRAINED** — refused by preflight on all 8 dispatches |
| latest artifacts | `/Users/durso/s5-runs/tss_pilot/20260916-020800/` |
| latest logs | `/Users/durso/s5-runs/tss_pilot/logs/20260916-020800/` |

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
| 8 | `6000b26` | Part A refused; Part B complete | **cause found**: a weakly typed leaf retraced the step. 2560.91 → **3.00 ms**. Then refused on budget overhead: 128.5 s needed against 107 s left |

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

## 3. Part A — NOT TRAINED

No arm has been trained for a single update. What is established about Part A
is its setup and its cost; nothing about its outcome.

**Verified at run time, on every dispatch:**

* task structure — 192 evaluation sequences, one content symbol per step,
  exactly one write and one query, cue present at the write step, query after
  the cue and inside the window, no cue marker at the query, signal variance
  `1.038`, delays `64/64/64`, classes `24` each;
* leakage probe — held-out `0.1302` against a permutation-null threshold of
  `0.1790` (chance `0.1250`), **PASS**, with the deliberately leaking positive
  control failing it as it must;
* the TSS-matched sector maps to `gamma = 0`, **not admissible** under
  `M <= gamma T`;
* parameter counts 1145 / 689 / 1417 / 689 and temporal state counts
  0 / 16 / 16 / 16, as declared.

**Corrected steady-state cost, `6000b26`:**

| arm | compile | step | projected arm total |
|---|---|---|---|
| `ideal_prospective` | 2.8 s | 2.81 ms | 5.3 s |
| `tss_finite_adaptation` | 4.0 s | 25.47 ms | 26.9 s |
| `tss_memory_then_prospective` | 7.7 s | **3.00 ms** | 10.4 s |
| `retained_compartment` | 4.1 s | 24.22 ms | 25.9 s |

`PREFLIGHT_A_PROJECTED_TOTAL_S = 128.5`, refused against 107 s remaining.

## 4. Provenance to preserve

Part B is complete and is **not** re-run to accompany Part A. There is no
scientific requirement that both parts execute in one invocation: Part B is a
frozen-parameter probe with fixed inputs, seeds and filters, and it has
reproduced identically five times.

| | value |
|---|---|
| Part B commit | **`6000b26cd8c982ca75e3ffe2e57c8932f3e56ace`** |
| Part B artifacts | `/Users/durso/s5-runs/tss_pilot/20260916-020800/part_b/` |
| Part B logs | `/Users/durso/s5-runs/tss_pilot/logs/20260916-020800/part_b.log` |
| Part B status | `TSS_B_STATUS=COMPLETE`, 9/9 rows, 33 s |

The Part-A continuation records this reference in its own status file, so the
two halves stay linked without either being re-executed.

## 5. After Part A executes

Record its actual result, favourable or not: per-seed scores and paired
differences for all four arms, recall by delay, signal MSE, the interventions,
the trained-parameter correctness checks, timings, and every declared
limitation in `docs/TSS_PILOT_PROTOCOL.md` s8. Every arm landing near the
12.5 % chance level on recall remains a declared outcome, not a failure to
hide.
