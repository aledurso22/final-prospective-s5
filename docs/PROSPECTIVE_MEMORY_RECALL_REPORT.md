# Memory-recall study: results

Protocol committed before execution:
`docs/PROSPECTIVE_MEMORY_RECALL_PROTOCOL.md`.

**Outcome: the predeclared screen FAILED.** The generalized prospective
recurrence did not improve recall over the literature mechanisms by the
declared margin, and an ordinary S5 with twice the stored modes beat it in
every seed. Two findings survive and are worth keeping; both are stated below
with their limits.

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

| arm | worst core impulse rel | query-logit rel | function-matched? |
|---|---|---|---|
| `gp_rho` | **1.99e-03** | 1.17e-03 | yes, by measurement |
| `gp_rho_frozen` | **1.99e-03** | 1.17e-03 | yes, by measurement |
| `rawat` | **2.09e+00** | 1.48e+00 | **no** |
| `professor` | **7.57e+00** | 1.18e+00 | **no** |

Gate **PASSED** for the generalized arms at 1.99e-03 against the predeclared
1e-2, so they genuinely start from the warm-up function rather than merely
having `rho` near 1.

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

**The screen FAILED.** The target was at least +0.3 pp over **both**
references. Against `ordinary` the difference is consistently positive but
averages **0.098 pp** — about **two examples out of the 2,048 evaluated** — and
is below the target in every seed. Against `rawat` the sign flips across seeds.

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
(+58 %)**. So the honest statement is that at equal carry and greater parameter
count, plain extra ordinary modes bought substantially more recall on this task
than the derived response did. This does not isolate every possible benefit of
the exact coefficient ties, and the brief said so in advance; it does mean the
simplest capacity explanation is not ruled out here — it is favoured.

## 5. What learning `rho` actually did — and a correction to my own protocol

| seed | `rho` init | final min | final median | modes that FELL |
|---|---|---|---|---|
| 100 | 0.9998 | 0.9497 | **0.9999** | 4 / 32 |
| 101 | 0.9998 | 0.9323 | **0.9999** | 5 / 32 |
| 102 | 0.9998 | 0.8212 | **0.9999** | 4 / 32 |

**The median final `rho` is 0.9999, which is the declared ceiling.** So 27-28 of
32 modes moved *up* and are pinned at the bound; only 4-5 modes per seed fell,
though those fell substantially (to 0.82-0.95).

**This corrects a statement I put in the protocol before the run.** I wrote that
`rho` "can effectively only fall" because it starts adjacent to its ceiling. That
was wrong: there was 1.0e-4 of headroom in log space and the optimizer used it,
driving most modes into the clip. The clip is therefore **active for roughly
85 % of modes at the end of training**, which is a bound interaction that has to
be reported, not a free exploration of the family.

`rho -> 1` is the **ordinary memory-bearing SSM limit** (the transfer reduces to
`b/(p+j)` with a rescaled clock). So the dominant learned direction was *towards*
ordinary S5, not away from it. The derived mass barely moved in the median:
`mu = T rho` went from 4.9990 to 4.9995. The minority of modes that fell reached
`mu` of 4.11-4.75.

The `gp_rho` vs `gp_rho_frozen` gap of **+0.911 pp, positive in all three
seeds**, is the one comparison that favours the mechanism. Given the `rho`
distribution, the most it supports is that *allowing* `rho` to move helped
relative to holding it at 0.9998 — and the movement was mostly toward the
ordinary limit with a handful of modes departing from it. It is **not** evidence
that a distinctly generalized response is what helped, and with three seeds and
most modes pinned at a bound it is not a population claim.

`gp_rho_frozen` scored **0.82 pp below `ordinary`** on average despite being
function-matched to 2e-3 at initialization. Two arms that start within 0.2 % of
each other diverged by more than the effect being screened for, which is a
useful caution about the resolution of this setup.

Effective clocks were essentially identical across `ordinary`, `gp_rho` and
`gp_rho_frozen` (median 0.0071, range 0.0012-0.105), so the arms did not
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
are reported, and they do not distinguish the arms** — the generalized
recurrence is not more cue-selective than ordinary S5 here.

## 8. Costs

| arm | params | physical carry | auxiliary | input buffer | s per continuation |
|---|---|---|---|---|---|
| `ordinary` | 7,176 | 64 | 0 | 0 | 10.8 |
| `rawat` | 7,176 | 64 | 0 | 64 | 9.7 |
| `professor` | 7,176 | **0** | 0 | 0 | 5.3 |
| `gp_rho` | 7,208 | 64 | **64** | 0 | 15.3 |
| `gp_rho_frozen` | 7,208 | 64 | 64 | 0 | 14.9 |
| `ordinary_2x` | 11,368 | **128** | 0 | 0 | 10.1 |

`gp_rho` adds 32 parameters (one `rho` per stored mode per layer) and doubles
the carry, and is the slowest memory-bearing arm at 1.4x `ordinary`.

## 9. Limitations

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
**did not** improve recall over ordinary S5 or Rawat's prospective-input S5 by
the predeclared margin. Its edge over ordinary S5 was +0.098 pp, roughly two
examples; against Rawat it was inconsistent in sign.

The most informative comparison is the one that did not involve the mechanism
at all: **doubling the ordinary recurrent state gained 3.8 pp over the
generalized recurrence in every seed, at equal total carry**. Whatever this task
rewards, extra ordinary modes supplied it more effectively than the derived
response did.

Allowing `rho` to learn helped relative to freezing it, in all three seeds. But
the learned direction was overwhelmingly *toward* the ordinary-SSM limit, with
most modes pinned at the bound, so that result does not support the derived
response being the active ingredient.

The supported next question is whether any response-shaping in this family can
compete with simply adding ordinary state on tasks of this kind, and that would
need the matched response-parameterization control this batch deliberately did
not include. **No larger run and no rescue sweep follows from this result.** The
earlier Speech Commands studies and their verdicts are unchanged.
