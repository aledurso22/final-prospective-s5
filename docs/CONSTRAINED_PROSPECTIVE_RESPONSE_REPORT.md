# Constrained learned response: results

Contract and bounded plan, committed before execution:
`docs/CONSTRAINED_PROSPECTIVE_RESPONSE_PROTOCOL.md`.

**Outcome: the predeclared screen FAILED, and letting the response learn made
it slightly worse than freezing it.** Reported, not adjusted.

## 1. Execution identity

| item | value |
|---|---|
| checkout | `/Local/durso/final-prospective-s5` |
| branch | `adaptive-recurrence` |
| executed commit | **`4848ee4989474bdb2bd1ddaacc9095b6301bb507`** |
| host | `pgi15-gpu3.iff.kfa-juelich.de`, RTX 3090 |
| SLURM job | 65870, `CUDA_VISIBLE_DEVICES=0` |
| backend | `gpu`, `CudaDevice(id=0)`, jax 0.11.0, flax 0.12.8, optax 0.2.8 |
| data | SC10 MFCC, train `73caf4e4...`, val `dc4f65d6...`, split seed 0 |
| test split | **never opened** |
| artifacts | `/Users/durso/s5-runs/constrained/20260915-193049/` |
| logs | `/Users/durso/s5-runs/constrained/logs/20260915-193049/` |
| status | **`CONSTRAINED_STATUS=PASS`, `CONSTRAINED_EXIT=0`** |
| wall time | 334 s of a 1200 s hard budget |

Command:

```
bash /Local/durso/final-prospective-s5/bin/run_experiments/cluster_constrained.sh
```

Training: seed 100, lr 1e-3, batch 32, 10-epoch cosine schedule, full BPTT,
highest matmul precision, deterministic XLA, validation only — identical to
Stage 2.

## 2. Correctness checks: 30 passed

`tests/test_constrained_response.py`, run on the GPU inside the same budget,
**30 passed in 111 s**, exit 0. Covered: component identities including the
reference point; the horizon fixed and the mass derived; per-mode row
broadcasting at `P != H`; unchanged common initialization; frozen-response
equivalence in **outputs and shared-parameter gradients**; both leaves
receiving nonzero gradients and moving under a real update with weight decay
also set to zero; forward clip and post-update projection; finite matrix
exponential and derivatives at interior and boundary corners; conjugate
sharing; scan, chunk and reset consistency; per-arm response policy; the
professor core's zero history; and float32/complex64 gradients for both leaves.

### Two earlier failures, both resolved by measurement

The first cluster run failed two checks and the launcher **refused to train**.
Neither was a defect in the model, and no tolerance was loosened.

1. **A bound I declared too wide.** `rho` down to `1e-4` with
   `gamma_n = 1e-2` gives `mu = 5e-6`, where the block matrix exponential is
   not finite. The measured frontier is **identical in float32 and float64**,
   so this is stiffness, not precision: smallest finite `mu` `5e-5`, largest
   non-finite `5e-6`. The `rho` lower bound was raised to `1e-2`, putting the
   worst corner at `mu = 5e-4` — ten times the smallest measured-good value.
   The excluded corner is kept as a regression test asserting it really is
   non-finite.
2. **A probe that did not match production precision.** The float32 gradient
   check reported a 17 % discrepancy. Cause: it never set the matmul
   precision, so it ran at the GPU's TF32 default while training runs at
   `highest`. TF32's ~10-bit mantissa puts `~5e-4` relative error on the loss
   against a finite-difference signal of `2hg/loss ~ 6e-5` — the same order, so
   the check measured the matmul mode. At `highest` precision both leaves trace
   the same U-curve as float64 and reach **2.3e-3 and 1.6e-3 at step 1e-2**,
   under the **unchanged** 5e-3 tolerance.

The analytic gradient was independently confirmed correct: in float64 the
relative error against central differences falls
`8.9e-3 -> 8.0e-4 -> 9.4e-5 -> 1.9e-5` as the step shrinks and rises again at
`1e-4`, the signature of an O(h^2) difference converging on a correct gradient.

A third failure was a reporting-path bug of mine: the professor arm's
`state_counts` carries a descriptive note and `arm_state_counts` called `int()`
on every value, so the preflight crashed on the **second** arm. Fixed, with a
regression test that exercises the helper for every arm.

## 3. Results

Validation only; the test split was never opened.

| arm | val accuracy | val CE | params | state/layer | s/epoch | provenance |
|---|---|---|---|---|---|---|
| `alpha_p_s5` | 95.00 % | 0.28953 | 35,050 | 32 | 2.70 | historical, reused |
| `gain_clip_s5` | 94.60 % | 0.29908 | 35,050 | 32 | 2.80 | historical, reused |
| `gp_fixed_mass` | **94.26 %** | 0.30020 | 35,050 | 64 | 6.85 | historical, reused |
| **`gp_learned_response`** | **94.21 %** | 0.30038 | **35,178** | 64 | 6.95 | **new run** |
| `native_s5` | 94.17 % | 0.30953 | 35,050 | 32 | 2.75 | historical, reused |
| **`prospective_recurrence`** | **84.85 %** | 0.58897 | 35,050 | **0** | 2.30 | **new run** |

Best epochs: `gp_learned_response` 9, `prospective_recurrence` 7.

### The predeclared screen

| comparison | delta | verdict |
|---|---|---|
| learned vs `gain_clip_s5` (matched ordinary) | **-0.398 pp** | fails the +0.3 pp target |
| learned vs `alpha_p_s5` (Rawat reference) | **-0.795 pp** | fails the +0.3 pp target |
| learned vs `native_s5` | +0.035 pp | — |
| **learned vs `gp_fixed_mass` (frozen, same family)** | **-0.052 pp** | **learning did not help** |

**The screen FAILED.** The fixed-versus-learned comparison is the one that
isolates allowing response learning within this family, and it is the cleanest
negative available: the two arms are the same model, the same initialization,
the same data order and the same schedule, differing only in whether `gamma_n`
and `rho` may move. Allowing them to move cost **0.052 pp** and **128
parameters**.

### Reuse of the references, and why it is valid

The four references are **reused** at commit `227648e`, not rerun. Reuse is
justified by measurement, not assertion: the frozen-response tests show the
learned arm at its initialization reproduces `gp_fixed_mass` in **outputs** and
in **shared-parameter gradients** to `1e-10`, and adding the two leaves left the
common parameter tree **bit-identical** under the same key. Data digests,
schedule, optimizer and seed are unchanged. Had any of those failed, the
affected comparator would have been rerun or the comparison labelled
incomplete.

## 4. The learned response actually moved

This matters: a null result means something different if the parameters never
left their initialization. They did.

All four layers, 64 stored modes:

| quantity | init | min | median | max | std |
|---|---|---|---|---|---|
| `gamma_n` | 1.00 | 0.662 | **0.853** | 1.249 | 0.111 |
| `rho` | 0.75 | 0.615 | **0.840** | 0.998 | 0.096 |
| `mu` (derived) | 3.75 | 2.352 | **3.473** | 5.341 | 0.657 |
| `c_d` | 7.50 | 4.966 | 6.401 | 9.366 | 0.833 |
| `G_s = G_d` | 2.00 | 1.502 | 1.786 | 2.438 | 0.222 |
| `h` | 1.00 | 0.060 | 0.714 | 1.512 | 0.325 |
| `g_L` | 1.00 | 0.926 | 1.072 | 1.443 | 0.110 |

Largest movement `|d log gamma_n| = 0.41`, `|d log rho| = 0.29`. **No mode sat
at any declared bound** in any layer (`0/0` at every layer), so the amended box
never constrained the outcome.

The component identities hold on the **trained** values: maximum residual
`1.8e-15`, all components strictly positive. Every learned pair therefore still
has a positive physical realization.

Structure worth recording, as an observation rather than a claim: `gamma_n`
moved **down** ~15 % in median while `rho` moved **up** ~12 %, leaving the
derived mass `mu` only ~7 % below its initial value. The optimizer moved the
two learned quantities in partly compensating directions, largely along a level
set of the derived mass, rather than driving the mass anywhere in particular.
Some modes approach `rho -> 1`, where the axial coupling `h = G_s sqrt(1-rho)`
becomes small — `h` reaches 0.060 — i.e. toward near-decoupled compartments.
That is a boundary of the physical family, though not a declared numerical
bound.

**Arithmetic note:** `tau_d` and `mu` are numerically identical in the table
because `tau_d = gamma_ref gamma_n rho` and `mu = T gamma_n rho` with
`T = gamma_ref = 5` in this reference. That is a coincidence of the chosen
constants, not an error.

## 5. The professor control

`prospective_recurrence` implements `r + T r' = 0` exactly as
`s_k = J^-1 b x_k`, by diagonal division. Verified: **zero driven history at
nonzero lag** on the isolated core, no dependence on earlier inputs with the
current input held, and nonzero gradients to `B`, `C`, `D`, `Lambda_re` and
`log_step`. It carries **zero recurrent state**.

It reached **84.85 %**. Low accuracy was declared in advance not to be a
correctness condition, and it is not treated as one. The number is informative
in a different way: a completely memoryless recurrence, on this pooled
classification task, gets within about 10 points of every memory-bearing arm.
The whole span that the recurrent mechanisms compete over here is roughly
**84.85 % to 95.00 %**.

This arm illustrates the fully matched recurrence placement. It is **not** the
TSS memory architecture, which includes memory-bearing non-prospective neurons.

## 6. Costs, disclosed

`gp_learned_response` adds **128 real parameters** (35,178 vs 35,050) and keeps
the **already doubled** carry of the positive-mass family, 64 real coordinates
per layer against 32 for ordinary S5. Measured 10.55 ms/step against 9.51 ms
for the structurally identical fixed-mass arm — about 11 % more, consistent
with two extra scalars per mode entering the block coefficients — and 6.95
s/epoch against 2.70-2.80 for the one-state arms.

**This screen therefore cannot establish an advantage independent of parameter
or state capacity**, and it did not need to: the arm did not win. No dummy
parameters were inserted into any control to manufacture capacity matching.

## 7. Limitations

* **One seed, one configuration, 10 epochs, validation only.** The differences
  being discussed are 0.05-0.80 pp. No single-seed outcome is statistical
  confirmation, and no variance estimate exists.
* The fixed-versus-learned comparison is the internally controlled one. The
  literature comparisons measure complete constructions that differ in more
  than the response.
* The bounds were amended once, from a measurement, **before** training and
  before any score. They never bound the outcome.
* Weight decay acts in **log coordinates** on the response leaves — an
  optimizer choice inherited from the paper-reproduction policy, not a derived
  plasticity law.
* Reused reference scores are historical measurements at `227648e`, validated
  for equivalence but not re-executed in this batch.
* This is a classification screen. It says nothing about meta-learning, memory
  allocation, or unseen temporal regimes.

## 8. Conclusion

Within this bounded, predeclared screen, allowing the two constrained physical
response parameters to learn **did not help**: 94.21 % against 94.26 % for the
same model with the response frozen, at a cost of 128 parameters, and 0.40 and
0.80 pp below the matched ordinary and Rawat references.

The result is not that learning failed to move anything. The response moved
substantially — median `gamma_n` down 15 %, median `rho` up 12 %, `mu` spread
over 2.35 to 5.34 against a fixed 3.75, no mode at a bound — and the model was
no better for it. On this task, at this size and budget, the derived response
appears to sit on a flat region of the objective: the optimizer explored the
two-parameter family and found nothing materially better than the reference
point the circuit prescribed.

The supported next question is therefore **not** a wider response search, which
this result gives no reason to expect would pay. It is whether the objective is
sensitive to the response at all in this regime — a question that the flat
outcome here, and the 84.85 % memoryless floor in s5, both bear on, and which
would need a task where recurrent memory carries more of the decision than
roughly ten accuracy points.

No larger run and no rescue sweep follows from this. The previous Stage 2
verdict and the fixed positive-mass result are unchanged.
