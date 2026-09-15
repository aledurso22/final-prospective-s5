# Constrained learned response: results

Contract and bounded plan, committed before execution:
`docs/CONSTRAINED_PROSPECTIVE_RESPONSE_PROTOCOL.md`.

**Outcome: the predeclared screen FAILED.** This run scored 0.052 pp lower with
the response learned than with it frozen. Reported, not adjusted.

> **CORRECTIONS APPLIED 15 September 2026** after the coordinator's post-run
> algebra review. The recorded scores and execution provenance are unchanged;
> several *interpretations* below were too strong and are corrected in place,
> and one analytic finding is added in s0.

## 0. An exact redundancy, found analytically after the run

`(Delta, gamma_n, rho)` is input-output equivalent to `(Delta/gamma_n, 1, rho)`
for every admissible `rho`, not merely in a limit: dividing the whole equation
by `gamma_n` and setting `hat_Delta = Delta/gamma_n` leaves the same continuous
state and input matrices, hence the same ZOH blocks, scan and output
trajectory, with the same carry.

So of the 128 added leaves, **only the 64 `rho` coordinates add per-mode
response-shape freedom** relative to the already-trainable clock; the 64
`gamma_n` coordinates re-parameterize that clock. In the interior of the bounds
and for the differentiable data loss,
`dL/d(log Delta_i) = -dL/d(log gamma_n_i)`.

The equivalence concerns the forward model and the data loss. It does **not**
make the two training procedures identical: AdamW, log-coordinate weight decay,
bounds and global gradient clipping can behave differently when a redundant
coordinate is present. The observable clock coordinate is
`log_step - log(gamma_n)`, alongside `rho` — not `gamma_n` alone.

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
   not finite. Identical in float32 and float64, so not a
   precision effect: smallest finite `mu` `5e-5`, largest non-finite `5e-6`
   **on the probed matrices**. **Scope correction:** this is a non-finite
   result of the TESTED numerical exponential implementation on THOSE matrices,
   not a physical mass threshold and not a universal frontier determined by
   mass alone. A finite matrix has a finite mathematical exponential; matrix
   norm, pole scale, conditioning and the implementation's scaling-and-squaring
   limits all matter. The `rho` lower bound was raised to `1e-2`, putting the
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
| **learned vs `gp_fixed_mass` (frozen, same family)** | **-0.052 pp** | this run scored lower |

**The screen FAILED.** The shortfalls against the +0.3 pp target are
**0.698 pp** (`gain_clip_s5`) and **1.095 pp** (`alpha_p_s5`); those are
distinct quantities from the raw accuracy differences of -0.398 and -0.795 pp
and must not be conflated.

The fixed-versus-learned comparison is the internally controlled one: the two
arms are the same model, the same initialization, the same data order and the
same schedule, differing only in whether `gamma_n` and `rho` may move. **This
run scored 0.052 pp lower with them free** — three examples out of 5,783 — at a
cost of 128 parameters, of which only 64 add response-shape freedom (s0).
**A difference of three examples on one seed establishes neither degradation
nor improvement.**

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

Largest movement `|d log gamma_n| = 0.41`, `|d log rho| = 0.29`. **In the saved best
checkpoint no mode sits at any declared bound** (`0/0` at every layer). That is
a statement about this checkpoint only: no trajectory was recorded, so it does
not establish that the bounds were never active during training.

The component identities hold on the **trained** values: maximum residual
`1.8e-15`, all components strictly positive. Every learned pair therefore still
has a positive physical realization.

Observed MARGINAL distributions: the median `gamma_n` is below its initial
value and the median `rho` above it, with the median derived mass ~7 % below
its initial value. These are marginals. They do **not** establish per-mode
compensatory motion, a joint within-mode relationship, or an approximately
constant-mass trajectory; that would need the per-mode joint values and the
trajectory, which were not recorded.

Some modes reach `rho` near 1, where `h = G_s sqrt(1-rho)` becomes small — `h`
reaches 0.060. At `rho = 1` the transfer reduces to an ordinary memory-bearing
SSM response with its clock rescaled, **not** to the professor equation's
memoryless map. Movement toward that limit in part of the population does not
show that the trained network became equivalent to ordinary S5.

**Arithmetic note:** `tau_d` and `mu` are numerically identical in the table
because `tau_d = gamma_ref gamma_n rho` and `mu = T gamma_n rho` with
`T = gamma_ref = 5` in this reference. That is a coincidence of the chosen
constants, not an error.

## 5. The professor control

`prospective_recurrence` implements `r + T r' = 0` exactly as
`s_k = J^-1 b x_k`, by diagonal division. Verified: **zero driven history at
nonzero lag** on the isolated core, and no dependence on earlier inputs with
the current input held. It carries **zero recurrent state**.

**Correction.** The report previously listed a nonzero `log_step` gradient as a
verified property. That is wrong as a property of this model: the map is
`-Delta B_c/(Delta lambda) = -B_c/lambda`, so its exact data-loss gradient with
respect to `log_step` is **ZERO**. A test asserting merely `> 0` was passing on
floating-point cancellation residue, not on a genuine gradient path. The
ordinary `B`, `C`, `D` and pole degrees of freedom can still learn. The executed
run and its score are unchanged and are not rerun for this correction.

It reached **84.85 %**. Low accuracy was declared in advance not to be a
correctness condition, and it is not treated as one.

**What this number is not.** It is one trained architecture and configuration on
this pooled classification task. It is **not** an additive decomposition of
accuracy into static and memory contributions, **not** a performance floor, and
**not** a ceiling on what recurrence could contribute.

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

The response did move and the final accuracies are close.

**Withdrawn.** An earlier version of this section inferred from that pairing
that the objective is flat in this family and that the optimizer "explored the
family and found nothing better". Similar final accuracy does not establish
objective curvature or search completeness, and neither claim is supported. The
only flatness established here is the exact reparameterization direction in s0,
which concerns the `gamma_n`/clock coordinate and says nothing comparable about
the `rho` direction or the full training objective.

What stands: the predeclared screen failed, and this single-seed run does not
distinguish the learned and frozen responses.

No larger run and no rescue sweep follows from this. The previous Stage 2
verdict and the fixed positive-mass result are unchanged.
