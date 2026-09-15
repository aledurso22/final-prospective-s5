# Constrained learned response: model contract and bounded run plan

**Committed before numerical execution.** Branch `adaptive-recurrence`, from
`3a06cd34d7a1cb21a07a2cd924d4410dd4d7e3e1`.

## 0. Provenance of the scope change

The user clarified that the **literature-analogous coefficient policy comes
first**: keep the derived law and the fixed horizon, learn designated
constrained physical parameters by full BPTT, derive the rest. That
clarification **superseded the activity-dependent adaptation brief BEFORE any
numerical execution of this study**.

The superseded within-sequence work is **retained, not deleted**:
`s5/adaptive_circuit.py`, `tests/test_adaptive_recurrence.py`,
`tests/adaptive_float32_probe.py`, and the arms `gp_adaptive_mass`,
`gp_frozen_adaptive`, `ordinary_adaptive` remain selectable and tested. **They
are not part of this batch.** Nothing from that direction was ever executed
numerically — no cluster job was submitted from it — so there is no superseded
result to label.

## 1. The model

Existing clipped S5 poles, gain-scaled input map, output timing, learned native
steps and full BPTT are unchanged. Clock absorbed once:
`F0 = diag(Delta Lambda)`, `B0 = diag(Delta) B_c`, `r_n = -F0 s - B0 x`.

Per stored complex mode `i`, two real learned quantities shared with the
conjugate partner:

```
mu_i s_i'' + gamma_n_i s_i' + r_n_i + T r_n_i' = 0 ,   mu_i = T gamma_n_i rho_i
```

**`T = 5` stays FIXED** at the published-comparison horizon. **The mass is
DERIVED**, never learned separately. Initialization `gamma_n = 1`,
`rho = 0.75`, so `mu = 3.75` exactly matches the prior fixed positive-mass
model. `gamma_n` is **not** renormalized to 1 after an update: that would
change which physical coupling is held fixed.

## 2. Positive component realization

`s5/constrained_response.py`. With `kappa_0 = 3/2` and the FIXED whole-equation
normalization `gamma_ref = 5`:

```
c_s = T kappa_0 = 7.5            c_d = gamma_ref gamma_n kappa_0 = 7.5 gamma_n
G_s = G_d = kappa_0 / rho        h   = G_s sqrt(1 - rho)
g_L = G_s - h = kappa_0 / (1 + sqrt(1 - rho)) > 0
```

giving identically `kappa = kappa_0`, `T = 5`,
`tau_d = gamma_ref gamma_n rho`, `gamma_phys = gamma_ref gamma_n`,
`M_phys = T tau_d`, and `mu = M_phys/gamma_ref = T gamma_n rho`.

Verified by arithmetic before implementation: at `gamma_n = 1, rho = 0.75` the
components are `c_s = c_d = 7.5`, `h = g_L = 1`, `G_s = G_d = 2` — the original
symmetric reference — and every identity holds to `1.1e-13` across the whole
admissible box with **strictly positive** components at every corner.

Learning therefore departs from equality of the two capacitances and from
equality of leak and axial conductance. Those equalities were **initialization
and modeling assumptions, not laws**.

**Scope, stated plainly:** this is a family within the previously declared
operating-point model with its prospective-source and learned additive-current
convention. A positive component manifold does **not** make arbitrary complex
S5 feedback, or BPTT, a biological circuit or a plasticity rule.

## 3. Parameterization, bounds and optimizer

Leaves `log_response_gamma`, `log_response_rho`, each shape `(P,)`, declared
**last** in `setup()` so the common parameter draw is unchanged (asserted by
test, not assumed).

```
gamma_n = exp(eta),  eta  init 0          rho = exp(zeta), zeta init log(0.75)
```

**Declared numerical admissibility bounds, not swept:**
`1e-2 <= gamma_n <= 1e2`, `1e-2 <= rho <= 1 - 1e-4`. These are numerical design
choices, **not** physiological measurements and **not** numbers from NLA.

**AMENDMENT, 15 September 2026, from a measurement, before any training and
before any validation score.** The first declaration allowed `rho` down to
`1e-4`. With `gamma_n = 1e-2` that gives a derived mass `mu = 5e-6`, and at
that corner the block matrix exponential is **not finite** — caught by the
boundary test in the first cluster check run, which correctly stopped the batch
before training. The measured frontier
(`experiments/gp/constrained_numerics_probe.py`), **identical in float32 and
float64**, so this is stiffness rather than precision:

| mu | 499.9 | 50 | 5 | 0.5 | 0.05 | 5e-3 | 5e-4 | 5e-5 | 5e-6 |
|---|---|---|---|---|---|---|---|---|---|
| value and gradient finite | yes | yes | yes | yes | yes | yes | yes | **yes** | **NO** |

Smallest `mu` measured finite: `5e-5`. Largest measured non-finite: `5e-6`. The
`rho` lower bound is therefore raised to `1e-2`, putting the **worst corner** of
the box at `mu = 5 * 1e-2 * 1e-2 = 5e-4` — ten times the smallest measured-good
value and a hundred times the measured failure. `gamma_n` is unchanged; its
extreme corner `mu = 499.9` was measured finite. The box still leaves the mass
free over three decades below its initial `3.75`.

This bound was set **from measurement**, on the same principle as the
second-order prototype's `MU_RATIO_MIN`. It was not chosen to improve a score,
and the excluded corner is kept as a regression test asserting it really is
non-finite, so the bound stays a measured limit rather than unnecessary
caution.

**Gradient correctness was checked, not assumed.** The same probe compared the
analytic directional derivative against central differences at six step sizes
in float64: the relative error falls `8.9e-3 -> 8.0e-4 -> 9.4e-5 -> 1.9e-5` as
the step shrinks and rises again at `1e-4` where rounding dominates, for both
leaves. That is the signature of a CORRECT gradient with an O(h^2) finite
difference, not of a dropped term.

* forward safety guard: raw logs are **clipped before exponentiation**;
* projection policy: after every optimizer update the raw leaves are clipped
  back into the interval. **Optimizer state is untouched** — only parameters
  are projected.

The new leaves use the **existing** paper-reproduction AdamW policy: the same
learning rate and weight decay, no response-specific search. **Weight decay
acts in log coordinates** — an optimizer choice, not a derived plasticity law.

**Only `gp_learned_response` may carry these leaves.** Historical fixed arms
keep the blanket prohibition; the new arm gets an exact allowed-name, shape and
count check. The prohibition is not relaxed globally
(`assert_response_policy`).

Expected: **128 added real parameters** at `P = 16`, 4 layers, i.e. **35,178
against 35,050**, verified from the actual tree. **No new recurrent state**
relative to the fixed positive-mass arm.

## 4. Numerical reuse and the broadcasting pitfall

`mass_block_generator`, `mass_block_zoh`, the block associative scan and the
physical-state readout are reused. Coefficients are computed once per layer per
update, not per token; trajectories stay linear within a layer.

`gp_fixed.py` previously assumed scalar `gamma`/`rho`: dividing a `(P, H)`
input matrix by a `(P,)` vector broadcasts along the **feature** axis, which is
silently valid only when `H == P`. Fixed by an explicit `(P, 1)` row factor,
with a test at `P != H`. `T` stays scalar. Traced learned arrays are **never**
wrapped in the host-side `PhysicalResponse` dataclass and its Python-value
validation is never called on tracers.

## 5. Arms

| arm | role | response leaves | state/mode | provenance |
|---|---|---|---|---|
| `gp_learned_response` | **treatment**, constrained learned response | 2 x `(P,)` | 2 | new run |
| `prospective_recurrence` | professor equation `r + T r' = 0`, `s_k = J^-1 b x_k` | none | 0 | new run |
| `gp_fixed_mass` | frozen response, same family | none | 2 | historical, reused |
| `alpha_p_s5` | Rawat prospective-input reference | none | 1 + buffer | historical, reused |
| `gain_clip_s5` | matched gain-scaled clipped ordinary | none | 1 | historical, reused |
| `native_s5` | native S5 | none | 1 | historical, reused |

Historical scores at commit `227648e`, data digests `73caf4e4...` (train) /
`dc4f65d6...` (val), seed 100, lr 1e-3, 10 epochs: `native_s5` 94.17 %,
`alpha_p_s5` 95.00 %, `gain_clip_s5` 94.60 %, `gp_fixed_mass` 94.26 %.

Reuse is conditional on verified equivalence: the frozen-response tests prove
that the learned arm at its initialization reproduces the fixed arm's
**outputs and shared-parameter gradients**, and that adding leaves did not
change the common initialization. If equivalence failed, the affected
comparator would be rerun within the cap or the comparison labelled incomplete.
**Scores from incompatible configurations are never spliced.**

## 6. Training configuration — unchanged from Stage 2

Cached SC10 MFCC, 161 frames, 20 features, split seed 0, **test split never
opened**. Depth 4, width 32, 8 HiPPO blocks, batch 32, seed 100, lr 1e-3, the
exact existing **10-epoch cosine** schedule, full BPTT, highest precision,
deterministic XLA, **validation only**. No zero-mass rerun, no new task, no
sweep, no within-sequence adaptation.

## 7. Budget rule

**HARD TOTAL 20 minutes** of cluster wall time covering focused checks,
compilation, preflight and both training runs, with a watchdog and cleanup
reserve. Throughput is **measured** in the preflight before the batch launches;
the recurrence and carry are unchanged from the fixed-mass arm, so reuse is
substantial, but no runtime is stated as fact before measurement. Training
starts only if the checks pass **and** the measured projection fits. If it does
not fit, the cost is reported and no run is shortened, started or extended.

Fixed arm order so a partial batch stays interpretable:
`gp_learned_response`, then `prospective_recurrence`.

## 8. Screening target and what a result would mean

Retained: at least **+0.3 percentage points** over **both** `alpha_p_s5` and
`gain_clip_s5`. All observed differences are reported, including the
**fixed-versus-learned** comparison, which is what isolates allowing response
learning within this model family; the literature comparisons measure complete
constructions instead.

Disclosed with any result: **128 additional parameters** and the **already
doubled carry** versus ordinary S5. This screen therefore does **not** establish
an advantage independent of parameter or state capacity. No dummy parameters
are inserted into any control to manufacture capacity matching.

**Low accuracy is not a correctness condition for the professor control**: a
pooled speech classifier can be useful without recurrent memory, and that arm
illustrates the fully matched recurrence placement, not the whole TSS memory
architecture.

**No single-seed outcome is statistical confirmation.** No larger run or rescue
sweep follows automatically.
