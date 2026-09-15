# Adaptive prospective recurrence: model contract and bounded run plan

**Committed before any training.** A NEW development experiment informed by
Stage 2, not a reopening of Stage 2's failed screen, which stands.

Branch `adaptive-recurrence`, from
`3a06cd34d7a1cb21a07a2cd924d4410dd4d7e3e1`.

## 1. Model contract

Full derivation and every declared assumption: `s5/adaptive_circuit.py`
docstring, and the three levels are kept separate there.

**Level 1, published circuit** (NLA App. 6, Eqs. 80-81) with `E_L` as
reference, `g_sd = g_ds = h`:

```
G_s    = g_L + h                     FIXED
G_d(t) = g_L + h + g_E(t) + g_I(t)   VARIES with activity
I_d(t) = g_E(t) E_E + g_I(t) E_I     the synaptic DRIVE CURRENT
```

Each synapse therefore has **both** effects: it raises `G_d` (shunting, a
shorter dendritic time constant) and it injects current.

**Level 2, finite-dendrite elimination.** With `I_s = 0` the somatic equation
gives `v = (c u' + G_s u)/h` with **constant** coefficients, so

```
c^2 u'' + c(G_s + G_d(t)) u' + (G_s G_d(t) - h^2) u = h I_d(t)
```

and **no `dG_d/dt` term appears**. This is why the realization is in physical
coordinates: making the computational `(s, w)` block's coefficients time
dependent does NOT satisfy the same law — differentiating that block leaves
`-T(A_c' s' + B_c' w)` on the right. Here there is nothing to correct.

**Prospective source, time-dependent version.** The horizon goes INSIDE the
derivative:

```
b = b_bar + d(T(t) b_bar)/dt ,   r = u - b_bar/kappa(t)
=>  M(t) u'' + gamma(t) u' + r + T(t) r' = 0     exactly, no extra terms
```

because `T(t) kappa(t) = c_s` is constant. The alternative `b = b_bar + T b_bar'`
leaves `+(T kappa'/kappa)(u - r)`. The operator form `(1 + D o T)` is therefore
**forced by the algebra**, not chosen for convenience, and the reason is
recorded.

**Tied coefficients**, all functions of the single varying `G_d`:

```
tau_d = c/G_d,  kappa = G_s - h^2/G_d,  T = c/kappa,
gamma = G_s tau_d/kappa,  M = T tau_d,  rho = M/(gamma T)
```

**Level 3, computational mapping.** `r = gamma_0 (J s - beta x)`,
`J = -diag(a)`, `a = Delta*Lambda`, `beta = Delta*B_c`, native clock absorbed
exactly once. This substitution is a **computational extension**: the passive
plant does not prescribe it, and a complex S5 weight is not a nonnegative
conductance. A passive reciprocal two-compartment circuit has **real**
eigenvalues, so the complex pole is learned feedback and is not claimed to be
biological.

**Quadrature sharing.** Conductances are real and nonnegative; a complex mode's
two quadratures are the same compartment and share one real `G_d(t)`.

**Realization integrated:** physical `z = (s, s')`. The map to `(u, v)` is
`v = (c s' + G_s s)/h` with constant coefficients, so there is **no
parameter-dependent coordinate change** and no state correction when the
modulation jumps between tokens. Within a token the modulation is constant, so
the update is an exact matrix exponential and the `x'` term is impulsive at the
boundary, entering as a jump in `s'`. Because the modulation depends only on the
**layer input**, every per-token block is precomputable and the **associative
scan is retained**.

### Declared assumptions, fixed before any validation score

* `c_s = c_d = c = 1`, `g_L = h = g = 2/15`, time in input intervals. Not free:
  these are the unique symmetric-reference values reproducing the contract
  tuple `T = 5`, `tau_d = 3.75`, `gamma_phys = 5`, `M_phys = 18.75`,
  `rho = 3/4`. Enforced by `validate_reference()`.
* Modulation, reusing the EXISTING synaptic weights with **zero added
  parameters**:
  `drive = |beta_p . x_k|`,
  `q = MOD_SCALE * G_d0 * drive/(DRIVE_REF + drive)`, so `0 <= q < G_d0`:
  `G_d` at most doubles and `tau_d` at least halves. Boundedness stands in for
  finite synaptic resources.
  **`MOD_SCALE = 1.0` and `DRIVE_REF = 1.0` are dimensionless calibration
  constants declared here.** They are not from NLA and are **not swept**.
* The current effect is the existing complex drive `beta.x` (the level-3
  computational source); the conductance effect is `q`, from the same weights.
  Both effects are present. **Not claimed:** that the complex drive equals
  `g_E E_E + g_I E_I` with physical reversal potentials.

## 2. Arms

All on the SAME gain-scaled, clipped substrate, identical trainable parameter
counts (verified by test), depth 4, width 32.

| arm | role | state (real, per layer) | new |
|---|---|---|---|
| `gp_adaptive_mass` | **primary treatment** | 2 x 32 | yes |
| `gp_frozen_adaptive` | within-family control: the NEW model with modulation frozen | 2 x 32 | yes |
| `prospective_recurrence` | professor equation `r + T r' = 0`, exact cancellation control | **0** | yes |
| `ordinary_adaptive` | matched ordinary adaptive SSM, SAME modulation information | 1 x 32 | yes |
| `alpha_p_s5` | Rawat prospective-input reference | 1 x 32 + buffer | historical |
| `gain_clip_s5` | matched gain-scaled clipped ordinary reference | 1 x 32 | historical |
| `native_s5` | native S5 reference | 1 x 32 | historical |
| `gp_fixed_mass` | fixed positive-mass generalized recurrence | 2 x 32 | historical |

`gp_frozen_adaptive` exists because the old `gp_fixed_mass` result may only
substitute for it if the two are **identical**, which is a test
(`test_frozen_modulation_reproduces_the_fixed_mass_arm`), not an assumption. It
is counted in the budget either way.

**Cost mismatch, stated:** `ordinary_adaptive` carries one state per mode
against the adaptive treatment's two. It separates general adaptivity from the
prospective law; it does not equalize state or compute.

## 3. Reuse versus rerun of references

Historical Stage 2 scores are reused **only if** the configuration, data
identity, schedule, initialization and comparator code path are unchanged. The
preflight verifies the unchanged paths at parameter, forward and gradient level
against the saved checkpoints. If shared code changed a comparator's behaviour,
that comparator is rerun within the cap or the comparison is reported
incomplete. **Scores from incompatible configurations are never spliced.**

Historical references, all at commit `227648e` with data digest
`73caf4e4...` (train) / `dc4f65d6...` (val), seed 100, 10 epochs, lr 1e-3:
`native_s5` 94.17 %, `alpha_p_s5` 95.00 %, `gain_clip_s5` 94.60 %,
`gp_fixed_mass` 94.26 %.

## 4. Training configuration — unchanged from Stage 2

Cached SC10 MFCC, 161 frames, 20 features, split seed 0, **test split not
opened**. Depth 4, width 32, 8 HiPPO blocks, conjugate symmetry, same
normalization, residual, half-GLU and readout. Batch 32, **full BPTT**, seed
100, lr 1e-3, **the exact existing 10-epoch cosine schedule**, AdamW with the
published settings, highest matmul precision, deterministic XLA.

**No new learning-rate, horizon or physical-scale sweep.** No spatial-only BP,
no separate TSS experiment, no MAML loop.

## 5. Numerical criteria, predeclared

`tests/test_adaptive_recurrence.py` header holds the tolerances:
TIE `1e-12`, FROZEN `1e-10`, ODE `2e-6`, SCAN `1e-10`, GRAD `5e-3`, F32 `1e-4`.

Required coverage: conductance admissibility and boundedness; coefficient ties
including `T kappa = c`; causal modulation; **independent scipy integration of
the UNREDUCED two-compartment circuit with both input and conductance
changing**, plus a guard proving that a frozen-coefficient shortcut would FAIL
that reference; frozen-limit identity; scan-vs-sequential and reset
equivalence; float32/complex64 production dtypes and gradients **through** the
modulation with nothing detached; professor-control zero driven history and
nonzero ordinary gradients; and no added trainable parameters.

**Low accuracy is not a correctness condition for the professor control.** Its
history-cancellation checks run on the isolated core, so pooling and
training-time normalization cannot obscure them.

## 6. Bounded run plan and the budget rule

**HARD TOTAL: 20 minutes of cluster wall time** covering focused checks,
compilation, preflight and training, with a watchdog and cleanup reserve.

1. **Preflight** (inside the budget): focused tests, then two optimizer updates
   per new arm with measured steady-state step time and peak memory.
2. **Projection.** From the measured steady-state time, project the cost of the
   training batch at 843 steps/epoch x 10 epochs per arm, plus validation.
3. **Gate.** Training starts **only if** the checks pass **and** the projection
   fits the remaining budget. If it does not fit, the cost is reported and the
   correctness work is finished; **no arm is silently shortened, no long run is
   started, and the budget is never widened automatically.**

The previous nine-run Stage 2 batch took 716 s, but that is **not** an estimate
for an unimplemented adaptive kernel; only the measured projection decides.

Arms are trained in this fixed order, so a partial batch is interpretable:
`gp_adaptive_mass`, `gp_frozen_adaptive`, `ordinary_adaptive`,
`prospective_recurrence`.

## 7. Screening target and what a result would mean

Retained: at least **+0.3 percentage points** over **both** `alpha_p_s5` and
`gain_clip_s5`. Additionally reported: the comparison against
`ordinary_adaptive`, which is what separates general adaptivity from the
prospective law.

**No single-seed outcome is statistical confirmation.** No automatic larger run
or rescue sweep follows. This screen measures classification under adaptive
dynamics; it does **not** establish meta-learning, causally useful memory
allocation, or performance on unseen temporal regimes. Adaptation variability
alone is not evidence of useful memory selection, and ordinary neural
adaptation is not by itself evidence of meta-learning.
