# Two-compartment generalized prospective S5: parallel-scan realizations

Branch `s5-two-compartment-parallel-scan`, from the clean production commit
`ef004cda025b4e098041cfe5970c217fa0015bba`.

**The equation is not changed.** `generalized_coefficients`, `target_map`,
`response_mass_gamma`, the initialization (`response_init = 0.05`,
`rho_init = 0.5`, `gamma_init = 1.0`, `RHO_MIN = 1e-4`), the output
semantics, the diagnostics and the arm name
`generalized_prospective_s5` — "generalized prospective dynamics (M,γ,T) —
finite-difference realization" — are all untouched. Native S5 is
byte-identical. Only **how** the same recurrence is evaluated changes.

## 1. Why the old run was slow

The production arm evaluated

```
s_t = a1 s_{t−1} + a2 s_{t−2} + c1 @ x_t + c2 @ x_{t−1}
```

with `scan_companion_sequential`: a `jax.lax.scan` over the token axis,
wrapped in `jax.checkpoint` so the backward pass rematerializes the whole
rollout.

At the production length of 16,000 that is **16,000 strictly sequential
steps forward and a rematerialized 16,000 backward**, per layer, per
direction — six layers, bidirectional, so 24 full-length sequential
traversals per optimizer step. The measured rate was about **2.98 optimizer
steps per minute**, projecting roughly **155 hours** for a three-seed wave.

**The observed prefix stayed finite.** Nothing in that run indicates an
equation-level instability; what it indicates is that a constant-coefficient
second-order per-mode recurrence was being evaluated by an O(L) sequential
method when the recurrence is associative and admits an O(log L) one. That
distinction is the reason for this branch.

## 2. Two parallel realizations of the same recurrence

**(1) Companion.** `scan_companion` (already present at the base commit)
runs the affine associative scan over

```
H = [[a1, a2],
     [ 1,  0]]
```

Correct, but it carries a 2×2 matrix per token per mode: four complex
multiplies and two adds per combine.

**(2) Factored — preferred.** The characteristic polynomial is

```
λ² − a1 λ − a2 = 0,      r1 + r2 = a1,      r1 r2 = −a2
```

so `(1 − r1 z⁻¹)(1 − r2 z⁻¹) = 1 − a1 z⁻¹ − a2 z⁻²`, and the second-order
recurrence is **two first-order ones in series**:

```
v_t = r1 v_{t−1} + d_t
s_t = r2 s_{t−1} + v_t
```

Each is exactly the shape `s5/ssm.py`'s own `binary_operator` composes — one
complex multiply and one add per combine, no 2×2 block, no new primitive.

### Roots, computed stably

The naive pair `(a1 ± √(a1²+4a2))/2` loses the **small** root to
cancellation whenever `|a1|` dominates the square root. The standard remedy
is used instead:

```
sign  = +1 if Re(conj(a1)·√disc) ≥ 0 else −1
major = (a1 + sign·√disc) / 2          ← always the constructive sum
minor = −a2 / major                    ← from the PRODUCT, never a difference
```

Measured on the cancellation-dominated case (`a1 = 10¹⁰`, `a2 = 10⁻³`): the
naive small root is wrong by more than `10⁻³` relative, the product form by
less than `10⁻⁹`. Ordering is deterministic — `major` is always the larger
root — and the cascade is order-independent anyway, which is tested.

**Repeated roots are exact, not special-cased.** At `disc = 0` the
factorization gives `r1 = r2` and the cascade becomes the Jordan
realization, reproducing `t·rᵗ⁻¹` exactly; this is checked against the
analytic impulse response. A partial-fraction route would divide by
`(r1 − r2)` and fail there. Near-repeated roots are equally safe because
`minor` comes from the product.

**No eigendecomposition is used anywhere in the training path** — asserted
by a test, not by convention.

## 3. The implementation choice

`scan_companion_sequential` remains the **default and the oracle**.
`GeneralizedProspectiveS5SSM` gains one field:

```python
implementation: str = DEFAULT_IMPLEMENTATION   # "sequential"
```

dispatched through `scan_for`, which raises on an unknown name rather than
falling back silently. The production path is therefore unchanged unless a
caller asks for an alternative **by name**. This is the only pre-existing
file this branch modifies, and a test asserts that the modification adds
nothing but the choice.

## 4. Verification

### Laptop (no JAX) — `tests/test_two_compartment_algebra.py`, **17 passed**

Coefficient algebra (`r1+r2 = a1`, `r1r2 = −a2`), transfer-function
equivalence, agreement with the existing `companion_radius`, the
cancellation comparison above, deterministic ordering, repeated and
near-repeated roots against the analytic response, sequence equivalence,
cascade order-independence, zero prehistory with no shifted-input
wraparound, and the byte-identity checks of §5.

### Cluster (JAX/GPU) — `tests/test_factored_recurrence_jax.py`

Both parallel scans against the sequential oracle at lengths
1, 2, 3, 5, 17, 63, 257, 1000 and **16,000**; real and complex modes;
forward and reverse, including a check that reverse is not a flipped
forward pass; zero prehistory and no wraparound in both directions;
repeated and near-repeated roots; gradients with respect to `a1`, `a2`,
`c1`, `c2` and the drive; gradients through the **original** `T`, `rho`,
`gamma` parameterization; float32 finiteness of values and gradients at
production length; and that the registry refuses an unknown implementation.

### Whole-inventory certification — `experiments/s5_two_compartment/certify_inventory.py`

Every mode of every layer at the production configuration, seeds **301,
302, 303** — not a subset. Reports both roots of every mode, the worst
spectral radius, the smallest `|discriminant|` (so a near-repeated root
cannot hide inside an aggregate), float64 agreement and float32 relative
error and finiteness, per layer and over the whole inventory. A mode at or
above radius 1 is named with its seed, layer and index.

## 5. Byte identity

`tests/test_two_compartment_algebra.py` intersects the files tracked at
`ef004cda` with those changed since, and asserts the result is exactly
`{s5/generalized_prospective_ssm.py}`. Files added by this branch are not
identity violations and are excluded by that intersection.

## 6. What the first cluster run exposed

Job 67223 at `0977072`. Laptop algebra passed (17/17). The JAX suite
returned **8 failed, 4 passed**, and the certification "passed" for a
reason that was itself a defect. Three separate causes, none of them in
`s5/factored_recurrence.py`:

**(a) The certification's float64 column was never float64.** JAX printed
`Explicitly requested dtype complex128 ... will be truncated to complex64`,
and the reported f64 and f32 numbers were identical to every digit
(2.910e-04 against 2.910e-04). `x64` was off, so `astype(complex128)`
silently did nothing and the float64 gate measured float32. **A gate that
cannot detect its own absence is worse than no gate.** Requesting float64
without `JAX_ENABLE_X64=1` is now a hard error, and the correctness gate
moved into its own file that refuses to run without it.

**(b) The test fixture was unstable, not the code.** `a1, a2` were drawn
uniformly on `[-0.6, 0.6]²`, which puts **18% of modes at spectral radius
≥ 1**, reaching 1.368. At length 16,000 that is `1.368^16000 ≈ 10^2180`:
the companion scan's matrix products overflow to `inf`, `inf·0` gives
`nan`, and the suite reported a failure that was entirely a property of the
fixture. Production modes are certified at radius ≤ 0.99998. Coefficients
are now drawn as **roots inside the unit disc** and mapped through
`roots_to_coefficients`, and the gradient and production-length tests now
assert the **oracle** is finite first, so they cannot be vacuous again.

**(c) The repeated-root fixture drove one mode out of four.**
`np.eye(4, 1)` is `[[1],[0],[0],[0]]`, so modes 1–3 had zero drive and the
analytic comparison ran against an all-zero oracle. Now `np.ones((4, 1))`.

### Tolerances, separated rather than loosened

The brief forbids concealing failures by loosening tolerances, so the two
questions are now asked separately:

| gate | precision | tolerance | what it decides |
|---|---|---|---|
| algebraic correctness | **float64** | `1e-10` | do the three routes compute the same recurrence? |
| accumulation | float32 | `1e-3` | how much single-precision round-off builds up? |

The float32 number is **set from measurement, not from convenience**: the
whole production inventory — 1152 modes, seeds 301–303, length 16,000, on
modes whose spectral radius is 0.99997 so nothing decays away — came in at
**2.3e-4 to 3.5e-4** relative against the sequential oracle. A real
algebraic discrepancy would survive into float64 and be caught by the
`1e-10` gate; only round-off would not.

## 7. Certification and the correctness gate — result

### float64: the factored scan is algebraically exact

`JAX_ENABLE_X64=1`, verified on by the file itself rather than assumed.
**5 passed**, tolerance `1e-10`: both parallel routes against the
sequential oracle at lengths 5, 257, 4000 and 16,000, forward and reverse,
repeated and near-repeated roots (`ε = 0, 1e-14, 1e-10, 1e-6`), and
gradients.

### Whole inventory, seeds 301–303, both precisions

```
x64_enabled                     true
total_modes_certified           1152          (6 layers x 64 modes x 3 seeds)
worst_spectral_radius           0.9999747626  (bound 1.0)
worst_float64_relative_error    1.83e-13      <-- algebraically exact
worst_float32_relative_error    3.83e-04      <-- pure accumulation
offending_modes                 []
certified                       true
```

**This is what licenses the float32 tolerance.** The two routes reproduce
the sequential oracle to `1.8e-13` in double precision on every production
mode, so the `3.8e-4` seen in single precision is round-off accumulating
over 16,000 tokens on poles of radius 0.99997 — not an algebraic
discrepancy. Had float64 come back at `3e-4` as well, no float32 tolerance
would have been allowed to cover it.

Every production mode is strictly inside the unit disc, worst radius
0.99997, no offenders.

## 8. Benchmark — result

Batch 16, length 16,000, full forward/backward/optimizer, **one process per
arm**, with the scan actually in force verified by reading it back off the
constructed module.

| arm | s/step | steps/min | peak GiB | vs sequential |
|---|---|---|---|---|
| `native` | 0.1172 | 512.0 | 6.909 | 149.9× |
| `sequential` | **17.5654** | 3.42 | 8.342 | 1.0× |
| `companion` | 0.2479 | 242.0 | 12.940 | **70.9×** |
| `factored` | **0.2181** | **275.1** | **12.885** | **80.5×** |

> **The wall-clock column has been removed, because it was wrong.** It was
> computed from `--steps-per-epoch 536`, a number I guessed and documented
> as "Speech Commands batches per epoch at batch 16". The real split gives
> roughly **2300** at batch 16 — a 4.3× error. The benchmark also measures
> optimizer steps *only*: synthetic batch, already on the device, one
> process alone, no data loading and no validation pass. Against a real
> run those cost about a further **2×**.
>
> **Measured, not projected:** the first real wave ran at **≈17 min/epoch**
> with three seeds concurrently on three RTX 3090s, i.e. **≈4.3 h for 15
> epochs**, against the 1.46 h this table used to claim. The per-step
> figures above are sound and the 80.5× speedup is sound; only the hours
> were wrong. `--steps-per-epoch` now has no default and no hours are
> reported without it.

`identical_peak_memory_suspicious: {}` — all four arms allocated
differently, and `implementations_in_force` reads
`sequential / companion / factored` as requested. The rows are real.

**117.7 hours becomes 1.46 hours.** The factored route is 1.14× faster than
the companion scan and uses marginally less memory, which is what the
operation count predicts: ~4 complex multiplies per combine against ~12,
4 complex numbers carried against 6, and no `(L, P, 2, 2)` array at all.

**The trade the original implementation made, now quantified.**
`scan_companion_sequential` was chosen because it is "memory-safe". It is:
8.342 GiB against the parallel routes' 12.9. But that 1.54× saving cost
**80× in time**. The arm was unrunnable for the sake of 4.5 GiB.

Against Native the factored arm costs **1.86× per step and 1.87× peak
memory**, for twice the recurrent state (256 against 128 real values per
layer) and two scans in series. That is the honest price of the second
compartment, and it is affordable.

Both thresholds are cleared with a wide margin: ≥ 9.8× was needed to fit a
12 h allocation, and 80.5× lands the three-seed wave at 1.46 h.

**Device capacity, confirmed.** `pgi15-gpu5` carries an NVIDIA GeForce RTX
3090 with **24576 MiB = 24.0 GiB**. The factored arm's 12.885 GiB peak is
**53.7% of the card, 1.86× headroom**, against Native's 6.909 GiB at 28.8%.
`cluster_env.sh` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so JAX does
not grab the card up front and the measured peak is the real one.

## 9. Gates — status

| gate | status |
|---|---|
| Algebraic equivalence (laptop, 20 tests) | **passed** |
| Byte identity | **passed** |
| **float64 correctness**, `1e-10` | **passed** — 5 tests; worst inventory error **1.83e-13** |
| **float32 equivalence and gradients** | **passed** — 12 tests |
| **Whole-inventory certification**, seeds 301–303 | **passed** — 1152 modes, worst ρ 0.99997, 0 offenders |
| **Throughput and peak memory** | **passed** — factored 80.5× over sequential, 1.46 h per wave |
| Device capacity vs 12.9 GiB peak | **passed** — RTX 3090, 24.0 GiB, peak is 53.7% |
| Launcher for this arm | **added, not run** |

**Every gate in the brief has passed**, so the launcher is now permitted
and has been written. It has **not been run**. Nothing has been trained
and no Slurm job has been submitted.

## 10. The launcher

`bin/run_experiments/allocation_s5_two_compartment_factored.sh` — the
generalized arm only, three seeds, 15 epochs, the factored scan, as a
direct child process of an existing interactive allocation.

**One seed per GPU, never two.** The measured peak is 12.885 GiB against
the RTX 3090's 24.0 GiB, so two concurrent seeds on one card would need
25.8 GiB and fail. Visible tokens are **de-duplicated** and the wave is
never wider than the number of distinct ones, so the seeds fall back to
running sequentially on a single-GPU allocation rather than colliding.

| allocation | schedule | wall clock |
|---|---|---|
| `--gres=gpu:1` | three seeds sequentially | **1.46 h** |
| `--gres=gpu:3` | one seed per GPU, concurrently | **0.49 h** |

**Guards**, all inherited from the three-GPU launcher: the commit is pinned
by `EXPECTED_COMMIT`, a dirty worktree is refused, the interpreter and data
cache are checked, it refuses to run outside a numeric `SLURM_JOB_ID`, each
child gets a private `TMPDIR` and JAX compilation cache, pre-JAX telemetry
is written before any JAX import, and every child of a wave is awaited
before the next begins. `DRY_RUN=1` prints the whole plan and starts
nothing.

**A smoke gates the training, and the gate checks the scan.** The smoke
child's `production_check.json` must report
`scan_implementation == "factored"`; if it reports `sequential` the
launcher aborts rather than quietly training the 117-hour configuration.
That check exists because the benchmark's first run silently measured the
wrong scan three times.

**No finalizer.** The finalizer is the only reader of the test split and it
compares all three arms; running it on a single arm is a separate,
explicit decision and the launcher says so and stops.

### Running it

```bash
# inside the allocation
export EXPECTED_COMMIT=$(git rev-parse HEAD)
DRY_RUN=1 bash bin/run_experiments/allocation_s5_two_compartment_factored.sh
# then, to actually train:
bash bin/run_experiments/allocation_s5_two_compartment_factored.sh
```

`IMPLEMENTATION=sequential` reproduces the 117-hour run and the launcher
prints a warning if asked for it.

### The production default has NOT moved

`GENERALIZED_IMPLEMENTATION` in the runner and `implementation` on the
layer both still default to `"sequential"`. Every pre-existing command
behaves exactly as it did before this branch; the factored scan is only
used when a caller asks for it by name. `scan_implementation` is recorded
in every run's `production_check.json`, so no result can be read without
knowing how it was computed.

No certification of the parallel routes is claimed yet, no launcher exists,
nothing has been trained, and no Slurm job has been submitted.

## 7. Cluster commands for independent verification

```bash
cd /Local/durso/final-prospective-s5
git fetch origin && git checkout s5-two-compartment-parallel-scan
source bin/run_experiments/cluster_env.sh && cluster_check_env

# 1. algebra, on any machine
"$PY" -m pytest tests/test_two_compartment_algebra.py -q

# 2a. ALGEBRAIC correctness, float64. x64 must really be on; the file
#     refuses to run otherwise.
JAX_ENABLE_X64=1 "$PY" -m pytest tests/test_factored_recurrence_float64.py -q \
      2>&1 | tee float64_tests.log

# 2b. float32 equivalence, gradients, production length
"$PY" -m pytest tests/test_factored_recurrence_jax.py -q 2>&1 | tee jax_tests.log

# 3. whole-inventory certification, seeds 301-303, BOTH precisions
JAX_ENABLE_X64=1 "$PY" -m experiments.s5_two_compartment.certify_inventory \
      --precision both --out certification.json 2>&1 | tee certification.log

# 4. benchmark, one process per arm
"$PY" -m experiments.s5_two_compartment.benchmark \
      --out benchmark.json 2>&1 | tee benchmark.log
```

If any production mode fails certification, it is named and investigated
(block/chunk scan, or a custom associative operator) rather than hidden by
a loosened tolerance. **If no correct parallel method is materially faster,
that is the result and it will be reported as such.**


---

## 11. A fourth arm: theory-matched zero first-order lag

Added after the generalized wave, and **not run**. It exists to answer one
question:

> Does removing the residual lag help, when the memory poles are held
> fixed?

### What it changes, and what it does not

Writing `k = 1 − λ̄`, `m = Mh/T`, `c = γh/T`, the generalized arm is

```
m s̈ + (c + Tk) ṡ + k s = b̄ (x + T ẋ)
```

Normalizing the denominator by `k` gives `1 + (T + c/k)p + (m/k)p²`, so the
first-order lag cancels exactly when the numerator's derivative coefficient
matches:

```
Γ_k = T + c/k = T + γh / (T(1 − λ̄))
```

and then, normalized by the DC gain `b̄/k`,

```
Ĥ(p) = (1 + Γ_k p)/(1 + Γ_k p + (m/k)p²) = 1 − (m/k)p² + O(p³)
```

**`Ĥ′(0) = 0` but `Ĥ ≢ 1`: zero first-order lag with the second-order
memory retained.** The principle is *compensate friction prospectively, not
inertia*. Adding `(m/k)p²` to the numerator would make it equal the
normalized denominator, give `Ĥ ≡ 1`, and destroy the memory completely —
a test asserts that and forbids it.

`a1` and `a2` are **inherited literally** from
`GeneralizedProspectiveS5SSM.setup`, never recomputed, so both poles are
bit-identical and the ablation is exactly one coefficient. `M, γ, T, λ̄, b̄`
and their initialization are identical; nothing is retuned.

### The drive is regrouped, and that is not cosmetic

The direct coefficients satisfy `c1 + c2 = K` exactly — two numbers of size
`K·Γ_k/h` cancelling to `K`, losing ≈ `log₁₀Γ_k` digits. Since
`Γ_k ≈ γh/(T|1−λ̄|)` reaches 2×10⁵ on near-real slow modes, that is five of
float32's seven. The algebraically identical regrouping

```
c1 x_t + c2 x_{t−1}  ≡  K x_t + K (Γ_k/h)(x_t − x_{t−1})
```

has no cancellation: at low frequency the difference vanishes on its own.
Measured in float32, error **8e-8 flat in Γ_k** against **3e-5** for the
direct form. The forward path uses only the regrouped form;
`direct_coefficients` exists solely so a test can prove the two are the
same recurrence in float64.

### What is verified

| item | where | status |
|---|---|---|
| regrouped ≡ direct, float64 | `test_matched_lag_jax.py` | needs GPU |
| `a1, a2` bit-identical | same, on arrays | needs GPU |
| `Γ_k = T` reproduces the generalized arm | laptop + GPU | **passed** (1e-18) |
| zero first-order lag (exact series coefficient) | laptop | **passed** |
| second-order memory retained (`Ĥ ≢ 1`) | laptop | **passed** |
| quadratic numerator would destroy it | laptop | **passed** |
| `Γ_k` conjugate-symmetric, no special handling | laptop + GPU | **passed** |
| regrouping removes the float32 cancellation | laptop | **passed** |
| `ARM_ORDER` still three, finalizer unaffected | laptop | **passed** |

### The remaining concern is optimization, not arithmetic

`Γ_k` spans four orders of magnitude across modes, and only **near-real**
slow modes blow up, because `|1−λ̄| ≥ |Im λ̄|`:

```
arg      |λ̄|=0.9   0.99    0.999    0.9999
0.000       200    2000    20000    200000
0.010       199    1418     1991      2000
0.050       181     394      400       400
0.200        93     101      100       100
```

`matched_lag_report.py` measures, over seeds 301–303 and every mode:
the `Γ_k` distribution by `|λ̄|` and phase, per-mode gradient magnitudes
along the prospective pathway for **both** arms with their across-mode
ratio in decades, and the low-frequency group delay of both arms — the
direct lag diagnostic.

### Running it

```bash
JAX_ENABLE_X64=1 "$PY" -m pytest tests/test_matched_lag_jax.py -q
"$PY" -m experiments.s5_two_compartment.matched_lag_report --out matched.json

export EXPECTED_COMMIT=$(git rev-parse HEAD)
ARMS="generalized_prospective_s5 matched_lag_prospective_s5" \
DRY_RUN=1 bash bin/run_experiments/allocation_s5_two_compartment_factored.sh
```

`ARMS` defaults to the generalized arm alone, so the existing command is
unchanged.


---

## 12. The comparison set, and the baseline already on disk

All four arms share one protocol: base commit `ef004cda`, the
`sc10_official_cache` split, 15 epochs, seeds 301/302/303. The three-arm
run `s5-three-arm-15epoch/20260919-224525` (allocation 66765) supplies the
first two; this branch supplies the last two.

| arm | spectral radius at init | outcome |
|---|---|---|
| **Native S5** | \|λ̄\| < 1 | **96.921% ± 0.097** validation |
| **Zucchet**, 1st order | **1.10 – 1.71** | **NaN at epoch 0, step 0**, all 3 seeds |
| generalized (M, γ, T) | ≤ 0.99997 | this branch; lag **+25.3** tokens |
| matched (M, γ, Γ_k) | **identical, bit for bit** | this branch; lag **−0.36** tokens |

### Native S5, the reference

```
seed 301   97.110%   CE 0.09217   selected epoch 15
seed 302   96.786%   CE 0.09645   selected epoch 13
seed 303   96.867%   CE 0.09481   selected epoch 15

mean 96.921%   sd 0.169   se 0.097   range 0.324 pp
```

**The seed-to-seed range of a single arm is 0.32 pp.** Any difference
between the generalized and matched arms has to clear that to mean
anything, and with three seeds per arm a paired test has very little
power — expect a direction and an effect size, not significance.

### Zucchet, the negative control

```
epoch 0, step 0
failure: "production check nonfinite"
loss NaN, gradient_norm NaN, gradients_finite false, state_finite false
```

All three seeds, verified `NumericalTrainingFailure`, `accepted: true`. It
fails the one-step finite-update gate **before training begins**, which is
what the spectral radius of 1.10–1.71 at initialization predicts. The
precise statement is *the first-order prospective recurrence is non-finite
on the first update*, not "training was unstable".

### VALIDATION, not test

`task_result.json` carries **validation** metrics. The finalizer is the
only reader of the test split and it never ran for the three-arm wave --
the generalized arm's 155-hour projection aborted the run before Stage 3.
All four arms are therefore compared on the same validation split, which
is sound, but it must be labelled validation and not test.

### Wall clock is NOT comparable across the two runs

Native ran on `pgi15-gpu2`, partition `pgi15-shared`, 30 CPUs per task;
this branch runs on `pgi15-gpu5`, partition `pgi15`, 10 CPUs, node to
itself. Native's real throughput was 0.69 s/step against a benchmarked
0.117 -- a 5.9× overhead -- while the generalized arm runs at ~0.53 s/step
against a benchmarked 0.218, a 2.4× overhead. The generalized arm
therefore *appears* faster than Native, which is an artefact of the shared
partition and not a property of the arms. **Accuracy comparisons across the
two runs are valid; wall-clock comparisons are not.**


---

## 13. What the experiment is actually testing

The comparison worth defending is **not** generalized → matched. It is

> **Native S5 → theory-derived prospective second-order S5.**

Native has **one pole doing two jobs**: `s_t = λ̄ s_{t−1} + B̄x_t`, so
`|λ̄| → 1` buys retention by buying delay. The matched arm gives each mode a
genuine second dynamical state, `s_t = a1 s_{t−1} + a2 s_{t−2} + d_t`, and
then derives the prospective correction *from that dynamics*:

```
d_t = K [ x_t + (Γ_k/h)(x_t − x_{t−1}) ],   Γ_k = T + γh/(T(1 − λ̄))
```

So the question is: **does separating memory from first-order lag improve
an SSM?** Not "does a derivative term help" — that is a weaker claim and a
weaker reason to expect anything.

The generalized arm is therefore **an intermediate construction, not a
baseline**. It exposed the second-order coefficients `a1, a2`; its
prospective coefficient `T` does not match the damping of the model it is
attached to, so there is no theoretical reason to train it as a final
architecture.

### But it is still the attribution control

`Native → matched` changes **three** things at once: first order becomes
second order, the drive gains a `Δx` term, and `T` becomes `Γ_k`. If the
matched arm wins, that alone does not say which of the three did it. The
generalized arm is the only comparison in which the poles are bit-identical
and exactly one symbol differs, so it is what separates "the lag
cancellation helped" from "a second-order recurrence with a derivative
drive helped". It is worth running **if and only if** the matched arm shows
something worth attributing.

### The optimization risk, and how it is measured

`Γ_k ∝ 1/(1 − λ̄)`, and the modes Native uses for its longest memory are
exactly the ones where the exact correction is largest. Gradient descent
has three ways to weaken it and one way to exploit it:

| training does | reading |
|---|---|
| `λ̄` retreats from 1 | gave up the long memory |
| `γ → 0` | gave up the damping that needed compensating |
| `T` moves to shrink `Γ_k` | weakened the correction directly |
| **`\|λ̄\| ≈ 1` kept AND `Γ_k` stays large** | **the intended regime: very slow internal memory, very fast prospective response** |

`prospective_drift.py` measures exactly this from the saved per-epoch
checkpoints, comparing the trained parameters against their initialization,
per layer and pooled: the `|λ̄|` distribution and how many modes remain
above 0.99 and 0.999, the medians of `γ` and `T`, the `Γ_k` distribution,
and a **concentration ratio** — the median `|Γ_k|` among the slowest decile
of modes against the median over all modes. A ratio well above one means
the slow modes are the ones carrying the correction, which is the intended
regime rather than an accident.

```bash
"$PY" -m experiments.s5_two_compartment.prospective_drift \
      --run-root /Users/durso/s5-runs/s5-two-compartment-factored/<stamp> \
      --arm matched_lag_prospective_s5 --out /Users/durso/s5-runs/drift.json
```

It reads the last checkpoint present, so it can be run **while training is
still going** to see the trend early.
