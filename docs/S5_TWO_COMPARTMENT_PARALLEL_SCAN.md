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

## 7. Certification result (first run, float32 only)

`certification.json`, seeds 301, 302, 303, all six layers, 64 modes each:

```
total_modes_certified   1152
worst_spectral_radius   0.9999748468399048      (bound 1.0)
offending_modes         []
certified               true
```

Every production mode is strictly inside the unit disc. **The float64
column of that run is void** for reason (a) and must be re-measured with
`JAX_ENABLE_X64=1`.

## 8. Benchmark (partial)

Native, one process, batch 16, length 16,000, full forward/backward/optimizer:

```
seconds_per_step   0.11722
steps_per_minute   511.9
peak_device        6.909 GiB
projection         0.262 h/seed, 0.785 h for three seeds at 15 epochs
```

The old sequential arm measured ≈2.98 steps/min, i.e. ≈20.1 s/step —
**≈172× Native's per-step cost**. The remaining rows are pending.

## 9. Gates — status

| gate | status |
|---|---|
| Algebraic equivalence (laptop, 17 tests) | **passed** |
| Byte identity | **passed** |
| Whole-inventory certification, seeds 301–303, radius | **passed** — 1152 modes, worst 0.99997 |
| float64 correctness gate | **not run** — needs `JAX_ENABLE_X64=1`; the first attempt was void |
| JAX float32 equivalence and gradients | **re-run pending** after the fixture fixes |
| Throughput and peak-memory benchmark | **partial** — Native measured, rest pending |
| Launcher for this arm | **not added**, and must not be until the gates above pass |

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
