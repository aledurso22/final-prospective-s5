# Generalized prospective dynamics in S5 — implementation report

File-based coordination interface for the theory coordinator. Keeps CPU
correctness, user-reported GPU evidence, smoke evidence and benchmark evidence
strictly distinct.

| | |
|---|---|
| Implementation checkout | `/Users/alessandrodurso/Documents/final-prospective-s5` |
| Branch | `generalized-prospective-s5` |
| Branched from | `main` @ `156e428` (verified clean before branching) |
| Brief | `CODING_AGENT_BRIEF.md`, sha256 `f9bd9134…cdbf0a` — **verified matches `inspection_record.json`** |
| Milestones | B and C implemented and tested on CPU. A is complete for state (de)serialization and **incomplete for epoch-boundary resume** (R4). D prepared, **not executed**. |
| Evidence level | **CPU correctness only.** No GPU run has been performed. No training benchmark is claimed. |
| Revision | 2, after coordinator review of `2abde98`. Dispositions in §7. |

Branches preserved untouched: `main`, `professor-state-pc`, `prospective-lead`,
`closure-prospectivity`, `learning-access-prospectivity`,
`backup/prospective-wip-20260901`, `upstream/main`. `s5/ssm.py` is **byte
identical** to `main`.

---

## 1. Confirmed interpretation

### 1.1 Clock mapping — absorbed exactly once

`a_i = Delta_i Lambda_i`, `b_i = Delta_i B_tilde_i`, with
`Delta_i = exp(log_step_i)`. Native S5 is then the **unit-interval ZOH** of
`sdot = F0 s + B0 x`. Verified against the stock `discretize_zoh`:

```
max |a_bar - Lambda_bar| = 0.0            (exact)
max |b_bar - B_bar|      = 6.2e-17
```

No second multiplication by `Delta` occurs anywhere. Both `a_i` and `b_i` are
computed *from* `log_step`, so autodiff carries the step gradient through
**both**, as required.

### 1.2 The realized equation

With `gamma = 1`, `step_rescale = 1`, residual `r = J s - B x`, `J = -F0`,
`B = B0`:

```
(I - T F0) sdot = F0 s + B0 x + T B0 xdot
```

Per stored complex mode, shared across its conjugate pair:

```
m_i = 1 - t_i a_i      a_eff_i = a_i/m_i
b_hist_i = b_i/m_i^2   d_x_i   = t_i b_i/m_i
a_bar = exp(a_eff)     b_bar   = phi1(a_eff) b_hist
```

### 1.3 Output timing — the native convention, adopted unchanged

```
h_k = a_bar h_{k-1} + b_bar x_k        h_{-1} = 0
s_k = h_k + D_x x_k                    tied state feedthrough
y_k = 2 Re{C_tilde s_k} + D_native * x_k      (factor 2 iff conj_sym)
```

`x_k` is held over the k-th unit interval; the output is produced **after**
consuming the current token. The initial jump implied by `s = h + D_x x` from a
zero prehistory is included. The native learned featurewise `D` is **retained**
alongside the new tied feedthrough; both are differentiated. Parallel,
streaming, impulse and dense-reference paths all use this one convention, and
it is pinned by tests so a future reset mask cannot change it silently. The
start-of-interval convention used in some theory examples is **not** used, and
no silent comparison against it is made.

### 1.4 Stability

An exact identity, derived and tested:

```
Re(a_eff) = (sigma - t |a|^2) / |m|^2 ,     a = sigma + i omega
```

so for `sigma < 0` and `t >= 0` the effective pole is strictly stable for
**every** admissible `t`. No constraint beyond a stable native pole is needed.
`clip_eigs` is **not** silently changed; its plain default `False` is preserved.

### 1.5 Predeclared validation criteria

Fixed before running. A failure is reported, not loosened.

| Gate | Scope | Tolerance |
|---|---|---|
| `GATE_EXACT` | plain / fixed-zero bypass identity | `atol = 0` |
| `GATE_REF_X64` | coefficients vs dense real-pair reference | `1e-9` normalized |
| `GATE_ZERO_X64` | generalized formula at `t -> 0` vs plain | `1e-12` normalized |
| `GATE_JVP_X64` | directional JVP vs finite differences | `1e-6` relative |
| `GATE_STREAM_X64` | parallel vs streaming, float64 | `1e-11` relative |
| `GATE_STREAM_F32` | parallel vs streaming, float32 production | `1e-5` relative |
| `GATE_C64_VALUE` | complex64 `phi1` value vs complex128 reference | `2e-7` |
| `GATE_C64_DERIV` | complex64 `phi1` derivative vs reference | `2e-7` |

The two 32-bit gates are ~float32 machine epsilon (1.19e-7). No tighter gate is
achievable in complex64 and none is claimed.

These are **new, separately scoped** gates. Historical exact-zero gates in
`docs/GATES.md` remain exact and are not reused or rewritten; that file is
untouched on this branch.

---

## 2. Files added and changed

**Added**

| File | Purpose |
|---|---|
| `s5/gp_coefficients.py` | pure coefficient builders, `phi1`, clock absorption, stability identity |
| `s5/gp_ssm.py` | `GPSSM` + `init_gp_ssm` factory; reuses the untouched scan |
| `s5/gp_diagnostics.py` | poles, margins, DC, direct/history split, impulse response, Hankel SVs |
| `s5/checkpointing.py` | opt-in checkpoints, full-precision JSONL metrics, provenance |
| `experiments/gp/run_diagnostics.py` | Milestone C diagnostics runner |
| `tests/test_gp_prospective.py` | 43 tests — Milestone B gates + theory diagnostics |
| `tests/test_gp_infrastructure.py` | 14 tests — Milestone A + C |
| `bin/run_experiments/gp_provenance.sh`, `gp_gpu_checks.sh`, `gp_smoke.sh` | Milestone D entrypoints |

**Changed**

| File | Change |
|---|---|
| `run_train.py` | `--ssm_mechanism`, `--gp_init_scale`, `--checkpoint_dir` |
| `s5/train.py` | routes through `init_gp_ssm`; opt-in checkpoint/metrics hooks |
| `s5/train_helpers.py` | `gp_response_raw` added to the **ssm** optimizer group |

**`s5/ssm.py` is NOT modified.** `binary_operator` and
`jax.lax.associative_scan` are imported and reused unchanged.

### Optimizer group — a deliberate choice

`gp_response_raw` is assigned to the **`ssm`** group: `optax.adam` at `ssm_lr`,
**no weight decay**.

**Corrected (R7.3):** an earlier version of this note claimed AdamW decay would
drive the response to zero. That is wrong. The stored parameter is *raw*, with
`t = softplus(raw)`, and decay pulls `raw` toward 0 where `softplus(0) = log 2
~ 0.693`. From the `t = 0.05` initialization `raw` is negative, so decay alone
would **increase** the response. The real objection is that decay imposes an
arbitrary preferred response scale unrelated to the task, which is why the
parameter is kept out of the decayed group. The production grouping — not a
re-declared copy of it — is now what the test inspects.

### Parameter counts (H=4, P=4 complex modes, one layer)

| mechanism | params | extra |
|---|---|---|
| `plain` | 80 | — |
| `gp_scalar` | 81 | +1 |
| `gp_diagonal` | 84 | +P |
| `prospective_input` | 81 | +1 |
| `full_state_pc` | 81 | +1 |

---

## 3. Actual test results (CPU, float64 unless stated)

**`pytest tests/` → 70 passed**, comprising 13 pre-existing `test_s5_baseline`,
43 `test_gp_prospective`, 14 `test_gp_infrastructure`.

### Check 1 — frozen plain / fixed-zero bypass

`init_gp_ssm(mechanism='plain')` returns the **original `S5SSM`**, not a
subclass. Parameter tree, forward output, gradients and one optimizer update
are compared with `assert_array_equal` (`atol = 0`) against stock `init_S5SSM`:
**all exact**. No `gp_*` parameter is allocated in the bypass, and random-key
consumption is unchanged.

### Check 2 — independent dense reference

Reference: **dense real-pair augmented matrix exponential via scipy**, built
with linear solves, never touching the closed-form complex algebra.
80 randomized cases covering zero and nonzero imaginary poles, `sigma` over
`10^-4 … 10^0.5`, `t` over `0 … 10`, including the exact zero-`T` limit:

```
max normalized error  a_bar = 1.1e-15
max normalized error  b_bar = 2.7e-13     (gate: 1e-9)
max normalized error  d_x   = 1.4e-16
```

The generalized formula is **also** verified at `t -> 0` against plain
(`1e-12` gate) so the bypass cannot conceal a wrong coefficient builder.
`phi1` value and derivative are checked at small argument: `phi1(0) = 1`,
`phi1'(0) = 0.5` to `1e-9`, and the gradient is finite at `z = 1e-13`.

### Check 3 — directional gradients across the full family

`jax.jvp` vs central finite differences in a random direction over **all**
parameters (`Lambda_re`, `Lambda_im`, `B`, `C`, `D`, `log_step`,
`gp_response_raw`) for all four nonzero mechanisms: relative agreement within
`1e-6`. The response parameter receives a finite, nonzero gradient. The tied
feedthrough is verified **active** — zeroing `D_x` changes the output.

### Check 4 — parallel vs streaming, boundaries, input patterns

4 mechanisms × 4 patterns (random, first-token impulse, constant, alternating)
against a sequential reference using the native timing convention: all within
`1e-11`. No state is carried between independent examples under `vmap`.

**Corrected (R1, R5):** the test previously labelled "float32 production" ran
under a module that enables x64 globally and therefore executed in
float64/complex128 — it never exercised production arithmetic. Genuine
production coverage is now a separate x64-disabled subprocess
(`tests/gp_float32_probe.py`), and carry/reset coverage has been added. See the
review dispositions below.

### Check 5 — stability and the matched-TSS control

300 randomized modes: `Re(a_eff) < 0` and `|a_bar| < 1` for every `t >= 0`
with `sigma < 0`. `full_state_pc` has `b_hist` and `b_bar` **identically
zero**, static branch exactly `-b/a = J^-1 B`, and impulse support **1**.
`prospective_input` leaves `a_eff == a` exactly — the generator does not move —
while `gp_*` does move it.

### Check 6 — rejected configurations

`bidirectional=True`, `discretization='bilinear'`, `step_rescale != 1.0` and
`gp_init_scale = 0` all raise `ValueError` with explicit messages rather than
silently changing the model.

### Milestone A — checkpointing

Restore reproduces a fixed-batch evaluation **and** the next optimizer update
exactly (`assert_array_equal`), for both `plain` and `gp_diagonal`.
Checkpointing consumes **no** randomness (verified: an identical key draws an
identical sample before and after saving). Metrics are written at full float
precision. Run directories are unique. Provenance records commit, branch,
dirty flag, a **diff sha256** identifying which uncommitted state produced a
development run, host, SLURM job id, JAX version, backend, devices and
`XLA_FLAGS`.

**Declared resume boundary:** epoch boundary, in-process, same host and
backend. Mid-epoch resume, cross-process/host continuation and resume across a
different JAX/CUDA build are **not** claimed and **not** checked.

### Milestone C — diagnostics and training smoke

Single-layer linear core, `gp_init_scale = 0.3`, one random initialization:

| mechanism | max&#124;a_bar&#124; | margin | eff. e-fold | history frac | impulse supp | Hankel top | DC |
|---|---|---|---|---|---|---|---|
| `plain` | 0.999435 | 0.0006 | 1769.77 | 0.1144 | 512 | 5.963e-01 | 1.9008 |
| `gp_scalar` | 0.999435 | 0.0006 | 1769.70 | 0.1076 | 512 | 5.822e-01 | 1.9008 |
| `gp_diagonal` | 0.999435 | 0.0006 | 1769.70 | 0.1076 | 512 | 5.822e-01 | 1.9008 |
| `prospective_input` | 0.999435 | 0.0006 | 1769.77 | 0.1142 | 512 | 5.943e-01 | 1.9008 |
| `full_state_pc` | 0.035674 | 0.9643 | 0.30 | 0.0000 | **1** | **0.000** | 1.9008 |

**Two theory diagnostics corroborated numerically, now pinned as tests:**

1. **DC invariance.** `H(0) = J^-1 B` is preserved across *every* mechanism to
   **2.22e-16**, including the matched-TSS control. Cancellation removes the
   history, not the static branch.
2. **A fixture-specific observation, NOT a monotonicity result.** On this
   HiPPO initialization the Hankel top falls as `gp_init_scale` grows
   (`1e-6 -> 0.3 -> 1.0`). **This does not generalize and the earlier claim was
   wrong.** The coordinator's counterexample (`a = -0.1 + 2*pi*i`, `b = C = 1`,
   readout `2 Re`) has the Hankel top *increase* with response —
   `0.0024019 -> 0.0060234 -> 0.0147512` for `t = 0 / 0.01 / 0.1` — with every
   effective pole still stable. Reproduced to `1e-6` as a regression test.
   `sigma_memory = |K0| tau_0/(2 tau)` concerns a fixed scalar *reciprocal
   continuous-time* mode and says nothing about a sampled complex core. No rule
   to maximize or minimize Hankel strength is inferred from either observation.

**Training smoke:** a fixed synthetic batch, 200 steps at `lr = 1e-2`, for
`gp_scalar`, `gp_diagonal` and `prospective_input`. Every declared parameter of
**those three mechanisms** receives a nonzero update, including
`gp_response_raw`, and the loss falls `1.5815 -> 1.8e-4`.

**`full_state_pc` is deliberately excluded** and must not be covered by an
"every mechanism" statement: with `b_hist == 0` its driven map is the static
`D_x = J^-1 B`, which does not depend on the response parameter, so that
parameter has no gradient for zero-history inference. That is a property of the
negative control, not a defect, and no artificial gradient is forced into it.

*Method note, recorded as engineering iteration and NOT as a preregistered
result:* the overfit setup was changed after observing that at the
`create_train_state` default `lr = 1e-3` the loss falls only ~8% in 60 steps.
`lr = 1e-2` and 200 steps were then selected, applied identically to the three
mechanisms. The fixed numerical gates above were predeclared and are separate
from this selected setup.

---

## 4. Bugs found and fixed during implementation

1. **`float()` on a traced value in `setup()`.** `inverse_softplus` was
   implemented with `jnp`; inside `jax.jit` every `jnp` op — even on a Python
   constant — is staged into the jaxpr and returns a tracer, so `float()` raised
   `ConcretizationTypeError` during `train_step`. It passed initial smoke tests
   only because those called `.init()` outside `jit`. Now computed host-side
   with `math`. This is why an eager-only check is insufficient for initializers.

---

## 5. Unresolved issues and explicit non-claims

- **No GPU evidence exists.** Everything above is CPU. The float32 gate is a
  local-CPU float32 check, not a GPU check.
- **No benchmark claim.** No comparison to the 2026 preprint, no sMNIST result,
  no Speech Commands result.
- The one-epoch plain sMNIST regression was **not** completed locally (slow, and
  the harness raced the redirect). Plain identity is nonetheless established
  exactly at the parameter/gradient/update level. The end-to-end regression is
  the first item of the Milestone D GPU batch.
- **Coupled `T`** (§6 of the brief) is not implemented. Its diagonal-scan cost
  claim would not hold and is not made.
- **Causal auxiliaries** (`L`, `Lambda`) are not implemented; `L = 0` throughout.
- **Resolution transfer** is not implemented; non-unit `step_rescale` is rejected.
- The `gp_scalar` and `gp_diagonal` diagnostics above are indistinguishable
  because all modes share one initial scale; they diverge only after training.
- A single random initialization was used for the diagnostics table.

---

## 6. Milestone D — short GPU launch commands (prepared, not executed)

Cluster workflow per the established F-002 protocol. Artifacts go to
`/Users/durso/s5-runs/...` (persistent NFS), **not** node-local scratch. The
shared cluster virtualenv is not modified.

```bash
cd /Local/durso/final-prospective-s5
git -c http.version=HTTP/1.1 fetch --no-tags origin generalized-prospective-s5
git switch generalized-prospective-s5 || git checkout -b generalized-prospective-s5 origin/generalized-prospective-s5
git pull --ff-only && git rev-parse HEAD
```
```bash
source /Local/durso/prospective_ssm_project/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
GP="$HOME/s5-runs/$(date +%Y%m%d-%H%M%S)-gp" && mkdir -p "$GP" && echo "$GP"
```
```bash
./bin/run_experiments/gp_provenance.sh "$GP"
./bin/run_experiments/gp_gpu_checks.sh "$GP"
```
```bash
# (a) frozen HISTORICAL smoke: original unclipped plain configuration
( time ./bin/run_experiments/gp_smoke.sh "$GP/plain" plain ) > "$GP/plain.log" 2>&1 ; echo "exit=$?"
```
```bash
# (b) MATCHED pair: identical explicit stability settings on BOTH arms
( time ./bin/run_experiments/gp_matched_pair.sh "$GP/m_plain" plain ) > "$GP/m_plain.log" 2>&1 ; echo "exit=$?"
( time ./bin/run_experiments/gp_matched_pair.sh "$GP/m_gp" gp_diagonal --gp_init_scale=0.05 ) > "$GP/m_gp.log" 2>&1 ; echo "exit=$?"
```
```bash
grep -hE "Trainable Parameters|Train Loss:|real" "$GP"/plain.log "$GP"/gp_diag.log
python -m experiments.gp.run_diagnostics --outdir "$GP/diagnostics"
```

**Expected of the plain run:** 26,058 parameters and
`Train Loss 1.40519 / Val 0.36394 / 0.8950 / Test 0.33751 / 0.9001`, the
deterministic `E2-004` record.

**Corrected (R7.1):** these historical values are a *sanity check*, not a causal
test. They are rounded console metrics from an earlier environment, so a
mismatch triggers investigation but **does not by itself identify a cause**.
To attribute a regression, run old-`main` and new-`plain` in the *same* verified
environment with identical data order, flags, dtype and deterministic settings,
and compare full-precision metrics.

The `gp_diagonal` run validates integration only. **It is not evidence for the
research claim** and must not be reported as a benchmark comparison.


---

# 7. Coordinator review — dispositions (revision 2)

Reviewed revision `2abde98`. All 70 CPU tests were independently reproduced by
the coordinator, who then found the defects below in separate probes. The
recurrent equation, clock mapping, coefficient algebra and output timing were
approved and are **unchanged**. The original probe evidence in
`generalized_s5_review/` is preserved; the checks here are new regression tests
written against the corrected public behaviour.

**Test totals after the revision: 123 passed** (13 `test_s5_baseline`,
44 `test_gp_prospective`, 14 `test_gp_infrastructure`, 52
`test_gp_review_fixes`), plus the standalone x64-disabled probe.

```
.venv/bin/python -m pytest tests/ -q            # 123 passed
.venv/bin/python tests/gp_float32_probe.py      # X64_DISABLED_OK / DTYPES_OK / PRODUCTION_OK
.venv/bin/python -m experiments.gp.run_diagnostics
```

## R1 — complex64 arithmetic and its validation — **FIXED**

*Defect.* `phi1` switched to its Taylor series at a fixed `1e-4`. In complex64
the direct `(exp(z)-1)/z` branch suffers cancellation, catastrophically so in
the derivative. At `z = -0.000101` the derivative was `-1.914` against a
reference `+0.49997` — **the wrong sign**. Separately, the test advertised as
"float32 production" imported a module enabling x64 globally and actually ran
in float64/complex128, so the gate had never been exercised.

*Fix.* A cancellation-safe complex `expm1` (`expm1(x)cos y - 2 sin^2(y/2)`,
`exp(x) sin y`), a 14-term Horner series, and a **dtype-dependent** switch
(`1.0` for 32-bit, `1e-2` for 64-bit) chosen from measured error rather than
guessed. Thresholds were selected by measuring both error sources; in
complex64 the direct branch's derivative error is `1.0e+03` at `|z|=1e-4`,
`5.2e-2` at `1e-2` and still `1.4e-3` at `0.1`.

*Measured after the fix*, genuine complex64 vs an mpmath/complex128 reference,
over `|z|` from `1e-6` to `5`, on both sides of the switch and with imaginary
components:

| | worst error |
|---|---|
| value | **6.1e-08** |
| derivative | **5.6e-08** |

At the flagged point `z = -0.000101`: value error `2.4e-4 -> 1.3e-8`,
derivative `-1.914 (wrong sign) -> error 1.1e-8`. Both inside the predeclared
`2e-7` gates, which are float32 machine epsilon.

Production coverage is now `tests/gp_float32_probe.py`, run as a **subprocess
with x64 disabled**, asserting `input float32`, `Lambda_re float32`,
`gp_response_raw float32`, `a_bar/b_bar/d_x complex64`, `output float32`, and
checking the parallel path against an independent NumPy complex64 sequential
reference (`rel = 4.7e-08`). The existing x64 gates are retained unchanged.

## R2 — admissibility not enforced — **FIXED**

*Defect.* `Re(a_eff) < 0` is conditional on `Re(a) < 0`. Neither the module nor
the launch enforced it; a probe reached `max|a_bar| = 1.00349 > 1`.

*Fix.* `GPSSM` now **rejects `clip_eigs=False`** with an explicit message. The
historical plain default (`clip_eigs=False`) is untouched.
`bin/run_experiments/gp_matched_pair.sh` runs treatment and control with
`--clip_eigs=True` on **both** arms, and is kept separate from the frozen
historical smoke so a clipped GP run is never compared against an unclipped
plain run.

## R3 — diagnostics described a different system — **FIXED**

*Defect.* `core_from_module` rebuilt `Lambda` from raw parameters and ignored
`clip_eigs`; with raw poles past the boundary it reported `max|a_bar| = 1.0175`
for a model whose realized value was `0.99999977`, and its impulse response was
wrong by `1.4e-2`. It also assumed conjugate symmetry independently of the
module.

*Fix.* Coefficients and readout are now read from the **bound, executed module**
via `module.apply(..., method=...)`, so `setup()` — and therefore clipping — is
what produces them; `conj_sym` is read from the module. Regression test drives
raw poles to `+0.5` with `clip_eigs=True` and asserts the diagnostic Markov
parameters equal the **actual forward** impulse response to `1e-9`, for
`plain` and `gp_diagonal`, and for both conjugate conventions.

## R4 — checkpoint coverage — **PARTLY FIXED, remainder explicitly DEFERRED**

*Defect.* `TrainState.step` was not restored (saved 3, restored 0). The old
test saved before any update and compared only parameters, so it missed this.

*Fixed.* `step` is serialized and restored. New test performs **three real
updates**, saves, restores into a **freshly initialized template with a
different seed and zero moments**, and asserts the parameters, the optimizer
moments, the restored `step == 3`, and that the **next** update agrees — which
Adam bias correction makes step-sensitive.

*Deferred, and now stated in the module docstring rather than implied:* there is
**no wired epoch-resume entrypoint**; the training RNG and dataloader RNG state
are not captured (a seed is not a current state); loop counters (`best_acc`,
early-stop count, lr step) are not captured. This module **loads and stores
state; it does not restart the training loop.** Milestone A is therefore
complete for parameter/optimizer serialization and **incomplete** for
epoch-boundary continuation.

## R5 — missing gates — **FIXED**

*Defect.* The sequential reference was a NumPy loop: it could not validate
gradients, could not take a non-zero carry, and the chunk test only confirmed
that an independent call *loses* history. No reset masks existed. The optimizer
test re-declared its own label rule and would have passed if production
regressed.

*Fix.* Added `gp_scan_sequential` (a differentiable `lax.scan` with `h0` and
`reset_mask`) and `gp_scan_reset` (reset in the **parallel** path, by zeroing
the transition factor at reset indices — no second scan). New tests: parallel
vs sequential agreement in **value and gradient** (`1e-9`); state-carrying
chunks reproducing the full sequence (`1e-12`); non-zero initial carry honoured;
reset masks making the tail identical to an independently run segment, in
**both** scan paths. The no-leakage-between-examples test is kept. The optimizer
test now inspects the **production** rule in `create_train_state`.

## R6 — Hankel claim overstated — **CORRECTED**

The broad claim and the test name are gone. The old test is renamed
`test_fixture_observation_hankel_decreases_on_THIS_configuration` and documents
that it is a fixture-specific observation. The coordinator's counterexample is
added as a **regression test** reproducing `0.0024019 / 0.0060234 / 0.0147512`
to `1e-6`. The counterexample is preserved as a valid theoretical
counterexample, not engineered away; the repository's Hankel utility already
agreed with the independent calculation, so only the claim was wrong.

## R7 — reporting and protocol — **CORRECTED**

1. **Historical regression.** The claim that any deviation from rounded
   `E2-004` metrics proves changed routing is removed; see §6.
2. **GPU checks.** `gp_gpu_checks.sh` now **exits 2** on a CPU-only JAX
   backend (verified locally: `backend=cpu ... FATAL ... exit=2`), records
   backend and devices, and runs the complex64 probe separately. NumPy/SciPy
   references remain CPU computations even inside a GPU run, and are labelled
   as such.
3. **Weight decay.** Corrected in code, test and report: decay of `raw` drives
   `t` toward `log 2 ~ 0.693`, not zero, and from `t = 0.05` would *increase*
   it.
4. **Softplus initialization.** Corrected: softplus is a bijection onto
   `(0, inf)` so no finite raw gives `t = 0` and `inverse_softplus(0)` is not
   finite; `softplus'(0) = 1/2`, so the "zero tangent" argument belongs to a
   squared parameterization, not this one.
5. **Full matched control.** Statements are scoped to the three tested
   mechanisms; `full_state_pc`'s response parameter is inactive for zero-history
   inference and is deliberately excluded.
6. **Provenance and run names.** The dirty identity now hashes `git diff HEAD`
   (covering staged) **plus the bytes of every untracked file**; `make_run_dir`
   uses `mkdtemp`, verified unique for the same tag twice in one second.
7. **Development tuning.** The overfit-smoke selection is recorded as
   engineering iteration, separate from the predeclared numerical gates.

## Remaining scope after this revision

- **No GPU evidence.** Everything above is CPU. The complex64 probe is a
  local-CPU float32 check.
- **Epoch-boundary resume is not implemented** (R4), only state (de)serialization.
- Coupled `T`, causal auxiliaries (`L`, `Lambda`) and resolution transfer remain
  unimplemented; non-unit `step_rescale`, bilinear and bidirectional are rejected.
- `prospective_input` is an **ingredient control**, not a reproduction of
  alpha-P-S5's gain scaling, pole constraints, preprocessing and tuning.
- No benchmark or learning-advantage claim is made.
