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
- **On sMNIST's suitability.** sMNIST *is* a sequential task and a model may
  well need history to solve it; an earlier phrasing implying otherwise is
  withdrawn. The reason it is not the right instrument here is narrower: it
  does not **isolate** current-input processing from delayed recall, because
  the two are entangled in a single pooled classification target. The
  cue/distractor/recall task in `tasks/cue_recall.py` separates them by
  construction (a per-token current-input head and a recall target scored only
  at the query token).
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
.venv/bin/python -m pytest tests/ -q            # 124 passed  (revision 3)
.venv/bin/python tests/gp_float32_probe.py      # X64_DISABLED_OK / DTYPES_OK / PRODUCTION_OK
.venv/bin/python -m experiments.gp.run_diagnostics
```

**Correction.** On revision `88f8ad9` the third command **did not pass**: it
printed the `plain` row and then exited 1, because the runner still built every
mechanism with `ssm_kwargs()` (default `clip_eigs=False`) while the new R2 guard
correctly rejects that for GP. The R2 fix broke the runner and it was not re-run
before reporting. Fixed in revision 3; the actual output is in §8.

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


---

# 8. Follow-up review of 88f8ad9 — revision 3

The coordinator independently reproduced all 123 tests on `88f8ad9`, the
x64-disabled probe (`rel = 4.748e-8`), the small-pole derivative
(`1.1233e-8` at `z = -0.000101`), and the CPU rejection of `gp_gpu_checks.sh`
(exit 2). One functional defect and several wording/scope items were raised.

**Test total after revision 3: 124 passed.**

## F1 — diagnostics runner crashed — **FIXED**

`experiments/gp/run_diagnostics.py:47` built every mechanism with
`ssm_kwargs()`, whose default is `clip_eigs=False`; the R2 guard then rejected
the first GP mechanism, so the documented command emitted only the `plain` row
and exited 1. The coordinator's one-line patch (`ssm_kwargs(clip_eigs=True)`)
was verified with `git apply --check`, applied unmodified, and the CLI re-run.

**Actual output, revision 3, `--init-scale 0.3`, one random initialization:**

| mechanism | max&#124;a_bar&#124; | margin | eff. e-fold | history frac | impulse supp | Hankel top | DC |
|---|---|---|---|---|---|---|---|
| `plain` | 0.999435 | 0.0006 | 1769.77 | 0.1144 | 512 | 5.9626e-01 | 1.9008 |
| `gp_scalar` | 0.999435 | 0.0006 | 1769.70 | 0.1076 | 512 | 5.8219e-01 | 1.9008 |
| `gp_diagonal` | 0.999435 | 0.0006 | 1769.70 | 0.1076 | 512 | 5.8219e-01 | 1.9008 |
| `prospective_input` | 0.999435 | 0.0006 | 1769.77 | 0.1142 | 512 | 5.9432e-01 | 1.9008 |
| `full_state_pc` | 0.035674 | 0.9643 | 0.30 | 0.0000 | **1** | **0.0000** | 1.9008 |

All five mechanisms complete. The matched cancellation control retains zero
history and impulse support 1. DC gain is invariant across all five.

These values are **identical** to those recorded before the R2 guard, because
the HiPPO-initialized poles are already strictly stable, so `clip_eigs=True`
does not bind at initialization. It would bind only if training pushed a raw
pole past the boundary — which is exactly the case the guard exists for.

## F2 — resume wording — **CORRECTED**

`s5/checkpointing.py` no longer says "DECLARED RESUME BOUNDARY". It now states
that it **serializes and restores training state** and **does not resume a
training loop**, with the checked property expressed as reproducing a
fixed-batch evaluation and the next update *given externally supplied inputs
and randomness*. The checkpoint metadata key `resume_boundary` is replaced by
`scope` carrying the same sentence. Epoch-boundary resume stays deferred, which
is acceptable for the short fresh runs planned.

**Also corrected:** restoring `TrainState.step` was reported as what repaired
Adam's bias correction. That is wrong — optax Adam keeps its own count inside
`opt_state`, so restoring `opt_state` preserves the bias correction. Restoring
`TrainState.step` is a separate, independently necessary fix. Both are done and
the code comment now says so.

## F3 — optimizer test was a text search — **FIXED**

Replaced with a **behavioural** test that exercises the real optimizer built by
`create_train_state`: with an all-zero gradient and fresh moments, an Adam
(no-decay) group leaves parameters exactly unchanged, while an AdamW group
still moves them by `-lr * weight_decay * param`. The test asserts the response
parameter's update is **exactly 0.0** and includes a control asserting that a
decayed parameter did move — otherwise the test could not distinguish the
groups at all.

## F4 — plain diagnostic scope — **FIXED**

`core_from_module` now raises `NotImplementedError` for **bidirectional** models
(a reverse scan gives `C_tilde` 2P columns and needs a separate formulation) and
for **bilinear** discretization (the continuous rate is not `log(pole)/step`, so
`effective_pole_real` would mix two notions of rate). The utility is causal-ZOH
only and no longer advertised beyond that. The planned runs are unidirectional
ZOH.

## F5 — precision/backend wording — **CORRECTED**

The misnamed test is renamed
`test_6_parallel_vs_streaming_under_the_suite_x64_setting`, with a docstring
saying it is **not** a float32 gate. Genuine complex64 coverage remains the
separate subprocess. The probe asserts x64 is off and **does not force it off**,
so `JAX_ENABLE_X64=1` makes it fail loudly rather than silently relabel float64
numbers; the runbook note is in the probe itself: invoke with `JAX_ENABLE_X64`
unset. Correcting the earlier over-broad statement: **GPU-side JAX x64 tests are
not automatically CPU computations** — only the NumPy/SciPy references are
CPU-only. Each process should record its observed backend.

## F6 — instability probe wording — **CORRECTED**

The `clip_eigs` rejection message said an unconstrained run "was measured to
reach" `|a_bar| = 1.0035`. That value came from a **constructed
admissible-API configuration with an unstable raw pole**, not an observed
training run. The message now says so explicitly.

## Evidence status, kept separate

> **HISTORICAL TABLE — state as of revision 3, superseded.** Retained as a
> record of what was known then. The current status is the table at the end of
> §10.

| Class | Status |
|---|---|
| CPU correctness | 124 tests + x64-disabled probe + diagnostics CLI, all passing locally |
| **GPU correctness** | **not yet run** |
| **Integration smokes** | **not yet run** (three short runs, purposes kept separate) |
| **Benchmark / learning advantage** | **not attempted, not claimed** |


---

# 9. GPU correctness gate — first execution (revision 3)

**Run on the RTX 3090, `pgi15-gpu3`, commit `39242b1`, clean tree, artifacts
`/Users/durso/s5-runs/20260914-151709-gp/`.** Environment:
`CUDA_VISIBLE_DEVICES=0`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`WANDB_MODE=offline`, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`,
`JAX_ENABLE_X64` unset.

`backend=gpu devices=[CudaDevice(id=0)]`, `GPU_BACKEND_OK`.

## Result: 123 passed, 1 FAILED

```
tests/test_gp_review_fixes.py::test_R1_production_float32_subprocess_asserts_dtypes
  AssertionError: 7.497774e-05          (predeclared gate: 1e-5)
1 failed, 123 passed in 222.78s
```

Every x64 gate, both complex64 `phi1` gates, the clipped-diagnostics checks,
the checkpoint/step restoration, the differentiable streaming and reset checks,
and the Hankel counterexample **all passed on GPU**. The single failure was the
production parallel-vs-sequential comparison.

## Diagnosis — measured, not assumed

| condition | rel error |
|---|---|
| CPU, backend default | 4.748e-08 |
| **GPU, backend default** | **7.498e-05** |
| **GPU, `JAX_DEFAULT_MATMUL_PRECISION=highest`** | **4.748e-08** |

**What this number is.** It is the relative discrepancy of **one fixture**: a
length-24 sequence through the **linear core alone**, compared against a NumPy
complex64 reference that consumes the **same already-rounded coefficients**.
It is therefore a comparison of two float32 evaluations of the same recurrence,
not a bound on the distance between the implementation and exact arithmetic,
and not a bound on error accumulated through a full training run (different
length, different magnitudes, nonlinearities, normalization, optimizer state).

At controlled precision the GPU reproduces the CPU value exactly, digit for
digit. That establishes that **this fixture is precision-sensitive** and that
the backend's default float32 matmul mode (TF32 on Ampere) accounts for the
discrepancy **here**. It does **not** universally rule out scan-order effects:
a reduction-order difference whose effect is itself of matmul-precision size
would also be removed by raising precision, and other shapes or lengths are not
covered by this single fixture. `--xla_gpu_deterministic_ops=true` was set
throughout and does not affect TF32.

## Protocol change, stated explicitly

Stated plainly: **the original gate, evaluated at the backend default
precision, FAILED** (7.498e-05 against a `1e-5` tolerance). It was not
retroactively loosened — the tolerance is unchanged — but the gate was
**re-scoped** to a different measurement condition, and **the newly scoped
highest-precision gate is the one that passed**. Both facts are reported.

The probe now makes two distinct measurements:

1. **GATE — implementation correctness**, evaluated under
   `jax.default_matmul_precision("highest")`, tolerance unchanged at `1e-5`.
   This measures our algebra rather than the backend's default matmul mode.
2. **RECORD — backend characteristic**, at the backend default, reported and
   **not gated**, because reduced-precision matmul is a property of the
   hardware path and not of this code.

## What this does and does not mean for the smokes

TF32 here is **deterministic**: `E2-004` showed four paired runs bit-identical
under it. So this is **not** run-to-run noise and does not reintroduce `F-002`.

What it does mean is narrow: on **this fixture**, two float32 evaluations of
the same linear recurrence differ by ~7.5e-5 relative when TF32 is enabled.
It is **not** an exact-arithmetic error bar for a trained model, and it must
**not** be placed next to training losses or accuracies to argue that some
observed difference is or is not TF32. Ruling TF32 in or out for a training
comparison requires re-running that comparison at controlled precision, which
is why the new controlled experiment (`docs/GP_EXPERIMENT_PROTOCOL.md` §7)
fixes highest precision on both arms.

**Recommendation for the three integration smokes: keep the backend default.**
The historical `E2-002`/`E2-004` records were produced under it, and changing
precision now would break comparability with the frozen historical smoke. The
deviation is recorded here so any later effect size can be judged against it.

Also fixed in this revision: `gp_gpu_checks.sh` used `set -e`, so a pytest
failure aborted the script before its summary printed, leaving only an exit
code. It now captures the status and always prints the log tail.

## GPU correctness gate, re-run at `be225d4`: **PASSED**

`backend=gpu devices=[CudaDevice(id=0)]`, `GPU_BACKEND_OK`, **124 passed in
224.99s**, `pytest exit=0`. Probe on GPU:

```
parallel vs sequential, complex64, HIGHEST matmul:  rel = 4.748e-08   (gate 1e-5)
parallel vs sequential, complex64, backend DEFAULT: rel = 7.498e-05   (recorded)
```

Artifacts: `/Users/durso/s5-runs/20260914-152851-gp/`.

## Smoke attempt at `be225d4`: **CRASHED — my bug**

Both smokes exited 1 after ~24 s, before the epoch loop:

```
File "s5/train.py", line 182, in train
    x.size for x in jax.tree_util.tree_leaves(state.params)),
NameError: name 'jax' is not defined
```

`s5/train.py` imports `jax.numpy as np` and `from jax import random` but never
`import jax`, while the checkpoint hook I added calls
`jax.tree_util.tree_leaves`.

**Why the tests missed it.** `save_checkpoint`/`restore_checkpoint` were unit
tested against a state built by `create_train_state`, but `run_train.py` was
**never run with `--checkpoint_dir`** — the one path both smoke scripts use.
Same class of gap as the diagnostics runner: a feature validated in isolation
and never exercised through its documented entrypoint.

**Fixes.** `import jax` added. A regression test
(`test_F7_train_module_has_every_name_its_hooks_use`) asserts the names the
hooks depend on are bound in `s5.train`. Reproduced end to end locally on CPU
before pushing: `exit=0`, full epoch, checkpoint and metrics artifacts written
(`config.json`, `best/last.msgpack`, `*.meta.json`, `metrics.jsonl` at full
precision, `step=843`).

**Separately:** `gp_smoke.sh` no longer passes `--checkpoint_dir`. The
`E2-002`/`E2-004` records were produced without checkpointing, so supplying it
made the "frozen" historical run not actually frozen. Matched runs
(`gp_matched_pair.sh`) still enable it.

*Local CPU end-to-end validation of the fix (integration evidence, NOT a
benchmark and NOT comparable to the GPU records):* `gp_diagonal`,
`gp_init_scale=0.05`, `clip_eigs=True`, seed 1919, 1 epoch — train loss
1.41934, test accuracy 0.8906, 26,058 + P parameters.


---

# 10. Integration smokes — first successful execution

Commit `91f4988`, RTX 3090 `pgi15-gpu3`, clean tree, artifacts
`/Users/durso/s5-runs/20260914-154718-gp/`. Environment as in §9
(`XLA_FLAGS=--xla_gpu_deterministic_ops=true`, backend default matmul
precision). All three `exit=0`.

| run | script | clip_eigs | params | train loss | val loss | val acc | test loss | test acc | wall |
|---|---|---|---|---|---|---|---|---|---|
| 1 historical plain | `gp_smoke.sh` | False (historical) | 26,058 | 1.40522 | 0.36394 | 0.8950 | 0.33751 | 0.9001 | 67.4 s |
| 2 matched plain | `gp_matched_pair.sh` | **True** | 26,058 | 1.40522 | 0.36394 | 0.8950 | 0.33751 | 0.9001 | 67.7 s |
| 3 matched `gp_diagonal` | `gp_matched_pair.sh` | **True** | **26,122** | 1.41597 | 0.39181 | 0.8870 | 0.36801 | 0.8908 | 71.6 s |

## Run 1 vs the historical record

Run 1 reproduces `E2-004` on **every printed digit**. `E2-004`'s full-precision
train loss `1.4052175283432007` prints as `1.40522`; val `0.36394 / 0.8950`,
test `0.33751 / 0.9001` likewise.

**Correction to an earlier instruction.** The expectation was stated as
"Train Loss 1.40519". That is `E2-002`'s value — the *ordinary*,
non-deterministic run — not `E2-004`'s. **`E2-002` was itself a GPU run**, not
a CPU-era one; an earlier version of this paragraph said otherwise and is
corrected here. The two records differ in their determinism protocol, not in
their backend. The two differ in the fifth
decimal and `E2-004` is the correct comparator for a deterministic GPU run.
The mistake was in the instruction, not in the result.

Per R7.1 this remains a **sanity check**, not a causal test: agreement is
consistent with unchanged routing but does not by itself prove it.

## Run 1 vs run 2 — does clipping bind?

**No difference is visible at the reported precision** on any logged metric.
That is consistent with the clip never activating — the HiPPO-initialized poles
are strictly stable, and the diagnostics in §8 saw no active clipping — but
matching rounded metrics is **not proof** that clipping never activated during
training: a clip that bound briefly, or bound on modes that barely affect the
loss, could leave the five-digit summaries unchanged. The runs did not log
clipping statistics, so the question is **unresolved by this evidence**.
New runs under `docs/GP_EXPERIMENT_PROTOCOL.md` §6 log per-step clipping and
stability statistics so this can be answered directly instead of inferred.

## Run 2 vs run 3 — the only comparable pair

Identical seed, data order, architecture, clipping and flags. `gp_diagonal`
adds **+64 parameters** (P = 32 complex modes x 2 layers), i.e. +0.25 %.

| metric | matched plain | matched `gp_diagonal` | delta |
|---|---|---|---|
| train loss | 1.40522 | 1.41597 | **+0.01075** |
| val loss | 0.36394 | 0.39181 | +0.02787 |
| val accuracy | 0.8950 | 0.8870 | **-0.0080** |
| test loss | 0.33751 | 0.36801 | +0.03050 |
| test accuracy | 0.9001 | 0.8908 | **-0.0093** |
| wall clock | 67.7 s | 71.6 s | +5.8 % |

**On this smoke the generalized mechanism is WORSE than plain on every
metric**, despite having more parameters. Stated plainly rather than buried.

Every delta is two to three orders of magnitude above the 7.5e-5
reduced-precision deviation recorded in §9, so the differences are resolvable
and are not an artifact of TF32.

## What this does and does not establish

**Does:** the generalized recurrent mechanism trains end to end through the
real S5 stack on GPU under the deterministic protocol, at a cost of
+0.25 % parameters and the wall-clock difference recorded below.

**Withdrawn from this paragraph:** "with BPTT reaching the response
parameters". That is true of the *gradient* — it is established by the CPU
directional-gradient tests in §3, Check 3 — but it was **not** demonstrated
*from this GPU smoke*, which saved no initial/final checkpoints and logged no
response values. Attributing it to the smoke was unsupported. The new runner
records initial and final response parameters per run
(`docs/GP_EXPERIMENT_PROTOCOL.md` §6), so the next runs can show the change
directly rather than assert it.
That was the purpose of the integration smokes and it is met.

The **+5.8 % wall clock is a single pair of timings** from one run each on a
shared cluster node. It is an observation about these two runs, not a measured
overhead of the mechanism; no repeated timing, no isolation of the node, and no
variance estimate were obtained.

**Does not:** support a general claim about whether the mechanism helps. But
the earlier wording here was too strong in the other direction, and is
withdrawn. This run **is** evidence — **unfavourable evidence for the
configuration actually tested**: at one epoch, one seed, d_model 64, 2 layers,
`gp_init_scale = 0.05`, sMNIST, `gp_diagonal` was worse than the matched plain
control on every recorded metric.

What limits it is **scope, not direction**: it is a single configuration with
no tuning on either arm, so it does not generalize to other sizes, budgets or
initializations. The following claims are **removed** as unsupported:

- that it "is not evidence" about whether the mechanism helps — it is, for this
  configuration;
- that the result is *necessarily* an early-optimization transient — no
  longer-horizon run was performed to establish that;
- that the mechanism *would be expected* to need a different learning-rate or
  initialization regime — that is a hypothesis, and it is precisely what the
  predeclared development budget in `docs/GP_EXPERIMENT_PROTOCOL.md` §4 tests,
  not something this run establishes.

The honest statement is: *integration works; the one configuration tested is
unfavourable to the mechanism; a controlled temporal task with a predeclared,
equal tuning budget is the next step, and its outcome is not prejudged.*

## Evidence classes, kept distinct

Each row names the commit the evidence was actually produced at. These are
**different commits** and the counts are **not interchangeable**: the GPU suite
was 124 tests at `be225d4`. It has **not** been re-run at any later commit, so
nothing here may be relabelled as a larger GPU test count at HEAD.

| Class | Count | Commit evidence was produced at | Status |
|---|---|---|---|
| CPU correctness (revision 3) | 125 tests + x64-disabled probe + diagnostics CLI | `2b454ed` | PASSED, local CPU |
| **GPU correctness** | **124 tests** | **`be225d4`** | **PASSED**, exit 0, user-executed on RTX 3090 (§9) |
| Integration smokes | 3 runs | `91f4988` | **PASSED**, exit 0, user-executed (this section) |
| Benchmark / learning advantage | — | — | not attempted; the one configuration tested is unfavourable |

The report text describing these results is committed separately at `2b454ed`;
a report commit is not evidence of a re-run.

---

# 11. Revision 4 — controlled task, matched control, finite-inertia prototype

Implementing `NEXT_CODING_BRIEF_2026_09_14.md`. Nothing in this section is GPU
evidence; every number below is a **local CPU** result.

## 11.1 Implementation checkout

| item | value |
|---|---|
| checkout | `/Users/alessandrodurso/Documents/final-prospective-s5` |
| branch | `generalized-prospective-s5` |
| base commit | `2b454ed273a8621e8599a42b28dd16516fb86039` |
| public repo | `https://github.com/aledurso22/final-prospective-s5` |
| backend | CPU (`darwin`), x64 enabled unless a probe disables it |
| Python / JAX / Flax / optax | 3.14.4 / 0.11.1 / 0.12.9 / 0.2.8 |

`gp_diagonal` is **unchanged**. It remains the valid `M = 0, gamma = 1` member
of the extended family, and full BPTT is retained everywhere. Nothing added in
this revision replaces it.

## 11.2 Equations as implemented

**Clock absorption**, applied exactly once, `a_i = Delta_i lambda_i`,
`b_i = Delta_i Btilde_i`, so native S5 is the unit-interval ZOH of
`sdot = F0 s + B0 x`.

**Generalized first order** (`s5/gp_coefficients.py`, unchanged):

```
(I - T F0) sdot = F0 s + B0 x + T B0 xdot
m      = 1 - t a          a_eff = a / m        b_hist = b / m^2
d_x    = t b / m          abar  = exp(a_eff)   bbar   = phi1(a_eff) b_hist
h_k    = abar h_{k-1} + bbar x_k ,  h_{-1} = 0
s_k    = h_k + d_x x_k
y_k    = 2 Re{ Ctilde s_k } + D x_k
Re(a_eff) = (sigma - t |a|^2) / |m|^2  < 0  whenever sigma < 0, t >= 0
```

**Matched conventional modal SSM** (`s5/modal_ssm.py`, NEW). Same ODE class as
ordinary S5 written in modal form, with a **static** input feedthrough inside
the readout so the parameter count matches exactly:

```
d   = 1 - t a
alpha = a / d        B_h = b / d^2        z = t d
hdot  = alpha h + B_h x
y     = 2 Re{ C [ h + diag(z) B_h x ] } + D x
```

Parameters are `alpha_re, alpha_im, z_re, z_im, B, C, D` — **no `log_step`**;
the clock is absorbed at initialization, so this arm has the same temporal
degrees of freedom as `gp_diagonal` but **no prospective coupling**: `z` is a
free static gain, not `t` tied to the recurrence.

**Finite inertia, second order** (`s5/gp_second_order.py`, NEW prototype):

```
mu sddot + (1 + t j) sdot + j s = b x + t b xdot ,   j = -a ,   0 < mu <= t
w = mu sdot + t (j s - b x)
d/dt [s; w] = F [s; w] + B x ,
F = [[ -t j / mu , 1/mu ], [ (t/mu - 1) j , -1/mu ]] ,
B = [ t b / mu , (1 - t/mu) b ]
```

Discretized by **`expm` on the augmented matrix** under `jax.vmap` — never by
eigendecomposition, so defective/near-defective blocks are handled. The scan is
a 2x2 block affine associative scan with
`(A_i, b_i) o (A_j, b_j) = (A_j A_i, A_j b_i + b_j)`. The readout takes the
first component `s`; at positive mass there is **no `D_x` term**, because the
derivative feedthrough is produced by the dynamics rather than added
algebraically.

## 11.3 Controlled task

`tasks/cue_recall.py`: length 256, 12 channels — `[0:8]` payload +-1 resampled
at every token, `[8:10]` two current +-1 bits, `[10]` cue flag, `[11]` query
flag. Cue ~ U{8..31}; delay bucket uniform over
`(16,31) (32,63) (64,95) (96,128)`; query = cue + delay, max 159 < 256.
Recall target = the 8 cue bits, scored **only at the query token**. Current
target = XOR of the two current bits, scored at **every** token.
`L = L_recall + L_current`, separately averaged. Splits TRAIN/DEV/TEST/EXTRAP
use disjoint folded key domains; `EXTRAP_DELAYS = (192, 256, 384)` at length
512 and is **never** used for selection.

`s5/tokenwise_model.py` wraps the existing `StackedEncoderModel` with an
8-logit recall head and a 1-logit current head, token-wise — **no pooling and
no temporal batchnorm**, so no information crosses tokens outside the SSM.

## 11.4 Actual test results (local CPU)

```
.venv/bin/python -m pytest tests/ -q     ->  178 passed in 162.66s
.venv/bin/python tests/gp_float32_probe.py  ->  X64_DISABLED_OK DTYPES_OK PRODUCTION_OK
.venv/bin/python tests/so_float32_probe.py  ->  SO_PRODUCTION_OK
```

| file | tests |
|---|---|
| `tests/test_s5_baseline.py` (and pre-existing) | 13 |
| `tests/test_gp_prospective.py` | 44 |
| `tests/test_gp_infrastructure.py` | 14 |
| `tests/test_gp_review_fixes.py` | 53 |
| `tests/test_cue_recall_task.py` (NEW) | 38 |
| `tests/test_gp_second_order.py` (NEW) | 15 |
| **total** | **178** |

Previous total was 125; this revision adds 53. (An intermediate run of this
suite gave 163; the extra 15 are the runner entry-point tests added after the
bug in 11.4a was found.)

### 11.4a Two bugs found by running the entry point, not the units

**The runner crashed on every mechanism.** `_update` was `jit`-ed with only
`model` static; the optax `GradientTransformation` `tx` — a NamedTuple of
functions — was passed as a traced argument, so jit tried to abstractify it:
`TypeError: Cannot interpret value of type <class 'function'> as an abstract
array ... at path tx.init`. Every component test passed while the entry point
could not complete a single update. Fixed by `static_argnums=(3, 4)`, and the
eval path is now jitted the same way. `tests/test_cue_recall_task.py` now runs
`main()` end-to-end for all six mechanisms. This is the **same class of miss**
as the earlier `import jax` crash: units tested, entry point never run.

**The diagnostic described a different system than the one that ran** — review
item R3 recurring in new code. My first `spectral_stats` read `alpha_re`
directly for `modal_ssm` and reported `|abar| = 1.00076`, i.e. a marginally
*unstable* control arm. `ModalSSM.coefficients()` applies the same
`Re(alpha) <= -1e-4` clip as the GP arms, so the executed model was strictly
stable throughout; the *diagnostic* was wrong, not the model. Fixed to read the
clipped pole, with a regression test asserting no arm is ever reported with
`|abar| >= 1`.

**A finding this produced.** With the corrected diagnostic, `clip_eigs`
**does** activate for `modal_ssm`: after two updates, 1 of 16 modes in layer 1
is clipped (`clipped_fraction = 0.0625`), while the GP arms show `0` clipped at
the same point. This is exactly the statistic §10 could not answer from rounded
metrics, and it is now logged per run. It is also an asymmetry to watch: the
control arm reaches the stability boundary and the treatment arm does not, and
that must be reported alongside any difference in loss rather than discovered
afterwards.

Measured checks worth naming:

- **Parameter match.** `gp_diagonal` 2,144 params vs `modal_ssm` 2,144;
  temporal parameters 64 vs 64 (4P at P = 16). Exact, not approximate.
- **Conversion fidelity.** `gp_to_modal` reproduces the generalized coefficients
  to `abar` error `0.0`, `bbar` error `0.0`, `d_x` error `2.42e-19` (float64).
- **Every arm runs the entry point.** `plain`, `modal_ssm`, `gp_diagonal`,
  `gp_scalar`, `prospective_input`, `full_state_pc` and
  `gp_diagonal --freeze_response` each complete updates and write
  `summary.json`. `--freeze_response` is verified to leave the response
  bit-identical while other parameters move.
- **Task/oracle agreement.** The generator is verified against an independent
  oracle; cue and query flags sum to 1; maximum query index 159 < 256.
- **Second order vs first order.** The `mu -> 0` limit recovers the
  generalized first-order response, with the measured breakdown table below.
- **Block scan vs sequential.** The associative block scan matches `lax.scan`.

**Measured tiny-mass breakdown** (this is why `MU_RATIO_MIN = 1e-4`):

| `mu/t` | 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|---|
| error vs first order | 1.5e-2 | 1.5e-3 | 1.4e-4 | 1.4e-5 | 1.4e-6 | **NaN** |

The `1/mu` entries of `F` overflow before the limit is reached. The floor is
set at `1e-4`, one decade inside the last value that was measured to work, and
it is a **measured** bound, not a guess.

## 11.5 Limitations of this revision — stated, not deferred

- **No GPU evidence for anything in this section.** All 178 tests are local
  CPU. The GPU suite remains 124 tests at `be225d4` and has not been re-run.
- **No learning result on the new task.** Not one training run has been
  executed on `cue_recall`, on any arm. The protocol is predeclared precisely
  so that it cannot be adjusted after seeing one.
- **The second-order model is a prototype and is NOT wired into the
  comparison.** It has correctness tests and a measured numerical floor; it has
  no training evidence, no stability analysis across the full admissible
  region, and no gates of its own yet. It must not enter the primary
  comparison until it does.
- **`modal_ssm` is parameter-matched, not capability-matched.** Matching counts
  does not prove the two arms have equal expressive capacity on this task; it
  removes the crudest confound, nothing more.
- **Scalar `prospective_input` is not capacity-matched to `gp_diagonal`** and
  is compared only against `gp_scalar`. It remains an ingredient/placement
  control, not a reproduction of the preprint's full alpha-P-S5 recipe.
- **Peak memory is not recorded on CPU.** `memory_stats()` returns `None` on
  the CPU backend, so `peak_memory_bytes` is `null` in every local run. The
  protocol requires it; it will only be populated on the GPU runs.
- **Coupled `T`** (brief §6) is still not implemented; its diagonal-scan cost
  claim is still not made.
- **The development budget is deliberately unequal in one respect**, stated in
  advance: `gp_diagonal` spends half its four configurations on response
  initialization and therefore explores two learning rates where the other arms
  explore four. Declared in `docs/GP_EXPERIMENT_PROTOCOL.md` §4 before any run.

## 11.6 Short cluster commands (prepared, NOT executed)

I have no cluster access; these are for the user to run. One line each.

```bash
git fetch origin && git checkout generalized-prospective-s5 && git pull
```
```bash
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
```
```bash
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -5
```
```bash
.venv/bin/python tests/gp_float32_probe.py 2>&1 | tail -8
```
```bash
.venv/bin/python tests/so_float32_probe.py 2>&1 | tail -5
```
```bash
.venv/bin/python -m experiments.gp.cue_recall_runner --mechanism plain --updates 750 --seed 0 --lr 1e-3 --ssm_lr 1e-3 --matmul_precision highest
```
```bash
.venv/bin/python -m experiments.gp.cue_recall_runner --mechanism modal_ssm --updates 750 --seed 0 --lr 1e-3 --ssm_lr 1e-3 --matmul_precision highest
```
```bash
.venv/bin/python -m experiments.gp.cue_recall_runner --mechanism gp_diagonal --updates 750 --seed 0 --lr 1e-3 --ssm_lr 1e-3 --gp_init_scale 0.05 --matmul_precision highest
```

Remaining development candidates are the rest of the table in
`docs/GP_EXPERIMENT_PROTOCOL.md` §4 — same command, changed `--lr`, `--ssm_lr`,
`--gp_init_scale`. Confirmation runs use `--updates 2000 --batch 64 --seed 1..5`
on each primary arm. **Report every configuration that is run, including the
ones that lose.**

---

# 12. Revision 5 — fixed physical coefficients, Rawat reference, cluster prep

Implementing `docs/handoff_2026_09_15/CLUSTER_CODING_BRIEF.md` under
`DERIVATION_CONTRACT.md`, both copied into the repository at
`docs/handoff_2026_09_15/`.

**No GPU evidence exists in this revision. No training was run, locally or on
the cluster. No cluster job was submitted.** Everything below is local CPU
static verification and preparation.

## 12.1 Checkout

| item | value |
|---|---|
| checkout | `/Users/alessandrodurso/Documents/final-prospective-s5` |
| branch | `cluster-gp-rawat` (new; `generalized-prospective-s5` untouched) |
| base commit | `a4c5b12599c247923ee0425c25a09d92e8556ce3` |
| Python / JAX / Flax / optax | 3.14.4 / 0.11.1 / 0.12.9 / 0.2.8 |
| backend | CPU only |

Nothing existing was replaced: `gp_diagonal`, `modal_ssm`, the second-order
prototype, the cue/recall task and every historical result remain as they were.
The new arms are additional mechanisms with new names.

## 12.2 Coefficient policy, as implemented

Resolved by the user as "just do as in nla and/or tss": the added physical
coefficients are **fixed declared hyperparameters**; ordinary S5 parameters,
including the native learned steps, train with full BPTT.

`s5/physical_coefficients.py` holds them in a **frozen dataclass**, not in a
trainable parameter with a nominal zero learning rate:

```
T = 5 intervals      gamma_n = 1      rho = 3/4
mu = M/gamma = 3.75  M_physical = 18.75   gamma_physical = 5
```

Derived from the symmetric intrinsic reference (`c_s = c_d = c`, `g_L = h = g`,
no background synaptic conductance), with the tie `M = rho gamma T` enforced by
`validate()`. The full circuit-to-code chain is `docs/DERIVATION_TRACE.md`.

Verified behaviourally, not by inspection:

* for both fixed arms the parameter tree is exactly
  `{B, C, D, Lambda_im, Lambda_re, log_step}` — **no response parameter
  exists**, so no gradient, optimizer slot or weight-decay term can reach the
  physical coefficients;
* the runner independently **refuses to start** if a response-like parameter
  appears (`assert_response_is_not_trainable`);
* and the converse, so the claim is not empty: **changing `T` or `rho` changes
  the model output**.

## 12.3 Equations implemented

M = 0 (`gp_fixed_m0`), general solve form and the executed scalar reduction:

```
W = gamma I + T J    D_x = W^-1 T B    F = -W^-1 J    B_h = W^-1(B - J D_x)
J = -a diagonal, T = 5I scalar:
  W = gamma - T a    F = a/W    D_x = T b/W    B_h = gamma b/W^2
  a_bar = exp(F)     b_bar = phi1(F) B_h
  h_k = a_bar h_{k-1} + b_bar x_k ,  s_k = h_k + D_x x_k
```

M > 0 (`gp_fixed_mass`), carry `z = (s, v)`:

```
gamma rho s' = -J s - gamma(1-rho) v + B x        T v' = s' - v
A = [[-J/(gamma rho), -(1-rho)I/rho], [-T^-1 J/(gamma rho), -T^-1/rho]]
B_blk = [B/(gamma rho) ; T^-1 B/(gamma rho)]
expm([[A, I2],[0,0]]) = [[A_bar, Phi],[0,I]] ,  B_bar = Phi B_blk
z_k = A_bar z_{k-1} + B_bar x_k ,  s_k = z_k[0] ,  no D_x at positive mass
```

Per the brief, the prototype's `(2+H)x(2+H)` augmented exponential is replaced
by a **per-mode 4x4** exponential for transition and integral, then multiplied
by the 2xH drive, so cost no longer scales with feature width. `expm`
throughout; never an eigendecomposition.

## 12.4 Actual test results (local CPU)

```
.venv/bin/python -m pytest tests/ -q             ->  212 passed in 143.72s
.venv/bin/python tests/cluster_float32_probe.py  ->  CLUSTER_FLOAT32_OK
```

212 at the time of that local run, up from 178 at `a4c5b12`: 33 new in
`tests/test_cluster_gp.py` plus one resumable-checkpoint round-trip test. Four
further dataset front-end tests were added afterwards (see 12.10), bringing the
suite to **216**, which is the count the GPU gate ran.

New file `tests/test_cluster_gp.py`, **33 tests**, tolerances predeclared in
the module header and unchanged since: ALGEBRAIC 1e-10, ODE 1e-6, GRADIENT
1e-5, FLOAT32 1e-4.

Measured results worth naming:

| check | result |
|---|---|
| `gp_fixed_m0` vs the audited M=0 law at `t = T` | `a_bar` and `b_bar` differ by **0.0**; `d_x` by 1.0e-15 |
| `rho = 1`, zero prehistory, vs ordinary S5 | **6.2e-16** (float64), **3.9e-07** (float32) |
| `(s,v)` block vs the target law, SciPy | satisfied at `M = 3.75`, derivatives taken from the ODE |
| block scan vs sequential | < 1e-10, all three fixtures |
| chunked streaming vs full sequence | < 1e-10 |
| reset / nonzero initialization | < 1e-10 |
| repeated (coalescing) poles | value and gradient both finite |
| parameter and **clock** gradients vs central differences | within 1e-5 |
| `native_s5` vs upstream `S5SSM` | params identical; output max|diff| **0.0** |
| alpha-P two-tap vs sequential reference | **4.4e-16** |
| production float32, worst of three arms | **3.9e-07** against a 1e-4 gate |
| trainable parameter count, all five arms | **identical** |

**Float32 note.** The brief records that the old dense positive-mass code
failed pure float32 checks. The per-mode 4x4 formulation passes at 3.9e-07.
That is a per-fixture agreement between two precisions of the same recurrence;
it is **not** an exact-arithmetic error bound and not a bound for a training
run, and it must not be compared against losses or accuracies.

## 12.5 Rawat reference

`docs/RAWAT_BASELINE_MAP.md`. The linked reference repository
`Sequel-Institute/prospective-rqf` returns **404 — repository not found**, so
**this port is paper-based and no reference commit can be pinned**; that status
is recorded rather than worked around. The paper itself was retrieved and
Appendix E.3/E.4 and Table 6 are quoted verbatim in the map.

Implemented as published: `B_c = diag(-Re lambda) B_tilde`; pole clipping at
`-1e-4` before forming alpha; `B_plus = B_bar + 5 diag(Delta) A_bar B_c`,
`B_minus = -5 diag(Delta) A_bar B_c`; delayed input starting at zero;
instantaneous `D`; no trainable parameters added.

**Architecture reconstruction check:** depth 4, width 32, MFCC gives
**35,050 parameters** against the paper's reported 35.1k.

Recorded differences and findings, all in the map: the half-GLU variant and MLP
dropout placement are unspecified and declared; weight decay on SSM parameters
follows the **paper**, which differs from upstream S5; the published MFCC
configuration leaves **2 of 64 mel filters empty**; and the stratified
70/15/15 split is **not speaker-disjoint**, unlike the official Speech Commands
lists, which inflates absolute accuracy but is shared by every arm.

Because the published alpha-P-S5 row changes gain, clipping and the second tap
together — as the paper states — a fifth arm `gain_clip_s5` isolates the tap.

## 12.6 Runner infrastructure: both reported gaps fixed

**Checkpointing.** `cue_recall_runner.py` imported `save_checkpoint` and never
called it. It now writes an **atomic** `best` checkpoint at every improvement,
so the selected parameters can actually be re-evaluated later; a summary scalar
is not a checkpoint. `s5/checkpointing.py` gained atomic writes (temp file plus
`os.replace`) and a real `loop_state`: epoch/step counters, dropout RNG,
selection and early-stopping counters. Data order needs no RNG snapshot because
it is a pure function of `(data_seed, epoch)`. **A resumed epoch restarts at its
first batch** — stated in the docstring and in the checkpoint metadata rather
than implied.

The R4 test that asserted resume was DEFERRED was **updated, not deleted**: the
honest record is that the scope changed, and a new behavioural test round-trips
a resumable checkpoint.

**State counts.** `n_state_coords = 2 * SSM_SIZE_BASE * N_LAYERS` double
counted. With conjugate symmetry a layer stores `P = SSM_SIZE_BASE/2` complex
modes, which is `2P = SSM_SIZE_BASE` real coordinates per layer. The two-layer
width-32 model has **64** recurrent real coordinates, not 128. Counts are now
derived from the executed carry and reported per kind — physical, auxiliary,
previous-input buffer — never summed into one number.

**Pipelines.** `bin/run_experiments/cluster_gpu_checks.sh` saves the log,
captures the real process status and only then prints a tail; it never reports
`tail`'s exit code as success.

**Nonlinear end-to-end checks include the executed runner:** all five arms run
`main()` end to end, plus train, resume and evaluate modes.

## 12.7 Two bugs found by running things, recorded

**Resume crashed on the first attempt.** `_to_jsonable` did not recurse into
containers, so `loop_state["best"]` was serialized as the *text* of a dict and
the resume path failed with `TypeError: string indices must be integers`. Fixed
by recursing; covered by the new round-trip test. Found by actually resuming a
run, not by reading the code.

**Four tests in the new suite failed on first execution.** All four were
defects in the tests, not the model: `float()` inside a differentiated
function, a duplicated `bidirectional` keyword, and — the one worth keeping —
a central second difference whose `O(h^2)` truncation error exceeded the
predeclared ODE tolerance at the repeated-pole fixture. **The tolerance was not
loosened.** The measurement was replaced with derivatives taken from the ODE
itself, which is what the test was supposed to check.

## 12.8 Limitations, stated before any training

* **No GPU evidence, no training, no cluster job, no benchmark result.** The
  earlier GPU evidence remains attached to its own commits: 124 tests at
  `be225d4`, smokes at `91f4988`. Nothing here may be relabelled as GPU
  evidence at this commit.
* The measured throughput and memory numbers in the protocol are **CPU,
  synthetic-data, structural indications**, not cluster measurements. Peak
  memory is `null` on CPU because the backend exposes no `memory_stats`.
* `gp_fixed_m0` is a **reduced/constitutive ablation**, not the fast-dendrite
  circuit limit, which sends gamma and M to zero together.
* The Rawat port is **paper-based**; the reference implementation could not be
  inspected.
* Only the identical-cell `T = 5I` model is implemented. No full-matrix
  coupled-`T` mechanism, and no arbitrary fixed off-diagonal elements were
  added to manufacture a coupled candidate.
* The scalar-`rho` stability construction is **not** extended to learned mass
  matrices.
* **The derivation gap is open and is recorded before training:** the passive
  plant supplies the tied coefficients but does not prescribe the prospective
  closed-loop source. Substituting a learned S5 residual is a computational
  extension; a nonsymmetric S5 residual is not automatically the gradient of
  the original NLA mismatch energy. Stability here rests on the declared
  constraints (`T` SPD, `sym(J) > 0` via native clipping, scalar `gamma > 0`,
  scalar `rho` in (0,1]), not on a global biological derivation. Per the
  contract this gap is **reported, not patched with a trainable correction**.
* The September-14 cue/recall protocol remains **predeclared and unexecuted**.
  This batch does not run it.
* Three confirmation seeds are preliminary and differ from the paper's
  five-seed aggregate.

## 12.9 Evidence classes at this commit

| class | scope | where | status |
|---|---|---|---|
| Historical local positives (learned response) | width-64 and depth-2 archived runs | earlier reports | historical, not reproduced here |
| Unsuccessful fixed-coefficient confirmation | pole-preserving study, 30.24% worse, 0/3 paired wins | earlier reports | preserved, unfavourable |
| CPU numerical verification | this revision | `tests/test_cluster_gp.py` (33), float32 probe | PASSED locally |
| GPU correctness | 124 tests | `be225d4` | PASSED, earlier commit |
| Integration smokes | 3 runs | `91f4988` | PASSED, earlier commit |
| GPU integration for the new arms | — | — | **not run** |
| Validation screening | — | — | **not run** |
| Final benchmark comparison | — | — | **not run** |


## 12.10 GPU correctness gate — EXECUTED AND PASSED

First real GPU evidence for this revision. Run by the user on the cluster;
I have no cluster access and did not execute it.

| item | value |
|---|---|
| tested tree | **`129f73e0e36c06c7c2df6ca8c697bcfc9dbaf739`** |
| host | `pgi15-gpu3.iff.kfa-juelich.de`, RTX 3090 |
| SLURM job | 65870, `CUDA_VISIBLE_DEVICES=0` |
| backend | `gpu`, `CudaDevice(id=0)`, jax 0.11.0 |
| artifacts | `/Users/durso/s5-runs/gpu_checks/20260915-152008/` |
| result | **216 passed in 866.81 s, exit 0** |
| `GPU_CHECKS_EXIT` | **0** |

Commits `4dac0ea` and `1f88a10` came after this run. They changed **only**
`docs/GP_RAWAT_CLUSTER_PROTOCOL.md` and `bin/run_experiments/cluster_gpu_checks.sh`
— verified with `git diff --name-only 129f73e..HEAD`, which lists nothing
outside `docs/` and that script. No library or test code differs. The gate is
nonetheless recorded against `129f73e`, the tree that actually ran, and is not
relabelled to HEAD.

Probe results on the GPU:

| probe | measurement | value | gate |
|---|---|---|---|
| `cluster_float32_probe` | `gp_fixed_m0` float32 vs float64 reference | 1.103e-07 | — |
| | `gp_fixed_mass` block scan vs sequential | 1.992e-07 | — |
| | `rho = 1` reduction in float32 | 5.385e-07 | — |
| | **worst** | **5.385e-07** | 1e-04, PASSED |
| `gp_float32_probe` | parallel vs sequential, HIGHEST matmul | 4.748e-08 | 1e-05, PASSED |
| | parallel vs sequential, backend DEFAULT | 7.498e-05 | recorded, not gated |
| `so_float32_probe` | second-order block scan vs sequential | 7.295e-05 | PASSED |

Two observations, stated at their actual scope:

* The positive-mass arm **passes a pure float32 check on real GPU hardware**
  (worst 5.385e-07 against a 1e-4 gate). The brief records that the older dense
  positive-mass code failed pure float32. This is a per-fixture agreement
  between two precisions of the same recurrence — **not** an exact-arithmetic
  error bound, **not** a bound for a training run, and not to be compared
  against losses or accuracies.
* The GPU float32 figures are slightly larger than the same probes on CPU
  (5.385e-07 vs 3.919e-07 worst). Both sit three orders of magnitude inside the
  predeclared gate. No conclusion is drawn from the difference.
* `gp_float32_probe` reproduced its historical backend-default value
  **7.498e-05 exactly**, consistent with the TF32 characterization in s9.

### Frozen data identity, produced before any training

| split | examples | SHA-256 of the filename+label list |
|---|---|---|
| train | 26,984 | `73caf4e4cd760aa4972a3f1553c37260a357dd0eb06f17bcddafc4847e37a345` |
| val | 5,783 | `dc4f65d666d832303364553d36f3a90c3e70e6b42aefa443fc7ad203407202e4` |
| test | 5,779 | `d458e62d0e40fcc25bcc62e214702a4e6c9f311539eb3e4967cdb9ad845c9568` |

38,546 clips, exactly the published 70/15/15 proportions; **843 optimizer steps
per epoch** at batch 32. The documented empty-mel-filter warning appeared as
expected.

### What this does and does not establish

**Does:** the fixed-coefficient arms and the Rawat port are numerically correct
on the actual target hardware, in the production dtype, under the deterministic
protocol, with every predeclared tolerance met and none loosened.

**Does not:** anything whatsoever about accuracy, learning or benchmark
standing. No training has been run. Stage 1 of
`docs/GP_RAWAT_CLUSTER_PROTOCOL.md` has not been executed.

### Updated evidence classes

| class | scope | commit | status |
|---|---|---|---|
| CPU numerical verification | 212 tests | local, this revision | PASSED |
| **GPU correctness, this revision** | **216 tests + 3 probes** | **`129f73e`** | **PASSED, exit 0** |
| GPU correctness, earlier revision | 124 tests | `be225d4` | PASSED |
| Integration smokes | 3 runs | `91f4988` | PASSED |
| GPU integration for the new arms (stage 1) | — | — | **not run** |
| Validation screening (stage 2) | — | — | **not run** |
| Final benchmark comparison (stage 3) | — | — | **not run** |

## 12.11 Stage 1 — GPU integration and cost, EXECUTED

Run by the user; `STAGE1_EXIT=0`, all five arms.
Artifacts `/Users/durso/s5-runs/stage1/20260915-153757`, host `pgi15-gpu3`,
SLURM 65870, backend `gpu`.

| arm | params | ms/step | vs native | peak memory | physical + auxiliary + buffer state |
|---|---|---|---|---|---|
| `native_s5` | 35,050 | 4.927 | 1.00x | 34.9 MB | 128 + 0 + 0 |
| `alpha_p_s5` | 35,050 | 4.657 | 0.95x | 68.4 MB | 128 + 0 + 128 |
| `gain_clip_s5` | 35,050 | 4.962 | 1.01x | 34.9 MB | 128 + 0 + 0 |
| `gp_fixed_m0` | 35,050 | 4.789 | 0.97x | 68.4 MB | 128 + 0 + 0 |
| `gp_fixed_mass` | 35,050 | **9.512** | **1.93x** | 68.4 MB | 128 + **128** + 0 |

Confirmed on hardware: identical trainable parameter counts across all five
arms, and the corrected state counts (32 physical real coordinates per layer,
four layers).

**The positive-mass cost on GPU is 1.93x per step, not the ~2.6x the CPU
smoke suggested.** The protocol's disclosed cost table was updated to the
measured value.

**The reported `acc` of 0.28-0.31 in stage 1 means nothing.** Integration mode
applies three updates to the SAME batch, so it measures memorization of one
batch, not learning. It is recorded only to show gradients flow.

**Budget decision, recorded before development:** nine stage-2 runs cost 603 s
of compute (0.17 GPU-h) against a 2 GPU-hour cap; even at 4x data-pipeline
overhead, 0.67 GPU-h. The epoch budget is **not** reduced. Stage 3 at 300
epochs and 2x overhead would be 10.96 GPU-h against its 8 GPU-hour cap, so the
stage 3 budget will be fixed from the actual `epoch_s` measured in stage 2 and
recorded before stage 3 starts.

Still true at this point: **no accuracy result exists, and stage 2 has not
run.**

## 12.12 Stage 2 — validation screening: the predeclared criterion FAILED

Executed by the user on `pgi15-gpu3`, SLURM 65870, 15 September 2026.
`STAGE2 done. elapsed 716s`, all nine runs `exit: 0`.
Artifacts `/Users/durso/s5-runs/stage2/`, manifest `manifest.jsonl`.
Development seed 100, 10 epochs, validation only. The test split was never
touched.

Selected configuration per arm by the predeclared rule (validation accuracy,
then validation cross entropy, then declared candidate order). Every family's
better learning rate was 1e-3; 3e-4 was worse for all four, by 4.7 to 6.8
points.

| arm | lr | val accuracy | val cross entropy |
|---|---|---|---|
| `alpha_p_s5` | 1e-3 | **95.00 %** | 0.28953 |
| `gain_clip_s5` | 1e-3 | 94.60 % | 0.29908 |
| `gp_fixed_mass` | 1e-3 | 94.26 % | 0.30020 |
| `native_s5` | 1e-3 | 94.17 % | 0.30953 |
| `gp_fixed_m0` | 1e-3 | 93.60 % | 0.32600 |

Cross entropy orders the arms identically to accuracy, so the tie-break never
had to be used.

### The criterion, and the outcome

Predeclared (s6 of the protocol): mean accuracy improvement over **both** the
reproduced `alpha_p_s5` **and** the matched `gain_clip_s5` control, screening
target at least 0.3 percentage points.

| comparison | delta | verdict |
|---|---|---|
| `gp_fixed_mass` vs `gain_clip_s5` (its MATCHED substrate control) | **-0.35 pp** | fails |
| `gp_fixed_mass` vs `alpha_p_s5` (reproduced baseline) | **-0.74 pp** | fails |
| `gp_fixed_m0` vs `gain_clip_s5` | **-1.00 pp** | fails, worst arm overall |

**The criterion FAILED. It is reported, not loosened.**

`gp_fixed_mass` is +0.09 pp above `native_s5`, and that number must not be
quoted as a success. `gp_fixed_mass` runs ON the gain-scaled, clipped
substrate, so `gain_clip_s5` is its control; comparing it to `native_s5`
credits the generalized prospective law with the input-gain and clipping gain
that belongs to the substrate. That confound is precisely why `gain_clip_s5`
was added as an arm.

### Consequences, per the protocol

* **Stage 3 confirmation is NOT triggered.** It was conditional on the primary
  physical candidate improving on validation. It did not.
* **No rescue sweep.** The protocol forbids an undeclared sweep after seeing
  the screen, and none was run.
* `gp_fixed_m0` succeeding alone could have triggered a separately labelled
  reduced-model confirmation. It is the worst arm, so that is not triggered
  either.
* The bounded batch therefore ends here with a **negative screening result for
  the physical candidate**.

### The reproduction, as a secondary observation

| step | delta |
|---|---|
| `gain_clip_s5` over `native_s5` (input gain + pole clipping) | +0.43 pp |
| `alpha_p_s5` over `gain_clip_s5` (the second tap alone) | +0.40 pp |
| `alpha_p_s5` over `native_s5` (the complete construction) | +0.83 pp |

Directionally consistent with Table 6, which reports 96.31 vs 95.84 (+0.47 pp)
at depth 4, width 32, MFCC. Ours is **10 epochs and one seed** against the
paper's 300 epochs and five seeds, and the absolute values are correspondingly
lower (95.00 vs 96.31). This is agreement in direction only; it is not a
reproduction of the published numbers, and the paired-seed comparison that
would support a quantitative claim was not run.

Worth recording separately: the split of alpha-P's advantage into roughly half
input-gain-plus-clipping and half second tap is a measurement the published
table cannot make, because those three changes move together there. It rests
on one seed at 10 epochs and is not asserted as more than an indication.

### What this result does and does not establish

**Does:** within this bounded, predeclared batch, at depth 4, width 32, MFCC,
10 epochs, one development seed, neither fixed-coefficient generalized
prospective arm reached its matched control, and the positive-mass arm did so
while costing 1.93x the per-step time and doubling the recurrent carry.

**Does not:** establish that the mechanism is worse in general. One seed at 10
epochs cannot resolve differences of a few tenths of a point; no variance
estimate exists; the 10-epoch regime is not the paper's 300-epoch regime. The
honest statement is that the candidate **did not pass the screen it was
predeclared against**, not that it has been refuted.

A properly powered comparison (paired seeds, full schedule) is a **separate
study that would have to be declared before it is run**. It is not started
here, because starting it now — after seeing an unfavourable screen — is
exactly the undeclared rescue sweep the protocol prohibits.

### Measured epoch times, for any future budget

| arm | s/epoch (real, including data pipeline) |
|---|---|
| `alpha_p_s5` | 2.70 |
| `native_s5` | 2.75 |
| `gain_clip_s5` | 2.80 |
| `gp_fixed_mass` | **6.85** |

Real training epochs are FASTER than the stage-1 compute-only projection
(2.70-2.80 s against 4.15 s), because stage 1 measured two steps immediately
after compilation and the training loop benefits from asynchronous dispatch.
The positive-mass ratio in real training is **2.49x**, higher than the 1.93x
per-step ratio measured in stage 1.

For the record, a stage 3 at these measured rates would have fit its cap
comfortably: 12 runs at 100 epochs is 1.26 GPU-h and at the full 300 epochs
3.77 GPU-h, against the 8 GPU-hour cap. **Budget was not the reason stage 3 did
not run; the failed criterion was.**

### Evidence classes, updated

| class | scope | commit / artifact | status |
|---|---|---|---|
| GPU correctness, this revision | 216 tests + 3 probes | `129f73e`, `gpu_checks/20260915-152008` | PASSED |
| GPU integration, stage 1 | 5 arms | `stage1/20260915-153757` | PASSED, exit 0 |
| **Validation screening, stage 2** | **9 runs, seed 100** | **`stage2/manifest.jsonl`** | **EXECUTED — criterion FAILED** |
| Final benchmark comparison, stage 3 | — | — | **not triggered** |

## 12.13 Stage 2 diagnostic — executed, report linked

A read-only diagnostic of the saved Stage 2 checkpoints was run on the cluster
(`pgi15-gpu3`, SLURM 65870, commit `cc4c752`, output
`/Users/durso/s5-runs/stage2-diagnostics/20260915-172533`).

**Full findings: [`docs/GP_STAGE2_DIAGNOSTIC_REPORT.md`](GP_STAGE2_DIAGNOSTIC_REPORT.md).**

**The Stage 2 verdict in s12.12 is unchanged: the predeclared screen FAILED.**
The diagnostic explains the result; it does not revise it.

Verification: restored validation counts matched the saved values **exactly**
for all five arms with cross-entropy difference `0.00e+00`; the response adapter
reproduced the executed core to **5e-08** over every layer and input; future
input sensitivity was **0.0**; and the Stage 2 source file hashes were
**unchanged**. Status `INCOMPLETE/3` for one reason only — `matplotlib` is
absent, so the optional plots did not run; no required check failed and no
phase was dropped for budget.

Two distinct diagnoses, not one:

* **`gp_fixed_m0`** changes the learned temporal filtering excessively. The
  contract's own bound predicts that `a_eff = a/(1 - T a)` confines effective
  modes to a disk of radius `1/(2T)`; at `T = 5` the measured trained poles sit
  inside it, frequencies compressed **2.88x**, and the end-to-end history share
  falls to **3.7 %** against **27.5 %** for its matched control.
* **`gp_fixed_mass`** changes the computation **least** of the three
  interventions — a 10-16 % counterfactual response change against 38-60 % for
  alpha-P, and a dynamical current tap no larger than the ordinary control —
  while costing ~2.5x the epoch time and double the recurrent carry.

No gradient pathology, normalization effect or implementation defect was found.

## 12.14 Constrained learned response — executed, report linked

The literature-analogous coefficient policy: keep the derived law and the fixed
horizon `T = 5`, learn two constrained physical parameters per stored mode
(`gamma_n`, `rho`) by full BPTT, derive the mass `mu = T gamma_n rho`. Executed
on the cluster at commit `4848ee4`, `CONSTRAINED_STATUS=PASS`, artifacts
`/Users/durso/s5-runs/constrained/20260915-193049/`.

**Full findings: [`docs/CONSTRAINED_PROSPECTIVE_RESPONSE_REPORT.md`](CONSTRAINED_PROSPECTIVE_RESPONSE_REPORT.md).**

**The predeclared screen FAILED.** `gp_learned_response` reached **94.21 %**
against **94.26 %** for the same model with the response frozen — this run
scored **0.052 pp lower**, three examples out of 5,783, which establishes
neither degradation nor improvement — and -0.398 pp against the matched
ordinary control and -0.795 pp against the Rawat reference (shortfalls against
the +0.3 pp target of 0.698 and 1.095 pp respectively).

**Analytic finding recorded after the run:** `(Delta, gamma_n, rho)` is
input-output equivalent to `(Delta/gamma_n, 1, rho)` for every admissible
`rho`, so only the 64 `rho` coordinates added response-shape freedom; the 64
`gamma_n` coordinates re-parameterized the already-learned clock. The observable
clock coordinate is `log_step - log(gamma_n)`.

The response did move: median `gamma_n` 1.00 -> 0.853, median `rho`
0.75 -> 0.840, `mu` spread 2.35-5.34 against a fixed 3.75, with **no mode at any
declared bound** and the component identities holding on the trained values to
`1.8e-15`. The two quantities moved in partly compensating directions, leaving
the derived mass near its initial value.

The professor control `prospective_recurrence` (`r + T r' = 0`, zero recurrent
state, verified zero driven history) reached **84.85 %**. That is one trained
configuration on a mean-pooled task; it is not a decomposition of accuracy, a
floor, or a ceiling on what recurrence could contribute. Its exact data-loss
gradient with respect to `log_step` is ZERO, correcting an earlier claim.

30 focused GPU checks passed. Two earlier check failures were resolved by
measurement before training — a bound declared too wide, and a probe that ran
at TF32 default instead of production precision — with no tolerance loosened.

## 12.15 Memory-recall study — executed, report linked

A controlled recall task (length 128, two marked cues among distractors, target
is the latest cue, query token carries no symbol) with six arms on one causal
stack, paired continuation from a shared warm-up, seeds 100/101/102. Executed
at commit `b5d7211`; `RECALL_STATUS=PASS`, 18/18 rows, 739 s of a 1200 s cap.

**Full findings: [`docs/PROSPECTIVE_MEMORY_RECALL_REPORT.md`](PROSPECTIVE_MEMORY_RECALL_REPORT.md).**

**The predeclared screen FAILED.** `gp_rho` beat `ordinary` by **+0.098 pp**
(about two examples of 2,048) and was inconsistent in sign against `rawat`,
against a +0.3 pp target.

**The dominant result is a control:** `ordinary_2x`, ordinary S5 with twice the
stored modes, beat `gp_rho` by **2.9-5.0 pp in every seed at equal total
recurrent carry** (128 real coordinates each), using 58 % more parameters.

The memoryless professor control scored **0.1203 against a 0.125 chance level**,
with exactly zero logit response to any input change. That confirms the task is
a real recall probe, and it puts the earlier 84.85 % on mean-pooled speech in
context: the same mechanism is at chance here.

Learning `rho` helped against freezing it in all three seeds (+0.911 pp), but
the median final `rho` is the declared **ceiling** — 27-28 of 32 modes moved
*up* into the clip, toward the ordinary-SSM limit — so it does not support the
generalized response being the active ingredient. A protocol statement of mine
that `rho` could "effectively only fall" is corrected in the report.
