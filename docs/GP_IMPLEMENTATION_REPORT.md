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
| Milestones | A, B, C implemented and tested on CPU. D prepared, **not executed**. |
| Evidence level | **CPU correctness only.** No GPU run has been performed. No training benchmark is claimed. |

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
**no weight decay**. The `regular` group is AdamW with `weight_decay`, which
would shrink the response coefficient toward zero — that is, toward the plain
baseline — and would bias every treatment/control comparison. Tested that the
label fires and that an actual optimizer update reaches the parameter.

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
`1e-11`. float32 production path within `1e-5`. Chunk-boundary semantics pinned
(a chunk restarts from zero prehistory); no state is carried between
independent examples under `vmap`.

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
2. **Prospectivity does not increase Hankel memory.** Hankel top falls
   monotonically with `gp_init_scale` (`1e-6 -> 0.3 -> 1.0`), consistent with
   `sigma_memory = |K0| tau_0 / (2 tau)`.

Reported as measured relations on this configuration and initialization, not
as proofs.

**Training smoke:** a fixed synthetic batch, 200 steps at `lr = 1e-2`. Every
declared parameter receives a nonzero update, including `gp_response_raw`, and
the loss falls `1.5815 -> 1.8e-4`. This is correctness evidence for the
gradient path. **It is not a benchmark and not an improvement claim.**

*Method note:* at the `create_train_state` default `lr = 1e-3` the loss falls
only ~8% in 60 steps, which would make any threshold arbitrary; `1e-2` was
chosen so the check is informative, applied identically to all mechanisms.

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
( time ./bin/run_experiments/gp_smoke.sh "$GP/plain" plain ) > "$GP/plain.log" 2>&1 ; echo "exit=$?"
( time ./bin/run_experiments/gp_smoke.sh "$GP/gp_diag" gp_diagonal --gp_init_scale=0.05 ) > "$GP/gp_diag.log" 2>&1 ; echo "exit=$?"
```
```bash
grep -hE "Trainable Parameters|Train Loss:|real" "$GP"/plain.log "$GP"/gp_diag.log
python -m experiments.gp.run_diagnostics --outdir "$GP/diagnostics"
```

**Expected of the plain run, as a regression gate:** 26,058 parameters and
`Train Loss 1.40519 / Val 0.36394 / 0.8950 / Test 0.33751 / 0.9001`, matching
the deterministic `E2-004` record exactly. Any deviation means the routing
changed the default path and must be investigated before anything else.

The `gp_diagonal` run validates integration only. **It is not evidence for the
research claim** and must not be reported as a benchmark comparison.
