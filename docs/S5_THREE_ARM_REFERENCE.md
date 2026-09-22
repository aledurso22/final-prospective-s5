# Part 1 — The three S5 arms: protocol, equations, implementation, results

*Part 2, the Momentum DeltaNet bridge, is `prospective/MDN_BRIDGE_REFERENCE.md`
in `github.com/aledurso22/MomentumDeltaNet-prospective`.*

A single reference for the Speech Commands S5 study: what is shared, what
differs, how each arm is evaluated, and what the numbers are. Written to be
read against the source — every claim below names the file it comes from, and
every code block is copied from the tree rather than paraphrased.

Base commit for the protocol: `ef004cda`. Scan work and the generalized wave:
branch `s5-two-compartment-parallel-scan`. Three-arm run
`s5-three-arm-15epoch/20260919-224525`; generalized wave
`s5-two-compartment-factored/20260921-175242`.

---

## 1. The three arms in one sentence each

| arm | recurrence | extra parameters | recurrent state / layer |
|---|---|---|---|
| **Native S5** | `s_t = λ̄ s_{t-1} + B̄ x_t` | none | 128 real |
| **Zucchet prospective** | `s_t = a₁s_{t-1} + a₂s_{t-2} + c₁x_t + c₂x_{t-1}`, at `M = γ = 0` | `T` | 128 real |
| **Generalized prospective** | the same second-order recurrence, `M, γ, T` all free | `T, ρ, γ` | 256 real |

The second and third are **the same equation at two points of one coefficient
map** (§4.3). That is the central structural fact of the study.

---

## 2. The shared protocol

Everything in this section is identical across arms by construction — one frozen
dataclass, one factory, one training loop.

### 2.1 Configuration

`experiments/s5_three_arm_full/runner.py`:

```python
@dataclass(frozen=True)
class SharedS5Config:
    d_model: int = 96
    ssm_size: int = 128
    blocks: int = 16
    n_layers: int = 6
    input_dim: int = 1
    seq_len: int = 16000
    batch_size: int = 16
    epochs: int = 40
    lr: float = 0.008
    ssm_lr: float = 0.002
    lr_final: float = 1e-6
    weight_decay: float = 0.04
    warmup_end: int = 1
    dropout: float = 0.1
    c_init: str = "lecun_normal"
    discretization: str = "zoh"
    conj_sym: bool = True
    clip_eigs: bool = True
    bidirectional: bool = True
```

**Runs used `epochs = 15`, not the dataclass default of 40.** The schedule
adapts to the shorter horizon rather than truncating a 40-epoch cosine (§2.5).

### 2.2 Dataset

`dataloaders/speech_commands10.py`. Speech Commands, 10-word subset:

```python
WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
SAMPLE_RATE = 16000
```

* **Raw waveform**, not MFCC. `input_dim = 1`, `seq_len = 16000` — one second of
  16 kHz audio, one channel, 16 000 timesteps. The loader *can* produce 20-coef
  MFCCs with a 200-sample FFT; this study does not use that path.
* **Stratified 70/15/15 file split**, `split_seed = 0`, deterministic in the seed
  alone. Measured validation set size: **n = 3703** (from the per-epoch
  `validation` record).
* **Feature-wise standardization computed on the training split only.**
* Everything is cached to `.npy` with a manifest of filenames and SHA-256
  digests, so every arm and every seed consumes byte-identical tensors.

> **Known deviation, stated rather than hidden.** The file split is *not*
> speaker-disjoint, unlike the official Speech Commands split. The loader's own
> docstring says so. Numbers here are therefore not directly comparable to
> official-split literature. Every arm shares the split, so the *between-arm*
> comparison is unaffected; the absolute level is optimistic.

### 2.3 Model

```python
BatchClassificationModel(
    ssm=ssm_factory(arm), d_output=len(SC.WORDS), d_model=96, n_layers=6,
    padded=False, activation="half_glu1", dropout=0.1, mode="pool",
    prenorm=True, batchnorm=True, bn_momentum=0.95)
```

Six layers, `d_model = 96`, pre-norm with BatchNorm (momentum 0.95),
`half_glu1` activation, mean pooling into a 10-way head.

### 2.4 SSM initialization — identical for every arm

```python
def ssm_kwargs(arm):
    block = SSM_SIZE_BASE // BLOCKS          # 128 // 16 = 8
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block)
    block //= 2                              # 4, conjugate pairs
    P = SSM_SIZE_BASE // 2                   # 64 complex modes
    Lambda, V = Lambda[:block], V[:, :block]
    Vc = V.conj().T
    Lambda = (Lambda * np.ones((BLOCKS, block))).ravel()
    V  = block_diag(*([V]  * BLOCKS))
    Vc = block_diag(*([Vc] * BLOCKS))
    return dict(H=96, P=P, Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
                V=V, Vinv=Vc, C_init="lecun_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=True,
                clip_eigs=True, bidirectional=True)
```

Block-diagonal HiPPO-LegS (DPLR), 16 blocks of 4 conjugate modes → **P = 64
complex modes = 128 real dimensions per layer**. `conj_sym=True` so only half
the spectrum is stored and the readout takes `2·Re(C̃ s)`. `clip_eigs=True`
holds `Re Λ ≤ −1e-4`. `bidirectional=True`, so `C̃` has `2P` columns and every
arm runs its recurrence forward and reversed.

`dt_min=0.001, dt_max=0.1` → `log_step` is initialized log-uniform on that
range, per mode.

**The arms receive this dictionary unchanged.** `ssm_factory` differs only in
which constructor consumes it:

```python
def ssm_factory(arm):
    kw = ssm_kwargs(arm)
    if arm == "native_matched_s5":
        return init_S5SSM(**kw)
    if arm == "zucchet_prospective_s5":
        return init_prospective_S5SSM(response_init=0.05, **kw)
    if arm == "generalized_prospective_s5":
        return init_generalized_prospective_S5SSM(
            response_init=0.05, rho_init=0.5, gamma_init=1.0,
            implementation=GENERALIZED_IMPLEMENTATION, **kw)
```

### 2.5 Optimizer and schedule

```python
create_train_state(model_cls_for(arm), jax.random.PRNGKey(seed), padded=False,
                   retrieval=False, in_dim=1, bsz=16, seq_len=16000,
                   weight_decay=0.04, batchnorm=True, opt_config="noBCdecay",
                   ssm_lr=0.002, lr=0.008, dt_global=False)
```

`opt_config="noBCdecay"` is the standard S5 grouping: **`B` and `C` are excluded
from weight decay**, SSM parameters (`Λ`, `B`, `log_step`) train at `ssm_lr =
0.002` without decay, everything else at `lr = 0.008` with `weight_decay =
0.04`. `dt_global=False` → per-mode timesteps.

```python
def apply_scheduled_learning_rate(state, step, steps_per_epoch, epochs=EPOCHS):
    warmup_steps = steps_per_epoch * WARMUP_END          # one epoch
    if step < warmup_steps:
        decay, schedule_step, end_step = linear_warmup, step, warmup_steps
    else:
        decay, schedule_step, end_step = (
            cosine_annealing, step - warmup_steps,
            steps_per_epoch * (epochs - WARMUP_END))
    return update_learning_rate_per_step(
        (decay, SSM_LR, LR, schedule_step, end_step, "noBCdecay", LR_FINAL),
        state)[0]
```

One epoch of linear warm-up, then cosine annealing to `lr_final = 1e-6` over the
remaining `epochs - 1`. Passing `epochs=15` makes the cosine complete inside the
run instead of stopping at 37.5% of a 40-epoch curve — a real confound if left
unfixed.

> `steps_per_epoch` is derived at run time from the split, not hard-coded. It is
> deliberately not quoted here: an earlier wall-clock projection in this project
> was wrong by ~9× because that number was guessed rather than measured.

**Seeds 301, 302, 303**, three per arm. The same PRNG key seeds every arm, so
the shared backbone is bit-identical at initialization; `shared_parameter_digest`
hashes every leaf to verify it.

### 2.6 Where the arms are *not* matched — read this before comparing

| | Native | Zucchet | Generalized |
|---|---|---|---|
| extra scalars per mode | 0 | 1 (`T`) | 3 (`T, ρ, γ`) |
| extra parameters, whole model | 0 | **+384** | **+1152** |
| recurrent state, real values / layer | 128 | 128 | **256** |

`recurrent_state_size()` in the runner makes the last row explicit. So the
generalized arm carries **twice the recurrent state and 1152 more parameters**
than Native. This is not a capacity-matched comparison, and it matters for how
the result reads: the generalized arm loses by 1 pp **while holding more
capacity**, which strengthens rather than weakens the negative conclusion.

---

## 3. What each arm computes

### 3.1 Native S5 — first-order, diagonal, associative

`s5/ssm.py`. ZOH discretization of `ṡ = Λs + B̃x`:

```python
def discretize_zoh(Lambda, B_tilde, Delta):
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar
```

giving `λ̄ = exp(ΛΔ)`, `B̄ = Λ⁻¹(λ̄ − 1)B̃`, and the recurrence

```
s_t = λ̄ s_{t-1} + B̄ x_t
```

evaluated by a first-order associative scan:

```python
@jax.vmap
def binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

_, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
```

`O(L)` work, `O(log L)` depth. One complex multiply and one add per combine.
This is the operator everything else is measured against.

### 3.2 Zucchet prospective — second-order, at zero mass and zero damping

`s5/prospective_ssm.py`:

```python
class ProspectiveS5SSM(S5SSM):
    response_init: float = 0.05

    def setup(self):
        super().setup()
        raw = np.log(np.expm1(self.response_init)).astype(np.float32)
        self.prospective_T_raw = self.param(
            "prospective_T_raw", lambda rng, shape: np.full(shape, raw), (self.P,))
        response = jax.nn.softplus(self.prospective_T_raw)
        b_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        lambda_bar, b_bar = discretize_zoh(self.Lambda, b_tilde, step)
        self.prospective_a1, self.prospective_a2, \
        self.prospective_c1, self.prospective_c2 = (
            zucchet_coefficients(lambda_bar, b_bar, response))
```

with, in `s5/discrete_recurrence.py`:

```python
def zucchet_coefficients(lambda_bar, b_bar, response, h=1.0):
    F, G = target_map(lambda_bar, b_bar, response, h)
    k = response.astype(lambda_bar.dtype) / h
    return ((1.0 - k) + (1.0 + k) * lambda_bar,
            (k - 1.0) - k * lambda_bar,
            (1.0 + k)[..., None] * b_bar, -k[..., None] * b_bar)
```

Note `F, G` are computed and then unused — the returned closed form is the same
expression already simplified. Harmless, but it reads as dead code.

### 3.3 Generalized prospective — the full two-compartment law

`s5/generalized_prospective_ssm.py` and `s5/discrete_recurrence.py`:

```python
def target_map(lambda_bar, b_bar, response, h=1.0):
    k = response.astype(lambda_bar.dtype) / h
    return 1.0 + k * (lambda_bar - 1.0), k[..., None] * b_bar


def generalized_coefficients(lambda_bar, b_bar, response, mass, gamma, h=1.0):
    F, G = target_map(lambda_bar, b_bar, response, h)
    q = mass + h * (gamma + response)
    alpha, beta, delta = mass / q, h * h / q, h * response / q
    return ((1.0 + alpha - beta) + (beta + delta) * F,
            -alpha - delta * F,
            (beta + delta)[..., None] * G, -delta[..., None] * G)
```

Both arms then run

```
s_t = a₁ s_{t-1} + a₂ s_{t-2} + c₁ x_t + c₂ x_{t-1}
```

with zero prehistory, forward and reversed, and read out `2·Re(C̃ [s_fwd; s_rev])`
plus the skip `D ⊙ x` — the native readout, unchanged.

---

## 4. The mathematics, and why the two prospective arms are one family

### 4.1 The continuous law

The generalized arm is the semi-implicit discretization of

```
M s̈ + (γ + T) ṡ = −s + Y + T Ẏ ,     Y = s + (T/h)(f − s),   f = λ̄ s + B̄ x
```

`M` mass, `γ` damping, `T` prospective horizon. `Y` is the *prospective target*:
a `T/h`-weighted step from the current state toward the S5 target `f`. Writing
`Y = F s_{t-1} + G x_t` gives exactly `target_map`:

```
F = 1 + (T/h)(λ̄ − 1)        G = (T/h) B̄
```

### 4.2 The discrete filter

With `q = M + h(γ+T)`, `α = M/q`, `β = h²/q`, `δ = hT/q`, the second-order filter
driven by an arbitrary target `u` is

```
y_t = (1 + α − β) y_{t-1} − α y_{t-2} + (β + δ) u_t − δ u_{t-1}
```

Substituting the *closed-loop* target `u_t = Y_t = F s_{t-1} + G x_t` and
collecting terms reproduces `generalized_coefficients` line for line. This is
the step that distinguishes **recurrence placement** from **drive placement**:
because `λ̄` enters through `F`, the prospective parameters and the SSM mode
share one characteristic polynomial.

### 4.3 Zucchet is the `M = γ = 0` face of Generalized

Not approximately — identically. Verified numerically on random `λ̄` inside the
unit disc, random `B̄`, random `T`:

```
zucchet_coefficients(λ̄, B̄, T)  ==  generalized_coefficients(λ̄, B̄, T, mass=0, gamma=0)

  a1  max|diff| = 5.55e-16        c1  max|diff| = 8.88e-16
  a2  max|diff| = 2.22e-16        c2  max|diff| = 0.0
```

Sketch: at `M = γ = 0`, `q = hT`, so `α = 0`, `β = h/T`, `δ = 1`. Then
`a₂ = −δF = −F = (k−1) − kλ̄` with `k = T/h`, matching directly; `c₁ = (β+δ)G =
(1 + 1/k)·k B̄ = (1+k)B̄`; `c₂ = −δG = −k B̄`; and `a₁ = (1 − 1/k) + (1 + 1/k)F`
expands to `(1−k) + (1+k)λ̄`.

**Consequence:** the arm that fails and the arm that trains are one equation at
two parameter settings. Any comparison between them is a controlled comparison
over `(M, γ)` with everything else — including `T` and its initialization —
held fixed.

### 4.4 Why zero mass fails: the poles leave the unit disc

At the production initialization `T = 0.05` with `h = 1`, the companion roots of
the Zucchet arm have magnitude

```
max root magnitude per mode: [1.63 1.59 1.59 1.63 1.39 1.16]
```

which is the **ρ = 1.10–1.71** measured at initialization in the production run.
Nothing damps the second-order recurrence, so the state diverges on the first
update. Finite `M` and `γ` pull the roots inside the disc: the epoch-15
checkpoints show `|r_slow| ≈ 0.999` and `|r_fast| ≈ 0.05`.

### 4.5 Parameterization of `(M, γ, T)`

```python
RHO_MIN = 1e-4

def response_mass_gamma(T_raw, rho_raw, gamma_raw):
    T = jax.nn.softplus(T_raw)
    rho = RHO_MIN + (1.0 - RHO_MIN) * jax.nn.sigmoid(rho_raw)
    gamma = jax.nn.softplus(gamma_raw)
    return T, rho * gamma * T, gamma, rho
```

**`M = ρ γ T` with `ρ ∈ (RHO_MIN, 1)`.** Mass is not a free positive scalar; it
is a *fraction* of `γT`. Two consequences worth defending to a referee:

1. `M ≥ 0` and `M < γT` hold by construction, with no projection step, so the
   discriminant and the pole locations stay in a controlled region throughout
   training.
2. `M` cannot be raised without also raising `γ` or `T`, which couples the
   damping and the mass. An unconstrained `M` is a different model and was not
   tested.

Initialization `T = 0.05, ρ = 0.5, γ = 1.0` → **`M = 0.025`**. Per mode, per
layer: `self.param(..., (self.P,))` with `P = 64`.

---

## 5. How the recurrence is evaluated — three scans, one equation

This section is about implementation only. **The equation, the parameterization
and the coefficients are identical for every choice; only the evaluation order
differs.**

### 5.1 The default is the oracle

```python
SCAN_IMPLEMENTATIONS = {
    "sequential": scan_companion_sequential,
    "companion":  scan_companion,
    "factored":   scan_factored,
}
DEFAULT_IMPLEMENTATION = "sequential"
```

An unknown name raises rather than silently substituting.

### 5.2 Sequential — correct, and the reason the wave was slow

```python
def scan_companion_sequential(a1, a2, c1, c2, inputs, reverse=False):
    sequence = inputs[::-1] if reverse else inputs

    def rollout(values):
        def body(carry, value):
            state, previous_state, previous_input = carry
            next_state = (a1 * state + a2 * previous_state
                          + c1 @ value + c2 @ previous_input)
            return (next_state, state, value), next_state
        initial = (np.zeros_like(a1), np.zeros_like(a1),
                   np.zeros_like(values[:1])[0])
        _, states = jax.lax.scan(body, initial, values)
        return states

    states = jax.checkpoint(rollout)(sequence)
    return states[::-1] if reverse else states
```

`lax.scan` over 16 000 tokens, wrapped in `jax.checkpoint` so the backward pass
rematerializes instead of storing every state. That is **O(L) sequential depth**
with a recompute, against Native's `O(log L)`. Measured at **2.98 optimizer
steps per minute**, projecting **≈155 hours** for a three-seed wave.

The bottleneck is *latency*, not arithmetic: each of ~576 000 sequential
iterations per step costs ~30 µs of kernel-launch overhead to perform ~0.8 ns of
useful work. **This is an implementation property, not an instability in the
equation** — the run stayed finite over the observed prefix.

> Note an asymmetry in the production configuration: the **Zucchet arm uses
> `scan_companion`** (the parallel 2×2 route) while the **generalized arm
> defaulted to `scan_companion_sequential`**. Both are exact; they differ in
> floating-point summation order, and only the generalized arm paid the
> sequential cost.

### 5.3 Companion — parallel, but carries a 2×2 per token

```python
A = A.at[:, :, 0, 0].set(a1)
A = A.at[:, :, 0, 1].set(a2)
A = A.at[:, :, 1, 0].set(1.0)
b = np.stack((drive, np.zeros_like(drive)), axis=-1)
_, states = jax.lax.associative_scan(_block_operator, (A, b))
```

Correct and parallel, but four complex multiplies and two adds per combine
against one multiply and one add for a scalar scan.

### 5.4 Factored — the preferred route

Factor the characteristic polynomial:

```
λ² − a₁λ − a₂ = 0,     r₁ + r₂ = a₁,     r₁r₂ = −a₂
(1 − r₁z⁻¹)(1 − r₂z⁻¹) = 1 − a₁z⁻¹ − a₂z⁻²
```

so the second-order recurrence is **two first-order ones in series**:

```
v_t = r₁ v_{t-1} + d_t
s_t = r₂ s_{t-1} + v_t
```

each of which is exactly the shape `s5/ssm.py`'s own `binary_operator` composes
— no new scan primitive, no 2×2 block:

```python
def scan_factored(a1, a2, c1, c2, inputs, reverse=False):
    sequence = inputs[::-1] if reverse else inputs
    drive = _drive(c1, c2, sequence).astype(a1.dtype)
    first, second = companion_roots(a1, a2)
    states = _first_order(second, _first_order(first, drive))
    return states[::-1] if reverse else states
```

**Roots computed stably.** The naive pair `(a₁ ± √(a₁²+4a₂))/2` loses the small
root to cancellation whenever `|a₁|` dominates the square root:

```python
def companion_roots(a1, a2):
    root = np.sqrt(discriminant(a1, a2).astype(a1.dtype))
    aligned = (np.conj(a1) * root).real if np.iscomplexobj(a1) else a1 * root
    sign = np.where(aligned >= 0, 1.0, -1.0).astype(a1.dtype)
    major = (a1 + sign * root) / 2.0                 # constructive sum
    safe = np.where(np.abs(major) < ROOT_FLOOR, np.ones_like(major), major)
    minor = np.where(np.abs(major) < ROOT_FLOOR, np.zeros_like(major),
                     -a2 / safe)                     # from the PRODUCT
    return major, minor
```

Form the root where the square root *adds*, then get the other from
`r₁r₂ = −a₂`, which never subtracts.

**Repeated roots are exact, not special-cased.** At `disc = 0` the cascade
becomes `v_t = r v_{t-1} + d_t`, `s_t = r s_{t-1} + v_t`, the Jordan realization,
which reproduces `t·r^{t-1}` exactly. A partial-fraction route would divide by
`(r₁ − r₂)` and fail there. Near-repeated roots are equally safe because `minor`
comes from a product.

**No eigendecomposition anywhere in the training path.**

### 5.5 Measured

| implementation | s / step | steps / min | peak GiB | speedup |
|---|---|---|---|---|
| `sequential` | 17.5654 | 3.42 | 8.342 | 1.0× |
| `factored` | **0.2181** | **275.1** | 12.885 | **80.5×** |

Agreement against the sequential oracle in float64, across all 1152 production
modes: **worst relative error 1.83e-13** — algebraically exact.

Memory: the parallel routes cost 12.885 GiB against 8.342, a 1.54× increase, and
80.5× turns a 155-hour wave into **1.46 h**. On a 24.0 GiB RTX 3090 that is one
seed per GPU, never two.

> The float64 certification is guarded: `x64_enabled()` refuses to report a
> float64 column if JAX's x64 mode is off. An earlier version of this check
> silently truncated to float32 and reported f64 and f32 as identical to every
> digit.

---

## 6. Failure gates

The runner raises `NumericalTrainingFailure` on a non-finite update, checked
before training proceeds:

```python
def _all_finite(tree):
    return bool(all(bool(jnp.all(jnp.isfinite(value)))
                    for value in jax.tree_util.tree_leaves(tree)
                    if hasattr(value, "dtype")))
```

The Zucchet arm trips this at **epoch 0, step 0** on all three seeds:

```
failure: "production check nonfinite"
loss NaN, gradient_norm NaN, gradients_finite false, state_finite false
```

The precise statement is *the first-order prospective recurrence is non-finite on
the first update*, not "training was unstable". `companion_spectral_radius`
reconstructs the roots from the checkpoint to confirm ρ = 1.10–1.71 at init.

---

## 7. Results

### 7.1 Validation accuracy at the selected epoch

| seed | Native | Generalized | difference | Native CE | Gen. CE |
|---|---|---|---|---|---|
| 301 | 97.110% | 96.138% | −0.972 | 0.09217 | 0.12461 |
| 302 | 96.786% | 95.895% | −0.891 | 0.09645 | 0.12754 |
| 303 | 96.867% | 95.733% | −1.134 | 0.09481 | 0.13464 |
| **mean** | **96.921%** | **95.922%** | **−0.999** | 0.09448 | 0.12893 |

Native: sd 0.169, se 0.097, seed-to-seed range **0.324 pp**.

**Native wins on 3/3 seeds by 1.00 pp, 95% CI [0.69, 1.31], paired t = −14.0**
against a critical 4.303 at df = 2. Significant with three seeds only because
the difference is so consistent (paired sd 0.124 pp). The gap is **3.1×** the
single-arm seed spread, so it is not noise.

Zucchet: no number. NaN before epoch 1.

### 7.2 Learning curves, per-epoch mean over seeds

| epoch | Native | Generalized | gap |
|---|---|---|---|
| 1 | 79.33 | 63.14 | 16.19 |
| 5 | 93.01 | 89.36 | 3.65 |
| 6 | 94.27 | **83.59** | — |
| 10 | 96.18 | 94.72 | 1.46 |
| 15 | 96.90 | 95.88 | 1.02 |

The generalized arm starts 16 pp behind and closes to 1. Both are flat over the
final three epochs, so neither is undertrained. The epoch-6 dip is a **single
seed reaching 64.95%**, fully recovered by epoch 7 (93.30% min across seeds).

### 7.3 The learned regime is the predicted one

Epoch-15 checkpoints, median over modes, per layer:

| seed | T | γ | ρ | \|r_slow\| | \|r_fast\| | in regime |
|---|---|---|---|---|---|---|
| 301 | 0.0426 | 1.107 | 0.426 | 0.99950 | 0.0521 | 62/64 |
| 302 | 0.0444 | 1.080 | 0.447 | 0.99905 | 0.0573 | 61/64 |
| 303 | 0.0461 | 1.054 | 0.485 | 0.99886 | 0.0580 | 55/64 |

"In regime" = slow root > 0.99 **and** fast root < 0.5. One slow pole carrying
long memory, one fast response pole, in 55–62 of 64 modes, every seed, stable
from epoch 3 through 15.

**γ rose from its initial 1.0 to 1.05–1.11** — away from the stability boundary,
not toward it. The predicted two-timescale separation is achieved; it costs
1 pp.

---

## 8. What a referee should push on

Listed because they are real, not to pre-empt them.

1. **The split is not speaker-disjoint.** Absolute accuracies are optimistic
   relative to official-split results. Between-arm comparison is unaffected.
2. **Three seeds.** The paired t is significant only because the effect is
   unusually consistent. A single anomalous seed would change the interval
   substantially.
3. **Validation, not test.** No test-set number is reported; model selection
   used the same validation split.
4. **Not capacity-matched** (§2.6). Generalized carries 2× the recurrent state
   and +1152 parameters and still loses. This cuts against the prospective arm,
   so it is not a threat to the negative result — but a *positive* result under
   this configuration would have been confounded.
5. **The arms use different scans.** Zucchet ran `scan_companion` (parallel),
   Generalized ran `scan_companion_sequential`. Both exact to 1.83e-13 in
   float64, but summation order differs; float32 training trajectories are not
   bitwise comparable across implementations.
6. **Wall clock is not comparable across implementations**, and the benchmark
   measures optimizer steps only — roughly 2× less overhead than a full epoch
   with evaluation.
7. **`M = ργT` is a modelling choice**, not a neutral parameterization (§4.5).
   An unconstrained `M ≥ 0` is a different model and was not run.
8. **`zucchet_coefficients` computes `F, G` and discards them.** Dead code, not
   a defect, but it invites the question of whether the intended expression was
   the one shipped.
9. **The published Zucchet Eq. (1) and `zucchet_coefficients` agree only at
   `k = 1`.** Production runs `T = 0.05`, so `k = 0.05`. This discrepancy was
   found, recorded, and deliberately not acted on; it does not affect the
   `M = γ = 0` equivalence of §4.3, which is internal to this codebase.

---

## 9. Related result, different placement

The same `(M, γ, T)` filter was ported into the official Momentum DeltaNet
codebase and placed on the **write residual** instead of the recurrence, where
`a, b, c, d` are gate-independent constants and the memory recurrence is
untouched. Over eight seeds on an MQAR-with-revision probe:

* zero mass never trains — **0/8 seeds above chance**, and generalized beats it
  **8/8, p = 0.0039** on all five metrics;
* generalized does **not** beat the native baseline — 3/8 seeds, p = 0.86 — and
  is indistinguishable from literal Nesterov (mean difference −0.04).

So in neither architecture and at neither placement does the prospective law
outperform the model it is added to. What mass and damping buy is a law that
runs at all rather than one that diverges at initialization or never leaves
chance.

See `docs/MDN_RECURRENT_BRIDGE_PLAN.md` on branch `mdn-recurrent-bridge`, and
the implementation at `github.com/aledurso22/MomentumDeltaNet-prospective`.
