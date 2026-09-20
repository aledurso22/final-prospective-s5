# WWJ prospective S5: derivation, realization, and the gates before training

20 September 2026. Branch `s5-three-arm-discrete-prospective`, on top of
`ef004cd`. Additive: **no existing file is modified**. Native S5 is the
control and is byte-identical; the old two-compartment generalized arm is
frozen and keeps its own name and its own results.

## 1. What the previous run showed

Slurm job `66765`, commit `ef004cd`, run root
`/Users/durso/s5-runs/s5-three-arm-15epoch/20260919-224525`, node
`pgi15-gpu2`, three RTX 3090s, one seed per GPU.

- the concurrent topology smoke passed on three distinct physical GPUs;
- Native S5 finished all three seeds and wrote `task_result.json`;
- the three Zucchet runs produced the expected `NumericalTrainingFailure`
  artifacts and passed the declared-failure gate;
- the **old generalized arm was operationally unusable**: all three seeds
  still in epoch 1, seed 301 sampled at ≈ **2.98 steps/minute**, ≈ 27,622
  steps remaining, ≈ **154.73 h** projected for that wave against ≈ 5.6 h of
  allocation left.

It did not fail numerically at initialization. Its recurrence was evaluated
by `scan_companion_sequential`, a rematerialized sequential `lax.scan` over
16,000 tokens, which is orders of magnitude too slow. Those artifacts stay
where they are and keep their labels: **they are not WWJ results**.

## 2. The continuous law

The Euclidean WWJ/Bregman residual equation applies one second-order
operator to the **complete** residual:

$$P(D)(s-f)=0,\qquad P(D)=1+\tau D+MD^{2},\qquad M=\varepsilon\tau^{2},$$

that is

$$M\ddot s+\tau\dot s+s \;=\; f+\tau\dot f+M\ddot f .$$

The `M f̈` term on the right is exactly what the older two-compartment
generalized equation (`s5/generalized_prospective_ssm.py`, an
(M, γ, T) system) does not have. They are different equations, so they are
different arms, and neither is a version of the other.

## 3. The realization is a mixed-stencil discretization, not that equation

The implemented update is an explicit **causal, mixed-stencil (IMEX-type)**
discretization, consistent with §2 but **not** the exact finite-step
identity $P(D_h)(s-f)=0$: the state uses forward stencils and the target
uses backward ones.

$$\dot s_t\approx\frac{s_{t+1}-s_t}{h},\quad
\ddot s_t\approx\frac{s_{t+1}-2s_t+s_{t-1}}{h^{2}},\quad
\dot f_t\approx\frac{f_t-f_{t-1}}{h},\quad
\ddot f_t\approx\frac{f_t-2f_{t-1}+f_{t-2}}{h^{2}}.$$

Multiplying the equation by $h^2$ and collecting terms (rederived, not
copied — the same derivation is in the module docstring and is verified in
exact rational arithmetic by `tests/test_wwj_algebra.py`):

$$(M+h\tau)s_{t+1}+(h^{2}-2M-h\tau)s_t+Ms_{t-1}
=(M+h\tau+h^{2})f_t-(2M+h\tau)f_{t-1}+Mf_{t-2},$$

so with $q=M+h\tau$,

$$s_{t+1}=a s_t-b s_{t-1}+c_0f_t+c_1f_{t-1}+c_2f_{t-2},$$
$$a=\frac{2M+h\tau-h^{2}}{q},\quad b=\frac{M}{q},\quad
c_0=\frac{M+h\tau+h^{2}}{q},\quad c_1=-\frac{2M+h\tau}{q},\quad
c_2=\frac{M}{q}.$$

**Boundary.** $M=0$ gives exactly
$s_{t+1}=(1-\tfrac h\tau)s_t+(1+\tfrac h\tau)f_t-f_{t-1}$, which is checked
as an exact rational identity.

$h=1$ **token**. It is deliberately **not** identified with S5's learned
continuous-time step $\Delta$, which enters only through $\bar A,\bar B$.

**Passive factorization.** $P(D)=(1+t_+D)(1+t_-D)$ with
$t_\pm=\tfrac\tau2(1\pm\sqrt{1-4\varepsilon})$ is an exact **continuous**
identity on $0\le\varepsilon\le\tfrac14$. It is *not* used as an
implementation shortcut: two discrete first-order factors reproduce the
update above only if both use the same discrete derivative, and here the
state and target stencils differ. The factorization is tested as an identity
about $\tau$ and $M$, and used for nothing else.

## 4. Native-matched target and the order-3 recurrence

With $f_t=F_\tau s_t+G_\tau x_t$, $F_\tau=I+\tfrac\tau h(\bar A-I)$,
$G_\tau=\tfrac\tau h\bar B$, substitution gives

$$s_{t+1}=A_0s_t+A_1s_{t-1}+A_2s_{t-2}+C_0x_t+C_1x_{t-1}+C_2x_{t-2},$$
$$A_0=aI+c_0F_\tau,\quad A_1=-bI+c_1F_\tau,\quad A_2=c_2F_\tau,\quad
C_i=c_iG_\tau .$$

Both forms are checked coefficient-by-coefficient in exact arithmetic by
evaluating the (linear) difference on a basis of
$(s_t,s_{t-1},s_{t-2},x_t,x_{t-1},x_{t-2})$.

**Prehistory** is zero: $s_{-1}=s_{-2}=0$, $x_{-1}=x_{-2}=0$, and the tests
check that an impulse at $t$ influences nothing before $t$ — no wraparound.
For **bidirectional** layers the same causal recurrence runs on the reversed
input sequence and is flipped back; no forward-history shift is ever applied
to a concatenated state.

## 5. The scan: a real parallel prefix, no per-token matrices

$F_\tau$ is diagonal in the mode basis, so $A_0,A_1,A_2$ are diagonal and
each mode carries a scalar order-3 recurrence whose companion matrix

$$H=\begin{bmatrix}A_0&A_1&A_2\\1&0&0\\0&1&0\end{bmatrix}$$

is **constant in time**. The associative composition
$(H_2,d_2)\circ(H_1,d_1)=(H_2H_1,H_2d_1+d_2)$ then collapses: a
Hillis–Steele doubling

$$v\leftarrow v+H^{(2^{k})}\,\mathrm{shift}(v,2^{k}),\qquad
H^{(2^{k+1})}=\big(H^{(2^{k})}\big)^{2}$$

computes every prefix in $\lceil\log_2 L\rceil$ levels while carrying **one
3×3 matrix per mode per level** — never an $L\times P\times3\times3$ tensor.
The invariant
$v^{(k)}_t=\sum_{j=t-2^{k}+1}^{t}H^{t-j}d_j$ is proved in the module
docstring and verified exactly, in rational arithmetic, against a sequential
rollout for lengths 1…100 including non-powers of two.

The production path calls neither `scan_companion` nor
`scan_companion_sequential`, contains no loop over tokens, and does not
rematerialize the rollout per step.

**Memory: three different quantities, only one of them established.**

| quantity | status |
|---|---|
| forward live storage | $O(LP\cdot 3)$: the lifted state and one shifted copy. Established by construction. |
| backward (autodiff residual) storage | **not** $O(LP)$. With `remat="level"` (the default) each level recomputes its internals, but reverse mode still needs every level **boundary**, so residuals are bounded by about $\lceil\log_2L\rceil$ arrays of size $LP\cdot3$, i.e. $O(LP\log L)$, unless XLA elides some. Per-level checkpointing improves the constant, not the log factor. |
| measured peak device memory | **unknown until the GPU benchmark runs.** It is the authority. |

An earlier version of this document claimed the $(L,P,3)$ state was the only
large array under per-level checkpointing. That was wrong and is withdrawn.

`remat="whole"` is available and keeps only the scan's inputs, giving
$O(LP\cdot3)$ residuals at roughly twice the forward flops. If the benchmark
shows the residuals are too large, the response is that switch, a coarser
rematerialization region, or a custom VJP for the structured recurrence —
**never** a change to the mathematics.

**No eigendecomposition** is used anywhere in the forward or backward path.
The companion spectral radius, a diagnostic, is computed as
$\lVert H^{n}\rVert_F^{1/n}$ with $n=1024$ by repeated squaring with the
magnitude carried in the log: submultiplicativity means it never
*understates* the radius, it needs no branch choice, it cannot overflow for
unstable modes, and `jnp.linalg.eigvals` (unavailable on the GPU backend) is
avoided entirely.

## 6. The arms

| code identifier | what it is |
|---|---|
| `native_matched_s5` | Native S5. Control. **Unchanged, byte for byte, including its code path.** |
| `generalized_prospective_s5` | the OLD two-compartment (M, γ, T) arm. Frozen, kept for comparison, keeps its own partial results. Not a WWJ arm. |
| `wwj_critical_s5` | new. $M=\tau^{2}/4$, $\varepsilon$ fixed at the critical value. |
| `wwj_passive_s5` | new. $M=\varepsilon\tau^{2}$, $\varepsilon=\tfrac14\sigma(\cdot)\in(0,\tfrac14)$ learned. |
| `wwj_unconstrained_s5_diagnostic` | **diagnostic only**, $\varepsilon$ unbounded above, possibly underdamped. Not the biological model; any result from it is reported as a diagnostic. |

Parameterization: $\tau=\mathrm{TAU\_MIN}+\mathrm{softplus}(\cdot)>0$, so
$q=M+h\tau\ge h\cdot 10^{-3}$ and can never approach zero.
**Granularity is declared**: one $\tau$ (and one $\varepsilon$) per S5
**layer**, broadcast across that layer's modes — deliberately different from
the older per-mode arms, and stated rather than silently changed.

Recurrence eigenvalues are **never** clipped or projected after an update.
If a projection is ever needed it will be defined mathematically and run as
its own declared intervention.

Optimizer safeguards for $\tau,\varepsilon$ (in
`experiments/s5_wwj/wwj_model.py`, WWJ-only): their own optimizer group at
**0.1×** `ssm_lr`, **no weight decay** (decay pulls the *raw* value toward an
arbitrary timescale, not toward zero), everything else exactly the
production `noBCdecay` configuration, including the absence of global
gradient clipping.

## 7. Initialization is chosen before any performance is seen

`experiments/s5_wwj/init_grid.py` sweeps $k=\tau/h\in\{0.05,0.1,0.25,0.5,1\}$
and $\varepsilon\in\{0,1/16,1/4\}$ over the **actual initialized S5 modes**,
records the companion spectral radius of **every mode of every layer**, and
applies a rule declared in that file's docstring: admissible = max radius
≤ 0.98 and all coefficients finite; choose the **largest admissible $k$ at
the critical $\varepsilon=1/4$**; the passive arm starts at that $k$ with
$\varepsilon=1/16$. If nothing is admissible it reports
`NO_ADMISSIBLE_INITIALIZATION` and selects nothing. No validation or test
number is consulted.

## 8. Gates before any training

**Both principal arms are gated separately.** `wwj_critical_s5` and
`wwj_passive_s5` each run the JAX correctness suite, the performance gate
and the developmental gate, into their own artifact directories
(`<run>/<arm>/benchmark.json`, `<run>/<arm>/dev_gate/dev_gate.json`). A
critical-arm pass authorizes **nothing** about the passive arm. The
initialization grid runs once and is shared, because it depends only on the
initialized S5 modes.

**Performance gate** (`experiments/s5_wwj/benchmark.py`), Native vs one WWJ
arm on
the same GPU, same batch 16, length 16,000, width 96, depth 6,
bidirectional, same update and precision. It reports compile time
separately, steady-state steps/minute, peak GPU memory, forward time,
backward time, the ratio to Native, and the projected 15-epoch wall clock
with three seeds concurrent. It authorizes training only if: the
production-shaped step completes; peak memory ≤ 80% of the device; loss,
gradients and states finite; the scan matches the sequential oracle; the
projection fits the allocation with ≥ 25% margin; and WWJ throughput is
≥ 0.5× Native. **Below that ratio the instruction is to optimize the scan,
not to start training.**

**Developmental gate** (`experiments/s5_wwj/dev_gate.py`): production-shaped
initialization and compilation, two genuine optimizer updates, a fixed
training subset swept three times, checkpoint save and reload, and
validation — never the test split. It passes only if everything stays
finite, the subset loss falls by ≥ 2%, the companion radii stay ≤ 1.05, and
the checkpoint restores. **This is a developmental result and is reported
separately from any scientific one.**

Diagnostics recorded per task and checkpoint: $\tau$, $\varepsilon$,
$M=\varepsilon\tau^2$, the companion spectral radius of every mode, state
and gradient norms, nonfinite counters, throughput, peak GPU memory,
validation accuracy and cross-entropy, the recurrence coefficients, the
continuous $P(\lambda)=1+\tau\lambda+M\lambda^{2}$, and — separately
labelled — the discrete-time transfer function of the *implemented*
recurrence,
$S(z)/U(z)=(c_0+c_1z^{-1}+c_2z^{-2})/(z-A_0-A_1z^{-1}-A_2z^{-2})$.

$P(\lambda)=0$ is **not** a target: exact cancellation would delete that
mode's memory. The hypothesis is partial temporal compensation with useful
long-delay information preserved.

## 9. Optional continuation diagnostic

If direct WWJ training is unstable, the update may be interpolated from
Native, $(1-g)\,\text{Native}+g\,\text{WWJ}$ with $g$ scheduled to reach
exactly 1. It is **not** implemented as a principal arm here, changes the
training protocol, and would be reported separately.

## 10. What has not been measured yet

No benchmark number exists: producing one needs an RTX 3090, and it is the
cluster's job, not this commit's. The JAX-level tests
(`tests/test_wwj_recurrence.py`) likewise need JAX, which the development
machine does not have; the coefficient algebra and the scan algorithm are
nevertheless proved here in exact rational arithmetic without JAX
(`tests/test_wwj_algebra.py`), so what remains to be confirmed on the
cluster is the JAX implementation of proven algebra, not the algebra.

Nothing in this commit launches anything.
