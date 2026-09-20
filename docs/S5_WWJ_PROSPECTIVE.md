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

## 2. The mixed-stencil realization was REJECTED

The first WWJ realization discretized `P(D)(s−f)=0` with **different**
stencils on the state and the target. It was rejected on the cluster at
commit `bfe53fe758c40f5217617c5f93f7b3c4c0498f77`, on the actual
production-initialized S5 modes:

```
status: NO_ADMISSIBLE_INITIALIZATION      (nothing selected)
critical eps = 0.25:
  k=0.05  max companion radius 1.7054537181
  k=0.10  max companion radius 1.7050817610
  k=0.25  max companion radius 1.7061925973
  k=0.50  max companion radius 1.7181166321
  k=1.00  max companion radius 1.8767736156
float32 production path: NaN. float64 states reached about 1e81.
```

Every coefficient was finite; **every declared cell was unstable**, including
the smallest timescale. This was not a tolerance problem, not an
initialization search that needed widening, and not something a smaller grid
value fixes.

**Why, mathematically.** `P(D)(s−f)=0` is `P(D)s = P(D)f`, so for matched
initial conditions the transfer function is

$$S/F = 1 .$$

The exactly matched residual law controls only **homogeneous residual
transients**; it creates **no persistent prospective transformation of $f$**.
(And if $f$ depends on the current state, an exact discrete form additionally
risks becoming an implicit target-manifold constraint.) The mixed-stencil
construction escaped that cancellation only by applying different discrete
derivatives to $s$ and to $f$ — so its extra poles were **artefacts of the
discretization**, and the cluster showed those artefacts are unstable for the
real S5 modes.

It is preserved, with this evidence, as a failed diagnostic/ablation:
`s5/wwj_mixed_stencil.py`, `s5/wwj_mixed_stencil_ssm.py`, arm identifier
`wwj_mixed_stencil_unstable_diagnostic`, tests in
`tests/test_wwj_mixed_stencil_algebra.py` (the algebra, which is still
correct) and `tests/test_wwj_mixed_stencil_rejected.py` (which now **asserts
the instability** instead of expecting finiteness). It is not a principal
arm, `experiments/s5_wwj/wwj_model.py` refuses to construct it, and the
launcher refuses it by name. It is **not production viable** and must never
be described as such.

## 3. The corrected architecture: Native memory, exact discrete WWJ readout

The Native S5 memory is untouched,

$$s_{t+1}=\bar A s_t+\bar B x_t,$$

and the WWJ operator is applied to that trajectory with **one consistent
backward derivative**,

$$D_hs_t=\frac{s_t-s_{t-1}}{h},\qquad
D_h^2s_t=\frac{s_t-2s_{t-1}+s_{t-2}}{h^2},$$

which satisfies $D_h(D_hs)=D_h^2s$ exactly — the property the mixed-stencil
form lacked. With $k=\tau/h$ and $m=M/h^2=\varepsilon k^2$,

$$z_t=P(D_h)s_t=(1+k+m)s_t-(k+2m)s_{t-1}+ms_{t-2}.$$

This is the **exact discrete WWJ operator**, not an approximation of one. It
is FIR: three taps, finite support, **zeros and no poles**.

**Exact recurrent realization.** With $q_t=[s_t;s_{t-1};s_{t-2}]$,

$$q_{t+1}=\begin{bmatrix}\bar A&0&0\\ I&0&0\\ 0&I&0\end{bmatrix}q_t
+\begin{bmatrix}\bar Bx_t\\0\\0\end{bmatrix},\qquad
z_t=\begin{bmatrix}(1+k+m)I&-(k+2m)I&mI\end{bmatrix}q_t,$$

so this is a genuine recurrent state-space S5 layer carrying the WWJ
generalized-prospective coordinate. The transition is block lower triangular
with diagonal blocks $\bar A,0,0$, so its characteristic polynomial is
$(z-\lambda)z^2$: **the recurrent poles are exactly the Native poles plus two
zero history-shift poles.** WWJ cannot destabilize what Native S5 does not
already do. `tests/test_wwj_operator_algebra.py` computes that determinant
rather than asserting it in prose.

**Implementation.** The augmented matrix is never scanned. The ordinary,
optimized Native S5 parallel scan runs first — literally
`jax.lax.associative_scan(s5.ssm.binary_operator, …)`, the same primitive
`apply_ssm` uses — and the three-tap is applied to its states, an $O(LP)$
elementwise pass with no loop, no scan and no rematerialization. Equality
with the augmented realization is proved in exact rational arithmetic and
checked numerically against a sequential oracle at production width.

**Boundaries.** Zero prehistory, $s_{-1}=s_{-2}=0$, with no wraparound at
either end. For **bidirectional** layers the Native suffix scan is used and
the three-tap is applied in that direction's own causal order, then flipped
back; no forward-history shift touches a concatenated state. $h=1$ **token**,
never identified with S5's learned $\Delta$.

**Passive factorization.** $P(D_h)=(1+t_+D_h)(1+t_-D_h)$ with
$t_\pm=\tfrac\tau2(1\pm\sqrt{1-4\varepsilon})$ is **exact here** — both
stages use the same discrete derivative — and applying the two stages
sequentially with zero prehistory reproduces the three-tap exactly. Proved,
not assumed.

**$\tau\to0$ recovers Native S5 exactly**: $k=m=0$ makes the operator the
identity.

## 4. The arms

| code identifier | what it is |
|---|---|
| `native_matched_s5` | Native S5. Control. **Byte-identical, untouched.** |
| `generalized_prospective_s5` | the OLD two-compartment (M, γ, T) arm. Frozen, keeps its own partial results. Not a WWJ arm. |
| `wwj_critical_s5` | **principal.** $M=\tau^2/4$, $\varepsilon=1/4$ fixed. |
| `wwj_passive_s5` | **principal.** $M=\varepsilon\tau^2$, $\varepsilon=\tfrac14\sigma(\cdot)\in(0,\tfrac14)$ learned. |
| `wwj_gated_recoverable_s5_diagnostic` | **diagnostic.** $\tilde s_t=s_t+g[k(s_t-s_{t-1})+m(s_t-2s_{t-1}+s_{t-2})]$, i.e. $s+g(z-s)$. A **separate intervention**: the direct arm only at $g=1$, Native only at $g=0$. Never described as identical to the direct arm. |
| `wwj_mixed_stencil_unstable_diagnostic` | **REJECTED**, §2. Failed ablation, never in a launcher. |

Readout for the principal arms: $y_t=Cz_t+Dx_t$ (with the conjugate-symmetry
factor `apply_ssm` uses).

Parameterization: $\tau=\text{TAU\_MIN}+\text{softplus}(\cdot)>0$,
$\varepsilon=\tfrac14\sigma(\cdot)$, **one $\tau$ and one $\varepsilon$ per
layer**, declared. No eigenvalue is projected after an update — with an FIR
operator there is nothing to project. The WWJ parameters keep their own
optimizer group at **0.1×** `ssm_lr` with **no weight decay**; the rest is
the production `noBCdecay` configuration.

## 5. Initialization: FIR gain and gradients, not companion radii

Because the operator adds no poles, `experiments/s5_wwj/init_grid.py` no
longer selects on companion radii. It sweeps $k\in\{0.01,0.05,0.1,0.25,0.5,1\}$
and $\varepsilon\in\{0,1/16,1/4\}$ over the actual initialized modes and
records: the **Native** $|\bar A_j|$ radii (unchanged by WWJ), the maximum
FIR gain
$\max_\omega|P_h(e^{i\omega})|$ with
$P_h(e^{i\omega})=1+k(1-e^{-i\omega})+m(1-e^{-i\omega})^2$ over a dense grid,
finiteness of the forward pass, and the **finite, non-vanishing** gradients
of a probe loss with respect to the raw WWJ parameters.

Declared rule: admissible = finite forward ∧ max FIR gain ≤ **3.0** ∧ finite
WWJ gradients with relative magnitude > **1e-8**; among admissible cells at
the critical $\varepsilon=1/4$ take the **smallest** $k$ — start close to
Native, since $\tau\to0$ *is* Native, with the gradient floor preventing
"close to Native" from meaning "cannot learn". The passive arm starts at that
$k$ with $\varepsilon=1/16$. Nothing admissible → `NO_ADMISSIBLE_INITIALIZATION`
and no selection. No validation or test number enters the rule.

## 6. Gates before any training

Both principal arms are gated **separately**, into their own artifact
directories; a critical pass authorizes nothing about the passive arm. The
initialization grid runs once and is shared.

**Performance gate** (`experiments/s5_wwj/benchmark.py`): Native vs one WWJ
arm, same GPU, batch 16, length 16 000, width 96, depth 6, bidirectional,
same update and precision. Reports compile time separately, steady-state
steps/minute, peak GPU memory, forward and backward time, the ratio to
Native, and the projected 15-epoch wall clock. Authorizes training only if
the production-shaped step completes; peak memory ≤ 80% of the device;
everything finite; the optimized path matches **both** the sequential oracle
and the augmented realization; the projection fits with ≥ 25% margin; and
throughput ≥ **0.80×** Native — raised from 0.50 because the expensive part
is now the existing optimized Native scan and WWJ adds only an $O(LP)$
three-tap. Below the floor: optimize, do not train.

**What the projection covers** is stated, not implied:
`hours_one_arm_wave_three_concurrent_seeds`,
`hours_both_wwj_arms_sequential`, `waves_planned`, `hours_planned`, `covers`,
`excludes` (the Native and Zucchet waves are in neither number). `--waves`
declares the plan and the time gate applies to `hours_planned`.

**Developmental gate** (`experiments/s5_wwj/dev_gate.py`): production-shaped
initialization and compilation, two genuine optimizer updates, a fixed subset
swept three times, checkpoint save and reload, validation — never the test
split. Passes only if everything stays finite, the subset loss falls by ≥ 2%,
the FIR gain stays ≤ 3.0, the Native radii stay inside the unit disc, and the
checkpoint restores. Reported as **developmental only**.

Diagnostics per task and checkpoint: $\tau$, $\varepsilon$, $M$, $k$, $m$,
the max FIR gain and the response on a coarse frequency grid, the **Native**
per-mode radii, state/output/gradient norms, nonfinite counters, throughput,
peak GPU memory, validation accuracy and cross-entropy, and the gate value
for the gated diagnostic arm.

$P(\lambda)=0$ is **not** a target: exact cancellation would delete that
mode's memory. The hypothesis is partial temporal compensation with useful
long-delay information preserved.

## 7. What has not been measured

No throughput, memory or learning number exists for this architecture: that
needs the RTX 3090 and is the cluster's job, not this commit's. Peak device
memory is likewise **unknown until the GPU benchmark runs**; it is the
authority. The JAX-level tests (`tests/test_wwj_ssm.py`) need JAX, which the
development machine does not have; the mathematics is nevertheless proved
here in exact rational arithmetic without JAX
(`tests/test_wwj_operator_algebra.py`), so what the cluster confirms is the
implementation of proven algebra.

Nothing in this commit launches anything.
