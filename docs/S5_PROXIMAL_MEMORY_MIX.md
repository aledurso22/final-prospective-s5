# Proximal NLA with a positive fast/slow direction memory

Date: 2026-11-18. This is a predeclared research candidate, not an established
accuracy improvement. It changes the S5 state recurrence; the outer training
optimizer, input data and selection protocol are unchanged. The source of the
proposal is an exact interpretation of P followed by a different, published
principle for using old gradients.

## What existing P already computes

For a fixed quadratic objective, let `g=H x-d`, `R=I+mu H` and
`g_mu=R^-1 g`. The proximal correction solves `y+mu g(y)=x`, so its force
is `g_mu(x)=g(y)`. For a fixed-input linear layer with squared loss, the
Hessian block is an activation Gram matrix and this is FOOF's ridge-inverse
direction, up to a scalar. A general Hessian is not an activation Gram matrix.

The current P equation, with `m=ell=tau` and `c=1`, is

\[
\tau\ddot x+\dot x=-g_\mu-\tau\dot q,
\qquad \tau\dot q=g_\mu-q.
\]

Let `d_F=-g_mu`, `v=xdot` and `q_d=-q`. Eliminating the auxiliary state gives

\[
v=K_P(\partial_t)d_F,\qquad
K_P(p)=\frac{1+2\tau p}{(1+\tau p)^2}
=\frac2{1+\tau p}-\frac1{(1+\tau p)^2}.
\]

Thus P applies a continuous double-exponential moving-average extrapolation
to its proximal/FOOF direction. The causal kernel is
`(2/tau-t/tau^2) exp(-t/tau)`: its tail becomes negative after `2*tau`.
It compensates first-order lag (`K_P(0)=1`, `K_P'(0)=0`), rather than retaining
an independently positive slow contribution. This is exact for the fixed
metric and matched filter initialization. For two cascaded traces `a,b`,
`v=2a-b` requires `a(0)=q_d(0)`, `b(0)=2q_d(0)-v(0)`.

For a fixed quadratic, multiplication by R and transformation `Q=R q`
also gives exactly `R(tau*xdd+xd)=-g-tau*Qdot`, `tau*Qdot=g-Q`.
Proximal-force and fixed-metric formulations coincide in this setting.
The older proximal report's statement that they are distinct is too broad.
If R varies, derivatives of R appear and this identity cannot be assumed.

## Literature and the one change

Pagliardini, Ablin and Grangier's **AdEMAMix** (ICLR 2025), section 3, uses
two *parallel* gradient EMAs with separate fast and slow horizons. Appendix
C.3.1 explicitly contrasts this with DEMA: subtracting a second EMA removes
old-gradient weight, whereas their aim is sensitivity to recent gradients
alongside a substantial old-gradient contribution. This is the motivation
for changing the direction-memory kernel, rather than adding another force.

Keep exactly the same proximal force and use

\[
\boxed{\begin{aligned}
d_F&=-R^{-1}g,\\
\tau_f\dot v_f+v_f&=d_F,\\
\tau_s\dot v_s+v_s&=d_F,\\
\dot x&=(1-w)v_f+w v_s,\quad 0<w<1,\quad 0<\tau_f<\tau_s.
\end{aligned}}
\]

The normalization preserves unit DC gain of the direction filter. It is
our adaptation, not the full AdEMAMix optimizer. Published AdEMAMix uses
`m1_hat+alpha*m2`, an adaptive second-moment denominator, weight decay and
warmup schedules. Its appendix C.1.9 discusses why normalization and
learning-rate schedules must be considered together. No published AdEMAMix
accuracy result is claimed for this S5 state recurrence.

This candidate retains the fixed-quadratic FOOF direction for constant input
direction, and in the vanishing-memory limit. Eliminating `v_f,v_s` yields
an explicit change relative to NLA history:

\[
\tau_f\ddot x+\dot x
=-g_\mu+w(\tau_s-\tau_f)\dot q,
\qquad \tau_s\dot q=g_\mu-q.
\]

The history-derivative sign is opposite to P. That is intentional: the
positive mixture retains old directions instead of extrapolating through
their change. It should not be described as additional Hessian damping.

For a fixed SPD R, the energy

\[
\mathcal V=E(x)-E_*+
\tfrac12(1-w)\tau_f v_f^T Rv_f+
\tfrac12w\tau_s v_s^T Rv_s
\]

obeys

\[
\dot{\mathcal V}=-(1-w)v_f^T Rv_f-wv_s^T Rv_s\le0.
\]

This identity follows directly by substituting the equations. For a strictly
convex quadratic, the only invariant zero-dissipation state has `g=0`.
It does not give a convergence-rate improvement over FOOF, and a complex
S5 resolvent is not an SPD Hessian metric. S5 needs the separate proof below.

## Complex S5 mapping and sharp uniform bound

Native continuous S5 in token units is
`sdot=-kappa*s+b`, where `kappa=-Lambda*Delta` and `b=Delta*B*u`.
For `Re(kappa)>0`, its two-real-coordinate residual is strongly monotone,
although generally not the gradient of a real potential. Its resolvent gives

\[
r_\mu(s,u)=\frac{\kappa s-b(u)}{1+\mu\kappa}.
\]

Both the state coefficient and physical input are divided by the same
denominator. Substitute `d_F=-r_mu` in the fast/slow system above, initialize
its three states to zero in each sequence direction, and discretize that
complete driven system with exact physical-input zero-order hold.
The construction has three complex states per native mode, as does P.

Let

\[
K(p)=\frac{1-w}{1+\tau_f p}+\frac{w}{1+\tau_s p},\qquad
\bar\tau=(1-w)\tau_f+w\tau_s.
\]

The characteristic polynomial is

\[
(1+\mu\kappa)p(1+\tau_f p)(1+\tau_s p)
+\kappa[1+((1-w)\tau_s+w\tau_f)p]=0.
\]

Write it as `1+kappa*F(p)=0` with `F=mu+K(p)/p`. Then

\[
\Re F(i\omega)=\mu-
\frac{(1-w)\tau_f}{1+\tau_f^2\omega^2}-
\frac{w\tau_s}{1+\tau_s^2\omega^2}.
\]

Consequently **mu >= bar_tau** makes F positive real: its finite poles away
from zero are in the left half-plane, its zero-frequency pole has positive
residue, and its boundary real part is nonnegative. Since `Re(kappa)>0`,
`-1/kappa` has negative real part and cannot equal F in the closed right
half-plane. The polynomial is nonzero at zero, and `1+mu*kappa` cannot vanish.
Every homogeneous pole is therefore strictly stable. The threshold is sharp
for this uniform statement: if `mu<bar_tau`, sufficiently small nonzero
omega gives `Re F(i omega)<0`; choosing `kappa=-1/F(i omega)` gives an
operator in the open right half-plane with an imaginary-axis closed-loop pole.

This is our calculation, not a coefficient copied from AdEMAMix or FOOF.
The guarantee concerns fixed parameters and homogeneous poles. Input-driven
gain, transient amplification, floating-point rounding and learning remain
separate questions. Exact ZOH preserves strict stability mathematically;
it does not provide a uniform numerical margin near `Re(kappa)=0`.

## The declared comparison isolates the memory shape

Take the already-used P horizon `h=1/abs(kappa)` and keep `mu=h/2`:

| Quantity | Existing P, alpha=1 | Declared positive mix |
|---|---:|---:|
| Proximal coefficient mu | h/2 | h/2 |
| First direction-memory horizon | h | h/10 |
| Second direction-memory horizon | h | h |
| Direction combination | 2 EMA - nested EMA | (5/9) fast + (4/9) slow |
| Minimum sufficient uniform mu | h/2 | (5/9)(h/10)+(4/9)h = h/2 |
| Complex states per native mode | 3 | 3 |
| Added trainable parameters | 0 | 0 |

The horizon ratios 0.1 and 1 reuse the two scales already investigated in
H/P. They are an explicit modeling choice, not theoretically optimal.
We choose the **largest slow weight allowed by the uniform stability bound**
at those horizons and the unchanged regularization budget; that gives `w=4/9`
algebraically. Smaller weights are also allowed. The point is to compare memory shape at the same proximal
force, not to choose the best ratio after looking at validation accuracy.

The state transfer is

\[
H(p)=\frac{K(p)}{(1+\mu\kappa)p+\kappa K(p)}.
\]

For both P and the positive mix, `H(0)=1/kappa` and
`-H'(0)/H(0)=1/kappa+mu`. Thus both have exactly the same continuous DC gain
and delay. The positive velocity filter's own delay `bar_tau` cancels between
input and residual feedback at first order; it must not be added a second
time to the state-transfer delay. Their finite-frequency resonances and
transients need not be similar.

## Read-only diagnostic and interpretation

`experiments/s5_proximal_memory_mix/diagnostic.py` draws each seed using the
existing subprocess helper with x64 disabled. It then computes kappa using
the production float32 JAX operations and evaluates independent SciPy
float64 state matrices. Seeds 301/302, all six layers and all 384 modes per
seed are retained. No train or validation example is loaded.

It reports continuous and exact-ZOH poles, continuous residues, repeated-pole
proximity, sampled signed-band frequency peaks, actual B/C norm coupling
proxies, and every-token 16,000-step impulse/step metrics. Modal component
energy is labeled as such and uses the exact total energy in its denominator;
component numerator energies omit interference and their shares may exceed 1.
Frequency peaks use a full [-pi,pi] grid and additional grids centered on each
pole's frequency. They are sampled maxima, not certified H-infinity bounds.

A predeclared sensitivity grid uses fast ratios {0.05,0.1,0.25}, slow ratios
{1,2,4}, and `w=(0.5-fast)/(slow-fast)`. Every cell retains the same mu and
regularization budget. This grid is diagnostic; it does not select the
training candidate by accuracy.

Reproduce from the repository root, using the compatible Python environment:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest \
  tests/test_proximal_memory_mix_diagnostic.py -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m \
  experiments.s5_proximal_memory_mix.diagnostic \
  --out results/s5_proximal_memory_mix/diagnostic_recheck.json \
  --raw-out results/s5_proximal_memory_mix/diagnostic_recheck_raw.json
```

The diagnostic refuses an existing output file. Its wall time is diagnostic
cost, not training throughput. Full-size float32 GPU smoke and training cost
must be measured separately by the launcher.

### Completed local initialization diagnostic

The independent reference run completed in 21.8 seconds on the local CPU.
Both seeds cover all 384 modes across all six layers. Native, P and the
declared mix have zero unstable continuous or exact-ZOH modes, as do all
eight additional predeclared sensitivity cells. This is initialization
evidence only; it does not inspect any trained checkpoint.

| Per-mode median (seed 301; seed 302 agrees to shown precision) | Native | P, alpha=1 | Declared mix |
|---|---:|---:|---:|
| Slowest decay time / native | 1.000 | 1.546 | 0.865 |
| Sampled peak frequency gain / native peak | 1.000 | 1.115 | 0.532 |
| 16,000-token impulse energy / native | 1.000 | 0.817 | 0.332 |
| Impulse relative L2 difference from native | 0.000 | 0.535 | 0.797 |
| Auxiliary component energy / exact total (proxy numerator) | 0.000 | 0.065 | 0.048 |
| Step-response peak / DC magnitude | 1.606 | 1.505 | 1.364 |

The declared mix substantially reshapes the response: it damps the native
resonant peak and shortens the median slowest decay time. The positive
slow-direction branch does **not** establish longer or more useful observable
S5 memory. In particular, its auxiliary component-energy proxy remains small
at initialization. Its larger departure from native is a material empirical
risk, even though its equilibrium and continuous DC delay match P.

The sensitivity grid makes this tradeoff visible rather than hiding it.
Increasing the slow-horizon ratio to 2 or 4 at the fixed regularization budget
reduces its allowed slow weight; the median peak/native gain falls further
(approximately 0.33--0.47), and median slow/native decay is approximately
0.60--0.66. These measurements do not select another launch candidate.

The checked-in `docs/evidence/s5_proximal_memory_mix/initial_diagnostic.json`
contains compact overall and per-layer summaries, configurations, and a
SHA-256 of the full raw evidence. The full per-mode arrays were preserved in
gitignored `results/s5_proximal_memory_mix/initial_diagnostic_raw.json`.
The same command produces both files; `--raw-out` changes the raw destination.
Reproducing requires new output paths because both outputs are protected
against overwrite. The source diagnostic's ten tests passed locally.

## Primary sources and limits of attribution

- Benzing (ICML 2022), *Gradient Descent on Neurons and its Link to Approximate
  Second-Order Optimization*, especially the regularized neuron-update
  least-squares construction:
  https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf
- Pagliardini, Ablin and Grangier (ICLR 2025), *The AdEMAMix Optimizer: Better,
  Faster, Older*, section 3, appendix C.1.9 and C.3.1:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/a2cf225ba392627529efef14dc857e22-Paper-Conference.pdf
- Chen et al. (ICML 2023), *Bidirectional Looking with A Novel Double
  Exponential Moving Average to Adaptive and Non-adaptive Momentum
  Optimizers*, section 2.2, gives a related DEMA variant, not this state law:
  https://proceedings.mlr.press/v202/chen23r/chen23r.pdf
- Attouch and Laszlo (2020), *Newton-like Inertial Dynamics and Proximal
  Algorithms Governed by Maximally Monotone Operators*, DOI
  https://doi.org/10.1137/20M1333316 supplies the broader Yosida-regularized
  inertial-operator precedent. Its convergence theorem is not imported here.

The restricted FOOF algebra, positive-memory adaptation, energy identity and
complex stability calculation are separated deliberately. None establishes
a classification gain, a new best optimizer, or a two-to-three-point S5 gain.

## Training handoff

The new entry point is `experiments.s5_proximal_memory_mix.runner`. The existing
constructors, shared runner and all prior results remain unchanged. Training
uses the original official-list SC10 cache, batch 16, 15 epochs, optimizer,
dropout, and validation selection rule. No test-set evaluation is performed.

`bin/run_experiments/submit_s5_proximal_memory_mix.sh` submits seeds 301 and 302
on two separately assigned RTX 4090 GPUs on **pgi14-gpu1**, partition **pgi14**.
It requests 16 CPUs, 100 GB RAM and an eight-hour limit. The limit is a resource
reservation, not a measured runtime prediction. Each job clones the pinned
published commit into its own node-local directory and writes to
`/Users/durso/s5-runs/s5-proximal-memory-mix/job-<jobid>/`.

From a clean checkout at the reviewed, published commit on the cluster:

```sh
EXPECTED_COMMIT="$(git rev-parse HEAD)" DRY_RUN=1 \
  bash bin/run_experiments/submit_s5_proximal_memory_mix.sh
EXPECTED_COMMIT="$(git rev-parse HEAD)" \
  bash bin/run_experiments/submit_s5_proximal_memory_mix.sh
```

The submission script does not publish local code. Before submitting, its
commit must be present on the remote repository. `STAGE=smoke` runs only the
profile and smoke stage. The default `STAGE=train` profiles both tasks and
requires both full-size GPU smokes to pass before either fresh training run.
Those gates bind the exact source, data, seed, recurrence coefficients,
precision, checkpoint restoration, actual block stability and finite gradients.
Compiler memory accounting is checked where available; successful full-size
GPU execution is still required. Local CPU checks do not establish GPU success.

The primary comparison is paired selected-validation accuracy against existing
P alpha=1 for seeds 301/302, with native S5 as the practical reference. Report
cross-entropy, per-seed differences, stability, runtime and memory alongside
accuracy. This pilot tests a memory principle; it is not predeclared to win.

Local validation commands (separate precision processes):

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest \
  tests/test_proximal_memory_mix.py \
  tests/test_proximal_memory_mix_runner.py \
  tests/test_proximal_memory_mix_diagnostic.py \
  tests/test_proximal_memory_mix_launcher.py \
  tests/test_proximal_memory_mix_float32.py -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest \
  tests/test_proximal_memory_mix.py -q
```
