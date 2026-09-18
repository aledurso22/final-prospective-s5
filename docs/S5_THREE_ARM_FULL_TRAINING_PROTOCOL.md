# Three-arm full-training raw-audio S5 recurrence experiment

Status: preregistered; full preflight complete; no training run has been launched.

This protocol compares exactly three scientific arms on one full, from-scratch
Speech Commands experiment. It does not include prospective input coupling,
learned-mismatch as a separate arm, adaptive-current dynamics, doubled-mode
S5, Nesterov, QHM, or any historical checkpoint.

## Scientific arms

| Scientific name | Code identifier | Executed recurrence |
|---|---|---|
| Native S5 recurrence under the shared stability constraint | `native_matched_s5` | Native S5 ZOH recurrence |
| Zucchet prospective S5 recurrence | `zucchet_prospective_s5` | The (M=0,\gamma=1) member below, with positive response time (T) |
| Generalized prospective S5 recurrence \((M,\gamma,T)\) | `generalized_prospective_s5` | The finite-inertia second-order member below, with \(M>0\), \(\gamma=1\), and positive \(T\) |

The code identifiers are metadata only. Results are reported with the
scientific names above.

## Exact equations

The upstream S5 continuous-time pole and input row are clock-absorbed as
\(a=\Delta\lambda\) and \(b=\Delta\widetilde B\), where the learned S5 step
\(\Delta\) is measured in raw-audio sample intervals and
\(\operatorname{Re}(a)<0\). The physical sample interval is
\(h=1/16000\) seconds; the implementation uses the equivalent computational
clock \(h=1\) per raw sample.

\(\phi_1(a)=(e^a-1)/a\), evaluated by the repository's cancellation-safe
implementation and defined continuously as \(\phi_1(0)=1\).

The discrete raw waveform is represented as a piecewise-constant held input
inside each sample interval. Thus \(\dot x=0\) in the interval interior and
sample jumps are represented by the derivative-free prospective feedthrough
term. This is the interpolation assumption under which the ZOH transition is
exact; a piecewise-linear interpolation would be a different experiment.

### Native matched S5

\[
\dot s = a s + b x,\qquad
s_{k+1}=e^a s_k+\phi_1(a)b x_k,
\]
where \(\phi_1(a)=(e^a-1)/a\), with \(\phi_1(0)=1\).

### Zucchet prospective S5 recurrence

This is the ordinary prospective boundary \(M=0,\gamma=1\), not input-side
prospectivity. With response time \(T>0\),

\[
(I-Ta)\dot s = a s + b x + T b\dot x.
\]

For held tokens, its derivative-free realization is

\[
\dot s = a_0s+b_0x+d_0x,
\quad a_0=\frac{a}{1-Ta},\quad
b_0=\frac{b}{(1-Ta)^2},\quad
d_0=\frac{Tb}{1-Ta},
\]

or, equivalently, \(s=u+d_0x\) with

\[
\dot u=a_0u+b_0x.
\]

The production recurrence is therefore

\[
u_{k+1}=e^{a_0}u_k+\phi_1(a_0)b_0x_k,
\qquad s_k=u_k+d_0x_k.
\]

This explicitly declares the Zucchet discretization: exact ZOH of the
derivative-free state realization, not Euler, bilinear, input-side coupling,
or adaptive-current dynamics.

### Generalized prospective S5 recurrence \((M,\gamma,T)\)

\[
M\ddot s+(\gamma-Ta)\dot s-a s=b x+T b\dot x.
\]

For the production arm \(\gamma=1\), define the carried state
\(z=(s,w)\), where \(w=M\dot s-T(as+bx)\). For held tokens,

\[
\frac{d}{dt}\begin{bmatrix}s\\w\end{bmatrix}
=
\begin{bmatrix}
Ta/M & 1/M\\
(1-T/M)a & -\gamma/M
\end{bmatrix}
\begin{bmatrix}s\\w\end{bmatrix}
+
\begin{bmatrix}Tb/M\\(1-T/M)b\end{bmatrix}x.
\]

The production transition uses the exact 4×4 augmented construction over one
unit interval:

\[
K=\begin{bmatrix}A&I_2\\0&0\end{bmatrix},\quad E=e^K,\quad
A_{bar}=E_{1:2,1:2},\quad G=E_{1:2,3:4},\quad B_{bar}=GB.
\]

\[
z_{k+1}=e^{A}z_k+A^{-1}(e^A-I)B x_k,
\qquad y_k=s_k.
\]

The declared mathematical boundary is \(M=0\), where elimination of the
second-order state gives the Zucchet prospective recurrence above. Production
uses \(M>0\) because the finite-dimensional second-order parameterization is
singular at exactly zero mass. The equation/identity tests verify the exact
zero-mass coefficient boundary and convergence of the production transition
to it as \(M\downarrow0\) within the declared numerical domain. In production,
\(\gamma=1\) is fixed as the normalized damping gauge; both prospective arms
share `T_INIT`, and the generalized arm learns
\(\rho=RHO_{MIN}+(1-RHO_{MIN})\sigma(\rho_{raw})\) with \(M=\rho T\).
Allowing \(\gamma\) to learn would
add a fourth recurrence degree of freedom and is not part of this comparison.

## Training contract

All arms use the upstream S5 Speech Commands architecture: bidirectional S5,
depth 6, feature width `H=96`, nominal latent size `P=128`, 16 HiPPO blocks,
`half_glu1`, batch normalization, dropout `0.1`, and mean pooling. The decoder
is the upstream decoder with only its output width changed from 35 to 10.
Only the recurrence and its necessary recurrence parameters differ. `clip_eigs=True`
is shared across all three arms as a stability requirement for the prospective
recurrences; this is the sole shared recurrence-setting deviation from the
published Speech Commands configuration, which uses `clip_eigs=False`.

Speech Commands v0.02 uses the ten selected keywords, raw waveforms of length
16,000, and the official `validation_list.txt` and `testing_list.txt`
assignments. Training-derived raw-audio normalization follows upstream S5's
`normalize_all_data` convention: one training statistic per input channel over
examples and time, recorded in a dedicated cache manifest. Training and
validation arrays are opened by array
tasks; the test array is opened once, only by the finalizer after all nine
tasks succeed.

Each seed trains every arm from scratch. The same seed initializes corresponding
shared parameters wherever shapes permit and produces the same epoch-wise
training permutation for all three arms. No checkpoint or historical metric is
reused.

The fixed schedule is 40 epochs, batch size 16, global learning rate `0.008`,
SSM learning rate `0.002`, upstream `noBCdecay` parameter-group exceptions,
weight decay `0.04`, one warm-up epoch over exactly `steps_per_epoch` steps,
then cosine annealing over the remaining steps, and paired seeds `301`, `302`,
`303`. The prospective response leaves (`prospective_T_raw`,
`generalized_T_raw`, and `generalized_rho_raw`) are explicitly in the SSM Adam
group and receive no AdamW decay.
The checkpoint is the first epoch attaining the highest validation accuracy;
validation cross-entropy breaks ties. Test accuracy and test cross-entropy are
computed only from that selected checkpoint.

## Decision rule and outputs

The primary contrast is Generalized prospective S5 recurrence \((M,\gamma,T)\)
minus Zucchet prospective S5 recurrence. Native S5 recurrence under the shared
stability constraint is the baseline.
For each seed and the across-seed mean, the result schema records test
accuracy, selected epoch, test cross-entropy, parameter count, recurrent-state
size, training seconds, examples/second, paired differences, paired standard
errors, and 95% paired confidence intervals.

The report must not call the generalized method better unless it beats Zucchet
prospective S5 recurrence consistently across seeds and does not materially
worsen test accuracy or cross-entropy relative to Native matched S5.

## Slurm execution and preflight

The production launch is one array with exactly nine tasks (`3 arms × 3
seeds`) and at most four concurrent tasks. Each task requests partition
`pgi15-single-gpu`, node `pgi15-gpu3`, one RTX 3090, 10 CPUs, and 50 GB RAM.
The array script gives every task separate temporary, JAX compilation-cache,
log, checkpoint, and output directories. The finalizer is submitted with an
`afterok` dependency; it verifies all nine task manifests, selects checkpoints,
opens the test split once, and writes the paired report.

### Authoritative isolated preflight

The isolated full preflight passed at commit
`507f4d26905cb1b6c1c99eb28190e375cd85edea`.
It used artifact directory
`/Users/durso/s5-runs/s5-three-arm-preflight/isolated-final-20260918-233014`.
The measurements below are in scientific/reporting order; all gradients and
complete states were finite.

| Scientific arm | Steady step (s) | Peak VRAM (bytes) |
|---|---:|---:|
| Native S5 recurrence under the shared stability constraint | 0.1261024214 | 7,418,702,080 |
| Zucchet prospective S5 recurrence | 0.1564851347 | 7,419,634,688 |
| Generalized prospective S5 recurrence \((M,\gamma,T)\) | 0.2704331186 | 12,219,229,952 |

The Slurm dispatch map is intentionally longest-first for scheduler
efficiency—generalized seeds `301`, `302`, `303`, then Zucchet seeds `301`,
`302`, `303`, then Native seeds `301`, `302`, `303`. This changes dispatch
order only; scientific/reporting order, equations, initialization, data,
metrics, and finalization behavior are unchanged.

Prepare the cache, then submit without changing the checkout:

```bash
DATA_ROOT=/Local/durso/speech_commands_v0.02 \
  bash bin/run_experiments/cluster_prepare_s5_three_arm_full_data.sh
bash bin/run_experiments/cluster_s5_three_arm_full.sh
```

The exact-shape raw-audio preflight has completed successfully in isolated
processes; the authoritative measurements are recorded above. No training
array has been launched.

The preflight-only command is:

```bash
EXPECTED_COMMIT=<authoritative-commit> \
  bash bin/run_experiments/cluster_s5_three_arm_preflight.sh
```

It performs the three ordered exact-shape batch-16, sequence-16,000 stages per
arm, records `telemetry_compile_seconds`, `normal_compile_seconds`, and
`steady_step_seconds`, plus peak VRAM and finite-gradient/state status, and
never opens the test split. Full training remains a separate command and must
not be submitted until this preflight succeeds.
