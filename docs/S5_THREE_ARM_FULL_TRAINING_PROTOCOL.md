# Three-arm full-training S5 recurrence experiment

Status: preregistered; no training run has been launched.

This protocol compares exactly three scientific arms on one full, from-scratch
Speech Commands experiment. It does not include prospective input coupling,
learned-mismatch as a separate arm, adaptive-current dynamics, doubled-mode
S5, Nesterov, QHM, or any historical checkpoint.

## Scientific arms

| Scientific name | Code identifier | Executed recurrence |
|---|---|---|
| Native matched S5 | `native_matched_s5` | Native S5 ZOH recurrence |
| Zucchet prospective S5 recurrence | `zucchet_prospective_s5` | The (M=0,\gamma=1) member below, with positive response time (T) |
| Generalized prospective S5 recurrence \((M,\gamma,T)\) | `generalized_prospective_s5` | The finite-inertia second-order member below, with \(M>0\), \(\gamma=1\), and positive \(T\) |

The code identifiers are metadata only. Results are reported with the
scientific names above.

## Exact equations

The S5 continuous-time pole and input row are clock-absorbed as
\(a=\Delta\lambda\) and \(b=\Delta\widetilde B\), with \(\operatorname{Re}(a)<0\).
Each token input is held over one unit interval and discretized with an exact
unit-interval zero-order hold.

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

The production transition is the exact augmented matrix exponential over one
unit interval:

\[
z_{k+1}=e^{A}z_k+A^{-1}(e^A-I)B x_k,
\qquad y_k=s_k.
\]

The declared mathematical boundary is \(M=0\), where elimination of the
second-order state gives the Zucchet prospective recurrence above. Production
uses \(M>0\) because the finite-dimensional second-order parameterization is
singular at exactly zero mass. The equation/identity tests verify the exact
zero-mass coefficient boundary and convergence of the production transition
to it as \(M\downarrow0\) within the declared numerical domain.

## Training contract

All arms use the same outer architecture: four causal unidirectional S5
layers, `d_model=32`, nominal S5 size 32, conjugate symmetry, ZOH
discretization, featurewise standardized 20-coefficient MFCC inputs, mean
pooling, GELU/half-GLU configuration fixed in the launcher, no dropout and
tokenwise LayerNorm. Only the recurrence and its necessary recurrence
parameters differ.

Speech Commands v0.02 uses the repository's 10-word subset, deterministic
split seed 0, 70/15/15 stratification, and the cached MFCC manifest. The
training and validation arrays are opened before training; the test arrays are
opened exactly once, after validation checkpoint selection.

Each seed trains every arm from scratch. The same seed initializes corresponding
shared parameters wherever shapes permit and produces the same epoch-wise
training permutation for all three arms. No checkpoint or historical metric is
reused.

The fixed schedule is 40 epochs, batch size 32, AdamW, global learning rate
`1e-3`, SSM learning rate `1e-3`, weight decay `1e-4`, global gradient clip 1.0,
one warm-up epoch, cosine decay to `1e-6`, and paired seeds `301, 302, 303`.
The checkpoint is the first epoch attaining the highest validation accuracy;
validation cross-entropy breaks ties. Test accuracy and test cross-entropy are
computed only from that selected checkpoint.

## Decision rule and outputs

The primary contrast is Generalized prospective S5 recurrence \((M,\gamma,T)\)
minus Zucchet prospective S5 recurrence. Native matched S5 is the baseline.
For each seed and the across-seed mean, the result schema records test
accuracy, selected epoch, test cross-entropy, parameter count, recurrent-state
size, training seconds, examples/second, paired differences, paired standard
errors, and 95% paired confidence intervals.

The report must not call the generalized method better unless it beats Zucchet
prospective S5 recurrence consistently across seeds and does not materially
worsen test accuracy or cross-entropy relative to Native matched S5.
