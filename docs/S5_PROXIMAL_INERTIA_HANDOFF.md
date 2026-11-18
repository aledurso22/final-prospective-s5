# Proximal regularization can expose the existing inertial timescale

Date: 2026-11-18. Theory and coding handoff. No new training has been run.

## Decision

Investigate one **unfiltered proximal inertial** candidate, with an exact
first-order proximal control. This follows the user's request to expose the
additional timescale in ordinary inertial NLA. It does not add a history filter,
another memory mixture, a readout, or an outer optimizer.

The proposed fixed settings are

\[
\kappa=-\Lambda\Delta,\quad h=0.1/|\kappa|,\quad
\tau=h,\quad\mu=h/2,\quad m=3h/2,\quad c=1.
\]

The matched control changes only m to h. Its zero-prehistory input/output
response is exactly first-order proximal flow. Existing N and P are additional
comparators, but neither replaces this control for attributing an effect to the
released inertial mode.

The constants 0.1 and 1.5 are modeling choices, not accuracy-optimal constants
derived from FOOF. Their relation mu=m-tau is theoretically motivated below.
A read-only initialization screen supports 0.1 over reusing P's alpha=1:
the latter substantially shortens native-like dominant decay in some modes.
No validation scores were used to choose this screen's settings.

## 1. Keep ordinary inertial NLA; regularize its force

Normalize ordinary damping to c=1 and write

\[
m\ddot x+\dot x=-g(x)-\tau\frac{d}{dt}g(x).
\]

Define the proximal point and regularized force by

\[
y+\mu g(y)=x,\qquad g_\mu(x)=g(y).
\]

The proposed equation is simply

\[
\boxed{m\ddot x+\dot x=-g_\mu(x)-\tau\frac{d}{dt}g_\mu(x).}
\]

There is no filtered-gradient state q. The second state is the inertia already
present in this second-order equation. For a smooth convex gradient, g_mu is
the gradient of the Moreau envelope. For fixed coefficients, no external input,
and a twice continuously differentiable convex E_mu, its energy satisfies

\[
\frac{d}{dt}\left(E_\mu(x)+\frac m2\|\dot x\|^2\right)
=-\|\dot x\|^2-\tau\dot x^T\nabla^2E_\mu(x)\dot x\le0.
\]

This convex-gradient statement is separate from the complex linear stability
calculation below. It does not prove convergence of a general nonlinear S5
network or predict classification accuracy.

## 2. The restricted FOOF correspondence is retained

For a fixed convex quadratic g(x)=Hx-d with H=H^T>=0 and mu>0,

\[
g_\mu=(I+\mu H)^{-1}g.
\]

For E(W)=||WA-Y||_F^2/2 with fixed A and S=AA^T,

\[
G_\mu=G_W(I+\mu S)^{-1}
=\mu^{-1}G_W(S+\mu^{-1}I)^{-1}.
\]

This agrees with FOOF's direction at lambda=1/mu, up to a scalar time/step
factor. It is an exact restricted algebraic correspondence, not identification
of arbitrary Hessians with activation Gram matrices. Finite-inertia velocities
are not equal to FOOF's instantaneous updates in general.

Equivalently, for constant R=I+mu H,

\[
mR\ddot x+R\dot x=-g-\tau\dot g.
\]

Thus the same regularized geometry multiplies mass and ordinary damping. A
state-dependent or time-dependent R would require additional analysis.

The S5 implementation uses a separate complex monotone residual and per-mode
mu; it is an operator-resolvent extension, not the literal activation-covariance
FOOF weight optimizer with one global ridge.

## 3. Why inertia can disappear from the driven response

For a frozen affine residual r=kappa*x-b(t), Re(kappa)>0,

\[
r_\mu=\frac{\kappa x-b(t)}{1+\mu\kappa}.
\]

Using total time differentiation, including the changing input, the causal
zero-prehistory continuous transfer is

\[
H(p)=\frac{x(p)}{b(p)}=
\frac{1+\tau p}
{(1+\mu\kappa)(mp^2+p)+\kappa(1+\tau p)}.
\]

At m=tau the denominator factors exactly:

\[
D(p)=(1+\tau p)[(1+\mu\kappa)p+\kappa],\qquad
H(p)=\frac1{(1+\mu\kappa)p+\kappa}.
\]

The extra recurrent pole is canceled by the forcing zero, for every mu. Larger
proximal regularization alone cannot expose it. Arbitrary initial conditions
can still excite an unforced transient; cancellation concerns the driven
zero-prehistory response, not all possible trajectories.

For m!=tau and tau>0,

\[
D(-1/\tau)=(1+\mu\kappa)(m-\tau)/\tau^2\ne0.
\]

The pole-zero cancellation is removed. This is an exact property of the
unfiltered continuous family, not a retroactive explanation of every A/C
finite-difference channel or P's three-state filtered dynamics.

## 4. Proximal regularization opens a stable larger-inertia regime

Define

\[
F(p)=\mu+\frac{1+\tau p}{p(1+mp)}
=\mu+\frac1p+\frac{\tau-m}{1+mp}.
\]

Away from its poles, a characteristic root requires F(p)=-1/kappa. On the
imaginary axis,

\[
\operatorname{Re}F(i\omega)=
\mu+\frac{\tau-m}{1+m^2\omega^2}.
\]

For m>0, tau>=0 and mu>=0, the sharp uniform condition is

\[
\boxed{\mu\ge\max(0,m-\tau).}
\]

It gives strictly left-half-plane homogeneous poles for every Re(kappa)>0.
To see sufficiency directly, if m<=tau each term of F has nonnegative real
part, with Re(1/p)>0 in the open right half-plane. If delta=m-tau>0, rewrite

\[
F(p)=(\mu-\delta)+\frac1p+\frac{\delta mp}{1+mp}.
\]

Under mu>=delta its real part is again positive there, whereas Re(-1/kappa)<0.
Nonzero imaginary-axis roots are excluded by the boundary formula. p=0 is not
a root because D(0)=kappa; the leading coefficient m(1+mu*kappa) cannot vanish.

For necessity, if mu<m-tau, sufficiently small nonzero omega makes
Re F(i omega)<0. Choosing kappa=-1/F(i omega) gives Re(kappa)>0 and a marginal
root; moving p slightly into the open right half-plane gives an admissible
unstable example. One checked example is m=1.5,tau=1,mu=0.4,
kappa=0.00023385579949444118-0.05009230721178928i, with root
p=0.00001+0.05i (characteristic residual 4.34e-19).

Without regularization, uniform stability requires m<=tau. With mu>0, the
new interval tau<m<=tau+mu becomes available. This is the concrete role of the
proximal force: it permits finite inertia beyond the cancellation boundary.
Equality gives no uniform numerical margin as Re(kappa) approaches zero.

## 5. Choose the boundary to retain the dominant pole to low order

Take m=tau+mu. Write a=m/tau>1, nu=mu/tau=a-1, chi=tau*kappa, z=tau*p.
The characteristic equation is

\[
(1+\nu\chi)(az^2+z)+\chi(1+z)=0.
\]

For small |chi|, the branch near the native pole is

\[
z_{\rm slow}=-\chi-a(a-1)\chi^3+O(\chi^4).
\]

The O(chi^2) error cancels. The other pole tends to -1/a. Define the scaled
dimensionless transfer G(z)=(1+z)/[az^2+(1+k)z+k], k=chi/(1+nu chi).
The physical transfer satisfies H(z/tau)/tau=G(z)/(1+nu chi). The residues
of G tend to 1 and -(a-1)/a; the common proximal factor also tends to 1.
Thus the auxiliary contribution need not vanish algebraically.
This is a local expansion, not a uniform relative-decay guarantee near purely
imaginary kappa, nor a promise of a large full-layer energy contribution.

All masses in this family retain

\[
H(0)=1/\kappa,\qquad
H(p)=\kappa^{-1}[1-(\kappa^{-1}+\mu)p+O(p^2)].
\]

Continuous DC delay is Re(1/kappa)+mu. Exact ZOH sampled delay must be measured
separately. With different mu, DC delay is not matched to the existing P alpha=1.

## 6. Preliminary numerical screen, not a trained result

Independent float64 equations and exact physical-input ZOH were evaluated on
the six layers of reconstructed production float32 initializations, seeds 301
and 302, 384 complex modes per seed. These are today's CPU reconstructions,
not saved historical epoch-zero checkpoints. Learned checkpoint P auditing
and true float32 GPU construction remain outstanding.

For alpha=0.1, a=1.5, nu=0.5:

| Quantity | Seed 301 | Seed 302 |
|---|---:|---:|
| Median dominant decay/native ratio | 1.0087 | 1.0087 |
| Range of dominant decay/native ratio | 0.9796–1.0175 | 0.9796–1.0175 |
| Median auxiliary decay, samples | 4.47 | 4.04 |
| p95 auxiliary decay, samples | 103.9 | 92.0 |
| Maximum auxiliary decay, samples | 214.5 | 226.5 |
| Median auxiliary component/mode energy | 0.284% | 0.289% |
| p95 auxiliary component/mode energy | 2.073% | 2.072% |

The energy statistic includes real B/C coupling within each mode and pools the
two readout directions. It is a finite 16,000-lag component ratio, not an additive
share of layer energy and not a measure of classification usefulness. It must
not be treated as an apples-to-apples trained comparison with the A/C audit.

At alpha=1, the initial dominant decay/native ratio instead ranges down to
about 0.148, with median about 0.726. A separate near-imaginary example,
chi=exp(1.56i), gives ratio 0.06582; alpha=0.1 at the same angle gives 0.92846.
The proposed scale is therefore 0.1, not 1. Simply maximizing auxiliary energy
can substantially distort the primary dynamics.

The reproducible screen is experiments/s5_predictor_corrector/proximal_inertia_screen.py;
its saved evidence is docs/evidence/s5_proximal_inertia/screen.json. It also
checks the derivative-free transfer against an independent augmented exponential.
Its maximum impulse discrepancy is 1.55e-14, explicit real-kernel energy
relative discrepancy is 4.01e-16, and cancellation discrepancy is 2.01e-16.
The identical decay-ratio summaries across seeds reflect their shared
deterministic initial Lambda spectrum; they are not independent confirmations.

Reproduce from the repository root, using the production Python environment:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m experiments.s5_predictor_corrector.proximal_inertia_screen \
  --out /path/to/new-screen.json
```

The screen refuses to overwrite an existing output.

## 7. Exact S5 realization for the coding agent

Use kappa=-Lambda*Delta and b=Delta*Btilde*u, as in P. Let
d=1+mu*kappa, k=kappa/d and define w=m*s_dot+tau*r_mu. Then

\[
\frac d{dt}\begin{pmatrix}s\\w\end{pmatrix}=
\begin{pmatrix}-\tau k/m&1/m\\(\tau/m-1)k&-1/m\end{pmatrix}
\begin{pmatrix}s\\w\end{pmatrix}
+\frac1d\begin{pmatrix}\tau/m\\1-\tau/m\end{pmatrix}b.
\]

Use an augmented 3x3 exponential per mode for the transition and scalar-input
column, then apply Delta*Btilde. Keep original kappa in the horizon definition.
Both the state force and physical input are regularized. Do not drop the total
input derivative in an (s,velocity) implementation: w is continuous across
token jumps, while the corresponding physical velocity may jump.

Keep the native position readout, conjugate factor, bidirectional resets and
D skip. Do not add a second direct feedthrough. At m=tau, zero-prehistory w
stays zero and the control is exactly first-order proximal flow. At additionally
mu=0 it is native continuous S5 and should match its discretized response under
the same clock and timing, within declared arithmetic tolerances.

Existing s5/gp_second_order.py supplies the same coordinate algebra, but its
constructor constrains mass/tau below1 and uses a different parameter mapping.
Do not reuse that constructor. Its legacy ZOH path can exponentiate
(2+feature_width)-square matrices and its scan broadcasts a transition per token.
Use a width-2 stationary scan with a validated adjoint or another memory-tested
implementation; no eigendecomposition in the differentiable forward path.

For old C, the continuous target mapping gives m=M/gamma=rho*T, tau=T,
rho<1, and its implemented residual uses 1-exp(Lambda*Delta) and Bbar. Therefore
this is a new proximal optimization extension, not a one-constant rewrite of
the old positive two-compartment C constructor.

## 8. Execution brief

1. Preserve all existing runs. Work in an isolated branch/checkout and write
   outputs to fresh directories. Complete a trained-P pole/impulse audit from
   existing checkpoints in parallel with implementation. P requires 3x3 block
   poles and Delta*Btilde input normalization, not A/C's quadratic-root path.
2. Implement exactly one candidate: alpha=0.1, mass_ratio=1.5,
   proximal_ratio=0.5, no filtered history. Implement the exact first-order
   proximal control at the same alpha/proximal_ratio. Add no trainable dynamics
   parameters, readouts, gates, optimizer changes or hyperparameter sweep.
3. Verify the cancellation control, transfer identity, native limit, physical
   input timing, independent SciPy augmented-exponential reference, finite
   parameter gradients, sequential/parallel agreement, and the sharp stability
   bound including a below-bound counterexample. Verify native parameter draws,
   names and optimizer groups remain unchanged.
4. Reproduce the initial screen in true float32 and float64, and produce
   complete B/C-coupled layer impulse responses with interference. Report all
   growing/boundary poles, response norms and new auxiliary residues; do not
   clip poles or equate raw hidden-state magnitude with observable memory.
5. Profile and smoke both arms at full sequence/batch/model shape on assigned
   RTX 4090s. Record GPU memory and steady-state step timing. Reuse validated
   caches; do not reduce batch 16 or the 15-epoch protocol to make it fit.
6. If those checks succeed, provide one pinned launch command on pgi14-gpu1
   for candidate/control x seeds 301/302 (four independent runs). Use the
   existing official train/validation split and selection rule; do not read
   the test split. Publish implementation and checks before providing the
   cluster command. No training was launched by this theory handoff.
7. Compare candidate vs its control to test whether the extra inertial state
   helps, and vs existing same-seed N and P for practical value. Report paired
   accuracy/CE, cost and trained impulse changes. A gain over P would not alone
   prove a FOOF mechanism: the history construction differs as well.

P checkpoint locations (under /Users/durso/s5-runs):

- s5-proximal-history/job-76878/alpha1/{301,302}
- s5-morning-followup/job-76913/fresh/proximal_history_s5/alpha1/{303,304,305}

The audit's task is to explain existing trained dynamics. Counterfactual
coefficients at those weights are untrained responses and must be labeled so.

## Sources and attribution

- Benzing, ICML2022, *Gradient Descent on Neurons and its Link to Approximate
  Second-Order Optimization*, Eqs5–6: https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf
- Attouch–Laszlo, *Newton-like Inertial Dynamics and Proximal Algorithms Governed
  by Maximally Monotone Operators*, DOI10.1137/20M1333316. The author abstract
  explicitly describes Yosida regularization in inertial Hessian-damped methods:
  https://www.cs.ubbcluj.ro/~grupanopt/22-10-20-Laszlo.pdf
- Adly–Attouch–Vo2021, sections1.2–1.4, for gradient/nonpotential distinctions
  and the Yosida-inertial precedent: https://link.springer.com/article/10.1186/s13663-021-00702-7

The cancellation analysis, sharp frozen-linear bound, small-chi expansion and
specific S5 experiment here are derived for this handoff. These sources do not
prove a gain for this architecture or establish this particular implementation
as a novel optimizer. The chosen family needs empirical validation.
