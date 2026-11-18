# Proximal correction inside inertial NLA/history, then S5

Date: 2026-11-18. Research pilot; no accuracy improvement is established.
This changes S5 state dynamics. The outer training optimizer and the completed
N/A/C/D/history experiments are unchanged.

## The small optimization modification

Original inertial NLA is

\[
m\ddot x+c\dot x=-g(x)-\tau\frac{d}{dt}g(x).
\]

Rather than explicitly applying the force evaluated at the uncorrected point,
evaluate it at the point resulting from a proximal correction:

\[
y+\mu g(y)=x,\qquad g_\mu(x)=g(y)=\frac{x-y}{\mu}.
\]

For a convex objective this is the gradient of its Moreau envelope. More
generally it is the Yosida approximation of a maximally monotone operator.
Replacing the force in inertial dynamics by its Yosida approximation is an
established optimization construction, not a gate or a new arbitrary force.

Retain the same history modification:

\[
m\ddot x+c\dot x=-g_\mu(x)-\tau\dot q,
\qquad \ell\dot q=g_\mu(x)-q.
\]

For a fixed affine gradient g(x)=Hx-d,

\[
g_\mu(x)=(I+\mu H)^{-1}(Hx-d).
\]

For a fixed-input squared-loss neuron problem H=AA^T (with the appropriate
row/column convention), this is the same ridge-inverse family as FOOF:
(I+mu H)^(-1)=mu^(-1)(H+mu^(-1)I)^(-1). This identifies the exact restricted
FOOF correspondence; it does not identify generic loss Hessians with activation
Gram matrices. This proximal-force family is also distinct from projecting the
whole NLA equation into a fixed FOOF metric: that other family instead has
(m Wddot+c Wdot)R=-G_W-tau Gdot_W. The chosen experiment follows the
regularized-force branch because it supplies a direct cheap S5 state map.

Sources:

- Benzing (ICML 2022), *Gradient Descent on Neurons and its Link to Approximate
  Second-Order Optimization*, Eqs. (5)-(6):
  https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf
- Attouch and Laszlo (2020), *Newton-like Inertial Dynamics and Proximal
  Algorithms Governed by Maximally Monotone Operators*, SIAM J. Optim.
  30(4), 3252-3283, DOI 10.1137/20M1333316. The accessible author abstract
  explicitly describes Yosida regularization of the monotone operator in
  Hessian-damped inertial dynamics:
  https://www.cs.ubbcluj.ro/~grupanopt/22-10-20-Laszlo.pdf
  Only this author abstract and the publisher abstract were available here;
  the HAL full-text request timed out. Their convergence theorem is not used
  as a theorem for our fixed-coefficient filtered-history extension.
- Adly, Attouch and Vo (2021), *Asymptotic behavior of Newton-like inertial
  dynamics involving the sum of potential and nonpotential terms*, full text,
  especially sections 1.2-1.4:
  https://link.springer.com/article/10.1186/s13663-021-00702-7

## Why the S5 mapping is legitimate without calling rotation a Hessian

In token units, native S5 has ds/dt=-kappa*s+b, kappa=-Lambda*Delta,
b=Delta*Btilde*u. Its real two-dimensional residual has

\[
\langle r(s)-r(t),s-t\rangle
=\operatorname{Re}(\kappa)|s-t|^2>0.
\]

Thus it is a strongly monotone real operator for Re(kappa)>0, although it is
not generally a convex gradient. Its resolvent and Yosida approximation are
well defined. For the affine residual, the proximal equation can be solved
exactly and cheaply:

\[
\boxed{r_\mu(s,u)=\frac{\kappa s-b(u)}{1+\mu\kappa}.}
\]

Use the existing history ties m=tau, c=1, ell=tau:

\[
\boxed{\tau\ddot s+\dot s=-r_\mu-\tau\dot q,
\qquad \tau\dot q=r_\mu-q.}
\]

Both the restoring coefficient AND the input are divided by 1+mu*kappa.
Dividing only the state coefficient would change the target incorrectly.
No rotating frame, full covariance, extra learned parameters, or outer
optimizer modification is introduced. There are still three complex states
per original complex mode, exactly as in the current history arm.

The time-varying input makes this a driven system. Static optimization
convergence theorems do not prove classification accuracy or tracking quality.
For each frozen set of trained weights the homogeneous poles are analyzed below.

## A coefficient obtained from the stability calculation

Set mu=nu*tau, z=tau*p, and q=tau*kappa (q here is a scalar coefficient,
not the history state). The homogeneous characteristic polynomial is

\[
z(z+1)^2+q\{\nu z(z+1)^2+2z+1\}=0.
\]

For nonzero z this can be written 1+q F_nu(z)=0, where

\[
F_\nu(z)=\nu+\frac{2z+1}{z(z+1)^2},\qquad
\operatorname{Re}F_\nu(i\omega)
=\nu-\frac{2\omega^2}{(1+\omega^2)^2}.
\]

The negative term has maximum magnitude 1/2. Consequently nu>=1/2
makes F positive real: its only boundary pole is at zero with positive
residue, its other poles are at -1, and its real part is nonnegative on the
imaginary boundary and positive in the open right half-plane. If Re(q)>0,
-1/q has negative real part and cannot equal F there. At z=0 the
characteristic polynomial is q, not zero. The leading coefficient 1+nu*q
cannot vanish. Therefore every homogeneous pole has negative real part.

The bound is sharp for a uniform statement over Re(q)>0: for nu<1/2 choose
q=-1/[(nu-1/2)-i]. Then Re(q)>0 and z=i is a root.

The pilot uses the smallest sufficient universal value, **mu=tau/2**.
This is our calculation for this linear filtered family, not a quoted
constant from the papers. Exact ZOH preserves these poles inside the unit
circle mathematically. Floating-point construction is separately monitored;
there is no claim of a uniform numerical margin as Re(kappa) approaches zero.

## What changes, and what is preserved

- Setting proximal_ratio=0 recovers the existing history equation at the same
  alpha, including its input coefficient.
- Setting alpha=0 invokes the exact native constructor explicitly.
- The equilibrium/DC gain is native, b/kappa, since the residual has the same
  zeros. The continuous DC group delay adds mu; finite-history trajectories
  and sampled group delay are not claimed equal to native.
- The force is now implicit/proximal but computed by an explicit division.
  History remains a dynamic trace, rather than being replaced by a static filter.
- With positive finite tau and mu, the native model is NOT reproduced exactly.
  Alpha controls a substantive representation change, and it must earn its run.

## Fixed overnight protocol

Use tau=ell=alpha/|kappa| as in the current history implementation. The
one-GPU job runs sequentially:

1. Main candidate: alpha=1, proximal_ratio=1/2, seed 301.
2. Same candidate, seed 302.
3. Matched-horizon control: alpha=0.1, proximal_ratio=1/2, seed 301.

Alpha=1 gives the history horizon the native inverse-generator timescale;
this is a declared modeling choice, not a uniquely optimal theoretical value.
The matched control changes only the proximal correction relative to the old
history arm. The main candidate tests whether the newly stable longer history
is useful. A control at just one seed is diagnostic, not a replicated claim.

Full training uses the existing architecture, batch 16, 15 epochs, seeds,
data split, outer optimizer, and selected-validation rule. No test split is
read. Each task must first pass its own full-shape GPU smoke with matching
sources, seed, alpha, proximal ratio, and data manifest. The job uses a new
checkout and output directory. Existing results are never overwritten.

This is an exploratory state-recurrence experiment. Neither a two-to-three
percentage-point gain nor any positive gain is established before training.

## Local evidence before publication

`experiments/s5_proximal_history/diagnostic.py` independently constructs the
physical matrices with SciPy and reconstructs the actual float32 native
initialization in isolated subprocesses. Evidence is saved in
`docs/evidence/s5_proximal_history/initial_diagnostic.json`.

Across all 768 initialized modes from seeds 301 and 302, the old history,
matched proximal history, and main proximal history have no unstable
continuous or float64 exact-ZOH poles. Median slow-decay-time/native ratios
are respectively 0.9911, 0.7413, and 1.5459. Median peak gain measured on an
adaptive band around the native resonance, relative to native peak gain,
is 1.0144, 0.7553, and 1.1146. This last quantity is a band measurement, not
a global frequency-domain norm or a learned-layer output comparison.

An independent 12,000-point right-half-plane coefficient grid has no
nonnegative-real-part pole at proximal ratio 1/2. The analytic bound, rather
than the sampled grid, is the basis of the universal frozen-mode claim.

Fourteen constructor/recurrence tests pass separately in true float32 and
float64. They check the native and old-history controls, parameters, physical
ZOH reference, state/parameter gradients, bidirectional resets, continuous DC
gain and delay, the sharp stability bound, and a 16,000-token scan. Production
GPU execution is not verified locally; the submitted job must pass that gate.

Twelve separate float32 integration tests also pass: scoped registration and
restoration, source/configuration-bound smoke identity, identical native
parameter and outer-optimizer initialization, an actual tiny CPU smoke through
the CLI (two updates, validation, checkpoint restoration), both update-profile
compilation paths, shell syntax, dry runs, and refusal of a two-GPU allocation.

Reproduce locally with separate processes:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest tests/test_proximal_history.py -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_proximal_history.py -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest tests/test_proximal_history_runner.py -q
```

From a clean checkout of the published pilot commit on the cluster head node,
set EXPECTED_COMMIT to its complete SHA and run
`bash bin/run_experiments/submit_s5_proximal_history.sh`. This submits one
RTX4090 on pgi14-gpu1, eight CPUs, 48 GB host RAM, and a 12-hour ceiling.
The job log is `/Users/durso/s5-runs/s5-proximal-history-<jobid>.log`; results
are `/Users/durso/s5-runs/s5-proximal-history/job-<jobid>/<alpha-tag>/<seed>/`.
