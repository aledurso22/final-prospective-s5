# Adaptive proximal-history S5

This experiment adds one learned positive horizon multiplier per complex mode
to the existing proximal-history arm P at alpha=1. Its purpose is to test whether
allocating history and proximal regularization separately across modes improves
the classifier. It does not assume that longer or stronger auxiliary responses
are always better, and it does not predict a particular accuracy gain.

## Evidence and hypothesis

The completed P audit found persistent auxiliary poles and appreciable individual
auxiliary component energies. Those component energies are not additive shares
of layer output energy: both cross-pole and cross-mode interference matter.
Thus P does not have the nearly invisible fast-pole problem seen in A/C, but the
audit has not established that its fixed horizon rule is optimal for the task.
The observed five-seed selected-validation means were 96.9862% for P alpha=1 and
96.8188% for native. This is preliminary motivation, not evidence of a large gain.

The restriction tested here is mathematical: P enforces tau_i |kappa_i| = 1
for every mode, even while Lambda and Delta train. This fixes the dimensionless
history horizon. The new scalar frees that constraint. A change to B/C alone
cannot absorb the resulting changes to recurrent poles and transfer zeros.

## One change to the equation of motion

For each diagonal mode, retain P's continuous equations:

    tau s'' + s' = -r_mu - tau q'
    tau q' = r_mu - q
    r_mu = (kappa s - b) / (1 + mu kappa).

Replace only the horizon rule:

    old: tau = 1 / |kappa|,       mu = tau / 2
    new: tau = exp(xi) / |kappa|, mu = tau / 2, xi_init = 0.

Here kappa=-Lambda*Delta and b=Delta*B_tilde*u retain their existing definitions.
The inertia m, memory horizon ell, and prospective coefficient remain tied to
tau. The new xi is one real parameter per mode, shared between the forward and
backward scans. It is fixed throughout a sequence; it changes only when the
outer training optimizer updates the model. This is not token-dependent gating.

There are 6*64=384 added real parameter values. The candidate trains them in the
same Adam group and learning-rate schedule as log_step, without weight decay.
The control stores identical zero-valued leaves but freezes them. All old
parameters, their initialization and optimizer groups retain their definitions.

The parameterization matters. Directly replacing tau by an independently stored
log-tau would remove the old dependence of tau on kappa and change shared-parameter
gradients at initialization. Multiplying the existing horizon by exp(xi) instead
gives a nested baseline: xi=0 reproduces old P at every shared-parameter value.

## What is preserved from proximal optimization and FOOF

The force still follows from the implicit proximal equation y+mu*r(y)=s.
For a scalar real convex quadratic r(x)=h*x-d it is

    r_mu(x) = (h*x-d)/(1+mu*h)
            = (1/mu)*(h+1/mu)^(-1)*(h*x-d).

Learning xi therefore also learns the regularization 1/mu, linked to the
inertial/history horizon. In commuting quadratic eigendirections this gives
mode-dependent regularized inverse factors.

The precise FOOF connection remains restricted. For the fixed-input quadratic
E(W)=0.5*||W*A-Y||_F^2 and a common scalar mu, the proximal gradient is
G(W)*(I+mu*A*A^T)^(-1), proportional to FOOF's direction with lambda=1/mu.
Benzing's original FOOF construction derives this activation-Gram inverse from
regularized matching of desired neuron changes (equations 5-6):
https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf

This S5 experiment does not estimate that activation Gram matrix or use FOOF as
the weight optimizer. Its complex residual is a monotone operator, generally
not the Hessian of a real scalar energy. The experiment retains the proximal
operator mechanism and extends its diagonal regularization; it is not a claim
of equivalence to general neural-network FOOF.

## Stability and implementation

For frozen coefficients and Re(kappa)>0, the existing P stability result applies
for every positive tau when m=ell=tau and mu>=tau/2. The new parameterization
keeps mu=tau/2 exactly. Hence learning xi does not leave that continuous-time
stability family. The exact zero-order-hold discrete poles are exp(p).

This statement is about homogeneous linear modes with fixed model parameters.
It does not prove training convergence, finite-precision stability, or task
accuracy. Very large/small learned scales can cause conditioning or nonfinite
arithmetic; the runner reports actual float32 and float64 transitions, and fails
on nonfinite values. No pole or horizon is silently clipped.

Reuse the three-state exact-ZOH construction, stationary scan, explicit adjoint,
and position-only readout. Scaling both kappa and b by the same proximal divisor
preserves the constant-input equilibrium s=b/kappa.

## Paired experiment

Run exactly these four fresh tasks:

- adaptive_proximal_history_s5, seeds 301 and 302;
- adaptive_proximal_history_control_s5, seeds 301 and 302.

Both arms start from alpha=1 P, with matching shared parameters and xi=0. Only
the candidate updates xi. Keep batch 16, 15 epochs, data split, dropout, learning
rate schedule and selected-validation rule unchanged. The test split is not read.
Do not replace older P scores with these controls or select a favorable seed.

All four full-shape GPU smoke checks must pass before any training begins.
Compile memory accounting is labelled as an estimate; successful real GPU smoke
execution is not presented as a measured peak-memory benchmark.

Every epoch records learned multipliers, physical horizons and actual pole
diagnostics. The post-training audit uses the selected checkpoints and saved xi,
checks both auxiliary pole decays, and computes the full B/C-coupled kernel
difference when both auxiliary responses are removed together. That squared
kernel difference includes interference; it is not a sum of component shares or
a validation-accuracy ablation. Source checkpoints remain unchanged.

Interpret candidate versus frozen control as the effect of learning the linked
horizon/regularization parameter. It does not separately identify effects of
inertia, history, ridge regularization, or the additional trainable capacity.
The experiment does not promise a 2-3 percentage-point gain.

## Execution

The submission script is bin/run_experiments/submit_s5_adaptive_proximal_history.sh.
It requests four RTX 4090 GPUs on pgi14-gpu1. The cluster wrapper uses a new pinned
checkout and new output directory. Submission is performed by the user on the
cluster head node; preparing or pushing this branch does not launch a job.

Local tests and cluster GPU checks are separate. See the accompanying validation
record for the checks actually run before submission.
