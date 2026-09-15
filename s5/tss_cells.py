"""Temporal cells for the TSS-based pilot: four cores, one clock, one integrator.

All timescales are in SAMPLE STEPS with dt = 1. Nothing here imports the S5
substrate: the pilot is a separate small-scale probe and must not perturb the
running SSM study.

THE COEFFICIENT SECTORS, AND WHY ARMS 2 AND 4 ARE NOT THE SAME POINT
--------------------------------------------------------------------
Our unnormalized law, for constant scalar coefficients,

    M s'' + gamma s' + (1 + T D)(s - f) = 0      i.e.   R + T R' terms,
    equivalently   M s'' + (gamma + T) s' + s = f + T f'.                 (L)

The finite-adaptation realization (TSS Eqs. 6-7 shape) is

    tau_m s' = -s + (1 + tau_p/eps) f - (tau_p/eps) a,
    eps a'   = -a + f,

which eliminates EXACTLY (no Taylor expansion) to

    tau_m eps s'' + (tau_m + eps) s' + s = f + (eps + tau_p) f'.          (A)

Matching (L) and (A) term by term gives the mapping

    M = tau_m eps,   gamma + T = tau_m + eps,   T = eps + tau_p.          (MAP)

Two consequences are used throughout this pilot.

1. Our circuit point has an EXACT adaptation twin. With the declared symmetric
   reference gamma = T, M = 3T^2/4, the roots of Q are t_pm = (gamma+T +/-
   sqrt((gamma+T)^2 - 4M))/2 = {3T/2, T/2}, so tau_m = 3T/2, eps = T/2 and
   tau_p = T - eps = T/2 reproduce arm 4 exactly. That is an implementation
   CORRECTNESS CHECK, not a fifth competitor, and `equivalent_adaptation`
   below constructs it.

2. TSS's own matching prescription tau_p = tau_m is a DIFFERENT point, and it
   lies OUTSIDE our admissible set. Substituting tau_p = tau_m into (MAP),

       T = eps + tau_m  and  gamma + T = tau_m + eps   =>   gamma = 0,

   so M <= gamma T fails for any M > 0. Arms 2 and 4 are therefore neither
   equivalent realizations nor different model classes: they are two declared
   coefficient sectors of ONE family, and the circuit inequality excludes the
   TSS-matched sector. `tss_gamma_equivalent` returns that gamma so the claim
   is measured rather than asserted.

To keep the comparison about response SHAPE rather than about who was handed
the longer memory, arm 2 is given the same two poles as arm 4 (tau_m = 3T/2,
eps = T/2). The arms then differ only in the prospective zero: arm 4's drive is
f + T f', arm 2's is f + 2T f'.

INTEGRATION
-----------
Every driven core is nonlinear in s (f depends on s), so no exact
discretization applies and a method must be declared: classical RK4 with
`n_sub` equal substeps per input step, input held constant across the step
(zero-order hold). `refinement_error` performs the declared step-refinement
check. The ideal cancellation arm has no ODE at all; it solves its fixed point.
"""

import jax
import jax.numpy as jnp

#: the pilot's absolute clock: one input step is one unit of model time
DT = 1.0

#: Declared physical horizon, in steps. NOT imported from the S5 studies: those
#: used 5 sample intervals for a 161-frame speech task, which is not a
#: biological constant and carries no authority here. T = 8 is chosen so the
#: slow pole 3T/2 = 12 steps is of the order of the SHORT recall delay (8) and
#: a fraction of the long one (32), i.e. the temporal layer must actually use
#: its recurrence to span the delays rather than having them handed to it by a
#: single sufficiently slow leak. It is declared before execution and is not
#: tuned on any score.
T_HORIZON = 8.0


def symmetric_reference(T=T_HORIZON):
    """The declared positive circuit point: gamma = T, M = 3T^2/4.

    This is the same symmetric reference used by the SSM studies, restated in
    UNNORMALIZED form: rho = M/(gamma T) = 3/4, so M <= gamma T holds strictly.
    No independent fit of M, gamma and T is performed anywhere in this pilot.
    """
    return dict(T=float(T), gamma=float(T), M=0.75 * float(T) ** 2,
                rho=0.75, label="symmetric-reference")


def q_roots(gamma, T, M):
    """t_-, t_+ : the roots of Q(p) = 1 + (gamma+T) p + M p^2, as timescales.

    Returned smallest first. Real and positive whenever (gamma+T)^2 >= 4M,
    which the circuit inequality M <= gamma T implies.
    """
    a0 = gamma + T
    disc = a0 * a0 - 4.0 * M
    if disc < 0:
        raise ValueError(f"complex timescales: (gamma+T)^2 = {a0 * a0} < 4M = "
                         f"{4 * M}; the circuit inequality M <= gamma*T is "
                         f"violated or the coefficients are inadmissible")
    r = disc ** 0.5
    return (a0 - r) / 2.0, (a0 + r) / 2.0


def equivalent_adaptation(gamma, T, M):
    """The finite-adaptation twin of OUR law, via (MAP). A correctness check.

    Returns (tau_m, eps, tau_p) with tau_m*eps = M, tau_m+eps = gamma+T and
    eps+tau_p = T. `tests/test_tss_pilot.py` checks that the twin reproduces
    arm 4's trajectory AND its gradients, not merely its transfer function.
    """
    t_minus, t_plus = q_roots(gamma, T, M)
    tau_m, eps = t_plus, t_minus
    tau_p = T - eps
    if tau_p < 0:
        raise ValueError(f"tau_p = {tau_p} < 0; outside the admissible sector")
    return tau_m, eps, tau_p


def tss_gamma_equivalent(tau_m, eps, tau_p):
    """The gamma our law would need to reproduce a given adaptation cell.

    From (MAP): gamma = (tau_m + eps) - (eps + tau_p) = tau_m - tau_p. Under
    TSS's matching tau_p = tau_m this is exactly 0, and M = tau_m*eps > 0 then
    violates M <= gamma*T. Reported, not assumed.
    """
    T_eq = eps + tau_p
    gamma_eq = (tau_m + eps) - T_eq
    M_eq = tau_m * eps
    return dict(gamma=gamma_eq, T=T_eq, M=M_eq,
                admissible=bool(gamma_eq > 0.0 and 0.0 < M_eq <= gamma_eq * T_eq))


# --------------------------------------------------------------- drives ----
def drive(params, s, x):
    """f_theta(s, x) = W phi(s) + U x + b, with phi = tanh. Shared by all arms.

    The arms differ ONLY in the temporal operator applied to this same drive,
    which is what makes the comparison about the temporal law.
    """
    return params["W"] @ jnp.tanh(s) + params["U"] @ x + params["b"]


def _rk4(vf, z, x, n_sub, dt=DT):
    """Classical RK4 over one input step, input held (ZOH), `n_sub` substeps."""
    h = dt / n_sub

    def one(z, _):
        k1 = vf(z, x)
        k2 = vf(z + 0.5 * h * k1, x)
        k3 = vf(z + 0.5 * h * k2, x)
        k4 = vf(z + h * k3, x)
        return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

    z, _ = jax.lax.scan(one, z, None, length=n_sub)
    return z


# ------------------------------------------- arm 1: ideal cancellation -----
#: iterations of the fixed-point solve, and the contraction cap that makes the
#: solution unique. The cap is INTRINSIC to this arm's definition ("a unique
#: fixed point"), not a handicap imposed on it, and it is applied only where a
#: fixed point is solved. 0.8^40 = 1.3e-4, and the executed residual is
#: measured and reported rather than assumed.
IDEAL_ITERS = 40
IDEAL_SPECTRAL_CAP = 0.8


def ideal_fixed_point(params, x, iters=IDEAL_ITERS, s0=None):
    """s = f_theta(s, x), solved by unrolled iteration. NO temporal state.

    Differentiating the unrolled iteration is exact for the computed solution
    to the same accuracy as the solve itself; the residual is returned so the
    approximation is visible. There is deliberately no carry between steps: the
    ideal prospective limit cancels the memory, which is the property this
    control exists to exhibit.
    """
    s = jnp.zeros_like(params["b"]) if s0 is None else s0

    def it(s, _):
        return drive(params, s, x), None

    s, _ = jax.lax.scan(it, s, None, length=iters)
    return s, jnp.max(jnp.abs(drive(params, s, x) - s))


def project_contraction(W, cap=IDEAL_SPECTRAL_CAP):
    """Scale W so ||W||_2 <= cap. Applied after each update, for fixed-point
    arms only, so the solved equation keeps a unique solution throughout."""
    sv = jnp.linalg.norm(W, ord=2)
    return jnp.where(sv > cap, W * (cap / sv), W)


# ------------------------------------ arm 2: TSS finite adaptation ---------
def tss_vector_field(params, tau_m, eps, tau_p):
    """z = (s, a) with tau_m s' = -s + (1+tau_p/eps) f - (tau_p/eps) a,
    eps a' = -a + f."""
    g = tau_p / eps

    def vf(z, x):
        s, a = z[0], z[1]
        f = drive(params, s, x)
        return jnp.stack([(-s + (1.0 + g) * f - g * a) / tau_m,
                          (-a + f) / eps])

    return vf


# --------------------------- arm 4: retained-compartment (our law) ---------
def retained_vector_field(params, gamma, T, M):
    """z = (s, p) with p = M s' + T R, R = s - f. Derivative-free realization:

        s' = (p - T R)/M,     p' = -(gamma/M) p + (gamma T/M - 1) R.

    This is the lift of the SAME equation, not a different neuron; the
    equivalence to the (s, v) and adaptation realizations is checked.
    """
    def vf(z, x):
        s, p = z[0], z[1]
        R = s - drive(params, s, x)
        return jnp.stack([(p - T * R) / M,
                          -(gamma / M) * p + (gamma * T / M - 1.0) * R])

    return vf


# ---------------------------------------- arm 3: leaky memory layer --------
def leaky_vector_field(params, tau_mem):
    """tau_mem s' = -s + f_theta(s, x). One state per unit, memory-bearing."""
    def vf(s, x):
        return (-s + drive(params, s, x)) / tau_mem

    return vf


# ------------------------------------------------------------- rollouts ----
def rollout_ode(vf, z0, xs, n_sub):
    """Integrate a driven core over a sequence. Returns the state at each step.

    The state is recorded AFTER consuming each input step, so step k's output
    depends on inputs 0..k and never on the future.
    """
    def step(z, x):
        z = _rk4(vf, z, x, n_sub)
        return z, z

    zT, zs = jax.lax.scan(step, z0, xs)
    return zs, zT


def rollout_ideal(params, xs, iters=IDEAL_ITERS):
    """Arm 1: solve the fixed point independently at every step."""
    def step(carry, x):
        s, r = ideal_fixed_point(params, x, iters)
        return carry, (s, r)

    _, (ss, res) = jax.lax.scan(step, 0.0, xs)
    return ss, jnp.max(res)


def refinement_error(vf, z0, xs, n_sub, factor=4):
    """Declared step-refinement check: relative change from refining the step.

    Returns ||z(n_sub) - z(factor*n_sub)||_inf / (||z(fine)||_inf + 1), the
    figure the protocol's tolerance is stated against. RK4 is fourth order, so
    a factor-4 refinement should reduce the local error by ~256; a coarse step
    that has not converged shows up here rather than silently biasing an arm.
    """
    coarse, _ = rollout_ode(vf, z0, xs, n_sub)
    fine, _ = rollout_ode(vf, z0, xs, n_sub * factor)
    num = jnp.max(jnp.abs(coarse - fine))
    return num / (jnp.max(jnp.abs(fine)) + 1.0)
