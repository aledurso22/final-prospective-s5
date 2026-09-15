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

R7. To keep the comparison about response SHAPE rather than about who was
handed the longer memory, arm 2 is given the same MATCHED OPEN-LOOP DENOMINATOR
as arm 4 (tau_m = 3T/2, eps = T/2), and the arms then differ only in the
prospective zero: arm 4's drive is f + T f', arm 2's is f + 2T f'.

That matching is open-loop, i.e. it holds when f is an EXTERNAL drive. Part A
closes the loop with f = W tanh(s) + U x + b, and for a scalar eigenvalue `a`
of the frozen local recurrent Jacobian the characteristic polynomials are

    retained:        M p^2 + (gamma + T - a T) p + (1 - a)
                     = 48 p^2 + (16 - 8 a) p + (1 - a)
    TSS adaptation:  48 p^2 + (16 - 16 a) p + (1 - a)

so EQUAL OPEN-LOOP DENOMINATORS DO NOT GIVE EQUAL CLOSED-LOOP POLES OR EQUAL
MEMORY, even before training. `closed_loop_polynomial` computes both, and the
runner reports the local Jacobian spectra descriptively. The open-loop matching
remains useful - it removes one gross asymmetry - but it is not a claim that
the arms have the same memory.

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


#: reporting metadata, kept OUT of anything that reaches a jitted call
REFERENCE_LABEL = "symmetric-reference"


def symmetric_reference(T=T_HORIZON):
    """The declared positive circuit point: gamma = T, M = 3T^2/4.

    NUMERIC ONLY. R1: an earlier revision carried `label="symmetric-reference"`
    in this dict, and the dict is passed as a dynamic pytree into the jitted
    training and evaluation steps. A string is not a valid dynamic JAX leaf, so
    the first production call would have been blocked by the configuration's
    structure even though no mathematics reads the string. Metadata now lives
    in `reference_metadata()` and never enters a traced call.

    Restated in UNNORMALIZED form: rho = M/(gamma T) = 3/4, so M <= gamma T
    holds strictly. No independent fit of M, gamma and T anywhere in this pilot.
    """
    return dict(T=float(T), gamma=float(T), M=0.75 * float(T) ** 2,
                rho=0.75)


def reference_metadata(T=T_HORIZON):
    """Reporting-only description of the coefficient point."""
    r = symmetric_reference(T)
    return dict(label=REFERENCE_LABEL, dt=DT,
                clock="one input step is one unit of model time", **r)


def assert_numeric_pytree(tree, where=""):
    """Refuse a configuration that cannot be a dynamic JIT argument.

    R1 again, as an executable guard rather than a convention: any non-numeric
    leaf reaching a traced call is an error at construction time, not a
    surprise at the first production step.
    """
    import jax as _jax
    bad = []
    for path, leaf in _jax.tree_util.tree_flatten_with_path(tree)[0]:
        if isinstance(leaf, (str, bytes)) or leaf is None:
            bad.append((_jax.tree_util.keystr(path), type(leaf).__name__))
    if bad:
        raise TypeError(f"non-numeric leaves in {where or 'config'}: {bad}. "
                        f"Metadata must not travel into a jitted call.")
    return tree


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


def closed_loop_polynomial(a, gamma=None, T=None, M=None, tau_m=None,
                           eps=None, tau_p=None):
    """Closed-loop characteristic coefficients [p^2, p^1, p^0] at Jacobian `a`.

    R7: the matched quantity is the OPEN-LOOP denominator Q. Closing the loop
    with f = a s moves the linear coefficient differently in the two sectors,
    because the prospective zero multiplies the drive.
    """
    if gamma is not None:
        return [M, (gamma + T) - a * T, 1.0 - a]
    return [tau_m * eps, (tau_m + eps) - a * (eps + tau_p), 1.0 - a]


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

    Written with trailing-axis contractions (`@ W.T`) so the SAME code runs on
    one sequence, shape (n,), and on a batch, shape (B, n). That is what lets
    the model be batch-native instead of wrapped in `vmap`; see
    `tss_models.batched_forward` for why that matters.
    """
    return (jnp.tanh(s) @ params["W"].T + x @ params["U"].T + params["b"])


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


# ------------------------------- TSS-style linear complex leaky memory -----
#: declared initial memory timescale, in steps: the same slow open-loop
#: timescale as the other arms' t_+ = 3T/2 = 12
MEMORY_TAU_INIT = 12.0


def _phi1(a):
    """(exp(a) - 1)/a, with the removable singularity handled at small |a|."""
    small = jnp.abs(a) < 1e-6
    safe = jnp.where(small, jnp.ones_like(a), a)
    return jnp.where(small, 1.0 + a / 2.0, (jnp.exp(safe) - 1.0) / safe)


def complex_memory_lambda(params):
    """lambda = -exp(log_decay) + i omega, stable by construction."""
    return -jnp.exp(params["log_decay"]) + 1j * params["omega"]


def complex_memory_step(params, z, u, dt=DT):
    """EXACT zero-order-hold update of an independent complex leaky unit.

    R2. This is the small version of TSS Section 3.3's memory structure:
    INDEPENDENT COMPLEX LINEAR leaky units, not a dense nonlinear tanh
    recurrence. Being linear with constant coefficients over the step, it gets
    the exact discretization the design asks for where one is available, so no
    integration error is introduced on this arm at all.
    """
    lam = complex_memory_lambda(params) * dt
    return jnp.exp(lam) * z + _phi1(lam) * u * dt


def complex_memory_coefficients(params, dt=DT):
    """Per-unit discrete coefficients, computed ONCE outside any recurrence.

    Returns (A_block, Phi) with A_block the REAL 2x2 rotation-scaling that
    exp(lambda*dt) performs on (Re z, Im z), and Phi = phi1(lambda*dt) the
    complex input weight. Every complex operation lives here, elementwise and
    outside the scan.
    """
    lam = complex_memory_lambda(params) * dt
    A = jnp.exp(lam)
    re, im = jnp.real(A), jnp.imag(A)
    A_block = jnp.stack([jnp.stack([re, -im], axis=-1),
                         jnp.stack([im, re], axis=-1)], axis=-2)   # (P,2,2)
    return A_block, _phi1(lam)


def complex_memory_rollout(params, es, dt=DT):
    """Run the memory bank over an encoded sequence. Returns (L, 2*P) REAL.

    The readout is the real and imaginary parts, which is why P complex units
    cost exactly 2P real temporal coordinates.

    IMPLEMENTATION, and why it is not the obvious one. The memory is linear
    with constant per-unit coefficients, so the recurrence
    z_k = exp(lambda) z_{k-1} + phi1(lambda) u_k is an affine scan. Written as
    a sequential `lax.scan` with a COMPLEX carry it measured 1259 ms per
    training update on the cluster, against 27 ms for the two ODE arms that
    take 512 sequential steps to this one's 64 - so neither sequential depth
    nor arithmetic volume explains it, and the complex carry is what differs.

    It is therefore written as a REAL 2x2 block associative scan: exp(lambda)
    acting on (Re z, Im z) is a rotation-scaling, the affine composition is
    associative, and `associative_scan` has logarithmic depth. The complex
    arithmetic is confined to the elementwise coefficient computation outside
    the scan. The mathematics is identical and is checked against a plain
    sequential reference.
    """
    from .gp_second_order import block_binary_operator
    Wr, Wi = params["W_in_re"], params["W_in_im"]
    A_block, Phi = complex_memory_coefficients(params, dt)
    ur = es @ Wr.T                                   # (L,P)
    ui = es @ Wi.T
    pr, pi = jnp.real(Phi), jnp.imag(Phi)
    br = (pr * ur - pi * ui) * dt                    # Re(Phi u) dt
    bi = (pr * ui + pi * ur) * dt
    b = jnp.stack([br, bi], axis=-1)                 # (L,P,2)
    A_elems = jnp.broadcast_to(A_block, es.shape[:-1] + A_block.shape)
    _, zs = jax.lax.associative_scan(block_binary_operator, (A_elems, b))
    return jnp.concatenate([zs[..., 0], zs[..., 1]], axis=-1)      # (L, 2P)


def complex_memory_rollout_sequential(params, es, dt=DT):
    """The same recurrence, written plainly. Reference for the scan above."""
    A_block, Phi = complex_memory_coefficients(params, dt)
    Wr, Wi = params["W_in_re"], params["W_in_im"]

    def step(z, e):
        u = (Wr @ e) + 1j * (Wi @ e)
        bb = Phi * u * dt
        z = ((A_block @ z[..., None])[..., 0]
             + jnp.stack([jnp.real(bb), jnp.imag(bb)], axis=-1))
        return z, jnp.concatenate([z[..., 0], z[..., 1]], axis=-1)

    z0 = jnp.zeros(A_block.shape[:1] + (2,), dtype=A_block.dtype)
    _, out = jax.lax.scan(step, z0, es)
    return out


# --------------------------------------- discrete transpose convolution ----
def discrete_transpose(rollout_fn, c):
    """The EXACT discrete adjoint of a causal LTI rollout, by reversal.

    R5. For a causal linear map s = K * f with zero initial state, the
    derivative of sum_t c_t s_t with respect to f is

        (K^T c)_t = sum_{sigma >= t} K_{sigma - t} c_sigma
                  = reverse( K * reverse(c) )_t

    so reversing the input, applying THE SAME EXECUTED discrete operator and
    reversing the output is the transpose convolution exactly - no sample/hold
    or end-of-step timing mismatch, and no continuous-time transfer function
    involved. The previous revision compared a sampled operator against a
    CONTINUOUS H(-i omega) at a tight tolerance, which mixes two different
    transfer functions; refining RK4 substeps cannot remove that difference.
    """
    return rollout_fn(c[::-1])[::-1]
