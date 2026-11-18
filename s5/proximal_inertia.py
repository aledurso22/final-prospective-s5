"""Unfiltered proximal inertial NLA as an S5 state recurrence (two states/mode).

Per complex mode, in token time, with kappa=-Lambda*Delta (Re kappa>0) and
physical drive b=Delta*B_tilde*u:

    m s'' + s' = -r_mu - tau d/dt r_mu,     r_mu = (kappa s - b)/(1 + mu kappa),

with the fixed per-mode ratios

    tau = alpha/|kappa|,  mu = proximal_ratio*tau,  m = mass_ratio*tau.

The production arm uses alpha=0.1, proximal_ratio=0.5, mass_ratio=1.5; the
matched control uses mass_ratio=1.0. These are fixed ratios, not parameters:
tau, mu and m follow Lambda and Delta differentiably and nothing is added to
the parameter tree.

Derivative-free coordinates. With d=1+mu*kappa, k=kappa/d and
w = m s' + tau r_mu, the total input derivative is absorbed exactly:

    d/dt [s, w] = [[-tau k/m, 1/m], [(tau/m - 1) k, -1/m]] [s, w]
                  + (1/d) [tau/m, 1 - tau/m] b.

w (and s) are continuous across zero-order-hold input jumps, while the
physical velocity s' may jump; nothing from the input derivative is dropped.
One-token exact physical-input ZOH comes from ONE augmented 3x3 exponential
per mode, [[A, g], [0, 0]], so no inversion of A, no distinct-pole
assumption and no eigendecomposition enters the differentiable path.

Properties verified in tests/test_proximal_inertia.py:
  * mass_ratio=1: the w row decouples (A_ws=0, g_w=0 exactly) and zero-prehistory
    responses are EXACTLY first-order proximal flow s'=-r_mu;
  * mass_ratio=1, proximal_ratio=0: exactly native S5 for any alpha;
  * sharp homogeneous stability for every Re(kappa)>0 iff
    proximal_ratio >= max(0, mass_ratio - 1); the candidate sits ON this bound
    (0.5 = 1.5 - 1), so no uniform numerical margin exists as Re(kappa)->0.
    Poles are monitored; nothing is clipped or damped.

Readout, conjugate symmetry, bidirectional zero starts and the D skip are the
native S5 ones; the position s is read out, never w.

INPUT-LEAD ABLATION (``input_lead`` = sigma). Expanding tau d/dt r_mu separates
the explicit input derivative:

    m s'' + s' = -r_mu - tau kappa/d s' + sigma tau/d b'.

sigma=1 is the model above (and runs the ORIGINAL code path unchanged);
sigma=0 removes only the prospective input lead. With
w_sigma = m s' + tau kappa s/d - sigma tau b/d the generator A is unchanged and
only the input column changes: g_sigma = [sigma tau/m, 1 - sigma tau/m]/d.
Input jumps then satisfy m Delta(s') = sigma tau Delta(b)/d. The transfer is
(1 + sigma tau p)/[(1+mu kappa)(m p^2+p) + kappa(1+tau p)]: same poles and DC
gain, different residues, and sigma=0 adds tau to the continuous DC delay.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from .ssm import S5SSM

#: the fixed production constants (candidate); the control changes MASS_RATIO only
ALPHA, PROXIMAL_RATIO, MASS_RATIO, CONTROL_MASS_RATIO = 0.1, 0.5, 1.5, 1.0
WIDTH = 2


def _check(alpha, mass_ratio, proximal_ratio, input_lead=1.0):
    if isinstance(input_lead, bool) or not isinstance(input_lead, (int, float)) \
            or not math.isfinite(input_lead):
        raise ValueError("input_lead must be a finite static number")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(mass_ratio) or mass_ratio <= 0:
        raise ValueError("mass_ratio must be finite and positive")
    if not math.isfinite(proximal_ratio) or proximal_ratio < 0:
        raise ValueError("proximal_ratio must be finite and nonnegative")


def inertia_generator(kappa, tau, mu, m, input_lead=1.0):
    """Continuous (A, g) for state (s, w_sigma); g multiplies the scalar drive b.

    Arrays broadcast over modes. tau and m must be positive, mu >= 0.
    input_lead (sigma) is static; sigma == 1 evaluates the original expression.
    """
    kappa, tau, mu, m = jnp.broadcast_arrays(kappa, tau, mu, m)
    dtype = jnp.result_type(kappa, jnp.complex64)
    kappa, tau, mu, m = (x.astype(dtype) for x in (kappa, tau, mu, m))
    d = 1 + mu * kappa
    k = kappa / d
    ratio = tau / m
    A = jnp.stack((jnp.stack((-ratio * k, 1 / m), -1),
                   jnp.stack(((ratio - 1) * k, -1 / m), -1)), -2)
    if input_lead == 1:      # the original model, the original arithmetic
        g = jnp.stack((ratio / d, (1 - ratio) / d), -1)
    else:
        lead = input_lead * ratio
        g = jnp.stack((lead / d, (1 - lead) / d), -1)
    return A, g


def augmented_zoh(A, g):
    """exp([[A, g], [0, 0]]) -> (Ad, bd): exact one-token ZOH, jointly."""
    n = A.shape[-1]
    aug = jnp.zeros(A.shape[:-2] + (n + 1, n + 1), A.dtype)
    aug = aug.at[..., :n, :n].set(A).at[..., :n, n].set(g)
    flat = aug.reshape((-1, n + 1, n + 1))
    disc = jax.vmap(expm)(flat).reshape(aug.shape)
    return disc[..., :n, :n], disc[..., :n, n]


def proximal_inertia_zoh(kappa, alpha=ALPHA, mass_ratio=MASS_RATIO,
                         proximal_ratio=PROXIMAL_RATIO, input_lead=1.0):
    """(Ad, bd, tau) per mode; bd multiplies the physical drive b."""
    _check(alpha, mass_ratio, proximal_ratio, input_lead)
    tau = jnp.asarray(alpha, kappa.real.dtype) / jnp.abs(kappa)
    A, g = inertia_generator(kappa, tau, proximal_ratio * tau, mass_ratio * tau, input_lead)
    Ad, bd = augmented_zoh(A, g)
    return Ad, bd, tau


# ------------------------------------------------- stationary block scan ---
# Width-generic copy of s5.gradient_history's stationary scan (which is fixed
# at width 3 and is left untouched so the P/H arms are unchanged).

def _matvec(A, x):
    return sum(A[..., :, j] * x[..., j, None] for j in range(A.shape[-1]))


def _matmul(A, B):
    return sum(A[..., :, j, None] * B[..., None, j, :] for j in range(A.shape[-1]))


def _compose(left, right):
    A0, b0 = left
    A1, b1 = right
    return _matmul(A1, A0), _matvec(A1, b0) + b1


def _stationary_scan(A, drive):
    """x_t = A x_{t-1} + drive_t, x_{-1}=0, for a time-invariant A.

    Pair adjacent inputs, recurse with A^2, recover the even states: O(L)
    block matvecs, O(log L) depth, no per-token matrix tape.
    """
    length = drive.shape[0]
    if length <= 1:
        return drive
    pairs = length // 2
    paired = _matvec(A, drive[:2 * pairs:2]) + drive[1:2 * pairs:2]
    odd = _stationary_scan(_matmul(A, A), paired)
    previous_odd = jnp.concatenate((jnp.zeros_like(drive[:1]), odd), axis=0)
    even = drive[::2] + _matvec(A, previous_odd[:(length + 1) // 2])
    inter = jnp.stack((even[:pairs], odd), axis=1).reshape((2 * pairs,) + drive.shape[1:])
    if length % 2:
        inter = jnp.concatenate((inter, even[-1:]), axis=0)
    return inter


@jax.custom_vjp
def _scan_adjoint(A, b, drive):
    """Forward scan with a recomputing linear-recurrence VJP (no AD tape)."""
    return _stationary_scan(A, drive[..., None] * b[None, ...])


def _scan_adjoint_fwd(A, b, drive):
    return _stationary_scan(A, drive[..., None] * b[None, ...]), (A, b, drive)


def _cot(value, primal):
    if not jnp.issubdtype(primal.dtype, jnp.complexfloating):
        value = value.real
    return value.astype(primal.dtype)


def _scan_adjoint_bwd(res, cot):
    A, b, drive = res
    states = _stationary_scan(A, drive[..., None] * b[None, ...])
    # bilinear (JAX) transpose: lambda_t = g_t + A^T lambda_{t+1}
    adj = _stationary_scan(jnp.swapaxes(A, -1, -2), cot[::-1])[::-1]
    grad_A = jnp.einsum("tpi,tpj->pij", adj[1:], states[:-1])
    grad_b = jnp.sum(adj * drive[..., None], axis=0)
    grad_drive = jnp.sum(adj * b[None, ...], axis=-1)
    return tuple(_cot(g, x) for g, x in ((grad_A, A), (grad_b, b), (grad_drive, drive)))


_scan_adjoint.defvjp(_scan_adjoint_fwd, _scan_adjoint_bwd)

SCANS = ("parallel", "parallel_reference", "associative", "sequential")


def scan_blocks(Ad, bd, drive, *, reverse=False, implementation="parallel"):
    """All block states for drive (length, modes); zero start per direction."""
    seq = drive[::-1] if reverse else drive
    if implementation == "parallel":
        states = _scan_adjoint(Ad, bd, seq)
    elif implementation == "parallel_reference":
        states = _stationary_scan(Ad, seq[..., None] * bd[None, ...])
    elif implementation == "associative":
        inputs = seq[..., None] * bd[None, ...]
        mats = jnp.broadcast_to(Ad, (seq.shape[0],) + Ad.shape)
        _, states = jax.lax.associative_scan(_compose, (mats, inputs))
    elif implementation == "sequential":
        inputs = seq[..., None] * bd[None, ...]

        def body(x, v):
            x = _matvec(Ad, x) + v
            return x, x
        _, states = jax.lax.scan(body, jnp.zeros_like(bd), inputs)
    else:
        raise ValueError(f"unknown scan {implementation!r}")
    return states[::-1] if reverse else states


class ProximalInertialS5SSM(S5SSM):
    """Native S5 parameters/readout; proximal inertial two-state recurrence.

    ``history_alpha`` is the horizon constant alpha (name shared with the
    history runner's factory). No parameter is added to the native tree.
    """
    history_alpha: float = ALPHA
    mass_ratio: float = MASS_RATIO
    proximal_ratio: float = PROXIMAL_RATIO
    input_lead: float = 1.0
    implementation: str = "parallel"

    def setup(self):
        _check(self.history_alpha, self.mass_ratio, self.proximal_ratio, self.input_lead)
        if self.discretization != "zoh":
            raise ValueError("proximal inertial S5 requires ZOH discretization")
        if self.implementation not in SCANS:
            raise ValueError(f"unknown scan {self.implementation!r}")
        S5SSM.setup(self)
        step = self.step_rescale * jnp.exp(self.log_step[:, 0])
        kappa = -self.Lambda * step
        self.inertia_Ad, self.inertia_bd, self.inertia_tau = proximal_inertia_zoh(
            kappa, self.history_alpha, self.mass_ratio, self.proximal_ratio, self.input_lead)
        self.inertia_B = step[:, None] * (self.B[..., 0] + 1j * self.B[..., 1])

    def __call__(self, input_sequence):
        drive = jax.vmap(lambda u: self.inertia_B @ u)(input_sequence)
        state = scan_blocks(self.inertia_Ad, self.inertia_bd, drive,
                            implementation=self.implementation)[..., 0]
        if self.bidirectional:
            back = scan_blocks(self.inertia_Ad, self.inertia_bd, drive, reverse=True,
                               implementation=self.implementation)[..., 0]
            state = jnp.concatenate((state, back), axis=-1)
        factor = 2 if self.conj_sym else 1
        ys = factor * jax.vmap(lambda s: (self.C_tilde @ s).real)(state)
        return ys + input_sequence * self.D
