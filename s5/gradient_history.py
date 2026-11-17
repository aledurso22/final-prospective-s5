"""Experimental finite-response gradient-history S5, with exact ZOH blocks.

This constructor does not change the existing S5 or prospective arms. Time is
in tokens, kappa=-Lambda*Delta, b=Delta*B_tilde*u. The continuous law is

    tau*s'' + s' = -r - (tau/ell)*(r-q),   ell*q' = r-q,
    r = kappa*s-b.

Thus c=1 and m=tau; the formal ell -> 0 input/output limit is native S5.
We use the equivalent scaled state (s, tau*s', tau*q), and discretize its
affine dynamics by a 4x4 augmented matrix exponential. No eigendecomposition
or pole-residue division is used in the differentiable path.

The deliberately restricted first experiment fixes tau=ell=alpha/|kappa|
with alpha=0.10. Horizons change differentiably as Lambda and Delta learn;
there are no extra learned parameters. This normalization/equal-horizon tie
is a modeling choice, not a consequence of the energy argument. Stability
of the initialized modes does NOT guarantee stability after training.

alpha=0 is an explicit native-constructor control, not a numerical evaluation
of the singular three-state equation. All positive-alpha runs retain three
complex states per native mode. Scans start each direction independently at
zero. Complex64 construction is genuinely complex64 (no hidden x64 cast).
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from .ssm import S5SSM


def history_generator(kappa, tau, ell):
    """Continuous A and unit-b drive, for scaled state (s,tau*v,tau*q).

    Inputs broadcast over modes; tau and ell must be strictly positive.
    Keeping independent horizons here supports mathematical diagnostics, not
    additional parameters in ``NormalizedGradientHistoryS5SSM``.
    """
    kappa, tau, ell = jnp.broadcast_arrays(kappa, tau, ell)
    dtype = jnp.result_type(kappa, jnp.complex64)
    kappa, tau, ell = (a.astype(dtype) for a in (kappa, tau, ell))
    zero, one = jnp.zeros_like(kappa), jnp.ones_like(kappa)
    ratio = tau / ell
    A = jnp.stack((jnp.stack((zero, one/tau, zero), -1),
                   jnp.stack((-(one+ratio)*kappa, -one/tau, one/ell), -1),
                   jnp.stack((ratio*kappa, zero, -one/ell), -1)), -2)
    drive = jnp.stack((zero, one+ratio, -ratio), -1)
    return A, drive


def history_zoh(kappa, tau, ell):
    """Exact one-token ZOH blocks (Ad, bd), bd multiplying the drive b.

    Expm has the usual floating-point approximation error. The input column
    is obtained jointly with Ad, without inverting A or assuming distinct
    poles; kappa=0 and repeated poles are allowed for this pure function.
    """
    A, drive = history_generator(kappa, tau, ell)
    aug = jnp.zeros(A.shape[:-2] + (4, 4), A.dtype)
    aug = aug.at[..., :3, :3].set(A).at[..., :3, 3].set(drive)
    flat = aug.reshape((-1, 4, 4))
    discrete = jax.vmap(expm)(flat).reshape(aug.shape)
    return discrete[..., :3, :3], discrete[..., :3, 3]


def normalized_history_zoh(kappa, alpha=0.10):
    """Fixed dimensionless horizon; kappa must be nonzero, alpha positive.

    No epsilon/clipping silently changes the law at kappa=0. The constructor
    rejects alpha<=0 except for its separately labeled exact native branch.
    Native S5's strictly negative clipped Lambda_re keeps kappa nonzero in
    the intended training protocol, provided Delta is finite and positive.
    """
    horizon = jnp.asarray(alpha, kappa.real.dtype) / jnp.abs(kappa)
    Ad, bd = history_zoh(kappa, horizon, horizon)
    return Ad, bd, horizon


def _matvec(A, x):
    # Tiny fixed blocks: explicit elementwise operations avoid batched GEMM
    # overhead (and permit fusion) on these 3x3/3x1 products.
    return sum(A[..., :, j] * x[..., j, None] for j in range(3))


def _matmul(A, B):
    return sum(A[..., :, j, None] * B[..., None, j, :] for j in range(3))


def _compose(left, right):
    A0, b0 = left
    A1, b1 = right
    return _matmul(A1, A0), _matvec(A1, b0) + b1


def _stationary_scan(A, drive):
    """Affine parallel scan specialized to a time-independent transition.

    Pair adjacent inputs (A*d_even+d_odd), recurse using A^2, then recover
    the missing even states. Only one transition per mode is squared at
    each level: there is no per-token matrix product or matrix scan tape.
    This performs O(length) block matvecs with O(log length) parallel depth.
    The algebra is exactly the same affine composition as associative_scan.
    """
    length = drive.shape[0]
    if length <= 1:
        return drive
    pairs = length // 2
    paired_drive = _matvec(A, drive[:2*pairs:2]) + drive[1:2*pairs:2]
    odd = _stationary_scan(_matmul(A, A), paired_drive)
    previous_odd = jnp.concatenate((jnp.zeros_like(drive[:1]), odd), axis=0)
    even = drive[::2] + _matvec(A, previous_odd[:(length+1)//2])
    interleaved = jnp.stack((even[:pairs], odd), axis=1).reshape(
        (2*pairs,) + drive.shape[1:])
    if length % 2:
        interleaved = jnp.concatenate((interleaved, even[-1:]), axis=0)
    return interleaved


@jax.custom_vjp
def _history_scan_adjoint(A, b, drive):
    """Same forward scan with a recomputed, explicit linear-recurrence VJP.

    The custom-VJP residuals contain only the scalar input drive and small
    transition/input blocks. Backward re-evaluates the forward states instead
    of retaining the recursive scan's automatic-differentiation tape. XLA may
    still reschedule or reuse buffers; actual memory is checked separately.
    """
    return _stationary_scan(A, drive[..., None] * b[None, ...])


def _history_scan_adjoint_fwd(A, b, drive):
    states = _stationary_scan(A, drive[..., None] * b[None, ...])
    return states, (A, b, drive)


def _cotangent_dtype(value, primal):
    # custom_vjp must also support real inputs to the public scan. The
    # production constructor uses complex A/b/drive, but a real primal's
    # cotangent is the real part of this complex-linear pullback.
    if not jnp.issubdtype(primal.dtype, jnp.complexfloating):
        value = value.real
    return value.astype(primal.dtype)


def _history_scan_adjoint_bwd(residuals, cotangent):
    A, b, drive = residuals
    states = _stationary_scan(A, drive[..., None] * b[None, ...])
    # JAX represents complex cotangents with a bilinear transpose, not a
    # Hermitian transpose: lambda_t = g_t + A.T @ lambda_{t+1}.
    adjoint = _stationary_scan(jnp.swapaxes(A, -1, -2), cotangent[::-1])[::-1]
    # x_{-1}=0, so no transition contribution from token zero. Contract
    # directly rather than allocating a (length,modes,3,3) outer-product.
    grad_A = jnp.einsum("tpi,tpj->pij", adjoint[1:], states[:-1])
    grad_b = jnp.sum(adjoint * drive[..., None], axis=0)
    grad_drive = jnp.sum(adjoint * b[None, ...], axis=-1)
    return tuple(_cotangent_dtype(g, x) for g, x in
                 ((grad_A, A), (grad_b, b), (grad_drive, drive)))


_history_scan_adjoint.defvjp(_history_scan_adjoint_fwd,
                           _history_scan_adjoint_bwd)


def scan_history_blocks(Ad, bd, drive, *, reverse=False, implementation="parallel"):
    """Return all three states for drive shape (length,modes), zero start.

    Ad shape (modes,3,3), bd shape (modes,3). The input/output sampling is
    post-update, matching native S5. Reverse direction processes reversed
    inputs from its own zero state and restores the original token order.
    ``parallel`` has an explicit reverse-mode rule; ``parallel_reference``
    is the same forward computation with ordinary autodiff, also available
    when direct forward-mode differentiation is needed.
    """
    sequence = drive[::-1] if reverse else drive
    if implementation == "parallel":
        states = _history_scan_adjoint(Ad, bd, sequence)
    elif implementation == "parallel_reference":
        states = _stationary_scan(Ad, sequence[..., None] * bd[None, ...])
    elif implementation == "associative":
        inputs = sequence[..., None] * bd[None, ...]
        matrices = jnp.broadcast_to(Ad, (sequence.shape[0],) + Ad.shape)
        _, states = jax.lax.associative_scan(_compose, (matrices, inputs))
    elif implementation == "sequential":
        inputs = sequence[..., None] * bd[None, ...]
        def body(state, value):
            state = _matvec(Ad, state) + value
            return state, state
        _, states = jax.lax.scan(body, jnp.zeros_like(bd), inputs)
    else:
        raise ValueError(f"unknown gradient-history scan {implementation!r}")
    return states[::-1] if reverse else states


class NormalizedGradientHistoryS5SSM(S5SSM):
    """Fixed-alpha finite-response NLA extension, native backbone/readout.

    Callers must monitor complex-block stability during training. This
    constructor deliberately does not clip poles or add hidden damping.
    """
    history_alpha: float = 0.10
    implementation: str = "parallel"

    def setup(self):
        if not math.isfinite(self.history_alpha) or self.history_alpha < 0:
            raise ValueError("history_alpha must be finite and nonnegative")
        if self.discretization != "zoh":
            raise ValueError("gradient-history S5 requires ZOH discretization")
        if self.implementation not in ("parallel", "parallel_reference", "associative", "sequential"):
            raise ValueError("gradient-history scan must be parallel, parallel_reference, associative or sequential")
        super().setup()
        if self.history_alpha == 0:
            return
        step = self.step_rescale * jnp.exp(self.log_step[:, 0])
        kappa = -self.Lambda * step
        self.history_Ad, self.history_bd, self.history_horizon = normalized_history_zoh(
            kappa, self.history_alpha)
        self.history_B = step[:, None] * (self.B[..., 0] + 1j*self.B[..., 1])

    def __call__(self, input_sequence):
        if self.history_alpha == 0:
            return super().__call__(input_sequence)
        drive = jax.vmap(lambda u: self.history_B @ u)(input_sequence)
        state = scan_history_blocks(self.history_Ad, self.history_bd, drive,
                                    implementation=self.implementation)[..., 0]
        if self.bidirectional:
            backward = scan_history_blocks(self.history_Ad, self.history_bd, drive,
                                            reverse=True,
                                            implementation=self.implementation)[..., 0]
            state = jnp.concatenate((state, backward), axis=-1)
        factor = 2 if self.conj_sym else 1
        ys = factor * jax.vmap(lambda s: (self.C_tilde @ s).real)(state)
        return ys + input_sequence * self.D
