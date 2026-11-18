"""Positive fast/slow memory of a resolvent-corrected S5 direction.

For r=kappa*s-b and d=-r/(1+mu*kappa), the continuous law is

    tau_f*v_f' = d-v_f,  tau_s*v_s' = d-v_s,
    s' = (1-weight)*v_f + weight*v_s.

The fixed-quadratic direction d has the same regularized inverse as proximal
history. This changes its temporal filter: positive fast/slow averaging replaces
the extrapolating gradient-history filter. It is not an acceleration claim.
For complex monotone modes Re(kappa)>0, the sharp uniform stability bound is
mu >= (1-weight)*tau_f + weight*tau_s. The production constructor enforces it.
Both state and physical input are regularized, preserving the native DC gain;
the continuous DC delay adds mu, just as in proximal history.

The fixed pilot uses tau_f=.1*h, tau_s=h, weight=4/9, mu=.5*h and
h=alpha/abs(kappa). Thus its regularization and horizon normalization match P,
while its weighted response time attains the stability bound. No parameters
are added. A 4x4 augmented exponential gives exact physical-input ZOH blocks
in the conditioned coordinates (s,h*v_f,h*v_s), up to floating-point error.
The existing stationary block scan and its memory-saving adjoint are reused.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from .gradient_history import NormalizedGradientHistoryS5SSM
from .ssm import S5SSM


def _validate_options(alpha, fast_ratio, slow_ratio, slow_weight,
                      proximal_ratio, *, allow_native=False,
                      enforce_stability=False):
    if not math.isfinite(alpha) or alpha < 0 or (alpha == 0 and not allow_native):
        raise ValueError("history_alpha must be finite and positive (or zero for the native constructor)")
    if (not math.isfinite(fast_ratio) or not math.isfinite(slow_ratio)
            or fast_ratio <= 0 or slow_ratio <= 0 or fast_ratio > slow_ratio):
        raise ValueError("memory ratios must be finite with 0 < fast_ratio <= slow_ratio")
    if not math.isfinite(slow_weight) or not 0 <= slow_weight <= 1:
        raise ValueError("slow_weight must be finite and in [0, 1]")
    if not math.isfinite(proximal_ratio) or proximal_ratio < 0:
        raise ValueError("proximal_ratio must be finite and nonnegative")
    minimum = (1-slow_weight)*fast_ratio + slow_weight*slow_ratio
    # Comparison tolerance only: retain the supplied coefficient exactly at a
    # floating-point representation of the analytic boundary; never project it.
    if (enforce_stability and alpha > 0 and proximal_ratio < minimum
            and not math.isclose(proximal_ratio, minimum, rel_tol=1e-12, abs_tol=1e-15)):
        raise ValueError(
            "proximal_ratio violates the uniform stability bound: "
            f"requires >= (1-slow_weight)*fast_ratio + slow_weight*slow_ratio = {minimum!r}")


def positive_memory_generator(kappa, tau_f, tau_s, mu, weight):
    """Return physical-coordinate A and unit-b drive for (s,v_f,v_s).

    Arguments broadcast over modes. Positive response times and 0<=weight<=1
    define the intended law. This pure algebraic helper deliberately permits
    below-bound mu for stability diagnostics; it performs no projection.
    kappa=0, equal response times and endpoint weights are well defined here.
    """
    kappa, tau_f, tau_s, mu, weight = jnp.broadcast_arrays(
        kappa, tau_f, tau_s, mu, weight)
    dtype = jnp.result_type(kappa, jnp.complex64)
    kappa, tau_f, tau_s, mu, weight = (
        a.astype(dtype) for a in (kappa, tau_f, tau_s, mu, weight))
    zero, one = jnp.zeros_like(kappa), jnp.ones_like(kappa)
    regularizer = one + mu*kappa
    fast_input = one/(regularizer*tau_f)
    slow_input = one/(regularizer*tau_s)
    A = jnp.stack((jnp.stack((zero, one-weight, weight), -1),
                   jnp.stack((-kappa*fast_input, -one/tau_f, zero), -1),
                   jnp.stack((-kappa*slow_input, zero, -one/tau_s), -1)), -2)
    drive = jnp.stack((zero, fast_input, slow_input), -1)
    return A, drive


def positive_memory_zoh(kappa, alpha=1., fast_ratio=.1, slow_ratio=1.,
                        slow_weight=4/9, proximal_ratio=.5):
    """Return one-token (Ad, physical-input bd, horizon), scaled as (s,h*v).

    kappa must be nonzero; no epsilon changes the normalization. Static options
    are checked, while the uniform stability bound is enforced by the model,
    not this diagnostic helper. alpha=0 is a separate exact native model branch.
    """
    _validate_options(alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio)
    kappa = jnp.asarray(kappa)
    horizon = jnp.asarray(alpha, kappa.real.dtype)/jnp.abs(kappa)
    A, drive = positive_memory_generator(
        kappa, fast_ratio*horizon, slow_ratio*horizon,
        proximal_ratio*horizon, slow_weight)
    scale = jnp.stack((jnp.ones_like(horizon), horizon, horizon), -1)
    A = A*scale[..., :, None]/scale[..., None, :]
    drive = drive*scale
    augmented = jnp.zeros(A.shape[:-2] + (4, 4), A.dtype)
    augmented = augmented.at[..., :3, :3].set(A).at[..., :3, 3].set(drive)
    discrete = jax.vmap(expm)(augmented.reshape((-1, 4, 4))).reshape(augmented.shape)
    return discrete[..., :3, :3], discrete[..., :3, 3], horizon


class ProximalMemoryMixS5SSM(NormalizedGradientHistoryS5SSM):
    """Positive two-timescale direction memory with the native S5 backbone."""

    history_alpha: float = 1.
    fast_ratio: float = .1
    slow_ratio: float = 1.
    slow_weight: float = 4/9
    proximal_ratio: float = .5

    def setup(self):
        _validate_options(self.history_alpha, self.fast_ratio, self.slow_ratio,
                          self.slow_weight, self.proximal_ratio,
                          allow_native=True, enforce_stability=True)
        if self.discretization != "zoh":
            raise ValueError("proximal memory mix requires ZOH discretization")
        if self.implementation not in ("parallel", "parallel_reference", "associative", "sequential"):
            raise ValueError("unknown proximal-memory-mix scan implementation")
        # Call native setup directly: retain every parameter name/RNG draw and
        # avoid constructing the parent's different gradient-history blocks.
        S5SSM.setup(self)
        if self.history_alpha == 0:
            return
        step = self.step_rescale*jnp.exp(self.log_step[:, 0])
        kappa = -self.Lambda*step
        self.history_Ad, self.history_bd, self.history_horizon = positive_memory_zoh(
            kappa, self.history_alpha, self.fast_ratio, self.slow_ratio,
            self.slow_weight, self.proximal_ratio)
        self.history_B = step[:, None]*(self.B[..., 0]+1j*self.B[..., 1])
