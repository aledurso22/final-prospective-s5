"""Resolvent-filtered NLA history, preserving the S5 diagonal-mode scan.

For kappa=-Lambda*Delta with Re(kappa)>0 and b=Delta*B*u, replace the
history residual kappa*s-b by its Yosida/proximal regularization

    r_mu = (kappa*s-b)/(1+mu*kappa),  mu=proximal_ratio*tau.

The existing law is otherwise unchanged: tau*s''+s'=-r_mu-tau*q',
tau*q'=r_mu-q, with tau=alpha/abs(kappa).  Input and state coefficients
are both scaled; scaling only kappa would change DC gain.  The augmented
exponential applies exact physical-input ZOH (up to floating-point error).

proximal_ratio=0 exactly delegates to the previous history construction.
history_alpha=0 delegates to native S5.  For every positive horizon and
Re(kappa)>0, ratios >= 1/2 give strictly stable continuous homogeneous
poles; 1/2 is the sharp universal bound over the right half-plane.
The continuous DC gain is native, while its low-frequency delay gains mu.
This is a resolvent of a complex monotone mode, not a claim that its
rotation is the Hessian of a real convex energy.  No trained parameters
are added, and no pole clipping is used.
"""

import math

import jax.numpy as jnp

from .gradient_history import (NormalizedGradientHistoryS5SSM, history_zoh,
                              normalized_history_zoh)
from .ssm import S5SSM


def proximal_history_zoh(kappa, alpha=.1, proximal_ratio=.5):
    """Return (Ad, physical-input bd, horizon) for positive static alpha.

    kappa must have positive real part.  No epsilon or projection modifies
    its value. The constructor handles alpha=0 using the separate native arm.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("proximal history blocks require finite alpha > 0")
    if not math.isfinite(proximal_ratio) or proximal_ratio < 0:
        raise ValueError("proximal_ratio must be finite and nonnegative")
    if proximal_ratio == 0:
        return normalized_history_zoh(kappa, alpha)
    horizon = jnp.asarray(alpha, kappa.real.dtype) / jnp.abs(kappa)
    denominator = 1 + proximal_ratio*horizon*kappa
    A, b = history_zoh(kappa/denominator, horizon, horizon)
    return A, b/denominator[..., None], horizon


class ProximalGradientHistoryS5SSM(NormalizedGradientHistoryS5SSM):
    """Existing history scan/readout with a regularized corrective residual."""

    proximal_ratio: float = .5

    def setup(self):
        if not math.isfinite(self.history_alpha) or self.history_alpha < 0:
            raise ValueError("history_alpha must be finite and nonnegative")
        if not math.isfinite(self.proximal_ratio) or self.proximal_ratio < 0:
            raise ValueError("proximal_ratio must be finite and nonnegative")
        if self.discretization != "zoh":
            raise ValueError("proximal-history S5 requires ZOH discretization")
        if self.implementation not in ("parallel", "parallel_reference", "associative", "sequential"):
            raise ValueError("unknown proximal-history scan implementation")
        # Preserve every native parameter name and RNG draw; do not run the
        # parent history setup, which would build the unregularized blocks.
        S5SSM.setup(self)
        if self.history_alpha == 0:
            return
        step = self.step_rescale*jnp.exp(self.log_step[:, 0])
        kappa = -self.Lambda*step
        self.history_Ad, self.history_bd, self.history_horizon = proximal_history_zoh(
            kappa, self.history_alpha, self.proximal_ratio)
        self.history_B = step[:, None]*(self.B[..., 0]+1j*self.B[..., 1])
