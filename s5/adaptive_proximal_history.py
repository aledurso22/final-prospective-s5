"""Learn per-mode horizons within the existing proximal-history equation.

For one complex mode, with physical drive b and kappa=-Lambda*Delta,

    tau = (alpha / abs(kappa)) * exp(log_horizon_scale)
    mu = proximal_ratio * tau
    r_mu = (kappa*s - b) / (1 + mu*kappa)
    tau*s'' + s' = -r_mu - tau*q',   tau*q' = r_mu - q.

The new real parameter is constant along an input sequence. It unties the
dimensionless horizon tau*abs(kappa) from the fixed alpha without changing
the three-state scan, output map, equilibrium, or proximal operator. For
positive finite tau, Re(kappa)>0 and proximal_ratio>=1/2 the existing frozen
continuous-mode stability result applies. Finite precision and nonfinite
parameters must still be monitored; no pole or parameter is clipped here.

At a zero log_horizon_scale this is exactly the existing P construction,
including its dependence on the shared Lambda and log_step parameters.
The frozen control keeps the same parameter tree but stops the gradient of
the new leaf. Its optimizer should additionally exclude that leaf from
weight decay (and may explicitly zero its updates).
"""

import math

import jax
import jax.numpy as jnp

from .gradient_history import history_zoh
from .proximal_history import ProximalGradientHistoryS5SSM
from .ssm import S5SSM


def adaptive_proximal_history_zoh(kappa, log_horizon_scale, alpha=1.,
                                 proximal_ratio=.5):
    """Return exact-ZOH (Ad, physical-input bd, tau) for each mode.

The exponential only approximates the matrix exponential numerically; the
underlying physical-input hold and the input scaling match P. Dynamic
nonfinite inputs are not silently repaired, so the runner can record them.
"""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("adaptive proximal history requires finite alpha > 0")
    if not math.isfinite(proximal_ratio) or proximal_ratio < 0:
        raise ValueError("proximal_ratio must be finite and nonnegative")
    kappa = jnp.asarray(kappa)
    xi = jnp.asarray(log_horizon_scale)
    if jnp.issubdtype(xi.dtype, jnp.complexfloating):
        raise ValueError("log_horizon_scale must be real")
    xi = xi.astype(kappa.real.dtype)
    horizon = (jnp.asarray(alpha, kappa.real.dtype) / jnp.abs(kappa)) * jnp.exp(xi)
    if proximal_ratio == 0:
        A, b = history_zoh(kappa, horizon, horizon)
        return A, b, horizon
    denominator = 1 + proximal_ratio*horizon*kappa
    A, b = history_zoh(kappa/denominator, horizon, horizon)
    return A, b/denominator[..., None], horizon


class AdaptiveProximalHistoryS5SSM(ProximalGradientHistoryS5SSM):
    """P plus one zero-initialized real log-horizon multiplier per mode."""

    history_alpha: float = 1.
    trainable_horizon: bool = True

    def setup(self):
        if not math.isfinite(self.history_alpha) or self.history_alpha <= 0:
            raise ValueError("history_alpha must be positive and finite")
        if not math.isfinite(self.proximal_ratio) or self.proximal_ratio < 0:
            raise ValueError("proximal_ratio must be finite and nonnegative")
        if self.discretization != "zoh":
            raise ValueError("adaptive proximal-history S5 requires ZOH discretization")
        if self.implementation not in ("parallel", "parallel_reference", "associative", "sequential"):
            raise ValueError("unknown adaptive proximal-history scan implementation")
        # Append the new leaf after all native RNG draws. The inherited
        # position-only readout and stationary custom-adjoint scan are reused.
        S5SSM.setup(self)
        self.log_horizon_scale = self.param(
            "log_horizon_scale", lambda key, shape: jnp.zeros(shape, self.log_step.dtype),
            (self.P,))
        xi = self.log_horizon_scale
        if not self.trainable_horizon:
            xi = jax.lax.stop_gradient(xi)
        step = self.step_rescale*jnp.exp(self.log_step[:, 0])
        kappa = -self.Lambda*step
        self.history_Ad, self.history_bd, self.history_horizon = adaptive_proximal_history_zoh(
            kappa, xi, self.history_alpha, self.proximal_ratio)
        self.history_B = step[:, None]*(self.B[..., 0]+1j*self.B[..., 1])
