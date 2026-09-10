"""The exact three-state learning-access system.

    z_{k+1} = (rho + theta) z_k + x_k          intended computation
    b_{k+1} = beta b_k + v_k                   hidden physical mode
    s_{k+1} = a s_k + (1-a) (z_{k+1} + theta b_{k+1})    physical realization

The local prospective inverse is EXACT:

    w_k = (s_k - a s_{k-1}) / (1 - a) = z_k + theta b_k

because s_k - a s_{k-1} = (1-a)(z_k + theta b_k) identically.

Consequences, and the whole point of this module:

  * at theta = 0 the realization is INFERENCE-PERFECT: w_k = z_k exactly, for
    any x, v, z_0, b_0.
  * its parameter tangent is NOT the intended one:

        d w_k / d theta |_0 = d z_k / d theta |_0 + b_k

    The extra term b_k is a physical direction that never appears in the
    nominal output, so no function of the observed prospective history can
    recover it.

Written with JAX so the same code supports autodiff, and with a pure-numpy
reference path so autodiff can be checked against something independent.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ThreeStateConfig:
    rho: float = 0.9
    beta: float = 0.8
    a: float = 0.7
    z0: float = 1.0
    b0: float = -4.0

    def pole(self, theta):
        """The INTENDED recurrent pole that learning is supposed to find."""
        return self.rho + theta


def rollout(cfg, theta, x, v):
    """Simulate the three states. Returns dict of (T+1,) arrays indexed 0..T.

    `x` and `v` are the drives applied at steps 1..T, so len(x) == len(v) == T.
    `s` is initialized on the manifold, s_0 = z_0 + theta b_0, which is the
    consistent choice: it makes w_k exact from k = 1 onwards.
    """
    T = len(x)
    z = jnp.zeros(T + 1).at[0].set(cfg.z0)
    b = jnp.zeros(T + 1).at[0].set(cfg.b0)
    s = jnp.zeros(T + 1).at[0].set(cfg.z0 + theta * cfg.b0)

    def step(carry, inp):
        zk, bk, sk = carry
        xk, vk = inp
        z1 = (cfg.rho + theta) * zk + xk
        b1 = cfg.beta * bk + vk
        s1 = cfg.a * sk + (1.0 - cfg.a) * (z1 + theta * b1)
        return (z1, b1, s1), (z1, b1, s1)

    (_, _, _), (zs, bs, ss) = jax.lax.scan(
        step, (z[0], b[0], s[0]), (jnp.asarray(x), jnp.asarray(v)))
    return dict(z=jnp.concatenate([z[:1], zs]),
                b=jnp.concatenate([b[:1], bs]),
                s=jnp.concatenate([s[:1], ss]))


def prospective_inverse(cfg, s):
    """w_k = (s_k - a s_{k-1}) / (1 - a), defined for k >= 1.

    Returns an array aligned with s[1:], i.e. w[i] corresponds to step i+1.
    This is the ONLY quantity a local causal corrector can see.
    """
    s = jnp.asarray(s)
    return (s[1:] - cfg.a * s[:-1]) / (1.0 - cfg.a)


def w_of(cfg, theta, x, v):
    """Convenience: the local prospective observable for a given theta."""
    return prospective_inverse(cfg, rollout(cfg, theta, x, v)["s"])


def side_measurement(cfg, theta, x, v, delta, noise=None):
    """m_k = delta b_k + n_k, the weak-access side channel (k >= 1)."""
    b = rollout(cfg, theta, x, v)["b"][1:]
    m = delta * b
    if noise is not None:
        m = m + jnp.asarray(noise)
    return m


def corrected_estimate(cfg, theta, x, v, delta, noise=None, gain=None):
    """z_hat_k = w_k - theta * gain * m_k.

    With the exact gain 1/delta and no noise this reproduces z for the FULL
    parameterized family, not merely at theta = 0:

        z_hat = (z + theta b) - (theta/delta)(delta b) = z.

    `gain=None` selects the exact 1/delta.
    """
    g = (1.0 / delta) if gain is None else gain
    w = w_of(cfg, theta, x, v)
    m = side_measurement(cfg, theta, x, v, delta, noise)
    return w - theta * g * m


# ------------------------------------------------------------ analytic refs

def dz_dtheta_analytic(cfg, x):
    """d z_k / d theta at theta = 0, in closed form.

    z_{k+1} = (rho+theta) z_k + x_k  =>  dz_{k+1} = z_k + rho dz_k, dz_0 = 0,
    so dz_k = sum_{j<k} rho^{k-1-j} z_j with z_j the theta=0 trajectory.
    """
    T = len(x)
    z = np.empty(T + 1)
    z[0] = cfg.z0
    for k in range(T):
        z[k + 1] = cfg.rho * z[k] + x[k]
    dz = np.zeros(T + 1)
    for k in range(T):
        dz[k + 1] = z[k] + cfg.rho * dz[k]
    return dz


def dw_dtheta_analytic(cfg, x, v):
    """d w_k / d theta |_0 = d z_k / d theta |_0 + b_k, for k >= 1."""
    dz = dz_dtheta_analytic(cfg, x)
    T = len(x)
    b = np.empty(T + 1)
    b[0] = cfg.b0
    for k in range(T):
        b[k + 1] = cfg.beta * b[k] + v[k]
    return dz[1:] + b[1:]


def numpy_rollout(cfg, theta, x, v):
    """Independent pure-numpy reference, so JAX is never checked against itself."""
    T = len(x)
    z = np.empty(T + 1); b = np.empty(T + 1); s = np.empty(T + 1)
    z[0], b[0] = cfg.z0, cfg.b0
    s[0] = cfg.z0 + theta * cfg.b0
    for k in range(T):
        z[k + 1] = (cfg.rho + theta) * z[k] + x[k]
        b[k + 1] = cfg.beta * b[k] + v[k]
        s[k + 1] = cfg.a * s[k] + (1.0 - cfg.a) * (z[k + 1] + theta * b[k + 1])
    return dict(z=z, b=b, s=s)
