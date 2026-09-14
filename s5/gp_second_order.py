"""PROTOTYPE: full second-order (finite-inertia) generalized prospective mode.

    mu_i s_ddot_i + (1 + t_i j_i) s_dot_i + j_i s_i = b_i x + t_i b_i x_dot,
    j_i = -a_i,  Re(j_i) > 0,  0 < mu_i <= t_i

SEPARATE MECHANISM. It does NOT modify `gp_diagonal`, which remains the
validated M=0, gamma=1 member of the family. Do not reinterpret existing
checkpoints through it, and do not expect its outputs to equal the M=0 model at
finite mass - they are different members, not an approximation and its target.

Naming: `mu` is the MASS. The existing `m = 1 - t a` in gp_coefficients is a
denominator, not a mass; the names are kept distinct on purpose.

Derivative-free state
---------------------
With w_i = mu_i s_dot_i + t_i (j_i s_i - b_i x),

    d/dt (s, w) = [[-t j / mu,      1/mu     ],  (s, w)
                   [ (t/mu - 1) j, -1/mu     ]]
                + [ t b / mu, (1 - t/mu) b ] x

exactly as in the brief. Unit-interval ZOH is taken with an AUGMENTED MATRIX
EXPONENTIAL, never an eigendecomposition: admissible exceptional points exist
(t=1, mu=0.75, j=0.5+i*sqrt(3)/2) where eigenvectors coalesce although the
exponential and its parameter derivatives stay analytic.

Endpoints
---------
* mu = t : row two becomes w_dot = -w/t with no forcing, so from w0 = 0 the
  response is EXACTLY plain S5 for every t.
* mu -> 0 : approaches the M=0 model under common ZOH timing, with a boundary
  layer. Exactly zero mass is NOT a differentiable point; it selects the
  first-order implementation instead.

Timing
------
Consume the held token, advance the block state, read the FIRST component s.
Native learned D and the conjugate factor two are retained. For positive mass
the tied first-order `D_x x` is NOT added: the state transfer is strictly
proper and adding it would double-count a different model's feedthrough.
"""

import jax
import jax.numpy as np
from flax import linen as nn
from jax.scipy.linalg import expm

from .gp_coefficients import absorb_clock
from .ssm import S5SSM

#: uniform sufficient STABILITY bound from the brief: mu <= t
MU_RATIO_MAX = 1.0

#: MEASURED numerical lower bound on mu/t, not assumed from stability.
#:
#: The generator contains 1/mu, so tiny mass makes the augmented exponential
#: overflow even though the mathematics stays stable. Measured on a 4-mode
#: x64 fixture (t = 0.5, stable complex poles), error against the M = 0 model
#: after the boundary layer:
#:
#:     mu/t   1e-1   1e-2     1e-3     1e-4     1e-5     1e-6
#:     err    1.5e-2 1.5e-3   1.4e-4   1.4e-5   1.4e-6   NaN (overflow)
#:
#: Convergence is linear in mu/t and breaks down at 1e-6. The parameterization
#: is floored at 1e-4, a decade of margin above the last good value. This is a
#: NUMERICAL limit of this prototype, not a mathematical one.
MU_RATIO_MIN = 1e-4


def block_binary_operator(q_i, q_j):
    """Affine 2x2 block composition in time order: (A2 A1, A2 b1 + b2).

    The diagonal elementwise `binary_operator` in s5/ssm.py cannot be reused on
    matrices, and s5/ssm.py is deliberately left unchanged.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return (A_j @ A_i, (A_j @ b_i[..., None])[..., 0] + b_j)


def second_order_generator(a, b, t, mu):
    """Continuous (F, B) per mode, in clocked units. Shapes (P,2,2), (P,2,H)."""
    j = -a
    tm = (t / mu).astype(a.dtype)
    inv_mu = (1.0 / mu).astype(a.dtype)
    t_c = t.astype(a.dtype)
    F = np.stack([
        np.stack([-t_c * j * inv_mu, inv_mu * np.ones_like(j)], axis=-1),
        np.stack([(tm - 1.0) * j, -inv_mu * np.ones_like(j)], axis=-1),
    ], axis=-2)                                             # (P, 2, 2)
    B = np.stack([t_c[:, None] * b * inv_mu[:, None],
                  (1.0 - tm)[:, None] * b], axis=-2)        # (P, 2, H)
    return F, B


def second_order_zoh(a, b, t, mu):
    """Exact unit-interval ZOH via an augmented matrix exponential."""
    F, B = second_order_generator(a, b, t, mu)
    P, _, H = B.shape
    n = 2 + H
    aug = np.zeros((P, n, n), dtype=F.dtype)
    aug = aug.at[:, :2, :2].set(F).at[:, :2, 2:].set(B)
    E = jax.vmap(expm)(aug)
    return dict(A_bar=E[:, :2, :2], B_bar=E[:, :2, 2:], F=F, B=B)


def second_order_scan(A_bar, B_bar, input_sequence, h0=None, reset_mask=None):
    """h_k = A_bar h_{k-1} + B_bar x_k, over 2x2 blocks, via a block scan."""
    L = input_sequence.shape[0]
    P = A_bar.shape[0]
    A_elems = np.broadcast_to(A_bar, (L, P, 2, 2))
    if reset_mask is not None:
        keep = (~np.asarray(reset_mask, dtype=bool))[:, None, None, None]
        A_elems = A_elems * keep.astype(A_bar.dtype)
    B_elems = jax.vmap(lambda u: (B_bar @ u))(input_sequence)     # (L, P, 2)
    if h0 is not None:
        # fold a non-zero prehistory into the first element
        B_elems = B_elems.at[0].add((A_elems[0] @ h0[..., None])[..., 0])
    _, hs = jax.lax.associative_scan(block_binary_operator, (A_elems, B_elems))
    return hs                                                     # (L, P, 2)


def second_order_sequential(A_bar, B_bar, input_sequence, h0=None,
                            reset_mask=None):
    """Differentiable sequential reference for the same recurrence."""
    P = A_bar.shape[0]
    init = np.zeros((P, 2), dtype=A_bar.dtype) if h0 is None else h0
    mask = (np.zeros(input_sequence.shape[0], dtype=bool)
            if reset_mask is None else np.asarray(reset_mask, dtype=bool))

    def step(h, inp):
        x_k, r_k = inp
        h_prev = np.where(r_k, np.zeros_like(h), h)
        nxt = (A_bar @ h_prev[..., None])[..., 0] + B_bar @ x_k
        return nxt, nxt

    _, hs = jax.lax.scan(step, init, (input_sequence, mask))
    return hs


def second_order_readout(hs, C_tilde, D, input_sequence, conj_sym):
    """Read the FIRST block component s; no tied D_x at positive mass."""
    s = hs[:, :, 0]
    if conj_sym:
        ys = jax.vmap(lambda si: 2 * (C_tilde @ si).real)(s)
    else:
        ys = jax.vmap(lambda si: (C_tilde @ si).real)(s)
    return ys + jax.vmap(lambda u: D * u)(input_sequence)


class SecondOrderGPSSM(S5SSM):
    """Prototype module. Adds a mass parameter; never used by `gp_diagonal`."""

    gp_init_scale: float = 0.3
    mu_ratio_init: float = 0.5          # mu = rho * t, rho in (0, 1)

    def setup(self):
        super().setup()
        if self.bidirectional or self.discretization != "zoh" \
                or self.step_rescale != 1.0:
            raise ValueError("second-order prototype: causal, ZOH, unit clock "
                             "only")
        if not self.clip_eigs:
            raise ValueError("second-order prototype requires clip_eigs=True: "
                             "stability assumes Re(j) > 0")
        if not (MU_RATIO_MIN <= self.mu_ratio_init < 1.0):
            raise ValueError(
                f"mu_ratio_init must lie in [{MU_RATIO_MIN}, 1); mu = 0 is not "
                f"a differentiable mass point, and below {MU_RATIO_MIN} the "
                f"augmented exponential was MEASURED to overflow.")
        from .gp_coefficients import inverse_softplus
        import math
        self.t_raw = self.param(
            "so_response_raw",
            lambda rng, s: np.full(s, inverse_softplus(self.gp_init_scale),
                                   dtype=np.float32), (self.P,))
        self.rho_raw = self.param(
            "so_mu_ratio_raw",
            lambda rng, s: np.full(
                s, math.log(self.mu_ratio_init / (1.0 - self.mu_ratio_init)),
                dtype=np.float32), (self.P,))

    def response_and_mass(self):
        """t = softplus(raw) > 0 and mu = rho t with rho in (MU_RATIO_MIN, 1).

        The floor is a measured numerical bound (see MU_RATIO_MIN), not a
        mathematical one. Exactly zero mass is excluded by construction: it is
        a different model, selected by the first-order implementation.
        """
        t = jax.nn.softplus(self.t_raw)
        rho = MU_RATIO_MIN + (MU_RATIO_MAX - MU_RATIO_MIN) * jax.nn.sigmoid(
            self.rho_raw)
        return t, rho * t

    def coefficients(self):
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        a, b = absorb_clock(self.Lambda, B_tilde, step)
        t, mu = self.response_and_mass()
        out = second_order_zoh(a, b, t, mu)
        out.update(a=a, b=b, t=t, mu=mu)
        return out

    def __call__(self, input_sequence, reset_mask=None):
        c = self.coefficients()
        hs = second_order_scan(c["A_bar"], c["B_bar"], input_sequence,
                               reset_mask=reset_mask)
        return second_order_readout(hs, self.C_tilde, self.D, input_sequence,
                                    self.conj_sym)


def init_second_order_ssm(gp_init_scale=0.3, mu_ratio_init=0.5, **s5_kwargs):
    from functools import partial
    return partial(SecondOrderGPSSM, gp_init_scale=gp_init_scale,
                   mu_ratio_init=mu_ratio_init, **s5_kwargs)
