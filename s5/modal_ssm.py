"""Matched conventional modal SSM control (brief section 4).

A standard modal SSM can contain the GP realized map WITHOUT a full dense
feedthrough matrix:

    hdot = alpha h + B_h x
    y    = 2 Re{ C [ h + diag(z) B_h x ] } + D_native x

Initialized from the SAME GP initialization, with d_i = 1 - t_i a_i:

    alpha_i = a_i / d_i        B_h,i = b_i / d_i**2        z_i = t_i d_i

so `diag(z) B_h` equals the GP tied state feedthrough exactly:

    z_i B_h,i = t_i d_i * b_i / d_i**2 = t_i b_i / d_i = d_x,i     (GP)

Parameter accounting, deliberately matched:

    GP    : complex native pole (2P) + real log_step (P) + real response (P) = 4P
    modal : complex alpha (2P)       + complex z (2P)                        = 4P

with identical input/readout/native-D dimensions. `log_step` is NOT carried:
the clock is already absorbed into alpha and B_h, and an unused log_step would
be a silent extra parameter.

Same unit-interval ZOH, same phi1, same diagonal scan. Trained as free complex
modal parameters, so independent Adam runs need not track GP exactly - that is
the point, it tests parameterization and constraints. Transporting every GP
update through the exact conversion, by contrast, must preserve the realized
function.

This control is NOT expected to lose. The claim it tests is the opposite one:
that the new physics does not create linear representational power an ordinary
SSM lacks.
"""

from functools import partial

import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .gp_coefficients import phi1
from .ssm import binary_operator
from .ssm_init import init_CV, init_log_steps, init_VinvB, trunc_standard_normal

MODAL_TEMPORAL_PARAMS = ("alpha_re", "alpha_im", "z_re", "z_im")
#: minimum decay imposed on Re(alpha) when clipping is requested
ALPHA_CLIP = -1e-4


def gp_to_modal(a, b, t):
    """Exact conversion from clocked GP quantities to modal coordinates."""
    t_c = t.astype(a.dtype)
    d = 1.0 - t_c * a
    return dict(alpha=a / d, B_h=b / (d ** 2)[:, None], z=t_c * d)


def matched_init(Lambda_re_init, Lambda_im_init, Vinv, H, P, gp_init_scale,
                 rng, dt_min=0.001, dt_max=0.1, conj_sym=True,
                 clip_eigs=False):
    """Reproduce the S5 initialization, then convert it to modal coordinates.

    Returns the modal init AND the underlying (Lambda, log_step, B) it came
    from, so a test can build the matching GP model and verify the conversion
    rather than trusting it.
    """
    local_P = 2 * P if conj_sym else P
    k_step, k_B = jax.random.split(rng)
    Lambda = np.asarray(Lambda_re_init) + 1j * np.asarray(Lambda_im_init)
    if clip_eigs:
        Lambda = np.clip(np.asarray(Lambda_re_init), None, ALPHA_CLIP) \
            + 1j * np.asarray(Lambda_im_init)
    log_step = init_log_steps(k_step, (P, dt_min, dt_max))
    step = np.exp(log_step[:, 0])
    B = init_VinvB(lecun_normal(), k_B, (local_P, H), Vinv)
    B_tilde = B[..., 0] + 1j * B[..., 1]
    a = step * Lambda
    b = step[:, None] * B_tilde
    t = np.full((P,), float(gp_init_scale))
    modal = gp_to_modal(a, b, t)
    return dict(modal=modal, Lambda=Lambda, log_step=log_step, B=B,
                a=a, b=b, t=t)


class ModalSSM(nn.Module):
    """Free complex modal SSM with a tied modal feedthrough. No log_step."""

    Lambda_re_init: np.ndarray
    Lambda_im_init: np.ndarray
    V: np.ndarray
    Vinv: np.ndarray
    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0
    gp_init_scale: float = 0.05

    def setup(self):
        if self.bidirectional:
            raise ValueError("modal_ssm control is causal; set "
                             "bidirectional=False")
        if self.discretization != "zoh":
            raise ValueError("modal_ssm control supports ZOH only")
        if self.step_rescale != 1.0:
            raise ValueError("modal_ssm uses the unit sample clock; "
                             "step_rescale must be 1.0")
        local_P = 2 * self.P if self.conj_sym else self.P

        def _init(rng):
            return matched_init(self.Lambda_re_init, self.Lambda_im_init,
                                self.Vinv, self.H, self.P, self.gp_init_scale,
                                rng, self.dt_min, self.dt_max, self.conj_sym,
                                self.clip_eigs)

        # All three temporal quantities come from ONE rng draw, so alpha, B_h
        # and z are mutually consistent and really are the GP initialization.
        self.alpha_re = self.param(
            "alpha_re", lambda rng, s: _init(rng)["modal"]["alpha"].real, (self.P,))
        self.alpha_im = self.param(
            "alpha_im", lambda rng, s: _init(rng)["modal"]["alpha"].imag, (self.P,))
        self.z_re = self.param(
            "z_re", lambda rng, s: _init(rng)["modal"]["z"].real, (self.P,))
        self.z_im = self.param(
            "z_im", lambda rng, s: _init(rng)["modal"]["z"].imag, (self.P,))
        self.B = self.param(
            "B", lambda rng, s: np.stack(
                [_init(rng)["modal"]["B_h"].real,
                 _init(rng)["modal"]["B_h"].imag], axis=-1), (local_P, self.H, 2))

        if self.C_init == "trunc_standard_normal":
            C_init = trunc_standard_normal
        elif self.C_init == "lecun_normal":
            C_init = lecun_normal()
        else:
            raise NotImplementedError(self.C_init)
        self.C = self.param("C", lambda rng, shape: init_CV(C_init, rng, shape,
                                                            self.V),
                            (self.H, local_P, 2))
        self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

    def coefficients(self):
        alpha = self.alpha_re + 1j * self.alpha_im
        if self.clip_eigs:
            alpha = np.clip(self.alpha_re, None, ALPHA_CLIP) + 1j * self.alpha_im
        z = self.z_re + 1j * self.z_im
        B_h = self.B[..., 0] + 1j * self.B[..., 1]
        return dict(a_eff=alpha, a_bar=np.exp(alpha),
                    b_hist=B_h, b_bar=phi1(alpha)[:, None] * B_h,
                    d_x=z[:, None] * B_h, m=np.ones_like(alpha))

    def __call__(self, input_sequence, reset_mask=None):
        from .gp_ssm import gp_readout, gp_scan_reset
        c = self.coefficients()
        hs = gp_scan_reset(c["a_bar"], c["b_bar"], input_sequence, reset_mask)
        ys = gp_readout(hs, c["d_x"], self.C_tilde, input_sequence,
                        self.conj_sym)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


def init_modal_ssm(gp_init_scale=0.05, **s5_kwargs):
    return partial(ModalSSM, gp_init_scale=gp_init_scale, **s5_kwargs)
