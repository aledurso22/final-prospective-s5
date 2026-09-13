"""Generalized prospective SSM for S5.

`s5/ssm.py` is NOT modified: `binary_operator` and the diagonal associative
scan are imported and reused unchanged, so the trusted reference stays byte
identical and easy to diff against.

Mechanisms
----------
plain              handled by the factory, which returns the original S5SSM.
gp_scalar          one response coefficient t per layer, shared by all modes.
gp_diagonal        one nonnegative t per stored complex mode.
prospective_input  input-side law sdot = F0 s + B0 x + tau_p B0 xdot. The
                   GENERATOR IS UNCHANGED; only input coupling and feedthrough
                   move. Comparator for the 2026 preprint's intervention.
full_state_pc      fully matched TSS, gamma = 0. Diagnostic negative control:
                   its driven history is exactly zero.

Output timing (adopted from native `apply_ssm`, unchanged)
----------------------------------------------------------
    h_k = a_bar h_{k-1} + b_bar x_k          h_{-1} = 0
    s_k = h_k + D_x x_k                      tied state feedthrough
    y_k = 2 Re{C_tilde s_k} + D_native * x_k       (factor 2 iff conj_sym)

`x_k` is held over the k-th unit interval. The native learned featurewise D is
RETAINED alongside the new tied feedthrough; both are differentiated.

Unsupported configurations are rejected loudly rather than silently changing
the model: non-unit `step_rescale`, bilinear discretization and bidirectional
all raise.
"""

import jax
import jax.numpy as np

from .gp_coefficients import (absorb_clock, gp_response_coefficients,
                              inverse_softplus, matched_tss_coefficients,
                              prospective_input_coefficients,
                              softplus_response)
from .ssm import S5SSM, binary_operator

GP_MECHANISMS = ("plain", "gp_scalar", "gp_diagonal", "prospective_input",
                 "full_state_pc")
#: mechanisms whose response parameter is a single shared scalar
_SCALAR_MECHANISMS = ("gp_scalar", "prospective_input", "full_state_pc")
#: the parameter name, used by the optimizer label rule in train_helpers
GP_PARAM_NAME = "gp_response_raw"


def gp_scan(a_bar, b_bar, input_sequence):
    """h_k = a_bar h_{k-1} + b_bar x_k via the ORIGINAL diagonal scan."""
    Lambda_elements = a_bar * np.ones((input_sequence.shape[0], a_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: b_bar @ u)(input_sequence)
    _, hs = jax.lax.associative_scan(binary_operator,
                                     (Lambda_elements, Bu_elements))
    return hs


def gp_scan_reset(a_bar, b_bar, input_sequence, reset_mask=None):
    """Parallel scan with optional per-interval reset.

    `reset_mask[k] == True` means the carry is dropped BEFORE interval k, so
    h_k = b_bar x_k. Implemented by zeroing the transition factor at those
    positions, which the associative operator handles exactly - no second scan
    and no sequential fallback.
    """
    L = input_sequence.shape[0]
    Lambda_elements = a_bar * np.ones((L, a_bar.shape[0]))
    if reset_mask is not None:
        keep = (~np.asarray(reset_mask, dtype=bool))[:, None].astype(a_bar.dtype)
        Lambda_elements = Lambda_elements * keep
    Bu_elements = jax.vmap(lambda u: b_bar @ u)(input_sequence)
    _, hs = jax.lax.associative_scan(binary_operator,
                                     (Lambda_elements, Bu_elements))
    return hs


def gp_scan_sequential(a_bar, b_bar, input_sequence, h0=None, reset_mask=None):
    """DIFFERENTIABLE sequential recurrence, with carry and resets.

    Uses `lax.scan`, so unlike a NumPy loop it can validate GRADIENTS against
    the parallel path, and it accepts a NON-ZERO initial carry so chunked
    evaluation can be checked.
    """
    L = input_sequence.shape[0]
    h_init = (np.zeros(a_bar.shape[0], dtype=a_bar.dtype) if h0 is None
              else np.asarray(h0, dtype=a_bar.dtype))
    mask = (np.zeros(L, dtype=bool) if reset_mask is None
            else np.asarray(reset_mask, dtype=bool))

    def step(h, inp):
        x_k, r_k = inp
        h_prev = np.where(r_k, np.zeros_like(h), h)
        h_new = a_bar * h_prev + b_bar @ x_k
        return h_new, h_new

    _, hs = jax.lax.scan(step, h_init, (input_sequence, mask))
    return hs


def gp_readout(hs, d_x, C_tilde, input_sequence, conj_sym):
    """s_k = h_k + D_x x_k, then the ordinary conjugate-symmetric readout."""
    s = hs + jax.vmap(lambda u: d_x @ u)(input_sequence)
    if conj_sym:
        return jax.vmap(lambda si: 2 * (C_tilde @ si).real)(s)
    return jax.vmap(lambda si: (C_tilde @ si).real)(s)


def build_coefficients(mechanism, a, b, t):
    """Dispatch to the pure coefficient builders. `t` is already positive."""
    if mechanism == "prospective_input":
        return prospective_input_coefficients(a, b, t[0])
    if mechanism == "full_state_pc":
        return matched_tss_coefficients(a, b, t[0])
    if mechanism in ("gp_scalar", "gp_diagonal"):
        t_full = t if t.shape[0] == a.shape[0] else np.broadcast_to(t, a.shape)
        return gp_response_coefficients(a, b, t_full)
    raise ValueError(f"unknown generalized mechanism: {mechanism}")


class GPSSM(S5SSM):
    """S5SSM plus one generalized prospective response parameter group.

    Subclasses S5SSM so every ordinary parameter (Lambda_re, Lambda_im, B, C,
    D, log_step) is initialized identically; the ONLY addition is
    `gp_response_raw`, which makes the extra parameter count explicit.
    """

    mechanism: str = "gp_diagonal"
    gp_init_scale: float = 0.05

    def setup(self):
        super().setup()
        if self.mechanism not in GP_MECHANISMS or self.mechanism == "plain":
            raise ValueError(
                f"GPSSM does not handle mechanism={self.mechanism!r}; the "
                f"factory returns the original S5SSM for 'plain'.")
        if self.bidirectional:
            raise ValueError(
                "generalized prospective response is causal and supports only "
                "unidirectional S5; set bidirectional=False.")
        if self.discretization != "zoh":
            raise ValueError(
                f"generalized response currently supports ZOH only, got "
                f"{self.discretization!r}. Resolution transfer is a separate, "
                f"untested protocol.")
        if self.step_rescale != 1.0:
            raise ValueError(
                f"step_rescale must be 1.0 for the generalized mechanism (got "
                f"{self.step_rescale}). Sample-clock T would have to rescale "
                f"as 1/c; that protocol is not implemented or tested.")
        if not self.clip_eigs:
            raise ValueError(
                "generalized prospective response requires clip_eigs=True.\n"
                "The stability identity Re(a_eff) = (sigma - t|a|^2)/|m|^2 < 0 "
                "is CONDITIONAL on Re(a) < 0. A stable initialization does not "
                "constrain later optimizer updates: a CONSTRUCTED admissible-API "
                "configuration with an unstable raw pole reaches "
                "|a_bar| = 1.0035 > 1. (That was a constructed probe, not an "
                "observed training run.) The historical "
                "plain default (clip_eigs=False) is deliberately unchanged; "
                "for a matched treatment/control comparison set clip_eigs=True "
                "on BOTH arms.")
        if self.gp_init_scale <= 0.0:
            raise ValueError(
                "gp_init_scale must be strictly positive. softplus is a "
                "bijection onto (0, inf), so no finite raw value gives t = 0 "
                "exactly and inverse_softplus(0) is not finite; the target "
                "scale must therefore be positive. (softplus'(0) = 1/2, so "
                "raw = 0 is a perfectly good tangent - the zero-tangent "
                "problem belongs to a SQUARED parameterization, not to this "
                "one.) Use mechanism='plain' for the exact identity baseline.")

        n_response = 1 if self.mechanism in _SCALAR_MECHANISMS else self.P
        raw0 = inverse_softplus(self.gp_init_scale)  # host-side float
        self.gp_response_raw = self.param(
            GP_PARAM_NAME,
            lambda rng, shape: np.full(shape, raw0, dtype=np.float32),
            (n_response,))

    def response(self):
        """The positive response coefficient(s) t, in sample-clock units."""
        return softplus_response(self.gp_response_raw)

    def coefficients(self, input_dtype=None):
        """All realized coefficients, for diagnostics and tests."""
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        a, b = absorb_clock(self.Lambda, B_tilde, step)
        return build_coefficients(self.mechanism, a, b, self.response())

    def __call__(self, input_sequence, reset_mask=None):
        c = self.coefficients()
        hs = gp_scan_reset(c["a_bar"], c["b_bar"], input_sequence, reset_mask)
        ys = gp_readout(hs, c["d_x"], self.C_tilde, input_sequence,
                        self.conj_sym)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


def init_gp_ssm(mechanism="gp_diagonal", gp_init_scale=0.05, **s5_kwargs):
    """Factory mirroring `init_S5SSM`.

    For `mechanism='plain'` this returns the ORIGINAL S5SSM partial with no
    extra parameters and no change to random-key consumption, which guarantees
    the intended baseline identity by construction.
    """
    from functools import partial

    from .ssm import init_S5SSM
    if mechanism == "plain":
        return init_S5SSM(**s5_kwargs)
    if mechanism not in GP_MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected one of "
                         f"{GP_MECHANISMS}")
    return partial(GPSSM, mechanism=mechanism, gp_init_scale=gp_init_scale,
                   **s5_kwargs)
