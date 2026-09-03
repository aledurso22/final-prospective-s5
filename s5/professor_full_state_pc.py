"""PROFESSOR IDEA 1 - literal full-state prospective coding inside S5.

This is the professor's original proposal implemented verbatim. It is NOT the
readout lead on `prospective-lead`, and it is NOT the projected/gated Idea 2.

Continuous mechanism:

    tau ds/dt = -s + f(s,x) + tau d/dt f(s,x),        f(s,x) = A s + B x

Mapping to S5's own parameterization. S5 writes the continuous plant as

    dx/dt = Lambda x + B_tilde u

so matching against  tau ds/dt = (A - I) s + B x  gives

    Lambda = (A - I)/tau,      B_tilde = B/tau,
    K = (I - A)^-1 B = -Lambda^-1 B_tilde       (equilibrium gain, tau-free)

Substituting f and collecting ds/dt (see docs/PROFESSOR_MECHANISM.md):

    tau ds/dt = -s + K u + tau K du/dt                                  (*)

Two consequences, both of which this file makes literally true in S5:

1. every pole becomes -1/tau. The learned Lambda no longer sets the
   homogeneous dynamics at all.
2. with s = K u + r, (*) gives tau dr/dt = -r, so from a zero initial state
   the SSM output is EXACTLY s_k = K u_k: the state-space model collapses to
   an instantaneous, memoryless linear map. The scan below still computes it
   the general way (it is needed for a non-zero initial state and it keeps the
   realization faithful), but the analytic prediction is that it is
   feedforward.

Discretization is the theory-derived first-order residual form, the same one
validated in `tests/test_mechanism.py`. With rho_i = exp(-step_i / tau):

    s_k = rho s_{k-1} + K u_k - rho K u_{k-1}

There is NO finite-difference derivative and therefore no parasitic
second-order root.

`s5/ssm.py` is untouched: `binary_operator` and `jax.lax.associative_scan` are
imported from it and used unchanged, so the recurrence stays scan-parallel.
"""

import jax
import jax.numpy as np

from .ssm import S5SSM, binary_operator


def prospective_gain(Lambda, B_tilde):
    """K = -Lambda^-1 B_tilde, the continuous equilibrium gain (I-A)^-1 B."""
    return -B_tilde / Lambda[:, None]


def apply_full_state_pc(rho, K, C_tilde, input_sequence, conj_sym):
    """s_k = rho s_{k-1} + K u_k - rho K u_{k-1}, then the ordinary readout.

    The forcing term is built with a vectorized causal shift, never a second
    scan and never `jnp.roll` (which would wrap the last token into the first).
    """
    Ku = jax.vmap(lambda u: K @ u)(input_sequence)              # (L, P)
    Ku_prev = np.concatenate((np.zeros_like(Ku[:1]), Ku[:-1]), axis=0)
    forcing = Ku - rho[None, :] * Ku_prev

    rho_elements = rho[None, :] * np.ones((input_sequence.shape[0],
                                           rho.shape[0]))
    _, xs = jax.lax.associative_scan(binary_operator, (rho_elements, forcing))

    if conj_sym:
        return jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


class FullStatePCSSM(S5SSM):
    """S5 with professor Idea 1 substituted for the state recurrence.

    Subclasses S5SSM so every parameter (Lambda_re, Lambda_im, B, C, D,
    log_step) is initialized identically - the parameter tree and count are
    unchanged from plain S5, which is what makes the trained comparison fair.
    Only `__call__` differs.
    """

    prospective_tau: float = 1.0

    def __call__(self, input_sequence):
        if self.bidirectional:
            raise ValueError(
                "professor Idea 1 (full state PC) is causal; it requires a "
                "unidirectional S5. Set bidirectional=False.")
        if self.prospective_tau <= 0.0:
            raise ValueError("prospective_tau must be positive")

        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        K = prospective_gain(self.Lambda, B_tilde)

        step = self.step_rescale * np.exp(self.log_step[:, 0])
        rho = np.exp(-step / self.prospective_tau).astype(self.Lambda.dtype)

        ys = apply_full_state_pc(rho, K, self.C_tilde, input_sequence,
                                 self.conj_sym)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


def init_full_state_pc_ssm(prospective_tau=1.0, **s5_kwargs):
    """Mirror of `init_S5SSM` returning the Idea-1 SSM instead."""
    from functools import partial
    return partial(FullStatePCSSM, prospective_tau=prospective_tau, **s5_kwargs)
