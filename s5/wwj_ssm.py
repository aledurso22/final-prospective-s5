"""The principal WWJ-S5 arms: Native memory, exact discrete WWJ readout.

Each layer runs the unmodified Native S5 recurrence and reads out the WWJ
generalized-prospective coordinate

    z_t = P(D_h) s_t = (1 + k + m) s_t - (k + 2m) s_{t-1} + m s_{t-2},
    y_t = C z_t + D x_t,

with k = tau/h, m = eps*k^2, h = 1 token. The recurrent poles are exactly the
Native poles plus two zero history-shift poles: WWJ adds zeros, not poles.
`s5/ssm.py` is untouched and is imported, not reimplemented.

Arms:

  wwj_critical_s5   eps = 1/4 fixed (M = tau^2/4)
  wwj_passive_s5    eps = (1/4) sigmoid(raw), learned inside (0, 1/4)
  wwj_gated_recoverable_s5_diagnostic
                    s~ = s + g(z - s), a SEPARATE intervention: the direct
                    arm only at g = 1, Native S5 only at g = 0. Diagnostic.

The rejected mixed-stencil realization lives in `s5/wwj_mixed_stencil.py` and
`s5/wwj_mixed_stencil_ssm.py` under the identifier
`wwj_mixed_stencil_unstable_diagnostic`. It is not a principal arm and never
enters a production launcher.

Parameterization: tau = TAU_MIN + softplus(raw) > 0, eps = (1/4)sigmoid(raw),
one tau and one eps per LAYER (declared, not silently per mode). No
eigenvalue is ever projected after an update -- and with an FIR operator
there is nothing to project.
"""

from functools import partial

import jax
import jax.numpy as np

from .ssm import S5SSM, discretize_zoh
from .wwj_operator import (EPS_CRITICAL, EPS_MAX, H_TOKEN, TAU_MIN, k_and_m,
                           gated_states, max_fir_gain, native_radii,
                           three_tap, wwj_states)


def tau_from_raw(raw):
    """tau > 0, floored so the prospective gradients cannot vanish."""
    return TAU_MIN + jax.nn.softplus(raw)


def eps_passive_from_raw(raw):
    """eps in (0, 1/4): the passive, non-oscillatory range of P(D)."""
    return EPS_MAX * jax.nn.sigmoid(raw)


def raw_from_tau(tau):
    tau = float(tau)
    if tau <= TAU_MIN:
        raise ValueError(f"tau_init must exceed TAU_MIN={TAU_MIN}: {tau}")
    return float(np.log(np.expm1(tau - TAU_MIN)))


def raw_from_eps(eps):
    eps = float(eps)
    if not 0.0 < eps < EPS_MAX:
        raise ValueError(f"eps_init must lie in (0, {EPS_MAX}): {eps}")
    ratio = eps / EPS_MAX
    return float(np.log(ratio / (1.0 - ratio)))


class _WWJBase(S5SSM):
    """Native scan plus the WWJ readout; subclasses supply `eps`."""

    tau_init: float = 0.05
    h_token: float = H_TOKEN

    def _setup_wwj(self):
        raw = np.asarray(raw_from_tau(self.tau_init), dtype=np.float32)
        self.wwj_tau_raw = self.param("wwj_tau_raw",
                                      lambda rng, shape: np.full(shape, raw),
                                      ())
        b_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        self.lambda_bar, self.b_bar = discretize_zoh(self.Lambda, b_tilde, step)

    @property
    def tau(self):
        """One scalar per layer."""
        return tau_from_raw(self.wwj_tau_raw)

    @property
    def eps(self):
        raise NotImplementedError

    @property
    def k_m(self):
        return k_and_m(self.tau, self.eps, self.h_token)

    def _project(self, states):
        """The layer's readout, conjugate symmetry handled as in `apply_ssm`."""
        if self.conj_sym:
            return jax.vmap(lambda z: 2 * (self.C_tilde @ z).real)(states)
        return jax.vmap(lambda z: (self.C_tilde @ z).real)(states)

    def _coordinate(self, states, k, m):
        """The WWJ coordinate of one direction. Overridden by the gated arm."""
        return three_tap(states, k, m)

    def diagnostics(self):
        """FIR diagnostics; the operator has no poles to report."""
        k, m = self.k_m
        return {"tau": self.tau, "eps": self.eps,
                "mass": self.eps * self.tau ** 2, "k": k, "m": m,
                "max_fir_gain": max_fir_gain(k, m),
                "native_radius_max": np.max(native_radii(self.lambda_bar)),
                "native_radius_min": np.min(native_radii(self.lambda_bar))}

    def __call__(self, input_sequence):
        k, m = self.k_m
        states = wwj_states(self.lambda_bar, self.b_bar, input_sequence, k, m)
        if self.bidirectional:
            # the reverse branch uses the Native suffix scan and applies the
            # operator in ITS causal order, then is concatenated
            reverse = wwj_states(self.lambda_bar, self.b_bar, input_sequence,
                                 k, m, reverse=True)
            states = np.concatenate((states, reverse), axis=-1)
        return (self._project(states)
                + jax.vmap(lambda value: self.D * value)(input_sequence))


class WWJCriticalS5SSM(_WWJBase):
    """M = tau^2/4: the critically damped WWJ operator. eps is not learned."""

    def setup(self):
        super().setup()
        self._setup_wwj()

    @property
    def eps(self):
        return np.asarray(EPS_CRITICAL, dtype=np.float32)


class WWJPassiveS5SSM(_WWJBase):
    """M = eps*tau^2 with eps learned inside the passive range (0, 1/4)."""

    eps_init: float = 0.0625

    def setup(self):
        super().setup()
        self._setup_wwj()
        raw = np.asarray(raw_from_eps(self.eps_init), dtype=np.float32)
        self.wwj_eps_raw = self.param("wwj_eps_raw",
                                      lambda rng, shape: np.full(shape, raw),
                                      ())

    @property
    def eps(self):
        return eps_passive_from_raw(self.wwj_eps_raw)


class WWJGatedRecoverableS5SSM(WWJPassiveS5SSM):
    """DIAGNOSTIC: s~ = s + g(z - s), with g learned in (0, 1).

    A separate intervention, NOT the direct arm: it coincides with it only at
    g = 1 and with Native S5 only at g = 0. Reported under its own name.
    """

    gate_init: float = 0.5

    def setup(self):
        super().setup()
        raw = np.asarray(float(np.log(self.gate_init
                                      / (1.0 - self.gate_init))),
                         dtype=np.float32)
        self.wwj_gate_raw = self.param("wwj_gate_raw",
                                       lambda rng, shape: np.full(shape, raw),
                                       ())

    @property
    def gate(self):
        return jax.nn.sigmoid(self.wwj_gate_raw)

    def _coordinate(self, states, k, m):
        return gated_states(states, k, m, self.gate)

    def __call__(self, input_sequence):
        k, m = self.k_m
        from .wwj_operator import native_states
        forward = self._coordinate(
            native_states(self.lambda_bar, self.b_bar, input_sequence), k, m)
        if self.bidirectional:
            backward = native_states(self.lambda_bar, self.b_bar,
                                     input_sequence, reverse=True)
            backward = self._coordinate(backward[::-1], k, m)[::-1]
            forward = np.concatenate((forward, backward), axis=-1)
        return (self._project(forward)
                + jax.vmap(lambda value: self.D * value)(input_sequence))


def init_wwj_critical_S5SSM(tau_init=0.05, **s5_kwargs):
    return partial(WWJCriticalS5SSM, tau_init=tau_init, **s5_kwargs)


def init_wwj_passive_S5SSM(tau_init=0.05, eps_init=0.0625, **s5_kwargs):
    return partial(WWJPassiveS5SSM, tau_init=tau_init, eps_init=eps_init,
                   **s5_kwargs)


def init_wwj_gated_S5SSM(tau_init=0.05, eps_init=0.0625, gate_init=0.5,
                         **s5_kwargs):
    return partial(WWJGatedRecoverableS5SSM, tau_init=tau_init,
                   eps_init=eps_init, gate_init=gate_init, **s5_kwargs)


#: the PRINCIPAL arms. The rejected mixed-stencil arm is deliberately absent.
WWJ_PRINCIPAL_ARMS = {
    "wwj_critical_s5": init_wwj_critical_S5SSM,
    "wwj_passive_s5": init_wwj_passive_S5SSM,
}
WWJ_DIAGNOSTIC_ARMS = {
    "wwj_gated_recoverable_s5_diagnostic": init_wwj_gated_S5SSM,
}
WWJ_SCIENTIFIC_NAMES = {
    "wwj_critical_s5":
        "WWJ prospective dynamics, critical M = tau^2/4 — exact discrete "
        "operator on the Native S5 trajectory",
    "wwj_passive_s5":
        "WWJ prospective dynamics, learned passive M = eps*tau^2, "
        "0 < eps < 1/4 — exact discrete operator on the Native S5 trajectory",
    "wwj_gated_recoverable_s5_diagnostic":
        "WWJ gated recoverable readout (DIAGNOSTIC; a separate intervention, "
        "not the direct arm)",
}
