"""WWJ prospective S5 arms: critical, learned-passive, and a diagnostic.

These are NEW arms. They do not replace and do not relabel the older
two-compartment generalized arm in `s5/generalized_prospective_ssm.py`, which
solves a different equation, is kept frozen, and whose partial results stay
its own. `s5/ssm.py` (Native S5) is untouched.

Parameter granularity is DECLARED and deliberately different from the older
arms: one `tau` (and, for the passive arm, one `eps`) per S5 LAYER, broadcast
over that layer's modes, rather than one per mode. Per-mode timescales are a
separate experiment, not a silent default.

Stable parameterization:

    tau = TAU_MIN + softplus(wwj_tau_raw) > 0     so q = M + h*tau >= h*TAU_MIN
    eps = EPS_MAX * sigmoid(wwj_eps_raw) in (0, 1/4)    (passive arm)
    eps = 1/4 exactly                                   (critical arm)
    M   = eps * tau^2

The eigenvalues of the recurrence are NEVER projected or clipped after an
update. If such a projection is ever needed it must be defined mathematically
and run as its own declared intervention.
"""

from functools import partial

import jax
import jax.numpy as np

from .ssm import S5SSM, discretize_zoh
from .wwj_recurrence import (EPS_CRITICAL, EPS_MAX, H_TOKEN, TAU_MIN,
                             companion_spectral_radius, continuous_response,
                             mass_from_eps, state_coefficients, wwj_states)


def tau_from_raw(raw):
    """tau > 0, floored so q = M + h*tau cannot approach zero."""
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
    """Shared WWJ machinery; subclasses only supply `eps`."""

    tau_init: float = 0.25
    h_token: float = H_TOKEN
    remat_scan: bool = True

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
        """One scalar per layer, broadcast to this layer's modes."""
        return tau_from_raw(self.wwj_tau_raw) * np.ones(self.P)

    @property
    def eps(self):
        raise NotImplementedError

    @property
    def mass(self):
        return mass_from_eps(self.tau, self.eps)

    def coefficients(self):
        """((A0, A1, A2), (C0, C1, C2)) of this layer, for diagnostics."""
        return state_coefficients(self.lambda_bar, self.b_bar, self.tau,
                                  self.mass, self.h_token)

    def diagnostics(self):
        """Per-layer WWJ diagnostics; never part of the forward computation."""
        (A0, A1, A2), _ = self.coefficients()
        return {"tau": self.tau, "eps": self.eps, "mass": self.mass,
                "companion_spectral_radius":
                    companion_spectral_radius(A0, A1, A2),
                "continuous_response":
                    continuous_response(self.Lambda, self.tau, self.mass)}

    def __call__(self, input_sequence):
        states = wwj_states(self.lambda_bar, self.b_bar, self.tau, self.mass,
                            input_sequence, reverse=False, h=self.h_token,
                            remat=self.remat_scan)
        if self.bidirectional:
            # the reverse branch sees the REVERSED sequence and is flipped
            # back; no forward-history shift touches a concatenated state
            reverse = wwj_states(self.lambda_bar, self.b_bar, self.tau,
                                 self.mass, input_sequence, reverse=True,
                                 h=self.h_token, remat=self.remat_scan)
            states = np.concatenate((states, reverse), axis=-1)
        ys = jax.vmap(lambda state: 2 * (self.C_tilde @ state).real)(states)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


class WWJCriticalS5SSM(_WWJBase):
    """M = tau^2/4: the critically damped WWJ recurrence. eps is not learned."""

    def setup(self):
        super().setup()
        self._setup_wwj()

    @property
    def eps(self):
        return np.asarray(EPS_CRITICAL, dtype=np.float32) * np.ones(self.P)


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
        return eps_passive_from_raw(self.wwj_eps_raw) * np.ones(self.P)


class WWJUnconstrainedS5SSM(_WWJBase):
    """DIAGNOSTIC ONLY: eps unbounded above, so P(D) may be underdamped.

    Not the biological model and not a principal arm. It exists to measure
    what the passive bound costs, and any result from it must be reported as
    a diagnostic.
    """

    eps_init: float = 0.0625

    def setup(self):
        super().setup()
        self._setup_wwj()
        raw = np.asarray(float(np.log(np.expm1(self.eps_init))),
                         dtype=np.float32)
        self.wwj_eps_raw = self.param("wwj_eps_raw",
                                      lambda rng, shape: np.full(shape, raw),
                                      ())

    @property
    def eps(self):
        return jax.nn.softplus(self.wwj_eps_raw) * np.ones(self.P)


def init_wwj_critical_S5SSM(tau_init=0.25, **s5_kwargs):
    return partial(WWJCriticalS5SSM, tau_init=tau_init, **s5_kwargs)


def init_wwj_passive_S5SSM(tau_init=0.25, eps_init=0.0625, **s5_kwargs):
    return partial(WWJPassiveS5SSM, tau_init=tau_init, eps_init=eps_init,
                   **s5_kwargs)


def init_wwj_unconstrained_S5SSM(tau_init=0.25, eps_init=0.0625, **s5_kwargs):
    return partial(WWJUnconstrainedS5SSM, tau_init=tau_init,
                   eps_init=eps_init, **s5_kwargs)


#: the WWJ constructors, kept out of `s5/three_arm_factory.py` so that the
#: three frozen production arms and their factory stay byte-identical
WWJ_ARMS = {
    "wwj_critical_s5": init_wwj_critical_S5SSM,
    "wwj_passive_s5": init_wwj_passive_S5SSM,
    "wwj_unconstrained_s5_diagnostic": init_wwj_unconstrained_S5SSM,
}

WWJ_SCIENTIFIC_NAMES = {
    "wwj_critical_s5":
        "WWJ prospective dynamics, critical M = tau^2/4 — mixed-stencil "
        "causal realization",
    "wwj_passive_s5":
        "WWJ prospective dynamics, learned passive M = eps*tau^2, "
        "0 < eps < 1/4 — mixed-stencil causal realization",
    "wwj_unconstrained_s5_diagnostic":
        "WWJ prospective dynamics, unconstrained eps (DIAGNOSTIC ONLY) — "
        "mixed-stencil causal realization",
}
