"""Generalized prospective dynamics (M,gamma,T) — finite-difference realization."""

from functools import partial
import jax
import jax.numpy as np

from .discrete_recurrence import (_block_operator, companion_radius,
                                  generalized_coefficients,
                                  scan_companion_sequential)
from .factored_recurrence import DEFAULT_IMPLEMENTATION, scan_for
from .ssm import S5SSM, discretize_zoh

RHO_MIN = 1e-4


def response_mass_gamma(T_raw, rho_raw, gamma_raw):
    T = jax.nn.softplus(T_raw)
    rho = RHO_MIN + (1.0 - RHO_MIN) * jax.nn.sigmoid(rho_raw)
    gamma = jax.nn.softplus(gamma_raw)
    return T, rho * gamma * T, gamma, rho


class GeneralizedProspectiveS5SSM(S5SSM):
    response_init: float = 0.05
    rho_init: float = 0.5
    gamma_init: float = 1.0
    #: WHICH SCAN evaluates the recurrence. The equation, the
    #: parameterization and the coefficients are identical for every
    #: choice; only the evaluation order differs. "sequential" is the
    #: default and the oracle, so the production path is unchanged unless
    #: a caller asks for something else by name.
    implementation: str = DEFAULT_IMPLEMENTATION

    def setup(self):
        super().setup()
        t_raw = np.log(np.expm1(self.response_init)).astype(np.float32)
        rho_raw = np.log(self.rho_init / (1.0 - self.rho_init)).astype(np.float32)
        gamma_raw = np.log(np.expm1(self.gamma_init)).astype(np.float32)
        self.generalized_T_raw = self.param(
            "generalized_T_raw", lambda rng, shape: np.full(shape, t_raw), (self.P,))
        self.generalized_rho_raw = self.param(
            "generalized_rho_raw", lambda rng, shape: np.full(shape, rho_raw), (self.P,))
        self.generalized_gamma_raw = self.param(
            "generalized_gamma_raw", lambda rng, shape: np.full(shape, gamma_raw), (self.P,))
        T, mass, gamma, _ = response_mass_gamma(
            self.generalized_T_raw, self.generalized_rho_raw,
            self.generalized_gamma_raw)
        b_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        lambda_bar, b_bar = discretize_zoh(self.Lambda, b_tilde, step)
        self.generalized_a1, self.generalized_a2, self.generalized_c1, self.generalized_c2 = (
            generalized_coefficients(lambda_bar, b_bar, T, mass, gamma))

    def __call__(self, input_sequence):
        scan = scan_for(self.implementation)
        states = scan(
            self.generalized_a1, self.generalized_a2,
            self.generalized_c1, self.generalized_c2, input_sequence)
        if self.bidirectional:
            reverse = scan(
                self.generalized_a1, self.generalized_a2,
                self.generalized_c1, self.generalized_c2,
                input_sequence, reverse=True)
            states = np.concatenate((states, reverse), axis=-1)
        ys = jax.vmap(lambda state: 2 * (self.C_tilde @ state).real)(states)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


def init_generalized_prospective_S5SSM(response_init=0.05, rho_init=0.5,
                                       gamma_init=1.0,
                                       implementation=DEFAULT_IMPLEMENTATION,
                                       **s5_kwargs):
    return partial(GeneralizedProspectiveS5SSM,
                   response_init=response_init, rho_init=rho_init,
                   gamma_init=gamma_init, implementation=implementation,
                   **s5_kwargs)


__all__ = ["RHO_MIN", "_block_operator", "companion_radius",
           "response_mass_gamma", "GeneralizedProspectiveS5SSM",
           "init_generalized_prospective_S5SSM",
           "DEFAULT_IMPLEMENTATION", "scan_for"]
