"""Generalized prospective S5 recurrence on the native S5 parameter path."""

from functools import partial

import jax
import jax.numpy as np
from jax.scipy.linalg import expm

from .ssm import S5SSM


def _block_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, (A_j @ b_i[..., None])[..., 0] + b_j


def generalized_zoh_coefficients(Lambda, B_tilde, step, response, mass):
    """Exact held-input transition for ``M s_ddot+(1-Ta)s_dot-a s=...``."""
    a = Lambda * step
    b = step[..., None] * B_tilde
    A = np.stack([
        np.stack([response * a / mass, 1.0 / mass], axis=-1),
        np.stack([(1.0 - response / mass) * a, -1.0 / mass], axis=-1),
    ], axis=-2)
    B = np.stack([response[:, None] * b / mass[:, None],
                  (1.0 - response / mass)[:, None] * b], axis=-2)
    augmented = np.zeros((A.shape[0], 2 + B.shape[-1], 2 + B.shape[-1]),
                         dtype=A.dtype)
    augmented = augmented.at[:, :2, :2].set(A)
    augmented = augmented.at[:, :2, 2:].set(B)
    exponential = jax.vmap(expm)(augmented)
    return exponential[:, :2, :2], exponential[:, :2, 2:]


class GeneralizedProspectiveS5SSM(S5SSM):
    """Exact-ZOH ``(M,gamma=1,T)`` recurrence with two carried states."""

    response_init: float = 0.3
    mass_init: float = 0.15

    def setup(self):
        super().setup()
        response_raw = np.log(np.expm1(self.response_init)).astype(np.float32)
        mass_raw = np.log(np.expm1(self.mass_init)).astype(np.float32)
        self.generalized_T_raw = self.param(
            "generalized_T_raw",
            lambda rng, shape: np.full(shape, response_raw, dtype=np.float32),
            (self.P,))
        self.generalized_M_raw = self.param(
            "generalized_M_raw",
            lambda rng, shape: np.full(shape, mass_raw, dtype=np.float32),
            (self.P,))
        response = jax.nn.softplus(self.generalized_T_raw)
        mass = jax.nn.softplus(self.generalized_M_raw)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        self.generalized_A_bar, self.generalized_B_bar = generalized_zoh_coefficients(
            self.Lambda, B_tilde, step, response, mass)

    def __call__(self, input_sequence):
        def scan(reverse=False):
            A = np.broadcast_to(self.generalized_A_bar,
                                (input_sequence.shape[0], self.P, 2, 2))
            Bu = jax.vmap(lambda value: self.generalized_B_bar @ value)(
                input_sequence)
            _, states = jax.lax.associative_scan(
                _block_operator, (A, Bu), reverse=reverse)
            return states

        states = scan()
        if self.bidirectional:
            states = np.concatenate((states, scan(reverse=True)), axis=1)
        observed = states[:, :, 0]
        ys = jax.vmap(lambda state: 2 * (self.C_tilde @ state).real)(observed)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


def init_generalized_prospective_S5SSM(response_init=0.3, mass_init=0.15,
                                       **s5_kwargs):
    return partial(GeneralizedProspectiveS5SSM,
                   response_init=response_init, mass_init=mass_init,
                   **s5_kwargs)
