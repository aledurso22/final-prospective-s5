"""Zucchet prospective dynamics — finite-difference realization."""

from functools import partial
import jax
import jax.numpy as np

from .discrete_recurrence import scan_companion, zucchet_coefficients
from .ssm import S5SSM, discretize_zoh


class ProspectiveS5SSM(S5SSM):
    response_init: float = 0.05

    def setup(self):
        super().setup()
        raw = np.log(np.expm1(self.response_init)).astype(np.float32)
        self.prospective_T_raw = self.param(
            "prospective_T_raw", lambda rng, shape: np.full(shape, raw), (self.P,))
        response = jax.nn.softplus(self.prospective_T_raw)
        b_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        lambda_bar, b_bar = discretize_zoh(self.Lambda, b_tilde, step)
        self.prospective_a1, self.prospective_a2, self.prospective_c1, self.prospective_c2 = (
            zucchet_coefficients(lambda_bar, b_bar, response))

    def __call__(self, input_sequence):
        states = scan_companion(self.prospective_a1, self.prospective_a2,
                                self.prospective_c1, self.prospective_c2,
                                input_sequence)
        if self.bidirectional:
            reverse = scan_companion(self.prospective_a1, self.prospective_a2,
                                     self.prospective_c1, self.prospective_c2,
                                     input_sequence, reverse=True)
            states = np.concatenate((states, reverse), axis=-1)
        ys = jax.vmap(lambda state: 2 * (self.C_tilde @ state).real)(states)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


def init_prospective_S5SSM(response_init=0.05, **s5_kwargs):
    return partial(ProspectiveS5SSM, response_init=response_init, **s5_kwargs)
