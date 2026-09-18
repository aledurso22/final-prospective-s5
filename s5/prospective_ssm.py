"""Zucchet prospective S5 recurrence on the native S5 parameter path."""

from functools import partial

import jax
import jax.numpy as np

from .ssm import S5SSM, binary_operator


def _clocked_coefficients(Lambda, B_tilde, step, response):
    a = Lambda * step
    b = step[..., None] * B_tilde
    denominator = 1.0 - response.astype(a.dtype) * a
    a_eff = a / denominator
    b_hist = b / denominator[:, None] ** 2
    d_x = response[:, None].astype(a.dtype) * b / denominator[:, None]
    a_bar = np.exp(a_eff)
    b_bar = (np.expm1(a_eff) / a_eff)[:, None] * b_hist
    return a_bar, b_bar, d_x


class ProspectiveS5SSM(S5SSM):
    """Exact-ZOH realization of ``(I-T a)s_dot=a s+b x+T b x_dot``."""

    response_init: float = 0.05

    def setup(self):
        super().setup()
        raw = np.log(np.expm1(self.response_init)).astype(np.float32)
        self.prospective_T_raw = self.param(
            "prospective_T_raw",
            lambda rng, shape: np.full(shape, raw, dtype=np.float32),
            (self.P,))
        response = jax.nn.softplus(self.prospective_T_raw)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        self.prospective_a_bar, self.prospective_b_bar, self.prospective_d_x = (
            _clocked_coefficients(self.Lambda, B_tilde, step, response))

    def __call__(self, input_sequence):
        def scan(reverse=False):
            A = self.prospective_a_bar * np.ones(
                (input_sequence.shape[0], self.P))
            Bu = jax.vmap(lambda value: self.prospective_b_bar @ value)(
                input_sequence)
            _, states = jax.lax.associative_scan(
                binary_operator, (A, Bu), reverse=reverse)
            return states

        states = scan()
        d_x = self.prospective_d_x
        if self.bidirectional:
            states = np.concatenate((states, scan(reverse=True)), axis=-1)
            d_x = np.concatenate((d_x, d_x), axis=0)
        observed = states + jax.vmap(lambda value: d_x @ value)(input_sequence)
        ys = jax.vmap(lambda state: 2 * (self.C_tilde @ state).real)(observed)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


def init_prospective_S5SSM(response_init=0.05, **s5_kwargs):
    return partial(ProspectiveS5SSM, response_init=response_init, **s5_kwargs)
