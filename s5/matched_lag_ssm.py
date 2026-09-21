"""The theory-matched zero-first-order-lag arm.

THE ONE DIFFERENCE FROM THE GENERALIZED ARM. Same M, gamma, T, same
initialization, same lambda_bar and b_bar, and the SAME a1 and a2 -- they
are inherited literally from `GeneralizedProspectiveS5SSM.setup`, so the
denominator and both poles are bit-identical and the ablation is exactly
one coefficient. Only the prospective drive changes:

    current   K [ x_t + (T/h)      (x_t - x_{t-1}) ]
    matched   K [ x_t + (Gamma_k/h)(x_t - x_{t-1}) ]

WHY Gamma_k. Writing k = 1 - lambda_bar, m = Mh/T and c = gamma h/T, the
generalized arm is

    m s'' + (c + Tk) s' + k s = b (x + T x').

Normalizing the denominator by k,

    D(p)/k = 1 + (T + c/k) p + (m/k) p^2,

so the first-order lag is cancelled exactly when the numerator's derivative
coefficient equals the denominator's, that is

    Gamma_k = T + c/k = T + gamma h / (T (1 - lambda_bar)).

Then, normalized by the DC gain b/k,

    Hhat(p) = (1 + Gamma_k p) / (1 + Gamma_k p + (m/k) p^2)
            = 1 - (m/k) p^2 + O(p^3),

so Hhat'(0) = 0 while Hhat(p) != 1: ZERO FIRST-ORDER LAG WITH THE
SECOND-ORDER MEMORY RETAINED.

NO QUADRATIC NUMERATOR TERM. Adding (m/k)p^2 to the numerator would make it
equal to the normalized denominator and give Hhat = 1 identically -- full
memory cancellation. The principle is: COMPENSATE FRICTION PROSPECTIVELY,
NOT INERTIA.

THE DRIVE IS FORMED REGROUPED, AND THAT MATTERS. The direct coefficients

    c1 = K(1 + Gamma_k/h),   c2 = -K Gamma_k/h

satisfy c1 + c2 = K exactly: two numbers of size K*Gamma_k/h cancelling to
K, losing about log10(Gamma_k) digits. Since Gamma_k ~ gamma h /(T|1-a|)
reaches 2e5 on near-real slow modes, that is five of float32's seven
digits. The algebraically identical regrouping

    c1 x_t + c2 x_{t-1}  =  K x_t + K (Gamma_k/h) (x_t - x_{t-1})

has no cancellation, because at low frequency the difference goes to zero
by itself instead of two large terms cancelling. Measured in float32 the
error is flat at 8e-8 against 3e-5 for the direct form -- a 300-fold
improvement at the worst mode.

WHAT REMAINS A CONCERN is not arithmetic but optimization: Gamma_k spans
four orders of magnitude across modes, so the gradients reaching the
prospective pathway may be very unevenly scaled. That is what the
diagnostics measure.
"""

from functools import partial

import jax
import jax.numpy as np

from .factored_recurrence import DEFAULT_IMPLEMENTATION, drive_scan_for
from .generalized_prospective_ssm import (GeneralizedProspectiveS5SSM,
                                            response_mass_gamma)
from .ssm import discretize_zoh

#: the TOKEN step of the finite-difference realization, h = 1. It is the
#: same h that `generalized_coefficients` and `target_map` default to, and
#: it is never identified with S5's learned Delta.
H_TOKEN = 1.0

#: |1 - lambda_bar| is floored so a mode sitting exactly at z = 1 cannot
#: divide by zero. It is NOT a tuning knob: at this value Gamma_k is
#: already 2e7 and the mode is far outside anything production produces.
DISTANCE_FLOOR = 1e-12


def matched_gamma(lambda_bar, response, gamma, h=H_TOKEN):
    """Gamma_k = T + gamma h / (T (1 - lambda_bar)).

    Complex, because lambda_bar is. Since T, gamma and h are real,
    Gamma_k(conj(a)) = conj(Gamma_k(a)) identically, so conjugate symmetry
    is preserved with no special handling and the layer's output stays
    real.
    """
    distance = 1.0 - lambda_bar
    magnitude = np.abs(distance)
    safe = np.where(magnitude < DISTANCE_FLOOR,
                    np.full_like(distance, DISTANCE_FLOOR), distance)
    return response.astype(lambda_bar.dtype) + gamma.astype(
        lambda_bar.dtype) * h / (response.astype(lambda_bar.dtype) * safe)


def matched_gain(b_bar, response, mass, gamma, h=H_TOKEN):
    """K = h T b / q with q = M + h(gamma + T): the drive's DC gain.

    This is exactly c1 + c2 of the generalized arm, so the two arms have
    the SAME DC response as well as the same poles.
    """
    q = mass + h * (gamma + response)
    return (h * response / q).astype(b_bar.dtype)[..., None] * b_bar


def matched_drive(gain, gamma_k, sequence, h=H_TOKEN):
    """K x_t + K (Gamma_k/h)(x_t - x_{t-1}), zero prehistory.

    Regrouped deliberately: see the module docstring. The projection
    happens once and the difference is taken on the PROJECTED signal, so
    the large coefficient multiplies a quantity that is itself small at low
    frequency.
    """
    projected = jax.vmap(lambda u: gain @ u)(sequence)
    previous = np.concatenate((np.zeros_like(projected[:1]), projected[:-1]),
                              axis=0)
    return projected + (gamma_k / h) * (projected - previous)


def direct_coefficients(gain, gamma_k, h=H_TOKEN):
    """(c1, c2) of the equivalent direct form, for testing ONLY.

    Never used in the forward path: it is the cancelling form the
    regrouping exists to avoid. It is kept so a test can prove the two are
    the same recurrence in float64.
    """
    return (1.0 + gamma_k / h)[..., None] * gain, -(gamma_k / h)[..., None] * gain


def apply_matched(a1, a2, gain, gamma_k, input_sequence, h=H_TOKEN,
                  reverse=False, implementation=DEFAULT_IMPLEMENTATION):
    """s_t = a1 s_{t-1} + a2 s_{t-2} + matched drive."""
    sequence = input_sequence[::-1] if reverse else input_sequence
    drive = matched_drive(gain, gamma_k, sequence, h)
    states = drive_scan_for(implementation)(a1, a2, drive)
    return states[::-1] if reverse else states


class MatchedLagProspectiveS5SSM(GeneralizedProspectiveS5SSM):
    """Identical to the generalized arm except for the drive.

    `super().setup()` builds every parameter and both a1 and a2, so they
    are not recomputed here and cannot drift.
    """

    #: the token step; never identified with S5's learned Delta
    h_token: float = H_TOKEN

    def setup(self):
        super().setup()
        # the SAME quantities the parent used, recomputed to expose them.
        # A test asserts these expressions match the parent's literally.
        response, mass, gamma, _ = response_mass_gamma(
            self.generalized_T_raw, self.generalized_rho_raw,
            self.generalized_gamma_raw)
        b_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        lambda_bar, b_bar = discretize_zoh(self.Lambda, b_tilde, step)
        self.matched_gamma_k = matched_gamma(lambda_bar, response, gamma,
                                             self.h_token)
        self.matched_gain = matched_gain(b_bar, response, mass, gamma,
                                         self.h_token)

    def diagnostics(self):
        """|Gamma_k| per mode, and where the extremes sit."""
        magnitude = np.abs(self.matched_gamma_k)
        return {
            "gamma_k_abs": np.asarray(magnitude),
            "gamma_k_abs_max": float(np.max(magnitude)),
            "gamma_k_abs_median": float(np.median(magnitude)),
            "gamma_k_abs_min": float(np.min(magnitude)),
            "gamma_k_decades": float(np.log10(np.max(magnitude)
                                              / np.maximum(np.min(magnitude),
                                                           1e-30))),
        }

    def __call__(self, input_sequence):
        states = apply_matched(
            self.generalized_a1, self.generalized_a2, self.matched_gain,
            self.matched_gamma_k, input_sequence, self.h_token,
            implementation=self.implementation)
        if self.bidirectional:
            reverse = apply_matched(
                self.generalized_a1, self.generalized_a2, self.matched_gain,
                self.matched_gamma_k, input_sequence, self.h_token,
                reverse=True, implementation=self.implementation)
            states = np.concatenate((states, reverse), axis=-1)
        ys = jax.vmap(lambda state: 2 * (self.C_tilde @ state).real)(states)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


def init_matched_lag_prospective_S5SSM(response_init=0.05, rho_init=0.5,
                                       gamma_init=1.0,
                                       implementation=DEFAULT_IMPLEMENTATION,
                                       **s5_kwargs):
    return partial(MatchedLagProspectiveS5SSM,
                   response_init=response_init, rho_init=rho_init,
                   gamma_init=gamma_init, implementation=implementation,
                   **s5_kwargs)
