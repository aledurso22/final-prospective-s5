"""One S5 substrate; four recurrent response laws; explicit attribution knobs.

Everything the comparison needs to hold fixed lives in ONE module so that
"same substrate, only the recurrent response law changes" is true in code and
not merely asserted in a report.

Two substrate knobs, both from Rawat et al. Appendix E.3
(arXiv:2609.04134v1), quoted in `docs/RAWAT_BASELINE_MAP.md`:

``input_gain``
    ``"native"``  B_c = B_tilde                       (native S5)
    ``"alpha"``   B_c = diag(alpha) B_tilde, alpha_p = -Re(lambda_p) > 0
                  "the alpha gain remains tied to the physical pole decay rate
                  and is not learned separately"
``clip_eigs``
    the paper's pole clipping, Re(lambda) <= -1e-4, applied BEFORE forming
    alpha. "In the reference alpha-P-S5 configuration, real pole parts are
    clipped to at most -1e-4 before forming alpha, whereas the native one-tap
    S5 configuration does not apply this clipping and uses the unscaled
    learned input matrix."

Because both knobs move together in the published alpha-P-S5 row, gain and
clipping are ATTRIBUTION CONTROLS, not incidental settings: `gain_clip_s5`
below is the gain-scaled, clipped, one-tap arm that isolates the second tap.
The paper states this limitation itself: Tables 2 and 6 "do not isolate the
second tap while holding the continuous-time input map and pole clipping
fixed."

Four response laws
------------------
``one_tap``           h_k = A_bar h_{k-1} + B_bar x_k                  (S5)
``alpha_p_two_tap``   h_k = A_bar h_{k-1} + B_plus x_k + B_minus x_{k-1}
                      B_plus  = B_bar + 5 diag(Delta) A_bar B_c
                      B_minus =        -5 diag(Delta) A_bar B_c
                      delayed input starts at zero; D feedthrough stays
                      instantaneous; no trainable parameters added.
``gp_fixed_m0``       M = 0 member of M s'' + gamma s' + r + T r' = 0
``gp_fixed_mass``     M = rho gamma T > 0 member, (s, v) carry

The GP arms absorb the clock once, a = Delta * lambda and b = Delta * B_c, so
they consume the SAME gain-scaled input map as alpha-P-S5 when
``input_gain="alpha"``. T = 5 native-clock intervals is the same shared horizon
as the paper's tau = 5h; the factor h cancels in both derivations.

No arm adds trainable parameters beyond native S5's. The added physical
coefficients are frozen configuration (`s5.physical_coefficients`), reachable
by no gradient, optimizer state or weight decay.
"""

import jax
import jax.numpy as np

from .gp_coefficients import phi1
from .gp_fixed import (fixed_m0_coefficients, mass_block_zoh, mass_scan,
                       state_counts)
from .gp_ssm import gp_readout, gp_scan_reset
from .physical_coefficients import SYMMETRIC_REFERENCE, PhysicalResponse
from .ssm import S5SSM

INPUT_GAINS = ("native", "alpha")
RESPONSES = ("one_tap", "alpha_p_two_tap", "gp_fixed_m0", "gp_fixed_mass")

#: the paper's clipping bound, Appendix E.3
POLE_CLIP = -1e-4


def input_gain_matrix(Lambda, B_tilde, input_gain):
    """B_c in the native clock."""
    if input_gain == "native":
        return B_tilde
    if input_gain == "alpha":
        alpha = -Lambda.real                       # tied to the poles, not learned
        return alpha[:, None] * B_tilde
    raise ValueError(f"unknown input_gain {input_gain!r}; expected "
                     f"{INPUT_GAINS}")


def two_tap_coefficients(Lambda, B_c, Delta, horizon=5.0):
    """Eq. (E.11): A_bar, B_bar, B_plus, B_minus in the native clock."""
    A_bar = np.exp(Lambda * Delta)
    B_bar = (1.0 / Lambda * (A_bar - 1.0))[:, None] * B_c
    impulse = (horizon * Delta * A_bar)[:, None] * B_c
    return dict(A_bar=A_bar, B_bar=B_bar, B_plus=B_bar + impulse,
                B_minus=-impulse)


def two_tap_drive(B_plus, B_minus, input_sequence):
    """B_plus x_k + B_minus x_{k-1}, with x_{-1} = 0 (delayed input at zero)."""
    cur = jax.vmap(lambda u: B_plus @ u)(input_sequence)
    prev_x = np.concatenate(
        [np.zeros_like(input_sequence[:1]), input_sequence[:-1]], axis=0)
    prev = jax.vmap(lambda u: B_minus @ u)(prev_x)
    return cur + prev


def diagonal_scan_drive(a_bar, drive, reset_mask=None):
    """h_k = a_bar h_{k-1} + drive_k, reusing the original diagonal scan."""
    from .ssm import binary_operator
    L = drive.shape[0]
    A = a_bar * np.ones((L, a_bar.shape[0]))
    if reset_mask is not None:
        keep = (~np.asarray(reset_mask, dtype=bool))[:, None].astype(a_bar.dtype)
        A = A * keep
    _, hs = jax.lax.associative_scan(binary_operator, (A, drive))
    return hs


class SubstrateSSM(S5SSM):
    """S5SSM with a selectable input gain and recurrent response law.

    Adds NO parameters: `input_gain`, `response` and `physical` are static
    configuration. `clip_eigs` is inherited and carries the paper's clipping.
    """

    input_gain: str = "native"
    response: str = "one_tap"
    physical: PhysicalResponse = SYMMETRIC_REFERENCE
    prospective_horizon: float = 5.0

    def setup(self):
        super().setup()
        if self.input_gain not in INPUT_GAINS:
            raise ValueError(f"input_gain must be one of {INPUT_GAINS}")
        if self.response not in RESPONSES:
            raise ValueError(f"response must be one of {RESPONSES}")
        if self.bidirectional:
            raise ValueError("this substrate is causal; bidirectional=False")
        if self.discretization != "zoh":
            raise ValueError("ZOH only; the published construction is ZOH")
        if self.step_rescale != 1.0:
            raise ValueError(
                "step_rescale must be 1.0. The fixed horizon T is expressed in "
                "native-clock intervals; rescaling the clock without rescaling "
                "T silently changes the model.")
        if self.response.startswith("gp_fixed"):
            self.physical.validate()
            if not self.clip_eigs:
                raise ValueError(
                    "the fixed generalized response requires clip_eigs=True: "
                    "sym(J) > 0 is what makes the construction stable, and a "
                    "free unstable pole is outside the supported region.")

    def _native(self):
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        Delta = self.step_rescale * np.exp(self.log_step[:, 0])
        B_c = input_gain_matrix(self.Lambda, B_tilde, self.input_gain)
        return self.Lambda, B_c, Delta

    def coefficients(self):
        """Every realized coefficient, for diagnostics and tests."""
        Lambda, B_c, Delta = self._native()
        if self.response in ("one_tap", "alpha_p_two_tap"):
            return two_tap_coefficients(Lambda, B_c, Delta,
                                        self.prospective_horizon)
        a, b = Lambda * Delta, Delta[:, None] * B_c      # clock absorbed once
        if self.response == "gp_fixed_m0":
            return fixed_m0_coefficients(a, b, self.physical.T,
                                         self.physical.gamma)
        return mass_block_zoh(a, b, self.physical.T, self.physical.gamma,
                              self.physical.rho)

    def state_counts(self):
        """Executed carry sizes in REAL coordinates, derived from the law."""
        c = state_counts(self.P, self.conj_sym,
                         "gp_fixed_mass" if self.response == "gp_fixed_mass"
                         else "other")
        if self.response == "alpha_p_two_tap":
            c = dict(c, previous_input_buffer=self.H,
                     total_real=c["total_real"] + self.H)
        return c

    def __call__(self, input_sequence, reset_mask=None):
        c = self.coefficients()
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)

        if self.response == "gp_fixed_mass":
            zs = mass_scan(c["A_bar"], c["B_bar"], input_sequence,
                           reset_mask=reset_mask)
            s = zs[..., 0]
            if self.conj_sym:
                ys = jax.vmap(lambda si: 2 * (self.C_tilde @ si).real)(s)
            else:
                ys = jax.vmap(lambda si: (self.C_tilde @ si).real)(s)
            return ys + Du

        if self.response == "gp_fixed_m0":
            hs = gp_scan_reset(c["a_bar"], c["b_bar"], input_sequence,
                               reset_mask)
            ys = gp_readout(hs, c["d_x"], self.C_tilde, input_sequence,
                            self.conj_sym)
            return ys + Du

        if self.response == "alpha_p_two_tap":
            drive = two_tap_drive(c["B_plus"], c["B_minus"], input_sequence)
        else:
            drive = jax.vmap(lambda u: c["B_bar"] @ u)(input_sequence)
        hs = diagonal_scan_drive(c["A_bar"], drive, reset_mask)
        if self.conj_sym:
            ys = jax.vmap(lambda h: 2 * (self.C_tilde @ h).real)(hs)
        else:
            ys = jax.vmap(lambda h: (self.C_tilde @ h).real)(hs)
        return ys + Du


#: The five arms of the benchmark comparison, as (input_gain, clip_eigs,
#: response). Names are stable identifiers used by the manifest and the runner.
ARMS = {
    # complete published recipes
    "native_s5":     dict(input_gain="native", clip_eigs=False,
                          response="one_tap"),
    "alpha_p_s5":    dict(input_gain="alpha", clip_eigs=True,
                          response="alpha_p_two_tap"),
    # attribution control: gain + clipping, WITHOUT either prospective law
    "gain_clip_s5":  dict(input_gain="alpha", clip_eigs=True,
                          response="one_tap"),
    # the new candidates, on the same gain-scaled clipped substrate
    "gp_fixed_m0":   dict(input_gain="alpha", clip_eigs=True,
                          response="gp_fixed_m0"),
    "gp_fixed_mass": dict(input_gain="alpha", clip_eigs=True,
                          response="gp_fixed_mass"),
}


def init_substrate_ssm(arm, physical=SYMMETRIC_REFERENCE, **s5_kwargs):
    """Factory mirroring `init_S5SSM`; `arm` is a key of `ARMS`."""
    from functools import partial
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    cfg = dict(ARMS[arm])
    s5_kwargs.pop("clip_eigs", None)          # the arm definition owns it
    return partial(SubstrateSSM, physical=physical, **cfg, **s5_kwargs)
