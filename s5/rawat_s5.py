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

import math

import jax
import jax.numpy as np

from .gp_coefficients import phi1
from .gp_fixed import (PROSPECTIVE_INPUT_HORIZON, fixed_m0_coefficients,
                       mass_block_two_tap, mass_block_zoh, mass_scan,
                       mass_scan_two_tap, state_counts)
from .gp_ssm import gp_readout, gp_scan_reset
from .physical_coefficients import SYMMETRIC_REFERENCE, PhysicalResponse
from .ssm import S5SSM

INPUT_GAINS = ("native", "alpha")
RESPONSES = ("one_tap", "alpha_p_two_tap", "gp_fixed_m0", "gp_fixed_mass",
             "gp_learned_response", "prospective_recurrence",
             # current parameterization: gamma_n removed as redundant with the
             # learned clock, rho the only learned response quantity
             "gp_rho", "gp_rho_frozen",
             # Rawat's prospective INPUT correction composed with the
             # generalized prospective RECURRENCE (combined-model study)
             "gp_rho_prospin",
             # learned per-mode response TIMESCALE T_i alongside rho_i; the
             # `_fixed` variant stores eta but never updates it
             "gp_rho_T", "gp_rho_T_fixed",
             # superseded by the constrained-response brief of 2026-09-15;
             # RETAINED and still selectable, but not part of that batch.
             "gp_adaptive_mass", "gp_frozen_adaptive", "ordinary_adaptive")

#: the two constrained learned response leaves of the SUPERSEDED
#: gp_learned_response arm; retained so its executed run stays reproducible
RESPONSE_PARAM_NAMES = ("log_response_gamma", "log_response_rho")

#: The CURRENT response leaf. gamma_n is gone: (Delta, gamma_n, rho) is
#: input-output equivalent to (Delta/gamma_n, 1, rho) for every admissible rho,
#: so gamma_n only re-parameterized the already-learned clock. With gamma_n = 1
#: and hat_Delta = Delta/gamma_n absorbed into log_step the law is
#:
#:     rho T s'' + s' + r + T r' = 0 ,  r = -hat_Delta lambda s - hat_Delta B_c x
#:
#: and the mass is derived, mu = T rho. One real leaf per stored complex mode,
#: shared with its conjugate partner.
RHO_ONLY_PARAM_NAME = "log_response_rho_only"
#: every response that carries the single rho leaf. Adding a member here is the
#: ONLY place a new rho-only arm has to be registered for parameter creation,
#: coefficient dispatch and the trainers' projection whitelist.
RHO_ONLY_RESPONSES = ("gp_rho", "gp_rho_frozen", "gp_rho_prospin",
                      "gp_rho_T", "gp_rho_T_fixed")

#: The learned RESPONSE TIMESCALE leaf, one per stored complex mode, shared
#: with its conjugate partner. Parameterized multiplicatively about the
#: reference horizon:
#:
#:     T_i = T_REFERENCE * exp(eta_i),   eta_i in LOG_T_BOUNDS
#:
#: so eta = 0 is exactly the reference T = 5 and the two generalized arms start
#: from the SAME function. The mass stays derived, mu_i = rho_i T_i; there is
#: no independent mass parameter and gamma_n stays 1.
#:
#: NOTE, recorded rather than hidden: AdamW's decoupled decay shrinks eta
#: toward 0, which biases T toward its REFERENCE value 5, not toward 0. Decay
#: in a log coordinate is not coordinate invariant, and the direction it pulls
#: is a property of this parameterization.
T_ONLY_PARAM_NAME = "log_response_T_only"
T_RESPONSES = ("gp_rho_T", "gp_rho_T_fixed")
T_REFERENCE = 5.0
#: DECLARED NUMERICAL GUARDRAILS for this bounded study, in sample intervals.
#: T in [0.05, 500], i.e. eta in [log(0.01), log(100)]. These are numerical
#: bounds for a bounded experiment; they are NOT physiological limits and were
#: NOT selected from accuracy. A guardrail that fails a numerical check is
#: reported and amended with numerical justification, never from a score.
T_BOUNDS = (T_REFERENCE * 1e-2, T_REFERENCE * 1e2)
LOG_T_BOUNDS = (math.log(1e-2), math.log(1e2))
#: Declared initialization for the learned-timescale study: the earlier
#: SYMMETRIC reference, not the near-one recall/composition initialization.
#: At rho = 1 the state transfer is b/(p + j), INDEPENDENT of T, so
#: d G/d T = b(1-rho)p^2 / [...]^2 vanishes there: starting at rho_0 = 0.9998
#: would begin this parameter-freedom test almost at an unidentifiable limit.
RHO_INIT_TIMESCALE = 0.75
#: responses whose carry is the (s, v) block and therefore accept a z0 carry
_BLOCK_CARRY_RESPONSES = ("gp_fixed_mass", "gp_learned_response", "gp_rho",
                          "gp_rho_frozen", "gp_rho_prospin",
                          "gp_rho_T", "gp_rho_T_fixed")
#: declared initialization for the recall study; a declared choice, not a
#: physiological measurement
RHO_INIT_RECALL = 0.9998
#: Declared numerical admissibility bounds.
#:
#: AMENDED 2026-09-15 from a MEASUREMENT, before any training and before any
#: validation score. The first declaration allowed rho down to 1e-4, which with
#: gamma_n = 1e-2 gives a derived mass mu = T gamma_n rho = 5e-6. At that corner
#: the block matrix exponential is NOT finite. Measured frontier, identical in
#: float32 and float64, so it is stiffness and not precision
#: (experiments/gp/constrained_numerics_probe.py):
#:
#:     mu      499.9  50   5    0.5   0.05   5e-3   5e-4*  5e-5   5e-6
#:     finite  yes    yes  yes  yes   yes    yes    yes    yes    NO
#:
#: The smallest mu that worked was 5e-5; the largest that failed was 5e-6. The
#: rho lower bound is therefore raised to 1e-2, which makes the WORST corner of
#: the box mu = 5 * 1e-2 * 1e-2 = 5e-4, ten times the smallest measured-good mu
#: and a hundred times the measured failure. gamma_n is unchanged: its extreme
#: corner mu = 499.9 was measured finite.
#:
#: This is a numerical bound set from measurement, on the same principle as the
#: second-order prototype's MU_RATIO_MIN. It is NOT a physiological claim and
#: it was NOT chosen to improve a score. It still leaves mu free over more than
#: three decades below its initial value of 3.75.
GAMMA_N_BOUNDS = (1e-2, 1e2)
RHO_BOUNDS = (1e-2, 1.0 - 1e-4)
#: measured, for the record and for the tests
MEASURED_SMALLEST_FINITE_MU = 5e-5
MEASURED_LARGEST_NONFINITE_MU = 5e-6
LOG_GAMMA_BOUNDS = (math.log(GAMMA_N_BOUNDS[0]), math.log(GAMMA_N_BOUNDS[1]))
LOG_RHO_BOUNDS = (math.log(RHO_BOUNDS[0]), math.log(RHO_BOUNDS[1]))

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
    #: initialization for the gp_rho / gp_rho_frozen response
    rho_init: float = RHO_INIT_RECALL

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
        if self.response.startswith(("gp_fixed", "gp_adaptive", "gp_frozen",
                                     "gp_rho")) \
                or self.response == "prospective_recurrence":
            self.physical.validate()
            if not self.clip_eigs:
                raise ValueError(
                    "the generalized and prospective-recurrence responses "
                    "require clip_eigs=True: sym(J) > 0 is what makes the "
                    "construction stable, and a free unstable pole is outside "
                    "the supported region. For the prospective recurrence it "
                    "also keeps J invertible, which its exact realization "
                    "s = J^-1 b x needs.")
        if self.response.startswith(("gp_adaptive", "gp_frozen")):
            from .adaptive_circuit import validate_reference
            validate_reference()
        if self.response in RHO_ONLY_RESPONSES:
            # Declared LAST, after super().setup(), so the common parameter
            # draw is bit-identical to the ordinary arm under the same key.
            self.physical.validate()
            self.log_response_rho_only = self.param(
                RHO_ONLY_PARAM_NAME,
                lambda rng, shape: np.full(shape, math.log(self.rho_init),
                                           dtype=np.float32),
                (self.P,))
        if self.response in T_RESPONSES:
            # Declared AFTER rho, so both generalized arms draw an identical
            # tree in an identical order. eta_0 = 0 exactly, computed on the
            # host, so T = T_REFERENCE exactly and the fixed and learned arms
            # start from the SAME forward function.
            self.log_response_T_only = self.param(
                T_ONLY_PARAM_NAME,
                lambda rng, shape: np.zeros(shape, dtype=np.float32),
                (self.P,))
        if self.response == "gp_learned_response":
            # Declared LAST, after super().setup() has drawn every ordinary
            # parameter, so the common parameter tree is bit-identical to the
            # fixed arm under the same key. That is asserted by a test, not
            # assumed.
            self.physical.validate()                 # host-side floats only
            self.log_response_gamma = self.param(
                "log_response_gamma",
                lambda rng, shape: np.zeros(shape, dtype=np.float32),
                (self.P,))                           # gamma_n = exp(0) = 1
            self.log_response_rho = self.param(
                "log_response_rho",
                lambda rng, shape: np.full(shape, math.log(0.75),
                                           dtype=np.float32),
                (self.P,))                           # rho = 0.75

    def _native(self):
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        Delta = self.step_rescale * np.exp(self.log_step[:, 0])
        B_c = input_gain_matrix(self.Lambda, B_tilde, self.input_gain)
        return self.Lambda, B_c, Delta

    def response_timescale(self):
        """The learned per-mode horizon T_i, clipped before exponentiation.

        Returned as a (P,) array so every downstream user - coefficients,
        derived mass, diagnostics - reads the EXECUTED value rather than the
        static `physical.T` dataclass field.
        """
        eta = np.clip(self.log_response_T_only, *LOG_T_BOUNDS)
        return T_REFERENCE * np.exp(eta)

    def rho_only(self):
        """The single learned response quantity, clipped before exponentiation.

        gamma_n is fixed at 1 by construction; the mass is DERIVED, mu = T rho.
        """
        zeta = np.clip(self.log_response_rho_only, *LOG_RHO_BOUNDS)
        return np.exp(zeta)

    def learned_response(self):
        """(gamma_n, rho) per stored complex mode, both strictly positive.

        gamma_n = exp(eta), rho = exp(zeta), with the raw logs CLIPPED before
        exponentiation as a forward safety guard. The horizon T stays FIXED at
        the published-comparison value and the mass is DERIVED,
        mu = T gamma_n rho; neither is a separate learned quantity.

        The conjugate partner of a stored mode shares these scalars, because
        they are properties of one physical compartment pair.
        """
        eta = np.clip(self.log_response_gamma, *LOG_GAMMA_BOUNDS)
        zeta = np.clip(self.log_response_rho, *LOG_RHO_BOUNDS)
        return np.exp(eta), np.exp(zeta)

    def derived_mass(self):
        """mu = T gamma_n rho, derived and never learned independently."""
        if self.response in T_RESPONSES:
            # gamma_n = 1, and T is the EXECUTED learned horizon
            return self.response_timescale() * self.rho_only()
        if self.response in ("gp_rho", "gp_rho_frozen", "gp_rho_prospin"):
            return self.physical.T * self.rho_only()
        g_n, rho = self.learned_response()
        return self.physical.T * g_n * rho

    def clock_absorbed(self):
        """(a, b) with the native clock absorbed EXACTLY once."""
        Lambda, B_c, Delta = self._native()
        return Lambda * Delta, Delta[:, None] * B_c

    def coefficients(self):
        """Every realized coefficient, for diagnostics and tests."""
        Lambda, B_c, Delta = self._native()
        if self.response in ("one_tap", "alpha_p_two_tap"):
            return two_tap_coefficients(Lambda, B_c, Delta,
                                        self.prospective_horizon)
        a, b = self.clock_absorbed()
        if self.response == "gp_fixed_m0":
            return fixed_m0_coefficients(a, b, self.physical.T,
                                         self.physical.gamma)
        if self.response == "gp_fixed_mass":
            return mass_block_zoh(a, b, self.physical.T, self.physical.gamma,
                                  self.physical.rho)
        if self.response in RHO_ONLY_RESPONSES:
            rho = self.rho_only()
            if self.response == "gp_rho_frozen":
                rho = jax.lax.stop_gradient(rho)
            # gamma_n = 1: the clock coordinate already carries hat_Delta
            if self.response in T_RESPONSES:
                # per-mode horizon. `mass_block_generator` already shapes a
                # (P,) denominator to (P,1) before dividing the (P,H) input
                # row, so this is correct for P != H as well as P == H.
                T = self.response_timescale()
                d = mass_block_zoh(a, b, T, np.ones_like(rho), rho)
                return dict(d, executed_T=T, executed_rho=rho,
                            derived_mu=T * rho,
                            # dimensionless per-mode product T_i j_i, j = -a
                            T_times_j=T.astype(a.dtype) * (-a))
            if self.response == "gp_rho_prospin":
                # the SAME recurrence, driven by x + T_in x_dot. The extra
                # input correction COMPOSES with the recurrence's own
                # prospective residual; it does not replace it.
                return mass_block_two_tap(a, b, self.physical.T,
                                          np.ones_like(rho), rho,
                                          self.prospective_horizon)
            return mass_block_zoh(a, b, self.physical.T,
                                  np.ones_like(rho), rho)
        if self.response == "gp_learned_response":
            g_n, rho = self.learned_response()
            return mass_block_zoh(a, b, self.physical.T, g_n, rho)
        if self.response == "prospective_recurrence":
            # r + T r_dot = 0 with r = J s - b x, J = -diag(a). For invertible
            # J the driven realization is EXACT and memoryless:
            #     s_k = J^-1 b x_k = -(Delta B_c)/(Delta lambda) = -B_c/lambda
            # Formed DIRECTLY from B_c and lambda, so Delta cancels
            # STRUCTURALLY rather than numerically. log_step is then unused by
            # this arm and its exact data-loss gradient with respect to
            # log_step is ZERO - a property of the model, not a defect. Exact
            # diagonal division, never a backward difference, which would add a
            # spurious recurrent root.
            return dict(s_gain=-B_c / Lambda[:, None], a=a, b=b)
        if self.response == "ordinary_adaptive":
            # matched ordinary adaptive control: the SAME modulation
            # information rescales the ordinary mode's time constant by
            # tau_d(q)/tau_d(0) = G_d0/G_d(q); no prospective term, one state.
            return dict(a=a, b=b)
        # gp_adaptive_mass / gp_frozen_adaptive are token dependent, so their
        # coefficients are formed inside __call__ where the input is available.
        return dict(a=a, b=b, adaptive=True, frozen=(
            self.response == "gp_frozen_adaptive"))

    def state_counts(self):
        """Executed carry sizes in REAL coordinates, derived from the law."""
        two_state = self.response in ("gp_fixed_mass", "gp_learned_response",
                                      "gp_rho", "gp_rho_frozen",
                                      "gp_rho_prospin",
                                      "gp_rho_T", "gp_rho_T_fixed",
                                      "gp_adaptive_mass", "gp_frozen_adaptive")
        c = state_counts(self.P, self.conj_sym,
                         "gp_fixed_mass" if two_state else "other")
        if self.response == "prospective_recurrence":
            c = dict(c, physical_real=0, total_real=0,
                     note="memoryless by construction: s_k = J^-1 b x_k")
        if self.response in ("alpha_p_two_tap", "gp_rho_prospin"):
            # the second tap carries the PREVIOUS token, which is real carried
            # state even though it costs no parameters
            c = dict(c, previous_input_buffer=self.H,
                     total_real=c["total_real"] + self.H)
        return c

    def _readout(self, s, Du):
        if self.conj_sym:
            ys = jax.vmap(lambda si: 2 * (self.C_tilde @ si).real)(s)
        else:
            ys = jax.vmap(lambda si: (self.C_tilde @ si).real)(s)
        return ys + Du

    def __call__(self, input_sequence, reset_mask=None, z0=None, prev_x=None):
        """`z0` / `prev_x` are the STREAMING carries, used by chunked calls.

        Both default to zero prehistory, the declared convention. Only the
        two-tap block response consumes `prev_x`; passing it to a response with
        no second tap is an error rather than a silent no-op.
        """
        if prev_x is not None and self.response != "gp_rho_prospin":
            raise ValueError(
                f"response {self.response!r} has no delayed-input tap, so a "
                f"prev_x carry would be silently discarded")
        if z0 is not None and self.response not in _BLOCK_CARRY_RESPONSES:
            raise ValueError(
                f"response {self.response!r} does not accept a block carry z0; "
                f"refusing to discard it silently")
        c = self.coefficients()
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)

        if self.response == "prospective_recurrence":
            # zero driven history beyond lag zero, by construction
            s = jax.vmap(lambda u: c["s_gain"] @ u)(input_sequence)
            return self._readout(s, Du)

        if self.response == "ordinary_adaptive":
            from .adaptive_circuit import (conductance_from_input,
                                           coefficients_from_conductance)
            from .gp_coefficients import phi1
            a, b = c["a"], c["b"]
            q = conductance_from_input(b, input_sequence)          # (L,P)
            cc = coefficients_from_conductance(q)
            scale = (cc["G_d"] / adaptive_G_D0())                  # >= 1
            a_k = a[None, :] * scale                               # (L,P)
            a_bar = np.exp(a_k)
            b_bar = phi1(a_k)[..., None] * b[None, :, :]           # (L,P,H)
            drive = np.einsum("lph,lh->lp", b_bar, input_sequence)
            hs = time_varying_diagonal_scan(a_bar, drive, reset_mask)
            return self._readout(hs, Du)

        if self.response in ("gp_adaptive_mass", "gp_frozen_adaptive"):
            from .adaptive_circuit import (adaptive_generator, adaptive_scan,
                                           adaptive_zoh,
                                           conductance_from_input)
            a, b = c["a"], c["b"]
            L = input_sequence.shape[0]
            if self.response == "gp_frozen_adaptive":
                q = np.zeros((L, a.shape[0]))
            else:
                q = conductance_from_input(b, input_sequence)
            A, Bx, d_jump, _ = adaptive_generator(a, b, q)
            A_bar, B_bar = adaptive_zoh(A, Bx)
            zs = adaptive_scan(A_bar, B_bar, d_jump, input_sequence,
                               reset_mask=reset_mask)
            return self._readout(zs[..., 0], Du)

        if self.response == "gp_rho_prospin":
            # Exact two-tap interval law. `prev_x` and the block carry are BOTH
            # reset together: clearing only the block state would leak the
            # previous sequence's last token through the second tap.
            zs = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"],
                                   input_sequence, z0=z0, prev_x=prev_x,
                                   reset_mask=reset_mask)
            return self._readout(zs[..., 0], Du)

        if self.response in ("gp_fixed_mass", "gp_learned_response",
                             "gp_rho", "gp_rho_frozen",
                             "gp_rho_T", "gp_rho_T_fixed"):
            zs = mass_scan(c["A_bar"], c["B_bar"], input_sequence, z0=z0,
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


def adaptive_G_D0():
    from .adaptive_circuit import G_D0
    return G_D0


def time_varying_diagonal_scan(a_bar, drive, reset_mask=None):
    """h_k = a_bar_k h_{k-1} + drive_k with a PER-TOKEN diagonal factor."""
    from .ssm import binary_operator
    A = a_bar
    if reset_mask is not None:
        keep = (~np.asarray(reset_mask, dtype=bool))[:, None].astype(A.dtype)
        A = A * keep
    _, hs = jax.lax.associative_scan(binary_operator, (A, drive))
    return hs


#: The arms of the comparison, as (input_gain, clip_eigs, response). Names are
#: stable identifiers used by the manifest and the runner.
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
    # --- constrained learned response: the current treatment
    "gp_learned_response": dict(input_gain="alpha", clip_eigs=True,
                                response="gp_learned_response"),
    # --- current parameterization, used by the recall study
    "gp_rho": dict(input_gain="alpha", clip_eigs=True, response="gp_rho"),
    "gp_rho_frozen": dict(input_gain="alpha", clip_eigs=True,
                          response="gp_rho_frozen"),
    # --- combined model: Rawat's prospective INPUT + the generalized
    #     prospective RECURRENCE, on the same gain-scaled clipped substrate
    "gp_rho_prospin": dict(input_gain="alpha", clip_eigs=True,
                           response="gp_rho_prospin"),
    # --- learned response TIMESCALE study. rho_0 = 0.75 (the symmetric
    #     reference), NOT the near-one recall initialization, because
    #     dG/dT vanishes at rho = 1 and the test would start unidentifiable.
    "gp_rho_T": dict(input_gain="alpha", clip_eigs=True, response="gp_rho_T",
                     rho_init=RHO_INIT_TIMESCALE),
    "gp_rho_T_fixed": dict(input_gain="alpha", clip_eigs=True,
                           response="gp_rho_T_fixed",
                           rho_init=RHO_INIT_TIMESCALE),
    # --- RETAINED but superseded: within-sequence adaptation arms. Kept
    #     selectable and tested; NOT part of the constrained-response batch.
    "gp_adaptive_mass": dict(input_gain="alpha", clip_eigs=True,
                             response="gp_adaptive_mass"),
    # frozen version of the NEW model, its own within-family control: the old
    # gp_fixed_mass result cannot substitute for it unless they are identical,
    # which is a test, not an assumption.
    "gp_frozen_adaptive": dict(input_gain="alpha", clip_eigs=True,
                               response="gp_frozen_adaptive"),
    # the professor equation, r + T r_dot = 0, as an exact cancellation control
    "prospective_recurrence": dict(input_gain="alpha", clip_eigs=True,
                                   response="prospective_recurrence"),
    # ordinary adaptive SSM using the SAME modulation information
    "ordinary_adaptive": dict(input_gain="alpha", clip_eigs=True,
                              response="ordinary_adaptive"),
}


def init_substrate_ssm(arm, physical=SYMMETRIC_REFERENCE, **s5_kwargs):
    """Factory mirroring `init_S5SSM`; `arm` is a key of `ARMS`."""
    from functools import partial
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    cfg = dict(ARMS[arm])
    s5_kwargs.pop("clip_eigs", None)          # the arm definition owns it
    return partial(SubstrateSSM, physical=physical, **cfg, **s5_kwargs)
