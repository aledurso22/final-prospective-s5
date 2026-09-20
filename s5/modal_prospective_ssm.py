"""The heterogeneous modal prospective S5 layer.

Native S5 memory is untouched: the layer runs `s5/ssm.py`'s own scan for
q_{j,t+1} = lambda_bar_j q_{j,t} + b_bar_j x_t, then applies a PER-MODE
cascade of first-order stages whose poles are d/(h+d) in (0, 1) by
construction, and reads out as Native does.

Per mode j and stage l the learned quantities are

    d_lj  = D_MIN + softplus(raw)            the stage pole parameter
    g_lj  = sigmoid(gate_raw) in (0, 1)      the regime gate
    n_lj  = d_lj + g_lj * softplus(delta)    the numerator parameter

so g = 0 makes the stage an exact identity. With both gates off the layer IS
Native S5; with one on it is ordinary prospectivity; with both on it is the
generalized WWJ numerator 1 + Gamma_n D + M_n D^2 over stable poles.

Gates are per MODE, deliberately: the point of the construction is that
different modes may choose different regimes. A gate penalty keeps the
model from switching mechanics on everywhere for free, and
`native_mode_gain` is reported and penalized as it approaches zero so a
cancelled native mode cannot pass silently.
"""

from functools import partial

import jax
import jax.numpy as np

from .modal_prospective import (D_MIN, GENERALIZED, H_TOKEN, NATIVE, ORDINARY,
                                apply_cascade, gamma_and_mass,
                                native_mode_gain, native_states,
                                numerator_from_gate, stage_pole)
from .ssm import S5SSM, discretize_zoh


def d_from_raw(raw):
    """d > 0, so the stage pole d/(h+d) is strictly inside (0, 1)."""
    return D_MIN + jax.nn.softplus(raw)


def gate_from_raw(raw):
    """g in (0, 1); g = 0 is the exact Native identity."""
    return jax.nn.sigmoid(raw)


class ModalProspectiveS5SSM(S5SSM):
    """Native S5 plus a two-stage per-mode cascade."""

    stages: int = 2
    h_token: float = H_TOKEN
    #: initial gate logit; strongly negative starts every mode at Native
    gate_init: float = -4.0
    d_init: float = 1.0
    delta_init: float = 0.5
    #: weight of the gate penalty and of the cancellation penalty
    gate_penalty: float = 1e-3
    cancellation_penalty: float = 1e-2
    #: |L(lambda_bar)| below this is treated as cancelling a native mode
    cancellation_floor: float = 0.05

    def setup(self):
        super().setup()
        shape = (self.P,)
        raw_d = float(np.log(np.expm1(self.d_init)))
        raw_delta = float(np.log(np.expm1(self.delta_init)))
        self.stage_d_raw = [
            self.param(f"stage{index}_d_raw",
                       lambda rng, s, value=raw_d: np.full(s, value), shape)
            for index in range(self.stages)]
        self.stage_delta_raw = [
            self.param(f"stage{index}_delta_raw",
                       lambda rng, s, value=raw_delta: np.full(s, value),
                       shape)
            for index in range(self.stages)]
        self.stage_gate_raw = [
            self.param(f"stage{index}_gate_raw",
                       lambda rng, s, value=self.gate_init:
                       np.full(s, value), shape)
            for index in range(self.stages)]
        b_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        self.lambda_bar, self.b_bar = discretize_zoh(self.Lambda, b_tilde,
                                                     step)

    # ------------------------------------------------------- parameters --
    @property
    def stage_parameters(self):
        """[(d, n)] per stage, in the working complex dtype."""
        out = []
        for index in range(self.stages):
            d = d_from_raw(self.stage_d_raw[index])
            gate = gate_from_raw(self.stage_gate_raw[index])
            n = numerator_from_gate(d, gate, self.stage_delta_raw[index])
            out.append((d.astype(self.lambda_bar.dtype),
                        n.astype(self.lambda_bar.dtype)))
        return out

    @property
    def gates(self):
        return [gate_from_raw(raw) for raw in self.stage_gate_raw]

    def penalty(self):
        """Gate cost plus a cancellation cost, both reported per layer.

        The gate cost is the mean gate: switching prospective mechanics on
        is not free. The cancellation cost grows as |L(lambda_bar)| falls
        below the declared floor, so annihilating a native mode is penalized
        rather than silently permitted.
        """
        gate_cost = sum(np.mean(gate) for gate in self.gates)
        gain = native_mode_gain(self.lambda_bar, self.stage_parameters,
                                self.h_token)
        shortfall = np.maximum(self.cancellation_floor - gain, 0.0)
        return (self.gate_penalty * gate_cost
                + self.cancellation_penalty * np.mean(shortfall ** 2))

    def diagnostics(self):
        """Per-mode gates, poles, zeros, Gamma_n, M_n and native-mode gain."""
        parameters = self.stage_parameters
        gates = self.gates
        gain = native_mode_gain(self.lambda_bar, parameters, self.h_token)
        gamma, mass = gamma_and_mass(parameters[0][1], parameters[1][1]) \
            if self.stages >= 2 else (parameters[0][1],
                                      np.zeros_like(parameters[0][1]))
        return {
            "gates": [np.asarray(gate) for gate in gates],
            "mean_gate": [float(np.mean(gate)) for gate in gates],
            "stage_poles": [np.asarray(stage_pole(d.real, self.h_token))
                            for d, _ in parameters],
            "max_stage_pole": float(np.max(np.stack(
                [stage_pole(d.real, self.h_token) for d, _ in parameters]))),
            "gamma_numerator": np.asarray(gamma.real),
            "mass_numerator": np.asarray(mass.real),
            "native_mode_gain_min": float(np.min(gain)),
            "native_mode_gain_mean": float(np.mean(gain)),
            "modes_below_cancellation_floor":
                int(np.sum(gain < self.cancellation_floor)),
            "regime_per_mode": self.regime_per_mode(),
        }

    def regime_per_mode(self, threshold=0.05):
        """Which regime each mode has selected, as counts."""
        first, second = (np.asarray(gate) > threshold for gate in self.gates) \
            if self.stages >= 2 else (np.asarray(self.gates[0]) > threshold,
                                      np.zeros((self.P,), dtype=bool))
        return {NATIVE: int(np.sum(~first & ~second)),
                ORDINARY: int(np.sum(first ^ second)),
                GENERALIZED: int(np.sum(first & second))}

    # ---------------------------------------------------------- forward --
    def _one_direction(self, input_sequence, reverse):
        states = native_states(self.lambda_bar, self.b_bar, input_sequence,
                               reverse=reverse)
        return apply_cascade(states, self.stage_parameters, self.h_token,
                             reverse=reverse)

    def __call__(self, input_sequence):
        states = self._one_direction(input_sequence, reverse=False)
        if self.bidirectional:
            # the reverse branch uses the Native suffix scan and its cascade
            # runs in that direction's own causal order
            states = np.concatenate(
                (states, self._one_direction(input_sequence, reverse=True)),
                axis=-1)
        if self.conj_sym:
            ys = jax.vmap(lambda value: 2 * (self.C_tilde @ value).real)(states)
        else:
            ys = jax.vmap(lambda value: (self.C_tilde @ value).real)(states)
        return ys + jax.vmap(lambda value: self.D * value)(input_sequence)


def init_modal_prospective_S5SSM(stages=2, gate_init=-4.0, d_init=1.0,
                                 delta_init=0.5, **s5_kwargs):
    return partial(ModalProspectiveS5SSM, stages=stages, gate_init=gate_init,
                   d_init=d_init, delta_init=delta_init, **s5_kwargs)


MODAL_ARMS = {"modal_prospective_s5": init_modal_prospective_S5SSM}
MODAL_SCIENTIFIC_NAMES = {
    "modal_prospective_s5":
        "heterogeneous many-body WWJ-inspired modal prospective cascade "
        "(learned per-mode gates; explicit model construction)",
}
