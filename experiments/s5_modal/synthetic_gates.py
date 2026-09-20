"""A tiny synthetic task where different modes should choose different gates.

THE TASK. Two channels, and a target that needs two different things at
once:

  * a LONG-DELAY memory component: the target depends on an impulse seen
    `delay` tokens ago, which a slow native mode can carry;
  * a LEAD component: the target depends on the edge of a switch signal,
    which a prospective (numerator) stage produces and a plain low-pass
    mode lags behind.

A cascade whose gates are per mode can satisfy both by opening gates on some
modes and leaving others Native. That is the ONLY claim this experiment
makes: that the construction can DIFFERENTIATE. It is not evidence of
benefit on Speech Commands, and the report says so.

SUCCESS CRITERION, corrected after the first run. The scientific claim needs
BOTH retained long-delay memory AND reduced response lag, so the criterion
tests both components, not the total. The first run improved the total by
1.2% while the long-delay component DEGRADED by 3.9% and the lead component
improved by 13.8% -- a trade, not a win -- and the old criterion would have
called that "beats the baseline". It no longer can.

STAGE SYMMETRY, fixed after the first run. Both stages were initialized
identically, so they received identical gradients and stayed tied to six
decimal places: n1 = n2 for ever, which locks the cascade to the CRITICAL
branch M = (Gamma/2)^2 and makes the general passive branch unreachable.
The stages are now initialized asymmetrically, and n1 versus n2 is reported
so the tie cannot recur unnoticed.

Runs on CPU in seconds. Trains nothing that touches Speech Commands.
"""

import argparse
import json

import jax
import jax.numpy as jnp
import optax

from s5 import modal_prospective as MP

MODES = 8
LENGTH = 256
BATCH = 16
DELAY = 32


def make_batch(key, batch=BATCH, length=LENGTH, delay=DELAY, channels=2):
    """(inputs, target).

    Inputs: channel 0 carries impulses, channel 1 a switch.

    Target with `channels = 2` (the default): channel 0 is the LONG-DELAY
    memory term and channel 1 is the LEAD term, on SEPARATE outputs. The
    single-channel version summed them, so one shared linear readout had to
    serve both demands at once and per-mode specialization could not be
    exploited -- which is why the first probe may have forced a trade rather
    than measured one. `channels = 1` reproduces that original task for
    comparison.
    """
    impulse_key, switch_key = jax.random.split(key)
    positions = jax.random.randint(impulse_key, (batch,), 0, length - delay - 1)
    impulses = jnp.zeros((batch, length)).at[
        jnp.arange(batch), positions].set(1.0)
    switch_times = jax.random.randint(switch_key, (batch,), length // 4,
                                      3 * length // 4)
    ramp = jnp.arange(length)[None, :]
    switch = (ramp >= switch_times[:, None]).astype(jnp.float32)
    inputs = jnp.stack((impulses, switch), axis=-1)
    # the long-delay term: the impulse, `delay` tokens later
    delayed = jnp.zeros((batch, length)).at[
        jnp.arange(batch), positions + delay].set(1.0)
    # the lead term: the switch EDGE, which a lagging filter misses
    edge = jnp.concatenate(
        (switch[:, :1] * 0.0, switch[:, 1:] - switch[:, :-1]), axis=1)
    if channels == 1:
        return inputs, (delayed + edge)[..., None]
    return inputs, jnp.stack((delayed, edge), axis=-1)


def model_apply(params, inputs, use_gates):
    """One modal layer: native scan, per-mode cascade, linear readout."""
    lambda_bar = jnp.clip(params["lambda_re"], -0.9, -1e-3) \
        + 1j * params["lambda_im"]
    lambda_bar = jnp.exp(lambda_bar)
    b_bar = params["b_re"] + 1j * params["b_im"]
    d = MP.D_MIN + jax.nn.softplus(params["d_raw"])
    gates = [jax.nn.sigmoid(raw) if use_gates else jnp.zeros_like(raw)
             for raw in params["gate_raw"]]
    stages = [(d[index].astype(lambda_bar.dtype),
               MP.numerator_from_gate(d[index], gates[index],
                                      params["delta_raw"][index]
                                      ).astype(lambda_bar.dtype))
              for index in range(len(gates))]

    def one(sequence):
        states = MP.native_states(lambda_bar, b_bar, sequence)
        return MP.apply_cascade(states, stages)

    states = jax.vmap(one)(inputs)
    features = jnp.concatenate((states.real, states.imag), axis=-1)
    return (jnp.einsum("blf,fc->blc", features, params["readout"]),
            gates, stages)


def loss_fn(params, inputs, target, use_gates, gate_penalty):
    prediction, gates, _ = model_apply(params, inputs, use_gates)
    error = jnp.mean((prediction - target) ** 2)
    cost = gate_penalty * sum(jnp.mean(gate) for gate in gates)
    return error + cost, error


def initial_params(key, stages=2, channels=2):
    """Asymmetric across stages AND across modes.

    Identical stages receive identical gradients and stay tied for ever,
    which pins n1 = n2 and so M = (Gamma/2)^2 -- the critical branch only.
    Identical modes cannot differentiate either. Both symmetries are broken
    here, by initialization rather than by any change to the mathematics.
    """
    keys = jax.random.split(key, 8)
    stage_offset = jnp.linspace(-0.5, 0.5, stages)[:, None]
    mode_offset = jnp.linspace(-0.3, 0.3, MODES)[None, :]
    return {
        "lambda_re": -0.05 - 0.4 * jax.random.uniform(keys[0], (MODES,)),
        "lambda_im": jax.random.uniform(keys[1], (MODES,), minval=-2.0,
                                        maxval=2.0),
        "b_re": jax.random.normal(keys[2], (MODES, 2)) * 0.5,
        "b_im": jax.random.normal(keys[3], (MODES, 2)) * 0.5,
        "d_raw": stage_offset + mode_offset,
        "delta_raw": 0.5 + stage_offset
                     + 0.1 * jax.random.normal(keys[5], (stages, MODES)),
        "gate_raw": -2.0 + stage_offset
                    + 0.5 * jax.random.normal(keys[6], (stages, MODES)),
        "readout": jax.random.normal(keys[4], (2 * MODES, channels)) * 0.1,
    }


def train(use_gates, steps, seed, gate_penalty, channels=2,
          learning_rate=3e-2):
    key = jax.random.PRNGKey(seed)
    params = initial_params(key, channels=channels)
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, batch_key):
        inputs, target = make_batch(batch_key, channels=channels)
        (_, error), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, inputs, target, use_gates, gate_penalty)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state, error

    errors = []
    for index in range(steps):
        key, batch_key = jax.random.split(key)
        params, state, error = step(params, state, batch_key)
        errors.append(float(error))
    return params, errors


def evaluate(params, use_gates, seed=99, channels=2):
    inputs, target = make_batch(jax.random.PRNGKey(seed), channels=channels)
    prediction, gates, stages = model_apply(params, inputs, use_gates)
    error = float(jnp.mean((prediction - target) ** 2))
    if channels == 2:
        # the two demands are on SEPARATE outputs, so their errors are read
        # off directly rather than split by a mask
        memory_error = float(jnp.mean((prediction[..., 0]
                                       - target[..., 0]) ** 2))
        lead_error = float(jnp.mean((prediction[..., 1]
                                     - target[..., 1]) ** 2))
    else:
        delayed_part = target[..., 0] * (jnp.arange(LENGTH)[None, :] > DELAY)
        residual = prediction[..., 0] - target[..., 0]
        memory_error = float(jnp.mean((residual * (delayed_part > 0)) ** 2))
        lead_error = float(jnp.mean((residual * (delayed_part == 0)) ** 2))
    gate_values = [jnp.asarray(gate) for gate in gates]
    numerators = [jnp.asarray(n.real) for _, n in stages]
    stage_tie = (float(jnp.max(jnp.abs(numerators[0] - numerators[1])))
                 if len(numerators) > 1 else float("nan"))

    # DOES A MODE'S GATE TRACK WHICH CHANNEL IT SERVES? The readout weight
    # of mode j on each channel says what that mode is used for; the gate
    # says how prospective it is. A positive correlation between the gate
    # and the LEAD channel's share is direct evidence of specialization.
    readout = jnp.asarray(params["readout"])
    weight = jnp.abs(readout[:MODES]) + jnp.abs(readout[MODES:])
    specialization = None
    if channels == 2:
        share = weight[:, 1] / (jnp.sum(weight, axis=1) + 1e-12)
        mean_gate = jnp.mean(jnp.stack(gate_values), axis=0)
        centred_share = share - jnp.mean(share)
        centred_gate = mean_gate - jnp.mean(mean_gate)
        denominator = (jnp.linalg.norm(centred_share)
                       * jnp.linalg.norm(centred_gate) + 1e-12)
        specialization = {
            "lead_channel_share_per_mode": [float(v) for v in share],
            "mean_gate_per_mode": [float(v) for v in mean_gate],
            "gate_vs_lead_share_correlation":
                float(jnp.sum(centred_share * centred_gate) / denominator),
        }
    return {
        "mean_squared_error": error,
        "long_delay_component_error": memory_error,
        "lead_component_error": lead_error,
        "gates_per_mode": [[float(value) for value in gate]
                           for gate in gate_values],
        "gate_spread": [float(jnp.max(gate) - jnp.min(gate))
                        for gate in gate_values],
        "gate_ratio_across_modes": [
            float(jnp.max(gate) / jnp.maximum(jnp.min(gate), 1e-12))
            for gate in gate_values],
        "numerator_n1_minus_n2_max_abs": stage_tie,
        "stages_are_tied": bool(stage_tie < 1e-4),
        "gamma_numerator_mean": (float(jnp.mean(numerators[0] + numerators[1]))
                                 if len(numerators) > 1 else float("nan")),
        "mass_numerator_mean": (float(jnp.mean(numerators[0] * numerators[1]))
                                if len(numerators) > 1 else float("nan")),
        "max_stage_pole": float(jnp.max(jnp.stack(
            [MP.stage_pole(d.real) for d, _ in stages]))),
        "specialization": specialization,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-penalty", type=float, default=1e-3)
    parser.add_argument("--spread-threshold", type=float, default=0.1)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=2,
                        help="2 (default): memory and lead on SEPARATE "
                             "outputs, so per-mode specialization can be "
                             "used. 1: the original summed target.")
    parser.add_argument("--component-margin", type=float, default=0.02,
                        help="relative change counted as real, per "
                             "component")
    args = parser.parse_args()

    gated_params, gated_errors = train(True, args.steps, args.seed,
                                       args.gate_penalty, args.channels)
    native_params, native_errors = train(False, args.steps, args.seed,
                                         args.gate_penalty, args.channels)
    gated = evaluate(gated_params, True, channels=args.channels)
    native = evaluate(native_params, False, channels=args.channels)
    differentiated = any(spread > args.spread_threshold
                         for spread in gated["gate_spread"])
    # BOTH components, not the total: the claim is retained long-delay
    # memory AND reduced lag, so a trade between them is not a success
    lead_change = ((gated["lead_component_error"]
                    - native["lead_component_error"])
                   / max(native["lead_component_error"], 1e-12))
    memory_change = ((gated["long_delay_component_error"]
                      - native["long_delay_component_error"])
                     / max(native["long_delay_component_error"], 1e-12))
    lead_improved = lead_change < -args.component_margin
    memory_retained = memory_change < args.component_margin
    better = gated["mean_squared_error"] < native["mean_squared_error"]
    report = {
        "schema": "s5-modal/synthetic-gates-v1",
        "task": ("long-delay memory and switch-edge lead on separate "
                 "outputs" if args.channels == 2 else
                 "long-delay memory plus switch-edge lead summed into one "
                 "output (the original probe)"),
        "channels": args.channels,
        "steps": args.steps, "seed": args.seed,
        "gate_penalty": args.gate_penalty,
        "gated": gated, "all_native_baseline": native,
        "final_training_error": {"gated": gated_errors[-1],
                                 "native": native_errors[-1]},
        "gates_are_heterogeneous": bool(differentiated),
        "beats_all_native_baseline_on_total": bool(better),
        "component_changes": {"lead_relative": lead_change,
                              "long_delay_relative": memory_change,
                              "margin": args.component_margin},
        "lead_improved": bool(lead_improved),
        "long_delay_memory_retained": bool(memory_retained),
        "status": ("DIFFERENTIATION_DEMONSTRATED"
                   if differentiated and lead_improved and memory_retained
                   else "NOT_DEMONSTRATED"),
        "criterion": ("heterogeneous gates AND reduced lead error AND "
                      "long-delay memory not degraded; a trade between the "
                      "two components is not a success"),
        "scope": ("a synthetic demonstration that per-mode gates can "
                  "differentiate; NOT evidence of benefit on Speech "
                  "Commands, and no scientific claim follows from it alone"),
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in
                      ("status", "channels", "criterion",
                       "gates_are_heterogeneous",
                       "lead_improved", "long_delay_memory_retained",
                       "component_changes", "gated",
                       "all_native_baseline")}, indent=2))


if __name__ == "__main__":
    main()
