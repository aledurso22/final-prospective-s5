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

MODES = 16
LENGTH = 256
BATCH = 16
DELAY = 32
#: the memory channel is an exponential TRACE of timescale DELAY, not a
#: delayed delta: a bank of exponential modes represents a trace naturally
#: and cannot represent a delay line, so the old target asked for something
#: the model class does not have
MEMORY_TIMESCALE = float(DELAY)
#: the lead channel's input is BLURRED, so sharpening it needs the
#: prospective mechanics; the old target was the first difference of an
#: input channel, which a two-tap readout solves exactly
BLUR_TIMESCALE = 8.0
#: the baseline must explain at least this much of a channel for a
#: comparison on it to mean anything. There is NO upper ceiling: a first
#: version rejected a baseline at R2 = 0.9984 as "already solved", while the
#: gated arm was cutting that residual twelve-fold. R2 near 1 is the wrong
#: way to ask whether headroom exists; the right way is whether the
#: difference survives seed-to-seed variation, which is now measured.
POWER_FLOOR = 0.20


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

    def smooth(signal, timescale):
        """Causal exponential smoothing, as a scan over the token axis."""
        decay = jnp.exp(-1.0 / timescale)

        def step(carry, value):
            carry = decay * carry + (1.0 - decay) * value
            return carry, carry

        return jax.lax.scan(step, jnp.zeros((signal.shape[0],)),
                            signal.T)[1].T
    switch_times = jax.random.randint(switch_key, (batch,), length // 4,
                                      3 * length // 4)
    ramp = jnp.arange(length)[None, :]
    switch = (ramp >= switch_times[:, None]).astype(jnp.float32)
    # the lead channel's INPUT is a blurred switch: the sharp transition has
    # to be reconstructed, which is what a lead filter is for
    blurred = smooth(switch, BLUR_TIMESCALE)
    inputs = jnp.stack((impulses, blurred), axis=-1)
    # the memory term: an exponential TRACE of the impulse, timescale
    # MEMORY_TIMESCALE, which a slow mode represents and a lead filter
    # attenuates
    delayed = smooth(impulses, MEMORY_TIMESCALE) * MEMORY_TIMESCALE
    # the lead term: the SHARP switch, against the blurred input
    edge = switch
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


def initial_params(key, stages=2, channels=2, modes=MODES):
    """Asymmetric across stages AND across modes.

    Identical stages receive identical gradients and stay tied for ever,
    which pins n1 = n2 and so M = (Gamma/2)^2 -- the critical branch only.
    Identical modes cannot differentiate either. Both symmetries are broken
    here, by initialization rather than by any change to the mathematics.
    """
    keys = jax.random.split(key, 8)
    stage_offset = jnp.linspace(-0.5, 0.5, stages)[:, None]
    mode_offset = jnp.linspace(-0.3, 0.3, modes)[None, :]
    return {
        # the slowest representable mode must outlast the memory timescale:
        # the previous range topped out at 20 tokens for a 32-token task, so
        # the probe could not hold what it asked the model to recall
        "lambda_re": -(0.002 + 0.3 * jax.random.uniform(keys[0], (modes,))),
        "lambda_im": jax.random.uniform(keys[1], (modes,), minval=-2.0,
                                        maxval=2.0),
        "b_re": jax.random.normal(keys[2], (modes, 2)) * 0.5,
        "b_im": jax.random.normal(keys[3], (modes, 2)) * 0.5,
        "d_raw": stage_offset + mode_offset,
        "delta_raw": 0.5 + stage_offset
                     + 0.1 * jax.random.normal(keys[5], (stages, modes)),
        "gate_raw": -2.0 + stage_offset
                    + 0.5 * jax.random.normal(keys[6], (stages, modes)),
        "readout": jax.random.normal(keys[4], (2 * modes, channels)) * 0.1,
    }


def train(use_gates, steps, seed, gate_penalty, channels=2, modes=MODES,
          learning_rate=3e-2):
    key = jax.random.PRNGKey(seed)
    params = initial_params(key, channels=channels, modes=modes)
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


def variance_explained(prediction, target):
    """R^2 against the target's own variance: the power check."""
    residual = float(jnp.mean((prediction - target) ** 2))
    spread = float(jnp.mean((target - jnp.mean(target)) ** 2))
    return 1.0 - residual / spread if spread > 0 else float("nan")


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
    modes = readout.shape[0] // 2
    weight = jnp.abs(readout[:modes]) + jnp.abs(readout[modes:])
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
    r_squared = ({"long_delay": variance_explained(prediction[..., 0],
                                                   target[..., 0]),
                  "lead": variance_explained(prediction[..., 1],
                                             target[..., 1])}
                 if channels == 2 else
                 {"total": variance_explained(prediction, target)})
    return {
        "mean_squared_error": error,
        "variance_explained": r_squared,
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


def effective_parameters(modes, gated, channels=2, stages=2):
    """Parameters that actually influence the output.

    With the gates off, d_raw, delta_raw and gate_raw are inert, so the
    all-Native arm has FEWER effective parameters than the gated one. A
    capacity-matched Native control is therefore run with more modes.
    """
    base = 2 * modes + 2 * modes * 2 + 2 * modes * channels
    return base + (3 * stages * modes if gated else 0)


def matched_modes(modes, channels=2, stages=2):
    target = effective_parameters(modes, True, channels, stages)
    width = modes
    while effective_parameters(width, False, channels, stages) < target:
        width += 1
    return width


def run_arm(use_gates, modes, args):
    """One arm across every seed.

    Returns the per-seed evaluations and the final training error of each
    seed, so the report can show both what was learned and how far training
    got.
    """
    out, finals = [], []
    for seed in args.seeds:
        params, errors = train(use_gates, args.steps, seed,
                               args.gate_penalty, args.channels, modes)
        out.append(evaluate(params, use_gates, channels=args.channels))
        finals.append(errors[-1])
    return out, finals


def summarize(runs, key, sub=None):
    values = [(run[key][sub] if sub else run[key]) for run in runs]
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / 2.0
    return {"mean": mean, "half_range": spread, "per_seed": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="the improvement must survive seed variation")
    parser.add_argument("--modes", type=int, default=MODES)
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

    wide = matched_modes(args.modes, args.channels)
    runs = {"gated": run_arm(True, args.modes, args),
            "native": run_arm(False, args.modes, args),
            "native_capacity_matched": run_arm(False, wide, args)}
    arms = {name: evaluations for name, (evaluations, _) in runs.items()}
    final_errors = {name: finals for name, (_, finals) in runs.items()}
    gated, native = arms["gated"][0], arms["native"][0]
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
    # POWER CHECK, before any verdict: a channel the baseline cannot learn
    # at all carries no information about whether the cascade helps. The
    # first probe compared two models on a memory channel neither arm
    # learned (R2 0.111 -> -0.000), and no verdict follows from that. The
    # baseline_power used below is averaged over seeds.
    #
    # every comparison is against BOTH Native arms, and must exceed the
    # seed-to-seed half-range of the baseline to count
    components = {"long_delay": "long_delay_component_error",
                  "lead": "lead_component_error"}
    comparison, survives = {}, True
    for name, field in components.items():
        row = {arm: summarize(runs, field) for arm, runs in arms.items()}
        for baseline in ("native", "native_capacity_matched"):
            change = ((row["gated"]["mean"] - row[baseline]["mean"])
                      / max(row[baseline]["mean"], 1e-12))
            margin = row[baseline]["half_range"] / max(row[baseline]["mean"],
                                                       1e-12)
            row[f"vs_{baseline}"] = {
                "relative_change": change,
                "baseline_seed_half_range_relative": margin,
                "improved_beyond_seed_spread": bool(change < -margin
                                                    and change
                                                    < -args.component_margin)}
            survives = survives and row[f"vs_{baseline}"][
                "improved_beyond_seed_spread"]
        comparison[name] = row
    baseline_power = {name: summarize(arms["native"], "variance_explained",
                                      name)["mean"]
                      for name in components}
    underpowered = {name: value for name, value in baseline_power.items()
                    if value < POWER_FLOOR}
    report = {
        "schema": "s5-modal/synthetic-gates-v2",
        "seeds": args.seeds,
        "modes": args.modes,
        "capacity_matched_native_modes": wide,
        "effective_parameters": {
            "gated": effective_parameters(args.modes, True, args.channels),
            "native": effective_parameters(args.modes, False, args.channels),
            "native_capacity_matched":
                effective_parameters(wide, False, args.channels)},
        "component_comparison": comparison,
        "improvement_survives_seeds_and_capacity_control": bool(survives),
        "task": ("long-delay memory and switch-edge lead on separate "
                 "outputs" if args.channels == 2 else
                 "long-delay memory plus switch-edge lead summed into one "
                 "output (the original probe)"),
        "channels": args.channels,
        "steps": args.steps,
        "gate_penalty": args.gate_penalty,
        "gated": gated, "all_native_baseline": native,
        "final_training_error": {
            name: sum(values) / len(values)
            for name, values in final_errors.items()},
        "gates_are_heterogeneous": bool(differentiated),
        "beats_all_native_baseline_on_total": bool(better),
        "component_changes": {"lead_relative": lead_change,
                              "long_delay_relative": memory_change,
                              "margin": args.component_margin},
        "lead_improved": bool(lead_improved),
        "long_delay_memory_retained": bool(memory_retained),
        "baseline_variance_explained": baseline_power,
        "underpowered_channels": underpowered,
        "power_floor": POWER_FLOOR,
        "status": ("PROBE_UNDERPOWERED" if underpowered else
                   "DEMONSTRATED_AGAINST_BOTH_CONTROLS"
                   if differentiated and survives
                   else "NOT_DEMONSTRATED"),
        "criterion": ("heterogeneous gates AND both components improved "
                      "beyond the baseline's seed-to-seed spread, against "
                      "BOTH an equal-mode Native arm and a "
                      "capacity-matched one"),
        "scope": ("a synthetic demonstration that per-mode gates can "
                  "differentiate; NOT evidence of benefit on Speech "
                  "Commands, and no scientific claim follows from it alone"),
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in
                      ("status", "channels", "criterion", "seeds",
                       "effective_parameters",
                       "capacity_matched_native_modes",
                       "component_comparison",
                       "improvement_survives_seeds_and_capacity_control",
                       "baseline_variance_explained",
                       "underpowered_channels", "gates_are_heterogeneous",
                       "lead_improved", "long_delay_memory_retained",
                       "component_changes", "gated",
                       "all_native_baseline")}, indent=2))


if __name__ == "__main__":
    main()
