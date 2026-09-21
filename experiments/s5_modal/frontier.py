"""The memory-versus-lag FRONTIER, and the one-stage versus two-stage
ablation that decides whether generalized WWJ mechanics are needed at all.

WHAT THE PREVIOUS PROBE SETTLED AND WHAT IT DID NOT. Three seeds at a
single delay of 32 showed that the cascade's LEAD improvement survives a
capacity-matched control on every seed, and that its apparent MEMORY
improvement does not: a plain Native arm with more modes matched it. Three
seeds is too few to test either claim properly, one delay is a point rather
than a frontier, and the two-stage cascade was never compared against the
one-stage ordinary-prospective filter it contains. This experiment fixes
all three.

THE ARMS. Five, so that both equal-mode and equal-parameter comparisons are
available for every question:

  native                      gates off, 16 modes         -- the reference
  native_capacity_matched     gates off, more modes       -- equal parameters
  one_stage                   1 stage, 16 modes           -- equal modes
  one_stage_capacity_matched  1 stage, more modes         -- equal parameters
  two_stage                   2 stages, 16 modes          -- the construction

`one_stage` realizes ORDINARY prospectivity, a single numerator zero.
`two_stage` realizes the GENERALIZED numerator 1 + Gamma D + M D^2. They
share their mode count, and for a given seed they share the draw of Lambda,
B and the readout exactly -- those tensors do not depend on the stage count
-- so the equal-mode ablation differs in the second stage and nothing else.

THE CRUCIAL ABLATION, stated before the run:

  * if one stage gives the same lead improvement and the same memory
    retention, ordinary prospectivity is sufficient and the second stage is
    unjustified;
  * if two stages retain memory better, or reach a better lead at equal
    memory, that specifically supports the generalized mechanics;
  * if NEITHER is memory non-inferior, the selective gates are not doing
    what they were designed to do, whatever happens to the lead.

WHAT IS PREDECLARED. The memory non-inferiority margin (10%, with 5%
reported alongside), the significance level (0.05), the delays, the seed
count, and the direction of every test. None of these is to be moved after
seeing the numbers.

NORMALIZED ERROR. Both components are reported as MSE divided by that
channel's own variance, so a number is a fraction of the signal rather than
an absolute MSE whose scale changes with the delay. 1.0 means "no better
than predicting the mean"; the frontier is only interpretable this way,
because the memory target's variance changes by two orders of magnitude
across the delays being swept.

MODE INITIALIZATION, corrected for this sweep. The previous probe drew
decay rates uniformly on [0.002, 0.302], whose 16-sample minimum sits near
0.02 -- a 51-token timescale, which cannot hold a 256-token delay. Rates
are now drawn LOG-uniformly across the range, so slow modes are reliably
present at every delay in the sweep. This is identical in every arm and so
favours none of them.
"""

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import optax

from s5 import modal_prospective as MP
from experiments.s5_modal import paired as PAIRED
from experiments.s5_modal.frontier_design import (
    POWER_FLOOR, available_comparisons, decide, effective_parameters,
    matched_modes, select_arms)

MODES = 16
BATCH = 16
#: fixed across the whole sweep, so only the delay changes between points
LENGTH = 1024
DELAYS = (16, 32, 64, 128, 256)
#: the lead channel's input is blurred by this much; sharpening it is what
#: a numerator zero is for
BLUR_TIMESCALE = 8.0
CHANNELS = 2
#: log-uniform decay rates: the slowest mode must outlast the longest delay
RATE_MIN = 1.0 / (2.0 * max(DELAYS))
RATE_MAX = 0.3
# ------------------------------------------------------------------ task --
def make_batch(key, delay, batch=BATCH, length=LENGTH):
    """(inputs, target) for one delay.

    Channel 0 of the input carries impulses; channel 1 carries a BLURRED
    switch. Channel 0 of the target is an exponential trace of the impulse
    at timescale `delay` -- what a slow native mode represents and a lead
    filter attenuates. Channel 1 is the SHARP switch, which the blurred
    input lags.
    """
    impulse_key, switch_key = jax.random.split(key)
    positions = jax.random.randint(impulse_key, (batch,), 0, length // 2)
    impulses = jnp.zeros((batch, length)).at[
        jnp.arange(batch), positions].set(1.0)
    switch_times = jax.random.randint(switch_key, (batch,), length // 4,
                                      3 * length // 4)
    switch = (jnp.arange(length)[None, :]
              >= switch_times[:, None]).astype(jnp.float32)
    blurred = smooth(switch, BLUR_TIMESCALE)
    inputs = jnp.stack((impulses, blurred), axis=-1)
    trace = smooth(impulses, float(delay)) * float(delay)
    return inputs, jnp.stack((trace, switch), axis=-1)


def smooth(signal, timescale):
    """Causal exponential smoothing along the token axis."""
    decay = jnp.exp(-1.0 / timescale)

    def step(carry, value):
        carry = decay * carry + (1.0 - decay) * value
        return carry, carry

    return jax.lax.scan(step, jnp.zeros((signal.shape[0],)), signal.T)[1].T


# ----------------------------------------------------------------- model --
def model_apply(params, inputs, use_gates):
    """Native scan, per-mode cascade, linear readout.

    `use_gates=False` forces every gate to zero, which makes each stage an
    exact identity: that arm IS Native S5's recurrence, not an approximation
    of it.
    """
    lambda_bar = jnp.exp(jnp.clip(params["lambda_re"], -0.9, -1e-4)
                         + 1j * params["lambda_im"])
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
        return MP.apply_cascade(MP.native_states(lambda_bar, b_bar, sequence),
                                stages)

    states = jax.vmap(one)(inputs)
    features = jnp.concatenate((states.real, states.imag), axis=-1)
    return jnp.einsum("blf,fc->blc", features, params["readout"]), gates


def loss_fn(params, inputs, target, use_gates, gate_penalty):
    prediction, gates = model_apply(params, inputs, use_gates)
    error = jnp.mean((prediction - target) ** 2)
    return error + gate_penalty * sum(jnp.mean(g) for g in gates), error


def initial_params(key, stages, modes):
    """Identical across arms wherever the shapes allow it.

    Lambda, B and the readout do not depend on the stage count, so for one
    seed the one-stage and two-stage arms at the same mode count start from
    exactly the same recurrence and the same readout. The arms with a
    different mode count necessarily draw different tensors; that is the
    price of matching parameters and it is stated in the report.

    Stages and modes are both initialized ASYMMETRICALLY. Identical stages
    receive identical gradients and stay tied for ever, pinning n1 = n2 and
    with it the critical branch M = (Gamma/2)^2, so the general passive
    branch would be unreachable.
    """
    keys = jax.random.split(key, 8)
    rate = jnp.exp(jax.random.uniform(keys[0], (modes,),
                                      minval=math.log(RATE_MIN),
                                      maxval=math.log(RATE_MAX)))
    offset = (jnp.zeros((1, 1)) if stages == 1
              else jnp.linspace(-0.5, 0.5, stages)[:, None])
    mode_offset = jnp.linspace(-0.3, 0.3, modes)[None, :]
    return {
        "lambda_re": -rate,
        "lambda_im": jax.random.uniform(keys[1], (modes,), minval=-2.0,
                                        maxval=2.0),
        "b_re": jax.random.normal(keys[2], (modes, 2)) * 0.5,
        "b_im": jax.random.normal(keys[3], (modes, 2)) * 0.5,
        "readout": jax.random.normal(keys[4], (2 * modes, CHANNELS)) * 0.1,
        "d_raw": offset + mode_offset,
        "delta_raw": 0.5 + offset
                     + 0.1 * jax.random.normal(keys[5], (stages, modes)),
        "gate_raw": -2.0 + offset
                    + 0.5 * jax.random.normal(keys[6], (stages, modes)),
    }


def train_all_seeds(use_gates, stages, modes, delay, seeds, steps,
                    gate_penalty, learning_rate=3e-2):
    """Every seed of one arm at once, under vmap.

    Seeds are independent runs, so mapping them is exact rather than an
    approximation, and it is what makes fifteen seeds across five delays
    and five arms affordable. The key schedule is identical in every arm,
    so seed s sees the SAME data stream in all of them -- which is what
    makes every comparison in this file paired.
    """
    keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    params = jax.vmap(lambda key: initial_params(key, stages, modes))(keys)
    optimizer = optax.adam(learning_rate)
    state = jax.vmap(optimizer.init)(params)

    def one_step(params, state, batch_key):
        inputs, target = make_batch(batch_key, delay)
        (_, error), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, inputs, target, use_gates, gate_penalty)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state, error

    step = jax.jit(jax.vmap(one_step))
    # the batch key is fold_in(seed_key, step_index) rather than a carried
    # split chain: it depends on the seed and the step number ALONE, so
    # seed s sees byte-identical data at step i in every one of the five
    # arms. That is what makes the comparisons paired rather than merely
    # simultaneous, and it does not depend on the arms running in any
    # particular order.
    fold = jax.jit(jax.vmap(jax.random.fold_in, in_axes=(0, None)))
    error = jnp.zeros((len(seeds),))
    for index in range(steps):
        params, state, error = step(params, state, fold(keys, index))
    return params, [float(value) for value in error]


def run_arm(use_gates, stages, modes, delay, seeds, args):
    """One arm, in seed chunks, with the per-seed measurements collected.

    Seeds are independent, so chunking is exact rather than an
    approximation. It exists because the mapped batch is
    `chunk * BATCH` sequences of length 1024 and a whole sweep should not
    be lost to one allocation.
    """
    chunk = args.seed_chunk or len(seeds)
    memory, lead, finals, gates = [], [], [], []
    for start in range(0, len(seeds), chunk):
        group = seeds[start:start + chunk]
        params, group_finals = train_all_seeds(
            use_gates, stages, modes, delay, group, args.steps,
            args.gate_penalty)
        group_memory, group_lead = jax.vmap(
            lambda p: normalized_errors(p, use_gates, delay,
                                        args.eval_seed))(params)
        group_gates = jax.vmap(lambda p: gate_report(p, use_gates))(params)
        memory.extend(float(value) for value in group_memory)
        lead.extend(float(value) for value in group_lead)
        finals.extend(group_finals)
        gates.append(group_gates)
    merged = {key: jnp.concatenate([group[key] for group in gates], axis=0)
              for key in gates[0]}
    return {
        "memory": memory, "lead": lead,
        "median_memory": PAIRED.median(memory),
        "median_lead": PAIRED.median(lead),
        "final_training_error": sum(finals) / len(finals),
        "modes": modes, "stages": stages if use_gates else 0,
        "effective_parameters": effective_parameters(
            modes, stages if use_gates else 0),
        "gates": {key: PAIRED.median([float(v) for v in value])
                  for key, value in merged.items() if value.ndim == 1},
        "gates_seed_zero": {key: [float(v) for v in value[0]]
                            for key, value in merged.items()
                            if value.ndim == 2},
    }


# -------------------------------------------------------------- measuring --
def normalized_errors(params, use_gates, delay, eval_seed):
    """Per-channel MSE divided by that channel's variance.

    Absolute MSEs are not comparable across delays -- the memory target's
    variance changes by two orders of magnitude over this sweep -- so every
    number the frontier is drawn from is a fraction of the signal.
    """
    inputs, target = make_batch(jax.random.PRNGKey(eval_seed), delay)
    prediction, _ = model_apply(params, inputs, use_gates)

    def one(index):
        channel = target[..., index]
        spread = jnp.mean((channel - jnp.mean(channel)) ** 2)
        return jnp.mean((prediction[..., index] - channel) ** 2) / spread

    return one(0), one(1)


def gate_report(params, use_gates):
    """Per-mode gates against per-mode timescales, as ARRAYS.

    The construction predicts SELECTIVITY: slow modes, which are the ones
    carrying long-delay memory, should keep their gates near zero, while
    faster modes open theirs to produce the lead. The correlation between a
    mode's gate and its DECAY RATE is therefore predicted positive, and the
    slow and fast halves are reported separately so the correlation is not
    the only evidence.

    Everything here stays a jnp array: this function is vmapped over seeds,
    and a `float()` inside a trace raises ConcretizationTypeError. The
    conversion happens in `main`, which is not traced.
    """
    rate = jnp.abs(jnp.clip(params["lambda_re"], -0.9, -1e-4))
    gates = (jax.nn.sigmoid(params["gate_raw"]) if use_gates
             else jnp.zeros_like(params["gate_raw"]))
    mean_gate = jnp.mean(gates, axis=0)
    order = jnp.argsort(rate)
    half = rate.shape[0] // 2
    slow, fast = order[:half], order[half:]
    centred_rate = rate - jnp.mean(rate)
    centred_gate = mean_gate - jnp.mean(mean_gate)
    denominator = (jnp.linalg.norm(centred_rate)
                   * jnp.linalg.norm(centred_gate) + 1e-12)
    return {
        "mean_gate": jnp.mean(mean_gate),
        "mean_gate_slow_half": jnp.mean(mean_gate[slow]),
        "mean_gate_fast_half": jnp.mean(mean_gate[fast]),
        "gate_vs_decay_rate_correlation":
            jnp.sum(centred_rate * centred_gate) / denominator,
        "gate_spread": jnp.max(mean_gate) - jnp.min(mean_gate),
        "timescales": 1.0 / rate,
        "gates_per_mode": mean_gate,
    }


def compare(results, treatment, baseline, alpha, margin):
    """Every paired test between two arms, in both components."""
    memory = {arm: results[arm]["memory"] for arm in (treatment, baseline)}
    lead = {arm: results[arm]["lead"] for arm in (treatment, baseline)}
    return {
        "treatment": treatment, "baseline": baseline,
        "lead_improvement": PAIRED.paired_improvement(
            lead[treatment], lead[baseline], alpha),
        "memory_improvement": PAIRED.paired_improvement(
            memory[treatment], memory[baseline], alpha),
        "memory_non_inferiority": PAIRED.paired_noninferiority(
            memory[treatment], memory[baseline], margin, alpha),
        "memory_non_inferiority_strict": PAIRED.paired_noninferiority(
            memory[treatment], memory[baseline], PAIRED.SECONDARY_MARGIN,
            alpha),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=15,
                        help="paired seeds; 10-20 is the declared range")
    parser.add_argument("--delays", type=int, nargs="+", default=list(DELAYS))
    parser.add_argument("--modes", type=int, default=MODES)
    parser.add_argument("--arms", nargs="+", default=None,
                        help="run only these arms. The point of a subset is "
                             "the POWER CHECK: one arm at full steps across "
                             "every delay answers whether the memory channel "
                             "is learnable at all, for a fifth of the cost "
                             "of the sweep. A subset that lacks the arms the "
                             "decision rule reads yields PARTIAL_ARM_SUBSET "
                             "rather than a verdict.")
    parser.add_argument("--gate-penalty", type=float, default=1e-3)
    parser.add_argument("--eval-seed", type=int, default=99)
    parser.add_argument("--seed-chunk", type=int, default=0,
                        help="seeds trained per vmapped group; 0 means all "
                             "at once. Lower it if the device runs out of "
                             "memory -- seeds are independent, so chunking "
                             "changes nothing but the allocation size.")
    parser.add_argument("--alpha", type=float, default=PAIRED.ALPHA)
    parser.add_argument("--margin", type=float,
                        default=PAIRED.MEMORY_NONINFERIORITY_MARGIN,
                        help="PREDECLARED memory non-inferiority margin")
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    arms = select_arms(args.arms, args.modes)
    comparisons = available_comparisons([name for name, _, _, _ in arms])

    frontier = []
    for delay in args.delays:
        results = {}
        for name, use_gates, stages, modes in arms:
            began = time.time()
            results[name] = run_arm(use_gates, stages, modes, delay, seeds,
                                    args)
            results[name]["wall_seconds"] = time.time() - began
            print(f"delay {delay:4d}  {name:28s} "
                  f"memory {results[name]['median_memory']:.4g}  "
                  f"lead {results[name]['median_lead']:.4g}  "
                  f"[{results[name]['wall_seconds']:.0f}s]", flush=True)
        # POWER: a channel the reference arm cannot learn carries no
        # information about whether the cascade helps on it
        reference = results["native_capacity_matched"]
        power = {"memory": 1.0 - reference["median_memory"],
                 "lead": 1.0 - reference["median_lead"]}
        row = {
            "delay": delay,
            "arms": results,
            "variance_explained_by_reference": power,
            "underpowered_channels": {name: value
                                      for name, value in power.items()
                                      if value < POWER_FLOOR},
            "comparisons": {
                f"{treatment}_vs_{baseline}":
                    compare(results, treatment, baseline, args.alpha,
                            args.margin)
                for treatment, baseline in comparisons},
        }
        row["verdict"] = decide(row)
        frontier.append(row)
        print(f"delay {delay:4d}  VERDICT {row['verdict']}", flush=True)

    verdicts = [row["verdict"] for row in frontier]
    report = {
        "schema": "s5-modal/frontier-v1",
        "predeclared": {
            "memory_non_inferiority_margin": args.margin,
            "secondary_margin": PAIRED.SECONDARY_MARGIN,
            "alpha": args.alpha,
            "delays": args.delays,
            "paired_seeds": len(seeds),
            "primary_test": "exact paired sign test",
            "secondary_test": "Wilcoxon signed-rank, normal approximation",
            "power_floor": POWER_FLOOR,
        },
        "arms": {name: {"gates": use_gates, "stages": stages, "modes": modes,
                        "effective_parameters": effective_parameters(
                            modes, stages if use_gates else 0)}
                 for name, use_gates, stages, modes in arms},
        "steps": args.steps,
        "length": LENGTH,
        "wall_seconds_total": sum(row["arms"][name]["wall_seconds"]
                                  for row in frontier for name in row["arms"]),
        "gate_penalty": args.gate_penalty,
        "frontier": frontier,
        "verdict_per_delay": dict(zip(args.delays, verdicts)),
        "ablation_question": (
            "one stage realizes ordinary prospectivity, two stages the "
            "generalized WWJ numerator 1 + Gamma D + M D^2; if one stage "
            "matches two on both components the second stage is "
            "unjustified"),
        "scope": ("a synthetic frontier measurement; NOT evidence of "
                  "benefit on Speech Commands, and no scientific claim "
                  "follows from it alone"),
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"verdict_per_delay": report["verdict_per_delay"],
                      "arms": report["arms"],
                      "predeclared": report["predeclared"]}, indent=2))


if __name__ == "__main__":
    main()
