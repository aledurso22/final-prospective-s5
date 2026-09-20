"""Whole-model certification: every layer, every direction, every mode.

WHY THIS EXISTS. Cells were previously certified on a handful of randomly
drawn modes. That is not certification: at tau = 2, eps = 0 a subset of six
to eight modes reported a maximum radius of 0.951, while the mode
Abar = 0.9 e^{-2i} in the SAME region has radius 1.4416 and makes the model
diverge at token 242 in float32 -- the sequential path too, not only a scan
(docs/analysis/direct_prospective_tau2_diagnosis.txt). A cell is eligible
only if EVERY production mode passes.

Bidirectionality: S5's reverse branch is the same recurrence run on the
reversed sequence, so both directions share a layer's Lambda_bar and B_bar.
The inventory therefore enumerates each layer once and records that both
directions are covered by it; nothing about the reverse branch escapes
certification.
"""

import jax
import jax.numpy as jnp
import numpy
from flax.traverse_util import flatten_dict

from experiments.s5_three_arm_full import runner as RUNNER
from s5 import direct_prospective as DP
from s5.ssm import discretize_zoh

#: chunk sizes considered, smallest first. C = 1 is NOT parallel and must be
#: benchmarked honestly if it is the only one that passes.
CHUNK_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
#: |H^C| ceiling, from the measured relationship between |H^C| and the
#: float32 error of the block scan
TRANSITION_NORM_CEILING = 10.0
#: A radius above 1 is rejected outright. The earlier 1 + 1e-5 allowance is
#: WITHDRAWN: a cell with rho > 1 is unstable, whether or not a particular
#: 4000-token rollout happens to stay finite, and tau = 5 with eps = 0
#: printed 1.0063 on one fixture and 1.0787247827275395 on another while
#: being called stable. This threshold is a tightening, never a loosening.
RADIUS_BOUND = 1.0
#: the tau values worth certifying. This is a CANDIDATE list, not a list of
#: stable values: every one of them is certified, and none is assumed.
TAU_CANDIDATES = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
#: the production seeds
SEEDS = (301, 302, 303)
#: (order, eps) of each model
MODEL_SPECS = {"professor_tss": (2, 0.0), "wwj_generalized_tss": (3, 0.25)}


def production_mode_inventory(seed=301):
    """Every (layer, mode) of the production-initialized model at `seed`.

    Returns a list of layers, each with its full Lambda_bar and B_bar. No
    subsetting, no sampling: this is the inventory the model actually uses.
    """
    state = RUNNER.init_state("native_matched_s5", seed)
    flat = flatten_dict(state.params)
    layers = []
    for key in sorted(k for k in flat if k[-1] == "Lambda_re"):
        prefix = key[:-1]
        lambda_continuous = (jnp.clip(flat[prefix + ("Lambda_re",)], None,
                                      -1e-4)
                             + 1j * flat[prefix + ("Lambda_im",)])
        b = flat[prefix + ("B",)]
        step = jnp.exp(flat[prefix + ("log_step",)][:, 0])
        lambda_bar, b_bar = discretize_zoh(
            lambda_continuous, b[..., 0] + 1j * b[..., 1], step)
        layers.append({"layer": "/".join(prefix),
                       "modes": int(lambda_bar.shape[0]),
                       "lambda_bar": lambda_bar, "b_bar": b_bar,
                       "directions": ("forward", "reverse"),
                       "directions_share_modes": True})
    return layers


def cell_coefficients(lambda_bar, b_bar, tau_value, eps, order, dtype):
    tau = jnp.full((lambda_bar.shape[0],), tau_value, dtype=dtype)
    if order == 2:
        return DP.professor_tss_state_coefficients(
            lambda_bar, b_bar, tau, DP.PROFESSOR_LINEAR_TARGET)
    mass = DP.mass_from_eps(tau, jnp.asarray(eps, dtype=dtype))
    return DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass,
                                         DP.PROFESSOR_LINEAR_TARGET)


def exact_radii(coefficients_A):
    """max |root| per mode on the host, from numpy."""
    arrays = [numpy.asarray(value) for value in coefficients_A]
    radii = []
    for mode in range(arrays[0].shape[0]):
        poly = [1.0] + [-complex(array[mode]) for array in arrays]
        radii.append(float(numpy.max(numpy.abs(numpy.roots(poly)))))
    return numpy.asarray(radii)


def gate_1_stability(layers, tau_value, eps, order):
    """EVERY mode of EVERY layer, not a subset. Reports the worst one."""
    worst = {"radius": -1.0}
    finite = True
    for entry in layers:
        A, _ = cell_coefficients(entry["lambda_bar"], entry["b_bar"],
                                 tau_value, eps, order, jnp.float64)
        finite = finite and all(bool(jnp.all(jnp.isfinite(value)))
                                for value in A)
        radii = exact_radii(A)
        index = int(numpy.argmax(radii))
        if float(radii[index]) > worst["radius"]:
            value = complex(numpy.asarray(entry["lambda_bar"])[index])
            worst = {"radius": float(radii[index]), "layer": entry["layer"],
                     "mode_index": index,
                     "lambda_bar": [value.real, value.imag],
                     "abs_lambda_bar": abs(value)}
    return {"coefficients_finite": finite,
            "max_radius_over_all_modes": worst["radius"],
            "worst_mode": worst,
            "passes": bool(finite and worst["radius"] <= RADIUS_BOUND),
            "bound": RADIUS_BOUND,
            "modes_certified": sum(entry["modes"] for entry in layers),
            "layers_certified": len(layers)}


def transition_norms(layers, tau_value, eps, order, chunks=CHUNK_CANDIDATES):
    """max |H^C| over EVERY mode, per chunk size."""
    out = {}
    for chunk in chunks:
        peak = 0.0
        for entry in layers:
            A, _ = cell_coefficients(entry["lambda_bar"], entry["b_bar"],
                                     tau_value, eps, order, jnp.float32)
            transition = DP.chunk_transition(tuple(A), chunk)
            peak = max(peak, float(jnp.max(jnp.abs(transition))))
        out[chunk] = peak
    return out


def select_chunk(norms, ceiling=TRANSITION_NORM_CEILING):
    """The LARGEST chunk whose |H^C| over every mode is within the ceiling.

    Returns None when not even C = 1 qualifies. C = 1 carries no parallelism
    at all and must be benchmarked, never assumed useful.
    """
    eligible = [chunk for chunk, norm in sorted(norms.items())
                if norm <= ceiling]
    if not eligible:
        return None
    chosen = max(eligible)
    return {"chunk": chosen, "transition_norm": norms[chosen],
            "ceiling": ceiling,
            "parallel": chosen > 1,
            "note": ("C = 1 is a plain sequential scan with no parallelism; "
                     "benchmark it, do not assume it is useful"
                     if chosen == 1 else
                     f"sequential depth is about C + L/C = {chosen} + L/{chosen}")}


def certify_seeds(tau_value, eps, order, seeds=SEEDS, inventories=None):
    """Gate 1 over EVERY layer and mode of EVERY production seed.

    A cell is eligible only if every mode of every seed passes. The worst
    mode across all seeds is named. This function, and nothing else, decides
    production eligibility: tests, the chunk study and any eventual training
    selection consume its output rather than maintaining a list of their own.
    """
    inventories = inventories or {seed: production_mode_inventory(seed)
                                  for seed in seeds}
    per_seed, worst = {}, None
    for seed, layers in inventories.items():
        stability = gate_1_stability(layers, tau_value, eps, order)
        per_seed[seed] = stability
        if worst is None or stability["max_radius_over_all_modes"] > \
                worst["max_radius_over_all_modes"]:
            worst = dict(stability, seed=seed)
    return {"tau": tau_value, "eps": eps, "order": order,
            "target_construction": DP.PROFESSOR_LINEAR_TARGET,
            "seeds": list(inventories),
            "per_seed": per_seed,
            "worst_over_seeds": worst,
            "max_radius": worst["max_radius_over_all_modes"],
            "bound": RADIUS_BOUND,
            "passes": all(value["passes"] for value in per_seed.values()),
            "verdict": ("PASS" if all(value["passes"]
                                      for value in per_seed.values())
                        else "REJECT")}


def eligible_cells(tau_candidates=TAU_CANDIDATES, seeds=SEEDS,
                   models=None):
    """Certify every (model, tau) candidate. Returns all rows and the
    eligible subset -- which may be EMPTY, and that is a result to record,
    not a reason to invent a candidate."""
    models = models or MODEL_SPECS
    inventories = {seed: production_mode_inventory(seed) for seed in seeds}
    rows = []
    for model, (order, eps) in models.items():
        for tau_value in tau_candidates:
            row = certify_seeds(tau_value, eps, order, seeds, inventories)
            row["model"] = model
            rows.append(row)
    eligible = [row for row in rows if row["passes"]]
    return {"rows": rows, "eligible": eligible,
            "status": "ELIGIBLE_CELLS_FOUND" if eligible
                      else "NO_ELIGIBLE_CELL_IN_THE_PRODUCTION_INVENTORY"}


def certify(layers, tau_value, eps, order):
    """Gate 1 over one inventory, then the chunk selection."""
    stability = gate_1_stability(layers, tau_value, eps, order)
    report = {"tau": tau_value, "eps": eps, "order": order,
              "target_construction": DP.PROFESSOR_LINEAR_TARGET,
              "gate_1_stability": stability}
    if not stability["passes"]:
        report["status"] = "REJECTED_BY_GATE_1"
        report["chunk_selection"] = None
        return report
    norms = transition_norms(layers, tau_value, eps, order)
    report["transition_norms"] = norms
    report["chunk_selection"] = select_chunk(norms)
    report["status"] = ("CHUNK_SELECTED" if report["chunk_selection"]
                        else "NO_CHUNK_WITHIN_TRANSITION_NORM_CEILING")
    return report
