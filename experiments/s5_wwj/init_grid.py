"""Choose the WWJ initialization from the ACTUAL initialized S5 modes.

The rule below is declared here, before any run, and is evaluated only on
initialized modes and their companion spectral radii. Validation and test
performance are never consulted: choosing an initialization after looking at
validation results would make the comparison meaningless.

DECLARED SELECTION RULE
  1. Sweep k = tau/h over GRID_K and eps over GRID_EPS on the modes of a
     production-initialized model (all layers, all modes, both the critical
     and the passive arms' eps values).
  2. A cell is ADMISSIBLE when the maximum companion spectral radius over
     every mode of every layer is <= RADIUS_CEILING (0.98), i.e. strictly
     inside the unit disc with margin, and every coefficient is finite.
  3. Among admissible cells at the CRITICAL eps = 1/4 -- the principal arm --
     take the LARGEST k, because the hypothesis is about the strength of
     prospective compensation and a larger tau compensates more.
  4. The passive arm initializes at that same k with eps = EPS_INIT_PASSIVE
     (1/16, the middle declared grid value) and learns eps from there.
  5. If no cell at eps = 1/4 is admissible, the script reports
     NO_ADMISSIBLE_INITIALIZATION and selects nothing. It does not widen the
     grid or relax the ceiling on its own.

Reading the report never changes the rule; the rule is applied by code.
"""

import argparse
import json

import jax.numpy as np
from flax.traverse_util import flatten_dict

from experiments.s5_wwj.wwj_model import create_wwj_train_state
from s5.ssm import discretize_zoh
from s5.wwj_recurrence import (companion_spectral_radius, mass_from_eps,
                               state_coefficients)

GRID_K = (0.05, 0.1, 0.25, 0.5, 1.0)
GRID_EPS = (0.0, 0.0625, 0.25)
RADIUS_CEILING = 0.98
EPS_CRITICAL = 0.25
EPS_INIT_PASSIVE = 0.0625


def layer_modes(state):
    """(lambda_bar, b_bar) of every layer of an initialized model."""
    flat = flatten_dict(state.params)
    out = {}
    for key in flat:
        if key[-1] != "Lambda_re":
            continue
        prefix = key[:-1]
        lambda_continuous = (np.clip(flat[key], None, -1e-4)
                             + 1j * flat[prefix + ("Lambda_im",)])
        b = flat[prefix + ("B",)]
        step = np.exp(flat[prefix + ("log_step",)][:, 0])
        out["/".join(prefix)] = discretize_zoh(
            lambda_continuous, b[..., 0] + 1j * b[..., 1], step)
    return out


def sweep(modes):
    """The declared grid, with the radii it produces."""
    cells = []
    for k in GRID_K:
        for eps in GRID_EPS:
            radii = {}
            finite = True
            for name, (lambda_bar, b_bar) in modes.items():
                tau = np.full((lambda_bar.shape[0],), k, dtype=np.float32)
                mass = mass_from_eps(tau, np.asarray(eps, dtype=np.float32))
                (A0, A1, A2), C = state_coefficients(lambda_bar, b_bar, tau,
                                                     mass)
                values = companion_spectral_radius(A0, A1, A2)
                radii[name] = [float(value) for value in values]
                finite = finite and all(
                    bool(np.all(np.isfinite(array)))
                    for array in (A0, A1, A2, *C))
            worst = max(max(values) for values in radii.values())
            cells.append({"k": k, "eps": eps, "finite": finite,
                          "max_companion_spectral_radius": worst,
                          "admissible": bool(finite
                                             and worst <= RADIUS_CEILING),
                          "per_layer_max": {name: max(values)
                                            for name, values in radii.items()},
                          "per_mode": radii})
    return cells


def select(cells):
    """Apply the declared rule. Returns the selection or a refusal."""
    critical = [cell for cell in cells
                if cell["eps"] == EPS_CRITICAL and cell["admissible"]]
    if not critical:
        return {"status": "NO_ADMISSIBLE_INITIALIZATION",
                "rule": "max companion spectral radius <= "
                        f"{RADIUS_CEILING} at eps = {EPS_CRITICAL}",
                "selected": None}
    chosen = max(critical, key=lambda cell: cell["k"])
    return {"status": "SELECTED", "rule": "largest admissible k at the "
                                          "critical eps",
            "selected": {"tau_init": chosen["k"],
                         "eps_init_passive": EPS_INIT_PASSIVE,
                         "eps_critical": EPS_CRITICAL,
                         "max_companion_spectral_radius":
                             chosen["max_companion_spectral_radius"],
                         "radius_ceiling": RADIUS_CEILING}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=301)
    args = parser.parse_args()
    # any WWJ arm gives the same initialized S5 modes; tau/eps are swept here
    state = create_wwj_train_state("wwj_critical_s5", args.seed)
    cells = sweep(layer_modes(state))
    report = {"schema": "s5-wwj/init-grid-v1", "seed": args.seed,
              "grid_k": list(GRID_K), "grid_eps": list(GRID_EPS),
              "cells": cells, **select(cells)}
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in
                      ("status", "rule", "selected")}, indent=2))
    if report["status"] != "SELECTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
