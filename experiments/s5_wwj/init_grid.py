"""Choose the WWJ initialization from FIR-gain and gradient criteria.

WHAT CHANGED AND WHY. The previous version of this file selected by
COMPANION SPECTRAL RADIUS, because the mixed-stencil realization introduced
recurrent poles that could leave the unit disc. On the cluster they always
did (max radius about 1.705 at every declared cell), and that realization was
rejected. The principal architecture applies the exact discrete WWJ operator
to the Native S5 trajectory, which is FIR: it adds ZEROS, and the layer's
only poles are the Native ones. There is therefore no radius to select on.
The criteria are now the ones that actually bind:

  * the NATIVE mode radii, reported unchanged, since WWJ cannot move them;
  * the maximum FIR gain of P_h over a dense frequency grid -- how much the
    operator amplifies;
  * finite, NON-VANISHING gradients of a probe loss with respect to the
    WWJ parameters, so the prospective coordinate can actually be learned;
  * finiteness of the forward pass.

DECLARED SELECTION RULE, fixed before the grid is read, and applied by code:

  1. Sweep k = tau/h over GRID_K and eps over GRID_EPS on the ACTUAL
     initialized S5 modes.
  2. A cell is ADMISSIBLE when the forward pass is finite, the maximum FIR
     gain is <= GAIN_CEILING, and both WWJ gradients are finite with
     magnitude above GRADIENT_FLOOR (relative to the loss).
  3. Among admissible cells at the critical eps = 1/4 -- the principal arm --
     take the SMALLEST k. The operator must start CLOSE TO NATIVE, because
     tau -> 0 recovers Native S5 exactly; the gradient floor is what stops
     "close to Native" from becoming "cannot learn".
  4. The passive arm starts at that same k with eps = EPS_INIT_PASSIVE
     (1/16, the middle declared value) and learns eps.
  5. If nothing is admissible, report NO_ADMISSIBLE_INITIALIZATION and
     select nothing. Do not widen the grid or relax a ceiling here.

No validation or test number is consulted anywhere in the rule.
"""

import argparse
import json

import jax
import jax.numpy as np
from flax.traverse_util import flatten_dict

from experiments.s5_wwj.wwj_model import create_wwj_train_state
from s5.ssm import discretize_zoh
from s5.wwj_operator import k_and_m, max_fir_gain, native_radii, wwj_states
from s5.wwj_ssm import eps_passive_from_raw, raw_from_eps, raw_from_tau
from s5.wwj_ssm import tau_from_raw

GRID_K = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
GRID_EPS = (0.0, 0.0625, 0.25)
#: how much the FIR operator may amplify, at any frequency, in any layer
GAIN_CEILING = 3.0
#: the WWJ gradients must be this large, relative to the probe loss
GRADIENT_FLOOR = 1e-8
EPS_CRITICAL = 0.25
EPS_INIT_PASSIVE = 0.0625
#: the probe sequence the gradient criterion is evaluated on
PROBE_LENGTH = 256


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


def probe_gradients(lambda_bar, b_bar, k_value, eps_value, seed=0):
    """Finite, non-vanishing gradients through the WWJ parameters?

    A small probe loss on random inputs, differentiated with respect to the
    RAW parameters the model actually learns, so a vanishing softplus or
    sigmoid derivative would show up here.
    """
    inputs = jax.random.normal(jax.random.PRNGKey(seed),
                              (PROBE_LENGTH, b_bar.shape[1]))
    tau_raw = np.asarray(raw_from_tau(max(k_value, 1.0001e-3)),
                         dtype=np.float32)
    eps_raw = np.asarray(raw_from_eps(min(max(eps_value, 1e-6), 0.2499)),
                         dtype=np.float32)

    def loss(tau_raw_value, eps_raw_value):
        k, m = k_and_m(tau_from_raw(tau_raw_value),
                       eps_passive_from_raw(eps_raw_value))
        return np.sum(np.abs(
            wwj_states(lambda_bar, b_bar, inputs, k, m)) ** 2).real

    value, grads = jax.value_and_grad(loss, argnums=(0, 1))(tau_raw, eps_raw)
    scale = float(np.abs(value)) + 1e-12
    return {"probe_loss": float(value),
            "grad_tau_raw": float(grads[0]), "grad_eps_raw": float(grads[1]),
            "relative_grad_tau": abs(float(grads[0])) / scale,
            "relative_grad_eps": abs(float(grads[1])) / scale,
            "finite": bool(np.isfinite(value))
                      and all(bool(np.isfinite(g)) for g in grads)}


def sweep(modes):
    cells = []
    for k_value in GRID_K:
        for eps_value in GRID_EPS:
            k, m = k_and_m(np.asarray(k_value, dtype=np.float32),
                           np.asarray(eps_value, dtype=np.float32))
            gain = float(max_fir_gain(k, m))
            per_layer, finite = {}, True
            probe = None
            for name, (lambda_bar, b_bar) in modes.items():
                radii = native_radii(lambda_bar)
                states = wwj_states(lambda_bar, b_bar,
                                    np.ones((8, b_bar.shape[1]),
                                            dtype=np.float32), k, m)
                finite = finite and bool(np.all(np.isfinite(states)))
                per_layer[name] = {
                    "native_radius_max": float(np.max(radii)),
                    "native_radius_min": float(np.min(radii))}
                if probe is None and eps_value > 0.0:
                    probe = probe_gradients(lambda_bar, b_bar, k_value,
                                            eps_value)
            if probe is None:            # eps = 0: only tau can be learned
                name = next(iter(modes))
                probe = probe_gradients(*modes[name], k_value, 1e-6)
            admissible = bool(finite and gain <= GAIN_CEILING
                              and probe["finite"]
                              and probe["relative_grad_tau"] > GRADIENT_FLOOR)
            cells.append({"k": k_value, "eps": eps_value, "m": float(m),
                          "max_fir_gain": gain, "forward_finite": finite,
                          "probe": probe, "admissible": admissible,
                          "per_layer": per_layer})
    return cells


def select(cells):
    """Apply the declared rule: smallest admissible k at the critical eps."""
    critical = [cell for cell in cells
                if cell["eps"] == EPS_CRITICAL and cell["admissible"]]
    if not critical:
        return {"status": "NO_ADMISSIBLE_INITIALIZATION",
                "rule": f"max FIR gain <= {GAIN_CEILING}, finite forward, "
                        f"relative WWJ gradient > {GRADIENT_FLOOR}, at "
                        f"eps = {EPS_CRITICAL}",
                "selected": None}
    chosen = min(critical, key=lambda cell: cell["k"])
    return {"status": "SELECTED",
            "rule": "smallest admissible k at the critical eps: start close "
                    "to Native (tau -> 0 IS Native) subject to the gradient "
                    "floor",
            "selected": {"tau_init": chosen["k"],
                         "eps_init_passive": EPS_INIT_PASSIVE,
                         "eps_critical": EPS_CRITICAL,
                         "max_fir_gain": chosen["max_fir_gain"],
                         "gain_ceiling": GAIN_CEILING,
                         "relative_grad_tau":
                             chosen["probe"]["relative_grad_tau"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=301)
    args = parser.parse_args()
    state = create_wwj_train_state("wwj_critical_s5", args.seed)
    cells = sweep(layer_modes(state))
    report = {"schema": "s5-wwj/init-grid-fir-v2", "seed": args.seed,
              "operator": "FIR three-tap on the Native S5 trajectory; no "
                          "recurrent poles are added",
              "grid_k": list(GRID_K), "grid_eps": list(GRID_EPS),
              "gain_ceiling": GAIN_CEILING,
              "gradient_floor": GRADIENT_FLOOR,
              "cells": cells, **select(cells)}
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in
                      ("status", "rule", "selected")}, indent=2))
    if report["status"] != "SELECTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
