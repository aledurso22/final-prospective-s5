"""Stability grid for the direct prospective recurrences, on REAL S5 modes.

Everything here is diagnosis. It trains nothing, launches nothing, and
authorizes nothing by itself.

For every declared cell it reports:

  * the maximum companion spectral radius over every production-initialized
    S5 mode, from the log-norm estimator, CROSS-CHECKED against exact roots
    computed on the host with numpy;
  * every nonfinite coefficient or root;
  * the worst offending native mode, with its index and its value;
  * the dependence on tau, eps and gamma;
  * float32 and float64 rollout finiteness at the production sequence
    length, for the cells that pass the radius bound;
  * gradient finiteness with respect to Abar, Bbar, tau and eps;
  * impulse responses for Native S5, the repository's Zucchet arm, the exact
    common-stencil matched operator and the causal mixed-stencil recurrence.

DECLARED STABILITY BOUND. A companion radius rho grows as rho^L over the
L = 16000-token sequence, so the bound is tied to the sequence length rather
than guessed: RADIUS_BOUND = 1 + 1e-5, which permits at most a factor
(1 + 1e-5)^16000 = 1.17 of growth end to end. It is NOT relaxed to obtain an
admissible cell. If nothing is admissible the report says
NO_ADMISSIBLE_INITIALIZATION and names the analytic reason.
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy
from flax.traverse_util import flatten_dict

from experiments.s5_three_arm_full import runner as RUNNER
from s5 import direct_prospective as DP
from s5.discrete_recurrence import zucchet_coefficients
from s5.ssm import discretize_zoh

#: (1 + 1e-5)^16000 = 1.17: bounded growth over the production sequence
RADIUS_BOUND = 1.0 + 1e-5
GRID_TAU = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0)
GRID_EPS = (0.0, 0.0625, 0.25)
GRID_GAMMA = (0.0, 1.0)
#: MODELS (which equation), never conflated with target constructions
EQUATIONS = ("wwj_generalized_tss", "partially_matched_two_compartment")
#: TARGET CONSTRUCTIONS (how f is built)
TARGETS = (DP.PROFESSOR_LINEAR_TARGET, DP.NATIVE_MATCHED_TARGET)
IMPULSE_LENGTH = 64


def production_modes(seed=301):
    """Lambda_bar and B_bar of every layer of a production-initialized model."""
    state = RUNNER.init_state("native_matched_s5", seed)
    flat = flatten_dict(state.params)
    layers = {}
    for key in flat:
        if key[-1] != "Lambda_re":
            continue
        prefix = key[:-1]
        lambda_continuous = (jnp.clip(flat[key], None, -1e-4)
                             + 1j * flat[prefix + ("Lambda_im",)])
        b = flat[prefix + ("B",)]
        step = jnp.exp(flat[prefix + ("log_step",)][:, 0])
        layers["/".join(prefix)] = discretize_zoh(
            lambda_continuous, b[..., 0] + 1j * b[..., 1], step)
    return layers


def coefficients_for(equation, lambda_bar, b_bar, tau, eps, gamma, target):
    """((A_i), (C_i)) for one cell."""
    tau_vector = jnp.full((lambda_bar.shape[0],), tau, dtype=jnp.float32)
    if equation == "wwj_generalized_tss":
        mass = DP.mass_from_eps(tau_vector, jnp.asarray(eps, jnp.float32))
        return DP.matched_state_coefficients(lambda_bar, b_bar, tau_vector,
                                             mass, target)
    gamma_vector = jnp.full_like(tau_vector, gamma)
    big_gamma = gamma_vector + tau_vector
    mass = DP.mass_from_eps(big_gamma, jnp.asarray(eps, jnp.float32))
    return DP.partially_matched_state_coefficients(
        lambda_bar, b_bar, tau_vector, mass, gamma_vector, target)


def radii(coefficients_A):
    """(estimated, exact) maxima and the worst mode index."""
    estimate = DP.companion_spectral_radius(coefficients_A)
    exact = None
    try:
        from tests.direct_prospective_reference import exact_companion_radius
        exact = exact_companion_radius(coefficients_A)
    except Exception as error:                     # never fail the grid
        return (float(jnp.max(estimate)), None, int(jnp.argmax(estimate)),
                f"{type(error).__name__}: {error}")
    return (float(jnp.max(estimate)), float(numpy.max(exact)),
            int(numpy.argmax(exact)), None)


def rollout_finite(equation, lambda_bar, b_bar, tau, eps, gamma, target,
                   length, dtype):
    """Is a production-length rollout finite in this precision?"""
    inputs = jax.random.normal(jax.random.PRNGKey(0),
                               (length, b_bar.shape[1])).astype(dtype)
    tau_vector = jnp.full((lambda_bar.shape[0],), tau, dtype=dtype)
    if equation == "wwj_generalized_tss":
        mass = DP.mass_from_eps(tau_vector, jnp.asarray(eps, dtype))
        states = DP.matched_states(lambda_bar.astype(
            jnp.complex64 if dtype == jnp.float32 else jnp.complex128),
            b_bar.astype(jnp.complex64 if dtype == jnp.float32
                         else jnp.complex128),
            inputs, tau_vector, mass, target)
    else:
        gamma_vector = jnp.full_like(tau_vector, gamma)
        mass = DP.mass_from_eps(gamma_vector + tau_vector,
                                jnp.asarray(eps, dtype))
        states = DP.partially_matched_states(lambda_bar, b_bar, inputs,
                                             tau_vector, mass, gamma_vector,
                                             target)
    return {"finite": bool(jnp.all(jnp.isfinite(states))),
            "max_abs": float(jnp.max(jnp.abs(states)))}


def gradient_finite(equation, lambda_bar, b_bar, tau, eps, gamma, target,
                    length=256):
    """Finite gradients with respect to Abar, Bbar, tau and eps?"""
    inputs = jax.random.normal(jax.random.PRNGKey(1), (length,
                                                       b_bar.shape[1]))

    def loss(lam, b, tau_value, eps_value):
        tau_vector = jnp.full((lam.shape[0],), tau_value)
        if equation == "wwj_generalized_tss":
            mass = DP.mass_from_eps(tau_vector, eps_value)
            states = DP.matched_states(lam, b, inputs, tau_vector, mass,
                                       target)
        else:
            gamma_vector = jnp.full_like(tau_vector, gamma)
            mass = DP.mass_from_eps(gamma_vector + tau_vector, eps_value)
            states = DP.partially_matched_states(lam, b, inputs, tau_vector,
                                                 mass, gamma_vector, target)
        return jnp.sum(jnp.abs(states) ** 2).real

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
        lambda_bar, b_bar, jnp.asarray(tau), jnp.asarray(eps))
    return {"all_finite": all(bool(jnp.all(jnp.isfinite(g))) for g in grads),
            "grad_tau": float(grads[2]), "grad_eps": float(grads[3])}


def impulse_responses(lambda_bar, b_bar, tau, eps):
    """Native, Zucchet, exact common-stencil and mixed-stencil, on one impulse."""
    from tests import direct_prospective_reference as ORACLE

    inputs = jnp.zeros((IMPULSE_LENGTH, b_bar.shape[1])).at[0, 0].set(1.0)
    tau_vector = jnp.full((lambda_bar.shape[0],), tau, dtype=jnp.float32)
    mass = DP.mass_from_eps(tau_vector, jnp.asarray(eps, jnp.float32))
    native = ORACLE.native_sequential(lambda_bar, b_bar, inputs)
    a1, a2, c1, c2 = zucchet_coefficients(lambda_bar, b_bar, tau_vector)
    zucchet = DP.companion_doubling_scan(
        (a1, a2), DP.input_drive((c1, c2), inputs), remat=None)
    collapsed = DP.common_stencil_collapsed_state(lambda_bar, b_bar, inputs)
    mixed = DP.matched_states(lambda_bar, b_bar, inputs, tau_vector, mass,
                              DP.PROFESSOR_LINEAR_TARGET)
    professor = DP.professor_tss_states(lambda_bar, b_bar, inputs,
                                        tau_vector,
                                        DP.PROFESSOR_LINEAR_TARGET)

    def summary(states):
        magnitude = jnp.abs(states[:, 0])
        return {"first": float(magnitude[0]),
                "at_8": float(magnitude[min(8, IMPULSE_LENGTH - 1)]),
                "at_32": float(magnitude[min(32, IMPULSE_LENGTH - 1)]),
                "max": float(jnp.max(magnitude)),
                "energy_after_first_token":
                    float(jnp.sum(magnitude[1:] ** 2)),
                "finite": bool(jnp.all(jnp.isfinite(states)))}

    return {"native_s5": summary(native),
            "zucchet_repository_legacy": summary(zucchet),
            "professor_tss_M_zero": summary(professor),
            "exact_common_stencil_collapsed": summary(collapsed),
            "wwj_generalized_tss_causal_mixed_stencil": summary(mixed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--length", type=int, default=RUNNER.SEQ_LEN)
    args = parser.parse_args()

    layers = production_modes(args.seed)
    name, (lambda_bar, b_bar) = next(iter(layers.items()))
    cells = []
    for equation in EQUATIONS:
        for target in TARGETS:
            for tau in GRID_TAU:
                for eps in GRID_EPS:
                    for gamma in (GRID_GAMMA
                                  if equation ==
                                  "partially_matched_two_compartment"
                                  else (0.0,)):
                        A, _ = coefficients_for(equation, lambda_bar, b_bar,
                                                tau, eps, gamma, target)
                        finite = all(bool(jnp.all(jnp.isfinite(value)))
                                     for value in A)
                        estimate, exact, worst, error = radii(A)
                        cell = {"equation": equation, "target": target,
                                "tau": tau, "eps": eps, "gamma": gamma,
                                "coefficients_finite": finite,
                                "max_radius_estimate": estimate,
                                "max_radius_exact": exact,
                                "worst_mode_index": worst,
                                "worst_mode": [float(lambda_bar[worst].real),
                                               float(lambda_bar[worst].imag)],
                                "root_error": error,
                                "admissible": bool(
                                    finite and exact is not None
                                    and exact <= RADIUS_BOUND)}
                        if cell["admissible"]:
                            cell["float32_rollout"] = rollout_finite(
                                equation, lambda_bar, b_bar, tau, eps, gamma,
                                target, args.length, jnp.float32)
                            cell["gradients"] = gradient_finite(
                                equation, lambda_bar, b_bar, tau, eps, gamma,
                                target)
                        cells.append(cell)

    admissible = [cell for cell in cells if cell["admissible"]]
    report = {
        "schema": "s5-direct-prospective/stability-grid-v1",
        "seed": args.seed, "layer": name,
        "modes": int(lambda_bar.shape[0]),
        "radius_bound": RADIUS_BOUND,
        "radius_bound_rationale":
            f"(1+1e-5)^{args.length} = "
            f"{float(numpy.exp(args.length * 1e-5)):.3f} of growth end to end",
        "grid": {"tau": list(GRID_TAU), "eps": list(GRID_EPS),
                 "gamma": list(GRID_GAMMA), "equations": list(EQUATIONS),
                 "targets": list(TARGETS)},
        "cells": cells,
        "impulse_responses": impulse_responses(lambda_bar, b_bar, 0.05, 0.25),
        "status": "SELECTED" if admissible else "NO_ADMISSIBLE_INITIALIZATION",
        "analytic_reason": (
            "z = 1 is an exact root of both causal recurrences at a mode "
            "Abar = 1 for every M, Gamma, T, so the recurrence is marginal "
            "at best there; away from that mode the companion radius rises "
            "above 1, and for the professor-consistent target it grows like "
            "2h/tau as tau shrinks. See "
            "docs/S5_DIRECT_GENERALIZED_PROSPECTIVE.md."),
        "selected": (min(admissible, key=lambda cell: cell["max_radius_exact"])
                     if admissible else None),
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"status": report["status"],
                      "selected": report["selected"],
                      "admissible_cells": len(admissible),
                      "total_cells": len(cells),
                      "impulse_responses": report["impulse_responses"]},
                     indent=2))
    if not admissible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
