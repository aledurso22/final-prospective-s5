"""Whole-inventory certification of the two-compartment recurrence.

Every mode of every layer, at the PRODUCTION configuration and the
production seeds 301, 302 and 303. A subset is not certification: three
earlier errors on this project came from measuring a handful of modes and
calling the result production-eligible.

What it reports, per seed and per layer and then over the whole inventory:

  * both companion roots of every mode, and the worst spectral radius;
  * float64 agreement between the sequential oracle and each parallel
    route;
  * float32 relative error and finiteness for the same comparison;
  * the discriminant's distance from zero, so a near-repeated root cannot
    hide inside an aggregate.

    python -m experiments.s5_two_compartment.certify_inventory --out cert.json
"""

import argparse
import json

import jax
import jax.numpy as np
import numpy as onp
from flax.traverse_util import flatten_dict

from s5 import factored_recurrence as FR
from s5.discrete_recurrence import (companion_radius, generalized_coefficients,
                                    scan_companion, scan_companion_sequential)
from s5.generalized_prospective_ssm import response_mass_gamma
from s5.ssm import discretize_zoh

ARM = "generalized_prospective_s5"
#: the production seeds, as the runner declares them
SEEDS = (301, 302, 303)
#: the radius bound. The recurrence is only usable if every mode is
#: strictly inside the unit disc over the whole inventory.
RADIUS_BOUND = 1.0


def _layer_groups(params):
    """Every SSM layer's parameters, keyed by its module path.

    Found by locating `generalized_T_raw`, so the grouping follows the
    actual parameter tree rather than an assumed layout.
    """
    flat = flatten_dict(params)
    groups = {}
    for path, value in flat.items():
        if path[-1] == "generalized_T_raw":
            groups["/".join(path[:-1])] = {}
    for path, value in flat.items():
        prefix = "/".join(path[:-1])
        if prefix in groups:
            groups[prefix][path[-1]] = value
    return groups


def _coefficients_for(group, clip_eigs=True):
    """Reproduce the layer's a1, a2 exactly as `setup` does."""
    lambda_re, lambda_im = group["Lambda_re"], group["Lambda_im"]
    if clip_eigs:
        lambda_value = np.clip(lambda_re, None, -1e-4) + 1j * lambda_im
    else:
        lambda_value = lambda_re + 1j * lambda_im
    b_tilde = group["B"][..., 0] + 1j * group["B"][..., 1]
    step = np.exp(group["log_step"][:, 0])
    lambda_bar, b_bar = discretize_zoh(lambda_value, b_tilde, step)
    T, mass, gamma, _ = response_mass_gamma(group["generalized_T_raw"],
                                            group["generalized_rho_raw"],
                                            group["generalized_gamma_raw"])
    a1, a2, c1, c2 = generalized_coefficients(lambda_bar, b_bar, T, mass,
                                              gamma)
    return a1, a2, c1, c2


def x64_enabled():
    return bool(jax.config.read("jax_enable_x64"))


def _compare(a1, a2, c1, c2, length, key, dtype):
    """Sequential oracle against both parallel routes, at one precision.

    REGRESSION. The first run reported a float64 column identical to the
    float32 one to every digit, because x64 was off and
    `astype(complex128)` silently truncated back to complex64 -- JAX even
    said so in a warning that was not read. A precision gate that cannot
    detect its own absence is worse than none, so requesting float64
    without x64 is now an error rather than a quiet downgrade.
    """
    if dtype == "float64" and not x64_enabled():
        raise SystemExit(
            "float64 was requested but JAX x64 is OFF, so complex128 would "
            "be truncated to complex64 and the column would silently "
            "measure float32. Re-run with JAX_ENABLE_X64=1.")
    values = jax.random.normal(key, (length, c1.shape[-1]), dtype=np.float32)
    cast = (lambda v: v.astype(np.complex128)) if dtype == "float64" \
        else (lambda v: v.astype(np.complex64))
    a1c, a2c, c1c, c2c = cast(a1), cast(a2), cast(c1), cast(c2)
    oracle = scan_companion_sequential(a1c, a2c, c1c, c2c, values)
    scale = float(np.max(np.abs(oracle))) or 1.0
    out = {"oracle_finite": bool(np.all(np.isfinite(np.abs(oracle)))),
           "oracle_max_abs": float(np.max(np.abs(oracle)))}
    for name, scan in (("companion", scan_companion),
                       ("factored", FR.scan_factored)):
        states = scan(a1c, a2c, c1c, c2c, values)
        out[name] = {
            "relative_error": float(np.max(np.abs(oracle - states))) / scale,
            "finite": bool(np.all(np.isfinite(np.abs(states)))),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--length", type=int, default=16000,
                        help="production sequence length")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--precision", choices=("float32", "float64", "both"),
                        default="float32",
                        help="float64 and both REQUIRE JAX_ENABLE_X64=1 and "
                             "fail loudly without it")
    arguments = parser.parse_args()

    from experiments.s5_three_arm_full import runner

    precisions = (("float32",) if arguments.precision == "float32"
                  else ("float64",) if arguments.precision == "float64"
                  else ("float64", "float32"))
    if "float64" in precisions and not x64_enabled():
        raise SystemExit(
            "float64 certification requires JAX_ENABLE_X64=1; refusing to "
            "report a float64 column measured in float32.")
    report = {"schema": "s5-two-compartment/certification-v2",
              "arm": ARM, "seeds": arguments.seeds,
              "length": arguments.length, "precisions": list(precisions),
              "x64_enabled": x64_enabled(),
              "radius_bound": RADIUS_BOUND, "seeds_detail": {}}
    worst_radius, worst_float32, worst_float64 = 0.0, 0.0, 0.0
    total_modes, offending = 0, []

    for seed in arguments.seeds:
        state = runner.init_state(ARM, seed)
        groups = _layer_groups(state.params)
        assert groups, "no generalized layers found in the parameter tree"
        seed_detail = {}
        for name, group in sorted(groups.items()):
            a1, a2, c1, c2 = _coefficients_for(group)
            major, minor = FR.companion_roots(a1, a2)
            radius = onp.maximum(onp.abs(onp.asarray(major)),
                                 onp.abs(onp.asarray(minor)))
            existing = onp.asarray(companion_radius(a1, a2))
            discriminant = onp.abs(onp.asarray(FR.discriminant(a1, a2)))
            total_modes += int(radius.size)
            for index in range(radius.size):
                if radius[index] >= RADIUS_BOUND:
                    offending.append({"seed": seed, "layer": name,
                                      "mode": index,
                                      "radius": float(radius[index])})
            key = jax.random.PRNGKey(seed)
            measured = {name: _compare(a1, a2, c1, c2, arguments.length, key,
                                       name) for name in precisions}
            comparison64 = measured.get("float64")
            comparison32 = measured.get("float32")
            worst_radius = max(worst_radius, float(radius.max()))
            if comparison64:
                worst_float64 = max(worst_float64,
                                    max(comparison64[k]["relative_error"]
                                        for k in ("companion", "factored")))
            if comparison32:
                worst_float32 = max(worst_float32,
                                    max(comparison32[k]["relative_error"]
                                        for k in ("companion", "factored")))
            seed_detail[name] = {
                "modes": int(radius.size),
                "max_spectral_radius": float(radius.max()),
                "max_abs_difference_from_companion_radius":
                    float(onp.max(onp.abs(radius - existing))),
                "min_abs_discriminant": float(discriminant.min()),
                "roots_major_abs": [float(v) for v in
                                    onp.abs(onp.asarray(major))],
                "roots_minor_abs": [float(v) for v in
                                    onp.abs(onp.asarray(minor))],
                "float64": comparison64,
                "float32": comparison32,
            }
            shown = "  ".join(
                f"{label} {max(row[k]['relative_error'] for k in ('companion', 'factored')):.3e}"
                for label, row in measured.items())
            print(f"seed {seed} {name:52s} modes {radius.size:4d} "
                  f"rho_max {radius.max():.6f}  {shown}", flush=True)
        report["seeds_detail"][str(seed)] = seed_detail

    report.update({
        "total_modes_certified": total_modes,
        "worst_spectral_radius": worst_radius,
        "worst_float64_relative_error": (worst_float64
                                         if "float64" in precisions else None),
        "worst_float32_relative_error": (worst_float32
                                         if "float32" in precisions else None),
        "offending_modes": offending,
        "certified": bool(not offending),
    })
    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("x64_enabled", "precisions", "total_modes_certified",
                       "worst_spectral_radius",
                       "worst_float64_relative_error",
                       "worst_float32_relative_error", "offending_modes",
                       "certified")}, indent=2))


if __name__ == "__main__":
    main()
