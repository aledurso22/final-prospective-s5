"""Per-mode diagnostics for the theory-matched zero-lag arm.

Answers, over the PRODUCTION inventory at seeds 301-303 -- every mode of
every layer, not a subset:

  (3) the distribution and magnitude of Gamma_k;
  (4) how it depends on |lambda_bar| and on phase, especially near-real
      slow modes, where Gamma_k ~ gamma h /(T |1 - lambda_bar|) diverges;
  (5) gradient statistics along the prospective pathway, because after the
      regrouping the remaining concern is optimization scale, not
      arithmetic;
  (7) the low-frequency group delay of BOTH arms, which is the direct
      lag diagnostic: the current arm should be positive and the matched
      arm approximately zero.

It also re-checks (2): a1 and a2 are bit-identical between the arms.

    python -m experiments.s5_two_compartment.matched_lag_report --out m.json
"""

import argparse
import json

import jax
import jax.numpy as np
import numpy as onp

from s5 import matched_lag_ssm as ML
from s5.discrete_recurrence import generalized_coefficients
from s5.generalized_prospective_ssm import response_mass_gamma
from s5.ssm import discretize_zoh
from experiments.s5_two_compartment.certify_inventory import (ARM, SEEDS,
                                                              _layer_groups)

H = 1.0


def _quantities(group, clip_eigs=True):
    """lambda_bar, b_bar and both arms' coefficients, from one layer."""
    lambda_re, lambda_im = group["Lambda_re"], group["Lambda_im"]
    value = (np.clip(lambda_re, None, -1e-4) if clip_eigs else lambda_re) \
        + 1j * lambda_im
    b_tilde = group["B"][..., 0] + 1j * group["B"][..., 1]
    lambda_bar, b_bar = discretize_zoh(value, b_tilde,
                                       np.exp(group["log_step"][:, 0]))
    T, mass, gamma, _ = response_mass_gamma(group["generalized_T_raw"],
                                            group["generalized_rho_raw"],
                                            group["generalized_gamma_raw"])
    a1, a2, c1, c2 = generalized_coefficients(lambda_bar, b_bar, T, mass,
                                              gamma)
    gamma_k = ML.matched_gamma(lambda_bar, T, gamma, H)
    gain = ML.matched_gain(b_bar, T, mass, gamma, H)
    return lambda_bar, b_bar, a1, a2, c1, c2, gamma_k, gain


def _group_delay(a1, a2, first, second, omega=1e-3, step=1e-4):
    """-dphi/domega at low frequency, per mode, for a drive c1 + c2 z^-1.

    Positive is LAG. Computed on the FIRST input channel only; the phase of
    a single-input-channel transfer is what the lag question is about.
    """
    def phase(w):
        z = onp.exp(1j * w)
        top = onp.asarray(first)[:, 0] + onp.asarray(second)[:, 0] / z
        bottom = 1.0 - onp.asarray(a1) / z - onp.asarray(a2) / (z * z)
        return onp.angle(top / bottom)

    ahead, behind = phase(omega + step), phase(omega - step)
    # unwrap the pair so a branch cut does not masquerade as a huge delay
    difference = onp.angle(onp.exp(1j * (ahead - behind)))
    return -difference / (2 * step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--length", type=int, default=2048,
                        help="sequence length for the gradient probe")
    arguments = parser.parse_args()

    from experiments.s5_three_arm_full import runner

    report = {"schema": "s5-two-compartment/matched-lag-report-v1",
              "arm": ARM, "seeds": arguments.seeds, "h": H, "seeds_detail": {}}
    every_gamma, every_magnitude, every_phase = [], [], []
    lag_current, lag_matched = [], []
    identical = True

    for seed in arguments.seeds:
        state = runner.init_state(ARM, seed)
        detail = {}
        for name, group in sorted(_layer_groups(state.params).items()):
            lam, b_bar, a1, a2, c1, c2, gamma_k, gain = _quantities(group)
            # (2) a1, a2 are shared by construction; re-derive and compare
            again = generalized_coefficients(
                lam, b_bar, *response_mass_gamma(
                    group["generalized_T_raw"], group["generalized_rho_raw"],
                    group["generalized_gamma_raw"])[:3])
            identical = identical and bool(np.all(a1 == again[0])) \
                and bool(np.all(a2 == again[1]))

            magnitude = onp.abs(onp.asarray(gamma_k))
            lam_abs = onp.abs(onp.asarray(lam))
            lam_phase = onp.angle(onp.asarray(lam))
            every_gamma.extend(magnitude.tolist())
            every_magnitude.extend(lam_abs.tolist())
            every_phase.extend(onp.abs(lam_phase).tolist())

            d1, d2 = ML.direct_coefficients(gain, gamma_k, H)
            cur = _group_delay(a1, a2, c1, c2)
            mat = _group_delay(a1, a2, d1, d2)
            lag_current.extend(cur.tolist())
            lag_matched.extend(mat.tolist())

            detail[name] = {
                "modes": int(magnitude.size),
                "gamma_k_abs": {"min": float(magnitude.min()),
                                "median": float(onp.median(magnitude)),
                                "max": float(magnitude.max())},
                "lambda_abs": {"min": float(lam_abs.min()),
                               "max": float(lam_abs.max())},
                "min_abs_one_minus_lambda":
                    float(onp.abs(1.0 - onp.asarray(lam)).min()),
                "group_delay_current": {"median": float(onp.median(cur)),
                                        "max": float(cur.max())},
                "group_delay_matched": {"median": float(onp.median(mat)),
                                        "max": float(onp.abs(mat).max())},
            }
            print(f"seed {seed} {name:34s} modes {magnitude.size:3d}  "
                  f"|Gk| {magnitude.min():9.1f}..{magnitude.max():11.1f}  "
                  f"lag cur {onp.median(cur):8.2f}  mat {onp.median(mat):7.2f}",
                  flush=True)

            # (5) gradient statistics along the prospective pathway
            raw = (group["generalized_T_raw"], group["generalized_rho_raw"],
                   group["generalized_gamma_raw"])
            values = jax.random.normal(jax.random.PRNGKey(seed),
                                       (arguments.length, b_bar.shape[-1]),
                                       dtype=np.float32)

            def make_loss(matched):
                def loss(t_raw, rho_raw, gamma_raw):
                    T, mass, gamma, _ = response_mass_gamma(t_raw, rho_raw,
                                                            gamma_raw)
                    A1, A2, C1, C2 = generalized_coefficients(lam, b_bar, T,
                                                              mass, gamma)
                    if matched:
                        gk = ML.matched_gamma(lam, T, gamma, H)
                        gn = ML.matched_gain(b_bar, T, mass, gamma, H)
                        states = ML.apply_matched(A1, A2, gn, gk, values, H,
                                                  implementation="factored")
                    else:
                        from s5.factored_recurrence import scan_factored
                        states = scan_factored(A1, A2, C1, C2, values)
                    return np.mean(np.abs(states) ** 2)
                return loss

            for label, matched in (("current", False), ("matched", True)):
                grads = jax.grad(make_loss(matched), argnums=(0, 1, 2))(*raw)
                per_mode = onp.abs(onp.asarray(grads[0]))
                finite = bool(onp.all(onp.isfinite(per_mode)))
                spread = (float(per_mode.max()
                                / max(float(per_mode.min()), 1e-30)))
                detail[name][f"gradient_{label}"] = {
                    "T_raw_abs_min": float(per_mode.min()),
                    "T_raw_abs_median": float(onp.median(per_mode)),
                    "T_raw_abs_max": float(per_mode.max()),
                    "across_mode_ratio": spread,
                    "decades": float(onp.log10(max(spread, 1.0))),
                    "finite": finite,
                }
        report["seeds_detail"][str(seed)] = detail

    gamma = onp.asarray(every_gamma)
    mag = onp.asarray(every_magnitude)
    phase = onp.asarray(every_phase)
    report.update({
        "total_modes": int(gamma.size),
        "a1_a2_identical_between_arms": bool(identical),
        "gamma_k_abs": {
            "min": float(gamma.min()), "median": float(onp.median(gamma)),
            "p90": float(onp.percentile(gamma, 90)),
            "p99": float(onp.percentile(gamma, 99)),
            "max": float(gamma.max()),
            "decades": float(onp.log10(gamma.max()
                                       / max(gamma.min(), 1e-30)))},
        "gamma_k_by_phase_decile": [
            {"phase_below": float(edge),
             "median_gamma_k": float(onp.median(gamma[phase <= edge]))
             if bool((phase <= edge).any()) else None}
            for edge in (0.01, 0.05, 0.2, 0.5, 1.0, 3.2)],
        "gamma_k_by_lambda_magnitude": [
            {"lambda_abs_above": float(edge),
             "median_gamma_k": float(onp.median(gamma[mag >= edge]))
             if bool((mag >= edge).any()) else None}
            for edge in (0.0, 0.9, 0.99, 0.999)],
        "group_delay_current": {
            "median": float(onp.median(lag_current)),
            "max": float(onp.max(lag_current))},
        "group_delay_matched": {
            "median": float(onp.median(lag_matched)),
            "max_abs": float(onp.max(onp.abs(lag_matched)))},
    })
    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("total_modes", "a1_a2_identical_between_arms",
                       "gamma_k_abs", "gamma_k_by_phase_decile",
                       "gamma_k_by_lambda_magnitude",
                       "group_delay_current", "group_delay_matched")},
                     indent=2))


if __name__ == "__main__":
    main()
