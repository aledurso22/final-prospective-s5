"""Read-only derivation checks and real float32-initialized S5 mode inventory."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from experiments.s5_predictor_corrector.production_init import production_init_flat


def quant(x):
    return dict(zip(("min", "p5", "median", "p95", "max"),
                    map(float, np.percentile(np.asarray(x), [0, 5, 50, 95, 100]))))


def reference(kappa, alpha, ratio):
    tau = alpha / abs(kappa)
    divisor = 1 + ratio * tau * kappa
    k = kappa / divisor
    A = np.array([[0, 1/tau, 0], [-2*k, -1/tau, 1/tau],
                  [k, 0, -1/tau]], complex)
    b = np.array([0, 2, -1], complex) / divisor
    aug = np.zeros((4, 4), complex)
    aug[:3, :3], aug[:3, 3] = A, b
    E = expm(aug)
    return A, b, E[:3, :3], E[:3, 3], tau


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[301, 302])
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    rng = np.random.default_rng(616)
    q = np.exp(rng.uniform(-8, 8, 12000)) * np.exp(1j*rng.uniform(-1.57079, 1.57079, 12000))
    spectral = {}
    for nu in (0., .49, .5, 1.):
        roots = np.array([np.roots([1+nu*u, 2*(1+nu*u),
                                   1+(nu+2)*u, u]) for u in q])
        spectral[str(nu)] = {"n": len(q), "nonnegative_max_real": int(np.sum(roots.real.max(1) >= 0)),
                             "max_real": float(roots.real.max())}
    w = np.geomspace(1e-5, 1e5, 10000)
    z = 1j*w
    F = .5 + (2*z+1)/(z*(z+1)**2)
    identity_error = float(np.max(abs(F.real-(.5-2*w*w/(1+w*w)**2))))
    assert identity_error < 1e-9
    assert spectral["0.5"]["nonnegative_max_real"] == 0
    # Below the bound there is an explicit right-half-plane q with a unit imaginary root.
    nu = .49
    boundary_q = -1/((nu-.5)-1j)
    polynomial_at_i = (1+nu*boundary_q)*1j**3 + 2*(1+nu*boundary_q)*1j**2 + (1+(nu+2)*boundary_q)*1j + boundary_q
    assert boundary_q.real > 0 and abs(polynomial_at_i) < 1e-12
    records = []
    for seed in args.seeds:
        flat = production_init_flat("native_matched_s5", seed)
        ks = []
        for key in sorted(flat):
            if key[-1] != "Lambda_re":
                continue
            p = key[:-1]
            re = np.minimum(np.asarray(flat[key], float), -1e-4)
            im = np.asarray(flat[p+("Lambda_im",)], float)
            delta = np.exp(np.asarray(flat[p+("log_step",)], float)[:, 0])
            ks.extend(-(re+1j*im)*delta)
        ks = np.asarray(ks)
        for name, alpha, ratio in (("history", .1, 0.), ("proximal_matched", .1, .5),
                                    ("proximal_main", 1., .5)):
            max_re, radii, decay, dc_error, gain_ratio, tau_values = [], [], [], [], [], []
            for k in ks:
                A, b, Ad, bd, tau = reference(k, alpha, ratio)
                poles = np.linalg.eigvals(A)
                max_re.append(poles.real.max())
                radii.append(max(abs(np.linalg.eigvals(Ad))))
                decay.append(k.real / -poles.real.max())
                dc_error.append(abs((-np.linalg.solve(A,b)[0])*k-1))
                # Adaptive frequency grid resolves the native resonance; continuous transfer only.
                frequencies = -k.imag + k.real*np.linspace(-20, 20, 1001)
                s = 1j*frequencies
                ke = k/(1+ratio*tau*k)
                H = (1+2*tau*s)/((1+ratio*tau*k)*(tau*tau*s**3+2*tau*s*s+(1+2*tau*ke)*s+ke))
                gain_ratio.append(max(abs(H))*k.real)
                tau_values.append(tau)
            records.append({"seed": seed, "variant": name, "alpha": alpha, "proximal_ratio": ratio,
                "n_modes": len(ks), "continuous_unstable": int(np.sum(np.array(max_re)>=0)),
                "discrete_unstable": int(np.sum(np.array(radii)>=1)),
                "max_radius": max(radii), "slow_decay_time_vs_native": quant(decay),
                "continuous_dc_gain_relative_error_max": max(dc_error),
                "near_native_resonance_gain_vs_native_peak": quant(gain_ratio),
                "horizon_tokens": quant(tau_values)})
    report = {"scope": "No training; independent float64 equations on float32 production initialization.",
              "seeds": args.seeds, "stability_grid": spectral,
              "positive_real_identity_error": identity_error,
              "below_bound_boundary_example": {"ratio": nu, "q": [boundary_q.real, boundary_q.imag],
                                                "polynomial_error": abs(polynomial_at_i)},
              "initialized_modes": records}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    for row in records:
        print(row["seed"], row["variant"], "unstable", row["continuous_unstable"],
              "decay/native", row["slow_decay_time_vs_native"]["median"],
              "near-resonance gain/native", row["near_native_resonance_gain_vs_native_peak"]["median"])
    print("wrote", out)


if __name__ == "__main__":
    main()
