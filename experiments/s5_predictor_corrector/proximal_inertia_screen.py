"""Read-only analytical screen of unfiltered proximal inertial NLA.

Run from the repository root, using the production Python environment::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 OPENBLAS_NUM_THREADS=4 \
      python -m experiments.s5_predictor_corrector.proximal_inertia_screen \
      --out /path/to/new-screen.json

No model is trained and no dataset/checkpoint is opened. Native production
initialization is reconstructed in a separate float32 CPU process. All
candidate calculations are independent NumPy/SciPy float64 equations; they
do not verify a production GPU implementation of this proposed model.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

# This diagnostic deliberately reconstructs initialization on CPU only.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "0"

import numpy as np
from scipy.linalg import expm

from .checkpoint_compare import _complex, layer_groups, mode_quantities
from .production_init import production_init_flat

ROOT = Path(__file__).resolve().parents[2]
NATIVE = "native_matched_s5"
MASS_RATIO = 1.5
PROXIMAL_RATIO = 0.5


def quantiles(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or not values.size:
        raise ValueError("nonfinite or empty diagnostic summary")
    return dict(zip(("min", "p5", "median", "p95", "max"),
                    map(float, np.quantile(values, [0, .05, .5, .95, 1]))))


def geometric_sum(exponent, horizon):
    """Sum exp(n*exponent), n=0..horizon-1, including its exact zero limit."""
    exponent = np.asarray(exponent)
    return np.divide(np.expm1(horizon * exponent), np.expm1(exponent),
                     out=np.full(exponent.shape, horizon, dtype=exponent.dtype),
                     where=(exponent != 0))


def candidate_modes(kappa, alpha, mass_ratio=MASS_RATIO,
                    proximal_ratio=PROXIMAL_RATIO):
    """Physical-time poles and impulse residues for held physical input b.

    x/b = (1+tau*p) / [(1+mu*kappa)(m*p^2+p)+kappa(1+tau*p)].
    Continuous residues multiply exp(p*t); ZOH residues multiply exp(p*n)
    after sampling at the END of each unit-width input interval.
    """
    kappa = np.asarray(kappa, complex).reshape(-1)
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    if not np.isfinite(kappa).all() or not np.all(kappa.real > 0):
        raise ValueError("the screen requires finite Re(kappa)>0")
    tau = alpha / abs(kappa)
    chi = tau * kappa
    k = chi / (1 + proximal_ratio * chi)
    z = np.array([np.roots([mass_ratio, 1 + value, value]) for value in k])
    z = np.take_along_axis(z, np.argsort(z.real, axis=1)[:, ::-1], axis=1)
    if np.any(abs(z[:, 0] - z[:, 1]) < 1e-10):
        raise ValueError("near-repeated poles require a non-residue calculation")
    residues_z = ((1 + z) /
                  (2 * mass_ratio * z + 1 + k[:, None]))
    p = z / tau[:, None]
    residues_continuous = residues_z / (1 + proximal_ratio * chi[:, None])
    residues_zoh = residues_continuous * np.expm1(p) / p
    return p, residues_continuous, residues_zoh, tau


def component_energies(p, residues, B, C, horizon):
    """Real B/C-coupled energies per mode, retaining cross-root interference.

    These exclude cross-mode interference and are NOT additive fractions of
    whole-layer energy. The secondary-component numerator omits cross-root
    interference; its denominator contains the exact two-root mode response.
    """
    G = np.sum(abs(B) ** 2, axis=1) * np.sum(abs(C) ** 2, axis=1)
    H = np.sum(B ** 2, axis=1) * np.sum(C ** 2, axis=1)
    conjugate = geometric_sum(p[:, :, None] + p[:, None, :].conj(), horizon)
    ordinary = geometric_sum(p[:, :, None] + p[:, None, :], horizon)
    abs_energy = np.einsum("pi,pj,pij->p", residues, residues.conj(), conjugate).real
    square_sum = np.einsum("pi,pj,pij->p", residues, residues, ordinary)
    total = 2 * (G * abs_energy + (H * square_sum).real)
    sec_abs = abs(residues[:, 1]) ** 2 * geometric_sum(2 * p[:, 1].real, horizon).real
    sec_square = residues[:, 1] ** 2 * geometric_sum(2 * p[:, 1], horizon)
    secondary = 2 * (G * sec_abs + (H * sec_square).real)
    if (not np.isfinite(total).all() or not np.isfinite(secondary).all()
            or np.any(total <= 0) or np.any(secondary < 0)):
        raise ValueError("invalid component energy; no clipping is performed")
    return total, secondary


def numerical_oracles():
    """Independent ZOH, dense-energy and exact-cancellation checks."""
    max_impulse_error = 0.
    max_energy_relative_error = 0.
    rng = np.random.default_rng(241118)
    for kappa in (.01 + .3j, .5 + .2j, .001 + 1j):
        for alpha in (.1, 1.):
            p, _, d, tau = candidate_modes([kappa], alpha)
            tau = float(tau[0])
            mu, mass = PROXIMAL_RATIO * tau, MASS_RATIO * tau
            a, forcing = kappa / (1 + mu * kappa), 1 / (1 + mu * kappa)
            # v=m*x_dot+tau*r_mu; no derivative of the held input is taken.
            A = np.array([[-tau * a / mass, 1 / mass],
                          [(tau / mass - 1) * a, -1 / mass]], complex)
            b = forcing * np.array([tau / mass, 1 - tau / mass])
            augmented = np.zeros((3, 3), complex)
            augmented[:2, :2], augmented[:2, 2] = A, b
            discrete = expm(augmented)
            state = discrete[:2, 2].copy()
            values = []
            for lag in range(64):
                analytic = np.sum(d[0] * np.exp(p[0] * lag))
                max_impulse_error = max(max_impulse_error, abs(state[0] - analytic))
                values.append(analytic)
                state = discrete[:2, :2] @ state
            B = rng.normal(size=(1, 2)) + 1j * rng.normal(size=(1, 2))
            C = rng.normal(size=(1, 3)) + 1j * rng.normal(size=(1, 3))
            total, secondary = component_energies(p, d, B, C, 64)
            h_secondary = d[0, 1] * np.exp(p[0, 1] * np.arange(64))
            for h, expected in ((np.asarray(values), total[0]),
                                (h_secondary, secondary[0])):
                dense = 2 * (h[:, None, None] * C[0, None, :, None] *
                             B[0, None, None, :]).real
                exact = np.sum(dense ** 2)
                max_energy_relative_error = max(max_energy_relative_error,
                                                abs(exact - expected) / exact)
    z = np.array([.1 + .2j, .4 + 2j, 2. + .3j])
    k = (.01 + .8j) / (1 + .5 * (.01 + .8j))
    cancelled = (1 + z) / (z * z + (1 + k) * z + k)
    first_order = 1 / (z + k)
    cancellation_error = float(np.max(abs(cancelled - first_order)))
    if max_impulse_error > 1e-10 or max_energy_relative_error > 1e-10 or cancellation_error > 1e-12:
        raise AssertionError("independent numerical oracle failed")
    return {"augmented_expm_ZOH_impulse_max_abs_error": float(max_impulse_error),
            "dense_real_kernel_energy_max_relative_error": float(max_energy_relative_error),
            "mass_equals_tau_transfer_cancellation_max_abs_error": cancellation_error}


def parameter_hash(flat):
    digest = hashlib.sha256()
    for key, value in sorted(flat.items()):
        array = np.ascontiguousarray(value)
        digest.update("/".join(key).encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def screen_seed(seed, alphas, horizon):
    flat = production_init_flat(NATIVE, seed)
    groups = layer_groups(flat, NATIVE, f"reconstructed seed {seed}")
    kappas, Bs, C1s, C2s = [], [], [], []
    for group in groups.values():
        modes = mode_quantities(group)
        kappas.extend(-modes["LambdaDelta"])
        Bs.extend(modes["Delta"][:, None] * _complex(group["B"]))
        C1s.extend(_complex(group["C1"]).T)
        C2s.extend(_complex(group["C2"]).T)
    kappa, B, C1, C2 = map(np.asarray, (kappas, Bs, C1s, C2s))
    rows = []
    for alpha in alphas:
        p, _, d, tau = candidate_modes(kappa, alpha)
        fractions = []
        for C in (C1, C2):
            total, secondary = component_energies(p, d, B, C, horizon)
            fractions.extend(secondary / total)
        rows.append({
            "alpha": alpha,
            "n_complex_channels": len(kappa),
            "continuous_poles_with_nonnegative_real_part": int(np.sum(p.real >= 0)),
            "max_discrete_radius_float64": float(np.exp(p.real).max()),
            "dominant_decay_over_native": quantiles(kappa.real / -p[:, 0].real),
            "secondary_decay_samples": quantiles(1 / -p[:, 1].real),
            "secondary_component_energy_over_exact_mode_energy": quantiles(fractions),
            "secondary_over_dominant_ZOH_residue_magnitude": quantiles(abs(d[:, 1]) / abs(d[:, 0])),
            "tau_samples": quantiles(tau),
        })
    return {"seed": seed, "reconstructed_native_full_parameter_sha256": parameter_hash(flat),
            "layer_names": list(groups), "variants": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[301, 302])
    parser.add_argument("--alphas", type=float, nargs="+", default=[.1, 1.])
    parser.add_argument("--horizon", type=int, default=16000)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.horizon <= 0:
        raise SystemExit("horizon must be positive")
    sources = [Path(__file__), ROOT / "experiments/s5_predictor_corrector/production_init.py",
               ROOT / "experiments/s5_predictor_corrector/checkpoint_compare.py",
               ROOT / "experiments/s5_three_arm_full/runner.py"]
    report = {
        "scope": "NO TRAINING. Float64 analytical proposal evaluated on current reconstructed float32 native CPU initialization; not saved historical epoch-zero or trained checkpoints, not a tested GPU constructor.",
        "equations": "m*x_ddot+x_dot=-r_mu-tau*r_mu_dot; r_mu=(kappa*x-b)/(1+mu*kappa); tau=alpha/abs(kappa); m=1.5*tau; mu=0.5*tau",
        "mass_ratio": MASS_RATIO, "proximal_ratio": PROXIMAL_RATIO,
        "horizon_samples": args.horizon,
        "metric_scope": "Per-mode real B/C-coupled finite-window impulse energies. Both directions pooled for summaries. Exact cross-root interference in denominator; no cross-mode interference. Ratios are not additive shares of layer energy. Pole ordering is by decay rate, not a claim of mode identity across parameters.",
        "limitations": [
            "Initialization agreement is not historical reconstruction proof.",
            "Both seeds share the deterministic initial Lambda spectrum; matching decay-ratio summaries are not independent evidence.",
            "Near-native dominant decay at alpha=0.1 is an initialization result, not a uniform statement for trained kappa approaching the imaginary axis.",
            "Positive component energy does not establish task-relevant memory or an accuracy gain.",
            "No assertion is made that this is a mass-only change from calibrated arm C or a literal FOOF optimizer.",
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "jax", "jaxlib", "flax")},
                        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},
        "oracles": numerical_oracles(),
        "seeds": [screen_seed(seed, args.alphas, args.horizon) for seed in args.seeds],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
