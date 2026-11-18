"""Proximal inertial S5 (candidate) and its exact first-order control.

    proximal_inertial_s5          m = 1.5 tau  (releases the inertial mode)
    proximal_inertial_control_s5  m = tau      (exactly first-order proximal flow)

Both use tau = 0.1/|kappa|, mu = tau/2, kappa = -Lambda*Delta, and the native
S5 parameters, readout, outer optimizer, batch 16, 15 epochs, split and
selection rule. The tested, smoke-gated history runner supplies everything
except the recurrence; registration is scoped to this process and restored
on exit (the same mechanism as the proximal-history arm P).

The inherited per-step key ``companion_spectral_radius`` reports the actual
2x2 block transition's spectral radius for these arms.
"""
import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.pole_diagnostics import BOUNDARY_TOL, classify
from s5 import proximal_inertia as PI

#: arm -> mass_ratio; everything else is shared and fixed
ARMS = {"proximal_inertial_s5": PI.MASS_RATIO,
        "proximal_inertial_control_s5": PI.CONTROL_MASS_RATIO}
ALPHA, PROXIMAL_RATIO = PI.ALPHA, PI.PROXIMAL_RATIO
RECURRENCE = "unfiltered proximal inertial NLA; derivative-free (s, w) coordinates; exact physical-input ZOH"


def transitions(params, mass_ratio):
    return jnp.concatenate([PI.proximal_inertia_zoh(k, ALPHA, mass_ratio, PROXIMAL_RATIO)[0]
                            for _, k in H._kappas(params)], axis=0)


def reference_blocks(kappa, mass_ratio, alpha=ALPHA, proximal_ratio=PROXIMAL_RATIO):
    """Float64 SciPy (Ad, bd, continuous A) for one mode, written independently."""
    tau = alpha / abs(kappa)
    mu, m = proximal_ratio * tau, mass_ratio * tau
    d = 1 + mu * kappa
    A = np.array([[-tau * kappa / (m * d), 1 / m],
                  [(tau / m - 1) * kappa / d, -1 / m]], np.complex128)
    g = np.array([tau / m, 1 - tau / m], np.complex128) / d
    aug = np.zeros((3, 3), np.complex128)
    aug[:2, :2], aug[:2, 2] = A, g
    E = expm(aug)
    return E[:2, :2], E[:2, 2], A


def pole_summary(params, arm):
    mass_ratio = ARMS[arm]
    host = jax.device_get(params)
    host64 = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), host)
    kappas = np.concatenate([k for _, k in H._kappas(host64, np)])
    mats32 = np.asarray(jax.device_get(transitions(params, mass_ratio)))
    poles32 = np.linalg.eigvals(mats32.astype(np.complex128))
    radius32 = np.max(np.abs(poles32), axis=-1)
    radius_solver = np.max(np.abs(np.asarray(jax.device_get(
        jnp.linalg.eigvals(jnp.asarray(mats32))))), axis=-1)
    radius64, cont, dom, aux = [], [], [], []
    for k in kappas:
        Ad, _, A = reference_blocks(k, mass_ratio)
        z = np.linalg.eigvals(Ad)
        z = z[np.argsort(-abs(z))]
        radius64.append(abs(z[0])); dom.append(abs(z[0])); aux.append(abs(z[1]))
        cont.append(np.exp(np.max(np.linalg.eigvals(A).real)))
    radius64, dom, aux = map(np.asarray, (radius64, dom, aux))
    ok, edge, bad = classify(radius32, radius64)
    finite = np.isfinite(radius32) & np.isfinite(radius64)
    ok &= finite; edge &= finite; bad &= finite
    native = np.abs(np.exp(-kappas))
    with np.errstate(divide="ignore", invalid="ignore"):
        dom_ratio = np.log(native) / np.log(dom)
        aux_decay = -1 / np.log(aux)
    return {"arm": arm, "n_modes": int(len(kappas)), "history_alpha": ALPHA,
            "mass_ratio": mass_ratio, "proximal_ratio": PROXIMAL_RATIO,
            "scan_implementation": "history_stationary_block", "block_width": PI.WIDTH,
            "radius_semantics": "actual exact-ZOH 2x2 (s, w) transition; inherited per-step key is companion_spectral_radius",
            "stable": int(np.sum(finite & (radius64 < 1))),
            "unstable": int(np.sum(finite & (radius64 >= 1))),
            "nonfinite_modes": int(np.sum(~finite)),
            "min_margin": float(1 - radius64.max()),
            "dominant_decay_over_native": H._quant(dom_ratio),
            "auxiliary_decay_tokens": H._quant(aux_decay),
            "radius_reference": H._quant(radius64),
            "max_expm_vs_generator_radius_difference": float(np.max(np.abs(radius64 - np.asarray(cont)))),
            "production_precision": {
                "dtype": "complex64", "boundary_tol": BOUNDARY_TOL,
                "stable": int(ok.sum()), "numerical_boundary": int(edge.sum()),
                "unstable": int(bad.sum()), "nonfinite": int((~finite).sum()),
                "max_radius_production": float(radius32.max()),
                "max_radius_production_eigensolver": float(radius_solver.max()),
                "max_radius_float64": float(radius64.max()),
                "max_abs_radius_difference_production_vs_float64": float(np.max(abs(radius32 - radius64))),
                "max_eigensolver_radius_difference": float(np.max(abs(radius_solver - radius32))),
                "min_margin_production": float(1 - radius32.max())}}


def _check_arm(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown proximal inertial arm {arm!r}")


@contextmanager
def configured(arm):
    """Scope every patch to this process; restore old arms and defaults."""
    _check_arm(arm)
    mass_ratio = ARMS[arm]
    old = {key: getattr(H, key) for key in (
        "ARM", "ALPHA", "NormalizedGradientHistoryS5SSM", "production_transitions",
        "history_pole_summary", "protocol_identity", "registered_history")}

    def identity(data_cache, seed):
        result = old["protocol_identity"](data_cache, seed)
        result.update(mass_ratio=mass_ratio, proximal_ratio=PROXIMAL_RATIO,
                      block_width=PI.WIDTH, horizon_normalization="magnitude",
                      recurrence=RECURRENCE, outer_optimizer="unchanged")
        return result

    @contextmanager
    def registered():
        with old["registered_history"]():
            saved = (R.recurrent_state_size, R.SCIENTIFIC_NAMES, R.ARM_CONFIGS)
            base_size = R.recurrent_state_size

            def size(a):
                if a != arm:
                    return base_size(a)
                return {"complex_modes": R.SSM_SIZE_BASE // 2,
                        "real_values_total": R.N_LAYERS * 2 * R.SSM_SIZE_BASE,
                        "per_layer_real_values": 2 * R.SSM_SIZE_BASE,
                        "states_per_complex_mode": 2,
                        "note": "per direction; bidirectional scans have separate zero starts"}
            R.recurrent_state_size = size
            R.SCIENTIFIC_NAMES = {**R.SCIENTIFIC_NAMES, arm:
                f"proximal inertial S5; alpha={ALPHA}, mass_ratio={mass_ratio}, "
                f"proximal_ratio={PROXIMAL_RATIO}; exact ZOH"}
            R.ARM_CONFIGS = {**R.ARM_CONFIGS, arm: {
                "shared": R.SHARED_CONFIG, "recurrence_constructor": "ProximalInertialS5SSM",
                "extra_parameters": (), "state_dimension": 2 * R.SSM_SIZE_BASE,
                "history_alpha": ALPHA, "mass_ratio": mass_ratio,
                "proximal_ratio": PROXIMAL_RATIO}}
            try:
                yield
            finally:
                R.recurrent_state_size, R.SCIENTIFIC_NAMES, R.ARM_CONFIGS = saved

    H.ARM, H.ALPHA = arm, ALPHA
    H.NormalizedGradientHistoryS5SSM = partial(PI.ProximalInertialS5SSM, mass_ratio=mass_ratio,
                                              proximal_ratio=PROXIMAL_RATIO)
    H.production_transitions = lambda params: transitions(params, mass_ratio)
    H.history_pole_summary = lambda params: pole_summary(params, arm)
    H.protocol_identity = identity
    H.registered_history = registered
    try:
        yield arm
    finally:
        for key, value in old.items():
            setattr(H, key, value)


SOURCES = ("s5/proximal_inertia.py", "experiments/s5_proximal_inertia/runner.py")


def memory_profile(arm, seed, out):
    from experiments.s5_gradient_history import memory_profile as MP
    out = Path(out)
    if out.exists():
        raise ValueError(f"refusing existing output: {out}")
    with configured(arm):
        old_arms = MP.ARMS
        MP.ARMS = tuple(dict.fromkeys(old_arms + (arm,)))
        try:
            result = MP.compile_profile(arm, seed=seed, history_scan="parallel")
        finally:
            MP.ARMS = old_arms
    root = Path(__file__).resolve().parents[2]
    result.update(mass_ratio=ARMS[arm], proximal_ratio=PROXIMAL_RATIO, history_alpha=ALPHA)
    for rel in SOURCES:
        result["source_sha256"][rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["updates"]:
        print(row["kind"], row["memory"], flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=tuple(ARMS), required=True)
    p.add_argument("--seed", type=int, choices=(301, 302), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-cache")
    p.add_argument("--smoke-report")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--profile", action="store_true")
    args = p.parse_args(argv)
    if args.profile:
        memory_profile(args.arm, args.seed, args.out)
        return
    if not args.data_cache:
        p.error("--data-cache is required for smoke/train")
    forwarded = ["--data-cache", args.data_cache, "--out", args.out, "--seed", str(args.seed),
                 "--smoke" if args.smoke else "--train"]
    if args.smoke_report:
        forwarded += ["--smoke-report", args.smoke_report]
    with configured(args.arm):
        H.main(forwarded)
    for name in ("production_check.json", "smoke_result.json", "task_result.json"):
        path = Path(args.out) / name
        if path.exists():
            result = json.loads(path.read_text())
            result.update(history_alpha=ALPHA, mass_ratio=ARMS[args.arm],
                          proximal_ratio=PROXIMAL_RATIO, block_width=PI.WIDTH,
                          recurrence=RECURRENCE, outer_optimizer="unchanged")
            path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
