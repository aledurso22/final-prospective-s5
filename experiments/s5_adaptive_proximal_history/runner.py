"""A nested, smoke-gated test of learning P's per-mode horizon multipliers.

The candidate and control contain the same zero-initialized scalar per complex
mode. Only the candidate updates it. The shared parameters retain their original
optimizer groups; the new candidate leaf uses log_step's no-decay SSM Adam group.
The control additionally assigns the leaf to the zero-learning-rate group.

All registration and optimizer-label changes are scoped to this entry point.
Batching, dropout, checkpointing, validation selection and the 15-epoch schedule
come from the existing runner. No test split is read and no previous run is edited.
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
from flax.traverse_util import flatten_dict
from scipy.linalg import expm

from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.pole_diagnostics import classify, BOUNDARY_TOL
from s5 import train_helpers as TH
from s5.adaptive_proximal_history import (
    AdaptiveProximalHistoryS5SSM, adaptive_proximal_history_zoh,
)

ARM = "adaptive_proximal_history_s5"
CONTROL = "adaptive_proximal_history_control_s5"
ARMS = (ARM, CONTROL)
ALPHA = 1.0
PROXIMAL_RATIO = 0.5
HORIZON_KEY = "log_horizon_scale"


def validate_arm(arm):
    if arm not in ARMS:
        raise ValueError(f"unsupported adaptive proximal arm {arm!r}")


def parameter_counts(params, arm):
    validate_arm(arm)
    stored = R.parameter_count(params)
    extra = sum(v.size for k, v in flatten_dict(params).items() if k[-1] == HORIZON_KEY)
    frozen = extra if arm == CONTROL else 0
    return {"stored_parameter_count": stored,
            "trainable_parameter_count": stored - frozen,
            "frozen_parameter_count": frozen}


def mode_parameters(params, xp=jnp):
    """Recover kappa and the actual stored horizon leaf in matching mode order."""
    flat = flatten_dict(params)
    for prefix, kappa in H._kappas(params, xp):
        theta = xp.asarray(flat[prefix + (HORIZON_KEY,)])
        if theta.shape != kappa.shape:
            raise ValueError(f"horizon shape differs from kappa at {'/'.join(prefix)}")
        yield prefix, kappa, theta


def transitions(params):
    return jnp.concatenate([
        adaptive_proximal_history_zoh(k, theta, ALPHA, PROXIMAL_RATIO)[0]
        for _, k, theta in mode_parameters(params)
    ], axis=0)


def pole_summary(params, arm=ARM):
    """Actual learned blocks, independently reconstructed in float64."""
    validate_arm(arm)
    host = jax.device_get(params)
    host64 = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), host)
    groups = list(mode_parameters(host64, np))
    kappas = np.concatenate([k for _, k, _ in groups])
    theta = np.concatenate([t for _, _, t in groups])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        multiplier = np.exp(theta)
        horizon = ALPHA / np.abs(kappas) * multiplier
    matrices = np.asarray(jax.device_get(transitions(params)))
    radius32 = np.full(len(kappas), np.nan)
    finite_blocks = np.isfinite(matrices).all(axis=(-2, -1))
    if finite_blocks.any():
        radius32[finite_blocks] = np.max(np.abs(np.linalg.eigvals(
            matrices[finite_blocks].astype(np.complex128))), axis=-1)
    radius64 = np.full(len(kappas), np.nan)
    continuous_radius = np.full(len(kappas), np.nan)
    for i, (k, tau) in enumerate(zip(kappas, horizon)):
        if not (np.isfinite(k) and np.isfinite(tau) and tau > 0):
            continue
        d = 1 + PROXIMAL_RATIO * tau * k
        ke = k / d
        gen = np.array([[0, 1 / tau, 0], [-2 * ke, -1 / tau, 1 / tau],
                        [ke, 0, -1 / tau]], dtype=np.complex128)
        aug = np.zeros((4, 4), dtype=np.complex128)
        aug[:3, :3], aug[:3, 3] = gen, np.array([0, 2, -1]) / d
        block = expm(aug)[:3, :3]
        if np.isfinite(block).all():
            radius64[i] = np.max(np.abs(np.linalg.eigvals(block)))
            continuous_radius[i] = np.exp(np.max(np.linalg.eigvals(gen).real))
    finite = np.isfinite(radius32) & np.isfinite(radius64)
    stable, boundary, unstable = classify(radius32, radius64)
    stable &= finite
    boundary &= finite
    unstable &= finite

    def finite_max(values):
        values = np.asarray(values)
        return float(values[np.isfinite(values)].max()) if np.isfinite(values).any() else None

    max32, max64 = finite_max(radius32), finite_max(radius64)
    theta_finite = bool(np.isfinite(theta).all())
    positive_horizons = bool(np.isfinite(horizon).all() and (horizon > 0).all())
    return {
        "arm": arm, "n_modes": len(kappas), "history_alpha": ALPHA,
        "proximal_ratio": PROXIMAL_RATIO, "trainable_horizon": arm == ARM,
        "scan_implementation": "history_stationary_block",
        "radius_semantics": "actual learned-horizon exact-ZOH 3x3 blocks",
        "stable": int(np.sum(finite & (radius64 < 1))),
        "unstable": int(np.sum(finite & (radius64 >= 1))),
        "nonfinite_modes": int((~finite).sum()),
        "min_margin": None if max64 is None else 1 - max64,
        "horizon_tokens": H._quant(horizon),
        "horizon_multiplier": H._quant(multiplier),
        "positive_finite_horizons": positive_horizons,
        "log_horizon_scale": H._quant(theta),
        "log_horizon_scale_max_abs": float(np.max(np.abs(theta))) if theta_finite else None,
        "log_horizon_scale_values": {
            "/".join(prefix): [float(x) if np.isfinite(x) else None for x in values]
            for prefix, _, values in groups
        },
        "control_horizon_exactly_zero": bool(theta_finite and (theta == 0).all()),
        "radius_reference": H._quant(radius64),
        "max_expm_vs_generator_radius_difference": finite_max(abs(radius64 - continuous_radius)),
        "production_precision": {
            "dtype": "complex64", "boundary_tol": BOUNDARY_TOL,
            "stable": int(stable.sum()), "numerical_boundary": int(boundary.sum()),
            "unstable": int(unstable.sum()), "nonfinite": int((~finite).sum()),
            "max_radius_production": max32, "max_radius_float64": max64,
            "max_abs_radius_difference_production_vs_float64": finite_max(abs(radius32 - radius64)),
            "min_margin_production": None if max32 is None else 1 - max32,
        },
    }


def horizon_gate(summary, arm, *, after_updates=False):
    """Fail closed on invalid horizons or a control that did not stay fixed."""
    if not summary.get("positive_finite_horizons"):
        raise ValueError("horizons must remain positive and finite")
    if arm == CONTROL and not summary.get("control_horizon_exactly_zero"):
        raise ValueError("frozen control horizon parameter moved away from zero")
    if after_updates and arm == ARM:
        maximum = summary.get("log_horizon_scale_max_abs")
        if maximum is None or not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("candidate smoke did not update any horizon parameter")


@contextmanager
def configured(arm):
    """Scope registration and the new leaf's optimizer group; preserve old arms."""
    validate_arm(arm)
    trainable = arm == ARM
    keys = ("ARM", "ALPHA", "NormalizedGradientHistoryS5SSM", "production_transitions",
            "history_pole_summary", "protocol_identity", "registered_history", "check_smoke_gate")
    old = {key: getattr(H, key) for key in keys}
    old_group = TH.no_bc_decay_group

    def group(key):
        if key == HORIZON_KEY:
            return "ssm" if trainable else "none"
        return old_group(key)

    def identity(data_cache, seed):
        result = old["protocol_identity"](data_cache, seed)
        result.update(
            proximal_ratio=PROXIMAL_RATIO, trainable_horizon=trainable,
            recurrence="proximal history with learned per-mode log horizon multiplier",
            horizon_parameter=HORIZON_KEY, horizon_initial_value=0.0,
            horizon_optimizer_group="ssm" if trainable else "none",
            common_parameter_optimizer="unchanged",
        )
        return result

    def smoke_gate(report, identity):
        old["check_smoke_gate"](report, identity)
        horizon_gate(report.get("poles_after_smoke_steps", {}), arm, after_updates=True)

    @contextmanager
    def registered():
        with old["registered_history"]():
            R.SCIENTIFIC_NAMES[arm] = (
                "adaptive proximal history, learned horizons" if trainable else
                "adaptive proximal history, frozen zero-multiplier control")
            R.ARM_CONFIGS[arm] = {
                "shared": R.SHARED_CONFIG,
                "recurrence_constructor": "AdaptiveProximalHistoryS5SSM",
                "extra_parameters": (HORIZON_KEY,),
                "extra_parameter_count": R.N_LAYERS * (R.SSM_SIZE_BASE // 2),
                "extra_trainable_parameters": R.N_LAYERS * (R.SSM_SIZE_BASE // 2) if trainable else 0,
                "extra_frozen_parameters": 0 if trainable else R.N_LAYERS * (R.SSM_SIZE_BASE // 2),
                "state_dimension": 3 * R.SSM_SIZE_BASE,
                "history_alpha": ALPHA, "proximal_ratio": PROXIMAL_RATIO,
                "trainable_horizon": trainable,
            }
            old_production, old_init = R.production_check, R.init_state

            def initialize(selected_arm, seed):
                state = old_init(selected_arm, seed)
                if selected_arm == arm:
                    counts = parameter_counts(state.params, arm)
                    print(f"[{arm}] stored parameters: {counts['stored_parameter_count']}; "
                          f"trainable: {counts['trainable_parameter_count']}; "
                          f"frozen: {counts['frozen_parameter_count']}", flush=True)
                return state

            def production(*args, **kwargs):
                old_production(*args, **kwargs)
                if args[0] == arm:
                    path = Path(args[2]) / "production_check.json"
                    report = json.loads(path.read_text())
                    report.update(parameter_counts(args[4].params, arm))
                    path.write_text(json.dumps(report, indent=2) + "\n")
                    horizon_gate(report["initial_poles"], arm)

            R.production_check, R.init_state = production, initialize
            try:
                yield
            finally:
                R.production_check, R.init_state = old_production, old_init

    H.ARM, H.ALPHA = arm, ALPHA
    H.NormalizedGradientHistoryS5SSM = partial(
        AdaptiveProximalHistoryS5SSM, proximal_ratio=PROXIMAL_RATIO,
        trainable_horizon=trainable)
    H.production_transitions = transitions
    H.history_pole_summary = lambda params: pole_summary(params, arm)
    H.protocol_identity, H.registered_history = identity, registered
    H.check_smoke_gate = smoke_gate
    TH.no_bc_decay_group = group
    try:
        yield arm
    finally:
        TH.no_bc_decay_group = old_group
        for key, value in old.items():
            setattr(H, key, value)


def memory_profile(arm, seed, out):
    from experiments.s5_gradient_history import memory_profile as MP
    out = Path(out)
    if out.exists():
        raise ValueError(f"refusing existing output: {out}")
    with configured(arm):
        previous = MP.ARMS
        MP.ARMS = tuple(dict.fromkeys(previous + (arm,)))
        try:
            result = MP.compile_profile(arm, seed=seed, history_scan="parallel")
        finally:
            MP.ARMS = previous
    result.update(history_alpha=ALPHA, proximal_ratio=PROXIMAL_RATIO,
                  trainable_horizon=arm == ARM,
                  extra_parameter_count=R.N_LAYERS * (R.SSM_SIZE_BASE // 2))
    root = Path(__file__).resolve().parents[2]
    for rel in ("s5/adaptive_proximal_history.py", "s5/proximal_history.py",
                "experiments/s5_adaptive_proximal_history/runner.py"):
        result["source_sha256"][rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(301, 302), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-cache")
    parser.add_argument("--smoke-report")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    if args.profile:
        memory_profile(args.arm, args.seed, args.out)
        return
    if not args.data_cache:
        parser.error("--data-cache is required for smoke/train")
    forwarded = ["--seed", str(args.seed), "--out", args.out,
                 "--data-cache", args.data_cache, "--smoke" if args.smoke else "--train"]
    if args.smoke_report:
        forwarded += ["--smoke-report", args.smoke_report]
    with configured(args.arm):
        H.main(forwarded)
    extra = R.N_LAYERS * (R.SSM_SIZE_BASE // 2)
    production_path = Path(args.out) / "production_check.json"
    production = json.loads(production_path.read_text()) if production_path.exists() else {}
    counts = {key: production[key] for key in
              ("stored_parameter_count", "trainable_parameter_count", "frozen_parameter_count")
              if key in production}
    for filename in ("production_check.json", "smoke_result.json", "task_result.json"):
        path = Path(args.out) / filename
        if path.exists():
            result = json.loads(path.read_text())
            result.update(history_alpha=ALPHA, proximal_ratio=PROXIMAL_RATIO,
                          trainable_horizon=args.arm == ARM,
                          extra_parameter_count=extra,
                          extra_trainable_parameters=extra if args.arm == ARM else 0,
                          extra_frozen_parameters=0 if args.arm == ARM else extra)
            result.update(counts)
            path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
