"""Isolated, smoke-gated positive-memory proximal S5 training.

The native backbone, optimizer, data split, batch size, 15-epoch schedule and
validation-selection rule are inherited unchanged. Only the state recurrence
is replaced. No test-set evaluation or shared-runner mutation persists after
this module's configuration context exits.

The legacy per-step key ``companion_spectral_radius`` measures this arm's
actual three-state ZOH transition; it is not a second-order companion radius.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.pole_diagnostics import BOUNDARY_TOL, classify
from s5.proximal_memory_mix import ProximalMemoryMixS5SSM, positive_memory_zoh


ARM = "proximal_memory_mix_s5"
ALPHA = 1.0
FAST_RATIO = 0.1
SLOW_RATIO = 1.0
SLOW_WEIGHT = 4.0 / 9.0
PROXIMAL_RATIO = 0.5
EPOCHS = 15
SCAN = "history_stationary_block"


def validate_config(alpha=ALPHA, fast_ratio=FAST_RATIO, slow_ratio=SLOW_RATIO,
                    slow_weight=SLOW_WEIGHT, proximal_ratio=PROXIMAL_RATIO):
    values = (alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("memory-mix parameters must be finite")
    if alpha <= 0 or fast_ratio <= 0 or slow_ratio <= 0:
        raise ValueError("alpha and both horizon ratios must be positive")
    if fast_ratio > slow_ratio:
        raise ValueError("fast_ratio must not exceed slow_ratio")
    if not 0 <= slow_weight <= 1:
        raise ValueError("slow_weight must lie in [0, 1]")
    bound = (1 - slow_weight) * fast_ratio + slow_weight * slow_ratio
    # Permit representation rounding at an analytically exact boundary. The
    # supplied coefficients are never changed, clipped or rounded upward.
    if proximal_ratio < bound and not math.isclose(
            proximal_ratio, bound, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("proximal_ratio violates the uniform stability bound "
                         "(1-slow_weight)*fast_ratio + slow_weight*slow_ratio")
    return bound


def _settings(alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio):
    bound = validate_config(alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio)
    return dict(history_alpha=alpha, fast_ratio=fast_ratio, slow_ratio=slow_ratio,
                slow_weight=slow_weight, proximal_ratio=proximal_ratio,
                uniform_stability_bound_ratio=bound,
                horizon_normalization="magnitude",
                recurrence="positive fast/slow memory of the proximal S5 residual",
                outer_optimizer="unchanged")


def transitions(params, alpha=ALPHA, fast_ratio=FAST_RATIO, slow_ratio=SLOW_RATIO,
                slow_weight=SLOW_WEIGHT, proximal_ratio=PROXIMAL_RATIO):
    return jnp.concatenate([positive_memory_zoh(
        k, alpha=alpha, fast_ratio=fast_ratio, slow_ratio=slow_ratio,
        slow_weight=slow_weight, proximal_ratio=proximal_ratio)[0]
        for _, k in H._kappas(params)], axis=0)


def pole_summary(params, alpha=ALPHA, fast_ratio=FAST_RATIO, slow_ratio=SLOW_RATIO,
                 slow_weight=SLOW_WEIGHT, proximal_ratio=PROXIMAL_RATIO):
    """Compare actual rounded blocks with independently constructed float64 blocks."""
    settings = _settings(alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio)
    host = jax.device_get(params)
    host64 = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), host)
    kappas = np.concatenate([k for _, k in H._kappas(host64, np)])
    matrices = np.asarray(jax.device_get(transitions(
        params, alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio)))
    # Host eigensolve of rounded complex64 blocks isolates construction error.
    radius32 = np.max(np.abs(np.linalg.eigvals(matrices.astype(np.complex128))), axis=-1)
    radius_solver = np.max(np.abs(np.asarray(jax.device_get(
        jnp.linalg.eigvals(jnp.asarray(matrices))))), axis=-1)
    horizon = alpha / np.abs(kappas)
    reference, generator_radius = [], []
    for k, h in zip(kappas, horizon):
        divisor = 1 + proximal_ratio * h * k
        # State (s, h*v_fast, h*v_slow). This independent host construction
        # does not call the production generator or cast its coefficients.
        generator = np.array([
            [0, (1-slow_weight)/h, slow_weight/h],
            [-k/(fast_ratio*divisor), -1/(fast_ratio*h), 0],
            [-k/(slow_ratio*divisor), 0, -1/(slow_ratio*h)],
        ], dtype=np.complex128)
        augmented = np.zeros((4, 4), dtype=np.complex128)
        augmented[:3, :3] = generator
        augmented[:3, 3] = [0, 1/(fast_ratio*divisor), 1/(slow_ratio*divisor)]
        reference.append(np.max(np.abs(np.linalg.eigvals(expm(augmented)[:3, :3]))))
        generator_radius.append(np.exp(np.max(np.linalg.eigvals(generator).real)))
    radius64 = np.asarray(reference)
    ok, edge, bad = classify(radius32, radius64)
    finite = np.isfinite(radius32) & np.isfinite(radius64)
    ok &= finite; edge &= finite; bad &= finite
    return {
        "arm": ARM, "n_modes": int(len(kappas)), **settings,
        "scan_implementation": SCAN,
        "radius_semantics": "actual positive-memory proximal exact-ZOH 3x3 transition; "
                            "inherited per-step key is companion_spectral_radius",
        "stable": int(np.sum(finite & (radius64 < 1))),
        "unstable": int(np.sum(finite & (radius64 >= 1))),
        "nonfinite_modes": int(np.sum(~finite)),
        "min_margin": float(1-radius64.max()),
        "horizon_tokens": H._quant(horizon),
        "fast_horizon_tokens": H._quant(fast_ratio*horizon),
        "slow_horizon_tokens": H._quant(slow_ratio*horizon),
        "proximal_mu_tokens": H._quant(proximal_ratio*horizon),
        "radius_reference": H._quant(radius64),
        "max_expm_vs_generator_radius_difference": float(np.max(
            np.abs(radius64-np.asarray(generator_radius)))),
        "production_precision": {
            "dtype": "complex64", "boundary_tol": BOUNDARY_TOL,
            "stable": int(ok.sum()), "numerical_boundary": int(edge.sum()),
            "unstable": int(bad.sum()), "nonfinite": int((~finite).sum()),
            "max_radius_production": float(radius32.max()),
            "max_radius_production_eigensolver": float(radius_solver.max()),
            "max_radius_float64": float(radius64.max()),
            "max_abs_radius_difference_production_vs_float64": float(np.max(abs(radius32-radius64))),
            "max_eigensolver_radius_difference": float(np.max(abs(radius_solver-radius32))),
            "min_margin_production": float(1-radius32.max()),
        },
    }


@contextmanager
def configured(alpha=ALPHA, fast_ratio=FAST_RATIO, slow_ratio=SLOW_RATIO,
               slow_weight=SLOW_WEIGHT, proximal_ratio=PROXIMAL_RATIO, *, seed=None):
    """Bind this research constructor and diagnostics, restoring on all exits."""
    settings = _settings(alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio)
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0):
        raise ValueError("seed must be a positive integer")
    keys = ("ARM", "ALPHA", "NormalizedGradientHistoryS5SSM", "production_transitions",
            "history_pole_summary", "protocol_identity", "registered_history")
    old = {key: getattr(H, key) for key in keys}
    old_seeds = R.SEEDS
    config = (alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio)

    def identity(data_cache, task_seed):
        result = old["protocol_identity"](data_cache, task_seed)
        result.update(settings, scan_implementation=SCAN)
        return result

    @contextmanager
    def registered():
        with old["registered_history"]():
            R.SCIENTIFIC_NAMES[ARM] = (
                "positive fast/slow proximal-memory S5; "
                f"alpha={alpha}, fast={fast_ratio}, slow={slow_ratio}, "
                f"slow_weight={slow_weight}, proximal_ratio={proximal_ratio}; exact ZOH")
            R.ARM_CONFIGS[ARM] = {
                "shared": R.SHARED_CONFIG,
                "recurrence_constructor": "ProximalMemoryMixS5SSM",
                "extra_parameters": (), "state_dimension": 3*R.SSM_SIZE_BASE,
                **settings,
            }
            yield

    H.ARM, H.ALPHA = ARM, alpha
    H.NormalizedGradientHistoryS5SSM = partial(
        ProximalMemoryMixS5SSM, fast_ratio=fast_ratio, slow_ratio=slow_ratio,
        slow_weight=slow_weight, proximal_ratio=proximal_ratio)
    H.production_transitions = lambda params: transitions(params, *config)
    H.history_pole_summary = lambda params: pole_summary(params, *config)
    H.protocol_identity = identity
    H.registered_history = registered
    if seed is not None:
        R.SEEDS = tuple(dict.fromkeys(old_seeds + (seed,)))
    try:
        yield ARM
    finally:
        for key, value in old.items():
            setattr(H, key, value)
        R.SEEDS = old_seeds


def check_smoke_gate(report, production, identity):
    """Require the same configured two-update GPU smoke before training."""
    if report.get("protocol_identity") != identity:
        raise ValueError("smoke protocol, source, data or seed differs")
    if report.get("backend") != "gpu":
        raise ValueError("training requires a GPU smoke")
    if (report.get("status") != "SMOKE_PASS" or not report.get("checkpoint_restored")
            or report.get("training_steps") != 2):
        raise ValueError("smoke must complete two finite updates and checkpoint restoration")
    if not report.get("telemetry_check", {}).get("values_agree"):
        raise ValueError("smoke telemetry values disagree")
    for key, low, high in (("validation_accuracy", 0.0, 1.0),
                           ("validation_cross_entropy", 0.0, math.inf)):
        value = report.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high):
            raise ValueError("smoke validation metric is invalid")
    for item in (report, production):
        if item.get("code_identifier") != ARM or item.get("seed") != identity["seed"]:
            raise ValueError("smoke arm or seed mismatch")
        if item.get("scan_implementation") != SCAN:
            raise ValueError("smoke scan mismatch")
        if item.get("epochs_requested") != EPOCHS:
            raise ValueError("smoke schedule mismatch")
        for key in ("history_alpha", "fast_ratio", "slow_ratio", "slow_weight", "proximal_ratio"):
            if item.get(key) != identity.get(key):
                raise ValueError("smoke memory configuration mismatch")
    if production.get("x64_enabled") is not False:
        raise ValueError("production smoke did not use float32")
    if not production.get("gradients_finite") or not production.get("state_finite"):
        raise ValueError("production smoke has nonfinite state or gradients")
    for key in ("loss", "accuracy", "gradient_norm"):
        value = production.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("production smoke has nonfinite observables")
    for poles in (production.get("initial_poles", {}), report.get("poles_after_smoke_steps", {})):
        precision = poles.get("production_precision", {})
        if precision.get("unstable", 1) or precision.get("nonfinite", 1):
            raise ValueError("smoke contains unstable or nonfinite modes")
        for key in ("max_radius_production", "max_radius_float64"):
            value = precision.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("smoke pole radius is invalid")


def memory_profile(alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio, seed, out):
    """Compile existing exact-shape updates without loading data or training."""
    from experiments.s5_gradient_history import memory_profile as MP
    out = Path(out)
    if out.exists():
        raise ValueError(f"refusing existing output: {out}")
    with configured(alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio, seed=seed) as arm:
        old_arms = MP.ARMS
        MP.ARMS = tuple(dict.fromkeys(old_arms + (arm,)))
        try:
            result = MP.compile_profile(arm, seed=seed, history_scan="parallel")
        finally:
            MP.ARMS = old_arms
    result.update(_settings(alpha, fast_ratio, slow_ratio, slow_weight, proximal_ratio))
    root = Path(__file__).resolve().parents[2]
    for relative in ("s5/proximal_memory_mix.py", "experiments/s5_proximal_memory_mix/runner.py"):
        result["source_sha256"][relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for row in result["updates"]:
        print(row["kind"], row["memory"], flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--fast-ratio", type=float, default=FAST_RATIO)
    parser.add_argument("--slow-ratio", type=float, default=SLOW_RATIO)
    parser.add_argument("--slow-weight", type=float, default=SLOW_WEIGHT)
    parser.add_argument("--proximal-ratio", type=float, default=PROXIMAL_RATIO)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-cache")
    parser.add_argument("--smoke-report")
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("profile", "smoke", "train"):
        mode.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args(argv)
    config = (args.alpha, args.fast_ratio, args.slow_ratio, args.slow_weight, args.proximal_ratio)
    try:
        settings = _settings(*config)
        if args.seed <= 0:
            raise ValueError("seed must be a positive integer")
    except ValueError as error:
        parser.error(str(error))
    if jax.config.read("jax_enable_x64"):
        parser.error("production memory-mix requires JAX_ENABLE_X64=0")
    out = Path(args.out)
    if out.exists():
        parser.error("output must be new; existing results are never reused")
    if args.profile:
        return memory_profile(*config, args.seed, out)
    if not args.data_cache:
        parser.error("--data-cache is required for smoke/train")
    with configured(*config, seed=args.seed):
        identity = H.protocol_identity(args.data_cache, args.seed)
        if args.train:
            if not args.smoke_report:
                parser.error("--train requires --smoke-report from this task's GPU smoke")
            smoke_path = Path(args.smoke_report)
            smoke = json.loads(smoke_path.read_text())
            production = json.loads((smoke_path.parent/"production_check.json").read_text())
            check_smoke_gate(smoke, production, identity)
            if jax.default_backend() != "gpu":
                parser.error("full training requires the verified GPU backend")
        out.mkdir(parents=True)
        (out/"memory_mix_protocol.json").write_text(json.dumps(identity, indent=2)+"\n")
        args.arm, args.epochs = ARM, EPOCHS
        try:
            with H.registered_history():
                R.run_task(args)
            for filename in ("production_check.json", "smoke_result.json", "task_result.json"):
                path = out/filename
                if path.exists():
                    result = json.loads(path.read_text())
                    result.update(settings, scan_implementation=SCAN)
                    if filename != "production_check.json":
                        result.update(protocol_identity=identity, backend=jax.default_backend())
                    path.write_text(json.dumps(result, indent=2)+"\n")
            if args.smoke and jax.default_backend() == "gpu":
                check_smoke_gate(json.loads((out/"smoke_result.json").read_text()),
                                 json.loads((out/"production_check.json").read_text()), identity)
        except Exception as error:
            (out/"failure.json").write_text(json.dumps({
                "arm": ARM, "seed": args.seed, **settings,
                "error_type": type(error).__name__, "failure": str(error),
            }, indent=2)+"\n")
            raise


if __name__ == "__main__":
    main()
