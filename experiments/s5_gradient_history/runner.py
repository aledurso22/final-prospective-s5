"""Separate, smoke-gated training entry point for fixed-alpha gradient history.

All dataset, optimizer, batching, dropout, selection and checkpoint operations
are reused from the completed S5 runner. Registration exists only within this
entry point's context and restores every patched reference on exit. The old
runner source, its CLI, and all old-arm defaults remain unchanged.

The inherited JSON key ``companion_spectral_radius`` reports the actual 3x3
history transition's spectral radius here, not a second-order companion root.
Epoch records label this explicitly and compare complex64 construction with
float64 reference construction. No pole clipping or added damping is used.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm
from flax.traverse_util import flatten_dict

from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full import step_telemetry as ST
from experiments.s5_three_arm_full.pole_diagnostics import classify, BOUNDARY_TOL
from s5.gradient_history import (NormalizedGradientHistoryS5SSM,
                                 normalized_history_zoh)

ARM = "gradient_history_s5"
ALPHA = 0.10
SCAN = "parallel"
EPOCHS = 15


def _kappas(params, xp=jnp):
    flat = flatten_dict(params)
    for key in sorted(flat):
        if key[-1] != "Lambda_re":
            continue
        prefix = key[:-1]
        re = xp.asarray(flat[key])
        im = xp.asarray(flat[prefix + ("Lambda_im",)])
        delta = xp.exp(xp.asarray(flat[prefix + ("log_step",)])[..., 0])
        yield prefix, -(xp.minimum(re, -1e-4) + 1j * im) * delta


def production_transitions(params):
    return jnp.concatenate([normalized_history_zoh(k, ALPHA)[0]
                            for _, k in _kappas(params)], axis=0)


def history_radius(params):
    """Read-only actual block radius; eigenvectors/gradients are not used."""
    poles = jnp.linalg.eigvals(production_transitions(params))
    return jnp.max(jnp.abs(poles))


def _quant(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(zip(("min", "p5", "p50", "p95", "max"),
                    map(float, np.percentile(values, (0, 5, 50, 95, 100)))))


def history_pole_summary(params):
    # Convert parameter leaves BEFORE computing kappa for a real f64 reference.
    host = jax.device_get(params)
    host64 = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), host)
    kappa64 = np.concatenate([k for _, k in _kappas(host64, np)])
    matrices32 = np.asarray(jax.device_get(production_transitions(params)))
    # Eigendecompose the actual rounded production matrices in host float64;
    # this avoids conflating matrix-construction error with eigensolver error.
    poles32_host64 = np.linalg.eigvals(matrices32.astype(np.complex128))
    poles32_solver = np.asarray(jax.device_get(jnp.linalg.eigvals(jnp.asarray(matrices32))))
    radius32 = np.max(np.abs(poles32_host64), axis=-1)
    radius32_solver = np.max(np.abs(poles32_solver), axis=-1)
    radius64, continuous_radius = [], []
    horizon = ALPHA / np.abs(kappa64)
    for k, h in zip(kappa64, horizon):
        gen = np.array([[0, 1 / h, 0], [-2 * k, -1 / h, 1 / h],
                        [k, 0, -1 / h]], dtype=np.complex128)
        aug = np.zeros((4, 4), np.complex128)
        aug[:3, :3] = gen
        aug[:3, 3] = [0, 2, -1]
        radius64.append(np.max(np.abs(np.linalg.eigvals(expm(aug)[:3, :3]))))
        continuous_radius.append(np.exp(np.max(np.linalg.eigvals(gen).real)))
    radius64 = np.asarray(radius64)
    ok, edge, bad = classify(radius32, radius64)
    finite = (np.isfinite(radius32) & np.isfinite(radius64))
    # classify() is intended for finite inputs; never label NaNs stable.
    ok &= finite
    edge &= finite
    bad &= finite
    return {
        "arm": ARM, "n_modes": int(len(radius64)), "history_alpha": ALPHA,
        "scan_implementation": "history_stationary_block",
        "radius_semantics": "actual exact-ZOH 3x3 transition; inherited per-step key is companion_spectral_radius",
        "stable": int(np.sum(finite & (radius64 < 1))),
        "unstable": int(np.sum(finite & (radius64 >= 1))),
        "nonfinite_modes": int(np.sum(~finite)),
        "min_margin": float(1 - radius64.max()),
        "horizon_tokens": _quant(horizon),
        "radius_reference": _quant(radius64),
        "max_expm_vs_generator_radius_difference": float(np.max(np.abs(radius64 - continuous_radius))),
        "production_precision": {
            "dtype": "complex64", "boundary_tol": BOUNDARY_TOL,
            "stable": int(ok.sum()), "numerical_boundary": int(edge.sum()),
            "unstable": int(bad.sum()), "nonfinite": int((~finite).sum()),
            "max_radius_production": float(radius32.max()),
            "max_radius_production_eigensolver": float(radius32_solver.max()),
            "max_radius_float64": float(radius64.max()),
            "max_abs_radius_difference_production_vs_float64": float(np.max(np.abs(radius32 - radius64))),
            "max_eigensolver_radius_difference": float(np.max(np.abs(radius32_solver - radius32))),
            "min_margin_production": float(1 - radius32.max()),
        },
    }


@contextmanager
def registered_history():
    """Register only in this process and restore even after exceptions."""
    saved = []

    def replace(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    old_factory, old_size = R.ssm_factory, R.recurrent_state_size
    old_radius, old_st_radius, old_poles = R.radius_array, ST.radius_array, R.pole_summary
    old_production = R.production_check

    def factory(arm):
        if arm != ARM:
            return old_factory(arm)
        return partial(NormalizedGradientHistoryS5SSM, history_alpha=ALPHA,
                       implementation=SCAN, **R.ssm_kwargs(arm))

    def state_size(arm):
        if arm != ARM:
            return old_size(arm)
        return {"complex_modes": R.SSM_SIZE_BASE // 2,
                "real_values_total": R.N_LAYERS * 3 * R.SSM_SIZE_BASE,
                "per_layer_real_values": 3 * R.SSM_SIZE_BASE,
                "states_per_complex_mode": 3,
                "note": "same per-direction convention as existing runner; bidirectional scans have separate zero starts"}

    def production(*args, **kwargs):
        old_production(*args, **kwargs)
        arm, _, out = args[:3]
        if arm == ARM:
            path = Path(out) / "production_check.json"
            report = json.loads(path.read_text())
            report["scan_implementation"] = "history_stationary_block"
            report["history_alpha"] = ALPHA
            path.write_text(json.dumps(report, indent=2) + "\n")
            pp = report["initial_poles"]["production_precision"]
            if pp["unstable"] or pp["nonfinite"]:
                raise RuntimeError("history initialization has unstable or nonfinite modes")

    replace(R, "ssm_factory", factory)
    replace(R, "recurrent_state_size", state_size)
    replace(R, "radius_array", lambda p, a: history_radius(p) if a == ARM else old_radius(p, a))
    replace(ST, "radius_array", lambda p, a: history_radius(p) if a == ARM else old_st_radius(p, a))
    replace(R, "pole_summary", lambda p, a: history_pole_summary(p) if a == ARM else old_poles(p, a))
    replace(R, "production_check", production)
    replace(R, "RUNNABLE_ARMS", R.RUNNABLE_ARMS + (ARM,))
    replace(R, "SCIENTIFIC_NAMES", {**R.SCIENTIFIC_NAMES, ARM:
             "finite-response gradient-history S5, exact ZOH, fixed normalized alpha=0.10"})
    replace(R, "ARM_CONFIGS", {**R.ARM_CONFIGS, ARM: {
        "shared": R.SHARED_CONFIG, "recurrence_constructor": "NormalizedGradientHistoryS5SSM",
        "extra_parameters": (), "state_dimension": 3 * R.SSM_SIZE_BASE}})
    try:
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def protocol_identity(data_cache, seed):
    root = Path(__file__).resolve().parents[2]
    # Bind the smoke to all local Python implementation sources, including
    # dataloaders and telemetry; a change outside the constructor also matters.
    sources = sorted(str(p.relative_to(root)) for directory in
                     ("s5", "dataloaders", "experiments")
                     for p in (root / directory).rglob("*.py"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    checksum = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return {"arm": ARM, "seed": seed, "alpha": ALPHA, "epochs": EPOCHS,
            "batch_size": R.BATCH_SIZE, "scan": SCAN,
            "data_cache": str(Path(data_cache).resolve()),
            "data_manifest_sha256": checksum(Path(data_cache) / "manifest.json"),
            "source_sha256": {s: checksum(root / s) for s in sources},
            "git_commit": commit,
            "jax_version": jax.__version__, "jax_enable_x64": False}


def check_smoke_gate(report, identity):
    if report.get("protocol_identity") != identity:
        raise ValueError("smoke protocol, source, data or seed differs from requested training")
    if report.get("backend") != "gpu":
        raise ValueError("training requires a GPU smoke; CPU checks cannot authorize it")
    if report.get("status") != "SMOKE_PASS" or not report.get("checkpoint_restored"):
        raise ValueError("smoke did not pass checkpoint restoration")
    if not report.get("telemetry_check", {}).get("values_agree"):
        raise ValueError("smoke telemetry values disagree")
    poles = report.get("poles_after_smoke_steps", {}).get("production_precision", {})
    if poles.get("unstable", 1) or poles.get("nonfinite", 1):
        raise ValueError("smoke produced unstable or nonfinite modes")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, choices=(301, 302, 303), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--smoke-report", help="passed GPU smoke_result.json, required for --train")
    args = parser.parse_args(argv)
    if jax.config.read("jax_enable_x64"):
        parser.error("production history training and initialization require JAX_ENABLE_X64=0")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("output must be new or empty; existing results are never reused")
    identity = protocol_identity(args.data_cache, args.seed)
    if args.train:
        if not args.smoke_report:
            parser.error("--train requires --smoke-report from this seed's GPU smoke")
        check_smoke_gate(json.loads(Path(args.smoke_report).read_text()), identity)
        if jax.default_backend() != "gpu":
            parser.error("full history training requires the verified GPU backend")
    args.arm, args.epochs = ARM, EPOCHS
    out.mkdir(parents=True, exist_ok=True)
    (out / "history_protocol.json").write_text(json.dumps(identity, indent=2) + "\n")
    try:
        with registered_history():
            R.run_task(args)
        if args.smoke:
            path = out / "smoke_result.json"
            report = json.loads(path.read_text())
            report.update(protocol_identity=identity, backend=jax.default_backend(),
                          scan_implementation="history_stationary_block")
            path.write_text(json.dumps(report, indent=2) + "\n")
            if report["backend"] == "gpu":
                check_smoke_gate(report, identity)
    except Exception as error:
        (out / "failure.json").write_text(json.dumps({"arm": ARM, "seed": args.seed,
            "error_type": type(error).__name__, "failure": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
