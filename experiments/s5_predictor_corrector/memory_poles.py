"""Read-only N/A/C pole-decay audit: reconstructed float32 init vs selected weights.

No dataset is opened and no training or evaluation is performed. Pole decay is
an internal e-folding timescale, not a measurement of task-relevant memory.
"""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from . import checkpoint_compare as CC
from . import selected_checkpoint_audit as SA

ARMS = SA.ARMS[:3]
LABELS = dict(zip(ARMS, ("N", "A", "C")))
STAGES = ("reconstructed_init", "trained")
TOL = 1e-5


def plot_preflight(mode):
    """Check optional plotting dependencies before expensive checkpoint work."""
    if mode == "none":
        return {"status": "skipped", "reason": "--plots none"}
    try:
        importlib.import_module("matplotlib.backends.backend_agg")
        importlib.import_module("matplotlib.figure")
    except ImportError as exc:
        reason = f"Plotting dependency unavailable: {exc}"
        if mode == "required":
            raise SA.AuditError(reason + "; install Matplotlib or use --plots auto/none") from exc
        return {"status": "unavailable", "reason": reason}
    return {"status": "ready", "reason": "plotting imports succeeded"}


def summarize_poles(roots, radius=None):
    """Each row is one complex channel; columns are its recurrent poles.

    A stable secondary pole is retained even if its companion is unstable.
    Zero, neutral, growing and nonfinite poles have no finite positive decay
    time; they are counted separately rather than plotted or clipped.
    """
    roots = np.asarray(roots, np.complex128)
    radius = np.abs(roots) if radius is None else np.asarray(radius, float)
    if roots.ndim != 2 or roots.shape != radius.shape or not roots.size:
        raise ValueError("expected nonempty (channels, poles) arrays")
    finite = np.isfinite(roots)
    finite &= np.isfinite(radius)
    valid = finite & (radius > 0) & (radius < 1)
    decay = np.full(radius.shape, np.nan)
    decay[valid] = -1 / np.log(radius[valid])
    ge = finite & (radius >= 1)
    summary = {
        "n_channels": roots.shape[0], "n_poles": roots.size,
        "poles_finite_decay": int(valid.sum()),
        "poles_zero": int((finite & (radius == 0)).sum()),
        "poles_equal_one": int((finite & (radius == 1)).sum()),
        "poles_gt_one": int((finite & (radius > 1)).sum()),
        "poles_ge_one": int(ge.sum()), "fraction_poles_ge_one": float(ge.mean()),
        "channels_ge_one": int(ge.any(axis=1).sum()),
        "fraction_channels_ge_one": float(ge.any(axis=1).mean()),
        "poles_nonfinite": int((~finite).sum()),
        "channels_nonfinite": int((~finite).any(axis=1).sum()),
        "poles_boundary_band": int((finite & (abs(radius - 1) <= TOL)).sum()),
        "poles_gt_one_plus_tol": int((finite & (radius > 1 + TOL)).sum()),
        "max_radius": float(np.max(radius[finite])) if finite.any() else None,
        "decay_samples_by_root": [CC.q(decay[:, j]) for j in range(roots.shape[1])],
        "decay_exceeds_16000_by_root": [int((decay[:, j] > 16000).sum())
                                         for j in range(roots.shape[1])],
    }
    return {"roots": roots, "radius": radius, "decay_samples": decay, "summary": summary}


def coefficients64(arm, group):
    """Actual constructor algebra, float64 values from saved float32 leaves."""
    m, co, hyper = SA.coefficients_for_arm(arm, group)
    if arm == ARMS[1]:
        # Keep A's original grouping, rather than substituting C's module.
        T, M, gamma = (hyper[k] for k in ("T", "M", "gamma"))
        F = 1 + T * (m["lambda_bar"] - 1)
        Q = M + gamma + T
        a, b, d = M / Q, 1 / Q, T / Q
        co = ((1 + a - b) + (b + d) * F, -a - d * F,
              (b + d) * T, -d * T)
    return m, tuple(np.asarray(x, complex) for x in co), hyper


def production_poles(arm, group):
    """Same JAX coefficient/root functions, on CPU float32, not GPU replay."""
    import jax
    import jax.numpy as jnp
    from experiments.s5_three_arm_full.pole_diagnostics import layer_coefficients
    from s5.factored_recurrence import companion_roots
    if jax.config.read("jax_enable_x64"):
        raise SA.AuditError("run with JAX_ENABLE_X64=0; float64 analysis uses NumPy")
    flat = {(k,): jnp.asarray(v) for k, v in group.items()}
    lb, a1, a2 = layer_coefficients(flat, (), arm)
    z = lb[:, None] if arm == ARMS[0] else jnp.stack(companion_roots(a1, a2), axis=-1)
    radius = jnp.abs(z)
    order = jnp.argsort(-radius, axis=1)
    return tuple(np.asarray(jax.device_get(x)) for x in (
        jnp.take_along_axis(z, order, axis=1),
        jnp.take_along_axis(radius, order, axis=1)))


def analyze_layer(arm, name, group, impulse_horizon=16000):
    from s5.predictor_corrector import roots
    if any(np.asarray(v).dtype != np.float32 for v in group.values()):
        raise SA.AuditError(f"{name}: expected saved float32 parameters")
    m, co, hyper = coefficients64(arm, group)
    z = co[0][:, None] if arm == ARMS[0] else np.stack(roots(co[0], co[1]), axis=-1)
    z = np.take_along_axis(z, np.argsort(-abs(z), axis=1), axis=1)
    out = {"layer": name, **summarize_poles(z), "coefficients64": co,
           "input_coefficient_scope": "b0/b1 multipliers per unit native B_bar",
           "parameters64": {"Lambda": m["Lambda"], "Delta": m["Delta"], **hyper}}
    zp, rp = production_poles(arm, group)
    out["float32_cpu"] = summarize_poles(zp, rp)
    r64 = out["radius"]
    out["float32_cpu"]["strict_classification_disagreements"] = int(
        ((r64 < 1) != (rp < 1)).sum())
    out["float32_cpu"]["max_radius_difference"] = float(np.max(abs(r64 - rp)))
    if z.shape[1] == 2:
        separation = abs(z[:, 0] - z[:, 1]) / np.maximum(1, np.max(abs(z), axis=1))
        out["relative_pole_separation"] = separation
        out["summary"]["near_coincident_channels"] = int((separation <= 1e-6).sum())
    else:
        out["summary"]["near_coincident_channels"] = 0
    from .memory_impulses import analyze_impulses
    out["impulse_response"] = analyze_impulses(arm, group, co, horizon=impulse_horizon)
    return out


def manifest_for_inputs(native, original, calibrated, expected_epochs=15, all_epochs=False):
    tasks, seeds, checkpoints = [], [], []
    for arm, directories in zip(ARMS, (native, original, calibrated)):
        seen = set()
        for directory in directories:
            task = Path(directory).resolve()
            try:
                seed = int(task.name)
            except ValueError as exc:
                raise SA.AuditError(f"invalid seed directory: {task}") from exc
            if seed in seen:
                raise SA.AuditError(f"duplicate seed {seed} for {arm}")
            seen.add(seed)
            row = SA.resolve_selected(task, arm, seed, expected_epochs)
            meta_path = task.parent.parent / "run_metadata.txt"
            meta = CC._run_metadata(task)
            if (not meta or not meta.get("authoritative_commit")
                    or meta.get("scan_implementation") != "factored"
                    or meta.get("epochs_requested") != str(expected_epochs)
                    or meta.get("batch_size") != "16"):
                raise SA.AuditError(f"missing/unsupported protocol metadata: {meta_path}")
            row["run_metadata"] = meta
            row["source_files"]["run_metadata"] = {
                "path": str(meta_path), "sha256": CC._sha256(meta_path)}
            if all_epochs:
                for epoch in range(1, expected_epochs + 1):
                    path = task / f"checkpoint_epoch_{epoch:02d}.msgpack"
                    if not path.is_file():
                        raise SA.AuditError(f"missing historical checkpoint: {path}")
                    sha = CC._sha256(path)
                    row["source_files"][f"epoch_{epoch:02d}"] = {"path": str(path), "sha256": sha}
                    checkpoints.append({"arm": arm, "seed": seed, "epoch": epoch,
                                        "checkpoint": str(path), "checkpoint_sha256": sha,
                                        "selected_for_validation": epoch == row["epoch"]})
            tasks.append(row)
        seeds.append(seen)
    if not seeds[0] or not seeds[0] == seeds[1] == seeds[2]:
        raise SA.AuditError(f"unmatched seeds across N/A/C: {seeds}")
    if len({r["validation"]["n"] for r in tasks}) != 1:
        raise SA.AuditError("validation population differs across inputs")
    return {"tasks": sorted(tasks, key=lambda r: (r["seed"], ARMS.index(r["arm"]))),
            "seeds": sorted(seeds[0]), "selection_rule": SA.SELECTION_RULE,
            "all_epoch_checkpoints": checkpoints}


def assert_sources_unchanged(manifest):
    for task in manifest["tasks"]:
        for name, source in task["source_files"].items():
            if CC._sha256(source["path"]) != source["sha256"]:
                raise SA.AuditError(f"source changed during audit: {name} {source['path']}")


def output_guard(out, manifest):
    out = Path(out).resolve()
    if out.exists():
        raise SA.AuditError(f"refusing existing output: {out}")
    for task in manifest["tasks"]:
        root = Path(task["task_dir"]).parent.parent.resolve()
        if out == root or root in out.parents:
            raise SA.AuditError("output must be outside every source run directory")
    return out


def paired_distributions(tasks):
    """Pair by seed/layer/initialized channel index; never pool seeds for a test."""
    out = []
    by = {(t["arm"], t["seed"]): t for t in tasks}
    for seed in sorted({t["seed"] for t in tasks}):
        for stage in STAGES:
            native = by[ARMS[0], seed]["stages"][stage]["layers"]
            for arm in ARMS[1:]:
                candidate = by[arm, seed]["stages"][stage]["layers"]
                for n, c in zip(native, candidate):
                    if n["layer"] != c["layer"]:
                        raise SA.AuditError("paired layer identity differs")
                    nd, cd = n["decay_samples"][:, 0], c["decay_samples"][:, 0]
                    valid = np.isfinite(nd) & np.isfinite(cd) & (nd > 0) & (cd > 0)
                    ratio = cd[valid] / nd[valid]
                    out.append({"arm": arm, "seed": seed, "stage": stage, "layer": n["layer"],
                                "n_paired_finite_decay": int(valid.sum()),
                                "dominant_decay_over_native": CC.q(ratio),
                                "abs_log10_ratio": CC.q(abs(np.log10(ratio)))})
    return out


def write_csv(report, path):
    fields = ("arm", "seed", "epoch", "stage", "layer", "channel", "pole",
              "real", "imag", "radius64", "decay_samples", "radius32_cpu",
              "strict_ge_one64", "strict_ge_one32_cpu", "boundary_band64",
              "log_amplitude_factor_16000", "near_coincident_channel")
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for task in report["tasks"]:
            for stage, data in task["stages"].items():
                for layer in data["layers"]:
                    for i, row in enumerate(layer["roots"]):
                        for j, z in enumerate(row):
                            r = layer["radius"][i, j]
                            r32 = layer["float32_cpu"]["radius"][i, j]
                            decay = layer["decay_samples"][i, j]
                            w.writerow(dict(arm=task["arm"], seed=task["seed"],
                                epoch=0 if stage == STAGES[0] else task["epoch"],
                                stage=stage, layer=layer["layer"], channel=i,
                                pole="dominant" if j == 0 else "secondary", real=z.real,
                                imag=z.imag, radius64=r, decay_samples=decay if np.isfinite(decay) else "",
                                radius32_cpu=r32, strict_ge_one64=bool(r >= 1),
                                strict_ge_one32_cpu=bool(r32 >= 1), boundary_band64=bool(abs(r - 1) <= TOL),
                                log_amplitude_factor_16000=16000 * np.log(r) if r > 0 else "",
                                near_coincident_channel=bool(layer.get("relative_pole_separation", [np.inf] * len(layer["roots"]))[i] <= 1e-6)))


def digest(report):
    lines = ["N/A/C pole e-folding timescales (samples), NOT usable memory.",
             "Initialization is a current float32 reconstruction, not a saved epoch-0 checkpoint.",
             "Counts: strict radius>=1 channels / total; poles counted separately in JSON/CSV."]
    for task in report["tasks"]:
        lines.append(f"\n{LABELS[task['arm']]} seed {task['seed']} selected epoch {task['epoch']}")
        for n, t in zip(*(task["stages"][s]["layers"] for s in STAGES)):
            def median(layer, j):
                q = layer["summary"]["decay_samples_by_root"]
                return "—" if j >= len(q) or q[j] is None else f"{q[j]['p50']:.4g}"
            s = t["summary"]
            lines.append(f"  {t['layer']}: dominant {median(n,0)} -> {median(t,0)}; "
                         f"secondary {median(n,1)} -> {median(t,1)}; "
                         f"strict>=1 {s['channels_ge_one']}/{s['n_channels']}; "
                         f"max radius {s['max_radius']:.10g}")
            impulse = t.get("impulse_response", {})
            if impulse:
                summary = impulse["branch_sum"]["summary"]
                lines.append("    B/C-coupled branch-summed impulse energy: " +
                             json.dumps(SA._jsonable(summary), allow_nan=False))
    return "\n".join(lines) + "\n"


def write_epoch_analyses(report, initial, out, impulse_horizon):
    """Stream complete training history; selection never changes."""
    summaries = []
    selected = {(t["arm"], t["seed"], t["epoch"]): t["stages"]["trained"]["layers"]
                for t in report["tasks"]}
    records = report["manifest"]["all_epoch_checkpoints"]
    for number, source in enumerate(records, 1):
        arm, seed, epoch = (source[k] for k in ("arm", "seed", "epoch"))
        print(f"History {number}/{len(records)}: {LABELS[arm]}/{seed} epoch {epoch}", flush=True)
        layers = selected.get((arm, seed, epoch))
        if layers is None:
            grouping = ARMS[0] if arm == ARMS[0] else ARMS[1]
            groups = CC.layer_groups(CC.read_params(source["checkpoint"]), grouping, source["checkpoint"])
            CC.check_against_init(groups, initial[arm, seed], source["checkpoint"])
            layers = [analyze_layer(arm, name, g, impulse_horizon) for name, g in groups.items()]
        relative = Path("checkpoint_analyses") / arm / str(seed) / f"epoch_{epoch:02d}.json"
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(SA._jsonable({**source, "layers": layers}), f, indent=1, allow_nan=False)
        report["all_epoch_checkpoints"].append({**source, "analysis_file": str(relative)})
        summaries.append({**source, "layers": [
            {"layer": layer["layer"], "poles": layer["summary"],
             "impulse": layer["impulse_response"]["branch_sum"]["summary"]} for layer in layers]})
    with (out / "epoch_summaries.json").open("x") as f:
        json.dump(SA._jsonable(summaries), f, indent=1, allow_nan=False)
    with (out / "epoch_summaries.csv").open("x", newline="") as f:
        fields = ("arm", "seed", "epoch", "selected_for_validation", "layer", "channels_ge_one",
                  "poles_ge_one", "poles_gt_one_plus_tol", "max_radius", "dominant_decay_median", "secondary_decay_median")
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in summaries:
            for layer in row["layers"]:
                s = layer["poles"]
                q = s["decay_samples_by_root"]
                writer.writerow({**{k: row[k] for k in fields[:4]}, "layer": layer["layer"],
                                 **{k: s[k] for k in fields[5:9]},
                                 "dominant_decay_median": q[0]["p50"] if q[0] else "",
                                 "secondary_decay_median": q[1]["p50"] if len(q) > 1 and q[1] else ""})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ("native", "original", "calibrated"):
        ap.add_argument("--" + key, nargs="+", required=True, metavar="TASK_DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-epochs", type=int, default=15)
    ap.add_argument("--all-epochs", action="store_true", help="also analyze every saved epoch in the completed runs")
    ap.add_argument("--impulse-horizon", type=int, default=16000, help="finite impulse horizon in waveform samples")
    ap.add_argument("--plots", choices=("auto", "required", "none"), default="auto",
                    help="auto: save numerical results even if Matplotlib is absent; required: fail before analysis")
    args = ap.parse_args(argv)
    if args.impulse_horizon < 2:
        raise SA.AuditError("impulse horizon must be at least two samples")
    plot_plan = plot_preflight(args.plots)
    print(f"Plot preflight: {plot_plan['status']} — {plot_plan['reason']}", flush=True)
    import jax
    if jax.config.read("jax_enable_x64") or jax.default_backend() != "cpu":
        raise SA.AuditError("set JAX_PLATFORMS=cpu JAX_ENABLE_X64=0")
    manifest = manifest_for_inputs(args.native, args.original, args.calibrated, args.expected_epochs, args.all_epochs)
    out = output_guard(args.out, manifest)
    from .production_init import production_init_flat
    from s5 import checkpointing
    report = {"schema_version": 1, "manifest": manifest, "analysis_provenance": checkpointing.provenance(),
              "tasks": [], "all_epoch_checkpoints": [], "impulse_horizon": args.impulse_horizon,
              "plot_plan": plot_plan, "scope": {
                  "initialization": "Current CPU float32 runner reconstruction in x64-off subprocess; not proof of historical epoch-0 identity across JAX versions/backends.",
                  "trained": "Actual already-selected N/A/C checkpoint weights, no counterfactual coefficient substitution.",
                  "precision": "NumPy float64 equation analysis plus CPU JAX float32 coefficient/root calculation; not bit-identical GPU replay.",
                  "units": "raw waveform samples (tokens), not seconds",
                  "counts": "64 independent complex channels per layer; N has one pole, A/C two. Bidirectional and conjugate copies not counted twice.",
                  "stable_distributions": "Each pole included only when 0<radius<1; stable secondary poles retained in otherwise unstable channels.",
                  "limits": "Pole decay and finite-horizon linear SSM impulse responses. Learned B/C, conjugate pairing and within-direction cross-mode interference are included. Branch sums exclude cross-direction interference; an additional interior bilateral recurrent-kernel summary includes the zero-lag directional cross term. Direct skip D and nonlinear layers are excluded. Channels are not independent experimental replicates.",
                  "data_access": "No dataset access or accuracy reevaluation."}}
    initial = {}
    for source in manifest["tasks"]:
        arm, seed = source["arm"], source["seed"]
        print(f"Analysing {LABELS[arm]}/{seed}, selected epoch {source['epoch']} ...", flush=True)
        grouping = ARMS[0] if arm == ARMS[0] else ARMS[1]
        ig = CC.layer_groups(production_init_flat(arm, seed), grouping, f"reconstructed {arm}/{seed}")
        tg = CC.layer_groups(CC.read_params(source["checkpoint"]), grouping, source["checkpoint"])
        CC.check_against_init(tg, ig, source["checkpoint"])
        if set(tg) != {f"encoder/layers_{i}/seq" for i in range(6)} or any(len(g["Lambda_re"]) != 64 for g in tg.values()):
            raise SA.AuditError("expected six layers of 64 complex channels")
        initial[arm, seed] = ig
        task = {"arm": arm, "seed": seed, "epoch": source["epoch"], "stages": {}}
        for stage, groups in ((STAGES[0], ig), (STAGES[1], tg)):
            task["stages"][stage] = {"layers": [analyze_layer(arm, name, g, args.impulse_horizon) for name, g in groups.items()]}
        report["tasks"].append(task)
    report["reconstructed_ssm_init_agreement"] = {}
    for seed in manifest["seeds"]:
        n = initial[ARMS[0], seed]
        agree = all(np.array_equal(n[layer][k], initial[arm, seed][layer][k])
                    for arm in ARMS[1:] for layer in n for k in CC.LAYER_KEYS)
        report["reconstructed_ssm_init_agreement"][str(seed)] = bool(agree)
        if not agree:
            raise SA.AuditError(f"reconstructed SSM backbone differs for seed {seed}")
    report["paired_layer_summaries"] = paired_distributions(report["tasks"])
    assert_sources_unchanged(manifest)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(report, out / "poles.csv")
    if args.all_epochs:
        write_epoch_analyses(report, initial, out, args.impulse_horizon)
    assert_sources_unchanged(manifest)
    with (out / "memory_poles.json").open("x") as f:
        json.dump(SA._jsonable(report), f, indent=1, allow_nan=False)
    text = digest(report)
    (out / "summary.txt").write_text(text)
    print(text, flush=True)
    print(f"Numerical audit complete: {out}", flush=True)
    status = dict(plot_plan)
    try:
        if plot_plan["status"] == "ready":
            from .memory_pole_plots import plot_report
            from .memory_impulses import plot_impulse_report
            files = (plot_report(report, out) or []) + (plot_impulse_report(report, out) or [])
            status = {"status": "completed", "files": files}
    except Exception as exc:
        status = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                  "numerical_reports_preserved": True}
        raise
    finally:
        with (out / "plot_status.json").open("x") as f:
            json.dump(status, f, indent=2)
        print(f"Plot status: {status['status']}. Numerical reports are preserved.", flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    except (SA.AuditError, CC.CheckpointError) as exc:
        sys.exit(f"memory-pole audit refused: {exc}")
