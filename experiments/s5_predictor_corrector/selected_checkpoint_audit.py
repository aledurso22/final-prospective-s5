"""Audit completed N/A/C/D results and their ACTUAL selected checkpoints.

No data are loaded, no epochs are reselected, and no original files are written.
The selected epoch recorded by training must equal the existing protocol's
maximum (validation accuracy, negative cross-entropy, negative epoch). All
requested epochs must be present and no failure.json may exist. This makes
``task_result.json`` completion evidence rather than assuming a status key that
the historical runner does not write.

The manifest is also the fixed input list for a separate, uniformly applied
BatchNorm diagnostic. A recalibrated metric must not replace primary scores.
Mode analysis uses the trained arm's own coefficient formula, in float64; it
is not a replay of the float32 GPU arithmetic. No counterfactual arm is analysed.

  JAX_PLATFORMS=cpu python -m experiments.s5_predictor_corrector.selected_checkpoint_audit \
    --run-root /Users/durso/s5-runs/s5-timescale-2x2/20261117-195818 \
    --seeds 301 302 --out /Users/durso/s5-runs/selected-audit/<new-name>
"""

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from . import checkpoint_compare as CC

ARMS = ("native_matched_s5", "generalized_prospective_s5",
        "calibrated_old_s5", "calibrated_prediction_s5")
VARIANTS = {ARMS[1]: ("legacy", "old"), ARMS[2]: ("calibrated", "old"),
            ARMS[3]: ("calibrated", "prediction")}
SELECTION_RULE = "max validation accuracy, then min cross-entropy, then earliest epoch"


class AuditError(ValueError):
    pass


def _read_json(path):
    try:
        with Path(path).open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc


def _validation(v, where):
    try:
        n, acc, ce = int(v["n"]), float(v["accuracy"]), float(v["cross_entropy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"{where}: invalid validation record") from exc
    if n <= 0 or v["n"] != n or not (math.isfinite(acc) and 0 <= acc <= 1) or not (math.isfinite(ce) and ce >= 0):
        raise AuditError(f"{where}: non-finite or invalid validation record")
    if abs(acc * n - round(acc * n)) > 1e-7:
        raise AuditError(f"{where}: accuracy is inconsistent with an integer correct count")
    return {"accuracy": acc, "cross_entropy": ce, "n": n, "correct": round(acc * n)}


def resolve_selected(task, arm, seed, expected_epochs):
    task = Path(task).resolve()
    result_path, metrics_path = task / "task_result.json", task / "metrics.jsonl"
    if task.name != str(seed) or task.parent.name != arm:
        raise AuditError(f"directory identity does not match {arm}/{seed}: {task}")
    if (task / "failure.json").exists():
        raise AuditError(f"{task}: failure.json exists")
    result = _read_json(result_path)
    if result.get("code_identifier") != arm or result.get("seed") != seed:
        raise AuditError(f"{result_path}: arm/seed identity mismatch")
    if result.get("epochs_requested") != expected_epochs:
        raise AuditError(f"{result_path}: epochs_requested differs from {expected_epochs}")
    if result.get("status") not in (None, "success", "completed", "SUCCESS", "COMPLETED"):
        raise AuditError(f"{result_path}: unsuccessful status {result['status']!r}")
    records = {}
    try:
        with metrics_path.open() as f:
            for line in f:
                row = json.loads(line)
                epoch = row["epoch"]
                if not isinstance(epoch, int) or epoch in records:
                    raise AuditError(f"{metrics_path}: duplicate or invalid epoch {epoch}")
                records[epoch] = _validation(row["validation"], f"{metrics_path} epoch {epoch}")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AuditError(f"cannot read complete metrics from {metrics_path}: {exc}") from exc
    if set(records) != set(range(1, expected_epochs + 1)):
        raise AuditError(f"{task}: incomplete epoch history: {sorted(records)}")
    if len({v["n"] for v in records.values()}) != 1:
        raise AuditError(f"{task}: validation population changes between epochs")
    expected = max(records, key=lambda e: (records[e]["accuracy"], -records[e]["cross_entropy"], -e))
    if result.get("selected_epoch") != expected:
        raise AuditError(f"{task}: selected_epoch {result.get('selected_epoch')} disagrees with selection rule ({expected})")
    validation = _validation(result.get("validation", {}), str(result_path))
    if validation != records[expected]:
        raise AuditError(f"{task}: selected validation differs from metrics.jsonl")
    checkpoint = task / f"checkpoint_epoch_{expected:02d}.msgpack"
    if not checkpoint.is_file():
        raise AuditError(f"missing selected checkpoint {checkpoint}")
    declared = result.get("selected_checkpoint")
    if declared is not None and Path(declared).name != checkpoint.name:
        raise AuditError(f"{task}: selected_checkpoint basename differs from selected epoch")
    files = {"checkpoint": checkpoint, "task_result": result_path, "metrics": metrics_path}
    hashes = {name: CC._sha256(path) for name, path in files.items()}
    return {"arm": arm, "seed": seed, "epoch": expected, "task_dir": str(task),
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashes["checkpoint"],
            "task_result_sha256": hashes["task_result"],
            "checkpoint_bytes": checkpoint.stat().st_size,
            "declared_checkpoint": declared, "validation": validation,
            "epochs_completed": expected_epochs,
            "source_files": {k: {"path": str(v), "sha256": hashes[k]} for k, v in files.items()},
            "training_seconds": result.get("training_seconds"),
            "parameter_count": result.get("parameter_count"),
            "recurrent_state": result.get("recurrent_state")}


def build_manifest(run_root, seeds=(301, 302), expected_epochs=15):
    """Validate all four arms and return a fixed, hash-addressed task list."""
    root = Path(run_root).resolve()
    if not seeds or len(set(seeds)) != len(seeds) or any(int(s) != s for s in seeds):
        raise AuditError("seeds must be a nonempty unique integer list")
    metadata_path = root / "run_metadata.txt"
    if not metadata_path.is_file():
        raise AuditError(f"missing {metadata_path}")
    meta = CC._run_metadata(root / ARMS[0] / str(seeds[0]))
    if not meta.get("authoritative_commit") or not meta.get("scan_implementation"):
        raise AuditError("run metadata must record commit and scan provenance")
    if meta.get("epochs_requested") != str(expected_epochs) or meta.get("batch_size") != "16":
        raise AuditError("run metadata must record the requested epochs and batch_size=16")
    tasks = [resolve_selected(root / arm / str(seed), arm, int(seed), expected_epochs)
             for arm in ARMS for seed in seeds]
    if len({r["validation"]["n"] for r in tasks}) != 1:
        raise AuditError("validation population differs across tasks")
    return {"schema_version": 1, "run_root": str(root), "seeds": list(seeds),
            "selection_rule": SELECTION_RULE, "epochs_requested": expected_epochs,
            "run_metadata": meta, "run_metadata_sha256": CC._sha256(metadata_path),
            "tasks": tasks,
            "scope": "Already-selected validation checkpoints; no reselection, test access, or counterfactual training claims."}


def assert_manifest_unchanged(manifest):
    root = Path(manifest["run_root"])
    if CC._sha256(root / "run_metadata.txt") != manifest["run_metadata_sha256"]:
        raise AuditError("run_metadata.txt changed during audit")
    for row in manifest["tasks"]:
        for name, source in row["source_files"].items():
            if CC._sha256(source["path"]) != source["sha256"]:
                raise AuditError(f"{row['arm']}/{row['seed']}: {name} changed during audit")


def performance_summary(manifest):
    by = {(r["arm"], r["seed"]): r for r in manifest["tasks"]}
    scores = []
    for arm in ARMS:
        vals = [by[arm, seed]["validation"] for seed in manifest["seeds"]]
        scores.append({"arm": arm, "mean_accuracy": float(np.mean([v["accuracy"] for v in vals])),
                       "mean_cross_entropy": float(np.mean([v["cross_entropy"] for v in vals])),
                       "seeds": {str(s): by[arm, s]["validation"] for s in manifest["seeds"]}})
    pairs = []
    for left, right in itertools.combinations(ARMS, 2):
        differences = {str(s): 100 * (by[left, s]["validation"]["accuracy"] - by[right, s]["validation"]["accuracy"])
                       for s in manifest["seeds"]}
        pairs.append({"left": left, "right": right, "difference_pp_by_seed": differences,
                      "mean_difference_pp": float(np.mean(list(differences.values())))})
    return {"observed_ranking": sorted(scores, key=lambda r: -r["mean_accuracy"]),
            "paired_differences": pairs,
            "interpretation": "Descriptive validation ranking only; no significance, equivalence, or equal selection-bias claim."}


def coefficients_for_arm(arm, group):
    """Float64 coefficients of the arm actually trained; B_bar factored out."""
    from s5 import predictor_corrector as PC
    P = np.asarray(group["Lambda_re"]).size
    expected = {"Lambda_re": (P,), "Lambda_im": (P,), "log_step": (P, 1)}
    B, C1, C2 = (np.asarray(group[k]) for k in ("B", "C1", "C2"))
    if B.ndim != 3 or B.shape[0] != P or B.shape[-1] != 2 or C1.shape != (B.shape[1], P, 2) or C2.shape != C1.shape:
        raise AuditError("unsupported S5 input/readout parameter shapes")
    if arm != ARMS[0]:
        expected.update({k: (P,) for k in CC.GEN_KEYS})
    if any(np.shape(group.get(k)) != shape for k, shape in expected.items()):
        raise AuditError("unsupported S5 parameter shapes")
    if any(not np.isfinite(v).all() for v in group.values()):
        raise AuditError("non-finite selected checkpoint parameters")
    m = CC.mode_quantities(group)
    if arm == ARMS[0]:
        return m, PC.native_coefficients(m["lambda_bar"]), {}
    if arm not in VARIANTS:
        raise AuditError(f"unsupported arm {arm}")
    T, M, gamma, rho = CC.generalized_params(group)
    return m, PC.scalar_coefficients(m["lambda_bar"], T, M, gamma, *VARIANTS[arm]), {
        "T": T, "M": M, "gamma": gamma, "rho": rho, "T_over_gamma": T / gamma}


def response_analysis(coefficients, native=False):
    """Poles, residues and exact per-mode energy. Coincident residues omitted.

    Total energy uses a stable 2-state Lyapunov realization, so a repeated root
    does not require a singular residue decomposition. Component energies omit
    interference; their ratios to exact total energy can exceed one.
    """
    from scipy.linalg import solve_discrete_lyapunov
    from s5 import predictor_corrector as PC
    a1, a2, g1, g2 = (np.asarray(x, complex) for x in coefficients)
    z1, z2 = PC.roots(a1, a2)
    radius = np.maximum(abs(z1), abs(z2))
    finite = np.isfinite(radius)
    stable = PC.schur_cohn(a1, a2) & finite & (radius < 1)
    separation = abs(z1 - z2) / np.maximum(1, np.maximum(abs(z1), abs(z2)))
    resolved = separation > 1e-6
    R1 = np.full_like(z1, np.nan); R2 = np.full_like(z2, np.nan)
    R1[resolved] = (g1[resolved] * z1[resolved] + g2[resolved]) / (z1[resolved] - z2[resolved])
    R2[resolved] = -(g1[resolved] * z2[resolved] + g2[resolved]) / (z1[resolved] - z2[resolved])
    total, e1, e2 = (np.full(z1.shape, np.nan) for _ in range(3))
    energy_residual = np.full(z1.shape, np.nan)
    for i in np.flatnonzero(stable):
        A = np.array([[a1[i], 1], [a2[i], 0]])
        B = np.array([g1[i], g2[i]])
        Q = np.outer(B, B.conj())
        P = solve_discrete_lyapunov(A, Q)
        residual = np.linalg.norm(P - A @ P @ A.conj().T - Q) / max(np.linalg.norm(P), np.linalg.norm(Q), 1e-300)
        energy_residual[i] = residual
        if np.isfinite(P).all() and residual < 1e-8 and P[0, 0].real >= 0:
            total[i] = P[0, 0].real
    mask = stable & resolved
    e1[mask] = abs(R1[mask]) ** 2 / (1 - abs(z1[mask]) ** 2)
    e2[mask] = abs(R2[mask]) ** 2 / (1 - abs(z2[mask]) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay1 = np.where(stable, -1 / np.log(abs(z1)), np.nan)
        decay2 = np.where(stable, -1 / np.log(abs(z2)), np.nan)
        delay = np.where(stable, PC.group_delay_dc(a1, a2, g1, g2), np.nan)
        residue_ratio = abs(R2) / abs(R1)
        proxy = e2 / (e1 + e2)
        energy_ratio = e2 / total
    if native:
        # The second zero pole belongs only to the analysis realization.
        R2[:] = 0; e2[stable] = 0; proxy[stable] = 0; energy_ratio[stable] = 0
        residue_ratio[:] = 0
    return {"stable": stable, "radius": radius, "slow_pole": z1, "fast_pole": z2,
            "residue_slow": R1, "residue_fast": R2, "residue_resolved": resolved,
            "relative_pole_separation": separation, "slow_decay": decay1, "fast_decay": decay2,
            "dc_delay": delay, "total_impulse_energy": total,
            "energy_lyapunov_relative_residual": energy_residual,
            "slow_component_energy": e1, "fast_component_energy": e2,
            "fast_component_energy_proxy": proxy, "fast_energy_over_exact_total": energy_ratio,
            "residue_fast_over_slow": residue_ratio}


def _quant(values):
    vals = np.asarray(values).ravel()
    return {"quantiles": CC.q(vals), "finite": int(np.isfinite(vals).sum()), "total": int(vals.size)}


def analyze_checkpoint(row):
    # C/D carry A's parameter names, but their coefficient routing is different.
    arm = row["arm"]
    groups = CC.layer_groups(CC.read_params(row["checkpoint"]),
                             ARMS[0] if arm == ARMS[0] else ARMS[1], row["checkpoint"])
    layers = []
    for name, group in groups.items():
        m, co, hyper = coefficients_for_arm(arm, group)
        r = response_analysis(co, native=(arm == ARMS[0]))
        coupling = m["b_bar_norm"] * m["c_norm"]
        layers.append({"layer": name, **m, **hyper, **r,
                       "slow_component_gain": coupling * np.sqrt(r["slow_component_energy"]),
                       "fast_component_gain": coupling * np.sqrt(r["fast_component_energy"]),
                       "total_mode_gain": coupling * np.sqrt(r["total_impulse_energy"])})
    cat = lambda key: np.concatenate([r[key] for r in layers])
    stable = cat("stable").astype(bool)
    resolved = cat("residue_resolved").astype(bool)
    summary = {"n_modes": int(stable.size), "stable": int(stable.sum()),
               "not_strictly_stable": int((~stable).sum()),
               "near_coincident_modes": int((~resolved).sum()),
               "max_radius": float(np.max(cat("radius"))),
               "coefficient_formula": "native" if arm == ARMS[0] else list(VARIANTS[arm]),
               "analysis_precision": "float64 equations; not production float32 replay",
               "native_analysis_auxiliary_pole_is_artificial": arm == ARMS[0]}
    for k in ("Delta", "b_bar_norm", "c_norm", "relative_pole_separation"):
        summary[k] = _quant(cat(k))
    for k in ("slow_decay", "fast_decay", "dc_delay", "total_impulse_energy", "total_mode_gain"):
        summary[k + "_stable_modes_only"] = _quant(cat(k)[stable])
    for k in ("residue_fast_over_slow", "fast_component_energy_proxy", "fast_energy_over_exact_total",
              "slow_component_gain", "fast_component_gain"):
        summary[k + "_stable_resolved_modes_only"] = _quant(cat(k)[stable & resolved])
    for k in ("T", "M", "gamma", "rho", "T_over_gamma"):
        if k in layers[0]:
            summary[k] = _quant(cat(k))
    return {"arm": arm, "seed": row["seed"], "epoch": row["epoch"],
            "actual_trained_arm": True, "summary": summary, "layers": layers}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"real": _jsonable(obj.real), "imag": _jsonable(obj.imag)}
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[301, 302])
    ap.add_argument("--expected-epochs", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-only", action="store_true", help="validate results and hashes without reading parameter arrays")
    args = ap.parse_args(argv)
    root, out = Path(args.run_root).resolve(), Path(args.out).resolve()
    if out == root or root in out.parents:
        raise AuditError("output must be outside the original run directory")
    if out.exists():
        raise AuditError(f"refusing to reuse existing output: {out}")
    manifest = build_manifest(root, args.seeds, args.expected_epochs)
    report = {"manifest": manifest, "performance": performance_summary(manifest),
              "mode_analysis_scope": "Independent per-mode B/C coupling proxies omit cross-mode and cross-direction interference. Near-coincident residue decompositions are withheld. Infinite-horizon energy uses stable modes only."}
    if not args.manifest_only:
        report["actual_selected_checkpoints"] = [analyze_checkpoint(r) for r in manifest["tasks"]]
    assert_manifest_unchanged(manifest)
    out.mkdir(parents=True, exist_ok=False)
    for filename, payload in (("manifest.json", manifest), ("selected_checkpoint_audit.json", report)):
        with (out / filename).open("x") as f:
            json.dump(_jsonable(payload), f, indent=2, allow_nan=False)
    for row in report["performance"]["observed_ranking"]:
        print(f"{row['arm']:28s} mean selected val {100 * row['mean_accuracy']:.4f}%  CE {row['mean_cross_entropy']:.6f}")
    for row in report.get("actual_selected_checkpoints", []):
        s = row["summary"]
        print(f"{row['arm']}/{row['seed']} selected epoch {row['epoch']}: stable {s['stable']}/{s['n_modes']}; near-coincident {s['near_coincident_modes']}")
    print(f"wrote {out / 'selected_checkpoint_audit.json'}")
    return report


if __name__ == "__main__":
    try:
        main()
    except (AuditError, CC.CheckpointError) as exc:
        raise SystemExit(f"selected-checkpoint audit refused: {exc}")
