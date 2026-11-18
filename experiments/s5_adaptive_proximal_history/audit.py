"""Read-only adaptive-P pole and joint auxiliary-response audit.

The learned per-mode log_horizon_scale is used in BOTH float64 reference and
float32 production blocks. Components are ordered by decreasing pole modulus;
both auxiliary poles are reported. The joint auxiliary kernel is formed before
the real B/C projection, retaining cross-pole, cross-mode and conjugate-pair
interference. Its energy ratio is a squared kernel change, not an additive
share, usable-memory measurement or classifier-ablation result. Directions are
separate; the D skip and nonlinear backbone are excluded. No dataset is read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from experiments.s5_predictor_corrector import checkpoint_compare as CC
from experiments.s5_predictor_corrector import memory_impulses as MI
from experiments.s5_predictor_corrector import selected_checkpoint_audit as SA
from experiments.s5_predictor_corrector.memory_poles import summarize_poles
from experiments.s5_proximal_inertia import block_audit as BA

ARM = "adaptive_proximal_history_s5"
CONTROL = "adaptive_proximal_history_control_s5"
ARMS = (ARM, CONTROL)
SCALE_KEY = "log_horizon_scale"
AuditError = BA.AuditError


def resolve_task(task, expected_epochs=15):
    task = Path(task).resolve()
    result_path, metrics_path = task / "task_result.json", task / "metrics.jsonl"
    if (task / "failure.json").exists():
        raise AuditError(f"{task}: failure.json exists")
    result = SA._read_json(result_path)
    arm, seed = result.get("code_identifier"), result.get("seed")
    if arm not in ARMS or task.name != str(seed):
        raise AuditError(f"{task}: unsupported arm or mismatched seed")
    if result.get("epochs_requested") != expected_epochs:
        raise AuditError(f"{task}: epochs_requested != {expected_epochs}")
    if result.get("history_alpha") != 1.0 or result.get("proximal_ratio") != 0.5:
        raise AuditError(f"{task}: expected history_alpha=1 and proximal_ratio=.5")
    records = {}
    with metrics_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            epoch = row["epoch"]
            if epoch in records:
                raise AuditError(f"{metrics_path}: duplicate epoch {epoch}")
            records[epoch] = SA._validation(row["validation"], str(metrics_path))
    if set(records) != set(range(1, expected_epochs + 1)):
        raise AuditError(f"{task}: incomplete epoch history")
    selected = max(records, key=lambda e: (records[e]["accuracy"],
                                          -records[e]["cross_entropy"], -e))
    if result.get("selected_epoch") != selected:
        raise AuditError(f"{task}: selected_epoch disagrees with original rule")
    if SA._validation(result["validation"], str(result_path)) != records[selected]:
        raise AuditError(f"{task}: selected validation disagrees with metrics")
    checkpoints = {e: task / f"checkpoint_epoch_{e:02d}.msgpack"
                   for e in sorted({selected, expected_epochs})}
    if not all(p.is_file() for p in checkpoints.values()):
        raise AuditError(f"{task}: selected or final checkpoint missing")
    sources = [result_path, metrics_path, *checkpoints.values()]
    return {"task": str(task), "arm": arm, "seed": int(seed),
            "history_alpha": 1.0, "proximal_ratio": .5,
            "selected_epoch": selected, "validation": records[selected],
            "final_validation": records[expected_epochs],
            "checkpoints": {e: str(p) for e, p in checkpoints.items()},
            "source_sha256": {str(p): CC._sha256(p) for p in sources}}


def validate_group(arm, group):
    if SCALE_KEY not in group:
        raise AuditError(f"missing {SCALE_KEY}")
    if any(np.asarray(v).dtype != np.float32 for v in group.values()):
        raise AuditError("expected float32 checkpoint leaves")
    xi = np.asarray(group[SCALE_KEY], np.float64)
    if xi.shape != np.asarray(group["Lambda_re"]).shape or not np.isfinite(xi).all():
        raise AuditError("invalid horizon scale shape or nonfinite scale")
    if arm == CONTROL and np.any(xi != 0):
        raise AuditError("frozen control has nonzero log_horizon_scale")
    return xi


def blocks64(kappa, xi):
    """Independent SciPy exact-ZOH blocks for the saved, learned horizons."""
    kappa, xi = np.asarray(kappa), np.asarray(xi, np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        tau = np.exp(xi) / np.abs(kappa)
    if not np.isfinite(tau).all() or np.any(tau <= 0):
        raise AuditError("nonfinite or nonpositive learned horizon")
    matrices, drives = [], []
    for k, h in zip(kappa, tau):
        d = 1 + .5 * h * k
        ke = k / d
        aug = np.zeros((4, 4), np.complex128)
        aug[:3, :3] = [[0, 1/h, 0], [-2*ke, -1/h, 1/h], [ke, 0, -1/h]]
        aug[:3, 3] = np.asarray([0, 2, -1]) / d
        E = expm(aug)
        matrices.append(E[:3, :3]); drives.append(E[:3, 3])
    return np.asarray(matrices), np.asarray(drives), tau


def blocks32(group):
    import jax
    import jax.numpy as jnp
    from s5.adaptive_proximal_history import adaptive_proximal_history_zoh
    if jax.config.read("jax_enable_x64"):
        raise AuditError("run the production comparison with JAX_ENABLE_X64=0")
    lam = jnp.minimum(jnp.asarray(group["Lambda_re"]), -1e-4) + 1j*jnp.asarray(group["Lambda_im"])
    kappa = -lam * jnp.exp(jnp.asarray(group["log_step"])[:, 0])
    A, b, tau = adaptive_proximal_history_zoh(kappa, jnp.asarray(group[SCALE_KEY]))
    return tuple(np.asarray(jax.device_get(v)) for v in (A, b, tau))


def joint_auxiliary_response(h, waves, B, C, resolved):
    """Layer-level kernel deletion metric, retaining all interference.

    Any unresolved modal decomposition withholds the whole-layer deletion
    metric. It is never silently replaced by a partial-mode result.
    """
    full, full_audit, _, _ = MI._real_kernel_energy(h, B, C)
    out = {"full": MI._energy_summary(full), "full_numerical_audit": full_audit,
           "status": "withheld_ill_conditioned_components",
           "joint_auxiliary_over_full_energy": None,
           "scope": "Squared change of the complete real recurrent kernel when both auxiliary poles are removed; not an additive energy share."}
    if not np.all(resolved):
        return out
    aux = np.sum(np.asarray(waves[1:]), axis=0)
    reconstruction = waves[0] + aux
    scale = np.maximum(np.max(np.abs(h), axis=0), np.finfo(float).tiny)
    error = float(np.max(np.max(np.abs(reconstruction-h), axis=0) / scale))
    out["component_reconstruction_relative_error_max"] = error
    if not np.isfinite(error) or error > 1e-6:
        out["status"] = "withheld_component_reconstruction_error"
        return out
    # Sum auxiliary WAVES before the B/C Gram calculation. In particular,
    # never sum component energies; cancellation belongs in this quantity.
    removed, removed_audit, _, _ = MI._real_kernel_energy(aux, B, C)
    dominant, dominant_audit, _, _ = MI._real_kernel_energy(waves[0], B, C)
    out.update(joint_auxiliary=MI._energy_summary(removed),
               dominant_only=MI._energy_summary(dominant),
               joint_numerical_audit=removed_audit,
               dominant_numerical_audit=dominant_audit)
    if out["full"]["status"] != "finite":
        out["status"] = "undefined_full_energy"
    elif out["joint_auxiliary"]["status"] not in ("finite", "zero_response"):
        out["status"] = "invalid_joint_auxiliary_energy"
    else:
        out["joint_auxiliary_over_full_energy"] = (
            out["joint_auxiliary"]["total_energy"] / out["full"]["total_energy"])
        out["status"] = "finite"
    return out


def analyze_layer(arm, name, group, horizon):
    xi = validate_group(arm, group)
    kappa, delta = BA.kappa64(group)
    A, b, tau = blocks64(kappa, xi)
    A32, b32, tau32 = blocks32(group)
    z64 = np.linalg.eigvals(A)
    z64 = np.take_along_axis(z64, np.argsort(-abs(z64), axis=1), axis=1)
    z32 = np.linalg.eigvals(A32.astype(np.complex128))
    z32 = np.take_along_axis(z32, np.argsort(-abs(z32), axis=1), axis=1)
    r64, r32 = abs(z64[:, 0]), abs(z32[:, 0])
    finite = np.isfinite(r64) & np.isfinite(r32)
    stable = finite & (r64 < 1-BA.TOL) & (r32 < 1-BA.TOL)
    prod = {"stable": int(stable.sum()),
            "numerical_boundary": int((finite & ~stable & (r64 <= 1+BA.TOL)).sum()),
            "unstable": int((finite & (r64 > 1+BA.TOL)).sum()),
            "nonfinite": int((~finite).sum()), "boundary_tol": BA.TOL,
            "max_radius_float64": float(r64.max()), "max_radius_float32": float(r32.max()),
            "max_abs_radius_difference": float(np.max(abs(r64-r32))),
            "max_abs_block_difference": float(np.max(abs(A-A32)))}
    poles = summarize_poles(z64)
    h = BA.impulse(A, b, horizon)
    fixed_A, fixed_b, _ = blocks64(kappa, np.zeros_like(xi))
    fixed_h = BA.impulse(fixed_A, fixed_b, horizon)
    _, residues, condition, waves = BA.components(A, b, horizon)
    resolved = np.isfinite(condition) & (condition < BA.COND_LIMIT)
    B = delta[:, None] * CC._complex(group["B"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_residue = abs(residues) / np.sum(abs(residues), axis=1, keepdims=True)
        dom_ratio = np.where((abs(z64[:, 0]) > 0) & (abs(z64[:, 0]) < 1),
                             -kappa.real / np.log(abs(z64[:, 0])), np.nan)
    out = {"layer": name, "n_modes": len(kappa), "block_width": 3,
           "log_horizon_scale": xi, "log_horizon_scale_summary": CC.q(xi),
           "horizon_multiplier_summary": CC.q(np.exp(xi)),
           "tau_samples": tau, "tau_samples_summary": CC.q(tau),
           "mu_samples_summary": CC.q(.5*tau),
           "max_abs_tau_float32_difference": float(np.max(abs(tau-tau32))),
           "production_precision": prod, "poles_float64": poles["summary"],
           "poles": {"float64": z64, "float32": z32},
           "pole_residues": residues, "component_resolved": resolved,
           "decay_samples": poles["decay_samples"],
           "dominant_decay_over_native_counterfactual": CC.q(dom_ratio),
           "auxiliary_decay_samples": [CC.q(poles["decay_samples"][:, j]) for j in (1, 2)],
           "auxiliary_relative_residue": [CC.q(rel_residue[:, j]) for j in (1, 2)],
           "eigenvector_condition": CC.q(condition),
           "components_withheld_ill_conditioned": int((~resolved).sum()),
           "directions": {}}
    for direction, key in (("forward", "C1"), ("backward", "C2")):
        C = CC._complex(group[key])
        energy, _, G, H = MI._real_kernel_energy(h, B, C)
        fixed_energy, _, _, _ = MI._real_kernel_energy(fixed_h, B, C)
        changed_energy, _, _, _ = MI._real_kernel_energy(h-fixed_h, B, C)
        full_summary = MI._energy_summary(energy)
        change_summary = MI._energy_summary(changed_energy)
        fixed_change_ratio = None
        if (full_summary["status"] == "finite" and
                change_summary["status"] in ("finite", "zero_response")):
            fixed_change_ratio = change_summary["total_energy"] / full_summary["total_energy"]
        mode_total, _ = MI._mode_energy(h, G, H)
        components = []
        for j, wave in enumerate(waves):
            ce, _ = MI._mode_energy(np.where(resolved[None, :], wave, 0), G, H)
            ce = np.where(resolved, ce, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = ce / mode_total
            components.append({"pole_rank": j, "mode_energy": ce,
                               "fraction_of_mode_energy": CC.q(ratio)})
        out["directions"][direction] = {
            "summary": full_summary, "components": components,
            "mode_total_energy": mode_total,
            "fixed_scale_same_weights_counterfactual": {
                "scope": "Untrained counterfactual: set xi=0 at these trained weights; not the separately trained control.",
                "summary": MI._energy_summary(fixed_energy),
                "kernel_change_summary": change_summary,
                "squared_kernel_change_over_actual_energy": fixed_change_ratio},
            "joint_auxiliary_removal": joint_auxiliary_response(h, waves, B, C, resolved)}
    return out


def stage_report(arm, params, horizon, where):
    groups = CC.layer_groups(params, "", where)
    layers = [analyze_layer(arm, name, group, horizon) for name, group in sorted(groups.items())]
    totals = {k: sum(l["production_precision"][k] for l in layers)
              for k in ("stable", "numerical_boundary", "unstable", "nonfinite")}
    return {"layers": layers, "production_precision_totals": totals}


def init_params(seed):
    from flax.traverse_util import flatten_dict, unflatten_dict
    flat = flatten_dict(BA.init_params(seed))
    for key in list(flat):
        if key[-1] == "Lambda_re":
            flat[key[:-1] + (SCALE_KEY,)] = np.zeros_like(flat[key], dtype=np.float32)
    return unflatten_dict(flat)


def digest(tasks):
    lines = ["Selected-checkpoint adaptive-P audit; all times are samples.",
             "Joint aux/full is squared full-layer kernel change, NOT an additive share."]
    def fmt(v):
        return "n/a" if v is None else f"{v:.5g}"
    for task in tasks:
        stage = task["stages"].get("selected") or task["stages"]["selected_and_final"]
        lines.append(f"\n{task['arm']} seed {task['seed']} epoch {task['selected_epoch']} "
                     f"val={100*task['validation']['accuracy']:.4f}%")
        for layer in stage["layers"]:
            aux = layer["auxiliary_decay_samples"]
            p = layer["production_precision"]
            parts = [f"{d} joint={fmt(layer['directions'][d]['joint_auxiliary_removal']['joint_auxiliary_over_full_energy'])}"
                     for d in ("forward", "backward")]
            lines.append(f"  {layer['layer']}: scale50={fmt(layer['horizon_multiplier_summary']['p50'])} "
                         f"tau50={fmt(layer['tau_samples_summary']['p50'])} "
                         f"aux1/aux2 decay50={fmt(aux[0]['p50'] if aux[0] else None)}/"
                         f"{fmt(aux[1]['p50'] if aux[1] else None)} "
                         f"stable/boundary/unstable={p['stable']}/{p['numerical_boundary']}/{p['unstable']} "
                         + " ".join(parts))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon", type=int, default=16000)
    args = parser.parse_args(argv)
    if args.horizon < 1:
        raise AuditError("horizon must be positive")
    out = Path(args.out).resolve()
    task_paths = [Path(t).resolve() for t in args.task]
    if any(out == t or out.is_relative_to(t) or t.is_relative_to(out) for t in task_paths):
        raise AuditError("audit output must be separate from source task directories")
    if out.exists():
        raise AuditError(f"refusing existing output: {out}")
    rows = [resolve_task(t) for t in task_paths]
    tasks = []
    for row in rows:
        print(f"Auditing {row['arm']}/{row['seed']} ...", flush=True)
        task = {k: v for k, v in row.items() if k != "checkpoints"}
        task["stages"] = {"reconstructed_init": stage_report(
            row["arm"], init_params(row["seed"]), args.horizon, "current float32 reconstruction")}
        for epoch, path in sorted(row["checkpoints"].items()):
            label = "selected" if epoch == row["selected_epoch"] else "final"
            if len(row["checkpoints"]) == 1:
                label = "selected_and_final"
            task["stages"][label] = stage_report(row["arm"], CC.read_params(path), args.horizon, path)
            task["stages"][label]["epoch"] = epoch
        tasks.append(task)
    changed = [path for row in rows for path, sha in row["source_sha256"].items()
               if CC._sha256(path) != sha]
    if changed:
        raise AuditError(f"source files changed while auditing: {changed}")
    report = {"scope": __doc__, "horizon": args.horizon, "tasks": tasks,
              "sources_unchanged": True, "test_split_read": False,
              "initialization_scope": "Current float32 reconstruction; not a saved epoch-zero checkpoint.",
              "production_scope": "Production constructor float32 on CPU; not a GPU replay."}
    out.mkdir(parents=True)
    with (out / "block_audit.json").open("x") as fh:
        json.dump(SA._jsonable(report), fh, indent=1, allow_nan=False)
    summary = digest(tasks)
    (out / "summary.txt").write_text(summary)
    print(summary, flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FAIL: {exc}")
