"""Read-only pole and B/C-coupled impulse audit for block-recurrence S5 arms.

Covers the trained proximal-history arm P (3x3 blocks, ``proximal_history_s5``)
and the proximal inertial candidate/control (2x2 blocks). For every task:

  * the selected checkpoint is re-derived from the complete validation history
    (max accuracy, then min cross-entropy, then earliest epoch) and must equal
    task_result.json; nothing is reselected;
  * stages: reconstructed float32 production initialization (native draw; the
    block arms' parameters are the native ones), the selected checkpoint, and
    the final epoch when it differs;
  * per layer and mode: the actual exact-ZOH block transition in float64
    (SciPy, from the saved float32 leaves) and in production float32 (the
    arm's own JAX constructor on CPU); poles ordered by modulus; counts of
    growing / boundary / nonfinite poles; nothing is clipped;
  * the native pole exp(-kappa) at the SAME Lambda, Delta is reported as an
    untrained counterfactual reference only;
  * the observable kernel 2 Re(C h_t (Delta B~)) per direction over 16,000
    lags, its energy profile, per-mode energies, per-pole components with
    residues from the block eigendecomposition (withheld when ill-conditioned)
    and the cross-pole interference that makes component shares non-additive.

D skip and the nonlinear backbone are excluded. No dataset is read, nothing is
trained or evaluated, and source checkpoints are hashed before and after.

  JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m experiments.s5_proximal_inertia.block_audit \
      --task <run>/alpha1/301 --task ... --out <new directory> [--allow-missing]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from experiments.s5_predictor_corrector import checkpoint_compare as CC
from experiments.s5_predictor_corrector import memory_impulses as MI
from experiments.s5_predictor_corrector import selected_checkpoint_audit as SA
from experiments.s5_predictor_corrector.memory_poles import summarize_poles

P_ARM = "proximal_history_s5"
PI_ARMS = ("proximal_inertial_s5", "proximal_inertial_control_s5")
ARMS = (P_ARM,) + PI_ARMS
TOL = 1e-5
COND_LIMIT = 1e8


class AuditError(ValueError):
    pass


# ------------------------------------------------------------- inputs ---

def resolve_task(task, expected_epochs=15):
    task = Path(task).resolve()
    result_path, metrics_path = task / "task_result.json", task / "metrics.jsonl"
    if (task / "failure.json").exists():
        raise AuditError(f"{task}: failure.json exists")
    if not result_path.is_file():
        raise AuditError(f"{task}: no task_result.json (unfinished or failed)")
    result = SA._read_json(result_path)
    arm, seed = result.get("code_identifier"), result.get("seed")
    if arm not in ARMS:
        raise AuditError(f"{result_path}: unsupported arm {arm!r}")
    if str(seed) != task.name:
        raise AuditError(f"{task}: directory seed differs from result seed {seed}")
    if result.get("epochs_requested") != expected_epochs:
        raise AuditError(f"{result_path}: epochs_requested != {expected_epochs}")
    config = {"history_alpha": result.get("history_alpha"),
              "proximal_ratio": result.get("proximal_ratio")}
    if arm in PI_ARMS:
        config["mass_ratio"] = result.get("mass_ratio")
    for k, v in config.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise AuditError(f"{result_path}: missing or invalid {k}")
    records = {}
    with metrics_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            e = row["epoch"]
            if e in records:
                raise AuditError(f"{metrics_path}: duplicate epoch {e}")
            records[e] = SA._validation(row["validation"], f"{metrics_path} epoch {e}")
    if set(records) != set(range(1, expected_epochs + 1)):
        raise AuditError(f"{task}: incomplete epoch history {sorted(records)}")
    selected = max(records, key=lambda e: (records[e]["accuracy"], -records[e]["cross_entropy"], -e))
    if result.get("selected_epoch") != selected:
        raise AuditError(f"{task}: selected_epoch {result.get('selected_epoch')} != rule {selected}")
    if SA._validation(result["validation"], str(result_path)) != records[selected]:
        raise AuditError(f"{task}: selected validation differs from metrics.jsonl")
    epochs = sorted({selected, expected_epochs})
    ckpts = {e: task / f"checkpoint_epoch_{e:02d}.msgpack" for e in epochs}
    for e, c in ckpts.items():
        if not c.is_file():
            raise AuditError(f"missing {c}")
    return {"task": str(task), "arm": arm, "seed": int(seed), "config": config,
            "selected_epoch": selected, "validation": records[selected],
            "final_validation": records[expected_epochs],
            "checkpoints": {e: str(c) for e, c in ckpts.items()},
            "sha256": {e: CC._sha256(c) for e, c in ckpts.items()},
            "task_result_sha256": CC._sha256(result_path)}


# ------------------------------------------------------- block algebra ---

def kappa64(group):
    re = np.minimum(np.asarray(group["Lambda_re"], np.float64), -1e-4)
    lam = re + 1j * np.asarray(group["Lambda_im"], np.float64)
    delta = np.exp(np.asarray(group["log_step"], np.float64)[:, 0])
    return -lam * delta, delta


def blocks64(arm, config, kappa):
    """Float64 SciPy exact-ZOH (Ad, bd) per mode, independent of the JAX code."""
    alpha, pr = config["history_alpha"], config["proximal_ratio"]
    Ads, bds = [], []
    for k in kappa:
        h = alpha / abs(k)
        if arm == P_ARM:   # tau s''+s'=-r_mu-tau q', tau q'=r_mu-q; state (s, tau s', tau q)
            d = 1 + pr * h * k
            ke = k / d
            A = np.array([[0, 1 / h, 0], [-2 * ke, -1 / h, 1 / h], [ke, 0, -1 / h]], complex)
            g = np.array([0, 2, -1], complex) / d
        else:              # (s, w = m s' + tau r_mu)
            tau, mu, m = h, pr * h, config["mass_ratio"] * h
            d = 1 + mu * k
            A = np.array([[-tau * k / (m * d), 1 / m], [(tau / m - 1) * k / d, -1 / m]], complex)
            g = np.array([tau / m, 1 - tau / m], complex) / d
        n = len(g)
        aug = np.zeros((n + 1, n + 1), complex)
        aug[:n, :n], aug[:n, n] = A, g
        E = expm(aug)
        Ads.append(E[:n, :n]); bds.append(E[:n, n])
    return np.asarray(Ads), np.asarray(bds)


def blocks32(arm, config, group):
    """The arm's own JAX constructor in float32/complex64 on CPU."""
    import jax
    import jax.numpy as jnp
    if jax.config.read("jax_enable_x64"):
        raise AuditError("run with JAX_ENABLE_X64=0 (float64 analysis uses NumPy/SciPy)")
    lam = jnp.clip(jnp.asarray(group["Lambda_re"]), None, -1e-4) + 1j * jnp.asarray(group["Lambda_im"])
    kappa = -lam * jnp.exp(jnp.asarray(group["log_step"])[:, 0])
    if arm == P_ARM:
        from s5.proximal_history import proximal_history_zoh
        Ad, bd, _ = proximal_history_zoh(kappa, config["history_alpha"], config["proximal_ratio"])
    else:
        from s5.proximal_inertia import proximal_inertia_zoh
        Ad, bd, _ = proximal_inertia_zoh(kappa, config["history_alpha"], config["mass_ratio"],
                                         config["proximal_ratio"])
    return np.asarray(jax.device_get(Ad)), np.asarray(jax.device_get(bd))


def impulse(Ad, bd, horizon):
    """h[t, p] = (Ad^t bd)_s, the position response to a unit physical drive."""
    x = bd.copy()
    h = np.empty((horizon, Ad.shape[0]), complex)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(horizon):
            h[t] = x[:, 0]
            x = np.einsum("pij,pj->pi", Ad, x)
    return h


def components(Ad, bd, horizon):
    """Per-pole waves r_j z_j^t from the block eigendecomposition."""
    P, n = bd.shape
    z = np.full((P, n), np.nan + 0j)
    r = np.full((P, n), np.nan + 0j)
    cond = np.full(P, np.inf)
    for p in range(P):
        w, V = np.linalg.eig(Ad[p])
        order = np.argsort(-abs(w))
        w, V = w[order], V[:, order]
        c = np.linalg.cond(V)
        z[p] = w
        cond[p] = c
        if np.isfinite(c) and c < COND_LIMIT:
            r[p] = V[0] * np.linalg.solve(V, bd[p])
    t = np.arange(horizon)[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        waves = [r[:, j][None, :] * z[:, j][None, :] ** t for j in range(n)]
    return z, r, cond, waves


# ------------------------------------------------------------ analysis ---

def analyze_layer(arm, config, name, group, horizon):
    kappa, delta = kappa64(group)
    Ad, bd = blocks64(arm, config, kappa)
    Ad32, bd32 = blocks32(arm, config, group)
    z64 = np.linalg.eigvals(Ad)
    z64 = np.take_along_axis(z64, np.argsort(-abs(z64), axis=1), axis=1)
    z32 = np.linalg.eigvals(Ad32.astype(np.complex128))
    z32 = np.take_along_axis(z32, np.argsort(-abs(z32), axis=1), axis=1)
    r64, r32 = abs(z64[:, 0]), abs(z32[:, 0])
    finite = np.isfinite(r64) & np.isfinite(r32)
    native = np.exp(-kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        dom_ratio = np.log(abs(native)) / np.log(abs(z64[:, 0]))
    poles = summarize_poles(z64)
    prod = {"stable": int((finite & (r32 < 1 - TOL) & (r64 < 1 - TOL)).sum()),
            "numerical_boundary": int((finite & (r64 <= 1 + TOL) & ~((r32 < 1 - TOL) & (r64 < 1 - TOL))).sum()),
            "unstable": int((finite & (r64 > 1 + TOL)).sum()),
            "nonfinite": int((~finite).sum()),
            "max_radius_float32": float(np.nanmax(r32)), "max_radius_float64": float(np.nanmax(r64)),
            "max_abs_radius_difference": float(np.nanmax(abs(r32 - r64))),
            "max_abs_block_difference": float(np.nanmax(abs(Ad32 - Ad))),
            "boundary_tol": TOL}
    h = impulse(Ad, bd, horizon)
    z, res, cond, waves = components(Ad, bd, horizon)
    resolved = np.isfinite(cond) & (cond < COND_LIMIT)
    with np.errstate(over="ignore", invalid="ignore"):
        recon = np.sum(waves, axis=0)
        scale = np.max(abs(h), axis=0)
        recon_err = np.where(resolved, np.max(abs(recon - h), axis=0) / np.maximum(scale, 1e-300), np.nan)
        rel_res = abs(res) / np.sum(abs(res), axis=1, keepdims=True)
    B = delta[:, None] * CC._complex(np.asarray(group["B"]))       # physical drive Delta B~
    h_native = native[None, :] ** np.arange(horizon)[:, None] * ((1 - native) / kappa)[None, :]
    lag = np.unique(np.r_[0, np.rint(np.geomspace(1, horizon - 1, 96)).astype(int), horizon - 1])
    out = {"layer": name, "n_modes": int(len(kappa)), "block_width": int(bd.shape[1]),
           "poles_float64": poles["summary"], "production_precision": prod,
           "dominant_decay_over_native_counterfactual": CC.q(dom_ratio),
           "auxiliary_decay_samples": [CC.q(poles["decay_samples"][:, j])
                                       for j in range(1, bd.shape[1])],
           "auxiliary_relative_residue": [CC.q(np.where(resolved, rel_res[:, j], np.nan))
                                          for j in range(1, bd.shape[1])],
           "eigenvector_condition": CC.q(cond),
           "components_withheld_ill_conditioned": int((~resolved).sum()),
           "component_reconstruction_relative_error_max": float(np.nanmax(recon_err)) if resolved.any() else None,
           "poles": {"float64": z64, "float32": z32}, "kappa": kappa,
           "lag_samples": lag, "directions": {}}
    for direction, key in (("forward", "C1"), ("backward", "C2")):
        C = CC._complex(np.asarray(group[key]))
        energy, audit, G, Hm = MI._real_kernel_energy(h, B, C)
        total, mode_audit = MI._mode_energy(h, G, Hm)
        comp = []
        for j, wave in enumerate(waves):
            ce, _ = MI._mode_energy(np.where(resolved[None, :], wave, 0), G, Hm)
            ce = np.where(resolved, ce, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                comp.append({"pole_rank": j, "fraction_of_mode_energy": CC.q(ce / total)})
        nat_energy, _, _, _ = MI._real_kernel_energy(h_native, B, C)
        out["directions"][direction] = {
            "curve_energy": energy[lag], "summary": MI._energy_summary(energy),
            "numerical_audit": audit, "mode_energy": CC.q(total),
            "components": comp,
            "native_counterfactual_summary": MI._energy_summary(nat_energy)}
    return out


def stage_report(arm, config, params, horizon, where):
    groups = CC.layer_groups(params, "", where)
    for name, g in groups.items():
        if any(np.asarray(v).dtype != np.float32 for v in g.values()):
            raise AuditError(f"{where}/{name}: expected saved float32 parameters")
    layers = [analyze_layer(arm, config, n, g, horizon) for n, g in sorted(groups.items())]
    agg = {k: sum(l["production_precision"][k] for l in layers)
           for k in ("stable", "numerical_boundary", "unstable", "nonfinite")}
    return {"layers": layers, "production_precision_totals": agg,
            "max_radius_float64": max(l["production_precision"]["max_radius_float64"] for l in layers)}


def init_params(seed):
    from flax.traverse_util import unflatten_dict
    from experiments.s5_predictor_corrector.production_init import production_init_flat
    return unflatten_dict(production_init_flat("native_matched_s5", seed))


def analyze_task(row, horizon):
    out = {k: v for k, v in row.items() if k not in ("checkpoints",)}
    out["stages"] = {"reconstructed_init": stage_report(
        row["arm"], row["config"], init_params(row["seed"]), horizon, "init")}
    for e, path in sorted(row["checkpoints"].items()):
        label = "selected" if e == row["selected_epoch"] else "final"
        if e == row["selected_epoch"] and len(row["checkpoints"]) == 1:
            label = "selected_and_final"
        out["stages"][label] = stage_report(row["arm"], row["config"], CC.read_params(path),
                                            horizon, path)
        out["stages"][label]["epoch"] = e
    return out


def digest(tasks):
    lines = ["arm seed sel_epoch val_acc stage f64_ge1 f32[stable/boundary/unstable] "
             "max|z| dom/native_p50 aux_decay_p50 aux_decay_p95 fwd_L90_med"]
    for t in tasks:
        for stage, s in t["stages"].items():
            L = s["layers"]
            pp = s["production_precision_totals"]
            dom = np.median([l["dominant_decay_over_native_counterfactual"]["p50"] for l in L])
            aux = [l["auxiliary_decay_samples"][0] for l in L]
            a50 = np.median([a["p50"] for a in aux if a])
            a95 = np.median([a["p95"] for a in aux if a])
            l90 = np.median([l["directions"]["forward"]["summary"]["energy_quantile_lags"]["90"] or np.nan
                             for l in L])
            ge1 = sum(l["poles_float64"]["poles_ge_one"] for l in L)
            acc = "-" if t["validation"] is None else f"{t['validation']['accuracy']:.4f}"
            lines.append(f"{t['arm']} {t['seed']} {t['selected_epoch']} {acc} "
                         f"{stage} {ge1} {pp['stable']}/{pp['numerical_boundary']}/{pp['unstable']} "
                         f"{s['max_radius_float64']:.7f} {dom:.4f} {a50:.2f} {a95:.2f} {l90:.0f}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument("--init-arm", choices=PI_ARMS,
                    help="audit only the reconstructed production initialization of this arm")
    ap.add_argument("--init-seed", type=int, action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=16000)
    ap.add_argument("--allow-missing", action="store_true",
                    help="record tasks without a completed result as unavailable instead of failing")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise AuditError(f"refusing non-empty output {out}")
    if bool(args.task) == bool(args.init_arm):
        raise AuditError("give either --task ... or --init-arm with --init-seed")
    rows, unavailable = [], []
    if args.init_arm:
        from s5 import proximal_inertia as PIM
        mass = PIM.MASS_RATIO if args.init_arm == PI_ARMS[0] else PIM.CONTROL_MASS_RATIO
        for seed in args.init_seed or [301]:
            rows.append({"task": None, "arm": args.init_arm, "seed": seed,
                         "config": {"history_alpha": PIM.ALPHA, "proximal_ratio": PIM.PROXIMAL_RATIO,
                                    "mass_ratio": mass},
                         "selected_epoch": None, "validation": None, "checkpoints": {}, "sha256": {}})
    for t in args.task:
        try:
            rows.append(resolve_task(t))
        except (AuditError, OSError, json.JSONDecodeError) as exc:
            if not args.allow_missing:
                raise
            unavailable.append({"task": str(t), "reason": str(exc)})
    if not rows:
        raise AuditError(f"no auditable task: {unavailable}")
    tasks = [analyze_task(r, args.horizon) for r in rows]
    changed = [r["task"] for r in rows
               if {e: CC._sha256(p) for e, p in r["checkpoints"].items()} != r["sha256"]]
    if changed:
        raise AuditError(f"source checkpoints changed during the audit: {changed}")
    report = {"scope": __doc__, "horizon": args.horizon, "boundary_tol": TOL,
              "tasks": tasks, "unavailable": unavailable,
              "sources_unchanged": True, "test_split_read": False}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "block_audit.json").open("x") as fh:
        json.dump(SA._jsonable(report), fh, indent=1, allow_nan=False)
    text = digest(tasks)
    if unavailable:
        text += "unavailable:\n" + "".join(f"  {u['task']}: {u['reason']}\n" for u in unavailable)
    (out / "summary.txt").write_text(text)
    print(text, flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        raise SystemExit(f"FAIL: {exc}")
