"""Finite-window, observable linear SSM impulse responses from frozen weights.

The real conjugate-pair kernel is 2 Re(sum_p C_p h_p[t] Bbar_p).
Gram products retain interference between channels and within conjugate pairs.
Directions are reported separately; their energy sum is explicitly not the
energy of a bidirectional network output. D skips and the nonlinear backbone
are excluded. No dataset, model evaluation or training is involved.
"""

from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from . import checkpoint_compare as CC


NATIVE = "native_matched_s5"
TAIL_LAGS = (16, 128, 1024, 8192)


def _impulse_sequences(coefficients, horizon):
    a1, a2, g0, g1 = (np.asarray(v, complex).reshape(-1) for v in coefficients)
    if len({v.shape for v in (a1, a2, g0, g1)}) != 1:
        raise ValueError("per-channel coefficient shapes differ")
    impulse = np.zeros(horizon); impulse[0] = 1
    with np.errstate(over="ignore", invalid="ignore"):
        return np.column_stack([lfilter([g0[p], g1[p]], [1., -a1[p], -a2[p]], impulse)
                                for p in range(len(a1))])


def _sanitize_energy(raw, scale):
    """Correct only bounded negative roundoff, while exposing larger failures."""
    raw = np.asarray(raw, float)
    scale = np.broadcast_to(np.asarray(scale, float), raw.shape)
    tolerance = 128 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny)
    finite = np.isfinite(raw) & np.isfinite(scale)
    roundoff = finite & (raw < 0) & (raw >= -tolerance)
    negative = finite & (raw < -tolerance)
    cleaned = raw.copy()
    cleaned[roundoff] = 0.
    return cleaned, {"roundoff_negative_values_set_to_zero": int(roundoff.sum()),
                     "significant_negative_values": int(negative.sum()),
                     "nonfinite_values": int((~finite).sum()),
                     "minimum_raw_energy": float(np.min(raw)) if raw.size else None,
                     "maximum_roundoff_tolerance": float(np.max(tolerance)) if raw.size else None}


def _real_kernel_energy(h, B, C):
    """Exact Frobenius kernel energy via P x P Grams, not a T x H x H tensor."""
    G = (C.conj().T @ C) * (B.conj() @ B.T)
    H = (C.T @ C) * (B @ B.T)
    with np.errstate(over="ignore", invalid="ignore"):
        complex_norm = np.einsum("ti,ij,tj->t", h.conj(), G, h, optimize=True).real
        real_cross = np.einsum("ti,ij,tj->t", h, H, h, optimize=True).real
        energy = 2 * (complex_norm + real_cross)
        # Frobenius-norm bound includes cancellation inside the Gram forms.
        scale = 2 * np.sum(abs(h) ** 2, axis=1) * (np.linalg.norm(G) + np.linalg.norm(H))
    cleaned, audit = _sanitize_energy(energy, scale)
    return cleaned, audit, G, H


def _mode_energy(h, G, H):
    """Per-mode real kernel energy; mode energies do not sum to full energy."""
    gd, hd = np.diag(G).real, np.diag(H)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = 2 * (abs(h) ** 2 * gd + (h * h * hd).real)
        scale = 2 * abs(h) ** 2 * (abs(gd) + abs(hd))
    energy, audit = _sanitize_energy(raw, scale)
    valid = np.isfinite(energy).all(axis=0) & (energy >= 0).all(axis=0)
    totals = np.sum(energy, axis=0)
    totals[~valid] = np.nan
    return totals, audit


def _energy_summary(energy):
    energy = np.asarray(energy, float)
    base = {"horizon_samples": int(len(energy)),
            "energy_quantile_lags": {str(q): None for q in (50, 90, 99)},
            "fraction_energy_at_or_after_lag": {str(t): None for t in TAIL_LAGS}}
    if not np.isfinite(energy).all():
        return {**base, "status": "nonfinite_response", "total_energy": None}
    if np.any(energy < 0):
        return {**base, "status": "significant_negative_energy", "total_energy": None}
    cumulative = np.cumsum(energy)
    total = float(cumulative[-1])
    if not np.isfinite(total):
        return {**base, "status": "nonfinite_total", "total_energy": None}
    if total == 0:
        return {**base, "status": "zero_response", "total_energy": 0.}
    base["energy_quantile_lags"] = {
        str(q): int(np.searchsorted(cumulative, total * q / 100)) for q in (50, 90, 99)}
    base["fraction_energy_at_or_after_lag"] = {
        str(t): float(np.sum(energy[t:]) / total) for t in TAIL_LAGS}
    return {**base, "status": "finite", "total_energy": total}


def _components(coefficients, horizon, native):
    a1, a2, g0, g1 = (np.asarray(v, complex).reshape(-1) for v in coefficients)
    if native:
        roots, residues = a1[:, None], g0[:, None]
        resolved = np.ones(len(a1), bool)
        separation = np.full(len(a1), np.nan)
    else:
        disc = np.sqrt(a1 * a1 + 4 * a2)
        sign = np.where((a1.conj() * disc).real >= 0, 1., -1.)
        r0 = (a1 + sign * disc) / 2
        r1 = np.divide(-a2, r0, out=np.zeros_like(r0), where=(r0 != 0))
        roots = np.column_stack([r0, r1])
        separation = abs(r0 - r1) / np.maximum(1., np.maximum(abs(r0), abs(r1)))
        resolved = np.isfinite(separation) & (separation > 1e-6)
        residues = np.full(roots.shape, complex(np.nan, np.nan))
        residues[resolved, 0] = (g0[resolved] * r0[resolved] + g1[resolved]) / (r0[resolved] - r1[resolved])
        residues[resolved, 1] = -(g0[resolved] * r1[resolved] + g1[resolved]) / (r0[resolved] - r1[resolved])
    waves = []
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(roots.shape[1]):
            # Unresolved channels remain NaN; the direct response is unaffected.
            waves.append(residues[:, j][None, :] * roots[:, j][None, :] ** np.arange(horizon)[:, None])
    diag = {"resolved": resolved, "relative_separation": separation,
            "near_coincident_threshold": 1e-6,
            "native_single_pole": native,
            "relative_residue_cancellation_threshold": 1e-8}
    if not native:
        denominator = abs(residues).sum(axis=1)
        relative = np.divide(abs(residues[:, 1]), denominator,
                             out=np.full(len(a1), np.nan), where=(denominator > 0))
        diag.update(secondary_relative_residue=relative,
                    secondary_residue_negligible=resolved & (relative <= 1e-8))
    return waves, diag


def analyze_impulses(arm, group, coefficients64, horizon=16000):
    """Analyse a frozen layer over lags 0..horizon-1, even if poles grow.

    Coefficients factor out Bbar, matching selected_checkpoint_audit.  Component
    energies include real conjugate-pair interference but omit interference
    with the OTHER root; therefore their fractions need not sum to one.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    h = _impulse_sequences(coefficients64, horizon)
    quantities = CC.mode_quantities(group)
    B = CC._complex(np.asarray(group["B"]))
    B = ((quantities["lambda_bar"] - 1) / quantities["Lambda"])[:, None] * B
    if B.shape[0] != h.shape[1]:
        raise ValueError("input coupling and recurrence channel counts differ")
    lag = np.unique(np.r_[0, np.rint(np.geomspace(1, max(1, horizon - 1), 128)).astype(int),
                          [t for t in TAIL_LAGS if t < horizon], horizon - 1])
    lag = lag[lag < horizon]
    component_waves, residues = _components(coefficients64, horizon, arm == NATIVE)
    out = {"horizon": horizon, "lag_samples": lag,
           "scope": "Real linear SSM kernel only; D skip and nonlinear backbone excluded. Forward/backward are separate directions. Branch sum omits interference between directions. All energies are finite-window, not infinite-horizon stability claims.",
           "component_scope": "Per-mode root-component energies; real conjugate-pair interference included, cross-root interference excluded. Fractions of exact mode energy may exceed one. Near-coincident decompositions withheld.",
           "direct_response_nonfinite_values": int((~np.isfinite(h)).sum()),
           "residue_diagnostic": residues, "directions": {}}
    curves, zero_lag_kernels = [], []
    for direction, key in (("forward", "C1"), ("backward", "C2")):
        C = CC._complex(np.asarray(group[key]))
        if C.shape[1] != h.shape[1]:
            raise ValueError("readout coupling and recurrence channel counts differ")
        zero_lag_kernels.append(2 * ((C * h[0][None, :]) @ B).real)
        energy, audit, G, H = _real_kernel_energy(h, B, C)
        curves.append(energy)
        mode_total, mode_audit = _mode_energy(h, G, H)
        components = {}
        energies = []
        for name, wave in zip(("dominant", "secondary"), component_waves):
            component_total, component_audit = _mode_energy(wave, G, H)
            energies.append(component_total)
            ratio = np.divide(component_total, mode_total, out=np.full_like(mode_total, np.nan),
                              where=np.isfinite(mode_total) & (mode_total > 0))
            components[name] = {"mode_energy": component_total,
                                "fraction_of_mode_total": ratio,
                                "numerical_audit": component_audit}
        with np.errstate(invalid="ignore"):
            interference = mode_total - np.sum(energies, axis=0)
        out["directions"][direction] = {
            "curve_energy": energy[lag], "summary": _energy_summary(energy),
            "numerical_audit": audit, "mode_numerical_audit": mode_audit,
            "mode_total_energy": mode_total,
            "zero_mode_energy": np.isfinite(mode_total) & (mode_total == 0),
            "components": components,
            "mode_cross_root_interference_energy": interference}
    with np.errstate(over="ignore", invalid="ignore"):
        combined = curves[0] + curves[1]
    out["branch_sum"] = {"curve_energy": combined[lag], "summary": _energy_summary(combined),
                         "scope": "Sum of separate forward/backward kernel energies; not combined bidirectional output energy."}
    # For a unit impulse in the interior of a bilateral sequence, the two
    # directions occupy opposite lags and overlap only at lag zero.  This
    # includes that overlap exactly, while still excluding the D skip.
    bilateral = combined.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        bilateral[0] = np.sum((zero_lag_kernels[0] + zero_lag_kernels[1]) ** 2)
    out["bidirectional_recurrent"] = {
        "curve_energy_by_absolute_lag": bilateral[lag], "summary": _energy_summary(bilateral),
        "signed_lag_range": [-(horizon - 1), horizon - 1],
        "scope": "Interior bilateral recurrent kernel, grouped by absolute lag; forward/backward zero-lag interference included. Each direction truncated at horizon. Finite-sequence boundaries, D skip and nonlinear backbone excluded."}
    return out


def plot_impulse_report(report, out):
    """Selected/init per-layer normalized energy-vs-lag curves, one PNG/PDF per seed."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from .memory_pole_plots import ARMS, STAGES, _tasks_by_seed

    tasks = _tasks_by_seed(report)
    out = Path(out)
    outputs = [out / f"observable_impulse_seed_{seed}.{ext}"
               for seed in sorted(tasks) for ext in ("png", "pdf")]
    if any(path.exists() for path in outputs):
        raise ValueError("refusing to overwrite observable impulse figures")
    out.mkdir(parents=True, exist_ok=True)
    for seed, group in sorted(tasks.items()):
        fig = Figure(figsize=(12.5, 19)); FigureCanvasAgg(fig)
        axes = fig.subplots(6, 2, sharex=True, sharey=True)
        epochs = ", ".join(f"{label}: {group[arm]['epoch']}" for arm, label, _ in ARMS)
        fig.suptitle(f"Observable linear SSM impulse energy — seed {seed}\nSelected epochs: {epochs}",
                     fontsize=15, y=.988)
        fig.legend(handles=[Line2D([], [], color=color, label=label) for _, label, color in ARMS],
                   loc="upper center", bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
        for row in range(6):
            for column, stage in enumerate(STAGES):
                ax = axes[row, column]
                annotations = []
                for arm, label, color in ARMS:
                    layer = group[arm]["stages"][stage]["layers"][row]
                    impulse = layer["impulse_response"]
                    summary = impulse["branch_sum"]["summary"]
                    total = summary["total_energy"]
                    if summary["status"] == "finite":
                        x = np.asarray(impulse["lag_samples"]) + 1
                        y = np.asarray(impulse["branch_sum"]["curve_energy"]) / total
                        valid = np.isfinite(y) & (y > 0)
                        ax.plot(x[valid], y[valid], color=color, linewidth=1.5)
                        annotations.append(f"{label} E={total:.2g}; L90={summary['energy_quantile_lags']['90']}")
                    else:
                        annotations.append(f"{label}: {summary['status']}")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.grid(True, alpha=.2)
                title = "Reconstructed initialization" if column == 0 else "Trained selected checkpoint"
                layer_name = group[ARMS[0][0]]["stages"][stage]["layers"][row]["layer"]
                ax.set_title(f"{layer_name}\n{title}", fontsize=10)
                ax.text(.02, .03, "\n".join(annotations), transform=ax.transAxes, fontsize=7.5,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 2})
                if column == 0:
                    ax.set_ylabel("Energy at lag / finite-window total", fontsize=9)
                if row == 5:
                    ax.set_xlabel("Lag + 1 (samples; lag 0 retained)")
        fig.text(.5, .025,
                 "Each curve is normalized by its own full-window energy E, shown in the panel; sampled lags only.\n"
                 "Within each direction, the real kernel includes conjugate-pair and cross-mode interference.\n"
                 "The plotted sum combines direction energies, not bidirectional outputs. D skip/nonlinear backbone excluded.\n"
                 "L90 is the lag containing 90% of finite-window energy; pole e-folding time is a different quantity.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.10, right=.98, top=.89, bottom=.10, hspace=.48, wspace=.18)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"observable_impulse_seed_{seed}.{ext}", dpi=160)
        del axes, fig
    return [str(path) for path in outputs]
