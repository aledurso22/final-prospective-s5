"""Per-layer, per-seed pole-decay distributions for the checkpoint audit.

The ECDFs are conditional on a pole having a finite, positive decay time.
Excluded-pole and non-decaying-channel counts are shown explicitly.  These
figures measure homogeneous pole e-folding times, not accessible task memory.
"""

from pathlib import Path

import numpy as np


ARMS = (
    ("native_matched_s5", "N", "#343434"),
    ("generalized_prospective_s5", "A", "#c56b16"),
    ("calibrated_old_s5", "C", "#126ca6"),
)
STAGES = ("reconstructed_init", "trained")
HORIZON = 16000


def _decay_values(layer, branch):
    """Eligible decay values for ONE pole family, never a joint-stability mask."""
    radius = np.asarray(layer["radius"], float)
    decay = np.asarray(layer["decay_samples"], float)
    if radius.ndim != 2 or radius.shape != decay.shape:
        raise ValueError("radius and decay_samples must be matching 2-D arrays")
    if branch >= radius.shape[1]:
        raise ValueError("missing requested pole family")
    r, d = radius[:, branch], decay[:, branch]
    valid = np.isfinite(r) & (r > 0) & (r < 1) & np.isfinite(d) & (d > 0)
    return np.sort(d[valid]), int(r.size - valid.sum())


def _tasks_by_seed(report):
    tasks = {}
    known = {name for name, _, _ in ARMS}
    for task in report["tasks"]:
        if task["arm"] not in known:
            raise ValueError(f"unsupported plotting arm: {task['arm']}")
        seed = int(task["seed"])
        group = tasks.setdefault(seed, {})
        if task["arm"] in group:
            raise ValueError(f"duplicate plotting task: {task['arm']}/{seed}")
        group[task["arm"]] = task
    if not tasks:
        raise ValueError("no checkpoint tasks to plot")
    for seed, group in tasks.items():
        if set(group) != known:
            raise ValueError(f"seed {seed} must contain N, A and C")
        layer_names = None
        for arm, _, _ in ARMS:
            for stage in STAGES:
                layers = group[arm]["stages"][stage]["layers"]
                names = [layer["layer"] for layer in layers]
                if len(names) != len(set(names)):
                    raise ValueError("duplicate layer names")
                if layer_names is None:
                    layer_names = names
                if names != layer_names or len(names) != 6:
                    raise ValueError("plots require the same six ordered layers in every arm/stage")
                expected_poles = 1 if arm == ARMS[0][0] else 2
                for layer in layers:
                    radius = np.asarray(layer["radius"], float)
                    if radius.ndim != 2 or radius.shape[1] != expected_poles:
                        raise ValueError(f"{arm} requires {expected_poles} pole(s) per channel")
                    for branch in range(expected_poles):
                        _decay_values(layer, branch)
    return tasks


def plot_report(report, out):
    """Write one six-layer PNG/PDF pair per seed; return the new file paths.

    Inputs retain raw per-pole arrays.  Native's artificial zero pole from a
    companion realization is deliberately not accepted as a second pole.
    Existing figures are never overwritten.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    tasks = _tasks_by_seed(report)
    out = Path(out)
    outputs = [out / f"pole_memory_seed_{seed}.{ext}"
               for seed in sorted(tasks) for ext in ("png", "pdf")]
    if any(path.exists() for path in outputs):
        raise ValueError("refusing to overwrite existing pole-memory figures")
    out.mkdir(parents=True, exist_ok=True)
    for seed in sorted(tasks):
        group = tasks[seed]
        all_values = []
        for task in group.values():
            for stage in STAGES:
                for layer in task["stages"][stage]["layers"]:
                    for branch in range(np.asarray(layer["radius"]).shape[1]):
                        values, _ = _decay_values(layer, branch)
                        if values.size:
                            all_values.append(values)
        values = np.concatenate(all_values) if all_values else np.array([1.0])
        x_lo = min(float(values.min()), 1.0) / 1.3
        x_hi = max(float(values.max()), HORIZON) * 1.3

        fig = Figure(figsize=(12.5, 19))
        FigureCanvasAgg(fig)
        axes = fig.subplots(6, 2, sharex=True, sharey=True)
        epochs = ", ".join(f"{label}: {group[arm]['epoch']}" for arm, label, _ in ARMS)
        fig.suptitle(f"Pole decay distributions — seed {seed}\n"
                     f"Selected checkpoint epochs: {epochs}", fontsize=15, y=.988)
        handles = []
        for arm, label, color in ARMS:
            branches = 1 if label == "N" else 2
            for branch in range(branches):
                family = "dominant" if branch == 0 else "secondary"
                handles.append(Line2D([], [], color=color,
                                      linestyle="-" if branch == 0 else "--",
                                      label=f"{label} {family}"))
        handles.append(Line2D([], [], color="#888888", linestyle=":",
                              label="16,000-sample sequence"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .949),
                   ncol=3, frameon=False, fontsize=10)

        for row in range(6):
            for column, stage in enumerate(STAGES):
                ax = axes[row, column]
                labels, channel_counts = [], []
                for arm, label, color in ARMS:
                    layer = group[arm]["stages"][stage]["layers"][row]
                    radius = np.asarray(layer["radius"], float)
                    channel_counts.append(f"{label} {np.any(radius >= 1, axis=1).sum()}/{len(radius)}")
                    for branch in range(radius.shape[1]):
                        vals, excluded = _decay_values(layer, branch)
                        short = label if label == "N" else label + ("d" if branch == 0 else "s")
                        labels.append(f"{short} {excluded}/{len(radius)}")
                        if vals.size:
                            # Include the tails explicitly: the conditional ECDF
                            # is zero below the minimum, one above the maximum.
                            x = np.r_[x_lo, vals, x_hi]
                            y = np.r_[0., np.arange(1, len(vals) + 1) / len(vals), 1.]
                            ax.step(x, y, where="post", color=color, linewidth=1.5,
                                    linestyle="-" if branch == 0 else "--")
                ax.axvline(HORIZON, color="#888888", linestyle=":", linewidth=1)
                ax.set_xscale("log")
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(0, 1.04)
                ax.grid(True, alpha=.2)
                title = "Reconstructed float32 initialization" if column == 0 else "Trained selected checkpoint"
                layer_name = group[ARMS[0][0]]["stages"][stage]["layers"][row]["layer"]
                ax.set_title(f"{layer_name}\n{title}", fontsize=10, pad=28)
                ax.text(0, 1.025, "Channels |pole| ≥ 1: " + "; ".join(channel_counts),
                        transform=ax.transAxes, fontsize=7.5, va="bottom")
                ax.text(.015, .035, "Excluded L: " + "; ".join(labels),
                        transform=ax.transAxes, fontsize=7, va="bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 2})
                if column == 0:
                    ax.set_ylabel("Conditional cumulative fraction", fontsize=9)
                if row == 5:
                    ax.set_xlabel("Pole amplitude e-folding time L (samples)", fontsize=10, labelpad=10)
        fig.text(.5, .016,
                 "ECDF denominator: stable, nonzero poles with finite positive L within each displayed family.\n"
                 "Excluded L counts are per family (d/s = dominant/secondary), with no pole clipping. Float64 equations.\n"
                 "Initialization is reconstructed, not a saved initial checkpoint. Pole decay is not usable memory;\n"
                 "input excitation, readout coupling, cancellations and repeated-root transients require separate analysis.",
                 ha="center", va="bottom", fontsize=9)
        fig.subplots_adjust(left=.09, right=.98, top=.885, bottom=.105, hspace=.70, wspace=.19)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"pole_memory_seed_{seed}.{ext}", dpi=160)
        # Figure is not registered with pyplot; release it without resetting
        # shared log axes (Figure.clear would emit a spurious xlim warning).
        del axes, fig
    return [str(path) for path in outputs]
