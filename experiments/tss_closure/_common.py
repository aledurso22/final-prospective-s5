"""Shared plumbing: deterministic seeds, config/result saving, plot style."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

RESULTS_ROOT = "results/tss_closure"

# validated categorical palette (dataviz validator: all checks PASS)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def outdir(name):
    d = os.path.join(RESULTS_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def save(name, config, results):
    d = outdir(name)
    payload = {"config": config, "results": results}
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=_enc)
    return os.path.join(d, "summary.json")


def _enc(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, complex): return {"re": o.real, "im": o.imag}
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    return str(o)


def style(ax, title=None, xlabel=None, ylabel=None, legend=False):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    if title: ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if legend: ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
