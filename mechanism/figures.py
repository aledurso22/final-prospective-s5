"""Figures for the professor-mechanism study.

Palette is the validated categorical set (blue/orange/aqua/yellow), checked
with the dataviz validator: lightness band, chroma floor, CVD separation and
normal-vision floor all PASS. Contrast vs surface WARNs for aqua/yellow, so
every series carries a visible label AND a distinct linestyle - identity is
never colour alone.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

PLAIN = "#2a78d6"
FULL = "#eb6834"
PROJ = "#1baf7a"
STATIC = "#eda100"

LW = 2.0


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=8, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def mechanism_overview(system, models, outdir):
    """THE figure: memory (top) and tracking (bottom) for the three mechanisms.

    Columns are mechanisms so the reader compares like with like vertically.
    """
    from mechanism import metrics

    cols = [("A. Plain SSM", "A_plain", PLAIN),
            ("B. Full state PC (Idea 1)", "B_full_state_pc", FULL),
            ("C. Projected state PC (Idea 2)", "C_projected_state_pc", PROJ)]

    L, t_on = 900, 20
    t = np.arange(L) * system.dt
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), facecolor=SURFACE,
                             sharey="row")

    for j, (label, key, colour) in enumerate(cols):
        m = models[key]
        imp = metrics.impulse_response(m, length=L)
        step = metrics.step_response(m, length=L, t_on=t_on)

        ax = axes[0, j]
        ref = metrics.impulse_response(models["A_plain"], length=L)[:, 0]
        ax.plot(t, ref / ref[0], color=MUTED, lw=1.2, ls=(0, (4, 3)),
                label="plain reference")
        h0 = imp[0, 0] if imp[0, 0] != 0 else 1.0
        ax.plot(t, imp[:, 0] / h0, color=colour, lw=LW, label="memory mode")
        _style(ax, title=label,
               ylabel="memory-mode impulse\n(normalised)" if j == 0 else None)
        ax.set_ylim(-0.1, 1.15)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
        retained = imp[-1, 0] / h0
        ax.text(0.97, 0.16, f"retained at {t[-1]:.0f}s: {retained:.2f}",
                transform=ax.transAxes, ha="right", fontsize=8.5,
                color=INK if retained > 0.05 else "#d03b3b")

        ax = axes[1, j]
        target = np.zeros(L); target[t_on:] = system.K[1]
        ax.plot(t, target, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="target")
        ax.plot(t, step[:, 1], color=colour, lw=LW, label="tracking mode")
        _style(ax, xlabel="time (s)",
               ylabel="tracking-mode\nstep response" if j == 0 else None)
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.15, 2.05)          # headroom so the label clears the trace
        ttc = metrics.time_to_correct(step[:, 1], system.K[1], system.dt,
                                      tol=0.05, t_on=t_on)
        ax.text(0.97, 0.90, f"time-to-correct: {ttc:.2f}s",
                transform=ax.transAxes, ha="right", fontsize=8.5, color=INK)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2,
                      loc="lower right")

    fig.suptitle("Memory is preserved only when prospectivity is projected onto "
                 "the tracking mode", color=INK, fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(outdir, "fig1_mechanism_overview.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def hostile_control(system, models, outdir):
    """Are projected PC, static bypass and a matched readout lead one system?"""
    rng = np.random.default_rng(11)
    x = rng.normal(size=400)
    t = np.arange(len(x)) * system.dt

    proj = models["C_projected_state_pc"].run(x)[:, 1]
    static = models["D_static_bypass"].run(x)[:, 1]
    lead = models["E1_readout_lead_matched"].run(x)[:, 1]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5.6), facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [2, 1]})
    win = slice(0, 90)
    ax.plot(t[win], proj[win], color=PROJ, lw=3.2, label="C. Projected state PC")
    ax.plot(t[win], static[win], color=STATIC, lw=2.0, ls=(0, (5, 3)),
            label="D. Static equilibrium bypass")
    ax.plot(t[win], lead[win], color=PLAIN, lw=1.4, ls=(0, (1.5, 2)),
            label="E1. Readout lead, matched alpha")
    _style(ax, title="Tracking mode: three mechanisms, one trajectory",
           ylabel="tracking-mode state")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3,
              loc="upper center")

    ax2.plot(t[win], np.abs(proj - static)[win], color=STATIC, lw=LW,
             label="|C - D|")
    ax2.plot(t[win], np.abs(proj - lead)[win], color=PLAIN, lw=LW, ls=(0, (3, 2)),
             label="|C - E1|")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-18, 1e-12)
    _style(ax2, xlabel="time (s)", ylabel="absolute\ndifference",
           title="Differences sit at float64 rounding - the three are one system")
    ax2.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2,
               loc="lower left")

    fig.suptitle("Hostile control: projected state PC is matched exactly by a "
                 "static bypass and by an output-side lead",
                 color=INK, fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(outdir, "fig2_hostile_control.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
