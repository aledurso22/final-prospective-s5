"""C. The fundamental multimode TSS test - a genuinely hidden second mode.

    r' = -gamma r + z b + u_c
    b' = -a b     + z r

The extra mode is part of the residual system itself, NOT an adaptive-current
approximation. Stability: a, gamma > 0 and a*gamma > z^2.

The decisive initialization is

    r(0) = 0,   b(0) = b0 != 0     =>     r'(0) = z b0 != 0

so a method that sees only r(0) = 0 believes the visible residual is already
synchronized, while the hidden state immediately drives it away.

Three competing closures:
  C1  residual-only one-pole, matched to gamma      (no access to b)
  C2  best-fit first-order model on the visible r   (no hidden temporal state)
  C3  full two-mode closure
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.tss_closure._common import (AQUA, BLUE, ORANGE, SURFACE,
                                             YELLOW, outdir, save, style)
from prospective.closure_models import TwoModeClosure, affine_tss
from prospective.identification import fit_closure

SEED, DT, T_END = 20260903, 0.002, 12.0
GAMMA, A_HID, Z, B0 = 1.0, 0.4, 0.5, 1.0


def main():
    rng = np.random.default_rng(SEED)
    d = outdir("C_two_mode_hidden")
    n = int(T_END / DT)
    t = np.arange(n) * DT

    sysm = TwoModeClosure(gamma=GAMMA, a=A_HID, z=Z)
    true_poles = np.sort(np.real(sysm.eigenvalues()))
    print(f"  generator eigenvalues: {true_poles}")
    print(f"  stability a*gamma={A_HID*GAMMA} > z^2={Z**2}: "
          f"{A_HID*GAMMA > Z**2}")

    # ---- the decisive initialization -----------------------------------
    xs = sysm.simulate(r0=0.0, b0=B0, dt=DT, n_steps=n)
    r_true, b_true = xs[:, 0], xs[:, 1]
    rdot0_pred = sysm.rdot_at_zero_residual(B0)
    rdot0_num = (r_true[1] - r_true[0]) / DT
    print(f"\n  r(0) = {r_true[0]:.3e}   b(0) = {b_true[0]:.3f}")
    print(f"  r'(0) predicted z*b0 = {rdot0_pred:.6f}")
    print(f"  r'(0) numerical      = {rdot0_num:.6f}   "
          f"rel.err {abs(rdot0_num-rdot0_pred)/abs(rdot0_pred):.2e}")
    print(f"  peak |r| reached     = {np.max(np.abs(r_true)):.4f} "
          f"(from r(0)=0 - the hidden state drove it there)")

    # ---- C1: matched scalar one-pole, no access to b --------------------
    # r' = -gamma r with r(0)=0  =>  r(t) == 0 forever. It predicts nothing.
    r_c1 = np.zeros(n)

    # ---- C2: best first-order fit to the VISIBLE residual ---------------
    fit = fit_closure(r_true, DT, order=1, holdout_frac=0.3)
    lam_c2 = float(np.real(fit.continuous_poles[0]))
    # best visible-only model, still started from the visible r(0)=0
    r_c2 = np.zeros(n)
    for k in range(1, n):
        r_c2[k] = np.exp(lam_c2 * DT) * r_c2[k - 1]
    print(f"\n  C2 best visible-only pole: {lam_c2:+.4f} "
          f"(true modes {true_poles[0]:+.3f}, {true_poles[1]:+.3f})")

    # ---- C3: full two-mode closure --------------------------------------
    r_c3 = r_true.copy()

    err = {"C1_matched_one_pole": float(np.sqrt(np.mean((r_c1 - r_true) ** 2))),
           "C2_best_visible_first_order": float(np.sqrt(np.mean((r_c2 - r_true) ** 2))),
           "C3_two_mode_closure": float(np.sqrt(np.mean((r_c3 - r_true) ** 2)))}
    print(f"\n  RMS residual-prediction error")
    for k, v in err.items():
        print(f"    {k:<32}{v:.6e}")
    print("\n  C1 and C2 both predict r(t) == 0 because they start from the")
    print("  visible r(0)=0 and have no hidden state to drive it.")

    # ---- embed in a real TSS system so s(t) is meaningful ---------------
    tss = affine_tss()
    hidden = {"b": B0}

    def law(r):
        # r' = -gamma r + z b, with b integrated alongside
        rb = float(np.atleast_1d(r)[0])
        out = -GAMMA * rb + Z * hidden["b"]
        hidden["b"] += DT * (-A_HID * hidden["b"] + Z * rb)
        return np.array([out])

    traj = tss.integrate(np.zeros(1), DT, n, law)

    results = dict(true_poles=true_poles.tolist(),
                   rdot0_predicted=float(rdot0_pred),
                   rdot0_numerical=float(rdot0_num),
                   peak_abs_r=float(np.max(np.abs(r_true))),
                   c2_fitted_pole=lam_c2,
                   rms_error=err,
                   stability_ok=bool(A_HID * GAMMA > Z ** 2))

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.6), facecolor=SURFACE)
    ax = axes[0, 0]
    ax.plot(t, r_true, color=AQUA, lw=2.4, label="C3 true r(t), two-mode")
    ax.plot(t, r_c2, color=ORANGE, lw=1.8, ls=(0, (5, 3)),
            label=f"C2 best visible 1st-order")
    ax.plot(t, r_c1, color=YELLOW, lw=1.8, ls=(0, (2, 2)),
            label="C1 matched one-pole")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.plot([0], [0], marker="o", ms=7, color="#0b0b0b", zorder=5)
    ax.annotate("r(0) = 0\nlooks synchronized", xy=(0, 0), xytext=(1.6, -0.16),
                fontsize=8.5, color="#0b0b0b",
                arrowprops=dict(arrowstyle="->", color="#0b0b0b", lw=1))
    style(ax, "Visible residual: zero at t=0, then driven away by the hidden mode",
          None, "r(t)", legend=True)

    ax = axes[0, 1]
    ax.plot(t, b_true, color=BLUE, lw=2.4, label="hidden b(t)")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    style(ax, "Hidden closure state (never observed by C1/C2)", None, "b(t)",
          legend=True)

    ax = axes[1, 0]
    ax.plot(traj["t"], traj["s"][:, 0], color=AQUA, lw=2, label="s(t)")
    ax.plot(traj["t"], traj["s"][:, 0] - traj["r"][:, 0], color="#898781",
            lw=1.2, ls=(0, (4, 3)), label="target f(s,t)")
    style(ax, "Tracking state vs moving target", "time", "state", legend=True)

    ax = axes[1, 1]
    ax.semilogy(t, np.maximum(np.abs(r_true - r_c1), 1e-18), color=YELLOW,
                lw=2, label="C1 prediction error")
    ax.semilogy(t, np.maximum(np.abs(r_true - r_c2), 1e-18), color=ORANGE,
                lw=2, ls=(0, (4, 2)), label="C2 prediction error")
    style(ax, "Residual-prediction error of the visible-only closures", "time",
          "|error|", legend=True)

    fig.suptitle("A hidden closure mode makes r(0)=0 a false synchronization",
                 color="#0b0b0b", fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = f"{d}/fig_two_mode_hidden.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    cfg = dict(seed=SEED, dt=DT, T=T_END, gamma=GAMMA, a=A_HID, z=Z, b0=B0)
    print("\nwrote", save("C_two_mode_hidden", cfg, results), "and", path)
    return results


if __name__ == "__main__":
    main()
