"""D. Active synchronization / minimum intervention.

With  r' = -gamma r + z b + u_c,  b' = -a b + z r  and r(0)=0, b(0)=b0, the
control that holds r(t) == 0 exactly is

    u_c(t) = -z b0 exp(-a t)

because with r == 0 the hidden state decays freely, b(t) = b0 exp(-a t), and
r' = z b(t) + u_c must vanish.

The demonstration:

    zero visible tracking error  =/=>  zero hidden mismatch or control effort

Analytic control effort:

    V = int_0^T 0.5 u_c^2 dt = z^2 b0^2 (1 - exp(-2 a T)) / (4 a)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.tss_closure._common import (AQUA, BLUE, ORANGE, SURFACE,
                                             outdir, save, style)
from prospective.closure_models import TwoModeClosure

SEED, DT, T_END = 20260903, 0.001, 15.0
GAMMA, A_HID, Z, B0 = 1.0, 0.4, 0.5, 1.0


def main():
    d = outdir("D_active_sync")
    n = int(T_END / DT)
    t = np.arange(n) * DT
    sysm = TwoModeClosure(gamma=GAMMA, a=A_HID, z=Z)

    u = sysm.oracle_control(B0, t)
    xs = sysm.simulate(r0=0.0, b0=B0, dt=DT, n_steps=n, u_c=u)
    r, b = xs[:, 0], xs[:, 1]

    V_num = float(np.trapezoid(0.5 * u ** 2, t))
    V_ana = float(sysm.oracle_control_effort(B0, T_END))
    b_ana = B0 * np.exp(-A_HID * t)

    print(f"  max |r(t)| under oracle control : {np.max(np.abs(r)):.3e}")
    print(f"  max |b(t)|                      : {np.max(np.abs(b)):.4f}  (NOT zero)")
    print(f"  max |u_c(t)|                    : {np.max(np.abs(u)):.4f}  (NOT zero)")
    print(f"  max |b_num - b_analytic|        : {np.max(np.abs(b - b_ana)):.3e}")
    print(f"\n  control effort V numerical      : {V_num:.9f}")
    print(f"  control effort V analytic       : {V_ana:.9f}")
    print(f"  relative error                  : {abs(V_num-V_ana)/V_ana:.3e}")

    # The residual is not EXACTLY zero because the oracle control is applied
    # under zero-order hold: u_c decays continuously inside each step. That is
    # a discretization artifact, so verify it vanishes with dt (first order)
    # rather than asserting an arbitrary absolute threshold.
    conv = []
    for dtc in (0.008, 0.004, 0.002, 0.001):
        nc = int(T_END / dtc)
        tc = np.arange(nc) * dtc
        xc = sysm.simulate(0.0, B0, dtc, nc, u_c=sysm.oracle_control(B0, tc))
        conv.append((dtc, float(np.max(np.abs(xc[:, 0])))))
    print("\n  ZOH convergence of the residual (should scale ~ dt):")
    for dtc, mx in conv:
        print(f"    dt={dtc:<8.4f} max|r| = {mx:.3e}   ratio to dt = {mx/dtc:.4f}")
    ratios = [mx / dtc for dtc, mx in conv]
    first_order = max(ratios) / min(ratios) < 1.3
    ok = first_order and conv[-1][1] < 1e-3
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} - r is held at zero up to a "
          f"first-order ZOH error that vanishes with dt, while b and u_c stay "
          f"non-zero")

    results = dict(max_abs_r=float(np.max(np.abs(r))),
                   max_abs_b=float(np.max(np.abs(b))),
                   max_abs_u=float(np.max(np.abs(u))),
                   b_vs_analytic_max_err=float(np.max(np.abs(b - b_ana))),
                   control_effort_numerical=V_num,
                   control_effort_analytic=V_ana,
                   control_effort_rel_error=float(abs(V_num - V_ana) / V_ana),
                   residual_held_at_zero=bool(ok),
                   zoh_convergence=[{"dt": a, "max_abs_r": b_} for a, b_ in conv],
                   zoh_first_order=bool(first_order))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), facecolor=SURFACE)
    axes[0].plot(t, r, color=AQUA, lw=2)
    axes[0].set_ylim(-1e-6, 1e-6)
    style(axes[0], "Visible residual r(t) ~ 0", "time", "r(t)")
    axes[0].text(0.96, 0.9, f"max|r| = {np.max(np.abs(r)):.1e}",
                 transform=axes[0].transAxes, ha="right", fontsize=8.5)

    axes[1].plot(t, b, color=BLUE, lw=2.4, label="b(t) numerical")
    axes[1].plot(t, b_ana, color="#0b0b0b", lw=1, ls=(0, (3, 3)),
                 label="b0 exp(-a t)")
    style(axes[1], "Hidden mismatch b(t) is NOT zero", "time", "b(t)", legend=True)

    axes[2].plot(t, u, color=ORANGE, lw=2.4, label="u_c(t) oracle")
    style(axes[2], "Control effort is NOT zero", "time", "u_c(t)", legend=True)
    axes[2].text(0.96, 0.12, f"V = {V_num:.6f}\nanalytic {V_ana:.6f}",
                 transform=axes[2].transAxes, ha="right", fontsize=8.5)

    fig.suptitle("Zero visible tracking error coexists with non-zero hidden "
                 "mismatch and non-zero control cost",
                 color="#0b0b0b", fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = f"{d}/fig_active_sync.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    cfg = dict(seed=SEED, dt=DT, T=T_END, gamma=GAMMA, a=A_HID, z=Z, b0=B0)
    print("wrote", save("D_active_sync", cfg, results), "and", path)
    return results


if __name__ == "__main__":
    main()
