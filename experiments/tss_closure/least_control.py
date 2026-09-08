"""E-A. Hidden control effort / Least-Control learning signal.

Objective is CONTROL EFFORT, not squared tracking error:

    V(z) = 0.5 int_0^T u_c(t)^2 dt

with the oracle control that holds r(t) == 0 exactly. Analytically

    V(z)        = z^2 b0^2 (1 - exp(-2 a T)) / (4 a)
    dV/dz       = z   b0^2 (1 - exp(-2 a T)) / (2 a)

The claim under test:

    zero visible tracking error can coexist with non-zero control cost and a
    non-zero Least-Control learning signal.

A reduced one-pole model has no hidden state, so from r(0)=0 it needs no
control at all: V_reduced == 0 and dV_reduced/dz == 0 identically. The full
model's gradient is non-zero. Gradients are compared three ways: autodiff,
finite differences, and the closed form.
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.tss_closure._common import (AQUA, BLUE, ORANGE, SURFACE,
                                             YELLOW, outdir, save, style)

jax.config.update("jax_enable_x64", True)

SEED, DT, T_END = 20260903, 0.001, 15.0
A_HID, B0, Z0 = 0.4, 1.0, 0.5


def V_full(z, dt=DT, T=T_END, a=A_HID, b0=B0):
    """Control effort of the exact zero-residual control, through the FULL model.

    With r held at 0 the hidden state decays freely, and the control that
    cancels its influence is u = -z*b. Both the control and the objective
    therefore depend on z, even though r(t) == 0 throughout.
    """
    n = int(T / dt)
    k = jnp.arange(n)
    b = b0 * jnp.exp(-a * k * dt)          # r == 0  =>  b' = -a b
    u = -z * b                             # exact cancellation of z*b
    sq = u ** 2
    # trapezoidal quadrature: O(dt^2), so the comparison against the closed
    # form tests the theory rather than the integration rule
    return 0.5 * dt * (jnp.sum(sq) - 0.5 * (sq[0] + sq[-1]))


def V_reduced(z, dt=DT, T=T_END):
    """Reduced one-pole model: no hidden state, so no control is ever needed."""
    return 0.0 * z


def V_analytic(z, T=T_END, a=A_HID, b0=B0):
    return z ** 2 * b0 ** 2 * (1.0 - np.exp(-2.0 * a * T)) / (4.0 * a)


def dV_analytic(z, T=T_END, a=A_HID, b0=B0):
    return z * b0 ** 2 * (1.0 - np.exp(-2.0 * a * T)) / (2.0 * a)


def main():
    d = outdir("EA_least_control")

    v_num = float(V_full(Z0))
    v_ana = V_analytic(Z0)
    print(f"  V(z={Z0})  numerical {v_num:.9f}   analytic {v_ana:.9f}   "
          f"rel {abs(v_num-v_ana)/v_ana:.2e}")

    g_auto = float(jax.grad(V_full)(Z0))
    g_ana = dV_analytic(Z0)
    eps = 1e-6
    g_fd = float((V_full(Z0 + eps) - V_full(Z0 - eps)) / (2 * eps))
    g_red = float(jax.grad(V_reduced)(Z0))

    print(f"\n  dV/dz  autodiff (full)     : {g_auto:+.9f}")
    print(f"  dV/dz  finite difference   : {g_fd:+.9f}")
    print(f"  dV/dz  analytic            : {g_ana:+.9f}")
    print(f"  dV/dz  REDUCED one-pole    : {g_red:+.9f}   <-- structurally zero")
    print(f"\n  |autodiff - analytic|      : {abs(g_auto-g_ana):.3e}")
    print(f"  |autodiff - finite diff|   : {abs(g_auto-g_fd):.3e}")
    print(f"  |reduced  - analytic|      : {abs(g_red-g_ana):.3e}   <-- the failure")

    print("\n  quadrature convergence of dV/dz towards the closed form:")
    for dtc in (0.008, 0.004, 0.002, 0.001):
        gc = float(jax.grad(lambda z: V_full(z, dt=dtc))(Z0))
        print(f"    dt={dtc:<8.4f} |autodiff - analytic| = {abs(gc-g_ana):.3e}")

    ok = (abs(g_auto - g_ana) / abs(g_ana) < 1e-5
          and abs(g_auto - g_fd) / abs(g_ana) < 1e-4
          and abs(g_red) < 1e-12 and abs(g_ana) > 1e-3)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} - r(t)=0 throughout, yet the "
          f"full model has a non-zero learning signal that the reduced model "
          f"reports as exactly zero")

    zs = np.linspace(0.05, 0.9, 40)
    v_curve = [float(V_full(z)) for z in zs]
    g_curve = [float(jax.grad(V_full)(z)) for z in zs]

    results = dict(z0=Z0, V_numerical=v_num, V_analytic=v_ana,
                   grad_autodiff=g_auto, grad_finite_difference=g_fd,
                   grad_analytic=g_ana, grad_reduced=g_red,
                   err_autodiff_vs_analytic=float(abs(g_auto - g_ana)),
                   err_autodiff_vs_fd=float(abs(g_auto - g_fd)),
                   err_reduced_vs_analytic=float(abs(g_red - g_ana)),
                   verdict_pass=bool(ok))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    axes[0].plot(zs, v_curve, color=AQUA, lw=2.4, label="V(z) full model")
    axes[0].plot(zs, [V_analytic(z) for z in zs], color="#0b0b0b", lw=1,
                 ls=(0, (3, 3)), label="analytic")
    axes[0].plot(zs, np.zeros_like(zs), color=YELLOW, lw=2, ls=(0, (2, 2)),
                 label="V(z) reduced one-pole")
    style(axes[0], "Control effort depends on z although r(t) = 0", "z",
          "V = 0.5 " + r"$\int u_c^2$", legend=True)

    axes[1].plot(zs, g_curve, color=AQUA, lw=2.4, label="dV/dz autodiff (full)")
    axes[1].plot(zs, [dV_analytic(z) for z in zs], color="#0b0b0b", lw=1,
                 ls=(0, (3, 3)), label="analytic")
    axes[1].plot(zs, np.zeros_like(zs), color=YELLOW, lw=2, ls=(0, (2, 2)),
                 label="reduced one-pole")
    style(axes[1], "Least-Control learning signal", "z", "dV/dz", legend=True)

    fig.suptitle("Zero visible error, non-zero learning signal - the reduced "
                 "closure reports exactly zero", color="#0b0b0b", fontsize=12,
                 x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = f"{d}/fig_least_control.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    cfg = dict(seed=SEED, dt=DT, T=T_END, a=A_HID, b0=B0, z0=Z0)
    print("wrote", save("EA_least_control", cfg, results), "and", path)
    return results


if __name__ == "__main__":
    main()
