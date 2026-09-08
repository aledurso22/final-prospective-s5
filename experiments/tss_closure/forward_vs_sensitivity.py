"""E-B. Forward response vs parameter sensitivity - a reduced model can match
the nominal response and still fail its parameter tangent.

    H_theta(p) = 1/(p+1) + theta/(p+3)

At theta = 0 the forward model has only the first mode,

    H_0(p) = 1/(p+1),            impulse response  h_0(t) = exp(-t)

but

    d/dtheta H|_0 = 1/(p+3),     tangent           exp(-3 t)

So the second mode is INVISIBLE in the nominal response and ESSENTIAL for the
parameter sensitivity. Forward-fit quality and sensitivity quality are
measured independently, and closures are calibrated on one input and evaluated
on a HELD-OUT input (impulse -> step).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.tss_closure._common import (AQUA, BLUE, ORANGE, SURFACE,
                                             YELLOW, outdir, save, style)
from prospective.identification import fit_closure, rollout

SEED, DT, T_END, EPS = 20260903, 0.01, 8.0, 0.05
POLE_A, POLE_B = 1.0, 3.0


def true_impulse(theta, t):
    return np.exp(-POLE_A * t) + theta * np.exp(-POLE_B * t)


def true_impulse_tangent(t):
    """d/dtheta of the impulse response at any theta: exp(-3t), exactly."""
    return np.exp(-POLE_B * t)


def step_from_impulse(h, dt):
    """Held-out input: the step response is the running integral of h."""
    return np.cumsum(h) * dt


def fit_and_rollout(h, dt, order, n_out):
    f = fit_closure(h, dt, order=order, holdout_frac=0.2)
    pred = rollout(f, h[:order], n_out - order)   # chronological
    return f, np.concatenate([h[:order], pred])


def main():
    d = outdir("EB_forward_vs_sensitivity")
    n = int(T_END / DT)
    t = np.arange(n) * DT

    h0 = true_impulse(0.0, t)
    hp = true_impulse(+EPS, t)
    hm = true_impulse(-EPS, t)
    tangent_true = true_impulse_tangent(t)

    print(f"  H_theta(p) = 1/(p+{POLE_A:.0f}) + theta/(p+{POLE_B:.0f})")
    print(f"  nominal theta=0 -> single mode at -{POLE_A:.0f}")
    print(f"  tangent          -> single mode at -{POLE_B:.0f} (invisible nominally)\n")

    rows = {}
    curves = {}
    for order, label in ((1, "reduced one-mode"), (2, "two-mode closure")):
        f0, r0 = fit_and_rollout(h0, DT, order, n)
        fp, rp = fit_and_rollout(hp, DT, order, n)
        fm, rm = fit_and_rollout(hm, DT, order, n)

        fwd_err = float(np.sqrt(np.mean((r0 - h0) ** 2)))
        tangent_model = (rp - rm) / (2 * EPS)
        sens_err = float(np.sqrt(np.mean((tangent_model - tangent_true) ** 2)))

        # held-out input: step response and its tangent
        step_true = step_from_impulse(h0, DT)
        step_model = step_from_impulse(r0, DT)
        step_fwd_err = float(np.sqrt(np.mean((step_model - step_true) ** 2)))
        step_tan_true = step_from_impulse(tangent_true, DT)
        step_tan_model = step_from_impulse(tangent_model, DT)
        step_sens_err = float(np.sqrt(np.mean((step_tan_model - step_tan_true) ** 2)))

        poles = np.sort(np.real(f0.continuous_poles))
        rows[label] = dict(
            order=order, n_params=f0.n_params,
            nominal_poles=poles.tolist(),
            forward_rmse_impulse=fwd_err, sensitivity_rmse_impulse=sens_err,
            forward_rmse_step_heldout=step_fwd_err,
            sensitivity_rmse_step_heldout=step_sens_err,
            sensitivity_to_forward_ratio=float(sens_err / max(fwd_err, 1e-15)))
        curves[label] = dict(forward=r0, tangent=tangent_model)

        print(f"  {label:<20} k={f0.n_params}  nominal poles "
              f"{np.round(poles, 3)}")
        print(f"    forward RMSE  (impulse, calibrated) : {fwd_err:.3e}")
        print(f"    forward RMSE  (step, HELD OUT)      : {step_fwd_err:.3e}")
        print(f"    SENSITIVITY RMSE (impulse)          : {sens_err:.3e}")
        print(f"    SENSITIVITY RMSE (step, HELD OUT)   : {step_sens_err:.3e}\n")

    one, two = rows["reduced one-mode"], rows["two-mode closure"]
    fwd_ratio = one["forward_rmse_impulse"] / max(two["forward_rmse_impulse"], 1e-18)
    sens_ratio = one["sensitivity_rmse_impulse"] / max(
        two["sensitivity_rmse_impulse"], 1e-18)
    print(f"  one-mode / two-mode  FORWARD error ratio     : {fwd_ratio:8.2f}x")
    print(f"  one-mode / two-mode  SENSITIVITY error ratio : {sens_ratio:8.2f}x")

    ok = sens_ratio > 10 * max(fwd_ratio, 1.0)
    print(f"\n  VERDICT: {'PASS' if ok else 'CHECK'} - the reduced model matches "
          f"the nominal response far better than it matches its parameter tangent")

    results = dict(models=rows, forward_error_ratio=float(fwd_ratio),
                   sensitivity_error_ratio=float(sens_ratio),
                   verdict_pass=bool(ok), eps=EPS)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), facecolor=SURFACE)
    axes[0].plot(t, h0, color="#0b0b0b", lw=2.4, label="true h(t), theta=0")
    axes[0].plot(t, curves["reduced one-mode"]["forward"], color=YELLOW, lw=1.8,
                 ls=(0, (4, 2)), label="one-mode fit")
    axes[0].plot(t, curves["two-mode closure"]["forward"], color=AQUA, lw=1.4,
                 ls=(0, (1.5, 2)), label="two-mode fit")
    style(axes[0], "Nominal forward response: both match", "time", "h(t)",
          legend=True)

    axes[1].plot(t, tangent_true, color="#0b0b0b", lw=2.4,
                 label=r"true $\partial_\theta h = e^{-3t}$")
    axes[1].plot(t, curves["reduced one-mode"]["tangent"], color=YELLOW, lw=2,
                 ls=(0, (4, 2)), label="one-mode tangent")
    axes[1].plot(t, curves["two-mode closure"]["tangent"], color=AQUA, lw=1.6,
                 ls=(0, (1.5, 2)), label="two-mode tangent")
    style(axes[1], "Parameter tangent: the reduced model fails", "time",
          r"$\partial_\theta h$", legend=True)

    labels = ["forward\n(impulse)", "forward\n(step, held out)",
              "sensitivity\n(impulse)", "sensitivity\n(step, held out)"]
    one_v = [one["forward_rmse_impulse"], one["forward_rmse_step_heldout"],
             one["sensitivity_rmse_impulse"], one["sensitivity_rmse_step_heldout"]]
    two_v = [two["forward_rmse_impulse"], two["forward_rmse_step_heldout"],
             two["sensitivity_rmse_impulse"], two["sensitivity_rmse_step_heldout"]]
    x = np.arange(4)
    axes[2].bar(x - 0.19, one_v, width=0.36, color=YELLOW, label="one-mode")
    axes[2].bar(x + 0.19, two_v, width=0.36, color=AQUA, label="two-mode")
    axes[2].set_yscale("log"); axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=7)
    style(axes[2], "Forward vs sensitivity error (log)", None, "RMSE", legend=True)

    fig.suptitle("A hidden mode can be irrelevant for nominal inference and "
                 "essential for parameter sensitivity", color="#0b0b0b",
                 fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = f"{d}/fig_forward_vs_sensitivity.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    cfg = dict(seed=SEED, dt=DT, T=T_END, eps=EPS, pole_a=POLE_A, pole_b=POLE_B)
    print("wrote", save("EB_forward_vs_sensitivity", cfg, results), "and", path)
    return results


if __name__ == "__main__":
    main()
