"""B2. Collective first-order closure: four model classes, exactly first-order data.

    scalar one-pole -> diagonal first-order -> collective first-order
                    -> augmented hidden-state closure

The data obey  r' = -Gamma r  with

    Gamma = [[1, c], [c, 3]]

exactly. There is NO hidden state. Yet a scalar (lambda I) or diagonal model
cannot represent it, because the two residual components are COUPLED.

The point this experiment makes:

    a zero closure defect does NOT mean scalar one-pole dynamics,
    it means the chosen OBSERVABLE SPACE IS CLOSED.

The full first-order model must recover the system without hidden states, and
the augmented model (an extra lag, i.e. hidden state per component) must be
REJECTED by model selection - it is included only as an overparameterized
control.

Calibration and evaluation use DIFFERENT initial conditions.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

from experiments.tss_closure._common import (AQUA, BLUE, ORANGE, SURFACE,
                                             YELLOW, outdir, save, style)
from prospective.identification import (STRUCTURES, compare_structures,
                                        evaluate_held_out, fit_matrix_closure)

SEED, DT, N, NOISE = 20260903, 0.01, 1200, 2e-3
C_COUPLING = 0.8


def gamma_matrix(c=C_COUPLING):
    return np.array([[1.0, c], [c, 3.0]])


def make_trajectories(Gamma, n_traj, rng, dt=DT, n=N, noise=NOISE):
    Ad = expm(-Gamma * dt)
    out = []
    for _ in range(n_traj):
        r = np.zeros((n, 2))
        r[0] = rng.normal(size=2)
        for k in range(1, n):
            r[k] = Ad @ r[k - 1] + noise * rng.normal(size=2)
        out.append(r)
    return out


def main():
    rng = np.random.default_rng(SEED)
    d = outdir("B2_collective_first_order")
    Gamma = gamma_matrix()
    true_poles = np.sort(np.real(-np.linalg.eigvals(Gamma)))
    print(f"  true Gamma =\n{Gamma}")
    print(f"  true continuous poles: {true_poles}")
    print(f"  (a scalar model has ONE pole and cannot represent two)")

    train = make_trajectories(Gamma, 16, rng)
    test = make_trajectories(Gamma, 10, rng)          # held-out initial states

    rows = {}
    print(f"\n  {'structure':>12}{'k':>4}{'fitted poles':>26}"
          f"{'held-out MSE':>15}{'BIC':>13}{'pole err':>11}")
    fits = {}
    for s in STRUCTURES:
        f = fit_matrix_closure(train, DT, structure=s, holdout_frac=0.25)
        fits[s] = f
        ho_mse, ho_max = evaluate_held_out(f, test)
        got = np.sort(np.real(f.continuous_poles))[:2]
        pole_err = float(np.max(np.abs(got - true_poles))) if len(got) >= 2 else np.nan
        rows[s] = dict(f.summary(), held_out_rollout_mse=ho_mse,
                       held_out_rollout_max=ho_max, pole_error=pole_err)
        pol = ", ".join(f"{p:+.3f}" for p in got)
        print(f"  {s:>12}{f.n_params:>4}{pol:>26}{ho_mse:>15.3e}"
              f"{f.bic:>13.1f}{pole_err:>11.3f}")

    best_bic = min(fits, key=lambda s: fits[s].bic)
    best_ho = min(rows, key=lambda s: rows[s]["held_out_rollout_mse"])

    # An MSE difference of a fraction of a percent is not evidence for extra
    # modes. Require the augmented model to beat `full` by a MATERIAL margin
    # before its lower error counts for anything.
    mse_full = rows["full"]["held_out_rollout_mse"]
    mse_aug = rows["augmented"]["held_out_rollout_mse"]
    aug_gain = (mse_full - mse_aug) / mse_full
    aug_material = aug_gain > 0.05
    print(f"\n  BIC selects                     : {best_bic}")
    print(f"  lowest held-out rollout MSE     : {best_ho}")
    print(f"  augmented vs full held-out gain : {aug_gain:+.2%} "
          f"({'material' if aug_material else 'NOT material, < 5%'})")
    print(f"  pole error, full                : {rows['full']['pole_error']:.4f}")
    print(f"  pole error, augmented           : {rows['augmented']['pole_error']:.1f}")

    if best_bic == "full" and not aug_material:
        verdict = ("PASS - collective first-order recovered WITHOUT hidden "
                   "states; the augmented model's held-out edge is immaterial "
                   "and its poles are spurious")
    elif aug_material:
        verdict = "FAIL - augmented model materially better; data are not first-order"
    else:
        verdict = f"CHECK - BIC selected {best_bic}, expected full"
    print(f"  VERDICT: {verdict}")

    results = dict(true_gamma=Gamma.tolist(), true_poles=true_poles.tolist(),
                   fits=rows, selected_by_bic=best_bic,
                   lowest_heldout=best_ho, augmented_gain=float(aug_gain),
                   augmented_material=bool(aug_material), verdict=verdict)

    # ---- figure: held-out rollout for each model class ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), facecolor=SURFACE)
    tr = test[0]
    t = np.arange(len(tr)) * DT
    colours = dict(scalar=YELLOW, diagonal=ORANGE, full=AQUA, augmented=BLUE)
    from prospective.identification import rollout
    for comp, ax in zip((0, 1), axes[:2]):
        ax.plot(t, tr[:, comp], color="#0b0b0b", lw=2.4, label="truth")
        for s in STRUCTURES:
            f = fits[s]
            pred = (rollout(f, tr[1], len(tr) - 2, r_minus1=tr[0])
                    if f.A2_discrete is not None else rollout(f, tr[0], len(tr) - 1))
            off = 2 if f.A2_discrete is not None else 1
            ax.plot(t[off:], pred[:, comp], color=colours[s], lw=1.6,
                    ls=(0, (4, 2)) if s != "full" else "-", label=s)
        style(ax, f"held-out rollout, r[{comp}]", "time", "residual",
              legend=(comp == 0))
        ax.set_xlim(0, 3)

    ho = [rows[s]["held_out_rollout_mse"] for s in STRUCTURES]
    axes[2].bar(list(STRUCTURES), ho, color=[colours[s] for s in STRUCTURES],
                width=0.6)
    axes[2].set_yscale("log")
    style(axes[2], "held-out rollout error (log)", "model class", "MSE")
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("Exactly first-order data: only the COLLECTIVE model recovers it - "
                 "no hidden state needed", color="#0b0b0b", fontsize=12,
                 x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = f"{d}/fig_collective_first_order.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    cfg = dict(seed=SEED, dt=DT, n_steps=N, noise=NOISE, coupling=C_COUPLING,
               n_train=16, n_test=10)
    print("wrote", save("B2_collective_first_order", cfg, results), "and", path)
    return results


if __name__ == "__main__":
    main()
