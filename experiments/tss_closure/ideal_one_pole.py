"""B. Ideal-TSS sanity check: one-pole data MUST select a one-mode closure.

If the true residual obeys tau r' = -r, the identification procedure has to
recover a one-pole model. If it does not, the method is broken and everything
downstream is meaningless. This is a gate, not a result.

Two parts:
  1. DETERMINISTIC: integrate the real TSS system s' = J_r^-1 [f_t - r/tau]
     and confirm the residual decays at exactly the prescribed rate.
  2. IDENTIFICATION: add a small stochastic drive (an OU residual) so that
     order selection is a genuine estimation problem rather than a degenerate
     fit to a noiseless exponential, then check that AR(1) wins on BIC and
     that the innovations are white.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.tss_closure._common import (AQUA, BLUE, MUTED, ORANGE,
                                             SURFACE, outdir, save, style)
from prospective.closure_models import affine_tss, nonlinear_tss, two_dim_affine_tss
from prospective.identification import (closure_defect, evaluate_held_out,
                                        select_order)

SEED, TAU, DT, N = 20260903, 0.35, 0.005, 4000


def deterministic_check(system, tau=TAU, dt=DT, n=N, s0=None):
    """Integrate the TSS law and measure the realized residual decay rate."""
    law = lambda r: -r / tau
    s0 = np.zeros(system.dim) if s0 is None else s0
    traj = system.integrate(s0, dt, n, law)
    r = traj["r"]
    mag = np.linalg.norm(r, axis=1)
    keep = mag > 1e-12
    idx = np.arange(len(mag))[keep][:1500]
    slope = np.polyfit(idx * dt, np.log(mag[keep][:1500]), 1)[0]
    return traj, dict(prescribed_rate=-1.0 / tau, measured_rate=float(slope),
                      rel_error=float(abs(slope + 1.0 / tau) * tau))


def main():
    rng = np.random.default_rng(SEED)
    d = outdir("B_ideal_one_pole")
    results = {}

    # ---- 1. deterministic: does the residual decay at exactly -1/tau? ----
    det = {}
    systems = {"affine": affine_tss(), "nonlinear_tanh": nonlinear_tss(),
               "affine_2d": two_dim_affine_tss()}
    trajs = {}
    for name, sysm in systems.items():
        s0 = np.full(sysm.dim, 0.8)          # deliberately off the manifold
        traj, stats = deterministic_check(sysm, s0=s0)
        trajs[name] = traj
        det[name] = stats
        print(f"  {name:<16} prescribed {stats['prescribed_rate']:+.4f}  "
              f"measured {stats['measured_rate']:+.4f}  "
              f"rel.err {stats['rel_error']:.2e}")
    results["deterministic_decay"] = det

    # ---- 2. identification on a stochastically driven ideal residual ----
    # tau r' = -r + noise  ->  discrete AR(1) with pole exp(-dt/tau).
    lam = -1.0 / TAU
    a1 = np.exp(lam * DT)
    n_train, n_test = 12, 8
    train = [np.cumsum(np.zeros(1))] * 0
    def make(m):
        out = []
        for _ in range(m):
            r = np.zeros(N)
            r[0] = rng.normal()
            for k in range(1, N):
                r[k] = a1 * r[k - 1] + 0.02 * rng.normal()
            out.append(r.reshape(-1, 1))
        return out
    train, test = make(n_train), make(n_test)

    best, fits = select_order([t[:, 0] for t in train], DT, orders=(1, 2, 3),
                              criterion="bic")
    rows = {}
    print(f"\n  true continuous pole: {lam:+.4f}")
    print(f"  {'order':>6}{'k':>4}{'poles':>34}{'BIC':>12}{'held-out MSE':>14}{'white':>8}")
    for k, f in fits.items():
        ho_mse, ho_max = evaluate_held_out(f, test)
        rows[k] = dict(f.summary(), held_out_rollout_mse=ho_mse,
                       held_out_rollout_max=ho_max,
                       defect=closure_defect(f, true_poles=[lam] if k == 1 else None))
        pol = ", ".join(f"{p.real:+.3f}" for p in np.sort_complex(f.continuous_poles))
        print(f"  {k:>6}{f.n_params:>4}{pol:>34}{f.bic:>12.1f}"
              f"{ho_mse:>14.3e}{str(f.whiteness_pass):>8}")
    results["identification"] = rows
    results["selected_order"] = best.order
    results["true_pole"] = lam
    print(f"\n  SELECTED ORDER = {best.order}   "
          f"{'PASS - ideal TSS => one mode' if best.order == 1 else 'FAIL'}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURFACE)
    tr = trajs["affine"]
    axes[0].plot(tr["t"], tr["s"][:, 0], color=BLUE, lw=2, label="s(t)")
    axes[0].plot(tr["t"], tr["s"][:, 0] - tr["r"][:, 0], color="#898781", lw=1.2, ls=(0, (4, 3)), label="f(s,t) target")
    style(axes[0], "Tracking on the ideal TSS law", "time", "state", legend=True)

    for name, c in (("affine", BLUE), ("nonlinear_tanh", ORANGE), ("affine_2d", AQUA)):
        m = np.linalg.norm(trajs[name]["r"], axis=1)
        axes[1].semilogy(trajs[name]["t"][:1200], np.maximum(m[:1200], 1e-16),
                         color=c, lw=2, label=name)
    axes[1].semilogy(tr["t"][:1200], np.abs(np.linalg.norm(trajs["affine"]["r"][0]))
                     * np.exp(-tr["t"][:1200] / TAU), color="#0b0b0b", lw=1,
                     ls=(0, (2, 2)), label="exp(-t/tau)")
    style(axes[1], "Residual decays at exactly -1/tau", "time", "|r|", legend=True)

    ks = sorted(fits); bics = [fits[k].bic for k in ks]
    axes[2].bar([str(k) for k in ks], bics, color=[AQUA if k == best.order else BLUE
                                                   for k in ks], width=0.55)
    style(axes[2], "BIC prefers one mode (lower is better)", "closure order", "BIC")
    fig.tight_layout()
    path = f"{d}/fig_ideal_one_pole.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    cfg = dict(seed=SEED, tau=TAU, dt=DT, n_steps=N, n_train=n_train, n_test=n_test)
    print("wrote", save("B_ideal_one_pole", cfg, results), "and", path)
    return results


if __name__ == "__main__":
    main()
