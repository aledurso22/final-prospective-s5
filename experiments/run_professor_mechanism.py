"""Deterministic mechanism test for professor Ideas 1 and 2.

No SGD, no S5, no GPU. A tiny diagonal linear system with one slow memory
mode and one fast tracking mode, integrated exactly, so every theoretical
claim can be checked against a closed-form prediction.

    python experiments/run_professor_mechanism.py [--outdir DIR]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechanism import metrics
from mechanism.full_state_pc import FullStatePC
from mechanism.plain_ssm import PlainSSM
from mechanism.projected_state_pc import ProjectedStatePC
from mechanism.readout_lead_control import (ReadoutLeadPerMode,
                                            ReadoutLeadShared, matched_alpha)
from mechanism.static_equilibrium_bypass import StaticEquilibriumBypass
from mechanism.system import two_timescale_system


def build(system):
    return {
        "A_plain": PlainSSM(system),
        "B_full_state_pc": FullStatePC(system),
        "C_projected_state_pc": ProjectedStatePC(system),
        "D_static_bypass": StaticEquilibriumBypass(system),
        "E1_readout_lead_matched": ReadoutLeadPerMode(system),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/professor_mechanism")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sys_ = two_timescale_system()
    models = build(sys_)
    report = {"system": {}, "poles": {}, "memory": {}, "tracking": {},
              "equivalences": {}, "robustness": {}}

    # ---------------------------------------------------------------- system
    report["system"] = dict(
        tau=sys_.tau, dt=sys_.dt, rho=sys_.rho,
        modes=[{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in row.items()} for row in sys_.describe()],
    )
    print(f"system: tau={sys_.tau} dt={sys_.dt} rho=exp(-dt/tau)={sys_.rho:.9f}")
    for row in sys_.describe():
        print(f"  mode {row['mode']} [{row['role']:>8}] a={row['a']:.3f} "
              f"K={row['K']:.4f} plain_pole={row['plain_continuous_pole']:+.4f} "
              f"lam={row['plain_discrete_pole']:.6f} "
              f"half_life={float(row['plain_half_life']):.3f}")

    # ----------------------------------------------------------------- poles
    print("\n=== POLES: theory vs measured (from HOMOGENEOUS response) ===")
    print("    plus forced-impulse support: how many samples the impulse")
    print("    response lasts (1 = no forced memory, s = K x exactly)")
    print(f"{'model':<26}{'mode':>5}{'theory disc':>14}{'measured':>12}"
          f"{'theory cont':>14}{'half-life':>11}{'imp.supp':>10}")
    for key, m in models.items():
        hom = metrics.homogeneous_response(m, sys_, length=3000)
        imp = metrics.impulse_response(m, length=3000)
        th_d = np.atleast_1d(m.discrete_poles())
        th_c = np.atleast_1d(m.continuous_poles())
        rows = []
        for i in range(sys_.n_modes):
            meas = metrics.empirical_discrete_pole(hom[:, i], sys_.dt)
            hl = metrics.half_life_from_impulse(hom[:, i], sys_.dt)
            supp = metrics.forced_impulse_support(imp[:, i])
            rows.append(dict(mode=i, theory_discrete=float(th_d[i]),
                             measured_discrete=float(meas),
                             theory_continuous=float(th_c[i]),
                             half_life=float(hl),
                             forced_impulse_support=supp))
            ms = "  collapsed" if not np.isfinite(meas) else f"{meas:>12.6f}"
            print(f"{key:<26}{i:>5}{th_d[i]:>14.6f}{ms}"
                  f"{th_c[i]:>14.4f}{hl:>11.3f}{supp:>10d}")
        report["poles"][key] = rows

    # ---------------------------------------------------------------- memory
    print("\n=== MEMORY: impulse retention |h[lag]|/|h[0]| per mode ===")
    lags = (1, 20, 100, 400)
    print(f"{'model':<26}{'mode':>5}" + "".join(f"{('lag'+str(l)):>12}" for l in lags))
    for key, m in models.items():
        ret = metrics.impulse_retention(m, lags)
        rows = []
        for i in range(sys_.n_modes):
            vals = {l: float(ret[l][i]) for l in lags}
            rows.append(dict(mode=i, role=("tracking" if sys_.tracking_mask[i]
                                           else "memory"), retention=vals))
            print(f"{key:<26}{i:>5}" + "".join(f"{vals[l]:>12.3e}" for l in lags))
        report["memory"][key] = rows

    # -------------------------------------------------------------- tracking
    print("\n=== TRACKING: step response of the TRACKING mode (mode 1) ===")
    t_on, L = 20, 600
    for key, m in models.items():
        step = metrics.step_response(m, length=L, t_on=t_on)[:, 1]
        final = sys_.K[1]
        ttc = metrics.time_to_correct(step, final, sys_.dt, tol=0.05, t_on=t_on)
        err = metrics.tracking_error(step[t_on:], np.full(L - t_on, final))
        report["tracking"][key] = dict(final_value=float(step[-1]),
                                       target=float(final),
                                       time_to_correct=float(ttc),
                                       rmse_to_target=float(err))
        print(f"  {key:<26} final={step[-1]:8.4f} target={final:8.4f} "
              f"time_to_correct={ttc:6.3f}s  rmse={err:.5f}")

    # ---------------------------------------------------- hostile equivalences
    print("\n=== HOSTILE CONTROLS: are these mechanisms the same system? ===")
    rng = np.random.default_rng(7)
    x = rng.normal(size=1500)
    traj = {k: m.run(x) for k, m in models.items()}

    def cmp(a, b, col=None, skip=0):
        A = traj[a][skip:], traj[b][skip:]
        u, v = (A[0][:, col], A[1][:, col]) if col is not None else A
        return float(np.max(np.abs(u - v)))

    eq = {
        "C_projected_vs_D_static (tracking mode)":
            cmp("C_projected_state_pc", "D_static_bypass", col=1),
        "C_projected_vs_D_static (memory mode)":
            cmp("C_projected_state_pc", "D_static_bypass", col=0),
        "B_full_vs_D_static (tracking mode)":
            cmp("B_full_state_pc", "D_static_bypass", col=1),
        "E1_readout_lead_vs_C_projected (tracking mode, skip 1)":
            cmp("E1_readout_lead_matched", "C_projected_state_pc", col=1, skip=1),
        "E1_readout_lead_vs_D_static (tracking mode, skip 1)":
            cmp("E1_readout_lead_matched", "D_static_bypass", col=1, skip=1),
        "B_full_vs_A_plain (memory mode)":
            cmp("B_full_state_pc", "A_plain", col=0),
    }
    for k, v in eq.items():
        verdict = "IDENTICAL" if v < 1e-9 else "different"
        print(f"  max|d| = {v:.3e}  {verdict:>9}   {k}")
    report["equivalences"] = eq
    report["matched_alpha_tracking_mode"] = float(matched_alpha(sys_.lam[1]))
    report["matched_alpha_memory_mode"] = float(matched_alpha(sys_.lam[0]))
    print(f"\n  matched alpha, tracking mode (lam={sys_.lam[1]:.6f}): "
          f"{matched_alpha(sys_.lam[1]):.4f}")
    print(f"  matched alpha, memory   mode (lam={sys_.lam[0]:.6f}): "
          f"{matched_alpha(sys_.lam[0]):.4f}   <-- one shared alpha cannot be both")

    # ------------------------------------------- shared-alpha readout control
    print("\n=== E2: ONE shared alpha on the mixed readout (what S5 does) ===")
    target_state = traj["C_projected_state_pc"]
    target_y = target_state @ sys_.C.T
    plain_y = traj["A_plain"] @ sys_.C.T
    a_opt, rmse_opt = metrics.optimal_shared_alpha(plain_y, target_y)
    sig = float(np.std(target_y[5:, 0]))
    shared = {}
    for alpha in (0.0, 0.25, 1.0, 4.0, a_opt, matched_alpha(sys_.lam[1]),
                  matched_alpha(sys_.lam[0])):
        y = ReadoutLeadShared(sys_, alpha).run_readout(x)
        err = float(np.sqrt(np.mean((y[5:, 0] - target_y[5:, 0]) ** 2)))
        shared[f"alpha_{alpha:.4f}"] = err
        tag = "  <-- least-squares optimum" if np.isclose(alpha, a_opt) else ""
        print(f"  alpha={alpha:10.4f}   rmse vs projected-PC readout = "
              f"{err:10.6f}   rel={err/sig:6.1%}{tag}")
    report["readout_shared_alpha_rmse"] = shared
    report["readout_shared_alpha_optimum"] = dict(
        alpha=a_opt, rmse=rmse_opt, relative=rmse_opt / sig,
        target_std=sig)
    print(f"\n  least-squares optimal shared alpha = {a_opt:.4f}")
    print(f"  best achievable rmse = {rmse_opt:.6f} vs target std {sig:.4f} "
          f"-> {rmse_opt/sig:.1%} relative error")
    print("  Reading: one shared alpha does NOT reproduce projected state PC")
    print("  exactly (per-mode matching is exact; shared is not), but it gets")
    print("  within a few percent RMS here. The reason is that the difference")
    print("  operator (y - y_prev) is small for the slow memory mode, so a")
    print("  shared lead acts almost selectively on the fast tracking mode.")
    print("  The gap would widen if memory and tracking timescales were closer.")

    # ------------------------------------------------------------ robustness
    print("\n=== ROBUSTNESS ===")
    for key, m in models.items():
        ng = metrics.noise_gain(m, sys_)
        rt = metrics.reset_transient(m, sys_, length=200)
        settle = metrics.time_to_correct(np.abs(rt[:, 1]), 0.0, sys_.dt)
        report["robustness"][key] = dict(
            noise_gain=float(ng),
            reset_transient_peak=float(np.max(np.abs(rt[:, 1]))),
            reset_transient_final=float(np.abs(rt[-1, 1])),
        )
        print(f"  {key:<26} noise_gain={ng:8.4f}  "
              f"reset|peak|={np.max(np.abs(rt[:,1])):7.4f}  "
              f"reset|final|={np.abs(rt[-1,1]):.3e}")

    # ----------------------------------------------------------- figures
    try:
        from mechanism import figures
        f1 = figures.mechanism_overview(sys_, models, args.outdir)
        f2 = figures.hostile_control(sys_, models, args.outdir)
        report["figures"] = [f1, f2]
        print(f"\nfigures: {f1}\n         {f2}")
    except ImportError as exc:
        print(f"\n(figures skipped: {exc})")

    path = os.path.join(args.outdir, "mechanism_report.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nwrote {path}")
    return report


if __name__ == "__main__":
    main()
