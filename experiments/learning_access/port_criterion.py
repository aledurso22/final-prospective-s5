"""4. The general port-access criterion, validated on constructed cases.

    w = z + sum_a theta_a U_a b + O(||theta||^2),   m = R b

    faithful learning possible  <=>  ker R subset of intersection_a ker U_a
                                <=>  exists L_a with L_a R = U_a

and, for instantaneous measurements, p_learning = rank([U_1; ...; U_d]).

Treated as synthetic theorem validation, not a universal neural port count.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.learning_access._common import (AQUA, BLUE, ORANGE, SURFACE,
                                                 YELLOW, outdir, save, style)
from prospective.learning_access import (check_port_access, corrected_tangent,
                                         minimal_port)

SEED = 20260910


def case_three_state(delta):
    """The scalar three-state system: U = [1], R = [delta]."""
    return [np.array([[1.0]])], np.array([[delta]])


def build_cases(rng):
    cases = {}
    # the three-state system, strong and absent access
    cases["three_state_delta=1"] = case_three_state(1.0)
    cases["three_state_delta=0 (no access)"] = case_three_state(0.0)

    # b in R^3, one parameter, port sees only the first two coordinates
    U1 = np.array([[1.0, -0.5, 2.0]])
    cases["3d_port_misses_coord3"] = ([U1], np.array([[1.0, 0, 0], [0, 1.0, 0]]))
    cases["3d_full_port"] = ([U1], np.eye(3))
    # the minimal sufficient port
    Rm, _ = minimal_port([U1])
    cases["3d_minimal_port"] = ([U1], Rm)

    # two parameters, rank-2 requirement, rank-1 port -> must fail
    U_a = np.array([[1.0, 0.0, 0.0]])
    U_b = np.array([[0.0, 1.0, 0.0]])
    cases["2param_rank1_port"] = ([U_a, U_b], np.array([[1.0, 0.0, 0.0]]))
    cases["2param_rank2_port"] = ([U_a, U_b],
                                  np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    return cases


def main():
    rng = np.random.default_rng(SEED)
    d = outdir("4_port_criterion")
    cases = build_cases(rng)
    rows, all_ok = {}, True

    print(f"  {'case':<34}{'holds':>7}{'rank R':>8}{'p_learn':>9}"
          f"{'residual tangent':>18}")
    for name, (Us, R) in cases.items():
        res = check_port_access(Us, R)
        row = res.summary()

        if res.holds:
            # verify the constructed corrector kills the contamination for
            # RANDOM b and random parameter values
            worst = 0.0
            for _ in range(200):
                b = rng.normal(size=R.shape[1])
                th = rng.normal(size=len(Us)) * 0.1
                resid = corrected_tangent(Us, R, res.L, th, b)
                worst = max(worst, float(np.max(np.abs(resid))))
            row["max_residual_tangent"] = worst
            ok = worst < 1e-10
            note = f"{worst:.2e}"
        else:
            # the witness must be invisible through the port and visible to U
            b = res.witness_b
            m = R @ b
            ok = (float(np.max(np.abs(m))) < 1e-10
                  and float(np.max(np.abs(res.witness_Ub))) > 1e-6)
            row["witness_port_output"] = float(np.max(np.abs(m)))
            note = f"witness |Ub|={np.max(np.abs(res.witness_Ub)):.3f}"
        # rank formula
        stacked = np.vstack(Us)
        expected_p = int(np.linalg.matrix_rank(stacked))
        ok = ok and res.p_learning == expected_p
        row["rank_formula_ok"] = bool(res.p_learning == expected_p)
        rows[name] = row
        all_ok = all_ok and ok
        print(f"  {name:<34}{str(res.holds):>7}{res.rank_R:>8}"
              f"{res.p_learning:>9}{note:>18}")

    # explicit report of the failing directions
    print("\n  explicit null-space witnesses where the criterion FAILS:")
    for name, (Us, R) in cases.items():
        res = check_port_access(Us, R)
        if not res.holds:
            print(f"    {name}")
            print(f"      b in ker R      = {np.round(res.witness_b, 4)}")
            print(f"      R b             = {np.round((R @ res.witness_b), 12)}")
            print(f"      U_{res.witness_index} b          = "
                  f"{np.round(res.witness_Ub, 4)}  <- learning needs this")

    print(f"\n  VERDICT: {'PASS' if all_ok else 'FAIL'} - criterion, correctors, "
          f"witnesses and the rank formula all behave as predicted")

    # ---- figure: access strength vs recoverable tangent -----------------
    deltas = np.logspace(-3, 1, 40)
    holds = [1.0 if check_port_access(*case_three_state(dl)).holds else 0.0
             for dl in deltas]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    axes[0].semilogx(deltas, holds, color=AQUA, lw=2.6)
    axes[0].set_ylim(-0.1, 1.2)
    style(axes[0], "Criterion holds for any delta != 0 (a knife edge)",
          r"$\delta$", "holds (1 = yes)")
    axes[0].text(0.03, 0.45, "delta = 0 exactly:\nker R is everything,\n"
                             "no corrector exists", transform=axes[0].transAxes,
                 fontsize=8.5, color="#d03b3b")

    names = list(rows)
    vals = [1.0 if rows[n]["holds"] else 0.0 for n in names]
    axes[1].barh(range(len(names)), vals,
                 color=[AQUA if v else ORANGE for v in vals], height=0.6)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    axes[1].set_xlim(0, 1.35)
    for i, n in enumerate(names):
        axes[1].text(vals[i] + 0.03, i, f"p={rows[n]['p_learning']}",
                     va="center", fontsize=7.5, color="#0b0b0b")
    style(axes[1], "Constructed cases (green = faithful learning possible)",
          "criterion holds", None)

    fig.suptitle("The port-access criterion predicts exactly which channel is "
                 "needed", color="#0b0b0b", fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = f"{d}/fig_port_criterion.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    print("wrote", save("4_port_criterion", dict(seed=SEED),
                        dict(cases=rows, verdict_pass=bool(all_ok))), "and", path)
    return rows


if __name__ == "__main__":
    main()
