"""ONE consolidated certification table, over seeds 301, 302 and 303.

For Professor/TSS (order two, eps = 0) and WWJ generalized TSS (order three,
eps = 1/4), for every candidate tau, over EVERY layer and EVERY mode of the
production initialization -- with both directions covered, because S5's
reverse branch shares a layer's Lambda_bar and B_bar.

Columns: model, tau, worst seed, worst layer, worst mode index, that mode's
Abar, the maximum companion radius, and PASS or REJECT against rho <= 1.

A cell is eligible only if EVERY mode of EVERY seed passes. If nothing is
eligible the table says so, in one line, and that is the result -- no
fabricated candidate, no relaxed threshold, no silent skip.

This runs no training, submits nothing, and does not run the chunk study.
"""

import argparse
import json

from experiments.s5_direct_prospective import certification as CERT


def build(tau_candidates, seeds):
    return CERT.eligible_cells(tau_candidates=tau_candidates, seeds=seeds)


def render(report):
    lines = [f"{'model':>22} {'tau':>8} {'verdict':>8} {'max radius':>14} "
             f"{'seed':>6} {'layer':>28} {'mode':>6} {'Abar':>26}"]
    for row in report["rows"]:
        worst = row["worst_over_seeds"]["worst_mode"]
        value = complex(*worst["lambda_bar"])
        lines.append(
            f"{row['model']:>22} {row['tau']:>8.1f} {row['verdict']:>8} "
            f"{row['max_radius']:>14.10f} "
            f"{row['worst_over_seeds']['seed']:>6} "
            f"{worst['layer'][-28:]:>28} {worst['mode_index']:>6} "
            f"{value.real:>12.6f}{value.imag:+.6f}j")
    lines.append("")
    lines.append(f"status: {report['status']}")
    if report["eligible"]:
        lines.append("eligible cells: " + ", ".join(
            f"{row['model']} tau={row['tau']}" for row in report["eligible"]))
    else:
        lines.append("eligible cells: NONE. No (model, tau) candidate passes "
                     "gate 1 over every mode of every seed.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--taus", type=float, nargs="*",
                        default=list(CERT.TAU_CANDIDATES))
    parser.add_argument("--seeds", type=int, nargs="*",
                        default=list(CERT.SEEDS))
    args = parser.parse_args()
    report = build(tuple(args.taus), tuple(args.seeds))
    report["schema"] = "s5-direct-prospective/certification-table-v1"
    report["radius_bound"] = CERT.RADIUS_BOUND
    report["table"] = render(report)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(report["table"])
    # an empty table is a RESULT, not an error, so the exit status is 0 and
    # the status field carries the verdict


if __name__ == "__main__":
    main()
