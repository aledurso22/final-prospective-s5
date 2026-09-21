"""Read a frontier report and print it. No JAX, no heredoc.

This exists because analysing a run by pasting a multi-line python heredoc
into the cluster shell has now failed twice: the terminal indents the
pasted lines, which breaks python's indentation and, when it indents the
delimiter, leaves the shell waiting inside an unterminated heredoc. A file
in the repo is immune to both.

    python -m experiments.s5_modal.report_frontier frontier.json
"""

import argparse
import json


def _pct(value):
    return "n/a" if value is None else f"{value:.4f}"


def show_gates(arms):
    """What the gates actually did, per arm.

    The construction predicts slow modes near g = 0 and faster modes
    opening theirs, so the gate-versus-decay-rate correlation is predicted
    POSITIVE. A correlation near zero means the gates never specialized,
    which is a different failure from specializing and not helping.
    """
    for name, arm in arms.items():
        gates = arm.get("gates") or {}
        if not gates or arm.get("stages", 0) == 0:
            continue
        print(f"   {name:28s} gate mean {gates['mean_gate']:.3f}  "
              f"slow half {gates['mean_gate_slow_half']:.3f}  "
              f"fast half {gates['mean_gate_fast_half']:.3f}  "
              f"corr(gate,rate) {gates['gate_vs_decay_rate_correlation']:+.3f}")


def show_modes(arms):
    """The mode basis each arm ended up with."""
    for name, arm in arms.items():
        gates = arm.get("gates") or {}
        if "slowest_timescale" not in gates:
            continue
        print(f"   {name:28s} slowest tau {gates['slowest_timescale']:9.1f}  "
              f"its |omega| {gates['omega_of_slowest_mode']:.5f}  "
              f"min |omega| {gates['min_abs_omega']:.5f}")


def show_comparison(name, comparison):
    memory = comparison["memory_non_inferiority"]
    strict = comparison["memory_non_inferiority_strict"]
    lead = comparison["lead_improvement"]
    print(f"   {name}")
    print(f"      memory non-inferiority  "
          f"{memory['pairs_within_margin']:2d}/{memory['n_pairs']} within "
          f"{memory['threshold_ratio']:.2f}x   "
          f"median ratio {memory['median_ratio']:.3f}   "
          f"p {_pct(memory['sign_test']['p_value'])}   "
          f"-> {'NON-INFERIOR' if memory['non_inferior'] else 'NOT ESTABLISHED'}")
    print(f"      (at the 5% margin)      "
          f"{strict['pairs_within_margin']:2d}/{strict['n_pairs']} within "
          f"{strict['threshold_ratio']:.2f}x   "
          f"p {_pct(strict['sign_test']['p_value'])}   "
          f"-> {'NON-INFERIOR' if strict['non_inferior'] else 'NOT ESTABLISHED'}")
    print(f"      lead improvement        "
          f"{lead['wins']:2d}/{lead['n_pairs']} wins          "
          f"median {lead['median_factor_baseline_over_treatment']:.2f}x    "
          f"p {_pct(lead['sign_test']['p_value'])}   "
          f"wilcoxon p {_pct(lead['wilcoxon']['p_value'])}   "
          f"-> {'IMPROVED' if lead['significant_at_alpha'] else 'NOT SIGNIFICANT'}")


def show_per_seed(arms, ablation=("two_stage", "one_stage")):
    """Every seed, both components, plus the ablation's paired ratios.

    This is what separates a real paired effect from a marginal-median
    artifact. The two can disagree: a median OF RATIOS and a ratio OF
    MEDIANS are different statistics, and two outlier seeds are enough to
    move a 15-value median by two ranks while leaving the paired sign test
    untouched. The paired view is the correct one here -- the seeds are
    matched -- but it should be visible rather than asserted.
    """
    for component in ("memory", "lead"):
        for name, arm in arms.items():
            print(f"      {component:6s} {name:28s} "
                  + " ".join(f"{value:.4f}" for value in arm[component]))
        treatment, baseline = ablation
        if treatment in arms and baseline in arms:
            pairs = zip(arms[baseline][component], arms[treatment][component])
            factors = [b / t if t > 0 else float("inf") for b, t in pairs]
            wins = sum(1 for value in factors if value > 1.0)
            print(f"      {component:6s} {baseline}/{treatment} per seed  "
                  + " ".join(f"{value:.2f}" for value in factors)
                  + f"   ({wins}/{len(factors)} favour {treatment})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--per-seed", action="store_true",
                        help="print every seed's normalized errors, which is "
                             "what decides whether a failed test is a real "
                             "trade or a few heavy-tailed seeds")
    arguments = parser.parse_args()
    with open(arguments.path) as handle:
        report = json.load(handle)

    declared = report["predeclared"]
    print(f"{report['schema']}   {declared['paired_seeds']} paired seeds   "
          f"margin {declared['memory_non_inferiority_margin']:.0%}   "
          f"alpha {declared['alpha']}   "
          f"{report['steps']} steps   length {report['length']}")
    print()
    for row in report["frontier"]:
        print(f"delay {row['delay']:4d}   VERDICT {row['verdict']}")
        print("   " + "  ".join(
            f"{name}: mem {arm['median_memory']:.4f} lead {arm['median_lead']:.4f}"
            for name, arm in row["arms"].items()))
        show_gates(row["arms"])
        show_modes(row["arms"])
        for name, comparison in row["comparisons"].items():
            show_comparison(name, comparison)
        if arguments.per_seed:
            show_per_seed(row["arms"])
        print()
    print("verdicts:", report["verdict_per_delay"])


if __name__ == "__main__":
    main()
