"""Paired statistics for the frontier experiment. Pure Python, no JAX.

Every comparison here is PAIRED: arm A and arm B are run on the same seed,
which means the same data stream and -- where the shapes allow it -- the
same initialization draw. The difference between two arms on one seed is
therefore a much quieter quantity than either arm's spread across seeds,
and the earlier three-seed run showed exactly why that matters: one
baseline seed was 180x worse than its siblings, and a mean-versus-spread
test let that single blow-up veto a real effect.

Two tests are reported for every comparison, deliberately:

  * the SIGN test, which is exact and assumes nothing but independence of
    the seeds. It is the conservative one and it is what the verdicts use.
  * the WILCOXON signed-rank test, which uses the magnitudes as well and so
    has more power, at the cost of a symmetry assumption on the paired
    differences and a normal approximation that needs n >= 10.

Non-inferiority is a separate question from improvement and is tested as
such: the margin is declared before the run, and the test asks whether the
ratio is below that margin, not whether it differs from one.
"""

import math

#: predeclared before the run, and not to be moved afterwards
MEMORY_NONINFERIORITY_MARGIN = 0.10
SECONDARY_MARGIN = 0.05
ALPHA = 0.05
#: the Wilcoxon normal approximation is not trustworthy below this
WILCOXON_MIN_SAMPLES = 10


def _binomial_tail(successes, trials):
    """P(X >= successes) for X ~ Binomial(trials, 1/2). Exact."""
    total = sum(math.comb(trials, k) for k in range(successes, trials + 1))
    return total / (2 ** trials)


def sign_test(differences, alternative="less"):
    """Exact sign test on paired differences, zeros discarded.

    `alternative="less"` asks whether the differences are below zero more
    often than chance, which is the form every question here takes once it
    is written as `treatment - baseline` or as `ratio - margin`.
    """
    nonzero = [value for value in differences if value != 0.0]
    trials = len(nonzero)
    if trials == 0:
        return {"n": 0, "wins": 0, "p_value": 1.0, "test": "sign"}
    wins = sum(1 for value in nonzero if value < 0.0)
    if alternative == "less":
        p_value = _binomial_tail(wins, trials)
    elif alternative == "greater":
        p_value = _binomial_tail(trials - wins, trials)
    else:
        smaller = min(wins, trials - wins)
        p_value = min(1.0, 2.0 * _binomial_tail(trials - smaller, trials))
    return {"n": trials, "wins": wins, "p_value": p_value, "test": "sign",
            "discarded_zeros": len(differences) - trials}


def _ranks_with_ties(values):
    """Average ranks, 1-based, ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while (stop + 1 < len(order)
               and values[order[stop + 1]] == values[order[position]]):
            stop += 1
        shared = (position + stop) / 2.0 + 1.0
        for index in range(position, stop + 1):
            ranks[order[index]] = shared
        position = stop + 1
    return ranks


def _normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def wilcoxon_signed_rank(differences, alternative="less"):
    """Signed-rank test, normal approximation with tie and continuity
    corrections. Returns p_value None below WILCOXON_MIN_SAMPLES rather
    than a number that would not mean anything."""
    nonzero = [value for value in differences if value != 0.0]
    trials = len(nonzero)
    if trials < WILCOXON_MIN_SAMPLES:
        return {"n": trials, "p_value": None, "test": "wilcoxon",
                "reason": f"fewer than {WILCOXON_MIN_SAMPLES} nonzero pairs"}
    magnitudes = [abs(value) for value in nonzero]
    ranks = _ranks_with_ties(magnitudes)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0.0)
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0.0)
    mean = trials * (trials + 1) / 4.0
    variance = trials * (trials + 1) * (2 * trials + 1) / 24.0
    counts = {}
    for magnitude in magnitudes:
        counts[magnitude] = counts.get(magnitude, 0) + 1
    tie_correction = sum(size ** 3 - size for size in counts.values()) / 48.0
    variance -= tie_correction
    if variance <= 0.0:
        return {"n": trials, "p_value": None, "test": "wilcoxon",
                "reason": "degenerate variance under ties"}
    statistic = negative if alternative == "less" else positive
    # continuity correction, in the direction that makes the test smaller
    z = (statistic - mean - 0.5) / math.sqrt(variance)
    if alternative == "two-sided":
        z = (min(negative, positive) - mean + 0.5) / math.sqrt(variance)
        p_value = min(1.0, 2.0 * _normal_cdf(z))
    else:
        p_value = 1.0 - _normal_cdf(z)
    return {"n": trials, "p_value": p_value, "test": "wilcoxon",
            "negative_rank_sum": negative, "positive_rank_sum": positive,
            "z": z}


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return float("nan")
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def ratios(treatment, baseline):
    """treatment / baseline, per pair, guarded against a zero baseline."""
    return [t / b if b > 0.0 else float("inf")
            for t, b in zip(treatment, baseline)]


def paired_improvement(treatment, baseline, alpha=ALPHA):
    """Is `treatment` lower than `baseline`, seed by seed?

    Reported on the ratio scale as well as the difference scale, because a
    factor is what the earlier runs were quoted in and a difference is what
    the tests operate on.
    """
    differences = [t - b for t, b in zip(treatment, baseline)]
    factor = ratios(baseline, treatment)
    sign = sign_test(differences, "less")
    wilcoxon = wilcoxon_signed_rank(differences, "less")
    return {
        "n_pairs": len(differences),
        "wins": sign["wins"],
        "median_factor_baseline_over_treatment": median(factor),
        "median_ratio_treatment_over_baseline": median(ratios(treatment,
                                                              baseline)),
        "sign_test": sign,
        "wilcoxon": wilcoxon,
        "significant_at_alpha": bool(sign["p_value"] < alpha),
        "alpha": alpha,
    }


def paired_noninferiority(treatment, baseline,
                          margin=MEMORY_NONINFERIORITY_MARGIN, alpha=ALPHA):
    """Is `treatment` no worse than `baseline` by more than `margin`?

    H0: the paired ratio is at or above 1 + margin.
    H1: it is below.

    Rejecting H0 is what licenses the word "non-inferior"; failing to
    reject is NOT evidence of non-inferiority and the report says so
    rather than reading a null result as a pass.
    """
    paired = ratios(treatment, baseline)
    threshold = 1.0 + margin
    shifted = [value - threshold for value in paired]
    sign = sign_test(shifted, "less")
    wilcoxon = wilcoxon_signed_rank(shifted, "less")
    below = sum(1 for value in paired if value < threshold)
    return {
        "margin": margin,
        "threshold_ratio": threshold,
        "n_pairs": len(paired),
        "pairs_within_margin": below,
        "median_ratio": median(paired),
        "sign_test": sign,
        "wilcoxon": wilcoxon,
        "non_inferior": bool(sign["p_value"] < alpha),
        "alpha": alpha,
    }
