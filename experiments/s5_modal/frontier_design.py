"""The frontier experiment's DESIGN: arms, capacity matching and the
predeclared decision rule. No JAX, deliberately.

Everything here is arithmetic and branching, which is exactly the part that
has repeatedly reached the cluster broken -- a missing constructor argument,
a dangling name, a verdict rule that could not express the answer. Keeping
it in a JAX-free module means it is tested on a laptop in milliseconds
instead of after a GPU round trip.
"""

#: a channel the reference arm cannot learn carries no information about
#: whether the cascade helps on it, so no verdict is issued below this
POWER_FLOOR = 0.20
CHANNELS = 2
MODES = 16


#: the Lagrangian arm's four per-mode action constants: Gamma, M, tau_s, m_s
ACTION_CONSTANTS = 4


def effective_parameters(modes, stages, channels=CHANNELS, action=False):
    """Parameters that actually influence the output.

    With the gates off, d_raw, delta_raw and gate_raw are inert, so a Native
    arm at the same mode count has FEWER effective parameters. `stages=0`
    names that arm.
    """
    extra = ACTION_CONSTANTS if action else 3 * stages
    return (2 * modes + 4 * modes + 2 * modes * channels + channels
            + extra * modes)


def matched_modes(target_parameters, stages, channels=CHANNELS,
                  action=False):
    """The fewest modes that reach `target_parameters` at `stages`."""
    modes = 1
    while effective_parameters(modes, stages, channels,
                               action) < target_parameters:
        modes += 1
    return modes


#: arm kinds. "native" and "cascade" keep S5's own recurrence and add a
#: gated cascade on top; "action" REPLACES it with the Euler-Lagrange
#: recurrence of the per-mode WWJ action, where `a` enters only through
#: |1 - a|^2 and the poles are the action's own memory times.
NATIVE, CASCADE, ACTION = "native", "cascade", "action"


def arm_table(modes=MODES):
    """(name, use_gates, stages, modes, kind) for every arm.

    The capacity-matched arms are sized against the two-stage arm, and the
    action arm is deliberately NOT given extra modes: at 14 modes' worth of
    parameters per mode against the cascade's 16 it is already the smaller
    model, which is the conservative direction.
    """
    target = effective_parameters(modes, 2)
    return (("native", False, 1, modes, NATIVE),
            ("native_capacity_matched", False, 1, matched_modes(target, 0),
             NATIVE),
            ("one_stage", True, 1, modes, CASCADE),
            ("one_stage_capacity_matched", True, 1, matched_modes(target, 1),
             CASCADE),
            ("two_stage", True, 2, modes, CASCADE),
            ("lagrangian_action", True, 2, modes, ACTION))


# ------------------------------------------------------------- verdicts --
COMPARISONS = (
    # equal parameters, and the question the whole sweep exists to answer
    ("two_stage", "native_capacity_matched"),
    ("one_stage_capacity_matched", "native_capacity_matched"),
    ("two_stage", "one_stage_capacity_matched"),
    # equal modes, identical Lambda/B/readout draw: the pure ablation
    ("two_stage", "one_stage"),
    ("two_stage", "native"),
    # the action arm against the construction it was derived to replace,
    # and against the untouched recurrence
    ("lagrangian_action", "two_stage"),
    ("lagrangian_action", "native"),
    ("lagrangian_action", "native_capacity_matched"),
)


def select_arms(names, modes=MODES):
    """The arm table filtered to `names`, in table order.

    A typo here would silently run four arms instead of five and produce a
    verdict from the wrong comparison set, so an unknown name is an error
    rather than an omission.
    """
    table = arm_table(modes)
    if not names:
        return table
    known = {row[0] for row in table}
    unknown = sorted(set(names) - known)
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; known: {sorted(known)}")
    return tuple(row for row in table if row[0] in set(names))


def available_comparisons(names):
    """Only the comparisons whose BOTH arms were actually run."""
    present = set(names)
    return tuple((treatment, baseline) for treatment, baseline in COMPARISONS
                 if treatment in present and baseline in present)


#: the three comparisons `decide` reads; without all of them there is no
#: verdict to give, and the report says so rather than inventing one
REQUIRED_FOR_VERDICT = ("two_stage_vs_native_capacity_matched",
                        "one_stage_capacity_matched_vs_native_capacity_matched",
                        "two_stage_vs_one_stage")


def decide(delay_row):
    """The predeclared decision rule, applied to one delay.

    The rule takes no thresholds: every threshold was applied when the
    paired tests were computed, and is recorded in the row. It reads in
    order -- whether the reference arm learned the channels at all, then
    whether EITHER gated arm retains memory, then whether the lead
    improves, and only then whether the second stage earns its place.

    The three outcomes the ablation was declared to distinguish are
    GENERALIZED_SUPPORTED, ORDINARY_SUFFICIENT and GATES_NOT_SELECTIVE.
    The others exist so that a run which answers none of them cannot be
    forced into one that it did not.
    """
    if not all(name in delay_row["comparisons"]
               for name in REQUIRED_FOR_VERDICT):
        # a subset run: a power check or a resumed sweep, not a failure
        return "PARTIAL_ARM_SUBSET"
    if delay_row["underpowered_channels"]:
        return "UNDERPOWERED"
    two = delay_row["comparisons"]["two_stage_vs_native_capacity_matched"]
    one = delay_row["comparisons"][
        "one_stage_capacity_matched_vs_native_capacity_matched"]
    ablation = delay_row["comparisons"]["two_stage_vs_one_stage"]
    two_ok = two["memory_non_inferiority"]["non_inferior"]
    one_ok = one["memory_non_inferiority"]["non_inferior"]
    if not two_ok and not one_ok:
        # the designed behaviour is selectivity; without it there is nothing
        # to trade off, whatever the lead does
        return "GATES_NOT_SELECTIVE"
    two_lead = two["lead_improvement"]["significant_at_alpha"]
    one_lead = one["lead_improvement"]["significant_at_alpha"]
    if not two_lead and not one_lead:
        return "NO_LEAD_IMPROVEMENT"
    # the second stage has to beat the one-stage filter it contains, on one
    # component or the other, at equal modes and an identical draw
    second_stage_adds = (
        ablation["lead_improvement"]["significant_at_alpha"]
        or ablation["memory_improvement"]["significant_at_alpha"])
    two_works, one_works = two_ok and two_lead, one_ok and one_lead
    if two_works and second_stage_adds:
        return "GENERALIZED_SUPPORTED"
    if one_works and not second_stage_adds:
        return "ORDINARY_SUFFICIENT"
    if two_works:
        return "TWO_STAGE_WORKS_ONE_STAGE_DOES_NOT"
    return "INCONCLUSIVE"
