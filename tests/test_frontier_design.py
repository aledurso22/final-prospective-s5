"""The frontier experiment's statistics and decision rule, on a laptop.

Nothing here imports JAX. The arithmetic, the paired tests and the verdict
branching are the parts that have repeatedly reached the cluster broken,
and they are the parts that do not need a GPU to check.
"""

import ast
import io
import itertools
import math
import os

from experiments.s5_modal import paired as P
from experiments.s5_modal import frontier_design as D
from tests import source_introspection as SI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTIER = os.path.join(REPO, "experiments/s5_modal/frontier.py")
DESIGN = os.path.join(REPO, "experiments/s5_modal/frontier_design.py")
PAIRED = os.path.join(REPO, "experiments/s5_modal/paired.py")


# ------------------------------------------------------------ statistics --
def test_the_sign_test_is_exact_against_the_binomial_definition():
    """Enumerated, not approximated: for twelve pairs the p-value of k
    wins is the exact upper binomial tail at 1/2."""
    for wins in range(13):
        differences = [-1.0] * wins + [1.0] * (12 - wins)
        expected = sum(math.comb(12, k)
                       for k in range(wins, 13)) / 2 ** 12
        assert abs(P.sign_test(differences)["p_value"] - expected) < 1e-15


def test_twelve_unanimous_wins_are_significant_and_seven_are_not():
    """The threshold the verdicts depend on, pinned. At alpha = 0.05 the
    sign test needs ten of twelve; nine is not enough."""
    def p(wins, n=12):
        return P.sign_test([-1.0] * wins + [1.0] * (n - wins))["p_value"]
    assert p(12) < 0.05 and p(11) < 0.05 and p(10) < 0.05
    assert p(9) > 0.05, p(9)
    # and this is why three seeds could never have settled anything: even a
    # clean sweep of three is not significant at 0.05
    assert p(3, 3) > 0.05, p(3, 3)
    assert p(10, 10) < 0.05


def test_zeros_are_discarded_rather_than_counted_as_wins():
    result = P.sign_test([0.0, 0.0, -1.0, -1.0])
    assert result["n"] == 2 and result["wins"] == 2
    assert result["discarded_zeros"] == 2


def test_wilcoxon_refuses_to_report_below_ten_pairs():
    """A normal approximation on nine pairs is a number, not evidence."""
    assert P.wilcoxon_signed_rank([-1.0] * 9)["p_value"] is None
    assert P.wilcoxon_signed_rank([-float(i) for i in range(1, 11)]
                                  )["p_value"] is not None


def test_wilcoxon_uses_magnitudes_where_the_sign_test_cannot():
    """Ten small losses and two huge wins: the signs favour the baseline,
    and the ranks must too. The point of carrying both tests is that they
    can disagree, so neither is allowed to be the only one reported."""
    differences = [0.01] * 10 + [-50.0, -60.0]
    assert P.sign_test(differences)["wins"] == 2
    assert P.sign_test(differences)["p_value"] > 0.5
    assert P.wilcoxon_signed_rank(differences)["p_value"] > 0.05


def test_ties_lower_the_wilcoxon_variance():
    tied = P.wilcoxon_signed_rank([-1.0] * 12)
    untied = P.wilcoxon_signed_rank([-float(i) for i in range(1, 13)])
    assert tied["negative_rank_sum"] == untied["negative_rank_sum"]
    # same rank sum, smaller variance under ties, so a larger |z|
    assert tied["z"] > untied["z"]


def test_non_inferiority_is_not_the_same_question_as_improvement():
    """A treatment 5% worse on every seed is NON-INFERIOR at a 10% margin
    and is NOT an improvement. Conflating the two is the error the margin
    exists to prevent."""
    treatment, baseline = [1.05] * 12, [1.0] * 12
    assert P.paired_noninferiority(treatment, baseline, 0.10)["non_inferior"]
    assert not P.paired_noninferiority(treatment, baseline,
                                       0.02)["non_inferior"]
    assert not P.paired_improvement(treatment, baseline
                                    )["significant_at_alpha"]


def test_a_null_result_is_not_read_as_non_inferiority():
    """Six pairs inside the margin and six outside rejects nothing, and the
    flag must say so rather than defaulting to a pass."""
    treatment = [1.0] * 6 + [2.0] * 6
    result = P.paired_noninferiority(treatment, [1.0] * 12, 0.10)
    assert result["pairs_within_margin"] == 6
    assert not result["non_inferior"]
    assert result["sign_test"]["p_value"] > 0.05


def test_the_margin_and_alpha_are_declared_in_the_module():
    assert P.MEMORY_NONINFERIORITY_MARGIN == 0.10
    assert P.SECONDARY_MARGIN == 0.05
    assert P.ALPHA == 0.05
    assert P.WILCOXON_MIN_SAMPLES == 10


def test_a_zero_baseline_cannot_manufacture_an_improvement():
    assert P.ratios([1.0], [0.0]) == [float("inf")]
    assert P.median([]) != P.median([])  # nan


# ----------------------------------------------------------------- arms --
def test_both_controls_have_at_least_as_many_parameters_as_two_stage():
    """The whole point of the controls. The previous run's memory result
    turned out to be capacity, so a control that is even slightly smaller
    would be worthless."""
    arms = dict((name, (gates, stages, modes))
                for name, gates, stages, modes in D.arm_table())
    two = D.effective_parameters(*(arms["two_stage"][2],
                                   arms["two_stage"][1]))
    assert two == 256
    for name in ("native_capacity_matched", "one_stage_capacity_matched"):
        gates, stages, modes = arms[name]
        assert D.effective_parameters(modes, stages if gates else 0) >= two


def test_the_two_gated_arms_share_their_mode_count():
    """The pure ablation needs the recurrence to be identical, so
    `one_stage` and `two_stage` must differ in the stage count ONLY."""
    arms = {name: (gates, stages, modes)
            for name, gates, stages, modes in D.arm_table()}
    assert arms["one_stage"][2] == arms["two_stage"][2] == D.MODES
    assert arms["one_stage"][1] == 1 and arms["two_stage"][1] == 2


def test_the_native_arms_carry_no_stage_parameters():
    for name, gates, stages, modes in D.arm_table():
        if not gates:
            assert D.effective_parameters(modes, 0) == 10 * modes


def test_every_comparison_names_arms_that_exist():
    names = {name for name, _, _, _ in D.arm_table()}
    for treatment, baseline in D.COMPARISONS:
        assert treatment in names and baseline in names
    # the three the decision rule reads must be among them
    for pair in (("two_stage", "native_capacity_matched"),
                 ("one_stage_capacity_matched", "native_capacity_matched"),
                 ("two_stage", "one_stage")):
        assert pair in D.COMPARISONS, pair


def test_an_arm_subset_is_filtered_and_a_typo_is_refused():
    """A subset is how the power check runs -- one arm at full steps across
    every delay, for a fifth of the sweep's cost. A misspelt name must not
    silently drop an arm and change which comparisons exist."""
    assert D.select_arms(None) == D.arm_table()
    subset = D.select_arms(["two_stage", "native_capacity_matched"])
    assert [name for name, _, _, _ in subset] == ["native_capacity_matched",
                                                  "two_stage"]
    try:
        D.select_arms(["natve_capacity_matched"])
    except SystemExit as error:
        assert "unknown arm" in str(error)
    else:
        raise AssertionError("a misspelt arm name was accepted")


def test_comparisons_are_dropped_when_an_arm_was_not_run():
    assert D.available_comparisons(["native_capacity_matched"]) == ()
    both = D.available_comparisons(["two_stage", "one_stage"])
    assert both == (("two_stage", "one_stage"),)
    assert D.available_comparisons(
        [name for name, _, _, _ in D.arm_table()]) == D.COMPARISONS


# -------------------------------------------------------------- verdicts --
def _row(two_memory, two_lead, one_memory, one_lead, ablation_lead=False,
         ablation_memory=False, underpowered=None):
    def comparison(non_inferior, lead, memory=False):
        return {"memory_non_inferiority": {"non_inferior": non_inferior},
                "lead_improvement": {"significant_at_alpha": lead},
                "memory_improvement": {"significant_at_alpha": memory}}
    return {"underpowered_channels": underpowered or {},
            "comparisons": {
                "two_stage_vs_native_capacity_matched":
                    comparison(two_memory, two_lead),
                "one_stage_capacity_matched_vs_native_capacity_matched":
                    comparison(one_memory, one_lead),
                "two_stage_vs_one_stage":
                    comparison(False, ablation_lead, ablation_memory)}}


def test_neither_arm_retaining_memory_is_reported_as_such():
    """The declared third outcome: if neither is memory non-inferior, the
    gates are not selective, and a large lead gain does not change that."""
    assert D.decide(_row(False, True, False, True,
                         ablation_lead=True)) == "GATES_NOT_SELECTIVE"


def test_one_stage_matching_two_stage_means_ordinary_is_sufficient():
    assert D.decide(_row(True, True, True, True)) == "ORDINARY_SUFFICIENT"


def test_the_second_stage_must_beat_the_one_it_contains():
    """GENERALIZED_SUPPORTED requires the two-stage arm to beat the
    one-stage arm at equal modes, not merely to beat Native."""
    assert D.decide(_row(True, True, True, True,
                         ablation_memory=True)) == "GENERALIZED_SUPPORTED"
    assert D.decide(_row(True, True, True, True,
                         ablation_lead=True)) == "GENERALIZED_SUPPORTED"


def test_two_stage_working_alone_is_not_called_generalized_support():
    """If the second stage does not beat the first at equal modes, the
    result is named for what it is and not promoted."""
    assert D.decide(_row(True, True, False, False)
                    ) == "TWO_STAGE_WORKS_ONE_STAGE_DOES_NOT"


def test_a_subset_run_yields_no_verdict_rather_than_a_wrong_one():
    """The power check runs one arm. It must not produce a verdict from
    comparisons that were never computed."""
    row = _row(True, True, True, True)
    del row["comparisons"]["two_stage_vs_one_stage"]
    assert D.decide(row) == "PARTIAL_ARM_SUBSET"


def test_an_underpowered_channel_blocks_every_verdict():
    assert D.decide(_row(True, True, True, True, ablation_lead=True,
                         underpowered={"memory": 0.01})) == "UNDERPOWERED"


def test_no_lead_improvement_is_its_own_outcome():
    assert D.decide(_row(True, False, True, False)) == "NO_LEAD_IMPROVEMENT"


def test_the_rule_returns_a_declared_outcome_for_every_input():
    """Exhaustive over the six booleans it reads: no combination may fall
    through to something undeclared."""
    declared = {"UNDERPOWERED", "GATES_NOT_SELECTIVE", "NO_LEAD_IMPROVEMENT",
                "GENERALIZED_SUPPORTED", "ORDINARY_SUFFICIENT",
                "TWO_STAGE_WORKS_ONE_STAGE_DOES_NOT", "INCONCLUSIVE"}
    seen = set()
    for flags in itertools.product((False, True), repeat=6):
        seen.add(D.decide(_row(*flags)))
    assert seen <= declared, seen - declared
    # and the three outcomes the ablation was declared to distinguish are
    # all reachable
    assert {"GENERALIZED_SUPPORTED", "ORDINARY_SUFFICIENT",
            "GATES_NOT_SELECTIVE"} <= seen


# ------------------------------------------------------------ structure --
def test_nothing_in_the_frontier_converts_an_array_inside_a_vmap():
    """REGRESSION in kind. `float()` of a tracer raises
    ConcretizationTypeError, which has already cost one GPU round trip in
    this workstream. `gate_report` and `normalized_errors` are vmapped, so
    neither may convert."""
    tree = SI.parse(FRONTIER)
    for name in ("gate_report", "normalized_errors", "model_apply",
                 "loss_fn", "initial_params"):
        node = SI.function_node(tree, name)
        assert node is not None, name
        offenders = [n for n in ast.walk(node)
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "float"]
        assert offenders == [], (name, len(offenders))
    # `make_batch` is exempt and stays exempt for a stated reason: its
    # float() converts the Python int `delay`, never a traced array
    batch = ast.get_source_segment(io.open(FRONTIER).read(),
                                   SI.function_node(tree, "make_batch")) or ""
    assert "float(delay)" in batch


def test_the_frontier_sweeps_the_declared_delays_and_reports_normalized_error():
    source = io.open(FRONTIER).read()
    tree = SI.parse(FRONTIER)
    assert "DELAYS = (16, 32, 64, 128, 256)" in source
    assert SI.defines_function(tree, "normalized_errors")
    # the memory target's variance changes by orders of magnitude across
    # the sweep, so the frontier must divide by it
    node = SI.function_node(tree, "normalized_errors")
    body = ast.get_source_segment(source, node) or ""
    assert "/ spread" in body


def test_the_decay_rate_is_learned_in_the_log_not_additively():
    """REGRESSION, from the power check at 7df3006. Adam moves every
    parameter by about the learning rate per step, so an ADDITIVE rate
    parameter takes a 3e-2 step that is an eightfold overshoot in timescale
    at a rate of 1/256 and a 50 percent change at 1/16. The reference arm's
    normalized memory error was 0.008, 0.004, 0.77, 1.11, 1.44 across
    delays 16 to 256 -- it degraded monotonically with the delay, which is
    the signature of a step size that does not scale.

    Nothing in the frontier may read an additive `lambda_re` parameter.
    """
    source = io.open(FRONTIER).read()
    tree = SI.parse(FRONTIER)
    assert SI.defines_function(tree, "decay_rate")
    assert 'params["log_rate"]' in source
    assert 'params["lambda_re"]' not in source, (
        "the additive rate parameter is what the power check falsified")
    initial = ast.get_source_segment(
        source, SI.function_node(tree, "initial_params")) or ""
    assert '"log_rate"' in initial and '"lambda_re"' not in initial
    # the DISTRIBUTION must be unchanged: only the geometry moves
    assert "math.log(RATE_MIN)" in initial and "math.log(RATE_MAX)" in initial


def test_the_slowest_mode_can_outlast_the_longest_delay():
    """The bug that made the previous probe unable to hold what it asked
    the model to recall: uniform decay rates whose 16-sample minimum was a
    51-token timescale, for a task needing 256."""
    source = io.open(FRONTIER).read()
    assert "RATE_MIN = 1.0 / (2.0 * max(DELAYS))" in source
    assert "math.log(RATE_MIN)" in source, "rates must be LOG-uniform"
    # and the clip must not cut off the slow end it just made reachable
    assert "1e-4" in source and 1e-4 < 1.0 / (2.0 * 256)


def test_the_design_and_statistics_modules_import_no_jax():
    """They are the laptop-testable half and must stay that way."""
    for path in (DESIGN, PAIRED):
        modules = {module for module, _ in SI.imported_names(SI.parse(path))}
        assert not any(name.startswith(("jax", "flax", "optax"))
                       for name in modules), (path, modules)


def test_no_frontier_file_reads_a_name_nothing_binds():
    for path in (FRONTIER, DESIGN, PAIRED):
        assert SI.undefined_names(path) == {}, (path,
                                                SI.undefined_names(path))
