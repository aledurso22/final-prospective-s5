"""Structural checks for the modal workstream, by AST. No JAX.

These run on a laptop, which is where a constructor-signature mistake
should be caught. One already was not: `layer_comparison` omitted
`bidirectional`, which `init_S5SSM` requires, and the error only appeared
after a four-minute GPU run.
"""

import ast
import io
import os

from tests import source_introspection as SI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE = os.path.join(REPO, "s5/modal_prospective.py")
LAYER = os.path.join(REPO, "s5/modal_prospective_ssm.py")
BENCHMARK = os.path.join(REPO, "experiments/s5_modal/benchmark.py")
SYNTHETIC = os.path.join(REPO, "experiments/s5_modal/synthetic_gates.py")
NATIVE_SSM = os.path.join(REPO, "s5/ssm.py")


def _kwargs_keys_in(function_node, variable):
    """The keys of a `variable = dict(...)` literal inside a function."""
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == variable
                for target in node.targets):
            call = node.value
            if isinstance(call, ast.Call):
                return {keyword.arg for keyword in call.keywords}
            if isinstance(call, ast.Dict):
                return {key.value for key in call.keys
                        if isinstance(key, ast.Constant)}
    return set()


def test_the_layer_benchmark_supplies_every_argument_init_S5SSM_requires():
    """REGRESSION. `init_S5SSM` takes thirteen arguments and none has a
    default, so omitting one is a TypeError at call time. The benchmark's
    kwargs must cover all of them -- checked by parsing both files, so it
    fails here rather than after a GPU run."""
    native = SI.function_node(SI.parse(NATIVE_SSM), "init_S5SSM")
    required = {argument.arg for argument in native.args.args}
    assert len(native.args.defaults) == 0, "init_S5SSM has no defaults"
    comparison = SI.function_node(SI.parse(BENCHMARK), "layer_comparison")
    supplied = _kwargs_keys_in(comparison, "kwargs")
    missing = required - supplied
    assert missing == set(), sorted(missing)
    # the bug that motivated this test, stated concretely
    assert "bidirectional" in required and "bidirectional" in supplied


def test_the_benchmark_measures_each_arm_in_its_own_process():
    tree = SI.parse(BENCHMARK)
    assert SI.calls(tree, "subprocess.run")
    main = SI.function_node(tree, "main")
    assert "--arm" in io.open(BENCHMARK).read()
    assert SI.defines_function(tree, "layer_comparison")


def test_the_cascade_builds_no_dense_companion_and_reuses_the_s5_scan():
    tree = SI.parse(CASCADE)
    assert any(name == "binary_operator"
               for _, name in SI.imported_names(tree))
    assert SI.calls(tree, "jax.lax.associative_scan")
    called = SI.called_names(tree)
    for forbidden in ("np.linalg.matrix_power", "companion_matrix",
                      "np.linalg.eigvals"):
        assert forbidden not in called, forbidden
    for name in ("stage_coefficients", "stage_pole", "apply_stage",
                 "apply_cascade", "native_mode_gain", "added_poles"):
        assert SI.defines_function(tree, name), name


def test_the_synthetic_criterion_requires_both_components():
    """A trade between lead and memory must not read as a success."""
    source = io.open(SYNTHETIC).read()
    main = SI.function_node(SI.parse(SYNTHETIC), "main")
    text = ast.get_source_segment(source, main) or source
    for token in ("lead_improved", "memory_retained", "component_margin"):
        assert token in text, token
    assert "differentiated and lead_improved and memory_retained" in text


def test_the_layer_reports_the_diagnostics_the_construction_promises():
    tree = SI.parse(LAYER)
    node = SI.function_node(tree, "diagnostics")
    source = ast.get_source_segment(io.open(LAYER).read(), node) or ""
    for key in ("gates", "stage_poles", "max_stage_pole", "gamma_numerator",
                "mass_numerator", "native_mode_gain_min",
                "modes_below_cancellation_floor", "regime_per_mode"):
        assert key in source, key
    assert SI.defines_function(tree, "penalty")
