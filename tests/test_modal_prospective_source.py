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


def test_the_synthetic_criterion_requires_both_components_and_controls():
    """A trade must not read as a success, an improvement inside seed noise
    must not either, and neither must extra capacity."""
    source = io.open(SYNTHETIC).read()
    tree = SI.parse(SYNTHETIC)
    main = SI.function_node(tree, "main")
    text = ast.get_source_segment(source, main) or source
    # both components, both controls, and the seed spread
    for token in ("long_delay", "lead", "native_capacity_matched",
                  "improved_beyond_seed_spread", "component_margin",
                  "half_range"):
        assert token in text, token
    assert "differentiated and survives" in text
    # the controls exist as code, not as prose
    for name in ("effective_parameters", "matched_modes", "run_arm",
                 "summarize", "variance_explained"):
        assert SI.defines_function(tree, name), name
    # the power floor survives; the ceiling that rejected a real result does
    # not
    assert "POWER_FLOOR" in source
    assert "POWER_CEILING" not in source


def test_the_layer_reports_the_diagnostics_the_construction_promises():
    tree = SI.parse(LAYER)
    node = SI.function_node(tree, "diagnostics")
    source = ast.get_source_segment(io.open(LAYER).read(), node) or ""
    for key in ("gates", "stage_poles", "max_stage_pole", "gamma_numerator",
                "mass_numerator", "native_mode_gain_min",
                "modes_below_cancellation_floor", "regime_per_mode"):
        assert key in source, key
    assert SI.defines_function(tree, "penalty")


def test_setup_never_forces_a_jnp_value_to_a_python_float():
    """REGRESSION. `setup()` runs inside the trace whenever `apply` is
    jitted, so `float(jnp.something)` raises ConcretizationTypeError there
    while working fine under an unjitted `init`. Initialization constants
    must therefore be computed with `math`, not with `jnp`.

    Checked by parsing, so it fails on a laptop rather than after a GPU run.
    """
    tree = SI.parse(LAYER)
    setup = SI.function_node(tree, "setup")
    offenders = []
    for node in ast.walk(setup):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "float"):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Call):
                dotted = SI._dotted(argument.func) or ""
                if dotted.startswith(("np.", "jnp.", "jax.")):
                    offenders.append(dotted)
    assert offenders == [], offenders
    # and the constants really are computed with math
    source = ast.get_source_segment(io.open(LAYER).read(), setup) or ""
    assert "math.log(math.expm1(" in source
    assert ("math", "math") in SI.imported_names(tree)


def test_only_reporting_methods_convert_arrays_to_python_floats():
    """`diagnostics()` may call float() -- it builds a JSON report and is
    never jitted -- but the forward path may not."""
    tree = SI.parse(LAYER)
    for name in ("setup", "__call__", "_one_direction", "penalty"):
        node = SI.function_node(tree, name)
        if node is None:
            continue
        calls = [n for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "float"]
        assert calls == [], (name, len(calls))


#: every file this workstream owns, checked for dangling references
WORKSTREAM_FILES = (CASCADE, LAYER, BENCHMARK, SYNTHETIC,
                    os.path.join(REPO, "experiments/s5_modal/production_step.py"),
                    os.path.join(REPO, "experiments/s5_modal/frontier.py"),
                    os.path.join(REPO, "experiments/s5_modal/frontier_design.py"),
                    os.path.join(REPO, "experiments/s5_modal/paired.py"),
                    os.path.join(REPO,
                                 "experiments/s5_modal/report_frontier.py"),
                    os.path.join(REPO,
                                 "experiments/s5_modal/why_generalized.py"),
                    os.path.join(REPO, "s5/prospective_euler.py"),
                    os.path.join(REPO,
                                 "experiments/s5_modal/wwj_lagrangian.py"))


def test_no_file_reads_a_name_nothing_binds():
    """The guard for the commonest edit accident: replacing the code that
    produced a value and leaving a reader behind.

    Three of those reached the cluster in three commits -- `bidirectional`
    missing from a constructor call, `POWER_CEILING` after its definition
    was removed, and `gated_errors` after the statement that built it was
    replaced. Each cost a GPU round trip. This finds them in milliseconds.
    """
    for path in WORKSTREAM_FILES:
        assert SI.undefined_names(path) == {}, (path,
                                               SI.undefined_names(path))


def test_the_guard_actually_detects_a_dangling_reference():
    """The check is worth nothing if it cannot fail."""
    import tempfile

    broken = ("def main():\n"
              "    values = compute()\n"
              "    return summary[-1]\n"
              "def compute():\n"
              "    return [1]\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(broken)
        path = handle.name
    try:
        assert SI.undefined_names(path) == {"main": ["summary"]}
    finally:
        os.unlink(path)
    # and a closure over an enclosing function's variable is NOT a defect
    fine = ("def outer(scale):\n"
            "    def inner(value):\n"
            "        return value * scale\n"
            "    return inner\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(fine)
        path = handle.name
    try:
        assert SI.undefined_names(path) == {}
    finally:
        os.unlink(path)
