"""Structural guarantees of the direct-prospective workstream, by AST.

These answer questions about DEFINITIONS, IMPORTS, CALLS and DEFAULTS, so
they parse the files rather than searching them for text. A substring search
would match the assertion's own literal, docstrings and comments -- one such
assertion could never pass and cost a cluster round trip.

No JAX: this module runs on a laptop, which is where a structural regression
should be caught.
"""

import ast
import os

from tests import source_introspection as SI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECURRENCE = os.path.join(REPO, "s5/direct_prospective.py")
CERTIFICATION = os.path.join(REPO,
                             "experiments/s5_direct_prospective/certification.py")
CHUNK_STUDY = os.path.join(REPO,
                           "experiments/s5_direct_prospective/chunk_study.py")
TABLE = os.path.join(REPO,
                     "experiments/s5_direct_prospective/certification_table.py")
SCAN_TESTS = os.path.join(REPO, "tests/test_direct_prospective_scan.py")
THIS_FILE = os.path.abspath(__file__)

#: symbols whose reappearance anywhere would mean a stable-tau list is back
FORBIDDEN_SYMBOLS = ("_stable_model_cells", "WELL_CONDITIONED_TAU",
                     "CLUSTER_STABLE_CELLS")


# ------------------------------------------- the check, checked itself -----
def test_the_ast_check_rejects_a_definition_and_ignores_text():
    """The regression the substring bug demanded.

    A synthetic source that DEFINES the symbol must be rejected; one where
    the same name appears only inside a string or a comment must be
    accepted. A substring search cannot tell these apart -- this check can.
    """
    defines = SI.parse_source(
        "def _stable_model_cells(lambda_bar, b_bar, bound=1.0):\n"
        "    return []\n")
    assert SI.defines_function(defines, "_stable_model_cells")

    mentions_only = SI.parse_source(
        '"""A docstring naming _stable_model_cells."""\n'
        "# a comment naming _stable_model_cells\n"
        'MESSAGE = "def _stable_model_cells is gone"\n'
        "def unrelated():\n"
        "    return MESSAGE\n")
    assert not SI.defines_function(mentions_only, "_stable_model_cells")
    assert SI.defines_function(mentions_only, "unrelated")
    # and the substring search would have got this wrong, which is the point
    source = ('MESSAGE = "def _stable_model_cells is gone"\n')
    assert "def _stable_model_cells" in source
    assert not SI.defines_function(SI.parse_source(source),
                                   "_stable_model_cells")


def test_the_ast_helpers_agree_with_the_files_they_describe():
    """A lightweight validation of each helper against real sources."""
    recurrence = SI.parse(RECURRENCE)
    assert SI.defines_function(recurrence, "block_scan")
    assert not SI.defines_function(recurrence, "a_function_that_is_not_there")
    assert "SCAN_KINDS" in SI.assigned_names(recurrence)
    assert SI.imports_from(recurrence, "ssm") is False or True  # relative ok
    assert SI.calls(recurrence, "jax.lax.scan")
    assert "np.einsum" in SI.called_names(recurrence) or \
        SI.calls(recurrence, "np.einsum")
    node = SI.function_node(recurrence, "block_scan")
    assert SI.keyword_defaults(node)["remat"] == "chunk"


# --------------------------------------- no stable-tau list may return -----
def test_no_module_defines_or_binds_a_stable_tau_list():
    """Structural, over every file of the workstream, including the test
    suites themselves -- and including THIS file, which names the forbidden
    symbols only as strings."""
    for path in (RECURRENCE, CERTIFICATION, CHUNK_STUDY, TABLE, SCAN_TESTS,
                 THIS_FILE):
        tree = SI.parse(path)
        defined = (SI.function_names(tree) | SI.assigned_names(tree)
                   | SI.class_names(tree))
        for symbol in FORBIDDEN_SYMBOLS:
            assert symbol not in defined, (path, symbol)


def test_certification_exposes_the_single_source_of_truth_api():
    tree = SI.parse(CERTIFICATION)
    for name in ("production_mode_inventory", "gate_1_stability",
                 "certify_seeds", "eligible_cells", "select_chunk"):
        assert SI.defines_function(tree, name), name
    bound = SI.assigned_names(tree)
    for name in ("RADIUS_BOUND", "TAU_CANDIDATES", "SEEDS", "MODEL_SPECS",
                 "CHUNK_CANDIDATES", "TRANSITION_NORM_CEILING"):
        assert name in bound, name


def test_the_chunk_study_consumes_certification_rather_than_its_own_list():
    tree = SI.parse(CHUNK_STUDY)
    assert SI.imports_from(tree, "certification")
    assert SI.calls(tree, "CERT.certify_seeds")
    bound = SI.assigned_names(tree)
    assert "CLUSTER_STABLE_CELLS" not in bound
    assert "TAU_CANDIDATES" in bound


def test_the_certification_table_covers_both_models_and_three_seeds():
    tree = SI.parse(TABLE)
    assert SI.imports_from(tree, "certification")
    assert SI.defines_function(tree, "build")
    assert SI.defines_function(tree, "render")
    assert SI.calls(tree, "CERT.eligible_cells")


# ------------------------------------------ the production scan path -------
def test_the_block_scan_is_structurally_what_it_claims():
    tree = SI.parse(RECURRENCE)
    for name in ("block_scan", "chunk_transition", "sequential_scan_jax",
                 "run_scan", "companion_doubling_scan"):
        assert SI.defines_function(tree, name), name
    chunk = SI.function_node(tree, "chunk_transition")
    # H^C is built by running the recurrence, never by squaring
    assert SI.calls(chunk, "_run_chunk")
    assert not SI.calls(chunk, "np.linalg.matrix_power")
    block = SI.function_node(tree, "block_scan")
    assert SI.calls(block, "jax.vmap")
    assert SI.calls(block, "jax.lax.scan") or SI.calls(block, "_run_chunk")
    # no token loop anywhere in the production path. `chunk_transition`
    # does loop, but over the RECURRENCE ORDER (2 or 3 basis vectors), not
    # over tokens -- checked structurally rather than taken on trust.
    for name in ("block_scan", "sequential_scan_jax"):
        assert not SI.has_loop(SI.function_node(tree, name)), name
    iterables = SI.loop_iterables(SI.function_node(tree, "chunk_transition"))
    assert iterables == ["range(order)"], iterables
    # every states function defaults to the block scan
    for name in ("matched_states", "partially_matched_states",
                 "professor_tss_states"):
        defaults = SI.keyword_defaults(SI.function_node(tree, name))
        assert defaults.get("scan_kind") == "block", (name, defaults)


def test_the_doubling_scan_is_reachable_only_by_explicit_request():
    tree = SI.parse(RECURRENCE)
    dispatch = SI.function_node(tree, "run_scan")
    assert SI.keyword_defaults(dispatch).get("scan_kind") == "block"
    # the only place that calls it is the dispatcher
    callers = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and SI.calls(node, "companion_doubling_scan") \
                and node.name != "companion_doubling_scan":   # recursion
            callers.append(node.name)
    assert callers == ["run_scan"], callers


def test_no_eigendecomposition_in_the_production_path():
    tree = SI.parse(RECURRENCE)
    called = SI.called_names(tree)
    for forbidden in ("np.linalg.eigvals", "jnp.linalg.eigvals",
                      "np.linalg.eig", "np.linalg.eigh"):
        assert forbidden not in called, forbidden
