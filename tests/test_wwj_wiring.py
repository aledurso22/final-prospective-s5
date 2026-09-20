"""Structural guarantees of the WWJ work that need no JAX.

Runs anywhere, including a laptop without JAX: it checks what the WWJ code
IS ALLOWED TO TOUCH, how it is wired, and that the frozen science has not
moved. The numerical behaviour of the JAX implementation is checked by
`test_wwj_recurrence.py` on the cluster; the mathematics itself is proved
exactly by `test_wwj_algebra.py`, here and everywhere.
"""

import ast
import io
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the commit whose entire existing tree must be untouched by this work
FROZEN_BASE = "ef004cda025b4e098041cfe5970c217fa0015bba"
OPERATOR = os.path.join(REPO, "s5/wwj_operator.py")
ARMS = os.path.join(REPO, "s5/wwj_ssm.py")
REJECTED = os.path.join(REPO, "s5/wwj_mixed_stencil.py")
REJECTED_ARMS = os.path.join(REPO, "s5/wwj_mixed_stencil_ssm.py")
LAUNCHER = os.path.join(REPO, "bin/run_experiments/allocation_s5_wwj_gates.sh")
BENCHMARK = os.path.join(REPO, "experiments/s5_wwj/benchmark.py")
DEV_GATE = os.path.join(REPO, "experiments/s5_wwj/dev_gate.py")
INIT_GRID = os.path.join(REPO, "experiments/s5_wwj/init_grid.py")
WWJ_MODEL = os.path.join(REPO, "experiments/s5_wwj/wwj_model.py")


def _text(path):
    return io.open(path).read()


#: the files this work is allowed to add, and nothing else
NEW_WWJ_FILES = (
    # the principal architecture
    "s5/wwj_operator.py", "s5/wwj_ssm.py",
    "tests/wwj_operator_reference.py", "tests/test_wwj_operator_algebra.py",
    "tests/test_wwj_ssm.py",
    # the REJECTED mixed-stencil realization, kept as a failed ablation
    "s5/wwj_mixed_stencil.py", "s5/wwj_mixed_stencil_ssm.py",
    "tests/wwj_sequential_reference.py",
    "tests/test_wwj_mixed_stencil_algebra.py",
    "tests/test_wwj_mixed_stencil_rejected.py",
    # experiment layer, launcher, documentation, wiring
    "experiments/s5_wwj/__init__.py", "experiments/s5_wwj/wwj_model.py",
    "experiments/s5_wwj/init_grid.py", "experiments/s5_wwj/benchmark.py",
    "experiments/s5_wwj/dev_gate.py",
    "bin/run_experiments/allocation_s5_wwj_gates.sh",
    "docs/S5_WWJ_PROSPECTIVE.md", "tests/test_wwj_wiring.py",
)


def _git(*arguments):
    return subprocess.run(["git", "-C", REPO, *arguments],
                          capture_output=True, text=True).stdout


def _tracked_at(commit):
    return set(_git("ls-tree", "-r", "--name-only", commit).split())


#: pre-existing files this work is allowed to touch, each with its reason.
#: Nothing outside `tests/` may ever appear here: the scientific tree, the
#: launchers and the documentation of earlier work stay byte-identical.
ALLOWED_PRE_EXISTING_CHANGES = {
    "tests/test_s5_three_gpu_allocation_run.py":
        "asserted the order in which CONCURRENT children record themselves, "
        "which is racy; the assertions now check the arm/seed/GPU assignment "
        "instead. No production behaviour is involved.",
}


def test_no_file_that_existed_before_this_work_has_changed():
    """Additions are allowed; touching anything that existed is not.

    The earlier version of this test compared `git diff` against the empty
    list, which was true only while the new files were still untracked: it
    could not have passed once they were committed. It now intersects the
    CHANGED paths with the files TRACKED AT THE BASE COMMIT, which catches a
    modification, a deletion, a rename or a type change of a pre-existing
    file while permitting genuinely new ones.
    """
    base_files = _tracked_at(FROZEN_BASE)
    assert base_files, "the base commit must be reachable"
    # names on both sides of a rename, and every status letter
    changed = set()
    for entry in _git("diff", "--name-status", "-M", FROZEN_BASE,
                      "--").splitlines():
        parts = entry.split("\t")
        changed.update(parts[1:])
    touched_pre_existing = changed & base_files
    # nothing outside tests/ may be touched at all
    assert {name for name in touched_pre_existing
            if not name.startswith("tests/")} == set(), \
        sorted(touched_pre_existing)
    # and inside tests/, only what is declared above, with its reason
    assert touched_pre_existing <= set(ALLOWED_PRE_EXISTING_CHANGES), \
        sorted(touched_pre_existing - set(ALLOWED_PRE_EXISTING_CHANGES))
    # belt and braces: the same question asked of git's own filters
    touched = set(_git("diff", "--name-only", "--diff-filter=MDRTC", "-M",
                       FROZEN_BASE, "--").split())
    assert touched <= set(ALLOWED_PRE_EXISTING_CHANGES), sorted(touched)
    # every file that existed then still exists, and is byte-identical unless
    # it is one of the declared exceptions
    for relative in sorted(base_files):
        path = os.path.join(REPO, relative)
        assert os.path.exists(path), f"deleted: {relative}"
        if relative in ALLOWED_PRE_EXISTING_CHANGES:
            continue
        blob = _git("rev-parse", f"{FROZEN_BASE}:{relative}").strip()
        now = _git("hash-object", path).strip()
        assert blob == now != "", relative
    # and the control is named explicitly, so the intent is readable
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py",
                     "s5/generalized_prospective_ssm.py",
                     "experiments/s5_three_arm_full/runner.py"):
        assert relative in base_files


def test_every_declared_exception_is_a_test_file_and_really_changed():
    """An allowlist that quietly accumulates entries is worthless: each one
    must name a test file that genuinely differs from the base commit."""
    for relative, reason in ALLOWED_PRE_EXISTING_CHANGES.items():
        assert relative.startswith("tests/"), relative
        assert len(reason) > 40, relative        # a reason, not a label
        assert relative in _tracked_at(FROZEN_BASE), relative
        blob = _git("rev-parse", f"{FROZEN_BASE}:{relative}").strip()
        now = _git("hash-object", os.path.join(REPO, relative)).strip()
        assert blob != now, f"stale exception, remove it: {relative}"
    # the scientific tree is never eligible for an exception
    assert not any(name.startswith(("s5/", "experiments/", "bin/", "docs/",
                                    "dataloaders/"))
                   for name in ALLOWED_PRE_EXISTING_CHANGES)


def test_the_new_wwj_files_exist_and_are_tracked():
    """The other half of the invariant: the additions are really there, and
    committed, not merely sitting in the worktree."""
    tracked = set(_git("ls-files").split())
    for relative in NEW_WWJ_FILES:
        assert os.path.exists(os.path.join(REPO, relative)), relative
        assert relative in tracked, f"untracked: {relative}"
        assert relative not in _tracked_at(FROZEN_BASE), relative
    # nothing else was added either
    added = set(_git("diff", "--name-only", "--diff-filter=A", FROZEN_BASE,
                     "--").split())
    assert added == set(NEW_WWJ_FILES), sorted(added ^ set(NEW_WWJ_FILES))


def test_the_principal_operator_is_fir_and_reuses_the_native_scan():
    """The architecture: Native scan untouched, WWJ as a three-tap on its
    trajectory. No new recurrence, so no new poles."""
    source = _text(OPERATOR)
    assert "from .ssm import binary_operator" in source
    assert "associative_scan(binary_operator" in source
    for forbidden in ("scan_companion", "companion_matrix", "doubling",
                      "eigvals"):
        assert forbidden not in source, forbidden
    assert "(1.0 + k + m) * states" in source       # the three-tap itself
    assert "- (k + 2.0 * m) * _shift(states, 1)" in source
    assert "+ m * _shift(states, 2)" in source
    # and the claim about poles is a function, not a sentence
    assert "def recurrent_poles(" in source


def test_the_rejected_realization_is_quarantined():
    """It is preserved with its evidence, labelled, and unreachable from the
    principal arms or any launcher."""
    for path in (REJECTED, REJECTED_ARMS):
        text = _text(path)
        assert "REJECTED REALIZATION" in text
        assert "1.7054537181" in text or "1.705" in text
        assert "bfe53fe" in text
    names = _text(REJECTED_ARMS)
    assert "wwj_mixed_stencil_unstable_diagnostic" in names
    assert "WWJ_REJECTED_ARMS" in names
    # the principal modules never IMPORT it; naming it in a docstring, to
    # say what was rejected and why, is exactly what they should do
    for path in (OPERATOR, ARMS):
        imports = [line for line in _text(path).splitlines()
                   if line.startswith(("import ", "from "))]
        assert not any("mixed_stencil" in line for line in imports), path
    # and the launcher refuses it by name
    launcher = _text(LAUNCHER)
    assert "*mixed_stencil*)" in launcher
    assert "REJECTED mixed-stencil realization" in launcher
    model = _text(WWJ_MODEL)
    assert "mixed_stencil" in model and "cannot be constructed here" in model


def test_the_selection_criteria_are_fir_not_companion_radii():
    grid = _text(INIT_GRID)
    assert "GAIN_CEILING = 3.0" in grid
    assert "GRADIENT_FLOOR" in grid
    assert "max_fir_gain" in grid and "probe_gradients" in grid
    assert "companion_spectral_radius" not in grid
    assert "NO_ADMISSIBLE_INITIALIZATION" in grid
    # the rule prefers the SMALLEST admissible k: start close to Native
    selection = grid[grid.index("def select("):grid.index("def main(")]
    assert "min(critical" in selection
    dev = _text(DEV_GATE)
    assert "FIR_GAIN_CEILING" in dev and "RADIUS_CEILING" not in dev


def test_native_s5_cannot_even_see_the_wwj_code():
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py",
                     "s5/generalized_prospective_ssm.py"):
        assert "wwj" not in _text(os.path.join(REPO, relative)).lower()


def test_the_old_generalized_arm_keeps_its_own_name():
    """The old arm is frozen and separately labelled; the WWJ names never
    reuse its identifier and never call it a WWJ result."""
    names = _text(ARMS)
    assert "generalized_prospective_s5" not in names
    for identifier in ("wwj_critical_s5", "wwj_passive_s5",
                       "wwj_gated_recoverable_s5_diagnostic"):
        assert identifier in names
    rejected = _text(REJECTED_ARMS)
    assert "wwj_mixed_stencil_unstable_diagnostic" in rejected
    documentation = _text(os.path.join(REPO, "docs/S5_WWJ_PROSPECTIVE.md"))
    assert "old two-compartment" in documentation.lower()
    assert "they are not WWJ results" in documentation


def test_no_eigendecomposition_in_the_principal_path():
    code = [line for line in _text(OPERATOR).splitlines()
            if not line.strip().startswith("#")]
    for forbidden in ("eigvals(", "eigh(", "linalg.eig"):
        assert not any(forbidden in line for line in code), forbidden


def test_the_parameterization_is_stable_and_declared():
    source = _text(ARMS)
    assert "TAU_MIN + jax.nn.softplus(raw)" in source
    assert "EPS_MAX * jax.nn.sigmoid(raw)" in source
    operator = _text(OPERATOR)
    assert "EPS_MAX = 0.25" in operator
    assert "TAU_MIN = 1e-3" in operator
    assert "H_TOKEN = 1.0" in operator and "learned Delta" in operator
    # one scalar per layer, and said so
    assert "per LAYER" in source or "per layer" in source.lower()
    # no post-update eigenvalue projection anywhere
    for path in (ARMS, OPERATOR, WWJ_MODEL):
        assert "clip_eigs=True" not in _text(path)


def test_the_gates_are_declared_with_numbers():
    benchmark = _text(BENCHMARK)
    for constant in ("MEMORY_CEILING_FRACTION = 0.80", "TIME_MARGIN = 0.25",
                     "THROUGHPUT_FLOOR = 0.80"):
        assert constant in benchmark, constant
    assert "2.98" in benchmark and "154.73" in benchmark   # the old number
    for metric in ("compile_seconds", "steps_per_minute", "peak_gpu_bytes",
                   "forward_seconds", "backward_seconds",
                   "throughput_ratio_to_native", "projection"):
        assert metric in benchmark, metric
    dev = _text(DEV_GATE)
    assert "REQUIRED_LOSS_REDUCTION" in dev and "FIR_GAIN_CEILING" in dev
    assert "DEV_GATE_PASS" in dev and "developmental_only" in dev


def test_the_initialization_rule_is_declared_before_the_grid_is_read():
    source = _text(INIT_GRID)
    assert "GRID_K = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)" in source
    assert "GRID_EPS = (0.0, 0.0625, 0.25)" in source
    assert "GAIN_CEILING = 3.0" in source
    assert "NO_ADMISSIBLE_INITIALIZATION" in source
    # the rule is applied by code, and validation is never consulted
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)}
    assert {"sweep", "select"} <= names
    # the rule uses companion radii and finiteness, and nothing else: no
    # accuracy, loss or dataset enters the selection
    selection = source[source.index("def select("):source.index("def main(")]
    for forbidden in ("accuracy", "loss", "validation", "test"):
        assert forbidden not in selection.lower(), forbidden


def test_no_wwj_path_loads_the_test_split():
    """Every data load in the WWJ code asks for train and val only. The test
    split is opened once, by the finalizer of a real experiment, and none of
    this is one."""
    loads = 0
    for path in (BENCHMARK, DEV_GATE, INIT_GRID, WWJ_MODEL):
        source = _text(path)
        for match in re.finditer(r"load_official_raw\(", source):
            call = source[match.start():match.start() + 160]
            assert '("train", "val")' in call, path
            assert '"test"' not in call.split(")")[0] + ")", path
            loads += 1
    assert loads >= 1                     # the dev gate really does load data


def test_the_gate_launcher_starts_no_slurm_job_and_no_training():
    text = _text(LAUNCHER)
    code = "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("#"))
    for command in ("sbatch", "salloc", "srun", "scancel"):
        assert command not in code, command
    # it runs the three gates and nothing else
    assert "experiments.s5_wwj.init_grid" in code
    assert "experiments.s5_wwj.benchmark" in code
    assert "experiments.s5_wwj.dev_gate" in code
    assert "s5_three_arm_full.runner" not in code
    assert "allocation_s5_three_arm_three_gpu" not in code
    # and it guards the same way the production launchers do
    for guard in ("EXPECTED_COMMIT", "git status --porcelain", "SLURM_JOB_ID",
                  "CUDA_VISIBLE_DEVICES"):
        assert guard in text, guard


def test_every_new_python_file_parses_and_declares_an_entry_point():
    for path in (OPERATOR, ARMS, REJECTED, REJECTED_ARMS, WWJ_MODEL,
                 BENCHMARK, DEV_GATE, INIT_GRID):
        source = _text(path)
        ast.parse(source)
        if path in (BENCHMARK, DEV_GATE, INIT_GRID):
            assert 'if __name__ == "__main__":' in source, path
            assert "def main(" in source, path


def test_both_principal_arms_are_gated_separately():
    """A critical-arm pass must not authorize passive training: each arm is
    benchmarked and dev-gated on its own, into its own directory."""
    text = _text(LAUNCHER)
    assert 'S5_WWJ_ARMS:-wwj_critical_s5 wwj_passive_s5' in text
    loop = text[text.index('for ARM in "${WWJ_ARMS[@]}"'):]
    assert "experiments.s5_wwj.benchmark" in loop
    assert "experiments.s5_wwj.dev_gate" in loop
    # separate artifact directories, per arm
    assert '--out "$RUN_ROOT/$ARM/benchmark.json"' in loop
    assert '--out "$RUN_ROOT/$ARM/dev_gate"' in loop
    assert '"$ARM-benchmark"' in loop and '"$ARM-dev_gate"' in loop
    # the initialization grid runs once, before the loop, and is shared
    assert text.index("experiments.s5_wwj.init_grid") < text.index(
        'for ARM in "${WWJ_ARMS[@]}"')
    assert "authorizes nothing about the passive arm" in text


def test_the_projection_states_what_it_covers():
    """A one-arm wave is not the wall time of the WWJ experiment."""
    benchmark = _text(BENCHMARK)
    for field in ("hours_one_arm_wave_three_concurrent_seeds",
                  "hours_both_wwj_arms_sequential", "waves_planned",
                  "hours_planned", "covers", "excludes"):
        assert field in benchmark, field
    # the gate is applied to the planned scope, not to the one-arm number
    gate = benchmark[benchmark.index("def gate("):benchmark.index("def main(")]
    assert 'projection["hours_planned"]' in gate
    assert "hours_one_arm_wave" not in gate
    # and the launcher passes the number of waves it is actually gating
    launcher = _text(LAUNCHER)
    assert '--waves "$WAVES"' in launcher
    assert 'WAVES="${S5_WWJ_WAVES:-${#WWJ_ARMS[@]}}"' in launcher


def test_the_memory_claim_is_qualified():
    """The rejected path's memory discussion is kept honest, and the
    principal path adds no scan to argue about."""
    rejected = _text(REJECTED)
    assert "FORWARD LIVE STORAGE" in rejected
    assert "BACKWARD (AUTODIFF RESIDUAL) STORAGE IS NOT O(L*P*3)" in rejected
    assert "MEASURED PEAK DEVICE MEMORY" in rejected
    documentation = _text(os.path.join(REPO, "docs/S5_WWJ_PROSPECTIVE.md"))
    assert "unknown until the GPU benchmark runs" in documentation


def test_the_throughput_floor_reflects_the_new_architecture():
    benchmark = _text(BENCHMARK)
    assert "THROUGHPUT_FLOOR = 0.80" in benchmark
    assert "O(L*P) three-tap" in benchmark
