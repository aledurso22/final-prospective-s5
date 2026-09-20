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
RECURRENCE = os.path.join(REPO, "s5/wwj_recurrence.py")
ARMS = os.path.join(REPO, "s5/wwj_prospective_ssm.py")
LAUNCHER = os.path.join(REPO, "bin/run_experiments/allocation_s5_wwj_gates.sh")
BENCHMARK = os.path.join(REPO, "experiments/s5_wwj/benchmark.py")
DEV_GATE = os.path.join(REPO, "experiments/s5_wwj/dev_gate.py")
INIT_GRID = os.path.join(REPO, "experiments/s5_wwj/init_grid.py")
WWJ_MODEL = os.path.join(REPO, "experiments/s5_wwj/wwj_model.py")


def _text(path):
    return io.open(path).read()


#: the files this work is allowed to add, and nothing else
NEW_WWJ_FILES = (
    "s5/wwj_recurrence.py", "s5/wwj_prospective_ssm.py",
    "experiments/s5_wwj/__init__.py", "experiments/s5_wwj/wwj_model.py",
    "experiments/s5_wwj/init_grid.py", "experiments/s5_wwj/benchmark.py",
    "experiments/s5_wwj/dev_gate.py",
    "bin/run_experiments/allocation_s5_wwj_gates.sh",
    "docs/S5_WWJ_PROSPECTIVE.md", "tests/wwj_sequential_reference.py",
    "tests/test_wwj_algebra.py", "tests/test_wwj_wiring.py",
    "tests/test_wwj_recurrence.py",
)


def _git(*arguments):
    return subprocess.run(["git", "-C", REPO, *arguments],
                          capture_output=True, text=True).stdout


def _tracked_at(commit):
    return set(_git("ls-tree", "-r", "--name-only", commit).split())


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
    assert changed & base_files == set(), sorted(changed & base_files)
    # belt and braces: the same question asked of git's own filters
    touched = _git("diff", "--name-only", "--diff-filter=MDRTC", "-M",
                   FROZEN_BASE, "--").split()
    assert touched == [], touched
    # every file that existed then still exists and is byte-identical now
    for relative in sorted(base_files):
        path = os.path.join(REPO, relative)
        assert os.path.exists(path), f"deleted: {relative}"
        blob = _git("rev-parse", f"{FROZEN_BASE}:{relative}").strip()
        now = _git("hash-object", path).strip()
        assert blob == now != "", relative
    # and the control is named explicitly, so the intent is readable
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py",
                     "s5/generalized_prospective_ssm.py",
                     "experiments/s5_three_arm_full/runner.py"):
        assert relative in base_files


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
                       "wwj_unconstrained_s5_diagnostic"):
        assert identifier in names
    documentation = _text(os.path.join(REPO, "docs/S5_WWJ_PROSPECTIVE.md"))
    assert "old two-compartment" in documentation.lower()
    assert "they are not WWJ results" in documentation


def test_the_production_path_never_uses_a_sequential_scan():
    """The failure being fixed was operational: a rematerialized sequential
    rollout. Neither it nor any token loop may appear in the new path."""
    source = _text(RECURRENCE)
    for forbidden in ("scan_companion", "scan_companion_sequential",
                      "lax.scan", "for token", "for t in range"):
        assert forbidden not in source, forbidden
    assert "jax.checkpoint" in source            # levels, not the rollout
    tree = ast.parse(source)
    scan = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "wwj_scan")
    # exactly one loop, over the O(log L) doubling levels
    loops = [node for node in ast.walk(scan)
             if isinstance(node, (ast.For, ast.While))]
    assert len(loops) == 1 and isinstance(loops[0], ast.While)
    assert "distance *= 2" in source
    # the oracle is a reference for tests and for the pre-training gate; the
    # MODEL never sees it, and neither does any launcher
    # source files only: a __pycache__ entry is a build artifact, not an
    # import path, and matching one made this assertion depend on whether
    # anything had been byte-compiled
    reachable = subprocess.run(
        ["grep", "-rl", "--include=*.py", "wwj_sequential_reference", "s5",
         "bin"], capture_output=True, text=True, cwd=REPO)
    assert reachable.stdout.strip() == ""
    users = subprocess.run(
        ["grep", "-rl", "--include=*.py", "wwj_sequential_reference",
         "experiments"], capture_output=True, text=True, cwd=REPO).stdout.split()
    assert users == ["experiments/s5_wwj/benchmark.py"], users


def test_no_eigendecomposition_in_the_recurrence():
    # no eigendecomposition CALL; the docstring may explain why not
    code = [line for line in _text(RECURRENCE).splitlines()
            if not line.strip().startswith("#")]
    for forbidden in ("eigvals(", "eigh(", "linalg.eig"):
        assert not any(forbidden in line and '"""' not in line
                       and "`" not in line for line in code), forbidden
    assert "Frobenius" in _text(RECURRENCE)


def test_the_parameterization_is_stable_and_declared():
    source = _text(ARMS)
    assert "TAU_MIN + jax.nn.softplus(raw)" in source
    assert "EPS_MAX * jax.nn.sigmoid(raw)" in source
    recurrence = _text(RECURRENCE)
    assert "EPS_MAX = 0.25" in recurrence
    assert "TAU_MIN = 1e-3" in recurrence
    # one scalar per layer, and said so
    assert "per S5 LAYER" in source or "per layer" in source.lower()
    # no post-update eigenvalue projection anywhere
    for path in (ARMS, RECURRENCE, WWJ_MODEL):
        assert "clip_eigs=True" not in _text(path)


def test_the_gates_are_declared_with_numbers():
    benchmark = _text(BENCHMARK)
    for constant in ("MEMORY_CEILING_FRACTION = 0.80", "TIME_MARGIN = 0.25",
                     "THROUGHPUT_FLOOR = 0.50"):
        assert constant in benchmark, constant
    assert "2.98" in benchmark and "154.73" in benchmark   # the old number
    for metric in ("compile_seconds", "steps_per_minute", "peak_gpu_bytes",
                   "forward_seconds", "backward_seconds",
                   "throughput_ratio_to_native", "projection"):
        assert metric in benchmark, metric
    dev = _text(DEV_GATE)
    assert "REQUIRED_LOSS_REDUCTION" in dev and "RADIUS_CEILING" in dev
    assert "DEV_GATE_PASS" in dev and "developmental_only" in dev


def test_the_initialization_rule_is_declared_before_the_grid_is_read():
    source = _text(INIT_GRID)
    assert "GRID_K = (0.05, 0.1, 0.25, 0.5, 1.0)" in source
    assert "GRID_EPS = (0.0, 0.0625, 0.25)" in source
    assert "RADIUS_CEILING = 0.98" in source
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
    for path in (RECURRENCE, ARMS, WWJ_MODEL, BENCHMARK, DEV_GATE, INIT_GRID):
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
    """The earlier O(L*P) backward-memory claim was wrong; the document and
    the module must distinguish forward, residual and measured memory."""
    recurrence = _text(RECURRENCE)
    assert "FORWARD LIVE STORAGE" in recurrence
    assert "BACKWARD (AUTODIFF RESIDUAL) STORAGE IS NOT O(L*P*3)" in recurrence
    assert "MEASURED PEAK DEVICE MEMORY" in recurrence
    assert 'remat="whole"' in recurrence
    documentation = _text(os.path.join(REPO, "docs/S5_WWJ_PROSPECTIVE.md"))
    assert "withdrawn" in documentation
    assert "unknown until the GPU benchmark runs" in documentation
