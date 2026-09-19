"""The production submission topology: nine tasks, one at a time.

These checks read the real production launcher and the real array sbatch
script, and execute their guards. They need no JAX, no GPU and no Slurm.

Why serialization: jobs 66683 (Native S5, COMPLETED, full SMOKE_PASS) and
66684 (generalized prospective dynamics, FAILED 0:9, empty stderr, no
failure.json) were submitted as two SEPARATE non-array jobs, with separate
job cgroups, and their resolved process bindings show two DIFFERENT physical
GPUs; both ended at exactly 00:01:13, and the generalized arm had already
completed a finite production update. Job 66688 ran the same generalized arm
alone and passed. Array membership and GPU sharing are therefore excluded as
unique causes, and concurrent execution on this node is the implicated
condition.
"""

import io
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(REPO, "bin/run_experiments/cluster_s5_three_arm_full.sh")
ARRAY_SBATCH = os.path.join(REPO, "bin/slurm/s5_three_arm_full_array.sbatch")
FINALIZE_SBATCH = os.path.join(REPO, "bin/slurm/s5_three_arm_full_finalize.sbatch")
RUNNER = os.path.join(REPO, "experiments/s5_three_arm_full/runner.py")
#: the last commit whose scientific behaviour is authoritative
SCIENTIFIC_BASE = "c575dddbb8d9006577e6685eceaca47bff8f9229"
#: the nine tasks, in fixed scientific reporting order
EXPECTED_MAPPING = [("native_matched_s5", seed) for seed in (301, 302, 303)] \
    + [("zucchet_prospective_s5", seed) for seed in (301, 302, 303)] \
    + [("generalized_prospective_s5", seed) for seed in (301, 302, 303)]


def _text(path):
    return io.open(path).read()


def test_the_submitted_array_range_is_exactly_0_8_with_throttle_1():
    launcher = _text(LAUNCHER)
    assert 'ARRAY_SPEC="0-8%1"' in launcher
    assert '--array="$ARRAY_SPEC"' in launcher
    assert "%4" not in launcher and "%2" not in launcher
    # the ONLY --array argument reaching sbatch is the checked spec
    begin = launcher.index("array_submit=(")
    submit = launcher[begin:launcher.index('"$ARRAY_SBATCH")', begin)]
    assert re.findall(r"--array=?\S*", submit) == ['--array="$ARRAY_SPEC"']
    # the sbatch default cannot fall back to concurrency either
    assert "#SBATCH --array=0-8%1" in _text(ARRAY_SBATCH)


def test_the_launcher_refuses_if_the_sbatch_default_disagrees(tmp_path):
    """The two places that carry the throttle are compared before submitting,
    so a later edit to one of them cannot silently widen the run."""
    launcher = _text(LAUNCHER)
    start = launcher.index('ARRAY_SPEC="0-8%1"')
    end = launcher.index('echo "scientific arms', start)
    block = launcher[start:end]
    stub = tmp_path / "stub.sbatch"

    def run(directive):
        stub.write_text(f"#!/usr/bin/env bash\n#SBATCH --array={directive}\n")
        script = (f'set -euo pipefail\nREPO_ROOT="{REPO}"\n'
                  + block.replace(
                      '"$REPO_ROOT/bin/slurm/s5_three_arm_full_array.sbatch"',
                      f'"{stub}"'))
        return subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True)

    assert run("0-8%1").returncode == 0
    for widened in ("0-8%4", "0-8", "0-8%2"):
        refused = run(widened)
        assert refused.returncode == 1, widened
        assert "refusing to submit" in refused.stderr


def _run_serialization_guard(running_lines, squeue_exit=0):
    """Execute the array script's own guard against a stubbed squeue."""
    text = _text(ARRAY_SBATCH)
    start = text.index("if ! RUNNING_SIBLINGS=")
    end = text.index('TASK_ROOT="$S5_THREE_ARM_RUN_ROOT', start)
    body = ("set -euo pipefail\n"
            "squeue() { printf '%s' \"$SQUEUE_OUT\"; return $SQUEUE_RC; }\n"
            "export -f squeue 2>/dev/null || true\n"
            "SLURM_ARRAY_TASK_ID=0\nSLURM_ARRAY_JOB_ID=70001\n"
            + text[start:end])
    env = dict(os.environ,
               SQUEUE_OUT="".join(f"70001_{i}\n" for i in range(running_lines)),
               SQUEUE_RC=str(squeue_exit))
    return subprocess.run(["bash", "-c", body], capture_output=True,
                          text=True, env=env)


def test_no_two_experimental_array_elements_can_run_concurrently():
    """A task refuses to start while a sibling element is RUNNING. This is
    enforced in the job, so the property does not depend on the throttle
    surviving a resubmission or a scheduler setting."""
    alone = _run_serialization_guard(running_lines=1)   # only this task
    assert alone.returncode == 0, alone.stderr
    for siblings in (2, 3, 9):
        blocked = _run_serialization_guard(running_lines=siblings)
        assert blocked.returncode == 1
        assert f"{siblings} elements of array" in blocked.stderr
        assert "requires exactly one" in blocked.stderr
    # zero running is also not the expected state (this task must be running)
    assert _run_serialization_guard(running_lines=0).returncode == 1
    # and an unusable squeue fails CLOSED, never open
    broken = _run_serialization_guard(running_lines=1, squeue_exit=1)
    assert broken.returncode == 1
    assert "cannot verify serialization" in broken.stderr


def test_arm_and_seed_mapping_is_unchanged_for_all_nine_elements():
    """Executed, not pattern-matched: the script's own indexing produces the
    nine (arm, seed) pairs in scientific order."""
    text = _text(ARRAY_SBATCH)
    start = text.index("DISPATCH_ARMS=(")
    end = text.index("# HARD SERIALIZATION GUARD", start)
    observed = []
    for task_id in range(9):
        script = (f"set -euo pipefail\nSLURM_ARRAY_TASK_ID={task_id}\n"
                  + text[start:end] + '\nprintf "%s %s" "$ARM" "$SEED"\n')
        done = subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True)
        assert done.returncode == 0, done.stderr
        arm, seed = done.stdout.split()
        observed.append((arm, int(seed)))
    assert observed == EXPECTED_MAPPING


def test_finalizer_waits_for_every_array_element():
    launcher = _text(LAUNCHER)
    assert '--dependency="afterany:${ARRAY_JOB}"' in launcher
    # the dependency is on the ARRAY JOB ID, which is satisfied only when all
    # nine elements are terminal, and it is never a per-element id
    assert "afterany:${ARRAY_JOB}_" not in launcher
    assert launcher.index("array_submit=(") < launcher.index("ARRAY_JOB=\"$(")
    assert launcher.index("ARRAY_JOB=\"$(") < launcher.index("finalizer_submit=(")
    assert "s5_three_arm_full_finalize.sbatch" in launcher
    assert launcher.count("afterany:") == 1


def test_private_task_paths_and_guards_are_preserved():
    text = _text(ARRAY_SBATCH)
    for anchor in ('TASK_ROOT="$S5_THREE_ARM_RUN_ROOT/$ARM/$SEED"',
                   'export TMPDIR="$TASK_ROOT/tmp"',
                   'export JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/jax-cache"',
                   'test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"',
                   "git status --porcelain"):
        assert anchor in text, anchor
    launcher = _text(LAUNCHER)
    assert 'test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"' in launcher
    assert "Refusing dirty worktree" in launcher


def _blob_hash(path):
    relative = os.path.relpath(path, REPO)
    return subprocess.run(["git", "-C", REPO, "rev-parse",
                           f"{SCIENTIFIC_BASE}:{relative}"],
                          capture_output=True, text=True).stdout.strip()


def _worktree_hash(path):
    return subprocess.run(["git", "-C", REPO, "hash-object", path],
                          capture_output=True, text=True).stdout.strip()


def test_no_scientific_behaviour_changed():
    """Byte identity against the authoritative commit for every file that can
    express a recurrence, the data, the optimizer, the schedule, the
    checkpoint format or the evaluation."""
    for relative in ("s5/ssm.py", "s5/discrete_recurrence.py",
                     "s5/prospective_ssm.py",
                     "s5/generalized_prospective_ssm.py",
                     "s5/three_arm_factory.py",
                     "experiments/s5_three_arm_full/data.py",
                     "experiments/s5_three_arm_full/runner.py",
                     "experiments/s5_three_arm_full/finalize.py",
                     "bin/slurm/s5_three_arm_full_finalize.sbatch"):
        path = os.path.join(REPO, relative)
        assert _blob_hash(path) == _worktree_hash(path) != "", relative


def test_the_only_change_to_the_array_script_is_the_topology():
    """The array script may differ from the authoritative commit only in the
    throttle and the serialization guard, never in what is executed."""
    diff = subprocess.run(
        ["git", "-C", REPO, "diff", SCIENTIFIC_BASE, "--unified=0", "--",
         "bin/slurm/s5_three_arm_full_array.sbatch"],
        capture_output=True, text=True).stdout
    changed = [line for line in diff.splitlines()
               if re.match(r"^[+-][^+-]", line)]
    assert changed, "expected the throttle change"
    for line in changed:
        body = line[1:].strip()
        allowed = (body.startswith("#")            # comment or SBATCH directive
                   or "RUNNING_SIBLINGS" in body
                   or "squeue" in body
                   or body.startswith("echo \"Refusing to run")
                   or body.startswith("echo \"this production run")
                   or body in ("exit 1", "fi", ""))
        assert allowed, line
    # the runner invocation and its arguments are untouched
    assert "--- experiments" not in diff
    for anchor in ("experiments.s5_three_arm_full.runner", "--data-cache",
                   "--arm", "--seed"):
        assert anchor in _text(ARRAY_SBATCH), anchor
