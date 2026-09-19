"""The 15-epoch, three-GPU run inside one existing Slurm allocation.

Nothing here needs JAX, a GPU or Slurm. The launcher tests EXECUTE the real
script against a throwaway git fixture and a stub interpreter, so the wave
ordering, the gating and the failure policy are observed rather than
pattern-matched, and no experiment can start.
"""

import ast
import io
import json
import os
import re
import shutil
import subprocess

from experiments.s5_three_arm_full import failure_gate as FG
from experiments.s5_three_arm_full import topology_report as TR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(
    REPO, "bin/run_experiments/allocation_s5_three_arm_three_gpu.sh")
RUNNER = os.path.join(REPO, "experiments/s5_three_arm_full/runner.py")
FINALIZE = os.path.join(REPO, "experiments/s5_three_arm_full/finalize.py")
#: the commit whose scientific content must not move
SCIENTIFIC_BASE = "7841390ad74b3ae5cc73c51f0ab342887c51c425"
ARMS = ("native_matched_s5", "zucchet_prospective_s5",
        "generalized_prospective_s5")

STUB_PYTHON = r'''#!/usr/bin/env python3
"""Stand-in for the venv interpreter: records what each child was given."""
import json, os, sys

argv = sys.argv[1:]
if "-" in argv[:2]:                       # the data-cache validation heredoc
    sys.stdin.read()
    print("validated official raw cache")
    sys.exit(0)


def value(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


environment = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "TMPDIR": os.environ.get("TMPDIR"),
               "JAX_COMPILATION_CACHE_DIR":
                   os.environ.get("JAX_COMPILATION_CACHE_DIR")}

if "experiments.s5_three_arm_full.gpu_telemetry" in argv:
    write(value("--out"), {"label": value("--label"), "env": environment})
    sys.exit(0)

if "experiments.s5_three_arm_full.topology_report" in argv:
    sys.exit(int(os.environ.get("STUB_GATE_RC", "0")))

if "experiments.s5_three_arm_full.failure_gate" in argv:
    # delegate to the REAL gate, so the launcher is tested against it
    sys.path.insert(0, os.environ["STUB_REAL_REPO"])
    from experiments.s5_three_arm_full import failure_gate
    sys.argv = ["failure_gate"] + argv[argv.index(
        "experiments.s5_three_arm_full.failure_gate") + 1:]
    failure_gate.main()
    sys.exit(0)

if "experiments.s5_three_arm_full.finalize" in argv:
    root = value("--task-root")
    write(os.path.join(value("--out"), "results.json"),
          {"finalized": True, "task_root": root,
           "test_opened_once_by_finalizer": True})
    with open(os.path.join(os.path.dirname(root.rstrip("/")),
                           "finalizer_calls.txt"), "a") as handle:
        handle.write(root + "\n")
    sys.exit(0)

if "experiments.s5_three_arm_full.runner" in argv:
    out, arm, seed = value("--out"), value("--arm"), value("--seed")
    record = {"code_identifier": arm, "seed": int(seed),
              "epochs_requested": int(value("--epochs")),
              "smoke": "--smoke" in argv, "env": environment,
              "order_marker": len(open(os.environ["STUB_ORDER"]).readlines())
              if os.path.exists(os.environ["STUB_ORDER"]) else 0}
    with open(os.environ["STUB_ORDER"], "a") as handle:
        handle.write(f"{arm} {seed} {environment['CUDA_VISIBLE_DEVICES']} "
                     f"{'smoke' if record['smoke'] else 'train'}\n")
    failing = os.environ.get("STUB_FAIL_ARMS", "").split(",")
    if not record["smoke"]:
        failing += os.environ.get("STUB_FAIL_TRAIN_ARMS", "").split(",")
    if arm in failing:
        # the shape runner.py actually writes for the declared failure,
        # unless the case under test asks for a different one
        failure = dict(record, failure="stubbed numerical failure",
                       error_type=os.environ.get("STUB_ERROR_TYPE",
                                                 "NumericalTrainingFailure"),
                       record=dict(record, epoch=1, step=0,
                                   failure="nonfinite"))
        for key in os.environ.get("STUB_FAILURE_DROP", "").split(","):
            failure.pop(key, None)
        if os.environ.get("STUB_FAILURE_ARM"):
            failure["code_identifier"] = os.environ["STUB_FAILURE_ARM"]
        if os.environ.get("STUB_FAILURE_SEED"):
            failure["seed"] = int(os.environ["STUB_FAILURE_SEED"])
        if os.environ.get("STUB_FAILURE_CORRUPT"):
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "failure.json"), "w").write("{not json")
            sys.exit(1)
        write(os.path.join(out, "failure.json"), failure)
        sys.exit(1)
    if record["smoke"]:
        write(os.path.join(out, "smoke_result.json"),
              dict(record, status="SMOKE_PASS", training_steps=2,
                   checkpoint_restored=True))
    else:
        write(os.path.join(out, "task_result.json"),
              dict(record, selected_epoch=3, validation={"accuracy": 0.5}))
    sys.exit(0)

sys.exit(f"stub interpreter: unexpected argv {argv}")
'''


def _fixture(tmp_path):
    """A private repo, a stub interpreter and a runs directory."""
    repo = tmp_path / "repo"
    shutil.copytree(os.path.join(REPO, "bin"), repo / "bin")
    git = ["git", "-C", str(repo)]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["-c", "user.email=t@t", "-c", "user.name=t",
                          "commit", "-qm", "fixture"], check=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(STUB_PYTHON)
    python.chmod(0o755)
    (tmp_path / "data").mkdir()
    return repo, head, python


def _run(tmp_path, dry_run=False, devices="0,1,2,3", job_id="66760", **stub):
    repo, head, python = _fixture(tmp_path)
    order = tmp_path / "order.txt"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(PROSPECTIVE_REPO=str(repo), EXPECTED_COMMIT=head,
               PROSPECTIVE_VENV=str(python.parent.parent),
               PROSPECTIVE_RUNS=str(tmp_path / "runs"),
               S5_THREE_ARM_DATA=str(tmp_path / "data"),
               RUN_ID="fixture", STUB_ORDER=str(order),
               STUB_REAL_REPO=REPO, **stub)
    if job_id is not None:
        env["SLURM_JOB_ID"] = job_id
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    if dry_run:
        env["DRY_RUN"] = "1"
    done = subprocess.run(
        ["bash", str(repo / "bin/run_experiments/"
                     "allocation_s5_three_arm_three_gpu.sh")],
        capture_output=True, text=True, env=env)
    launched = [line.split() for line in
                (order.read_text().splitlines() if order.exists() else [])]
    return done, launched, tmp_path / "runs/s5-three-arm-15epoch/fixture"


# --------------------------------------------------------------- epochs ----
def test_the_runner_still_defaults_to_forty_epochs():
    """The old protocol cannot change silently: --epochs is optional and its
    default is the shared configuration's 40."""
    tree = ast.parse(io.open(RUNNER).read())
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SharedS5Config":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.value is not None:
                    defaults[item.target.id] = ast.literal_eval(item.value)
    assert defaults["epochs"] == 40
    for name in ("apply_scheduled_learning_rate", "train_arm",
                 "production_check", "full_path_smoke"):
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == name)
        argument_names = [argument.arg for argument in function.args.args]
        assert "epochs" in argument_names, name
        # the trailing defaults line up with the trailing arguments
        offset = len(argument_names) - len(function.args.defaults)
        default = function.args.defaults[argument_names.index("epochs") - offset]
        assert isinstance(default, ast.Name) and default.id == "EPOCHS", name
    source = io.open(RUNNER).read()
    assert 'parser.add_argument("--epochs", type=int, default=EPOCHS)' in source


def test_the_cosine_schedule_uses_the_requested_epochs_not_forty():
    source = io.open(RUNNER).read()
    assert "steps_per_epoch * (epochs - WARMUP_END))" in source
    assert "steps_per_epoch * (EPOCHS - WARMUP_END))" not in source
    assert "for epoch in range(epochs):" in source
    assert "for epoch in range(EPOCHS):" not in source
    # the horizon reaches every artifact that reports a result
    for anchor in ('"epochs_requested": epochs',
                   '"epochs_requested": getattr(args, "epochs", EPOCHS)'):
        assert anchor in source, anchor


def test_the_launcher_enforces_fifteen_epochs(tmp_path):
    assert "EPOCHS=15" in io.open(LAUNCHER).read()
    done, launched, _ = _run(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "epochs_requested: 15" in done.stdout
    # every child, smoke and training alike, was given exactly 15
    root = tmp_path / "runs/s5-three-arm-15epoch/fixture"
    results = [json.load(open(os.path.join(base, name)))
               for base, _, files in os.walk(root) for name in files
               if name in ("task_result.json", "smoke_result.json")]
    assert results and all(r["epochs_requested"] == 15 for r in results)
    assert "epochs_requested=15" in (root / "run_metadata.txt").read_text()


# ---------------------------------------------------------- the topology ---
def test_three_distinct_gpu_tokens_and_private_paths(tmp_path):
    done, launched, root = _run(tmp_path)
    assert done.returncode == 0, done.stderr
    training = [row for row in launched if row[3] == "train"]
    assert len(training) == 9
    for arm in ARMS:                      # one wave, three seeds, three GPUs
        wave = [row for row in training if row[0] == arm]
        assert [row[1] for row in wave] == ["301", "302", "303"]
        assert sorted(row[2] for row in wave) == ["0", "1", "2"]
        assert len({row[2] for row in wave}) == 3        # distinct tokens
    assert all(row[2] != "3" for row in launched)        # token 3 unused
    # each child had its own task directory, TMPDIR and compilation cache
    seen = set()
    for arm in ARMS:
        for seed in (301, 302, 303):
            record = json.load(open(root / arm / str(seed) /
                                    "task_result.json"))
            environment = record["env"]
            assert environment["TMPDIR"].endswith(f"{arm}/{seed}/tmp")
            assert environment["JAX_COMPILATION_CACHE_DIR"].endswith(
                f"{arm}/{seed}/jax-cache")
            assert "," not in environment["CUDA_VISIBLE_DEVICES"]
            for path in environment.values():
                assert path not in seen or path == environment[
                    "CUDA_VISIBLE_DEVICES"]
            seen.update(v for k, v in environment.items()
                        if k != "CUDA_VISIBLE_DEVICES")
            assert os.path.exists(root / arm / str(seed) /
                                  "gpu_telemetry.json")


def test_it_refuses_fewer_than_four_visible_gpu_tokens(tmp_path):
    for devices in ("0,1,2", "0", "", None):
        done, _, _ = _run(tmp_path / f"case{devices}", devices=devices)
        assert done.returncode == 1
        assert "at least 4 visible GPU tokens" in done.stderr


def test_it_refuses_to_run_outside_the_allocation(tmp_path):
    done, _, _ = _run(tmp_path, job_id=None)
    assert done.returncode == 1
    assert "not a numeric job id" in done.stderr


def test_no_slurm_command_is_ever_invoked():
    """Every task is a direct child of this shell: a child job or step could
    terminate while another task trains, which is the condition under
    diagnosis."""
    text = io.open(LAUNCHER).read()
    code = "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("#"))
    for command in ("sbatch", "salloc", "srun", "scancel"):
        assert command not in code, command
    assert "&\n" in code and "wait " in code


# ------------------------------------------------------- gating and waves --
def test_the_smoke_runs_first_concurrently_and_gates_training(tmp_path):
    done, launched, root = _run(tmp_path)
    smoke = [row for row in launched if row[3] == "smoke"]
    assert len(smoke) == 3
    assert [row[0] for row in smoke[:2]] == ["native_matched_s5",
                                             "generalized_prospective_s5"]
    assert sorted(row[2] for row in smoke) == ["0", "1", "2"]
    # the smoke precedes every training child
    assert all(launched.index(row) < min(launched.index(other)
                                         for other in launched
                                         if other[3] == "train")
               for row in smoke)
    assert os.path.exists(root / "topology_smoke/native_matched_s5/301/"
                          "smoke_result.json")


def test_a_failed_smoke_gate_stops_the_run_before_training(tmp_path):
    """The gate is the artifact gate (SMOKE_PASS, two steps, restored
    checkpoint, resolved DISTINCT GPUs), not the exit status alone."""
    done, launched, _ = _run(tmp_path, STUB_GATE_RC="1")
    assert done.returncode == 1
    assert "did not pass its artifact" in done.stderr
    assert "was NOT started" in done.stderr
    assert [row for row in launched if row[3] == "train"] == []


def test_waves_are_sequential_and_awaited(tmp_path):
    done, launched, _ = _run(tmp_path)
    training = [row for row in launched if row[3] == "train"]
    assert [row[0] for row in training] == [arm for arm in ARMS
                                            for _ in range(3)]
    text = io.open(LAUNCHER).read()
    # the wave loop waits for every child before the next arm starts
    loop = text[text.index('for arm in "${WAVE_ARMS[@]}"'):]
    assert loop.index("start_child") < loop.index("wait_for_wave")
    assert "wait \"$pid\"" in text


def test_expected_zucchet_failures_do_not_stop_the_generalized_arm(tmp_path):
    """Zucchet's numerical failure is the declared negative control: it is
    recorded and carried on, and the generalized wave still runs."""
    done, launched, root = _run(tmp_path,
                                STUB_FAIL_ARMS="zucchet_prospective_s5")
    assert done.returncode == 0, done.stderr
    assert "expected numerical failure recorded" in done.stdout
    generalized = [row for row in launched
                   if row[0] == "generalized_prospective_s5"
                   and row[3] == "train"]
    assert len(generalized) == 3
    for seed in (301, 302, 303):
        assert os.path.exists(root / "zucchet_prospective_s5" / str(seed) /
                              "failure.json")
        assert os.path.exists(root / "generalized_prospective_s5" /
                              str(seed) / "task_result.json")
    assert os.path.exists(root / "final/results.json")


def test_an_unexpected_failure_stops_the_run_and_the_finalizer(tmp_path):
    for arm in ("native_matched_s5", "generalized_prospective_s5"):
        # the smoke passes; the failure appears in the training wave
        done, launched, root = _run(tmp_path / arm, STUB_FAIL_TRAIN_ARMS=arm)
        assert done.returncode == 1
        assert "unexpected failure in wave" in done.stderr
        assert "artifacts are preserved" in done.stderr
        assert not os.path.exists(root / "final/results.json")
        if arm == "native_matched_s5":       # later waves never started
            assert [row for row in launched
                    if row[0] != "native_matched_s5"
                    and row[3] == "train"] == []
        # the failed arm's own artifacts survive
        assert os.path.exists(root / arm / "301/failure.json")


def test_the_finalizer_runs_once_and_is_the_only_test_reader(tmp_path):
    done, _, root = _run(tmp_path)
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "runs/s5-three-arm-15epoch"
             / "finalizer_calls.txt").read_text().splitlines()
    assert calls == [str(root)]                 # exactly one finalizer call
    launcher = io.open(LAUNCHER).read()
    assert launcher.count("experiments.s5_three_arm_full.finalize") == 2  # +dry
    # the test split is opened by the finalizer and by nothing else: the
    # runner's only data load asks for train and val
    assert '("test",)' in io.open(FINALIZE).read()
    runner_loads = [line for line in io.open(RUNNER).read().splitlines()
                    if "load_official_raw(" in line]
    assert runner_loads
    assert all('("train", "val")' in line and '"test"' not in line
               for line in runner_loads)


# ------------------------------------------------------- what cannot move --
def _blob(path):
    relative = os.path.relpath(path, REPO)
    return subprocess.run(["git", "-C", REPO, "rev-parse",
                           f"{SCIENTIFIC_BASE}:{relative}"],
                          capture_output=True, text=True).stdout.strip()


def _worktree(path):
    return subprocess.run(["git", "-C", REPO, "hash-object", path],
                          capture_output=True, text=True).stdout.strip()


def test_recurrences_and_data_are_byte_identical():
    for relative in ("s5/ssm.py", "s5/discrete_recurrence.py",
                     "s5/prospective_ssm.py",
                     "s5/generalized_prospective_ssm.py",
                     "s5/three_arm_factory.py",
                     "experiments/s5_three_arm_full/data.py"):
        path = os.path.join(REPO, relative)
        assert _blob(path) == _worktree(path) != "", relative


def test_optimizer_model_seed_and_arm_definitions_are_unchanged():
    """The runner changed only to accept a horizon: every block that defines
    the science is textually identical to the authoritative commit."""
    before = subprocess.run(
        ["git", "-C", REPO, "show", f"{SCIENTIFIC_BASE}:"
         "experiments/s5_three_arm_full/runner.py"],
        capture_output=True, text=True).stdout
    now = io.open(RUNNER).read()
    for block in ("@dataclass(frozen=True)\nclass SharedS5Config:",
                  "ARM_CONFIGS = {", "SCIENTIFIC_NAMES = {"):
        start_before, start_now = before.index(block), now.index(block)
        end_before = before.index("\n\n\n", start_before)
        end_now = now.index("\n\n\n", start_now)
        assert before[start_before:end_before] == now[start_now:end_now], block
    assert "SEEDS = (301, 302, 303)" in now
    assert "T_INIT, RHO_INIT = 0.05, 0.5" in now


# ------------------------------------------------------------ the gate -----
def _binding(pid, uuid, status="resolved"):
    record = {"pid": pid, "status": status}
    if status == "resolved":
        record["gpu_uuid"] = uuid
    return record


def _smoke_dir(tmp_path, name, uuid, status="resolved", passed=True):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "smoke_result.json").write_text(json.dumps(
        {"status": "SMOKE_PASS" if passed else "SMOKE_FAIL",
         "training_steps": 2 if passed else 1, "checkpoint_restored": passed}))
    (directory / "gpu_process_binding.json").write_text(
        json.dumps(_binding(1, uuid, status)))
    return str(directory)


def test_the_topology_gate_requires_passes_and_distinct_resolved_gpus(tmp_path):
    good = [_smoke_dir(tmp_path, "a", "GPU-a"), _smoke_dir(tmp_path, "b",
                                                           "GPU-b"),
            _smoke_dir(tmp_path, "c", "GPU-c")]
    assert TR.gate(TR.report(good)) == []
    shared = [_smoke_dir(tmp_path / "s", "a", "GPU-a"),
              _smoke_dir(tmp_path / "s", "b", "GPU-a")]
    problems = TR.gate(TR.report(shared))
    assert any("shared physical GPU GPU-a" in problem for problem in problems)
    unresolved = [_smoke_dir(tmp_path / "u", "a", "GPU-a"),
                  _smoke_dir(tmp_path / "u", "b", None, status="unresolved")]
    problems = TR.gate(TR.report(unresolved))
    assert any("not resolved" in problem for problem in problems)
    failed = [_smoke_dir(tmp_path / "f", "a", "GPU-a", passed=False),
              _smoke_dir(tmp_path / "f", "b", "GPU-b")]
    problems = TR.gate(TR.report(failed))
    assert any("SMOKE_PASS" in problem for problem in problems)


# ------------------------------------ the Zucchet negative-control gate ----
def _failure(tmp_path, **overrides):
    """A `failure.json` of the shape runner.py writes, with edits."""
    record = {"arm": "Zucchet prospective dynamics — finite-difference "
                     "realization",
              "code_identifier": "zucchet_prospective_s5", "seed": 301,
              "epoch": 1, "step": 0, "failure": "nonfinite"}
    failure = {"scientific_name": record["arm"],
               "code_identifier": "zucchet_prospective_s5", "seed": 301,
               "epochs_requested": 15,
               "error_type": "NumericalTrainingFailure",
               "failure": json.dumps(record), "record": record}
    failure.update(overrides)
    for key in [k for k, v in failure.items() if v is _ABSENT]:
        del failure[key]
    path = tmp_path / "failure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failure))
    return str(path)


class _Absent:
    pass


_ABSENT = _Absent()


def test_the_declared_numerical_failure_is_accepted_for_every_seed(tmp_path):
    for seed in (301, 302, 303):
        record = {"code_identifier": "zucchet_prospective_s5", "seed": seed,
                  "epoch": 1, "step": 0, "failure": "nonfinite"}
        path = _failure(tmp_path / str(seed), seed=seed, record=record)
        verdict = FG.classify(path, "zucchet_prospective_s5", seed)
        assert verdict["accepted"] is True, verdict["reasons"]
        assert verdict["reasons"] == []


def test_every_other_kind_of_failure_is_rejected(tmp_path):
    """Presence of failure.json is not a scientific classification."""
    cases = {
        "runtime error": {"error_type": "RuntimeError"},
        "cuda error": {"error_type": "XlaRuntimeError",
                       "failure": "CUDA_ERROR_OUT_OF_MEMORY"},
        "configuration error": {"error_type": "ValueError",
                                "failure": "invalid task identity"},
        "generic failure": {"error_type": _ABSENT},
        "wrong arm": {"code_identifier": "generalized_prospective_s5"},
        "wrong seed": {"seed": 999},
        "no telemetry record": {"record": _ABSENT},
        "record of another task": {
            "record": {"code_identifier": "native_matched_s5", "seed": 301}},
    }
    for name, overrides in cases.items():
        path = _failure(tmp_path / name.replace(" ", "_"), **overrides)
        verdict = FG.classify(path, "zucchet_prospective_s5", 301)
        assert verdict["accepted"] is False, name
        assert verdict["reasons"], name
    # missing and corrupt artifacts
    missing = FG.classify(str(tmp_path / "absent/failure.json"),
                          "zucchet_prospective_s5", 301)
    assert missing["accepted"] is False and "missing" in missing["reasons"][0]
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "failure.json").write_text("{not json")
    broken = FG.classify(str(corrupt / "failure.json"),
                         "zucchet_prospective_s5", 301)
    assert broken["accepted"] is False and "unreadable" in broken["reasons"][0]


def test_the_runner_records_the_exception_class(tmp_path):
    source = io.open(RUNNER).read()
    assert '"error_type": type(error).__name__,' in source
    # and only the declared failure is written as failure.json at all
    assert source.index('"error_type": type(error).__name__,') < \
        source.index('with open(os.path.join(args.out, "failure.json"), "w")')


def test_a_rejected_zucchet_failure_stops_before_the_generalized_wave(
        tmp_path):
    for name, stub in (("runtime error", {"STUB_ERROR_TYPE": "RuntimeError"}),
                       ("cuda error", {"STUB_ERROR_TYPE": "XlaRuntimeError"}),
                       ("no error_type",
                        {"STUB_FAILURE_DROP": "error_type"}),
                       ("no record", {"STUB_FAILURE_DROP": "record"}),
                       ("wrong arm",
                        {"STUB_FAILURE_ARM": "native_matched_s5"}),
                       ("wrong seed", {"STUB_FAILURE_SEED": "999"}),
                       ("corrupt artifact", {"STUB_FAILURE_CORRUPT": "1"})):
        done, launched, root = _run(tmp_path / name.replace(" ", "_"),
                                    STUB_FAIL_TRAIN_ARMS=
                                    "zucchet_prospective_s5", **stub)
        assert done.returncode == 1, name
        assert "not the declared numerical failure" in done.stderr, name
        assert "artifacts are preserved" in done.stderr, name
        # the generalized wave never started and nothing was finalized
        assert [row for row in launched
                if row[0] == "generalized_prospective_s5"
                and row[3] == "train"] == [], name
        assert not os.path.exists(root / "final/results.json"), name
        # every artifact survives
        assert os.path.exists(root / "zucchet_prospective_s5/301/failure.json")
        assert os.path.exists(root / "native_matched_s5/301/task_result.json")


def test_native_and_generalized_failures_are_still_never_excused(tmp_path):
    """No failure.json of any shape makes a Native or generalized failure
    acceptable: the gate applies to the declared control arm only."""
    for arm in ("native_matched_s5", "generalized_prospective_s5"):
        done, _, root = _run(tmp_path / arm, STUB_FAIL_TRAIN_ARMS=arm)
        assert done.returncode == 1
        assert "unexpected failure in wave" in done.stderr
        assert not os.path.exists(root / "final/results.json")
