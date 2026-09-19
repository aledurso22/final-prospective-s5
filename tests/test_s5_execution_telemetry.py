"""Execution-topology telemetry and submission topology.

Diagnosis only: nothing here touches a recurrence, the data or the protocol.
These checks need no JAX and no GPU, so they run on the cluster and off it.

The governing distinction, tested repeatedly below: a `--query-gpu` inventory
lists the node's devices and says nothing about which GPU a process used;
only a `--query-compute-apps` row matching the runner's own pid does.
"""

import ast
import io
import json
import os
import subprocess

from experiments.s5_three_arm_full import gpu_telemetry as GT
from experiments.s5_three_arm_full import topology_report as TR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARRAY_SBATCH = os.path.join(REPO, "bin/slurm/s5_three_arm_full_array.sbatch")
ONE_TASK_SBATCH = os.path.join(REPO, "bin/slurm/s5_three_arm_one_task.sbatch")
SMOKE_LAUNCHER = os.path.join(
    REPO, "bin/run_experiments/cluster_s5_three_arm_independent_smoke.sh")
RUNNER = os.path.join(REPO, "experiments/s5_three_arm_full/runner.py")

UUIDS = [f"GPU-0000000{i}-1111-2222-3333-44444444444{i}" for i in range(4)]
#: a four-GPU node inventory: what every task on this node sees, identically
FAKE_INVENTORY = [{"index": str(i), "uuid": UUIDS[i],
                   "pci.bus_id": f"00000000:{0x1A + i:02X}:00.0",
                   "name": "NVIDIA RTX 3090", "memory.total": "24576 MiB",
                   "memory.used": "0 MiB", "compute_mode": "Default"}
                  for i in range(4)]


def _apps(*pid_uuid):
    return [{"pid": str(pid), "gpu_uuid": uuid, "used_gpu_memory": "512 MiB",
             "process_name": "python"} for pid, uuid in pid_uuid]


# ------------------------------------------------- pre-JAX inventory record
def test_collect_records_every_required_identifier(monkeypatch):
    required = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
    for key in required:
        monkeypatch.setenv(key, f"value-of-{key}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    record = GT.collect(smi=lambda: FAKE_INVENTORY)
    for key in required:
        assert record["slurm"][key] == f"value-of-{key}"
    assert record["cuda"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert record["pid"] > 0 and record["pgid"] > 0
    assert record["hostname"] and record["timestamp_utc"].endswith("Z")


def test_the_pre_jax_record_labels_its_scope_honestly():
    """The inventory is stored as an inventory and the pid as the telemetry
    process's, because neither identifies the runner's device."""
    record = GT.collect(smi=lambda: FAKE_INVENTORY)
    assert "gpus" not in record                     # the misleading old key
    assert record["gpu_inventory"]["rows"] == FAKE_INVENTORY
    scope = record["gpu_inventory"]["scope"].lower()
    assert "inventory" in scope and "not" in scope
    assert "runner" in record["pid_scope"]
    assert record["pid"] == os.getpid()             # telemetry process only
    # inventory uuids are reportable, but are not an identity
    assert GT.inventory_uuids(record) == UUIDS
    assert GT.inventory_uuids(None) == []
    assert not hasattr(GT, "physical_gpu_ids")


def test_nvidia_smi_failure_is_recorded_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("nvidia-smi not found")
    monkeypatch.setattr(GT.subprocess, "run", boom)
    for probe in (GT._nvidia_smi, GT._compute_apps):
        result = probe()
        assert "error" in result and "nvidia-smi not found" in result["error"]


def test_write_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(GT, "_nvidia_smi", lambda timeout=30: FAKE_INVENTORY)
    path = tmp_path / "nested" / "gpu_telemetry.json"
    written = GT.write(str(path), label="pre_jax_test")
    loaded = GT.load(str(path))
    assert loaded == written and loaded["label"] == "pre_jax_test"
    assert GT.load(str(tmp_path / "absent.json")) is None
    (tmp_path / "broken.json").write_text("{not json")
    assert GT.load(str(tmp_path / "broken.json")) is None


# --------------------------------------------- process-specific GPU binding
def test_one_pid_on_a_four_gpu_node_resolves_to_its_own_device():
    """The inventory lists four devices; the compute-apps row for this pid
    picks out the one the process actually ran on."""
    binding = GT.resolve_process_gpu(pid=4242,
                                     apps=_apps((4242, UUIDS[2]),
                                                (99, UUIDS[0])))
    assert binding["status"] == GT.STATUS_RESOLVED
    assert binding["gpu_uuid"] == UUIDS[2]
    assert GT.resolved_gpu_uuid(binding) == UUIDS[2]


def test_the_same_inventory_with_different_pid_mappings_means_different_gpus():
    """Two tasks see the identical four-GPU inventory. Only the per-pid rows
    distinguish them, and here they ran on different devices."""
    apps = _apps((100, UUIDS[1]), (200, UUIDS[3]))
    a = GT.resolve_process_gpu(pid=100, apps=apps)
    b = GT.resolve_process_gpu(pid=200, apps=apps)
    verdict = GT.classify_shared_gpu(a, b)
    assert verdict["classification"] == "different_physical_gpus"
    assert (verdict["gpu_uuid_a"], verdict["gpu_uuid_b"]) == (UUIDS[1],
                                                              UUIDS[3])


def test_identical_process_uuids_mean_a_shared_physical_gpu():
    apps = _apps((100, UUIDS[1]), (200, UUIDS[1]))
    verdict = GT.classify_shared_gpu(GT.resolve_process_gpu(100, apps),
                                     GT.resolve_process_gpu(200, apps))
    assert verdict["classification"] == "same_physical_gpu"
    assert verdict["gpu_uuid_a"] == verdict["gpu_uuid_b"] == UUIDS[1]


def test_missing_pid_query_failure_and_ambiguity_never_guess():
    absent = GT.resolve_process_gpu(pid=7, apps=_apps((100, UUIDS[0])))
    assert absent["status"] == GT.STATUS_UNRESOLVED
    assert "not present" in absent["reason"]
    failed = GT.resolve_process_gpu(pid=7, apps={"error": "no nvidia-smi"})
    assert failed["status"] == GT.STATUS_UNRESOLVED
    assert failed["query_result"] == {"error": "no nvidia-smi"}
    ambiguous = GT.resolve_process_gpu(pid=7, apps=_apps((7, UUIDS[0]),
                                                         (7, UUIDS[2])))
    assert ambiguous["status"] == GT.STATUS_MULTIPLE
    assert ambiguous["gpu_uuids"] == [UUIDS[0], UUIDS[2]]
    for unusable in (absent, failed, ambiguous, None, {}):
        assert GT.resolved_gpu_uuid(unusable) is None
        # an unresolved side makes the comparison inconclusive, never a guess
        good = GT.resolve_process_gpu(100, _apps((100, UUIDS[0])))
        verdict = GT.classify_shared_gpu(good, unusable)
        assert verdict["classification"] == "inconclusive"
        assert "unresolved" in verdict["reason"]


def test_the_binding_artifact_is_written_before_it_is_needed(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(GT, "_compute_apps",
                        lambda timeout=30: _apps((os.getpid(), UUIDS[3])))
    path = tmp_path / "gpu_process_binding.json"
    written = GT.write_process_binding(str(path))
    loaded = GT.load(str(path))
    assert loaded == written
    assert loaded["status"] == GT.STATUS_RESOLVED
    assert loaded["gpu_uuid"] == UUIDS[3] and loaded["pid"] == os.getpid()
    assert GT.summary(loaded) == {"status": "resolved", "pid": os.getpid(),
                                  "gpu_uuid": UUIDS[3]}


# ------------------------------------------------------ cgroup resolution --
def _cgroup_v2(tmp_path, relative="/system.slice/slurmstepd.scope/job_66583"):
    mount = tmp_path / "cgroup2"
    job = mount / relative.strip("/")
    job.mkdir(parents=True)
    (job / "memory.max").write_text("53687091200\n")
    (job / "memory.current").write_text("1600000000\n")
    (job / "memory.peak").write_text("2190000000\n")
    proc = tmp_path / "self_cgroup"
    proc.write_text(f"0::{relative}/step_0/user/task_0\n")
    mounts = tmp_path / "mounts"
    mounts.write_text(f"cgroup2 {mount} cgroup2 rw,nsdelegate 0 0\n")
    return proc, mounts, job


def test_cgroup_v2_reports_the_jobs_own_limit_not_the_root(tmp_path):
    proc, mounts, job = _cgroup_v2(tmp_path)
    out = GT.resolve_cgroup_memory(str(proc), str(mounts))
    assert out["status"] == GT.STATUS_RESOLVED and out["version"] == 2
    assert out["is_job_specific"] is True
    assert out["limit_path"] == str(job / "memory.max")
    assert (out["memory_limit"], out["memory_current"], out["memory_peak"]) \
        == ("53687091200", "1600000000", "2190000000")
    assert out["cgroup_path"].endswith("/step_0/user/task_0")


def test_cgroup_v1_reports_the_memory_controller_limit(tmp_path):
    mount = tmp_path / "cgroup" / "memory"
    job = mount / "slurm/uid_1000/job_66583/step_0"
    job.mkdir(parents=True)
    (job / "memory.limit_in_bytes").write_text("53687091200\n")
    (job / "memory.usage_in_bytes").write_text("1600000000\n")
    (job / "memory.max_usage_in_bytes").write_text("2190000000\n")
    proc = tmp_path / "self_cgroup"
    proc.write_text("9:memory:/slurm/uid_1000/job_66583/step_0\n"
                    "3:cpuset:/slurm/uid_1000/job_66583/step_0\n")
    mounts = tmp_path / "mounts"
    mounts.write_text(f"cgroup {mount} cgroup rw,memory 0 0\n")
    out = GT.resolve_cgroup_memory(str(proc), str(mounts))
    assert out["status"] == GT.STATUS_RESOLVED and out["version"] == 1
    assert out["is_job_specific"] is True
    assert out["memory_limit"] == "53687091200"
    assert out["memory_peak"] == "2190000000"


def test_a_root_only_limit_is_never_labelled_the_jobs_limit(tmp_path):
    """The defect this replaces: reading /sys/fs/cgroup/memory.max and
    calling it the task's limit."""
    mount = tmp_path / "cgroup2"
    (mount / "system.slice/slurmstepd.scope").mkdir(parents=True)
    (mount / "memory.max").write_text("max\n")       # hierarchy root only
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/system.slice/slurmstepd.scope\n")
    mounts = tmp_path / "mounts"
    mounts.write_text(f"cgroup2 {mount} cgroup2 rw 0 0\n")
    out = GT.resolve_cgroup_memory(str(proc), str(mounts))
    assert out["status"] == "hierarchy_root"
    assert out["is_job_specific"] is False
    assert "NOT the job" in out["note"]


def test_cgroup_resolution_failure_is_recorded_not_raised(tmp_path):
    out = GT.resolve_cgroup_memory(str(tmp_path / "absent"),
                                   str(tmp_path / "absent"))
    assert out["status"] == GT.STATUS_UNRESOLVED and "error" in out
    assert json.loads(json.dumps(out))["status"] == GT.STATUS_UNRESOLVED


# ----------------------------------------------------- submission topology -
def test_telemetry_runs_before_the_runner_in_both_sbatch_scripts():
    """PRE-JAX: the telemetry call must precede the runner invocation, so the
    job identity exists even if the task is killed during training."""
    for path in (ARRAY_SBATCH, ONE_TASK_SBATCH):
        text = io.open(path).read()
        telemetry = text.index("experiments.s5_three_arm_full.gpu_telemetry")
        runner = text.index("experiments.s5_three_arm_full.runner")
        assert telemetry < runner, path
        assert "--out \"$TASK_ROOT/gpu_telemetry.json\"" in text, path


def test_the_one_task_sbatch_is_not_an_array_job():
    text = io.open(ONE_TASK_SBATCH).read()
    assert "--array" not in text
    assert "SLURM_ARRAY_TASK_ID" not in text
    for guard in ("S5_THREE_ARM_ARM", "S5_THREE_ARM_SEED", "EXPECTED_COMMIT",
                  "git status --porcelain"):
        assert guard in text


def test_smoke_launcher_submits_two_independent_jobs_in_scientific_order():
    text = io.open(SMOKE_LAUNCHER).read()
    assert "--array" not in text
    assert 'SMOKE_TASKS=("native_matched_s5:301" ' \
           '"generalized_prospective_s5:301")' in text
    # exactly one submission is built, inside the loop over the two tasks
    assert text.count("submit=(sbatch --parsable") == 1
    assert "S5_THREE_ARM_SMOKE=1" in text
    assert "s5_three_arm_one_task.sbatch" in text
    assert "s5_three_arm_full_array.sbatch" not in text
    # the inspection recipe reads the binding, never the inventory
    assert "topology_report" in text
    recipe = text.split("cat <<EOF")[1]
    # the printed COMMANDS (indented lines) must not read the inventory for
    # identity; the prose may name it to explain why it cannot be used
    commands = [line for line in recipe.splitlines()
                if line.startswith("  ") and line.strip()]
    assert commands and not any("gpu_telemetry.json" in line
                                for line in commands)
    assert any("topology_report" in line for line in commands)
    assert "inconclusive" in recipe


def _run_base_id_assertion(job_ids):
    """Execute the launcher's own distinct-base-id block against given ids."""
    text = io.open(SMOKE_LAUNCHER).read()
    start = text.index("BASE_IDS=()")
    end = text.index("printf 'run_root=", start)
    script = ("set -euo pipefail\nJOB_IDS=(%s)\n" %
              " ".join(f'"{job_id}"' for job_id in job_ids)) + text[start:end]
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True)


def test_the_launcher_fails_unless_two_distinct_base_job_ids_exist():
    """Printing the base ids was not enough: two array members of one job
    (12345_0, 12345_1) are exactly the topology under test, so they must
    abort the run."""
    same_job = _run_base_id_assertion(["12345_0", "12345_1"])
    assert same_job.returncode != 0
    assert "two distinct base job ids" in same_job.stderr
    assert "NOT created" in same_job.stderr
    one_only = _run_base_id_assertion(["12345"])
    assert one_only.returncode != 0
    distinct = _run_base_id_assertion(["12345", "12346"])
    assert distinct.returncode == 0, distinct.stderr
    assert "distinct: 2" in distinct.stdout


# ----------------------------------------------------- runner and reporting
def test_runner_binds_its_own_pid_before_the_smoke_steps():
    source = io.open(RUNNER).read()
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)}
    assert {"record_lifecycle", "_execution_identity",
            "bind_process_gpu"} <= names
    assert "physical_gpu_ids" not in source          # inventory identity gone
    for anchor in ('"slurm": _execution_identity()',
                   '"process_gpu": GPU_TELEMETRY.summary(binding)',
                   'record["pre_jax_telemetry"] = pre_jax'):
        assert anchor in source, anchor
    # the binding is written after the production check and before the smoke
    bind = source.index("binding = bind_process_gpu(args.out)")
    check = source.index("production_check(args.arm, args.seed, args.out")
    smoke = source.index("result = full_path_smoke(")
    assert check < bind < smoke


def test_topology_report_separates_the_pass_from_the_shared_gpu_question(
        tmp_path):
    native, generalized = tmp_path / "native", tmp_path / "gen"
    for directory, uuid, pid in ((native, UUIDS[0], 11),
                                 (generalized, UUIDS[0], 22)):
        directory.mkdir()
        (directory / "smoke_result.json").write_text(json.dumps(
            {"status": "SMOKE_PASS", "training_steps": 2,
             "checkpoint_restored": True}))
        (directory / "gpu_process_binding.json").write_text(json.dumps(
            GT.resolve_process_gpu(pid, _apps((pid, uuid)))))
    out = TR.report([str(native), str(generalized)])
    assert out["all_full_path_pass"] is True
    assert out["shared_gpu"]["classification"] == "same_physical_gpu"
    # an unresolved binding makes it inconclusive, and a missing artifact or
    # a short run is not a pass
    (generalized / "gpu_process_binding.json").write_text(json.dumps(
        GT.resolve_process_gpu(22, _apps((33, UUIDS[1])))))
    (native / "smoke_result.json").write_text(json.dumps(
        {"status": "SMOKE_PASS", "training_steps": 1,
         "checkpoint_restored": True}))
    out = TR.report([str(native), str(generalized)])
    assert out["all_full_path_pass"] is False
    assert out["shared_gpu"]["classification"] == "inconclusive"
    assert json.dumps(out)                            # serializable


def test_lifecycle_record_shape_is_json_serializable(tmp_path, monkeypatch):
    """`record_lifecycle` writes one JSON object per line; reproduced here
    without importing the runner (which needs JAX)."""
    monkeypatch.setattr(GT, "_nvidia_smi", lambda timeout=30: FAKE_INVENTORY)
    monkeypatch.setattr(GT, "_compute_apps",
                        lambda timeout=30: _apps((os.getpid(), UUIDS[1])))
    out = tmp_path / "task"
    out.mkdir()
    GT.write(str(out / "gpu_telemetry.json"))
    binding = GT.write_process_binding(str(out / "gpu_process_binding.json"))
    record = {"event": "after_production_check_before_full_path",
              "pid": os.getpid(),
              "slurm": {k: os.environ.get(k) for k in GT.SLURM_KEYS},
              "pre_jax_telemetry": GT.load(str(out / "gpu_telemetry.json")),
              "process_gpu": GT.summary(binding)}
    line = json.loads(json.dumps(record))
    assert line["process_gpu"]["gpu_uuid"] == UUIDS[1]
    assert line["process_gpu"]["pid"] == os.getpid()
