"""Execution-topology telemetry and submission topology.

Diagnosis only: nothing here touches a recurrence, the data or the protocol.
These checks need no JAX and no GPU, so they run on the cluster and off it.
"""

import ast
import io
import json
import os

from experiments.s5_three_arm_full import gpu_telemetry as GT

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARRAY_SBATCH = os.path.join(REPO, "bin/slurm/s5_three_arm_full_array.sbatch")
ONE_TASK_SBATCH = os.path.join(REPO, "bin/slurm/s5_three_arm_one_task.sbatch")
SMOKE_LAUNCHER = os.path.join(
    REPO, "bin/run_experiments/cluster_s5_three_arm_independent_smoke.sh")
RUNNER = os.path.join(REPO, "experiments/s5_three_arm_full/runner.py")

FAKE_SMI = [{"index": "0",
             "uuid": "GPU-1234abcd-5678-90ef-ghij-klmnopqrstuv",
             "pci.bus_id": "00000000:1A:00.0", "name": "NVIDIA RTX 3090",
             "memory.total": "24576 MiB", "memory.used": "0 MiB",
             "compute_mode": "Default"}]


def test_collect_records_every_required_identifier(monkeypatch):
    """The exact variables the diagnosis needs, plus the physical device."""
    required = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
    for key in required:
        monkeypatch.setenv(key, f"value-of-{key}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    record = GT.collect(smi=lambda: FAKE_SMI)
    for key in required:
        assert record["slurm"][key] == f"value-of-{key}"
    assert record["cuda"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert record["gpus"] == FAKE_SMI
    assert record["pid"] > 0 and record["pgid"] > 0
    assert record["hostname"] and record["timestamp_utc"].endswith("Z")


def test_physical_gpu_identity_is_extracted_and_survives_failure():
    assert GT.physical_gpu_ids({"gpus": FAKE_SMI}) == [
        (FAKE_SMI[0]["uuid"], FAKE_SMI[0]["pci.bus_id"])]
    assert GT.physical_gpu_ids({"gpus": {"error": "nvidia-smi missing"}}) == []
    assert GT.physical_gpu_ids(None) == []


def test_write_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(GT, "_nvidia_smi", lambda timeout=30: FAKE_SMI)
    path = tmp_path / "nested" / "gpu_telemetry.json"
    written = GT.write(str(path), label="pre_jax_test")
    loaded = GT.load(str(path))
    assert loaded == written and loaded["label"] == "pre_jax_test"
    assert GT.load(str(tmp_path / "absent.json")) is None
    (tmp_path / "broken.json").write_text("{not json")
    assert GT.load(str(tmp_path / "broken.json")) is None


def test_nvidia_smi_failure_is_recorded_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("nvidia-smi not found")
    monkeypatch.setattr(GT.subprocess, "run", boom)
    result = GT._nvidia_smi()
    assert "error" in result and "nvidia-smi not found" in result["error"]


def test_telemetry_runs_before_the_runner_in_both_sbatch_scripts():
    """PRE-JAX: the telemetry call must precede the runner invocation, so the
    record exists even if the task is killed during training."""
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
    assert text.count("sbatch --parsable") == 1      # one submit per loop pass
    assert "S5_THREE_ARM_SMOKE=1" in text
    assert "s5_three_arm_one_task.sbatch" in text
    # the launcher must not imply a full run
    assert "s5_three_arm_full_array.sbatch" not in text


def test_runner_attaches_execution_identity_to_artifacts():
    tree = ast.parse(io.open(RUNNER).read())
    source = io.open(RUNNER).read()
    assert "gpu_telemetry as GPU_TELEMETRY" in source
    assert "_execution_identity" in source
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)}
    assert {"record_lifecycle", "_execution_identity"} <= names
    # identity reaches the lifecycle log and every result artifact
    for anchor in ('"slurm": _execution_identity()',
                   '"physical_gpus": GPU_TELEMETRY.physical_gpu_ids(pre_jax)',
                   'record["pre_jax_telemetry"] = pre_jax'):
        assert anchor in source, anchor


def test_lifecycle_record_shape_is_json_serializable(tmp_path, monkeypatch):
    """`record_lifecycle` writes one JSON object per line; reproduced here
    without importing the runner (which needs JAX)."""
    monkeypatch.setattr(GT, "_nvidia_smi", lambda timeout=30: FAKE_SMI)
    out = tmp_path / "task"
    out.mkdir()
    GT.write(str(out / "gpu_telemetry.json"))
    pre = GT.load(str(out / "gpu_telemetry.json"))
    record = {"event": "before_training_initialization", "pid": os.getpid(),
              "slurm": {k: os.environ.get(k) for k in GT.SLURM_KEYS},
              "pre_jax_telemetry": pre,
              "physical_gpus": GT.physical_gpu_ids(pre)}
    line = json.dumps(record)
    assert json.loads(line)["physical_gpus"] == [
        [FAKE_SMI[0]["uuid"], FAKE_SMI[0]["pci.bus_id"]]]
