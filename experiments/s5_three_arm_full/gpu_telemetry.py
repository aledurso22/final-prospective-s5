"""PRE-JAX execution telemetry: which job, which task, which PHYSICAL GPU.

Imported and executed BEFORE any JAX import, so the record exists even when
the task is later killed by an external signal. It answers the question the
job-66583 smoke could not: whether two concurrently running tasks were placed
on the same physical device, or merely both saw logical index 0 after Slurm
renumbered their assigned GPU.

No JAX, no model, no dataset: standard library and one `nvidia-smi` call.
"""

import argparse
import json
import os
import socket
import subprocess
import time

#: every environment variable that identifies the job, the array member and
#: the GPU allocation
SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
              "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_JOB_NODELIST",
              "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_MEM_PER_NODE",
              "SLURM_CPUS_PER_TASK", "SLURM_PROCID", "SLURM_TASK_PID",
              "SLURM_SUBMIT_DIR")
CUDA_KEYS = ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
             "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS")
#: nvidia-smi fields, in query order
SMI_FIELDS = ("index", "uuid", "pci.bus_id", "name", "memory.total",
              "memory.used", "compute_mode")


def _nvidia_smi(timeout=30):
    """Physical GPU identity as nvidia-smi reports it to THIS process.

    With CUDA_VISIBLE_DEVICES set, nvidia-smi reports only the visible
    device(s), so the rows below are exactly the devices this task may use.
    Returns a list of dicts, or an error record; never raises.
    """
    query = ",".join(SMI_FIELDS)
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {"error": f"nvidia-smi returncode={completed.returncode}",
                "stderr": completed.stderr.strip()[:400]}
    rows = []
    for line in completed.stdout.strip().splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == len(SMI_FIELDS):
            rows.append(dict(zip(SMI_FIELDS, values)))
    return rows


def _host_memory():
    """Node memory and the cgroup limit, for the node-pressure hypothesis."""
    out = {}
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    key, value = line.split(":", 1)
                    out[key] = value.strip()
    except OSError as error:
        out["meminfo_error"] = str(error)
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as handle:
                out["cgroup_memory_limit"] = handle.read().strip()
                out["cgroup_memory_limit_path"] = path
                break
        except OSError:
            continue
    return out


def collect(label="pre_jax", smi=None):
    """The full pre-JAX record.

    `smi` is injectable for tests; it defaults to the module-level probe by
    LOOKUP, not by default-argument binding, so patching
    `gpu_telemetry._nvidia_smi` also affects `write`."""
    smi = _nvidia_smi if smi is None else smi
    return {"label": label,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "hostname": socket.gethostname(),
            "pid": os.getpid(), "ppid": os.getppid(),
            "pgid": os.getpgrp(),
            "slurm": {key: os.environ.get(key) for key in SLURM_KEYS},
            "cuda": {key: os.environ.get(key) for key in CUDA_KEYS},
            "gpus": smi(),
            "host_memory": _host_memory()}


def write(path, label="pre_jax"):
    record = collect(label)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
    return record


def load(path):
    """The record written earlier by `write`, or None."""
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def physical_gpu_ids(record):
    """(uuid, pci.bus_id) pairs, the identity two tasks would share."""
    gpus = (record or {}).get("gpus")
    if not isinstance(gpus, list):
        return []
    return [(gpu.get("uuid"), gpu.get("pci.bus_id")) for gpu in gpus]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="pre_jax")
    args = parser.parse_args()
    record = write(args.out, args.label)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
