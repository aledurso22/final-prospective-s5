"""Execution telemetry: which job, which task, and which PHYSICAL GPU.

Two records, deliberately separate, because they answer different questions
with different strength of evidence.

1. PRE-JAX, written before any JAX import so it survives a later external
   kill: the Slurm identity, the CUDA environment, the process ids, the
   resolved cgroup memory limit, and a NODE/NVML GPU INVENTORY.

   The inventory comes from `nvidia-smi --query-gpu`. That is an inventory
   query: it is NOT filtered to the device this task was assigned, and it is
   NOT evidence about which GPU any process used. Two tasks listing the same
   uuids prove nothing about sharing.

2. PROCESS BINDING, written after JAX/CUDA has initialized and the
   production update has run, from `nvidia-smi --query-compute-apps`, which
   reports the pid of each active compute process and the uuid of the GPU it
   is running on. Only this record can tie THIS runner's `os.getpid()` to a
   physical GPU, and it is written to a durable artifact before the smoke
   training steps so it survives a kill during them.

Every result is explicit: `resolved`, `multiple` or `unresolved`, the last
two carrying the command result or error. Nothing here guesses an identity
from the inventory.
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
#: inventory fields (node/NVML scope), in query order
INVENTORY_FIELDS = ("index", "uuid", "pci.bus_id", "name", "memory.total",
                    "memory.used", "compute_mode")
#: compute-application fields: the pid -> physical GPU mapping
COMPUTE_APP_FIELDS = ("pid", "gpu_uuid", "used_gpu_memory", "process_name")
STATUS_RESOLVED, STATUS_MULTIPLE, STATUS_UNRESOLVED = (
    "resolved", "multiple", "unresolved")


def _run_smi(query, timeout=30):
    """`nvidia-smi --query-<kind>` as CSV rows, or an error record."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-{query['kind']}={','.join(query['fields'])}",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}",
                "query": query["kind"]}
    if completed.returncode != 0:
        return {"error": f"nvidia-smi returncode={completed.returncode}",
                "query": query["kind"],
                "stderr": completed.stderr.strip()[:400]}
    rows = []
    for line in completed.stdout.strip().splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == len(query["fields"]):
            rows.append(dict(zip(query["fields"], values)))
    return rows


def _nvidia_smi(timeout=30):
    """NODE/NVML GPU INVENTORY. Not filtered to the assigned device, and not
    evidence about which GPU a process used."""
    return _run_smi({"kind": "gpu", "fields": INVENTORY_FIELDS}, timeout)


def _compute_apps(timeout=30):
    """Active compute processes: pid and the uuid of the GPU each runs on."""
    return _run_smi({"kind": "compute-apps", "fields": COMPUTE_APP_FIELDS},
                    timeout)


def inventory_uuids(record):
    """Uuids from the INVENTORY. Reporting only: never a sharing test."""
    gpus = (record or {}).get("gpu_inventory", {}).get("rows")
    if not isinstance(gpus, list):
        return []
    return [gpu.get("uuid") for gpu in gpus]


# ------------------------------------------------------------- cgroup ------
def _mount_points(mounts_path="/proc/mounts"):
    """(cgroup2 mount, {controller: mount}) from the mount table."""
    v2, v1 = None, {}
    try:
        with open(mounts_path) as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None, {}
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        _, mount, fstype, options = parts[0], parts[1], parts[2], parts[3]
        if fstype == "cgroup2" and v2 is None:
            v2 = mount
        elif fstype == "cgroup":
            for option in options.split(","):
                if option in ("memory", "memory,hugetlb", "hugetlb,memory"):
                    v1.setdefault("memory", mount)
    return v2, v1


def _read_first(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def resolve_cgroup_memory(proc_cgroup="/proc/self/cgroup",
                          mounts_path="/proc/mounts"):
    """THIS process's cgroup and its memory limit, v1 or v2.

    The process's own cgroup path is read from `/proc/self/cgroup` and joined
    to the mounted hierarchy. The limit is taken from the nearest ancestor
    that actually publishes one, and the path it came from is recorded. A
    value found at the hierarchy root is labelled `hierarchy_root`, never
    reported as the job's limit.
    """
    out = {"status": STATUS_UNRESOLVED}
    try:
        with open(proc_cgroup) as handle:
            entries = handle.read().splitlines()
    except OSError as error:
        out["error"] = f"{type(error).__name__}: {error}"
        return out
    v2_mount, v1_mounts = _mount_points(mounts_path)
    relative, version, mount = None, None, None
    for entry in entries:                       # "hid:controllers:path"
        parts = entry.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "" and v2_mount:      # unified hierarchy
            relative, version, mount = path, 2, v2_mount
            break
        if "memory" in controllers.split(",") and v1_mounts.get("memory"):
            relative, version, mount = path, 1, v1_mounts["memory"]
            break
    if relative is None:
        out["error"] = "no memory cgroup entry matched a mounted hierarchy"
        out["proc_self_cgroup"] = entries
        return out
    limit_name = "memory.max" if version == 2 else "memory.limit_in_bytes"
    current_name = "memory.current" if version == 2 else "memory.usage_in_bytes"
    peak_name = "memory.peak" if version == 2 else "memory.max_usage_in_bytes"
    out.update(version=version, mount=mount, cgroup_path=relative)
    segments = [seg for seg in relative.strip("/").split("/") if seg]
    for depth in range(len(segments), -1, -1):
        directory = os.path.join(mount, *segments[:depth])
        limit = _read_first(os.path.join(directory, limit_name))
        if limit is None:
            continue
        at_root = depth == 0
        out.update(status=("hierarchy_root" if at_root else STATUS_RESOLVED),
                   limit_path=os.path.join(directory, limit_name),
                   memory_limit=limit,
                   memory_current=_read_first(
                       os.path.join(directory, current_name)),
                   memory_peak=_read_first(os.path.join(directory, peak_name)),
                   is_job_specific=not at_root)
        if at_root:
            out["note"] = ("limit found only at the hierarchy root: this is "
                           "NOT the job's cgroup limit")
        return out
    out["error"] = f"no {limit_name} between {relative} and the hierarchy root"
    return out


def node_memory(meminfo_path="/proc/meminfo"):
    """Node-wide memory, clearly separate from the cgroup limit."""
    out = {}
    try:
        with open(meminfo_path) as handle:
            for line in handle:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    key, value = line.split(":", 1)
                    out[key] = value.strip()
    except OSError as error:
        out["error"] = str(error)
    return out


# ------------------------------------------------------ process binding ----
def resolve_process_gpu(pid=None, apps=None):
    """Bind a pid to a PHYSICAL GPU uuid from the compute-apps query.

    Returns `resolved` with one uuid, `multiple` with every uuid found for
    the pid, or `unresolved` with the query result or error. It never falls
    back to the inventory.
    """
    pid = os.getpid() if pid is None else int(pid)
    rows = _compute_apps() if apps is None else apps
    record = {"pid": pid,
              "query": "nvidia-smi --query-compute-apps=pid,gpu_uuid"}
    if not isinstance(rows, list):
        record.update(status=STATUS_UNRESOLVED, reason="query_failed",
                      query_result=rows)
        return record
    matches = [row for row in rows if str(row.get("pid")) == str(pid)]
    uuids = sorted({row.get("gpu_uuid") for row in matches
                    if row.get("gpu_uuid")})
    record["matched_rows"] = matches
    record["n_compute_apps"] = len(rows)
    if len(uuids) == 1:
        record.update(status=STATUS_RESOLVED, gpu_uuid=uuids[0])
    elif len(uuids) > 1:
        record.update(status=STATUS_MULTIPLE, gpu_uuids=uuids,
                      reason="pid maps to more than one GPU")
    else:
        record.update(status=STATUS_UNRESOLVED,
                      reason="pid not present in compute-apps output",
                      compute_apps=rows)
    return record


def resolved_gpu_uuid(binding):
    """The uuid only when the binding resolved; otherwise None."""
    if not isinstance(binding, dict):
        return None
    return binding.get("gpu_uuid") if binding.get("status") == \
        STATUS_RESOLVED else None


def classify_shared_gpu(binding_a, binding_b):
    """Did two runners use the same physical GPU?

    Only resolved process bindings can answer. Inventories are never used.
    """
    uuid_a, uuid_b = resolved_gpu_uuid(binding_a), resolved_gpu_uuid(binding_b)
    if uuid_a is None or uuid_b is None:
        unresolved = [name for name, binding in (("a", binding_a),
                                                 ("b", binding_b))
                      if resolved_gpu_uuid(binding) is None]
        return {"classification": "inconclusive",
                "reason": "process-specific GPU binding unresolved",
                "unresolved": unresolved,
                "status": {"a": (binding_a or {}).get("status"),
                           "b": (binding_b or {}).get("status")}}
    same = uuid_a == uuid_b
    return {"classification": ("same_physical_gpu" if same
                               else "different_physical_gpus"),
            "gpu_uuid_a": uuid_a, "gpu_uuid_b": uuid_b}


# ------------------------------------------------------------- records ----
def collect(label="pre_jax", smi=None):
    """The PRE-JAX record.

    `smi` is injectable for tests; it defaults to the module-level probe by
    LOOKUP, not by default-argument binding, so patching
    `gpu_telemetry._nvidia_smi` also affects `write`."""
    smi = _nvidia_smi if smi is None else smi
    inventory = smi()
    return {"label": label,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "hostname": socket.gethostname(),
            "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(),
            "pid_scope": ("this telemetry process only; the runner runs "
                          "later with a different pid"),
            "slurm": {key: os.environ.get(key) for key in SLURM_KEYS},
            "cuda": {key: os.environ.get(key) for key in CUDA_KEYS},
            "gpu_inventory": {
                "query": "nvidia-smi --query-gpu",
                "scope": ("node/NVML inventory: NOT filtered to the assigned "
                          "device and NOT evidence of which GPU a process "
                          "used"),
                "rows": inventory},
            "cgroup_memory": resolve_cgroup_memory(),
            "node_memory": node_memory()}


def _write_json(path, record):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
    return record


def write(path, label="pre_jax"):
    return _write_json(path, collect(label))


def write_process_binding(path, pid=None, apps=None, label="post_cuda_init"):
    """Durable pid -> physical GPU artifact, written before the smoke steps."""
    record = resolve_process_gpu(pid=pid, apps=apps)
    record.update(label=label, hostname=socket.gethostname(),
                  timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()),
                  slurm={key: os.environ.get(key) for key in SLURM_KEYS})
    return _write_json(path, record)


def load(path):
    """A record written earlier, or None."""
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def summary(binding):
    """Compact process-binding summary for lifecycle lines."""
    if not isinstance(binding, dict):
        return {"status": STATUS_UNRESOLVED, "reason": "no binding artifact"}
    out = {"status": binding.get("status"), "pid": binding.get("pid")}
    for key in ("gpu_uuid", "gpu_uuids", "reason"):
        if key in binding:
            out[key] = binding[key]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="pre_jax")
    parser.add_argument("--process-binding", action="store_true",
                        help="query compute-apps and bind THIS pid instead")
    args = parser.parse_args()
    record = (write_process_binding(args.out, label=args.label)
              if args.process_binding else write(args.out, args.label))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
