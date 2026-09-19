"""Read the concurrent-smoke artifacts and say what they do and do not show.

Two questions, kept apart:

1. Did both jobs pass the full path? Answered from `smoke_result.json`, never
   from the Slurm state.
2. Did the two runners use the same physical GPU? Answered only from the
   process-specific `gpu_process_binding.json` records. If either is not
   `resolved`, the answer is **inconclusive**: the node inventory in
   `gpu_telemetry.json` cannot substitute for it.
"""

import argparse
import json
import os

from experiments.s5_three_arm_full import gpu_telemetry as GPU_TELEMETRY

SMOKE_RESULT = "smoke_result.json"


def task_report(task_dir):
    smoke = GPU_TELEMETRY.load(os.path.join(task_dir, SMOKE_RESULT))
    binding = GPU_TELEMETRY.load(
        os.path.join(task_dir, "gpu_process_binding.json"))
    passed = bool(smoke and smoke.get("status") == "SMOKE_PASS"
                  and smoke.get("training_steps") == 2
                  and smoke.get("checkpoint_restored") is True)
    return {"task_dir": task_dir, "full_path_pass": passed,
            "smoke_result": smoke,
            "failure": GPU_TELEMETRY.load(os.path.join(task_dir,
                                                       "failure.json")),
            "process_gpu": GPU_TELEMETRY.summary(binding),
            "_binding": binding}


def report(task_dirs):
    tasks = [task_report(directory) for directory in task_dirs]
    out = {"tasks": [{k: v for k, v in task.items() if k != "_binding"}
                     for task in tasks],
           "all_full_path_pass": all(task["full_path_pass"] for task in tasks)}
    if len(tasks) == 2:
        out["shared_gpu"] = GPU_TELEMETRY.classify_shared_gpu(
            tasks[0]["_binding"], tasks[1]["_binding"])
    return out


def gate(out, require_pass=True, require_distinct_gpus=True):
    """Reasons this report does NOT clear a concurrent-topology smoke.

    An empty list means it does. `require_distinct_gpus` demands that every
    task's binding be `resolved` and that no two of them name the same
    physical GPU: an unresolved binding is never treated as distinct.
    """
    problems = []
    if require_pass:
        for task in out["tasks"]:
            if not task["full_path_pass"]:
                problems.append(f"{task['task_dir']}: no artifact-level "
                                "SMOKE_PASS with two steps and a restored "
                                "checkpoint")
            if task["failure"] is not None:
                problems.append(f"{task['task_dir']}: failure.json present")
    if require_distinct_gpus:
        uuids = {}
        for task in out["tasks"]:
            process_gpu = task["process_gpu"]
            if process_gpu.get("status") != "resolved":
                problems.append(f"{task['task_dir']}: process GPU binding is "
                                f"{process_gpu.get('status')}, not resolved")
                continue
            uuids.setdefault(process_gpu["gpu_uuid"], []).append(
                task["task_dir"])
        for uuid, dirs in uuids.items():
            if len(dirs) > 1:
                problems.append(f"shared physical GPU {uuid}: "
                                + ", ".join(dirs))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dirs", nargs="+")
    parser.add_argument("--require-pass", action="store_true",
                        help="exit 1 unless every task passed the full path")
    parser.add_argument("--require-distinct-gpus", action="store_true",
                        help="exit 1 unless every binding resolved to its own "
                             "physical GPU")
    args = parser.parse_args()
    out = report(args.task_dirs)
    print(json.dumps(out, indent=2))
    if args.require_pass or args.require_distinct_gpus:
        problems = gate(out, args.require_pass, args.require_distinct_gpus)
        for problem in problems:
            print("GATE FAILED: " + problem)
        if problems:
            raise SystemExit(1)
        print("GATE PASSED")


if __name__ == "__main__":
    main()
