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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dirs", nargs="+")
    args = parser.parse_args()
    print(json.dumps(report(args.task_dirs), indent=2))


if __name__ == "__main__":
    main()
