"""Is this `failure.json` the DECLARED numerical failure, or something else?

The Zucchet finite-difference recurrence is the experiment's negative
control: it is intentionally unrepaired, and a numerical training failure is
an expected scientific result that the run carries into the finalizer. Every
other failure — a CUDA error, a missing file, a bad configuration, an
unrelated Python exception, an artifact belonging to another task — is an
infrastructure problem that must stop the experiment.

The presence of `failure.json` cannot distinguish those, so this module
checks the artifact the runner actually writes: `error_type` must name the
declared exception, and the artifact must identify the very task that failed.
Nothing is inferred from the file's existence.
"""

import argparse
import json
import os

#: the only exception class the runner declares as a scientific outcome
EXPECTED_ERROR_TYPE = "NumericalTrainingFailure"


def classify(path, arm, seed):
    """Accept only the declared numerical failure of exactly this task."""
    result = {"failure_artifact": path, "expected_arm": arm,
              "expected_seed": int(seed),
              "expected_error_type": EXPECTED_ERROR_TYPE, "reasons": []}
    if not os.path.exists(path):
        result["reasons"].append("failure.json is missing")
        return dict(result, accepted=False)
    try:
        with open(path) as handle:
            failure = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        result["reasons"].append(f"failure.json is unreadable: {error}")
        return dict(result, accepted=False)
    if not isinstance(failure, dict):
        result["reasons"].append("failure.json is not a JSON object")
        return dict(result, accepted=False)

    result["error_type"] = failure.get("error_type")
    if failure.get("error_type") != EXPECTED_ERROR_TYPE:
        result["reasons"].append(
            f"error_type is {failure.get('error_type')!r}, not "
            f"{EXPECTED_ERROR_TYPE!r}")
    if failure.get("code_identifier") != arm:
        result["reasons"].append(
            f"artifact belongs to arm {failure.get('code_identifier')!r}, "
            f"not {arm!r}")
    if failure.get("seed") != int(seed):
        result["reasons"].append(
            f"artifact belongs to seed {failure.get('seed')!r}, not "
            f"{int(seed)!r}")
    record = failure.get("record")
    if not isinstance(record, dict):
        result["reasons"].append(
            "no numerical-failure record: the declared failure always "
            "carries the step telemetry that produced it")
    else:
        if record.get("code_identifier") != arm or \
                record.get("seed") != int(seed):
            result["reasons"].append(
                "the numerical-failure record identifies a different task: "
                f"{record.get('code_identifier')!r}/{record.get('seed')!r}")
    return dict(result, accepted=not result["reasons"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    verdict = classify(args.failure, args.arm, args.seed)
    print(json.dumps(verdict, indent=2))
    if not verdict["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
