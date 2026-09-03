#!/usr/bin/env python
"""Extract full-precision metrics and provenance from offline W&B runs.

Offline W&B (>=0.29) does NOT write `files/wandb-summary.json`; the metrics
live inside the binary `run-*.wandb` datastore. This reads that datastore and
prints, per run:

  - provenance recorded inside the artifact itself: git commit, host, and the
    exact `run_train.py` argument list;
  - full-precision history and summary metrics;
  - whether the run reached a clean exit.

Console output rounds losses to 5 dp and accuracies to 4 dp, so it cannot
establish identity. This is the machine-readable source of record.

Usage:
    python tools/wandb_extract.py <dir> [<dir> ...]
    python tools/wandb_extract.py "$B"          # a G2b base dir: scans wb/*/
"""

import glob
import json
import os
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

METRICS = ("Training Loss", "Val loss", "Val Accuracy", "Test Loss",
           "Test Accuracy")
SUMMARY_KEYS = ("Best Val Loss", "Best Val Accuracy", "Best Test Loss",
                "Best Test Accuracy", "Best Epoch")


def read_run(wandb_file):
    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)
    out = {"history": [], "summary": {}, "args": None, "git": None,
           "host": None, "exited": False}
    while True:
        raw = ds.scan_data()
        if raw is None:
            break
        rec = pb.Record()
        rec.ParseFromString(raw)
        kind = rec.WhichOneof("record_type")
        if kind == "environment":
            env = rec.environment
            if list(env.args):
                out["args"] = list(env.args)
                out["host"] = env.host
                out["git"] = env.git.commit
        elif kind == "history":
            row = {}
            for item in rec.history.item:
                key = item.key or (item.nested_key[0] if item.nested_key else "")
                row[key] = item.value_json
            out["history"].append(row)
        elif kind == "summary":
            for item in rec.summary.update:
                key = item.key or (item.nested_key[0] if item.nested_key else "")
                if key:
                    out["summary"][key] = item.value_json
        elif kind == "exit":
            out["exited"] = True
    return out


def report(label, wandb_file):
    run = read_run(wandb_file)
    print(f"=== {label}")
    print(f"    run_dir : {os.path.basename(os.path.dirname(wandb_file))}")
    print(f"    git     : {run['git']}")
    print(f"    host    : {run['host']}")
    print(f"    args    : {' '.join(run['args']) if run['args'] else '<none>'}")
    print(f"    exited  : {run['exited']}")

    merged = {}
    for row in run["history"]:
        for key in METRICS:
            if key in row:
                merged[key] = row[key]
    if not merged:
        print("    metrics : <none - run did not complete an epoch>")
    else:
        for key in METRICS:
            if key in merged:
                print(f"    {key:<16}= {merged[key]}")
    for key in SUMMARY_KEYS:
        if key in run["summary"]:
            print(f"    {key:<16}= {run['summary'][key]}")
    print()
    return run


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    targets = []
    for arg in argv:
        tagged = sorted(glob.glob(os.path.join(arg, "wb", "*", "wandb",
                                               "offline-run-*", "run-*.wandb")))
        if tagged:
            for path in tagged:
                tag = path.split(os.sep + "wb" + os.sep)[1].split(os.sep)[0]
                targets.append((tag, path))
            continue
        for path in sorted(glob.glob(os.path.join(arg, "**", "run-*.wandb"),
                                     recursive=True)):
            targets.append((os.path.basename(arg), path))
    if not targets:
        print(f"no run-*.wandb found under: {' '.join(argv)}")
        return 1
    for tag, path in targets:
        report(tag, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
