"""Old vs new `runner.train_arm`: byte-identical training, identical logs.

Imports the runner at --legacy-rev from git, trains both on the same small
synthetic dataset (32 train / 16 val examples, 2 epochs = 4 steps) and
compares epoch checkpoints byte for byte, every per-step telemetry field,
and the validation records. CPU is fine; it takes a few minutes per arm.

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m \
        experiments.s5_predictor_corrector.verify_runner_identity \
        --legacy-rev 15aa1a2 --arm generalized_prospective_s5 --out /tmp/identity
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-rev", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = subprocess.run(["git", "show", f"{args.legacy_rev}:experiments/s5_three_arm_full/runner.py"],
                         check=True, capture_output=True, text=True).stdout
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "runner_legacy.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("runner_legacy", path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    from experiments.s5_three_arm_full import runner as new
    shutil.rmtree(args.out, ignore_errors=True)
    rng = np.random.RandomState(0)
    data = {"train": (rng.normal(size=(32, 16000, 1)).astype(np.float32), rng.randint(0, 10, 32)),
            "val": (rng.normal(size=(16, 16000, 1)).astype(np.float32), rng.randint(0, 10, 16))}
    secs = {}
    for name, mod in (("old", old), ("new", new)):
        d = os.path.join(args.out, name); os.makedirs(d)
        t0 = time.perf_counter(); mod.train_arm(args.arm, 301, data, d, epochs=2)
        secs[name] = time.perf_counter() - t0
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    ok = True
    for ep in (1, 2):
        same = sha(os.path.join(args.out, "old", f"checkpoint_epoch_{ep:02d}.msgpack")) == \
            sha(os.path.join(args.out, "new", f"checkpoint_epoch_{ep:02d}.msgpack"))
        print(f"epoch {ep} checkpoint bytes identical: {same}"); ok &= same
    read = lambda n, f: [json.loads(l) for l in open(os.path.join(args.out, n, f))]
    lo, ln = read("old", "step_metrics.jsonl"), read("new", "step_metrics.jsonl")
    same = lo == ln
    print(f"step records old {len(lo)} new {len(ln)}, every field identical: {same}"); ok &= same
    same = read("old", "metrics.jsonl") == read("new", "metrics.jsonl")
    print(f"validation records identical: {same}"); ok &= same
    print(f"wall seconds old {secs['old']:.1f} new {secs['new']:.1f} (CPU, step-dominated)")
    print("IDENTICAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
