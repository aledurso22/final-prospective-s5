"""May the completed sigma=1 proximal_inertial_s5 runs serve as the matched baseline?

Reuse is supported only if, for every requested seed:
  * the baseline task completed: 15 epochs, no failure.json, and its recorded
    selected epoch equals the selection rule over the full validation history
    (block_audit.resolve_task);
  * it is the matched arm: proximal_inertial_s5 with alpha 0.1, mu = tau/2,
    m = 1.5 tau and the original input lead (NOT the m = tau control);
  * its recorded protocol identity (history_protocol.json) matches what the
    sigma=0 training will use in everything except arm and input_lead: batch,
    epochs, scan, alpha, ratios, the data manifest digest, the JAX version and
    float32; and it was trained at the pinned baseline commit.

That sigma=1 at THIS commit computes exactly what the baseline commit computed
is checked separately on the GPU by lead_identity.py. Nothing here reads data
splits or writes into the baseline directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax

from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_inertia import block_audit as BA
from experiments.s5_proximal_inertia import runner as PIR

BASE, ABL = "proximal_inertial_s5", "proximal_inertial_nolead_s5"
BASELINE_COMMIT = "acb4d850b7f868ddcfd85da498792c6ff5987749"
SHARED = ("batch_size", "epochs", "scan", "alpha", "mass_ratio", "proximal_ratio",
          "block_width", "horizon_normalization", "outer_optimizer", "jax_enable_x64",
          "data_manifest_sha256", "jax_version")


def check(baseline_root, data_cache, seeds=(301, 302), commit=BASELINE_COMMIT):
    root = Path(baseline_root)
    reasons, tasks = [], []
    meta = {}
    try:
        for line in (root / "run_metadata.txt").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
    except OSError as exc:
        reasons.append(f"run_metadata.txt unreadable: {exc}")
    if meta and meta.get("commit") != commit:
        reasons.append(f"baseline run commit {meta.get('commit')} != {commit}")
    with PIR.configured(ABL):
        current = H.protocol_identity(data_cache, seeds[0])
    for seed in seeds:
        task = root / BASE / str(seed)
        row = {"seed": seed, "task": str(task)}
        try:
            r = BA.resolve_task(task)
            row.update(selected_epoch=r["selected_epoch"], validation=r["validation"],
                       config=r["config"], checkpoint_sha256=r["sha256"])
            if r["arm"] != BASE:
                reasons.append(f"{task}: arm {r['arm']} is not {BASE}")
            want = {"history_alpha": .1, "proximal_ratio": .5, "mass_ratio": 1.5, "input_lead": 1.0}
            if r["config"] != want:
                reasons.append(f"{task}: configuration {r['config']} != matched {want}")
        except (BA.AuditError, OSError, ValueError, KeyError) as exc:
            reasons.append(f"{task}: not a completed, rule-consistent baseline: {exc}")
        try:
            ident = json.loads((task / "history_protocol.json").read_text())
            row["baseline_protocol"] = {k: ident.get(k) for k in SHARED + ("git_commit", "arm", "seed")}
            for k in SHARED:
                if ident.get(k) != current.get(k):
                    reasons.append(f"{task}: protocol {k} {ident.get(k)!r} != current {current.get(k)!r}")
            if ident.get("git_commit") != commit:
                reasons.append(f"{task}: trained at {ident.get('git_commit')} not {commit}")
            if ident.get("seed") != seed or ident.get("arm") != BASE:
                reasons.append(f"{task}: protocol arm/seed mismatch")
        except (OSError, ValueError) as exc:
            reasons.append(f"{task}: history_protocol.json unreadable: {exc}")
        tasks.append(row)
    return {"baseline_root": str(root), "baseline_commit": commit, "seeds": list(seeds),
            "current_protocol": {k: current.get(k) for k in SHARED + ("arm", "input_lead")},
            "differs_by_design": ["arm", "input_lead"], "tasks": tasks,
            "reuse_supported": not reasons, "reasons": reasons,
            "note": "If reuse is not supported, rerun proximal_inertial_s5 at seeds 301/302 "
                    "alongside the ablation instead of comparing across protocols."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--data-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        raise SystemExit(f"FAIL: refusing existing output {args.out}")
    if jax.config.read("jax_enable_x64"):
        raise SystemExit("FAIL: run with JAX_ENABLE_X64=0")
    rep = check(args.baseline_root, args.data_cache)
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({"reuse_supported": rep["reuse_supported"], "reasons": rep["reasons"]}, indent=1))
    if not rep["reuse_supported"]:
        raise SystemExit("FAIL: the completed sigma=1 runs do not support reuse as the matched baseline")
    return rep


if __name__ == "__main__":
    main()
