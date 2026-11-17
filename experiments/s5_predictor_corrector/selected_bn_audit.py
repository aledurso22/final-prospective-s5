"""Uniform BN sensitivity audit at the ORIGINAL selected checkpoints.

Consumes selected_checkpoint_audit's frozen manifest. Each task retains its
original selected epoch; neither recalibration variant can select an epoch.
The unchanged collapse diagnostic supplies training-only population moments
(upstream normalization uses batch statistics). Its sorted per-batch probe is
not used here: class-grouped caches confound that secondary measurement.
"""

import argparse
import json
from pathlib import Path
import sys

from .bn_collapse_diagnostic import DiagnosticError, main as diagnose, sha256
from .selected_checkpoint_audit import assert_manifest_unchanged


MODES = ("saved_stats", "recal_dropout_off", "recal_dropout_on")


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    tasks = manifest["tasks"]
    identities = [(t["arm"], int(t["seed"])) for t in tasks]
    if not tasks or len(set(identities)) != len(tasks):
        raise DiagnosticError("manifest has no tasks or repeated arm/seed identities")
    return path, manifest, sha256(path)


def check_sources(task):
    if sha256(task["checkpoint"]) != task["checkpoint_sha256"]:
        raise DiagnosticError(f"selected checkpoint changed: {task['checkpoint']}")
    result = Path(task["task_dir"]) / "task_result.json"
    if sha256(result) != task["task_result_sha256"]:
        raise DiagnosticError(f"selection record changed: {result}")


def validate_report(task, report):
    """Refuse summary aggregation if replay or provenance checks failed."""
    if (report["arm"], int(report["seed"])) != (task["arm"], int(task["seed"])):
        raise DiagnosticError("BN report arm/seed does not match manifest")
    epoch = str(task["epoch"])
    if set(report["epochs"]) != {epoch}:
        raise DiagnosticError("BN report did not use exactly the original selected epoch")
    row = report["epochs"][epoch]
    if row["checkpoint_sha256"] != task["checkpoint_sha256"]:
        raise DiagnosticError("BN report checkpoint hash mismatch")
    if not all(report["preservation"].values()):
        raise DiagnosticError("BN diagnostic did not preserve original artifacts")
    if not row["reproduction"]["within_atol"]:
        raise DiagnosticError("saved-statistics evaluation failed original-score reproduction")
    original = task["validation"]
    for k in ("accuracy", "cross_entropy", "n"):
        if row["recorded_validation"][k] != original[k]:
            raise DiagnosticError(f"BN record differs from frozen selection: {k}")
    if report["protocol"]["validation_used_for_statistics"] or report["protocol"]["test_split_read"]:
        raise DiagnosticError("unexpected evaluation-data use in BN statistics")
    scores = {mode: {k: row["validation"][mode][k]
                     for k in ("accuracy", "cross_entropy", "n")} for mode in MODES}
    if any(v["n"] != original["n"] for v in scores.values()):
        raise DiagnosticError("evaluation example counts differ")
    counts = {row["validation"][mode]["recalibration_examples"]
              for mode in MODES if mode != "saved_stats"}
    if len(counts) != 1:
        raise DiagnosticError("recalibration variants used different example counts")
    return {"arm": task["arm"], "seed": task["seed"], "selected_epoch": task["epoch"],
            "checkpoint_sha256": task["checkpoint_sha256"],
            "recorded_primary_validation": original, "scores": scores,
            "recalibration_examples": counts.pop(), "reproduction": row["reproduction"]}


def run_task(manifest_path, index, data_cache, output):
    path, manifest, digest = load_manifest(manifest_path)
    if index < 0 or index >= len(manifest["tasks"]):
        raise DiagnosticError("task index is outside manifest")
    task = manifest["tasks"][index]
    assert_manifest_unchanged(manifest)
    check_sources(task)
    out = Path(output).resolve()
    if out.exists():
        raise DiagnosticError(f"refusing existing output: {out}")
    epoch = str(task["epoch"])
    report = diagnose(["--task", task["task_dir"], "--epochs", epoch,
                       "--reproduce-epoch", epoch, "--data-cache", str(data_cache),
                       "--out", str(out), "--reproduce-atol", "1e-5"])
    check_sources(task)
    if sha256(path) != digest:
        raise DiagnosticError("manifest changed during evaluation")
    assert_manifest_unchanged(manifest)
    summary = validate_report(task, report)
    summary.update(manifest_sha256=digest, task_index=index,
                   epoch_reselection=False, original_results_replaced=False,
                   batch_stat_probe_used_for_inference=False)
    with (out / "selected_summary.json").open("x") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
    return summary


def aggregate(manifest_path, reports_root):
    path, manifest, digest = load_manifest(manifest_path)
    assert_manifest_unchanged(manifest)
    root = Path(reports_root).resolve()
    rows = []
    for index, task in enumerate(manifest["tasks"]):
        check_sources(task)
        folder = root / task["arm"] / str(task["seed"])
        summary = json.loads((folder / "selected_summary.json").read_text())
        if summary["manifest_sha256"] != digest or summary["task_index"] != index:
            raise DiagnosticError("task used a different frozen manifest")
        report = json.loads((folder / "bn_diagnostic.json").read_text())
        rows.append(validate_report(task, report))
    if len({r["recalibration_examples"] for r in rows}) != 1:
        raise DiagnosticError("tasks used different training sample counts for recalibration")
    arm_order = list(dict.fromkeys(r["arm"] for r in rows))
    seed_sets = [{r["seed"] for r in rows if r["arm"] == arm} for arm in arm_order]
    if any(s != seed_sets[0] for s in seed_sets):
        raise DiagnosticError("incomplete seed pairing across arms")
    means = {}
    for arm in arm_order:
        selected = [r for r in rows if r["arm"] == arm]
        means[arm] = {mode: {
            "accuracy": sum(r["scores"][mode]["accuracy"] for r in selected) / len(selected),
            "cross_entropy": sum(r["scores"][mode]["cross_entropy"] for r in selected) / len(selected)
        } for mode in MODES}
    result = {
        "purpose": "BN sensitivity at frozen original selections; secondary diagnostic",
        "manifest_sha256": digest, "tasks": rows, "means": means,
        "epoch_reselection": False, "test_split_read": False,
        "primary_results": "Unchanged selected validation with saved BatchNorm statistics",
        "limitations": [
            "Population moments are pooled with upstream layers using batch statistics.",
            "The collapse tool's sorted per-batch training probe is excluded from this interpretation.",
            "Neither recalibration mode is chosen by its validation score.",
            "Differences are descriptive; two seeds do not establish equivalence or significance."
        ]}
    if sha256(path) != digest:
        raise DiagnosticError("manifest changed during aggregation")
    assert_manifest_unchanged(manifest)
    with (root / "selected_bn_summary.json").open("x") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
    print("Mean validation accuracy (%); original selected epochs are fixed")
    print(f"{'arm':32s} {'saved':>9s} {'recal off':>10s} {'recal on':>10s}")
    for arm, values in means.items():
        print(f"{arm:32s} " + " ".join(f"{100*values[m]['accuracy']:9.4f}" for m in MODES))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task-index", type=int)
    ap.add_argument("--data-cache")
    ap.add_argument("--out")
    ap.add_argument("--aggregate", metavar="REPORTS_ROOT")
    args = ap.parse_args(argv)
    if args.aggregate:
        if args.task_index is not None or args.out or args.data_cache:
            ap.error("aggregation cannot also run a task")
        return aggregate(args.manifest, args.aggregate)
    if args.task_index is None or not args.data_cache or not args.out:
        ap.error("task execution requires --task-index, --data-cache, and --out")
    return run_task(args.manifest, args.task_index, args.data_cache, args.out)


if __name__ == "__main__":
    try:
        main()
    except (DiagnosticError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(2)
