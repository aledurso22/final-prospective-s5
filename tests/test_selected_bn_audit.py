import copy
import json
from pathlib import Path

import pytest

from experiments.s5_predictor_corrector import selected_bn_audit as B


def write_manifest(path, tasks):
    root = path.parent / "run"
    meta = root / "run_metadata.txt"
    meta.write_text("synthetic test metadata")
    enriched = []
    for task in tasks:
        item = copy.deepcopy(task)
        item["source_files"] = {"checkpoint": {"path": item["checkpoint"],
                                              "sha256": item["checkpoint_sha256"]},
                                "task_result": {"path": str(Path(item["task_dir"]) / "task_result.json"),
                                                "sha256": item["task_result_sha256"]}}
        enriched.append(item)
    path.write_text(json.dumps({"tasks": enriched, "run_root": str(root),
                                "run_metadata_sha256": B.sha256(meta)}))


def fixture_task(tmp_path, arm="native_matched_s5", seed=301):
    task_dir = tmp_path / "run" / arm / str(seed)
    task_dir.mkdir(parents=True)
    checkpoint = task_dir / "checkpoint_epoch_14.msgpack"
    checkpoint.write_bytes(b"synthetic frozen checkpoint")
    result = task_dir / "task_result.json"
    result.write_text('{"selected_epoch":14}')
    val = {"accuracy": .75, "cross_entropy": .6, "n": 16}
    task = {"task_dir": str(task_dir), "arm": arm, "seed": seed, "epoch": 14,
            "checkpoint": str(checkpoint), "checkpoint_sha256": B.sha256(checkpoint),
            "task_result_sha256": B.sha256(result), "validation": val}
    scores = {mode: {**val, "accuracy": .75 if mode == "saved_stats" else .8125,
                     "recalibration_examples": 32} for mode in B.MODES}
    report = {"arm": arm, "seed": seed,
              "preservation": {"checkpoints_unchanged": True, "validation_records_unchanged": True},
              "protocol": {"validation_used_for_statistics": False, "test_split_read": False},
              "epochs": {"14": {"checkpoint_sha256": task["checkpoint_sha256"],
                                "recorded_validation": val, "validation": scores,
                                "reproduction": {"within_atol": True, "bitwise_identical": True}}}}
    return task, report


def test_frozen_selected_report(tmp_path):
    task, report = fixture_task(tmp_path)
    B.check_sources(task)
    row = B.validate_report(task, report)
    assert row["selected_epoch"] == 14
    assert row["recorded_primary_validation"]["accuracy"] == .75
    assert row["scores"]["recal_dropout_off"]["accuracy"] == .8125


@pytest.mark.parametrize("corruption", ["epoch", "hash", "replay", "selection", "test", "count", "preservation"])
def test_rejects_invalid_recalibration_reports(tmp_path, corruption):
    task, report = fixture_task(tmp_path)
    row = report["epochs"]["14"]
    if corruption == "epoch":
        report["epochs"]["15"] = report["epochs"].pop("14")
    elif corruption == "hash":
        row["checkpoint_sha256"] = "wrong"
    elif corruption == "replay":
        row["reproduction"]["within_atol"] = False
    elif corruption == "selection":
        row["recorded_validation"] = {**row["recorded_validation"], "accuracy": .5}
    elif corruption == "test":
        report["protocol"]["test_split_read"] = True
    elif corruption == "count":
        row["validation"]["recal_dropout_on"]["n"] = 15
    else:
        report["preservation"]["checkpoints_unchanged"] = False
    with pytest.raises(B.DiagnosticError):
        B.validate_report(task, report)


def test_changed_selection_is_rejected(tmp_path):
    task, _ = fixture_task(tmp_path)
    (tmp_path / "run" / task["arm"] / "301" / "task_result.json").write_text("changed")
    with pytest.raises(B.DiagnosticError, match="selection record changed"):
        B.check_sources(task)


def test_complete_paired_aggregation_and_refuse_overwrite(tmp_path):
    pairs = [fixture_task(tmp_path, arm, seed)
             for arm in ("native_matched_s5", "calibrated_old_s5") for seed in (301, 302)]
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, [t for t, r in pairs])
    reports = tmp_path / "reports"
    for index, (task, report) in enumerate(pairs):
        folder = reports / task["arm"] / str(task["seed"])
        folder.mkdir(parents=True)
        (folder / "bn_diagnostic.json").write_text(json.dumps(report))
        (folder / "selected_summary.json").write_text(json.dumps({
            "manifest_sha256": B.sha256(manifest), "task_index": index}))
    result = B.aggregate(manifest, reports)
    assert result["epoch_reselection"] is False
    assert len(result["tasks"]) == 4
    assert result["means"]["calibrated_old_s5"]["saved_stats"]["accuracy"] == .75
    with pytest.raises(FileExistsError):
        B.aggregate(manifest, reports)
    (reports / "selected_bn_summary.json").unlink()
    (reports / "calibrated_old_s5" / "302" / "selected_summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        B.aggregate(manifest, reports)


def test_task_execution_freezes_epoch_and_uses_both_modes(tmp_path, monkeypatch):
    task, report = fixture_task(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, [task])
    out = tmp_path / "out"
    called = []

    def fake_diagnose(argv):
        called.extend(argv)
        out.mkdir()
        return copy.deepcopy(report)

    monkeypatch.setattr(B, "diagnose", fake_diagnose)
    summary = B.run_task(manifest, 0, tmp_path / "cache", out)
    assert called[called.index("--epochs") + 1] == "14"
    assert "--recal-examples" not in called  # full training batches, unchanged estimator
    assert summary["epoch_reselection"] is False
    assert (out / "selected_summary.json").exists()
    with pytest.raises(B.DiagnosticError, match="existing output"):
        B.run_task(manifest, 0, tmp_path / "cache", out)
