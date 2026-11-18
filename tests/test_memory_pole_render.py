"""Saved report rendering must not access checkpoint files or import training."""
import builtins
import json

import numpy as np
import pytest

from experiments.s5_predictor_corrector import memory_impulses as I
from experiments.s5_predictor_corrector import memory_pole_render as R
from experiments.s5_predictor_corrector.selected_checkpoint_audit import _jsonable
from tests.test_memory_pole_plots import report_fixture
from tests.test_memory_impulses import group


def test_render_saved_json_without_training_modules(tmp_path, monkeypatch):
    report = report_fixture()
    for task in report["tasks"]:
        co = [np.full(3, v) for v in ((.8, 0., 1., 0.) if task["arm"] == I.NATIVE else (1.1, -.24, 1., -.2))]
        impulse = I.analyze_impulses(task["arm"], group(), co, horizon=32)
        for stage in task["stages"].values():
            for layer in stage["layers"]:
                layer["impulse_response"] = impulse
    path = tmp_path / "memory_poles.json"
    path.write_text(json.dumps(_jsonable(report), allow_nan=False))
    original = path.read_bytes()
    real_import = builtins.__import__

    def no_training_import(name, *args, **kwargs):
        if name.split(".")[0] in ("jax", "flax", "torch"):
            raise AssertionError(f"renderer imported training dependency {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_training_import)
    out = tmp_path / "figures"
    files = R.main(["--report", str(path), "--out", str(out)])
    assert len(files) == 4
    assert json.loads((out / "render_status.json").read_text())["status"] == "completed"
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="existing"):
        R.main(["--report", str(path), "--out", str(out)])
