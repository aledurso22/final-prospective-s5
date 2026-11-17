"""Compile-only memory accounting does not execute a training update."""
from types import SimpleNamespace

import jax
import pytest

from experiments.s5_gradient_history import memory_profile as M


def test_memory_accounting_subtracts_donated_aliases():
    stats = SimpleNamespace(argument_size_in_bytes=10, output_size_in_bytes=7,
                            alias_size_in_bytes=3, temp_size_in_bytes=20)
    out = M.memory_fields(stats)
    assert out["buffer_accounting_total_bytes"] == 34


def test_compile_only_history_profile(monkeypatch):
    assert not jax.config.read("jax_enable_x64")
    for name, value in {"D_MODEL": 8, "SSM_SIZE_BASE": 16, "BLOCKS": 2,
                        "N_LAYERS": 2, "SEQ_LEN": 32, "BATCH_SIZE": 2}.items():
        monkeypatch.setattr(M.R, name, value)
    # If an actual training update is run, this test would need concrete data;
    # every array argument below is only an abstract ShapeDtypeStruct.
    report = M.compile_profile(M.H.ARM, kinds=("observable", "telemetry"))
    assert report["training_executed"] is False and report["data_read"] is False
    assert report["backend"] == jax.default_backend()
    assert report["sequence_length"] == 32 and report["batch_size"] == 2
    assert {r["kind"] for r in report["updates"]} == {"observable", "telemetry"}
    assert all(r["memory"]["buffer_accounting_total_bytes"] > 0 for r in report["updates"])
    assert M.H.ARM not in M.R.RUNNABLE_ARMS


def test_profile_refuses_unsupported_arm_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        M.compile_profile("invented_arm")
    out = tmp_path / "keep.json"
    out.write_text("keep")
    with pytest.raises(ValueError, match="existing output"):
        M.main(["--out", str(out)])
    assert out.read_text() == "keep"
