"""Real tiny-JAX interrupted/resumed training, separately with x64 off."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import serialization

from experiments.s5_gradient_history import runner as H
from experiments.s5_gradient_history import resume as M
from experiments.s5_three_arm_full import runner as R


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    assert not jax.config.read("jax_enable_x64")
    for key, value in {"D_MODEL": 4, "SSM_SIZE_BASE": 8, "BLOCKS": 2,
                       "N_LAYERS": 1, "SEQ_LEN": 16, "BATCH_SIZE": 2}.items():
        monkeypatch.setattr(R, key, value)
    monkeypatch.setattr(H, "EPOCHS", 3)
    rng = np.random.RandomState(16)
    data = {"train": (rng.randn(8, 16, 1).astype(np.float32), np.arange(8) % 10),
            "val": (rng.randn(4, 16, 1).astype(np.float32), np.arange(4) % 10)}
    cache = tmp_path / "data"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"synthetic_test": True}))
    monkeypatch.setattr(R.EXPERIMENT_DATA, "load_official_raw", lambda *_: (data, {}))
    monkeypatch.setattr(R, "bind_process_gpu", lambda _: {})
    return data, cache


def make_source_metadata(source, cache):
    identity = {"arm": H.ARM, "seed": 301, "alpha": H.ALPHA, "epochs": H.EPOCHS,
                "batch_size": R.BATCH_SIZE, "scan": H.SCAN, "jax_enable_x64": False,
                "jax_version": jax.__version__, "data_cache": str(cache.resolve()),
                "data_manifest_sha256": M.sha256(cache / "manifest.json"),
                "git_commit": M.SOURCE_COMMIT, "source_sha256": M.original_source_hashes()}
    (source / "history_protocol.json").write_text(json.dumps(identity))
    # Synthetic GPU provenance exercises the gate. This CPU unit test does
    # not claim to constitute a production GPU smoke.
    smoke = source.parents[1] / "smoke" / H.ARM / "301" / "smoke_result.json"
    smoke.parent.mkdir(parents=True)
    report = {"protocol_identity": identity, "backend": "gpu", "status": "SMOKE_PASS",
              "checkpoint_restored": True, "code_identifier": H.ARM, "seed": 301,
              "scan_implementation": "history_stationary_block", "training_steps": 2,
              "validation_accuracy": .5, "validation_cross_entropy": .7,
              "telemetry_check": {"values_agree": True},
              "poles_after_smoke_steps": {"production_precision": {
                  "unstable": 0, "nonfinite": 0, "max_radius_production": .9,
                  "max_radius_float64": .9}}}
    smoke.write_text(json.dumps(report))
    return identity


class Interrupted(RuntimeError):
    pass


def test_resume_matches_uninterrupted_full_state_metrics_and_selection(tiny, tmp_path, monkeypatch):
    data, cache = tiny
    full = tmp_path / "full"
    full.mkdir()
    source = tmp_path / "interrupted" / H.ARM / "301"
    source.mkdir(parents=True)
    with H.registered_history():
        uninterrupted = R.train_arm(H.ARM, 301, data, str(full), epochs=3)
        original_update = R.train_one_batch_observable

        def interrupt(state, rng, xb, yb, model, step, *args):
            if step == 4:
                raise Interrupted("simulated kill at the next epoch boundary")
            return original_update(state, rng, xb, yb, model, step, *args)

        with monkeypatch.context() as scoped:
            scoped.setattr(R, "train_one_batch_observable", interrupt)
            with pytest.raises(Interrupted):
                R.train_arm(H.ARM, 301, data, str(source), epochs=3)
    make_source_metadata(source, cache)
    before = M._snapshot(source)
    destination = tmp_path / "resumed" / H.ARM / "301"
    result = M.resume_task(source, destination, cache, 1, require_gpu=False)
    assert M._snapshot(source) == before
    # Full serialized state includes all optimizer moments/counts/rates, BN
    # statistics and update counter, rather than only network parameters.
    for epoch in (1, 2, 3):
        name = f"checkpoint_epoch_{epoch:02d}.msgpack"
        assert (destination / name).read_bytes() == (full / name).read_bytes()
    assert M._rows(destination / "metrics.jsonl") == M._rows(full / "metrics.jsonl")
    assert M._rows(destination / "step_metrics.jsonl") == M._rows(full / "step_metrics.jsonl")
    assert result["selected_epoch"] == uninterrupted["selected_epoch"]
    assert result["validation"] == uninterrupted["validation"]
    assert Path(result["selected_checkpoint"]).parent == destination
    assert result["epochs_requested"] == 3 and result["resumed_epochs"] == 2
    assert result["source_unchanged"] and result["selection_includes_source_epochs"]
    preflight = json.loads((destination / "resume_preflight.json").read_text())
    assert preflight["validation_exact"]
    assert [r["step"] for r in preflight["discarded_training_steps"]] == [4, 5]
    provenance = json.loads((destination / "resume_provenance.json").read_text())
    assert provenance["restored_step"] == 4
    assert provenance["total_schedule_epochs"] == 3
    assert provenance["source_training_seconds"] is None


def test_resume_preserves_earlier_selected_checkpoint(tiny, tmp_path, monkeypatch):
    data, _ = tiny
    original = tmp_path / "earlier.msgpack"
    with H.registered_history():
        state = R.init_state(H.ARM, 301)
        # Construct one complete epoch using the original three-epoch schedule.
        for step, indices in enumerate(R.SC.epoch_batches(8, 2, 301, 0, drop_last=True)):
            state, *_ = R.train_one_batch_observable(
                state, jax.random.PRNGKey(301000 + step),
                *R.batch_arrays(*data["train"], indices), R.model_for(H.ARM, True), step, 4, 3)
        original.write_bytes(serialization.to_bytes(state))
        prior = {"selection": (1., 0., 0), "epoch": 1,
                 "val": {"accuracy": 1., "cross_entropy": 0., "n": 4},
                 "checkpoint": str(original)}
        monkeypatch.setattr(R, "evaluate", lambda *_: {
            "accuracy": .5, "cross_entropy": 1., "n": 4})
        result = R.train_arm(H.ARM, 301, data, str(tmp_path), state=state,
                             epochs=3, start_epoch=1, previous_best=prior)
    assert result["selected_epoch"] == 1
    assert result["selected_checkpoint"] == str(original)
    assert result["validation"] == prior["val"]


def test_output_and_precision_refusals(tmp_path, monkeypatch):
    source = tmp_path / "original" / H.ARM / "301"
    source.mkdir(parents=True)
    for output in (source, source.parent / "302", source / "new", source.parents[1]):
        with pytest.raises(ValueError, match="outside the source"):
            M.resume_task(source, output, tmp_path, require_gpu=False)
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep").write_text("unchanged")
    with pytest.raises(ValueError, match="new or empty"):
        M.resume_task(source, out, tmp_path, require_gpu=False)
    assert (out / "keep").read_text() == "unchanged"
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(ValueError, match="one allocated GPU"):
        M.resume_task(source, tmp_path / "new", tmp_path)


def test_continuation_hook_refuses_reset_or_wrong_step(tiny, tmp_path):
    data, _ = tiny
    with H.registered_history():
        with pytest.raises(ValueError, match="full state"):
            R.train_arm(H.ARM, 301, data, str(tmp_path), epochs=3, start_epoch=1)
        state = R.init_state(H.ARM, 301)
        with pytest.raises(ValueError, match="step disagrees"):
            R.train_arm(H.ARM, 301, data, str(tmp_path), state=state,
                         epochs=3, start_epoch=1, previous_best={})


def test_source_protocol_refusals(tiny, tmp_path, monkeypatch):
    _, cache = tiny
    source = tmp_path / "old" / H.ARM / "301"
    source.mkdir(parents=True)
    identity = make_source_metadata(source, cache)
    for field, value in (("seed", 302), ("alpha", 1.), ("batch_size", 4),
                         ("epochs", 6), ("jax_enable_x64", True),
                         ("git_commit", "unexpected")):
        (source / "history_protocol.json").write_text(json.dumps({**identity, field: value}))
        with pytest.raises(ValueError):
            M.inspect_source(source, cache, 1)
    (source / "history_protocol.json").write_text(json.dumps(identity))
    monkeypatch.setattr(M, "_check_source_code", lambda _: {})
    M._write_rows(source / "metrics.jsonl", [{"epoch": 2, "validation": {
        "accuracy": .5, "cross_entropy": 1., "n": 4}}])
    with pytest.raises(ValueError, match="exactly the completed epochs"):
        M.inspect_source(source, cache, 1)


def test_restore_refuses_weights_only_shape_and_step(tiny):
    with H.registered_history():
        state = R.init_state(H.ARM, 301)
    with pytest.raises(Exception):
        M._restore(state, serialization.to_bytes(state.params), 0)
    with pytest.raises(ValueError, match="update count"):
        M._restore(state, serialization.to_bytes(state), 4)


def test_source_refuses_missing_checkpoint_and_cpu_smoke(tiny, tmp_path, monkeypatch):
    _, cache = tiny
    source = tmp_path / "old" / H.ARM / "301"
    source.mkdir(parents=True)
    make_source_metadata(source, cache)
    monkeypatch.setattr(M, "_check_source_code", lambda _: {})
    M._write_rows(source / "metrics.jsonl", [{"epoch": 1, "validation": {
        "accuracy": .5, "cross_entropy": 1., "n": 4}}])
    with pytest.raises(ValueError, match="missing completed-epoch checkpoint"):
        M.inspect_source(source, cache, 1)
    (source / "checkpoint_epoch_01.msgpack").write_bytes(b"placeholder")
    smoke = source.parents[1] / "smoke" / H.ARM / "301" / "smoke_result.json"
    report = json.loads(smoke.read_text())
    report["backend"] = "cpu"
    smoke.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="CPU checks"):
        M.inspect_source(source, cache, 1)


def test_source_refuses_changed_numerical_implementation(monkeypatch):
    fake_manifest = {"s5/gradient_history.py": "not-the-current-source-hash"}
    monkeypatch.setattr(M, "original_source_hashes", lambda: fake_manifest)
    identity = {"git_commit": M.SOURCE_COMMIT, "source_sha256": fake_manifest}
    with pytest.raises(ValueError, match="training implementation changed"):
        M._check_source_code(identity)
    with pytest.raises(ValueError, match="manifest does not match"):
        M._check_source_code({**identity, "source_sha256": {}})
