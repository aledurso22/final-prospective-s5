"""Full-state continuation checks on real tiny adaptive/frozen JAX training."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import serialization
from flax.traverse_util import flatten_dict

from experiments.s5_adaptive_proximal_history import runner as AP
from experiments.s5_adaptive_proximal_history import audit as AUDIT
from experiments.s5_adaptive_proximal_history import resume as M
from experiments.s5_gradient_history import runner as H
from experiments.s5_gradient_history import resume as HR
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
    pinned = M.original_source_hashes()
    monkeypatch.setattr(M, "original_source_hashes", lambda: pinned)
    return data, cache


def metadata(source, cache, arm):
    with AP.configured(arm):
        identity = H.protocol_identity(cache, 301)
    identity.update(git_commit=M.SOURCE_COMMIT, source_sha256=M.original_source_hashes())
    (source / "history_protocol.json").write_text(json.dumps(identity))
    smoke = source.parents[1] / "smoke" / arm / "301/smoke_result.json"
    smoke.parent.mkdir(parents=True, exist_ok=True)
    # Synthetic provenance exercises validation only. This CPU test does not
    # claim to provide the real GPU authorization required by the cluster job.
    report = {
        "protocol_identity": identity, "backend": "gpu", "status": "SMOKE_PASS",
        "checkpoint_restored": True, "code_identifier": arm, "seed": 301,
        "scan_implementation": "history_stationary_block", "training_steps": 2,
        "validation_accuracy": .5, "validation_cross_entropy": .7,
        "telemetry_check": {"values_agree": True},
        "poles_after_smoke_steps": {
            "positive_finite_horizons": True,
            "control_horizon_exactly_zero": arm == AP.CONTROL,
            "log_horizon_scale_max_abs": 0. if arm == AP.CONTROL else .01,
            "production_precision": {"unstable": 0, "nonfinite": 0,
                                     "max_radius_production": .9, "max_radius_float64": .9}},
    }
    smoke.write_text(json.dumps(report))
    return identity


class Interrupted(RuntimeError):
    pass


def interrupted_and_full(data, cache, tmp_path, monkeypatch, arm):
    full = tmp_path / "full"
    full.mkdir()
    source = tmp_path / "interrupted" / arm / "301"
    source.mkdir(parents=True)
    with AP.configured(arm), H.registered_history():
        full_result = R.train_arm(arm, 301, data, str(full), epochs=3)
        actual = R.train_one_batch_observable

        def interrupt(state, rng, xb, yb, model, step, *args):
            if step == 5:
                raise Interrupted("partial epoch two")
            return actual(state, rng, xb, yb, model, step, *args)

        with monkeypatch.context() as scoped:
            scoped.setattr(R, "train_one_batch_observable", interrupt)
            with pytest.raises(Interrupted):
                R.train_arm(arm, 301, data, str(source), epochs=3)
    # Include a complete partial-epoch update and a truncated tail, independent
    # of whether the terminated writer's buffering flushed the partial epoch.
    HR._write_rows(source / "step_metrics.jsonl", HR._rows(full / "step_metrics.jsonl")[:5])
    with (source / "step_metrics.jsonl").open("a") as handle:
        handle.write('{"partial":')
    metadata(source, cache, arm)
    return source, full, full_result


@pytest.mark.parametrize("arm", AP.ARMS)
def test_resume_matches_uninterrupted_full_state_and_metrics(tiny, tmp_path, monkeypatch, arm):
    data, cache = tiny
    source, full, full_result = interrupted_and_full(data, cache, tmp_path, monkeypatch, arm)
    before = HR._snapshot(source)
    dest = tmp_path / "resumed" / arm / "301"
    result = M.resume_task(source, dest, cache, 1, arm=arm, require_gpu=False)
    assert HR._snapshot(source) == before
    for epoch in (1, 2, 3):
        name = f"checkpoint_epoch_{epoch:02d}.msgpack"
        assert (dest / name).read_bytes() == (full / name).read_bytes()
    assert HR._rows(dest / "metrics.jsonl") == HR._rows(full / "metrics.jsonl")
    assert HR._rows(dest / "step_metrics.jsonl") == HR._rows(full / "step_metrics.jsonl")
    assert result["selected_epoch"] == full_result["selected_epoch"]
    assert result["validation"] == full_result["validation"]
    assert Path(result["selected_checkpoint"]).parent == dest
    assert result["epochs_requested"] == 3 and result["resumed_epochs"] == 2
    assert result["history_alpha"] == 1 and result["proximal_ratio"] == .5
    assert result["trainable_horizon"] == (arm == AP.ARM)
    assert result["source_unchanged"] and result["selection_includes_source_epochs"]
    assert AUDIT.resolve_task(dest, expected_epochs=3)["selected_epoch"] == result["selected_epoch"]
    preflight = HR._json(dest / "resume_preflight.json")
    assert preflight["validation_exact"] and preflight["source_unchanged"]
    assert [row["step"] for row in preflight["discarded_training_steps"]] == [4, 5]
    provenance = HR._json(dest / "resume_provenance.json")
    assert provenance["restored_step"] == 4
    assert provenance["total_schedule_epochs"] == 3
    assert provenance["discarded_partial_epoch_steps"] == 1
    assert provenance["source_training_seconds"] is None
    with AP.configured(arm), H.registered_history():
        template = R.init_state(arm, 301)
        trained = HR._restore(template, (dest / "checkpoint_epoch_03.msgpack").read_bytes(), 12)
    xi = [np.asarray(value) for key, value in flatten_dict(trained.params).items()
          if key[-1] == AP.HORIZON_KEY]
    assert any(np.any(value != 0) for value in xi) == (arm == AP.ARM)


def test_preflight_only_does_not_train_or_copy_checkpoints(tiny, tmp_path, monkeypatch):
    data, cache = tiny
    source, _, _ = interrupted_and_full(data, cache, tmp_path, monkeypatch, AP.ARM)
    before = HR._snapshot(source)
    dest = tmp_path / "preflight"
    monkeypatch.setattr(R, "train_arm", lambda *a, **kw: pytest.fail("preflight attempted training"))
    result = M.resume_task(source, dest, cache, 1, arm=AP.ARM,
                           require_gpu=False, preflight_only=True)
    assert result["status"] == "RESUME_PREFLIGHT_PASS"
    assert result["source_task"] == str(source.resolve())
    assert result["source_epoch"] == 1 and result["arm"] == AP.ARM
    assert not list(dest.glob("checkpoint*"))
    assert not (dest / "task_result.json").exists()
    assert not (dest / "metrics.jsonl").exists()
    assert HR._snapshot(source) == before


def test_resume_keeps_earlier_best_checkpoint(tiny, tmp_path, monkeypatch):
    data, cache = tiny
    source, _, _ = interrupted_and_full(data, cache, tmp_path, monkeypatch, AP.CONTROL)
    previous = HR._rows(source / "metrics.jsonl")[0]["validation"]
    actual_train = R.train_arm

    def worse_validation(*args, **kwargs):
        # Only continuation sees these scores; checkpoint reproduction remains
        # the real saved-weight evaluation and is still required to pass.
        with monkeypatch.context() as scoped:
            scoped.setattr(R, "evaluate", lambda *a: {
                **previous, "cross_entropy": previous["cross_entropy"] + 1.})
            return actual_train(*args, **kwargs)

    monkeypatch.setattr(R, "train_arm", worse_validation)
    dest = tmp_path / "resumed-earlier-best" / AP.CONTROL / "301"
    result = M.resume_task(source, dest, cache, 1, arm=AP.CONTROL, require_gpu=False)
    assert result["selected_epoch"] == 1 and result["validation"] == previous
    selected = Path(result["selected_checkpoint"])
    assert selected == dest / "checkpoint_epoch_01.msgpack"
    assert selected.read_bytes() == (source / selected.name).read_bytes()
    assert AUDIT.resolve_task(dest, expected_epochs=3)["selected_epoch"] == 1


def test_failed_reproduction_preserves_source_and_refuses_training(tiny, tmp_path, monkeypatch):
    data, cache = tiny
    source, _, _ = interrupted_and_full(data, cache, tmp_path, monkeypatch, AP.CONTROL)
    before = HR._snapshot(source)
    expected = HR._rows(source / "metrics.jsonl")[-1]["validation"]
    monkeypatch.setattr(R, "evaluate", lambda *a: {**expected, "cross_entropy": expected["cross_entropy"] + .1})
    monkeypatch.setattr(R, "train_arm", lambda *a, **kw: pytest.fail("bad reproduction trained"))
    dest = tmp_path / "bad-reproduction"
    with pytest.raises(ValueError, match="failed reproduction"):
        M.resume_task(source, dest, cache, 1, arm=AP.CONTROL, require_gpu=False)
    assert HR._snapshot(source) == before
    assert HR._json(dest / "failure.json")["source_unchanged"]
    assert not (dest / "task_result.json").exists()


def stub(tiny, tmp_path, arm=AP.ARM):
    _, cache = tiny
    source = tmp_path / "source" / arm / "301"
    source.mkdir(parents=True)
    identity = metadata(source, cache, arm)
    HR._write_rows(source / "metrics.jsonl", [{"epoch": 1, "validation": {
        "accuracy": .5, "cross_entropy": 1., "n": 4}}])
    (source / "checkpoint_epoch_01.msgpack").write_bytes(b"placeholder")
    return source, cache, identity


def test_protocol_source_manifest_and_control_configuration_binding(tiny, tmp_path):
    source, cache, identity = stub(tiny, tmp_path)
    for name, value in (("seed", 302), ("alpha", .1), ("epochs", 15),
                        ("batch_size", 16), ("trainable_horizon", False),
                        ("horizon_optimizer_group", "none"), ("proximal_ratio", .6),
                        ("data_manifest_sha256", "wrong"), ("git_commit", "wrong"),
                        ("source_sha256", {})):
        (source / "history_protocol.json").write_text(json.dumps({**identity, name: value}))
        with pytest.raises(ValueError):
            M.inspect_source(source, cache, 1, AP.ARM)
    (source / "history_protocol.json").write_text(json.dumps(identity))
    assert M.inspect_source(source, cache, 1, AP.ARM)["verified_source_file_count"] > 0


def test_missing_checkpoint_and_cpu_smoke_refused(tiny, tmp_path):
    source, cache, _ = stub(tiny, tmp_path)
    checkpoint = source / "checkpoint_epoch_01.msgpack"
    checkpoint.unlink()
    with pytest.raises(ValueError, match="missing completed-epoch checkpoint"):
        M.inspect_source(source, cache, 1, AP.ARM)
    checkpoint.write_bytes(b"placeholder")
    smoke = source.parents[1] / "smoke" / AP.ARM / "301/smoke_result.json"
    report = HR._json(smoke)
    report["backend"] = "cpu"
    smoke.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="CPU checks"):
        M.inspect_source(source, cache, 1, AP.ARM)


def test_no_original_source_change_is_allowed(monkeypatch):
    manifest = {"s5/adaptive_proximal_history.py": "wrong-hash"}
    monkeypatch.setattr(M, "original_source_hashes", lambda: manifest)
    with pytest.raises(ValueError, match="training implementation changed"):
        M.check_source_code({"git_commit": M.SOURCE_COMMIT, "source_sha256": manifest})


def test_output_arm_and_gpu_refusals(tmp_path, monkeypatch):
    source = tmp_path / "original" / AP.ARM / "301"
    source.mkdir(parents=True)
    for out in (source, source.parent / "302", source / "new", source.parents[1], tmp_path):
        with pytest.raises(ValueError, match="outside the source"):
            M.resume_task(source, out, tmp_path, arm=AP.ARM, require_gpu=False)
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep").write_text("preserved")
    with pytest.raises(ValueError, match="new or empty"):
        M.resume_task(source, out, tmp_path, arm=AP.ARM, require_gpu=False)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(ValueError, match="one allocated GPU"):
        M.resume_task(source, tmp_path / "new", tmp_path, arm=AP.ARM)
    with pytest.raises(ValueError, match="unsupported"):
        M.resume_task(source, tmp_path / "new", tmp_path, arm="wrong", require_gpu=False)


def test_restore_refuses_weights_only_and_wrong_step(tiny):
    with AP.configured(AP.ARM), H.registered_history():
        template = R.init_state(AP.ARM, 301)
    with pytest.raises(Exception):
        HR._restore(template, serialization.to_bytes(template.params), 0)
    with pytest.raises(ValueError, match="update count"):
        HR._restore(template, serialization.to_bytes(template), 4)
