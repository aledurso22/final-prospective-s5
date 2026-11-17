"""Scoped research runner and frozen production protocol checks (x64 off)."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from flax.traverse_util import flatten_dict

from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full import step_telemetry as ST


@pytest.fixture(autouse=True)
def require_float32():
    assert not jax.config.read("jax_enable_x64"), "run integration tests separately with x64 off"


@pytest.fixture
def tiny(monkeypatch):
    # Computational test fixture, not an experiment/CLI configuration.
    for name, value in {"D_MODEL": 8, "SSM_SIZE_BASE": 16, "BLOCKS": 2,
                        "N_LAYERS": 2, "SEQ_LEN": 32, "BATCH_SIZE": 2}.items():
        monkeypatch.setattr(R, name, value)


def test_import_is_inert_and_protocol_constants_unchanged():
    assert H.ARM not in R.RUNNABLE_ARMS
    assert R.BATCH_SIZE == 16 and H.EPOCHS == 15 and H.ALPHA == .10
    assert H.SCAN == "parallel"
    assert R.SCAN_ARMS == ("generalized_prospective_s5", "matched_lag_prospective_s5",
                          "calibrated_old_s5", "calibrated_prediction_s5")


def test_registration_restores_references_even_on_error():
    old = (R.RUNNABLE_ARMS, R.ssm_factory, R.radius_array, ST.radius_array,
           R.pole_summary, R.production_check, R.ARM_CONFIGS, R.SCIENTIFIC_NAMES)
    with pytest.raises(ValueError, match="intentional"):
        with H.registered_history():
            assert H.ARM in R.RUNNABLE_ARMS
            assert R.ssm_factory("generalized_prospective_s5").keywords["implementation"] == R.GENERALIZED_IMPLEMENTATION
            assert R.recurrent_state_size(H.ARM)["per_layer_real_values"] == 384
            raise ValueError("intentional")
    restored = (R.RUNNABLE_ARMS, R.ssm_factory, R.radius_array, ST.radius_array,
                R.pole_summary, R.production_check, R.ARM_CONFIGS, R.SCIENTIFIC_NAMES)
    assert all(a is b for a, b in zip(old, restored))


def test_history_shares_native_backbone_and_optimizer(tiny):
    native = R.init_state("native_matched_s5", 301)
    with H.registered_history():
        history = R.init_state(H.ARM, 301)
    pn, ph = flatten_dict(native.params), flatten_dict(history.params)
    assert set(pn) == set(ph)
    for key in pn:
        np.testing.assert_array_equal(pn[key], ph[key], err_msg=str(key))
    ln, lh = jax.tree_util.tree_leaves(native.opt_state), jax.tree_util.tree_leaves(history.opt_state)
    assert len(ln) == len(lh)
    for left, right in zip(ln, lh):
        np.testing.assert_array_equal(left, right)


def test_actual_block_poles_finite_update_and_checkpoint(tiny, tmp_path):
    rng = np.random.RandomState(5)
    train = (rng.randn(8, R.SEQ_LEN, 1).astype(np.float32), np.arange(8) % 10)
    val = (rng.randn(4, R.SEQ_LEN, 1).astype(np.float32), np.arange(4) % 10)
    with H.registered_history():
        state = R.init_state(H.ARM, 301)
        radius = float(H.history_radius(state.params))
        poles = R.pole_summary(state.params, H.ARM)
        assert poles["n_modes"] == 16
        assert poles["production_precision"]["unstable"] == 0
        assert poles["production_precision"]["nonfinite"] == 0
        assert radius == pytest.approx(poles["production_precision"]["max_radius_production"], abs=2e-6)
        # The old telemetry's stable key now has documented 3x3 semantics.
        telemetry = ST.make_state_telemetry(H.ARM)(state)
        assert float(telemetry[2]) == pytest.approx(radius, abs=2e-6)
        smoke_dir = tmp_path / "smoke"
        smoke_dir.mkdir()
        model = R.model_for(H.ARM, True)
        first = (jnp.asarray(train[0][:2]), jnp.asarray(train[1][:2]))
        R.production_check(H.ARM, 301, str(smoke_dir), first, state, model, epochs=15)
        production = json.loads((smoke_dir / "production_check.json").read_text())
        assert production["scan_implementation"] == "history_stationary_block"
        smoke = R.full_path_smoke(H.ARM, 301, {"train": train, "val": val},
                                  str(smoke_dir), state, model, epochs=15)
        assert smoke["checkpoint_restored"] and smoke["telemetry_check"]["values_agree"]
        result = R.train_arm(H.ARM, 301, {"train": train, "val": val}, str(tmp_path),
                             state=state, epochs=2)
        assert result["recurrent_state"]["states_per_complex_mode"] == 3
        assert result["epochs_requested"] == 2
        assert result["selected_epoch"] in (1, 2)
        assert result["validation"]["n"] == 4
        raw = Path(result["selected_checkpoint"]).read_bytes()
        restored = serialization.from_bytes(state, raw)
        assert R._all_finite(restored)
        records = [json.loads(s) for s in (tmp_path / "pole_diagnostics.jsonl").read_text().splitlines()]
        assert [r["epoch"] for r in records] == [0, 1, 2]
        assert all(r["radius_semantics"].startswith("actual exact-ZOH 3x3") for r in records)
        steps = [json.loads(s) for s in (tmp_path / "step_metrics.jsonl").read_text().splitlines()]
        assert len(steps) == 8 and not any(s["nonfinite"] for s in steps)


def test_smoke_gate_refuses_wrong_protocol_cpu_or_unstable():
    identity = {"seed": 301, "alpha": .1}
    good = {"protocol_identity": identity, "backend": "gpu", "status": "SMOKE_PASS",
            "code_identifier": H.ARM, "seed": 301,
            "scan_implementation": "history_stationary_block", "training_steps": 2,
            "validation_accuracy": .5, "validation_cross_entropy": .7,
            "checkpoint_restored": True, "telemetry_check": {"values_agree": True},
            "poles_after_smoke_steps": {"production_precision": {"unstable": 0, "nonfinite": 0,
                "max_radius_production": .999, "max_radius_float64": .999}}}
    H.check_smoke_gate(good, identity)
    with pytest.raises(ValueError, match="seed differs"):
        H.check_smoke_gate(good, {"seed": 302, "alpha": .1})
    with pytest.raises(ValueError, match="CPU checks"):
        H.check_smoke_gate({**good, "backend": "cpu"}, identity)
    with pytest.raises(ValueError, match="unstable"):
        H.check_smoke_gate({**good, "poles_after_smoke_steps": {
            "production_precision": {"unstable": 1, "nonfinite": 0}}}, identity)

    for changes, match in (({"code_identifier": "native_matched_s5"}, "arm or seed"),
                           ({"seed": 302}, "arm or seed"),
                           ({"scan_implementation": "factored"}, "stationary history scan"),
                           ({"training_steps": 1}, "exactly two"),
                           ({"validation_accuracy": float("nan")}, "must be finite"),
                           ({"validation_cross_entropy": float("inf")}, "must be finite"),
                           ({"validation_cross_entropy": -.1}, "valid range"),
                           ({"validation_accuracy": 1.1}, "valid range")):
        with pytest.raises(ValueError, match=match):
            H.check_smoke_gate({**good, **changes}, identity)
    for key in ("max_radius_production", "max_radius_float64"):
        for value in (None, float("nan"), float("inf"), -.1):
            pp = {**good["poles_after_smoke_steps"]["production_precision"], key: value}
            with pytest.raises(ValueError, match=key):
                H.check_smoke_gate({**good, "poles_after_smoke_steps": {
                    "production_precision": pp}}, identity)


def test_cli_never_reuses_output(tmp_path):
    out = tmp_path / "keep"
    out.mkdir()
    (out / "artifact").write_text("keep")
    with pytest.raises(SystemExit):
        H.main(["--data-cache", str(tmp_path), "--out", str(out), "--seed", "301", "--smoke"])
    assert (out / "artifact").read_text() == "keep"


def test_result_scan_metadata_preserves_selection_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint_epoch_14.msgpack"
    checkpoint.write_bytes(b"frozen trained checkpoint")
    original = {"code_identifier": H.ARM, "seed": 301, "selected_epoch": 14,
                "selected_checkpoint": str(checkpoint),
                "validation": {"accuracy": .875, "cross_entropy": .4, "n": 16},
                "scan_implementation": None, "epochs_requested": 15}
    result = tmp_path / "task_result.json"
    result.write_text(json.dumps(original))
    before = checkpoint.read_bytes()
    updated = H.stamp_training_result(result, 301)
    assert json.loads(result.read_text()) == updated
    assert updated["scan_implementation"] == "history_stationary_block"
    assert updated["history_alpha"] == H.ALPHA
    for key, value in original.items():
        if key != "scan_implementation":
            assert updated[key] == value
    assert checkpoint.read_bytes() == before
    result_before = result.read_bytes()
    with pytest.raises(ValueError, match="arm or seed"):
        H.stamp_training_result(result, 302)
    assert result.read_bytes() == result_before
