"""Independent numerical and provenance checks for the adaptive-P audit."""
import json

import numpy as np
import pytest
from flax import serialization

from experiments.s5_adaptive_proximal_history import audit as A
from experiments.s5_proximal_inertia import block_audit as BA


def group(xi=(0., 0.)):
    rng = np.random.default_rng(17)
    f = lambda x: np.asarray(x, dtype=np.float32)
    return {"Lambda_re": f([-.4, -.6]), "Lambda_im": f([1.2, 2.]),
            "log_step": f(np.log([[.1], [.2]])),
            "B": f(rng.normal(size=(2, 3, 2))),
            "C1": f(rng.normal(size=(3, 2, 2))),
            "C2": f(rng.normal(size=(3, 2, 2))),
            "log_horizon_scale": f(xi)}


def dense_kernel(h, B, C):
    return 2*np.einsum("op,tp,pi->toi", C, h, B).real


def test_joint_auxiliary_energy_includes_cross_pole_and_cross_mode_cancellation():
    rng = np.random.default_rng(3)
    B = rng.normal(size=(2, 3)) + 1j*rng.normal(size=(2, 3))
    C = rng.normal(size=(4, 2)) + 1j*rng.normal(size=(4, 2))
    t = np.arange(40)[:, None]
    dominant = np.asarray([1+.2j, .3-.7j])[None, :] * np.asarray([.9, .8])[None, :]**t
    first = np.asarray([.5-.3j, -.2+.4j])[None, :] * np.asarray([.7, .6])[None, :]**t
    second = -.93*first  # strong intentional cancellation
    h = dominant+first+second
    report = A.joint_auxiliary_response(h, [dominant, first, second], B, C, np.ones(2, bool))
    full = dense_kernel(h, B, C)
    removed = dense_kernel(first+second, B, C)
    expected = np.sum(removed**2)/np.sum(full**2)
    assert report["status"] == "finite"
    assert report["joint_auxiliary_over_full_energy"] == pytest.approx(expected, rel=1e-12)
    separate = sum(np.sum(dense_kernel(w, B, C)**2) for w in (first, second))/np.sum(full**2)
    assert separate > 100*expected
    assert report["dominant_only"]["total_energy"] == pytest.approx(
        np.sum(dense_kernel(dominant, B, C)**2), rel=1e-12)


def test_unresolved_component_withholds_complete_layer_deletion():
    h = np.ones((5, 2), complex)
    result = A.joint_auxiliary_response(h, [h/2, h/4, h/4],
                                      np.ones((2, 1)), np.ones((1, 2)), [True, False])
    assert result["status"] == "withheld_ill_conditioned_components"
    assert result["joint_auxiliary_over_full_energy"] is None


def test_learned_scale_is_used_by_both_precisions_and_both_auxiliaries_reported():
    g = group((.7, -.4))
    row = A.analyze_layer(A.ARM, "layer", g, 80)
    kappa, _ = BA.kappa64(g)
    assert np.allclose(row["tau_samples"], np.exp(g["log_horizon_scale"])/abs(kappa), rtol=1e-7)
    actual, _, _ = A.blocks64(kappa, g["log_horizon_scale"])
    zero, _, _ = A.blocks64(kappa, np.zeros(2))
    assert np.max(abs(actual-zero)) > 1e-2
    assert row["production_precision"]["max_abs_block_difference"] < 1e-6
    assert len(row["auxiliary_decay_samples"]) == 2
    assert len(row["auxiliary_relative_residue"]) == 2
    assert row["production_precision"]["unstable"] == 0
    for direction in ("forward", "backward"):
        data = row["directions"][direction]
        assert len(data["components"]) == 3
        assert data["joint_auxiliary_removal"]["status"] == "finite"
        assert data["fixed_scale_same_weights_counterfactual"]["squared_kernel_change_over_actual_energy"] > 0


def test_zero_scale_matches_original_p_and_control_checks_saved_values():
    g = group()
    kappa, _ = BA.kappa64(g)
    got = A.blocks64(kappa, np.zeros(2))
    ref = BA.blocks64(BA.P_ARM, {"history_alpha": 1., "proximal_ratio": .5}, kappa)
    # Independent float64 scalar/vector norm paths can differ by a few ulps.
    np.testing.assert_allclose(got[0], ref[0], rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(got[1], ref[1], rtol=2e-14, atol=2e-15)
    A.validate_group(A.CONTROL, g)
    with pytest.raises(A.AuditError, match="nonzero"):
        A.validate_group(A.CONTROL, group((.1, 0)))


def task(root, arm=A.ARM):
    folder = root/arm/"301"
    folder.mkdir(parents=True)
    params = {"encoder": {"seq": group((.2, -.1) if arm == A.ARM else (0, 0))}}
    state = {"params": params}
    for epoch in (12, 15):
        (folder/f"checkpoint_epoch_{epoch:02d}.msgpack").write_bytes(serialization.msgpack_serialize(state))
    rows = [{"epoch": e, "validation": {"accuracy": (3500+(20 if e == 12 else e))/3703,
                                        "cross_entropy": .2, "n": 3703}} for e in range(1, 16)]
    (folder/"metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (folder/"task_result.json").write_text(json.dumps({
        "code_identifier": arm, "seed": 301, "epochs_requested": 15,
        "selected_epoch": 12, "validation": rows[11]["validation"],
        "history_alpha": 1., "proximal_ratio": .5}))
    return folder


def test_end_to_end_preserves_sources_and_reports_init_selected_final(tmp_path, monkeypatch):
    source = task(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    monkeypatch.setattr(A, "init_params", lambda seed: {"encoder": {"seq": group()}})
    out = tmp_path/"audit"
    report = A.main(["--task", str(source), "--out", str(out), "--horizon", "48"])
    assert report["sources_unchanged"] and report["test_split_read"] is False
    assert set(report["tasks"][0]["stages"]) == {"reconstructed_init", "selected", "final"}
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert (out/"block_audit.json").is_file()
    assert "aux1/aux2" in (out/"summary.txt").read_text()
    with pytest.raises(A.AuditError, match="existing output"):
        A.main(["--task", str(source), "--out", str(out)])


def test_refuses_source_output_and_inconsistent_selection(tmp_path):
    source = task(tmp_path)
    with pytest.raises(A.AuditError, match="separate"):
        A.main(["--task", str(source), "--out", str(source/"audit")])
    result_path = source/"task_result.json"
    result = json.loads(result_path.read_text())
    result["selected_epoch"] = 15
    result_path.write_text(json.dumps(result))
    with pytest.raises(A.AuditError, match="selected_epoch"):
        A.resolve_task(source)


def test_source_mutation_during_audit_fails_before_writing(tmp_path, monkeypatch):
    source = task(tmp_path)
    monkeypatch.setattr(A, "init_params", lambda seed: {})
    def changed(*args):
        p = source/"metrics.jsonl"
        p.write_text(p.read_text()+"\n")
        return {}
    monkeypatch.setattr(A, "stage_report", changed)
    out = tmp_path/"audit"
    with pytest.raises(A.AuditError, match="source files changed"):
        A.main(["--task", str(source), "--out", str(out), "--horizon", "4"])
    assert not out.exists()
