"""Known-pole numerical checks and read-only, multi-run checkpoint audit."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.s5_predictor_corrector import memory_poles as M


def group(arm, modes=64, width=3):
    g = {
        "Lambda_re": np.full(modes, -.5, np.float32),
        "Lambda_im": np.linspace(.1, 15, modes, dtype=np.float32),
        "log_step": np.full((modes, 1), np.log(.02), np.float32),
        "B": np.ones((modes, width, 2), np.float32),
        "C1": np.ones((width, modes, 2), np.float32),
        "C2": np.full((width, modes, 2), .5, np.float32),
    }
    if arm != M.ARMS[0]:
        g.update({
            "generalized_T_raw": np.full(modes, np.log(np.expm1(.05)), np.float32),
            "generalized_rho_raw": np.zeros(modes, np.float32),
            "generalized_gamma_raw": np.full(modes, np.log(np.expm1(1.)), np.float32),
        })
    return g


def params(arm, trained=False):
    layers = {}
    for i in range(6):
        g = group(arm)
        if trained:
            g["log_step"] = g["log_step"] + np.float32(.01 * (i + 1))
        layers[f"layers_{i}"] = {"seq": g}
    return {"encoder": layers}


def make_inputs(tmp_path):
    from flax import serialization
    roots = (tmp_path / "seeds301302", tmp_path / "seed303")
    inputs = [[], [], []]
    for root, seeds in zip(roots, ((301, 302), (303,))):
        root.mkdir()
        (root / "run_metadata.txt").write_text(
            "epochs_requested=3\nbatch_size=16\nauthoritative_commit=synthetic\n"
            "scan_implementation=factored\n")
        for ai, arm in enumerate(M.ARMS):
            for seed in seeds:
                task = root / arm / str(seed)
                task.mkdir(parents=True)
                inputs[ai].append(task)
                vals = [dict(accuracy=.7, cross_entropy=.8, n=100),
                        dict(accuracy=.8, cross_entropy=.5, n=100),
                        dict(accuracy=.8, cross_entropy=.5, n=100)]
                (task / "metrics.jsonl").write_text("".join(
                    json.dumps({"epoch": i + 1, "validation": v}) + "\n"
                    for i, v in enumerate(vals)))
                for epoch in (1, 2, 3):
                    (task / f"checkpoint_epoch_{epoch:02d}.msgpack").write_bytes(
                        serialization.msgpack_serialize({"params": params(arm, True)}))
                checkpoint = task / "checkpoint_epoch_02.msgpack"
                (task / "task_result.json").write_text(json.dumps({
                    "code_identifier": arm, "seed": seed, "epochs_requested": 3,
                    "selected_epoch": 2, "selected_checkpoint": str(checkpoint),
                    "validation": vals[1]}))
    return roots, inputs


def snapshot(roots):
    return {str(p): M.CC._sha256(p) for root in roots for p in root.rglob("*") if p.is_file()}


def test_known_poles_decay_and_distinct_exclusion_counts():
    z = np.array([[.9 * np.exp(.3j), .5], [1, .2], [1.00002, .3],
                  [0, 0], [np.nan, .4], [.8, .8]], complex)
    result = M.summarize_poles(z)
    s, decay = result["summary"], result["decay_samples"]
    assert s["n_channels"] == 6 and s["n_poles"] == 12
    assert s["poles_finite_decay"] == 7
    assert s["poles_zero"] == 2
    assert s["poles_equal_one"] == s["poles_gt_one"] == 1
    assert s["poles_nonfinite"] == 1
    assert s["channels_ge_one"] == s["poles_ge_one"] == 2
    assert s["fraction_channels_ge_one"] == pytest.approx(2 / 6)
    assert s["fraction_poles_ge_one"] == pytest.approx(2 / 12)
    assert s["poles_boundary_band"] == 1
    assert s["poles_gt_one_plus_tol"] == 1
    assert decay[0, 0] == pytest.approx(-1 / np.log(.9))
    assert decay[2, 1] == pytest.approx(-1 / np.log(.3))
    assert np.isnan(decay[2, 0])
    assert np.isnan(decay[3]).all()
    assert decay[5, 0] == decay[5, 1]


def test_tiny_positive_pole_has_positive_finite_decay():
    d = M.summarize_poles(np.array([[1e-25]], complex))["decay_samples"][0, 0]
    assert d > 0
    assert d == pytest.approx(-1 / np.log(1e-25))


@pytest.mark.parametrize("z", [np.array([]), np.array([.5]), np.empty((0, 2))])
def test_bad_pole_array_refused(z):
    with pytest.raises(ValueError):
        M.summarize_poles(z)


@pytest.mark.parametrize("arm", M.ARMS)
def test_actual_arm_coefficients_against_literal_recurrence_and_companion(arm):
    g = group(arm, modes=4)
    m, co, hyper = M.coefficients64(arm, g)
    a1, a2, b0, b1 = co
    rng = np.random.default_rng(7)
    s1, s2, xt, xp = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    if arm == M.ARMS[0]:
        expected = m["lambda_bar"] * s1 + xt
    else:
        T, mass, gamma = (hyper[k] for k in ("T", "M", "gamma"))
        Q = mass + gamma + T
        k = T if arm == M.ARMS[1] else gamma
        mu, eta, zeta = (mass + T) / Q, k / Q, k * T / Q
        K = 1 - m["lambda_bar"]
        d = mu * (s1 - s2) - zeta * ((K * s1 - xt) - (K * s2 - xp))
        expected = s1 + d - eta * (K * s1 - xt)
    np.testing.assert_allclose(a1 * s1 + a2 * s2 + b0 * xt + b1 * xp,
                               expected, rtol=3e-14, atol=3e-14)
    layer = M.analyze_layer(arm, "test", g)
    assert layer["roots"].shape == (4, 1 if arm == M.ARMS[0] else 2)
    for i in range(4):
        expected = [a1[i]] if arm == M.ARMS[0] else np.linalg.eigvals(
            np.array([[a1[i], a2[i]], [1, 0]], complex))
        np.testing.assert_allclose(np.sort_complex(layer["roots"][i]),
                                   np.sort_complex(expected), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(layer["float32_cpu"]["radius"], layer["radius"],
                               rtol=5e-5, atol=4e-7)


def test_manifest_three_seeds_from_two_runs_and_nonlast_selected_checkpoint(tmp_path):
    roots, inputs = make_inputs(tmp_path)
    before = snapshot(roots)
    manifest = M.manifest_for_inputs(*inputs, expected_epochs=3)
    assert manifest["seeds"] == [301, 302, 303]
    assert len(manifest["tasks"]) == 9
    assert all(t["epoch"] == 2 for t in manifest["tasks"])
    assert all(t["checkpoint"].endswith("checkpoint_epoch_02.msgpack") for t in manifest["tasks"])
    M.assert_sources_unchanged(manifest)
    assert before == snapshot(roots)


@pytest.mark.parametrize("problem", ["unmatched", "duplicate", "wrong_arm", "selection", "metadata", "missing_checkpoint", "failure"])
def test_manifest_refuses_unsupported_or_incomplete_inputs(tmp_path, problem):
    roots, inputs = make_inputs(tmp_path)
    task = inputs[0][0]
    if problem == "unmatched":
        inputs[2] = inputs[2][:-1]
    elif problem == "duplicate":
        inputs[0].append(inputs[0][0])
    elif problem == "wrong_arm":
        inputs[0][0] = inputs[1][0]
    elif problem == "selection":
        p = task / "task_result.json"
        r = json.loads(p.read_text()); r["selected_epoch"] = 3
        p.write_text(json.dumps(r))
    elif problem == "metadata":
        (roots[1] / "run_metadata.txt").write_text("epochs_requested=3\n")
    elif problem == "missing_checkpoint":
        (task / "checkpoint_epoch_02.msgpack").unlink()
    elif problem == "failure":
        (task / "failure.json").write_text("{}")
    with pytest.raises((M.SA.AuditError, M.CC.CheckpointError)):
        M.manifest_for_inputs(*inputs, expected_epochs=3)


@pytest.mark.parametrize("source", ["checkpoint", "task_result", "metrics", "run_metadata"])
def test_source_mutation_detected(tmp_path, source):
    _, inputs = make_inputs(tmp_path)
    manifest = M.manifest_for_inputs(*inputs, expected_epochs=3)
    p = Path(manifest["tasks"][0]["source_files"][source]["path"])
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(M.SA.AuditError, match="source changed"):
        M.assert_sources_unchanged(manifest)


def test_output_must_be_new_and_outside_both_source_runs(tmp_path):
    roots, inputs = make_inputs(tmp_path)
    manifest = M.manifest_for_inputs(*inputs, expected_epochs=3)
    for root in roots:
        with pytest.raises(M.SA.AuditError, match="outside"):
            M.output_guard(root / "new-output", manifest)
    out = tmp_path / "output"
    assert M.output_guard(out, manifest) == out
    out.mkdir()
    with pytest.raises(M.SA.AuditError, match="existing"):
        M.output_guard(out, manifest)


def test_end_to_end_actual_checkpoint_io_readonly_finite_json_and_csv(tmp_path, monkeypatch):
    from experiments.s5_predictor_corrector import production_init as PI
    from experiments.s5_predictor_corrector import memory_pole_plots as plots
    from experiments.s5_predictor_corrector import memory_impulses as impulses
    roots, inputs = make_inputs(tmp_path)
    before = snapshot(roots)
    monkeypatch.setattr(PI, "production_init_flat", lambda arm, seed: params(arm))
    plotted = []
    monkeypatch.setattr(plots, "plot_report", lambda report, out: plotted.append(report))
    monkeypatch.setattr(impulses, "plot_impulse_report", lambda report, out: None)
    out = tmp_path / "out"
    args = []
    for label, directories in zip(("native", "original", "calibrated"), inputs):
        args += ["--" + label, *map(str, directories)]
    args += ["--out", str(out), "--expected-epochs", "3", "--all-epochs", "--impulse-horizon", "32"]
    report = M.main(args)
    assert len(plotted) == 1
    assert before == snapshot(roots)
    text = (out / "memory_poles.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    saved = json.loads(text)
    assert saved["reconstructed_ssm_init_agreement"] == {"301": True, "302": True, "303": True}
    assert len(saved["tasks"]) == 9
    assert len(saved["paired_layer_summaries"]) == 72
    assert len(saved["all_epoch_checkpoints"]) == 27
    assert sum(r["selected_for_validation"] for r in saved["all_epoch_checkpoints"]) == 9
    assert {r["epoch"] for r in saved["all_epoch_checkpoints"]} == {1, 2, 3}
    for row in saved["all_epoch_checkpoints"]:
        path = Path(row["analysis_file"])
        if not path.is_absolute():
            path = out / path
        assert path.is_file()
        payload = path.read_text()
        assert "NaN" not in payload and "Infinity" not in payload
        json.loads(payload)
        assert row["checkpoint_sha256"] == M.CC._sha256(row["checkpoint"])
    epochs = json.loads((out / "epoch_summaries.json").read_text())
    assert len(epochs) == 27
    assert all(len(row["layers"]) == 6 for row in epochs)
    with (out / "epoch_summaries.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 27 * 6
    assert all(t["epoch"] == 2 for t in report["tasks"])
    with (out / "poles.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 6 * 2 * (64 + 128 + 128)
    assert {r["epoch"] for r in rows if r["stage"] == "trained"} == {"2"}
    assert {r["epoch"] for r in rows if r["stage"] == "reconstructed_init"} == {"0"}
    native = [r for r in rows if r["arm"] == M.ARMS[0]]
    assert {r["pole"] for r in native} == {"dominant"}
    assert "not a saved epoch-0" in (out / "summary.txt").read_text()
    with pytest.raises(M.SA.AuditError, match="existing"):
        M.main(args)


def test_all_epochs_requires_unselected_historical_checkpoints(tmp_path):
    _, inputs = make_inputs(tmp_path)
    missing = inputs[2][2] / "checkpoint_epoch_01.msgpack"
    missing.unlink()
    args = []
    for label, directories in zip(("native", "original", "calibrated"), inputs):
        args += ["--" + label, *map(str, directories)]
    args += ["--out", str(tmp_path / "out"), "--expected-epochs", "3", "--all-epochs", "--impulse-horizon", "32"]
    with pytest.raises((M.SA.AuditError, M.CC.CheckpointError), match="missing|Missing"):
        M.main(args)
    assert not (tmp_path / "out").exists()


def test_all_epochs_detects_unselected_checkpoint_mutation_during_audit(tmp_path, monkeypatch):
    from experiments.s5_predictor_corrector import production_init as PI
    from experiments.s5_predictor_corrector import memory_pole_plots as plots
    from experiments.s5_predictor_corrector import memory_impulses as impulses
    _, inputs = make_inputs(tmp_path)
    monkeypatch.setattr(PI, "production_init_flat", lambda arm, seed: params(arm))
    old_checkpoint = inputs[1][1] / "checkpoint_epoch_01.msgpack"

    original = M.write_epoch_analyses

    def mutate_after_analysis(*args, **kwargs):
        original(*args, **kwargs)
        old_checkpoint.write_bytes(old_checkpoint.read_bytes() + b"changed")

    monkeypatch.setattr(M, "write_epoch_analyses", mutate_after_analysis)
    monkeypatch.setattr(plots, "plot_report", lambda report, out: None)
    monkeypatch.setattr(impulses, "plot_impulse_report", lambda report, out: None)
    args = []
    for label, directories in zip(("native", "original", "calibrated"), inputs):
        args += ["--" + label, *map(str, directories)]
    out = tmp_path / "out"
    args += ["--out", str(out), "--expected-epochs", "3", "--all-epochs", "--impulse-horizon", "32"]
    with pytest.raises((M.SA.AuditError, M.CC.CheckpointError), match="changed"):
        M.main(args)
    assert not (out / "memory_poles.json").exists()
