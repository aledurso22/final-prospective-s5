"""Frozen selection validation, exact arm routing, and observable energy checks."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.s5_predictor_corrector import selected_checkpoint_audit as A


def make_run(root, epochs=3):
    root.mkdir()
    (root / "run_metadata.txt").write_text(
        f"epochs_requested={epochs}\nbatch_size=16\nauthoritative_commit=test\nscan_implementation=factored\n")
    for arm in A.ARMS:
        for seed in (301, 302):
            task = root / arm / str(seed)
            task.mkdir(parents=True)
            # Equal maximum accuracy: CE selects epoch 2; epoch 3 is an exact
            # tie, where the earlier epoch wins.
            vals = [{"accuracy": .7, "cross_entropy": .8, "n": 100},
                    {"accuracy": .8, "cross_entropy": .5, "n": 100},
                    {"accuracy": .8, "cross_entropy": .5, "n": 100}]
            with (task / "metrics.jsonl").open("w") as f:
                for i, v in enumerate(vals[:epochs], 1):
                    f.write(json.dumps({"epoch": i, "validation": v}) + "\n")
            (task / "checkpoint_epoch_02.msgpack").write_bytes(b"synthetic checkpoint")
            (task / "task_result.json").write_text(json.dumps({
                "code_identifier": arm, "seed": seed, "epochs_requested": epochs,
                "selected_epoch": 2, "selected_checkpoint": "/original/checkpoint_epoch_02.msgpack",
                "validation": vals[1]}))
    return root


def change_json(path, **changes):
    v = json.loads(path.read_text()); v.update(changes)
    path.write_text(json.dumps(v))


def test_manifest_preserves_recorded_selection_and_hashes(tmp_path):
    root = make_run(tmp_path / "run")
    m = A.build_manifest(root, expected_epochs=3)
    assert [(r["arm"], r["seed"]) for r in m["tasks"]] == [(a, s) for a in A.ARMS for s in (301, 302)]
    assert all(r["epoch"] == 2 and r["validation"]["correct"] == 80 for r in m["tasks"])
    assert all(len(r["task_result_sha256"]) == 64 for r in m["tasks"])
    A.assert_manifest_unchanged(m)
    (root / A.ARMS[0] / "301" / "checkpoint_epoch_02.msgpack").write_bytes(b"changed")
    with pytest.raises(A.AuditError, match="checkpoint changed"):
        A.assert_manifest_unchanged(m)


@pytest.mark.parametrize("problem", ["selection", "score", "missing_epoch", "duplicate_epoch", "identity", "failure", "missing_result", "missing_checkpoint"])
def test_refuses_incomplete_or_inconsistent_tasks(tmp_path, problem):
    root = make_run(tmp_path / "run")
    t = root / A.ARMS[0] / "301"
    result = t / "task_result.json"
    if problem == "selection":
        change_json(result, selected_epoch=3)
    elif problem == "score":
        change_json(result, validation={"accuracy": .9, "cross_entropy": .5, "n": 100})
    elif problem == "missing_epoch":
        (t / "metrics.jsonl").write_text((t / "metrics.jsonl").read_text().splitlines()[0] + "\n")
    elif problem == "duplicate_epoch":
        with (t / "metrics.jsonl").open("a") as f:
            f.write((t / "metrics.jsonl").read_text().splitlines()[0] + "\n")
    elif problem == "identity":
        change_json(result, code_identifier=A.ARMS[1])
    elif problem == "failure":
        (t / "failure.json").write_text("{}")
    elif problem == "missing_result":
        result.unlink()
    elif problem == "missing_checkpoint":
        (t / "checkpoint_epoch_02.msgpack").unlink()
    with pytest.raises(A.AuditError):
        A.build_manifest(root, expected_epochs=3)


def test_report_ranking_and_output_guards_do_not_mutate_run(tmp_path):
    root = make_run(tmp_path / "run")
    t = root / A.ARMS[3] / "301"
    metrics = [json.loads(l) for l in (t / "metrics.jsonl").read_text().splitlines()]
    metrics[1]["validation"]["accuracy"] = .83
    (t / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    change_json(t / "task_result.json", validation=metrics[1]["validation"])
    before = {str(p): A.CC._sha256(p) for p in root.rglob("*") if p.is_file()}
    out = tmp_path / "audit"
    r = A.main(["--run-root", str(root), "--expected-epochs", "3", "--out", str(out), "--manifest-only"])
    assert r["performance"]["observed_ranking"][0]["arm"] == A.ARMS[3]
    pair = next(x for x in r["performance"]["paired_differences"] if x["left"] == A.ARMS[0] and x["right"] == A.ARMS[3])
    assert pair["mean_difference_pp"] == pytest.approx(-1.5)
    assert json.loads((out / "manifest.json").read_text())["tasks"][0]["epoch"] == 2
    assert before == {str(p): A.CC._sha256(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(A.AuditError, match="existing output"):
        A.main(["--run-root", str(root), "--out", str(out), "--manifest-only"])
    with pytest.raises(A.AuditError, match="outside"):
        A.main(["--run-root", str(root), "--out", str(root / "audit"), "--manifest-only"])


def group():
    P, H = 2, 3
    return {"Lambda_re": np.array([-.5, -.5], np.float32),
            "Lambda_im": np.array([1., 8.], np.float32),
            "log_step": np.log(np.array([[.02], [.1]], np.float32)),
            "B": np.ones((P, H, 2), np.float32),
            "C1": np.ones((H, P, 2), np.float32),
            "C2": np.ones((H, P, 2), np.float32),
            "generalized_T_raw": np.full(P, np.log(np.expm1(.05)), np.float32),
            "generalized_gamma_raw": np.full(P, np.log(np.expm1(1.)), np.float32),
            "generalized_rho_raw": np.zeros(P, np.float32)}


def test_actual_arm_coefficient_routing_against_literal_updates():
    rng = np.random.default_rng(3)
    g = group()
    s1, s2, xt, xp = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    seen = []
    for arm in A.ARMS:
        m, co, h = A.coefficients_for_arm(arm, g)
        a1, a2, g1, g2 = co
        actual = a1 * s1 + a2 * s2 + g1 * xt + g2 * xp
        if arm == A.ARMS[0]:
            expected = m["lambda_bar"] * s1 + xt
        else:
            T, M, gamma = h["T"], h["M"], h["gamma"]
            Q = M + gamma + T
            k = T if arm == A.ARMS[1] else gamma
            mu, eta, zeta = (M + T) / Q, k / Q, k * T / Q
            K = 1 - m["lambda_bar"]
            d = mu * (s1 - s2) - zeta * ((K * s1 - xt) - (K * s2 - xp))
            y = s1 + d
            expected = y - eta * (K * (y if arm == A.ARMS[3] else s1) - xt)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        seen.append(actual)
    assert all(not np.allclose(seen[i], seen[j]) for i in range(4) for j in range(i))


def direct_impulse(co, length=1000):
    a1, a2, g1, g2 = [complex(v[0]) for v in co]
    h = np.zeros(length, complex)
    h[0] = g1
    h[1] = a1 * h[0] + g2
    for t in range(2, length):
        h[t] = a1 * h[t - 1] + a2 * h[t - 2]
    return h


@pytest.mark.parametrize("repeated", [False, True])
def test_observable_total_energy_including_repeated_roots(repeated):
    p1, p2 = (.7, .7) if repeated else (.8 + .05j, .3 - .1j)
    co = tuple(np.array([x], complex) for x in (p1 + p2, -p1 * p2, .4, .1))
    r = A.response_analysis(co)
    assert r["stable"].all()
    np.testing.assert_allclose(r["total_impulse_energy"][0], np.sum(abs(direct_impulse(co)) ** 2), rtol=5e-13)
    if repeated:
        assert not r["residue_resolved"][0]
        assert np.isnan(r["fast_component_energy"][0])
    else:
        t = np.arange(1000)
        from_residues = r["residue_slow"][0] * r["slow_pole"][0] ** t + r["residue_fast"][0] * r["fast_pole"][0] ** t
        np.testing.assert_allclose(from_residues, direct_impulse(co), rtol=1e-12, atol=1e-14)


def test_unstable_energy_is_absent_and_json_finite():
    co = tuple(np.array([x], complex) for x in (1.6, -.55, .4, .1))  # roots 1.1, .5
    r = A.response_analysis(co)
    assert not r["stable"][0]
    assert np.isnan(r["total_impulse_energy"][0])
    assert np.isnan(r["fast_component_energy"][0])
    text = json.dumps(A._jsonable(r), allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_selected_parameter_analysis_labels_actual_formula(tmp_path):
    from flax import serialization
    g = group()
    for arm in A.ARMS:
        params = dict(g)
        if arm == A.ARMS[0]:
            params = {k: v for k, v in params.items() if not k.startswith("generalized_")}
        p = tmp_path / (arm + ".msgpack")
        p.write_bytes(serialization.msgpack_serialize({"params": {"layer": {"seq": params}}}))
        report = A.analyze_checkpoint({"arm": arm, "seed": 301, "epoch": 14, "checkpoint": str(p)})
        assert report["actual_trained_arm"]
        assert report["summary"]["coefficient_formula"] == ("native" if arm == A.ARMS[0] else list(A.VARIANTS[arm]))
        assert report["summary"]["n_modes"] == 2
        json.dumps(A._jsonable(report), allow_nan=False)
