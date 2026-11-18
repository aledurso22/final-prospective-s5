"""Distribution masks and end-to-end rendering, including undefined decay times."""

import copy

import numpy as np
import pytest

from experiments.s5_predictor_corrector import memory_pole_plots as P


def report_fixture():
    tasks = []
    for arm, _, _ in P.ARMS:
        poles = 1 if arm == P.ARMS[0][0] else 2
        stages = {}
        for stage in P.STAGES:
            layers = []
            for i in range(6):
                radius = np.array([[.5, .2], [1.00001, .1], [1., 0.]])[:, :poles]
                decay = np.full_like(radius, np.nan)
                good = (radius > 0) & (radius < 1)
                decay[good] = -1 / np.log(radius[good])
                layers.append({"layer": f"encoder/layers_{i}/seq", "roots": radius.astype(complex),
                               "radius": radius, "decay_samples": decay,
                               "summary": {"n_channels": 3, "channels_ge_one": 2,
                                           "n_poles": 3 * poles, "poles_ge_one": 2}})
            stages[stage] = {"layers": layers}
        tasks.append({"arm": arm, "seed": 301, "epoch": 14, "stages": stages})
    return {"tasks": tasks}


def test_per_pole_masks_keep_stable_secondary_when_dominant_unstable():
    layer = report_fixture()["tasks"][1]["stages"]["trained"]["layers"][0]
    dominant, excluded_d = P._decay_values(layer, 0)
    secondary, excluded_s = P._decay_values(layer, 1)
    assert len(dominant) == 1 and excluded_d == 2
    assert len(secondary) == 2 and excluded_s == 1
    np.testing.assert_allclose(secondary, np.sort(-1 / np.log([.2, .1])))


def test_invalid_radius_never_becomes_valid_by_supplied_positive_decay():
    layer = {"radius": np.array([[0.], [1.], [1.01], [np.nan], [.9]]),
             "decay_samples": np.ones((5, 1))}
    values, excluded = P._decay_values(layer, 0)
    assert values.tolist() == [1.] and excluded == 4


def test_six_layer_figures_render_and_refuse_overwrite(tmp_path):
    paths = P.plot_report(report_fixture(), tmp_path)
    assert len(paths) == 2
    assert (tmp_path / "pole_memory_seed_301.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "pole_memory_seed_301.pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="overwrite"):
        P.plot_report(report_fixture(), tmp_path)


def test_all_undefined_timescales_still_render(tmp_path):
    report = report_fixture()
    for task in report["tasks"]:
        for stage in task["stages"].values():
            for layer in stage["layers"]:
                layer["radius"][:] = 1.
                layer["decay_samples"][:] = np.nan
    P.plot_report(report, tmp_path)
    assert (tmp_path / "pole_memory_seed_301.pdf").stat().st_size > 1000


@pytest.mark.parametrize("problem", ["duplicate", "missing_arm", "wrong_layers", "native_auxiliary"])
def test_structural_refusals(tmp_path, problem):
    report = copy.deepcopy(report_fixture())
    if problem == "duplicate":
        report["tasks"].append(report["tasks"][0])
    elif problem == "missing_arm":
        report["tasks"].pop()
    elif problem == "wrong_layers":
        report["tasks"][0]["stages"]["trained"]["layers"].pop()
    else:
        layer = report["tasks"][0]["stages"]["trained"]["layers"][0]
        layer["radius"] = np.repeat(layer["radius"], 2, axis=1)
        layer["decay_samples"] = np.repeat(layer["decay_samples"], 2, axis=1)
    with pytest.raises(ValueError):
        P.plot_report(report, tmp_path)
