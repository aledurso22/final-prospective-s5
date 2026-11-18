"""Independent dense-kernel and literal-recurrence oracles for impulse auditing."""

import numpy as np
import pytest

from experiments.s5_predictor_corrector import memory_impulses as M


def packed(value):
    return np.stack([value.real, value.imag], axis=-1).astype(np.float32)


def group(p=3, inputs=2, outputs=4):
    rng = np.random.default_rng(15)
    return {"Lambda_re": np.full(p, -.5, np.float32),
            "Lambda_im": np.linspace(0, 2, p, dtype=np.float32),
            "log_step": np.full((p, 1), np.log(.1), np.float32),
            "B": packed(rng.normal(size=(p, inputs)) + 1j * rng.normal(size=(p, inputs))),
            "C1": packed(rng.normal(size=(outputs, p)) + 1j * rng.normal(size=(outputs, p))),
            "C2": packed(rng.normal(size=(outputs, p)) + 1j * rng.normal(size=(outputs, p)))}


@pytest.mark.parametrize("repeated", [False, True])
def test_direct_impulse_matches_literal_recurrence(repeated):
    roots = (.7, .7) if repeated else (.8 + .1j, .2 - .1j)
    a1, a2 = sum(roots), -np.prod(roots)
    co = [np.array([v]) for v in (a1, a2, .4, .12)]
    expected = np.zeros(31, complex)
    for t in range(31):
        expected[t] = (a1 * expected[t-1] if t else 0) + (a2 * expected[t-2] if t > 1 else 0)
        expected[t] += .4 if t == 0 else .12 if t == 1 else 0
    np.testing.assert_allclose(M._impulse_sequences(co, 31)[:, 0], expected, atol=1e-15)


def test_gram_energy_matches_explicit_real_dense_kernel_and_mode_interference():
    rng = np.random.default_rng(83)
    h = rng.normal(size=(17, 3)) + 1j * rng.normal(size=(17, 3))
    B = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    C = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    kernel = 2 * np.einsum("op,tp,pi->toi", C, h, B).real
    energy, audit, G, H = M._real_kernel_energy(h, B, C)
    np.testing.assert_allclose(energy, (kernel ** 2).sum(axis=(1, 2)), rtol=3e-15, atol=2e-13)
    assert audit["significant_negative_values"] == 0
    mode_total, _ = M._mode_energy(h, G, H)
    for p in range(3):
        mode_kernel = 2 * np.einsum("o,t,i->toi", C[:, p], h[:, p], B[p]).real
        assert mode_total[p] == pytest.approx((mode_kernel ** 2).sum(), rel=2e-15)
    assert not np.isclose(mode_total.sum(), energy.sum())


def test_full_report_preserves_input_and_readout_coupling():
    g = group()
    co = [np.full(3, value, complex) for value in (.95, -.12, .4, -.08)]
    report = M.analyze_impulses("calibrated_old_s5", g, co, horizon=48)
    h = M._impulse_sequences(co, 48)
    q = M.CC.mode_quantities(g)
    B = M.CC._complex(g["B"]) * ((q["lambda_bar"] - 1) / q["Lambda"])[:, None]
    totals = []
    for direction, key in (("forward", "C1"), ("backward", "C2")):
        C = M.CC._complex(g[key])
        kernel = 2 * np.einsum("op,tp,pi->toi", C, h, B).real
        expected = np.sum(kernel ** 2, axis=(1, 2))
        observed = report["directions"][direction]
        np.testing.assert_allclose(observed["curve_energy"], expected[report["lag_samples"]], rtol=1e-13, atol=1e-15)
        assert observed["summary"]["total_energy"] == pytest.approx(expected.sum())
        assert observed["summary"]["energy_quantile_lags"]["90"] == np.searchsorted(expected.cumsum(), .9 * expected.sum())
        totals.append(expected.sum())
    assert report["branch_sum"]["summary"]["total_energy"] == pytest.approx(sum(totals))


def test_pole_cancellation_and_repeated_root_have_different_meanings():
    g = group(p=1)
    co = [np.array([v]) for v in (1.1, -.24, 1., -.3)]  # roots .8,.3; secondary cancelled
    r = M.analyze_impulses("calibrated_old_s5", g, co, horizon=64)
    assert r["residue_diagnostic"]["secondary_residue_negligible"].all()
    assert r["directions"]["forward"]["components"]["secondary"]["mode_energy"][0] < 1e-27
    co = [np.array([v]) for v in (1.4, -.49, 1., .1)]
    r = M.analyze_impulses("calibrated_old_s5", g, co, horizon=64)
    assert not r["residue_diagnostic"]["resolved"].any()
    assert np.isnan(r["directions"]["forward"]["components"]["secondary"]["mode_energy"]).all()
    assert r["directions"]["forward"]["summary"]["status"] == "finite"


def test_real_readout_cancellation_zero_pole_and_growing_finite_window():
    g = group(p=1, inputs=1, outputs=1)
    g["B"] = packed(np.ones((1, 1), complex))
    g["C1"] = packed(np.full((1, 1), 1j))
    g["C2"] = packed(np.ones((1, 1), complex))
    co = [np.array([v]) for v in (0., 0., 1., 0.)]
    r = M.analyze_impulses(M.NATIVE, g, co, horizon=32)
    assert r["directions"]["forward"]["summary"]["status"] == "zero_response"
    assert r["directions"]["backward"]["summary"]["energy_quantile_lags"]["99"] == 0
    assert "secondary" not in r["directions"]["forward"]["components"]
    co[0] = np.array([1.01])
    r = M.analyze_impulses(M.NATIVE, g, co, horizon=32)
    assert r["directions"]["backward"]["summary"]["status"] == "finite"
    assert r["directions"]["backward"]["summary"]["fraction_energy_at_or_after_lag"]["16"] > .5


def test_negative_energy_guard_preserves_significant_failures():
    cleaned, audit = M._sanitize_energy(np.array([-1e-16, -1e-3, 2.]), np.ones(3))
    np.testing.assert_equal(cleaned, [0., -1e-3, 2.])
    assert audit["roundoff_negative_values_set_to_zero"] == 1
    assert audit["significant_negative_values"] == 1
    assert M._energy_summary(cleaned)["status"] == "significant_negative_energy"


def test_bilateral_kernel_includes_zero_lag_direction_cancellation():
    g = group(p=1, inputs=1, outputs=1)
    g["B"] = packed(np.ones((1, 1), complex))
    g["C1"] = packed(np.ones((1, 1), complex))
    g["C2"] = packed(-np.ones((1, 1), complex))
    co = [np.array([v]) for v in (0., 0., 1., 0.)]
    r = M.analyze_impulses(M.NATIVE, g, co, horizon=32)
    assert r["branch_sum"]["summary"]["total_energy"] > 0
    assert r["bidirectional_recurrent"]["summary"]["total_energy"] == 0
    assert r["bidirectional_recurrent"]["signed_lag_range"] == [-31, 31]


def test_nonfinite_response_is_flagged_not_hidden():
    g = group(p=1)
    co = [np.array([v]) for v in (1e10, 0., 1., 0.)]
    r = M.analyze_impulses(M.NATIVE, g, co, horizon=100)
    assert r["direct_response_nonfinite_values"] > 0
    assert r["branch_sum"]["summary"]["status"] == "nonfinite_response"


def test_plotting_consumes_full_report(tmp_path):
    from tests.test_memory_pole_plots import report_fixture
    report = report_fixture()
    g = group()
    for task in report["tasks"]:
        co = [np.full(3, v) for v in ((.8, 0., 1., 0.) if task["arm"] == M.NATIVE else (1.1, -.24, 1., -.2))]
        impulse = M.analyze_impulses(task["arm"], g, co, horizon=32)
        for stage in task["stages"].values():
            for layer in stage["layers"]:
                layer["impulse_response"] = impulse
    paths = M.plot_impulse_report(report, tmp_path)
    assert len(paths) == 2
    assert (tmp_path / "observable_impulse_seed_301.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "observable_impulse_seed_301.pdf").read_bytes().startswith(b"%PDF")
