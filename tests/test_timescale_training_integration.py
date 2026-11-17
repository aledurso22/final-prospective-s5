"""Training integration of the timescale study (N / A / C / D), float32.

Run in its own process WITHOUT JAX_ENABLE_X64 (production precision; several
other test modules switch x64 on at import):

    python -m pytest tests/test_timescale_training_integration.py -q
"""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from flax.traverse_util import flatten_dict

from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full import step_telemetry as ST
from experiments.s5_three_arm_full.pole_diagnostics import (pole_summary,
                                                             radius_array)
from s5 import predictor_corrector as PC
from s5.discrete_recurrence import (companion_radius, generalized_coefficients,
                                    zucchet_coefficients)
from s5.ssm import discretize_zoh

STUDY = ("native_matched_s5", "generalized_prospective_s5",
         "calibrated_old_s5", "calibrated_prediction_s5")


@pytest.fixture(autouse=True)
def _require_float32():
    assert not jax.config.read("jax_enable_x64"), "run without JAX_ENABLE_X64"


@pytest.fixture(scope="module")
def states():
    return {arm: R.init_state(arm, 301) for arm in STUDY}


def _legacy_radius(state, arm):
    """The ORIGINAL runner.companion_spectral_radius, copied verbatim."""
    flat = flatten_dict(state.params)
    radii = []
    for key in flat:
        if key[-1] != "Lambda_re":
            continue
        prefix = key[:-1]
        lam_re, lam_im = flat[key], flat[prefix + ("Lambda_im",)]
        step = jnp.exp(flat[prefix + ("log_step",)][..., 0])
        lam = jnp.clip(lam_re, None, -1e-4) + 1j * lam_im
        b = jnp.zeros((lam.shape[0], 1), dtype=lam.dtype)
        abar, bbar = discretize_zoh(lam, b, step)
        if arm == "native_matched_s5":
            radii.append(jnp.max(jnp.abs(abar)))
            continue
        if arm == "zucchet_prospective_s5":
            raw = flat[prefix + ("prospective_T_raw",)]
            coeff = zucchet_coefficients(abar, bbar, jax.nn.softplus(raw))
        else:
            t = jax.nn.softplus(flat[prefix + ("generalized_T_raw",)])
            rho = 1e-4 + (1.0 - 1e-4) * jax.nn.sigmoid(
                flat[prefix + ("generalized_rho_raw",)])
            gamma = jax.nn.softplus(flat[prefix + ("generalized_gamma_raw",)])
            coeff = generalized_coefficients(abar, bbar, t, rho * gamma * t, gamma)
        radii.append(jnp.max(companion_radius(coeff[0], coeff[1])))
    return float(jnp.max(jnp.asarray(radii)))


def test_arms_registered_and_A_constructor_unchanged():
    for arm in ("calibrated_old_s5", "calibrated_prediction_s5"):
        assert arm in R.RUNNABLE_ARMS and arm in R.SCAN_ARMS
        assert R.recurrent_state_size(arm)["per_layer_real_values"] == 256
    assert R.ARM_ORDER == ("native_matched_s5", "zucchet_prospective_s5",
                           "generalized_prospective_s5")
    a = R.ssm_factory("generalized_prospective_s5")
    from s5.generalized_prospective_ssm import GeneralizedProspectiveS5SSM
    assert a.func is GeneralizedProspectiveS5SSM
    c, d = R.ssm_factory("calibrated_old_s5"), R.ssm_factory("calibrated_prediction_s5")
    assert (c.keywords["scale"], c.keywords["corrector"]) == ("calibrated", "old")
    assert (d.keywords["scale"], d.keywords["corrector"]) == ("calibrated", "prediction")
    assert R.BATCH_SIZE == 16


def test_C_D_initialization_is_A_initialization_exactly(states):
    """Same PRNG key, names, values, optimizer groups: a matched init."""
    pa = flatten_dict(states["generalized_prospective_s5"].params)
    for arm in ("calibrated_old_s5", "calibrated_prediction_s5"):
        pc = flatten_dict(states[arm].params)
        assert sorted(pa) == sorted(pc)
        for k in pa:
            assert onp.array_equal(onp.asarray(pa[k]), onp.asarray(pc[k])), k
            assert onp.asarray(pc[k]).dtype in (onp.float32, onp.int32), k
    # native shares the backbone of the very same draw
    pn = flatten_dict(states["native_matched_s5"].params)
    for k in pn:
        assert onp.array_equal(onp.asarray(pn[k]), onp.asarray(pa[k])), k


def test_radius_matches_original_code_for_existing_arms(states):
    for arm in ("native_matched_s5", "generalized_prospective_s5"):
        assert R.companion_spectral_radius(states[arm], arm) == pytest.approx(
            _legacy_radius(states[arm], arm), rel=1e-7)


def test_C_D_radius_uses_their_own_coefficients(states):
    for arm, variant in (("calibrated_old_s5", ("calibrated", "old")),
                         ("calibrated_prediction_s5", ("calibrated", "prediction"))):
        got = R.companion_spectral_radius(states[arm], arm)
        flat = flatten_dict(jax.device_get(states[arm].params))
        want = 0.0
        for k in flat:
            if k[-1] != "Lambda_re":
                continue
            p = k[:-1]
            lam = (onp.minimum(onp.float64(flat[k]), -1e-4)
                   + 1j * onp.float64(flat[p + ("Lambda_im",)]))
            lb = onp.exp(lam * onp.exp(onp.float64(flat[p + ("log_step",)][:, 0])))
            sp = lambda v: onp.logaddexp(0.0, onp.float64(v))
            T, g = sp(flat[p + ("generalized_T_raw",)]), sp(flat[p + ("generalized_gamma_raw",)])
            rho = 1e-4 + (1 - 1e-4) / (1 + onp.exp(-onp.float64(flat[p + ("generalized_rho_raw",)])))
            a1, a2, *_ = PC.scalar_coefficients(lb, T, rho * g * T, g, *variant)
            want = max(want, float(onp.max(onp.maximum(*map(abs, PC.roots(a1, a2))))))
        assert got == pytest.approx(want, rel=2e-6)
        # and it is NOT the generalized (legacy-scale) radius
        assert abs(got - _legacy_radius(states[arm], arm)) > 1e-5


def test_initial_pole_summaries(states):
    s = {arm: pole_summary(states[arm].params, arm) for arm in STUDY}
    assert s["native_matched_s5"]["n_modes"] == 384
    for arm in STUDY[1:]:
        assert s[arm]["stable"] == 384 and s[arm]["unstable"] == 0
    assert s["generalized_prospective_s5"]["eta_over_1_minus_mu"]["p50"] == pytest.approx(0.05, rel=1e-5)
    for arm in ("calibrated_old_s5", "calibrated_prediction_s5"):
        assert s[arm]["eta_over_1_minus_mu"]["p50"] == pytest.approx(1.0, rel=1e-6)
    # the timescale story at the production init: A ~18x slower, C ~native
    assert s["generalized_prospective_s5"]["slow_tau_over_native"]["p50"] > 15
    assert s["calibrated_old_s5"]["slow_tau_over_native"]["p50"] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("arm", STUDY)
def test_fused_telemetry_matches_original_and_is_pure(states, arm):
    st = states[arm]
    before = jax.device_get(st.params)
    fused = jax.device_get(ST.make_state_telemetry(arm)(st))
    legacy = ST.legacy_state_telemetry(st, arm)
    assert bool(fused[0]) == legacy[0] is True
    assert float(fused[1]) == pytest.approx(legacy[1], rel=1e-5)
    assert float(fused[2]) == pytest.approx(legacy[2], rel=1e-6)
    after = jax.device_get(st.params)
    for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after)):
        assert onp.array_equal(a, b)


def test_fused_telemetry_detects_nonfinite(states):
    st = states["calibrated_prediction_s5"]
    bad = st.replace(params=jax.tree_util.tree_map(
        lambda v: v.at[(0,) * v.ndim].set(jnp.nan) if v.dtype == jnp.float32 else v,
        st.params))
    finite, _, _ = jax.device_get(ST.make_state_telemetry("calibrated_prediction_s5")(bad))
    assert not bool(finite)
    s = ST.host_scalars(jnp.float32(1), jnp.float32(1), jnp.float32(1), jnp.bool_(True),
                        ST.make_state_telemetry("calibrated_prediction_s5")(bad))
    assert ST.is_nonfinite(s)


@pytest.mark.parametrize("arm", ["generalized_prospective_s5", "calibrated_prediction_s5"])
def test_fused_telemetry_is_cheaper(states, arm):
    m = ST.measure_overhead(states[arm], arm)
    assert m["values_agree"]
    assert m["fused_seconds"] < m["legacy_seconds"], m


def test_classify_separates_boundary_from_real_instability():
    from experiments.s5_three_arm_full.pole_diagnostics import BOUNDARY_TOL, classify
    rp = onp.array([0.99, 1 - BOUNDARY_TOL / 2, 0.9999999, 1 + BOUNDARY_TOL / 2, 1.2, 0.99999])
    r64 = onp.array([0.99, 1 - BOUNDARY_TOL / 2, 1.0000001, 1 + BOUNDARY_TOL / 2, 1.2, 0.98])
    ok, edge, bad = classify(rp, r64)
    assert ok.tolist() == [True, False, False, False, False, False]
    assert edge.tolist() == [False, True, True, True, False, True]
    assert bad.tolist() == [False, False, False, False, True, False]
    assert (ok.astype(int) + edge + bad == 1).all()


def test_production_precision_section_per_variant(states):
    for arm in STUDY:
        pp = pole_summary(states[arm].params, arm)["production_precision"]
        assert pp["dtype"] == "float32"
        assert pp["stable"] == 384 and pp["numerical_boundary"] == 0 and pp["unstable"] == 0
        assert pp["max_abs_radius_difference_production_vs_float64"] < 1e-6
    # A sits closest to the unit circle at init (legacy scale, 18x slower)
    a = pole_summary(states["generalized_prospective_s5"].params,
                     "generalized_prospective_s5")["production_precision"]
    c = pole_summary(states["calibrated_old_s5"].params,
                     "calibrated_old_s5")["production_precision"]
    assert a["min_margin_production"] < c["min_margin_production"] / 10
