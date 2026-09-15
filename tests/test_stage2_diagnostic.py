"""Focused validation for the Stage 2 diagnostic. CLUSTER-ONLY.

Run by `bin/run_experiments/cluster_stage2_diagnostic.sh` inside the same
20-minute budget as the diagnostic itself. Not run locally: the governing brief
requires all numerical checks on the cluster.

Covers the coordinator's review of 8122ddd:
  R1  trained raw poles, not initializer fields; continuous vs aliased poles
  R2  the invalid geometric tail bound is gone; the exact window remainder
      identity holds, including on a nonnormal block
  R3  failed checks propagate to status and exit code
  R5  "core" is used for the linear recurrent core, never the whole layer
"""

import os
import sys

import jax
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax.scipy.linalg import block_diag                           # noqa: E402
from s5 import substrate_diagnostics as SD                        # noqa: E402
from s5.rawat_s5 import ARMS, SubstrateSSM, init_substrate_ssm    # noqa: E402
from s5.ssm import init_S5SSM                                     # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                           # noqa: E402

IMPULSE_TOL = 1e-10
REMAINDER_TOL = 1e-10   # RELATIVE, float64 identity
BANDS = ((0, 0), (1, 4), (5, 16), (17, 64), (65, 160), (161, 511))


def ssm_kw(H=4, ssm=8, blocks=2, dt_min=0.001, dt_max=0.1):
    blk = ssm // blocks
    Lam, _, _, V, _ = make_DPLR_HiPPO(blk)
    blk //= 2
    P = ssm // 2
    Lam, V = Lam[:blk], V[:, :blk]
    Vc = V.conj().T
    Lam = (Lam * np.ones((blocks, blk))).ravel()
    return dict(H=H, P=P, Lambda_re_init=Lam.real, Lambda_im_init=Lam.imag,
                V=block_diag(*([V] * blocks)), Vinv=block_diag(*([Vc] * blocks)),
                C_init="trunc_standard_normal", discretization="zoh",
                dt_min=dt_min, dt_max=dt_max, conj_sym=True,
                bidirectional=False)


class _Wrapper:
    """Exposes the `encoder.layers[i].seq` path `read_core` expects."""

    def __init__(self, seq):
        lyr = type("L", (), {"seq": seq})()
        self.encoder = type("E", (), {"layers": [lyr]})()


def _module(arm, H=4, **kw):
    mod = init_substrate_ssm(arm, **ssm_kw(H=H, **kw))()
    x = np.zeros((8, H))
    return mod, mod.init(jax.random.PRNGKey(0), x), x


def _core(mod, v, x):
    """read_core against a single bound SSM, via the same attribute path."""
    def read(m):
        Lam, B_c, Delta = m._native()
        B_tilde = m.B[..., 0] + 1j * m.B[..., 1]
        return dict(response=m.response, input_gain=m.input_gain,
                    clip_eigs=m.clip_eigs, conj_sym=m.conj_sym, P=m.P, H=m.H,
                    Lambda_clipped=Lam,
                    # R1: the TRAINED parameters, not the initializer fields
                    Lambda_raw_param=m.Lambda_re + 1j * m.Lambda_im,
                    Lambda_initializer_field=(m.Lambda_re_init
                                              + 1j * m.Lambda_im_init),
                    B_tilde=B_tilde, B_c=B_c, Delta=Delta,
                    a=Lam * Delta, b=Delta[:, None] * B_c,
                    C_tilde=m.C_tilde, D=m.D, coefficients=m.coefficients(),
                    physical=dict(T=m.physical.T, gamma=m.physical.gamma,
                                  rho=m.physical.rho, mass=m.physical.mass))
    return mod.apply(v, method=read)


# ----------------------------------------------------------------- R1 -----
@pytest.mark.parametrize("arm", ["gain_clip_s5", "gp_fixed_m0",
                                 "gp_fixed_mass"])
def test_trained_raw_poles_are_read_not_initializer_fields(arm):
    """R1: move BOTH raw parts away from initialization, with one real part
    ABOVE the clipping boundary, and check raw, clipped, the clipping count and
    the coefficients separately. A reader of `Lambda_re_init` cannot pass this.
    """
    mod, v, x = _module(arm)
    p = dict(v["params"])
    re = onp.asarray(p["Lambda_re"]).copy()
    im = onp.asarray(p["Lambda_im"]).copy()
    re_init, im_init = re.copy(), im.copy()
    re[0] = +0.5                       # ABOVE the clip boundary -1e-4
    re[1] = re[1] - 0.25               # still stable, moved
    im[0] = im[0] + 1.75               # imaginary part moved too
    p["Lambda_re"] = np.asarray(re)
    p["Lambda_im"] = np.asarray(im)
    v2 = dict(v); v2["params"] = p
    core = _core(mod, v2, x)

    raw = onp.asarray(core["Lambda_raw_param"])
    init = onp.asarray(core["Lambda_initializer_field"])
    clipped = onp.asarray(core["Lambda_clipped"])
    assert raw[0].real == pytest.approx(0.5)
    assert raw[0].imag == pytest.approx(im[0])
    assert float(onp.max(onp.abs(raw - init))) > 0.1        # genuinely moved
    assert float(onp.max(onp.abs(init - (re_init + 1j * im_init)))) == 0.0
    # the EXECUTED pole is clipped, the trained raw one is not
    assert clipped[0].real == pytest.approx(-1e-4)
    summary = SD.pole_summary(core)
    assert summary["n_raw_poles_clipped"] == 1
    assert summary["clip_active"] is True
    assert summary["trained_minus_initializer_max"] > 0.1
    assert summary["raw_minus_clipped_max"] > 0.4
    # coefficients follow the CLIPPED pole
    assert float(onp.max(onp.abs(onp.asarray(
        SD.discrete_poles(core))))) < 1.0


def test_continuous_poles_are_not_the_log_of_the_discrete_eigenvalue():
    """R1: a continuous frequency beyond pi per unit interval is ALIASED by the
    discrete angle. The exported continuous pole must keep the true value, and
    the aliasing must be flagged rather than relabelled."""
    mod, v, x = _module("gain_clip_s5", dt_min=0.5, dt_max=0.9)
    p = dict(v["params"])
    im = onp.asarray(p["Lambda_im"]).copy()
    im[0] = 9.0                       # with Delta ~ 0.5-0.9 this exceeds pi
    p["Lambda_im"] = np.asarray(im)
    v2 = dict(v); v2["params"] = p
    core = _core(mod, v2, x)
    cont = SD.continuous_poles(core)
    disc = SD.discrete_poles(core)
    k = int(onp.argmax(onp.abs(onp.imag(cont))))
    assert abs(onp.imag(cont)[k]) > onp.pi, "fixture must exceed Nyquist"
    aliased = onp.angle(disc)[k]
    assert abs(aliased) <= onp.pi + 1e-12
    assert abs(abs(onp.imag(cont)[k]) - abs(aliased)) > 1e-3
    s = SD.pole_summary(core)
    assert s["n_continuous_modes_aliased_by_discrete_angle"] >= 1
    assert s["discrete_angle_is_aliased_frequency"] is True
    assert abs(s["continuous_pole_im"][k]) > onp.pi


# ----------------------------------------------------------------- R2 -----
def test_the_invalid_geometric_tail_bound_is_gone():
    assert not hasattr(SD, "truncation_tail_bound")


def test_cancellation_counterexample_defeats_the_old_bound():
    """R2, analytical: K_l = (1/2)^l - 2(1/4)^l has K_1 = 0 but K_2 = 1/8, so
    ||K_last|| * rho/(1-rho) returns a ZERO tail for a nonzero one. Kept as an
    executable record of why the bound was removed rather than loosened."""
    l = onp.arange(8)
    K = (0.5) ** l - 2 * (0.25) ** l
    assert K[1] == pytest.approx(0.0, abs=1e-15)
    assert K[2] == pytest.approx(0.125)
    rho = 0.5
    old_bound = abs(K[1]) * rho / (1 - rho)         # window 0..1
    true_tail = float(onp.sum(onp.abs(K[2:])))
    assert old_bound == pytest.approx(0.0)
    assert true_tail > 0.1                           # bound understates it


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_exact_window_remainder_closes_the_finite_dft(arm):
    """R2 repair: H(w) == finite DFT over 0..N-1 PLUS the closed-form
    remainder. This is an identity, so it is a real check rather than an
    estimate, and it holds for the NONNORMAL 2x2 blocks of the mass arm."""
    H, N = 3, 64
    mod, v, x = _module(arm, H=H)
    core = _core(mod, v, x)
    w, Hf = SD.frequency_response(core, n_freq=17)
    K = SD.impulse_matrices(core, N)
    rem = SD.frequency_window_remainder(core, N, w)
    lags = onp.arange(N)
    dft = onp.zeros_like(Hf)
    for k, wk in enumerate(w):
        dft[k] = onp.tensordot(onp.exp(-1j * wk * lags), K, axes=(0, 0))
    # RELATIVE: at dt_min = 1e-3 the resolvent has denominators of order 1e-4,
    # so |H| itself is large and an absolute tolerance would test magnitude
    # rather than the identity.
    err = float(onp.max(onp.abs(Hf - (dft + rem))))
    scale = max(float(onp.max(onp.abs(Hf))), 1e-12)
    assert err / scale < REMAINDER_TOL, (arm, err, scale, err / scale)


def test_mass_block_is_nonnormal_so_the_remainder_test_is_not_vacuous():
    mod, v, x = _module("gp_fixed_mass")
    core = _core(mod, v, x)
    A = onp.asarray(core["coefficients"]["A_bar"])
    dev = max(float(onp.max(onp.abs(A[p] @ A[p].conj().T
                                    - A[p].conj().T @ A[p])))
              for p in range(A.shape[0]))
    assert dev > 1e-6, "fixture must contain a genuinely nonnormal block"


def test_band_energies_are_windowed_and_absolute_as_well_as_fractional():
    K = onp.zeros((512, 2, 2)); K[0] = 1.0; K[200] = 2.0
    r = SD.band_energy(K, BANDS)
    by = {(b["lo"], b["hi"]): b for b in r["bands"]}
    assert by[(0, 0)]["energy"] == pytest.approx(4.0)
    assert by[(161, 511)]["energy"] == pytest.approx(16.0)
    assert all("energy" in b and "fraction" in b for b in r["bands"])


# ----------------------------------------------------------------- core ----
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_impulse_identities_equal_the_executed_core(arm):
    H, L = 4, 40
    mod, v, _ = _module(arm, H=H)
    core = _core(mod, v, np.zeros((L, H)))
    K = SD.impulse_matrices(core, L)
    for h in range(H):
        x = onp.zeros((L, H)); x[0, h] = 1.0
        y = onp.asarray(mod.apply(v, np.asarray(x)))
        assert float(onp.max(onp.abs(y - K[:, :, h]))) < IMPULSE_TOL, (arm, h)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_superposition_holds_for_the_linear_recurrent_CORE(arm):
    """R5 wording: this validates the linear recurrent CORE, not the whole
    nonlinear residual SequenceLayer."""
    H, L = 4, 40
    mod, v, _ = _module(arm, H=H)
    core = _core(mod, v, np.zeros((L, H)))
    K = SD.impulse_matrices(core, L)
    x = onp.asarray(onp.random.RandomState(0).randn(L, H))
    y = onp.asarray(mod.apply(v, np.asarray(x)))
    conv = onp.zeros_like(y)
    for l in range(L):
        for m in range(l + 1):
            conv[l] += K[m] @ x[l - m]
    assert float(onp.max(onp.abs(y - conv))) < 1e-9, arm


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_two_re_of_the_complex_transfer_is_NOT_the_real_response(arm):
    H = 3
    mod, v, _ = _module(arm, H=H)
    core = _core(mod, v, np.zeros((8, H)))
    w, Hf = SD.frequency_response(core, n_freq=33)
    c = core["coefficients"]
    C = onp.asarray(core["C_tilde"]).astype(onp.complex128)
    if core["response"] in ("one_tap", "alpha_p_two_tap"):
        A = onp.asarray(c["A_bar"]).astype(onp.complex128)
        B = onp.asarray(c["B_bar"]).astype(onp.complex128)
    elif core["response"] == "gp_fixed_m0":
        A = onp.asarray(c["a_bar"]).astype(onp.complex128)
        B = onp.asarray(c["b_bar"]).astype(onp.complex128)
    else:
        pytest.skip("block realization: the scalar shortcut does not apply")
    naive = onp.zeros_like(Hf)
    for k, wk in enumerate(w):
        u = onp.exp(-1j * wk)
        naive[k] = 2 * onp.real(C @ (B / (1.0 - A * u)[:, None]))
    assert onp.max(onp.abs(Hf[1:-1] - naive[1:-1])) > 1e-8


def test_native_arm_is_bit_identical_to_upstream_s5():
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(13).randn(9, kw["H"]))
    k = jax.random.PRNGKey(0)
    base = init_S5SSM(clip_eigs=False, **kw)()
    sub = init_substrate_ssm("native_s5", **kw)()
    vb, vs = base.init(k, x), sub.init(k, x)
    assert float(onp.max(onp.abs(base.apply(vb, x) - sub.apply(vs, x)))) == 0.0


def test_adapter_refuses_an_unknown_response():
    mod, v, _ = _module("native_s5")
    bad = dict(_core(mod, v, np.zeros((8, 4))), response="something_else")
    with pytest.raises(ValueError):
        SD.impulse_matrices(bad, 4)
    with pytest.raises(ValueError):
        SD.frequency_response(bad, n_freq=8)
    with pytest.raises(ValueError):
        SD.frequency_window_remainder(bad, 8, onp.array([0.0]))


def test_old_gp_diagnostics_would_have_dispatched_wrongly():
    assert not hasattr(SubstrateSSM, "mechanism")
    assert "response" in SubstrateSSM.__annotations__


@pytest.mark.parametrize("arm", ["alpha_p_s5", "gp_fixed_m0", "gp_fixed_mass"])
def test_counterfactual_one_tap_differs_from_the_executed_law(arm):
    H, L = 4, 64
    mod, v, _ = _module(arm, H=H)
    core = _core(mod, v, np.zeros((L, H)))
    assert float(onp.max(onp.abs(SD.impulse_matrices(core, L)
                                 - SD.counterfactual_one_tap(core, L)))) > 1e-6


def test_counterfactual_reproduces_a_one_tap_core_exactly():
    H, L = 4, 64
    mod, v, _ = _module("gain_clip_s5", H=H)
    core = _core(mod, v, np.zeros((L, H)))
    assert float(onp.max(onp.abs(SD.impulse_matrices(core, L)
                                 - SD.counterfactual_one_tap(core, L)))) < 1e-9


# ----------------------------------------------------------------- R3 -----
def _status(tmp_path):
    from experiments.gp.stage2_diagnostic import Status
    return Status(str(tmp_path))


def test_failed_required_check_forces_FAILED_and_nonzero_exit(tmp_path):
    from experiments.gp.stage2_diagnostic import EXIT_CODES
    st = _status(tmp_path)
    st.record("restore_counts", False, arm="gp_fixed_mass")
    assert st.final(hashes_ok=True, budget_incomplete=False) == "FAILED"
    assert EXIT_CODES["FAILED"] != 0
    assert st.blocked("gp_fixed_mass")


def test_optional_failure_does_not_fail_the_run(tmp_path):
    st = _status(tmp_path)
    st.record("optional_thing", False, required=False, arm="native_s5")
    assert st.final(hashes_ok=True, budget_incomplete=False) == "PASS"
    assert not st.blocked("native_s5")


def test_changed_source_hashes_force_FAILED(tmp_path):
    st = _status(tmp_path)
    st.record("F_source_files_unchanged", False)
    assert st.final(hashes_ok=False, budget_incomplete=False) == "FAILED"


def test_unknown_hash_state_is_INCOMPLETE_not_pass(tmp_path):
    st = _status(tmp_path)
    assert st.final(hashes_ok=None, budget_incomplete=False) == "INCOMPLETE"


def test_budget_incomplete_is_INCOMPLETE_with_nonzero_exit(tmp_path):
    from experiments.gp.stage2_diagnostic import EXIT_CODES
    st = _status(tmp_path)
    assert st.final(hashes_ok=True, budget_incomplete=True) == "INCOMPLETE"
    assert EXIT_CODES["INCOMPLETE"] != 0


def test_skipped_check_is_recorded_and_not_silently_passed(tmp_path):
    st = _status(tmp_path)
    st.skip("D2_sensitivity", "budget exhausted", arm="native_s5")
    assert st.final(hashes_ok=True, budget_incomplete=False) == "INCOMPLETE"
    assert st.unexecuted and st.unexecuted[0]["status"] == "not_executed"


def test_checks_are_persisted_immediately(tmp_path):
    import json
    st = _status(tmp_path)
    st.record("something", True)
    with open(os.path.join(str(tmp_path), "checks.json")) as fh:
        assert json.load(fh)[0]["check"] == "something"


def test_budget_accepts_an_absolute_deadline_from_the_launcher():
    import time as _t
    from experiments.gp.stage2_diagnostic import Budget
    b = Budget(1200.0, deadline=_t.time() + 30.0)
    assert b.limit <= 30.5 and b.limit > 25.0
    assert not b.incomplete
    b.mark("x", "skipped_budget")
    assert b.incomplete is True


# ------------------------------------------------- offset helper (review 2) --
# The external-input FD test does NOT validate the offset helper: it
# differentiates a different function. These checks target the helper itself.

def _tiny_loaded(arm="gp_fixed_mass", d_model=16, ssm_size=16, n_layers=2,
                 batch=4, seed=0):
    """A small model + params in the shape the diagnostic restores.

    No Stage 2 checkpoint is touched; this builds a fresh module of the same
    class so the helper can be validated without the saved artifacts.
    """
    import dataloaders.speech_commands10 as SC
    from experiments.gp import rawat_benchmark as RB

    class A:
        pass
    a = A()
    a.arm, a.d_model, a.ssm_size, a.n_layers = arm, d_model, ssm_size, n_layers
    a.label_smoothing, a.seed, a.batch = 0.1, seed, batch
    model = RB.build_model(arm, d_model, ssm_size, n_layers, training=True)
    eval_model = RB.build_model(arm, d_model, ssm_size, n_layers,
                                training=False)
    L = 24
    x = np.asarray(jax.random.normal(jax.random.PRNGKey(seed + 5),
                                     (batch, L, SC.N_MFCC)))
    y = np.asarray(jax.random.randint(jax.random.PRNGKey(seed + 6), (batch,),
                                      0, RB.D_OUTPUT))
    v = model.init({"params": jax.random.PRNGKey(seed),
                    "dropout": jax.random.PRNGKey(seed + 1)},
                   x, np.ones((batch, L)), None)
    loaded = dict(args=a, model=model, eval_model=eval_model,
                  params=v["params"], batch_stats=v.get("batch_stats"),
                  init_params=v["params"],
                  init_batch_stats=v.get("batch_stats"))
    return loaded, x, y


def test_zero_offsets_reproduce_the_actual_inference_forward():
    """The helper must BE the model at zero offsets, not merely resemble it."""
    from experiments.gp import rawat_benchmark as RB
    from experiments.gp.stage2_diagnostic import block_activation_report
    loaded, x, y = _tiny_loaded()
    rep = block_activation_report(loaded, x, y, return_raw=True)
    logits_helper = onp.asarray(rep["_raw"]["logits"])
    loss_ref, (logits_ref, _) = RB.loss_and_logits(
        loaded["params"], loaded["batch_stats"], loaded["eval_model"],
        x, y, None, False, loaded["args"].label_smoothing)
    assert logits_helper.shape == onp.asarray(logits_ref).shape
    assert float(onp.max(onp.abs(logits_helper - onp.asarray(logits_ref)))) \
        < 1e-5
    assert abs(rep["loss"] - float(loss_ref)) < 1e-6


def test_offset_derivative_agrees_with_an_independent_finite_difference():
    """Differentiate the helper's own loss along a fixed offset direction."""
    from experiments.gp.stage2_diagnostic import block_activation_report
    loaded, x, y = _tiny_loaded()
    rep = block_activation_report(loaded, x, y, return_raw=True)
    raw = rep["_raw"]
    loss_fn, zeros, g = raw["loss_fn"], raw["zeros"], raw["grads"]
    rs = onp.random.RandomState(4242)
    # RMS-1 per site, so the float32 perturbation stays above resolution
    d = {k: np.asarray(rs.randn(*v.shape), dtype=v.dtype)
         for k, v in zeros.items()}
    d = {k: v / np.std(v) for k, v in d.items()}
    analytic = float(sum(float(np.sum(g[k] * d[k])) for k in zeros))
    h = 1e-2
    plus = float(loss_fn({k: zeros[k] + h * d[k] for k in zeros})[0])
    minus = float(loss_fn({k: zeros[k] - h * d[k] for k in zeros})[0])
    fd = (plus - minus) / (2 * h)
    assert abs(analytic - fd) / max(abs(fd), 1e-8) < 5e-3, (analytic, fd)


def test_shared_offset_cancels_opposite_per_example_gradients():
    """Why offsets carry a batch axis.

    A single offset shared across the batch differentiates to the SUM of signed
    per-example gradients, so +g and -g cancel exactly while both per-example
    magnitudes are large. Demonstrated analytically, then the reporting
    convention is checked on the real helper.
    """
    def shared(off):                       # one vector used by both examples
        return jnp.mean(jnp.stack([jnp.sum(off), -jnp.sum(off)]))

    gshared = jax.grad(shared)(jnp.ones((5,)))
    assert float(jnp.abs(gshared).max()) == 0.0        # total cancellation

    def per_example(off):                  # (2, 5), one slice per example
        return jnp.mean(jnp.stack([jnp.sum(off[0]), -jnp.sum(off[1])]))

    gper = jax.grad(per_example)(jnp.ones((2, 5)))
    assert float(jnp.abs(gper[0]).max()) > 0.0
    assert float(jnp.abs(gper[1]).max()) > 0.0
    assert float(jnp.sum(gper[0]) * jnp.sum(gper[1])) < 0.0   # opposite signs


def test_helper_reports_per_example_not_shared_offset_magnitudes():
    from experiments.gp.stage2_diagnostic import block_activation_report
    loaded, x, y = _tiny_loaded(batch=4)
    rep = block_activation_report(loaded, x, y, return_raw=True)
    g = rep["_raw"]["grads"]
    assert g["block"].shape[0] == 4, "offsets must carry a batch axis"
    assert g["pre_pool"].shape[0] == 4
    for s in rep["sites"]:
        assert "per_example_grad_norm_mean" in s
        assert s["batch_size"] == 4
        # the shared-offset equivalent is reported SEPARATELY and is not the
        # quantity called the per-example magnitude
        assert "shared_offset_equivalent_norm" in s


def test_site_names_distinguish_block_core_and_pre_pooling():
    from experiments.gp.stage2_diagnostic import block_activation_report
    loaded, x, y = _tiny_loaded()
    rep = block_activation_report(loaded, x, y)
    names = {s["site"] for s in rep["sites"]}
    assert names == {"block_input_before_norm_and_skip",
                     "recurrent_core_input_post_norm",
                     "recurrent_core_output",
                     "pre_pooling_encoder_output"}
    skip = {s["site"] for s in rep["sites"] if s["includes_identity_skip"]}
    assert skip == {"block_input_before_norm_and_skip"}
    for a_ in rep["activations"]:
        assert "block_output_rms" in a_ and "recurrent_core_output_rms" in a_


def test_block_input_and_core_input_are_different_measurements():
    """If they coincided, one of the two sites would be mislabelled."""
    from experiments.gp.stage2_diagnostic import block_activation_report
    loaded, x, y = _tiny_loaded()
    rep = block_activation_report(loaded, x, y)
    for layer in range(loaded["args"].n_layers):
        b = next(s for s in rep["sites"]
                 if s["site"] == "block_input_before_norm_and_skip"
                 and s["layer"] == layer)
        c = next(s for s in rep["sites"]
                 if s["site"] == "recurrent_core_input_post_norm"
                 and s["layer"] == layer)
        assert abs(b["per_example_grad_norm_mean"]
                   - c["per_example_grad_norm_mean"]) > 1e-9, layer


def test_runtime_error_is_FAILED_while_interruption_is_INCOMPLETE(tmp_path):
    from experiments.gp.stage2_diagnostic import Status
    st = Status(str(tmp_path))
    assert st.final(hashes_ok=True, budget_incomplete=False,
                    runtime_error=True) == "FAILED"
    assert st.final(hashes_ok=True, budget_incomplete=True,
                    runtime_error=False) == "INCOMPLETE"
