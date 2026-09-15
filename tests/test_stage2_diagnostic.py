"""Correctness of the Stage 2 diagnostic adapter. No training, no updates.

The adapter must equal the EXECUTED forward pass for every arm before any
number it produces is interpreted. `s5/gp_diagnostics.py` dispatches on a
`mechanism` attribute that `SubstrateSSM` does not have, and a dispatch miss
there silently returns inherited native coefficients - which is exactly the
failure mode these tests exist to exclude.
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
from s5.rawat_s5 import ARMS, init_substrate_ssm                  # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                           # noqa: E402

IMPULSE_TOL = 1e-10          # float64 algebraic identity
FREQ_TOL = 1e-6              # resolvent vs truncated-impulse DFT, with tail
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


class _OneLayer:
    """Minimal stand-in exposing the attribute path read_core expects."""

    def __init__(self, seq):
        self.encoder = type("E", (), {"layers": [type("L", (), {"seq": seq})()]})()


def _core_and_module(arm, H=4, L=40, **kwargs):
    kw = ssm_kw(H=H, **kwargs)
    mod = init_substrate_ssm(arm, **kw)()
    x = np.zeros((L, H))
    v = mod.init(jax.random.PRNGKey(0), x)

    def read(m):
        Lam, B_c, Delta = m._native()
        B_tilde = m.B[..., 0] + 1j * m.B[..., 1]
        return dict(response=m.response, input_gain=m.input_gain,
                    clip_eigs=m.clip_eigs, conj_sym=m.conj_sym, P=m.P, H=m.H,
                    Lambda_clipped=Lam,
                    Lambda_raw_param=m.Lambda_re_init + 1j * m.Lambda_im_init,
                    B_tilde=B_tilde, B_c=B_c, Delta=Delta,
                    a=Lam * Delta, b=Delta[:, None] * B_c,
                    C_tilde=m.C_tilde, D=m.D, coefficients=m.coefficients(),
                    physical=dict(T=m.physical.T, gamma=m.physical.gamma,
                                  rho=m.physical.rho, mass=m.physical.mass))
    return mod, v, mod.apply(v, method=read)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_impulse_identities_equal_the_executed_forward_pass(arm):
    H, L = 4, 40
    mod, v, core = _core_and_module(arm, H=H, L=L)
    K = SD.impulse_matrices(core, L)
    for h in range(H):
        x = onp.zeros((L, H)); x[0, h] = 1.0
        y = onp.asarray(mod.apply(v, np.asarray(x)))         # (L, H)
        assert float(onp.max(onp.abs(y - K[:, :, h]))) < IMPULSE_TOL, (
            arm, h, float(onp.max(onp.abs(y - K[:, :, h]))))


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_impulse_is_linear_so_the_core_really_is_the_whole_layer(arm):
    """If the extracted K reproduced only part of the layer, superposition
    against a random input would fail even though the impulse matched."""
    H, L = 4, 40
    mod, v, core = _core_and_module(arm, H=H, L=L)
    K = SD.impulse_matrices(core, L)
    x = onp.asarray(onp.random.RandomState(0).randn(L, H))
    y = onp.asarray(mod.apply(v, np.asarray(x)))
    conv = onp.zeros_like(y)
    for l in range(L):
        for m in range(l + 1):
            conv[l] += K[m] @ x[l - m]
    assert float(onp.max(onp.abs(y - conv))) < 1e-9, arm


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_frequency_response_matches_the_impulse_dft_within_the_tail(arm):
    """The resolvent form must agree with the DFT of a long impulse window.

    Disagreement larger than the truncation tail would mean the 2*Re trap: a
    complex-pair system's real response is NOT 2 Re of its complex transfer.
    """
    H, n_lags = 3, 4096
    # A WELL DAMPED fixture: at the production dt_min = 1e-3 these models have
    # NOT decayed by lag 4096 (tau can exceed 2000 frames), so a truncated DFT
    # cannot test the closed form there. The production-scale case is covered
    # by the geometric tail bound in the next test.
    mod, v, core = _core_and_module(arm, H=H, L=8, dt_min=0.05, dt_max=0.5)
    w, Hf = SD.frequency_response(core, n_freq=33)
    K = SD.impulse_matrices(core, n_lags)
    rho = SD.spectral_radius(core)
    bound = SD.truncation_tail_bound(K, rho)
    dft = onp.zeros_like(Hf)
    lags = onp.arange(n_lags)
    for k, wk in enumerate(w):
        dft[k] = onp.tensordot(onp.exp(-1j * wk * lags), K, axes=(0, 0))
    err = float(onp.max(onp.abs(Hf - dft)))
    assert err < max(FREQ_TOL, 10 * bound), (arm, err, bound, rho)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_production_scale_window_is_declared_truncated_not_negligible(arm):
    """At production dt the 512-lag window used by the report is NOT the whole
    response, and the report must say so rather than imply convergence."""
    H = 3
    _, _, core = _core_and_module(arm, H=H, L=8)          # production dt
    K = SD.impulse_matrices(core, 512)
    rho = SD.spectral_radius(core)
    bound = SD.truncation_tail_bound(K, rho)
    assert rho < 1.0
    energy_in_window = float(onp.sqrt(onp.sum(K ** 2)))
    # the tail is a real quantity here, not a rounding artefact
    assert bound > 0.0
    assert onp.isfinite(bound) and onp.isfinite(energy_in_window)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_two_re_of_the_complex_transfer_is_NOT_the_real_response(arm):
    """Pins the specific error the brief warns about, so a later refactor
    cannot reintroduce it silently."""
    H = 3
    _, _, core = _core_and_module(arm, H=H, L=8)
    w, Hf = SD.frequency_response(core, n_freq=33)
    c = core["coefficients"]
    C = onp.asarray(core["C_tilde"]).astype(onp.complex128)
    if core["response"] in ("one_tap", "alpha_p_two_tap"):
        A = onp.asarray(c["A_bar"]).astype(onp.complex128)
        B = onp.asarray(c.get("B_bar")).astype(onp.complex128)
    elif core["response"] == "gp_fixed_m0":
        A = onp.asarray(c["a_bar"]).astype(onp.complex128)
        B = onp.asarray(c["b_bar"]).astype(onp.complex128)
    else:
        pytest.skip("block realization: the scalar shortcut does not apply")
    naive = onp.zeros_like(Hf)
    for k, wk in enumerate(w):
        u = onp.exp(-1j * wk)
        naive[k] = 2 * onp.real(C @ (B / (1.0 - A * u)[:, None]))
    # they must NOT coincide except at w = 0 and w = pi, where u is real
    interior = onp.max(onp.abs(Hf[1:-1] - naive[1:-1]))
    assert interior > 1e-8, (core["response"], interior)


def test_band_energy_reports_absolute_and_fraction():
    K = onp.zeros((512, 2, 2))
    K[0] = 1.0
    K[200] = 2.0
    r = SD.band_energy(K, BANDS)
    assert r["total_energy"] == pytest.approx(4 * 1.0 + 4 * 4.0)
    by = {(b["lo"], b["hi"]): b for b in r["bands"]}
    assert by[(0, 0)]["energy"] == pytest.approx(4.0)
    assert by[(161, 511)]["energy"] == pytest.approx(16.0)
    assert sum(b["fraction"] for b in r["bands"]) == pytest.approx(1.0)


@pytest.mark.parametrize("arm", ["alpha_p_s5", "gp_fixed_m0", "gp_fixed_mass"])
def test_counterfactual_one_tap_differs_from_the_executed_law(arm):
    """The counterfactual must actually swap the law, not silently return the
    same response - otherwise it could not separate law from coadaptation."""
    H, L = 4, 64
    _, _, core = _core_and_module(arm, H=H, L=L)
    K = SD.impulse_matrices(core, L)
    Kc = SD.counterfactual_one_tap(core, L)
    assert float(onp.max(onp.abs(K - Kc))) > 1e-6, arm


def test_counterfactual_reproduces_the_one_tap_arm_exactly():
    """Sanity: on a one-tap checkpoint the counterfactual is the identity."""
    H, L = 4, 64
    _, _, core = _core_and_module("gain_clip_s5", H=H, L=L)
    K = SD.impulse_matrices(core, L)
    Kc = SD.counterfactual_one_tap(core, L)
    assert float(onp.max(onp.abs(K - Kc))) < 1e-9


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_pole_summary_reads_the_clipped_executed_pole(arm):
    _, _, core = _core_and_module(arm)
    s = SD.pole_summary(core)
    assert s["abs_a_bar_max"] < 1.0
    assert s["n_modes"] == core["P"]
    assert 0.0 < s["delta_min"] <= s["delta_max"] < 1.0


def test_adapter_refuses_an_unknown_response():
    """A dispatch miss must RAISE. gp_diagnostics.core_from_module silently
    falls back to inherited native coefficients for these arms; that is the
    bug this module was written to avoid, so it is pinned here."""
    _, _, core = _core_and_module("native_s5")
    bad = dict(core, response="something_else")
    with pytest.raises(ValueError):
        SD.impulse_matrices(bad, 4)
    with pytest.raises(ValueError):
        SD.frequency_response(bad, n_freq=8)


def test_old_gp_diagnostics_would_have_dispatched_wrongly():
    """Documents WHY the adapter exists, as an executable fact rather than a
    remark: the old helper keys off `mechanism`, which SubstrateSSM lacks."""
    from s5.rawat_s5 import SubstrateSSM
    assert not hasattr(SubstrateSSM, "mechanism")
    assert "response" in SubstrateSSM.__annotations__
