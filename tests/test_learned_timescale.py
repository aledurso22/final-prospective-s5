"""Focused checks for the LEARNED RESPONSE TIMESCALE study. CLUSTER-ONLY.

Run by `bin/run_experiments/cluster_timescale.sh` inside the same 1,200 s cap
as the screen. Not run locally: the brief requires all numerical work on the
cluster.

PREDECLARED TOLERANCES, by dtype:
    EXACT64  1e-10  per-mode generator/ZOH vs the independent dense reference
    LIMIT64  1e-10  T = 5 reproduces the fixed-horizon law; rho = 1 reduction
    IDENT64  1e-12  the two generalized arms are the same function at init
    FD64     1e-6   best central-difference agreement for d/d(log T)
    ZEROT    1e-9   |d(physical output)/dT| at the rho = 1 boundary
    F32      2e-4   production float32/complex64, in `timescale_float32_probe.py`

Declared numerical corners exercised here: T in {0.05, 500} (the guardrails)
and rho in {0.01, 0.9999} (the existing bounds), in every combination.
"""

import math
import os
import subprocess
import sys

import jax
import numpy as onp
import pytest
from scipy.linalg import expm as sp_expm

jax.config.update("jax_enable_x64", True)
import jax.numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5 import timescale_response as TR                           # noqa: E402
from s5.gp_fixed import (mass_block_generator, mass_block_zoh,    # noqa: E402
                         mass_scan, mass_scan_sequential)
from s5.rawat_s5 import (LOG_RHO_BOUNDS, LOG_T_BOUNDS,            # noqa: E402
                         RHO_INIT_TIMESCALE, RHO_ONLY_PARAM_NAME,
                         T_BOUNDS, T_ONLY_PARAM_NAME, T_REFERENCE,
                         T_RESPONSES, init_substrate_ssm,
                         two_tap_coefficients)
from s5.response_projection import (RESPONSE_LEAF_BOUNDS,          # noqa: E402
                                    project_response_leaves,
                                    projection_telemetry)

EXACT64, LIMIT64, IDENT64, FD64, ZEROT = 1e-10, 1e-10, 1e-12, 1e-6, 1e-9
#: the declared guardrail corners, exercised as corners and not as a sweep
T_CORNERS = (T_BOUNDS[0], 1.0, T_REFERENCE, T_BOUNDS[1])
RHO_CORNERS = (0.01, 0.25, 0.75, 0.9999)


# ------------------------------------------------- independent reference ---
def _to_real_pair(M):
    return onp.block([[onp.real(M), -onp.imag(M)],
                      [onp.imag(M), onp.real(M)]])


def _from_real_pair(R):
    n = R.shape[0] // 2
    return R[:n, :n] + 1j * R[n:, :n]


def _reference_block(a, b, T, rho, nodes=64):
    """Per-mode A, B, A_bar, B_bar computed INDEPENDENTLY of production.

    `scipy.linalg.expm` is a different exponential implementation from JAX's,
    the complex 2x2 is exponentiated through its REAL-PAIR embedding, and the
    integral is Gauss-Legendre quadrature rather than production's augmented
    4x4 trick. `T` and `rho` may be per-mode.
    """
    a = onp.asarray(a); b = onp.asarray(b)
    P, H = b.shape
    T = onp.broadcast_to(onp.asarray(T, dtype=float), (P,))
    rho = onp.broadcast_to(onp.asarray(rho, dtype=float), (P,))
    xg, wg = onp.polynomial.legendre.leggauss(nodes)
    ug, wg = 0.5 * (xg + 1.0), 0.5 * wg
    A = onp.zeros((P, 2, 2), dtype=complex)
    B = onp.zeros((P, 2, H), dtype=complex)
    A_bar = onp.zeros_like(A); Phi = onp.zeros_like(A)
    for p in range(P):
        J, r, t = -a[p], rho[p], T[p]
        A[p] = onp.array([[-J / r, -(1.0 - r) / r],
                          [-J / (r * t), -1.0 / (r * t)]], dtype=complex)
        B[p] = onp.stack([b[p] / r, b[p] / (r * t)], axis=0)
        Ar = _to_real_pair(A[p])
        A_bar[p] = _from_real_pair(sp_expm(Ar))
        Phi[p] = _from_real_pair(
            sum(w * sp_expm(Ar * u) for u, w in zip(ug, wg)))
    return dict(A=A, B=B, A_bar=A_bar, Phi=Phi,
                B_bar=onp.einsum("pij,pjh->pih", Phi, B))


def _modes(rs, P, H):
    a = onp.asarray(-onp.exp(rs.uniform(-3.0, -0.5, P))
                    + 1j * rs.uniform(-2.5, 2.5, P))
    b = onp.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    return a, b


@pytest.mark.parametrize("P,H", [(5, 3), (3, 7), (4, 4)])
def test_per_mode_T_generator_and_ZOH_match_an_independent_reference(P, H):
    """PER-MODE T, with P != H included deliberately.

    A (P,) denominator applied to a (P,H) input row broadcasts along the
    FEATURE axis, which is silently valid only when H == P. The T coordinate is
    new in that denominator, so this is the specific broadcasting the change
    could get wrong.
    """
    rs = onp.random.RandomState(P * 31 + H)
    a, b = _modes(rs, P, H)
    T = onp.exp(rs.uniform(math.log(0.2), math.log(50.0), P))
    rho = rs.uniform(0.05, 0.95, P)
    got = mass_block_zoh(np.asarray(a), np.asarray(b), np.asarray(T),
                         np.ones(P), np.asarray(rho))
    ref = _reference_block(a, b, T, rho)
    for k in ("A", "B", "A_bar", "B_bar"):
        err = onp.max(onp.abs(onp.asarray(got[k]) - ref[k]))
        scale = max(1.0, float(onp.max(onp.abs(ref[k]))))
        assert err / scale < EXACT64, (k, P, H, err / scale)


@pytest.mark.parametrize("T", T_CORNERS)
@pytest.mark.parametrize("rho", RHO_CORNERS)
def test_the_declared_numerical_corners_are_finite_in_value_AND_derivative(T, rho):
    """The guardrails must be usable at their corners, not only in the middle.

    A non-finite response or derivative at a declared corner is a failure to
    report and amend with NUMERICAL justification - never a licence to move the
    bound because a score improved.
    """
    rs = onp.random.RandomState(int(1000 * rho) + int(T))
    P, H, L = 5, 3, 16
    a, b = _modes(rs, P, H)
    x = np.asarray(rs.randn(L, H))

    def out(eta):
        TT = T_REFERENCE * np.exp(eta)
        d = mass_block_zoh(np.asarray(a), np.asarray(b), TT, np.ones(P),
                           np.full(P, rho))
        return np.real(np.sum(mass_scan(d["A_bar"], d["B_bar"], x)[..., 0]))

    eta0 = np.full(P, math.log(T / T_REFERENCE))
    assert math.log(1e-2) - 1e-12 <= float(eta0[0]) <= math.log(1e2) + 1e-12
    v = out(eta0)
    g = jax.grad(out)(eta0)
    assert onp.isfinite(float(v)), (T, rho, float(v))
    assert onp.all(onp.isfinite(onp.asarray(g))), (T, rho)
    ref = _reference_block(a, b, T, rho)
    err = onp.max(onp.abs(onp.asarray(
        mass_block_zoh(np.asarray(a), np.asarray(b), np.full(P, T),
                       np.ones(P), np.full(P, rho))["A_bar"]) - ref["A_bar"]))
    assert err / max(1.0, float(onp.max(onp.abs(ref["A_bar"])))) < EXACT64


def test_T_equals_five_reproduces_the_existing_fixed_horizon_law():
    """The learned parameterization must not perturb the existing model.

    eta = 0 gives T = T_REFERENCE exactly, so the per-mode-T coefficients must
    equal the scalar-T coefficients the fixed-horizon arms already use.
    """
    rs = onp.random.RandomState(101)
    P, H = 6, 4
    a, b = _modes(rs, P, H)
    rho = np.full(P, RHO_INIT_TIMESCALE)
    scalar = mass_block_zoh(np.asarray(a), np.asarray(b), T_REFERENCE,
                            np.ones(P), rho)
    per_mode = mass_block_zoh(np.asarray(a), np.asarray(b),
                              np.full(P, T_REFERENCE), np.ones(P), rho)
    for k in ("A", "B", "A_bar", "B_bar"):
        assert onp.max(onp.abs(onp.asarray(scalar[k])
                               - onp.asarray(per_mode[k]))) < LIMIT64, k


@pytest.mark.parametrize("T", [0.5, 5.0, 50.0])
def test_rho_one_is_ordinary_S5_for_EVERY_T_with_zero_T_sensitivity(T):
    """The mathematical boundary rho = 1, where v decouples from s.

    There the state transfer is b/(p + j), INDEPENDENT of T - which is exactly
    why this study does NOT initialize at rho near one. The physical output's
    sensitivity to T is asserted to be ZERO here; no nonzero-gradient assertion
    is made at this boundary, because none is true.
    """
    rs = onp.random.RandomState(7)
    P, H, L = 5, 3, 16
    a, b = _modes(rs, P, H)
    x = np.asarray(rs.randn(L, H))
    Delta = np.ones(P)

    def s_out(eta):
        TT = T_REFERENCE * np.exp(eta)
        d = mass_block_zoh(np.asarray(a), np.asarray(b), TT, np.ones(P),
                           np.ones(P))
        return mass_scan(d["A_bar"], d["B_bar"], x)[..., 0]

    eta0 = np.full(P, math.log(T / T_REFERENCE))
    got = s_out(eta0)
    # ordinary one-tap S5: horizon 0 in the existing Rawat helper
    c = two_tap_coefficients(np.asarray(a), np.asarray(b), Delta, 0.0)
    from s5.rawat_s5 import diagonal_scan_drive
    drive = jax.vmap(lambda u: c["B_bar"] @ u)(x)
    want = diagonal_scan_drive(c["A_bar"], drive)
    assert onp.max(onp.abs(onp.asarray(got) - onp.asarray(want))) < LIMIT64

    w = np.asarray(rs.randn(L, P) + 1j * rs.randn(L, P))
    g = jax.grad(lambda e: np.real(np.sum(w * s_out(e))))(eta0)
    assert onp.max(onp.abs(onp.asarray(g))) < ZEROT, float(
        onp.max(onp.abs(onp.asarray(g))))


def test_the_T_derivative_is_NONZERO_away_from_that_boundary():
    """Complementary to the previous check: the freedom is observable at
    rho = 0.75, which is why the study initializes there."""
    rs = onp.random.RandomState(9)
    P, H, L = 5, 3, 16
    a, b = _modes(rs, P, H)
    x = np.asarray(rs.randn(L, H))

    def loss(eta):
        TT = T_REFERENCE * np.exp(eta)
        d = mass_block_zoh(np.asarray(a), np.asarray(b), TT, np.ones(P),
                           np.full(P, RHO_INIT_TIMESCALE))
        return np.real(np.sum(mass_scan(d["A_bar"], d["B_bar"], x)[..., 0] ** 2))

    g = jax.grad(loss)(np.zeros(P))
    assert onp.min(onp.abs(onp.asarray(g))) > 0.0


def test_the_log_T_gradient_converges_against_central_differences():
    """An INDEPENDENT finite-difference check on a NONCONSTANT probe.

    A nonzero gradient is not a gradient-correctness test. This measures the
    directional derivative against central differences over a step ladder and
    requires O(h^2) convergence to the analytic value - the signature of a
    correct derivative rather than of an accidentally plausible number. The
    probe is a random input sequence, so the loss is not constant in T.
    """
    rs = onp.random.RandomState(11)
    P, H, L = 5, 3, 24
    a, b = _modes(rs, P, H)
    x = np.asarray(rs.randn(L, H))
    d_eta = np.asarray(rs.randn(P))

    def loss(eta):
        TT = T_REFERENCE * np.exp(eta)
        d = mass_block_zoh(np.asarray(a), np.asarray(b), TT, np.ones(P),
                           np.full(P, RHO_INIT_TIMESCALE))
        z = mass_scan(d["A_bar"], d["B_bar"], x)
        return np.real(np.sum(z[..., 0] ** 2) + np.sum(z[..., 1] ** 2))

    eta0 = np.zeros(P)
    analytic = float(np.sum(jax.grad(loss)(eta0) * d_eta))
    assert abs(analytic) > 0.0
    ladder = []
    for h in (1e-2, 3e-3, 1e-3):
        fd = float((loss(eta0 + h * d_eta) - loss(eta0 - h * d_eta))
                   / (2.0 * h))
        ladder.append((h, abs(fd - analytic) / abs(analytic)))
    print("  d/d(log T) ladder:",
          "  ".join(f"h={h:.0e} rel={r:.3e}" for h, r in ladder))
    assert min(r for _, r in ladder) < FD64, ladder
    # truncation-dominated at the coarse end, i.e. actually converging
    assert ladder[0][1] > ladder[-1][1], ladder


def test_the_gradient_flows_through_the_DERIVED_mass_mu_equals_rho_T():
    """mu is derived, so d(mu)/d(eta) = rho * T and d(mu)/d(zeta) = rho * T."""
    P = 4
    rho = onp.full(P, RHO_INIT_TIMESCALE)
    eta = onp.zeros(P)

    def mu(e, z):
        return np.sum(T_REFERENCE * np.exp(e) * np.exp(z))

    g_eta, g_zeta = jax.grad(mu, argnums=(0, 1))(
        np.asarray(eta), np.asarray(onp.log(rho)))
    want = T_REFERENCE * rho
    assert onp.max(onp.abs(onp.asarray(g_eta) - want)) < LIMIT64
    assert onp.max(onp.abs(onp.asarray(g_zeta) - want)) < LIMIT64


def test_scan_carries_and_resets_are_unaffected_by_the_per_mode_horizon():
    rs = onp.random.RandomState(13)
    P, H, L = 4, 3, 24
    a, b = _modes(rs, P, H)
    T = onp.exp(rs.uniform(math.log(0.5), math.log(20.0), P))
    d = mass_block_zoh(np.asarray(a), np.asarray(b), np.asarray(T),
                       np.ones(P), np.full(P, 0.6))
    x = np.asarray(rs.randn(L, H))
    z0 = np.asarray(rs.randn(P, 2) + 1j * rs.randn(P, 2))
    mask = onp.zeros(L, dtype=bool); mask[[0, 11]] = True
    for kw in (dict(), dict(z0=z0), dict(reset_mask=np.asarray(mask)),
               dict(z0=z0, reset_mask=np.asarray(mask))):
        par = mass_scan(d["A_bar"], d["B_bar"], x, **kw)
        seq = mass_scan_sequential(d["A_bar"], d["B_bar"], x, **kw)
        err = onp.max(onp.abs(onp.asarray(par) - onp.asarray(seq)))
        assert err < EXACT64, (sorted(kw), err)
    # chunked streaming equals one whole call
    whole = mass_scan(d["A_bar"], d["B_bar"], x)
    k = 9
    first = mass_scan(d["A_bar"], d["B_bar"], x[:k])
    second = mass_scan(d["A_bar"], d["B_bar"], x[k:], z0=first[-1])
    joined = np.concatenate([first, second], axis=0)
    assert onp.max(onp.abs(onp.asarray(joined)
                           - onp.asarray(whole))) < EXACT64


# ------------------------------------------------------ the circuit map ----
@pytest.mark.parametrize("T", T_CORNERS)
@pytest.mark.parametrize("rho", (0.01, 0.25, 0.75, 0.99))
def test_the_positive_component_map_reproduces_T_rho_and_the_masses(T, rho):
    res = TR.check_identities(T, rho)
    assert res["passed"], res
    c = TR.component_reconstruction(T, rho)
    assert float(c["kappa"]) == pytest.approx(TR.C_STAR / T, rel=1e-12)
    assert float(c["tau_d"]) == pytest.approx(T * rho, rel=1e-12)
    assert float(c["gamma_phys"]) == pytest.approx(T, rel=1e-12)
    assert float(c["M_phys"]) == pytest.approx(T * T * rho, rel=1e-12)
    # NORMALIZED and PHYSICAL are distinct quantities, not synonyms
    assert float(c["mu"]) == pytest.approx(T * rho, rel=1e-12)
    assert float(c["M_phys"] / c["gamma_phys"]) == pytest.approx(
        float(c["mu"]), rel=1e-12)


def test_the_map_recovers_the_symmetric_reference_at_T5_rho075():
    c = TR.component_reconstruction(5.0, 0.75)
    assert float(c["G_s"]) == pytest.approx(2.0, rel=1e-12)
    assert float(c["g_L"]) == pytest.approx(1.0, rel=1e-12)
    assert float(c["h"]) == pytest.approx(1.0, rel=1e-12)
    assert float(c["g_L"]) == pytest.approx(float(c["h"]), rel=1e-12)


# -------------------------------------------------- the executed module ----
def _ssm_kwargs(P, H):
    rs = onp.random.RandomState(29)
    return dict(H=H, P=P,
                Lambda_re_init=-onp.exp(rs.uniform(-3, -0.5, P)),
                Lambda_im_init=rs.uniform(-2, 2, P),
                V=onp.eye(2 * P, dtype=complex)[:, :P],
                Vinv=onp.eye(2 * P, dtype=complex)[:P],
                C_init="trunc_standard_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=True, bidirectional=False)


def _bind(arm, P=4, H=6, L=12, seed=0):
    ssm = init_substrate_ssm(arm, **_ssm_kwargs(P, H))()
    x = np.asarray(onp.random.RandomState(seed).randn(L, H))
    params = ssm.init(jax.random.PRNGKey(seed), x)["params"]
    return ssm, params, x


def test_both_generalized_arms_declare_both_leaves_at_the_declared_init():
    from flax.traverse_util import flatten_dict
    for arm in ("gp_rho_T", "gp_rho_T_fixed"):
        ssm, params, _ = _bind(arm)
        flat = flatten_dict(params)
        rho_leaf = [v for k, v in flat.items()
                    if k[-1] == RHO_ONLY_PARAM_NAME]
        eta_leaf = [v for k, v in flat.items() if k[-1] == T_ONLY_PARAM_NAME]
        assert len(rho_leaf) == 1 and len(eta_leaf) == 1, arm
        assert rho_leaf[0].shape == (4,) and eta_leaf[0].shape == (4,)
        assert rho_leaf[0].dtype == onp.float32, arm
        assert eta_leaf[0].dtype == onp.float32, arm
        rho = float(onp.exp(onp.asarray(rho_leaf[0])[0]))
        assert rho == pytest.approx(RHO_INIT_TIMESCALE, rel=1e-6), (arm, rho)
        assert float(onp.max(onp.abs(onp.asarray(eta_leaf[0])))) == 0.0
        T = ssm.apply({"params": params},
                      method=lambda m: m.response_timescale())
        assert onp.allclose(onp.asarray(T), T_REFERENCE, rtol=1e-6)
        mu = ssm.apply({"params": params}, method=lambda m: m.derived_mass())
        assert onp.allclose(onp.asarray(mu), T_REFERENCE * rho, rtol=1e-5)


def test_the_two_generalized_arms_are_the_SAME_function_at_initialization():
    """They differ only in whether eta is updated, so at step zero they must be
    identical - otherwise their later difference would not isolate learning the
    timescale."""
    from flax.traverse_util import flatten_dict
    ssm_f, p_f, x = _bind("gp_rho_T_fixed", seed=3)
    ssm_l, p_l, _ = _bind("gp_rho_T", seed=3)
    assert sorted("/".join(k) for k in flatten_dict(p_f)) == \
        sorted("/".join(k) for k in flatten_dict(p_l))
    y_f = onp.asarray(ssm_f.apply({"params": p_f}, x))
    y_l = onp.asarray(ssm_l.apply({"params": p_l}, x))
    assert onp.max(onp.abs(y_f - y_l)) < IDENT64
    # and their gradients with respect to the SHARED coordinates agree
    loss_f = jax.grad(lambda p: np.sum(ssm_f.apply({"params": p}, x) ** 2))(p_f)
    loss_l = jax.grad(lambda p: np.sum(ssm_l.apply({"params": p}, x) ** 2))(p_l)
    ff, fl = flatten_dict(loss_f), flatten_dict(loss_l)
    for k in ff:
        if k[-1] == T_ONLY_PARAM_NAME:
            continue                      # eta is the coordinate under test
        assert onp.max(onp.abs(onp.asarray(ff[k]) - onp.asarray(fl[k]))) \
            < IDENT64, "/".join(k)


def test_the_coefficients_and_diagnostics_read_the_EXECUTED_T():
    """A learned-timescale arm must not report the static dataclass horizon."""
    from s5 import substrate_diagnostics as SD
    ssm, params, x = _bind("gp_rho_T", P=4, H=5)
    from flax.traverse_util import flatten_dict, unflatten_dict
    flat = dict(flatten_dict(params))
    key = [k for k in flat if k[-1] == T_ONLY_PARAM_NAME][0]
    flat[key] = np.asarray(onp.linspace(-1.0, 1.0, 4), dtype=np.float32)
    moved = unflatten_dict(flat)
    c = ssm.apply({"params": moved}, method=lambda m: m.coefficients())
    want = T_REFERENCE * onp.exp(onp.linspace(-1.0, 1.0, 4))
    assert onp.allclose(onp.asarray(c["executed_T"]), want, rtol=1e-5)
    assert not onp.allclose(onp.asarray(c["executed_T"]), T_REFERENCE)
    mu = ssm.apply({"params": moved}, method=lambda m: m.derived_mass())
    assert onp.allclose(onp.asarray(mu),
                        want * onp.asarray(c["executed_rho"]), rtol=1e-5)
    # the block really was built with those horizons
    a = onp.asarray(ssm.apply({"params": moved},
                              method=lambda m: m.clock_absorbed()[0]))
    b = onp.asarray(ssm.apply({"params": moved},
                              method=lambda m: m.clock_absorbed()[1]))
    ref = _reference_block(a, b, want, onp.asarray(c["executed_rho"]))
    assert onp.max(onp.abs(onp.asarray(c["A_bar"]) - ref["A_bar"])) < 1e-6
    # T_i j_i, the dimensionless product the report uses
    assert onp.allclose(onp.asarray(c["T_times_j"]), want * (-a), rtol=1e-5)


def test_the_diagnostic_impulse_matches_an_actual_forward_call():
    """A dispatch miss would report another arm's dynamics as if executed."""
    from s5 import substrate_diagnostics as SD
    P, H, n_lags = 4, 5, 48
    ssm, params, _ = _bind("gp_rho_T", P, H, n_lags)
    core = ssm.apply({"params": params}, method=lambda m: dict(
        response=m.response, input_gain=m.input_gain, clip_eigs=m.clip_eigs,
        conj_sym=m.conj_sym, P=m.P, H=m.H, C_tilde=m.C_tilde, D=m.D,
        coefficients=m.coefficients()))
    K = SD.impulse_matrices(core, n_lags)
    for i in range(H):
        u = onp.zeros((n_lags, H)); u[0, i] = 1.0
        y = onp.asarray(ssm.apply({"params": params}, np.asarray(u)))
        assert onp.max(onp.abs(y - K[:, :, i])) < 1e-6, i
    w, Hf = SD.frequency_response(core, 33)
    for k, wk in enumerate(w):
        dft = sum(K[l] * onp.exp(-1j * wk * l) for l in range(n_lags))
        assert onp.max(onp.abs(dft - Hf[k])) < 1e-4, k


def test_state_counts_are_identical_for_the_two_generalized_arms():
    """More trainable coefficients do not imply more carried state."""
    counts = {}
    for arm in ("gp_rho_T", "gp_rho_T_fixed"):
        ssm, params, _ = _bind(arm, P=4, H=6)
        counts[arm] = ssm.apply({"params": params},
                                method=lambda m: m.state_counts())
    assert counts["gp_rho_T"] == counts["gp_rho_T_fixed"]
    c = counts["gp_rho_T"]
    assert c["physical_real"] == 8 and c["auxiliary_real"] == 8
    assert c["previous_input_buffer"] == 0     # no prospective input here
    assert c["total_real"] == 16


# -------------------------------------- production optimizer + projection --
def test_projection_covers_BOTH_response_coordinates():
    assert RHO_ONLY_PARAM_NAME in RESPONSE_LEAF_BOUNDS
    assert T_ONLY_PARAM_NAME in RESPONSE_LEAF_BOUNDS
    assert RESPONSE_LEAF_BOUNDS[T_ONLY_PARAM_NAME] == LOG_T_BOUNDS


@pytest.mark.parametrize("leaf,bounds", [(RHO_ONLY_PARAM_NAME, LOG_RHO_BOUNDS),
                                         (T_ONLY_PARAM_NAME, LOG_T_BOUNDS)])
@pytest.mark.parametrize("direction", ["above", "below"])
def test_forced_outward_then_inward_for_each_raw_coordinate(leaf, bounds,
                                                            direction):
    """The lockout regression, for BOTH coordinates and BOTH bounds.

    Outside the interval the forward clip flattens the map, so the task
    gradient through that leaf is exactly zero and nothing brings it back.
    After the production projection the gradient is nonzero again and a real
    optimizer step keeps the leaf inside.
    """
    import optax
    from flax.traverse_util import flatten_dict, unflatten_dict
    ssm, params, x = _bind("gp_rho_T", P=4, H=5, L=10)
    flat = dict(flatten_dict(params))
    key = [k for k in flat if k[-1] == leaf][0]
    dt = flat[key].dtype
    lo = np.asarray(bounds[0], dtype=dt)
    hi = np.asarray(bounds[1], dtype=dt)
    outside = dict(flat)
    outside[key] = (hi + np.asarray(0.05, dtype=dt) if direction == "above"
                    else lo - np.asarray(0.05, dtype=dt)) * np.ones_like(
                        flat[key])
    outside = unflatten_dict(outside)
    loss = lambda p: np.sum(ssm.apply({"params": p}, x) ** 2)   # noqa: E731
    g_out = flatten_dict(jax.grad(loss)(outside))[key]
    assert onp.max(onp.abs(onp.asarray(g_out))) == 0.0, (leaf, direction)

    projected = project_response_leaves(outside)
    pv = flatten_dict(projected)[key]
    assert pv.dtype == dt                      # dtype-correct comparison
    assert bool(np.all(pv <= hi)) and bool(np.all(pv >= lo))
    g_in = flatten_dict(jax.grad(loss)(projected))[key]
    assert onp.max(onp.abs(onp.asarray(g_in))) > 0.0, (leaf, direction)

    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(1e-3, weight_decay=1e-4))
    st = tx.init(projected)
    g = jax.grad(loss)(projected)
    upd, st = tx.update(g, st, projected)
    raw = optax.apply_updates(projected, upd)
    final = project_response_leaves(raw)
    fv = flatten_dict(final)[key]
    assert bool(np.all(fv <= hi)) and bool(np.all(fv >= lo))
    tel = projection_telemetry(raw, final, grads=g, updates=upd, names=(leaf,))
    assert float(tel["response_grad_norm"]) > 0.0


def test_a_REAL_production_update_moves_log_T_only_in_the_learned_arm():
    """The production step, optimizer and projector, on the production model.

    Arm 5 must move eta; arm 4 must leave it EXACTLY unchanged, including under
    decoupled weight decay - which is why eta is frozen by zeroing its updates
    rather than by merely omitting its gradient. rho and the native parameters
    must move in both.
    """
    from flax.traverse_util import flatten_dict
    from experiments.gp import timescale_study as TS
    rs = onp.random.RandomState(5)
    xb = np.asarray(rs.randn(4, 40, 20).astype(onp.float32))
    yb = np.asarray(rs.randint(0, 10, 4))
    moved = {}
    for arm in ("gp_rho_T", "gp_rho_T_fixed"):
        m, p, bs = TS.init_variables(arm, 0)
        tx, _ = TS.make_optimizer(10, 1, freeze_T=(arm == "gp_rho_T_fixed"))
        st = tx.init(p)
        q, st, bs2, loss, acc, gn, tel = TS.train_step(
            m, tx, p, st, bs, xb, yb, jax.random.PRNGKey(0))
        f0, f1 = flatten_dict(p), flatten_dict(q)
        moved[arm] = {k[-1]: float(onp.max(onp.abs(onp.asarray(f1[k])
                                                   - onp.asarray(f0[k]))))
                      for k in f0}
    assert moved["gp_rho_T"][T_ONLY_PARAM_NAME] > 0.0
    assert moved["gp_rho_T_fixed"][T_ONLY_PARAM_NAME] == 0.0
    for arm in ("gp_rho_T", "gp_rho_T_fixed"):
        assert moved[arm][RHO_ONLY_PARAM_NAME] > 0.0, arm
        assert moved[arm]["Lambda_re"] > 0.0, arm
        assert moved[arm]["log_step"] > 0.0, arm


def test_the_response_policy_admits_both_arms_and_still_refuses_strays():
    import experiments.gp.rawat_benchmark as RB
    _, params, _ = _bind("gp_rho_T")
    stacked = {"encoder": {"layers_0": {"seq": params}}}
    for arm in ("gp_rho_T", "gp_rho_T_fixed"):
        RB.assert_response_policy(stacked, arm, 1, 4)
    for arm in ("alpha_p_s5", "gp_fixed_mass"):
        with pytest.raises(SystemExit, match="not declared for this arm"):
            RB.assert_response_policy(stacked, arm, 1, 4)


def test_the_historical_fixed_coefficient_arms_are_untouched():
    """Adding a learned horizon must not relax the fixed arms or change them."""
    from flax.traverse_util import flatten_dict
    for arm in ("gp_fixed_mass", "alpha_p_s5", "gain_clip_s5", "native_s5"):
        ssm, params, x = _bind(arm, P=4, H=6)
        names = {k[-1] for k in flatten_dict(params)}
        assert T_ONLY_PARAM_NAME not in names and \
            RHO_ONLY_PARAM_NAME not in names, arm
        y = onp.asarray(ssm.apply({"params": params}, x))
        assert onp.all(onp.isfinite(y)), arm
        same = project_response_leaves(params)
        assert jax.tree_util.tree_all(jax.tree_util.tree_map(
            lambda a, b: bool(np.all(a == b)), params, same)), arm
    # gp_fixed_mass still reports the STATIC reference, not a learned horizon
    from s5 import substrate_diagnostics as SD
    ssm, params, _ = _bind("gp_fixed_mass")
    ex = ssm.apply({"params": params},
                   method=lambda m: SD._executed_response(m))
    assert ex["kind"] == "fixed_reference" and ex["T_is_per_mode"] is False


# --------------------------------------------------- production dtypes -----
def test_production_float32_and_complex64_in_its_own_process():
    """x64 is process-global, so the production-dtype probe runs separately."""
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "timescale_float32_probe.py")
    env = dict(os.environ, JAX_ENABLE_X64="0")
    r = subprocess.run([sys.executable, probe], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-4000:])
    assert r.returncode == 0, r.stdout + r.stderr
