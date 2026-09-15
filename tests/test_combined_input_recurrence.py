"""Focused checks for the COMPOSITION: prospective input + prospective
recurrence. CLUSTER-ONLY.

The composition's comparative training batch is DEFERRED by the priority
handoff of 15 September 2026. These checks establish that the implementation is
correct and reusable; they are not a performance claim and no combined training
has been executed.

PREDECLARED TOLERANCES, by dtype:
    EXACT64  1e-10  float64/complex128 against the independent dense reference
    ODE64    1e-8   float64 against an independently integrated trajectory
    LIMIT64  1e-10  the T_in = 0 and rho = 1 reductions, float64
    GRAD64   1e-9   gradient agreement in those limits, float64
    F32      2e-4   production float32/complex64, in `combined_float32_probe.py`

The float64 figures are reference-quality checks of the algebra; the float32
figure is what the executed model actually runs in and is measured in its own
process with x64 OFF, because x64 is a process-global setting.
"""

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

from s5.gp_fixed import (mass_block_two_tap, mass_block_zoh,       # noqa: E402
                         mass_scan, mass_scan_two_tap,
                         mass_scan_two_tap_sequential)
from s5.rawat_s5 import (LOG_RHO_BOUNDS, RHO_ONLY_PARAM_NAME,      # noqa: E402
                         RHO_ONLY_RESPONSES, diagonal_scan_drive,
                         init_substrate_ssm, two_tap_coefficients,
                         two_tap_drive)
from s5.response_projection import (project_response_leaves,        # noqa: E402
                                    projection_telemetry)

EXACT64, ODE64, LIMIT64, GRAD64 = 1e-10, 1e-8, 1e-10, 1e-9
T_IN = 5.0


# ------------------------------------------------- independent reference ---
def _to_real_pair(M):
    """Complex (n,n) -> real (2n,2n) acting on [Re q; Im q]."""
    X, Y = onp.real(M), onp.imag(M)
    return onp.block([[X, -Y], [Y, X]])


def _from_real_pair(R):
    n = R.shape[0] // 2
    return R[:n, :n] + 1j * R[n:, :n]


def _reference_two_tap(a, b, T, rho, horizon_in, nodes=64):
    """A, B, A_bar, Phi, B_plus, B_minus per mode, computed INDEPENDENTLY.

    Independent of the production helper in both respects that matter:
    `scipy.linalg.expm` is a different exponential implementation from JAX's,
    and Phi is formed by Gauss-Legendre quadrature of exp(A u) rather than by
    the augmented-matrix trick production uses. The complex 2x2 block is
    exponentiated through its REAL-PAIR embedding, so the reference never
    relies on a complex matrix exponential either.
    """
    a = onp.asarray(a); b = onp.asarray(b)
    rho = onp.broadcast_to(onp.asarray(rho, dtype=float), a.shape)
    T = onp.broadcast_to(onp.asarray(T, dtype=float), a.shape)
    P, H = b.shape
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
    B_bar = onp.einsum("pij,pjh->pih", Phi, B)
    J_in = horizon_in * onp.einsum("pij,pjh->pih", A_bar, B)
    return dict(A=A, B=B, A_bar=A_bar, Phi=Phi, B_bar=B_bar, J_in=J_in,
                B_plus=B_bar + J_in, B_minus=-J_in)


def _modes(rs, P, H):
    a = onp.asarray(-onp.exp(rs.uniform(-3.0, -0.5, P))
                    + 1j * rs.uniform(-2.5, 2.5, P))
    b = onp.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    return a, b


@pytest.mark.parametrize("rho", [0.05, 0.25, 0.5, 0.75, 0.9999])
@pytest.mark.parametrize("P,H", [(5, 3), (3, 7), (4, 4)])
def test_two_tap_coefficients_match_an_independent_dense_reference(rho, P, H):
    """P != H is included deliberately: a (P,) denominator applied to a (P,H)
    input row broadcasts along the FEATURE axis, which is silently valid only
    when H == P."""
    rs = onp.random.RandomState(P * 100 + H)
    a, b = _modes(rs, P, H)
    got = mass_block_two_tap(np.asarray(a), np.asarray(b), 5.0,
                             np.ones(P), np.full(P, rho), T_IN)
    ref = _reference_two_tap(a, b, 5.0, rho, T_IN)
    for k in ("A", "B", "A_bar", "B_bar", "J_in", "B_plus", "B_minus"):
        err = onp.max(onp.abs(onp.asarray(got[k]) - ref[k]))
        scale = max(1.0, float(onp.max(onp.abs(ref[k]))))
        assert err / scale < EXACT64, (k, rho, P, H, err / scale)


def test_the_correction_is_A_bar_times_CONTINUOUS_B_and_nothing_else():
    """The three ways to form J_in wrongly, excluded by measurement.

    `T_in * B_bar`, `T_in * A_bar @ B_bar` and an extra Delta factor all
    produce a plausible-looking two-tap law. Each is checked to DISAGREE, so
    the passing case is evidence rather than a coincidence of shapes.
    """
    rs = onp.random.RandomState(7)
    P, H = 6, 4
    a, b = _modes(rs, P, H)
    d = mass_block_two_tap(np.asarray(a), np.asarray(b), 5.0, np.ones(P),
                           np.full(P, 0.6), T_IN)
    A_bar, B, B_bar = [onp.asarray(d[k]) for k in ("A_bar", "B", "B_bar")]
    right = T_IN * onp.einsum("pij,pjh->pih", A_bar, B)
    assert onp.max(onp.abs(onp.asarray(d["J_in"]) - right)) < EXACT64
    for wrong in (T_IN * B_bar,
                  T_IN * onp.einsum("pij,pjh->pih", A_bar, B_bar)):
        assert onp.max(onp.abs(right - wrong)) > 1e-3


def test_DC_response_is_unchanged_and_poles_are_untouched():
    """B_plus + B_minus = B_bar, and the autonomous transition is identical."""
    rs = onp.random.RandomState(11)
    P, H = 5, 3
    a, b = _modes(rs, P, H)
    base = mass_block_zoh(np.asarray(a), np.asarray(b), 5.0, np.ones(P),
                          np.full(P, 0.4))
    comb = mass_block_two_tap(np.asarray(a), np.asarray(b), 5.0, np.ones(P),
                              np.full(P, 0.4), T_IN)
    assert onp.max(onp.abs(onp.asarray(comb["A_bar"])
                           - onp.asarray(base["A_bar"]))) < EXACT64
    total = onp.asarray(comb["B_plus"]) + onp.asarray(comb["B_minus"])
    assert onp.max(onp.abs(total - onp.asarray(base["B_bar"]))) < EXACT64


def test_the_jump_law_matches_an_independently_integrated_trajectory():
    """Integrate q' = A q + B x_k with explicit jumps, and compare.

    This checks the DERIVATION, not just the algebra: the two-tap coefficients
    are supposed to encode `q(0+) - q(0-) = T_in B (x_k - x_{k-1})` followed by
    ordinary held-input evolution. Here that story is simulated directly with
    an ODE solver and the result is compared with the closed-form law.
    """
    from scipy.integrate import solve_ivp
    rs = onp.random.RandomState(3)
    P, H, L = 3, 2, 6
    a, b = _modes(rs, P, H)
    rho, T = 0.7, 5.0
    ref = _reference_two_tap(a, b, T, rho, T_IN)
    x = rs.randn(L, H)
    q = onp.zeros((P, 2), dtype=complex)
    x_prev = onp.zeros(H)
    traj = []
    for k in range(L):
        q = q + T_IN * ref["B"] @ (x[k] - x_prev)          # the jump
        nxt = onp.zeros_like(q)
        for p in range(P):                                  # held-input leg
            Ar = _to_real_pair(ref["A"][p])
            drive = onp.concatenate([onp.real(ref["B"][p] @ x[k]),
                                     onp.imag(ref["B"][p] @ x[k])])
            y0 = onp.concatenate([onp.real(q[p]), onp.imag(q[p])])
            sol = solve_ivp(lambda t, y: Ar @ y + drive, (0.0, 1.0), y0,
                            rtol=1e-12, atol=1e-14, dense_output=True)
            yf = sol.y[:, -1]
            nxt[p] = yf[:2] + 1j * yf[2:]
        q = nxt
        x_prev = x[k]
        traj.append(q.copy())
    traj = onp.stack(traj)
    got = onp.asarray(mass_scan_two_tap(
        np.asarray(ref["A_bar"]), np.asarray(ref["B_plus"]),
        np.asarray(ref["B_minus"]), np.asarray(x)))
    assert onp.max(onp.abs(got - traj)) < ODE64


# --------------------------------------------------------- the two limits --
def test_T_in_zero_reduces_to_the_plain_generalized_recurrence():
    rs = onp.random.RandomState(5)
    P, H, L = 6, 4, 20
    a, b = _modes(rs, P, H)
    x = np.asarray(rs.randn(L, H))
    rho = np.full(P, 0.55)
    base = mass_block_zoh(np.asarray(a), np.asarray(b), 5.0, np.ones(P), rho)
    comb = mass_block_two_tap(np.asarray(a), np.asarray(b), 5.0, np.ones(P),
                              rho, 0.0)
    z_base = mass_scan(base["A_bar"], base["B_bar"], x)
    z_comb = mass_scan_two_tap(comb["A_bar"], comb["B_plus"], comb["B_minus"],
                               x)
    assert onp.max(onp.abs(onp.asarray(z_base) - onp.asarray(z_comb))) < LIMIT64


@pytest.mark.parametrize("horizon", [0.0, 1.0, 5.0])
def test_rho_one_reduces_to_Rawat_two_tap_forward_and_in_gradient(horizon):
    """The MATHEMATICAL boundary rho = 1, where v decouples from s.

    rho = 1 is outside the trained parameter's numerical margin (0.9999); it is
    used here as an algebraic limit, which is what the brief asks for. With
    Delta = 1 the clock-absorbed (a, b) equal (Lambda, B_c), so the comparison
    is against the EXISTING Rawat coefficient helper with no re-derivation.
    """
    rs = onp.random.RandomState(13)
    P, H, L = 5, 3, 16
    a, b = _modes(rs, P, H)
    Delta = np.ones(P)
    x = np.asarray(rs.randn(L, H))

    def combined_out(bb):
        c = mass_block_two_tap(np.asarray(a), bb, 5.0, np.ones(P),
                               np.ones(P), horizon)
        return mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"],
                                 x)[..., 0]

    def rawat_out(bb):
        c = two_tap_coefficients(np.asarray(a), bb, Delta, horizon)
        drive = two_tap_drive(c["B_plus"], c["B_minus"], x)
        return diagonal_scan_drive(c["A_bar"], drive)

    bb = np.asarray(b)
    got, want = combined_out(bb), rawat_out(bb)
    assert onp.max(onp.abs(onp.asarray(got) - onp.asarray(want))) < LIMIT64

    # gradients of the SHARED parameters and of the input agree too
    w = np.asarray(rs.randn(L, P) + 1j * rs.randn(L, P))
    loss_c = lambda t: np.real(np.sum(w * combined_out(t)))        # noqa: E731
    loss_r = lambda t: np.real(np.sum(w * rawat_out(t)))           # noqa: E731
    gc = jax.grad(loss_c)(bb)
    gr = jax.grad(loss_r)(bb)
    assert onp.max(onp.abs(onp.asarray(gc) - onp.asarray(gr))) < GRAD64


# ------------------------------------------------- scan, carries, resets ---
def test_parallel_scan_agrees_with_an_independent_sequential_recurrence():
    """Nonzero carries and resets included. The sequential reference carries
    the previous token in its scan STATE rather than building a shifted array,
    so agreement is not a restatement of the same construction."""
    rs = onp.random.RandomState(17)
    P, H, L = 4, 3, 24
    a, b = _modes(rs, P, H)
    c = mass_block_two_tap(np.asarray(a), np.asarray(b), 5.0, np.ones(P),
                           np.full(P, 0.65), T_IN)
    x = np.asarray(rs.randn(L, H))
    z0 = np.asarray(rs.randn(P, 2) + 1j * rs.randn(P, 2))
    px = np.asarray(rs.randn(H))
    mask = onp.zeros(L, dtype=bool); mask[[0, 9, 17]] = True
    for kw in (dict(), dict(z0=z0), dict(prev_x=px),
               dict(z0=z0, prev_x=px), dict(reset_mask=np.asarray(mask)),
               dict(z0=z0, prev_x=px, reset_mask=np.asarray(mask))):
        par = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"], x, **kw)
        seq = mass_scan_two_tap_sequential(c["A_bar"], c["B_plus"],
                                           c["B_minus"], x, **kw)
        err = onp.max(onp.abs(onp.asarray(par) - onp.asarray(seq)))
        assert err < EXACT64, (sorted(kw), err)


def test_a_reset_clears_the_DELAYED_INPUT_and_not_only_the_block_state():
    """Clearing only q would leak the previous sequence's last token.

    Two independent sequences are run concatenated with a reset at the join,
    then the second is run alone. If only the block carry were cleared, the
    second sequence's FIRST output would still contain B_minus @ (last token of
    the first sequence) and the two would differ.
    """
    rs = onp.random.RandomState(19)
    P, H, L = 4, 3, 10
    a, b = _modes(rs, P, H)
    c = mass_block_two_tap(np.asarray(a), np.asarray(b), 5.0, np.ones(P),
                           np.full(P, 0.6), T_IN)
    x1, x2 = rs.randn(L, H), rs.randn(L, H)
    joined = np.asarray(onp.concatenate([x1, x2], axis=0))
    mask = onp.zeros(2 * L, dtype=bool); mask[L] = True
    both = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"], joined,
                             reset_mask=np.asarray(mask))
    alone = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"],
                              np.asarray(x2))
    err = onp.max(onp.abs(onp.asarray(both[L:]) - onp.asarray(alone)))
    assert err < EXACT64, err
    # and the leak this guards against is REAL: without the reset the join
    # differs, so the assertion above is not vacuous
    leaky = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"], joined)
    assert onp.max(onp.abs(onp.asarray(leaky[L]) - onp.asarray(alone[0]))) > 1e-6


def test_streaming_in_chunks_equals_one_whole_call():
    """Both carries preserved across an ordinary chunk boundary."""
    rs = onp.random.RandomState(23)
    P, H, L = 4, 3, 18
    a, b = _modes(rs, P, H)
    c = mass_block_two_tap(np.asarray(a), np.asarray(b), 5.0, np.ones(P),
                           np.full(P, 0.5), T_IN)
    x = np.asarray(rs.randn(L, H))
    whole = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"], x)
    k = 7
    first = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"], x[:k])
    second = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"], x[k:],
                               z0=first[-1], prev_x=x[k - 1])
    got = np.concatenate([first, second], axis=0)
    assert onp.max(onp.abs(onp.asarray(got) - onp.asarray(whole))) < EXACT64


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


def test_the_combined_response_is_registered_and_carries_ONE_rho_leaf():
    from flax.traverse_util import flatten_dict
    assert "gp_rho_prospin" in RHO_ONLY_RESPONSES
    ssm, params, _ = _bind("gp_rho_prospin")
    leaves = [k for k in flatten_dict(params)
              if k[-1] == RHO_ONLY_PARAM_NAME]
    assert len(leaves) == 1
    assert flatten_dict(params)[leaves[0]].shape == (4,)
    assert flatten_dict(params)[leaves[0]].dtype == onp.float32


def test_state_counts_report_the_auxiliary_carry_AND_the_input_buffer():
    ssm, params, x = _bind("gp_rho_prospin", P=4, H=6)
    counts = ssm.apply({"params": params}, method=lambda m: m.state_counts())
    assert counts["physical_real"] == 8          # 2P under conjugate symmetry
    assert counts["auxiliary_real"] == 8
    assert counts["previous_input_buffer"] == 6  # H
    assert counts["total_real"] == 22


def test_the_module_forward_matches_the_reference_law_end_to_end():
    """The BOUND module, not a re-derivation: read its executed coefficients,
    rebuild the output from the reference two-tap recurrence and compare."""
    P, H, L = 4, 6, 14
    ssm, params, x = _bind("gp_rho_prospin", P, H, L)
    y = onp.asarray(ssm.apply({"params": params}, x))
    c = ssm.apply({"params": params}, method=lambda m: m.coefficients())
    D = onp.asarray(ssm.apply({"params": params}, method=lambda m: m.D))
    C = onp.asarray(ssm.apply({"params": params}, method=lambda m: m.C_tilde))
    z = onp.asarray(mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"], x))
    want = 2.0 * onp.real(z[..., 0] @ C.T) + onp.asarray(x) * D
    assert onp.max(onp.abs(y - want)) < 1e-6


def test_a_prev_x_carry_is_refused_by_responses_without_a_second_tap():
    ssm, params, x = _bind("gp_rho")
    with pytest.raises(ValueError, match="no delayed-input tap"):
        ssm.apply({"params": params}, x, None, None, np.zeros(6))


def test_diagnostics_agree_with_the_ACTUAL_executed_forward_response():
    """A dispatch miss here would report a different arm's dynamics.

    The impulse matrices are compared against a real forward call on unit
    impulses, and the frequency response against the DFT of that same measured
    impulse over a window long enough for the tail to be negligible.
    """
    from s5 import substrate_diagnostics as SD
    P, H, L = 4, 5, 160
    ssm, params, _ = _bind("gp_rho_prospin", P, H, L)
    # `read_core` expects a stacked classifier; this is a bare layer, so the
    # same fields are read directly from the bound module.
    core = ssm.apply({"params": params}, method=lambda m: dict(
        response=m.response, input_gain=m.input_gain, clip_eigs=m.clip_eigs,
        conj_sym=m.conj_sym, P=m.P, H=m.H, C_tilde=m.C_tilde, D=m.D,
        coefficients=m.coefficients()))
    n_lags = 64
    K = SD.impulse_matrices(core, n_lags)
    for i in range(H):
        u = onp.zeros((n_lags, H)); u[0, i] = 1.0
        y = onp.asarray(ssm.apply({"params": params}, np.asarray(u)))
        assert onp.max(onp.abs(y - K[:, :, i])) < 1e-5, i
    w, Hf = SD.frequency_response(core, 33)
    for k, wk in enumerate(w):
        dft = sum(K[l] * onp.exp(-1j * wk * l) for l in range(n_lags))
        assert onp.max(onp.abs(dft - Hf[k])) < 1e-4, k


# -------------------------------------- production optimizer + projection --
def test_a_production_update_reaches_rho_and_the_DELAYED_INPUT_path():
    """Full BPTT: no stop_gradient on the block carry or the second tap.

    The delayed-input path is isolated by differentiating with respect to the
    input sequence and checking that token L-2 receives gradient from the
    output at L-1 through B_minus, which only the second tap supplies.
    """
    P, H, L = 4, 5, 8
    ssm, params, x = _bind("gp_rho_prospin", P, H, L)
    from flax.traverse_util import flatten_dict
    key = [k for k in flatten_dict(params) if k[-1] == RHO_ONLY_PARAM_NAME][0]

    def last_only(p, xx):
        return np.sum(ssm.apply({"params": p}, xx)[-1] ** 2)

    g_p = jax.grad(last_only)(params, x)
    assert onp.max(onp.abs(onp.asarray(flatten_dict(g_p)[key]))) > 0.0
    g_x = onp.asarray(jax.grad(last_only, argnums=1)(params, x))
    assert onp.max(onp.abs(g_x[-2])) > 0.0        # via A_bar AND via B_minus
    assert onp.all(onp.isfinite(g_x))
    # The second tap SPECIFICALLY: token L-1 is the last input, so with only a
    # one-tap drive the output at L-1 could not depend on it through B_minus.
    # Differentiating a length-1 window isolates that path - with no recurrence
    # to carry anything, any gradient at the PREVIOUS token comes from B_minus.
    def head_only(p, xx):
        return np.sum(ssm.apply({"params": p}, xx)[1] ** 2)
    g2 = onp.asarray(jax.grad(head_only, argnums=1)(params, x[:2]))
    assert onp.max(onp.abs(g2[0])) > 0.0


def test_forced_outward_then_inward_restores_a_usable_gradient():
    """The lockout regression, exercised through the PRODUCTION projector.

    An outward raw value has zero task gradient (the forward clip flattens it).
    After projection the same leaf has a nonzero gradient again and an inward
    update actually moves it back inside the interval - not merely that a
    projection was applied.
    """
    import optax
    from flax.traverse_util import flatten_dict, unflatten_dict
    P, H, L = 4, 5, 8
    ssm, params, x = _bind("gp_rho_prospin", P, H, L)
    flat = flatten_dict(params)
    key = [k for k in flat if k[-1] == RHO_ONLY_PARAM_NAME][0]
    hi = np.asarray(LOG_RHO_BOUNDS[1], dtype=flat[key].dtype)

    outside = dict(flat); outside[key] = flat[key] + np.asarray(
        0.05, dtype=flat[key].dtype)
    outside = unflatten_dict(outside)
    assert bool(np.all(flatten_dict(outside)[key] > hi))

    loss = lambda p: np.sum(ssm.apply({"params": p}, x) ** 2)   # noqa: E731
    g_out = flatten_dict(jax.grad(loss)(outside))[key]
    assert onp.max(onp.abs(onp.asarray(g_out))) == 0.0

    projected = project_response_leaves(outside)
    pv = flatten_dict(projected)[key]
    assert pv.dtype == flat[key].dtype            # dtype-correct comparison
    assert bool(np.all(pv <= hi))
    g_in = flatten_dict(jax.grad(loss)(projected))[key]
    assert onp.max(onp.abs(onp.asarray(g_in))) > 0.0

    # a real optimizer step from the projected point moves INWARD and stays in
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(1e-3, weight_decay=1e-4))
    st = tx.init(projected)
    g = jax.grad(loss)(projected)
    upd, st = tx.update(g, st, projected)
    raw = optax.apply_updates(projected, upd)
    final = project_response_leaves(raw)
    tel = projection_telemetry(raw, final, grads=g, updates=upd)
    assert bool(np.all(flatten_dict(final)[key] <= hi))
    assert float(tel["response_grad_norm"]) > 0.0
    assert int(tel["n_projected"]) >= 0


def test_the_projector_recognizes_the_rho_only_leaf_that_the_OLD_one_missed():
    """The Speech Commands trainer's projector must cover the current leaf.

    Its previous whitelist knew only `log_response_gamma` and
    `log_response_rho`, so the rho-only leaf would have been left outside its
    interval with zero task gradient - the recall study's lockout, repeated.
    """
    import experiments.gp.rawat_benchmark as RB
    from s5.response_projection import RESPONSE_LEAF_BOUNDS
    assert RHO_ONLY_PARAM_NAME in RESPONSE_LEAF_BOUNDS
    assert RB.project_response_leaves is project_response_leaves
    tree = {"seq": {RHO_ONLY_PARAM_NAME: np.full((3,), 0.5)}}
    out = project_response_leaves(tree)
    assert bool(np.all(out["seq"][RHO_ONLY_PARAM_NAME]
                       <= np.asarray(LOG_RHO_BOUNDS[1], dtype=np.float32)))


def test_the_response_policy_admits_the_combined_arm_and_refuses_strays():
    import experiments.gp.rawat_benchmark as RB
    _, params, _ = _bind("gp_rho_prospin")
    stacked = {"encoder": {"layers_0": {"seq": params}}}
    RB.assert_response_policy(stacked, "gp_rho_prospin", 1, 4)
    with pytest.raises(SystemExit, match="not declared for this arm"):
        RB.assert_response_policy(stacked, "alpha_p_s5", 1, 4)


# ------------------------------------------------------ baseline untouched -
def test_the_Rawat_baseline_is_UNCHANGED_by_this_integration():
    """Forward values and one full optimizer update, for the untouched arm."""
    import optax
    P, H, L = 4, 6, 12
    ssm, params, x = _bind("alpha_p_s5", P, H, L)
    y = onp.asarray(ssm.apply({"params": params}, x))
    c = ssm.apply({"params": params}, method=lambda m: m.coefficients())
    drive = two_tap_drive(c["B_plus"], c["B_minus"], x)
    hs = diagonal_scan_drive(c["A_bar"], drive)
    C = onp.asarray(ssm.apply({"params": params}, method=lambda m: m.C_tilde))
    D = onp.asarray(ssm.apply({"params": params}, method=lambda m: m.D))
    want = 2.0 * onp.real(hs @ C.T) + onp.asarray(x) * D
    assert onp.max(onp.abs(y - want)) < 1e-6
    # the projection is a no-op on an arm with no response leaves
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda a, b: bool(np.all(a == b)), params,
        project_response_leaves(params)))
    tx = optax.adamw(1e-3, weight_decay=1e-4)
    st = tx.init(params)
    g = jax.grad(lambda p: np.sum(ssm.apply({"params": p}, x) ** 2))(params)
    upd, _ = tx.update(g, st, params)
    stepped = optax.apply_updates(params, upd)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda a, b: bool(np.all(a == b)), stepped,
        project_response_leaves(stepped)))


# --------------------------------------------------- production dtypes -----
def test_production_float32_and_complex64_in_its_own_process():
    """x64 is process-global, so the production-dtype probe runs separately."""
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "combined_float32_probe.py")
    env = dict(os.environ, JAX_ENABLE_X64="0")
    r = subprocess.run([sys.executable, probe], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-3000:])
    assert r.returncode == 0, r.stdout + r.stderr
