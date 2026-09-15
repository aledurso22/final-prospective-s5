"""Focused checks for the adaptive recurrence arms. CLUSTER-ONLY.

Run by `bin/run_experiments/cluster_adaptive.sh` inside the same hard budget as
the preflight and training. Not run locally: the brief requires all numerical
work on the cluster.

PREDECLARED TOLERANCES, fixed before any validation score:

    TIE        1e-12  algebraic ties among the circuit coefficients (float64)
    FROZEN     1e-10  frozen-modulation identity against the fixed mass arm
    ODE        2e-6   implemented discrete model vs INDEPENDENT scipy
                      integration of the UNREDUCED two-compartment circuit,
                      with both the input and the conductance changing; this is
                      an integration-error tolerance, set by the reference
                      solver's rtol 1e-11 accumulated over the token sequence
    SCAN       1e-10  associative scan vs sequential, float64
    GRAD       5e-3   derivatives through adaptation vs central differences in
                      PRODUCTION float32/complex64, RMS-1 direction, step 1e-2
    F32        1e-4   production dtype against the float64 path
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
from s5 import adaptive_circuit as AC                             # noqa: E402
from s5.rawat_s5 import ARMS, init_substrate_ssm                  # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                           # noqa: E402

TIE, FROZEN, ODE, SCAN, GRAD, F32 = 1e-12, 1e-10, 2e-6, 1e-10, 5e-3, 1e-4
NEW_ARMS = ("gp_adaptive_mass", "gp_frozen_adaptive",
            "prospective_recurrence", "ordinary_adaptive")


def ssm_kw(H=4, ssm=8, blocks=2):
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
                dt_min=0.001, dt_max=0.1, conj_sym=True, bidirectional=False)


def _mod(arm, H=4, L=16, seed=0):
    mod = init_substrate_ssm(arm, **ssm_kw(H=H))()
    x = np.asarray(onp.random.RandomState(seed).randn(L, H))
    return mod, mod.init(jax.random.PRNGKey(0), x), x


# ------------------------------------------------- 1. admissibility and ties
def test_intrinsic_constants_reproduce_the_contract_reference():
    got = AC.validate_reference()
    assert got == pytest.approx(dict(T=5.0, tau_d=3.75, gamma=5.0, M=18.75,
                                     rho=0.75), abs=TIE)


@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.5, 1.0, 3.0])
def test_coefficients_stay_tied_under_adaptation(ratio):
    """T, tau_d, gamma, M, rho are functions of ONE varying quantity, G_d.
    Their defining identities must hold at every modulation level."""
    q = onp.asarray(ratio * AC.G_D0)
    c = AC.coefficients_from_conductance(q)
    G_d = AC.G_D0 + q
    assert float(c["tau_d"]) == pytest.approx(AC.C_MEM / G_d, abs=TIE)
    assert float(c["kappa"]) == pytest.approx(
        AC.G_S - AC.H_COUPLE ** 2 / G_d, abs=TIE)
    assert float(c["T"]) == pytest.approx(AC.C_MEM / c["kappa"], abs=TIE)
    assert float(c["gamma"]) == pytest.approx(
        AC.G_S * c["tau_d"] / c["kappa"], abs=TIE)
    assert float(c["M"]) == pytest.approx(c["T"] * c["tau_d"], abs=TIE)
    # T * kappa = c_s is the identity that makes the (1 + D o T) convention
    # preserve the residual law under adaptation
    assert float(c["T"] * c["kappa"]) == pytest.approx(AC.C_MEM, abs=TIE)
    assert float(c["rho"]) <= 1.0 and float(c["rho"]) > 0.0


def test_conductance_is_nonnegative_bounded_and_uses_existing_weights():
    rs = onp.random.RandomState(1)
    b = np.asarray(rs.randn(6, 5) + 1j * rs.randn(6, 5))
    for scale in (0.0, 1.0, 50.0):
        x = np.asarray(scale * rs.randn(20, 5))
        q = AC.conductance_from_input(b, x)
        q = onp.asarray(q)
        assert q.shape == (20, 6)
        assert onp.all(q >= 0.0)
        assert onp.all(q < AC.MOD_SCALE * AC.G_D0 + 1e-12)   # saturating
    assert onp.all(onp.asarray(
        AC.conductance_from_input(b, np.zeros((3, 5)))) == 0.0)


def test_modulation_is_causal_in_the_layer_input():
    """q_k depends on x_k only: changing a LATER token must not change it."""
    rs = onp.random.RandomState(2)
    b = np.asarray(rs.randn(4, 3) + 1j * rs.randn(4, 3))
    x = onp.asarray(rs.randn(12, 3))
    q1 = onp.asarray(AC.conductance_from_input(b, np.asarray(x)))
    x2 = x.copy(); x2[7:] += 5.0
    q2 = onp.asarray(AC.conductance_from_input(b, np.asarray(x2)))
    assert onp.max(onp.abs(q1[:7] - q2[:7])) == 0.0
    assert onp.max(onp.abs(q1[7:] - q2[7:])) > 0.0


# ------------------------------------------- 2. independent reference circuit
def test_implemented_model_matches_the_unreduced_circuit_integration():
    """Scipy on the UNREDUCED two-compartment circuit versus the implemented
    reduced discrete model, with BOTH input and conductance varying."""
    rs = onp.random.RandomState(3)
    P, H, L = 3, 2, 12
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P))
                   + 1j * rs.uniform(-1, 1, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(L, H))
    q = AC.conductance_from_input(b, x)
    assert float(onp.max(onp.asarray(q))) > 0.0, "fixture must actually adapt"
    assert float(onp.std(onp.asarray(q))) > 0.0, "conductance must VARY"
    A, Bx, d_jump, _ = AC.adaptive_generator(a, b, q)
    A_bar, B_bar = AC.adaptive_zoh(A, Bx)
    zs = AC.adaptive_scan(A_bar, B_bar, d_jump, x)
    s_impl = onp.asarray(zs[..., 0])
    for p in range(P):
        s_ref = AC.reference_discrete_via_unreduced(
            complex(a[p]), onp.asarray(b[p]), onp.asarray(x),
            onp.asarray(q)[:, p])
        scale = max(float(onp.max(onp.abs(s_ref))), 1e-12)
        err = float(onp.max(onp.abs(s_impl[:, p] - s_ref))) / scale
        assert err < ODE, (p, err)


def test_reference_check_would_fail_on_a_frozen_coefficient_shortcut():
    """Guards the guard: if the model ignored the modulation, the unreduced
    reference must reject it. Otherwise the previous test proves nothing."""
    rs = onp.random.RandomState(4)
    P, H, L = 2, 2, 10
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-1, 1, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(3.0 * rs.randn(L, H))
    q = AC.conductance_from_input(b, x)
    A, Bx, dj, _ = AC.adaptive_generator(a, b, np.zeros_like(q))   # SHORTCUT
    A_bar, B_bar = AC.adaptive_zoh(A, Bx)
    s_frozen = onp.asarray(AC.adaptive_scan(A_bar, B_bar, dj, x)[..., 0])
    s_ref = AC.reference_discrete_via_unreduced(
        complex(a[0]), onp.asarray(b[0]), onp.asarray(x), onp.asarray(q)[:, 0])
    scale = max(float(onp.max(onp.abs(s_ref))), 1e-12)
    assert float(onp.max(onp.abs(s_frozen[:, 0] - s_ref))) / scale > 1e-3


# ------------------------------------------------------ 3. frozen-limit claim
def test_frozen_modulation_reproduces_the_fixed_mass_arm():
    """If this holds, the historical fixed-mass score is a valid within-family
    reference; if it fails, `gp_frozen_adaptive` must be trained as its own
    control. Either way it is measured, not assumed."""
    H, L = 4, 24
    kw = ssm_kw(H=H)
    x = np.asarray(onp.random.RandomState(5).randn(L, H))
    k = jax.random.PRNGKey(0)
    froz = init_substrate_ssm("gp_frozen_adaptive", **kw)()
    fixed = init_substrate_ssm("gp_fixed_mass", **kw)()
    vf, vx = froz.init(k, x), fixed.init(k, x)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda p, q: bool(np.array_equal(p, q)), vf, vx)), "params must match"
    yf, yx = froz.apply(vf, x), fixed.apply(vx, x)
    rel = float(np.max(np.abs(yf - yx)) / np.max(np.abs(yx)))
    assert rel < FROZEN, rel


def test_adaptive_actually_differs_from_its_frozen_control():
    H, L = 4, 24
    kw = ssm_kw(H=H)
    x = np.asarray(3.0 * onp.random.RandomState(6).randn(L, H))
    k = jax.random.PRNGKey(0)
    ad = init_substrate_ssm("gp_adaptive_mass", **kw)()
    fr = init_substrate_ssm("gp_frozen_adaptive", **kw)()
    v = ad.init(k, x)
    assert float(np.max(np.abs(ad.apply(v, x) - fr.apply(v, x)))) > 1e-6


# ------------------------------------------------- 4. scan, chunks and resets
def test_block_scan_matches_sequential_under_adaptation():
    rs = onp.random.RandomState(7)
    P, H, L = 4, 3, 20
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(L, H))
    q = AC.conductance_from_input(b, x)
    A, Bx, dj, _ = AC.adaptive_generator(a, b, q)
    A_bar, B_bar = AC.adaptive_zoh(A, Bx)
    par = AC.adaptive_scan(A_bar, B_bar, dj, x)
    seq = AC.adaptive_scan_sequential(A_bar, B_bar, dj, x)
    assert float(np.max(np.abs(par - seq))) < SCAN


def test_reset_drops_the_carry_under_adaptation():
    rs = onp.random.RandomState(8)
    P, H, L = 3, 2, 14
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(L, H))
    q = AC.conductance_from_input(b, x)
    A, Bx, dj, _ = AC.adaptive_generator(a, b, q)
    A_bar, B_bar = AC.adaptive_zoh(A, Bx)
    mask = onp.zeros(L, dtype=bool); mask[6] = True
    cut = AC.adaptive_scan(A_bar, B_bar, dj, x, reset_mask=np.asarray(mask))
    # a reset zeroes the carry; the tail then depends only on tokens >= 6,
    # with the SAME per-token coefficients, so re-running the tail alone with
    # its own coefficients and a zero prehistory must agree
    A2, Bx2, dj2, _ = AC.adaptive_generator(a, b, q[6:])
    A_bar2, B_bar2 = AC.adaptive_zoh(A2, Bx2)
    x2 = np.concatenate([np.zeros_like(x[:1]), x[7:]], axis=0)
    fresh = AC.adaptive_scan(A_bar2, B_bar2, dj2,
                             np.concatenate([x[6:7], x[7:]], axis=0))
    assert float(np.max(np.abs(cut[6:] - fresh))) < SCAN


# --------------------------------------------- 5. professor control behaviour
def test_prospective_recurrence_has_zero_driven_history():
    """r + T r_dot = 0 gives s_k = J^-1 b x_k: no dependence on earlier inputs
    with the current input fixed. Checked on the ISOLATED core, so pooling and
    training-time normalization cannot obscure it."""
    H, L = 4, 20
    mod, v, x = _mod("prospective_recurrence", H=H, L=L)
    y = onp.asarray(mod.apply(v, x))
    x2 = onp.asarray(x).copy()
    x2[:L - 1] += 7.0                      # change all PAST tokens
    y2 = onp.asarray(mod.apply(v, np.asarray(x2)))
    assert float(onp.max(onp.abs(y[-1] - y2[-1]))) < 1e-9
    # and an impulse has no tail beyond lag zero
    imp = onp.zeros((L, H)); imp[0, 0] = 1.0
    yi = onp.asarray(mod.apply(v, np.asarray(imp)))
    assert float(onp.max(onp.abs(yi[1:]))) < 1e-9
    assert float(onp.max(onp.abs(yi[0]))) > 0.0


def test_prospective_recurrence_still_has_nonzero_ordinary_gradients():
    """Its ordinary weights must still learn a static mapping. Low accuracy is
    NOT a correctness condition for this control."""
    H, L = 4, 12
    mod, v, x = _mod("prospective_recurrence", H=H, L=L)

    def loss(params):
        return np.sum(mod.apply({"params": params}, x) ** 2)

    g = jax.grad(loss)(v["params"])
    from flax.traverse_util import flatten_dict
    norms = {"/".join(k): float(np.linalg.norm(val))
             for k, val in flatten_dict(g).items()}
    for name in ("B", "C", "D", "Lambda_re", "log_step"):
        assert norms.get(name, 0.0) > 0.0, (name, norms)


def test_prospective_recurrence_uses_exact_division_not_a_difference():
    """s_k = -(b/a) x_k exactly; a backward difference would add a root."""
    H, L = 3, 8
    mod, v, x = _mod("prospective_recurrence", H=H, L=L)
    c = mod.apply(v, method=lambda m: m.coefficients())
    a, b = onp.asarray(c["a"]), onp.asarray(c["b"])
    assert float(onp.max(onp.abs(onp.asarray(c["s_gain"])
                                 - (-b / a[:, None])))) < 1e-12


# ------------------------------------------------- 6. gradients and dtypes
@pytest.mark.parametrize("arm", ["gp_adaptive_mass", "ordinary_adaptive"])
def test_gradients_flow_through_the_modulation_path(arm):
    """The coefficient/modulation path must NOT be detached: perturbing only
    the modulation's own dependence must change the gradient."""
    rs = onp.random.RandomState(9)
    H, L = 3, 10
    mod, v, x = _mod(arm, H=H, L=L, seed=9)

    def loss(xx):
        return np.sum(mod.apply(v, xx) ** 2)

    g = onp.asarray(jax.grad(loss)(x))
    assert onp.all(onp.isfinite(g)) and float(onp.max(onp.abs(g))) > 0.0
    d = rs.randn(*x.shape); d = np.asarray(d / d.std())
    h = 1e-4
    fd = (float(loss(x + h * d)) - float(loss(x - h * d))) / (2 * h)
    ana = float(np.sum(g * d))
    assert abs(ana - fd) / max(abs(fd), 1e-9) < 1e-4, (ana, fd)


@pytest.mark.parametrize("arm", NEW_ARMS)
def test_production_dtype_and_float32_gradients(arm):
    """Actual production float32/complex64, in a subprocess with x64 off."""
    import subprocess
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "adaptive_float32_probe.py")
    r = subprocess.run([sys.executable, probe, arm], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ADAPTIVE_F32_OK" in r.stdout, r.stdout


@pytest.mark.parametrize("arm", NEW_ARMS)
def test_new_arms_add_no_trainable_parameters(arm):
    from flax.traverse_util import flatten_dict
    kw = ssm_kw()
    x = np.zeros((8, kw["H"]))
    k = jax.random.PRNGKey(0)
    ref = init_substrate_ssm("gp_fixed_mass", **kw)()
    new = init_substrate_ssm(arm, **kw)()
    nref = sum(int(a.size) for a in
               jax.tree_util.tree_leaves(ref.init(k, x)["params"]))
    p = new.init(k, x)["params"]
    nnew = sum(int(a.size) for a in jax.tree_util.tree_leaves(p))
    assert nnew == nref, (arm, nnew, nref)
    assert set("/".join(q) for q in flatten_dict(p)) == {
        "B", "C", "D", "Lambda_im", "Lambda_re", "log_step"}


@pytest.mark.parametrize("arm", NEW_ARMS)
def test_state_counts_are_reported_for_the_new_arms(arm):
    kw = ssm_kw()
    x = np.zeros((8, kw["H"]))
    mod = init_substrate_ssm(arm, **kw)()
    v = mod.init(jax.random.PRNGKey(0), x)
    c = mod.apply(v, method=lambda m: m.state_counts())
    if arm == "prospective_recurrence":
        assert c["total_real"] == 0
    elif arm in ("gp_adaptive_mass", "gp_frozen_adaptive"):
        assert c["auxiliary_real"] == c["physical_real"] > 0
    else:
        assert c["auxiliary_real"] == 0 and c["physical_real"] > 0


def test_unknown_law_is_rejected_not_dispatched_to_the_mass_case():
    from s5.rawat_s5 import SubstrateSSM
    kw = ssm_kw()
    with pytest.raises(ValueError):
        SubstrateSSM(response="not_a_law", clip_eigs=True,
                     **kw).init(jax.random.PRNGKey(0), np.zeros((4, kw["H"])))
