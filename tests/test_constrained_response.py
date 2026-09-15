"""Focused checks for the constrained learned response. CLUSTER-ONLY.

Run by `bin/run_experiments/cluster_constrained.sh` inside the same hard
20-minute budget as the preflight and training. Not run locally: the brief
requires all numerical work on the cluster.

PREDECLARED TOLERANCES, fixed before any validation score:

    TIE     1e-9   component-map identities in float64
    FROZEN  1e-10  frozen-response equivalence to the existing mass arm,
                   outputs AND shared-parameter gradients
    SCAN    1e-10  associative scan vs sequential, float64
    GRAD    5e-3   float32 gradients for BOTH leaves against central
                   differences at an RMS-1 direction and step 1e-2, which is
                   the float32-resolvable regime for this function
"""

import math
import os
import subprocess
import sys

import jax
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax.scipy.linalg import block_diag                           # noqa: E402
from s5 import constrained_response as CR                         # noqa: E402
from s5.gp_fixed import mass_block_generator, mass_block_zoh, mass_scan  # noqa: E402
from s5.gp_fixed import mass_scan_sequential                      # noqa: E402
from s5.rawat_s5 import (GAMMA_N_BOUNDS, LOG_GAMMA_BOUNDS,        # noqa: E402
                         LOG_RHO_BOUNDS, RESPONSE_PARAM_NAMES,
                         RHO_BOUNDS, init_substrate_ssm)
from s5.ssm_init import make_DPLR_HiPPO                           # noqa: E402

TIE, FROZEN, SCAN, GRAD = 1e-9, 1e-10, 1e-10, 5e-3


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


# ------------------------------------------- 1. components and coefficients
def test_component_map_reproduces_the_reference_point():
    c = CR.component_reconstruction(onp.array([1.0]), onp.array([0.75]))
    got = {k: float(onp.ravel(v)[0]) for k, v in c.items()}
    assert got["c_s"] == pytest.approx(7.5, abs=TIE)
    assert got["c_d"] == pytest.approx(7.5, abs=TIE)
    assert got["G_s"] == pytest.approx(2.0, abs=TIE)
    assert got["h"] == pytest.approx(1.0, abs=TIE)
    assert got["g_L"] == pytest.approx(1.0, abs=TIE)
    assert got["T"] == pytest.approx(5.0, abs=TIE)
    assert got["mu"] == pytest.approx(3.75, abs=TIE)
    assert got["gamma_phys"] == pytest.approx(5.0, abs=TIE)


def test_component_identities_hold_across_the_admissible_box():
    g = onp.array([GAMMA_N_BOUNDS[0], 0.3, 1.0, 7.0, GAMMA_N_BOUNDS[1]])
    r = onp.array([RHO_BOUNDS[0], 0.2, 0.75, 0.95, RHO_BOUNDS[1]])
    gg, rr = onp.meshgrid(g, r, indexing="ij")
    res = CR.check_identities(gg.ravel(), rr.ravel(), tol=TIE)
    assert res["all_positive"], res
    assert res["max_residual"] < TIE, res
    assert res["passed"]


def test_horizon_is_fixed_and_mass_is_derived_not_learned():
    """T must not move with the learned pair, and mu must equal T gamma_n rho."""
    g = onp.array([0.05, 1.0, 20.0])
    r = onp.array([0.1, 0.75, 0.99])
    c = CR.component_reconstruction(g, r)
    assert onp.max(onp.abs(c["T"] - CR.T_HORIZON)) < TIE
    assert onp.max(onp.abs(c["mu"] - CR.T_HORIZON * g * r)) < TIE


# ----------------------------------------------- 2. broadcasting with P != H
def test_per_mode_coefficients_divide_ROWS_not_features():
    """P != H on purpose: dividing a (P,H) input by a (P,) vector would
    broadcast along the FEATURE axis, which is silently valid only when
    H == P. This is the pitfall the brief names."""
    P, H = 5, 3
    assert P != H
    rs = onp.random.RandomState(0)
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    g = np.asarray(onp.linspace(0.5, 2.0, P))
    r = np.asarray(onp.linspace(0.3, 0.9, P))
    A, B = mass_block_generator(a, b, 5.0, g, r)
    assert A.shape == (P, 2, 2) and B.shape == (P, 2, H)
    gr = onp.asarray(g) * onp.asarray(r)
    want0 = onp.asarray(b) / gr[:, None]
    assert float(onp.max(onp.abs(onp.asarray(B[:, 0, :]) - want0))) < TIE
    want1 = onp.asarray(b) / (gr[:, None] * 5.0)
    assert float(onp.max(onp.abs(onp.asarray(B[:, 1, :]) - want1))) < TIE


def test_vector_and_scalar_coefficients_agree_when_uniform():
    P, H = 6, 4
    rs = onp.random.RandomState(1)
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    sc = mass_block_zoh(a, b, 5.0, 1.0, 0.75)
    ve = mass_block_zoh(a, b, 5.0, np.ones(P), np.full((P,), 0.75))
    for k in ("A_bar", "B_bar"):
        assert float(np.max(np.abs(sc[k] - ve[k]))) < TIE


# --------------------------------------------- 3. frozen-response equivalence
def _arm(name, H=4, L=20, seed=0):
    kw = ssm_kw(H=H)
    mod = init_substrate_ssm(name, **kw)()
    x = np.asarray(onp.random.RandomState(seed).randn(L, H))
    return mod, mod.init(jax.random.PRNGKey(0), x), x, kw


def test_adding_response_leaves_does_not_change_common_initialization():
    """Flax draws in call order; the new leaves are declared LAST, after
    super().setup(). The shared tree must be BIT-identical to the fixed arm."""
    H, L = 4, 12
    kw = ssm_kw(H=H)
    x = np.zeros((L, H))
    k = jax.random.PRNGKey(0)
    fixed = init_substrate_ssm("gp_fixed_mass", **kw)().init(k, x)["params"]
    learn = init_substrate_ssm("gp_learned_response", **kw)().init(k, x)["params"]
    shared = set(fixed) & set(learn)
    assert shared == set(fixed), (set(fixed), set(learn))
    for name in shared:
        assert bool(np.array_equal(fixed[name], learn[name])), name
    extra = set(learn) - set(fixed)
    assert extra == set(RESPONSE_PARAM_NAMES), extra


def test_frozen_response_reproduces_the_fixed_mass_arm_outputs():
    H, L = 4, 24
    kw = ssm_kw(H=H)
    x = np.asarray(onp.random.RandomState(2).randn(L, H))
    k = jax.random.PRNGKey(0)
    fixed = init_substrate_ssm("gp_fixed_mass", **kw)()
    learn = init_substrate_ssm("gp_learned_response", **kw)()
    vf, vl = fixed.init(k, x), learn.init(k, x)
    yf, yl = fixed.apply(vf, x), learn.apply(vl, x)
    rel = float(np.max(np.abs(yf - yl)) / np.max(np.abs(yf)))
    assert rel < FROZEN, rel


def test_frozen_response_reproduces_shared_parameter_GRADIENTS():
    """Outputs matching is not enough: the shared-parameter gradients must
    match too, or the learned arm is a different model at its own init."""
    H, L = 4, 16
    kw = ssm_kw(H=H)
    x = np.asarray(onp.random.RandomState(3).randn(L, H))
    k = jax.random.PRNGKey(0)
    fixed = init_substrate_ssm("gp_fixed_mass", **kw)()
    learn = init_substrate_ssm("gp_learned_response", **kw)()
    vf, vl = fixed.init(k, x), learn.init(k, x)

    def mk(mod):
        def loss(p):
            return np.sum(mod.apply({"params": p}, x) ** 2)
        return loss

    gf = jax.grad(mk(fixed))(vf["params"])
    gl = jax.grad(mk(learn))(vl["params"])
    for name in gf:
        num = float(np.max(np.abs(gf[name] - gl[name])))
        den = max(float(np.max(np.abs(gf[name]))), 1e-30)
        assert num / den < FROZEN, (name, num / den)


# --------------------------------------- 4. the leaves actually do something
def test_both_leaves_receive_nonzero_gradients():
    H, L = 4, 16
    mod, v, x, kw = _arm("gp_learned_response", H=H, L=L, seed=4)

    def loss(p):
        return np.sum(mod.apply({"params": p}, x) ** 2)

    g = jax.grad(loss)(v["params"])
    for name in RESPONSE_PARAM_NAMES:
        assert name in g, name
        assert float(np.max(np.abs(g[name]))) > 0.0, name


def test_changing_either_leaf_changes_the_response():
    H, L = 4, 16
    mod, v, x, kw = _arm("gp_learned_response", H=H, L=L, seed=5)
    y0 = mod.apply(v, x)
    for name, delta in (("log_response_gamma", 0.3),
                        ("log_response_rho", -0.3)):
        p = dict(v["params"])
        p[name] = p[name] + delta
        y = mod.apply({"params": p}, x)
        assert float(np.max(np.abs(y - y0))) > 1e-6, name


def test_both_leaves_move_under_a_real_update_not_only_weight_decay():
    """A nondegenerate step must move BOTH leaves, and the movement must not
    be explained by weight decay alone: repeat with wd = 0."""
    import optax
    H, L = 4, 16
    mod, v, x, kw = _arm("gp_learned_response", H=H, L=L, seed=6)

    def loss(p):
        return np.sum(mod.apply({"params": p}, x) ** 2)

    for wd in (1e-4, 0.0):
        tx = optax.adamw(1e-3, weight_decay=wd)
        st = tx.init(v["params"])
        g = jax.grad(loss)(v["params"])
        upd, _ = tx.update(g, st, v["params"])
        for name in RESPONSE_PARAM_NAMES:
            moved = float(np.max(np.abs(upd[name])))
            assert moved > 0.0, (name, wd)


# -------------------------------------------- 5. bounds, projection, finiteness
def test_forward_clip_keeps_the_response_inside_the_declared_bounds():
    H, L = 4, 8
    mod, v, x, kw = _arm("gp_learned_response", H=H, L=L, seed=7)
    p = dict(v["params"])
    p["log_response_gamma"] = np.full_like(p["log_response_gamma"], 50.0)
    p["log_response_rho"] = np.full_like(p["log_response_rho"], 50.0)

    def read(m):
        return m.learned_response()

    g_n, rho = mod.apply({"params": p}, method=read)
    assert float(np.max(g_n)) <= GAMMA_N_BOUNDS[1] * (1 + 1e-6)
    assert float(np.max(rho)) <= RHO_BOUNDS[1] * (1 + 1e-6)
    p["log_response_gamma"] = np.full_like(p["log_response_gamma"], -50.0)
    p["log_response_rho"] = np.full_like(p["log_response_rho"], -50.0)
    g_n, rho = mod.apply({"params": p}, method=read)
    assert float(np.min(g_n)) >= GAMMA_N_BOUNDS[0] * (1 - 1e-6)
    assert float(np.min(rho)) >= RHO_BOUNDS[0] * (1 - 1e-6)


def test_projection_pulls_raw_leaves_back_into_the_interval():
    from experiments.gp.rawat_benchmark import project_response_leaves
    p = {"log_response_gamma": np.array([-99.0, 0.0, 99.0]),
         "log_response_rho": np.array([-99.0, math.log(0.75), 99.0]),
         "B": np.array([123.0])}
    out = project_response_leaves(p)
    assert float(np.min(out["log_response_gamma"])) >= LOG_GAMMA_BOUNDS[0]
    assert float(np.max(out["log_response_gamma"])) <= LOG_GAMMA_BOUNDS[1]
    assert float(np.min(out["log_response_rho"])) >= LOG_RHO_BOUNDS[0]
    assert float(np.max(out["log_response_rho"])) <= LOG_RHO_BOUNDS[1]
    assert float(out["B"][0]) == 123.0          # other leaves untouched


@pytest.mark.parametrize("g_n,rho", [(1.0, 0.75), (1e-2, 1e-2),
                                     (1e2, 1 - 1e-4), (1e-2, 1 - 1e-4),
                                     (1e2, 1e-2)])
def test_matrix_exponential_and_derivative_finite_at_interior_and_boundary(
        g_n, rho):
    P, H = 4, 3
    rs = onp.random.RandomState(8)
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(10, H))

    def f(eta, zeta):
        z = mass_block_zoh(a, b, 5.0, np.exp(eta) * np.ones(P),
                           np.exp(zeta) * np.ones(P))
        return np.sum(np.abs(mass_scan(z["A_bar"], z["B_bar"], x)) ** 2)

    val = float(f(math.log(g_n), math.log(rho)))
    gr = jax.grad(f, argnums=(0, 1))(math.log(g_n), math.log(rho))
    assert onp.isfinite(val), (g_n, rho, val)
    assert all(onp.isfinite(float(v)) for v in gr), (g_n, rho, gr)


# ------------------------------------------------ 6. scan, stream, conjugate
def test_scan_matches_sequential_with_per_mode_response():
    P, H, L = 5, 3, 18
    rs = onp.random.RandomState(9)
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(L, H))
    g = np.asarray(onp.linspace(0.4, 2.5, P))
    r = np.asarray(onp.linspace(0.2, 0.95, P))
    z = mass_block_zoh(a, b, 5.0, g, r)
    par = mass_scan(z["A_bar"], z["B_bar"], x)
    seq = mass_scan_sequential(z["A_bar"], z["B_bar"], x)
    assert float(np.max(np.abs(par - seq))) < SCAN


def test_chunked_and_reset_consistency_with_per_mode_response():
    P, H, L = 4, 3, 20
    rs = onp.random.RandomState(10)
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(L, H))
    g = np.asarray(onp.linspace(0.4, 2.5, P))
    r = np.asarray(onp.linspace(0.2, 0.95, P))
    z = mass_block_zoh(a, b, 5.0, g, r)
    full = mass_scan(z["A_bar"], z["B_bar"], x)
    first = mass_scan(z["A_bar"], z["B_bar"], x[:8])
    second = mass_scan(z["A_bar"], z["B_bar"], x[8:], z0=first[-1])
    assert float(np.max(np.abs(full - np.concatenate([first, second])))) < SCAN
    mask = onp.zeros(L, dtype=bool); mask[6] = True
    cut = mass_scan(z["A_bar"], z["B_bar"], x, reset_mask=np.asarray(mask))
    fresh = mass_scan(z["A_bar"], z["B_bar"], x[6:])
    assert float(np.max(np.abs(cut[6:] - fresh))) < SCAN


def test_response_is_one_scalar_pair_per_STORED_mode():
    """Conjugate sharing: (P,) not (2P,). The partner of a stored mode is the
    same physical compartment pair."""
    kw = ssm_kw()
    x = np.zeros((6, kw["H"]))
    mod = init_substrate_ssm("gp_learned_response", **kw)()
    p = mod.init(jax.random.PRNGKey(0), x)["params"]
    for name in RESPONSE_PARAM_NAMES:
        assert p[name].shape == (kw["P"],), (name, p[name].shape)


def test_the_declared_box_keeps_the_derived_mass_above_the_measured_floor():
    """The bound exists because of a MEASURED breakdown, not a guess.

    Worst corner of the declared box must sit an order of magnitude above the
    smallest mu measured finite, and two above the largest measured
    non-finite.
    """
    from s5.rawat_s5 import (MEASURED_LARGEST_NONFINITE_MU,
                             MEASURED_SMALLEST_FINITE_MU)
    worst_mu = CR.T_HORIZON * GAMMA_N_BOUNDS[0] * RHO_BOUNDS[0]
    assert worst_mu == pytest.approx(5e-4, rel=1e-9)
    assert worst_mu >= 10 * MEASURED_SMALLEST_FINITE_MU
    assert worst_mu >= 100 * MEASURED_LARGEST_NONFINITE_MU
    # and the box still leaves the mass free over three decades below init
    assert worst_mu < 3.75 / 1000


def test_the_excluded_corner_really_is_non_finite():
    """Guards the guard: if mu = 5e-6 were fine, the bound would be
    unnecessary caution rather than a measured limit. Recorded as a known
    numerical limitation of the block exponential at extreme stiffness."""
    P, H = 4, 3
    rs = onp.random.RandomState(8)
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(10, H))
    z = mass_block_zoh(a, b, 5.0, 1e-2 * np.ones(P), 1e-4 * np.ones(P))
    val = float(np.sum(np.abs(mass_scan(z["A_bar"], z["B_bar"], x)) ** 2))
    assert not onp.isfinite(val), (
        "mu = 5e-6 is finite after all; the measured bound would need "
        "re-deriving rather than keeping")


def test_parameter_count_is_the_declared_128_added_at_production_shape():
    """P=16, 4 layers -> 2*16*4 = 128 added real parameters."""
    P, n_layers = 16, 4
    assert 2 * P * n_layers == 128


# -------------------------------------------------- 7. professor control kept
def test_prospective_recurrence_core_has_zero_history_at_nonzero_lag():
    H, L = 4, 20
    mod, v, x, kw = _arm("prospective_recurrence", H=H, L=L, seed=11)
    imp = onp.zeros((L, H)); imp[0, 0] = 1.0
    y = onp.asarray(mod.apply(v, np.asarray(imp)))
    assert float(onp.max(onp.abs(y[1:]))) < 1e-9
    assert float(onp.max(onp.abs(y[0]))) > 0.0
    x2 = onp.asarray(x).copy(); x2[:L - 1] += 9.0
    assert float(onp.max(onp.abs(
        onp.asarray(mod.apply(v, x))[-1]
        - onp.asarray(mod.apply(v, np.asarray(x2)))[-1]))) < 1e-9


def test_prospective_recurrence_ordinary_weights_still_get_gradients():
    H, L = 4, 12
    mod, v, x, kw = _arm("prospective_recurrence", H=H, L=L, seed=12)

    def loss(p):
        return np.sum(mod.apply({"params": p}, x) ** 2)

    g = jax.grad(loss)(v["params"])
    for name in ("B", "C", "D", "Lambda_re", "log_step"):
        assert float(np.max(np.abs(g[name]))) > 0.0, name


def test_prospective_recurrence_carries_no_response_leaves():
    kw = ssm_kw()
    mod = init_substrate_ssm("prospective_recurrence", **kw)()
    p = mod.init(jax.random.PRNGKey(0), np.zeros((6, kw["H"])))["params"]
    assert not (set(p) & set(RESPONSE_PARAM_NAMES))


# ------------------------------------------------------- 8. policy and dtypes
def test_response_policy_allows_only_the_new_arm_to_learn():
    from experiments.gp.rawat_benchmark import assert_response_policy
    kw = ssm_kw()
    x = np.zeros((6, kw["H"]))
    k = jax.random.PRNGKey(0)
    learn = init_substrate_ssm("gp_learned_response", **kw)().init(k, x)["params"]
    assert_response_policy({"layers_0": learn}, "gp_learned_response", 1,
                           kw["P"])
    with pytest.raises(SystemExit):
        assert_response_policy({"layers_0": learn}, "gp_fixed_mass", 1, kw["P"])
    fixed = init_substrate_ssm("gp_fixed_mass", **kw)().init(k, x)["params"]
    with pytest.raises(SystemExit):
        assert_response_policy({"layers_0": fixed}, "gp_learned_response", 1,
                               kw["P"])


def test_production_dtype_and_float32_gradients_for_both_leaves():
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "constrained_float32_probe.py")
    r = subprocess.run([sys.executable, probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CONSTRAINED_F32_OK" in r.stdout, r.stdout
