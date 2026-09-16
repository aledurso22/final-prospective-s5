"""Focused checks for the stable generalized prospective continuation. CLUSTER.

Run by `bin/run_experiments/cluster_stable_gp.sh` inside the single 1200 s cap,
before any training. Never run locally.

PREDECLARED TOLERANCES (frozen before any execution):

    STAB64   eigenvalue sign agrees with S at rho = rho_max (1 -+ 1e-3)
    POLY64   1e-10   relative: characteristic polynomial and transfer identity
    COEF64   1e-10   relative: production two-tap block vs independent reference
    IDENT64  1e-9    relative: arm identities (logits, layer outputs)
    GRAD64   1e-8    relative per leaf: shared-parameter and input gradients
    UPD64    1e-8    max |u_x - u_ref| / lr: first update, added leaves frozen
    FD64     1e-6    relative: JVP vs central differences at BOTH h = 1e-5, 1e-6
    ZEROT64  1e-9    |dL/dt| relative to |dL/dq| at rho = 1
    STREAM64 1e-10   relative: chunked carries and resets vs one call
    TSS64    1e-12   absolute: equation-level discrete references
    TAN64    1e-8    relative: dG/dr at r = 0 vs the displayed closed form
    RES64    1e-5    relative: auxiliary-pole residue at delta = 1e-9
    FD_R64   1e-6    relative: r-ONLY JVP at the actual start, h = 1e-5, 1e-6

AMENDED BEFORE EXECUTION (review of 97cedfa, R0-R4): recurrent reference
horizon 10 (input 5); full-ratio projection margin with rho = 1 fallback;
r-only derivative at the actual start; mixed absolute/relative gradient
identity; optimizer-routing identity with copied gradients; executed-arithmetic
domain validation. See docs/STABLE_GP_CONTINUATION_PROTOCOL.md section 13.

Production float32/complex64 coverage runs in `stable_gp_float32_probe.py`, in
its own process with x64 off, using the study's declared gate tolerances.
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

from s5 import stable_gp as SG                                    # noqa: E402
from s5.gp_fixed import mass_block_generator                      # noqa: E402
from s5.rawat_s5 import STABLE_GP_RESPONSES, init_substrate_ssm   # noqa: E402
from tests.response_reference import (reference_block,            # noqa: E402
                                      ssm_kwargs as _ssm_kwargs)

POLY64, COEF64, IDENT64, GRAD64 = 1e-10, 1e-10, 1e-9, 1e-8
UPD64, FD64, ZEROT64, STREAM64, TSS64 = 1e-8, 1e-6, 1e-9, 1e-10, 1e-12
FD_STEPS = (1e-5, 1e-6)
TAN64, RES64, FD_R64 = 1e-8, 1e-5, 1e-6


def _cast(tree, dt=onp.float64):
    return jax.tree_util.tree_map(
        lambda v: (np.asarray(v, dtype=dt)
                   if np.issubdtype(np.asarray(v).dtype, np.floating) else v),
        tree)


def _rel(a, b):
    a = onp.asarray(a); b = onp.asarray(b)
    n = float(onp.sqrt(onp.sum(onp.abs(b) ** 2)))
    d = float(onp.sqrt(onp.sum(onp.abs(a - b) ** 2)))
    return d / n if n > 0 else d


def _set(params, name, fn):
    """Replace every leaf called `name` by fn(layer_index, old)."""
    from flax.traverse_util import flatten_dict, unflatten_dict
    flat = flatten_dict(params)
    i = 0
    for k in sorted(flat):
        if k[-1] == name:
            flat[k] = np.asarray(fn(i, flat[k]), dtype=flat[k].dtype)
            i += 1
    return unflatten_dict(flat)


# =========================================================================
#  1. Stability domain against the EXECUTED generator
# =========================================================================
def _random_modes(rs, P):
    # ranges keep the boundary crossing resolvable by a float64 eigensolver:
    # near rho_max the crossing root moves by ~ delta / (rho T), which must
    # stay far above eps * ||A||
    a = onp.exp(rs.uniform(math.log(1e-2), math.log(3.0), P))
    omega = rs.choice([-1.0, 1.0], P) * onp.exp(rs.uniform(math.log(5e-2),
                                                           math.log(5.0), P))
    T = onp.exp(rs.uniform(math.log(0.2), math.log(50.0), P))
    return a, omega, T


def _executed_poles(a, omega, T, rho):
    """Eigenvalues of production `mass_block_generator` for j = a + i omega.

    Production takes the clock-absorbed `a_code = Delta lambda = -j`."""
    P = a.size
    a_code = np.asarray(-(a + 1j * omega))
    b = np.ones((P, 1), dtype=complex)
    A, _ = mass_block_generator(a_code, b, np.asarray(T), np.ones(P),
                                np.asarray(rho))
    return onp.linalg.eigvals(onp.asarray(A))


def test_the_characteristic_polynomial_of_the_executed_block_is_the_law():
    """rho T det(pI - A) = rho T p^2 + (1 + T j) p + j, at random p."""
    rs = onp.random.RandomState(1)
    P = 7
    a, omega, T = _random_modes(rs, P)
    rho = onp.exp(rs.uniform(-2, 1.5, P))
    a_code = np.asarray(-(a + 1j * omega))
    A, _ = mass_block_generator(a_code, np.ones((P, 1), dtype=complex),
                                np.asarray(T), np.ones(P), np.asarray(rho))
    A = onp.asarray(A)
    j = a + 1j * omega
    for p in (0.3 + 0.7j, -1.1 + 2.0j, 4.0):
        got = rho * T * onp.array([onp.linalg.det(p * onp.eye(2) - A[k])
                                   for k in range(P)])
        want = rho * T * p * p + (1 + T * j) * p + j
        assert _rel(got, want) < POLY64, p


def test_the_transfer_identity_and_its_rho_one_factorization():
    """[(pI - A)^-1 B]_s (1 + T_in p) = G(p); at rho = 1 the denominator is
    (1 + T p)(p + j) for EVERY T, which is Rawat's input-horizon response."""
    rs = onp.random.RandomState(2)
    P, H = 5, 3
    a, omega, T = _random_modes(rs, P)
    b = rs.randn(P, H) + 1j * rs.randn(P, H)
    T_in = onp.exp(rs.uniform(0.5, 2.5, P))
    j = a + 1j * omega
    for rho in (onp.exp(rs.uniform(-1, 1, P)), onp.ones(P)):
        A, B = mass_block_generator(np.asarray(-j), np.asarray(b),
                                    np.asarray(T), np.ones(P), np.asarray(rho))
        A, B = onp.asarray(A), onp.asarray(B)
        for p in (0.2 + 0.5j, 1.5 - 0.3j):
            got = onp.stack([onp.linalg.solve(p * onp.eye(2) - A[k], B[k])[0]
                             for k in range(P)]) * (1 + T_in * p)[:, None]
            want = SG.transfer(p, j[:, None], b, T[:, None], rho[:, None],
                               T_in[:, None])
            assert _rel(got, want) < POLY64
            if onp.all(rho == 1.0):
                rawat = b * (1 + T_in * p)[:, None] / (p + j)[:, None]
                assert _rel(got, rawat) < POLY64


def test_analytic_domain_matches_executed_eigenvalues_on_both_sides():
    """Inside and outside rho_max, for complex modes; every rho for real."""
    rs = onp.random.RandomState(3)
    P = 40
    a, omega, T = _random_modes(rs, P)
    rmax = SG.rho_max_float64(a, omega, T)
    assert onp.all(rmax > 1.0), "rho = 1 must always be strictly inside"
    for side, fac in (("inside", 1.0 - 1e-3), ("outside", 1.0 + 1e-3)):
        rho = rmax * fac
        S = SG.stability_S(a, omega, T, rho)
        ev = _executed_poles(a, omega, T, rho).reshape(P, 2)
        stable = onp.max(ev.real, axis=1) < 0
        assert onp.all(stable == (S > 0)), side
        assert onp.all(stable) if side == "inside" else not onp.any(stable)
    # omega = 0: every positive rho stable, including very large mass
    for rho in (1e-3, 0.5, 1.0, 10.0, 1e4):
        ev = _executed_poles(a, onp.zeros(P), T, onp.full(P, rho))
        assert onp.max(ev.real) < 0, rho
        assert onp.all(onp.isinf(SG.rho_max_float64(a, onp.zeros(P), T)))


def test_the_old_passive_sector_is_a_strict_subset_of_the_domain():
    """rho <= 1 (M <= gamma T) is inside, and the domain extends beyond it."""
    rs = onp.random.RandomState(4)
    a, omega, T = _random_modes(rs, 30)
    assert onp.all(SG.stability_S(a, omega, T, onp.ones(30)) > 0)
    rmax = SG.rho_max_float64(a, omega, T)
    mid = 0.5 * (1.0 + rmax)
    assert onp.all(mid > 1.0) and onp.all(SG.stability_S(a, omega, T, mid) > 0)


# =========================================================================
#  2. The projection bound and its log arithmetic
# =========================================================================
def test_log_bound_equals_the_closed_form_interior_float64():
    rs = onp.random.RandomState(5)
    P = 50
    a, omega, T = _random_modes(rs, P)
    got = onp.asarray(SG.log_rho_upper(np.asarray(a), np.asarray(omega),
                                       np.asarray(T), onp.float64))
    c = 1 + T * a
    z = T * a + a * c ** 2 / (T * omega ** 2)
    Lz = onp.log1p(z)
    want = onp.maximum(0.0, Lz - SG.eps_num(onp.float64) * (1 + onp.abs(Lz)))
    assert onp.max(onp.abs(got - want) / onp.maximum(1.0, onp.abs(want))) \
        < 1e-12
    assert onp.all(onp.isinf(onp.asarray(SG.log_rho_upper(
        np.asarray(a), np.zeros(P), np.asarray(T), onp.float64))))


def _layer_tree(rs, P, r=None, t=None, lam_re=None, lam_im=None, log_step=None):
    lam_re = -onp.exp(rs.uniform(-3, 1, P)) if lam_re is None else lam_re
    lam_im = rs.uniform(-3, 3, P) if lam_im is None else lam_im
    log_step = (onp.log(rs.uniform(1e-3, 1e-1, P))[:, None]
                if log_step is None else log_step)
    # PRODUCTION float32 leaves: executed_domain_report checks the executed
    # generator's eigenvalues, and a float64 fixture projected to its own
    # 32 eps64 interior would sit at the eigensolver's resolution
    f32 = onp.float32
    return {"encoder": {"layers_0": {"seq": {
        "Lambda_re": np.asarray(lam_re, f32), "Lambda_im": np.asarray(lam_im, f32),
        "log_step": np.asarray(log_step, f32),
        SG.LEAF_T: np.asarray(onp.zeros(P) if t is None else t, f32),
        SG.LEAF_RHO: np.asarray(onp.zeros(P) if r is None else r, f32),
        SG.LEAF_T_IN: np.asarray(onp.zeros(P), f32)}}}}


def test_projection_uses_the_UPDATED_complete_layer_and_only_moves_r():
    """Simultaneous updates to the poles, the clock and T change the feasible
    set; the bound must be recomputed from all of them, not a fixed interval."""
    rs = onp.random.RandomState(6)
    P = 16
    tree = _layer_tree(rs, P, t=rs.uniform(-1, 1, P))
    seq = tree["encoder"]["layers_0"]["seq"]
    # propose an outward r against the UPDATED leaves
    a, w = SG.modal_j(seq["Lambda_re"], seq["Lambda_im"], seq["log_step"])
    T = SG.RECURRENT_T_REFERENCE * np.exp(seq[SG.LEAF_T])
    bound = SG.log_rho_upper(a, w, T, onp.float32)
    seq[SG.LEAF_RHO] = bound + 0.5
    out, tel = SG.project_stable_domain(tree)
    s2 = out["encoder"]["layers_0"]["seq"]
    assert onp.allclose(onp.asarray(s2[SG.LEAF_RHO]), onp.asarray(bound))
    for k in ("Lambda_re", "Lambda_im", "log_step", SG.LEAF_T):
        assert onp.array_equal(onp.asarray(s2[k]), onp.asarray(seq[k])), k
    assert int(tel["n_projected"]) == P
    assert float(tel["max_overshoot"]) == pytest.approx(0.5, rel=1e-5)
    assert SG.executed_domain_report(out)["passed"]
    # a different update of T and of the poles moves the bound with it
    seq2 = dict(s2, **{SG.LEAF_T: s2[SG.LEAF_T] + 1.0,
                       "Lambda_im": s2["Lambda_im"] * 3.0})
    tree2 = {"encoder": {"layers_0": {"seq": seq2}}}
    a2, w2 = SG.modal_j(seq2["Lambda_re"], seq2["Lambda_im"], seq2["log_step"])
    b2 = SG.log_rho_upper(a2, w2, SG.RECURRENT_T_REFERENCE * np.exp(seq2[SG.LEAF_T]),
                          onp.float32)
    out2, _ = SG.project_stable_domain(tree2)
    assert onp.allclose(onp.asarray(out2["encoder"]["layers_0"]["seq"]
                                    [SG.LEAF_RHO]),
                        onp.minimum(onp.asarray(seq2[SG.LEAF_RHO]),
                                    onp.asarray(b2)))


def test_inward_points_and_real_modes_are_never_moved():
    rs = onp.random.RandomState(7)
    P = 12
    lam_im = rs.uniform(-3, 3, P)
    lam_im[:4] = 0.0                                   # exactly real modes
    tree = _layer_tree(rs, P, lam_im=lam_im,
                       r=onp.concatenate([onp.full(4, 30.0),
                                          rs.uniform(-2, 0, P - 4)]))
    out, tel = SG.project_stable_domain(tree)
    assert onp.array_equal(onp.asarray(out["encoder"]["layers_0"]["seq"]
                                       [SG.LEAF_RHO]),
                           onp.asarray(tree["encoder"]["layers_0"]["seq"]
                                       [SG.LEAF_RHO]))
    assert int(tel["n_projected"]) == 0
    assert int(tel["n_real_modes"]) == 4


def test_projection_is_a_no_op_for_arms_without_the_rho_leaf():
    tree = {"encoder": {"layers_0": {"seq": {"Lambda_re": np.ones(3),
                                             SG.LEAF_T_IN: np.ones(3)}}}}
    out, tel = SG.project_stable_domain(tree)
    assert out is tree and int(tel["n_projected"]) == 0


# =========================================================================
#  3. Layer level: exact identities, carries, resets, first token
# =========================================================================
def _layer(arm, P=4, H=6, L=14, seed=0):
    ssm = init_substrate_ssm(arm, **_ssm_kwargs(P, H))()
    x = np.asarray(onp.random.RandomState(seed).randn(L, H))
    params = _cast(ssm.init(jax.random.PRNGKey(seed), x)["params"])
    return ssm, params, x


def test_leaves_are_declared_after_the_common_tree_zero_float32():
    from flax.traverse_util import flatten_dict
    for arm, leaves in STABLE_GP_RESPONSES.items():
        ssm = init_substrate_ssm(arm, **_ssm_kwargs(4, 6))()
        x = np.asarray(onp.random.RandomState(0).randn(8, 6))
        raw = ssm.init(jax.random.PRNGKey(0), x)["params"]
        names = list(raw)
        assert names[-len(leaves):] == list(leaves), (arm, names)
        for n in leaves:
            assert raw[n].shape == (4,) and raw[n].dtype == onp.float32
            assert float(onp.max(onp.abs(onp.asarray(raw[n])))) == 0.0
        ref = init_substrate_ssm("alpha_p_s5", **_ssm_kwargs(4, 6))()
        rp = flatten_dict(ref.init(jax.random.PRNGKey(0), x)["params"])
        fp = flatten_dict(raw)
        for k, v in rp.items():
            assert onp.array_equal(onp.asarray(v), onp.asarray(fp[k])), k


@pytest.mark.parametrize("P,H", [(4, 6), (5, 3)])
def test_production_block_coefficients_match_the_independent_reference(P, H):
    """Per-mode T, rho (including rho > 1 inside the domain) and T_in, P != H."""
    rs = onp.random.RandomState(P + 10 * H)
    ssm, params, x = _layer("sgp_learned_input", P=P, H=H)
    params = _set(params, SG.LEAF_T, lambda i, v: rs.uniform(-1, 1, P))
    params = _set(params, SG.LEAF_T_IN, lambda i, v: rs.uniform(-1, 1, P))
    params = _set(params, SG.LEAF_RHO, lambda i, v: rs.uniform(-1, 0.4, P))
    params, _ = SG.project_stable_domain({"x": params})
    params = params["x"]
    c = ssm.apply({"params": params}, method=lambda m: m.coefficients())
    a, b = ssm.apply({"params": params}, method=lambda m: m.clock_absorbed())
    T = SG.RECURRENT_T_REFERENCE * onp.exp(onp.asarray(params[SG.LEAF_T]))
    rho = onp.exp(onp.asarray(params[SG.LEAF_RHO]))
    T_in = SG.INPUT_T_REFERENCE * onp.exp(onp.asarray(params[SG.LEAF_T_IN]))
    ref = reference_block(onp.asarray(a), onp.asarray(b), T, rho)
    J_in = T_in[:, None, None] * onp.einsum("pij,pjh->pih", ref["A_bar"],
                                            ref["B"])
    for k, want in (("A_bar", ref["A_bar"]), ("B_bar", ref["B_bar"]),
                    ("J_in", J_in), ("B_plus", ref["B_bar"] + J_in),
                    ("B_minus", -J_in)):
        got = onp.asarray(c[k])
        assert _rel(got, want) < COEF64, k


@pytest.mark.parametrize("q_scale", [0.0, 0.7])
def test_C_at_rho_one_is_B_for_every_q_and_t_forward_and_gradient(q_scale):
    rs = onp.random.RandomState(11)
    sb, pb, x = _layer("rawat_learned_input")
    sc, pc, _ = _layer("sgp_learned_input")
    q = rs.uniform(-1, 1, 4) * q_scale
    pb = _set(pb, SG.LEAF_T_IN, lambda i, v: q)
    pc = _set(pc, SG.LEAF_T_IN, lambda i, v: q)
    for t in (onp.zeros(4), rs.uniform(-2, 2, 4)):
        pc_t = _set(pc, SG.LEAF_T, lambda i, v: t)
        yb = sb.apply({"params": pb}, x)
        yc = sc.apply({"params": pc_t}, x)
        assert _rel(yc, yb) < IDENT64
        w = np.asarray(rs.randn(*yb.shape))

        def lb(p, xx):
            return np.sum(w * sb.apply({"params": p}, xx))

        def lc(p, xx):
            return np.sum(w * sc.apply({"params": p}, xx))
        gb, gxb = jax.grad(lb, argnums=(0, 1))(pb, x)
        gc, gxc = jax.grad(lc, argnums=(0, 1))(pc_t, x)
        assert _rel(gxc, gxb) < GRAD64
        for k in gb:
            assert _rel(gc[k], gb[k]) < GRAD64, k      # includes log_T_in (q)
        # T is unidentifiable at rho = 1: its task gradient vanishes
        assert float(onp.max(onp.abs(onp.asarray(gc[SG.LEAF_T])))) <= \
            ZEROT64 * max(1.0, float(onp.max(onp.abs(onp.asarray(
                gc[SG.LEAF_T_IN])))))


def test_B_at_q_zero_is_Rawat_forward_and_gradient():
    sa, pa, x = _layer("alpha_p_s5")
    sb, pb, _ = _layer("rawat_learned_input")
    ya, yb = sa.apply({"params": pa}, x), sb.apply({"params": pb}, x)
    assert _rel(yb, ya) < IDENT64
    w = np.asarray(onp.random.RandomState(3).randn(*ya.shape))
    ga, gxa = jax.grad(lambda p, xx: np.sum(w * sa.apply({"params": p}, xx)),
                       argnums=(0, 1))(pa, x)
    gb, gxb = jax.grad(lambda p, xx: np.sum(w * sb.apply({"params": p}, xx)),
                       argnums=(0, 1))(pb, x)
    assert _rel(gxb, gxa) < GRAD64
    for k in ga:
        assert _rel(gb[k], ga[k]) < GRAD64, k


def test_rho_one_decouples_the_first_row_even_at_the_first_token():
    """At rho = 1 the s-row of A_bar has no v entry, so zero prehistory in v
    cannot leak into s at startup."""
    rs = onp.random.RandomState(12)
    sc, pc, x = _layer("sgp_learned_input")
    pc = _set(pc, SG.LEAF_T, lambda i, v: rs.uniform(-1, 1, 4))
    c = sc.apply({"params": pc}, method=lambda m: m.coefficients())
    assert float(onp.max(onp.abs(onp.asarray(c["A"])[:, 0, 1]))) == 0.0
    assert float(onp.max(onp.abs(onp.asarray(c["A_bar"])[:, 0, 1]))) < 1e-14


def _carry_B(ssm, params, x):
    c = ssm.apply({"params": params}, method=lambda m: m.coefficients())
    h = onp.zeros(onp.asarray(c["A_bar"]).shape, dtype=complex)
    xp = onp.zeros(x.shape[1])
    for k in range(x.shape[0]):
        h = (onp.asarray(c["A_bar"]) * h + onp.asarray(c["B_plus"]) @ x[k]
             + onp.asarray(c["B_minus"]) @ xp)
        xp = onp.asarray(x[k])
    return np.asarray(h)


def _carry_C(ssm, params, x):
    c = ssm.apply({"params": params}, method=lambda m: m.coefficients())
    A = onp.asarray(c["A_bar"]); Bp = onp.asarray(c["B_plus"])
    Bm = onp.asarray(c["B_minus"])
    z = onp.zeros((A.shape[0], 2), dtype=complex)
    xp = onp.zeros(x.shape[1])
    for k in range(x.shape[0]):
        z = onp.einsum("pij,pj->pi", A, z) + Bp @ x[k] + Bm @ xp
        xp = onp.asarray(x[k])
    return np.asarray(z)


@pytest.mark.parametrize("arm", ["rawat_learned_input", "sgp_learned_input"])
def test_streaming_carries_and_resets_clear_BOTH_state_and_delayed_input(arm):
    rs = onp.random.RandomState(13)
    ssm, params, x = _layer(arm, L=16)
    params = _set(params, SG.LEAF_T_IN, lambda i, v: rs.uniform(-1, 1, 4))
    if arm == "sgp_learned_input":
        params = _set(params, SG.LEAF_T, lambda i, v: rs.uniform(-1, 1, 4))
        params = _set(params, SG.LEAF_RHO, lambda i, v: rs.uniform(-1, 0.3, 4))
        params = SG.project_stable_domain({"x": params})[0]["x"]
    full = onp.asarray(ssm.apply({"params": params}, x))
    k = 7
    carry = (_carry_B if arm == "rawat_learned_input" else _carry_C)(
        ssm, params, onp.asarray(x[:k]))
    tail = onp.asarray(ssm.apply({"params": params}, x[k:], z0=carry,
                                 prev_x=x[k - 1]))
    assert _rel(tail, full[k:]) < STREAM64
    # a reset at k behaves exactly like a fresh call on x[k:], and that holds
    # only if the DELAYED INPUT is cleared as well as the state
    mask = onp.zeros(16, dtype=bool); mask[k] = True
    reset = onp.asarray(ssm.apply({"params": params}, x,
                                  reset_mask=np.asarray(mask)))
    fresh = onp.asarray(ssm.apply({"params": params}, x[k:]))
    assert _rel(reset[k:], fresh) < STREAM64
    assert _rel(reset[:k], full[:k]) < STREAM64


# =========================================================================
#  4. Full network: logits, gradients, a real tangent, frozen-extra updates
# =========================================================================
def _net(arm, P=4, H=6, L=20, seed=0, dt=onp.float64):
    from s5.rawat_model import RawatClassifier
    m = RawatClassifier(ssm=init_substrate_ssm(arm, **_ssm_kwargs(P, H)),
                        d_model=H, n_layers=2, d_output=3, readout_width=5,
                        mlp_hidden=7, training=False)
    x = np.asarray(onp.random.RandomState(seed + 1).randn(L, 2), dt)
    v = _cast(m.init(jax.random.PRNGKey(seed), x, np.ones(L, dt)), dt)
    return m, v, x


def _loss(m, v_bs, w):
    def f(params, xx):
        y = m.apply({"params": params, "batch_stats": v_bs}, xx,
                    np.ones(xx.shape[0]))
        return np.sum(w * y)
    return f


def test_full_network_logits_and_shared_gradients_nest_A_B_C():
    rs = onp.random.RandomState(21)
    ma, va, x = _net("alpha_p_s5")
    mb, vb, _ = _net("rawat_learned_input")
    mc, vc, _ = _net("sgp_learned_input")
    w = np.asarray(rs.randn(3))
    q = rs.uniform(-1, 1, 4)
    t = rs.uniform(-1, 1, 4)
    pb_q = _set(vb["params"], SG.LEAF_T_IN, lambda i, v: q)
    pc_q = _set(_set(vc["params"], SG.LEAF_T_IN, lambda i, v: q),
                SG.LEAF_T, lambda i, v: t)
    cases = (("B(q=0) vs A", mb, vb["params"], ma, va["params"], vb, va),
             ("C(r=0, q, t) vs B(q)", mc, pc_q, mb, pb_q, vc, vb))
    for name, mx, px, mr, pr, vx, vr in cases:
        fx, fr = _loss(mx, vx["batch_stats"], w), _loss(mr, vr["batch_stats"], w)
        assert _rel(mx.apply({"params": px, "batch_stats": vx["batch_stats"]},
                             x, np.ones(x.shape[0])),
                    mr.apply({"params": pr, "batch_stats": vr["batch_stats"]},
                             x, np.ones(x.shape[0]))) < IDENT64, name
        gx, gxx = jax.grad(fx, argnums=(0, 1))(px, x)
        gr, grx = jax.grad(fr, argnums=(0, 1))(pr, x)
        assert _rel(gxx, grx) < GRAD64, name
        from flax.traverse_util import flatten_dict
        fgx, fgr = flatten_dict(gx), flatten_dict(gr)
        for k, v in fgr.items():
            assert _rel(fgx[k], v) < GRAD64, (name, "/".join(k))


def test_a_nonzero_response_tangent_matches_central_differences():
    """Away from rho = 1, along a random direction in (q, r, t) of every layer.
    No static bypass and no straight-through estimator exists to hide behind."""
    from flax.traverse_util import flatten_dict, unflatten_dict
    rs = onp.random.RandomState(22)
    mc, vc, x = _net("sgp_learned_input")
    p = _set(vc["params"], SG.LEAF_RHO, lambda i, v: rs.uniform(-0.6, 0.3, 4))
    p = _set(p, SG.LEAF_T, lambda i, v: rs.uniform(-0.5, 0.5, 4))
    p = _set(p, SG.LEAF_T_IN, lambda i, v: rs.uniform(-0.5, 0.5, 4))
    p = SG.project_stable_domain(p)[0]
    w = np.asarray(rs.randn(3))
    f = _loss(mc, vc["batch_stats"], w)
    flat = flatten_dict(p)
    d = {k: (np.asarray(rs.randn(*onp.shape(v))) if k[-1] in SG.ADDED_LEAVES
             else np.zeros_like(v)) for k, v in flat.items()}
    dtree = unflatten_dict(d)
    jvp = float(jax.jvp(lambda pp: f(pp, x), (p,), (dtree,))[1])
    assert abs(jvp) > 1e-8, "the tangent must be genuinely nonzero"
    for h in FD_STEPS:
        plus = jax.tree_util.tree_map(lambda a, b: a + h * b, p, dtree)
        minus = jax.tree_util.tree_map(lambda a, b: a - h * b, p, dtree)
        fd = (float(f(plus, x)) - float(f(minus, x))) / (2 * h)
        assert abs(jvp - fd) / abs(fd) < FD64, (h, jvp, fd)
    # and T is identifiable here: its gradient is not zero
    g = flatten_dict(jax.grad(lambda pp: f(pp, x))(p))
    tg = max(float(onp.max(onp.abs(onp.asarray(v))))
             for k, v in g.items() if k[-1] == SG.LEAF_T)
    assert tg > 1e-8, tg


def test_routing_identity_uses_copied_gradients_and_labels_are_published():
    """Review R2: the optimizer-routing gate feeds the SAME shared gradient
    values into both trees (extra leaves frozen), so it tests grouping, decay
    and clipping routing - not gradient rounding."""
    from experiments.gp import stable_gp_study as ST
    from flax.traverse_util import flatten_dict
    rs = onp.random.RandomState(23)
    mb, vb, x = _net("rawat_learned_input")
    mc, vc, _ = _net("sgp_learned_input")
    w = np.asarray(rs.randn(3)) * 50.0          # large enough to engage clip
    tx, desc = ST.make_optimizer(steps_per_epoch=10, epochs=10)
    gb = jax.grad(_loss(mb, vb["batch_stats"], w))(vb["params"], x)
    r = ST.routing_identity(tx, vc["params"], vb["params"], gb)
    assert r["worst"] < UPD64 and r["passed"], r
    # the routed tree of C carries B's q gradient and zero on r and t
    routed = flatten_dict(ST.routed_gradients(vc["params"], gb))
    for k, v in routed.items():
        if k[-1] in (SG.LEAF_RHO, SG.LEAF_T):
            assert float(onp.max(onp.abs(onp.asarray(v)))) == 0.0
    assert desc["labels"]["response"].startswith("Adam(")
    for k, lab in flatten_dict(ST.label_tree(vc["params"])).items():
        assert lab == ("response" if k[-1] in SG.ADDED_LEAVES else "common"), k


def test_mixed_identity_accepts_rounding_and_catches_a_real_defect():
    """Review R2 fixtures for the absolute branch, exercised on the actual
    comparison function with both dtypes' floors."""
    from experiments.gp import stable_gp_study as ST
    rs = onp.random.RandomState(26)
    ref = {"a": rs.randn(50), "b": rs.randn(30), "zero": onp.zeros(8)}
    G = float(onp.sqrt(sum(onp.sum(v ** 2) for v in ref.values())))
    for dt in (onp.float32, onp.float64):
        eps = float(onp.finfo(dt).eps)
        # analytical zero-gradient leaf with rounding-level residuals: accepted
        ok = dict(ref, zero=onp.full(8, 10 * eps * G / 8))
        ok["a"] = ref["a"] * (1 + 10 * eps)
        r = ST.compare_shared(ok, ref, dt)
        assert r["passed"], r["leaves"]
        assert r["leaves"]["zero"]["branch"] == "absolute"
        assert r["leaves"]["zero"]["rel_err"] is None
        # a finite gradient on the zero leaf far above rounding: rejected
        bad = dict(ref, zero=onp.full(8, 1e-2 * G))
        assert not ST.compare_shared(bad, ref, dt)["passed"]
        # a 5% error on a resolved leaf: rejected
        bad = dict(ref, a=ref["a"] * 1.05)
        assert not ST.compare_shared(bad, ref, dt)["passed"]
        # non-finite: rejected, and the leaf is kept in the record
        bad = dict(ref, b=onp.full(30, onp.nan))
        rb = ST.compare_shared(bad, ref, dt)
        assert not rb["passed"] and rb["leaves"]["b"]["finite"] is False
    # the pair-specific shared set includes q for C vs B
    c_tree = {"Lambda_re": onp.ones(3), SG.LEAF_T_IN: onp.ones(3),
              SG.LEAF_RHO: onp.ones(3)}
    b_tree = {"Lambda_re": onp.ones(3), SG.LEAF_T_IN: onp.ones(3)}
    assert set(ST.compare_shared(c_tree, b_tree, onp.float64)["leaves"]) == \
        {"Lambda_re", SG.LEAF_T_IN}


def test_training_mode_null_direction_is_handled_by_the_absolute_branch():
    """Review R2: in TRAINING mode a constant shift of the encoder Dense bias
    survives the residual skips and is removed by batch normalization, so its
    exact task gradient is zero. Both arms must agree through the absolute
    branch, and the relative error there is not the deciding quantity."""
    from experiments.gp import stable_gp_study as ST
    from s5.rawat_model import BatchRawatClassifier
    from flax.traverse_util import flatten_dict
    import optax

    def batch_net(arm):
        m = BatchRawatClassifier(ssm=init_substrate_ssm(arm,
                                                        **_ssm_kwargs(4, 6)),
                                 d_model=6, n_layers=2, d_output=3,
                                 readout_width=5, mlp_hidden=7, training=True)
        x = np.asarray(onp.random.RandomState(5).randn(8, 20, 2))
        v = m.init({"params": jax.random.PRNGKey(0),
                    "dropout": jax.random.PRNGKey(1)}, x, np.ones((8, 20)),
                   None)
        return m, _cast(v), x

    y = np.asarray(onp.random.RandomState(6).randint(0, 3, 8))
    grads = {}
    for arm in ("rawat_learned_input", "sgp_learned_input"):
        m, v, x = batch_net(arm)

        def loss(p):
            logits, _ = m.apply({"params": p, "batch_stats": v["batch_stats"]},
                                x, np.ones((8, 20)), None,
                                rngs={"dropout": jax.random.PRNGKey(2)},
                                mutable=["batch_stats"])
            return optax.softmax_cross_entropy(
                logits, jax.nn.one_hot(y, 3)).mean()
        grads[arm] = jax.grad(loss)(v["params"])
    fb = flatten_dict(grads["rawat_learned_input"])
    null = [k for k in fb if k[-1] == "bias" and k[:2] == ("encoder",
                                                           "encoder")]
    assert null, sorted(fb)[:10]
    G = float(onp.sqrt(sum(float(onp.sum(onp.asarray(v) ** 2))
                           for v in fb.values())))
    assert float(onp.max(onp.abs(onp.asarray(fb[null[0]])))) < 1e-10 * G
    r = ST.compare_shared(grads["sgp_learned_input"],
                          grads["rawat_learned_input"], onp.float64,
                          rel_tol=GRAD64)
    assert r["passed"], {k: v for k, v in r["leaves"].items()
                         if not v["passed"]}


def test_a_real_optimizer_step_then_projection_keeps_an_inward_gradient():
    """Force an outward proposal on r through a real update, project it, and
    confirm the gradient at the projected point is finite and usable."""
    from experiments.gp import stable_gp_study as ST
    rs = onp.random.RandomState(24)
    # production float32: the report checks executed-generator eigenvalues
    # at the projected boundary, which float64's own interior cannot resolve
    mc, vc, x = _net("sgp_learned_input", dt=onp.float32)
    w = np.asarray(rs.randn(3), onp.float32)
    f = _loss(mc, vc["batch_stats"], w)
    tx, _ = ST.make_optimizer(steps_per_epoch=1, epochs=1)
    p = _set(vc["params"], SG.LEAF_RHO, lambda i, v: onp.full(4, 40.0))
    p, tel = SG.project_stable_domain(p)
    assert int(tel["n_projected"]) > 0
    assert SG.executed_domain_report(p)["passed"]
    g = jax.grad(lambda pp: f(pp, x))(p)
    st = tx.init(p)
    upd, _ = tx.update(g, st, p)
    p2 = jax.tree_util.tree_map(lambda a, b: a + b, p, upd)
    p2, _ = SG.project_stable_domain(p2)
    assert SG.executed_domain_report(p2)["passed"]
    from flax.traverse_util import flatten_dict
    rg = [onp.asarray(v) for k, v in flatten_dict(g).items()
          if k[-1] == SG.LEAF_RHO]
    assert all(onp.all(onp.isfinite(v)) for v in rg)
    assert max(float(onp.max(onp.abs(v))) for v in rg) > 0.0


def test_checkpoint_round_trip_reproduces_the_generalized_logits():
    from flax import serialization
    rs = onp.random.RandomState(25)
    mc, vc, x = _net("sgp_learned_input")
    p = _set(vc["params"], SG.LEAF_RHO, lambda i, v: rs.uniform(-0.5, 0.2, 4))
    p = SG.project_stable_domain(p)[0]
    blob = serialization.to_bytes(dict(params=p, batch_stats=vc["batch_stats"]))
    back = serialization.from_bytes(dict(params=p,
                                         batch_stats=vc["batch_stats"]), blob)
    y0 = mc.apply({"params": p, "batch_stats": vc["batch_stats"]}, x,
                  np.ones(x.shape[0]))
    y1 = mc.apply(back, x, np.ones(x.shape[0]))
    assert onp.array_equal(onp.asarray(y0), onp.asarray(y1))


# =========================================================================
#  4b. Review R0-R4 amendments
# =========================================================================
def _dG_dr(p, j, b, T, T_in):
    """Exact AD derivative of the transfer in r at r = 0 (real tangent)."""
    def G(r):
        return SG.transfer(p, j, b, T, np.exp(r), T_in)
    return jax.jvp(G, (np.asarray(0.0),), (np.asarray(1.0),))[1]


def test_R0_the_r_tangent_identity_and_the_equal_horizon_cancellation():
    """dG/dr|_0 = -b T p^2 (1 + T_in p) / [(1 + T p)(p + j)^2]. At T = T_in the
    auxiliary pole -1/T cancels (the brief's original choice); at T != T_in
    its residue is -b (1 - T_in/T) / [T^2 (j - 1/T)^2]."""
    j, b = 0.3 + 0.8j, 1.7 - 0.4j            # j != 1/T for both horizons
    for T, T_in in ((5.0, 5.0), (SG.RECURRENT_T_REFERENCE,
                                 SG.INPUT_T_REFERENCE)):
        for p in (0.2 + 0.5j, -0.4 + 1.3j, 2.0):
            got = complex(_dG_dr(p, j, b, T, T_in))
            want = -b * T * p ** 2 * (1 + T_in * p) / ((1 + T * p)
                                                       * (p + j) ** 2)
            assert abs(got - want) / abs(want) < TAN64, (T, T_in, p)
        delta = 1e-9
        p0 = -1.0 / T + delta
        res_num = complex(_dG_dr(p0, j, b, T, T_in)) * delta
        if T == T_in:
            assert abs(res_num) < 1e-5, "equal horizons: pole cancels"
        else:
            res = -b * (1 - T_in / T) / (T ** 2 * (j - 1 / T) ** 2)
            assert abs(res) > 1e-3
            assert abs(res_num - res) / abs(res) < RES64, (res_num, res)


def test_R0_the_module_starts_at_T10_Tin5_rho1_and_equals_B():
    sc, pc, x = _layer("sgp_learned_input")
    T = onp.asarray(sc.apply({"params": pc}, method=lambda m: m.recurrent_T()))
    Ti = onp.asarray(sc.apply({"params": pc},
                              method=lambda m: m.input_horizon()))
    rho = onp.asarray(sc.apply({"params": pc},
                               method=lambda m: m.recurrent_rho()))
    assert onp.all(T == 10.0) and onp.all(Ti == 5.0) and onp.all(rho == 1.0)
    sb, pb, _ = _layer("rawat_learned_input")
    assert _rel(sc.apply({"params": pc}, x), sb.apply({"params": pb}, x)) \
        < IDENT64


def _old_fractional_excess_bound(a, omega, T, eps):
    """The coordinator's ORIGINAL prescription, kept only as a regression."""
    c = 1 + T * a
    z = T * a + a * c ** 2 / (T * omega ** 2)
    return math.log1p((1 - 32 * eps) * z)


def test_R1_rounding_witness_old_margin_fails_new_margin_holds():
    """T = 1, omega = 1, a = 3 * 2^-26 in float32: rho_max - 1 is about 0.75 of
    a float32 step. The old margin rounds exp(bound) to 1 + 2^-23, beyond the
    boundary; the full-ratio margin returns r = 0, the exact rho = 1 fallback."""
    eps32 = float(onp.finfo(onp.float32).eps)
    a = onp.float32(3 * 2.0 ** -26)
    T = onp.float32(1.0); w = onp.float32(1.0)
    rmax = float(SG.rho_max_float64(a, w, T))
    old = onp.float32(_old_fractional_excess_bound(float(a), 1.0, 1.0, eps32))
    rho_old = onp.float32(onp.exp(onp.float64(old)))
    assert float(rho_old) == 1.0 + 2.0 ** -23
    assert float(rho_old) > rmax, "the witness: old interior is not interior"
    assert SG.stability_S(float(a), 1.0, 1.0, float(rho_old)) < 0
    new = onp.asarray(SG.log_rho_upper(np.asarray(a, dtype=onp.float32),
                                       np.asarray(w, dtype=onp.float32),
                                       np.asarray(T, dtype=onp.float32),
                                       onp.float32))
    assert float(new) == 0.0
    assert SG.stability_S(float(a), 1.0, 1.0, 1.0) > 0


def _net_resolvable(arm):
    """A nondegenerate fixture: larger clock so the r path is resolvable."""
    from s5.rawat_model import RawatClassifier
    kw = dict(_ssm_kwargs(4, 6), dt_min=0.1, dt_max=1.0)
    m = RawatClassifier(ssm=init_substrate_ssm(arm, **kw), d_model=6,
                        n_layers=2, d_output=3, readout_width=5, mlp_hidden=7,
                        training=False)
    x = np.asarray(onp.random.RandomState(8).randn(24, 2))
    v = _cast(m.init(jax.random.PRNGKey(3), x, np.ones(24)))
    return m, v, x


def test_R2_r_only_derivative_at_the_actual_start_float64():
    """At q = r = t = 0: an r-ONLY tangent against independent central
    differences at both declared steps, and a vanishing t derivative."""
    from flax.traverse_util import flatten_dict, unflatten_dict
    rs = onp.random.RandomState(27)
    m, v, x = _net_resolvable("sgp_learned_input")
    w = np.asarray(rs.randn(3))
    f = _loss(m, v["batch_stats"], w)
    p = v["params"]
    flat = flatten_dict(p)
    d = unflatten_dict({k: (np.asarray(rs.randn(*onp.shape(val)))
                            if k[-1] == SG.LEAF_RHO else np.zeros_like(val))
                        for k, val in flat.items()})
    jvp = float(jax.jvp(lambda pp: f(pp, x), (p,), (d,))[1])
    assert abs(jvp) > 1e-6, jvp
    for h in FD_STEPS:
        plus = jax.tree_util.tree_map(lambda a_, b_: a_ + h * b_, p, d)
        minus = jax.tree_util.tree_map(lambda a_, b_: a_ - h * b_, p, d)
        fd = (float(f(plus, x)) - float(f(minus, x))) / (2 * h)
        assert abs(jvp - fd) / abs(fd) < FD_R64, (h, jvp, fd)
    g = flatten_dict(jax.grad(lambda pp: f(pp, x))(p))
    gr = max(float(onp.max(onp.abs(onp.asarray(val))))
             for k, val in g.items() if k[-1] == SG.LEAF_RHO)
    gt = max(float(onp.max(onp.abs(onp.asarray(val))))
             for k, val in g.items() if k[-1] == SG.LEAF_T)
    assert gr > 1e-6 and gt <= ZEROT64 * gr, (gr, gt)


def test_R4_validation_detects_executed_overflow_a_float64_product_hides():
    """A real mode with rho and T each ~1e30 in float32: the float64 product is
    finite, the EXECUTED float32 mass overflows. The report must fail it."""
    P = 2
    tree = {"encoder": {"layers_0": {"seq": {
        "Lambda_re": np.asarray([-0.5, -0.5], dtype=onp.float32),
        "Lambda_im": np.asarray([0.0, 0.0], dtype=onp.float32),
        "log_step": np.asarray([[-2.0], [-2.0]], dtype=onp.float32),
        SG.LEAF_RHO: np.asarray([69.0, 0.0], dtype=onp.float32),
        SG.LEAF_T: np.asarray([68.0, 0.0], dtype=onp.float32),
        SG.LEAF_T_IN: np.zeros(P, dtype=onp.float32)}}}}
    rho64 = math.exp(69.0); T64 = 10 * math.exp(68.0)
    assert math.isfinite(rho64 * T64)
    rep_ = SG.executed_domain_report(tree)
    assert rep_["passed"] is False
    assert rep_["layers"][0]["executed_finite_positive"] is False
    # an underflowed input horizon fails for B as well
    btree = {"encoder": {"layers_0": {"seq": {
        SG.LEAF_T_IN: np.asarray([-200.0, 0.0], dtype=onp.float32)}}}}
    rb = SG.executed_domain_report(btree)
    assert rb["passed"] is False and rb["layers"][0]["kind"] == \
        "input_horizon_only"


# =========================================================================
#  5. Runner logic that decides the run: source selection and the screen
# =========================================================================
def test_source_selection_applies_the_original_rule(tmp_path):
    import json
    from experiments.gp import stable_gp_study as ST
    runs = [("alpha_p_s5__lr1e-3__seed100", 1e-3, [0.90, 0.95, 0.95]),
            ("alpha_p_s5__lr3e-4__seed100", 3e-4, [0.91, 0.95, 0.93]),
            ("native_s5__lr1e-3__seed100", 1e-3, [0.99, 0.99, 0.99])]
    with open(tmp_path / "manifest.jsonl", "w") as mf:
        for tag, lr, accs in runs:
            rd = tmp_path / tag
            rd.mkdir()
            with open(rd / "metrics.jsonl", "w") as fh:
                for e, acc in enumerate(accs):
                    fh.write(json.dumps(dict(mode="train", epoch=e,
                                             val_accuracy=acc,
                                             val_cross_entropy=0.3)) + "\n")
            for n in ("best.msgpack", "best.meta.json"):
                (rd / n).write_text("{}")
            (rd / "config.json").write_text(json.dumps({"config": {}}))
            arm = tag.split("__")[0]
            mf.write(json.dumps(dict(run=tag, arm=arm, lr=lr, seed=100,
                                     exit=0, run_dir=str(rd))) + "\n")
    sel = ST.select_source(str(tmp_path))
    # native_s5 is excluded; equal accuracy and CE fall to the declared order;
    # within the 1e-3 run the FIRST epoch reaching 0.95 is the best checkpoint
    assert sel["chosen"]["lr"] == 1e-3
    assert sel["chosen"]["best_epoch"] == 1
    assert len(sel["candidates"]) == 2


def _rows(acc, ce):
    from experiments.gp import stable_gp_study as ST
    return [dict(seed=s, arm=a, endpoint_val_accuracy=acc[a][i],
                 endpoint_val_cross_entropy=ce[a][i])
            for i, s in enumerate(ST.SEEDS) for a in ST.ARMS]


def test_the_screen_requires_both_baselines_every_stream_and_lower_CE():
    from experiments.gp import stable_gp_study as ST
    base = dict(A_rawat=[0.950, 0.951, 0.949],
                B_learned_input=[0.951, 0.950, 0.950],
                C_stable_generalized=[0.956, 0.955, 0.955])
    ce = dict(A_rawat=[0.30] * 3, B_learned_input=[0.30] * 3,
              C_stable_generalized=[0.29] * 3)
    assert ST.screen(_rows(base, ce))["development_success"] is True
    # one negative stream against B fails even with a large mean
    bad = dict(base, C_stable_generalized=[0.970, 0.970, 0.949])
    assert ST.screen(_rows(bad, ce))["development_success"] is False
    # higher CE against either baseline fails
    ce_bad = dict(ce, C_stable_generalized=[0.31] * 3)
    assert ST.screen(_rows(base, ce_bad))["development_success"] is False
    # a gain over A alone, not over B, fails
    only_a = dict(base, B_learned_input=[0.956, 0.955, 0.955])
    assert ST.screen(_rows(only_a, ce))["development_success"] is False
    # a missing pair cannot produce a verdict
    rows = [r for r in _rows(base, ce) if not (r["seed"] == ST.SEEDS[-1]
                                               and r["arm"] == "A_rawat")]
    assert ST.screen(rows)["development_success"] is False


# =========================================================================
#  6. Actual TSS Eq. (17) and its generalized discrete descendant
# =========================================================================
def test_generalized_discrete_equation_recovers_TSS_eq17_exactly():
    """M = gamma = 0 gives Eq. (17), including the previous-drive convention,
    on a NONLINEAR drive f_k = tanh(W s_k + x_k)."""
    rs = onp.random.RandomState(31)
    n, K, h, T = 5, 40, 0.3, 2.0
    W = rs.randn(n, n) * 0.6
    xs = rs.randn(K, n)
    s_a = s_b = rs.randn(n)
    s_prev = s_b.copy()
    f_prev = onp.tanh(W @ s_a + xs[0])            # declared startup convention
    for k in range(K):
        f = onp.tanh(W @ s_a + xs[k])
        s_a_next = SG.tss_eq17_step(s_a, f, f_prev, h, T)
        fb = onp.tanh(W @ s_b + xs[k])
        s_b_next = SG.generalized_discrete_step(s_b, s_prev, fb, f_prev, h,
                                                0.0, 0.0, T)
        assert onp.max(onp.abs(s_a_next - s_b_next)) < TSS64, k
        s_prev, s_a, s_b, f_prev = s_b, s_a_next, s_b_next, f


def test_the_other_exact_rows_of_the_contract():
    rs = onp.random.RandomState(32)
    s, sp, f, fp = rs.randn(4), rs.randn(4), rs.randn(4), rs.randn(4)
    h, g, M = 0.2, 1.7, 0.9
    # M = T = 0: Euler residual correction
    d = SG.generalized_discrete_step(s, sp, f, fp, h, 0.0, g, 0.0) - s
    assert onp.max(onp.abs(d - (-(h / g) * (s - f)))) < TSS64
    # T = 0, M > 0: heavy-ball-like correction
    d = SG.generalized_discrete_step(s, sp, f, fp, h, M, g, 0.0) - s
    want = (M / (M + h * g)) * (s - sp) - (h * h / (M + h * g)) * (s - f)
    assert onp.max(onp.abs(d - want)) < TSS64


def test_eq17_retains_history_unlike_the_memoryless_continuous_reduction():
    """f_k = a s_k + x_k. Eq. (17) has characteristic polynomial
    mu^2 - [1 - h/T + (1 + h/T) a] mu + a, so an input impulse has a nonzero
    response two and more steps later. The exact continuous reduction
    r + T r' = 0 (the old `prospective_recurrence` arm) has s_k = J^-1 b x_k
    and NO response after lag zero. They are not interchangeable."""
    a, h, T, K = 0.6, 0.4, 3.0, 12
    s, f_prev, resp = 0.0, 0.0, []
    for k in range(K):
        x = 1.0 if k == 0 else 0.0
        f = a * s + x
        s = SG.tss_eq17_step(s, f, f_prev, h, T)
        f_prev = f
        resp.append(s)
    assert max(abs(v) for v in resp[2:]) > 1e-3, resp
    # the recursion's roots are the stated polynomial's
    c1 = 1 - h / T + (1 + h / T) * a
    mu = onp.roots([1.0, -c1, a])
    seq = onp.asarray(resp[2:])
    # s_{k+1} = c1 s_k - a s_{k-1} once the impulse has passed
    assert onp.max(onp.abs(seq[2:] - (c1 * seq[1:-1] - a * seq[:-2]))) < TSS64
    assert onp.all(onp.abs(mu) > 0)
    # the memoryless reduction: zero beyond lag zero by construction
    memoryless = [1.0 / 0.5] + [0.0] * (K - 1)       # s_k = J^-1 b x_k
    assert max(abs(v) for v in memoryless[1:]) == 0.0


def test_unforced_increments_drift_in_eq17_and_decay_when_gamma_positive():
    h, T, M, g = 0.5, 2.0, 1.3, 0.8
    d_tss = d_gen = 1.0
    s_t, s_g = 0.0, 0.0
    for _ in range(5):
        # residual zero and f = s in the unforced direction
        s_t_next = SG.tss_eq17_step(s_t + d_tss, s_t + d_tss, s_t, h, T)
        d_tss_next = s_t_next - (s_t + d_tss)
        s_prev = s_g
        s_g = s_g + d_gen
        s_g_next = SG.generalized_discrete_step(s_g, s_prev, s_g, s_prev, h,
                                                M, g, T)
        ratio = (s_g_next - s_g) / d_gen
        assert abs(ratio - (M + h * T) / (M + h * (g + T))) < TSS64
        assert abs(d_tss_next - d_tss) < TSS64
        s_t, d_gen = s_t + d_tss, s_g_next - s_g


# =========================================================================
#  7. Production dtype, in its own process
# =========================================================================
def test_production_float32_probe_in_its_own_process():
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stable_gp_float32_probe.py")
    env = dict(os.environ, JAX_ENABLE_X64="0")
    r = subprocess.run([sys.executable, probe], env=env, capture_output=True,
                       text=True)
    print(r.stdout[-6000:]); print(r.stderr[-4000:])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
