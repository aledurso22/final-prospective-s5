"""Generalized prospective response: Milestone B/C correctness gates.

PREDECLARED TOLERANCES (fixed before running; a failure is reported, not
loosened without an explained new protocol):

  GATE_EXACT      plain / fixed-zero bypass identity            atol = 0
  GATE_REF_X64    coefficients vs dense real-pair reference     1e-9  normalized
  GATE_ZERO_X64   generalized formula at t -> 0 vs plain        1e-12 normalized
  GATE_JVP_X64    directional JVP vs finite differences         1e-6  relative
  GATE_STREAM_F32 parallel vs streaming, float32 production     1e-5  relative
  GATE_STREAM_X64 parallel vs streaming, float64               1e-11  relative

The reference in GATE_REF_X64 is an INDEPENDENT dense real-pair augmented
matrix exponential computed with scipy; it never touches the closed-form
complex algebra under test.
"""
import os
import sys

import jax
import numpy as onp
import pytest
from scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
import jax.numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5.gp_coefficients import (absorb_clock, gp_response_coefficients,     # noqa: E402
                                inverse_softplus, matched_tss_coefficients,
                                phi1, prospective_input_coefficients,
                                softplus_response)
from s5.gp_ssm import GP_MECHANISMS, GPSSM, init_gp_ssm                     # noqa: E402
from s5.ssm import S5SSM, discretize_zoh, init_S5SSM                        # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                                     # noqa: E402
from jax.scipy.linalg import block_diag                                     # noqa: E402

GATE_EXACT = 0.0
GATE_REF_X64 = 1e-9
GATE_ZERO_X64 = 1e-12
GATE_JVP_X64 = 1e-6
GATE_STREAM_F32 = 1e-5
GATE_STREAM_X64 = 1e-11

H, SSM_SIZE, BLOCKS, L = 4, 8, 2, 24


def ssm_kwargs(conj_sym=True, bidirectional=False, discretization="zoh"):
    block = SSM_SIZE // BLOCKS
    Lam, _, _, V, _ = make_DPLR_HiPPO(block)
    if conj_sym:
        block //= 2
        P = SSM_SIZE // 2
    else:
        P = SSM_SIZE
    Lam, V = Lam[:block], V[:, :block]
    Vc = V.conj().T
    Lam = (Lam * np.ones((BLOCKS, block))).ravel()
    return dict(H=H, P=P, Lambda_re_init=Lam.real, Lambda_im_init=Lam.imag,
                V=block_diag(*([V] * BLOCKS)), Vinv=block_diag(*([Vc] * BLOCKS)),
                C_init="trunc_standard_normal", discretization=discretization,
                dt_min=0.001, dt_max=0.1, conj_sym=conj_sym, clip_eigs=False,
                bidirectional=bidirectional)


def build(mechanism, gp_init_scale=0.05, **kw):
    fn = init_gp_ssm(mechanism=mechanism, gp_init_scale=gp_init_scale,
                     **ssm_kwargs(**kw))
    return fn(step_rescale=1.0)


def inputs(key=0, length=L, h=H):
    return jax.random.normal(jax.random.PRNGKey(key), (length, h))


# =====================================================================
# CHECK 1 - frozen plain / fixed-zero bypass is the ORIGINAL path
# =====================================================================

def test_1_plain_factory_returns_original_s5ssm():
    """No extra parameters, no random-key change, no new module type."""
    kw = ssm_kwargs()
    plain = init_gp_ssm(mechanism="plain", **kw)(step_rescale=1.0)
    stock = init_S5SSM(**kw)(step_rescale=1.0)
    assert type(plain) is type(stock) is S5SSM
    u = inputs()
    vp = plain.init(jax.random.PRNGKey(0), u)
    vs = stock.init(jax.random.PRNGKey(0), u)
    from flax.traverse_util import flatten_dict
    fp, fs = flatten_dict(vp["params"]), flatten_dict(vs["params"])
    assert sorted(fp) == sorted(fs)
    assert not any("gp_" in "/".join(k) for k in fp), "bypass allocated params"
    for k in fp:
        onp.testing.assert_array_equal(fp[k], fs[k])      # exact, atol = 0
    onp.testing.assert_array_equal(plain.apply(vp, u), stock.apply(vs, u))


def test_1_plain_gradients_and_one_optimizer_update_identical():
    import optax
    kw = ssm_kwargs()
    plain = init_gp_ssm(mechanism="plain", **kw)(step_rescale=1.0)
    stock = init_S5SSM(**kw)(step_rescale=1.0)
    u = inputs(1)
    vp = plain.init(jax.random.PRNGKey(0), u)
    vs = stock.init(jax.random.PRNGKey(0), u)

    def step(mod, v):
        loss = lambda p: np.sum(mod.apply({"params": p}, u) ** 2)
        g = jax.grad(loss)(v["params"])
        tx = optax.adam(1e-3)
        st = tx.init(v["params"])
        upd, _ = tx.update(g, st, v["params"])
        return g, optax.apply_updates(v["params"], upd)

    gp, pp = step(plain, vp)
    gs, ps = step(stock, vs)
    from flax.traverse_util import flatten_dict
    for a, b in ((gp, gs), (pp, ps)):
        fa, fb = flatten_dict(a), flatten_dict(b)
        assert sorted(fa) == sorted(fb)
        for k in fa:
            onp.testing.assert_array_equal(fa[k], fb[k])   # exact


# =====================================================================
# CHECK 2 - coefficients vs an INDEPENDENT dense real-pair reference
# =====================================================================

def _real_pair(z):
    return onp.array([[z.real, -z.imag], [z.imag, z.real]])


def _dense_reference(a, b_row, t):
    """Dense real-pair augmented matrix exponential (scipy), with solves."""
    F0 = _real_pair(a)
    B0 = onp.stack([b_row.real, b_row.imag])
    M = onp.eye(2) - t * F0
    F = onp.linalg.solve(M, F0)
    Dx = onp.linalg.solve(M, t * B0)
    Bh = onp.linalg.solve(M, B0) + F @ Dx
    n_in = B0.shape[1]
    aug = onp.zeros((2 + n_in, 2 + n_in))
    aug[:2, :2] = F
    aug[:2, 2:] = Bh
    E = expm(aug)
    return E[:2, :2], E[:2, 2:], Dx


@pytest.mark.parametrize("omega_on", [0.0, 1.0])
def test_2_coefficients_match_dense_real_pair_reference(omega_on):
    rng = onp.random.default_rng(11)
    worst = dict(a=0.0, b=0.0, d=0.0)
    n_cases = 0
    for _ in range(40):
        sigma = -10.0 ** rng.uniform(-4, 0.5)          # includes SMALL poles
        omega = omega_on * rng.normal() * 10.0 ** rng.uniform(-3, 0.5)
        a = complex(sigma, omega)
        b_row = rng.normal(size=3) + 1j * rng.normal(size=3)
        t = float(rng.choice([0.0, 10.0 ** rng.uniform(-4, 1.0)]))   # incl. ZERO-T
        ar, br, dr = _dense_reference(a, b_row, t)
        c = gp_response_coefficients(np.asarray([a]), np.asarray(b_row)[None, :],
                                     np.asarray([t]))
        got_a = _real_pair(complex(onp.asarray(c["a_bar"])[0]))
        got_b = onp.stack([onp.asarray(c["b_bar"])[0].real,
                           onp.asarray(c["b_bar"])[0].imag])
        got_d = onp.stack([onp.asarray(c["d_x"])[0].real,
                           onp.asarray(c["d_x"])[0].imag])
        nrm = lambda x: max(1.0, onp.linalg.norm(x))
        worst["a"] = max(worst["a"], onp.linalg.norm(got_a - ar) / nrm(ar))
        worst["b"] = max(worst["b"], onp.linalg.norm(got_b - br) / nrm(br))
        worst["d"] = max(worst["d"], onp.linalg.norm(got_d - dr) / nrm(dr))
        n_cases += 1
    assert n_cases == 40
    for k, v in worst.items():
        assert v < GATE_REF_X64, (k, v)


def test_2_clock_absorbed_exactly_once():
    """absorb_clock + t=0 must reproduce the stock ZOH bit-for-bit-close."""
    rng = onp.random.default_rng(3)
    P, n_in = 6, 3
    Lam = (-onp.abs(rng.normal(size=P)) - 0.05) + 1j * rng.normal(size=P)
    Bt = rng.normal(size=(P, n_in)) + 1j * rng.normal(size=(P, n_in))
    step = onp.exp(rng.uniform(-3, 0, size=P))
    Lb, Bb = discretize_zoh(np.asarray(Lam), np.asarray(Bt), np.asarray(step))
    a, b = absorb_clock(np.asarray(Lam), np.asarray(Bt), np.asarray(step))
    c = gp_response_coefficients(a, b, np.zeros(P))
    assert float(np.max(np.abs(c["a_bar"] - Lb))) < GATE_ZERO_X64
    assert float(np.max(np.abs(c["b_bar"] - Bb))) < GATE_ZERO_X64
    assert float(np.max(np.abs(c["d_x"]))) == 0.0


def test_2_generalized_formula_itself_is_correct_at_zero():
    """The bypass must not conceal a wrong coefficient builder."""
    rng = onp.random.default_rng(5)
    P, n_in = 5, 3
    a = np.asarray((-onp.abs(rng.normal(size=P)) - 0.02)
                   + 1j * rng.normal(size=P))
    b = np.asarray(rng.normal(size=(P, n_in)) + 1j * rng.normal(size=(P, n_in)))
    base = gp_response_coefficients(a, b, np.zeros(P))
    for t in (1e-12, 1e-10, 1e-8):
        c = gp_response_coefficients(a, b, np.full(P, t))
        for k in ("a_bar", "b_bar"):
            err = float(np.max(np.abs(c[k] - base[k]))) / max(
                1.0, float(np.max(np.abs(base[k]))))
            assert err < 1e-6, (k, t, err)


def test_2_phi1_small_argument_value_and_derivative():
    assert complex(phi1(np.asarray(0.0 + 0j))) == pytest.approx(1.0)
    d = jax.grad(lambda z: phi1(z).real)(0.0 + 0j)
    assert complex(d).real == pytest.approx(0.5, abs=1e-9)
    assert onp.isfinite(complex(jax.grad(lambda z: phi1(z).real)(1e-13 + 0j)).real)
    for z in (1e-6, 1e-5, 1e-3, 0.1, 1.0):
        assert complex(phi1(np.asarray(z + 0j))).real == pytest.approx(
            float(onp.expm1(z) / z), rel=1e-12)


# =====================================================================
# CHECK 3 - directional gradients across the FULL parameterized family
# =====================================================================

@pytest.mark.parametrize("mechanism", ["gp_scalar", "gp_diagonal",
                                       "prospective_input", "full_state_pc"])
def test_3_jvp_matches_finite_differences_all_parameters(mechanism):
    mod = build(mechanism)
    u = inputs(7)
    v = jax.tree_util.tree_map(lambda x: x.astype(np.float64),
                               mod.init(jax.random.PRNGKey(0), u))
    params = v["params"]
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, len(jax.tree_util.tree_leaves(params)))
    it = iter(keys)
    direction = jax.tree_util.tree_map(
        lambda x: jax.random.normal(next(it), x.shape, dtype=x.dtype), params)

    f = lambda p: np.sum(mod.apply({"params": p}, u) ** 2)
    val, tangent = jax.jvp(f, (params,), (direction,))
    eps = 1e-6
    shift = lambda s: jax.tree_util.tree_map(lambda p, d: p + s * eps * d,
                                             params, direction)
    fd = (f(shift(+1)) - f(shift(-1))) / (2 * eps)
    rel = abs(float(tangent) - float(fd)) / max(1.0, abs(float(fd)))
    assert rel < GATE_JVP_X64, (mechanism, rel, float(tangent), float(fd))


def test_3_response_parameter_receives_a_nonzero_gradient():
    for mech in ("gp_scalar", "gp_diagonal", "prospective_input"):
        mod = build(mech)
        u = inputs(8)
        v = mod.init(jax.random.PRNGKey(0), u)
        g = jax.grad(lambda p: np.sum(mod.apply({"params": p}, u) ** 2))(v["params"])
        gr = g["gp_response_raw"]
        assert np.all(np.isfinite(gr))
        assert float(np.max(np.abs(gr))) > 0.0, mech


def test_3_tied_feedthrough_is_actually_active():
    """Removing D_x must change the output: the term is not dead code."""
    mod = build("gp_diagonal", gp_init_scale=0.4)
    u = inputs(9)
    v = mod.init(jax.random.PRNGKey(0), u)
    c = mod.apply(v, method=GPSSM.coefficients)
    assert float(np.max(np.abs(c["d_x"]))) > 1e-6
    from s5.gp_ssm import gp_readout, gp_scan
    hs = gp_scan(c["a_bar"], c["b_bar"], u)
    with_dx = gp_readout(hs, c["d_x"], v["params"]["C"][..., 0]
                         + 1j * v["params"]["C"][..., 1], u, True)
    without = gp_readout(hs, np.zeros_like(c["d_x"]), v["params"]["C"][..., 0]
                         + 1j * v["params"]["C"][..., 1], u, True)
    assert float(np.max(np.abs(with_dx - without))) > 1e-6


# =====================================================================
# CHECK 4 - parallel vs streaming, boundaries, and input patterns
# =====================================================================

def _stream(c, C_tilde, D, u, conj_sym=True):
    """Reference sequential recurrence with the NATIVE timing convention:
    h_k = a_bar h_{k-1} + b_bar x_k, h_{-1} = 0, s_k = h_k + D_x x_k."""
    ab = onp.asarray(c["a_bar"]); bb = onp.asarray(c["b_bar"])
    dx = onp.asarray(c["d_x"]); C = onp.asarray(C_tilde)
    f = 2.0 if conj_sym else 1.0
    h = onp.zeros(ab.shape[0], dtype=complex)
    out = []
    for x in onp.asarray(u):
        h = ab * h + bb @ x
        out.append(f * onp.real(C @ (h + dx @ x)) + onp.asarray(D) * x)
    return onp.array(out)


@pytest.mark.parametrize("mechanism", ["gp_scalar", "gp_diagonal",
                                       "prospective_input", "full_state_pc"])
@pytest.mark.parametrize("pattern", ["random", "impulse", "constant",
                                     "alternating"])
def test_4_parallel_matches_streaming(mechanism, pattern):
    mod = build(mechanism, gp_init_scale=0.3)
    v = jax.tree_util.tree_map(lambda x: x.astype(np.float64),
                               mod.init(jax.random.PRNGKey(0), inputs()))
    if pattern == "random":
        u = inputs(12).astype(np.float64)
    elif pattern == "impulse":                       # FIRST-TOKEN impulse
        u = np.zeros((L, H), dtype=np.float64).at[0, 0].set(1.0)
    elif pattern == "constant":
        u = np.ones((L, H), dtype=np.float64)
    else:
        u = np.asarray(onp.tile([[1.0] * H, [-1.0] * H], (L // 2, 1)))
    par = onp.asarray(mod.apply(v, u))
    c = mod.apply(v, method=GPSSM.coefficients)
    C_tilde = v["params"]["C"][..., 0] + 1j * v["params"]["C"][..., 1]
    ref = _stream(c, C_tilde, v["params"]["D"], u)
    rel = onp.max(onp.abs(par - ref)) / max(1.0, onp.max(onp.abs(ref)))
    assert rel < GATE_STREAM_X64, (mechanism, pattern, rel)


def test_4_chunk_boundary_restarts_from_zero_prehistory():
    """A chunk processed alone restarts at h = 0, matching the reset semantics.

    Stock S5 has no within-sequence reset; each call starts from a zero
    prehistory. This pins that convention so a future reset mask cannot change
    it silently.
    """
    mod = build("gp_diagonal", gp_init_scale=0.3)
    v = jax.tree_util.tree_map(lambda x: x.astype(np.float64),
                               mod.init(jax.random.PRNGKey(0), inputs()))
    u = inputs(13).astype(np.float64)
    first = onp.asarray(mod.apply(v, u[:L // 2]))
    whole = onp.asarray(mod.apply(v, u))
    onp.testing.assert_allclose(first, whole[:L // 2], rtol=0, atol=1e-12)
    second_alone = onp.asarray(mod.apply(v, u[L // 2:]))
    # the second chunk alone is NOT the tail of the whole run: state is not carried
    assert onp.max(onp.abs(second_alone - whole[L // 2:])) > 1e-8


def test_4_no_state_carried_between_independent_examples():
    mod = build("gp_diagonal", gp_init_scale=0.3)
    v = mod.init(jax.random.PRNGKey(0), inputs())
    u1, u2 = inputs(14), inputs(15)
    batched = jax.vmap(lambda x: mod.apply(v, x))(np.stack([u1, u2]))
    onp.testing.assert_allclose(onp.asarray(batched[0]),
                                onp.asarray(mod.apply(v, u1)), rtol=1e-10,
                                atol=1e-12)
    onp.testing.assert_allclose(onp.asarray(batched[1]),
                                onp.asarray(mod.apply(v, u2)), rtol=1e-10,
                                atol=1e-12)


# =====================================================================
# CHECK 5 - stability and the matched-TSS negative control
# =====================================================================

def test_5_effective_pole_is_stable_for_every_nonnegative_t():
    """Re(a_eff) = (sigma - t|a|^2)/|m|^2 < 0 whenever sigma < 0, t >= 0."""
    from s5.gp_coefficients import effective_pole_real_part, is_admissible
    rng = onp.random.default_rng(21)
    for _ in range(300):
        a = np.asarray([complex(-10.0 ** rng.uniform(-4, 1),
                                rng.normal() * 10.0 ** rng.uniform(-3, 1))])
        t = np.asarray([10.0 ** rng.uniform(-5, 2)])
        assert bool(is_admissible(a, t)[0])
        assert float(effective_pole_real_part(a, t)[0]) < 0.0
        c = gp_response_coefficients(a, np.ones((1, 1), dtype=complex), t)
        assert float(np.abs(c["a_bar"])[0]) < 1.0      # discrete contraction


def test_5_matched_tss_cancels_the_driven_history():
    """full_state_pc has EXACTLY zero driven history: b_hist == 0."""
    rng = onp.random.default_rng(22)
    P, n_in = 5, 3
    a = np.asarray((-onp.abs(rng.normal(size=P)) - 0.05)
                   + 1j * rng.normal(size=P))
    b = np.asarray(rng.normal(size=(P, n_in)) + 1j * rng.normal(size=(P, n_in)))
    c = matched_tss_coefficients(a, b, 0.7)
    assert float(np.max(np.abs(c["b_hist"]))) == 0.0
    assert float(np.max(np.abs(c["b_bar"]))) == 0.0
    # the static branch is J^-1 B = -b/a
    onp.testing.assert_allclose(onp.asarray(c["d_x"]),
                                onp.asarray(-b / a[:, None]), rtol=1e-12)


def test_5_matched_tss_impulse_response_has_support_one():
    mod = build("full_state_pc", gp_init_scale=0.7)
    v = jax.tree_util.tree_map(lambda x: x.astype(np.float64),
                               mod.init(jax.random.PRNGKey(0), inputs()))
    imp = np.zeros((L, H), dtype=np.float64).at[0, 0].set(1.0)
    y = onp.asarray(mod.apply(v, imp))
    assert onp.max(onp.abs(y[1:])) < 1e-12, onp.max(onp.abs(y[1:]))
    assert onp.max(onp.abs(y[0])) > 1e-8


def test_5_prospective_input_leaves_the_generator_unchanged():
    """The 2026-style intervention must NOT move the transition."""
    rng = onp.random.default_rng(23)
    P, n_in = 4, 2
    a = np.asarray((-onp.abs(rng.normal(size=P)) - 0.1)
                   + 1j * rng.normal(size=P))
    b = np.asarray(rng.normal(size=(P, n_in)) + 1j * rng.normal(size=(P, n_in)))
    c = prospective_input_coefficients(a, b, 0.3)
    onp.testing.assert_allclose(onp.asarray(c["a_eff"]), onp.asarray(a), rtol=0)
    gp = gp_response_coefficients(a, b, np.full(P, 0.3))
    assert onp.max(onp.abs(onp.asarray(gp["a_eff"]) - onp.asarray(a))) > 1e-6


# =====================================================================
# CHECK 6 - rejected configurations, float32 production tolerance
# =====================================================================

@pytest.mark.parametrize("kw,match", [
    (dict(bidirectional=True), "unidirectional"),
    (dict(discretization="bilinear"), "ZOH"),
])
def test_6_unsupported_configurations_are_rejected(kw, match):
    mod = build("gp_diagonal", **kw)
    with pytest.raises(ValueError, match=match):
        mod.init(jax.random.PRNGKey(0), inputs())


def test_6_non_unit_step_rescale_is_rejected():
    fn = init_gp_ssm(mechanism="gp_diagonal", gp_init_scale=0.05, **ssm_kwargs())
    with pytest.raises(ValueError, match="step_rescale"):
        fn(step_rescale=2.0).init(jax.random.PRNGKey(0), inputs())


def test_6_zero_init_scale_is_rejected():
    with pytest.raises(ValueError, match="zero tangent"):
        build("gp_diagonal", gp_init_scale=0.0).init(jax.random.PRNGKey(0),
                                                     inputs())


def test_6_float32_production_parallel_vs_streaming():
    mod = build("gp_diagonal", gp_init_scale=0.3)
    v = mod.init(jax.random.PRNGKey(0), inputs())     # native float32 params
    u = inputs(31)
    par = onp.asarray(mod.apply(v, u))
    c = mod.apply(v, method=GPSSM.coefficients)
    C_tilde = v["params"]["C"][..., 0] + 1j * v["params"]["C"][..., 1]
    ref = _stream(c, C_tilde, v["params"]["D"], u)
    rel = onp.max(onp.abs(par - ref)) / max(1.0, onp.max(onp.abs(ref)))
    assert rel < GATE_STREAM_F32, rel


def test_6_conj_sym_false_drops_the_factor_two():
    mod = build("gp_diagonal", gp_init_scale=0.2, conj_sym=False)
    v = jax.tree_util.tree_map(lambda x: x.astype(np.float64),
                               mod.init(jax.random.PRNGKey(0), inputs()))
    u = inputs(32).astype(np.float64)
    c = mod.apply(v, method=GPSSM.coefficients)
    C_tilde = v["params"]["C"][..., 0] + 1j * v["params"]["C"][..., 1]
    ref = _stream(c, C_tilde, v["params"]["D"], u, conj_sym=False)
    rel = onp.max(onp.abs(onp.asarray(mod.apply(v, u)) - ref)) / max(
        1.0, onp.max(onp.abs(ref)))
    assert rel < GATE_STREAM_X64, rel


# =====================================================================
# THEORY DIAGNOSTIC - the DC invariant H(0) = J^-1 B
# =====================================================================

def test_theory_dc_gain_invariant_across_mechanisms():
    """For fixed J, B, C, gamma, changing the prospective kernel preserves
    the continuous DC gain H(0) = J^-1 B (overview.md section 2).

    This holds for EVERY mechanism here, including the matched-TSS control
    whose driven history is identically zero: cancellation removes the
    history, not the static branch.
    """
    from s5.gp_diagnostics import core_from_module, dc_response
    u = inputs(77)
    base = None
    for mech in ("plain", "gp_scalar", "gp_diagonal", "prospective_input",
                 "full_state_pc"):
        kw = dict(mechanism=mech, **ssm_kwargs())
        if mech != "plain":
            kw["gp_init_scale"] = 0.3
        mod = init_gp_ssm(**kw)(step_rescale=1.0)
        v = jax.tree_util.tree_map(lambda z: z.astype(np.float64),
                                   mod.init(jax.random.PRNGKey(0), u))
        core, _, _ = core_from_module(mod, v)
        dc = onp.asarray(dc_response(core)["dc_gain"])
        if base is None:
            base = dc
        else:
            assert onp.max(onp.abs(dc - base)) < 1e-12, mech


def test_theory_prospectivity_does_not_increase_hankel_memory():
    """Increasing instantaneous response does not increase the layer's
    past-to-future operator norm (overview.md section 3).

    Reported as a measured relation on this configuration, not as a proof.
    """
    from s5.gp_diagnostics import core_from_module, hankel_singular_values
    u = inputs(78)
    tops = {}
    for scale in (1e-6, 0.3, 1.0):
        mod = build("gp_diagonal", gp_init_scale=scale)
        v = jax.tree_util.tree_map(lambda z: z.astype(np.float64),
                                   mod.init(jax.random.PRNGKey(0), u))
        core, _, _ = core_from_module(mod, v)
        tops[scale] = hankel_singular_values(core, n=32)["hankel_top"]
    assert tops[1.0] <= tops[0.3] + 1e-12 <= tops[1e-6] + 1e-12, tops
