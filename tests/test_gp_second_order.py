"""Gates for the second-order (finite-inertia) prototype, brief section 6.

PREDECLARED TOLERANCES (fixed before running):
  GATE_SO_REF_X64    dense real-pair augmented exponential reference   1e-9
  GATE_SO_ENDPOINT   mu = t reproduces plain S5                        1e-11
  GATE_SO_SCAN_X64   block scan vs sequential, value and gradient      1e-10
  GATE_SO_JVP_X64    directional JVP vs finite differences             1e-6
  GATE_SO_F32        x64-disabled production, parallel vs sequential   1e-4

The mu -> 0 limit is REPORTED as a measured convergence, not gated to a fixed
number: it has a boundary layer and its accuracy near tiny mass must be
measured rather than assumed from mathematical stability.
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

from experiments.gp.cue_recall_runner import ssm_kwargs                # noqa: E402
from s5.gp_coefficients import gp_response_coefficients, phi1          # noqa: E402
from s5.gp_second_order import (SecondOrderGPSSM, block_binary_operator,  # noqa: E402
                                init_second_order_ssm, second_order_generator,
                                second_order_readout, second_order_scan,
                                second_order_sequential, second_order_zoh)

GATE_SO_REF_X64 = 1e-9
GATE_SO_ENDPOINT = 1e-11
GATE_SO_SCAN_X64 = 1e-10
GATE_SO_JVP_X64 = 1e-6
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real_rep_scalar(z):
    return onp.array([[z.real, -z.imag], [z.imag, z.real]])


def _real_rep_matrix(M):
    """Complex (n,n) -> real (2n,2n) with each entry a 2x2 rotation block."""
    n = M.shape[0]
    out = onp.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            out[2 * i:2 * i + 2, 2 * j:2 * j + 2] = _real_rep_scalar(M[i, j])
    return out


def _real_rep_input(B):
    """Complex (n,H) -> real (2n,H), rows interleaved (re, im)."""
    n, H = B.shape
    out = onp.zeros((2 * n, H))
    for i in range(n):
        out[2 * i] = B[i].real
        out[2 * i + 1] = B[i].imag
    return out


def _dense_reference(a, b_row, t, mu):
    """INDEPENDENT dense real-pair augmented exponential, via scipy."""
    F, B = second_order_generator(np.asarray([a]), np.asarray(b_row)[None, :],
                                  np.asarray([t]), np.asarray([mu]))
    Fr = _real_rep_matrix(onp.asarray(F)[0])
    Br = _real_rep_input(onp.asarray(B)[0])
    H = Br.shape[1]
    aug = onp.zeros((4 + H, 4 + H))
    aug[:4, :4] = Fr
    aug[:4, 4:] = Br
    E = sp_expm(aug)
    return E[:4, :4], E[:4, 4:]


@pytest.mark.parametrize("case", [
    ("complex", complex(-0.3, 2.1), 0.7, 0.35),
    ("real_pole", complex(-0.8, 0.0), 0.5, 0.25),
    ("small_pole", complex(-1e-3, 0.0), 0.4, 0.2),
    ("fast_osc", complex(-0.05, 6.2831853), 0.9, 0.45),
    ("small_mass", complex(-0.5, 1.0), 0.6, 1e-4),
    ("coalescing", complex(-0.5, -0.8660254037844386), 1.0, 0.75),
])
def test_so_zoh_matches_dense_real_pair_reference(case):
    """Includes the documented exceptional point t=1, mu=0.75,
    j=0.5+i*sqrt(3)/2 where eigenvectors coalesce; the exponential stays
    analytic, which is why no eigendecomposition is used."""
    name, a, t, mu = case
    rng = onp.random.default_rng(0)
    b_row = rng.normal(size=3) + 1j * rng.normal(size=3)
    Ar, Br = _dense_reference(a, b_row, t, mu)
    c = second_order_zoh(np.asarray([a]), np.asarray(b_row)[None, :],
                         np.asarray([t]), np.asarray([mu]))
    got_A = _real_rep_matrix(onp.asarray(c["A_bar"])[0])
    got_B = _real_rep_input(onp.asarray(c["B_bar"])[0])
    nA = max(1.0, onp.linalg.norm(Ar)); nB = max(1.0, onp.linalg.norm(Br))
    assert onp.linalg.norm(got_A - Ar) / nA < GATE_SO_REF_X64, (name, "A")
    assert onp.linalg.norm(got_B - Br) / nB < GATE_SO_REF_X64, (name, "B")


def test_so_endpoint_mu_equals_t_is_exactly_plain_s5():
    """At mu = t, w is unforced; from w0 = 0 the response is plain S5."""
    rng = onp.random.default_rng(3)
    P, H, L = 5, 3, 40
    a = np.asarray((-onp.abs(rng.normal(size=P)) - 0.05)
                   + 1j * rng.normal(size=P))
    b = np.asarray(rng.normal(size=(P, H)) + 1j * rng.normal(size=(P, H)))
    x = np.asarray(rng.normal(size=(L, H)))
    for t_val in (0.2, 1.0, 3.0):
        t = np.full((P,), t_val)
        c = second_order_zoh(a, b, t, t)                  # mu = t exactly
        hs = second_order_scan(c["A_bar"], c["B_bar"], x)
        s = onp.asarray(hs[:, :, 0])
        plain = gp_response_coefficients(a, b, np.zeros(P))
        h = onp.zeros(P, dtype=complex); ref = []
        for xk in onp.asarray(x):
            h = onp.asarray(plain["a_bar"]) * h + onp.asarray(plain["b_bar"]) @ xk
            ref.append(h.copy())
        ref = onp.array(ref)
        err = onp.max(onp.abs(s - ref)) / max(1.0, onp.max(onp.abs(ref)))
        assert err < GATE_SO_ENDPOINT, (t_val, err)


def test_so_small_mass_approaches_the_M0_model_and_convergence_is_measured():
    """mu -> 0 approaches the M=0 model, LINEARLY in mu/t.

    Measured, not assumed. Below mu/t = 1e-6 the augmented exponential
    overflows to NaN, which is why the parameterization is floored at
    MU_RATIO_MIN = 1e-4. That is a numerical limit of this prototype, not a
    mathematical one.
    """
    rng = onp.random.default_rng(5)
    P, H, L = 4, 3, 60
    a = np.asarray((-onp.abs(rng.normal(size=P)) - 0.2)
                   + 1j * rng.normal(size=P) * 0.5)
    b = np.asarray(rng.normal(size=(P, H)) + 1j * rng.normal(size=(P, H)))
    x = np.asarray(rng.normal(size=(L, H)))
    t = np.full((P,), 0.5)

    m0 = gp_response_coefficients(a, b, t)
    h = onp.zeros(P, dtype=complex); ref = []
    for xk in onp.asarray(x):
        h = onp.asarray(m0["a_bar"]) * h + onp.asarray(m0["b_bar"]) @ xk
        ref.append(h + onp.asarray(m0["d_x"]) @ xk)        # physical s = h + D_x x
    ref = onp.array(ref)

    errs = []
    for ratio in (0.5, 0.1, 0.01, 1e-3, 1e-4):
        c = second_order_zoh(a, b, t, t * ratio)
        s = onp.asarray(second_order_scan(c["A_bar"], c["B_bar"], x)[:, :, 0])
        # skip the boundary layer: compare after the transient
        errs.append(onp.max(onp.abs(s[5:] - ref[5:]))
                    / max(1.0, onp.max(onp.abs(ref[5:]))))
    assert errs[-1] < errs[0], errs          # it does converge
    assert errs[-1] < 1e-2, errs             # loose, declared, not tuned
    # convergence is linear: halving the ratio should roughly halve the error
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(r > 2.0 for r in ratios[1:]), ratios

    from s5.gp_second_order import MU_RATIO_MIN, second_order_zoh as _z
    below = _z(a, b, t, t * (MU_RATIO_MIN * 1e-2))
    assert not bool(onp.all(onp.isfinite(onp.asarray(below["A_bar"])))), (
        "if this now stays finite, re-measure MU_RATIO_MIN")


def test_so_block_scan_matches_sequential_value_and_gradient():
    rng = onp.random.default_rng(7)
    P, H, L = 4, 3, 48
    a0 = np.asarray((-onp.abs(rng.normal(size=P)) - 0.1)
                    + 1j * rng.normal(size=P))
    b0 = np.asarray(rng.normal(size=(P, H)) + 1j * rng.normal(size=(P, H)))
    x = np.asarray(rng.normal(size=(L, H)))

    def f(scan, t_raw):
        t = jax.nn.softplus(t_raw)
        c = second_order_zoh(a0, b0, t, 0.4 * t)
        hs = scan(c["A_bar"], c["B_bar"], x)
        return np.sum(np.abs(hs[:, :, 0]) ** 2)

    t_raw = np.full((P,), 0.3)
    va, vb = f(second_order_scan, t_raw), f(second_order_sequential, t_raw)
    assert abs(float(va) - float(vb)) / max(1.0, abs(float(vb))) < GATE_SO_SCAN_X64
    ga = jax.grad(lambda tr: f(second_order_scan, tr))(t_raw)
    gb = jax.grad(lambda tr: f(second_order_sequential, tr))(t_raw)
    onp.testing.assert_allclose(onp.asarray(ga), onp.asarray(gb),
                                rtol=1e-8, atol=1e-10)


def test_so_reset_and_carry_semantics():
    rng = onp.random.default_rng(9)
    P, H, L = 3, 2, 32
    a = np.asarray((-onp.abs(rng.normal(size=P)) - 0.2) + 1j * rng.normal(size=P))
    b = np.asarray(rng.normal(size=(P, H)) + 1j * rng.normal(size=(P, H)))
    x = np.asarray(rng.normal(size=(L, H)))
    t = np.full((P,), 0.5)
    c = second_order_zoh(a, b, t, 0.25 * t)

    # chunks with carry reproduce the full sequence (BOTH s and w are carried)
    full = second_order_sequential(c["A_bar"], c["B_bar"], x)
    half = L // 2
    f1 = second_order_sequential(c["A_bar"], c["B_bar"], x[:half])
    f2 = second_order_sequential(c["A_bar"], c["B_bar"], x[half:], h0=f1[-1])
    onp.testing.assert_allclose(onp.asarray(np.concatenate([f1, f2])),
                                onp.asarray(full), rtol=0, atol=1e-12)

    # a reset clears BOTH components before the interval
    jdx = L // 2
    mask = np.zeros(L, dtype=bool).at[jdx].set(True)
    for scan in (second_order_sequential, second_order_scan):
        out = scan(c["A_bar"], c["B_bar"], x, reset_mask=mask)
        indep = second_order_sequential(c["A_bar"], c["B_bar"], x[jdx:])
        onp.testing.assert_allclose(onp.asarray(out[jdx:]), onp.asarray(indep),
                                    rtol=0, atol=1e-11)


def test_so_module_jvp_matches_finite_differences():
    kw = ssm_kwargs()
    mod = init_second_order_ssm(gp_init_scale=0.3, mu_ratio_init=0.5,
                                **kw)(step_rescale=1.0)
    u = jax.random.normal(jax.random.PRNGKey(0), (24, kw["H"]))
    v = jax.tree_util.tree_map(lambda z: z.astype(np.float64),
                               mod.init(jax.random.PRNGKey(1), u))
    params = v["params"]
    keys = jax.random.split(jax.random.PRNGKey(2),
                            len(jax.tree_util.tree_leaves(params)))
    it = iter(keys)
    dirn = jax.tree_util.tree_map(
        lambda z: jax.random.normal(next(it), z.shape, dtype=z.dtype), params)
    f = lambda p: np.sum(mod.apply({"params": p}, u) ** 2)
    _, tangent = jax.jvp(f, (params,), (dirn,))
    eps = 1e-6
    shift = lambda s: jax.tree_util.tree_map(lambda p, d: p + s * eps * d,
                                             params, dirn)
    fd = (f(shift(+1)) - f(shift(-1))) / (2 * eps)
    assert abs(float(tangent) - float(fd)) / max(1.0, abs(float(fd))) \
        < GATE_SO_JVP_X64


def test_so_mass_and_response_parameters_receive_gradients():
    kw = ssm_kwargs()
    mod = init_second_order_ssm(**kw)(step_rescale=1.0)
    u = jax.random.normal(jax.random.PRNGKey(0), (24, kw["H"]))
    v = mod.init(jax.random.PRNGKey(1), u)
    g = jax.grad(lambda p: np.sum(mod.apply({"params": p}, u) ** 2))(v["params"])
    for k in ("so_response_raw", "so_mu_ratio_raw"):
        assert np.all(np.isfinite(g[k]))
        assert float(np.max(np.abs(g[k]))) > 0.0, k


def test_so_mass_bound_and_stability_are_enforced():
    kw = ssm_kwargs()
    mod = init_second_order_ssm(**kw)(step_rescale=1.0)
    u = jax.random.normal(jax.random.PRNGKey(0), (8, kw["H"]))
    v = jax.tree_util.tree_map(lambda z: z.astype(np.float64),
                               mod.init(jax.random.PRNGKey(1), u))
    # push the ratio parameter hard in both directions; mu must stay in (0, t)
    for fill in (-40.0, 0.0, 40.0):
        vv = dict(params=dict(v["params"],
                              so_mu_ratio_raw=np.full_like(
                                  v["params"]["so_mu_ratio_raw"], fill)))
        c = mod.apply(vv, method=SecondOrderGPSSM.coefficients)
        assert bool(np.all(c["mu"] > 0.0))
        assert bool(np.all(c["mu"] <= c["t"] * (1.0 + 1e-12)))
        ev = onp.linalg.eigvals(onp.asarray(c["A_bar"]))
        assert float(onp.max(onp.abs(ev))) < 1.0, (fill, onp.max(onp.abs(ev)))


def test_so_rejects_unconstrained_and_zero_mass_configurations():
    kw = ssm_kwargs(clip_eigs=False)
    with pytest.raises(ValueError, match="clip_eigs=True"):
        init_second_order_ssm(**kw)(step_rescale=1.0).init(
            jax.random.PRNGKey(0), np.ones((4, kw["H"])))
    kw2 = ssm_kwargs()
    with pytest.raises(ValueError, match="differentiable mass point"):
        init_second_order_ssm(mu_ratio_init=0.0, **kw2)(step_rescale=1.0).init(
            jax.random.PRNGKey(0), np.ones((4, kw2["H"])))
    with pytest.raises(ValueError, match="differentiable mass point"):
        init_second_order_ssm(mu_ratio_init=1e-9, **kw2)(step_rescale=1.0).init(
            jax.random.PRNGKey(0), np.ones((4, kw2["H"])))


def test_so_production_float32_subprocess():
    out = subprocess.run([sys.executable,
                          os.path.join(REPO, "tests", "so_float32_probe.py")],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "SO_X64_DISABLED_OK" in out.stdout
    assert "SO_PRODUCTION_OK" in out.stdout
