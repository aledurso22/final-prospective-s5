"""Numerical validation for the fixed-coefficient GP arms and the Rawat port.

TOLERANCES ARE PREDECLARED HERE, before any cluster run, and are not to be
loosened to clear a failure. They follow the independently audited coefficient
gates already in `tests/test_gp_prospective.py`:

    ALGEBRAIC   1e-10  exact identities in float64 (reductions, scan order)
    ODE         1e-6   against independent SciPy real-valued evolution
    GRADIENT    1e-5   parameter and clock gradients vs. central differences
    FLOAT32     1e-4   production dtype against the float64 reference

Coverage required by CLUSTER_CODING_BRIEF s6: initial and trained-like
admissible fixtures; repeated poles; input jumps; values and all
parameter/clock gradients against independent real-valued SciPy evolution;
sequential versus block scan; chunks/reset/nonzero initialization; fixed-buffer
invariance; ordinary/rho=1 and scalar-diagonal reductions; both the production
dtype and a high-precision reference.

These run on CPU here for development. The CLUSTER run of this file on GPU is
what counts as evidence; see `bin/run_experiments/cluster_gpu_checks.sh`.
"""

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
from s5.gp_coefficients import gp_response_coefficients, phi1     # noqa: E402
from s5.gp_fixed import (fixed_m0_coefficients, mass_block_generator,  # noqa: E402
                         mass_block_zoh, mass_scan, mass_scan_sequential,
                         state_counts)
from s5.physical_coefficients import (M0_REFERENCE, SYMMETRIC_REFERENCE,  # noqa: E402
                                      PhysicalResponse, coefficient_table)
from s5.rawat_s5 import (ARMS, SubstrateSSM, init_substrate_ssm,  # noqa: E402
                         two_tap_coefficients)
from s5.ssm import init_S5SSM                                     # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                           # noqa: E402

ALGEBRAIC, ODE, GRADIENT, FLOAT32 = 1e-10, 1e-6, 1e-5, 1e-4
R = SYMMETRIC_REFERENCE


# ------------------------------------------------------------------ fixtures
def admissible(kind="initial", P=4, H=3, seed=0):
    """Admissible (a, b) fixtures.

    `initial`      HiPPO-like: small |a|, the clock-absorbed scale at init.
    `trained_like` larger spread in decay and frequency, as after training.
    `repeated`     exactly repeated poles, to exercise coalescing eigenvectors.
    """
    rs = onp.random.RandomState(seed)
    if kind == "initial":
        a = -0.01 * (1 + rs.rand(P)) + 1j * 0.5 * rs.randn(P)
    elif kind == "trained_like":
        a = -onp.exp(rs.uniform(-4, -0.3, P)) + 1j * rs.uniform(-3, 3, P)
    elif kind == "repeated":
        a = onp.repeat(onp.array([-0.2 + 1.3j, -0.05 + 0.4j]), P // 2)
    else:
        raise ValueError(kind)
    b = rs.randn(P, H) + 1j * rs.randn(P, H)
    return np.asarray(a), np.asarray(b)


def ssm_kw(H=4, ssm=8, blocks=2, conj_sym=True):
    blk = ssm // blocks
    Lam, _, _, V, _ = make_DPLR_HiPPO(blk)
    if conj_sym:
        blk //= 2
        P = ssm // 2
    else:
        P = ssm
    Lam, V = Lam[:blk], V[:, :blk]
    Vc = V.conj().T
    Lam = (Lam * np.ones((blocks, blk))).ravel()
    return dict(H=H, P=P, Lambda_re_init=Lam.real, Lambda_im_init=Lam.imag,
                V=block_diag(*([V] * blocks)), Vinv=block_diag(*([Vc] * blocks)),
                C_init="trunc_standard_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=conj_sym,
                bidirectional=False)


def plain_reference(a, b, x):
    """Ordinary clock-absorbed S5 driven response, h_{-1} = 0."""
    a_bar, b_bar = np.exp(a), phi1(a)[:, None] * b
    h = np.zeros(a.shape, dtype=a.dtype)
    out = []
    for k in range(x.shape[0]):
        h = a_bar * h + b_bar @ x[k]
        out.append(h)
    return np.stack(out)


# -------------------------------------------------------------- coefficients
def test_coefficients_are_the_contract_values_and_immutable():
    assert (R.T, R.gamma, R.rho) == (5.0, 1.0, 0.75)
    assert R.mass == 3.75 and R.mass_physical == 18.75
    assert R.gamma_physical == 5.0
    with pytest.raises(Exception):
        R.T = 9.0                                    # frozen dataclass
    assert all(not row["trainable"] for row in coefficient_table())


def test_mass_tie_is_enforced_not_assumed():
    with pytest.raises(ValueError):
        PhysicalResponse(rho=1.5).validate()
    with pytest.raises(ValueError):
        PhysicalResponse(T=-1.0).validate()
    with pytest.raises(ValueError):
        PhysicalResponse(gamma=0.0).validate()


@pytest.mark.parametrize("kind", ["initial", "trained_like", "repeated"])
def test_m0_equals_the_validated_gp_diagonal_law_at_t_equals_T(kind):
    """The fixed M=0 arm must BE the audited M=0 law, not a re-derivation."""
    a, b = admissible(kind)
    new = fixed_m0_coefficients(a, b, R.T, 1.0)
    old = gp_response_coefficients(a, b, np.full(a.shape, R.T))
    for k in ("a_bar", "b_bar", "d_x"):
        assert float(np.max(np.abs(new[k] - old[k]))) < ALGEBRAIC


@pytest.mark.parametrize("kind", ["initial", "trained_like", "repeated"])
def test_m0_is_stable_wherever_the_poles_are(kind):
    a, b = admissible(kind)
    c = fixed_m0_coefficients(a, b, R.T, 1.0)
    assert float(np.max(np.abs(c["a_bar"]))) < 1.0
    assert float(np.max(c["a_eff"].real)) < 0.0


# ------------------------------------------------------------- positive mass
@pytest.mark.parametrize("kind", ["initial", "trained_like", "repeated"])
def test_rho_one_with_zero_prehistory_is_the_ordinary_driven_response(kind):
    """Stated in the brief as the reduction test; checked, not assumed."""
    a, b = admissible(kind)
    x = np.asarray(onp.random.RandomState(7).randn(12, b.shape[1]))
    z = mass_block_zoh(a, b, R.T, 1.0, 1.0)
    s = mass_scan(z["A_bar"], z["B_bar"], x)[..., 0]
    ref = plain_reference(a, b, x)
    assert float(np.max(np.abs(s - ref)) / np.max(np.abs(ref))) < ALGEBRAIC


@pytest.mark.parametrize("kind", ["initial", "trained_like", "repeated"])
def test_mass_block_satisfies_the_target_law_against_scipy(kind):
    """M s'' + gamma s' + r + T r' = 0 with M = rho gamma T, by INDEPENDENT
    real-valued SciPy evolution of the (s, v) system."""
    from scipy.integrate import solve_ivp
    a, b = admissible(kind)
    A, B = mass_block_generator(a, b, R.T, R.gamma, R.rho)
    x = onp.array([1.0, -0.5, 0.3])
    p = 0
    Ap, Bp = onp.array(A[p]), onp.array(B[p])
    ap, bp = complex(a[p]), onp.array(b[p]) @ x

    def f(t, y):                                   # REAL-valued state
        z = y[:2] + 1j * y[2:]
        d = Ap @ z + Bp @ x
        return onp.concatenate([d.real, d.imag])

    sol = solve_ivp(f, [0, 3], onp.zeros(4), rtol=1e-12, atol=1e-14,
                    dense_output=True)
    t0 = 1.7
    y = sol.sol(t0)
    z = onp.array([y[0] + 1j * y[2], y[1] + 1j * y[3]])
    # Derivatives come from the ODE itself, NOT from finite differences: a
    # central second difference carries O(h^2) truncation error that at the
    # repeated-pole fixture exceeds the predeclared ODE tolerance, which would
    # make the test measure the difference stencil rather than the model.
    zdot = Ap @ z + Bp @ x
    zddot = Ap @ zdot
    s0, sp, spp = z[0], zdot[0], zddot[0]
    J = -ap
    r, rp = J * s0 - bp, J * sp
    resid = R.mass * spp + R.gamma * sp + r + R.T * rp
    assert abs(resid) / max(abs(r), 1e-12) < ODE


@pytest.mark.parametrize("kind", ["initial", "trained_like", "repeated"])
def test_block_scan_matches_sequential(kind):
    a, b = admissible(kind)
    x = np.asarray(onp.random.RandomState(3).randn(17, b.shape[1]))
    z = mass_block_zoh(a, b, R.T, R.gamma, R.rho)
    par = mass_scan(z["A_bar"], z["B_bar"], x)
    seq = mass_scan_sequential(z["A_bar"], z["B_bar"], x)
    assert float(np.max(np.abs(par - seq))) < ALGEBRAIC


def test_nonzero_initialization_and_reset():
    a, b = admissible("trained_like")
    x = np.asarray(onp.random.RandomState(4).randn(9, b.shape[1]))
    z = mass_block_zoh(a, b, R.T, R.gamma, R.rho)
    z0 = np.asarray(onp.random.RandomState(5).randn(a.shape[0], 2)) + 0j
    par = mass_scan(z["A_bar"], z["B_bar"], x, z0=z0)
    seq = mass_scan_sequential(z["A_bar"], z["B_bar"], x, z0=z0)
    assert float(np.max(np.abs(par - seq))) < ALGEBRAIC
    # a reset at k drops the carry: the tail must equal a fresh run
    mask = onp.zeros(9, dtype=bool); mask[4] = True
    cut = mass_scan(z["A_bar"], z["B_bar"], x, reset_mask=np.asarray(mask))
    fresh = mass_scan(z["A_bar"], z["B_bar"], x[4:])
    assert float(np.max(np.abs(cut[4:] - fresh))) < ALGEBRAIC


def test_chunked_streaming_equals_the_full_sequence():
    a, b = admissible("trained_like")
    x = np.asarray(onp.random.RandomState(6).randn(20, b.shape[1]))
    z = mass_block_zoh(a, b, R.T, R.gamma, R.rho)
    full = mass_scan(z["A_bar"], z["B_bar"], x)
    first = mass_scan(z["A_bar"], z["B_bar"], x[:8])
    second = mass_scan(z["A_bar"], z["B_bar"], x[8:], z0=first[-1])
    joined = np.concatenate([first, second], axis=0)
    assert float(np.max(np.abs(full - joined))) < ALGEBRAIC


def test_input_jump_is_handled_at_the_first_token_and_at_a_step():
    """A source jump must not be smoothed away or double counted."""
    a, b = admissible("trained_like")
    H = b.shape[1]
    x = np.concatenate([np.zeros((5, H)), np.ones((5, H))], axis=0)
    z = mass_block_zoh(a, b, R.T, R.gamma, R.rho)
    s = mass_scan(z["A_bar"], z["B_bar"], x)
    assert float(np.max(np.abs(s[:5]))) < ALGEBRAIC      # zero input, at rest
    assert float(np.max(np.abs(s[5]))) > 0.0             # responds immediately
    seq = mass_scan_sequential(z["A_bar"], z["B_bar"], x)
    assert float(np.max(np.abs(s - seq))) < ALGEBRAIC
    # the M=0 arm keeps its current-input term at the FIRST token
    c = fixed_m0_coefficients(a, b, R.T, 1.0)
    x1 = np.ones((1, H))
    assert float(np.max(np.abs(c["d_x"] @ x1[0]))) > 0.0


def test_repeated_poles_do_not_break_the_exponential_or_its_gradient():
    """Coalescing eigenvectors: expm stays analytic, an eigendecomposition
    would not. The gradient must be finite, not merely the value."""
    a, b = admissible("repeated")
    x = np.asarray(onp.random.RandomState(8).randn(6, b.shape[1]))

    def f(scale):
        z = mass_block_zoh(a * scale, b, R.T, R.gamma, R.rho)
        return np.sum(np.abs(mass_scan(z["A_bar"], z["B_bar"], x)) ** 2)

    g = jax.grad(f)(1.0)
    assert onp.isfinite(float(f(1.0))) and onp.isfinite(float(g))


# ---------------------------------------------------------------- gradients
def test_parameter_and_clock_gradients_match_central_differences():
    """All declared trainable directions: pole real/imag, input, and the CLOCK.
    The fixed coefficients are NOT among them and are checked separately."""
    Lam_re = np.asarray([-0.3, -0.05])
    Lam_im = np.asarray([1.1, 0.4])
    B = np.asarray(onp.random.RandomState(9).randn(2, 3))
    logstep = np.asarray([-1.0, -2.0])
    x = np.asarray(onp.random.RandomState(10).randn(7, 3))

    def loss(lr, li, Bm, ls, response="gp_fixed_mass"):
        a = (lr + 1j * li) * np.exp(ls)
        b = np.exp(ls)[:, None] * Bm
        if response == "gp_fixed_mass":
            z = mass_block_zoh(a, b, R.T, R.gamma, R.rho)
            s = mass_scan(z["A_bar"], z["B_bar"], x)[..., 0]
        else:
            c = fixed_m0_coefficients(a, b, R.T, R.gamma)
            a_bar, b_bar = c["a_bar"], c["b_bar"]
            h = np.zeros(a.shape, dtype=a.dtype)
            out = []
            for k in range(x.shape[0]):
                h = a_bar * h + b_bar @ x[k]
                out.append(h + c["d_x"] @ x[k])
            s = np.stack(out)
        return np.sum(np.abs(s) ** 2).real

    for response in ("gp_fixed_m0", "gp_fixed_mass"):
        args = (Lam_re, Lam_im, B, logstep, response)
        for i in range(4):
            g = jax.grad(lambda *a_: loss(*a_, response), argnums=i)(
                Lam_re, Lam_im, B, logstep)
            flat = onp.asarray(g).ravel()
            base = [onp.asarray(v) for v in (Lam_re, Lam_im, B, logstep)]
            for j in range(min(2, flat.size)):
                eps = 1e-6
                up = [v.copy() for v in base]; dn = [v.copy() for v in base]
                up[i].ravel()[j] += eps
                dn[i].ravel()[j] -= eps
                num = float(loss(*[np.asarray(v) for v in up], response)
                            - loss(*[np.asarray(v) for v in dn], response)
                            ) / (2 * eps)
                den = max(abs(num), 1.0)
                assert abs(flat[j] - num) / den < GRADIENT, (
                    response, i, j, flat[j], num)


def test_fixed_coefficients_are_outside_the_gradient_and_the_optimizer():
    """The coefficient policy in code: T, gamma, rho and the mass must not be
    parameters, must receive no optimizer slot and no weight decay."""
    import optax
    from flax.traverse_util import flatten_dict
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(11).randn(6, kw["H"]))
    for arm in ("gp_fixed_m0", "gp_fixed_mass"):
        m = init_substrate_ssm(arm, **kw)()
        v = m.init(jax.random.PRNGKey(0), x)
        names = ["/".join(k) for k in flatten_dict(v["params"])]
        assert set(names) == {"B", "C", "D", "Lambda_im", "Lambda_re",
                              "log_step"}, names
        tx = optax.adamw(1e-3, weight_decay=1e-4)
        st = tx.init(v["params"])
        slots = ["/".join(k) for k in flatten_dict(
            jax.tree_util.tree_map(lambda a: a, st[0].mu))]
        assert all(n in names for n in slots)


def test_changing_a_fixed_coefficient_changes_the_model():
    """Guards against the coefficients being decorative: if T or rho had no
    effect, the 'fixed physical response' claim would be empty."""
    a, b = admissible("trained_like")
    x = np.asarray(onp.random.RandomState(12).randn(8, b.shape[1]))
    base = mass_block_zoh(a, b, R.T, R.gamma, R.rho)
    other_T = mass_block_zoh(a, b, 2.0, R.gamma, R.rho)
    other_rho = mass_block_zoh(a, b, R.T, R.gamma, 0.4)
    s0 = mass_scan(base["A_bar"], base["B_bar"], x)
    assert float(np.max(np.abs(
        s0 - mass_scan(other_T["A_bar"], other_T["B_bar"], x)))) > 1e-6
    assert float(np.max(np.abs(
        s0 - mass_scan(other_rho["A_bar"], other_rho["B_bar"], x)))) > 1e-6


# ------------------------------------------------------------- the Rawat port
def test_native_arm_is_bit_identical_to_upstream_s5():
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(13).randn(9, kw["H"]))
    k = jax.random.PRNGKey(0)
    base = init_S5SSM(clip_eigs=False, **kw)()
    sub = init_substrate_ssm("native_s5", **kw)()
    vb, vs = base.init(k, x), sub.init(k, x)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda p, q: bool(np.array_equal(p, q)), vb, vs))
    assert float(np.max(np.abs(base.apply(vb, x) - sub.apply(vs, x)))) == 0.0


def test_two_tap_law_matches_an_explicit_sequential_reference():
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(14).randn(9, kw["H"]))
    m = init_substrate_ssm("alpha_p_s5", **kw)()
    v = m.init(jax.random.PRNGKey(0), x)
    y = m.apply(v, x)
    c = m.apply(v, method=lambda mm: mm.coefficients())
    Ct = m.apply(v, method=lambda mm: mm.C_tilde)
    D = v["params"]["D"]
    h = np.zeros(c["A_bar"].shape, dtype=c["A_bar"].dtype)
    xprev = np.zeros(kw["H"])
    ref = []
    for t in range(x.shape[0]):
        h = c["A_bar"] * h + c["B_plus"] @ x[t] + c["B_minus"] @ xprev
        xprev = x[t]
        ref.append(2 * (Ct @ h).real + D * x[t])
    assert float(np.max(np.abs(y - np.stack(ref)))) < ALGEBRAIC


def test_delayed_tap_starts_at_zero():
    """x_{-1} = 0: the first token must see only the current tap."""
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(15).randn(4, kw["H"]))
    m = init_substrate_ssm("alpha_p_s5", **kw)()
    v = m.init(jax.random.PRNGKey(0), x)
    c = m.apply(v, method=lambda mm: mm.coefficients())
    from s5.rawat_s5 import two_tap_drive
    d = two_tap_drive(c["B_plus"], c["B_minus"], x)
    assert float(np.max(np.abs(d[0] - c["B_plus"] @ x[0]))) < ALGEBRAIC


def test_alpha_gain_and_clipping_are_separable_attribution_controls():
    """gain_clip_s5 must differ from native_s5 (gain+clip alone do something)
    and from alpha_p_s5 (the second tap does something)."""
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(16).randn(9, kw["H"]))
    k = jax.random.PRNGKey(0)
    ys = {}
    for arm in ("native_s5", "gain_clip_s5", "alpha_p_s5"):
        m = init_substrate_ssm(arm, **kw)()
        ys[arm] = m.apply(m.init(k, x), x)
    assert float(np.max(np.abs(ys["gain_clip_s5"] - ys["native_s5"]))) > 1e-6
    assert float(np.max(np.abs(ys["alpha_p_s5"] - ys["gain_clip_s5"]))) > 1e-6


def test_gp_arms_consume_the_same_gain_scaled_input_map():
    """The comparison is only matched if the GP arms sit on the alpha-scaled
    substrate, i.e. b = Delta * B_c and not Delta * B_tilde."""
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(17).randn(5, kw["H"]))
    m = init_substrate_ssm("gp_fixed_m0", **kw)()
    v = m.init(jax.random.PRNGKey(0), x)

    def read(mm):
        Lam, B_c, Delta = mm._native()
        B_tilde = mm.B[..., 0] + 1j * mm.B[..., 1]
        return B_c, B_tilde, Lam
    B_c, B_tilde, Lam = m.apply(v, method=read)
    expect = (-Lam.real)[:, None] * B_tilde
    assert float(np.max(np.abs(B_c - expect))) < ALGEBRAIC


def test_every_arm_has_the_same_trainable_parameter_count():
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(18).randn(5, kw["H"]))
    k = jax.random.PRNGKey(0)
    counts = {}
    for arm in ARMS:
        m = init_substrate_ssm(arm, **kw)()
        counts[arm] = sum(int(a.size) for a in
                          jax.tree_util.tree_leaves(m.init(k, x)["params"]))
    assert len(set(counts.values())) == 1, counts


def test_state_counts_are_reported_per_kind_and_not_double_counted():
    P, conj = 16, True
    plain = state_counts(P, conj, "other")
    mass = state_counts(P, conj, "gp_fixed_mass")
    assert plain["physical_real"] == 32            # 2P, NOT 2*ssm_size
    assert plain["auxiliary_real"] == 0
    assert mass["auxiliary_real"] == 32
    assert mass["total_real"] == 64
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(19).randn(5, kw["H"]))
    m = init_substrate_ssm("alpha_p_s5", **kw)()
    v = m.init(jax.random.PRNGKey(0), x)
    c = m.apply(v, method=lambda mm: mm.state_counts())
    assert c["previous_input_buffer"] == kw["H"]
    assert c["total_real"] == c["physical_real"] + kw["H"]


def test_unsupported_configurations_are_refused():
    kw = ssm_kw()
    x = np.asarray(onp.random.RandomState(20).randn(4, kw["H"]))
    with pytest.raises(ValueError):
        SubstrateSSM(response="gp_fixed_mass", clip_eigs=False,
                     **kw).init(jax.random.PRNGKey(0), x)
    with pytest.raises(ValueError):
        SubstrateSSM(response="one_tap", step_rescale=2.0, clip_eigs=True,
                     **kw).init(jax.random.PRNGKey(0), x)
    kw_bi = dict(kw); kw_bi.pop("bidirectional")
    with pytest.raises(ValueError):
        SubstrateSSM(response="one_tap", bidirectional=True, clip_eigs=True,
                     **kw_bi).init(jax.random.PRNGKey(0), x)


def test_float32_production_dtype_probe_runs_and_passes():
    """The production dtype is checked in a SUBPROCESS with x64 disabled: a
    double-precision fixture cannot establish a float32 property."""
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cluster_float32_probe.py")
    r = subprocess.run([sys.executable, probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CLUSTER_FLOAT32_OK" in r.stdout


# --------------------------------------------------------- dataset front end
def _write_wav(path, samples_int16, sr=16000):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(samples_int16.tobytes())


def test_wav_decoding_matches_scipy_bit_for_bit(tmp_path):
    """torchaudio 2.11 routes `load` through TorchCodec, absent on the shared
    cluster env. We decode with the standard library instead; this pins the
    convention (int16/32768) against an independent reader."""
    import scipy.io.wavfile as wf
    from dataloaders.speech_commands10 import _decode_wav
    x = (onp.random.RandomState(0).randn(16000) * 8000).astype(onp.int16)
    p = str(tmp_path / "a.wav")
    _write_wav(p, x)
    _, raw = wf.read(p)
    assert float(onp.abs(_decode_wav(p) - raw.astype(onp.float32) / 32768.0
                         ).max()) == 0.0


def test_wav_decoder_refuses_unexpected_formats(tmp_path):
    import wave
    from dataloaders.speech_commands10 import _decode_wav
    p = str(tmp_path / "stereo.wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(onp.zeros(32000, dtype=onp.int16).tobytes())
    with pytest.raises(ValueError):
        _decode_wav(p)
    q = str(tmp_path / "wrongrate.wav")
    _write_wav(q, onp.zeros(8000, dtype=onp.int16), sr=8000)
    with pytest.raises(ValueError):
        _decode_wav(q)


def test_prepare_and_load_round_trip_with_digest_verification(tmp_path):
    """Runs the ACTUAL prepare() entry point on a small synthetic word tree."""
    from dataloaders import speech_commands10 as SC
    root = tmp_path / "sc"
    for i, word in enumerate(SC.WORDS):
        d = root / word
        d.mkdir(parents=True)
        for j in range(4):
            n = 16000 if j % 2 else 11000            # include short clips
            x = (onp.random.RandomState(i * 50 + j).randn(n) * 5000).astype(
                onp.int16)
            _write_wav(str(d / f"s{j}_nohash_0.wav"), x)
    cache = tmp_path / "cache"
    m = SC.prepare(str(root), str(cache))
    assert sum(m["counts"].values()) == 40
    assert m["mfcc"]["n_mfcc"] == 20 and m["mfcc"]["n_fft"] == 200
    data, _ = SC.load(str(cache))
    for split, (x, y) in data.items():
        assert x.shape[1:] == (SC.N_FRAMES, SC.N_MFCC), (split, x.shape)
        assert x.shape[0] == y.shape[0]
    # standardization is fitted on TRAIN only
    xt = onp.asarray(data["train"][0])
    assert abs(float(xt.mean())) < 1e-4 and abs(float(xt.std()) - 1.0) < 1e-3
    # corrupting a cached file must be refused, not silently trained on
    with open(cache / "train_x.npy", "r+b") as fh:
        fh.write(b"\x00")
    with pytest.raises(RuntimeError):
        SC.load(str(cache))


def test_splits_are_deterministic_disjoint_and_cover_every_file(tmp_path):
    from dataloaders import speech_commands10 as SC
    root = tmp_path / "sc"
    for i, word in enumerate(SC.WORDS):
        d = root / word
        d.mkdir(parents=True)
        for j in range(10):
            _write_wav(str(d / f"s{j}.wav"),
                       onp.zeros(16000, dtype=onp.int16))
    a = SC.build_splits(str(root))
    b = SC.build_splits(str(root))
    assert a == b                                        # seed 0 alone decides
    files = {k: {p for p, _ in v} for k, v in a.items()}
    assert not (files["train"] & files["val"])
    assert not (files["train"] & files["test"])
    assert not (files["val"] & files["test"])
    assert sum(len(v) for v in files.values()) == 100
