"""Focused checks for generalized prospective memory around the delta boundary.
CLUSTER-ONLY; not run locally.

PREDECLARED TOLERANCES, frozen before any execution:
    NEST64   1e-9   relative: candidate(rho = 1) vs adaptive_delta logits
    GRAD64   1e-8   relative per leaf, absolute floor 1e3 eps64 G: shared grads
    ZEROT64  1e-9   |dL/d raw_tau| relative to G at rho = 1 (T unidentifiable)
    FD64     1e-6   relative: raw_r JVP vs central differences, h = 1e-5, 1e-6
    LYAP64   1e-10  relative: residual-velocity integral identity (audit eq. 7)
    STORE64  1e-12  relative slack: storage V non-increasing (audit eq. 11)
    EQ17_64  1e-12  absolute: actual TSS Eq. (17) step vs literal reference
    EXACT64  1e-12  absolute: exact identities (Z = 0 at rho = 1, generators)

Production float32 coverage: tests/meta_delta_float32_probe.py, own process.
"""

import math
import os
import subprocess
import sys

import jax
import numpy as onp
import pytest
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.adaptive_memory import dynamics as AD            # noqa: E402
from experiments.meta_delta import calibrate as CAL               # noqa: E402
from experiments.meta_delta import dynamics as MD                 # noqa: E402
from experiments.meta_delta import model as MM                    # noqa: E402
from experiments.nested_memory import model as NM                 # noqa: E402
from experiments.nested_memory import task as TK                  # noqa: E402

NEST64, GRAD64, ZEROT64, FD64 = 1e-9, 1e-8, 1e-9, 1e-6
LYAP64, STORE64, EQ17_64, EXACT64 = 1e-10, 1e-12, 1e-12, 1e-12
FD_STEPS = (1e-5, 1e-6)
F64 = onp.float64


def _ep(seed, i=0):
    b = TK.generate_batch(seed, 2)
    return {k: jnp.asarray(b[k][i]) for k in ("key_id", "val_id", "event",
                                               "label")}


def _loss(rule, ep):
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)

    def f(p):
        out = MM.rollout(rule, p, ep)
        import optax
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(lab, TK.N_VALUES,
                                          dtype=out["logits"].dtype)) * q
        return jnp.sum(ce) / jnp.sum(q)
    return f


def _rel(a, b):
    a, b = onp.asarray(a, F64), onp.asarray(b, F64)
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def _nonzero_gate_fixture(seed, rs):
    """A STRESS fixture, not the trained start: calibrated eta with a NONZERO
    random gate, a delta tree and its candidate twin. Returns (p_delta,
    p_cand)."""
    slot = CAL.configurations()["slots"]["adaptive_delta/A"]
    pd = MM.init_params("adaptive_delta", seed, init_coeffs=slot, dtype=F64)
    pd = dict(pd, gate_u=jnp.asarray(0.5 * rs.randn(31)),
              gate_b=jnp.asarray(0.5 * rs.randn(7)))
    return pd, MM.delta_to_two_sided(pd)


# ============================================== 1. exact delta nesting ======
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_rho_one_is_the_first_order_delta_model_forward_and_gradient(seed):
    """Arbitrary valid episodes (key changes, idle and query intervals), a
    nonzero gate: outputs and shared gradients (common, gate, eta) agree, and
    the unidentifiable raw_tau has zero task gradient."""
    rs = onp.random.RandomState(seed)
    pd, pc = _nonzero_gate_fixture(seed, rs)
    for i in range(2):
        ep = _ep(9000 + seed, i)
        yd = MM.rollout("adaptive_delta", pd, ep)["logits"]
        yc = MM.rollout("gp_two_sided", pc, ep)["logits"]
        assert _rel(yc, yd) < NEST64
        gd = jax.grad(_loss("adaptive_delta", ep))(pd)
        gc = jax.grad(_loss("gp_two_sided", ep))(pc)
        G = math.sqrt(sum(float(onp.sum(onp.asarray(v) ** 2))
                          for v in gd.values()))
        for k in gd:
            d = float(onp.linalg.norm(onp.asarray(gc[k]) - onp.asarray(gd[k])))
            n = float(onp.linalg.norm(onp.asarray(gd[k])))
            assert d <= max(GRAD64 * n, 1e3 * onp.finfo(F64).eps * G), (k, d, n)
        assert float(onp.max(onp.abs(onp.asarray(gc["raw_tau"])))) \
            <= ZEROT64 * G


def test_rho_one_keeps_Z_exactly_zero_on_every_token():
    rs = onp.random.RandomState(3)
    _, pc = _nonzero_gate_fixture(3, rs)
    out = MM.rollout("gp_two_sided", pc, _ep(9100))
    assert float(onp.max(onp.asarray(out["aux_norm"]))) <= EXACT64


def test_documented_map_from_a_saved_delta_tree():
    rs = onp.random.RandomState(4)
    pd, _ = _nonzero_gate_fixture(4, rs)
    pd = dict(pd, raw_eta=jnp.asarray([0.37]))            # a "trained" eta
    pc = MM.delta_to_two_sided(pd)
    for k in pd:
        assert onp.array_equal(onp.asarray(pc[k]), onp.asarray(pd[k])), k
    assert float(pc["raw_r"][0]) == 0.0
    assert float(pc["raw_tau"][0]) == math.log(MD.TAU0)
    ep = _ep(9200)
    assert _rel(MM.rollout("gp_two_sided", pc, ep)["logits"],
                MM.rollout("adaptive_delta", pd, ep)["logits"]) < NEST64
    with pytest.raises(ValueError):
        MM.delta_to_two_sided(dict(pd, stray=jnp.zeros(1)))


# ====================================== 2. the new direction at the start ===
def test_raw_r_tangent_at_a_nonzero_gate_fixture_matches_central_differences():
    rs = onp.random.RandomState(5)
    _, pc = _nonzero_gate_fixture(5, rs)
    f = _loss("gp_two_sided", _ep(9300))

    def fr(r):
        return f(dict(pc, raw_r=jnp.asarray([r])))
    jvp = float(jax.jvp(fr, (0.0,), (1.0,))[1])
    assert abs(jvp) > 1e-8, jvp
    for h in FD_STEPS:
        fd = (float(fr(h)) - float(fr(-h))) / (2 * h)
        assert abs(jvp - fd) / abs(fd) < FD64, (h, jvp, fd)
    # BOTH sides are usable: finite derivatives just below and above rho = 1
    for r in (-1e-3, 1e-3):
        g = float(jax.grad(fr)(r))
        assert onp.isfinite(g) and g != 0.0, (r, g)


def _actual_start(seed):
    """The UNCHANGED initialized trees that training uses: zero gate
    coordinates, calibrated eta, rho = 1, tau = 1 (review R1)."""
    slots = CAL.configurations()["slots"]
    pd = MM.init_params("adaptive_delta", seed,
                        init_coeffs=slots["adaptive_delta/A"], dtype=F64)
    pc = MM.init_params("gp_two_sided", seed,
                        init_coeffs=slots["gp_two_sided/A"], dtype=F64)
    return pd, pc


def test_the_actual_initialized_tree_is_delta_and_its_tangents():
    """At the actual start: gate zero, rho = 1, tau = 1. Nesting in value and
    shared gradients; the raw_r tangent against central differences; the
    raw_tau tangent vanishes. If |jvp| is below the declared float64
    resolvability threshold the limitation is REPORTED (printed and asserted
    as such), not treated as a dead parameter and not replaced by a fixture."""
    pd, pc = _actual_start(300)
    assert float(onp.max(onp.abs(onp.asarray(pc["gate_u"])))) == 0.0
    assert float(pc["raw_r"][0]) == 0.0 and float(pc["raw_tau"][0]) == 0.0
    ep = _ep(9150)
    assert _rel(MM.rollout("gp_two_sided", pc, ep)["logits"],
                MM.rollout("adaptive_delta", pd, ep)["logits"]) < NEST64
    f = _loss("gp_two_sided", ep)
    gd = jax.grad(_loss("adaptive_delta", ep))(pd)
    gc = jax.grad(f)(pc)
    G = math.sqrt(sum(float(onp.sum(onp.asarray(v) ** 2)) for v in gd.values()))
    for k in gd:
        d = float(onp.linalg.norm(onp.asarray(gc[k]) - onp.asarray(gd[k])))
        n = float(onp.linalg.norm(onp.asarray(gd[k])))
        assert d <= max(GRAD64 * n, 1e3 * onp.finfo(F64).eps * G), (k, d, n)
    assert abs(float(gc["raw_tau"][0])) <= ZEROT64 * G

    def fr(r):
        return f(dict(pc, raw_r=jnp.asarray([r])))
    jvp = float(jax.jvp(fr, (0.0,), (1.0,))[1])
    fval = float(fr(0.0))
    threshold = 100 * onp.finfo(F64).eps * max(abs(fval), 1.0) / min(FD_STEPS)
    resolvable = abs(jvp) >= threshold
    print(f"  actual start: raw_r jvp {jvp:.6e} (float64 resolvability "
          f"{threshold:.1e}: {resolvable}), raw_tau grad "
          f"{float(gc['raw_tau'][0]):.3e}")
    assert onp.isfinite(jvp)
    if resolvable:
        for h in FD_STEPS:
            fd = (float(fr(h)) - float(fr(-h))) / (2 * h)
            assert abs(jvp - fd) / abs(fd) < FD64, (h, jvp, fd)


# =============================== 3. mechanism on each side of rho = 1 =======
def _single(eta, tau, rho, W, Z, w, k, v):
    G = MD.two_sided_generator(jnp.asarray(eta), jnp.asarray(tau),
                               jnp.asarray(rho), jnp.asarray(w))
    F = AD.expm2(G)
    W2, Z2 = AD.two_state_step(W, Z, k, v, F, jnp.exp(-AD.H / tau))
    return W2, Z2


@pytest.mark.parametrize("rho,z_sign,idle_sign", [(0.5, 1, -1), (1.0, 0, 0),
                                                  (1.8, -1, 1)])
def test_Z_sign_and_idle_motion_on_each_side(rho, z_sign, idle_sign):
    """Negative residual on a unit key (desired positive update): rho < 1 gives
    Z > 0 and W gives back during idle; rho = 1 keeps Z = 0 and W fixed;
    rho > 1 gives Z < 0 and W continues the write during idle."""
    eta, tau = 0.9, 1.0
    W = jnp.zeros((1, 1)); Z = jnp.zeros((1, 1))
    k = jnp.asarray([1.0]); v = jnp.asarray([1.0])
    W1, Z1 = _single(eta, tau, rho, W, Z, 1.0, k, v)
    W2, Z2 = _single(eta, tau, rho, W1, Z1, 0.0, k, jnp.zeros(1))
    z = float(Z1[0, 0]); dW = float(W2[0, 0] - W1[0, 0])
    if z_sign == 0:
        assert abs(z) <= EXACT64 and abs(dW) <= EXACT64
    else:
        assert onp.sign(z) == z_sign and onp.sign(dW) == idle_sign, (z, dW)


@pytest.mark.parametrize("M,gamma,T,lam", [(0.75, 1.0, 0.5, 1.0),
                                           (2.0, 0.7, 3.0, 0.4),
                                           (0.3, 2.0, 0.1, 5.0)])
def test_residual_velocity_integral_identity(M, gamma, T, lam):
    """Audit eq. (7) against an independent Lyapunov solve, and its ratio to
    matched heavy ball (T = 0). The (0.75, 1, 0.5, 1) case is the audit's
    rho = 3/2 witness with ratio 2/3."""
    from scipy.linalg import solve_continuous_lyapunov
    v0 = 1.3

    def integral(TT):
        A = onp.array([[0.0, 1.0], [-lam / M, -(gamma + TT * lam) / M]])
        Gm = solve_continuous_lyapunov(A.T, -onp.diag([1.0, 0.0]))
        x0 = onp.array([0.0, v0])
        return float(x0 @ Gm @ x0)
    closed = M * M * v0 * v0 / (2 * lam * (gamma + T * lam))
    assert abs(integral(T) - closed) / closed < LYAP64
    ratio = integral(T) / integral(0.0)
    assert abs(ratio - gamma / (gamma + T * lam)) < LYAP64
    if (M, gamma, T, lam) == (0.75, 1.0, 0.5, 1.0):
        assert abs(ratio - 2.0 / 3.0) < LYAP64


# ====================================== 4. switching storage, wider side ====
def _pair_trajectory(eta, tau, rho, steps, rs, dk=4, dv=3):
    k_seq = [rs.randn(dk) for _ in range(steps)]
    k_seq = [kk / onp.linalg.norm(kk) for kk in k_seq]
    v_seq = [rs.randn(dv) for _ in range(steps)]
    w_seq = [0.0 if s % 5 == 4 else float(rs.uniform(0.0, 2.0))
             for s in range(steps)]
    W1, W2 = jnp.asarray(rs.randn(dv, dk)), jnp.asarray(rs.randn(dv, dk))
    Z1, Z2 = jnp.asarray(rs.randn(dv, dk)), jnp.asarray(rs.randn(dv, dk))
    if rho == 1.0:
        Z1 = Z2 = jnp.zeros((dv, dk))
    out = []
    for s in range(steps):
        out.append((W1, Z1, W2, Z2))
        k = jnp.asarray(k_seq[s]); v = jnp.asarray(v_seq[s])
        W1, Z1 = _single(eta, tau, rho, W1, Z1, w_seq[s], k, v)
        W2, Z2 = _single(eta, tau, rho, W2, Z2, w_seq[s], k, v)
    out.append((W1, Z1, W2, Z2))
    return out


@pytest.mark.parametrize("frac", [0.3, 0.95])
def test_storage_is_non_increasing_in_the_wider_sector(frac, seed=21):
    """Identical exogenous keys/values/gates including w = 0 idle, arbitrary
    key switches, the executed gate bound L = 2; rho > 1 with d L < gamma^2."""
    rs = onp.random.RandomState(seed + int(100 * frac))
    eta, tau = 0.9, 1.0
    x = eta * tau * MD.GATE_BOUND_L
    rho_max = x / (x - 1.0)
    rho = 1.0 + frac * (rho_max - 1.0)
    gamma, M, T = 1.0 / eta, tau / eta, tau / rho
    assert (M - gamma * T) * MD.GATE_BOUND_L < gamma * gamma
    traj = _pair_trajectory(eta, tau, rho, 40, rs)
    Vs = []
    for (W1, Z1, W2, Z2) in traj:
        X = onp.asarray(W1 - W2); Y = -gamma * onp.asarray(Z1 - Z2)  # P = -gamma Z
        Vs.append(MD.storage_V(X, Y, gamma, M, T))
    for a, b in zip(Vs, Vs[1:]):
        assert b <= a * (1 + STORE64) + 1e-300, (a, b)


def test_delta_boundary_is_non_expansive_separately():
    """At rho = 1 with Z = 0 the storage formula is singular; the exact delta
    step is checked directly: ||W1 - W2|| is non-increasing."""
    rs = onp.random.RandomState(31)
    traj = _pair_trajectory(0.9, 1.0, 1.0, 40, rs)
    ns = [float(onp.linalg.norm(onp.asarray(W1 - W2)))
          for (W1, _, W2, _) in traj]
    for a, b in zip(ns, ns[1:]):
        assert b <= a * (1 + STORE64), (a, b)


# ================================ 5. projection, optimizer, dtypes ==========
def test_projection_uses_updated_eta_tau_and_keeps_the_delta_fallback():
    rs = onp.random.RandomState(41)
    for _ in range(200):
        eta, tau = float(onp.exp(rs.uniform(-3, 2))), float(onp.exp(rs.uniform(-3, 2)))
        p = dict(raw_eta=jnp.asarray([math.log(eta)], onp.float32),
                 raw_tau=jnp.asarray([math.log(tau)], onp.float32),
                 raw_r=jnp.asarray([float(rs.uniform(-3, 6))], onp.float32))
        q, tel = MD.project_two_sided(p)
        b = float(MD.log_rho_upper(jnp.exp(p["raw_eta"]), jnp.exp(p["raw_tau"]),
                                   onp.float32)[0])
        assert float(q["raw_r"][0]) <= b and b >= 0.0
        if float(p["raw_r"][0]) <= b:
            assert float(q["raw_r"][0]) == float(p["raw_r"][0])
        rep = MD.domain_report(q)
        assert rep["passed"], rep
    # x <= 1: every rho > 1 allowed, nothing projected
    p = dict(raw_eta=jnp.asarray([math.log(0.2)], onp.float32),
             raw_tau=jnp.asarray([0.0], onp.float32),
             raw_r=jnp.asarray([5.0], onp.float32))
    q, tel = MD.project_two_sided(p)
    assert float(q["raw_r"][0]) == 5.0 and int(tel["n_projected"]) == 0


def test_a_real_update_with_an_outward_proposal_stays_certified():
    from experiments.meta_delta import study as ST
    slot = CAL.configurations()["slots"]["gp_two_sided/B"]
    p = MM.init_params("gp_two_sided", 300, init_coeffs=slot)
    p = dict(p, raw_r=jnp.asarray([4.0], jnp.float32))    # far outside
    eps = ST.to_jax(TK.generate_batch(ST.train_stream_seed(300, 0), 8))
    out = ST.train_step("gp_two_sided", p, ST.TX.init(p), eps,
                        jnp.asarray(0.01, jnp.float32))
    q, tel = out[0], out[8]
    assert int(tel["n_projected"]) == 1
    assert ST.validate_coefficients("gp_two_sided", q) is None
    assert ST.all_finite(q) and ST.all_finite(out[1])
    for k, v in q.items():
        assert onp.asarray(v).dtype == onp.float32, k
    g = jax.grad(lambda pp: ST.batch_loss("gp_two_sided", pp, eps)[0])(q)
    assert onp.isfinite(float(g["raw_r"][0]))


def test_scalar_finite_but_generator_nonfinite_is_rejected():
    """Review R2 counterexample: eta ~ 1e20, tau = 1, rho ~ 5e-19 in float32.
    Every response scalar is finite and positive and rho < 1 is the auto-
    accepted side, yet nu = eta/rho ~ 2e38 and the production generator entry
    -nu w overflows at the permitted gate endpoint w = L = 2."""
    p = dict(raw_eta=jnp.asarray([math.log(1e20)], onp.float32),
             raw_tau=jnp.asarray([0.0], onp.float32),
             raw_r=jnp.asarray([math.log(5e-19)], onp.float32))
    rep = MD.domain_report(p)
    assert rep["scalars_finite"] and rep["scalars_positive"], rep
    assert rep["generator_finite"] is False and rep["passed"] is False, rep
    assert onp.isfinite(rep["executed_nu"])


def test_zero_or_nonfinite_executed_scalars_fail_without_raising():
    for raw_eta in (-1e6, float("nan"), 1e6):
        p = dict(raw_eta=jnp.asarray([raw_eta], onp.float32),
                 raw_tau=jnp.asarray([0.0], onp.float32),
                 raw_r=jnp.asarray([0.0], onp.float32))
        rep = MD.domain_report(p)
        assert rep["passed"] is False and "failed" in rep, rep


def test_report_keeps_executed_and_certificate_values_separate():
    p = dict(raw_eta=jnp.asarray([-0.1], onp.float32),
             raw_tau=jnp.asarray([0.0], onp.float32),
             raw_r=jnp.asarray([0.3], onp.float32))
    rep = MD.domain_report(p)
    assert "executed_gamma" in rep and "certificate_f64_gamma" in rep
    assert "gamma" not in rep and "M" not in rep


def test_a_nonfinite_measured_preflight_scalar_refuses_training():
    from experiments.meta_delta import study as ST
    ok = dict(loss=0.5, accuracy=0.4, grad_norm=1.0, update_norm=0.1,
              w_norm=2.0, aux_norm=0.3)
    assert ST.measured_scalar_failures(ok) == []
    for field in ST.SCALAR_NAMES:
        bad = dict(ok, **{field: float("inf")})
        failed = ST.measured_scalar_failures(bad)
        assert failed == [field]
        d = ST.decide_after_preflight(
            10.0, False, [f"gp_two_sided: non-finite measured preflight "
                          f"scalars {failed}"], 500.0)
        assert d is not None and d[0] == 4 and field in d[2]


# ======================= 6. comparators: Eq. (17), attribution, literature ==
def test_eq17_step_matches_a_literal_reference_and_its_amplitudes():
    rs = onp.random.RandomState(51)
    eta, T, h = 0.7, 4.0, 1.0
    W = onp.zeros((3, 4)); f_prev = onp.zeros((3, 4))
    Wj, fj = jnp.asarray(W), jnp.asarray(f_prev)
    k = rs.randn(4); k /= onp.linalg.norm(k); v = rs.randn(3)
    seq = [(1.0, v)] + [(0.0, onp.zeros(3))] * 4
    hist = [W.copy()]
    for w, vv in seq:
        R = w * onp.outer(W @ k - vv, k)
        f = W - eta * R
        W_ref = W + (h / T) * (-W + f) + f - f_prev
        Wj, fj = MD.eq17_step(Wj, fj, jnp.asarray(k), jnp.asarray(vv),
                              jnp.asarray(w), eta, T)
        assert float(onp.max(onp.abs(onp.asarray(Wj) - W_ref))) < EQ17_64
        W, f_prev = W_ref, f
        hist.append(W.copy())
    unit = onp.outer(v, k)
    assert onp.allclose(hist[1], (1 + h / T) * eta * unit, atol=EQ17_64)
    assert onp.allclose(hist[2], (1 + 2 * h / T) * eta * unit, atol=EQ17_64)
    # D1: Delta W_{k+1} = Delta W_k - eta (1+h/T) R_k + eta R_{k-1}. The FIRST
    # idle interval after the write changes the increment (R_{k-1} != 0) ...
    d1, d2 = hist[1] - hist[0], hist[2] - hist[1]
    R0 = 1.0 * onp.outer(hist[0] @ k - v, k)
    assert onp.allclose(d2, d1 + eta * R0, atol=EQ17_64)
    assert not onp.allclose(d2, d1, atol=1e-6)
    # ... and it is preserved only after TWO consecutive zero-residual inputs
    for a, b, c in zip(hist[1:], hist[2:], hist[3:]):
        assert onp.allclose(c - b, b - a, atol=EQ17_64)
    # (I - Df)[X] = eta w (X k) k^T: singular off the key, zero when w = 0
    X = rs.randn(3, 4)
    u = rs.randn(4); u -= (u @ k) * k
    for w in (0.0, 1.0):
        op = eta * w * onp.outer(X @ k, k)
        assert onp.allclose(op @ u, 0.0, atol=EQ17_64)
        if w == 0.0:
            assert onp.allclose(op, 0.0)


def test_eq17_calibration_matches_beta_star():
    cfg = CAL.configurations()
    s = cfg["slots"]["tss_eq17/A"]
    assert abs((1 + 2 / s["T"]) * s["eta"] - cfg["beta_star"]) < 1e-12


def test_attribution_control_is_the_candidate_law_with_T_removed():
    """The (e, z) generator of the candidate tends to the heavy-ball generator
    as T = tau/rho -> 0 at fixed gamma and M; the control starts at the
    candidate's gamma and M."""
    eta, tau, w = 0.9, 1.0, 1.3
    Gc = onp.asarray(MD.two_sided_generator(jnp.asarray(eta), jnp.asarray(tau),
                                            jnp.asarray(1e9), jnp.asarray(w)))
    Gh = onp.asarray(AD.inertial_generator(jnp.asarray(eta), jnp.asarray(tau),
                                           jnp.asarray(w)))
    assert onp.max(onp.abs(Gc - Gh)) < 1e-8
    s = CAL.configurations()["slots"]
    assert s["heavy_ball_same_mass/A"]["raw_eta"] == s["gp_two_sided/A"]["raw_eta"]
    assert s["heavy_ball_same_mass/A"]["raw_tau"] == s["gp_two_sided/A"]["raw_tau"]


def test_candidate_starts_equal_to_the_delta_slot():
    s = CAL.configurations()["slots"]
    for tag in ("A", "B"):
        assert s[f"gp_two_sided/{tag}"]["raw_eta"] == s[f"adaptive_delta/{tag}"]["raw_eta"]
        assert s[f"gp_two_sided/{tag}"]["raw_r"] == 0.0
        assert s[f"gp_two_sided/{tag}"]["lr"] == s[f"adaptive_delta/{tag}"]["lr"]


def test_counts_carry_and_common_tensors():
    s = CAL.configurations()["slots"]
    want = dict(gp_two_sided=433, adaptive_delta=431, heavy_ball_same_mass=432,
                tss_eq17=432, gated_delta=480, momentum_delta=569)
    ref = MM.init_params("adaptive_delta", 7, init_coeffs=s["adaptive_delta/A"])
    for rule in MD.RULES:
        p = MM.init_params(rule, 7, init_coeffs=s[f"{rule}/A"])
        c = MM.parameter_counts(rule, p)
        assert c["total"] == want[rule] and c["common"] == 392, (rule, c)
        for k in MM.COMMON:
            assert onp.array_equal(onp.asarray(p[k]), onp.asarray(ref[k])), k
    assert MD.CARRY["gp_two_sided"] == MD.CARRY["momentum_delta"] == 128


def test_literature_arms_are_the_pinned_implementations():
    for rule in MD.LITERATURE:
        a = MM.init_params(rule, 8)
        b = NM.init_params(rule, 8)
        ep = _ep(9400)
        ya = onp.asarray(MM.rollout(rule, a, ep)["logits"])
        yb = onp.asarray(NM.rollout(rule, b, ep, NM.constants_for(rule))["logits"])
        assert onp.array_equal(ya, yb), rule


def test_streams_are_disjoint_from_each_other_and_previous_studies():
    from experiments.meta_delta import study as ST
    lo = ST.train_stream_seed(min(ST.DEV_SEED, *ST.FINAL_SEEDS), 0)
    hi = ST.train_stream_seed(max(ST.DEV_SEED, *ST.FINAL_SEEDS), ST.UPDATES)
    for s in list(ST.STREAM.values())[1:] + list(ST.PREVIOUS_STREAMS):
        assert not (lo <= s <= hi), s
    for a, b in ST.PREVIOUS_TRAIN_RANGES:
        assert hi < a or lo > b


def test_screens_are_separate_and_need_complete_pairs():
    from experiments.meta_delta import study as ST

    def rows(pc):
        out = []
        for s in ST.FINAL_SEEDS:
            for rule, pr in pc.items():
                out.append(dict(rule=rule, seed=s, heldout=dict(
                    primary=pr, retention_revision_untouched=0.5,
                    recall_overall=0.5)))
        return out
    base = dict(gp_two_sided=0.60, adaptive_delta=0.55,
                heavy_ball_same_mass=0.62, tss_eq17=0.40, gated_delta=0.55,
                momentum_delta=0.58)
    sc = ST.screen(rows(base))
    assert sc["literature_screen_passed"] is True
    assert sc["matched_delta_departure_passed"] is True
    assert sc["eq17_direct_fast_weight_comparison_passed"] is True
    assert "applicability-limited" in sc["eq17_label"]
    assert sc["heavy_ball_family_comparison_passed"] is False
    assert "not causal" in sc["heavy_ball_label"]
    # a literature win cannot hide a loss to the matched delta rule
    lose_delta = dict(base, adaptive_delta=0.62)
    sc2 = ST.screen(rows(lose_delta))
    assert sc2["literature_screen_passed"] is True
    assert sc2["matched_delta_departure_passed"] is False
    assert len(sc2["matched_delta"]["paired_primary_differences"]) == 3
    r = [x for x in rows(base) if not (x["rule"] == "momentum_delta"
                                       and x["seed"] == ST.FINAL_SEEDS[0])]
    assert ST.screen(r)["literature_screen_passed"] is False


def test_preflight_decision_refuses_invalid_state_and_over_budget():
    from experiments.meta_delta import study as ST
    assert ST.decide_after_preflight(10.0, False, ["x"], 500)[0] == 4
    assert ST.decide_after_preflight(float("nan"), False, [], 500)[0] == 4
    assert ST.decide_after_preflight(10.0, True, [], 500)[0] == 3
    assert ST.decide_after_preflight(900.0, False, [], 500)[0] == 3
    assert ST.decide_after_preflight(10.0, False, [], 500) is None


def test_production_float32_probe_in_its_own_process():
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "meta_delta_float32_probe.py")
    r = subprocess.run([sys.executable, probe],
                       env=dict(os.environ, JAX_ENABLE_X64="0"),
                       capture_output=True, text=True)
    print(r.stdout[-5000:]); print(r.stderr[-3000:])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
