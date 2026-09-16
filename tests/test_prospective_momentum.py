"""Focused checks for the prospective Momentum DeltaNet continuation.
CLUSTER-ONLY; not run locally.

PREDECLARED TOLERANCES, frozen before any execution (brief s8; reused where
they cover the same observable):
    ID64    1e-9   relative: float64 identities (native nesting, canonical
                   step, identity (7), transfer (8), QHM, streaming)
    GRAD64  1e-8   relative per leaf, absolute floor 1e3 eps64 G
    FD64    1e-6   relative: kappa JVP vs central differences, h = 1e-5, 1e-6
    EXACT64 1e-12  relative: executed 2x2 transition vs A_kappa and (11)

The restored-checkpoint checks need PM_SOURCE_DIR (set by the launcher). A
missing source is a FAILURE, never a skip.

Production float32 coverage: tests/prospective_momentum_float32_probe.py,
own process.
"""

import json
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

import optax                                                       # noqa: E402
from experiments.nested_memory import dynamics as NMD             # noqa: E402
from experiments.nested_memory import model as NM                 # noqa: E402
from experiments.nested_memory import task as TK                  # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import source as SRC        # noqa: E402

ID64, GRAD64, FD64, EXACT64 = 1e-9, 1e-8, 1e-6, 1e-12
FD_STEPS = (1e-5, 1e-6)
F64 = onp.float64
EPS64 = float(onp.finfo(F64).eps)


def _rel(a, b):
    a, b = onp.asarray(a, F64), onp.asarray(b, F64)
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def _ep(seed, i=0):
    b = TK.generate_batch(seed, 2)
    return {k: jnp.asarray(b[k][i]) for k in ("key_id", "val_id", "event",
                                               "label")}


def _slice(ep, lo, hi):
    return {k: v[lo:hi] for k, v in ep.items()}


def _f64(p):
    return {k: jnp.asarray(onp.asarray(v), dtype=jnp.float64)
            for k, v in p.items()}


def _source_native_f64():
    d = os.environ.get("PM_SOURCE_DIR")
    assert d, "PM_SOURCE_DIR not set: the restored checkpoint is REQUIRED"
    st, sel = SRC.load_metadata(d)
    SRC.verify_metadata("momentum_delta", st, sel)
    return _f64(SRC.restore("momentum_delta", d))


def _perturbed_native(seed):
    """A STRESS fixture: official initializer plus random gate perturbations."""
    rs = onp.random.RandomState(seed)
    p = _f64(NM.init_params("momentum_delta", seed))
    for k in ("a_proj", "b_proj", "m_proj", "e_proj"):
        p[k] = p[k] + jnp.asarray(0.7 * rs.randn(*p[k].shape))
    return p


def _gates(rs):
    return dict(alpha=rs.uniform(0.3, 1.0), beta=rs.uniform(0.05, 0.95),
                mu=rs.uniform(0.14, 0.95), eta=rs.uniform(0.1, 1.9))


def _loss_fn(rule, ep):
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)

    def f(p, c0):
        out = PM.rollout(rule, p, ep, carry0=c0)
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(lab, TK.N_VALUES,
                                          dtype=out["logits"].dtype)) * q
        return jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0)
    return f


def _grad_close(ga, gb, G):
    for k in gb:
        d = float(onp.linalg.norm(onp.asarray(ga[k], F64)
                                  - onp.asarray(gb[k], F64)))
        n = float(onp.linalg.norm(onp.asarray(gb[k], F64)))
        assert d <= max(GRAD64 * n, 1e3 * EPS64 * G), (k, d, n)


# ================================= 1. canonical step and its solution =======
@pytest.mark.parametrize("h,kappa", [(1.0, 0.0), (1.0, 0.37), (0.5, 2.5)])
def test_semi_implicit_canonical_step_is_equation_6(h, kappa):
    """Solve (3) as a linear system in (V, P) per entry, with M, gamma, s, T
    from (4), and compare with the implemented (6)."""
    rs = onp.random.RandomState(int(100 * kappa + 10 * h))
    for _ in range(5):
        g = _gates(rs)
        a, b, mu, eta = g["alpha"], g["beta"], g["mu"], g["eta"]
        M, gam, s, T = h / b, (1 - mu) / (b * mu), eta / (mu * h), kappa * h
        W, Q = rs.randn(3, 4), rs.randn(3, 4)
        k = rs.randn(4); k /= onp.linalg.norm(k)
        v = rs.randn(3)
        Wb = a * W
        R = onp.outer(Wb @ k - v, k)
        P_prev = -Q
        A = onp.array([[M, -1.0], [h * gam, 1.0]])
        V = onp.zeros_like(W); P = onp.zeros_like(W)
        for i in range(3):
            for j in range(4):
                V[i, j], P[i, j] = onp.linalg.solve(
                    A, [-T * s * R[i, j], P_prev[i, j] - h * s * R[i, j]])
        W_ref, Q_ref = Wb + h * V, -P
        W_new, Q_new = PD.prospective_step(
            (jnp.asarray(W), jnp.asarray(Q)), jnp.asarray(k), jnp.asarray(v),
            jnp.asarray(1.0), a, b, mu, eta, kappa)
        assert _rel(W_new, W_ref) < ID64 and _rel(Q_new, Q_ref) < ID64


# ============================== 2. identity (7), transfer (8), pulses =======
def _open_loop(R_seq, a, b, mu, eta, kappa, scalar_update=PD.prospective_update):
    W = onp.zeros_like(R_seq[0]); Q = onp.zeros_like(R_seq[0]); out = []
    for R in R_seq:
        W, Q = scalar_update(a * W, Q, R, b, mu, eta, kappa)
        W, Q = onp.asarray(W), onp.asarray(Q)
        out.append(W)
    return onp.array(out)


@pytest.mark.parametrize("kappa", [0.0, 0.6, 3.0])
def test_identity_7_and_transfer_8_on_a_prescribed_residual(kappa):
    from scipy.signal import lfilter
    rs = onp.random.RandomState(7)
    g = _gates(rs)
    a, b, mu, eta = g["alpha"], g["beta"], g["mu"], g["eta"]
    R = rs.randn(40, 2, 3)
    W = _open_loop(R, a, b, mu, eta, kappa)
    q = b * eta
    Wp = onp.concatenate([onp.zeros((2, 2, 3)), W])
    Rp = onp.concatenate([onp.zeros((1, 2, 3)), R])
    lhs = Wp[2:] - (a + mu) * Wp[1:-1] + a * mu * Wp[:-2]
    rhs = -q * (Rp[1:] + kappa * (Rp[1:] - Rp[:-1]))
    assert _rel(lhs, rhs) < ID64
    bnum = [-q * (1 + kappa), q * kappa]
    aden = [1.0, -(a + mu), a * mu]
    W_tf = lfilter(bnum, aden, R, axis=0)
    assert _rel(W, W_tf) < ID64


@pytest.mark.parametrize("kappa", [0.0, 0.5, 2.0])
def test_pulse_immediate_factor_and_unchanged_total_at_alpha_one(kappa):
    mu, b, eta = 0.5, 0.4, 1.3
    q = b * eta
    R = onp.zeros((400, 1, 1)); R[0] = 1.0
    W = _open_loop(R, 1.0, b, mu, eta, kappa)
    assert abs(W[0, 0, 0] - (-q * (1 + kappa))) <= ID64 * q
    assert abs(W[-1, 0, 0] - (-q / (1 - mu))) <= ID64 * q
    if kappa == 0.5:                 # the brief's symbolic example: 3/2
        W0 = _open_loop(R, 1.0, b, mu, eta, 0.0)
        assert abs(W[0, 0, 0] / W0[0, 0, 0] - 1.5) < ID64


def test_gain_changes_both_responses_kappa_only_the_immediate_one():
    mu, b, eta, gval = 0.6, 0.3, 1.1, 1.7
    q = b * eta
    R = onp.zeros((500, 1, 1)); R[0] = 1.0
    Wg = _open_loop(R, 1.0, b, mu, eta, gval, scalar_update=PD.gain_update)
    assert abs(Wg[0, 0, 0] + q * gval) <= ID64 * q
    assert abs(Wg[-1, 0, 0] + q * gval / (1 - mu)) <= ID64 * q
    Wk = _open_loop(R, 1.0, b, mu, eta, gval - 1.0)
    assert abs(Wk[0, 0, 0] - Wg[0, 0, 0]) <= ID64 * q       # same immediate
    assert abs(Wk[-1, 0, 0] + q / (1 - mu)) <= ID64 * q      # total unchanged
    assert abs(Wk[-1, 0, 0] - Wg[-1, 0, 0]) > 0.1 * q


# ========================================== 3. QHM equivalence (10) =========
@pytest.mark.parametrize("kappa", [0.2, 1.7])
def test_qhm_equivalence_closed_loop_with_matched_initial_state(kappa):
    """alpha = 1, fixed gates, state-dependent associative residual with key
    switches and idle tokens; nonzero incoming W and Q."""
    rs = onp.random.RandomState(31)
    mu, b, eta = 0.5, 0.35, 1.2
    nu = 1 - kappa * (1 - mu) / mu
    assert abs(nu) > 0.1                       # the state map needs nu != 0
    a_qhm = b * eta / (1 - mu)
    W0, Q0 = rs.randn(3, 5), rs.randn(3, 5)
    keys = rs.randn(60, 5); keys /= onp.linalg.norm(keys, axis=1)[:, None]
    vals = rs.randn(60, 3)
    m = (rs.rand(60) < 0.6).astype(float)
    W, Q = W0.copy(), Q0.copy()
    Wq, g = W0.copy(), (1 - mu) * Q0 / (eta * nu)
    for t in range(60):
        W, Q = PD.prospective_step((jnp.asarray(W), jnp.asarray(Q)),
                                   jnp.asarray(keys[t]), jnp.asarray(vals[t]),
                                   jnp.asarray(m[t]), 1.0, b, mu, eta, kappa)
        W, Q = onp.asarray(W), onp.asarray(Q)
        R = m[t] * onp.outer(Wq @ keys[t] - vals[t], keys[t])
        g = mu * g + (1 - mu) * R
        Wq = Wq - a_qhm * ((1 - nu) * R + nu * g)
        assert _rel(W, Wq) < ID64, t
        assert _rel((1 - mu) * Q / (eta * nu), g) < ID64, t


# ====================== 4. native nesting at kappa = 0 and g = 1 ============
def _nesting_check(p_native, seeds):
    rs = onp.random.RandomState(5)
    for rule in ("prospective_momentum", "gain_momentum"):
        pc = PM.convert_momentum(p_native, rule)
        for s in seeds:
            ep = _ep(s)
            c0 = (jnp.asarray(0.3 * rs.randn(NM.D_V, NM.D_K)),
                  jnp.asarray(0.3 * rs.randn(NM.D_V, NM.D_K)))
            on = PM.rollout("momentum_delta", p_native, ep, carry0=c0)
            oc = PM.rollout(rule, pc, ep, carry0=c0)
            assert _rel(oc["logits"], on["logits"]) < ID64
            for x, y in zip(oc["final_carry"], on["final_carry"]):
                assert _rel(x, y) < ID64
            # streaming: a chunk boundary inside the episode
            o1 = PM.rollout(rule, pc, _slice(ep, 0, 29), carry0=c0)
            o2 = PM.rollout(rule, pc, _slice(ep, 29, 64),
                            carry0=o1["final_carry"])
            assert _rel(jnp.concatenate([o1["logits"], o2["logits"]]),
                        oc["logits"]) < ID64
            for x, y in zip(o2["final_carry"], oc["final_carry"]):
                assert _rel(x, y) < ID64
            # shared parameter gradients and input (incoming-carry) gradients
            gn = jax.grad(_loss_fn("momentum_delta", ep), argnums=(0, 1))(
                p_native, c0)
            gc = jax.grad(_loss_fn(rule, ep), argnums=(0, 1))(pc, c0)
            G = math.sqrt(sum(float(onp.sum(onp.asarray(v, F64) ** 2))
                              for v in gn[0].values()))
            _grad_close(gc[0], gn[0], G)
            _grad_close(dict(enumerate(gc[1])), dict(enumerate(gn[1])), G)
            extra = PD.EXTRA_LEAF[rule]
            assert onp.isfinite(float(gc[0][extra][0]))


def test_kappa_zero_and_unit_gain_are_native_on_a_stress_fixture():
    _nesting_check(_perturbed_native(17), seeds=(9100, 9101))


def test_kappa_zero_and_unit_gain_are_native_on_the_restored_checkpoint():
    _nesting_check(_source_native_f64(), seeds=(9200,))


def test_idle_token_update_is_native_for_the_same_state_and_gates():
    rs = onp.random.RandomState(8)
    W, Q = jnp.asarray(rs.randn(3, 4)), jnp.asarray(rs.randn(3, 4))
    k = jnp.asarray(rs.randn(4)); v = jnp.asarray(rs.randn(3))
    g = _gates(rs)
    args = (k, v, jnp.asarray(0.0), g["alpha"], g["beta"], g["mu"], g["eta"])
    Wn, Qn = NMD.momentum_delta_step((W, Q), *args)
    for kappa in (0.0, 3.0):
        Wc, Qc = PD.prospective_step((W, Q), *args, kappa)
        assert onp.array_equal(onp.asarray(Wc), onp.asarray(Wn))
        assert onp.array_equal(onp.asarray(Qc), onp.asarray(Qn))
    Wg, Qg = PD.gain_step((W, Q), *args, 1.7)
    assert _rel(Wg, Wn) < EXACT64 and _rel(Qg, Qn) < EXACT64


# ================================= 5. kappa tangent, float64 reference ======
def _kappa_fd(p, kappa0, ep):
    f = _loss_fn("prospective_momentum", ep)
    fk = lambda kk: f(dict(p, kappa=jnp.asarray([kk])), None)  # noqa: E731
    jvp = float(jax.jvp(fk, (jnp.float64(kappa0),), (jnp.float64(1.0),))[1])
    assert onp.isfinite(jvp)
    for h in FD_STEPS:
        fd = (float(fk(kappa0 + h)) - float(fk(kappa0 - h))) / (2 * h)
        assert onp.isfinite(fd)
        print(f"  kappa={kappa0:.4g} jvp {jvp:.6e} fd(h={h}) {fd:.6e}")
        assert abs(jvp - fd) <= FD64 * max(abs(fd), 1e-12), (h, jvp, fd)
    return jvp


def test_kappa_tangent_at_the_restored_start_and_an_interior_point():
    pn = _source_native_f64()
    p = PM.convert_momentum(pn, "prospective_momentum")
    a, b, mu, eta = PD.table_gates(p, PD.write_table())
    kmax = float(jnp.min(PD.kappa_bound(a, b, mu, eta)))
    assert onp.isfinite(kmax) and kmax > 0
    ep = _ep(9300)
    j0 = _kappa_fd(p, 0.0, ep)
    assert j0 != 0.0, "kappa derivative identically zero at the start"
    _kappa_fd(p, 0.5 * kmax, ep)


# ================================= 6. frozen-token transition (11)-(12) =====
def test_jury_expressions_and_bound_on_the_executed_transition():
    rs = onp.random.RandomState(12)
    for i in range(40):
        g = _gates(rs)
        if i % 5 == 0:
            g["alpha"] = 1.0
        a, b, mu, eta = (jnp.asarray([g[k]]) for k in
                         ("alpha", "beta", "mu", "eta"))
        kb = float(PD.kappa_bound(a, b, mu, eta)[0])
        kbar = ((1 + g["alpha"]) * (1 + g["mu"]) - g["alpha"] * g["beta"]
                * g["eta"]) / (2 * g["alpha"] * g["beta"] * g["eta"])
        assert abs(kb - kbar) <= EXACT64 * kbar
        for kappa in (0.0, 0.5 * kb, kb * (1 - 1e-3), kb * (1 + 1e-3)):
            A = onp.asarray(PD.executed_transitions(
                PD.prospective_step, (a, b, mu, eta), jnp.asarray(kappa),
                jnp.float64))[0]
            Aa = PD.analytic_transition(g["alpha"], g["beta"], g["mu"],
                                        g["eta"], kappa)
            assert onp.max(onp.abs(A - Aa)) <= EXACT64 * max(1, onp.max(
                onp.abs(Aa))) * 10
            tr, det = A[0, 0] + A[1, 1], onp.linalg.det(A)
            J = PD.jury(g["alpha"], g["beta"], g["mu"], g["eta"], kappa)
            scale = 1 + abs(tr) + abs(det)
            for x, y in zip((1 - tr + det, 1 + tr + det, 1 - det), J):
                assert abs(x - y) <= 1e-10 * scale
            lab, _ = PD.classify(A)
            rho = onp.max(onp.abs(onp.linalg.eigvals(A)))
            if min(J) > 1e-9:
                assert lab == "stable" and rho < 1
            elif min(J) < -1e-9:
                assert lab == "unstable" and rho > 1
            if kappa > kb:
                assert lab == "unstable"
        Ag = onp.asarray(PD.executed_transitions(
            PD.gain_step, (a, b, mu, eta), jnp.asarray(1.3), jnp.float64))[0]
        assert onp.max(onp.abs(Ag - PD.analytic_gain_transition(
            g["alpha"], g["beta"], g["mu"], g["eta"], 1.3))) <= 1e-11


def test_zero_q_does_not_constrain_kappa_and_rounded_neutral_is_classified():
    a, b, mu, eta = (jnp.asarray([x]) for x in (0.8, 0.0, 0.5, 1.2))
    assert float(PD.kappa_bound(a, b, mu, eta)[0]) == float("inf")
    assert float(PD.gain_bound(a, b, mu, eta)[0]) == float("inf")
    A = onp.asarray(PD.executed_transitions(
        PD.prospective_step, (a, b, mu, eta), jnp.asarray(1e6),
        jnp.float64))[0]
    # beta = 0 removes the immediate effect on W, NOT the residual source in Q
    assert A[0, 1] == 0.0 and A[1, 0] != 0.0
    assert PD.classify(A)[0] == "stable"
    assert PD.classify([[1.0, 0.0], [1.2, 0.5]])[0] == "neutral"
    assert PD.classify([[1.5, 0.0], [0.0, 0.2]])[0] == "unstable"
    assert PD.classify([[float("nan"), 0.0], [0.0, 0.2]])[0] == "nonfinite"


def test_native_gates_satisfy_the_bound_in_exact_arithmetic():
    """(1+alpha)(1+mu) - alpha q > 0 whenever q < 2 and alpha <= 1, so the
    native kappa = 0 and g = 1 are strictly inside."""
    rs = onp.random.RandomState(13)
    for _ in range(200):
        a, mu = rs.uniform(0, 1), rs.uniform(math.exp(-2), 1)
        q = rs.uniform(0, 2)
        assert (1 + a) * (1 + mu) - a * q > 0
        assert a * q < (1 + a) * (1 + mu)


# ====================================== 7. projection with UPDATED gates ====
def _independent_caps(p):
    a, b, mu, eta = (onp.asarray(x, F64) for x in PD.table_gates(
        p, PD.write_table()))
    aq = a * b * eta
    kb = onp.min(onp.where(aq > 0, ((1 + a) * (1 + mu) - aq)
                           / (2 * onp.where(aq > 0, aq, 1)), onp.inf))
    gb = onp.min(onp.where(aq > 0, (1 + a) * (1 + mu)
                           / onp.where(aq > 0, aq, 1), onp.inf))
    return kb, gb


def test_projection_uses_updated_gates_and_keeps_zero_fallback():
    pn = _perturbed_native(19)
    p = PM.convert_momentum(pn, "prospective_momentum")
    kb, _ = _independent_caps(p)
    pq, tel = PD.project(dict(p, kappa=jnp.asarray([1e12])))
    assert abs(float(pq["kappa"][0]) - kb * (1 - PD.PROJ_REL_MARGIN)) \
        <= EXACT64 * kb * 10
    assert int(tel["n_projected"]) == 1
    pz, telz = PD.project(dict(p, kappa=jnp.asarray([-0.3])))
    assert float(pz["kappa"][0]) == 0.0 and int(telz["n_projected"]) == 1
    p0, tel0 = PD.project(p)
    assert float(p0["kappa"][0]) == 0.0 and int(tel0["n_projected"]) == 0
    # UPDATED gates: larger beta lowers the bound; the projection must see it
    p_up = dict(p, b_proj=p["b_proj"] + 3.0, kappa=jnp.asarray([1e12]))
    kb_up, _ = _independent_caps(p_up)
    assert kb_up < kb
    pq_up, _ = PD.project(p_up)
    assert float(pq_up["kappa"][0]) <= kb_up * (1 - PD.PROJ_REL_MARGIN) \
        * (1 + 10 * EXACT64)
    # PROJECTION BEHAVIOUR ONLY: an outward proposal is projected, and a
    # manual inward move of the scalar is then not blocked by the projection.
    # This is not an optimizer or task-gradient recovery test.
    inward = dict(pq, kappa=pq["kappa"] - 0.1 * float(pq["kappa"][0]))
    pin, telin = PD.project(inward)
    assert int(telin["n_projected"]) == 0
    assert float(pin["kappa"][0]) < float(pq["kappa"][0])
    # gain control
    pg = PM.convert_momentum(pn, "gain_momentum")
    _, gb = _independent_caps(pg)
    pgq, _ = PD.project(dict(pg, log_g=jnp.asarray([50.0])))
    assert abs(math.exp(float(pgq["log_g"][0]))
               - gb * (1 - PD.PROJ_REL_MARGIN)) <= 1e-10 * gb
    pg0, telg = PD.project(pg)
    assert float(pg0["log_g"][0]) == 0.0 and int(telg["n_projected"]) == 0
    # a tree without an extension scalar is untouched
    pn2, teln = PD.project(pn)
    assert SRC.jax_leaves_equal(pn2, pn) and int(teln["n_projected"]) == 0


def test_a_real_training_step_projects_from_the_updated_parameters():
    """The cap reported by the compiled step is the bound of the POST-update
    gates, not of the incoming ones. The step starts from a valid interior
    kappa; an outward INCOMING kappa is not used here because nothing
    guarantees a finite forward pass beyond the frozen-token bound. The
    outward-proposal / manual-inward-move sequence below checks projection
    behaviour only, not optimizer recovery (review, reporting)."""
    from experiments.prospective_momentum import study as ST
    p = PM.convert_momentum(_perturbed_native(23), "prospective_momentum")
    kb0, _ = _independent_caps(p)
    p = dict(p, kappa=jnp.asarray([0.5 * kb0]))
    opt = ST.TX.init(p)
    eps = {k: jnp.asarray(v) for k, v in TK.generate_batch(9400, 2).items()
           if k in ("key_id", "val_id", "event", "label")}
    o = ST.train_step("prospective_momentum", p, opt, eps,
                      jnp.asarray(0.01, dtype=jnp.float64))
    p1, tel = o[0], o[8]
    kb1, _ = _independent_caps(p1)
    cap = float(tel["cap"])
    assert abs(cap - kb1 * (1 - PD.PROJ_REL_MARGIN)) <= 1e-10 * kb1
    assert abs(cap - kb0 * (1 - PD.PROJ_REL_MARGIN)) > 1e-10 * kb0
    assert 0.0 <= float(p1["kappa"][0]) <= cap
    assert onp.isfinite(float(o[9])) and float(o[9]) != 0.0
    assert PD.transition_failure(PD.transition_report(
        p1, "prospective_momentum")) is None
    # projection behaviour only: proposal beyond the cap of these updated
    # gates, then a MANUAL inward move (not an optimizer update)
    pq, t1 = PD.project(dict(p1, kappa=jnp.asarray([2.0 * kb1])))
    assert int(t1["n_projected"]) == 1
    pin, t2 = PD.project(dict(pq, kappa=0.9 * pq["kappa"]))
    assert int(t2["n_projected"]) == 0


def test_transition_failure_rejects_kappa_at_or_beyond_the_bound():
    p = PM.convert_momentum(_perturbed_native(29), "prospective_momentum")
    kb, gb = _independent_caps(p)
    assert PD.transition_failure(PD.transition_report(p,
                                 "prospective_momentum")) is None
    bad = dict(p, kappa=jnp.asarray([kb * 1.01]))
    assert PD.transition_failure(PD.transition_report(
        bad, "prospective_momentum")) is not None
    pg = PM.convert_momentum(_perturbed_native(29), "gain_momentum")
    badg = dict(pg, log_g=jnp.asarray([math.log(gb * 1.01)]))
    assert PD.transition_failure(PD.transition_report(
        badg, "gain_momentum")) is not None


# ============================================ 8. contract and bookkeeping ===
def test_input_contract_no_query_value_label_or_future_input():
    pn = _perturbed_native(37)
    for rule in ("prospective_momentum", "gain_momentum"):
        p = PM.convert_momentum(pn, rule)
        if rule == "prospective_momentum":
            p = dict(p, kappa=jnp.asarray([0.4]))
        ep = _ep(9500)
        y = onp.asarray(PM.rollout(rule, p, ep)["logits"])
        q = onp.asarray(ep["event"]) == TK.QUERY
        ep2 = dict(ep, val_id=jnp.where(jnp.asarray(q), 3, ep["val_id"]),
                   label=jnp.full_like(ep["label"], 5))
        assert onp.array_equal(onp.asarray(PM.rollout(rule, p, ep2)["logits"]),
                               y)
        ep3 = dict(ep, key_id=ep["key_id"].at[41:].set(7),
                   event=ep["event"].at[41:].set(TK.WRITE),
                   val_id=ep["val_id"].at[41:].set(2))
        y3 = onp.asarray(PM.rollout(rule, p, ep3)["logits"])
        assert onp.array_equal(y3[:41], y[:41])


def test_counts_carry_and_conversion_refusal():
    pn = _perturbed_native(41)
    assert PM.parameter_counts("momentum_delta", pn)["total"] == 569
    for rule in ("prospective_momentum", "gain_momentum"):
        p = PM.convert_momentum(pn, rule)
        c = PM.parameter_counts(rule, p)
        assert c["total"] == 570 and c["carry_real_numbers"] == 128
        assert set(p) - set(pn) == {PD.EXTRA_LEAF[rule]}
        assert float(p[PD.EXTRA_LEAF[rule]][0]) == 0.0
        for k in pn:
            assert onp.array_equal(onp.asarray(p[k]), onp.asarray(pn[k]))
    with pytest.raises(ValueError):
        PM.convert_momentum(dict(pn, stray=jnp.zeros(1)), "gain_momentum")


def test_streams_are_disjoint_from_each_other_and_previous_studies():
    from experiments.prospective_momentum import study as ST
    assert ST.stream_overlaps() == []
    lo, hi = ST.new_streams()["train"]
    assert lo == 84_000_000 and hi == 84_030_199


def test_screens_are_separate_and_need_complete_pairs():
    from experiments.prospective_momentum import study as ST

    def rows(pc):
        return [dict(rule=r, seed=s, heldout=dict(
            primary=v, retention_revision_untouched=0.6, recall_overall=0.7))
            for s in ST.FINAL_SEEDS for r, v in pc.items()]
    base = dict(prospective_momentum=0.60, momentum_delta=0.55,
                gated_delta=0.55, gain_momentum=0.62, gp_two_sided=0.50,
                tss_eq17=0.595)
    sc = ST.screen(rows(base))
    assert sc["literature_screen_passed"] is True
    assert sc["gain_control_comparison_passed"] is False
    assert sc["old_generalized_comparison_passed"] is True
    assert sc["tss_eq17_comparison_passed"] is False
    sc2 = ST.screen(rows(dict(base, gated_delta=0.595)))
    assert sc2["literature_screen_passed"] is False        # joint screen
    r = [x for x in rows(base) if not (x["rule"] == "gated_delta"
                                       and x["seed"] == ST.FINAL_SEEDS[0])]
    assert ST.screen(r)["literature_screen_passed"] is False
    worse_ret = rows(base)
    for x in worse_ret:
        if x["rule"] == "prospective_momentum":
            x["heldout"] = dict(x["heldout"], retention_revision_untouched=0.58)
    assert ST.screen(worse_ret)["literature_screen_passed"] is False


def test_source_refusals_and_reproduction_rule(tmp_path):
    with pytest.raises(SRC.SourceRefusal):
        SRC.load_metadata(str(tmp_path))
    row = dict(tag="dev", rule="momentum_delta", config="B", seed=300)
    st = dict(run_id=SRC.SOURCE_RUN_ID, dev_seed=300, complete=True,
              development=[row])
    assert SRC.verify_metadata("momentum_delta", st,
                               dict(selected=dict(momentum_delta="B"))) == row
    with pytest.raises(SRC.SourceRefusal):
        SRC.verify_metadata("momentum_delta", st,
                            dict(selected=dict(momentum_delta="A")))
    with pytest.raises(SRC.SourceRefusal):
        SRC.verify_metadata("momentum_delta", dict(st, development=[]),
                            dict(selected=dict(momentum_delta="B")))
    with pytest.raises(SRC.SourceRefusal):
        SRC.verify_metadata("momentum_delta", dict(st, run_id="other"),
                            dict(selected=dict(momentum_delta="B")))
    (tmp_path / "status.json").write_text(json.dumps(st))
    (tmp_path / "selection.json").write_text(json.dumps(
        dict(selected=dict(momentum_delta="B"))))
    with pytest.raises(SRC.SourceRefusal):
        SRC.restore("momentum_delta", str(tmp_path))       # file missing

    def metrics(acc, ce):
        cat = {c: dict(accuracy=acc, cross_entropy=ce, n=100)
               for c in TK.CATEGORIES}
        return {f: dict(by_category=cat) for f in TK.FAMILIES} | dict(
            primary=acc, retention_revision_untouched=acc, recall_overall=acc)
    assert SRC.reproduction_differences(metrics(0.5, 1.0),
                                        metrics(0.51, 1.0))[0] == []
    assert SRC.reproduction_differences(metrics(0.5, 1.0),
                                        metrics(0.52, 1.0))[0] != []
    assert SRC.reproduction_differences(metrics(0.5, 1.0),
                                        metrics(0.5, 1.001))[0] != []


def test_the_restored_source_verifies_and_its_gates_are_valid():
    d = os.environ.get("PM_SOURCE_DIR")
    assert d, "PM_SOURCE_DIR not set"
    st, sel = SRC.load_metadata(d)
    for fam in SRC.SOURCE_FAMILIES:
        SRC.verify_metadata(fam, st, sel)
        SRC.restore(fam, d)
    pn = SRC.restore("momentum_delta", d)
    assert PD.gate_range_report(pn)["gates_valid"]


# ============================ 9. executed gain, finiteness, telemetry (R3) ===
def test_executed_gain_underflow_is_rejected_and_ordinary_gain_accepted():
    """float32 leaves inside the x64 test process: jnp.exp stays float32."""
    pn32 = NM.init_params("momentum_delta", 43)
    pg = PM.convert_momentum(pn32, "gain_momentum")
    assert pg["log_g"].dtype == jnp.float32
    under = dict(pg, log_g=jnp.asarray([-110.0], dtype=jnp.float32))
    rep = PD.transition_report(under, "gain_momentum")
    assert rep["executed_g"] == 0.0 and rep["reference_g_f64"] > 0.0
    assert rep["executed_g_dtype"] == "float32"
    assert PD.transition_failure(rep) is not None
    ok = dict(pg, log_g=jnp.asarray([-0.3], dtype=jnp.float32))
    rep_ok = PD.transition_report(ok, "gain_momentum")
    # exact equality only with the SAME executed value being reported (F3)
    g_exec = PD.executed_gain(ok)
    assert onp.asarray(g_exec).dtype == onp.float32
    assert rep_ok["executed_g"] == float(onp.asarray(g_exec))
    assert onp.isfinite(rep_ok["executed_g"]) and rep_ok["executed_g"] > 0
    assert PD.transition_failure(rep_ok) is None
    # independent exponential: the declared production float32 tolerance
    ref = math.exp(float(onp.float32(-0.3)))
    rel_err = abs(rep_ok["executed_g"] - ref) / ref
    print(f"  executed_g {rep_ok['executed_g']!r} vs float64 exp {ref!r}: "
          f"relative {rel_err:.2e} (TRAJ32 2e-5)")
    assert rel_err <= 2e-5


def _metrics_fixture():
    cat = {c: dict(accuracy=0.5, cross_entropy=1.0, n=10)
           for c in TK.CATEGORIES}
    m = {f: dict(accuracy=0.5, cross_entropy=1.0, macro_accuracy=0.5,
                 by_category=cat) for f in TK.FAMILIES}
    m.update(primary=0.5, retention_revision_untouched=0.5,
             recall_overall=0.5,
             state_norms=dict(W_frobenius_mean=1.0, W_frobenius_max=2.0,
                              aux_frobenius_mean=0.1, aux_frobenius_max=0.2,
                              aux_meaning="Q"))
    return m


def test_metrics_acceptance_includes_state_norms_and_observed_gates():
    from experiments.prospective_momentum import study as ST
    m = _metrics_fixture()
    assert ST.metrics_finite(m)
    bad = _metrics_fixture()
    bad["state_norms"]["aux_frobenius_max"] = float("nan")
    assert not ST.metrics_finite(bad)
    missing = _metrics_fixture()
    missing.pop("state_norms")
    assert not ST.metrics_finite(missing)
    gates = _metrics_fixture()
    gates["observed_rollout_gates"] = dict(finite=False)
    assert not ST.metrics_finite(gates)


def test_observed_rollout_gate_report_uses_supplied_gates():
    ev = onp.array([[TK.WRITE, TK.QUERY], [TK.IDLE, TK.WRITE]])
    a = onp.array([[0.5, 0.1], [0.05, 1.0]])
    b = onp.array([[0.2, 0.9], [0.9, 0.5]])
    mu = onp.array([[0.5, 0.9], [0.9, 0.2]])
    eta = onp.array([[1.0, 1.9], [1.9, 1.5]])
    rep = PD.observed_gate_report("prospective_momentum", (a, b, mu, eta), ev,
                                  kappa=0.1)
    assert rep["n_write_tokens"] == 2 and rep["finite"]
    kb = min(((1 + 0.5) * 1.5 - 0.5 * 0.2) / (2 * 0.5 * 0.2),
             ((1 + 1.0) * 1.2 - 0.75) / (2 * 0.75))
    assert abs(rep["kappa_bound_f64_min"] - kb) <= EXACT64 * kb
    assert rep["write_tokens"]["alpha"]["min"] == 0.5   # non-writes excluded
    assert rep["all_tokens"]["alpha"]["min"] == 0.05


def test_extension_summary_uses_each_parameterizations_units():
    from experiments.prospective_momentum import study as ST
    h = [dict(n_projected=0, pre=0.5, post=0.5, cap=1.0, bound=1.001,
              overshoot=0.0)]
    s1 = ST.summarize_history(h, "prospective_momentum")
    assert abs(s1["min_relative_margin_to_cap"] - 0.5) < 1e-12
    s2 = ST.summarize_history(h, "gain_momentum")
    assert abs(s2["min_log_slack_to_cap"] - 0.5) < 1e-12
    assert abs(s2["min_relative_gain_margin_to_cap"]
               - (1 - math.exp(-0.5))) < 1e-12
    assert "min_relative_margin_to_cap" not in s2
    s3 = ST.summarize_history(h, "gp_two_sided")
    assert "min_relative_gain_margin_to_cap" not in s3
    pn, tel = PD.project(_perturbed_native(47))
    assert all(onp.isnan(float(tel[k])) for k in
               ("pre", "post", "cap", "bound", "overshoot"))


# ================================ 10. finalization and terminal paths (R1/R2)
def _status(before):
    return ({} if before is None
            else dict(source=dict(hashes_at_restore=before)))


def test_finalizer_changed_checksum_forces_failed():
    from experiments.prospective_momentum import study as ST
    saved = []
    st = _status({"a": "1"})
    code, label = ST.finalize(st, 0, "PASS", lambda: {"a": "2"},
                              lambda x: saved.append(json.loads(json.dumps(
                                  x))))
    assert (code, label) == (4, "FAILED")
    assert st["source_unchanged"] is False and not st["integrity_verified"]
    assert st["computation_status"] == "PASS"
    assert saved and saved[-1]["study_status"] == "FAILED"
    st2 = _status({"a": "1"})
    assert ST.finalize(st2, 0, "PASS", lambda: {"a": None},
                       lambda x: None)[0] == 4


def test_finalizer_unavailable_verification_never_passes_and_keeps_reason():
    from experiments.prospective_momentum import study as ST

    def boom():
        raise OSError("disk gone")
    st = _status({"a": "1"})
    assert ST.finalize(st, 0, "PASS", boom, lambda x: None) == \
        (3, "INCOMPLETE")
    assert st["source_unchanged"] is None and not st["integrity_verified"]
    st2 = dict(_status({"a": "1"}), failed="original reason")
    assert ST.finalize(st2, 4, "FAILED", boom, lambda x: None) == \
        (4, "FAILED")
    assert st2["failed"] == "original reason"
    assert any("re-hash failed" in w for w in st2["integrity_failures"])
    st3 = _status(None)
    assert ST.finalize(st3, 0, "PASS", lambda: {"a": "1"},
                       lambda x: None)[0] == 3
    st4 = _status({"a": "1"})
    assert ST.finalize(st4, 0, "PASS", lambda: {"a": "1"},
                       lambda x: None) == (0, "PASS")
    assert st4["source_unchanged"] is True and st4["integrity_verified"]

    def cannot_write(x):
        raise OSError("read-only")
    assert ST.finalize(_status({"a": "1"}), 0, "PASS", lambda: {"a": "1"},
                       cannot_write)[0] == 4


def test_post_restore_runtime_exception_is_finalized():
    from experiments.prospective_momentum import study as ST
    saved = []
    st = dict(_status({"a": "1"}), incomplete=[])

    def body():
        st["development"] = ["partial"]
        raise RuntimeError("crash after restore")
    code, label = ST.guarded(body, st, lambda: {"a": "1"}, saved.append)
    assert (code, label) == (4, "FAILED")
    assert "crash after restore" in st["failed"]
    assert "Traceback" in st["runtime_failures"][0]["traceback"]
    assert st["source_unchanged"] is True and saved
    assert st["development"] == ["partial"]
    st2 = dict(_status({"a": "1"}), incomplete=[])

    def body2():
        st2["failed"] = "dev arm invalid"
        raise RuntimeError("second error")
    ST.guarded(body2, st2, lambda: {"a": "1"}, lambda x: None)
    assert st2["failed"] == "dev arm invalid"
    st3 = dict(_status({"a": "1"}), incomplete=[])

    def body3():
        raise ST.Terminated("signal 15")
    assert ST.guarded(body3, st3, lambda: {"a": "1"}, lambda x: None) == \
        (3, "INCOMPLETE")


def test_terminal_verdict_rules():
    from experiments.prospective_momentum.terminal import terminal_verdict as V
    ok = dict(study_status="PASS", study_exit=0)
    assert V("study", "completed", 0, 0, "complete", True, ok)["label"] == \
        "PASS"
    assert V("study", "completed", 0, 1, "complete", True, ok)["label"] == \
        "FAILED"
    assert V("study", "completed", 0, 2, "complete", True, ok)["label"] == \
        "INCOMPLETE"
    assert V("study", "completed", 0, 3, "complete", True, ok)["label"] == \
        "INCOMPLETE"
    assert V("study", "completed", 0, 4, "complete", True, ok)["label"] == \
        "FAILED"
    assert V("study", "completed", 0, 0, "omitted:no-time", True,
             ok)["label"] == "INCOMPLETE"
    assert V("checks", "completed", 1, 0, "omitted:no-status-json",
             False)["code"] == 4
    assert V("study", "completed", 125, 0, "complete", True, ok)["label"] == \
        "FAILED"                  # a command's own 125 is not "not started"
    v = V("study", "completed", 4, 1, "failed:timeout", True,
          dict(study_status="FAILED", study_exit=4, failed="x"))
    assert v["label"] == "FAILED" and len(v["reasons"]) == 4


def test_not_started_and_supervisor_failure_are_distinct():
    from experiments.prospective_momentum.terminal import terminal_verdict as V
    ns = V("study", "not_started", -1, 0, "omitted:no-status-json", False)
    sf = V("study", "supervisor_failure", -1, 0, "omitted:no-status-json",
           False)
    assert (ns["label"], sf["label"]) == ("INCOMPLETE", "FAILED")
    assert V("checks", "watchdog_kill", -1, 0, "omitted:no-status-json",
             False)["label"] == "INCOMPLETE"
    assert V("study", "completed", 0, 4, "complete", True,
             dict(study_status="PASS", study_exit=0))["label"] == "FAILED"


@pytest.mark.parametrize("saved,outcome,rc,expected", [
    (dict(study_status="FAILED", study_exit=4, failed="numerical failure X"),
     "watchdog_term", -1, "FAILED"),
    (dict(study_status="FAILED", study_exit=4, failed="numerical failure X"),
     "completed", 0, "FAILED"),
    (dict(study_status="PASS", study_exit=0), "watchdog_kill", -1,
     "INCOMPLETE"),
    (dict(study_status="INCOMPLETE", study_exit=3, incomplete=["over budget"]),
     "completed", 0, "INCOMPLETE"),
    (dict(development=[]), "completed", 0, "INCOMPLETE"),     # not finalized
    (dict(study_status="PASS", study_exit=4), "completed", 0, "FAILED"),
    ("{not json", "completed", 0, "FAILED"),
])
def test_terminal_merges_the_saved_study_verdict(tmp_path, saved, outcome, rc,
                                                 expected):
    from experiments.prospective_momentum import terminal as TM
    run, logs = tmp_path / "run", tmp_path / "logs"
    run.mkdir(); logs.mkdir()
    (run / "status.json").write_text(saved if isinstance(saved, str)
                                     else json.dumps(saved))
    TM.main(["--run_dir", str(run), "--log_dir", str(logs), "--stage",
             "study", "--stage_outcome", outcome, "--stage_rc", str(rc),
             "--integrity_rc", "0", "--digest", "complete"])
    v = json.load(open(logs / "terminal.json"))
    assert v["label"] == expected, v
    if isinstance(saved, dict) and saved.get("failed"):
        assert any("numerical failure X" in r for r in v["reasons"])
        assert v["saved_study"]["failed"] == "numerical failure X"
    if not isinstance(saved, str):
        merged = json.load(open(run / "status.json"))
        assert merged["terminal"]["label"] == expected
        for k, val in saved.items():                      # nothing erased
            assert merged[k] == val


# ---- supervisor (review F1): stdlib, dummy children only -------------------
SUPERVISOR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments", "prospective_momentum",
    "supervise.py")


def _executing(pid):
    """True if pid exists and is not a zombie (a surviving executable)."""
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                       capture_output=True, text=True)
    st = r.stdout.strip()
    return bool(st) and not st.startswith("Z")


def _gone_within(pid, seconds=1.5):
    import time
    end = time.time() + seconds
    while time.time() < end:
        if not _executing(pid):
            return True
        time.sleep(0.05)
    return not _executing(pid)


def _kill_quietly(pid_files):
    import signal as _sig
    for f in pid_files:
        try:
            os.kill(int(open(f).read().split()[0]), _sig.SIGKILL)
        except (OSError, ValueError, IndexError):
            pass


def _supervise(tmp_path, script, term_in, grace):
    import time
    out = tmp_path / "outcome"
    t0 = time.time()
    r = subprocess.run([sys.executable, SUPERVISOR, "--term_at",
                        str(t0 + term_in), "--grace", str(grace), "--outcome",
                        str(out), "--", "bash", "-c", script,
                        str(tmp_path)], capture_output=True, text=True,
                       timeout=60)
    elapsed = time.time() - t0
    line = open(out).read().split()
    rec = json.load(open(str(out) + ".json"))
    print(rec, elapsed, r.stderr[-500:])
    assert line == [rec["outcome"], str(rec["rc"])]
    return rec, elapsed


def test_supervisor_kills_when_leader_and_descendant_ignore_term(tmp_path):
    pidf = tmp_path / "gc.pid"
    try:
        rec, el = _supervise(tmp_path, 'trap "" TERM; (exec sleep 60) & '
                             'echo $! > "$0/gc.pid"; wait', 1.0, 1.0)
        assert rec["outcome"] == "watchdog_kill" and rec["rc"] == 137
        assert el <= 5.0
        assert _gone_within(int(pidf.read_text()))
    finally:
        _kill_quietly([pidf])


def test_supervisor_kills_descendant_after_leader_exits_on_term(tmp_path):
    """The GNU-timeout gap: the leader exits on TERM, a descendant ignores
    TERM. The group must still be KILLed at term + grace."""
    pidf = tmp_path / "gc.pid"
    try:
        rec, el = _supervise(tmp_path, '(trap "" TERM; exec sleep 60) & '
                             'echo $! > "$0/gc.pid"; wait', 1.0, 1.0)
        assert rec["term_sent"] and rec["kill_sent"]
        assert rec["outcome"] == "watchdog_kill"
        assert rec["leader_rc"] == 128 + 15
        assert el <= 5.0
        assert _gone_within(int(pidf.read_text()))
    finally:
        _kill_quietly([pidf])


def test_supervisor_cleans_members_left_by_a_completed_leader(tmp_path):
    pidf = tmp_path / "gc.pid"
    try:
        rec, el = _supervise(tmp_path, '(trap "" TERM; exec sleep 60) & '
                             'echo $! > "$0/gc.pid"; exit 0', 30.0, 1.0)
        assert rec["outcome"] == "completed" and rec["rc"] == 0
        assert rec["orphans_cleaned"] and rec["kill_sent"]
        assert el <= 5.0
        assert _gone_within(int(pidf.read_text()))
    finally:
        _kill_quietly([pidf])


def test_supervisor_ordinary_completion_and_start_failure(tmp_path):
    rec, el = _supervise(tmp_path, "exit 3", 30.0, 1.0)
    assert rec["outcome"] == "completed" and rec["rc"] == 3
    assert not rec["term_sent"] and el < 5.0
    out = tmp_path / "bad"
    subprocess.run([sys.executable, SUPERVISOR, "--term_at", "1e12",
                    "--grace", "1", "--outcome", str(out), "--",
                    str(tmp_path / "no-such-executable")], timeout=30)
    assert open(out).read().split() == ["supervisor_failure", "70"]


def test_launcher_terminal_paths_with_dummy_children(tmp_path):
    """Launcher library with the supervisor: explicit not-started state,
    supervisor failure distinct from it, source verification (unchanged,
    changed, no time, verifier failure), digest omission, and the terminal
    verdict merging a saved FAILED study with a watchdog outcome. Dummy
    commands only; no model."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib = os.path.join(repo, "bin", "run_experiments",
                       "prospective_momentum_terminal.sh")
    stub = tmp_path / "broken_supervisor.py"
    stub.write_text("import sys\nsys.exit(1)\n")
    script = r"""
set -u
source "$LIB"
REAL_SUP="$PM_SUPERVISOR"
START=$(date +%s); TOTAL_S=40; DEADLINE=$(( START + 40 ))
pm_bounded $(( $(date +%s) - 1 )) 1 true; echo NOTSTARTED=$PM_OUTCOME
pm_bounded $(( $(date +%s) + 20 )) 1 bash -c 'exit 4'; echo COMPLETED=$PM_OUTCOME/$PM_RC
PM_SUPERVISOR="$STUB"
pm_bounded $(( $(date +%s) + 20 )) 1 true; echo SUPFAIL=$PM_OUTCOME
mkdir -p "$SOURCE_DIR"; echo a > "$SOURCE_DIR/f"
(cd "$SOURCE_DIR" && sha256sum f) > "$LOG_DIR/source_sha256_before.txt"
pm_verify_source > /dev/null; echo VERIFY_SUPFAIL=$PM_INTEGRITY_RC
PM_SUPERVISOR="$REAL_SUP"
pm_verify_source > /dev/null; echo VERIFY_OK=$PM_INTEGRITY_RC
echo b > "$SOURCE_DIR/f"
pm_verify_source > /dev/null; echo VERIFY_CHANGED=$PM_INTEGRITY_RC
DEADLINE=$(( $(date +%s) + 10 ))
pm_verify_source > /dev/null; echo VERIFY_NOTIME=$PM_INTEGRITY_RC
mkdir -p "$LOG_DIR/run"
echo '{"study_status": "FAILED", "study_exit": 4, "failed": "saved failure Y"}' > "$LOG_DIR/run/status.json"
DEADLINE=$(( $(date +%s) + 5 ))
pm_digest "$LOG_DIR/run" > /dev/null; echo DIGEST=$PM_DIGEST
DEADLINE=$(( $(date +%s) + 30 )); PM_INTEGRITY_RC=0; PM_DIGEST=complete
pm_terminal "$LOG_DIR/run" study watchdog_term "" > /dev/null; echo TERMINAL=$PM_LABEL/$PM_CODE
"""
    env = dict(os.environ, LIB=lib, LOG_DIR=str(tmp_path),
               SOURCE_DIR=str(tmp_path / "src"), PY=sys.executable,
               STUB=str(stub))
    env.pop("PM_SUPERVISOR", None)
    r = subprocess.run(["bash", "-c", script], cwd=repo, env=env,
                       capture_output=True, text=True, timeout=90)
    print(r.stdout); print(r.stderr[-2000:])
    out = dict(tok.split("=", 1) for tok in r.stdout.split()
               if "=" in tok and tok.split("=", 1)[0].isupper())
    assert out["NOTSTARTED"] == "not_started"
    assert out["COMPLETED"] == "completed/4"
    assert out["SUPFAIL"] == "supervisor_failure"
    assert out["VERIFY_SUPFAIL"] == "4"
    assert out["VERIFY_OK"] == "0" and out["VERIFY_CHANGED"] == "1"
    assert out["VERIFY_NOTIME"] == "2"
    assert out["DIGEST"] == "omitted:no-time"
    assert out["TERMINAL"] == "FAILED/4"
    st = json.load(open(tmp_path / "run" / "status.json"))
    assert st["terminal"]["label"] == "FAILED"
    assert st["failed"] == "saved failure Y"
    assert any("saved failure Y" in x for x in st["terminal"]["reasons"])


def test_production_float32_probe_in_its_own_process():
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "prospective_momentum_float32_probe.py")
    r = subprocess.run([sys.executable, probe],
                       env=dict(os.environ, JAX_ENABLE_X64="0"),
                       capture_output=True, text=True)
    print(r.stdout[-6000:]); print(r.stderr[-3000:])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
