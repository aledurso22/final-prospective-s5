"""Focused checks for the nested associative-memory study. CLUSTER-ONLY.

PREDECLARED TOLERANCES, by dtype and route:
    EXACT64   1e-9   rank-one step vs a dense augmented-ODE matrix exponential
    TRAJ32    2e-5   the same over a 64-token float32 trajectory
    GRAD64    1e-5   relative, JVP vs central differences, with an absolute
                     near-zero branch
    GRAD32    2e-2   production float32, at BOTH declared perturbations
    IDENT64   1e-10  analytic limits and the literature reductions
    PARITY32  2e-5   literature rules vs separate literal references
    STREAM32  2e-5   chunked carry vs an unsplit sequence

Float64 checks run on cluster CPU as independent references; the production
path is float32 on GPU. Each is labelled where it matters.
"""

import os
import sys

import jax
import numpy as onp
import pytest
from scipy.linalg import expm as sp_expm

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.nested_memory import dynamics as D                # noqa: E402
from experiments.nested_memory import model as MD                  # noqa: E402
from experiments.nested_memory import task as TK                   # noqa: E402

EXACT64, TRAJ32, GRAD64, GRAD32 = 1e-9, 2e-5, 1e-5, 2e-2
IDENT64, PARITY32, STREAM32 = 1e-10, 2e-5, 2e-5
GRAD_PERTURBATIONS = (1e-2, 3e-3)


# ------------------------------- independent dense-ODE reference -----------
def _dense_interval(W, P, k, v, m, M, gamma, T, h):
    """Advance the FULL (W, P) ODE by an augmented affine matrix exponential.

    Independent of the rank-one formula: it vectorizes the whole matrix ODE,
    builds the linear operator and its constant term directly from

        Wdot = (P - T R)/M ,  Pdot = -gamma P/M + (gamma T/M - 1) R
        R    = m (W k - v) k^T

    and exponentiates `[[A, c], [0, 0]]`. No key normalization is assumed.
    """
    dv, dk = W.shape
    n = dv * dk
    A = onp.zeros((2 * n + 1, 2 * n + 1))
    K = onp.outer(k, k)                       # R = m (W K - v k^T)
    # d vec(W)/dt = -(T m/M) vec(W K) + (1/M) vec(P) + (T m/M) vec(v k^T)
    WK = onp.kron(K.T, onp.eye(dv))           # vec(W K) = (K^T kron I) vec(W)
    A[:n, :n] = -(T * m / M) * WK
    A[:n, n:2 * n] = onp.eye(n) / M
    A[:n, 2 * n] = (T * m / M) * onp.reshape(onp.outer(v, k), -1, order="F")
    c1 = (gamma * T / M - 1.0) * m
    A[n:2 * n, :n] = c1 * WK
    A[n:2 * n, n:2 * n] = -(gamma / M) * onp.eye(n)
    A[n:2 * n, 2 * n] = -c1 * onp.reshape(onp.outer(v, k), -1, order="F")
    z = onp.concatenate([onp.reshape(W, -1, order="F"),
                         onp.reshape(P, -1, order="F"), [1.0]])
    out = sp_expm(A * h) @ z
    return (onp.reshape(out[:n], (dv, dk), order="F"),
            onp.reshape(out[n:2 * n], (dv, dk), order="F"))


def _rank_one(W, P, k, v, m, c):
    W2, P2 = D.two_state_step((jnp.asarray(W), jnp.asarray(P)),
                              jnp.asarray(k), jnp.asarray(v),
                              jnp.asarray(float(m)),
                              jnp.asarray(c["F"]), c["a0"], c["b0"])
    return onp.asarray(W2), onp.asarray(P2)


@pytest.mark.parametrize("case", range(12))
def test_the_exact_step_matches_an_independent_dense_ODE(case):
    """Arbitrary nonzero carry, nonorthogonal key switches, idle intervals and
    several admissible coefficient sets. float64."""
    rs = onp.random.RandomState(case)
    dv, dk = (3, 4) if case % 2 else (4, 3)
    M, gamma, T = [(0.75, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.5, 1.0),
                   (0.25, 0.5, 2.0)][case % 4]
    c = D.prospective_constants(M, gamma, T, 1.0)
    W = rs.randn(dv, dk); P = rs.randn(dv, dk)
    worst = 0.0
    for step in range(5):
        k = rs.randn(dk); k /= onp.linalg.norm(k)      # nonorthogonal switches
        v = rs.randn(dv)
        m = 0.0 if step == 3 else 1.0                  # one idle interval
        Wr, Pr = _dense_interval(W, P, k, v, m, M, gamma, T, 1.0)
        Wg, Pg = _rank_one(W, P, k, v, m, c)
        scale = max(1.0, onp.max(onp.abs(Wr)), onp.max(onp.abs(Pr)))
        worst = max(worst, onp.max(onp.abs(Wg - Wr)) / scale,
                    onp.max(onp.abs(Pg - Pr)) / scale)
        W, P = Wr, Pr
    assert worst < EXACT64, (case, worst)


def test_the_exact_step_holds_over_a_64_token_float32_trajectory():
    """Production dtype, a full episode length. Reference in float64."""
    rs = onp.random.RandomState(7)
    dv = dk = 8
    c = D.prospective_constants()
    W64 = onp.zeros((dv, dk)); P64 = onp.zeros((dv, dk))
    W32 = jnp.zeros((dv, dk), jnp.float32); P32 = jnp.zeros((dv, dk), jnp.float32)
    F32 = jnp.asarray(c["F"], jnp.float32)
    worst = 0.0
    for t in range(64):
        k = rs.randn(dk); k /= onp.linalg.norm(k)
        v = rs.randn(dv); m = float(t % 4 != 3)
        W64, P64 = _dense_interval(W64, P64, k, v, m, c["M"], c["gamma"],
                                   c["T"], c["h"])
        W32, P32 = D.two_state_step(
            (W32, P32), jnp.asarray(k, jnp.float32), jnp.asarray(v, jnp.float32),
            jnp.float32(m), F32, jnp.float32(c["a0"]), jnp.float32(c["b0"]))
        assert W32.dtype == onp.float32 and P32.dtype == onp.float32
        scale = max(1.0, onp.max(onp.abs(W64)))
        worst = max(worst, onp.max(onp.abs(onp.asarray(W32) - W64)) / scale)
    print(f"  64-token float32 vs float64 dense ODE: {worst:.3e}")
    assert worst < TRAJ32, worst


# ------------------------------------------------------ analytic limits ----
def test_the_ordinary_gradient_limit_at_M_equals_gamma_T():
    """`gamma T = M` with `P0 = 0` keeps `P` zero for ANY history of changing
    keys and masks, and reduces to the declared delta tap."""
    c = D.prospective_constants(M=1.0, gamma=1.0, T=1.0)
    assert abs(c["F"][1, 0]) < IDENT64, "F21 must vanish at the boundary"
    assert abs((1 - c["F"][0, 0]) - D.ordinary_gradient_beta()) < IDENT64
    rs = onp.random.RandomState(3)
    W = rs.randn(4, 4); P = onp.zeros((4, 4))
    for step in range(6):
        k = rs.randn(4); k /= onp.linalg.norm(k)
        W, P = _rank_one(W, P, k, rs.randn(4), float(step % 3 != 2), c)
        assert onp.max(onp.abs(P)) < IDENT64, (step, onp.max(onp.abs(P)))


def test_one_write_from_rest_matches_the_declared_fixed_delta_tap():
    """`W' = (1 - F11) v k^T` from `W = P = 0`, which is the comparator's beta."""
    c = D.prospective_constants()
    k = onp.zeros(8); k[2] = 1.0
    v = onp.arange(1.0, 9.0)
    W, P = _rank_one(onp.zeros((8, 8)), onp.zeros((8, 8)), k, v, 1.0, c)
    assert onp.max(onp.abs(W - D.beta_match() * onp.outer(v, k))) < IDENT64
    # and the auxiliary left behind is exactly -F21 v k^T, per the audit
    assert onp.max(onp.abs(P + c["F"][1, 0] * onp.outer(v, k))) < IDENT64


def test_the_inertial_control_matches_its_OWN_dense_ODE():
    """The no-prospective arm is `M Wddot + gamma Wdot + R = 0`, checked
    against its own dense ODE - NOT against a residual-law approximation."""
    rs = onp.random.RandomState(11)
    c = D.inertial_constants()
    M, gamma = c["M"], c["gamma"]
    W = rs.randn(3, 3); P = rs.randn(3, 3)
    for step in range(4):
        k = rs.randn(3); k /= onp.linalg.norm(k)
        v = rs.randn(3); m = float(step != 2)
        Wr, Pr = _dense_interval(W, P, k, v, m, M, gamma, 0.0, 1.0)
        Wg, Pg = _rank_one(W, P, k, v, m, c)
        s = max(1.0, onp.max(onp.abs(Wr)))
        assert onp.max(onp.abs(Wg - Wr)) / s < EXACT64, step
        assert onp.max(onp.abs(Pg - Pr)) / s < EXACT64, step
        W, P = Wr, Pr


def test_the_zero_key_branch_is_defined_and_suppresses_the_write():
    """A raw vector below the floor is never treated as exactly unit."""
    x = jnp.zeros((3, 8)).at[0].set(1.0).at[1].set(1e-12)
    k, valid = D.safe_normalize(x)
    assert float(valid[0]) == 1.0 and float(valid[1]) == 0.0
    assert float(jnp.max(jnp.abs(k[1]))) == 0.0
    assert abs(float(jnp.linalg.norm(k[0])) - 1.0) < 1e-6
    g = jax.grad(lambda z: jnp.sum(D.safe_normalize(z)[0] ** 2))(x)
    assert bool(jnp.all(jnp.isfinite(g))), "the safe branch must not emit NaN"


# ----------------------------------------- analytical-supplement checks ----
def test_the_difference_Lyapunov_function_never_increases():
    """Audit s3: on IDENTICAL inputs, initial-state differences do not grow in
    the weighted norm `V`. Reported before and after each exact step."""
    c = D.prospective_constants()
    rs = onp.random.RandomState(5)
    W1, P1 = rs.randn(4, 4), rs.randn(4, 4)
    W2, P2 = rs.randn(4, 4), rs.randn(4, 4)
    V = [D.lyapunov_V(W1 - W2, P1 - P2)]
    for step in range(10):
        k = rs.randn(4); k /= onp.linalg.norm(k)
        v = rs.randn(4); m = float(step % 3 != 2)
        W1, P1 = _rank_one(W1, P1, k, v, m, c)
        W2, P2 = _rank_one(W2, P2, k, v, m, c)
        V.append(D.lyapunov_V(W1 - W2, P1 - P2))
    print("  V:", " ".join(f"{x:.4f}" for x in V))
    for a, b in zip(V, V[1:]):
        assert b <= a + 1e-9, (a, b)


def test_the_persistent_transient_coordinates_are_an_exact_change_of_basis():
    """`C = W + P/gamma`, `Z = -P/gamma`, `W = C + Z`. Coordinates of the
    existing system, not added states; the readout is unchanged."""
    rs = onp.random.RandomState(9)
    W, P = rs.randn(3, 5), rs.randn(3, 5)
    C, Z = D.persistent_transient(W, P)
    assert onp.max(onp.abs((C + Z) - W)) < IDENT64
    # on a no-write interval C is constant and Z decays by a0
    c = D.prospective_constants()
    W2, P2 = _rank_one(W, P, onp.eye(5)[0], onp.zeros(3), 0.0, c)
    C2, Z2 = D.persistent_transient(W2, P2)
    assert onp.max(onp.abs(C2 - C)) < 1e-9, "C must not move on an idle step"
    assert onp.max(onp.abs(Z2 - c["a0"] * Z)) < 1e-9


def test_the_response_coefficients_match_the_audit_algebra():
    """Audit s4: the comparator matches the WRITE-END response, and the
    first-query response is smaller. Recorded, not used to change beta."""
    r = D.response_coefficients()
    x, y = onp.exp(-2.0 / 3.0), onp.exp(-2.0)
    assert abs(r["F11"] - (x + y) / 2) < IDENT64
    assert abs(r["F21"] - (x - y) / 4) < IDENT64
    assert r["F21"] > 0
    assert r["beta_query"] < r["beta_write"], "F21 > 0 implies a smaller query"
    assert r["beta_settled"] < r["beta_query"]
    # the comparator uses beta_write exactly, unchanged
    assert abs(D.beta_match() - r["beta_write"]) < IDENT64


# --------------------------------------------------- literature parity ----
def _ref_gated_delta(W, k, v, m, alpha, beta):
    """A separate LITERAL reference: Wbar = alpha W, then the delta write."""
    Wb = alpha * W
    return Wb + beta * m * onp.outer(v - Wb @ k, k)


def _ref_momentum(W, Q, k, v, m, alpha, beta, mu, eta):
    """Literal transpose of the official recurrence:
    `M_t = mu M + (eta k) w^T`, `S_t = alpha S - beta M_t`, `w = alpha S^T k - v`."""
    S, Mm = W.T, Q.T                              # official orientation
    w = alpha * (S.T @ k) - v
    Mm = mu * Mm + onp.outer(eta * k, m * w)
    S = alpha * S - beta * Mm
    return S.T, Mm.T


def test_the_literature_rules_match_separate_literal_references():
    rs = onp.random.RandomState(13)
    dv = dk = 6
    W = rs.randn(dv, dk).astype(onp.float32)
    Q = rs.randn(dv, dk).astype(onp.float32)
    Wg, Wm, Qm = W.copy(), W.copy(), Q.copy()
    worst_g = worst_m = 0.0
    for t in range(32):
        k = rs.randn(dk).astype(onp.float32); k /= onp.linalg.norm(k)
        v = rs.randn(dv).astype(onp.float32); m = float(t % 3 != 2)
        alpha, beta = float(rs.uniform(0.5, 1.0)), float(rs.uniform(0.1, 1.0))
        mu, eta = float(rs.uniform(0.0, 0.9)), float(rs.uniform(0.2, 2.0))
        (g,) = D.gated_delta_step((jnp.asarray(Wg),), jnp.asarray(k),
                                  jnp.asarray(v), jnp.float32(m), alpha, beta)
        ref_g = _ref_gated_delta(Wg, k, v, m, alpha, beta)
        worst_g = max(worst_g, onp.max(onp.abs(onp.asarray(g) - ref_g))
                      / max(1.0, onp.max(onp.abs(ref_g))))
        Wg = onp.asarray(g)
        w2, q2 = D.momentum_delta_step((jnp.asarray(Wm), jnp.asarray(Qm)),
                                       jnp.asarray(k), jnp.asarray(v),
                                       jnp.float32(m), alpha, beta, mu, eta)
        ref_w, ref_q = _ref_momentum(Wm, Qm, k, v, m, alpha, beta, mu, eta)
        worst_m = max(worst_m, onp.max(onp.abs(onp.asarray(w2) - ref_w))
                      / max(1.0, onp.max(onp.abs(ref_w))))
        Wm, Qm = onp.asarray(w2), onp.asarray(q2)
    print(f"  gated parity {worst_g:.3e}   momentum parity {worst_m:.3e}")
    assert worst_g < PARITY32 and worst_m < PARITY32


def test_momentum_with_mu_zero_and_eta_one_reduces_to_gated_delta():
    rs = onp.random.RandomState(17)
    W = jnp.asarray(rs.randn(5, 5)); Q = jnp.zeros((5, 5))
    Wg = W
    for t in range(16):
        k = jnp.asarray(rs.randn(5) / onp.linalg.norm(rs.randn(5) * 0 + 1))
        k = k / jnp.linalg.norm(k)
        v = jnp.asarray(rs.randn(5)); m = jnp.float32(t % 4 != 3)
        a, b = 0.9, 0.4
        W, Q = D.momentum_delta_step((W, Q), k, v, m, a, b, 0.0, 1.0)
        (Wg,) = D.gated_delta_step((Wg,), k, v, m, a, b)
        assert float(jnp.max(jnp.abs(W - Wg))) < IDENT64, t


def test_the_gates_obey_the_pinned_official_parameterization():
    """Ranges and the official mu clamp, on the executed gate block."""
    rs = onp.random.RandomState(19)
    x = jnp.asarray(rs.randn(256, MD.GATE_DIM) * 3.0)
    p = MD.init_params("momentum_delta", 100)
    alpha, beta, mu, eta = MD._momentum_gates(p, x)
    occ = dict(alpha=[float(alpha.min()), float(alpha.max())],
               beta=[float(beta.min()), float(beta.max())],
               mu=[float(mu.min()), float(mu.max())],
               eta=[float(eta.min()), float(eta.max())],
               mu_at_clamp=float(jnp.mean(
                   jnp.abs(jnp.log(mu) - MD.MIN_LOG_MU) < 1e-6)))
    print("  gate ranges / boundary occupancy:", occ)
    assert 0.0 < alpha.min() and alpha.max() <= 1.0 + 1e-6
    assert 0.0 <= beta.min() and beta.max() <= 1.0 + 1e-6
    assert float(jnp.min(jnp.log(mu))) >= MD.MIN_LOG_MU - 1e-6
    assert 0.0 <= eta.min() and eta.max() <= 2.0 + 1e-6
    ga, gb = MD._gated_delta_gates(MD.init_params("gated_delta", 100), x)
    assert 0.0 < ga.min() and ga.max() <= 1.0 + 1e-6
    assert 0.0 <= gb.min() and gb.max() <= 1.0 + 1e-6


# ------------------------------------------ causality and data integrity --
@pytest.mark.parametrize("rule", list(D.RULES))
def test_a_future_token_cannot_change_an_earlier_prediction(rule):
    b = TK.generate_batch(4242, 1)
    ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event")}
    const = MD.constants_for(rule)
    p = MD.init_params(rule, 100)
    a = MD.rollout(rule, p, ep, const)["logits"]
    ep2 = dict(ep)
    ep2["key_id"] = ep["key_id"].at[40].set((int(ep["key_id"][40]) + 5) % 32)
    ep2["val_id"] = ep["val_id"].at[40].set(3)
    c = MD.rollout(rule, p, ep2, const)["logits"]
    assert float(jnp.max(jnp.abs(a[:40] - c[:40]))) == 0.0, rule
    assert float(jnp.max(jnp.abs(a[40:] - c[40:]))) > 0.0, rule


@pytest.mark.parametrize("rule", list(D.RULES))
def test_changing_only_the_oracle_labels_cannot_change_predictions(rule):
    b = TK.generate_batch(4243, 1)
    ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event")}
    const = MD.constants_for(rule)
    p = MD.init_params(rule, 100)
    a = MD.rollout(rule, p, ep, const)["logits"]
    b["label"] = (b["label"] + 1) % TK.N_VALUES          # oracle only
    c = MD.rollout(rule, p, ep, const)["logits"]
    assert float(jnp.max(jnp.abs(a - c))) == 0.0, rule


def test_the_task_supplies_no_value_at_a_query_and_the_oracle_is_exact():
    b = TK.generate_batch(4244, 8)
    s = TK.structure_check(b)
    assert s["query_value_field_is_absent"] and s["query_has_no_write_target"]
    assert s["every_query_has_a_label"] and s["no_label_outside_queries"]
    assert s["queries_per_sequence"] == [TK.N_QUERIES]
    assert set(s["category_counts"].values()) == {4 * 16}
    assert TK.oracle_is_exact(b)
    assert s["age_by_category"]["immediate_selected"]["max"] == 1


@pytest.mark.parametrize("rule", list(D.RULES))
def test_batched_and_per_example_execution_agree(rule):
    b = TK.generate_batch(4245, 2)
    const = MD.constants_for(rule)
    p = MD.init_params(rule, 100)
    eps = {k: jnp.asarray(b[k]) for k in ("key_id", "val_id", "event")}
    batched = jax.vmap(lambda e: MD.rollout(rule, p, e, const)["logits"])(eps)
    for i in range(b["event"].shape[0]):
        one = MD.rollout(rule, p, {k: eps[k][i] for k in eps}, const)["logits"]
        assert float(jnp.max(jnp.abs(one - batched[i]))) < IDENT64, (rule, i)


# ---------------------------------------- gradients, streaming, learning --
def _raw_embedding_probe(rule, p, ep, const, direction, eps):
    """Perturb the RAW key embeddings before normalization."""
    q = dict(p)
    q["key_raw"] = p["key_raw"] + eps * direction
    out = MD.rollout(rule, q, ep, const)
    return jnp.sum(out["logits"] ** 2)


@pytest.mark.parametrize("rule", list(D.RULES))
def test_gradients_match_finite_differences_on_raw_embeddings(rule):
    """JVP through the ENTIRE executed recurrence and readout against central
    differences on the raw embeddings, before normalization. float64."""
    b = TK.generate_batch(4246, 1)
    ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event")}
    const = MD.constants_for(rule)
    p = MD.init_params(rule, 100, dtype=onp.float64)
    rs = onp.random.RandomState(23)
    dirn = jnp.asarray(rs.randn(*p["key_raw"].shape))
    dirn = dirn / jnp.linalg.norm(dirn)
    f = lambda e: _raw_embedding_probe(rule, p, ep, const, dirn, e)  # noqa: E731
    ana = float(jax.grad(f)(0.0))
    best = None
    for h in (1e-4, 1e-5, 1e-6):
        num = float((f(h) - f(-h)) / (2 * h))
        rel = abs(num - ana) / max(abs(ana), 1e-300)
        best = rel if best is None else min(best, rel)
    print(f"  {rule}: |dL|={abs(ana):.3e}  best rel={best:.3e}")
    assert best < GRAD64 or abs(ana) < 1e-8, (rule, best, ana)


@pytest.mark.parametrize("rule", list(D.RULES))
def test_production_float32_gradients_at_both_declared_perturbations(rule):
    """Production dtype, BOTH declared perturbation sizes, both recorded."""
    b = TK.generate_batch(4247, 1)
    ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event")}
    const = MD.constants_for(rule)
    p = MD.init_params(rule, 100, dtype=onp.float32)
    assert p["key_raw"].dtype == onp.float32, "production leaves must be f32"
    rs = onp.random.RandomState(29)
    dirn = jnp.asarray(rs.randn(*p["key_raw"].shape).astype(onp.float32))
    dirn = dirn / jnp.linalg.norm(dirn)
    f = lambda e: _raw_embedding_probe(rule, p, ep, const, dirn, e)  # noqa: E731
    ana = float(jax.grad(f)(jnp.float32(0.0)))
    rels = {}
    for h in GRAD_PERTURBATIONS:
        num = float((f(jnp.float32(h)) - f(jnp.float32(-h))) / (2 * h))
        rels[h] = abs(num - ana) / max(abs(ana), 1e-30)
    print(f"  {rule}: f32 |dL|={abs(ana):.3e}  rels={rels}")
    assert min(rels.values()) < GRAD32 or abs(ana) < 1e-4, (rule, rels, ana)


@pytest.mark.parametrize("rule", list(D.RULES))
def test_chunked_streaming_reproduces_an_unsplit_sequence(rule):
    """Carries persist across a chunk boundary; only an episode resets them."""
    b = TK.generate_batch(4248, 1)
    ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event")}
    const = MD.constants_for(rule)
    p = MD.init_params(rule, 100)
    whole = MD.rollout(rule, p, ep, const)["logits"]
    cut = 37
    first = {k: v[:cut] for k, v in ep.items()}
    second = {k: v[cut:] for k, v in ep.items()}
    o1 = MD.rollout(rule, p, first, const)
    o2 = MD.rollout(rule, p, second, const, carry0=o1["final_carry"])
    joined = jnp.concatenate([o1["logits"], o2["logits"]], axis=0)
    err = float(jnp.max(jnp.abs(joined - whole)))
    scale = max(1.0, float(jnp.max(jnp.abs(whole))))
    print(f"  {rule}: chunked vs unsplit {err / scale:.3e}")
    assert err / scale < STREAM32, (rule, err / scale)


@pytest.mark.parametrize("rule", list(D.RULES))
def test_a_real_optimizer_update_reaches_every_trainable_leaf(rule):
    """Full BPTT to the common tables and the literature gates; the fixed
    physical coefficients appear in NO optimizer leaf."""
    import optax
    from experiments.nested_memory import study as S
    b = TK.generate_batch(4249, 2)
    eps = S.to_jax(b)
    const = MD.constants_for(rule)
    p = MD.init_params(rule, 100)
    tx = S.get_tx()
    opt = tx.init(p)
    q, opt2, loss, acc, gn, wn = S.train_step(rule, tx, p, opt, eps, const)
    assert onp.isfinite(float(loss)) and onp.isfinite(float(gn))
    moved = {k: float(jnp.max(jnp.abs(q[k] - p[k]))) for k in p}
    print(f"  {rule}: moved {[(k, '%.2e' % v) for k, v in moved.items()]}")
    for k in ("key_raw", "value_table", "readout_W"):
        assert moved[k] > 0.0, (rule, k)
    if rule == "gated_delta":
        for k in ("a_proj", "b_proj", "A_log", "dt_bias"):
            assert moved[k] > 0.0, (rule, k)
    if rule == "momentum_delta":
        for k in ("m_proj", "e_proj", "Mu_log", "mu_bias", "log_factor"):
            assert moved[k] > 0.0, (rule, k)
    forbidden = {"F", "a0", "b0", "M", "gamma", "T", "beta", "h"}
    assert not (set(p) & forbidden), "physical coefficients are not parameters"
    # the step must NOT retrace when fed its own output
    n0 = S.train_step._cache_size()
    for _ in range(3):
        q, opt2, loss, acc, gn, wn = S.train_step(rule, tx, q, opt2, eps, const)
    assert S.train_step._cache_size() == n0, rule


def test_a_checkpoint_round_trips():
    from flax import serialization
    from experiments.nested_memory import study as S
    p = MD.init_params("momentum_delta", 100)
    blob = serialization.to_bytes(jax.tree_util.tree_map(onp.asarray, p))
    back = serialization.from_bytes(jax.tree_util.tree_map(onp.asarray, p), blob)
    for k in p:
        assert onp.array_equal(onp.asarray(p[k]), onp.asarray(back[k])), k


def test_every_parameter_leaf_is_explicitly_typed():
    """A weakly typed leaf retraces a step that consumes its own output."""
    for rule in D.RULES:
        p = MD.init_params(rule, 100)
        weak = [k for k, v in p.items() if jnp.asarray(v).weak_type]
        assert not weak, (rule, weak)
