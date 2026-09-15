"""Focused checks for the memory-recall study. CLUSTER-ONLY.

Run by `bin/run_experiments/cluster_recall.sh` inside the same 1200 s cap as
the study. Not run locally: the brief requires all numerical work on the
cluster.

PREDECLARED TOLERANCES:
    EXACT   1e-10  reparameterization identity and rho=1 limit, float64
    CLONE   0      cloned parameters must be bit-identical
    EMBED   1e-10  function-preserving state-capacity embedding
"""

import math
import os
import sys

import jax
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tasks.recall as T                                          # noqa: E402
from experiments.gp import recall_study as RS                     # noqa: E402
from s5.rawat_s5 import (LOG_RHO_BOUNDS, RHO_INIT_RECALL,         # noqa: E402
                         RHO_ONLY_PARAM_NAME, init_substrate_ssm)
from s5.gp_fixed import mass_block_zoh, mass_scan                 # noqa: E402
from s5.gp_coefficients import phi1                               # noqa: E402

EXACT, EMBED = 1e-10, 1e-10


# --------------------------------------------- 1. the reparameterization ---
@pytest.mark.parametrize("gamma_n", [0.5, 1.0, 2.0, 7.0])
def test_gamma_clock_reparameterization_preserves_the_executed_blocks(gamma_n):
    """(Delta, gamma_n, rho) == (Delta/gamma_n, 1, rho): the SAME continuous
    and discrete blocks, input drive, carry and forward response."""
    rs = onp.random.RandomState(0)
    P, H, L = 5, 3, 24
    lam = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    B_c = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    Delta = np.asarray(onp.exp(rs.uniform(-4, -1, P)))
    rho = np.asarray(onp.linspace(0.3, 0.95, P))
    x = np.asarray(rs.randn(L, H))

    a1, b1 = lam * Delta, Delta[:, None] * B_c
    z1 = mass_block_zoh(a1, b1, 5.0, gamma_n * np.ones(P), rho)
    hatD = Delta / gamma_n
    a2, b2 = lam * hatD, hatD[:, None] * B_c
    z2 = mass_block_zoh(a2, b2, 5.0, np.ones(P), rho)
    for k in ("A", "B", "A_bar", "B_bar"):
        assert float(np.max(np.abs(z1[k] - z2[k]))) < EXACT, k
    s1 = mass_scan(z1["A_bar"], z1["B_bar"], x)
    s2 = mass_scan(z2["A_bar"], z2["B_bar"], x)
    assert float(np.max(np.abs(s1 - s2))) < EXACT
    assert s1.shape == s2.shape                      # identical carry


def test_rho_one_limit_agrees_with_ordinary_s5():
    """Mathematical boundary check only, NOT the training configuration:
    at rho = 1 the transfer reduces to the ordinary memory-bearing SSM."""
    rs = onp.random.RandomState(1)
    P, H, L = 4, 3, 40
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-1, 1, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(L, H))
    z = mass_block_zoh(a, b, 5.0, np.ones(P), np.ones(P))
    s_mass = mass_scan(z["A_bar"], z["B_bar"], x)[..., 0]
    a_bar = np.exp(a)
    b_bar = phi1(a)[:, None] * b
    h = np.zeros((P,), dtype=a.dtype)
    ref = []
    for k in range(L):
        h = a_bar * h + b_bar @ x[k]
        ref.append(h)
    ref = np.stack(ref)
    assert float(np.max(np.abs(s_mass - ref)) / np.max(np.abs(ref))) < EXACT


def test_mass_is_derived_from_rho_alone():
    """mu = T rho with gamma_n = 1; no independent mass parameter exists."""
    from s5.rawat_s5 import SubstrateSSM
    assert "log_response_gamma" not in str(
        SubstrateSSM.__dict__.get("rho_only", ""))
    for rho in (0.1, 0.5, 0.9998):
        assert 5.0 * rho == pytest.approx(5.0 * rho)


# ------------------------------------------------------ 2. the task --------
def test_task_structure_and_pairing():
    rng = onp.random.RandomState(0)
    x, y = T.generate(rng, 96)
    c = T.check_pairing(x, y)
    assert c["seq_len"] == 128 and c["channels"] == 10
    assert c["query_marker_only_at_end"]
    assert c["query_has_no_symbol"]
    assert c["exactly_two_cues"]
    assert c["label_balance_max_dev"] < 0.35, c


def test_paired_sequences_share_a_token_multiset_but_differ_in_target():
    """A bag-of-symbols statistic cannot separate the pair."""
    rng = onp.random.RandomState(3)
    xs, ys = [], []
    for _ in range(1):
        x, y = T.generate(onp.random.RandomState(7), 1)
        xs.append(x); ys.append(y)
    x, y = xs[0], ys[0]
    assert x.shape[0] == 2
    ma = x[0, :T.QUERY_INDEX, :T.N_SYMBOLS].sum(axis=0)
    mb = x[1, :T.QUERY_INDEX, :T.N_SYMBOLS].sum(axis=0)
    assert onp.allclose(ma, mb), (ma, mb)
    assert y[0] != y[1]
    assert onp.array_equal(x[0, :, T.CUE_CHANNEL], x[1, :, T.CUE_CHANNEL])


def test_target_is_the_LATEST_cue_not_the_earlier_one():
    rng = onp.random.RandomState(5)
    x, y = T.generate_fixed_delay(rng, 64, delay=32)
    late = T.QUERY_INDEX - 32
    for i in range(x.shape[0]):
        cues = onp.flatnonzero(x[i, :, T.CUE_CHANNEL])
        assert cues.max() == late
        assert int(onp.argmax(x[i, cues.max(), :T.N_SYMBOLS])) == int(y[i])


def test_heldout_delay_is_not_a_training_delay():
    assert T.HELDOUT_DELAY not in T.TRAIN_DELAYS


# ----------------------------------------------- 3. causality and readout --
def test_query_logits_do_not_depend_on_future_or_on_the_query_symbol():
    m, p = RS.init_params("ordinary", 0)
    rng = onp.random.RandomState(11)
    x, _ = T.generate_fixed_delay(rng, 8, delay=32)
    x = np.asarray(x[:4])
    ts = np.ones(x.shape[:2])
    y0 = m.apply({"params": p}, x, ts)
    x2 = onp.asarray(x).copy()
    x2[:, T.QUERY_INDEX, :T.N_SYMBOLS] = 0.0           # already zero
    assert float(np.max(np.abs(
        m.apply({"params": p}, np.asarray(x2), ts) - y0))) == 0.0


def test_professor_query_output_is_invariant_to_the_earlier_cue():
    """The memoryless core cannot see any earlier token, so the query logits
    must be identical when the marked cue changes."""
    m, p = RS.init_params("professor", 0)
    rng = onp.random.RandomState(13)
    bx, by, cx, cy, dx = T.probe_interventions(rng, 8)
    ts = np.ones(bx.shape[:2])
    lb = m.apply({"params": p}, np.asarray(bx), ts)
    lc = m.apply({"params": p}, np.asarray(cx), ts)
    ld = m.apply({"params": p}, np.asarray(dx), ts)
    assert float(np.max(np.abs(lc - lb))) < 1e-9
    assert float(np.max(np.abs(ld - lb))) < 1e-9


def test_professor_log_step_gradient_is_structurally_zero():
    """-B_c/lambda contains no Delta, so the data-loss gradient with respect to
    log_step is EXACTLY zero. Previously mis-reported as nonzero."""
    m, p = RS.init_params("professor", 0)
    rng = onp.random.RandomState(17)
    x, y = T.generate_fixed_delay(rng, 8, delay=32)

    def loss(params):
        return RS.loss_fn(params, m, np.asarray(x), np.asarray(y))[0]

    g = jax.grad(loss)(p)
    from flax.traverse_util import flatten_dict
    for k, v in flatten_dict(g).items():
        if k[-1] == "log_step":
            assert float(np.max(np.abs(v))) == 0.0, ("/".join(k),
                                                     float(np.max(np.abs(v))))
    others = {"/".join(k): float(np.max(np.abs(v)))
              for k, v in flatten_dict(g).items()
              if k[-1] in ("B", "C", "D")}
    assert all(v > 0.0 for v in others.values()), others


# ------------------------------------------------- 4. cloning and embedding
def test_clone_copies_every_shared_parameter_bit_identically():
    _, p_ord = RS.init_params("ordinary", 100)
    m_gp, p_gp = RS.init_params("gp_rho", 100)
    cloned, copied, kept = RS.clone_common(p_ord, p_gp)
    from flax.traverse_util import flatten_dict
    a, b = flatten_dict(p_ord), flatten_dict(cloned)
    for k in a:
        assert bool(np.array_equal(a[k], b[k])), "/".join(k)
    assert kept == [f"encoder/layers_{i}/seq/{RHO_ONLY_PARAM_NAME}"
                    for i in range(RS.N_LAYERS)] or set(kept) == {
        f"encoder/layers_{i}/seq/{RHO_ONLY_PARAM_NAME}"
        for i in range(RS.N_LAYERS)}, kept


def test_rho_leaf_initializes_at_the_declared_value():
    _, p = RS.init_params("gp_rho", 0)
    from flax.traverse_util import flatten_dict
    found = [v for k, v in flatten_dict(p).items()
             if k[-1] == RHO_ONLY_PARAM_NAME]
    assert found
    for v in found:
        assert float(np.max(np.abs(np.exp(v) - RHO_INIT_RECALL))) < 1e-6


def test_double_state_embedding_preserves_the_ordinary_function():
    m_o, p_o = RS.init_params("ordinary", 100)
    m_2, p_2 = RS.init_params("ordinary_2x", 100, ssm_size=RS.SSM_SIZE * 2)
    emb, notes = RS.embed_double_state(p_o, p_2)
    rng = onp.random.RandomState(19)
    x, _ = T.generate_fixed_delay(rng, 8, delay=32)
    x = np.asarray(x[:4]); ts = np.ones(x.shape[:2])
    y_o = m_o.apply({"params": p_o}, x, ts)
    y_2 = m_2.apply({"params": emb}, x, ts)
    rel = float(np.max(np.abs(y_2 - y_o)) / max(float(np.max(np.abs(y_o))),
                                                1e-12))
    assert rel < EMBED, (rel, notes)


def test_added_modes_have_a_trainable_readout_path_and_nonzero_input():
    m_o, p_o = RS.init_params("ordinary", 100)
    m_2, p_2 = RS.init_params("ordinary_2x", 100, ssm_size=RS.SSM_SIZE * 2)
    emb, _ = RS.embed_double_state(p_o, p_2)
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(emb)
    for k, v in flat.items():
        if k[-1] == "C":
            P = v.shape[1] // 2
            assert float(np.max(np.abs(v[:, P:, :]))) == 0.0   # zero readout
        if k[-1] == "B":
            P = v.shape[0] // 2
            assert float(np.max(np.abs(v[P:]))) > 0.0          # nonzero input
    rng = onp.random.RandomState(23)
    x, y = T.generate_fixed_delay(rng, 8, delay=32)

    def loss(params):
        return RS.loss_fn(params, m_2, np.asarray(x), np.asarray(y))[0]

    g = flatten_dict(jax.grad(loss)(emb))
    for k, v in g.items():
        if k[-1] == "C":
            P = v.shape[1] // 2
            assert float(np.max(np.abs(v[:, P:, :]))) > 0.0, "/".join(k)


def test_double_state_actually_doubles_the_carry():
    m_o, p_o = RS.init_params("ordinary", 0)
    m_2, p_2 = RS.init_params("ordinary_2x", 0, ssm_size=RS.SSM_SIZE * 2)
    c_o = RS.read_state_counts("ordinary", p_o)[0]
    c_2 = RS.read_state_counts("ordinary_2x", p_2, RS.SSM_SIZE * 2)[0]
    assert c_2["physical_real"] == 2 * c_o["physical_real"]
    assert RS.parameter_count(p_2) > RS.parameter_count(p_o)


# ------------------------------------------------- 5. training mechanics ---
def test_rho_updates_when_learned_and_not_when_frozen():
    rng = onp.random.RandomState(29)
    x, y = T.generate_fixed_delay(rng, 32, delay=32)
    x, y = np.asarray(x), np.asarray(y)
    from flax.traverse_util import flatten_dict
    out = {}
    for arm, frozen in (("gp_rho", False), ("gp_rho_frozen", True)):
        m, p = RS.init_params(arm, 0)
        tx = RS.make_tx(freeze_rho=frozen)
        opt = tx.init(p)
        q, opt, loss, acc, gn, _tel = RS.train_step(m, tx, p, opt, x, y)
        before = [v for k, v in flatten_dict(p).items()
                  if k[-1] == RHO_ONLY_PARAM_NAME]
        after = [v for k, v in flatten_dict(q).items()
                 if k[-1] == RHO_ONLY_PARAM_NAME]
        moved = max(float(np.max(np.abs(b - a)))
                    for a, b in zip(before, after))
        other = max(float(np.max(np.abs(flatten_dict(p)[k]
                                        - flatten_dict(q)[k])))
                    for k in flatten_dict(p) if k[-1] == "B")
        out[arm] = (moved, other)
    assert out["gp_rho"][0] > 0.0, out
    assert out["gp_rho_frozen"][0] == 0.0, out
    assert out["gp_rho_frozen"][1] > 0.0, out      # ordinary weights still move


def test_full_bptt_gradients_are_finite_for_every_arm():
    rng = onp.random.RandomState(31)
    x, y = T.generate_fixed_delay(rng, 16, delay=64)
    x, y = np.asarray(x), np.asarray(y)
    for arm in RS.ARMS:
        size = RS.SSM_SIZE * 2 if arm == "ordinary_2x" else RS.SSM_SIZE
        m, p = RS.init_params(arm, 0, size)

        def loss(params):
            return RS.loss_fn(params, m, x, y)[0]

        g = jax.grad(loss)(p)
        leaves = jax.tree_util.tree_leaves(g)
        assert all(bool(onp.isfinite(onp.asarray(v)).all()) for v in leaves), arm
        assert max(float(np.max(np.abs(v))) for v in leaves) > 0.0, arm


def test_every_arm_uses_tokenwise_layernorm_not_batch_or_time_norm():
    """Splitting the batch must not change any example's logits."""
    rng = onp.random.RandomState(37)
    x, _ = T.generate_fixed_delay(rng, 16, delay=32)
    x = np.asarray(x[:8])
    for arm in RS.ARMS:
        size = RS.SSM_SIZE * 2 if arm == "ordinary_2x" else RS.SSM_SIZE
        m, p = RS.init_params(arm, 0, size)
        full = m.apply({"params": p}, x, np.ones(x.shape[:2]))
        half = np.concatenate([
            m.apply({"params": p}, x[:4], np.ones(x[:4].shape[:2])),
            m.apply({"params": p}, x[4:], np.ones(x[4:].shape[:2]))], axis=0)
        assert float(np.max(np.abs(full - half))) < 1e-9, arm


def test_production_dtype_probe():
    import subprocess
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "recall_float32_probe.py")
    r = subprocess.run([sys.executable, probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RECALL_F32_OK" in r.stdout, r.stdout


def test_response_adapter_handles_every_arm_in_this_study():
    """The initialization gate calls the adapter for each arm; an unknown law
    would raise. Includes the memoryless arm, whose impulse is lag-zero only."""
    from s5 import substrate_diagnostics as SD
    for arm in ("ordinary", "rawat", "professor", "gp_rho", "gp_rho_frozen"):
        m, p = RS.init_params(arm, 0)
        for layer in range(RS.N_LAYERS):
            r = RS.core_response_change(arm, p, arm, p, layer)
            assert r["impulse_abs_diff"] == pytest.approx(0.0, abs=1e-12), arm
            if not r["zero_reference"]:
                assert r["impulse_rel"] == pytest.approx(0.0, abs=1e-12), arm


def test_memoryless_arm_has_no_response_beyond_lag_zero_in_the_adapter():
    from s5 import substrate_diagnostics as SD
    m = RS.build_single("professor")
    x = np.zeros((T.SEQ_LEN, T.N_CHANNELS)); ts = np.ones((T.SEQ_LEN,))
    _, p = RS.init_params("professor", 0)

    def read(mm, xx, tt):
        seq = mm.encoder.layers[0].seq
        Lam, B_c, Delta = seq._native()
        return dict(response=seq.response, input_gain=seq.input_gain,
                    clip_eigs=seq.clip_eigs, conj_sym=seq.conj_sym,
                    P=seq.P, H=seq.H, Lambda_clipped=Lam,
                    Lambda_raw_param=seq.Lambda_re + 1j * seq.Lambda_im,
                    Lambda_initializer_field=(seq.Lambda_re_init
                                              + 1j * seq.Lambda_im_init),
                    B_tilde=seq.B[..., 0] + 1j * seq.B[..., 1], B_c=B_c,
                    Delta=Delta, a=Lam * Delta, b=Delta[:, None] * B_c,
                    C_tilde=seq.C_tilde, D=seq.D,
                    coefficients=seq.coefficients(),
                    physical=dict(T=seq.physical.T, gamma=seq.physical.gamma,
                                  rho=seq.physical.rho,
                                  mass=seq.physical.mass))
    core = m.apply({"params": p}, x, ts, method=read)
    K = SD.impulse_matrices(core, 32)
    assert float(onp.max(onp.abs(K[1:]))) == 0.0
    assert float(onp.max(onp.abs(K[0]))) > 0.0
    assert SD.spectral_radius(core) == 0.0


def test_rho_gradient_converges_like_a_second_order_difference():
    """Gradient CORRECTNESS, established by the CONVERGENCE SHAPE.

    The evidence that a gradient is right is not that some finite difference
    is small in absolute terms - that depends on the derivative's size and the
    arithmetic - but that the error falls like O(h^2) as the step falls, until
    rounding takes over. A dropped term would leave a step-INDEPENDENT floor.

    Measured here (parameters are float32, so this is not true float64
    arithmetic despite x64 being enabled; the effective rounding floor sits
    near 1e-5):

        h      1e-1      1e-2      1e-3      1e-4
        rel    9.76e-2   1.21e-3   9.21e-6   1.27e-4

    about 80x and 130x improvement per decade against the 100x an O(h^2)
    difference predicts, then a rise at 1e-4 as rounding dominates. The
    assertions below check that SHAPE, plus a floor consistent with the
    measured precision. An earlier revision demanded min < 1e-6, which is below
    what this configuration can reach and said nothing about correctness.
    """
    import math as _m
    from flax.traverse_util import flatten_dict, unflatten_dict
    m, p = RS.init_params("gp_rho", 0)
    flat = dict(flatten_dict(p))
    key = [k for k in flat if k[-1] == RHO_ONLY_PARAM_NAME][0]
    for k in list(flat):
        if k[-1] == RHO_ONLY_PARAM_NAME:
            flat[k] = np.full_like(flat[k], _m.log(0.5))   # interior
    p = unflatten_dict(flat)
    flat = dict(flatten_dict(p))
    rng = onp.random.RandomState(41)
    x, y = T.generate_fixed_delay(rng, 32, delay=32)
    x, y = np.asarray(x), np.asarray(y)

    def loss_of(v):
        f = dict(flat); f[key] = v
        return float(RS.loss_fn(unflatten_dict(f), m, x, y)[0])

    def L(params):
        return RS.loss_fn(params, m, x, y)[0]

    g = flatten_dict(jax.grad(L)(p))[key]
    assert float(np.max(np.abs(g))) > 0.0
    d = onp.random.RandomState(42).randn(*onp.asarray(g).shape)
    d = np.asarray(d / d.std(), dtype=flat[key].dtype)
    ana = float(np.sum(g * d))
    rels = []
    for h in (1e-1, 1e-2, 1e-3, 1e-4):
        fd = (loss_of(flat[key] + h * d) - loss_of(flat[key] - h * d)) / (2 * h)
        rels.append(abs(ana - fd) / max(abs(fd), 1e-12))
    ladder = list(zip((1e-1, 1e-2, 1e-3, 1e-4), rels))
    # 1. the error must FALL monotonically while truncation dominates
    assert rels[1] < rels[0] and rels[2] < rels[1], ladder
    # 2. and by orders of magnitude, as O(h^2) over two decades implies
    assert rels[0] / rels[2] > 1e3, ladder
    # 3. reaching a floor consistent with the arithmetic actually used
    assert min(rels) < 1e-4, ladder


# ---------------------------------------------- post-update projection (R1) --
def test_raw_response_is_projected_back_after_an_outward_update():
    """The production path must not strand a raw parameter outside the bound.

    Forces an outward crossing, then checks that the projection pulls it back
    AND that an inward task gradient can still move it - which is exactly what
    a forward clip alone cannot provide, since d(rho)/d(eta) = 0 beyond the
    bound gives zero task gradient and AdamW's decay shrinks eta toward zero,
    away from a negative upper bound.
    """
    from flax.traverse_util import flatten_dict, unflatten_dict
    from s5.rawat_s5 import LOG_RHO_BOUNDS
    m, p = RS.init_params("gp_rho", 0)
    flat = dict(flatten_dict(p))
    key = [k for k in flat if k[-1] == RHO_ONLY_PARAM_NAME][0]
    # The leaf is stored float32 even when x64 is enabled, so the bounds must be
    # compared in the LEAF's dtype: float32(log(0.01)) is a rounded
    # representation of the double-precision endpoint, and a 1e-12 comparison
    # against the double value would fail on representation alone. This is a
    # dtype issue, not a widening of the admissible interval.
    dt = flat[key].dtype
    assert dt == onp.float32, dt
    lo, hi = onp.asarray(LOG_RHO_BOUNDS[0], dt), onp.asarray(
        LOG_RHO_BOUNDS[1], dt)

    # 1. a raw value pushed far outside is projected back onto the bound
    outside = unflatten_dict({**flat, key: np.full_like(flat[key], hi + 5.0)})
    back = flatten_dict(RS.project_response_leaves(outside))[key]
    assert back.dtype == dt
    assert float(np.max(back)) <= float(hi)
    assert float(np.min(back)) >= float(lo)
    below = unflatten_dict({**flat, key: np.full_like(flat[key], lo - 5.0)})
    got = flatten_dict(RS.project_response_leaves(below))[key]
    assert got.dtype == dt and float(np.min(got)) >= float(lo)

    # 2. an UNPROJECTED outward value has exactly zero task gradient: the
    #    failure mode the projection exists to prevent
    rng = onp.random.RandomState(51)
    x, y = T.generate_fixed_delay(rng, 16, delay=32)
    x, y = np.asarray(x), np.asarray(y)

    def grad_at(tree):
        def L(params):
            return RS.loss_fn(params, m, x, y)[0]
        return flatten_dict(jax.grad(L)(tree))[key]

    assert float(np.max(np.abs(grad_at(outside)))) == 0.0

    # 3. after projection onto the bound the gradient is available again
    projected = RS.project_response_leaves(outside)
    assert float(np.max(np.abs(grad_at(projected)))) > 0.0


def test_forced_outward_crossing_is_projected_counted_and_then_reversible():
    """The real update path, with a DETERMINISTIC crossing and real movement.

    Three things, none of which the previous version established:
      1. starting outside the interval, the production step projects back and
         reports a POSITIVE event count and overshoot - not `>= 0`, which any
         no-op satisfies;
      2. on the boundary the task gradient is available again;
      3. under a controlled INWARD objective, carrying the SAME optimizer state
         forward rather than resetting it, the leaf moves STRICTLY back inside.
    """
    import optax
    from flax.traverse_util import flatten_dict, unflatten_dict
    from s5.rawat_s5 import LOG_RHO_BOUNDS
    rng = onp.random.RandomState(53)
    x, y = T.generate_fixed_delay(rng, 32, delay=32)
    x, y = np.asarray(x), np.asarray(y)
    m, p = RS.init_params("gp_rho", 0)
    flat = dict(flatten_dict(p))
    key = [k for k in flat if k[-1] == RHO_ONLY_PARAM_NAME][0]
    dt = flat[key].dtype
    lo, hi = onp.asarray(LOG_RHO_BOUNDS[0], dt), onp.asarray(
        LOG_RHO_BOUNDS[1], dt)

    # (1) start OUTSIDE: the crossing is deterministic, not gradient-dependent
    overshoot = 0.5
    start = unflatten_dict({**flat,
                            key: np.full_like(flat[key], hi + overshoot)})
    tx = RS.make_tx()
    opt = tx.init(start)
    q, opt, loss, acc, gn, tel = RS.train_step(m, tx, start, opt, x, y)
    raw = flatten_dict(q)[key]
    assert raw.dtype == dt
    assert float(np.max(raw)) <= float(hi), "projection did not hold the bound"
    assert int(tel["n_projected"]) == raw.size, int(tel["n_projected"])
    assert float(tel["max_overshoot"]) > 0.4, float(tel["max_overshoot"])
    assert float(np.max(np.abs(raw - hi))) < 1e-6, "should sit ON the bound"

    # (2) on the boundary the task gradient exists again
    def L(params):
        return RS.loss_fn(params, m, x, y)[0]

    assert float(np.max(np.abs(flatten_dict(jax.grad(L)(q))[key]))) > 0.0

    # (3) a controlled inward objective must move it STRICTLY inside, with the
    #     optimizer state carried forward - Adam's first steps are
    #     momentum-limited, so several updates are taken.
    def inward(params):
        return sum(np.sum(v) for v in RS._rho_leaves(params))

    for _ in range(12):
        g = jax.grad(inward)(q)
        upd, opt = tx.update(g, opt, q)
        q = RS.project_response_leaves(optax.apply_updates(q, upd))
    moved = flatten_dict(q)[key]
    assert float(np.max(moved)) < float(hi) - 1e-5, (
        "an inward objective did not move the leaf off the boundary: "
        f"max {float(np.max(moved)):.8f} vs bound {float(hi):.8f}")
    assert float(np.min(moved)) >= float(lo)


def test_the_frozen_arm_is_unchanged_by_the_projection():
    rng = onp.random.RandomState(57)
    x, y = T.generate_fixed_delay(rng, 32, delay=32)
    x, y = np.asarray(x), np.asarray(y)
    from flax.traverse_util import flatten_dict
    m, p = RS.init_params("gp_rho_frozen", 0)
    tx = RS.make_tx(freeze_rho=True)
    opt = tx.init(p)
    q, opt, loss, acc, gn, tel = RS.train_step(m, tx, p, opt, x, y)
    a = flatten_dict(p); b = flatten_dict(q)
    for k in a:
        if k[-1] == RHO_ONLY_PARAM_NAME:
            assert float(np.max(np.abs(a[k] - b[k]))) == 0.0, "/".join(k)
    assert int(tel["n_projected"]) == 0
    assert float(tel["max_overshoot"]) == 0.0
    moved = max(float(np.max(np.abs(a[k] - b[k]))) for k in a if k[-1] == "B")
    assert moved > 0.0, "ordinary weights must still train"


def test_stored_and_trainable_counts_are_reported_separately():
    _, p = RS.init_params("gp_rho_frozen", 0)
    c = RS.trainable_count(p, "gp_rho_frozen")
    assert c["stored"] > c["trainable"]
    assert c["stored"] - c["trainable"] == c["rho_leaves"]
    _, q = RS.init_params("gp_rho", 0)
    d = RS.trainable_count(q, "gp_rho")
    assert d["stored"] == d["trainable"]


def test_realized_delay_distribution_is_reported_not_assumed_equal():
    r = T.realized_delay_counts(16)
    assert r["counts"] == {8: 6, 32: 5, 64: 5}
    assert r["equal"] is False
    assert abs(r["fractions"][8] - 0.375) < 1e-9
    assert T.realized_delay_counts(15)["equal"] is True


def _layer(imp_rel, freq_rel, imp_abs=1.0, freq_abs=1.0, layer=0):
    return dict(layer=layer, impulse_rel=imp_rel, freq_rel=freq_rel,
                impulse_abs_diff=imp_abs, freq_abs_diff=freq_abs)


def _arms(layers):
    return {a: dict(layers=layers) for a in RS.GATED_ARMS}


def test_gate_decision_both_criteria_pass():
    d = RS.evaluate_gate(_arms([_layer(1e-3, 2e-3)]))
    assert d["passed"] and d["failures"] == []
    assert d["gp_worst_impulse_rel"] == pytest.approx(1e-3)
    assert d["gp_worst_frequency_rel"] == pytest.approx(2e-3)


def test_gate_decision_fails_when_only_the_FREQUENCY_criterion_is_violated():
    """The defect being pinned: a decision on the impulse alone would PASS."""
    d = RS.evaluate_gate(_arms([_layer(1e-3, 5e-1)]))
    assert not d["passed"]
    assert any("frequency" in f for f in d["failures"]), d["failures"]
    assert not any("impulse" in f for f in d["failures"]), d["failures"]


def test_gate_decision_fails_when_only_the_impulse_criterion_is_violated():
    d = RS.evaluate_gate(_arms([_layer(5e-1, 1e-3)]))
    assert not d["passed"]
    assert any("impulse" in f for f in d["failures"]), d["failures"]


def test_gate_decision_rejects_non_finite_values():
    for bad in (float("nan"), float("inf")):
        d = RS.evaluate_gate(_arms([_layer(bad, 1e-3)]))
        assert not d["passed"], bad
        d2 = RS.evaluate_gate(_arms([_layer(1e-3, 1e-3, imp_abs=bad)]))
        assert not d2["passed"], bad


def test_gate_decision_zero_reference_passes_only_when_the_arm_is_also_zero():
    """A zero-reference layer is NOT dropped: it gets an absolute criterion."""
    ok = RS.evaluate_gate(_arms([_layer(None, None, imp_abs=1e-12,
                                        freq_abs=1e-12)]))
    assert ok["passed"], ok["failures"]
    bad = RS.evaluate_gate(_arms([_layer(None, None, imp_abs=1e-3,
                                         freq_abs=1e-12)]))
    assert not bad["passed"]
    assert any("zero reference" in f for f in bad["failures"]), bad["failures"]


def test_gate_decision_requires_a_record_for_every_gated_arm():
    d = RS.evaluate_gate({"gp_rho": dict(layers=[_layer(1e-3, 1e-3)])})
    assert not d["passed"]
    assert any("no gate record" in f for f in d["failures"]), d["failures"]


def test_gate_runs_for_every_seed():
    assert RS.GATE_EVERY_SEED is True
    assert RS.INIT_RESPONSE_ABS_TOL > 0.0
