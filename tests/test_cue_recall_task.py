"""CPU gates for the controlled cue/recall task and its models (brief section 5).

Covers: generator/oracle, target masks, no-future-input, no-history shortcut,
task-runner smoke, fixed-batch overfit, parameter update/freeze, and the
conventional-chart conversion with directional derivatives.
"""
import os
import sys

import jax
import numpy as onp
import optax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gp.cue_recall_runner import (build_model, evaluate,   # noqa: E402
                                              loss_fn, make_state,
                                              ssm_kwargs)
from flax.traverse_util import flatten_dict                            # noqa: E402
from s5.gp_coefficients import gp_response_coefficients, phi1          # noqa: E402
from s5.modal_ssm import init_modal_ssm, matched_init                  # noqa: E402
from tasks import cue_recall as T                                      # noqa: E402

CFG = T.TaskConfig()


# ------------------------------------------------------- generator / oracle

def test_config_is_self_consistent():
    assert CFG.validate()
    assert CFG.max_query() == 31 + 128 == 159 < CFG.seq_len


def test_generator_matches_its_oracle():
    b = T.generate(T.split_key(0, T.Split.TRAIN, 0), 64, CFG)
    onp.testing.assert_array_equal(onp.asarray(T.oracle_recall(b)),
                                   onp.asarray(b["recall_target"]))
    onp.testing.assert_array_equal(onp.asarray(T.oracle_current(b)),
                                   onp.asarray(b["current_target"]))


def test_generator_structure_and_ranges():
    b = T.generate(T.split_key(1, T.Split.TRAIN, 0), 256, CFG)
    x = onp.asarray(b["x"])
    assert x.shape == (256, CFG.seq_len, T.N_CHANNELS)
    # payload and current are strictly +-1
    assert set(onp.unique(x[..., :T.N_PAYLOAD + T.N_CURRENT])) == {-1.0, 1.0}
    # exactly one cue flag and one query flag per example
    onp.testing.assert_array_equal(x[..., T.CUE_FLAG].sum(1), onp.ones(256))
    onp.testing.assert_array_equal(x[..., T.QUERY_FLAG].sum(1), onp.ones(256))
    cue, delay, q = (onp.asarray(b[k]) for k in ("cue_idx", "delay", "query_idx"))
    assert cue.min() >= CFG.cue_lo and cue.max() <= CFG.cue_hi
    onp.testing.assert_array_equal(q, cue + delay)
    assert q.max() < CFG.seq_len
    # every delay falls inside its declared bucket
    for k, (lo, hi) in enumerate(CFG.buckets):
        m = onp.asarray(b["bucket"]) == k
        if m.any():
            assert delay[m].min() >= lo and delay[m].max() <= hi


def test_every_token_has_a_fresh_payload_so_later_ones_are_distractors():
    b = T.generate(T.split_key(2, T.Split.TRAIN, 0), 512, CFG)
    x = onp.asarray(b["x"]); q = onp.asarray(b["query_idx"])
    at_query = x[onp.arange(len(q)), q, :T.N_PAYLOAD]
    tgt = onp.asarray(b["recall_target"]) * 2 - 1
    # the query token's own payload carries no information about the answer
    agree = (at_query == tgt).mean()
    assert 0.45 < agree < 0.55, agree


def test_no_history_shortcut_from_the_query_token():
    """A same-token model must be at chance: the target is independent of the
    query token's inputs by construction."""
    b = T.generate(T.split_key(3, T.Split.TRAIN, 0), 4096, CFG)
    x = onp.asarray(b["x"]); q = onp.asarray(b["query_idx"])
    # only the non-constant channels: the cue flag is identically 0 at the
    # query token, so its correlation is 0/0 rather than informative
    feats = x[onp.arange(len(q)), q, :T.N_PAYLOAD + T.N_CURRENT]
    tgt = onp.asarray(b["recall_target"]) * 2 - 1
    c = onp.corrcoef(onp.concatenate([feats, tgt], axis=1).T)
    n_f = feats.shape[1]
    cross = c[:n_f, n_f:]
    assert onp.nanmax(onp.abs(cross)) < 0.08, onp.nanmax(onp.abs(cross))


def test_rng_domains_are_disjoint():
    a = T.generate(T.split_key(0, T.Split.TRAIN, 0), 8, CFG)
    for other in (T.Split.DEV, T.Split.TEST, T.Split.EXTRAP):
        b = T.generate(T.split_key(0, other, 0), 8, CFG)
        assert not onp.allclose(onp.asarray(a["x"]), onp.asarray(b["x"]))


def test_extrapolation_split_uses_the_declared_long_delays():
    b = T.generate(T.split_key(0, T.Split.EXTRAP, 0), 128, CFG,
                   seq_len=CFG.extrap_seq_len, delays=CFG.extrap_delays)
    d = set(onp.unique(onp.asarray(b["delay"]).tolist()))
    assert d.issubset(set(CFG.extrap_delays))
    assert onp.asarray(b["query_idx"]).max() < CFG.extrap_seq_len


# ------------------------------------------------------------ loss / masks

def test_recall_is_scored_only_at_the_query_token():
    b = T.generate(T.split_key(4, T.Split.TRAIN, 0), 16, CFG)
    r = jax.random.normal(jax.random.PRNGKey(0), (16, CFG.seq_len, 8))
    c = jax.random.normal(jax.random.PRNGKey(1), (16, CFG.seq_len))
    base = T.losses(r, c, b)
    q = onp.asarray(b["query_idx"])
    # perturbing a NON-query token must not change the recall loss
    other = (q + 7) % CFG.seq_len
    r2 = onp.asarray(r).copy(); r2[onp.arange(16), other] += 5.0
    assert float(T.losses(np.asarray(r2), c, b)["recall_bce"]) == \
        pytest.approx(float(base["recall_bce"]), rel=1e-12)
    # perturbing the query token MUST change it
    r3 = onp.asarray(r).copy(); r3[onp.arange(16), q] += 5.0
    assert abs(float(T.losses(np.asarray(r3), c, b)["recall_bce"])
               - float(base["recall_bce"])) > 1e-3


def test_two_heads_are_averaged_separately():
    """256 current targets must not drown out the single recall event."""
    b = T.generate(T.split_key(5, T.Split.TRAIN, 0), 8, CFG)
    r = np.zeros((8, CFG.seq_len, 8)); c = np.zeros((8, CFG.seq_len))
    out = T.losses(r, c, b)
    ln2 = float(onp.log(2.0))
    assert float(out["recall_bce"]) == pytest.approx(ln2, rel=1e-6)
    assert float(out["current_bce"]) == pytest.approx(ln2, rel=1e-6)
    assert float(out["loss"]) == pytest.approx(2 * ln2, rel=1e-6)


# --------------------------------------------------------- model behaviour

@pytest.mark.parametrize("mechanism", ["plain", "gp_diagonal", "modal_ssm"])
def test_model_is_causal_no_future_input(mechanism):
    model = build_model(mechanism, 0.05, training=False)
    b = T.generate(T.split_key(6, T.Split.TRAIN, 0), 4, CFG)
    p = model.init({"params": jax.random.PRNGKey(0),
                    "dropout": jax.random.PRNGKey(0)},
                   b["x"], np.ones(b["x"].shape[:2]))["params"]
    ts = np.ones(b["x"].shape[:2])
    r1, c1 = model.apply({"params": p}, b["x"], ts)
    cut = 100
    x2 = b["x"].at[:, cut:, :].set(0.0)
    r2, c2 = model.apply({"params": p}, x2, ts)
    onp.testing.assert_allclose(onp.asarray(r1[:, :cut]),
                                onp.asarray(r2[:, :cut]), rtol=1e-8, atol=1e-9)
    onp.testing.assert_allclose(onp.asarray(c1[:, :cut]),
                                onp.asarray(c2[:, :cut]), rtol=1e-8, atol=1e-9)


def test_full_state_pc_is_memoryless_so_recall_must_be_at_chance():
    """Negative control, checked structurally rather than by training.

    full_state_pc has zero driven history, so the whole stack is tokenwise:
    changing the CUE token cannot change the QUERY token's output. A model
    with that property cannot do better than chance on payload recall.
    """
    model = build_model("full_state_pc", 0.3, training=False)
    b = T.generate(T.split_key(7, T.Split.TRAIN, 0), 4, CFG)
    p = model.init({"params": jax.random.PRNGKey(0),
                    "dropout": jax.random.PRNGKey(0)},
                   b["x"], np.ones(b["x"].shape[:2]))["params"]
    ts = np.ones(b["x"].shape[:2])
    q = onp.asarray(b["query_idx"]); cue = onp.asarray(b["cue_idx"])
    r1, _ = model.apply({"params": p}, b["x"], ts)
    x2 = onp.asarray(b["x"]).copy()
    x2[onp.arange(4), cue, :T.N_PAYLOAD] *= -1.0          # flip the cue payload
    r2, _ = model.apply({"params": p}, np.asarray(x2), ts)
    at_q1 = onp.asarray(r1)[onp.arange(4), q]
    at_q2 = onp.asarray(r2)[onp.arange(4), q]
    onp.testing.assert_allclose(at_q1, at_q2, rtol=0, atol=1e-10)


@pytest.mark.parametrize("mechanism", ["plain", "gp_diagonal", "modal_ssm"])
def test_history_reaches_the_query_token(mechanism):
    """The complement of the control above: these models DO carry history."""
    model = build_model(mechanism, 0.05, training=False)
    b = T.generate(T.split_key(8, T.Split.TRAIN, 0), 4, CFG)
    p = model.init({"params": jax.random.PRNGKey(0),
                    "dropout": jax.random.PRNGKey(0)},
                   b["x"], np.ones(b["x"].shape[:2]))["params"]
    ts = np.ones(b["x"].shape[:2])
    q = onp.asarray(b["query_idx"]); cue = onp.asarray(b["cue_idx"])
    r1, _ = model.apply({"params": p}, b["x"], ts)
    x2 = onp.asarray(b["x"]).copy()
    x2[onp.arange(4), cue, :T.N_PAYLOAD] *= -1.0
    r2, _ = model.apply({"params": p}, np.asarray(x2), ts)
    d = onp.max(onp.abs(onp.asarray(r1)[onp.arange(4), q]
                        - onp.asarray(r2)[onp.arange(4), q]))
    assert d > 1e-9, (mechanism, d)


# ------------------------------------------------- runner smoke / training

@pytest.mark.parametrize("mechanism", ["plain", "gp_diagonal", "modal_ssm"])
def test_fixed_batch_overfit_and_all_parameters_update(mechanism):
    model = build_model(mechanism, 0.05, training=True)
    b = T.generate(T.split_key(9, T.Split.TRAIN, 0), 8, CFG)
    params, tx, opt = make_state(model, jax.random.PRNGKey(0), b, 3e-3, 3e-3)
    start = {k: onp.asarray(v).copy() for k, v in flatten_dict(params).items()}
    losses = []
    for _ in range(60):
        (l, _), g = jax.value_and_grad(loss_fn, has_aux=True)(params, model, b)
        upd, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, upd)
        losses.append(float(l))
    end = flatten_dict(params)
    never = [k for k in start
             if float(onp.max(onp.abs(onp.asarray(end[k]) - start[k]))) == 0.0]
    assert not never, (mechanism, never)
    assert losses[-1] < 0.9 * losses[0], (mechanism, losses[0], losses[-1])


def test_frozen_response_control_trains_everything_except_t():
    model = build_model("gp_diagonal", 0.05, training=True)
    b = T.generate(T.split_key(10, T.Split.TRAIN, 0), 8, CFG)
    params, tx, opt = make_state(model, jax.random.PRNGKey(0), b, 3e-3, 3e-3,
                                 freeze_response=True)
    start = {k: onp.asarray(v).copy() for k, v in flatten_dict(params).items()}
    for _ in range(10):
        (_, _), g = jax.value_and_grad(loss_fn, has_aux=True)(params, model, b)
        upd, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, upd)
    end = flatten_dict(params)
    resp = [k for k in start if k[-1] == "gp_response_raw"]
    assert resp
    for k in resp:
        onp.testing.assert_array_equal(onp.asarray(end[k]), start[k])
    moved = [k for k in start if k[-1] != "gp_response_raw"
             and float(onp.max(onp.abs(onp.asarray(end[k]) - start[k]))) > 0]
    assert moved, "frozen control froze everything, not just the response"


# --------------------------------------- conventional chart: conversion/JVP

def test_modal_chart_conversion_is_exact():
    kw = ssm_kwargs()
    mi = matched_init(kw["Lambda_re_init"], kw["Lambda_im_init"], kw["Vinv"],
                      kw["H"], kw["P"], 0.05, jax.random.PRNGKey(7))
    gp = gp_response_coefficients(mi["a"], mi["b"], mi["t"])
    al, Bh, z = (mi["modal"][k] for k in ("alpha", "B_h", "z"))
    onp.testing.assert_allclose(onp.asarray(np.exp(al)),
                                onp.asarray(gp["a_bar"]), rtol=0, atol=1e-14)
    onp.testing.assert_allclose(onp.asarray(phi1(al)[:, None] * Bh),
                                onp.asarray(gp["b_bar"]), rtol=0, atol=1e-14)
    onp.testing.assert_allclose(onp.asarray(z[:, None] * Bh),
                                onp.asarray(gp["d_x"]), rtol=0, atol=1e-14)


def test_modal_chart_directional_derivatives_agree_with_finite_differences():
    kw = ssm_kwargs()
    mod = init_modal_ssm(gp_init_scale=0.05, **kw)(step_rescale=1.0)
    u = jax.random.normal(jax.random.PRNGKey(0), (24, kw["H"]))
    v = jax.tree_util.tree_map(lambda x: x.astype(np.float64),
                               mod.init(jax.random.PRNGKey(1), u))
    params = v["params"]
    keys = jax.random.split(jax.random.PRNGKey(3),
                            len(jax.tree_util.tree_leaves(params)))
    it = iter(keys)
    dirn = jax.tree_util.tree_map(
        lambda x: jax.random.normal(next(it), x.shape, dtype=x.dtype), params)
    f = lambda p: np.sum(mod.apply({"params": p}, u) ** 2)
    _, tangent = jax.jvp(f, (params,), (dirn,))
    eps = 1e-6
    shift = lambda s: jax.tree_util.tree_map(
        lambda p, d: p + s * eps * d, params, dirn)
    fd = (f(shift(+1)) - f(shift(-1))) / (2 * eps)
    assert abs(float(tangent) - float(fd)) / max(1.0, abs(float(fd))) < 1e-6


def test_modal_control_has_no_log_step_and_matches_gp_parameter_count():
    from s5.gp_ssm import init_gp_ssm
    kw = ssm_kwargs()
    u = jax.random.normal(jax.random.PRNGKey(0), (8, kw["H"]))
    gp = init_gp_ssm(mechanism="gp_diagonal", gp_init_scale=0.05,
                     **kw)(step_rescale=1.0).init(jax.random.PRNGKey(1), u)
    mo = init_modal_ssm(gp_init_scale=0.05,
                        **kw)(step_rescale=1.0).init(jax.random.PRNGKey(1), u)
    assert "log_step" not in mo["params"], "unused log_step retained"
    n = lambda d: sum(x.size for x in jax.tree_util.tree_leaves(d["params"]))
    assert n(gp) == n(mo), (n(gp), n(mo))
    temporal = lambda d, names: sum(
        v.size for k, v in flatten_dict(d["params"]).items() if k[-1] in names)
    assert temporal(gp, ("Lambda_re", "Lambda_im", "log_step",
                         "gp_response_raw")) == \
        temporal(mo, ("alpha_re", "alpha_im", "z_re", "z_im")) == 4 * kw["P"]


# ---------------------------------------------------------------------------
# Runner entry point.
#
# These exist because unit-testing the pieces is NOT enough: the runner crashed
# on EVERY mechanism (jit tried to abstractify the optax GradientTransformation
# passed as a traced argument) while every component test passed. The same
# class of miss as the earlier `import jax` crash. Run the entry point.
# ---------------------------------------------------------------------------

import experiments.gp.cue_recall_runner as RUN


ALL_MECHANISMS = ("plain", "modal_ssm", "gp_diagonal", "gp_scalar",
                  "prospective_input", "full_state_pc")


def _tiny_run(tmp_path, mechanism, **kw):
    import sys
    argv = ["cue_recall_runner", "--mechanism", mechanism, "--updates", "2",
            "--batch", "8", "--eval_every", "2", "--eval_batches", "1",
            "--outdir", str(tmp_path)]
    for k, v in kw.items():
        argv += ([f"--{k}"] if v is True else [f"--{k}", str(v)])
    old = sys.argv
    sys.argv = argv
    try:
        return RUN.main()
    finally:
        sys.argv = old


@pytest.mark.parametrize("mechanism", ALL_MECHANISMS)
def test_runner_entry_point_runs_end_to_end(tmp_path, mechanism):
    s = _tiny_run(tmp_path, mechanism)
    assert s["best"]["step"] == 2
    assert np.isfinite(s["best"]["joint_bce"])
    assert s["best"]["update_norm"] > 0.0        # parameters actually moved


def test_runner_freeze_response_leaves_the_response_untouched(tmp_path):
    s = _tiny_run(tmp_path, "gp_diagonal", freeze_response=True)
    key = "encoder/layers_0/seq/gp_response_raw"
    assert s["response_initial"][key] == s["final_response"][key]
    moving = _tiny_run(tmp_path, "gp_diagonal")
    assert moving["response_initial"][key] != moving["final_response"][key]


def test_gp_and_modal_arms_are_exactly_parameter_matched(tmp_path):
    gp = _tiny_run(tmp_path, "gp_diagonal")
    md = _tiny_run(tmp_path, "modal_ssm")
    assert gp["n_params"] == md["n_params"]
    assert gp["n_state_coords"] == md["n_state_coords"]


@pytest.mark.parametrize("mechanism", ALL_MECHANISMS)
def test_spectral_stats_describe_the_system_that_actually_ran(mechanism):
    """R3 regression: report the CLIPPED pole, not the raw parameter.

    An earlier version of `spectral_stats` read `alpha_re` directly for
    `modal_ssm` and reported |a_bar| = 1.00076 for a model that in fact
    realizes a strictly stable pole, because `ModalSSM.coefficients()` applies
    the same clip. Every arm runs with clip_eigs=True, so no arm may ever be
    reported as marginally unstable.
    """
    model = RUN.build_model(mechanism, 0.05, training=True)
    batch = T.generate(T.split_key(0, T.Split.TRAIN, 0), 4, T.TaskConfig())
    params, _, _ = RUN.make_state(model, jax.random.PRNGKey(0), batch,
                                  1e-3, 1e-3)
    stats = RUN.spectral_stats(params)
    assert stats, "every arm has at least one SSM layer"
    for layer, st in stats.items():
        assert st["abs_abar_max"] < 1.0, (mechanism, layer, st)
        assert st["eff_pole_re_max"] < 0.0
        assert st["n_modes"] == 16


def test_prospective_coupling_is_reported_only_where_it_exists():
    """t|a| is meaningful for the GP family; for modal_ssm `z` is a free
    static gain and must NOT be reported as if it were a coupling."""
    batch = T.generate(T.split_key(0, T.Split.TRAIN, 0), 4, T.TaskConfig())

    def stats(mech):
        model = RUN.build_model(mech, 0.05, training=True)
        p, _, _ = RUN.make_state(model, jax.random.PRNGKey(0), batch, 1e-3, 1e-3)
        return RUN.spectral_stats(p)

    for st in stats("gp_diagonal").values():
        assert st["t_abs_a_max"] > 0.0
    for st in stats("modal_ssm").values():
        assert st["t_abs_a_max"] is None
        assert st["abs_z_max"] > 0.0
    for st in stats("plain").values():
        assert st["t_abs_a_max"] == 0.0        # no response parameter at all
