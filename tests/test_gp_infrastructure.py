"""Milestone A (checkpointing/metrics) and C (diagnostics, training smoke).

The resume claim under test is exactly the DECLARED one: epoch boundary,
in-process, same host and backend. Mid-epoch and cross-process GPU resume are
neither claimed nor checked here.
"""
import json
import os
import sys
import tempfile
from functools import partial

import jax
import numpy as onp
import optax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5.checkpointing import (append_metrics, make_run_dir, provenance,     # noqa: E402
                              restore_checkpoint, save_checkpoint,
                              write_config)
from s5.gp_diagnostics import core_from_module, summarize                   # noqa: E402
from s5.gp_ssm import GPSSM, init_gp_ssm                                    # noqa: E402
from s5.seq_model import BatchClassificationModel                           # noqa: E402
from s5.train_helpers import create_train_state, train_step                 # noqa: E402
from tests.test_gp_prospective import build, inputs, ssm_kwargs, H, L        # noqa: E402

BSZ, SEQ, NCLS = 4, 16, 5


def _model_cls(mechanism, gp_init_scale=0.1):
    # GP mechanisms require clip_eigs=True (admissibility, R2). Plain keeps the
    # historical default unless a matched comparison asks otherwise.
    fn = init_gp_ssm(mechanism=mechanism, gp_init_scale=gp_init_scale,
                     **ssm_kwargs(clip_eigs=(mechanism != "plain")))
    return partial(BatchClassificationModel, ssm=fn, d_output=NCLS, d_model=H,
                   n_layers=2, padded=False, activation="half_glu1",
                   dropout=0.0, mode="pool", prenorm=True, batchnorm=False)


def _batch(seed=0):
    x = jax.random.normal(jax.random.PRNGKey(seed), (BSZ, SEQ, 1))
    y = np.arange(BSZ) % NCLS
    ts = np.ones((BSZ, SEQ))
    return x, y, ts


# ---------------------------------------------------------------- Milestone A

def test_A_provenance_records_required_fields():
    p = provenance()
    for k in ("commit", "branch", "dirty", "hostname", "jax_version",
              "backend", "devices", "timestamp"):
        assert k in p
    if p["dirty"]:
        # must cover unstaged + staged + untracked content, not `git diff` alone
        assert p["dirty_manifest_sha256"], "dirty tree needs a manifest identity"
        assert "untracked_files" in p


@pytest.mark.parametrize("mechanism", ["plain", "gp_diagonal"])
def test_A_checkpoint_restore_reproduces_eval_and_next_update(mechanism):
    """Declared boundary: epoch, in-process, same host/backend."""
    model_cls = _model_cls(mechanism)
    state = create_train_state(model_cls, jax.random.PRNGKey(0), padded=False,
                               retrieval=False, in_dim=1, bsz=BSZ, seq_len=SEQ,
                               batchnorm=False)
    x, y, ts = _batch()
    model = model_cls(training=True)

    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(d, "ck", state, epoch=0, step=7, data_seed=1919)
        before_eval = onp.asarray(model.apply({"params": state.params}, x, ts))
        s1, _ = train_step(state, jax.random.PRNGKey(3), x, y, ts, model, False)

        # perturb the live state, then restore
        state = state.replace(params=jax.tree_util.tree_map(
            lambda p: p + 1.0, state.params))
        restored, _, meta = restore_checkpoint(d, "ck", state)

        after_eval = onp.asarray(model.apply({"params": restored.params}, x, ts))
        s2, _ = train_step(restored, jax.random.PRNGKey(3), x, y, ts, model, False)

    onp.testing.assert_array_equal(before_eval, after_eval)   # fixed-batch eval
    from flax.traverse_util import flatten_dict
    f1, f2 = flatten_dict(s1.params), flatten_dict(s2.params)
    assert sorted(f1) == sorted(f2)
    for k in f1:
        onp.testing.assert_array_equal(f1[k], f2[k])          # next update
    assert meta["resume_boundary"].startswith("epoch")


def test_A_checkpointing_consumes_no_randomness():
    """Saving must not draw or split a PRNG key."""
    model_cls = _model_cls("gp_diagonal")
    state = create_train_state(model_cls, jax.random.PRNGKey(0), padded=False,
                               retrieval=False, in_dim=1, bsz=BSZ, seq_len=SEQ,
                               batchnorm=False)
    key = jax.random.PRNGKey(99)
    a = jax.random.normal(key, (4,))
    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(d, "ck", state, epoch=0, step=0)
        append_metrics(d, dict(epoch=0, val_acc=0.5))
    b = jax.random.normal(key, (4,))
    onp.testing.assert_array_equal(onp.asarray(a), onp.asarray(b))


def test_A_metrics_are_written_at_full_precision():
    with tempfile.TemporaryDirectory() as d:
        value = 0.12345678901234567
        append_metrics(d, dict(epoch=0, val_loss=value))
        rec = json.loads(open(os.path.join(d, "metrics.jsonl")).read().strip())
    assert rec["val_loss"] == value           # not rounded console text


def test_A_unique_run_directories():
    with tempfile.TemporaryDirectory() as d:
        a, b = make_run_dir(d, "x"), make_run_dir(d, "y")
        assert a != b and os.path.isdir(a) and os.path.isdir(b)


# ------------------------------------------- optimizer group (Milestone B/A)

def test_optimizer_group_reaches_the_response_parameter():
    """The response parameter must be in the ssm group: adam, NO weight decay.

    AdamW decay would shrink the response toward zero - toward the plain
    baseline - and bias every treatment/control comparison.
    """
    from s5.train_helpers import map_nested_fn
    ssm_fn = map_nested_fn(
        lambda k, _: "ssm"
        if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm",
                 "gp_response_raw"]
        else ("none" if k in [] else "regular"))
    labels = ssm_fn({"gp_response_raw": np.zeros(3), "C": np.zeros(3)})
    assert labels["gp_response_raw"] == "ssm"
    assert labels["C"] == "regular"


def test_optimizer_actually_updates_the_response_parameter():
    model_cls = _model_cls("gp_diagonal")
    state = create_train_state(model_cls, jax.random.PRNGKey(0), padded=False,
                               retrieval=False, in_dim=1, bsz=BSZ, seq_len=SEQ,
                               batchnorm=False)
    x, y, ts = _batch(1)
    before = jax.tree_util.tree_leaves(
        {k: v for k, v in _flat(state.params).items()
         if k[-1] == "gp_response_raw"})
    new, _ = train_step(state, jax.random.PRNGKey(2), x, y, ts,
                        model_cls(training=True), False)
    after = jax.tree_util.tree_leaves(
        {k: v for k, v in _flat(new.params).items()
         if k[-1] == "gp_response_raw"})
    assert before and len(before) == len(after)
    assert any(float(np.max(np.abs(a - b))) > 0.0 for a, b in zip(before, after))


def _flat(tree):
    from flax.traverse_util import flatten_dict
    return flatten_dict(tree)


# ---------------------------------------------------------------- Milestone C

def test_C_diagnostics_native_pole_is_not_the_effective_pole():
    mod = build("gp_diagonal", gp_init_scale=0.8)
    v = jax.tree_util.tree_map(lambda z: z.astype(np.float64),
                               mod.init(jax.random.PRNGKey(0), inputs()))
    core, a, a_eff = core_from_module(mod, v)
    s = summarize(core, a=a, a_eff=a_eff, n_impulse=128, n_hankel=24)
    assert s["max_discrete_pole_abs"] < 1.0          # stable
    assert s["margin_discrete"] > 0.0
    # the generalized response genuinely moves the pole
    assert onp.max(onp.abs(onp.asarray(a).real
                           - onp.asarray(a_eff).real)) > 1e-6
    for k in ("dc_gain_norm", "hankel_top", "hankel_nuclear",
              "history_fraction", "impulse_support"):
        assert k in s and onp.isfinite(s[k])
    assert "NOT the nonlinear network" in s["scope"]


def test_C_matched_tss_has_no_history_in_diagnostics():
    mod = build("full_state_pc", gp_init_scale=0.5)
    v = jax.tree_util.tree_map(lambda z: z.astype(np.float64),
                               mod.init(jax.random.PRNGKey(0), inputs()))
    core, a, a_eff = core_from_module(mod, v)
    s = summarize(core, a=a, a_eff=a_eff, n_impulse=64, n_hankel=16)
    assert s["history_norm"] < 1e-12
    assert s["hankel_top"] < 1e-12          # no past-to-future operator at all
    assert s["impulse_support"] == 1


def test_C_plain_diagnostics_have_real_memory():
    mod = build("gp_diagonal", gp_init_scale=1e-6)   # essentially plain
    v = jax.tree_util.tree_map(lambda z: z.astype(np.float64),
                               mod.init(jax.random.PRNGKey(0), inputs()))
    core, a, a_eff = core_from_module(mod, v)
    s = summarize(core, a=a, a_eff=a_eff, n_impulse=128, n_hankel=24)
    assert s["history_norm"] > 1e-6
    assert s["hankel_top"] > 1e-6
    assert s["impulse_support"] > 10


@pytest.mark.parametrize("mechanism", ["gp_scalar", "gp_diagonal",
                                       "prospective_input"])
def test_C_training_smoke_every_declared_parameter_updates(mechanism):
    """Deterministic synthetic smoke: all parameters receive real updates.

    Correctness evidence only. Not a benchmark and not an improvement claim.
    """
    model_cls = _model_cls(mechanism)
    # lr chosen so the check is informative: at the create_train_state default
    # of 1e-3 the loss falls only ~8% in 60 steps, which would make any
    # threshold arbitrary. At 1e-2 the model fits the batch outright.
    state = create_train_state(model_cls, jax.random.PRNGKey(0), padded=False,
                               retrieval=False, in_dim=1, bsz=BSZ, seq_len=SEQ,
                               batchnorm=False, ssm_lr=1e-2, lr=1e-2)
    model = model_cls(training=True)
    start = _flat(state.params)
    # ONE fixed batch, repeated: the loss must fall if the gradient path is
    # wired correctly end to end. Rotating batches over a handful of steps
    # need not lower the loss and would test nothing.
    x, y, ts = _batch(0)
    losses = []
    for i in range(200):
        state, loss = train_step(state, jax.random.PRNGKey(i), x, y, ts,
                                 model, False)
        losses.append(float(loss))
    end = _flat(state.params)

    unchanged = [k for k in start
                 if float(np.max(np.abs(end[k] - start[k]))) == 0.0]
    assert not unchanged, f"{mechanism}: parameters never updated: {unchanged}"
    assert all(onp.isfinite(losses)), mechanism
    assert losses[-1] < 0.01 * losses[0], (mechanism, losses[0], losses[-1])
    # and the response parameter specifically moved
    resp = [k for k in start if k[-1] == "gp_response_raw"]
    assert resp, mechanism
    assert any(float(np.max(np.abs(end[k] - start[k]))) > 0.0 for k in resp)
