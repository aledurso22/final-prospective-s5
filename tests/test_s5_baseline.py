"""Real-S5 verification for the plain baseline branch.

These tests instantiate the actual S5SSM (HiPPO init, ZOH/bilinear
discretization, ``jax.lax.associative_scan``) and check that it runs forward,
compiles under JIT, produces finite gradients through the scan, and accepts an
optimizer update.  No stubs.
"""

import jax
import jax.numpy as np
import numpy as onp
import optax
import pytest
from flax import linen as nn

from s5.layers import SequenceLayer
from s5.seq_model import BatchClassificationModel, StackedEncoderModel
from s5.ssm import apply_ssm, binary_operator
from tests.s5_reference import make_ssm_init_fn

SEQ_LEN = 32
D_MODEL = 8
BSZ = 4


def _classification_model(training=False, **kwargs):
    ssm_init_fn, _ = make_ssm_init_fn(d_model=D_MODEL, **kwargs)
    return BatchClassificationModel(
        ssm=ssm_init_fn, d_output=10, d_model=D_MODEL, n_layers=2,
        padded=False, activation="half_glu1", dropout=0.0,
        training=training, mode="pool", prenorm=True, batchnorm=False,
    )


# ------------------------------------------------------------ (1) imports


def test_real_model_imports_and_constructs():
    """A real S5 stack builds and initializes with modern JAX."""
    model = _classification_model()
    x = np.ones((BSZ, SEQ_LEN, 1))
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        x, np.ones((BSZ, SEQ_LEN)),
    )
    params = variables["params"]
    flat = jax.tree_util.tree_leaves(params)
    assert flat, "model produced no parameters"
    assert all(np.all(np.isfinite(p)) for p in flat)


# ------------------------------------------------------ (2) forward pass


def test_real_s5_forward_pass():
    model = _classification_model()
    x = jax.random.normal(jax.random.PRNGKey(2), (BSZ, SEQ_LEN, 1))
    ts = np.ones((BSZ, SEQ_LEN))
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, ts
    )
    logits = model.apply(variables, x, ts)
    assert logits.shape == (BSZ, 10)
    assert np.all(np.isfinite(logits))
    # log_softmax output: each row must normalize
    onp.testing.assert_allclose(
        onp.asarray(np.exp(logits).sum(axis=-1)), onp.ones(BSZ), rtol=1e-5
    )


def test_ssm_layer_output_is_real_and_correct_shape():
    """apply_ssm returns real (L, H) preactivations from complex state."""
    ssm_init_fn, P = make_ssm_init_fn(d_model=D_MODEL)
    ssm = ssm_init_fn(step_rescale=1.0)
    u = jax.random.normal(jax.random.PRNGKey(3), (SEQ_LEN, D_MODEL))
    variables = ssm.init(jax.random.PRNGKey(0), u)
    y = ssm.apply(variables, u)
    assert y.shape == (SEQ_LEN, D_MODEL)
    assert not np.iscomplexobj(y)
    assert np.all(np.isfinite(y))


def test_associative_scan_matches_sequential_recurrence():
    """The parallel scan reproduces x_t = Lambda_bar x_{t-1} + B_bar u_t."""
    key = jax.random.PRNGKey(7)
    P, H, L = 6, 4, 16
    Lambda_bar = jax.random.uniform(key, (P,), minval=0.1, maxval=0.9).astype(
        np.complex64
    )
    B_bar = jax.random.normal(jax.random.PRNGKey(8), (P, H)).astype(np.complex64)
    C_tilde = jax.random.normal(jax.random.PRNGKey(9), (H, P)).astype(np.complex64)
    u = jax.random.normal(jax.random.PRNGKey(10), (L, H))

    parallel = apply_ssm(Lambda_bar, B_bar, C_tilde, u, conj_sym=False,
                         bidirectional=False)

    x = np.zeros((P,), dtype=np.complex64)
    sequential = []
    for t in range(L):
        x = Lambda_bar * x + B_bar @ u[t]
        sequential.append((C_tilde @ x).real)
    onp.testing.assert_allclose(
        onp.asarray(parallel), onp.asarray(np.stack(sequential)),
        rtol=1e-4, atol=1e-4,
    )


def test_binary_operator_is_associative():
    key = jax.random.PRNGKey(11)
    shape = (5,)
    elems = [
        (jax.random.uniform(jax.random.fold_in(key, 2 * i), shape).astype(np.complex64),
         jax.random.normal(jax.random.fold_in(key, 2 * i + 1), shape).astype(np.complex64))
        for i in range(3)
    ]
    a, b, c = elems
    left = binary_operator(binary_operator(a, b), c)
    right = binary_operator(a, binary_operator(b, c))
    for l, r in zip(left, right):
        onp.testing.assert_allclose(onp.asarray(l), onp.asarray(r),
                                    rtol=1e-5, atol=1e-6)


# -------------------------------------------------------------- (3) JIT


def test_jit_compiles_and_matches_eager():
    model = _classification_model()
    x = jax.random.normal(jax.random.PRNGKey(4), (BSZ, SEQ_LEN, 1))
    ts = np.ones((BSZ, SEQ_LEN))
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, ts
    )
    jitted = jax.jit(model.apply)
    onp.testing.assert_allclose(
        onp.asarray(jitted(variables, x, ts)),
        onp.asarray(model.apply(variables, x, ts)),
        rtol=1e-5, atol=1e-5,
    )


# --------------------------------------- (4) gradients through the scan


def test_gradients_flow_through_associative_scan():
    """Every S5 parameter, including Lambda and log_step, gets a finite grad."""
    model = _classification_model()
    x = jax.random.normal(jax.random.PRNGKey(5), (BSZ, SEQ_LEN, 1))
    ts = np.ones((BSZ, SEQ_LEN))
    labels = np.array([0, 1, 2, 3])
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, ts
    )

    def loss_fn(params):
        logits = model.apply({"params": params}, x, ts)
        return -np.mean(logits[np.arange(BSZ), labels])

    grads = jax.jit(jax.grad(loss_fn))(variables["params"])
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    assert flat
    for path, g in flat:
        name = jax.tree_util.keystr(path)
        assert np.all(np.isfinite(g)), f"non-finite gradient at {name}"

    names = [jax.tree_util.keystr(p) for p, _ in flat]
    joined = " ".join(names)
    for expected in ("Lambda_re", "Lambda_im", "log_step", "C", "D"):
        assert expected in joined, f"missing S5 parameter {expected} in {names}"

    # gradients must not be uniformly zero
    total = sum(float(np.sum(np.abs(g))) for _, g in flat)
    assert total > 0.0


def test_gradient_wrt_inputs_is_finite():
    model = _classification_model()
    x = jax.random.normal(jax.random.PRNGKey(6), (BSZ, SEQ_LEN, 1))
    ts = np.ones((BSZ, SEQ_LEN))
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, ts
    )
    g = jax.grad(lambda inp: np.sum(model.apply(variables, inp, ts) ** 2))(x)
    assert g.shape == x.shape
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


# ------------------------------------------------- (5) one optimizer step


def test_single_optimizer_update_decreases_loss():
    model = _classification_model()
    x = jax.random.normal(jax.random.PRNGKey(12), (BSZ, SEQ_LEN, 1))
    ts = np.ones((BSZ, SEQ_LEN))
    labels = np.array([0, 1, 2, 3])
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, ts
    )
    params = variables["params"]

    def loss_fn(p):
        logits = model.apply({"params": p}, x, ts)
        return -np.mean(logits[np.arange(BSZ), labels])

    tx = optax.adam(1e-2)
    opt_state = tx.init(params)

    before = loss_fn(params)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    after = loss_fn(params)

    assert np.isfinite(before) and np.isfinite(after)
    assert after < before, f"loss did not decrease: {before} -> {after}"


def test_create_train_state_and_train_step():
    """The repository's own optimizer plumbing runs end to end."""
    from functools import partial
    from s5.train_helpers import create_train_state, train_step

    ssm_init_fn, _ = make_ssm_init_fn(d_model=D_MODEL)
    model_cls = partial(
        BatchClassificationModel, ssm=ssm_init_fn, d_output=10, d_model=D_MODEL,
        n_layers=2, padded=False, activation="half_glu1", dropout=0.0,
        mode="pool", prenorm=True, batchnorm=False,
    )
    state = create_train_state(
        model_cls, jax.random.PRNGKey(0), padded=False, retrieval=False,
        in_dim=1, bsz=BSZ, seq_len=SEQ_LEN, batchnorm=False,
    )
    x = jax.random.normal(jax.random.PRNGKey(13), (BSZ, SEQ_LEN, 1))
    ts = np.ones((BSZ, SEQ_LEN))
    labels = np.array([0, 1, 2, 3])

    new_state, loss = train_step(
        state, jax.random.PRNGKey(1), x, labels, ts,
        model_cls(training=True), False,
    )
    assert np.isfinite(loss)
    assert new_state.step == state.step + 1


# ------------------------------------------------------------ bidirectional


def test_bidirectional_s5_runs():
    """Plain S5 supports bidirectional; the baseline must not regress it."""
    ssm_init_fn, _ = make_ssm_init_fn(d_model=D_MODEL, bidirectional=True)
    layer = SequenceLayer(ssm=ssm_init_fn, dropout=0.0, d_model=D_MODEL,
                          training=False)
    u = jax.random.normal(jax.random.PRNGKey(14), (SEQ_LEN, D_MODEL))
    variables = layer.init(jax.random.PRNGKey(0), u)
    y = layer.apply(variables, u)
    assert y.shape == (SEQ_LEN, D_MODEL) and np.all(np.isfinite(y))


@pytest.mark.parametrize("discretization", ["zoh", "bilinear"])
def test_both_discretizations_run(discretization):
    ssm_init_fn, _ = make_ssm_init_fn(d_model=D_MODEL,
                                      discretization=discretization)
    ssm = ssm_init_fn(step_rescale=1.0)
    u = jax.random.normal(jax.random.PRNGKey(15), (SEQ_LEN, D_MODEL))
    variables = ssm.init(jax.random.PRNGKey(0), u)
    assert np.all(np.isfinite(ssm.apply(variables, u)))
