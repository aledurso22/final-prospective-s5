"""Tests that pin the prospective branch against the plain-S5 baseline.

These complement ``tests/test_prospective.py`` (operator semantics) by
checking (a) that ``prospective_mode='off'`` is byte-for-byte the ``main``
code path, and (b) that the S5 state recurrence / associative scan in
``s5/ssm.py`` is identical to ``main``.

The reference is the ``main`` branch, not upstream ``3c18fdb``: ``main``
already carries the modern-JAX compatibility fixes, so diffing against it
isolates the prospective feature and nothing else.
"""

import hashlib
import importlib.util
import pathlib
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from flax.traverse_util import flatten_dict

from s5.layers import SequenceLayer
from s5.prospective import ProspectiveLead, apply_parallel, step
from s5.seq_model import StackedEncoderModel

REPO = pathlib.Path(__file__).resolve().parents[1]

# The plain-S5 baseline this branch must remain a pure superset of.
BASELINE_REF = "main"

# The official Linderman S5 commit, for the record.
UPSTREAM_COMMIT = "3c18fdb6b06414da35e77b94b9cd855f6a95ef17"

# sha256 of s5/ssm.py on main (upstream + the DeviceArray -> jax.Array
# annotation fix only).
SSM_SHA256 = "be64f943dd1a629f9967ffad9fd1807eb0becf43f28aff1f791a3a94284e0fac"


class _FakeSSM(nn.Module):
    """Deterministic stand-in for S5SSM with the same call signature.

    Used where a test needs an exactly reproducible, cheap sequence-to-sequence
    map so that "off vs lead(alpha=0)" comparisons are bit-exact and fast.
    The prospective operator on the *real* S5SSM is covered separately by
    ``test_real_s5_prospective_*`` below.
    """

    step_rescale: float = 1.0

    @nn.compact
    def __call__(self, sequence):
        return jnp.cumsum(nn.Dense(sequence.shape[-1])(sequence), axis=0)


# ---------------------------------------------------------------- integrity


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def test_ssm_source_digest_is_unchanged():
    """The S5 SSM implementation, discretization and scan are untouched."""
    digest = hashlib.sha256((REPO / "s5" / "ssm.py").read_bytes()).hexdigest()
    assert digest == SSM_SHA256


def test_ssm_and_scan_match_baseline_branch():
    """Belt-and-braces: git agrees s5/ssm.py is identical to main."""
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    diff = _git("diff", BASELINE_REF, "--", "s5/ssm.py")
    assert diff == "", "s5/ssm.py differs from main:\n" + diff

    source = (REPO / "s5" / "ssm.py").read_text()
    assert "def binary_operator(q_i, q_j):" in source
    assert source.count("jax.lax.associative_scan") == 2


def _load_baseline_layers():
    """Import main's s5/layers.py as a throwaway module."""
    source = _git("show", f"{BASELINE_REF}:s5/layers.py")
    module = types.ModuleType("baseline_layers")
    module.__dict__["__name__"] = "baseline_layers"
    exec(compile(source, "<main s5/layers.py>", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("activation", ["gelu", "half_glu1", "half_glu2", "full_glu"])
@pytest.mark.parametrize("prenorm", [True, False])
def test_off_mode_reproduces_baseline_layer(activation, prenorm):
    """prospective_mode='off' is main's computation and parameter tree."""
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    baseline = _load_baseline_layers()

    sequence = jax.random.normal(jax.random.PRNGKey(3), (16, 4))
    common = dict(
        ssm=_FakeSSM,
        dropout=0.0,
        d_model=4,
        activation=activation,
        training=False,
        prenorm=prenorm,
        batchnorm=False,
    )
    key = jax.random.PRNGKey(11)

    reference = baseline.SequenceLayer(**common)
    branch = SequenceLayer(**common, prospective_mode="off")

    reference_variables = reference.init(key, sequence)
    branch_variables = branch.init(key, sequence)

    assert sorted(flatten_dict(reference_variables["params"])) == sorted(
        flatten_dict(branch_variables["params"])
    )
    np.testing.assert_array_equal(
        branch.apply(branch_variables, sequence),
        reference.apply(reference_variables, sequence),
    )


def test_off_mode_stack_reproduces_baseline_stack():
    """A whole StackedEncoderModel in 'off' mode matches main exactly."""
    sequence = jax.random.normal(jax.random.PRNGKey(5), (12, 4))
    common = dict(
        ssm=_FakeSSM, d_model=4, n_layers=3, dropout=0.0, training=False,
        activation="half_glu1",
    )
    key = jax.random.PRNGKey(2)

    off = StackedEncoderModel(**common, prospective_mode="off")
    zero = StackedEncoderModel(
        **common,
        prospective_mode="lead",
        prospective_alpha=0.0,
        prospective_layers="all",
    )
    off_variables = off.init(key, sequence, None)
    zero_variables = zero.init(key, sequence, None)

    np.testing.assert_array_equal(
        zero.apply(zero_variables, sequence, None),
        off.apply(off_variables, sequence, None),
    )


# ---------------------------------------------------------------- placement


@pytest.mark.parametrize("placement,expected", [("all", 3), ("last", 1)])
def test_placement_learned_alpha_count(placement, expected):
    sequence = jnp.ones((6, 3))
    model = StackedEncoderModel(
        ssm=_FakeSSM, d_model=3, n_layers=3, dropout=0.0, training=False,
        prospective_mode="lead", prospective_alpha=0.25,
        prospective_alpha_learned=True, prospective_layers=placement,
    )
    variables = model.init(jax.random.PRNGKey(0), sequence, None)
    alphas = [p for p in flatten_dict(variables["params"]) if p[-1] == "alpha_logit"]
    assert len(alphas) == expected


def test_invalid_placement_is_rejected():
    model = StackedEncoderModel(
        ssm=_FakeSSM, d_model=3, n_layers=2, dropout=0.0, training=False,
        prospective_mode="lead", prospective_layers="middle",
    )
    with pytest.raises(ValueError, match="prospective_layers"):
        model.init(jax.random.PRNGKey(0), jnp.ones((4, 3)), None)


def test_bidirectional_stack_is_rejected():
    model = StackedEncoderModel(
        ssm=_FakeSSM, d_model=3, n_layers=2, dropout=0.0, training=False,
        prospective_mode="lead", prospective_alpha=0.25, bidirectional=True,
    )
    with pytest.raises(ValueError, match="unidirectional"):
        model.init(jax.random.PRNGKey(0), jnp.ones((4, 3)), None)


# ------------------------------------------------- streaming / jit / grads


def test_module_parallel_matches_streaming_steps():
    """One vectorized shift == repeated single-token streaming with a cache."""
    sequence = jax.random.normal(jax.random.PRNGKey(1), (20, 5))
    module = ProspectiveLead(alpha=0.4, learned=True, alpha_max=1.0)
    variables = module.init(jax.random.PRNGKey(0), sequence)

    parallel = module.apply(variables, sequence)

    cache = None
    streamed = []
    for token in sequence:
        out, cache = module.apply(
            variables, token, cache, False, method=ProspectiveLead.step
        )
        streamed.append(out)
        assert cache.shape == (5,)  # one H-dimensional cache only

    np.testing.assert_allclose(parallel, jnp.stack(streamed), rtol=0, atol=0)


def test_jit_layer_and_gradients_wrt_inputs_and_alpha():
    sequence = jax.random.normal(jax.random.PRNGKey(9), (10, 4))
    layer = SequenceLayer(
        ssm=_FakeSSM, dropout=0.0, d_model=4, training=False,
        prospective_mode="lead", prospective_alpha=0.3,
        prospective_alpha_learned=True, prospective_alpha_max=1.0,
    )
    variables = layer.init(jax.random.PRNGKey(0), sequence)

    apply_fn = jax.jit(layer.apply)
    np.testing.assert_allclose(
        apply_fn(variables, sequence), layer.apply(variables, sequence),
        rtol=1e-6, atol=1e-6,
    )

    def loss(params, inputs):
        return jnp.sum(layer.apply({"params": params}, inputs) ** 2)

    param_grads, input_grads = jax.jit(jax.grad(loss, argnums=(0, 1)))(
        variables["params"], sequence
    )
    flat = flatten_dict(param_grads)
    alpha_grad = [v for k, v in flat.items() if k[-1] == "alpha_logit"]
    assert len(alpha_grad) == 1
    assert jnp.isfinite(alpha_grad[0]) and alpha_grad[0] != 0.0
    assert jnp.all(jnp.isfinite(input_grads)) and jnp.any(input_grads != 0.0)


def test_layer_output_is_causal():
    """Perturbing a future input cannot change an earlier layer output."""
    sequence = jax.random.normal(jax.random.PRNGKey(4), (12, 4))
    layer = SequenceLayer(
        ssm=_FakeSSM, dropout=0.0, d_model=4, training=False, prenorm=True,
        prospective_mode="lead", prospective_alpha=0.9,
    )
    variables = layer.init(jax.random.PRNGKey(0), sequence)

    perturbed = sequence.at[8:].set(1e3)
    np.testing.assert_array_equal(
        layer.apply(variables, perturbed)[:8],
        layer.apply(variables, sequence)[:8],
    )


def test_learned_alpha_respects_alpha_max():
    module = ProspectiveLead(alpha=0.5, learned=True, alpha_max=0.5)
    variables = module.init(jax.random.PRNGKey(0), jnp.ones((3, 2)))
    params = jax.tree_util.tree_map(lambda x: x + 50.0, variables["params"])
    alpha = module.apply(
        {"params": params}, method=ProspectiveLead.effective_alpha
    )
    assert 0.0 < float(alpha) <= 0.5


def test_alpha_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="alpha"):
        ProspectiveLead(alpha=2.0, alpha_max=1.0).init(
            jax.random.PRNGKey(0), jnp.ones((2, 2))
        )


# ------------------------------------------------- real S5SSM (no stubs)


def _real_classification_model(training=False, **prospective):
    from s5.seq_model import BatchClassificationModel
    from tests.s5_reference import make_ssm_init_fn

    ssm_init_fn, _ = make_ssm_init_fn(d_model=8)
    return BatchClassificationModel(
        ssm=ssm_init_fn, d_output=10, d_model=8, n_layers=2, padded=False,
        activation="half_glu1", dropout=0.0, training=training, mode="pool",
        prenorm=True, batchnorm=False, **prospective,
    )


def test_real_s5_prospective_alpha_zero_matches_off():
    """On the real S5SSM, fixed alpha=0 is bit-identical to plain S5."""
    x = jax.random.normal(jax.random.PRNGKey(21), (4, 32, 1))
    ts = jnp.ones((4, 32))
    rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}

    off = _real_classification_model(prospective_mode="off")
    zero = _real_classification_model(
        prospective_mode="lead", prospective_alpha=0.0, prospective_layers="all"
    )
    off_variables = off.init(rngs, x, ts)
    zero_variables = zero.init(rngs, x, ts)

    np.testing.assert_array_equal(
        zero.apply(zero_variables, x, ts), off.apply(off_variables, x, ts)
    )


def test_real_s5_prospective_forward_jit_and_grad():
    """Real S5SSM + prospective lead: forward, JIT and gradients all work."""
    x = jax.random.normal(jax.random.PRNGKey(22), (4, 32, 1))
    ts = jnp.ones((4, 32))
    labels = jnp.array([0, 1, 2, 3])
    model = _real_classification_model(
        prospective_mode="lead", prospective_alpha=0.25,
        prospective_alpha_learned=True, prospective_layers="all",
    )
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, ts
    )

    logits = model.apply(variables, x, ts)
    assert logits.shape == (4, 10) and jnp.all(jnp.isfinite(logits))
    np.testing.assert_allclose(
        np.asarray(jax.jit(model.apply)(variables, x, ts)),
        np.asarray(logits), rtol=1e-5, atol=1e-5,
    )

    def loss_fn(params):
        return -jnp.mean(model.apply({"params": params}, x, ts)[jnp.arange(4), labels])

    grads = jax.jit(jax.grad(loss_fn))(variables["params"])
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    for path, g in flat:
        assert jnp.all(jnp.isfinite(g)), jax.tree_util.keystr(path)

    alpha_grads = [g for p, g in flat if p[-1].key == "alpha_logit"]
    assert len(alpha_grads) == 2  # one per block, prospective_layers="all"
    assert all(jnp.isfinite(g) and g != 0.0 for g in alpha_grads)

    # S5's own parameters must still receive gradients through the scan
    names = " ".join(jax.tree_util.keystr(p) for p, _ in flat)
    for expected in ("Lambda_re", "Lambda_im", "log_step", "C", "D"):
        assert expected in names


def test_real_s5_bidirectional_plus_prospective_is_rejected():
    from s5.seq_model import BatchClassificationModel
    from tests.s5_reference import make_ssm_init_fn

    ssm_init_fn, _ = make_ssm_init_fn(d_model=8, bidirectional=True)
    model = BatchClassificationModel(
        ssm=ssm_init_fn, d_output=10, d_model=8, n_layers=2, padded=False,
        activation="half_glu1", dropout=0.0, training=False, mode="pool",
        prenorm=True, batchnorm=False,
        prospective_mode="lead", prospective_alpha=0.25, bidirectional=True,
    )
    with pytest.raises(ValueError, match="unidirectional"):
        model.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
            jnp.ones((4, 32, 1)), jnp.ones((4, 32)),
        )
