"""Predictor-corrector variants in PRODUCTION float32 (x64 must be OFF).

    python -m pytest tests/test_predictor_corrector_float32.py -q
"""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from experiments.s5_three_arm_full import runner as R
from s5 import factored_recurrence as FR
from s5 import predictor_corrector as PC
from s5.discrete_recurrence import scan_companion, scan_companion_sequential

ALL = list(PC.VARIANTS.items())
SCANS = {"sequential": scan_companion_sequential, "companion": scan_companion,
         "factored": FR.scan_factored}


@pytest.fixture(autouse=True)
def _require_float32():
    assert not jax.config.read("jax_enable_x64"), (
        "this file checks production float32; run it WITHOUT JAX_ENABLE_X64")


def _init_modes(n=64):
    kw = R.ssm_kwargs("generalized_prospective_s5")
    lam = onp.exp((kw["Lambda_re_init"][:n] + 1j * kw["Lambda_im_init"][:n])
                  * onp.geomspace(1e-3, 1e-1, n))
    return lam


@pytest.mark.parametrize("name,variant", ALL)
def test_float32_scans_match_float64_reference(name, variant):
    lam = _init_modes(); n = lam.size
    rng = onp.random.default_rng(0)
    b = rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4))
    x = rng.normal(size=(2000, 4))
    T, M, g = 0.05 * onp.ones(n), 0.025 * onp.ones(n), onp.ones(n)
    ref = PC.direct_rollout(lam, b, T, M, g, x, *variant)       # numpy f64
    co = PC.pc_coefficients(np.asarray(lam, np.complex64),
                            np.asarray(b, np.complex64),
                            *(np.asarray(v, np.float32) for v in (T, M, g)),
                            *variant)
    assert co[0].dtype == np.complex64
    scale = onp.abs(ref).max()
    for key, scan in SCANS.items():
        got = onp.asarray(scan(*co, np.asarray(x, np.float32)))
        err = onp.abs(got - ref).max() / scale
        assert err < 2e-3, (key, err)


@pytest.mark.parametrize("name,variant", ALL)
def test_float32_gradients_match_float64_finite_differences(name, variant):
    """Each float32 scan's d loss / d(T, M, gamma) against a float64 central
    difference of the LITERAL predictor-corrector equations. Measured worst
    (relative to the largest gradient entry): 5.5e-3, the M-gradient of the
    legacy-scale variants through the SEQUENTIAL scan; <= 3e-4 elsewhere."""
    n = 16
    lam = _init_modes(n)
    x = onp.asarray(jax.random.normal(jax.random.PRNGKey(0), (500, 2)), float)
    b = onp.ones((n, 2), complex)
    base = (0.05 * onp.ones(n), 0.025 * onp.ones(n), onp.ones(n))

    def f64(*p):   # per-mode losses; modes are independent
        return (abs(PC.direct_rollout(lam, b, *p, x, *variant)) ** 2).sum(0)

    h, refs = 1e-6, []
    for i in range(3):
        up = [v + h * (j == i) for j, v in enumerate(base)]
        dn = [v - h * (j == i) for j, v in enumerate(base)]
        refs.append((f64(*up) - f64(*dn)) / (2 * h))

    def loss(scan, T, M, g):
        co = PC.pc_coefficients(np.asarray(lam, np.complex64),
                                np.asarray(b, np.complex64), T, M, g, *variant)
        return np.sum(np.abs(scan(*co, np.asarray(x, np.float32))) ** 2)

    args = tuple(np.asarray(v, np.float32) for v in base)
    for key, scan in SCANS.items():
        got = jax.grad(lambda *a: loss(scan, *a), argnums=(0, 1, 2))(*args)
        for g_, ref in zip(got, refs):
            g_ = onp.asarray(g_, float)
            assert onp.all(onp.isfinite(g_))
            assert onp.abs(g_ - ref).max() / onp.abs(ref).max() < 1e-2, key


@pytest.mark.parametrize("implementation", ["sequential", "factored"])
@pytest.mark.parametrize("name,variant", ALL)
def test_16000_steps_production_width_finite_gradients(name, variant,
                                                       implementation):
    """The production layer (H, P, bidirectional) over 16,000 tokens:
    outputs bounded and every gradient finite, at the matched init."""
    kw = R.ssm_kwargs("generalized_prospective_s5")
    mod = PC.PredictorCorrectorS5SSM(**kw, scale=variant[0],
                                     corrector=variant[1],
                                     implementation=implementation)
    x = jax.random.normal(jax.random.PRNGKey(1), (16000, kw["H"]))
    params = mod.init(jax.random.PRNGKey(0), x[:8])
    out, grads = jax.jit(jax.value_and_grad(
        lambda p: np.mean(mod.apply(p, x) ** 2)))(params)
    assert onp.isfinite(float(out)) and float(out) < 1e3
    for leaf in jax.tree_util.tree_leaves(grads):
        assert onp.all(onp.isfinite(onp.asarray(leaf)))
