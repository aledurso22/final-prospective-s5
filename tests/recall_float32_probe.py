"""Production-dtype probe for the recall arms. x64 OFF, own process.

Matches production matmul precision, for the reason recorded in the constrained
protocol: at the GPU's TF32 default a finite-difference signal of this size is
the same order as the arithmetic noise.
"""

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
import numpy as onp                                              # noqa: E402

assert not jax.config.read("jax_enable_x64"), "x64 must be OFF in this probe"
jax.config.update("jax_default_matmul_precision", "highest")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tasks.recall as T                                          # noqa: E402
from experiments.gp import recall_study as RS                     # noqa: E402
from s5.rawat_s5 import RHO_ONLY_PARAM_NAME                       # noqa: E402

TOL, STEP = 5e-3, 1e-2
rng = onp.random.RandomState(0)
x, y = T.generate_fixed_delay(rng, 32, delay=32)
x, y = jnp.asarray(x), jnp.asarray(y)
print(f"  backend={jax.default_backend()}  x {x.dtype}  y {y.dtype}")
assert x.dtype == onp.float32

for arm in RS.ARMS:
    size = RS.SSM_SIZE * 2 if arm == "ordinary_2x" else RS.SSM_SIZE
    m, p = RS.init_params(arm, 0, size)
    logits = m.apply({"params": p}, x, jnp.ones(x.shape[:2]))
    assert logits.dtype == onp.float32, (arm, logits.dtype)
    assert bool(onp.isfinite(onp.asarray(logits)).all()), arm
    loss, _ = RS.loss_fn(p, m, x, y)
    print(f"  {arm:16s} logits {logits.dtype} {logits.shape}  "
          f"loss {float(loss):.5f}")

# float32 gradient through the rho leaf, against a central difference
m, p = RS.init_params("gp_rho", 0)
from flax.traverse_util import flatten_dict, unflatten_dict                # noqa
flat = flatten_dict(p)
key = [k for k in flat if k[-1] == RHO_ONLY_PARAM_NAME][0]
print(f"  rho leaf {'/'.join(key)} dtype {flat[key].dtype} "
      f"shape {flat[key].shape}")


def loss_of(v):
    f = dict(flat); f[key] = v
    return float(RS.loss_fn(unflatten_dict(f), m, x, y)[0])


def grad_of():
    def L(params):
        return RS.loss_fn(params, m, x, y)[0]
    return flatten_dict(jax.grad(L)(p))[key]


g = onp.asarray(grad_of())
assert onp.isfinite(g).all() and onp.max(onp.abs(g)) > 0.0, "rho gradient"
d = onp.random.RandomState(1).randn(*g.shape)
d = jnp.asarray(d / d.std(), dtype=flat[key].dtype)
ana = float(jnp.sum(jnp.asarray(g) * d))
fd = (loss_of(flat[key] + STEP * d) - loss_of(flat[key] - STEP * d)) / (2 * STEP)
rel = abs(ana - fd) / max(abs(fd), 1e-8)
print(f"  rho gradient: analytic {ana:.6f}  fd {fd:.6f}  rel {rel:.3e} "
      f"(tol {TOL:.0e})")
assert rel < TOL, f"GATE FAILED: {rel}"
print("RECALL_F32_OK")
