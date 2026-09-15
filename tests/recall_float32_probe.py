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

# float32 gradient through the rho leaf, against a central difference.
#
# EVALUATED AT AN INTERIOR rho, NOT at the initialization value. rho_0 = 0.9998
# sits 1.0e-4 from the declared upper bound in log space, while a usable float32
# finite-difference step is 1e-2: every positive-direction component would cross
# the ceiling into the forward clip, the "+h" side would not move, and the
# central difference would return about HALF the true directional derivative.
# That is what the first cluster run measured (analytic 0.0549 vs fd 0.0281).
# Saturation is a separate property and is checked separately below; this check
# is about the gradient.
FD_RHO = 0.75
m, p = RS.init_params("gp_rho", 0)
from flax.traverse_util import flatten_dict, unflatten_dict                # noqa
flat = dict(flatten_dict(p))
key = [k for k in flat if k[-1] == RHO_ONLY_PARAM_NAME][0]
print(f"  rho leaf {'/'.join(key)} dtype {flat[key].dtype} "
      f"shape {flat[key].shape}")

# the initialization really does sit adjacent to the ceiling: record it
import math as _math
z0 = float(onp.max(onp.asarray(flat[key])))
print(f"  log rho_0 = {z0:.6e}, upper bound = {_math.log(1 - 1e-4):.6e}, "
      f"headroom = {_math.log(1 - 1e-4) - z0:.3e}")
assert z0 < _math.log(1 - 1e-4), "rho_0 must start inside the bound"

for k in list(flat):
    if k[-1] == RHO_ONLY_PARAM_NAME:
        flat[k] = jnp.full_like(flat[k], _math.log(FD_RHO))
p = unflatten_dict(flat)
flat = dict(flatten_dict(p))
print(f"  gradient check evaluated at rho = {FD_RHO} (interior)")


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
# saturation IS the intended behaviour at the ceiling, checked on its own
_, p2 = RS.init_params("gp_rho", 0)
f2 = dict(flatten_dict(p2))
k2 = [k for k in f2 if k[-1] == RHO_ONLY_PARAM_NAME][0]
big = unflatten_dict({**f2, k2: f2[k2] + 1.0})
m2 = RS.build_single("gp_rho")          # unbatched: nn.vmap cannot map this
r_big = m2.apply({"params": big},
                 jnp.zeros((T.SEQ_LEN, T.N_CHANNELS)),
                 jnp.ones((T.SEQ_LEN,)),
                 method=lambda mm, xx, tt: mm.encoder.layers[0].seq.rho_only())
print(f"  forward clip at the ceiling: max rho after +1.0 in log space = "
      f"{float(jnp.max(r_big)):.6f} (bound {1 - 1e-4})")
assert float(jnp.max(r_big)) <= (1 - 1e-4) * (1 + 1e-6)
print("RECALL_F32_OK")
