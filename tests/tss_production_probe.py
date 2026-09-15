"""Production ENTRYPOINT probe for the TSS pilot. x64 OFF, own process.

R1. The pilot's focused checks globally enable x64 and mostly exercise eager
module paths; Part A runs in the default float32 configuration and goes through
`pilot_a.train_step` and `pilot_a.eval_all`, which are jitted and receive the
coefficient configuration as a DYNAMIC pytree. A float64 fixture cannot
establish that path: an earlier revision carried a string in that configuration
and would have been blocked at the first production call while every existing
test passed.

So this probe calls the ACTUAL entrypoints, for EVERY arm, in the production
dtype, and checks that the results are finite float32.
"""

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as onp                                               # noqa: E402

assert not jax.config.read("jax_enable_x64"), "x64 must be OFF in this probe"
jax.config.update("jax_default_matmul_precision", "highest")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tasks.dual_recall as TK                                     # noqa: E402
from experiments.tss import pilot_a as PA                          # noqa: E402
from s5 import tss_models as TM                                    # noqa: E402
from s5.tss_cells import assert_numeric_pytree                     # noqa: E402

assert not jax.config.read("jax_enable_x64"), \
    "x64 was switched back ON by an import; this probe would measure float64"
assert jnp.zeros(1).dtype == onp.float32, jnp.zeros(1).dtype

cfg = assert_numeric_pytree(TM.coefficients(), "production cfg")
rng = onp.random.RandomState(0)
xs, y_sig, y_cls, q_idx, _ = TK.generate(rng, 4)
args = (jnp.asarray(xs), jnp.asarray(y_sig), jnp.asarray(y_cls),
        jnp.asarray(q_idx))
print(f"  backend={jax.default_backend()}  x64={jax.config.read('jax_enable_x64')}")

fails = []
for arm in TM.ARMS:
    p = TM.init_params(arm, 100)
    tx = PA.get_tx()
    opt = tx.init(p)
    # the ACTUAL jitted training entrypoint
    p2, opt2, loss, aux, gn = PA.train_step(arm, tx, p, opt, *args, cfg)
    # the ACTUAL jitted evaluation entrypoint
    ev = PA.eval_all(arm, p2, *args, cfg)
    vals = dict(loss=float(loss), mse=float(aux["mse"]), ce=float(aux["ce"]),
                acc=float(aux["acc"]), grad_norm=float(gn),
                eval_mse=float(ev["mse"]), eval_ce=float(ev["ce"]),
                residual=float(aux["residual"]))
    ok = all(onp.isfinite(v) for v in vals.values())
    dt_ok = (loss.dtype == onp.float32 and ev["mse"].dtype == onp.float32)
    moved = float(max(
        onp.max(onp.abs(onp.asarray(a) - onp.asarray(b)))
        for a, b in zip(jax.tree_util.tree_leaves(p),
                        jax.tree_util.tree_leaves(p2))))
    print(f"  {arm:<28} loss={vals['loss']:.4f} acc={vals['acc']:.3f} "
          f"|dparam|={moved:.3e} residual={vals['residual']:.2e} "
          f"dtype={loss.dtype} finite={ok}")
    if not ok:
        fails.append(("nonfinite", arm, vals))
    if not dt_ok:
        fails.append(("dtype", arm, str(loss.dtype), str(ev["mse"].dtype)))
    if moved <= 0.0:
        fails.append(("no parameter movement", arm))
    # the residual must be small for the arms that solve a fixed point
    if arm in ("ideal_prospective", "tss_memory_then_prospective") \
            and vals["residual"] > PA.MAX_FIXED_POINT_RESIDUAL:
        fails.append(("fixed point residual", arm, vals["residual"]))

# a second call must NOT recompile: the optimizer transform is a cached object
before = PA.train_step._cache_size() if hasattr(PA.train_step, "_cache_size") \
    else None
for arm in TM.ARMS:
    p = TM.init_params(arm, 101)
    tx = PA.get_tx()
    PA.train_step(arm, tx, p, tx.init(p), *args, cfg)
after = PA.train_step._cache_size() if hasattr(PA.train_step, "_cache_size") \
    else None
print(f"  jit cache size before/after a second seed: {before}/{after}")
if before is not None and after is not None and after != before:
    fails.append(("retrace on a second seed", before, after))

if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("tss production probe OK")
