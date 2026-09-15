"""Production-dtype probe for the LEARNED TIMESCALE arms. x64 OFF, own process.

x64 is a process-global JAX setting, so production float32/complex64 coverage
cannot share a process with the float64 reference checks.

Tolerance F32 = 2e-4 RELATIVE against the same independent dense reference the
float64 checks use, computed in float64 on the host. That is a dtype-resolution
figure for a complex 2x2 matrix exponential at float32 (eps = 1.19e-7), not a
loosened correctness criterion: the ALGEBRA is established at 1e-10 in
`test_learned_timescale.py`.

The declared numerical corners T in {0.05, 500} and rho in {0.01, 0.9999} are
exercised HERE too, in production dtypes, because a corner that is finite in
float64 and not in float32 would be a float32 failure.
"""

import math
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as onp                                               # noqa: E402

assert not jax.config.read("jax_enable_x64"), "x64 must be OFF in this probe"
jax.config.update("jax_default_matmul_precision", "highest")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flax.traverse_util import flatten_dict                        # noqa: E402
from s5.gp_fixed import mass_block_zoh, mass_scan                  # noqa: E402
from s5.rawat_s5 import (RHO_INIT_TIMESCALE, RHO_ONLY_PARAM_NAME,  # noqa: E402
                         T_BOUNDS, T_ONLY_PARAM_NAME, T_REFERENCE,
                         init_substrate_ssm)
# Imported from the NEUTRAL reference module. Importing these from the test
# module would execute its `jax.config.update("jax_enable_x64", True)` and this
# probe would measure float64 while reporting float32 - which is exactly what
# happened on the first cluster run.
from tests.response_reference import (reference_block as _reference_block,
                                      ssm_kwargs as _ssm_kwargs)  # noqa: E402

# Re-checked AFTER the imports: the first cluster run passed the pre-import
# assertion and still ran in float64, because an imported module had switched
# x64 back on. The invariant has to hold at the point of measurement.
assert not jax.config.read("jax_enable_x64"), \
    "x64 was switched back ON by an import; this probe would measure float64"
assert jnp.zeros(1).dtype == onp.float32, jnp.zeros(1).dtype
assert jnp.zeros(1, dtype=jnp.complex64).dtype == onp.complex64

F32 = 2e-4
print(f"  backend={jax.default_backend()}  "
      f"x64={jax.config.read('jax_enable_x64')}")

fails = []

# 1. per-mode T coefficients against the independent dense reference
for (P, H) in ((5, 3), (3, 7), (4, 4)):
    rs = onp.random.RandomState(P * 31 + H)
    a = onp.asarray(-onp.exp(rs.uniform(-3.0, -0.5, P))
                    + 1j * rs.uniform(-2.5, 2.5, P))
    b = onp.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    T = onp.exp(rs.uniform(math.log(0.2), math.log(50.0), P))
    rho = rs.uniform(0.05, 0.95, P)
    got = mass_block_zoh(jnp.asarray(a), jnp.asarray(b), jnp.asarray(T),
                         jnp.ones(P), jnp.asarray(rho))
    if got["A_bar"].dtype != onp.complex64:
        fails.append(("dtype A_bar", str(got["A_bar"].dtype)))
    ref = _reference_block(a, b, T, rho)
    for k in ("A", "B", "A_bar", "B_bar"):
        err = onp.max(onp.abs(onp.asarray(got[k]) - ref[k]))
        scale = max(1.0, float(onp.max(onp.abs(ref[k]))))
        rel = err / scale
        print(f"  P={P} H={H} {k:<6} rel={rel:.3e}")
        if not rel < F32:
            fails.append((P, H, k, rel))

# 2. the declared corners are finite in value AND derivative, in float32
for T in (T_BOUNDS[0], 1.0, T_REFERENCE, T_BOUNDS[1]):
    for rho in (0.01, 0.25, 0.75, 0.9999):
        rs = onp.random.RandomState(17)
        P, H, L = 5, 3, 16
        a = onp.asarray(-onp.exp(rs.uniform(-3.0, -0.5, P))
                        + 1j * rs.uniform(-2.5, 2.5, P))
        b = onp.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
        x = jnp.asarray(rs.randn(L, H).astype(onp.float32))

        def out(eta, a=a, b=b, x=x, rho=rho, P=P):
            TT = T_REFERENCE * jnp.exp(eta)
            d = mass_block_zoh(jnp.asarray(a), jnp.asarray(b), TT,
                               jnp.ones(P), jnp.full(P, rho))
            return jnp.real(jnp.sum(mass_scan(d["A_bar"], d["B_bar"],
                                              x)[..., 0]))

        eta0 = jnp.full(P, math.log(T / T_REFERENCE), dtype=jnp.float32)
        v = float(out(eta0))
        g = onp.asarray(jax.grad(out)(eta0))
        ok = onp.isfinite(v) and onp.all(onp.isfinite(g))
        print(f"  corner T={T:<8g} rho={rho:<7} finite={ok} "
              f"|g|max={onp.max(onp.abs(g)):.3e}")
        if not ok:
            fails.append(("nonfinite corner", T, rho, v))

# 3. the executed module: float32 leaves, float32 output, declared init
for arm in ("gp_rho_T", "gp_rho_T_fixed"):
    ssm = init_substrate_ssm(arm, **_ssm_kwargs(4, 6))()
    xx = jnp.asarray(onp.random.RandomState(0).randn(12, 6).astype(onp.float32))
    params = ssm.init(jax.random.PRNGKey(0), xx)["params"]
    flat = flatten_dict(params)
    rho_leaf = [v for k, v in flat.items() if k[-1] == RHO_ONLY_PARAM_NAME][0]
    eta_leaf = [v for k, v in flat.items() if k[-1] == T_ONLY_PARAM_NAME][0]
    y = ssm.apply({"params": params}, xx)
    T = onp.asarray(ssm.apply({"params": params},
                              method=lambda m: m.response_timescale()))
    print(f"  {arm}: out {y.dtype}  rho {rho_leaf.dtype}  eta {eta_leaf.dtype}"
          f"  T={T[0]:.6f}")
    if y.dtype != onp.float32 or rho_leaf.dtype != onp.float32 \
            or eta_leaf.dtype != onp.float32:
        fails.append(("dtype", arm, str(y.dtype), str(rho_leaf.dtype),
                      str(eta_leaf.dtype)))
    if not onp.all(onp.isfinite(onp.asarray(y))):
        fails.append(("nonfinite forward", arm))
    if abs(float(T[0]) - T_REFERENCE) > 1e-4:
        fails.append(("T init", arm, float(T[0])))
    rho0 = float(onp.exp(onp.asarray(rho_leaf)[0]))
    if abs(rho0 - RHO_INIT_TIMESCALE) > 1e-5:
        fails.append(("rho init", arm, rho0))

# 4. the two arms are the same function at initialization, in float32
s_f = init_substrate_ssm("gp_rho_T_fixed", **_ssm_kwargs(4, 6))()
s_l = init_substrate_ssm("gp_rho_T", **_ssm_kwargs(4, 6))()
xx = jnp.asarray(onp.random.RandomState(1).randn(12, 6).astype(onp.float32))
p_f = s_f.init(jax.random.PRNGKey(3), xx)["params"]
p_l = s_l.init(jax.random.PRNGKey(3), xx)["params"]
d = float(onp.max(onp.abs(onp.asarray(s_f.apply({"params": p_f}, xx))
                          - onp.asarray(s_l.apply({"params": p_l}, xx)))))
print(f"  generalized arms max|dy| at init = {d:.3e}")
if d != 0.0:
    fails.append(("arms differ at init", d))

if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("timescale float32 probe OK")
