"""Production-dtype probe for the COMBINED response. x64 OFF, own process.

x64 is a process-global JAX setting, so production float32/complex64 coverage
cannot share a process with the float64 reference checks. Executed on the
cluster only, inside the same cap.

Tolerance F32 = 2e-4 RELATIVE, against the same independent dense reference the
float64 checks use, computed in float64 on the host. That is the achievable
agreement for a complex 2x2 matrix exponential and a length-scale ~1 state at
float32 (eps = 1.19e-7) after a 2x2 solve and a scan; it is a dtype-resolution
figure, not a loosened correctness criterion. The ALGEBRA is established at
1e-10 in `test_combined_input_recurrence.py`.
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

from s5.gp_fixed import (mass_block_two_tap, mass_scan_two_tap,    # noqa: E402
                         mass_scan_two_tap_sequential)
from s5.rawat_s5 import RHO_ONLY_PARAM_NAME, init_substrate_ssm    # noqa: E402
# Imported from the NEUTRAL reference module: importing these from the test
# module would execute its `jax.config.update("jax_enable_x64", True)` and this
# probe would measure float64 while reporting float32.
from tests.response_reference import (reference_two_tap as _reference_two_tap,
                                      ssm_kwargs as _ssm_kwargs)  # noqa: E402

# Re-checked AFTER the imports, not only before them: an imported module can
# switch x64 back on, and then this probe would measure float64 silently.
assert not jax.config.read("jax_enable_x64"), \
    "x64 was switched back ON by an import; this probe would measure float64"
assert jnp.zeros(1).dtype == onp.float32, jnp.zeros(1).dtype

F32 = 2e-4
T_IN = 5.0
print(f"  backend={jax.default_backend()}  x64={jax.config.read('jax_enable_x64')}")

fails = []
for (P, H) in ((5, 3), (3, 7), (4, 4)):
    for rho in (0.05, 0.25, 0.75, 0.9999):
        rs = onp.random.RandomState(P * 100 + H)
        a = onp.asarray(-onp.exp(rs.uniform(-3.0, -0.5, P))
                        + 1j * rs.uniform(-2.5, 2.5, P))
        b = onp.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
        got = mass_block_two_tap(jnp.asarray(a), jnp.asarray(b), 5.0,
                                 jnp.ones(P), jnp.full(P, rho), T_IN)
        assert got["A_bar"].dtype == onp.complex64, got["A_bar"].dtype
        assert got["B_plus"].dtype == onp.complex64
        ref = _reference_two_tap(a, b, 5.0, rho, T_IN)
        for k in ("A_bar", "B_bar", "J_in", "B_plus", "B_minus"):
            err = onp.max(onp.abs(onp.asarray(got[k]) - ref[k]))
            scale = max(1.0, float(onp.max(onp.abs(ref[k]))))
            rel = err / scale
            print(f"  P={P} H={H} rho={rho:<7} {k:<8} rel={rel:.3e}")
            if not rel < F32:
                fails.append((P, H, rho, k, rel))

# the scan itself, parallel vs sequential, in production dtypes
rs = onp.random.RandomState(41)
P, H, L = 4, 3, 40
a = onp.asarray(-onp.exp(rs.uniform(-3, -0.5, P)) + 1j * rs.uniform(-2, 2, P))
b = onp.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
c = mass_block_two_tap(jnp.asarray(a), jnp.asarray(b), 5.0, jnp.ones(P),
                       jnp.full(P, 0.6), T_IN)
x = jnp.asarray(rs.randn(L, H).astype(onp.float32))
par = mass_scan_two_tap(c["A_bar"], c["B_plus"], c["B_minus"], x)
seq = mass_scan_two_tap_sequential(c["A_bar"], c["B_plus"], c["B_minus"], x)
rel = float(onp.max(onp.abs(onp.asarray(par) - onp.asarray(seq)))
            / max(1.0, float(onp.max(onp.abs(onp.asarray(seq))))))
print(f"  scan parallel-vs-sequential rel={rel:.3e}")
if not rel < F32:
    fails.append(("scan", rel))

# the executed module really runs in float32, and its response leaf is float32
from flax.traverse_util import flatten_dict                        # noqa: E402
ssm = init_substrate_ssm("gp_rho_prospin", **_ssm_kwargs(4, 6))()
xx = jnp.asarray(onp.random.RandomState(0).randn(12, 6).astype(onp.float32))
params = ssm.init(jax.random.PRNGKey(0), xx)["params"]
y = ssm.apply({"params": params}, xx)
leaf = [v for k, v in flatten_dict(params).items()
        if k[-1] == RHO_ONLY_PARAM_NAME][0]
print(f"  module out {y.dtype}   rho leaf {leaf.dtype} {leaf.shape}")
if y.dtype != onp.float32 or leaf.dtype != onp.float32:
    fails.append(("dtype", str(y.dtype), str(leaf.dtype)))
if not onp.all(onp.isfinite(onp.asarray(y))):
    fails.append(("nonfinite forward",))

if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("combined float32 probe OK")
