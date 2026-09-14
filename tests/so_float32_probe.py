"""Second-order prototype in genuinely x64-DISABLED production arithmetic.

Run as a SUBPROCESS: importing a test module that enables x64 would silently
promote everything and defeat the purpose. Invoke with JAX_ENABLE_X64 unset.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

assert not jax.config.jax_enable_x64, "probe must run without x64"
print("SO_X64_DISABLED_OK")

from experiments.gp.cue_recall_runner import ssm_kwargs                # noqa: E402
from s5.gp_second_order import (SecondOrderGPSSM, init_second_order_ssm,  # noqa: E402
                                second_order_readout, second_order_scan,
                                second_order_sequential)

kw = ssm_kwargs()
mod = init_second_order_ssm(gp_init_scale=0.3, mu_ratio_init=0.5,
                            **kw)(step_rescale=1.0)
u = jax.random.normal(jax.random.PRNGKey(0), (32, kw["H"]))
v = mod.init(jax.random.PRNGKey(1), u)
c = mod.apply(v, method=SecondOrderGPSSM.coefficients)
y = mod.apply(v, u)

expect = {"input": (u.dtype, jnp.float32),
          "so_response_raw": (v["params"]["so_response_raw"].dtype, jnp.float32),
          "so_mu_ratio_raw": (v["params"]["so_mu_ratio_raw"].dtype, jnp.float32),
          "A_bar": (c["A_bar"].dtype, jnp.complex64),
          "B_bar": (c["B_bar"].dtype, jnp.complex64),
          "output": (y.dtype, jnp.float32)}
bad = {k: str(g) for k, (g, w) in expect.items() if g != w}
for k, (g, w) in expect.items():
    print(f"  {k:<18} {str(g):>12}  expected {str(w)}")
assert not bad, f"dtype promotion: {bad}"
print("SO_DTYPES_OK")

C_t = v["params"]["C"][..., 0] + 1j * v["params"]["C"][..., 1]
par = second_order_readout(second_order_scan(c["A_bar"], c["B_bar"], u),
                           C_t, v["params"]["D"], u, True)
seq = second_order_readout(second_order_sequential(c["A_bar"], c["B_bar"], u),
                           C_t, v["params"]["D"], u, True)
rel = float(onp.max(onp.abs(onp.asarray(par) - onp.asarray(seq)))
            / max(1.0, onp.max(onp.abs(onp.asarray(seq)))))
print(f"  block scan vs sequential, complex64: rel = {rel:.3e}")
print(f"  backend={jax.default_backend()} devices={jax.devices()}")
assert rel < 1e-4, rel          # predeclared GATE_SO_F32
print("SO_PRODUCTION_OK")
