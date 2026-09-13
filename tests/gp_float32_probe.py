"""R1: production arithmetic in a genuinely x64-DISABLED process.

Run as a SUBPROCESS. Importing any test module that calls
`jax.config.update("jax_enable_x64", True)` would silently promote everything
to float64/complex128, which is exactly the defect this probe exists to catch.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

assert not jax.config.jax_enable_x64, "probe must run without x64"
print("X64_DISABLED_OK")

from jax.scipy.linalg import block_diag                            # noqa: E402
from s5.gp_ssm import GPSSM, init_gp_ssm                           # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                            # noqa: E402

H, SSM, BLOCKS, L = 4, 8, 2, 24
block = SSM // BLOCKS
Lam, _, _, V, _ = make_DPLR_HiPPO(block)
block //= 2
Lam, V = Lam[:block], V[:, :block]
Vc = V.conj().T
Lam = (Lam * jnp.ones((BLOCKS, block))).ravel()
kw = dict(H=H, P=SSM // 2, Lambda_re_init=Lam.real, Lambda_im_init=Lam.imag,
          V=block_diag(*([V] * BLOCKS)), Vinv=block_diag(*([Vc] * BLOCKS)),
          C_init="trunc_standard_normal", discretization="zoh", dt_min=0.001,
          dt_max=0.1, conj_sym=True, clip_eigs=True, bidirectional=False)

mod = init_gp_ssm(mechanism="gp_diagonal", gp_init_scale=0.3, **kw)(step_rescale=1.0)
u = jax.random.normal(jax.random.PRNGKey(0), (L, H))
v = mod.init(jax.random.PRNGKey(1), u)
c = mod.apply(v, method=GPSSM.coefficients)
y = mod.apply(v, u)

expect = {
    "input": (u.dtype, jnp.float32),
    "Lambda_re": (v["params"]["Lambda_re"].dtype, jnp.float32),
    "gp_response_raw": (v["params"]["gp_response_raw"].dtype, jnp.float32),
    "a_bar (carry)": (c["a_bar"].dtype, jnp.complex64),
    "b_bar": (c["b_bar"].dtype, jnp.complex64),
    "d_x": (c["d_x"].dtype, jnp.complex64),
    "output": (y.dtype, jnp.float32),
}
bad = {k: str(got) for k, (got, want) in expect.items() if got != want}
for k, (got, want) in expect.items():
    print(f"  {k:<18} {str(got):>12}  expected {str(want)}")
assert not bad, f"dtype promotion detected: {bad}"
print("DTYPES_OK")

# production parallel vs an INDEPENDENT NumPy sequential reference that does
# not reuse the implementation's scan
ab = onp.asarray(c["a_bar"]); bb = onp.asarray(c["b_bar"])
dx = onp.asarray(c["d_x"])
C_t = onp.asarray(v["params"]["C"][..., 0] + 1j * v["params"]["C"][..., 1])
D = onp.asarray(v["params"]["D"])
h = onp.zeros(ab.shape[0], dtype=onp.complex64)
ref = []
for x in onp.asarray(u):
    h = (ab * h + bb @ x).astype(onp.complex64)
    ref.append(2 * onp.real(C_t @ (h + dx @ x)) + D * x)
ref = onp.array(ref)
rel = onp.max(onp.abs(onp.asarray(y) - ref)) / max(1.0, onp.max(onp.abs(ref)))
print(f"  production parallel vs sequential, complex64: rel = {rel:.3e}")
assert rel < 1e-5, rel
print("PRODUCTION_OK")
