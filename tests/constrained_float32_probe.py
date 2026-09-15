"""Production-dtype probe for the constrained learned response.

x64 DISABLED, own process: a float64 fixture cannot establish a float32
property. Checks the ACTUAL dtypes and the float32 gradients for BOTH response
leaves against central differences, at a step chosen to exceed rounding noise.

Predeclared: relative error < 5e-3 at step 1e-2 with an RMS-1 direction.
"""

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
import numpy as onp                                              # noqa: E402

assert not jax.config.read("jax_enable_x64"), "x64 must be OFF in this probe"

# MATCH PRODUCTION. Training runs with --matmul_precision highest; without this
# line the probe ran at the GPU's TF32 default, whose ~10-bit mantissa puts a
# relative error of order 1e-4..1e-3 on the loss. The finite-difference signal
# here is only 2*h*g/loss ~ 6e-5, i.e. the SAME ORDER as that noise, which is
# how the first run produced a 17% discrepancy against a correct gradient. The
# other float32 probes in this repository already scope their measurement this
# way; this one did not.
jax.config.update("jax_default_matmul_precision", "highest")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax.scipy.linalg import block_diag                          # noqa: E402
from s5.rawat_s5 import RESPONSE_PARAM_NAMES, init_substrate_ssm  # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                          # noqa: E402

#: Tolerance UNCHANGED at 5e-3. The step is declared from the MEASURED float32
#: U-curve at highest precision (experiments/gp/constrained_numerics_probe.py):
#:
#:   step      1e-1     3e-2     1e-2     3e-3     1e-3     1e-4
#:   gamma   5.9e-2   5.4e-3   2.3e-3   3.4e-3   3.9e-2   5.8e-2
#:   rho     1.0e-2   1.0e-3   1.6e-3   7.2e-3   7.2e-3   8.8e-2
#:
#: truncation dominating above, rounding below, minimum near 1e-2 where BOTH
#: leaves sit under the tolerance. All three central steps are reported so a
#: future failure is diagnosable rather than just red.
TOL, STEP = 5e-3, 1e-2
REPORT_STEPS = (3e-2, 1e-2, 3e-3)
H, ssm, blocks, L = 4, 8, 2, 16
blk = ssm // blocks
Lam, _, _, V, _ = make_DPLR_HiPPO(blk)
blk //= 2
P = ssm // 2
Lam, V = Lam[:blk], V[:, :blk]
Vc = V.conj().T
Lam = (Lam * jnp.ones((blocks, blk))).ravel()
kw = dict(H=H, P=P, Lambda_re_init=Lam.real, Lambda_im_init=Lam.imag,
          V=block_diag(*([V] * blocks)), Vinv=block_diag(*([Vc] * blocks)),
          C_init="trunc_standard_normal", discretization="zoh",
          dt_min=0.001, dt_max=0.1, conj_sym=True, bidirectional=False)

mod = init_substrate_ssm("gp_learned_response", **kw)()
x = jnp.asarray(onp.random.RandomState(0).randn(L, H), dtype=jnp.float32)
v = mod.init(jax.random.PRNGKey(0), x)
p = v["params"]
print(f"  backend={jax.default_backend()}  input {x.dtype}")
for name in RESPONSE_PARAM_NAMES:
    print(f"  {name:22s} dtype {p[name].dtype} shape {p[name].shape}")
    assert p[name].dtype == onp.float32 and p[name].shape == (P,)

g_n, rho = mod.apply(v, method=lambda m: m.learned_response())
mu = mod.apply(v, method=lambda m: m.derived_mass())
print(f"  gamma_n {g_n.dtype} in [{float(g_n.min()):.4f},{float(g_n.max()):.4f}]"
      f"  rho in [{float(rho.min()):.4f},{float(rho.max()):.4f}]"
      f"  mu in [{float(mu.min()):.4f},{float(mu.max()):.4f}]")
assert g_n.dtype == onp.float32 and rho.dtype == onp.float32
assert abs(float(g_n.max()) - 1.0) < 1e-6 and abs(float(rho.max()) - 0.75) < 1e-6
assert abs(float(mu.max()) - 3.75) < 1e-5, float(mu.max())

y = mod.apply(v, x)
print(f"  output {y.dtype}")
assert y.dtype == onp.float32 and bool(onp.isfinite(onp.asarray(y)).all())


def loss(params):
    return jnp.sum(mod.apply({"params": params}, x) ** 2)


g = jax.grad(loss)(p)
rs = onp.random.RandomState(1)
worst = 0.0
for name in RESPONSE_PARAM_NAMES:
    gi = onp.asarray(g[name])
    assert onp.isfinite(gi).all() and onp.max(onp.abs(gi)) > 0.0, name
    d = rs.randn(*gi.shape)
    d = jnp.asarray(d / d.std(), dtype=jnp.float32)
    ana = float(jnp.sum(g[name] * d))
    rels = {}
    for h in REPORT_STEPS:
        plus = dict(p); plus[name] = p[name] + h * d
        minus = dict(p); minus[name] = p[name] - h * d
        fd = (float(loss(plus)) - float(loss(minus))) / (2 * h)
        rels[h] = (fd, abs(ana - fd) / max(abs(fd), 1e-8))
    rel = rels[STEP][1]
    worst = max(worst, rel)
    print(f"  {name:22s} analytic {ana:12.5f}  "
          + "  ".join(f"h={h:.0e} fd={v[0]:.5f} rel={v[1]:.2e}"
                      for h, v in rels.items()))
    assert rel < TOL, f"GATE FAILED for {name} at step {STEP}: {rel}"
print(f"  worst relative error {worst:.3e} (tol {TOL:.0e})")
print("CONSTRAINED_F32_OK")
