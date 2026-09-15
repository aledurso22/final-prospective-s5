"""Production-dtype probe for one adaptive arm. x64 DISABLED, own process.

A float64 fixture cannot establish a float32 property, and importing anything
that enables x64 first would silently invalidate the check.

Reports the ACTUAL dtypes of the modulation, coefficients and executed state,
and checks a float32 gradient THROUGH the adaptation against a central
difference. Predeclared: relative error < 5e-3 at an RMS-1 direction and step
1e-2, which is the float32-resolvable regime for this function.
"""

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
import numpy as onp                                              # noqa: E402

assert not jax.config.read("jax_enable_x64"), "x64 must be OFF in this probe"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax.scipy.linalg import block_diag                          # noqa: E402
from s5.rawat_s5 import init_substrate_ssm                       # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                          # noqa: E402

GRAD_TOL, STEP = 5e-3, 1e-2
arm = sys.argv[1]
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

mod = init_substrate_ssm(arm, **kw)()
x = jnp.asarray(onp.random.RandomState(0).randn(L, H), dtype=jnp.float32)
v = mod.init(jax.random.PRNGKey(0), x)
print(f"  arm={arm} backend={jax.default_backend()}")
print(f"  input dtype {x.dtype}")
assert x.dtype == onp.float32

y = mod.apply(v, x)
print(f"  output dtype {y.dtype}")
assert y.dtype == onp.float32, y.dtype
assert bool(onp.isfinite(onp.asarray(y)).all()), "non-finite output"

if arm in ("gp_adaptive_mass", "ordinary_adaptive", "gp_frozen_adaptive"):
    from s5 import adaptive_circuit as AC

    def read(m):
        a, b = m.clock_absorbed()
        return a, b
    a, b = mod.apply(v, method=read)
    q = AC.conductance_from_input(b, x)
    c = AC.coefficients_from_conductance(q)
    print(f"  a {a.dtype}  b {b.dtype}  q {q.dtype}  T {c['T'].dtype}  "
          f"M {c['M'].dtype}")
    assert q.dtype == onp.float32 and a.dtype == onp.complex64
    qn = onp.asarray(q)
    print(f"  q in [{qn.min():.5f}, {qn.max():.5f}] of bound "
          f"{AC.MOD_SCALE * AC.G_D0:.5f}")
    assert qn.min() >= 0.0 and qn.max() < AC.MOD_SCALE * AC.G_D0 + 1e-6
    A, Bx, dj, _ = AC.adaptive_generator(a, b, q)
    A_bar, _ = AC.adaptive_zoh(A, Bx)
    print(f"  A_bar {A_bar.dtype}  max|A_bar| {float(jnp.max(jnp.abs(A_bar))):.6f}")
    assert A_bar.dtype == onp.complex64


def loss(xx):
    return jnp.sum(mod.apply(v, xx) ** 2)


g = jax.grad(loss)(x)
assert bool(onp.isfinite(onp.asarray(g)).all()), "non-finite gradient"
d = onp.random.RandomState(1).randn(*x.shape)
d = jnp.asarray(d / d.std(), dtype=jnp.float32)
ana = float(jnp.sum(g * d))
fd = (float(loss(x + STEP * d)) - float(loss(x - STEP * d))) / (2 * STEP)
rel = abs(ana - fd) / max(abs(fd), 1e-8)
print(f"  float32 grad through adaptation: analytic {ana:.6f} fd {fd:.6f} "
      f"rel {rel:.3e} (tol {GRAD_TOL:.0e})")
assert rel < GRAD_TOL, f"GATE FAILED: {rel}"
print("ADAPTIVE_F32_OK")
