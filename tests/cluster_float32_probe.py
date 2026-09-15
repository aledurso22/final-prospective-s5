"""Production-dtype probe for the fixed GP arms and the Rawat port.

Runs with x64 DISABLED, in its own process. A float64 fixture cannot establish
a float32 property, and importing anything that enables x64 first would
silently invalidate the check - so this file is never imported by the suite;
`tests/test_cluster_gp.py` runs it as a subprocess.

It reports, separately and honestly:
  * the ACTUAL dtype of coefficients, parameters and executed state;
  * the relative difference between the float32 execution and a float64
    reference of the same recurrence, per arm.

Predeclared gate: relative difference < 1e-4 at controlled matmul precision.
That number is a per-fixture agreement of two evaluations of the same
recurrence at different precisions. It is NOT an exact-arithmetic error bound,
NOT a bound for a full training run, and must NOT be compared against training
losses or accuracies to rule a backend precision mode in or out.
"""

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax                                                       # noqa: E402
import jax.numpy as np                                           # noqa: E402
import numpy as onp                                              # noqa: E402

assert not jax.config.read("jax_enable_x64"), "x64 must be OFF in this probe"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5.gp_fixed import (fixed_m0_coefficients, mass_block_zoh,   # noqa: E402
                         mass_scan, mass_scan_sequential)
from s5.physical_coefficients import SYMMETRIC_REFERENCE as R     # noqa: E402

GATE = 1e-4
P, H, L = 8, 4, 24

rs = onp.random.RandomState(0)
a64 = (-onp.exp(rs.uniform(-4, -0.3, P)) + 1j * rs.uniform(-3, 3, P))
b64 = rs.randn(P, H) + 1j * rs.randn(P, H)
x64 = rs.randn(L, H)

a32 = np.asarray(a64.astype(onp.complex64))
b32 = np.asarray(b64.astype(onp.complex64))
x32 = np.asarray(x64.astype(onp.float32))

print(f"  backend={jax.default_backend()} devices={jax.devices()}")
for name, arr in (("a", a32), ("b", b32), ("x", x32)):
    print(f"  input {name:<2} dtype {arr.dtype}")
    assert arr.dtype in (onp.complex64, onp.float32), arr.dtype


def numpy_reference_m0(a, b, x):
    W = 1.0 - R.T * a
    F = a / W
    d_x = (R.T * b) / W[:, None]
    B_h = b / (W ** 2)[:, None]
    a_bar = onp.exp(F)
    with onp.errstate(divide="ignore", invalid="ignore"):
        phi = onp.where(onp.abs(F) < 1e-8, 1.0 + F / 2.0,
                        (onp.expm1(F.real) * onp.cos(F.imag)
                         - 2 * onp.sin(F.imag / 2) ** 2
                         + 1j * onp.exp(F.real) * onp.sin(F.imag)) / F)
    b_bar = phi[:, None] * B_h
    h = onp.zeros(a.shape, dtype=a.dtype)
    out = []
    for k in range(x.shape[0]):
        h = a_bar * h + b_bar @ x[k]
        out.append(h + d_x @ x[k])
    return onp.stack(out)


def rel(u, v):
    u, v = onp.asarray(u), onp.asarray(v)
    return float(onp.max(onp.abs(u - v)) / max(onp.max(onp.abs(v)), 1e-30))


with jax.default_matmul_precision("highest"):
    c32 = fixed_m0_coefficients(a32, b32, R.T, R.gamma)
    print(f"  gp_fixed_m0 coefficient dtype {c32['a_bar'].dtype}, "
          f"d_x {c32['d_x'].dtype}")
    assert c32["a_bar"].dtype == onp.complex64

    h = np.zeros((P,), dtype=np.complex64)
    out = []
    for k in range(L):
        h = c32["a_bar"] * h + c32["b_bar"] @ x32[k]
        out.append(h + c32["d_x"] @ x32[k])
    s32 = np.stack(out)
    print(f"  gp_fixed_m0 executed state dtype {s32.dtype}")
    r_m0 = rel(s32, numpy_reference_m0(a64, b64, x64))
    print(f"  gp_fixed_m0  float32 vs float64 reference: rel = {r_m0:.3e}")

    z32 = mass_block_zoh(a32, b32, R.T, R.gamma, R.rho)
    print(f"  gp_fixed_mass coefficient dtype {z32['A_bar'].dtype}")
    assert z32["A_bar"].dtype == onp.complex64
    par = mass_scan(z32["A_bar"], z32["B_bar"], x32)
    seq = mass_scan_sequential(z32["A_bar"], z32["B_bar"], x32)
    print(f"  gp_fixed_mass executed state dtype {par.dtype}")
    r_scan = rel(par, seq)
    print(f"  gp_fixed_mass block scan vs sequential: rel = {r_scan:.3e}")

    # rho = 1 must still reduce to the ordinary response IN float32
    z1 = mass_block_zoh(a32, b32, R.T, 1.0, 1.0)
    s_mass = mass_scan(z1["A_bar"], z1["B_bar"], x32)[..., 0]
    a_bar = np.exp(a32)
    F = a32
    phi = np.where(np.abs(F) < 1e-6, 1.0 + F / 2.0, (np.exp(F) - 1.0) / F)
    b_bar = phi[:, None] * b32
    hh = np.zeros((P,), dtype=np.complex64)
    ref = []
    for k in range(L):
        hh = a_bar * hh + b_bar @ x32[k]
        ref.append(hh)
    r_rho1 = rel(s_mass, np.stack(ref))
    print(f"  rho=1 reduction in float32:              rel = {r_rho1:.3e}")

worst = max(r_m0, r_scan, r_rho1)
assert worst < GATE, f"GATE FAILED: worst relative difference {worst:.3e}"
print(f"  worst = {worst:.3e} (gate {GATE:.0e})")
print("CLUSTER_FLOAT32_OK")
