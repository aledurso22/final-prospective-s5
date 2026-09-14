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

# This probe ASSERTS that x64 is off; it deliberately does NOT force it off.
# JAX_ENABLE_X64=1 in the environment would therefore make it fail loudly
# rather than silently report float64 numbers as production evidence.
# RUNBOOK: invoke with JAX_ENABLE_X64 unset (or 0).
assert not jax.config.jax_enable_x64, (
    "probe must run without x64; unset JAX_ENABLE_X64 (it is currently on, so "
    "these would be float64 numbers mislabelled as production evidence)")
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


def _rel(out):
    return float(onp.max(onp.abs(onp.asarray(out) - ref))
                 / max(1.0, onp.max(onp.abs(ref))))


# Two DISTINCT measurements, reported separately on purpose.
#
# (1) GATE - implementation correctness. Evaluated at full float32 matmul
#     precision, so it measures OUR algebra and not the backend's default
#     reduced-precision matmul mode. Predeclared tolerance 1e-5.
#
# (2) RECORD - backend characteristic, NOT gated. At the backend default this
#     is TF32 on Ampere, whose ~10-bit mantissa is a property of the hardware
#     path, not of this code. Reported so the number is on the record.
#
# SCOPE OF THESE NUMBERS - read before quoting them anywhere.
# Both are the relative discrepancy of ONE fixture: a length-24 sequence
# through the LINEAR CORE ALONE, against a reference that consumes the SAME
# already-rounded coefficients. They compare two float32 evaluations of the
# same recurrence. They are NOT a bound on the distance from exact arithmetic,
# NOT a bound on error accumulated through a full training run, and they must
# NOT be placed beside training losses or accuracies to argue that some
# observed difference is or is not TF32. To settle that for a training
# comparison, re-run the comparison itself at controlled precision.
#
# Protocol change, stated explicitly rather than applied silently: on
# 2026-09-14 the single default-precision measurement FAILED its 1e-5 gate on
# an RTX 3090 at rel = 7.498e-05. Re-running at highest precision reproduced
# the CPU value 4.748e-08 exactly, showing that THIS FIXTURE is
# precision-sensitive and that the backend matmul mode (TF32 on Ampere)
# accounts for the discrepancy here. That does NOT universally rule out
# scan-order effects: a reduction-order difference of matmul-precision size
# would also vanish under raised precision, and one fixture does not cover
# other shapes or lengths. The gate was therefore RE-SCOPED to the
# precision-controlled measurement -- the originally scoped gate failed, the
# newly scoped one passes -- and the default-precision value is recorded
# alongside it. The tolerance itself was NOT loosened.
rel_default = _rel(y)
with jax.default_matmul_precision("highest"):
    y_hi = mod.apply(v, u)
rel_highest = _rel(y_hi)

print(f"  parallel vs sequential, complex64, HIGHEST matmul: rel = {rel_highest:.3e}")
print(f"  parallel vs sequential, complex64, backend DEFAULT: rel = {rel_default:.3e}")
print(f"  backend={jax.default_backend()} devices={jax.devices()}")
assert rel_highest < 1e-5, (
    f"GATE FAILED at controlled precision: {rel_highest}")
if rel_default >= 1e-5:
    print(f"  NOTE: the backend default matmul mode contributes "
          f"{rel_default:.3e} of deviation from full float32 precision.")
    print("  This is NOT run-to-run noise: the mode is deterministic, and "
          "E2-004 showed four paired runs bit-identical under it.")
    print("  SCOPE: this is one length-24 linear-core fixture against a "
          "reference sharing the same rounded coefficients. It is NOT an "
          "exact-arithmetic error bound and NOT a bound for a full training "
          "run. Do not compare it against training losses or accuracies to "
          "rule TF32 in or out; re-run that comparison at controlled "
          "precision instead.")
print("PRODUCTION_OK")
