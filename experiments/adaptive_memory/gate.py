"""The positive source gate, shared by the three new rules.

    l(k, v) = (H32 u)_k + (H8 b)_v ,   a_psi(k, v) = 2 sigmoid(l(k, v))

`H32` and `H8` are FIXED orthonormal zero-sum contrast matrices, so the gate
carries no constant logit direction. A constant multiplier of `R` is absorbable
into the response coefficients, and learning one here as well would make the
parameterization unidentifiable; removing the constant directions is what
prevents that. `u = b = 0` at initialization, so `a = 1` on every write.

38 trainable reals, no intercept and no event coefficient.

INFORMATION RESTRICTIONS, which the exogenous convex-objective argument needs:
only the CURRENT observed key and value identities enter. A query has `m = 0`
and no value source. The gate never sees a query label, a future key, a
target/distractor identity, a position, an error norm or the memory state; in
particular it cannot know which keys will be queried later.

The bounded range `(0, 2)` is a declared design choice, not a theorem, and the
additive logit gauge is fixed rather than the post-training mean being claimed
to stay at one.
"""

import jax
import jax.numpy as jnp
import numpy as onp


def contrast_matrix(n):
    """`(n, n-1)` Helmert contrasts: orthonormal columns, each summing to zero.

    Column `j` has `1/sqrt((j+1)(j+2))` in rows `i <= j`,
    `-(j+1)/sqrt((j+1)(j+2))` in row `j+1`, and zero below. Deterministic, so
    the convention is reproducible rather than an arbitrary basis.
    """
    H = onp.zeros((n, n - 1), dtype=onp.float64)
    for j in range(n - 1):
        c = 1.0 / onp.sqrt((j + 1.0) * (j + 2.0))
        H[:j + 1, j] = c
        H[j + 1, j] = -(j + 1.0) * c
    return H


def check_contrast(H, tol=1e-12):
    """`H^T H = I` and `1^T H = 0`, returned as measured residuals."""
    n, m = H.shape
    return dict(orthonormal_residual=float(
        onp.max(onp.abs(H.T @ H - onp.eye(m)))),
        zero_sum_residual=float(onp.max(onp.abs(H.sum(axis=0)))),
        shape=[int(n), int(m)], passed=bool(
            onp.max(onp.abs(H.T @ H - onp.eye(m))) < tol
            and onp.max(onp.abs(H.sum(axis=0))) < tol))


def source_weight_table(params, H32, H8):
    """`a_psi(k, v)` for every `(key, value)` pair: shape `(32, 8)`.

    Computed ONCE per forward call, outside the temporal scan and outside the
    per-example vmap, then indexed causally by the observed token. The table
    holds a response for each POSSIBLE token, not an observation of a future
    one.
    """
    lk = H32 @ params["gate_u"]                      # (32,)
    lv = H8 @ params["gate_b"]                       # (8,)
    return 2.0 * jax.nn.sigmoid(lk[:, None] + lv[None, :])
