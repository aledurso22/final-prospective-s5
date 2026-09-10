"""The general port-access criterion for learning-faithful prospective realizations.

Setting. A prospective realization is nominally exact but its first-order
parameter expansion carries hidden physical directions:

    w = z + sum_a theta_a U_a b + O(||theta||^2),        m = R b

`w` is what a local causal corrector can see; `m` is whatever side access the
physics exposes. The learning-faithful estimate must remove every U_a b term
using only `m`.

Criterion. A correction z_hat = w - sum_a theta_a L_a m reproduces the intended
first-order tangent for ALL b if and only if

    ker R  subset of  intersection_a ker U_a

equivalently, there exist L_a with  L_a R = U_a.

If the condition fails there is a b in ker R with U_a b != 0 for some a: two
physical states indistinguishable through the port yet requiring different
learning corrections. No amount of local capacity can separate them, because
the information is absent, not merely unextracted.

Under the stated instantaneous-measurement setup the number of learning-
relevant ports is

    p_learning = rank([U_1; ...; U_d]).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _null_space(R, rtol=1e-10):
    R = np.atleast_2d(np.asarray(R, dtype=float))
    _, sv, Vh = np.linalg.svd(R)
    if sv.size == 0:
        return np.eye(R.shape[1])
    tol = max(R.shape) * np.finfo(float).eps * (sv[0] if sv[0] > 0 else 1.0)
    tol = max(tol, rtol * (sv[0] if sv[0] > 0 else 1.0))
    rank = int((sv > tol).sum())
    return Vh[rank:].T                      # columns span ker R


@dataclass
class PortAccessResult:
    holds: bool
    rank_R: int
    p_learning: int
    L: Optional[List[np.ndarray]]
    residuals: List[float]
    null_space: np.ndarray = field(repr=False, default=None)
    witness_b: Optional[np.ndarray] = None
    witness_index: Optional[int] = None
    witness_Ub: Optional[np.ndarray] = None

    def summary(self):
        d = dict(holds=bool(self.holds), rank_R=int(self.rank_R),
                 p_learning=int(self.p_learning),
                 max_residual=float(max(self.residuals)) if self.residuals else 0.0)
        if not self.holds:
            d.update(witness_b=self.witness_b.tolist(),
                     witness_parameter_index=int(self.witness_index),
                     witness_Ub=np.asarray(self.witness_Ub).ravel().tolist())
        return d


def check_port_access(U_list, R, tol=1e-9):
    """Test ker R subset of intersection_a ker U_a, and build the correctors.

    Args:
        U_list: list of (n_w, n_b) matrices, one per learned parameter.
        R: (n_m, n_b) measurement matrix, m = R b.

    Returns PortAccessResult with, on success, L_a satisfying L_a R = U_a; on
    failure, an explicit b in ker R with U_a b != 0.
    """
    Us = [np.atleast_2d(np.asarray(U, dtype=float)) for U in U_list]
    R = np.atleast_2d(np.asarray(R, dtype=float))
    N = _null_space(R)
    rank_R = int(np.linalg.matrix_rank(R))

    stacked = np.vstack(Us) if Us else np.zeros((0, R.shape[1]))
    p_learning = int(np.linalg.matrix_rank(stacked)) if stacked.size else 0

    # candidate correctors via the pseudo-inverse, then VERIFY L_a R == U_a
    Rp = np.linalg.pinv(R)
    L = [U @ Rp for U in Us]
    residuals = [float(np.max(np.abs(La @ R - U))) for La, U in zip(L, Us)]

    holds = all(r <= tol for r in residuals)
    if holds:
        return PortAccessResult(True, rank_R, p_learning, L, residuals, N)

    # failure: exhibit a concrete null-space direction the port cannot see
    witness_b, witness_idx, witness_Ub = None, None, None
    if N.size:
        best = -1.0
        for a, U in enumerate(Us):
            UN = U @ N                                  # (n_w, dim ker R)
            for j in range(UN.shape[1]):
                mag = float(np.linalg.norm(UN[:, j]))
                if mag > best:
                    best, witness_idx = mag, a
                    witness_b, witness_Ub = N[:, j].copy(), UN[:, j].copy()
    return PortAccessResult(False, rank_R, p_learning, None, residuals, N,
                            witness_b, witness_idx, witness_Ub)


def corrected_tangent(U_list, R, L_list, thetas, b):
    """First-order tangent of z_hat = w - sum_a theta_a L_a m, minus the intended.

    Returns the residual contamination sum_a theta_a (U_a - L_a R) b, which is
    exactly zero when the access criterion holds.
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    out = None
    for th, U, L in zip(thetas, U_list, L_list):
        term = th * ((np.atleast_2d(U) - np.atleast_2d(L) @ R) @ b)
        out = term if out is None else out + term
    return out


def minimal_port(U_list, tol=1e-9):
    """The smallest measurement that makes learning faithful.

    Taking R = stack(U_a) always satisfies the criterion, and its row rank is
    p_learning, so no smaller instantaneous port suffices.
    """
    Us = [np.atleast_2d(np.asarray(U, dtype=float)) for U in U_list]
    stacked = np.vstack(Us)
    u, sv, Vh = np.linalg.svd(stacked, full_matrices=False)
    rank = int((sv > max(stacked.shape) * np.finfo(float).eps * sv[0]).sum()) \
        if sv.size and sv[0] > 0 else 0
    return Vh[:rank], rank
