"""Closure identification: how many modes does a residual trajectory need?

    fit_closure(residual_trajectories, dt, order=k)

Fits a discrete AR(k) model to recorded residuals and reports the diagnostics
that decide whether an extra mode is JUSTIFIED, not merely better-fitting.

Why AR(k). If the residual system is the coupled pair

    r' = -gamma r + z b,      b' = -a b + z r

then eliminating the hidden state b gives a second-order ODE in r alone, whose
exact discrete-time marginal is AR(2). One mode -> AR(1), two modes -> AR(2).
So AR order IS closure order, and the AR roots map back to the continuous
eigenvalues by  lambda = log(root)/dt. That makes the fitted poles directly
comparable with the true generator eigenvalues.

Designed to be reused on S5/RQF residuals later: it only ever sees an array of
residual trajectories and a timestep.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ClosureFit:
    order: int
    coeffs: np.ndarray
    discrete_roots: np.ndarray
    continuous_poles: np.ndarray
    n_params: int
    n_obs: int
    train_mse: float
    onestep_mse: float
    rollout_mse: float
    aic: float
    bic: float
    ljung_box: float
    ljung_box_dof: int
    whiteness_pass: bool
    innovations: np.ndarray = field(repr=False, default=None)

    def summary(self):
        return dict(
            order=self.order,
            n_params=self.n_params,
            continuous_poles=[complex(p) for p in self.continuous_poles],
            onestep_mse=self.onestep_mse,
            rollout_mse=self.rollout_mse,
            aic=self.aic,
            bic=self.bic,
            ljung_box=self.ljung_box,
            whiteness_pass=bool(self.whiteness_pass),
        )


def _design(series, order):
    """Lagged design matrix for an AR(order) fit."""
    n = len(series)
    X = np.column_stack([series[order - 1 - i: n - 1 - i] for i in range(order)])
    y = series[order:]
    return X, y


def _ljung_box(resid, n_lags=12):
    """Ljung-Box Q statistic. Large Q means leftover temporal structure."""
    resid = np.asarray(resid, dtype=float)
    n = len(resid)
    resid = resid - resid.mean()
    denom = np.sum(resid ** 2)
    if denom == 0 or n <= n_lags + 1:
        return 0.0, n_lags
    q = 0.0
    for k in range(1, n_lags + 1):
        rk = np.sum(resid[k:] * resid[:-k]) / denom
        q += rk ** 2 / (n - k)
    return float(n * (n + 2) * q), n_lags


# chi-square 95% critical values, dof 1..24 (avoids a scipy.stats dependency)
_CHI2_95 = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 6: 12.59, 7: 14.07,
            8: 15.51, 9: 16.92, 10: 18.31, 11: 19.68, 12: 21.03, 13: 22.36,
            14: 23.68, 15: 25.00, 16: 26.30, 17: 27.59, 18: 28.87, 19: 30.14,
            20: 31.41, 21: 32.67, 22: 33.92, 23: 35.17, 24: 36.42}


def fit_closure(residual_trajectories, dt, order=1, holdout_frac=0.3,
                rollout_steps=None, lags=12):
    """Fit an order-k closure to residual data.

    Args:
        residual_trajectories: (T,) or (T, 1) array, or a list of such.
        dt: timestep, used to convert discrete roots to continuous poles.
        order: closure order k (1 = ideal one-pole TSS).
        holdout_frac: tail fraction reserved for held-out evaluation.
        rollout_steps: free-run horizon on held-out data (default: all of it).

    Returns:
        ClosureFit, with held-out rollout error, fitted poles, parameter count,
        AIC/BIC and a Ljung-Box whiteness test on the innovations.
    """
    if isinstance(residual_trajectories, (list, tuple)):
        trajs = [np.asarray(t, dtype=float).ravel() for t in residual_trajectories]
    else:
        arr = np.asarray(residual_trajectories, dtype=float)
        trajs = [arr.ravel()] if arr.ndim == 1 else [arr[:, 0]]

    Xs, ys, holds = [], [], []
    for tr in trajs:
        cut = int(len(tr) * (1.0 - holdout_frac))
        train, hold = tr[:cut], tr[cut:]
        if len(train) <= order + 2:
            raise ValueError("trajectory too short for this order")
        X, y = _design(train, order)
        Xs.append(X); ys.append(y); holds.append(hold)

    X = np.vstack(Xs); y = np.concatenate(ys)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

    innov = y - X @ coeffs
    train_mse = float(np.mean(innov ** 2))

    # one-step-ahead error on held-out data
    one_step = []
    for hold in holds:
        if len(hold) > order:
            Xh, yh = _design(hold, order)
            one_step.append(yh - Xh @ coeffs)
    onestep_mse = float(np.mean(np.concatenate(one_step) ** 2)) if one_step \
        else float("nan")

    # free-running rollout on held-out data: the real test of a closure
    rollouts = []
    for hold in holds:
        if len(hold) <= order:
            continue
        horizon = rollout_steps or (len(hold) - order)
        state = list(hold[:order][::-1])          # most recent first
        pred = []
        for _ in range(horizon):
            nxt = float(np.dot(coeffs, state))
            pred.append(nxt)
            state = [nxt] + state[:-1]
        rollouts.append(np.asarray(pred) - hold[order:order + horizon])
    rollout_mse = float(np.mean(np.concatenate(rollouts) ** 2)) if rollouts \
        else float("nan")

    roots = np.roots(np.concatenate(([1.0], -coeffs)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cont = np.log(roots.astype(complex)) / dt

    n = len(y)
    k = order
    sigma2 = max(train_mse, 1e-300)
    aic = n * np.log(sigma2) + 2 * k
    bic = n * np.log(sigma2) + k * np.log(n)

    q, dof = _ljung_box(innov, n_lags=lags)
    dof_eff = max(1, dof - order)
    crit = _CHI2_95.get(dof_eff, 1.5 * dof_eff)

    return ClosureFit(
        order=order, coeffs=coeffs, discrete_roots=roots, continuous_poles=cont,
        n_params=k, n_obs=n, train_mse=train_mse, onestep_mse=onestep_mse,
        rollout_mse=rollout_mse, aic=float(aic), bic=float(bic),
        ljung_box=q, ljung_box_dof=dof_eff, whiteness_pass=bool(q < crit),
        innovations=innov)


def select_order(residual_trajectories, dt, orders=(1, 2, 3), criterion="bic",
                 **kwargs):
    """Fit several orders and pick one by a penalized criterion.

    Returns (best_fit, {order: ClosureFit}). The criterion PENALIZES extra
    modes, so a two-mode model that merely overfits will not be selected.
    """
    fits = {k: fit_closure(residual_trajectories, dt, order=k, **kwargs)
            for k in orders}
    key = {"aic": lambda f: f.aic, "bic": lambda f: f.bic,
           "rollout": lambda f: f.rollout_mse}[criterion]
    best = min(fits.values(), key=key)
    return best, fits


def closure_defect(fit, true_poles=None):
    """A discrete analogue of the closure defect.

    Two complementary numbers:
      - `whiteness`: Ljung-Box Q on the innovations. If a closure of order k
        is sufficient, what is left is white and Q is small.
      - `pole_error`: distance from the fitted continuous poles to the true
        ones, when the truth is known.
    """
    out = {"ljung_box": fit.ljung_box, "whiteness_pass": bool(fit.whiteness_pass),
           "rollout_mse": fit.rollout_mse}
    if true_poles is not None:
        true = np.sort_complex(np.asarray(true_poles, dtype=complex))
        got = np.sort_complex(np.asarray(fit.continuous_poles, dtype=complex))
        m = min(len(true), len(got))
        out["pole_error"] = float(np.max(np.abs(true[:m] - got[:m])))
    return out


# =====================================================================
# Multivariate closures: the four model classes
#
#   scalar one-pole  ->  diagonal first-order  ->  collective first-order
#                    ->  augmented hidden-state closure
#
# A zero closure defect does NOT mean scalar one-pole dynamics. It means the
# CHOSEN OBSERVABLE SPACE IS CLOSED. A 2-D residual obeying r' = -Gamma r with
# a non-diagonal Gamma is exactly first-order and needs no hidden state, yet a
# scalar or diagonal model cannot represent it. These fitters separate "how
# many modes" from "how rich a first-order coupling".
# =====================================================================

from scipy.linalg import expm, logm                                  # noqa: E402

STRUCTURES = ("scalar", "diagonal", "full", "augmented")


@dataclass
class MatrixClosureFit:
    structure: str
    A_discrete: np.ndarray
    A2_discrete: Optional[np.ndarray]
    Gamma: Optional[np.ndarray]
    continuous_poles: np.ndarray
    n_params: int
    n_obs: int
    train_mse: float
    onestep_mse: float
    rollout_mse: float
    aic: float
    bic: float
    ljung_box: float
    whiteness_pass: bool

    def summary(self):
        return dict(structure=self.structure, n_params=self.n_params,
                    continuous_poles=[complex(p) for p in self.continuous_poles],
                    onestep_mse=self.onestep_mse, rollout_mse=self.rollout_mse,
                    aic=self.aic, bic=self.bic, ljung_box=self.ljung_box,
                    whiteness_pass=bool(self.whiteness_pass))


def _fit_A(X, Y, structure, n):
    """Least squares for Y ~ A X under a structural constraint.

    X, Y are (samples, n). Returns (A, n_params).
    """
    if structure == "scalar":
        a = float(np.sum(Y * X) / max(np.sum(X * X), 1e-300))
        return a * np.eye(n), 1
    if structure == "diagonal":
        diag = np.array([np.sum(Y[:, i] * X[:, i]) /
                         max(np.sum(X[:, i] ** 2), 1e-300) for i in range(n)])
        return np.diag(diag), n
    if structure == "full":
        A, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return A.T, n * n
    raise ValueError(structure)


def fit_matrix_closure(R, dt, structure="full", holdout_frac=0.3, lags=12):
    """Fit a first-order (or augmented second-order) closure to vector residuals.

    Args:
        R: (T, n) residual trajectory, or a list of them.
        structure: 'scalar' (lambda I), 'diagonal', 'full' (collective
            first-order), or 'augmented' (VAR(2): an extra lag, standing in
            for one hidden state per residual component). 'augmented' is
            included as an OVERPARAMETERIZED CONTROL - if model selection
            prefers it on exactly-first-order data, the selection is broken.
    """
    trajs = [np.atleast_2d(np.asarray(t, dtype=float)) for t in
             (R if isinstance(R, (list, tuple)) else [R])]
    trajs = [t if t.shape[0] >= t.shape[1] else t.T for t in trajs]
    n = trajs[0].shape[1]

    Xtr, Ytr, X2tr, holds = [], [], [], []
    for tr in trajs:
        cut = int(len(tr) * (1.0 - holdout_frac))
        train, hold = tr[:cut], tr[cut:]
        holds.append(hold)
        if structure == "augmented":
            Xtr.append(train[1:-1]); X2tr.append(train[:-2]); Ytr.append(train[2:])
        else:
            Xtr.append(train[:-1]); Ytr.append(train[1:])

    X = np.vstack(Xtr); Y = np.vstack(Ytr)
    if structure == "augmented":
        X2 = np.vstack(X2tr)
        Z = np.hstack([X, X2])
        W, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        A, A2 = W[:n].T, W[n:].T
        n_params = 2 * n * n
        pred = Z @ W
    else:
        A, n_params = _fit_A(X, Y, structure, n)
        A2 = None
        pred = X @ A.T

    innov = Y - pred
    train_mse = float(np.mean(innov ** 2))

    # continuous poles
    if structure == "augmented":
        companion = np.block([[A, A2], [np.eye(n), np.zeros((n, n))]])
        disc = np.linalg.eigvals(companion)
        Gamma = None
    else:
        disc = np.linalg.eigvals(A)
        try:
            Gamma = -np.real(logm(A)) / dt
        except Exception:
            Gamma = None
    with np.errstate(divide="ignore", invalid="ignore"):
        cont = np.log(disc.astype(complex)) / dt

    # held-out one-step and free-running rollout
    one, roll = [], []
    for hold in holds:
        if len(hold) < 3:
            continue
        if structure == "augmented":
            pr = hold[1:-1] @ A.T + hold[:-2] @ A2.T
            one.append(hold[2:] - pr)
            x1, x0 = hold[1].copy(), hold[0].copy()
            preds = []
            for _ in range(len(hold) - 2):
                nxt = A @ x1 + A2 @ x0
                preds.append(nxt); x0, x1 = x1, nxt
            roll.append(np.array(preds) - hold[2:])
        else:
            one.append(hold[1:] - hold[:-1] @ A.T)
            x = hold[0].copy(); preds = []
            for _ in range(len(hold) - 1):
                x = A @ x; preds.append(x)
            roll.append(np.array(preds) - hold[1:])

    onestep_mse = float(np.mean(np.concatenate(one) ** 2)) if one else float("nan")
    rollout_mse = float(np.mean(np.concatenate(roll) ** 2)) if roll else float("nan")

    n_obs = Y.shape[0] * n
    sigma2 = max(train_mse, 1e-300)
    aic = n_obs * np.log(sigma2) + 2 * n_params
    bic = n_obs * np.log(sigma2) + n_params * np.log(n_obs)
    q, dof = _ljung_box(innov[:, 0], n_lags=lags)
    crit = _CHI2_95.get(max(1, dof - 1), 1.5 * dof)

    return MatrixClosureFit(
        structure=structure, A_discrete=A, A2_discrete=A2, Gamma=Gamma,
        continuous_poles=cont, n_params=n_params, n_obs=n_obs,
        train_mse=train_mse, onestep_mse=onestep_mse, rollout_mse=rollout_mse,
        aic=float(aic), bic=float(bic), ljung_box=q,
        whiteness_pass=bool(q < crit))


def compare_structures(R, dt, structures=STRUCTURES, **kwargs):
    """Fit every model class and rank them by BIC (which penalizes parameters)."""
    fits = {s: fit_matrix_closure(R, dt, structure=s, **kwargs) for s in structures}
    best = min(fits.values(), key=lambda f: f.bic)
    return best, fits


def rollout(fit, r0, n_steps, r_minus1=None):
    """Free-run a fitted closure from an initial condition.

    Works for MatrixClosureFit (vector residual) and ClosureFit (scalar AR).

    CONVENTION: for a scalar AR(k) fit, `r0` is the first k samples in
    CHRONOLOGICAL order, [r_0, ..., r_{k-1}]. This function reverses them
    internally to match the coefficient ordering. Reversing before the call
    double-reverses and silently corrupts any order >= 2.
    """
    if isinstance(fit, MatrixClosureFit):
        A, A2 = fit.A_discrete, fit.A2_discrete
        x = np.atleast_1d(np.asarray(r0, dtype=float))
        if A2 is None:
            out = []
            for _ in range(n_steps):
                x = A @ x
                out.append(x)
            return np.array(out)
        prev = np.atleast_1d(np.asarray(
            r_minus1 if r_minus1 is not None else r0, dtype=float))
        out = []
        for _ in range(n_steps):
            nxt = A @ x + A2 @ prev
            out.append(nxt)
            prev, x = x, nxt
        return np.array(out)

    state = list(np.atleast_1d(np.asarray(r0, dtype=float))[::-1])
    out = []
    for _ in range(n_steps):
        nxt = float(np.dot(fit.coeffs, state))
        out.append(nxt)
        state = [nxt] + state[:-1]
    return np.asarray(out)


def evaluate_held_out(fit, trajectories):
    """Free-running rollout error on trajectories NOT used for calibration.

    This is the honest generalization test the specification asks for:
    calibrate the closure on one set, then evaluate response prediction on
    held-out initial states.
    """
    errs = []
    for tr in trajectories:
        tr = np.atleast_2d(np.asarray(tr, dtype=float))
        if tr.shape[0] < tr.shape[1]:
            tr = tr.T
        if isinstance(fit, MatrixClosureFit) and fit.A2_discrete is not None:
            pred = rollout(fit, tr[1], len(tr) - 2, r_minus1=tr[0])
            errs.append(pred - tr[2:])
        elif isinstance(fit, MatrixClosureFit):
            pred = rollout(fit, tr[0], len(tr) - 1)
            errs.append(pred - tr[1:])
        else:
            k = fit.order
            pred = rollout(fit, tr[:k, 0], len(tr) - k)
            errs.append(pred - tr[k:, 0])
    e = np.concatenate([x.ravel() for x in errs])
    return float(np.mean(e ** 2)), float(np.max(np.abs(e)))
