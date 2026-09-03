"""Interpretable measurements for the professor-mechanism comparison."""

import numpy as np


def homogeneous_response(model, system, length=600, s0=None):
    """Zero-input response from a non-zero initial state.

    Poles are a property of the HOMOGENEOUS dynamics, so this - not the
    impulse response - is the correct place to measure them. It matters here:
    a fully prospective mode has the forced response s = K x exactly, so its
    impulse response is K at k=0 and identically zero afterwards, and a decay
    fit to it is meaningless.
    """
    x = np.zeros(length)
    if s0 is None:
        s0 = np.ones(system.n_modes)
    return model.run(x, s0=s0)


def forced_impulse_support(series, tol=1e-12):
    """Number of samples of an impulse response that exceed tol.

    1 means the forced response has no memory at all (s = K x).
    """
    y = np.abs(np.asarray(series, dtype=float))
    nz = np.nonzero(y > tol * max(1.0, y.max()))[0]
    return int(nz[-1] + 1) if len(nz) else 0


def impulse_response(model, length=400):
    x = np.zeros(length); x[0] = 1.0
    return model.run(x)


def step_response(model, length=400, t_on=20):
    x = np.zeros(length); x[t_on:] = 1.0
    return model.run(x)


def ramp_response(model, length=400, slope=0.01):
    x = slope * np.arange(length, dtype=float)
    return model.run(x)


def empirical_discrete_pole(series, dt, skip=5, floor=1e-12):
    """Fit a decay rate to |series| by log-linear regression on the tail."""
    y = np.abs(np.asarray(series, dtype=float))[skip:]
    keep = y > floor
    if keep.sum() < 5:
        return np.nan
    idx = np.arange(len(y))[keep]
    slope = np.polyfit(idx, np.log(y[keep]), 1)[0]
    return float(np.exp(slope))


def half_life_from_impulse(series, dt, skip=5):
    """Continuous-time half-life measured from an impulse response."""
    pole = empirical_discrete_pole(series, dt, skip=skip)
    if not np.isfinite(pole) or pole <= 0 or pole >= 1:
        return np.inf if pole >= 1 else 0.0
    return float(dt * np.log(0.5) / np.log(pole))


def impulse_retention(model, lags, length=4000):
    """|h[lag]| / |h[0]| per mode, from the impulse response.

    The direct measure of how long an input keeps influencing the state.
    A value of 0 at every lag > 0 means the forced response carries no memory.

    (An earlier attempt measured R^2 of linearly predicting x[k-lag] from the
    state. That is not a usable memory metric for a 2-mode system: the linear
    capacity of a 2-dimensional state is about 2 delays regardless of pole
    placement, so it returned near-zero for every model including plain S5.)
    """
    h = impulse_response(model, length=length)
    h0 = np.abs(h[0])
    out = {}
    for lag in lags:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[lag] = np.where(h0 > 0, np.abs(h[lag]) / np.where(h0 > 0, h0, 1.0),
                                np.nan)
    return out


def optimal_shared_alpha(y, target, skip=5):
    """Closed-form least-squares alpha for y + alpha*(y - y_prev) ~= target.

    The lead is affine in alpha, so the optimum is exact, not a grid search.
    """
    y = np.asarray(y, float).reshape(len(y), -1)
    t = np.asarray(target, float).reshape(len(target), -1)
    prev = np.concatenate((y[:1], y[:-1]), axis=0)
    d = (y - prev)[skip:].ravel()
    r = (t - y)[skip:].ravel()
    denom = float(d @ d)
    if denom == 0:
        return 0.0, float(np.sqrt(np.mean(r ** 2)))
    a = float(d @ r / denom)
    resid = r - a * d
    return a, float(np.sqrt(np.mean(resid ** 2)))


def tracking_error(series, target, burn=0):
    d = np.asarray(series, float)[burn:] - np.asarray(target, float)[burn:]
    return float(np.sqrt(np.mean(d ** 2)))


def time_to_correct(series, final, dt, tol=0.05, t_on=0):
    """First time after t_on at which the series enters and stays within tol."""
    y = np.asarray(series, float)
    if final == 0:
        return np.nan
    within = np.abs(y - final) <= tol * abs(final)
    idx = None
    for k in range(len(y) - 1, t_on - 1, -1):
        if not within[k]:
            idx = k + 1
            break
        idx = k
    if idx is None or idx >= len(y):
        return np.nan
    return float((idx - t_on) * dt)


def noise_gain(model, system, length=4000, seed=1, use_readout=False):
    """Output std per unit input std, on white noise. An LTI gain measure."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=length)
    if use_readout and hasattr(model, "run_readout"):
        y = model.run_readout(x)[:, 0]
    else:
        y = model.run(x) @ system.C.T
        y = y[:, 0]
    return float(np.std(y[200:]) / np.std(x[200:]))


def reset_transient(model, system, length=200, s0=None):
    """Response to zero input from a non-zero initial state."""
    x = np.zeros(length)
    if s0 is None:
        s0 = np.ones(system.n_modes)
    return model.run(x, s0=s0)
