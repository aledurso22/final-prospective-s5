"""The synthetic diagonal test system used by the professor-mechanism study.

Deliberately tiny, linear and diagonal so that every claim can be checked in
closed form. Nothing here touches S5.

Continuous plant (the "plain" reference dynamics):

    tau * ds/dt = (A - I) s + B x

with A = diag(a_i). Per mode:

    continuous pole   p_i      = (a_i - 1) / tau
    equilibrium gain  K_i      = B_i / (1 - a_i)          [ (I-A)^-1 B ]
    ZOH discrete pole lam_i    = exp((a_i - 1) * dt / tau)
    ZOH input gain    bbar_i   = B_i * (lam_i - 1) / (a_i - 1)

The prospective residual pole, shared by every fully-prospective mode, is

    rho = exp(-dt / tau)          [ from tau * dr/dt = -r ]

A mode is designated MEMORY (projector P_M) or TRACKING (projector P_T) by
`tracking_mask`. P_M + P_T = I holds by construction because the mask is a
partition of the diagonal coordinates.
"""

import numpy as np


class DiagonalSystem:
    def __init__(self, a, B, C, tau=1.0, dt=0.05, tracking_mask=None,
                 name="system"):
        self.a = np.asarray(a, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.atleast_2d(np.asarray(C, dtype=np.float64))
        self.tau = float(tau)
        self.dt = float(dt)
        self.name = name
        if np.any(self.a >= 1.0):
            raise ValueError("a_i must be < 1 so that (a_i - 1) < 0 (stable)")
        if tracking_mask is None:
            tracking_mask = np.zeros_like(self.a, dtype=bool)
        self.tracking_mask = np.asarray(tracking_mask, dtype=bool)

    # ---- modal partition -------------------------------------------------
    @property
    def P_T(self):
        """Tracking projector, as a diagonal vector."""
        return self.tracking_mask.astype(np.float64)

    @property
    def P_M(self):
        """Memory projector. P_M + P_T = I by construction."""
        return 1.0 - self.P_T

    # ---- continuous-time quantities -------------------------------------
    @property
    def K(self):
        """Equilibrium gain (I - A)^-1 B."""
        return self.B / (1.0 - self.a)

    @property
    def plain_continuous_poles(self):
        return (self.a - 1.0) / self.tau

    @property
    def prospective_continuous_pole(self):
        return -1.0 / self.tau

    # ---- discrete-time quantities ---------------------------------------
    @property
    def lam(self):
        """Plain ZOH discrete pole per mode."""
        return np.exp((self.a - 1.0) * self.dt / self.tau)

    @property
    def bbar(self):
        """Plain ZOH discrete input gain per mode."""
        return self.B * (self.lam - 1.0) / (self.a - 1.0)

    @property
    def rho(self):
        """Prospective residual discrete pole, exp(-dt/tau)."""
        return float(np.exp(-self.dt / self.tau))

    @property
    def n_modes(self):
        return self.a.shape[0]

    def half_life(self, discrete_pole):
        """Continuous-time half-life implied by a discrete pole."""
        p = np.asarray(discrete_pole, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.where(p > 0, self.dt * np.log(0.5) / np.log(p), np.inf)

    def describe(self):
        rows = []
        for i in range(self.n_modes):
            rows.append(dict(
                mode=i,
                role="tracking" if self.tracking_mask[i] else "memory",
                a=self.a[i], B=self.B[i], K=self.K[i],
                plain_continuous_pole=self.plain_continuous_poles[i],
                plain_discrete_pole=self.lam[i],
                plain_half_life=self.half_life(self.lam[i]),
            ))
        return rows


def two_timescale_system(tau=1.0, dt=0.05):
    """The reference system: one slow memory mode, one fast tracking mode.

    mode 0: a = 0.98  -> very slow, long memory     -> MEMORY  (P_M)
    mode 1: a = 0.20  -> fast, used for tracking    -> TRACKING (P_T)
    """
    return DiagonalSystem(
        a=[0.98, 0.20],
        B=[1.0, 1.3],
        C=[[1.0, 1.0]],
        tau=tau, dt=dt,
        tracking_mask=[False, True],
        name="two_timescale",
    )
