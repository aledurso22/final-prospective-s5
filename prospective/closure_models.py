"""Closure models for the residual/error observable r.

The theoretical object is the map (L, g) -> P_g: the prospective law is derived
for whatever residual observable actually closes, instead of assuming the
scalar one-pole law (I + tau D) g = 0.

This module provides the GENERATIVE models (used to make data and to run the
controls). Fitting/identification lives in `prospective/identification.py`.

Nothing here imports S5.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------- one mode

@dataclass
class OneModeClosure:
    """Ideal TSS residual law:  tau r' = -r,  i.e.  r' = -Gamma r.

    `gamma` may be a scalar or a square matrix (the "best first-order
    matrix/visible-residual model", C2, with no hidden temporal state).
    """
    gamma: object
    name: str = "one_mode"

    @property
    def dim(self):
        g = np.atleast_2d(self.gamma)
        return 1 if g.size == 1 else g.shape[0]

    def generator(self):
        return -np.atleast_2d(np.asarray(self.gamma, dtype=float))

    def eigenvalues(self):
        return np.linalg.eigvals(self.generator())

    def simulate(self, r0, dt, n_steps, u=None):
        """Exact (matrix-exponential) propagation, with optional forcing u."""
        A = self.generator()
        Ad = expm(A * dt)
        r = np.atleast_1d(np.asarray(r0, dtype=float)).copy()
        out = np.empty((n_steps, r.shape[0]))
        for k in range(n_steps):
            out[k] = r
            r = Ad @ r
            if u is not None:
                r = r + dt * np.atleast_1d(u[k])
        return out


# ---------------------------------------------------------------- two mode

@dataclass
class TwoModeClosure:
    """Fundamentally two-mode residual system with ONE HIDDEN state.

        r' = -gamma r + z b + u_c
        b' = -a b     + z r

    The hidden mode `b` is NOT an adaptive-current approximation; it is part of
    the residual system itself. Stability of the uncontrolled system requires
    a > 0, gamma > 0 and a*gamma > z**2 (positive determinant).
    """
    gamma: float
    a: float
    z: float
    name: str = "two_mode"

    def __post_init__(self):
        if self.a <= 0 or self.gamma <= 0:
            raise ValueError("a and gamma must be positive")
        if self.a * self.gamma <= self.z ** 2:
            raise ValueError(
                f"uncontrolled system unstable: a*gamma={self.a*self.gamma} "
                f"must exceed z^2={self.z**2}")

    def generator(self):
        return np.array([[-self.gamma, self.z],
                         [self.z, -self.a]], dtype=float)

    def eigenvalues(self):
        return np.linalg.eigvals(self.generator())

    def simulate(self, r0, b0, dt, n_steps, u_c=None):
        """Exact propagation of the 2x2 linear system.

        Forcing enters only the r equation, held constant over each step
        (zero-order hold), which is exact for the piecewise-constant control
        used by the oracle test.
        """
        A = self.generator()
        Ad = expm(A * dt)
        # ZOH input matrix for forcing on the r channel only
        B = np.array([[1.0], [0.0]])
        Bd = np.linalg.solve(A, (Ad - np.eye(2)) @ B)

        x = np.array([float(r0), float(b0)])
        out = np.empty((n_steps, 2))
        for k in range(n_steps):
            out[k] = x
            x = Ad @ x
            if u_c is not None:
                x = x + (Bd @ np.atleast_1d(u_c[k])).ravel()
        return out          # columns: r, b

    # ---- the analytic facts the tests check -----------------------------
    def rdot_at_zero_residual(self, b0):
        """r'(0) when r(0)=0: equals z*b0, so the residual reappears."""
        return self.z * b0

    def oracle_control(self, b0, t):
        """u_c(t) = -z b0 exp(-a t): the exact control holding r(t) == 0.

        With r == 0 the hidden state decays freely, b(t) = b0 exp(-a t), so
        r' = z b(t) + u_c must vanish.
        """
        return -self.z * b0 * np.exp(-self.a * np.asarray(t))

    def oracle_control_effort(self, b0, T):
        """Analytic  V = int_0^T 0.5 u_c^2 dt  for the oracle control.

        V = z^2 b0^2 (1 - exp(-2 a T)) / (4 a)
        """
        return (self.z ** 2) * (b0 ** 2) * (1.0 - np.exp(-2.0 * self.a * T)) \
            / (4.0 * self.a)


# ------------------------------------------------- TSS residual embedding

@dataclass
class TSSSystem:
    """A tracking system defined through the residual r = s - f(s, t).

    Differentiating,  r' = (I - f_s) s' - f_t = J_r s' - f_t, so imposing a
    residual law r' = L(r, ...) gives the prediction-correction law

        s' = J_r^-1 [ f_t + L(r, ...) ]

    With the ideal TSS law L = -r/tau this is  s' = J_r^-1 [f_t - r/tau],
    which is the form in the specification.
    """
    f: object            # f(s, t) -> array
    f_s: object          # df/ds  -> Jacobian (n, n)
    f_t: object          # df/dt  -> array (n,)
    dim: int = 1
    name: str = "tss"

    def residual(self, s, t):
        return np.atleast_1d(s) - np.atleast_1d(self.f(s, t))

    def jacobian_r(self, s, t):
        return np.eye(self.dim) - np.atleast_2d(self.f_s(s, t))

    def s_dot(self, s, t, residual_law):
        """s' = J_r^-1 [ f_t + residual_law(r) ]."""
        r = self.residual(s, t)
        rhs = np.atleast_1d(self.f_t(s, t)) + np.atleast_1d(residual_law(r))
        return np.linalg.solve(self.jacobian_r(s, t), rhs)

    def integrate(self, s0, dt, n_steps, residual_law, t0=0.0):
        """RK4 on s, recording s, r and t."""
        s = np.atleast_1d(np.asarray(s0, dtype=float)).copy()
        t = t0
        S = np.empty((n_steps, self.dim))
        R = np.empty((n_steps, self.dim))
        T = np.empty(n_steps)
        for k in range(n_steps):
            S[k], R[k], T[k] = s, self.residual(s, t), t
            k1 = self.s_dot(s, t, residual_law)
            k2 = self.s_dot(s + 0.5 * dt * k1, t + 0.5 * dt, residual_law)
            k3 = self.s_dot(s + 0.5 * dt * k2, t + 0.5 * dt, residual_law)
            k4 = self.s_dot(s + dt * k3, t + dt, residual_law)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt
        return dict(s=S, r=R, t=T)


def affine_tss(c=0.4, omega=1.3, amp=1.0, drift=0.15):
    """Scalar affine f(s,t) = c s + m(t).  J_r = 1 - c is constant and != 0."""
    def f(s, t):
        return c * np.atleast_1d(s) + amp * np.sin(omega * t) + drift * t

    def f_s(s, t):
        return np.array([[c]])

    def f_t(s, t):
        return np.array([amp * omega * np.cos(omega * t) + drift])

    return TSSSystem(f=f, f_s=f_s, f_t=f_t, dim=1, name=f"affine_c{c}")


def nonlinear_tss(c=0.6, omega=1.1, amp=1.0):
    """Mildly nonlinear f(s,t) = c tanh(s) + m(t).

    |f_s| = |c (1 - tanh^2 s)| <= |c| < 1, so J_r = 1 - f_s stays nonsingular.
    """
    def f(s, t):
        return c * np.tanh(np.atleast_1d(s)) + amp * np.sin(omega * t)

    def f_s(s, t):
        return np.array([[c * (1.0 - np.tanh(float(np.atleast_1d(s)[0])) ** 2)]])

    def f_t(s, t):
        return np.array([amp * omega * np.cos(omega * t)])

    return TSSSystem(f=f, f_s=f_s, f_t=f_t, dim=1, name=f"tanh_c{c}")


def two_dim_affine_tss(C=None, omega=(0.9, 1.7), amp=(1.0, 0.6)):
    """2-D affine system; (I - C) must be invertible."""
    if C is None:
        C = np.array([[0.3, 0.2], [-0.15, 0.35]])
    C = np.asarray(C, dtype=float)
    if abs(np.linalg.det(np.eye(2) - C)) < 1e-8:
        raise ValueError("I - C is singular")

    def f(s, t):
        m = np.array([amp[0] * np.sin(omega[0] * t),
                      amp[1] * np.cos(omega[1] * t)])
        return C @ np.atleast_1d(s) + m

    def f_s(s, t):
        return C

    def f_t(s, t):
        return np.array([amp[0] * omega[0] * np.cos(omega[0] * t),
                         -amp[1] * omega[1] * np.sin(omega[1] * t)])

    return TSSSystem(f=f, f_s=f_s, f_t=f_t, dim=2, name="affine2d")
