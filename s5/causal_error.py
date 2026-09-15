"""Two causal approximations to the future-facing adjoint, and their audit.

The exact adjoint of the generalized forward node is future-facing:

    H(p) = (1 + tau p) / (1 + a0 p + M p^2),   a0 = gamma + tau,
    H_A(p) = H(-p) = (1 - tau p) / (1 - a0 p + M p^2),
    rho = H_A[c + A^T rho]        (terminal conditions, RHP poles).

H_A must NOT be simulated as a causal forward-time ODE. Both approximations
below are stable CAUSAL filters integrated forward in time whose low-frequency
expansion matches H_A; the phase lead that the true adjoint gets from looking
forward is supplied here by a lead numerator, and the price is the truncation
error quantified exactly by `remainder_bound`.

(i) `reciprocal` - a GLE-INSPIRED ordinary-prospective baseline, Sections 4-6
    of CAUSAL_LEARNING_AGENT: the regularized causal inverse E_eps, cascaded
    with R_delta so the error path is strictly proper and carries no algebraic
    feedthrough. R7: it is NOT a verbatim TSS or GLE learning rule, and it must
    be labelled GLE-inspired wherever it is reported. The exact UNREGULARIZED
    inverse 1/H shares the adjoint's phase on the Fourier axis for real
    coefficients; the EXECUTED product E_eps * R_delta generally does not, so
    no exact-phase claim is made for what actually runs. Its low-frequency
    error is O(omega^2).

(ii) `moment_matched` - Section 11, the preferred construction: a stable
     strictly proper filter matching H_A through second order in frequency,
     with an O(omega^3) error.

AUDIT POINTS CARRIED IN CODE, NOT ONLY IN PROSE
-----------------------------------------------
* Cubic asymptotic order does NOT imply a smaller error at every bandwidth.
  `remainder_bound` returns the exact finite-band figure for both, and the
  pilot reports both across predeclared bandwidths rather than choosing one.
  `measured_band_error` evaluates on a FINITE GRID and therefore returns a
  SAMPLED maximum, not a proven continuum supremum.
* Small eps buys accuracy at the cost of gain: the peak can grow like
  2 c2 / (3 sqrt 3 eps^2). `peak_gain_bound` returns (M6); the pilot reports it
  alongside the executed filter poles.
* Forward stability does NOT imply error-loop stability. `recurrent_witness`
  reproduces the Section 7 counterexample, where a stable forward node has an
  UNSTABLE reciprocal error loop. The pilot's Part B therefore uses a spatially
  FEEDFORWARD network, where the error coupling is triangular and no such loop
  exists; recurrent error dynamics are out of scope until validated.
* Neither filter supplies an initial-state derivative. The causal states start
  at zero, so nothing here estimates dJ/dz(0); the exact reference does, and
  the pilot reports that asymmetry instead of hiding it.
"""

import numpy as onp
from scipy.linalg import expm


def adjoint_moments(gamma, tau, M):
    """a0, c2, c3 of H_A(p) = 1 + gamma p + c2 p^2 + c3 p^3 + O(p^4)."""
    a0 = gamma + tau
    c2 = gamma * a0 - M
    c3 = a0 * c2 - M * gamma
    return a0, c2, c3


def moment_matched_weights(gamma, tau, M, eps):
    """w1, w2, w3 of (M3). They sum to one; w2 < 0 supplies the extrapolation."""
    _, c2, _ = adjoint_moments(gamma, tau, M)
    w1 = c2 / eps ** 2 + 3.0 * gamma / eps + 3.0
    w2 = -2.0 * c2 / eps ** 2 - 5.0 * gamma / eps - 3.0
    w3 = c2 / eps ** 2 + 2.0 * gamma / eps + 1.0
    return onp.array([w1, w2, w3])


def moment_matched_system(gamma, tau, M, eps):
    """(A, B, w) for zdot = A z + B k, rhohat = w . z. Three low-pass states."""
    A = onp.array([[-1.0 / eps, 0.0, 0.0],
                   [1.0 / eps, -1.0 / eps, 0.0],
                   [0.0, 1.0 / eps, -1.0 / eps]])
    B = onp.array([[1.0 / eps], [0.0], [0.0]])
    return A, B, moment_matched_weights(gamma, tau, M, eps)


def reciprocal_system(gamma, tau, M, eps, delta):
    """(A, B, w) for E_eps cascaded with R_delta. Four states, strictly proper.

    E_eps(p) = 1 + (M/tau) p/(1+eps p) + (gamma - M/tau) p/(1+tau p), realized
    without differentiating k as in (C3); R_delta = (1+2 delta p)/(1+delta p)^2
    removes the algebraic feedthrough as in (C5)-(C6).
    """
    g1 = M / (tau * eps)
    g2 = (gamma - M / tau) / tau
    beta0 = 1.0 + g1 + g2            # feedthrough of k into b
    beta_e = -g1                     # coefficient of h_eps in b
    beta_t = -g2                     # coefficient of h_tau in b
    # states: h_eps, h_tau, z1, z2
    A = onp.array([
        [-1.0 / eps, 0.0, 0.0, 0.0],
        [0.0, -1.0 / tau, 0.0, 0.0],
        [beta_e / delta, beta_t / delta, -1.0 / delta, 0.0],
        [0.0, 0.0, 1.0 / delta, -1.0 / delta]])
    B = onp.array([[1.0 / eps], [1.0 / tau], [beta0 / delta], [0.0]])
    w = onp.array([0.0, 0.0, 2.0, -1.0])
    return A, B, w


def zoh(A, B, dt=1.0):
    """Exact zero-order-hold discretization by one augmented exponential.

    The error filters are LINEAR with constant coefficients, so exact
    discretization is available and is used: the pilot's integration error
    should come from the nonlinear forward model alone, not from the filters
    that are being compared.
    """
    n, m = A.shape[0], B.shape[1]
    aug = onp.zeros((n + m, n + m))
    aug[:n, :n] = A
    aug[:n, n:] = B
    E = expm(aug * dt)
    return E[:n, :n], E[:n, n:]


def discretize(kind, gamma, tau, M, eps, delta, dt=1.0):
    """Precompute (Ad, Bd, w) once. R9: the pilot reuses these across every
    trajectory, bandwidth and seed instead of exponentiating per call."""
    A, B, w = (moment_matched_system(gamma, tau, M, eps) if kind == "moment"
               else reciprocal_system(gamma, tau, M, eps, delta))
    Ad, Bd = zoh(A, B, dt)
    return Ad, Bd, w


def run_filter_d(Ad, Bd, w, k):
    """rhohat over a sequence, from PRECOMPUTED discrete matrices.

    `k` is (L, ...): every axis after the first is treated as an independent
    channel, so a whole batch of trajectories advances in one pass. States
    start at ZERO, which is the causal prescription; the exact adjoint instead
    has a TERMINAL condition, so the two disagree near the sequence ends by
    construction and the pilot reports start/interior/end windows separately.
    """
    L = k.shape[0]
    flat = onp.asarray(k).reshape(L, -1)
    z = onp.zeros((Ad.shape[0], flat.shape[1]))
    out = onp.zeros_like(flat)
    for t in range(L):
        z = Ad @ z + Bd @ flat[t][None, :]
        out[t] = w @ z
    return out.reshape(k.shape)


def run_filter(A, B, w, k, dt=1.0):
    """Convenience wrapper that discretizes then runs. Prefer `run_filter_d`."""
    Ad, Bd = zoh(A, B, dt)
    return run_filter_d(Ad, Bd, w, k)


# ------------------------------------------------------------- the audit ---
def transfer(kind, gamma, tau, M, eps, delta, omega):
    """Executed filter response on the Fourier axis, from its own (A, B, w)."""
    A, B, w = (moment_matched_system(gamma, tau, M, eps) if kind == "moment"
               else reciprocal_system(gamma, tau, M, eps, delta))
    p = 1j * onp.asarray(omega)
    out = onp.zeros(p.shape, dtype=complex)
    I = onp.eye(A.shape[0])
    for i, pi in enumerate(onp.atleast_1d(p)):
        out[i] = (w @ onp.linalg.solve(pi * I - A, B))[0]
    return out


def exact_adjoint_transfer(gamma, tau, M, omega):
    """H_A(i omega) = conj(H(i omega)); the target both filters approximate."""
    p = 1j * onp.asarray(omega)
    return (1.0 - tau * p) / (1.0 - (gamma + tau) * p + M * p ** 2)


def remainder_bound(gamma, tau, M, eps, Omega):
    """(M4)-(M5): the EXACT moment-matched remainder and its band bound.

    The reciprocal's low-frequency error is O(omega^2) by (A4) and (C4); the
    moment-matched filter's is O(omega^3). Both are returned as measured
    sup-norm differences over the band as well, because the asymptotic order
    settles nothing at a finite bandwidth.
    """
    _, c2, c3 = adjoint_moments(gamma, tau, M)
    B3 = c3 + 3 * c2 * eps + 3 * gamma * eps ** 2 + eps ** 3
    B4 = M * (c2 + 3 * gamma * eps + 3 * eps ** 2) + tau * eps ** 3
    return dict(B3=float(B3), B4=float(B4),
                d_Omega=float(Omega ** 3 * onp.sqrt(B3 ** 2
                                                    + B4 ** 2 * Omega ** 2)),
                c2=float(c2), c3=float(c3),
                c2_ge_gamma_sq=bool(c2 >= gamma ** 2),
                c3_ge_gamma_cu=bool(c3 >= gamma ** 3))


def peak_gain_bound(gamma, tau, M, eps):
    """(M6), plus the small-eps peak estimate 2 c2 / (3 sqrt 3 eps^2)."""
    _, c2, _ = adjoint_moments(gamma, tau, M)
    return dict(
        bound=float(1.0 + (2.0 / (3.0 * onp.sqrt(3.0)))
                    * (c2 / eps ** 2 + 4.0 * gamma / eps + 6.0)),
        small_eps_peak_estimate=float(2.0 * c2 / (3.0 * onp.sqrt(3.0)
                                                  * eps ** 2)))


def filter_poles(kind, gamma, tau, M, eps, delta):
    """Executed backward-state poles. Audited, not assumed stable."""
    A, _, _ = (moment_matched_system(gamma, tau, M, eps) if kind == "moment"
               else reciprocal_system(gamma, tau, M, eps, delta))
    ev = onp.linalg.eigvals(A)
    return dict(poles=[complex(v) for v in ev],
                max_real_part=float(onp.max(ev.real)),
                stable=bool(onp.max(ev.real) < 0.0))


def measured_band_error(kind, gamma, tau, M, eps, delta, Omega, n=513):
    """SAMPLED max of |K(i w) - H_A(i w)| on a finite grid over |w| <= Omega.

    R7: a finite grid gives a sampled maximum, not a proven supremum over the
    continuum. It is reported as such.
    """
    w = onp.linspace(0.0, Omega, n)
    return float(onp.max(onp.abs(transfer(kind, gamma, tau, M, eps, delta, w)
                                 - exact_adjoint_transfer(gamma, tau, M, w))))


def recurrent_witness():
    """Section 7: a stable forward node whose reciprocal error loop is UNSTABLE.

    gamma = tau = 1, f(s) = 0.9 s. Forward pole -0.0909; the causal-inverse
    error loop has pole +0.125. Retained as a control so that "the forward
    model is stable" is never offered as evidence about the error dynamics.
    """
    gamma = tau = 1.0
    alpha = 0.9
    # H = (1+p)/(1+2p); closing f = 0.9 s gives (1+2p) - 0.9(1+p) = 0.1 + 1.1p
    forward_pole = -0.1 / 1.1
    # E = (1+2p)/(1+p); closing 0.9 gives (1+p) - 0.9(1+2p) = 0.1 - 0.8p
    error_pole = 0.1 / 0.8
    return dict(gamma=gamma, tau=tau, alpha=alpha,
                forward_pole=float(forward_pole),
                reciprocal_error_pole=float(error_pole),
                forward_stable=bool(forward_pole < 0),
                error_loop_stable=bool(error_pole < 0),
                note=("forward stability does not imply error-loop stability; "
                      "Part B therefore uses a spatially feedforward network "
                      "whose error coupling is triangular"))
