"""Independent reference implementations shared by checks and dtype probes.

**This module must never touch `jax.config`.** It is imported by the
production-dtype probes, which run with x64 OFF in their own process; a module
that enables x64 at import time would silently turn it back on and the probe
would measure float64 while asserting it had measured float32. That is exactly
what happened when the probes imported their helpers from the x64-enabling test
modules, so the helpers live here instead, and nothing here imports `jax` at
all.

Everything is numpy + scipy and independent of the production helpers in the
ways that matter: `scipy.linalg.expm` is a different exponential
implementation, the complex 2x2 blocks are exponentiated through their
REAL-PAIR embedding rather than as complex matrices, and the ZOH integral is
Gauss-Legendre quadrature rather than production's augmented-matrix trick.
"""

import numpy as onp
from scipy.linalg import expm as sp_expm


def to_real_pair(M):
    """Complex (n,n) -> real (2n,2n) acting on [Re q; Im q]."""
    return onp.block([[onp.real(M), -onp.imag(M)],
                      [onp.imag(M), onp.real(M)]])


def from_real_pair(R):
    n = R.shape[0] // 2
    return R[:n, :n] + 1j * R[n:, :n]


def modes(rs, P, H):
    """Stable complex poles and a complex input map, the usual test fixture."""
    a = onp.asarray(-onp.exp(rs.uniform(-3.0, -0.5, P))
                    + 1j * rs.uniform(-2.5, 2.5, P))
    b = onp.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    return a, b


def ssm_kwargs(P, H, seed=29):
    """Minimal conjugate-symmetric S5 configuration for a bare layer.

    `Vinv` is (P, 2P) and `V` is (2P, P): `init_VinvB` forms `Vinv @ B` from a
    (2P, H) draw, and `init_CV` forms `C @ V` from an (H, 2P) draw.
    """
    rs = onp.random.RandomState(seed)
    return dict(H=H, P=P,
                Lambda_re_init=-onp.exp(rs.uniform(-3, -0.5, P)),
                Lambda_im_init=rs.uniform(-2, 2, P),
                V=onp.eye(2 * P, dtype=complex)[:, :P],
                Vinv=onp.eye(2 * P, dtype=complex)[:P],
                C_init="trunc_standard_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=True, bidirectional=False)


def _generator(a, b, T, rho):
    """Per-mode continuous (A, B) for the (s, v) carry; T and rho per-mode."""
    a = onp.asarray(a); b = onp.asarray(b)
    P, H = b.shape
    T = onp.broadcast_to(onp.asarray(T, dtype=float), (P,))
    rho = onp.broadcast_to(onp.asarray(rho, dtype=float), (P,))
    A = onp.zeros((P, 2, 2), dtype=complex)
    B = onp.zeros((P, 2, H), dtype=complex)
    for p in range(P):
        J, r, t = -a[p], rho[p], T[p]
        A[p] = onp.array([[-J / r, -(1.0 - r) / r],
                          [-J / (r * t), -1.0 / (r * t)]], dtype=complex)
        B[p] = onp.stack([b[p] / r, b[p] / (r * t)], axis=0)
    return A, B


def reference_block(a, b, T, rho, nodes=64):
    """A, B, A_bar, Phi, B_bar per mode, computed independently of production."""
    A, B = _generator(a, b, T, rho)
    P = A.shape[0]
    xg, wg = onp.polynomial.legendre.leggauss(nodes)
    ug, wg = 0.5 * (xg + 1.0), 0.5 * wg
    A_bar = onp.zeros_like(A); Phi = onp.zeros_like(A)
    for p in range(P):
        Ar = to_real_pair(A[p])
        A_bar[p] = from_real_pair(sp_expm(Ar))
        Phi[p] = from_real_pair(
            sum(w * sp_expm(Ar * u) for u, w in zip(ug, wg)))
    return dict(A=A, B=B, A_bar=A_bar, Phi=Phi,
                B_bar=onp.einsum("pij,pjh->pih", Phi, B))


def reference_two_tap(a, b, T, rho, horizon_in, nodes=64):
    """The same, plus the prospective-input jump coefficients.

        J_in = T_in * A_bar @ B   (the CONTINUOUS B)
        B_+  = B_bar + J_in,   B_- = -J_in
    """
    d = reference_block(a, b, T, rho, nodes)
    J_in = horizon_in * onp.einsum("pij,pjh->pih", d["A_bar"], d["B"])
    return dict(d, J_in=J_in, B_plus=d["B_bar"] + J_in, B_minus=-J_in)


def block_conditioning(A_bar):
    """Per-mode eigenvector condition number of the executed transition.

    Reported alongside coefficient comparisons at the declared guardrail
    corners: a stiff (T, rho) corner makes the 2x2 block's eigenvectors nearly
    parallel, and two correct matrix-exponential implementations then differ by
    roughly `cond * eps` rather than by `eps`. That is a conditioning fact
    about the corner, not an error in either implementation, and it is the
    number a corner tolerance has to be justified against.
    """
    A_bar = onp.asarray(A_bar)
    out = []
    for p in range(A_bar.shape[0]):
        _, V = onp.linalg.eig(A_bar[p])
        out.append(float(onp.linalg.cond(V)))
    return onp.asarray(out)


def frequency_with_exact_tail(K, A_bar, C_tilde, D, fac, w, response,
                              coefficients):
    """H(w) from a MEASURED impulse window plus the EXACT remainder.

    A bare DFT of a finite impulse window is not the frequency response: with
    `Delta` as small as 1e-3 the executed block has barely decayed after a
    hundred lags, so the omitted tail is not negligible and truncating it is
    the invalid-tail-bound mistake in a different disguise. The remainder is
    therefore computed in closed form from the executed transition:

        sum_{l >= N} K_l u^l ,  K_l = fac * Re(C [A^l S_0]_s)

    and for every response here the state satisfies `S_l = A S_{l-1}` once
    l >= 2, so from the state at lag N-1,

        sum_{l >= N} u^l A^{l-N+1} S_{N-1} = u^{N-1} (uA)(I - uA)^{-1} S_{N-1}.

    Realness is handled exactly as the production reader does, by assembling
    conjugate partners rather than taking 2*Re of one complex transfer.
    """
    A = onp.asarray(A_bar).astype(onp.complex128)
    C = onp.asarray(C_tilde).astype(onp.complex128)
    n_lags = K.shape[0]
    if response == "gp_rho_prospin":
        S = onp.asarray(coefficients["B_plus"]).astype(onp.complex128)
        Bm = onp.asarray(coefficients["B_minus"]).astype(onp.complex128)
        for l in range(1, n_lags):
            S = onp.einsum("pij,pjh->pih", A, S)
            if l == 1:
                S = S + Bm
    else:
        S = onp.asarray(coefficients["B_bar"]).astype(onp.complex128)
        for _ in range(1, n_lags):
            S = onp.einsum("pij,pjh->pih", A, S)
    I2 = onp.eye(2, dtype=onp.complex128)

    def tail(u):
        X = onp.linalg.solve(I2[None] - A * u,
                             onp.einsum("pij,pjh->pih", A * u, S))
        return (u ** (n_lags - 1)) * (C @ X[:, 0, :])

    out = onp.zeros((len(w), K.shape[1], K.shape[2]), dtype=onp.complex128)
    for k, wk in enumerate(w):
        u, ub = onp.exp(-1j * wk), onp.exp(+1j * wk)
        dft = sum(K[l] * onp.exp(-1j * wk * l) for l in range(n_lags))
        out[k] = dft + (tail(u) + onp.conj(tail(ub))) * (fac / 2.0)
    return out
