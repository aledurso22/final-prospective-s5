"""Diagnostics for the linear core of a (generalized) S5 layer.

Scope, stated plainly: everything here describes ONE LAYER'S LINEAR CORE. It is
not the transfer of the complete nonlinear network. A layer's transfer is not
the network's transfer, and no claim here should be read as a statement about
the full stack. A nonlinear full-stack memory measurement requires input
perturbations about declared sequences and is a separate protocol.

Native `Lambda` is NOT the effective generalized pole. The quantity that
governs the generalized dynamics is `a_eff = a/(1 - t a)` in sample-clock
units, and its discrete counterpart `a_bar = exp(a_eff)`.
"""

import jax.numpy as np
import numpy as onp


def linear_core(a_bar, b_bar, d_x, C_tilde, D, conj_sym=True):
    """Package the realized discrete core: h_k = a_bar h_{k-1} + b_bar x_k,
    s_k = h_k + D_x x_k, y_k = f * Re(C_tilde s_k) + D * x_k."""
    return dict(a_bar=onp.asarray(a_bar), b_bar=onp.asarray(b_bar),
                d_x=onp.asarray(d_x), C_tilde=onp.asarray(C_tilde),
                D=onp.asarray(D), factor=2.0 if conj_sym else 1.0)


def poles(core, a=None, a_eff=None):
    """Discrete poles and stability margins.

    `margin_discrete` is 1 - max|a_bar|: positive means contraction per step.
    """
    ab = core["a_bar"]
    out = dict(discrete_pole_abs=onp.abs(ab).tolist(),
               max_discrete_pole_abs=float(onp.max(onp.abs(ab))),
               margin_discrete=float(1.0 - onp.max(onp.abs(ab))))
    if a is not None:
        out["clocked_native_pole_real"] = onp.asarray(a).real.tolist()
        out["max_clocked_native_real"] = float(onp.max(onp.asarray(a).real))
    if a_eff is not None:
        ae = onp.asarray(a_eff)
        out["effective_pole_real"] = ae.real.tolist()
        out["max_effective_real"] = float(onp.max(ae.real))
        out["effective_efolding_steps"] = [
            float(-1.0 / r) if r < 0 else float("inf") for r in ae.real]
    return out


def markov_parameters(core, n):
    """Exact discrete impulse response G_k, k = 0..n-1, shape (n, H_out, H_in).

    G_0 is the DIRECT term (tied feedthrough + native D); G_k for k >= 1 is the
    HISTORY term. Computed in closed form, not by running the scan.
    """
    ab, bb, dx, C, f = (core["a_bar"], core["b_bar"], core["d_x"],
                        core["C_tilde"], core["factor"])
    H_in = bb.shape[1]
    H_out = C.shape[0]
    G = onp.zeros((n, H_out, H_in))
    # k = 0: h_0 = b_bar x_0 and the tied feedthrough both act
    G[0] = f * onp.real(C @ (bb + dx)) + onp.diag(core["D"]) @ onp.eye(H_out, H_in) \
        if H_out == H_in else f * onp.real(C @ (bb + dx))
    if H_out == H_in:
        G[0] = f * onp.real(C @ (bb + dx)) + onp.diag(core["D"])
    for k in range(1, n):
        G[k] = f * onp.real(C @ ((ab ** k)[:, None] * bb))
    return G


def direct_history_decomposition(core, n=256):
    """Split the response into its instantaneous and history parts."""
    G = markov_parameters(core, n)
    direct = float(onp.linalg.norm(G[0]))
    history = float(onp.linalg.norm(G[1:]))
    return dict(direct_norm=direct, history_norm=history,
                history_fraction=float(history / max(direct + history, 1e-300)),
                impulse_support=int(
                    onp.max(onp.nonzero(
                        onp.linalg.norm(G.reshape(n, -1), axis=1)
                        > 1e-12 * max(1.0, onp.linalg.norm(G[0])))[0]) + 1)
                if onp.any(onp.linalg.norm(G.reshape(n, -1), axis=1) > 0) else 0)


def dc_response(core):
    """Steady-state gain for constant input: y_ss = [f Re(C ((I-abar)^-1 bbar
    + D_x)) + diag(D)] x."""
    ab, bb, dx, C, f = (core["a_bar"], core["b_bar"], core["d_x"],
                        core["C_tilde"], core["factor"])
    h_ss = bb / (1.0 - ab)[:, None]
    dc = f * onp.real(C @ (h_ss + dx))
    if dc.shape[0] == core["D"].shape[0] and dc.shape[1] == core["D"].shape[0]:
        dc = dc + onp.diag(core["D"])
    return dict(dc_gain_norm=float(onp.linalg.norm(dc)), dc_gain=dc.tolist())


def hankel_singular_values(core, n=64, k=None):
    """Finite block-Hankel singular values from the HISTORY Markov parameters.

    Slow poles alone do not certify memory. This measures the actual
    past-to-future operator of the layer's linear core, truncated at `n`.
    """
    G = markov_parameters(core, 2 * n + 1)[1:]          # drop the direct term
    H_out, H_in = G.shape[1], G.shape[2]
    blocks = onp.zeros((n * H_out, n * H_in))
    for i in range(n):
        for j in range(n):
            blocks[i * H_out:(i + 1) * H_out, j * H_in:(j + 1) * H_in] = G[i + j]
    sv = onp.linalg.svd(blocks, compute_uv=False)
    if k:
        sv = sv[:k]
    return dict(hankel_singular_values=sv.tolist(),
                hankel_top=float(sv[0]) if sv.size else 0.0,
                hankel_nuclear=float(onp.sum(sv)),
                hankel_effective_rank=float(
                    onp.sum(sv) ** 2 / max(float(onp.sum(sv ** 2)), 1e-300))
                if sv.size else 0.0)


def summarize(core, a=None, a_eff=None, n_impulse=256, n_hankel=48):
    out = {}
    out.update(poles(core, a=a, a_eff=a_eff))
    out.update(direct_history_decomposition(core, n=n_impulse))
    out.update(dc_response(core))
    out.update(hankel_singular_values(core, n=n_hankel))
    out["scope"] = ("single layer linear core; NOT the nonlinear network "
                    "transfer")
    return out


def core_from_module(module, variables):
    """Extract the realized core from the BOUND, EXECUTED module.

    This runs the module's own `setup()` through `module.apply`, so the
    coefficients are exactly those the forward pass uses. In particular
    `clip_eigs` is honoured.

    The previous implementation rebuilt Lambda from the raw parameters and
    ignored clipping. With clip_eigs=True and raw real poles past the
    boundary that reported max|a_bar| = 1.0175 - apparent instability - for a
    model whose realized max|a_bar| was 0.99999977, and its impulse response
    was wrong by 1.4e-2. Configuration is now read from the module, never
    re-derived.
    """
    def _read(m):
        C_tilde = m.C_tilde
        D = m.D
        conj = m.conj_sym
        if hasattr(m, "mechanism"):
            c = m.coefficients()
            Lam = m.Lambda                      # already clipped by setup()
            B_tilde = m.B[..., 0] + 1j * m.B[..., 1]
            step = m.step_rescale * np.exp(m.log_step[:, 0])
            a = step * Lam
            return (c["a_bar"], c["b_bar"], c["d_x"], C_tilde, D, conj, a,
                    c["a_eff"])
        # plain S5SSM: the realized ZOH coefficients, no tied feedthrough
        Lam = m.Lambda
        step = m.step_rescale * np.exp(m.log_step[:, 0])
        a = step * Lam
        zeros = np.zeros_like(m.B[..., 0] + 1j * m.B[..., 1])
        return (m.Lambda_bar, m.B_bar, zeros, C_tilde, D, conj, a, a)

    a_bar, b_bar, d_x, C_tilde, D, conj, a, a_eff = module.apply(
        variables, method=_read)
    return linear_core(a_bar, b_bar, d_x, C_tilde, D, conj), a, a_eff
