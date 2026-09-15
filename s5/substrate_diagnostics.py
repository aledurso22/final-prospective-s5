"""Read-only response extraction for the EXECUTED Stage 2 substrate.

Why this file exists instead of `s5/gp_diagnostics.py`
------------------------------------------------------
`gp_diagnostics.core_from_module` dispatches on a `mechanism` attribute.
`SubstrateSSM` has no `mechanism`; it has `response` and `coefficients()`. A
dispatch miss there does not raise - it falls through to inherited native S5
coefficients, so it would silently report the WRONG dynamics for four of the
five arms. Everything here reads the bound, executed module and is checked
against an actual forward impulse call before any of it is interpreted.

Nothing in this module trains, updates, mutates or writes. It takes bound
variables and returns numbers.

Impulse conventions (conjugate-symmetric realization, R(Z) = 2 Re(C_tilde Z))
----------------------------------------------------------------------------
lag 0 means the output after consuming the CURRENT held token.

    one_tap          K_0 = R(B_bar) + diag(D)
                     K_l = R(A_bar^l B_bar)                       l >= 1
    alpha_p_two_tap  K_0 = R(B_plus) + diag(D)
                     K_l = R(A_bar^(l-1) (A_bar B_plus + B_minus)) l >= 1
    gp_fixed_m0      K_0 = R(b_bar + d_x) + diag(D)
                     K_l = R(a_bar^l b_bar)                        l >= 1
    gp_fixed_mass    K_l = R( [A_block^l B_block]_s ), native D at l = 0 only

"Current tap" here includes the within-interval dynamic update. It is NOT a
synonym for continuous-time algebraic feedthrough.

Frequency response
------------------
For real input and y_k = 2 Re(C h_k), K_l is real and

    H(w) = sum_l K_l e^{-i w l}
         = C (I - A u)^-1 B  +  conj( C (I - A ubar)^-1 B ),  u = e^{-i w}

which is NOT 2 Re(C (I - A u)^-1 B): conj(T(e^{+iw})) != conj(T(e^{-iw})).
Taking 2 Re of a complex-frequency response is the specific error the brief
warns about, so the two terms are formed explicitly.
"""

import numpy as onp

RESPONSES = ("one_tap", "alpha_p_two_tap", "gp_fixed_m0", "gp_fixed_mass")


def _np(x):
    return onp.asarray(x)


def read_core(module, variables, layer):
    """Every realized quantity of one layer's recurrent core, read from the
    BOUND module. Raises rather than guessing if the response is unknown."""
    def _read(m):
        seq = m.encoder.layers[layer].seq
        Lambda_raw = seq.Lambda_re_init + 1j * seq.Lambda_im_init
        Lam, B_c, Delta = seq._native()
        B_tilde = seq.B[..., 0] + 1j * seq.B[..., 1]
        return dict(response=seq.response, input_gain=seq.input_gain,
                    clip_eigs=seq.clip_eigs, conj_sym=seq.conj_sym,
                    P=seq.P, H=seq.H,
                    Lambda_clipped=Lam, Lambda_raw_param=Lambda_raw,
                    B_tilde=B_tilde, B_c=B_c, Delta=Delta,
                    a=Lam * Delta, b=Delta[:, None] * B_c,
                    C_tilde=seq.C_tilde, D=seq.D,
                    coefficients=seq.coefficients(),
                    physical=dict(T=seq.physical.T, gamma=seq.physical.gamma,
                                  rho=seq.physical.rho,
                                  mass=seq.physical.mass))
    out = module.apply(variables, method=_read)
    if out["response"] not in RESPONSES:
        raise ValueError(f"unknown response {out['response']!r}; refusing to "
                         f"report dynamics that may not be the executed ones")
    return out


def impulse_matrices(core, n_lags):
    """K_l for l = 0..n_lags-1, shape (n_lags, H, H), REAL."""
    resp = core["response"]
    if resp not in RESPONSES:
        raise ValueError(
            f"unknown response {resp!r}. Refusing to fall through to another "
            f"arm's dynamics: a silent dispatch miss is the exact failure this "
            f"module exists to prevent.")
    c = core["coefficients"]
    C = _np(core["C_tilde"])
    D = _np(core["D"])
    conj = core["conj_sym"]
    fac = 2.0 if conj else 1.0

    def R(Z):
        return fac * onp.real(C @ Z)

    H = D.shape[0]
    K = onp.zeros((n_lags, H, H), dtype=onp.float64)

    if resp in ("one_tap", "alpha_p_two_tap"):
        A = _np(c["A_bar"])
        if resp == "one_tap":
            state = _np(c["B_bar"]).astype(onp.complex128)
        else:
            state = _np(c["B_plus"]).astype(onp.complex128)
            second = _np(c["B_minus"]).astype(onp.complex128)
        K[0] = R(state) + onp.diag(D)
        for l in range(1, n_lags):
            state = A[:, None] * state
            if resp == "alpha_p_two_tap" and l == 1:
                state = state + second
            K[l] = R(state)
    elif resp == "gp_fixed_m0":
        a_bar = _np(c["a_bar"]); b_bar = _np(c["b_bar"]); d_x = _np(c["d_x"])
        K[0] = R(b_bar + d_x) + onp.diag(D)
        state = b_bar.astype(onp.complex128)
        for l in range(1, n_lags):
            state = a_bar[:, None] * state
            K[l] = R(state)
    else:                                              # gp_fixed_mass
        A = _np(c["A_bar"]).astype(onp.complex128)     # (P,2,2)
        Bb = _np(c["B_bar"]).astype(onp.complex128)    # (P,2,H)
        state = Bb.copy()
        K[0] = R(state[:, 0, :]) + onp.diag(D)
        for l in range(1, n_lags):
            state = onp.einsum("pij,pjh->pih", A, state)
            K[l] = R(state[:, 0, :])
    return K


def frequency_response(core, n_freq=129):
    """Exact H(w) on a uniform grid 0..pi, shape (n_freq, H, H), COMPLEX."""
    resp = core["response"]
    if resp not in RESPONSES:
        raise ValueError(f"unknown response {resp!r}; refusing to guess")
    c = core["coefficients"]
    C = _np(core["C_tilde"]).astype(onp.complex128)
    D = _np(core["D"])
    fac = 2.0 if core["conj_sym"] else 1.0
    w = onp.linspace(0.0, onp.pi, n_freq)
    H = D.shape[0]
    out = onp.zeros((n_freq, H, H), dtype=onp.complex128)

    def term(u):
        """C (I - A u)^-1 B(u), for the diagonal and block cases."""
        if resp in ("one_tap", "alpha_p_two_tap"):
            A = _np(c["A_bar"]).astype(onp.complex128)
            if resp == "one_tap":
                Bu = _np(c["B_bar"]).astype(onp.complex128)
            else:
                Bu = (_np(c["B_plus"]).astype(onp.complex128)
                      + _np(c["B_minus"]).astype(onp.complex128) * u)
            return C @ (Bu / (1.0 - A * u)[:, None])
        if resp == "gp_fixed_m0":
            a = _np(c["a_bar"]).astype(onp.complex128)
            b = _np(c["b_bar"]).astype(onp.complex128)
            dx = _np(c["d_x"]).astype(onp.complex128)
            # s = h + d_x x, and h has transfer b/(1 - a u); d_x is lag 0
            return C @ (b / (1.0 - a * u)[:, None] + dx)
        A = _np(c["A_bar"]).astype(onp.complex128)      # (P,2,2)
        Bb = _np(c["B_bar"]).astype(onp.complex128)     # (P,2,H)
        P = A.shape[0]
        I2 = onp.eye(2, dtype=onp.complex128)
        M = I2[None] - A * u
        X = onp.linalg.solve(M, Bb)                     # (P,2,H)
        return C @ X[:, 0, :]

    for k, wk in enumerate(w):
        u = onp.exp(-1j * wk)
        ub = onp.exp(+1j * wk)
        # H(w) = T(e^{-iw}) + conj(T(e^{+iw})); NOT 2*Re(T(e^{-iw}))
        out[k] = (term(u) + onp.conj(term(ub))) * (fac / 2.0) + onp.diag(D)
    return w, out


def band_energy(K, bands):
    """Absolute and fractional response energy per lag band.

    Fractions alone can hide a collapse in total response, so both are
    returned, and the truncation of the window is reported by the caller.
    """
    tot = float(onp.sum(K ** 2))
    rows = []
    for lo, hi in bands:
        sl = K[lo:hi + 1]
        e = float(onp.sum(sl ** 2))
        rows.append(dict(lo=lo, hi=hi, energy=e,
                         fraction=(e / tot if tot > 0 else float("nan")),
                         rms=float(onp.sqrt(onp.mean(sl ** 2))) if sl.size else 0.0))
    return dict(total_energy=tot, bands=rows)


def pole_summary(core):
    """Native vs effective poles, decay times in frames, frequencies, clipping."""
    a = _np(core["a"])
    Lam_raw = _np(core["Lambda_raw_param"])
    Lam_clip = _np(core["Lambda_clipped"])
    c = core["coefficients"]
    resp = core["response"]
    if resp in ("one_tap", "alpha_p_two_tap"):
        a_bar = _np(c["A_bar"])
        a_eff = onp.log(a_bar)
    elif resp == "gp_fixed_m0":
        a_bar = _np(c["a_bar"]); a_eff = _np(c["a_eff"])
    else:
        A = _np(c["A_bar"])
        ev = onp.linalg.eigvals(A)                       # (P,2) discrete
        a_bar = ev
        with onp.errstate(divide="ignore"):
            a_eff = onp.log(ev + 0j)
    mag = onp.abs(a_bar)
    with onp.errstate(divide="ignore", invalid="ignore"):
        tau = onp.where(mag < 1.0, -1.0 / onp.log(onp.maximum(mag, 1e-300)),
                        onp.inf)
    n_clipped = int(onp.sum(Lam_raw.real > -1e-4)) if core["clip_eigs"] else 0
    return dict(
        n_modes=int(a.size),
        n_raw_poles_clipped=n_clipped,
        clipped_fraction=n_clipped / float(a.size),
        native_pole_re=a.real.tolist(), native_pole_im=a.imag.tolist(),
        eff_pole_re=onp.real(a_eff).ravel().tolist(),
        eff_pole_im=onp.imag(a_eff).ravel().tolist(),
        abs_a_bar_max=float(onp.max(mag)),
        decay_frames_median=float(onp.median(tau[onp.isfinite(tau)]))
        if onp.any(onp.isfinite(tau)) else float("inf"),
        decay_frames_max=float(onp.max(tau[onp.isfinite(tau)]))
        if onp.any(onp.isfinite(tau)) else float("inf"),
        osc_freq_cycles_per_frame=(onp.abs(onp.imag(a_eff)).ravel()
                                   / (2 * onp.pi)).tolist(),
        delta_min=float(onp.min(_np(core["Delta"]))),
        delta_max=float(onp.max(_np(core["Delta"]))),
        lambda_clip_active=bool(n_clipped > 0),
        lambda_raw_vs_clipped_max_shift=float(
            onp.max(onp.abs(Lam_raw - Lam_clip))),
    )


def counterfactual_one_tap(core, n_lags):
    """The ONE-TAP law evaluated on this checkpoint's own learned weights.

    Lambda, Delta, B_tilde, C_tilde, D, the input gain and the clipping are all
    held exactly as trained; only the recurrent response law is swapped. This
    is an UNTRAINED counterfactual response calculation - not a new accuracy
    score and not a trained baseline. It separates the direct effect of the
    equation from coadaptation of the learned weights.
    """
    from .gp_coefficients import phi1
    a = _np(core["a"]); b = _np(core["b"])
    C = _np(core["C_tilde"]); D = _np(core["D"])
    fac = 2.0 if core["conj_sym"] else 1.0
    a_bar = onp.exp(a)
    z = a.astype(onp.complex128)
    phi = onp.where(onp.abs(z) < 1e-8, 1.0 + z / 2.0, (onp.exp(z) - 1.0) / z)
    b_bar = phi[:, None] * b
    H = D.shape[0]
    K = onp.zeros((n_lags, H, H))
    state = b_bar.astype(onp.complex128)
    K[0] = fac * onp.real(C @ state) + onp.diag(D)
    for l in range(1, n_lags):
        state = a_bar[:, None] * state
        K[l] = fac * onp.real(C @ state)
    return K


def spectral_radius(core):
    """max |discrete pole| of the executed core, for truncation bounds."""
    c = core["coefficients"]
    resp = core["response"]
    if resp in ("one_tap", "alpha_p_two_tap"):
        return float(onp.max(onp.abs(_np(c["A_bar"]))))
    if resp == "gp_fixed_m0":
        return float(onp.max(onp.abs(_np(c["a_bar"]))))
    return float(onp.max(onp.abs(onp.linalg.eigvals(_np(c["A_bar"])))))


def truncation_tail_bound(K, rho):
    """Bound on sum_{l>=N} |K_l| given a geometric decay rate rho.

    ||K_N|| * rho/(1-rho). Reported with every windowed statistic: at
    Delta ~ 1e-3 the impulse of these models has NOT decayed by lag 4096, so a
    window is never assumed negligible.
    """
    if rho >= 1.0:
        return float("inf")
    last = float(onp.sqrt(onp.sum(K[-1] ** 2)))
    return last * rho / (1.0 - rho)
