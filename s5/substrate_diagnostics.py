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
    gp_rho_prospin   K_0 = R( [B_plus]_s ) + diag(D)
                     K_l = R( [A_bar^(l-1)(A_bar B_plus + B_minus)]_s ) l >= 1

The last one is the reason this module does not simply add the combined
response to the mass-like set: it is a TWO-tap block law, and extracting it
with the one-tap block formula would silently return a different response that
still looks plausible. Its frequency response is

    H(w) = C_block (I - A_bar u)^-1 (B_plus + B_minus u) + D,   u = e^{-i w}

assembled from conjugate partners exactly as the other arms are.

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

RESPONSES = ("one_tap", "alpha_p_two_tap", "gp_fixed_m0", "gp_fixed_mass",
             # same (s, v) block realization as gp_fixed_mass, with gamma_n = 1
             # and a per-mode learned rho
             "gp_rho", "gp_rho_frozen",
             # same ONE-tap (s, v) block, with a learned per-mode horizon T_i
             "gp_rho_T", "gp_rho_T_fixed",
             # the (s, v) block driven by x + T_in x_dot: a TWO-tap block law
             "gp_rho_prospin",
             # memoryless by construction: K_0 only
             "prospective_recurrence",
             # paired continuation study (16 Sep 2026): a DIAGONAL two-tap law
             # with a per-mode input horizon, and the two-tap (s, v) block law
             # with learned (rho, T) and the same per-mode input horizon
             "rawat_learned_input", "sgp_learned_input")
#: diagonal two-tap laws, extracted exactly as alpha-P-S5
_DIAG_TWO_TAP = ("alpha_p_two_tap", "rawat_learned_input")

#: the responses whose block realization is the ONE-tap (s, v) mass block
#: The learned-timescale arms belong here because their realization IS the
#: one-tap (s, v) block - only the horizon inside A and B changed, and the
#: coefficients dict carries the EXECUTED T. They are listed explicitly rather
#: than matched by prefix, so a future response cannot join by accident.
_MASS_LIKE = ("gp_fixed_mass", "gp_rho", "gp_rho_frozen",
              "gp_rho_T", "gp_rho_T_fixed")
#: the TWO-tap block law; extracted separately, never through _MASS_LIKE
_MASS_TWO_TAP = ("gp_rho_prospin", "sgp_learned_input")


def _np(x):
    return onp.asarray(x)


def _executed_response(seq):
    """The response quantities the bound module actually used.

    For a fixed-coefficient arm these are the static reference values. For a
    rho-only arm rho is learned and T is the fixed horizon. For a
    learned-timescale arm BOTH are per-mode learned arrays and the derived mass
    is mu_i = rho_i T_i. Returned as a dict of arrays or scalars, never as the
    dataclass field, so a caller cannot read a stale constant by accident.
    """
    from .rawat_s5 import RHO_ONLY_RESPONSES, T_RESPONSES
    if seq.response == "sgp_learned_input":
        T, rho = seq.recurrent_T(), seq.recurrent_rho()
        return dict(kind="stable_generalized_learned_rho_T_T_in", T=T,
                    rho=rho, gamma_n=1.0, mu=T * rho,
                    T_in=seq.input_horizon(), T_is_per_mode=True)
    if seq.response == "rawat_learned_input":
        return dict(kind="rawat_learned_T_in", T_in=seq.input_horizon(),
                    T_is_per_mode=True)
    if seq.response in T_RESPONSES:
        T = seq.response_timescale()
        rho = seq.rho_only()
        return dict(kind="learned_T_and_rho", T=T, rho=rho, gamma_n=1.0,
                    mu=T * rho, T_is_per_mode=True)
    if seq.response in RHO_ONLY_RESPONSES:
        rho = seq.rho_only()
        return dict(kind="learned_rho_fixed_T", T=seq.physical.T, rho=rho,
                    gamma_n=1.0, mu=seq.physical.T * rho,
                    T_is_per_mode=False)
    return dict(kind="fixed_reference", T=seq.physical.T,
                rho=seq.physical.rho, gamma_n=seq.physical.gamma,
                mu=seq.physical.mass, T_is_per_mode=False)


def read_core(module, variables, layer):
    """Every realized quantity of one layer's recurrent core, read from the
    BOUND module. Raises rather than guessing if the response is unknown."""
    def _read(m):
        seq = m.encoder.layers[layer].seq
        # R1: Lambda_re_init / Lambda_im_init are the INITIALIZER fields, i.e.
        # static configuration. The TRAINED raw pole is the parameter
        # seq.Lambda_re + 1j*seq.Lambda_im. Reading the initializer made
        # trained clipping counts and raw-vs-clipped shifts wrong even when the
        # executed impulse response was right.
        Lambda_raw = seq.Lambda_re + 1j * seq.Lambda_im
        Lambda_init = seq.Lambda_re_init + 1j * seq.Lambda_im_init
        Lam, B_c, Delta = seq._native()
        B_tilde = seq.B[..., 0] + 1j * seq.B[..., 1]
        return dict(response=seq.response, input_gain=seq.input_gain,
                    clip_eigs=seq.clip_eigs, conj_sym=seq.conj_sym,
                    P=seq.P, H=seq.H,
                    Lambda_clipped=Lam, Lambda_raw_param=Lambda_raw,
                    Lambda_initializer_field=Lambda_init,
                    B_tilde=B_tilde, B_c=B_c, Delta=Delta,
                    a=Lam * Delta, b=Delta[:, None] * B_c,
                    C_tilde=seq.C_tilde, D=seq.D,
                    coefficients=seq.coefficients(),
                    # the STATIC dataclass reference, kept for traceability
                    physical=dict(T=seq.physical.T, gamma=seq.physical.gamma,
                                  rho=seq.physical.rho,
                                  mass=seq.physical.mass),
                    # the EXECUTED response, which for a learned-timescale arm
                    # is NOT the static field above. Reporting `physical.T` for
                    # those arms would mislabel both the horizon and the
                    # derived mass.
                    executed_response=_executed_response(seq))
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

    if resp in ("one_tap",) + _DIAG_TWO_TAP:
        A = _np(c["A_bar"])
        if resp == "one_tap":
            state = _np(c["B_bar"]).astype(onp.complex128)
        else:
            state = _np(c["B_plus"]).astype(onp.complex128)
            second = _np(c["B_minus"]).astype(onp.complex128)
        K[0] = R(state) + onp.diag(D)
        for l in range(1, n_lags):
            state = A[:, None] * state
            if resp in _DIAG_TWO_TAP and l == 1:
                state = state + second
            K[l] = R(state)
    elif resp == "prospective_recurrence":
        # s_k = J^-1 b x_k: a single lag-zero tap and nothing after it.
        K[0] = R(_np(c["s_gain"]).astype(onp.complex128)) + onp.diag(D)
        return K
    elif resp == "gp_fixed_m0":
        a_bar = _np(c["a_bar"]); b_bar = _np(c["b_bar"]); d_x = _np(c["d_x"])
        K[0] = R(b_bar + d_x) + onp.diag(D)
        state = b_bar.astype(onp.complex128)
        for l in range(1, n_lags):
            state = a_bar[:, None] * state
            K[l] = R(state)
    elif resp in _MASS_TWO_TAP:
        A = _np(c["A_bar"]).astype(onp.complex128)      # (P,2,2)
        Bp = _np(c["B_plus"]).astype(onp.complex128)    # (P,2,H)
        Bm = _np(c["B_minus"]).astype(onp.complex128)
        state = Bp.copy()
        K[0] = R(state[:, 0, :]) + onp.diag(D)
        for l in range(1, n_lags):
            state = onp.einsum("pij,pjh->pih", A, state)
            if l == 1:
                state = state + Bm          # the delayed tap enters ONCE
            K[l] = R(state[:, 0, :])
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
        if resp in ("one_tap",) + _DIAG_TWO_TAP:
            A = _np(c["A_bar"]).astype(onp.complex128)
            if resp == "one_tap":
                Bu = _np(c["B_bar"]).astype(onp.complex128)
            else:
                Bu = (_np(c["B_plus"]).astype(onp.complex128)
                      + _np(c["B_minus"]).astype(onp.complex128) * u)
            return C @ (Bu / (1.0 - A * u)[:, None])
        if resp == "prospective_recurrence":
            return C @ _np(c["s_gain"]).astype(onp.complex128)
        if resp == "gp_fixed_m0":
            a = _np(c["a_bar"]).astype(onp.complex128)
            b = _np(c["b_bar"]).astype(onp.complex128)
            dx = _np(c["d_x"]).astype(onp.complex128)
            # s = h + d_x x, and h has transfer b/(1 - a u); d_x is lag 0
            return C @ (b / (1.0 - a * u)[:, None] + dx)
        A = _np(c["A_bar"]).astype(onp.complex128)      # (P,2,2)
        if resp in _MASS_TWO_TAP:
            Bb = (_np(c["B_plus"]).astype(onp.complex128)
                  + _np(c["B_minus"]).astype(onp.complex128) * u)
        else:
            Bb = _np(c["B_bar"]).astype(onp.complex128)  # (P,2,H)
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


def continuous_poles(core):
    """The ACTUAL continuous poles of the executed core.

    R1: `log(discrete eigenvalue)` returns a principal-branch value, so it
    ALIASES any continuous frequency with |Im| > pi. It is therefore not the
    original continuous pole and is never reported as one. Per response:

        one_tap / alpha_p_two_tap : a = Delta * Lambda_clipped
        gp_fixed_m0               : a_eff
        gp_fixed_mass             : eigenvalues of the executed CONTINUOUS
                                    block generator A (not of A_bar)
    """
    resp = core["response"]
    c = core["coefficients"]
    if resp in ("one_tap",) + _DIAG_TWO_TAP:
        return _np(core["a"]).ravel()
    if resp == "gp_fixed_m0":
        return _np(c["a_eff"]).ravel()
    if resp == "prospective_recurrence":
        return onp.zeros(0, dtype=complex)        # memoryless: no poles
    return onp.linalg.eigvals(_np(c["A"])).ravel()


def discrete_poles(core):
    """Discrete eigenvalues of the executed transition."""
    resp = core["response"]
    c = core["coefficients"]
    if resp in ("one_tap",) + _DIAG_TWO_TAP:
        return _np(c["A_bar"]).ravel()
    if resp == "gp_fixed_m0":
        return _np(c["a_bar"]).ravel()
    if resp == "prospective_recurrence":
        return onp.zeros(0, dtype=complex)
    return onp.linalg.eigvals(_np(c["A_bar"])).ravel()


def pole_summary(core):
    """Trained raw vs executed clipped poles, continuous and discrete kept
    SEPARATE, with clipping measured on the trained parameter."""
    raw = _np(core["Lambda_raw_param"])              # TRAINED
    init = _np(core["Lambda_initializer_field"])     # static config
    clipped = _np(core["Lambda_clipped"])            # executed
    cont = continuous_poles(core)
    disc = discrete_poles(core)
    mag = onp.abs(disc)
    with onp.errstate(divide="ignore", invalid="ignore"):
        tau = onp.where(mag < 1.0,
                        -1.0 / onp.log(onp.maximum(mag, 1e-300)), onp.inf)
    n_clipped = int(onp.sum(raw.real > -1e-4)) if core["clip_eigs"] else 0
    fin = tau[onp.isfinite(tau)]
    # a continuous |Im| above pi per unit interval is ALIASED by the discrete
    # angle; flagged rather than silently relabelled
    aliased = int(onp.sum(onp.abs(onp.imag(cont)) > onp.pi))
    return dict(
        n_modes=int(raw.size),
        n_raw_poles_clipped=n_clipped,
        clipped_fraction=n_clipped / float(raw.size),
        clip_active=bool(n_clipped > 0),
        trained_raw_re=raw.real.tolist(), trained_raw_im=raw.imag.tolist(),
        clipped_re=clipped.real.tolist(), clipped_im=clipped.imag.tolist(),
        raw_minus_clipped_max=float(onp.max(onp.abs(raw - clipped))),
        trained_minus_initializer_max=float(onp.max(onp.abs(raw - init))),
        continuous_pole_re=onp.real(cont).tolist(),
        continuous_pole_im=onp.imag(cont).tolist(),
        continuous_freq_cycles_per_frame=(onp.abs(onp.imag(cont))
                                          / (2 * onp.pi)).tolist(),
        n_continuous_modes_aliased_by_discrete_angle=aliased,
        discrete_abs=mag.tolist(),
        discrete_angle=onp.angle(disc).tolist(),
        discrete_angle_is_aliased_frequency=True,
        abs_a_bar_max=float(onp.max(mag)),
        decay_frames_median=float(onp.median(fin)) if fin.size else float("inf"),
        decay_frames_max=float(onp.max(fin)) if fin.size else float("inf"),
        delta_min=float(onp.min(_np(core["Delta"]))),
        delta_max=float(onp.max(_np(core["Delta"]))),
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
    """max |discrete pole| of the executed core. Descriptive only. 0 if none.

    R2: this is NOT used to bound an impulse tail. The formula
    ||K_last|| * rho/(1-rho) is invalid for a multimode output: modes can
    cancel exactly at the last measured sample and not at the next, e.g. the
    scalar two-mode response K_l = (1/2)^l - 2(1/4)^l has K_1 = 0 while
    K_2 = 1/8, so a two-sample window would report a zero tail for a nonzero
    one. A nonnormal block also need not contract in Euclidean norm at its
    spectral radius. The invalid bound has been removed rather than loosened.
    """
    d = discrete_poles(core)
    return float(onp.max(onp.abs(d))) if d.size else 0.0


def frequency_window_remainder(core, n_lags, w):
    """EXACT frequency-domain remainder of a finite impulse window.

    H(w) = sum_{l<N} K_l e^{-i w l} + remainder(w), computed in closed form
    from the resolvent rather than estimated. For a one-tap realization
    K_l = C A^l B the remainder after l = 0..N-1 is C (uA)^N (I - uA)^-1 B with
    u = e^{-iw}; native D has no tail. The two-tap lag-one drive
    (A B_plus + B_minus) and its index shift are handled explicitly, and
    complex pairs are realified exactly as in `frequency_response`.

    This replaces the invalid geometric bound: it is an identity, so it can be
    checked, and it makes a windowed frequency comparison exact instead of
    approximate.
    """
    resp = core["response"]
    if resp not in RESPONSES:
        raise ValueError(f"unknown response {resp!r}; refusing to guess")
    if resp in _MASS_TWO_TAP:
        # the block branch below is the ONE-tap block remainder; applying it
        # to a two-tap block law would return a plausible but wrong tail
        raise NotImplementedError(
            f"no exact remainder is implemented for the two-tap block law "
            f"{resp!r}; refusing to return the one-tap block formula")
    c = core["coefficients"]
    C = _np(core["C_tilde"]).astype(onp.complex128)
    fac = 2.0 if core["conj_sym"] else 1.0
    N = int(n_lags)

    def rem(u):
        if resp in ("one_tap",) + _DIAG_TWO_TAP:
            A = _np(c["A_bar"]).astype(onp.complex128)
            uA = u * A
            if resp == "one_tap":
                B = _np(c["B_bar"]).astype(onp.complex128)
                return C @ ((uA ** N / (1.0 - uA))[:, None] * B)
            G = (_np(c["A_bar"])[:, None] * _np(c["B_plus"])
                 + _np(c["B_minus"])).astype(onp.complex128)
            # sum_{l<N} state_l u^l with state_0 = B_plus,
            # state_l = A^(l-1) G  =>  remainder = u (uA)^(N-1) (I-uA)^-1 G
            return C @ ((u * uA ** (N - 1) / (1.0 - uA))[:, None] * G)
        if resp == "prospective_recurrence":
            return onp.zeros((C.shape[0], _np(core["D"]).shape[0]),
                             dtype=onp.complex128)     # no tail at all
        if resp == "gp_fixed_m0":
            a = _np(c["a_bar"]).astype(onp.complex128)
            b = _np(c["b_bar"]).astype(onp.complex128)
            ua = u * a
            return C @ ((ua ** N / (1.0 - ua))[:, None] * b)
        A = _np(c["A_bar"]).astype(onp.complex128)          # (P,2,2)
        B = _np(c["B_bar"]).astype(onp.complex128)          # (P,2,H)
        P = A.shape[0]
        I2 = onp.eye(2, dtype=onp.complex128)
        uA = u * A
        powN = onp.array([onp.linalg.matrix_power(uA[p], N) for p in range(P)])
        X = onp.linalg.solve(I2[None] - uA, B)              # (I-uA)^-1 B
        Y = onp.einsum("pij,pjh->pih", powN, X)
        return C @ Y[:, 0, :]

    w = onp.atleast_1d(w)
    H = _np(core["D"]).shape[0]
    out = onp.zeros((w.size, H, H), dtype=onp.complex128)
    for k, wk in enumerate(w):
        out[k] = (rem(onp.exp(-1j * wk))
                  + onp.conj(rem(onp.exp(+1j * wk)))) * (fac / 2.0)
    return out
