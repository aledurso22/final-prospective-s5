"""Positive component realization of every allowed (T, rho) pair.

`s5/constrained_response.py` realizes the superseded (gamma_n, rho) freedom at
a FIXED horizon T = 5. Learning a per-mode T needs a different map, because T
is now a coordinate rather than a constant, so this is a new module and the
historical one is left untouched.

For any T_i > 0 and 0 < rho_i < 1, with one fixed reference capacitance
c_* > 0 (c_* = 7.5, for continuity with the earlier reference):

    c_s,i = c_d,i = c_*
    G_s,i = G_d,i = c_* / (T_i rho_i)
    h_i   = G_s,i sqrt(1 - rho_i)
    g_L,i = G_s,i - h_i = c_* / (T_i (1 + sqrt(1 - rho_i)))   > 0

These give IDENTICALLY

    kappa_i      = G_s - h^2/G_d = G_s rho = c_*/T_i
    T_i(circuit) = c_s/kappa     = T_i               <- the requested horizon
    tau_d,i      = c_d/G_d       = T_i rho_i
    gamma_phys,i = G_s tau_d/kappa = T_i
    M_phys,i     = T_i tau_d     = T_i^2 rho_i
    mu_i         = M_phys/gamma_phys = T_i rho_i     <- the DERIVED mass

so dividing the WHOLE physical equation by gamma_phys,i = T_i reproduces the
implemented normalized law with gamma_n = 1 and mu_i = T_i rho_i.

At T = 5, rho = 3/4 this returns G_s = G_d = 2 and g_L = h = 1: exactly the
original symmetric reference, with equal leak and axial conductances. Learning
T therefore departs from that equality while keeping every displayed tie, and
the normalization divisor is now per-mode rather than a single fixed gamma_ref.
That is the substantive difference from the fixed-horizon map, and it is why
physical and normalized coefficients are reported separately below.

Scope, unchanged and repeated because it constrains claims rather than code:
the passive reciprocal plant with constant conductances and additive-current
coupling supplies these tied coefficients; it does NOT prescribe the
prospective closed-loop source. Fitting T and rho by BPTT between independent
sequences is an optimization of a declared model family. It is not a claim of
biological plasticity of these quantities, and it does not inherit NLA's local
learning theorem.
"""

import numpy as onp

#: reference capacitance, shared by both compartments. Fixed, not learned.
C_STAR = 7.5
#: the reference horizon about which T is parameterized multiplicatively
T_REFERENCE = 5.0
#: the symmetric-reference rho this study initializes at
RHO_SYMMETRIC = 0.75


def component_reconstruction(T, rho):
    """Positive components realizing (T, rho). Arrays or scalars."""
    t = onp.asarray(T, dtype=float)
    r = onp.asarray(rho, dtype=float)
    root = onp.sqrt(1.0 - r)
    G_s = C_STAR / (t * r)
    h = G_s * root
    g_L = C_STAR / (t * (1.0 + root))      # equals G_s - h, stably evaluated
    c_s = onp.broadcast_to(onp.asarray(C_STAR), onp.broadcast(t, r).shape)
    kappa = G_s - h ** 2 / G_s
    tau_d = c_s / G_s
    return dict(c_s=c_s.copy(), c_d=c_s.copy(), G_s=G_s, G_d=G_s, h=h,
                g_L=g_L, kappa=kappa, T=c_s / kappa, tau_d=tau_d,
                gamma_phys=G_s * tau_d / kappa,
                M_phys=(c_s / kappa) * tau_d,
                mu=t * r)


def check_identities(T, rho, tol=1e-9):
    """Every tie of the component map, as max absolute residuals."""
    t = onp.asarray(T, dtype=float)
    r = onp.asarray(rho, dtype=float)
    c = component_reconstruction(t, r)
    res = dict(
        kappa_equals_cstar_over_T=onp.max(onp.abs(c["kappa"] - C_STAR / t)),
        T_reproduced=onp.max(onp.abs(c["T"] - t)),
        tau_d=onp.max(onp.abs(c["tau_d"] - t * r)),
        gamma_phys_equals_T=onp.max(onp.abs(c["gamma_phys"] - t)),
        M_phys=onp.max(onp.abs(c["M_phys"] - t ** 2 * r)),
        mu_derived=onp.max(onp.abs(c["mu"] - t * r)),
        mu_from_physical=onp.max(onp.abs(c["M_phys"] / c["gamma_phys"]
                                         - c["mu"])),
        rho_recovered=onp.max(onp.abs(c["kappa"] / c["G_s"] - r)),
        g_L_equals_Gs_minus_h=onp.max(onp.abs(c["g_L"] - (c["G_s"] - c["h"]))),
    )
    res["all_positive"] = bool(
        onp.all(c["c_s"] > 0) and onp.all(c["G_s"] > 0)
        and onp.all(c["h"] > 0) and onp.all(c["g_L"] > 0))
    res["max_residual"] = float(max(v for k, v in res.items()
                                    if k != "all_positive"))
    res["passed"] = bool(res["all_positive"] and res["max_residual"] < tol)
    return res


def summarize(T, rho):
    """Distribution summary of the learned response and its components.

    NORMALIZED and PHYSICAL coefficients are kept distinct: `mu` is the
    normalized mass actually implemented, `M_phys` and `gamma_phys` are the
    unit-sample physical quantities before the whole-equation division.
    """
    t = onp.asarray(T, dtype=float).ravel()
    r = onp.asarray(rho, dtype=float).ravel()
    c = component_reconstruction(t, r)

    def st(v):
        v = onp.asarray(v, dtype=float).ravel()
        return dict(min=float(v.min()), median=float(onp.median(v)),
                    max=float(v.max()), mean=float(v.mean()),
                    std=float(v.std()))

    return dict(n_modes=int(t.size), T=st(t), rho=st(r),
                normalized=dict(gamma_n=1.0, mu=st(c["mu"])),
                physical=dict(kappa=st(c["kappa"]), tau_d=st(c["tau_d"]),
                              gamma_phys=st(c["gamma_phys"]),
                              M_phys=st(c["M_phys"]), G_s=st(c["G_s"]),
                              h=st(c["h"]), g_L=st(c["g_L"]),
                              c_s=float(C_STAR)),
                reference=dict(c_star=C_STAR, T_reference=T_REFERENCE,
                               rho_symmetric=RHO_SYMMETRIC,
                               note="at T=5, rho=3/4: G_s=G_d=2, g_L=h=1"))
