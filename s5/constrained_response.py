"""Positive component realization of every allowed (gamma_n, rho) pair.

The learned response is two real numbers per stored complex mode, gamma_n > 0
and rho in (0, 1). This module exhibits, for each such pair, a set of POSITIVE
capacitances and conductances in the existing reciprocal, equal-leak,
additive-current operating-point model that realizes it exactly:

    kappa_0   = 3/2      reference conductance unit
    gamma_ref = 5        FIXED whole-equation normalization
    T         = 5        FIXED published-comparison horizon (never learned)

    c_s   = T kappa_0            = 7.5
    c_d   = gamma_ref gamma_n kappa_0 = 7.5 gamma_n
    G_s   = G_d = kappa_0 / rho
    h     = G_s sqrt(1 - rho)
    g_L   = G_s - h = kappa_0 / (1 + sqrt(1 - rho))   > 0

These give IDENTICALLY

    kappa      = G_s - h^2/G_d = G_s rho = kappa_0
    T          = c_s/kappa     = 5                    (fixed, as required)
    tau_d      = c_d/G_d       = gamma_ref gamma_n rho
    gamma_phys = G_s tau_d/kappa = gamma_ref gamma_n
    M_phys     = T tau_d
    mu         = M_phys/gamma_ref = T gamma_n rho     (the DERIVED mass)

so dividing the WHOLE physical equation by the FIXED gamma_ref reproduces

    mu s'' + gamma_n s' + r_n + T r_n' = 0.

At gamma_n = 1, rho = 3/4 the components are c_s = c_d = 7.5, h = g_L = 1,
G_s = G_d = 2: the original symmetric reference. Learning therefore departs
from equality of the two capacitances, and from equality of leak and axial
conductance, while keeping the displayed relationships and the fixed horizon.
Those equalities were initialization and modeling assumptions, not laws.

Scope, stated plainly: this is a family WITHIN the previously declared
operating-point model with its prospective-source and learned additive-current
convention. A positive component manifold does not turn arbitrary complex S5
feedback, or BPTT, into a biological circuit or a plasticity rule.
"""

import numpy as onp

KAPPA_0 = 1.5
GAMMA_REF = 5.0
T_HORIZON = 5.0


def component_reconstruction(gamma_n, rho):
    """Positive components realizing (gamma_n, rho). Arrays or scalars."""
    g_n = onp.asarray(gamma_n, dtype=float)
    r = onp.asarray(rho, dtype=float)
    root = onp.sqrt(1.0 - r)
    G_s = KAPPA_0 / r
    h = G_s * root
    g_L = KAPPA_0 / (1.0 + root)          # equals G_s - h, stably evaluated
    c_s = onp.broadcast_to(onp.asarray(T_HORIZON * KAPPA_0), g_n.shape).copy()
    c_d = GAMMA_REF * g_n * KAPPA_0
    kappa = G_s - h ** 2 / G_s
    tau_d = c_d / G_s
    return dict(c_s=c_s, c_d=c_d, G_s=G_s, G_d=G_s, h=h, g_L=g_L,
                kappa=kappa, T=c_s / kappa, tau_d=tau_d,
                gamma_phys=G_s * tau_d / kappa, M_phys=(c_s / kappa) * tau_d,
                mu=T_HORIZON * g_n * r)


def check_identities(gamma_n, rho, tol=1e-9):
    """Every tie of the component map, as a dict of max absolute residuals."""
    g_n = onp.asarray(gamma_n, dtype=float)
    r = onp.asarray(rho, dtype=float)
    c = component_reconstruction(g_n, r)
    res = dict(
        kappa_equals_kappa0=onp.max(onp.abs(c["kappa"] - KAPPA_0)),
        T_fixed=onp.max(onp.abs(c["T"] - T_HORIZON)),
        tau_d=onp.max(onp.abs(c["tau_d"] - GAMMA_REF * g_n * r)),
        gamma_phys=onp.max(onp.abs(c["gamma_phys"] - GAMMA_REF * g_n)),
        mu_derived=onp.max(onp.abs(c["mu"] - T_HORIZON * g_n * r)),
        mu_from_physical=onp.max(onp.abs(c["M_phys"] / GAMMA_REF - c["mu"])),
        g_L_equals_Gs_minus_h=onp.max(onp.abs(c["g_L"] - (c["G_s"] - c["h"]))),
    )
    res["all_positive"] = bool(
        onp.all(c["c_s"] > 0) and onp.all(c["c_d"] > 0)
        and onp.all(c["G_s"] > 0) and onp.all(c["h"] > 0)
        and onp.all(c["g_L"] > 0))
    res["max_residual"] = float(max(v for k, v in res.items()
                                    if k != "all_positive"))
    res["passed"] = bool(res["all_positive"] and res["max_residual"] < tol)
    return res


def summarize(gamma_n, rho):
    """Distribution summary of the learned response and its components."""
    g_n = onp.asarray(gamma_n, dtype=float).ravel()
    r = onp.asarray(rho, dtype=float).ravel()
    c = component_reconstruction(g_n, r)

    def st(v):
        v = onp.asarray(v, dtype=float).ravel()
        return dict(min=float(v.min()), median=float(onp.median(v)),
                    max=float(v.max()), mean=float(v.mean()),
                    std=float(v.std()))

    return dict(n_modes=int(g_n.size), gamma_n=st(g_n), rho=st(r),
                mu=st(c["mu"]), tau_d=st(c["tau_d"]), c_d=st(c["c_d"]),
                G_s=st(c["G_s"]), h=st(c["h"]), g_L=st(c["g_L"]),
                gamma_phys=st(c["gamma_phys"]), M_phys=st(c["M_phys"]))
