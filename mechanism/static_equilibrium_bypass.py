"""D. MATCHED STATIC EQUILIBRIUM BYPASS - the hostile control.

Memory modes evolve with the ORIGINAL plain dynamics. Tracking modes are
replaced by the instantaneous equilibrium, with no dynamics at all:

    memory   : s_k = lam_i s_{k-1} + bbar_i x_k
    tracking : s_k = K_i x_k

Theory predicts a fully prospective tracking mode reduces, in forced
response, to exactly this plus a decaying initialization transient. If
`ProjectedStatePC` and this model agree, the prospective machinery is
buying nothing over a static bypass, and that is the finding.
"""

import numpy as np


class StaticEquilibriumBypass:
    mechanism = "static_equilibrium_bypass"
    label = "D. Static equilibrium bypass"

    def __init__(self, system):
        self.sys = system

    def discrete_poles(self):
        sys = self.sys
        return np.where(sys.tracking_mask, 0.0, sys.lam)

    def continuous_poles(self):
        sys = self.sys
        return np.where(sys.tracking_mask, -np.inf, sys.plain_continuous_poles)

    def run(self, x, s0=None):
        sys = self.sys
        s = np.zeros(sys.n_modes) if s0 is None else np.array(s0, float)
        out = np.empty((len(x), sys.n_modes))
        for k, xk in enumerate(x):
            s = np.where(sys.tracking_mask, sys.K * xk, sys.lam * s + sys.bbar * xk)
            out[k] = s
        return out
