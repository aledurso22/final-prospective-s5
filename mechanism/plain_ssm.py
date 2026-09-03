"""A. PLAIN SSM - the ordinary dynamics, no prospective term anywhere.

    tau ds/dt = (A - I) s + B x

Exact ZOH discretization, per mode:

    s_k = lam_i s_{k-1} + bbar_i x_k
"""

import numpy as np


class PlainSSM:
    mechanism = "plain_ssm"
    label = "A. Plain SSM"

    def __init__(self, system):
        self.sys = system

    def discrete_poles(self):
        return self.sys.lam.copy()

    def continuous_poles(self):
        return self.sys.plain_continuous_poles.copy()

    def run(self, x, s0=None):
        sys = self.sys
        s = np.zeros(sys.n_modes) if s0 is None else np.array(s0, float)
        out = np.empty((len(x), sys.n_modes))
        for k, xk in enumerate(x):
            s = sys.lam * s + sys.bbar * xk
            out[k] = s
        return out
