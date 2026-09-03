"""C. PROFESSOR IDEA 2 - PROJECTED / modal state prospectivity.

Continuous:

    tau ds/dt = -s + f(s,x) + tau P_T d/dt f(s,x),     P_M + P_T = I

For a diagonal system with a MODAL projector (P_T selects whole coordinates)
the system decouples exactly, and per mode:

  memory mode   (P_T,ii = 0):  tau ds_i/dt = (a_i - 1) s_i + B_i x
                               -> pole unchanged: (a_i - 1)/tau
                               -> lam_i^new = lam_i                     (theory)

  tracking mode (P_T,ii = 1):  tau (1 - a_i) ds_i/dt
                                   = (a_i - 1) s_i + B_i x + tau B_i dx/dt
                               -> tau ds_i/dt = -s_i + K_i x + tau K_i dx/dt
                               -> pole becomes -1/tau                   (theory)

Discretization, per mode, mixing the two exact schemes:

  memory   : s_k = lam_i s_{k-1} + bbar_i x_k          (original ZOH)
  tracking : s_k = rho  s_{k-1} + K_i x_k - rho K_i x_{k-1}

No finite-difference derivative is introduced anywhere, so no parasitic
second-order state appears.
"""

import numpy as np


class ProjectedStatePC:
    mechanism = "projected_state_pc"
    label = "C. Projected state PC (Idea 2)"

    def __init__(self, system):
        self.sys = system

    def discrete_poles(self):
        sys = self.sys
        return np.where(sys.tracking_mask, sys.rho, sys.lam)

    def continuous_poles(self):
        sys = self.sys
        return np.where(sys.tracking_mask,
                        sys.prospective_continuous_pole,
                        sys.plain_continuous_poles)

    def run(self, x, s0=None):
        sys = self.sys
        s = np.zeros(sys.n_modes) if s0 is None else np.array(s0, float)
        mask, rho, K, lam, bbar = (sys.tracking_mask, sys.rho, sys.K,
                                   sys.lam, sys.bbar)
        x_prev = 0.0
        out = np.empty((len(x), sys.n_modes))
        for k, xk in enumerate(x):
            memory = lam * s + bbar * xk
            tracking = rho * s + K * xk - rho * K * x_prev
            s = np.where(mask, tracking, memory)
            out[k] = s
            x_prev = xk
        return out
