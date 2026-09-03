"""B. PROFESSOR IDEA 1 - FULL prospective state dynamics (every mode).

Continuous:

    tau ds/dt = -s + f(s,x) + tau d/dt f(s,x),      f(s,x) = A s + B x

Substituting f and df/dt = A ds/dt + B dx/dt and collecting ds/dt:

    tau (I - A) ds/dt = (A - I) s + B x + tau B dx/dt

Left-multiplying by (I - A)^-1 and writing K = (I - A)^-1 B:

    tau ds/dt = -s + K x + tau K dx/dt                       (*)

Two consequences, both predicted by theory and both checked numerically:

1. the homogeneous dynamics are now -s/tau for EVERY mode. The learned
   mode structure A is gone from the homogeneous part: every pole becomes
   -1/tau. Slow memory is destroyed.
2. writing s = K x + r, (*) gives exactly tau dr/dt = -r, so

       s(t) = K x(t) + (s(0) - K x(0)) exp(-t/tau)

   i.e. the forced response is the INSTANTANEOUS equilibrium K x(t) plus a
   decaying initialization transient. Nothing else survives.

Exact discretization (no finite differencing, no parasitic second state).
With rho = exp(-dt/tau), the residual satisfies r_k = rho r_{k-1}, hence

    s_k = rho s_{k-1} + K x_k - rho K x_{k-1}
"""

import numpy as np


class FullStatePC:
    mechanism = "full_state_pc"
    label = "B. Full state PC (Idea 1)"

    def __init__(self, system):
        self.sys = system

    def discrete_poles(self):
        return np.full(self.sys.n_modes, self.sys.rho)

    def continuous_poles(self):
        return np.full(self.sys.n_modes, self.sys.prospective_continuous_pole)

    def run(self, x, s0=None):
        sys = self.sys
        s = np.zeros(sys.n_modes) if s0 is None else np.array(s0, float)
        rho, K = sys.rho, sys.K
        x_prev = 0.0
        out = np.empty((len(x), sys.n_modes))
        for k, xk in enumerate(x):
            s = rho * s + K * xk - rho * K * x_prev
            out[k] = s
            x_prev = xk
        return out
