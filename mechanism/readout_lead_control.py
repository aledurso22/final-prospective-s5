"""E. READOUT LEAD CONTROL - the EXISTING prospective-lead mechanism.

This is the mechanism already implemented on `prospective-lead`:

    b_pc[k] = b[k] + alpha (b[k] - b[k-1]),     P(z) = 1 + alpha(1 - z^-1)

applied AFTER the readout. It never touches the state recurrence. It is
included here purely as a control, to test whether state-level prospectivity
achieves anything an output-side lead cannot.

Two variants, because the distinction turns out to matter:

`ReadoutLeadPerMode`
    a separate alpha_i per mode, applied to each mode's own trajectory. For a
    first-order mode the matched value is

        alpha_i = lam_i / (1 - lam_i)

    because then

        P(z) S_plain(z) = (1/(1-lam)) (1 - lam z^-1) * bbar/(1 - lam z^-1) X(z)
                        = (bbar/(1-lam)) X(z) = K X(z)

    which is EXACTLY the fully prospective tracking response. This is not an
    approximation - the pole is cancelled by the lead's zero.

`ReadoutLeadShared`
    ONE scalar alpha applied to the mixed readout y = C s. This is what S5
    actually does: a single alpha per block, downstream of a readout that has
    already summed modes with different lam_i. One zero cannot cancel several
    distinct poles, so the exact per-mode equivalence breaks here.
"""

import numpy as np


def matched_alpha(lam):
    """alpha that makes the lead's zero cancel a discrete pole lam."""
    return lam / (1.0 - lam)


def apply_lead(sequence, alpha):
    """Causal lead. previous[0] := sequence[0], so the first correction is 0."""
    seq = np.asarray(sequence, dtype=np.float64)
    prev = np.concatenate((seq[:1], seq[:-1]), axis=0)
    return seq + np.asarray(alpha) * (seq - prev)


class ReadoutLeadPerMode:
    mechanism = "readout_lead_per_mode"
    label = "E1. Readout lead, per-mode matched alpha"

    def __init__(self, system, alpha=None, base=None):
        from mechanism.plain_ssm import PlainSSM
        self.sys = system
        self.base = base or PlainSSM(system)
        if alpha is None:
            alpha = np.where(system.tracking_mask, matched_alpha(system.lam), 0.0)
        self.alpha = np.asarray(alpha, dtype=np.float64)

    def discrete_poles(self):
        """With matched alpha the lead's zero cancels the plain pole, so the
        homogeneous response vanishes from the output after one sample."""
        cancelled = np.isclose(self.alpha, matched_alpha(self.sys.lam))
        return np.where(cancelled, 0.0, self.base.discrete_poles())

    def continuous_poles(self):
        cancelled = np.isclose(self.alpha, matched_alpha(self.sys.lam))
        return np.where(cancelled, -np.inf, self.sys.plain_continuous_poles)

    def run(self, x, s0=None):
        return apply_lead(self.base.run(x, s0), self.alpha)


class ReadoutLeadShared:
    mechanism = "readout_lead_shared"
    label = "E2. Readout lead, one shared alpha on the mixed readout"

    def __init__(self, system, alpha, base=None):
        from mechanism.plain_ssm import PlainSSM
        self.sys = system
        self.base = base or PlainSSM(system)
        self.alpha = float(alpha)

    def discrete_poles(self):
        return self.base.discrete_poles()

    def continuous_poles(self):
        return self.sys.plain_continuous_poles.copy()

    def run_readout(self, x, s0=None):
        """Returns the LEAD-CORRECTED READOUT y (T, n_out), not the state."""
        states = self.base.run(x, s0)
        y = states @ self.sys.C.T
        return apply_lead(y, self.alpha)
