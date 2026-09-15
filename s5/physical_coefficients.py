"""Fixed physical response coefficients, derived once and then IMMUTABLE.

Governed by `docs/handoff_2026_09_15/DERIVATION_CONTRACT.md`. Nothing here is
trainable, sweepable or selected from validation results. The values follow
from the declared symmetric intrinsic reference, not from any benchmark.

Circuit (NLA Appendix 6, Eqs. 80-81; reciprocal constant-conductance,
small-signal sector)::

    c u_dot  + G_s u - h v = I_s
    c_d v_dot + G_d v - h u = I_d

Retaining the dendritic capacitance `c_d` is the modification under test. With
kappa = G_s - h^2/G_d > 0, T = c/kappa and tau_d = c_d/G_d, exact elimination
at constant I_s gives::

    c tau_d u_ddot + (c + G_s tau_d) u_dot + kappa u = b,
    b = I_s + (h/G_d) I_d

and, with b = (1 + T D) bbar and r = u - bbar/kappa,

    M u_ddot + gamma u_dot + r + T r_dot = 0,
    gamma = G_s tau_d / kappa,   M = T tau_d,   rho = M/(gamma T) = kappa/G_s.

Declared symmetric reference (an IDEALIZATION, not an identified NLA cell)::

    c_s = c_d = c,  g_L = h = g > 0,  zero background synaptic conductance
    => G_s = G_d = 2g,  kappa = 3g/2,  tau_d = 3T/4,
       gamma = T,  M = 3T^2/4,  rho = 3/4

Model-time convention: T = 5 input intervals, chosen to align the
computational horizon with the benchmark. It is NOT fitted. At T = 5 the
unit-sample physical tuple is (T, gamma, M) = (5, 5, 18.75).

Exact S5 normalization. With J_phys = -gamma F0 and B_phys = gamma B0,
dividing the WHOLE equation by gamma gives

    (M/gamma) s_ddot + s_dot + r_n + T r_n_dot = 0,   r_n = -F0 s - B0 x

so normalized code uses gamma_n = 1, T = 5, mu = M/gamma = 3.75, rho = 0.75.
Dividing only one term would be a different model; `validate()` enforces the
tie M = rho * gamma * T.

Scope limits carried from the contract, repeated because they constrain what
may be claimed rather than what the code computes:

* The passive plant supplies these tied coefficients. It does NOT prescribe
  the prospective closed-loop source; substituting bbar/kappa = f_theta(u, x)
  is an explicit feedback convention, a computational extension.
* Component equalities (c_s = c_d, g_L = h) are declared idealizations.
* Fixed intrinsic/background conductances and learned ADDITIVE-CURRENT
  coupling are assumed. Literal conductance-synapse learning would need the
  unreduced circuit or a new derivation.
* Learned native Delta changes effective recurrent/input coupling; it does not
  retime the fixed compartment. Delta is absorbed ONCE into F0/B0.
"""

from dataclasses import dataclass

#: model-time convention from the contract: horizon in input intervals
T_INTERVALS = 5.0
#: rho = kappa/G_s = 3/4 for the symmetric reference
RHO_SYMMETRIC = 0.75


@dataclass(frozen=True)
class PhysicalResponse:
    """Immutable coefficient tuple. `frozen=True` is the enforcement."""

    T: float = T_INTERVALS
    #: normalized gamma. gamma_n = 1 by construction of the normalization;
    #: the PHYSICAL gamma is kept separately so physical checks stay possible.
    gamma: float = 1.0
    rho: float = RHO_SYMMETRIC
    #: physical gamma in unit-sample coefficients, for traceability only
    gamma_physical: float = T_INTERVALS
    label: str = "symmetric-reference-T5"

    @property
    def mass(self):
        """Normalized mass mu = M/gamma = rho * T (since gamma_n = 1)."""
        return self.rho * self.gamma * self.T

    @property
    def mass_physical(self):
        """Unit-sample physical M = rho * gamma_phys * T = 18.75 at T = 5."""
        return self.rho * self.gamma_physical * self.T

    def validate(self):
        """Check the ties the contract requires. Raises, never warns."""
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if not (0.0 < self.rho <= 1.0):
            raise ValueError(
                f"rho must lie in (0, 1]; got {self.rho}. Stability of the "
                f"scalar-rho block construction is supported for T SPD, "
                f"sym(J) > 0, scalar gamma > 0 and scalar rho in (0, 1] only.")
        if abs(self.mass - self.rho * self.gamma * self.T) > 1e-12:
            raise ValueError("M = rho*gamma*T tie violated")
        return self


#: THE reference used by the cluster experiment: gamma_n = 1, T = 5, mu = 3.75
SYMMETRIC_REFERENCE = PhysicalResponse().validate()

#: The M = 0 ablation. Same T and gamma; mass dropped. This is a REDUCED /
#: constitutive ablation, NOT the true fast-dendrite circuit limit, which
#: sends gamma and M to zero together at fixed intrinsic parameters.
M0_REFERENCE = PhysicalResponse(rho=1e-12, label="M0-ablation-T5")


def coefficient_table():
    """Source-to-code rows, emitted into the run manifest for traceability."""
    r = SYMMETRIC_REFERENCE
    return [
        dict(symbol="T", value=r.T, units="input intervals",
             source="model-time convention, DERIVATION_CONTRACT sec. 'Fixed "
                    "reference and exact S5 normalization'", trainable=False),
        dict(symbol="gamma_n", value=r.gamma, units="normalized",
             source="whole-equation division by gamma", trainable=False),
        dict(symbol="rho", value=r.rho, units="dimensionless",
             source="rho = kappa/G_s = 3/4, symmetric reference",
             trainable=False),
        dict(symbol="mu = M/gamma", value=r.mass, units="intervals^2",
             source="mu = rho*gamma_n*T", trainable=False),
        dict(symbol="M_physical", value=r.mass_physical, units="intervals^2",
             source="M = 3T^2/4 at T = 5", trainable=False),
        dict(symbol="gamma_physical", value=r.gamma_physical, units="intervals",
             source="gamma = G_s tau_d / kappa = T", trainable=False),
    ]
