"""The per-mode generalized WWJ action, IN the recurrence.

Not a filter on the readout and not a cascade bolted onto Native: the
recurrence below IS the Euler-Lagrange equation of a many-body action, one
body per S5 mode.

WHERE THE EARLIER FORMULATION WENT WRONG. Writing the whole action in the
prediction error alone,

    T = (1/2) M |e'|^2,  V = (1/2) k |e|^2,  R = (1/2) tau |e'|^2,

makes e = (1-a)s - bx the only coordinate. That is an affine change of
variables with CONSTANT Jacobian (1-a), Euler-Lagrange is covariant under
those, and (1-a) divides straight out: the dynamics of s are the roots of
the operator and `a` never appears. The memory is gone at every order in
D. No choice of P repairs it.

THE REPAIR IS IN THE ACTION, NOT IN P. A body has inertia and friction OF
ITS OWN, not only in its prediction error. Give it both:

    T = (1/2) M_s |s'|^2 + (1/2) M_e |e'|^2
    V = (1/2) k  |e|^2
    R = (1/2) tau_s |s'|^2 + (1/2) tau_e |e'|^2

Wirtinger Euler-Lagrange, d/dt(dT/ds'bar) + dV/dsbar + dR/ds'bar = 0, with
e = (1-a)s - bx and |1-a|^2 = (1-a)(1-abar):

  [M_s + M_e|1-a|^2] s'' + [tau_s + tau_e|1-a|^2] s' + k|1-a|^2 s
        = (1-abar) b [ M_e x'' + tau_e x' + k x ].

Read it:

  * the LEFT side is the body's own second-order dynamics. Its coefficients
    are REAL and POSITIVE, so Routh-Hurwitz makes it stable unconditionally
    -- no gate, no clipping, no threshold;
  * `a` enters through |1-a|^2, the mode's distance from z = 1, so the S5
    memory hierarchy is INHERITED rather than cancelled;
  * the RIGHT side is P(D) applied to the INPUT. That is the lead.

Dividing by k|1-a|^2 factors it into exactly the cascade this repository
already runs,

    (1 + U1 D)(1 + U2 D) s = [b/(1-a)] (1 + T1 D)(1 + T2 D) x,

with

    T1 + T2 = Gamma,              T1 T2 = M                  (LEAD)
    U1 + U2 = Gamma + tau_s/q,    U1 U2 = M + m_s/q,  q=|1-a|^2 (MEMORY)

so the two numerator times are the action's lead times and the two
denominator times are its memory times. The cascade was never arbitrary:
it is what this action gives.

THE TWO LIMITS, BOTH EXACT.

    tau_s = m_s = 0   =>  U = T identically, the stages CANCEL, and the
                          response is the memoryless map b/(1-a). That is
                          the professor's equation, and its defect appears
                          here as a literal pole-zero cancellation.
    Gamma = M = 0     =>  no zeros at all: memory, no lead.

A per-mode gate g in (0,1) scaling the error branch therefore slides each
body between "pure memory" and "pure prospection", and g is the many-body
variable: some bodies use the equation, some do not.

STABILITY, UNCONDITIONALLY. U1 + U2 > 0 and U1 U2 > 0 because Gamma, M,
tau_s, m_s are non-negative and q > 0. A quadratic with positive sum and
positive product has both roots in the right half plane, so every stage
pole U/(h+U) satisfies |U| < |h+U| and lies strictly inside the unit disc.
Nothing a parameter can do changes that.
"""

import jax
import jax.numpy as np

from .modal_prospective import H_TOKEN, apply_cascade

#: floors keeping the action's constants strictly positive
TIME_MIN = 1e-4
#: |1 - lambda_bar|^2 is floored so a mode sitting exactly at z = 1 cannot
#: divide by zero. It is the mode's distance from DC, not its magnitude.
DISTANCE_MIN = 1e-6


def distance_from_unity(lambda_bar):
    """q = |1 - lambda_bar|^2, the coefficient `a` enters the action by.

    Not |a|: a mode is slow RELATIVE TO DC when it sits near z = 1, and
    that is what |1 - a| measures.
    """
    return np.maximum(np.abs(1.0 - lambda_bar) ** 2, DISTANCE_MIN)


def _quadratic_times(total, product):
    """The two roots of U^2 - total U + product, as times.

    With total > 0 and product > 0 both roots have positive real part, so
    both stage poles are inside the unit disc.
    """
    spread = np.sqrt((total * total - 4.0 * product).astype(np.complex64)
                     if not np.iscomplexobj(total)
                     else total * total - 4.0 * product)
    return (total + spread) / 2.0, (total - spread) / 2.0


def lead_times(gamma, mass):
    """(T1, T2): the action's LEAD times, from the error branch alone."""
    return _quadratic_times(gamma, mass)


def memory_times(gamma, mass, tau_state, mass_state, distance):
    """(U1, U2): the action's MEMORY times.

    They exceed the lead times by exactly the state branch divided by the
    mode's distance from unity, which is how the S5 hierarchy is inherited:
    a mode near z = 1 has a small q and therefore a long memory.
    """
    return _quadratic_times(gamma + tau_state / distance,
                            mass + mass_state / distance)


def action_stages(lambda_bar, gamma, mass, tau_state, mass_state):
    """[(d, n)] per stage: d is a memory time, n is a lead time.

    Exactly the (d, n) pairs `apply_cascade` consumes, so the action runs
    on the repository's own scan with no new primitive.
    """
    distance = distance_from_unity(lambda_bar)
    first_lead, second_lead = lead_times(gamma, mass)
    first_memory, second_memory = memory_times(gamma, mass, tau_state,
                                               mass_state, distance)
    dtype = lambda_bar.dtype
    return [(first_memory.astype(dtype), first_lead.astype(dtype)),
            (second_memory.astype(dtype), second_lead.astype(dtype))]


def static_gain(lambda_bar, b_bar):
    """b/(1 - a): what the action leaves in front of the cascade."""
    return b_bar / (1.0 - lambda_bar)[:, None]


def apply_action(lambda_bar, b_bar, input_sequence, gamma, mass, tau_state,
                 mass_state, h=H_TOKEN, reverse=False):
    """The Euler-Lagrange recurrence itself, as two first-order scans."""
    gain = static_gain(lambda_bar, b_bar)
    drive = jax.vmap(lambda u: gain @ u)(input_sequence).astype(
        lambda_bar.dtype)
    if reverse:
        drive = drive[::-1]
    states = apply_cascade(drive, action_stages(lambda_bar, gamma, mass,
                                                tau_state, mass_state), h)
    return states[::-1] if reverse else states


def stage_poles(lambda_bar, gamma, mass, tau_state, mass_state, h=H_TOKEN):
    """U/(h+U) per stage: what the certification bounds."""
    return [d / (h + d) for d, _ in action_stages(lambda_bar, gamma, mass,
                                                  tau_state, mass_state)]


def certify(lambda_bar, gamma, mass, tau_state, mass_state, h=H_TOKEN):
    """Whole-inventory certification. The bound is 1 by construction; this
    measures how far inside it the inventory actually sits."""
    radii = [np.abs(pole) for pole in
             stage_poles(lambda_bar, gamma, mass, tau_state, mass_state, h)]
    worst = float(np.max(np.stack(radii)))
    return {
        "max_stage_pole": worst,
        "certified": bool(worst < 1.0),
        "modes_at_or_above_one": int(
            sum(int(np.sum(radius >= 1.0)) for radius in radii)),
    }


def regime_fractions(gamma, mass, tau_state, mass_state, distance,
                     threshold=1e-3):
    """How many bodies are prospective, how many are pure memory.

    `cancelling` counts the bodies that have drifted to the professor's
    singular limit, where the state branch has vanished and the stages
    cancel. It is reported rather than forbidden.
    """
    lead = (gamma + mass) > threshold
    state = (tau_state + mass_state) / distance > threshold
    return {
        "prospective_and_remembering": int(np.sum(lead & state)),
        "memory_only": int(np.sum(~lead & state)),
        "cancelling_professor_limit": int(np.sum(lead & ~state)),
        "inert": int(np.sum(~lead & ~state)),
    }
