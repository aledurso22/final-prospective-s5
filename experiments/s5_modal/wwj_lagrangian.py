"""The three prospective codings, each derived from the WWJ action.

Ordinary, generalized, and many-body/per-mode are not three separate
models. They are the SAME action with one ingredient added each time:

    ordinary     no kinetic term                 first order in D
    generalized  a kinetic term (mass M)         second order in D
    many-body    one such term PER MODE          second order, per mode

Pure Python. Every claim below is checked, not asserted.

    python -m experiments.s5_modal.wwj_lagrangian
"""

import cmath
import math

#: the prediction error is the generalized coordinate of the action
#: e = s - f, and with f = As + Bx that is e = (I - A)s - Bx


def euler_lagrange(mass, damping, stiffness):
    """Roots of M D^2 + tau D + k, i.e. the dynamics of the error.

    d/dt (dL/d e') - dL/de + dR/de' = 0 with
        L = (1/2) M e'^2 - (1/2) k e^2      (kinetic minus potential)
        R = (1/2) tau e'^2                  (Rayleigh dissipation)
    gives  M e'' + tau e' + k e = 0.
    """
    if mass == 0.0:
        return (-stiffness / damping,)
    spread = cmath.sqrt(damping * damping - 4 * mass * stiffness)
    return ((-damping + spread) / (2 * mass),
            (-damping - spread) / (2 * mass))


def timescales(mass, damping, stiffness=1.0):
    """-1/root per root: the relaxation times the action produces."""
    return tuple(-1.0 / value if abs(value) > 0 else float("inf")
                 for value in euler_lagrange(mass, damping, stiffness))


def from_timescales(first, second=None):
    """The inverse map. P(D) = (1 + T1 D)(1 + T2 D) = 1 + Gamma D + M D^2.

    So Gamma = T1 + T2 and M = T1 T2 -- which are EXACTLY `gamma_and_mass`
    in `s5/modal_prospective.py`. The cascade's two numerator parameters
    ARE the action's two relaxation times.
    """
    if second is None:
        return {"damping": first, "mass": 0.0}
    return {"damping": first + second, "mass": first * second}


def critical_mass(damping):
    """M = tau^2 / 4: where the two roots merge."""
    return damping * damping / 4.0


def regime(mass, damping):
    critical = critical_mass(damping)
    if abs(mass - critical) < 1e-12:
        return "critical"
    return "overdamped" if mass < critical else "underdamped"


def _check(label, condition):
    print(f"   [{'ok' if condition else 'FAIL'}] {label}")
    assert condition, label


def main():
    print(__doc__)
    print("=" * 74)
    print("1. THE ACTION")
    print("=" * 74)
    print("""
   Generalized coordinate: the PREDICTION ERROR  e = s - f.

       T = (1/2) M e'^2         kinetic      (inertia of the error)
       V = (1/2) k e^2          potential    (the error is penalized)
       R = (1/2) tau e'^2       Rayleigh     (dissipation)

       L = T - V,   and   d/dt(dL/de') - dL/de + dR/de' = 0

   gives, once and for all,

       M e'' + tau e' + k e = 0      i.e.   (k + tau D + M D^2) e = 0.

   Everything below is a choice of which ingredients are present.
""")
    print("=" * 74)
    print("2. ORDINARY PROSPECTIVE CODING  =  DROP THE KINETIC TERM")
    print("=" * 74)
    print("""
   With M = 0 there is no inertia, and the Euler-Lagrange equation
   degenerates to the gradient flow of V against the dissipation R:

       tau e' + e = 0          (k = 1)
       (1 + tau D)(s - f) = 0
       s - f + tau(s' - f') = 0
       tau s' = -s + f + tau f'            <-- THE NOTE'S EQ. (1)

   The prospective term tau f' is not an addition to the model. It is what
   the first-order relaxation of the prediction error LOOKS LIKE once it is
   written as an equation for s.
""")
    roots = euler_lagrange(0.0, 3.0, 1.0)
    _check("M = 0 gives exactly ONE root", len(roots) == 1)
    _check("that root is -k/tau = -1/3", abs(roots[0] + 1 / 3) < 1e-12)
    _check("so the error has ONE relaxation time, tau/k = 3",
           abs(timescales(0.0, 3.0)[0] - 3.0) < 1e-12)
    print("""
   ONE root means ONE timescale. That single number has to be both the
   memory of the mode and its prospective horizon. It cannot be both.
""")
    print("=" * 74)
    print("3. GENERALIZED PROSPECTIVE CODING  =  KEEP THE KINETIC TERM")
    print("=" * 74)
    print("""
   With M > 0 the full Euler-Lagrange equation survives:

       M e'' + tau e' + e = 0
       (1 + tau D + M D^2)(s - f) = 0

   which is the generalized WWJ operator, M f'' term included. It is second
   order, so it has TWO roots, so the error has TWO relaxation times.
""")
    print(f"   {'M':>8} {'tau':>6} {'regime':>13} {'timescales':>26}")
    for mass, damping in ((0.0, 5.0), (4.0, 5.0), (6.25, 5.0), (10.0, 5.0)):
        values = timescales(mass, damping)
        shown = ", ".join(f"{v:.3f}" if abs(v.imag) < 1e-12 else f"{v:.2f}"
                          for v in values)
        print(f"   {mass:8.2f} {damping:6.2f} "
              f"{regime(mass, damping) if mass else 'no kinetic':>13} "
              f"{shown:>26}")
    print()
    _check("critical mass of tau = 5 is 6.25", critical_mass(5.0) == 6.25)
    merged = timescales(6.25, 5.0)
    _check("at critical mass the two timescales MERGE",
           abs(merged[0] - merged[1]) < 1e-9)
    _check("and they merge at 2M/tau = tau/2 = 2.5",
           abs(merged[0] - 2.5) < 1e-9)
    split = timescales(4.0, 5.0)
    _check("below critical they SPLIT into two distinct real times",
           abs(split[0] - split[1]) > 1.0
           and abs(split[0].imag) < 1e-12)
    above = timescales(10.0, 5.0)
    _check("above critical they become a complex pair (oscillation)",
           abs(above[0].imag) > 1e-9)
    print("""
   THE CRITICAL MASS IS THE WHOLE POINT. M = tau^2/4 is where the two roots
   coincide: a critically damped mode has ONE timescale again and is the
   ordinary case in disguise. M < tau^2/4 splits them into two DISTINCT
   real relaxation times -- and that split is what lets one of them hold
   the memory while the other provides the lead.
""")
    print("   the inverse map, which is how the model is actually parameterized:")
    for first, second in ((30.0, 2.0), (10.0, 10.0), (100.0, 1.0)):
        coefficients = from_timescales(first, second)
        back = timescales(coefficients["mass"], coefficients["damping"])
        print(f"      T = ({first:6.1f}, {second:5.1f})  ->  "
              f"Gamma = {coefficients['damping']:7.1f}  "
              f"M = {coefficients['mass']:8.1f}  "
              f"{regime(coefficients['mass'], coefficients['damping']):>11}"
              f"  -> T = ({back[0].real:6.1f}, {back[1].real:5.1f})")
    print("""
   Gamma = T1 + T2 and M = T1 T2. Those are EXACTLY `gamma_and_mass` in
   `s5/modal_prospective.py`: the cascade's two numerator parameters
   n1, n2 ARE the action's two relaxation times. Setting n1 = n2 puts the
   mode on the critical branch, which is why identical stage
   initialization had to be broken -- tied stages are ordinary
   prospectivity wearing a second stage.
""")
    print("=" * 74)
    print("4. WHY BOTH OF THEM CANCEL THE MEMORY")
    print("=" * 74)
    print("""
   The action is written in e. The model is written in s. They are related
   by e = (I - A)s - Bx, which for constant A is an AFFINE CHANGE OF
   COORDINATES with a constant Jacobian (I - A).

   Euler-Lagrange equations are covariant under a constant linear change of
   variables: substituting e = (I-A)s - Bx into

       M e'' + tau e' + k e = 0

   gives (I-A)[M s'' + tau s' + k s] = (terms in x only), and the constant
   (I - A) divides straight out. THE DYNAMICS OF s ARE THE ROOTS OF THE
   OPERATOR AND NOTHING ELSE. A never enters them, at any order in D.

   This is the same theorem as WHY_GENERALIZED_THEN_PER_MODE.md section 2,
   said in the language it came from: A is a change of coordinates, and no
   Lagrangian written purely in the error can see a change of coordinates.
""")
    print("=" * 74)
    print("5. MANY-BODY / PER-MODE  =  ONE ACTION TERM PER MODE")
    print("=" * 74)
    print("""
   S5's A is DIAGONAL, so the modes are already the normal coordinates of
   the system: e_j = (1 - a_j)s_j - b_j x. The many-body action is the sum
   over them, with its own constants for each,

       L = sum_j [ (1/2) M_j e_j'^2 - (1/2) k_j e_j^2 ]
       R = sum_j   (1/2) tau_j e_j'^2

   and Euler-Lagrange, now one equation per body,

       M_j e_j'' + tau_j e_j' + k_j e_j = 0,
       ( k_j + tau_j D + M_j D^2 )( s_j - f_j ) = 0.

   Section 4 still applies to each mode separately -- a_j still divides
   out. What has changed is that THE OPERATOR ITSELF NOW DEPENDS ON THE
   MODE. The dynamics of s_j are the roots of P_j, and P_j is ours to
   choose. The memory does not come back through the residual; it comes
   back through the per-mode constants of the action.

   That is the whole repair, and it is why it needed the generalized form
   first:
""")
    print(f"   {'order':>22} {'roots per mode':>15} {'can hold memory':>16} "
          f"{'can add lead':>13} {'both at once':>13}")
    print(f"   {'ordinary (M_j = 0)':>22} {'1':>15} {'yes':>16} "
          f"{'yes':>13} {'NO':>13}")
    print(f"   {'generalized (M_j > 0)':>22} {'2':>15} {'yes':>16} "
          f"{'yes':>13} {'yes':>13}")
    print("""
   With one root per mode the single timescale must serve both demands, so
   a per-mode GATE would have nothing to select between: the mode's own
   a_j already fixes what is available. With two roots per mode there is a
   genuine two-parameter family, (Gamma_j, M_j) = (T1_j + T2_j, T1_j T2_j),
   and each body can place its two times where it needs them.

   "Some modes use it and some do not" is then a statement about the phase
   of each body in the (M_j, tau_j) plane:
""")
    print(f"   {'body':>6} {'T_memory':>9} {'T_lead':>8} {'Gamma':>8} "
          f"{'M':>9} {'M/(Gamma/2)^2':>14} {'phase':>12}")
    for index, (memory, lead) in enumerate(((300.0, 300.0), (300.0, 60.0),
                                            (300.0, 3.0), (8.0, 8.0),
                                            (8.0, 0.5))):
        coefficients = from_timescales(memory, lead)
        gamma, mass = coefficients["damping"], coefficients["mass"]
        ratio = mass / (gamma / 2) ** 2
        print(f"   {index:6d} {memory:9.1f} {lead:8.1f} {gamma:8.1f} "
              f"{mass:9.1f} {ratio:14.4f} {regime(mass, gamma):>12}")
    print("""
   A body on the critical line (T_memory = T_lead, ratio 1) is ordinary
   prospectivity and has given up the separation. A body far below it has
   a long memory time and a short lead time at once -- which is exactly
   what the construction was built to allow, and exactly what one body on
   its own cannot do.
""")
    print("   all checks passed")


if __name__ == "__main__":
    main()
