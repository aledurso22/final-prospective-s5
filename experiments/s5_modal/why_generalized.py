"""Why the generalized operator could not save the equation, and why it was
still the step that made a per-mode treatment possible.

Pure Python complex arithmetic. No JAX, no fitting, no seeds: every number
below is a property of the transfer functions and is reproducible exactly.

    python -m experiments.s5_modal.why_generalized
"""

import cmath
import math

#: group delay is read at this frequency unless stated otherwise
DC = 1e-3


def group_delay(transfer, omega=DC, step=1e-4):
    """-dphi/domega, in tokens. POSITIVE is lag, NEGATIVE is lead.

    This is the quantity the whole argument turns on: a prospective system
    is one whose output does not wait for its input, and negative group
    delay is what that means precisely.
    """
    ahead = cmath.phase(transfer(cmath.exp(1j * (omega + step))))
    behind = cmath.phase(transfer(cmath.exp(1j * (omega - step))))
    return -(ahead - behind) / (2 * step)


def native(a):
    """The untouched S5 mode, s_t = a s_{t-1} + x_t."""
    return lambda z: 1.0 / (1 - a / z)


def professor(a, epsilon):
    """The note's cell: A1 = (1-e) + (1+e)a, A2 = -a, xt = (1+e)x' - x''."""
    first = (1 - epsilon) + (1 + epsilon) * a
    return lambda z: ((1 + epsilon) / z - 1 / z ** 2) / (
        1 - first / z + a / z ** 2)


def roots(a, epsilon):
    first = (1 - epsilon) + (1 + epsilon) * a
    spread = cmath.sqrt(first * first - 4 * a)
    return (first + spread) / 2, (first - spread) / 2


def spectral_radius(a, epsilon):
    return max(abs(value) for value in roots(a, epsilon))


def epsilon_max(a):
    magnitude = abs(a) ** 2
    return (2 * (1 - magnitude) * abs(1 - a) ** 2
            / abs(2 * a - 1 - magnitude) ** 2)


def cascade(a, numerators, denominators):
    """(prod (1 + n_i D)) / (prod (1 + d_i D)) on top of the native mode,
    with D = 1 - z^-1. The native pole is NEVER touched."""
    def transfer(z):
        derivative = 1 - 1 / z
        top = 1.0
        for value in numerators:
            top *= (1 + value * derivative)
        bottom = 1.0
        for value in denominators:
            bottom *= (1 + value * derivative)
        return top / bottom / (1 - a / z)
    return transfer


def best_professor_lead(a, samples=4000):
    """The most lead the note's cell can reach while staying stable, and
    where its memory pole ends up when it does."""
    boundary = epsilon_max(a)
    best = None
    for index in range(1, samples):
        epsilon = index / samples * boundary
        radius = spectral_radius(a, epsilon)
        if radius >= 1.0:
            continue
        delay = group_delay(professor(a, epsilon))
        if best is None or delay < best[0]:
            best = (delay, radius, epsilon / boundary)
    return best


MODES = ((0.5, 0.2), (0.9, 0.2), (0.9, 1.0), (0.99, 0.2), (0.99, 1.0))


def main():
    print(__doc__)
    print("=" * 74)
    print("1. THE CANCELLATION IS STRUCTURAL AND DOES NOT CARE ABOUT P")
    print("=" * 74)
    print("""
   P(D)(s - f) = 0 with f = As + Bx.  The residual is

        s - f = (I - A)s - Bx,

   and (I - A) is a CONSTANT matrix, so it commutes with every D:

        P(D)[(I-A)s - Bx] = 0  =>  (I-A) P(D)s = P(D)Bx
                               =>  P(D)s = (I-A)^-1 P(D) B x.

   The homogeneous dynamics are P(D)s = 0.  A has left the dynamics and
   survives only as a static input gain.  The proof never reads P, so
   1 + tau D + M D^2 cancels the memory exactly as 1 + tau D does.
   GENERALIZING THE OPERATOR CANNOT SAVE THE EQUATION.  That is not a
   failure of the generalization; it is a property of the residual.
""")
    print("=" * 74)
    print("2. WHAT THE ONE KNOB ACTUALLY DOES: LEAD AND MEMORY ARE THE SAME")
    print("=" * 74)
    print("\n   Most lead the note's cell can reach before it leaves the disc,")
    print("   and where the memory pole has been dragged to when it gets there.\n")
    print(f"   {'mode':>16} {'native pole':>12} {'best lead':>11} "
          f"{'pole then':>10} {'timescale':>20}")
    for magnitude, omega in MODES:
        a = magnitude * cmath.exp(1j * omega)
        delay, radius, _ = best_professor_lead(a)
        before = -1.0 / math.log(abs(a))
        after = -1.0 / math.log(radius)
        print(f"   {magnitude:>6.2f} e^{{{omega:.1f}i}} {abs(a):12.4f} "
              f"{delay:11.3f} {radius:10.4f} "
              f"{before:8.1f} -> {after:<8.1f}")
    print("""
   The lead is small and the price is the whole memory: a mode at |a| = 0.5
   has a 1.4-token timescale and comes back with a 60-token one.  The pole
   IS the knob.  Pushing lead moves it, and eps_max(a) is where it stops.
""")
    print("=" * 74)
    print("3. WHAT GENERALIZING CHANGES: THE LEAD MOVES INTO THE ZEROS")
    print("=" * 74)
    print("\n   Same modes, a two-stage cascade with n = (8, 8), d = (1, 1).\n")
    print(f"   {'mode':>16} {'lead':>10} {'native pole':>12} {'added poles':>12}"
          f" {'vs professor':>13}")
    for magnitude, omega in MODES:
        a = magnitude * cmath.exp(1j * omega)
        delay = group_delay(cascade(a, (8.0, 8.0), (1.0, 1.0)))
        reference = best_professor_lead(a)[0]
        print(f"   {magnitude:>6.2f} e^{{{omega:.1f}i}} {delay:10.3f} "
              f"{abs(a):12.4f} {0.5:12.3f} {delay/reference:12.1f}x")
    print("""
   The native pole does not move at all, the added poles sit at
   d/(h+d) = 0.5 whatever the numerators do, and the lead is five to five
   hundred times larger.  Nothing here is a better approximation of the
   professor's law -- it is a different object: an operator whose poles and
   zeros are placed INDEPENDENTLY, rather than an operator applied to a
   residual that ties them together.
""")
    print("=" * 74)
    print("4. WHY THAT IS WHAT MAKES A PER-MODE CHOICE POSSIBLE")
    print("=" * 74)
    print("""
   With the first-order law a mode has ONE free parameter, eps, and its
   range is (0, eps_max(a)) -- a function of the mode itself.  The mode
   does not choose anything: `a` fixes how much lead is available and what
   it costs.  There is no regime to select, so there is nothing for a gate
   to gate.

   With poles and zeros placed independently the mode has a TWO-parameter
   family, (Gamma, M) = (n1 + n2, n1 n2), over a denominator that is stable
   whatever they are.  Now "how much memory" and "how much lead" are
   separate coordinates, a mode can hold one and vary the other, and
   "some modes use it and some do not" is an actual choice rather than a
   consequence of `a`.

   THAT is what the generalized operator bought.  Not a repair of the
   cancellation -- section 1 proves no operator repairs that -- but the
   separation of the two quantities the per-mode story needs to be about.
""")
    print("=" * 74)
    print("5. WHAT THIS ARGUMENT DOES *NOT* ESTABLISH")
    print("=" * 74)
    print("\n   At matched DC lead, one zero and two zeros are very similar:\n")
    a = 0.9 * cmath.exp(0.2j)
    for target in (-2.5, -5.0):
        low, high = 0.01, 400.0
        for _ in range(200):
            mid = (low + high) / 2
            if group_delay(cascade(a, (mid, 1.0), (1.0, 1.0))) > target:
                low = mid
            else:
                high = mid
        single = (low + high) / 2
        low, high = 0.01, 400.0
        for _ in range(200):
            mid = (low + high) / 2
            if group_delay(cascade(a, (mid, mid), (1.0, 1.0))) > target:
                low = mid
            else:
                high = mid
        double = (low + high) / 2
        print(f"   lead {target:+.1f} tokens:  one zero n = {single:7.3f}   "
              f"two zeros n = ({double:.3f}, {double:.3f})")
        for omega in (0.05, 0.2, 0.5, 1.0):
            one = group_delay(cascade(a, (single, 1.0), (1.0, 1.0)), omega)
            two = group_delay(cascade(a, (double, double), (1.0, 1.0)), omega)
            print(f"      at w = {omega:<5g}  one zero {one:8.3f}   "
                  f"two zeros {two:8.3f}")
    print("""
   A single zero can reach ANY DC lead on its own, and at matched lead the
   two profiles are close.  So this frequency-domain argument justifies
   moving the lead into the zeros -- it does NOT, by itself, justify the
   SECOND zero, which is the M f_ddot term.

   The evidence for the second zero is empirical, not analytic: in the
   frontier run the two-stage arm beat the one-stage arm it contains on
   lead at every delay (12-15 of 15 paired seeds, medians 1.34-1.63x)
   while staying memory non-inferior at every delay.  That is the
   justification, and it should be quoted as a measurement rather than
   dressed up as a derivation.
""")


if __name__ == "__main__":
    main()
