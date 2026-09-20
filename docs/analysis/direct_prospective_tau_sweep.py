"""Is there ANY stable tau for the causal mixed-stencil WWJ recurrence?

Sweeps tau over four decades for both target constructions, and also the
partially matched (no M f'') two-compartment form for comparison.
"""
"""Companion radius versus tau, for every direct prospective form.

Pure Python, complex arithmetic, Durand-Kerner roots. Compares, for the same
causal mixed-stencil recurrence:

  (i)  native-matched target   F = 1 + k(lam - 1),  G = k Bbar     [old]
  (ii) professor-consistent    F = lam,             G = Bbar       [asked for]

over a realistic locus of discrete S5 modes lam = r e^{i phi}.
"""
import cmath

def coefficients(tau, mass, h=1.0):
    q = mass + h * tau
    return ((2*mass + h*tau - h*h)/q, mass/q,
            (mass + h*tau + h*h)/q, -(2*mass + h*tau)/q, mass/q)

def roots_cubic(A0, A1, A2):
    # z^3 - A0 z^2 - A1 z - A2 ; Durand-Kerner
    coef = [1.0, -A0, -A1, -A2]
    def poly(z):
        out = 0
        for c in coef:
            out = out * z + c
        return out
    zs = [complex(0.4, 0.9) ** i for i in range(1, 4)]
    for _ in range(400):
        new = []
        for i, z in enumerate(zs):
            denom = 1.0
            for j, w in enumerate(zs):
                if i != j:
                    denom *= (z - w)
            new.append(z - poly(z) / denom if denom != 0 else z)
        if max(abs(a - b) for a, b in zip(new, zs)) < 1e-14:
            zs = new
            break
        zs = new
    return zs

def max_radius(target, k, eps, moduli, phases):
    tau, mass = k, eps * k * k          # h = 1: tau = k, M = eps tau^2
    a, b, c0, c1, c2 = coefficients(tau, mass)
    worst, worst_lam = 0.0, None
    for r in moduli:
        for p in range(phases):
            lam = cmath.rect(r, 2 * cmath.pi * p / phases)
            F = (1 + k * (lam - 1)) if target == "native_matched" else lam
            A0, A1, A2 = a + c0 * F, -b + c1 * F, c2 * F
            radius = max(abs(z) for z in roots_cubic(A0, A1, A2))
            if radius > worst:
                worst, worst_lam = radius, lam
    return worst, worst_lam



MODULI = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999)
PHASES = 128

def two_compartment(tau, mass, gamma, h=1.0):
    """M s'' + Gamma s' + s = f + T f' ; Gamma = gamma + T, T = tau.
    s_t = a s_{t-1} - b s_{t-2} + c0 f_{t-1} + c1 f_{t-2}."""
    Gamma = gamma + tau
    q = mass + h * Gamma
    return ((2*mass + h*Gamma - h*h)/q, mass/q, h*(h+tau)/q, -h*tau/q)

def radius_two_compartment(tau, eps, gamma=0.0):
    Gamma = gamma + tau
    mass = eps * Gamma * Gamma
    a, b, c0, c1 = two_compartment(tau, mass, gamma)
    worst = 0.0
    for r in MODULI:
        for p in range(PHASES):
            lam = cmath.rect(r, 2*cmath.pi*p/PHASES)
            A1, A2 = a + c0*lam, -b + c1*lam     # f = lam s + B x
            disc = cmath.sqrt(A1*A1 + 4*A2)
            worst = max(worst, abs((A1+disc)/2), abs((A1-disc)/2))
    return worst

print(f"{'tau':>9} | {'mixed native-matched':>20} {'mixed professor':>16} "
      f"| {'two-compartment (no Mf\")':>26}")
for tau in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
            100.0, 1000.0):
    n, _ = max_radius("native_matched", tau, 0.25, MODULI, PHASES)
    p, _ = max_radius("professor", tau, 0.25, MODULI, PHASES)
    t = radius_two_compartment(tau, 0.25)
    print(f"{tau:9.2f} | {n:20.4f} {p:16.4f} | {t:26.4f}")
print()
print("two-compartment, eps sweep at several tau (critical eps=1/4):")
for tau in (0.05, 0.5, 5.0, 50.0):
    row = [f"{radius_two_compartment(tau, e):.4f}" for e in (0.0, 0.0625, 0.25)]
    print(f"  tau={tau:7.2f}  eps=0: {row[0]}   eps=1/16: {row[1]}   eps=1/4: {row[2]}")
