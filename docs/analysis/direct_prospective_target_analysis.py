"""Does the TARGET CONSTRUCTION explain the 1.705-1.877 instability?

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
PHASES = 96
print(f"{'k':>7} {'eps':>7} | {'native-matched':>16} {'worst lam':>26} | "
      f"{'professor f=Abar s':>18} {'worst lam':>26}")
for k in (0.05, 0.1, 0.25, 0.5, 1.0):
    for eps in (0.25, 0.0625, 0.0):
        n, nl = max_radius("native_matched", k, eps, MODULI, PHASES)
        p, pl = max_radius("professor", k, eps, MODULI, PHASES)
        print(f"{k:7.2f} {eps:7.4f} | {n:16.4f} {str(round(nl.real,3))+'+'+str(round(nl.imag,3))+'j':>26} | "
              f"{p:18.4f} {str(round(pl.real,3))+'+'+str(round(pl.imag,3))+'j':>26}")
