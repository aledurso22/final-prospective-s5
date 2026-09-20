"""Max companion radius over the mode REGION, not a sampled subset.

This is the scan that falsified the "tau = 2, 5, 10 are well-conditioned"
claim: those cells are unstable at modes a small random subset never drew.
"""
import cmath
import math


def roots2(A0, A1):
    d = cmath.sqrt(A0 * A0 + 4 * A1)
    return [(A0 + d) / 2, (A0 - d) / 2]


def roots3(A0, A1, A2):
    coef = [1.0, -A0, -A1, -A2]
    zs = [complex(0.4, 0.9) ** i for i in (1, 2, 3)]
    for _ in range(500):
        new = []
        for i, z in enumerate(zs):
            p = 0
            for c in coef:
                p = p * z + c
            den = 1.0
            for j, w in enumerate(zs):
                if i != j:
                    den *= (z - w)
            new.append(z - p / den if den else z)
        zs = new
    return zs


def radius(tau, eps, lam):
    mass = eps * tau * tau
    q = mass + tau
    a, b = (2 * mass + tau - 1) / q, mass / q
    c0, c1, c2 = (mass + tau + 1) / q, -(2 * mass + tau) / q, mass / q
    if eps == 0:
        return max(abs(z) for z in roots2(a + c0 * lam, -b + c1 * lam))
    return max(abs(z) for z in roots3(a + c0 * lam, -b + c1 * lam, c2 * lam))


if __name__ == "__main__":
    print("max companion radius over mode regions "
          "(professor_linear_target)\n")
    print(f"{'tau':>7} {'eps':>6} | "
          f"{'|lam|<=0.9, |angle|<=2 (the failing test region)':>46} | "
          f"{'full disc':>22}")
    for tau in (2.0, 5.0, 10.0, 50.0, 100.0, 1000.0):
        for eps in (0.0, 0.25):
            worst_region = worst_disc = 0.0
            where_region = where_disc = None
            for i in range(40):
                r = 0.2 + 0.7 * i / 39
                for j in range(721):
                    angle = -math.pi + 2 * math.pi * j / 720
                    lam = cmath.rect(r, angle)
                    value = radius(tau, eps, lam)
                    if abs(angle) <= 2.0 and r <= 0.9 and value > worst_region:
                        worst_region, where_region = value, lam
                    if value > worst_disc:
                        worst_disc, where_disc = value, lam
            flag = "UNSTABLE" if worst_region > 1 else "ok"
            print(f"{tau:>7} {eps:>6.2f} | {worst_region:10.4f} at "
                  f"{where_region:.3f} {flag:>9} | {worst_disc:8.4f} at "
                  f"{where_disc:.3f}")
