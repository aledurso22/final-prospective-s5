"""Do the doubling scan and the sequential oracle agree BEFORE overflow?

Pure Python floats (same IEEE double as float64 JAX), unstable coefficients
of the kind the failing fixture produces (professor target, M = tau^2/4,
tau = 0.5, modes on the disc -> companion radius ~3.5).
"""
import math, cmath, random

def coefficients(tau, mass, h=1.0):
    q = mass + h*tau
    return ((2*mass + h*tau - h*h)/q, mass/q, (mass + h*tau + h*h)/q,
            -(2*mass + h*tau)/q, mass/q)

def sequential(A, drive):
    n = len(A); hist = [0j]*n; out = []
    for d in drive:
        v = sum(a*hh for a, hh in zip(A, hist)) + d
        out.append(v); hist = [v] + hist[:-1]
    return out

def doubling(A, drive):
    n = len(A); L = len(drive)
    state = [[d] + [0j]*(n-1) for d in drive]
    op = [list(A)] + [[1.0 if c == i else 0.0 for c in range(n)]
                      for i in range(n-1)]
    dist = 1
    while dist < L:
        shifted = [[0j]*n]*dist + state[:-dist]
        state = [[state[t][i] + sum(op[i][j]*shifted[t][j] for j in range(n))
                  for i in range(n)] for t in range(L)]
        op = [[sum(op[i][k]*op[k][j] for k in range(n)) for j in range(n)]
              for i in range(n)]
        dist *= 2
    return [row[0] for row in state]

def finite(z):
    return math.isfinite(z.real) and math.isfinite(z.imag)

random.seed(3)
tau, eps, L = 0.5, 0.25, 1000
mass = eps*tau*tau
a, b, c0, c1, c2 = coefficients(tau, mass)
for lam_desc, lam in (("0.9 e^{i2.0}", cmath.rect(0.9, 2.0)),
                      ("-0.95", complex(-0.95, 0.0)),
                      ("0.7 e^{-i1.2}", cmath.rect(0.7, -1.2))):
    A = [a + c0*lam, -b + c1*lam, c2*lam]
    # companion radius via numpy-free Durand-Kerner
    coef = [1.0, -A[0], -A[1], -A[2]]
    zs = [complex(0.4, 0.9)**i for i in (1, 2, 3)]
    for _ in range(500):
        new = []
        for i, z in enumerate(zs):
            p = 0
            for c in coef: p = p*z + c
            den = 1.0
            for j, w in enumerate(zs):
                if i != j: den *= (z - w)
            new.append(z - p/den if den != 0 else z)
        zs = new
    rho = max(abs(z) for z in zs)
    g = 1.0  # unit input coupling
    x = [complex(random.gauss(0, 1), 0) for _ in range(L)]
    drive = [c0*g*x[t] + (c1*g*x[t-1] if t >= 1 else 0j)
             + (c2*g*x[t-2] if t >= 2 else 0j) for t in range(L)]
    fast, slow = doubling(A, drive), sequential(A, drive)
    ff = [i for i, v in enumerate(fast) if not finite(v)]
    sf = [i for i, v in enumerate(slow) if not finite(v)]
    first_f = ff[0] if ff else None
    first_s = sf[0] if sf else None
    cut = min(first_f if first_f is not None else L,
              first_s if first_s is not None else L)
    errs = [(abs(fast[t] - slow[t]), abs(fast[t] - slow[t])/max(abs(slow[t]), 1e-300))
            for t in range(cut)]
    print(f"lam={lam_desc:14} rho={rho:8.4f} first_nonfinite fast={first_f} slow={first_s}")
    print(f"    common finite prefix = {cut} tokens; "
          f"max abs err {max(e[0] for e in errs):.3e}; "
          f"max rel err {max(e[1] for e in errs):.3e}")
    print(f"    |slow| at end of prefix = {abs(slow[cut-1]):.3e}")
