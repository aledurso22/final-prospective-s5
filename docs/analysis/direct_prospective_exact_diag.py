"""Scan vs oracle at the STABLE cell, against EXACT rational ground truth.

tau = 1000, eps = 1/4, professor_linear_target. Exact complex-rational
arithmetic gives the true sequence; both double-precision implementations are
measured against it, so the question "which one is wrong" is answered rather
than argued.
"""
import cmath, math, random
from fractions import Fraction as F

class CQ:                      # exact complex rational
    __slots__ = ("re", "im")
    def __init__(self, re=0, im=0): self.re, self.im = F(re), F(im)
    def __add__(s, o): return CQ(s.re + o.re, s.im + o.im)
    def __sub__(s, o): return CQ(s.re - o.re, s.im - o.im)
    def __mul__(s, o): return CQ(s.re*o.re - s.im*o.im, s.re*o.im + s.im*o.re)
    def to_complex(s): return complex(s.re, s.im)
    def __repr__(s): return f"CQ({float(s.re):.6g},{float(s.im):.6g})"

def exact_coefficients(tau, eps, lam):
    M = eps * tau * tau; q = M + tau
    a, b = F(2*M + tau - 1, 1)/q, F(M, 1)/q
    c0, c1, c2 = F(M + tau + 1, 1)/q, F(-(2*M + tau), 1)/q, F(M, 1)/q
    A0 = CQ(a, 0) + CQ(c0, 0)*lam
    A1 = CQ(-b, 0) + CQ(c1, 0)*lam
    A2 = CQ(c2, 0)*lam
    C = (CQ(c0, 0), CQ(c1, 0), CQ(c2, 0))       # b_bar = 1
    return (A0, A1, A2), C

def exact_sequential(A, drive):
    h = [CQ(), CQ(), CQ()]; out = []
    for d in drive:
        v = A[0]*h[0] + A[1]*h[1] + A[2]*h[2] + d
        out.append(v); h = [v, h[0], h[1]]
    return out

def double_sequential(A, drive):
    h = [0j, 0j, 0j]; out = []
    for d in drive:
        v = A[0]*h[0] + A[1]*h[1] + A[2]*h[2] + d
        out.append(v); h = [v, h[0], h[1]]
    return out

def double_doubling(A, drive):
    n, L = 3, len(drive)
    state = [[d, 0j, 0j] for d in drive]
    op = [list(A), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    dist = 1
    while dist < L:
        sh = [[0j]*n]*dist + state[:-dist]
        state = [[state[t][i] + sum(op[i][j]*sh[t][j] for j in range(n))
                  for i in range(n)] for t in range(L)]
        op = [[sum(op[i][k]*op[k][j] for k in range(n)) for j in range(n)]
              for i in range(n)]
        dist *= 2
    return [r[0] for r in state]

def ulps(x, y):
    """ULP distance between two doubles (component-wise max for complex)."""
    def one(a, b):
        if a == b: return 0
        if math.isnan(a) or math.isnan(b): return float("inf")
        import struct
        ia = struct.unpack("<q", struct.pack("<d", a))[0]
        ib = struct.unpack("<q", struct.pack("<d", b))[0]
        if ia < 0: ia = -9223372036854775808 - ia
        if ib < 0: ib = -9223372036854775808 - ib
        return abs(ia - ib)
    return max(one(x.real, y.real), one(x.imag, y.imag))

TAU, EPS = 1000, F(1, 4)
random.seed(11)
print(f"cell: tau={TAU} eps=1/4 target=professor_linear_target, b_bar=1\n")
for lam_q, label in ((CQ(F(1, 2), F(1, 4)), "0.5+0.25j"),
                     (CQ(F(9, 10), F(0)), "0.9"),
                     (CQ(F(-3, 5), F(2, 5)), "-0.6+0.4j")):
    Aq, Cq = exact_coefficients(TAU, EPS, lam_q)
    A = tuple(v.to_complex() for v in Aq)
    coef = [1.0, -A[0], -A[1], -A[2]]
    zs = [complex(0.4, 0.9)**i for i in (1, 2, 3)]
    for _ in range(600):
        new = []
        for i, z in enumerate(zs):
            p = 0
            for c in coef: p = p*z + c
            den = 1.0
            for j, w in enumerate(zs):
                if i != j: den *= (z - w)
            new.append(z - p/den if den else z)
        zs = new
    rho = max(abs(z) for z in zs)
    print(f"lam={label:12} rho={rho:.16f}  A0={A[0]:.6f} A1={A[1]:.6f} A2={A[2]:.6f}")
    for L in (3, 17, 257, 1000):
        xs = [F(random.randint(-8, 8), random.randint(1, 3)) for _ in range(L)]
        drive_q = [Cq[0]*CQ(xs[t]) + (Cq[1]*CQ(xs[t-1]) if t >= 1 else CQ())
                   + (Cq[2]*CQ(xs[t-2]) if t >= 2 else CQ()) for t in range(L)]
        drive = [d.to_complex() for d in drive_q]
        exact = [v.to_complex() for v in exact_sequential(Aq, drive_q)] \
            if L <= 257 else None
        fast, slow = double_doubling(A, drive), double_sequential(A, drive)
        diff = [abs(f - s) for f, s in zip(fast, slow)]
        denom = [max(abs(s), 1e-30) for s in slow]
        rel = [d / q for d, q in zip(diff, denom)]
        t_abs = max(range(L), key=lambda t: diff[t])
        t_rel = max(range(L), key=lambda t: rel[t])
        line = (f"   L={L:5} max|fast-slow|={max(diff):.3e} @t={t_abs} "
                f"(|slow|={abs(slow[t_abs]):.3e})  maxrel={max(rel):.3e} @t={t_rel}"
                f"  maxULP={max(ulps(f, s) for f, s in zip(fast, slow))}")
        if exact is not None:
            ef = max(abs(f - e)/max(abs(e), 1e-30) for f, e in zip(fast, exact))
            es = max(abs(s - e)/max(abs(e), 1e-30) for s, e in zip(slow, exact))
            line += f"\n         vs EXACT: scan {ef:.3e}   oracle {es:.3e}"
        print(line)
    print()
