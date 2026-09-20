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



# ---- analysis ----
"""Does tau = 1000 produce a prospective effect, or approach Native?"""
import math, cmath

TAU, EPS, L = 1000, F(1, 4), 4000

def roots_of(A):
    coef = [1.0, -A[0], -A[1], -A[2]]
    zs = [complex(0.4, 0.9)**i for i in (1, 2, 3)]
    for _ in range(800):
        new = []
        for i, z in enumerate(zs):
            p = 0
            for c in coef: p = p*z + c
            den = 1.0
            for j, w in enumerate(zs):
                if i != j: den *= (z - w)
            new.append(z - p/den if den else z)
        zs = new
    return zs

def professor_coefficients(tau, lam):      # M = 0
    a, c0, c1 = 1 - 1/tau, 1 + 1/tau, -1.0
    return (a + c0*lam, c1*lam, 0.0), (c0, c1, 0.0)

for lam, label in ((complex(0.9, 0.0), "0.9"), (complex(0.5, 0.25), "0.5+0.25j"),
                   (complex(0.99, 0.0), "0.99")):
    lam_q = CQ(F(lam.real).limit_denominator(10**6), F(lam.imag).limit_denominator(10**6))
    Aq, Cq = exact_coefficients(TAU, EPS, lam_q)
    A = tuple(v.to_complex() for v in Aq); C = tuple(v.to_complex() for v in Cq)
    Ap, Cp = professor_coefficients(TAU, lam)
    impulse = [1.0] + [0.0]*(L-1)
    def run(A, C):
        drive = [C[0]*impulse[t] + (C[1]*impulse[t-1] if t >= 1 else 0)
                 + (C[2]*impulse[t-2] if t >= 2 else 0) for t in range(L)]
        return double_sequential(A, drive)
    wwj = run(A, C)
    prof = run(Ap, Cp)
    native = []
    st = 0j
    for t in range(L):
        st = lam*st + (1.0 if t == 0 else 0.0); native.append(st)
    def energy(v): return math.sqrt(sum(abs(z)**2 for z in v))
    def centroid(v):
        w = [abs(z)**2 for z in v]; tot = sum(w)
        return sum(t*w[t] for t in range(len(v)))/tot if tot else float("nan")
    rel_native = energy([a-b for a, b in zip(wwj, native)]) / energy(native)
    rel_prof = energy([a-b for a, b in zip(wwj, prof)]) / energy(prof)
    zs = roots_of(A)
    times = [(-1/math.log(abs(z)) if 0 < abs(z) < 1 else float('inf')) for z in zs]
    print(f"lam={label}")
    print(f"  A0={A[0]:.9f} A1={A[1]:.9f} A2={A[2]:.9f}")
    print(f"  C0={C[0]:.9f} C1={C[1]:.9f} C2={C[2]:.9f}")
    print(f"  roots={[f'{z:.6f}' for z in zs]}")
    print(f"  |roots|={[f'{abs(z):.6f}' for z in zs]}  timescales(tokens)="
          f"{[f'{t:.1f}' for t in times]}")
    print(f"  native timescale = {-1/math.log(abs(lam)):.1f} tokens")
    print(f"  relative impulse-response difference:  vs Native {rel_native:.4e}"
          f"   vs Professor/TSS(M=0,tau=1000) {rel_prof:.4e}")
    print(f"  response centroid: WWJ {centroid(wwj):.2f}  Native {centroid(native):.2f}"
          f"  Professor {centroid(prof):.2f}  -> lag shift {centroid(wwj)-centroid(native):+.2f} tokens")
    print()
