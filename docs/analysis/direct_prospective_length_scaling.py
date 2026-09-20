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
"""How far does the doubling scan's accuracy survive at this stable cell?"""
import math, random

TAU, EPS = 1000, F(1, 4)
random.seed(11)
print(f"{'lam':>12} {'L':>7} {'rel err scan vs oracle':>24} {'digits kept':>12}")
for lam_q, label in ((CQ(F(9,10), F(0)), "0.9"), (CQ(F(1,2), F(1,4)), "0.5+0.25j")):
    Aq, Cq = exact_coefficients(TAU, EPS, lam_q)
    A = tuple(v.to_complex() for v in Aq)
    for L in (257, 1000, 2000, 4000, 8000, 16000):
        xs = [random.gauss(0, 1) for _ in range(L)]
        c0, c1, c2 = (c.to_complex() for c in Cq)
        drive = [c0*xs[t] + (c1*xs[t-1] if t >= 1 else 0) + (c2*xs[t-2] if t >= 2 else 0)
                 for t in range(L)]
        fast, slow = double_doubling(A, drive), double_sequential(A, drive)
        scale = max(abs(s) for s in slow)
        rel = max(abs(f-s) for f, s in zip(fast, slow)) / scale
        digits = -math.log10(rel) if rel > 0 else 17
        print(f"{label:>12} {L:>7} {rel:24.3e} {digits:12.1f}")
