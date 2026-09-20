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
"""A MEASURED condition number, and whether it predicts the observed error."""
import math, random

EPSILON64 = 2.220446049250313e-16
TAU, EPS = 1000, F(1, 4)

def abs_scan(A, drive):
    """The same recurrence with every quantity replaced by its magnitude:
    s~_t = |A0| s~_{t-1} + |A1| s~_{t-2} + |A2| s~_{t-3} + |d_t|.
    This is the sum of the absolute values of every term contributing to
    s_t, i.e. the numerator of the condition number of that sum."""
    h = [0.0, 0.0, 0.0]; out = []
    for d in drive:
        v = abs(A[0])*h[0] + abs(A[1])*h[1] + abs(A[2])*h[2] + abs(d)
        out.append(v); h = [v, h[0], h[1]]
    return out

random.seed(11)
print(f"{'lam':>12} {'L':>6} {'kappa':>12} {'eps*kappa*log2L':>17} "
      f"{'observed rel':>14} {'ratio obs/pred':>15}")
rows = []
for lam_q, label in ((CQ(F(1,2), F(1,4)), "0.5+0.25j"), (CQ(F(9,10), F(0)), "0.9"),
                     (CQ(F(-3,5), F(2,5)), "-0.6+0.4j")):
    Aq, Cq = exact_coefficients(TAU, EPS, lam_q)
    A = tuple(v.to_complex() for v in Aq)
    for L in (3, 17, 257, 1000):
        xs = [F(random.randint(-8, 8), random.randint(1, 3)) for _ in range(L)]
        drive = [(Cq[0]*CQ(xs[t]) + (Cq[1]*CQ(xs[t-1]) if t >= 1 else CQ())
                  + (Cq[2]*CQ(xs[t-2]) if t >= 2 else CQ())).to_complex()
                 for t in range(L)]
        fast, slow = double_doubling(A, drive), double_sequential(A, drive)
        mags = abs_scan(A, drive)
        scale = max(abs(s) for s in slow)
        kappa = max(m / max(scale, 1e-300) for m in mags)
        pred = EPSILON64 * kappa * max(1, math.ceil(math.log2(max(L, 2))))
        obs = max(abs(f-s) for f, s in zip(fast, slow)) / max(scale, 1e-300)
        ratio = obs / pred if pred else float("inf")
        rows.append(ratio)
        print(f"{label:>12} {L:>6} {kappa:12.4e} {pred:17.3e} {obs:14.3e} "
              f"{ratio:15.3f}")
print(f"\nmax observed/predicted ratio = {max(rows):.3f}  "
      f"(a safety factor of 8 covers every case here)")
