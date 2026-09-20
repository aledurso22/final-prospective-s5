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
"""Does the BLOCK scan fix float32 at the stable cell? Emulated, before JAX."""
import math, random, struct

TAU, EPS = 1000, F(1, 4)

def f32(z):
    return complex(struct.unpack("f", struct.pack("f", z.real))[0],
                   struct.unpack("f", struct.pack("f", z.imag))[0])

def seq(A, drive, r=lambda z: z):
    h = [0j]*len(A); out = []
    for d in drive:
        v = r(sum(r(A[i]*h[i]) for i in range(len(A))) + d)
        out.append(v); h = [v] + h[:-1]
    return out

def block(A, drive, C, r=lambda z: z):
    """chunked: in-chunk sequential, H^C by basis scans, boundary lax.scan."""
    n, L = len(A), len(drive)
    nchunks = (L + C - 1) // C
    padded = list(drive) + [0j]*(nchunks*C - L)

    def run_chunk(init, chunk):                    # init: list of n
        h = list(init); out = []
        for d in chunk:
            v = r(sum(r(A[i]*h[i]) for i in range(n)) + d)
            out.append(v); h = [v] + h[:-1]
        return h, out

    # H^C from n basis-vector zero-drive runs -- SEQUENTIAL, never squared
    Hc = []
    for j in range(n):
        init = [1.0 if i == j else 0j for i in range(n)]
        end, _ = run_chunk(init, [0j]*C)
        Hc.append(end)                             # column j
    # pass 1: chunk summaries from zero
    summaries = [run_chunk([0j]*n, padded[c*C:(c+1)*C])[0]
                 for c in range(nchunks)]
    # pass 2: sequential propagation over chunk boundaries
    boundaries, z = [], [0j]*n
    for c in range(nchunks):
        boundaries.append(list(z))
        z = [r(sum(r(Hc[j][i]*z[j]) for j in range(n)) + summaries[c][i])
             for i in range(n)]
    # pass 3: re-run each chunk from its boundary
    out = []
    for c in range(nchunks):
        _, tokens = run_chunk(boundaries[c], padded[c*C:(c+1)*C])
        out.extend(tokens)
    return out[:L]

random.seed(11)
print(f"{'lam':>11} {'L':>6} {'C':>5} {'block f32 vs f64':>18} "
      f"{'seq f32 vs f64':>16} {'doubling f32 vs f64':>21}")
for lam_q, label in ((CQ(F(9,10), F(0)), "0.9"), (CQ(F(1,2), F(1,4)), "0.5+0.25j")):
    Aq, Cq = exact_coefficients(TAU, EPS, lam_q)
    A = tuple(v.to_complex() for v in Aq); Cc = tuple(c.to_complex() for c in Cq)
    for L in (1000, 4000):
        xs = [random.gauss(0, 1) for _ in range(L)]
        drive = [Cc[0]*xs[t] + (Cc[1]*xs[t-1] if t >= 1 else 0)
                 + (Cc[2]*xs[t-2] if t >= 2 else 0) for t in range(L)]
        ref = seq(A, drive)                                   # float64 oracle
        scale = max(abs(s) for s in ref)
        s32 = max(abs(a-b) for a, b in zip(seq(A, drive, f32), ref))/scale
        # a PROPERLY rounded float32 doubling scan, for the contrast
        def dbl32(A, drive):
            n, L = 3, len(drive)
            state = [[f32(d), 0j, 0j] for d in drive]
            op = [[f32(a) for a in A], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            dist = 1
            while dist < L:
                sh = [[0j]*n]*dist + state[:-dist]
                state = [[f32(state[t][i] + f32(sum(f32(op[i][j]*sh[t][j])
                                                    for j in range(n))))
                          for i in range(n)] for t in range(L)]
                op = [[f32(sum(f32(op[i][k]*op[k][j]) for k in range(n)))
                       for j in range(n)] for i in range(n)]
                dist *= 2
            return [r[0] for r in state]
        d32 = max(abs(a-b) for a, b in zip(dbl32(A, drive), ref))/scale
        for C in (16, 64, 256):
            b32 = max(abs(a-b) for a, b in zip(block(A, drive, C, f32), ref))/scale
            print(f"{label:>11} {L:>6} {C:>5} {b32:18.3e} {s32:16.3e} {d32:21.3e}")
