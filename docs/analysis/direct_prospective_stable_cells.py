"""float32 accuracy of each scan across the CLUSTER-MEASURED stable cells.

The earlier study used only tau = 1000, where rho = 0.998 and the operator is
nearly marginal. The cluster found stable cells at tau = 2, 5, 10, 50, 100
and 1000, and the picture at the well-conditioned ones is completely
different: when |H^C| is O(1) the block scan matches the sequential
recurrence to within a small factor.
"""
import math
import random
import struct
from fractions import Fraction as F


class CQ:
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re, self.im = F(re), F(im)

    def __add__(s, o): return CQ(s.re + o.re, s.im + o.im)
    def __mul__(s, o): return CQ(s.re * o.re - s.im * o.im,
                                 s.re * o.im + s.im * o.re)
    def to_complex(s): return complex(s.re, s.im)


def f32(z):
    return complex(struct.unpack("f", struct.pack("f", z.real))[0],
                   struct.unpack("f", struct.pack("f", z.imag))[0])


def coefficients(tau, eps, lam):
    mass = eps * tau * tau
    q = mass + tau
    a, b = F(2 * mass + tau - 1, 1) / q, F(mass, 1) / q
    c0 = F(mass + tau + 1, 1) / q
    c1 = F(-(2 * mass + tau), 1) / q
    c2 = F(mass, 1) / q
    return ((CQ(a, 0) + CQ(c0, 0) * lam, CQ(-b, 0) + CQ(c1, 0) * lam,
             CQ(c2, 0) * lam), (CQ(c0, 0), CQ(c1, 0), CQ(c2, 0)))


def seq(A, drive, r=lambda z: z):
    h = [0j] * len(A)
    out = []
    for d in drive:
        v = r(sum(r(A[i] * h[i]) for i in range(len(A))) + d)
        out.append(v)
        h = [v] + h[:-1]
    return out


def block(A, drive, C, r=lambda z: z):
    n, L = len(A), len(drive)
    count = (L + C - 1) // C
    padded = list(drive) + [0j] * (count * C - L)

    def run(init, chunk):
        h, out = list(init), []
        for d in chunk:
            v = r(sum(r(A[i] * h[i]) for i in range(n)) + d)
            out.append(v)
            h = [v] + h[:-1]
        return h, out

    transition = []
    for j in range(n):
        init = [1.0 if i == j else 0j for i in range(n)]
        transition.append(run(init, [0j] * C)[0])
    summaries = [run([0j] * n, padded[c * C:(c + 1) * C])[0]
                 for c in range(count)]
    boundaries, z = [], [0j] * n
    for c in range(count):
        boundaries.append(list(z))
        z = [r(sum(r(transition[j][i] * z[j]) for j in range(n))
               + summaries[c][i]) for i in range(n)]
    out = []
    for c in range(count):
        out.extend(run(boundaries[c], padded[c * C:(c + 1) * C])[1])
    return out[:L]


def doubling(A, drive, r=lambda z: z):
    n, L = len(A), len(drive)
    state = [[r(d), 0j, 0j] for d in drive]
    op = [[r(a) for a in A], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    dist = 1
    while dist < L:
        sh = [[0j] * n] * dist + state[:-dist]
        state = [[r(state[t][i] + r(sum(r(op[i][j] * sh[t][j])
                                        for j in range(n))))
                  for i in range(n)] for t in range(L)]
        op = [[r(sum(r(op[i][k] * op[k][j]) for k in range(n)))
               for j in range(n)] for i in range(n)]
        dist *= 2
    return [row[0] for row in state]


def spectral_radius(A):
    coef = [1.0, -A[0], -A[1], -A[2]]
    zs = [complex(0.4, 0.9) ** i for i in (1, 2, 3)]
    for _ in range(600):
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
    return max(abs(z) for z in zs)


def transition_norm(A, C):
    H = [[A[0], A[1], A[2]], [1, 0, 0], [0, 1, 0]]
    R = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
    for _ in range(C):
        R = [[sum(H[i][k] * R[k][j] for k in range(3)) for j in range(3)]
             for i in range(3)]
    return max(abs(v) for row in R for v in row)


#: the cells the cluster measured as stable, for professor_linear_target
CLUSTER_STABLE_TAU = (2, 5, 10, 50, 100, 1000)
LENGTH = 4000

if __name__ == "__main__":
    random.seed(11)
    print(f"{'tau':>6} {'eps':>6} {'lam':>11} {'rho':>8} {'|H^64|':>10} | "
          f"{'block C=64':>11} {'block C=16':>11} {'sequential':>11} "
          f"{'doubling':>11}   (float32 relative to float64, L=4000)")
    for tau in CLUSTER_STABLE_TAU:
        for eps in (F(0), F(1, 4)):
            for lam_q, label in ((CQ(F(9, 10), F(0)), "0.9"),
                                 (CQ(F(1, 2), F(1, 4)), "0.5+0.25j")):
                Aq, Cq = coefficients(F(tau), eps, lam_q)
                A = tuple(v.to_complex() for v in Aq)
                C = tuple(v.to_complex() for v in Cq)
                xs = [random.gauss(0, 1) for _ in range(LENGTH)]
                drive = [C[0] * xs[t] + (C[1] * xs[t - 1] if t >= 1 else 0)
                         + (C[2] * xs[t - 2] if t >= 2 else 0)
                         for t in range(LENGTH)]
                reference = seq(A, drive)
                scale = max(abs(s) for s in reference)

                def rel(values):
                    return max(abs(a - b) for a, b in
                               zip(values, reference)) / scale

                print(f"{tau:>6} {float(eps):>6.2f} {label:>11} "
                      f"{spectral_radius(A):8.4f} {transition_norm(A, 64):10.2e} | "
                      f"{rel(block(A, drive, 64, f32)):11.3e} "
                      f"{rel(block(A, drive, 16, f32)):11.3e} "
                      f"{rel(seq(A, drive, f32)):11.3e} "
                      f"{rel(doubling(A, drive, f32)):11.3e}")
