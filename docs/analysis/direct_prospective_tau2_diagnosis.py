"""Diagnosis of the tau=2, eps=0, C=64, L=4000 float32 NaN.

Answers, for the offending mode: is the SEQUENTIAL path finite; where each
path first goes nonfinite; the coefficients in both precisions; the exact
roots; |H^C| for every chunk size; the last finite magnitude; whether the
drive is finite; and which chunk sizes survive.
"""
import cmath, math, struct

def _round32(value):
    """Round to float32, saturating to +-inf like the hardware does."""
    if math.isnan(value):
        return value
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def f32(z):
    return complex(_round32(z.real), _round32(z.imag))

def coefficients(tau, eps, h=1.0):
    M = eps*tau*tau; q = M + h*tau
    return ((2*M + h*tau - h*h)/q, M/q, (M + h*tau + h*h)/q,
            -(2*M + h*tau)/q, M/q)

def order2(tau, lam):
    a, _, c0, c1, _ = coefficients(tau, 0.0)
    return (a + c0*lam, c1*lam)

def roots2(A):
    d = cmath.sqrt(A[0]*A[0] + 4*A[1])
    return [(A[0]+d)/2, (A[0]-d)/2]

def seq(A, drive, r=lambda z: z):
    h = [0j]*len(A); out = []
    for d in drive:
        v = r(sum(r(A[i]*h[i]) for i in range(len(A))) + d)
        out.append(v); h = [v] + h[:-1]
    return out

def block(A, drive, C, r=lambda z: z):
    n, L = len(A), len(drive)
    count = (L + C - 1)//C
    padded = list(drive) + [0j]*(count*C - L)
    def run(init, chunk):
        h, out = list(init), []
        for d in chunk:
            v = r(sum(r(A[i]*h[i]) for i in range(n)) + d)
            out.append(v); h = [v] + h[:-1]
        return h, out
    T = []
    for j in range(n):
        T.append(run([1.0 if i == j else 0j for i in range(n)], [0j]*C)[0])
    summaries = [run([0j]*n, padded[c*C:(c+1)*C])[0] for c in range(count)]
    bounds, z = [], [0j]*n
    for c in range(count):
        bounds.append(list(z))
        z = [r(sum(r(T[j][i]*z[j]) for j in range(n)) + summaries[c][i])
             for i in range(n)]
    out = []
    for c in range(count):
        out.extend(run(bounds[c], padded[c*C:(c+1)*C])[1])
    return out[:L]

def transition_norm(A, C):
    n = len(A)
    H = [list(A)] + [[1.0 if j == i else 0.0 for j in range(n)]
                     for i in range(n-1)]
    R = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(C):
        R = [[sum(H[i][k]*R[k][j] for k in range(n)) for j in range(n)]
             for i in range(n)]
    return max(abs(v) for row in R for v in row)

def first_nonfinite(values):
    for index, value in enumerate(values):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            return index
    return None

TAU, EPS, LENGTH = 2.0, 0.0, 4000
#: the offending mode: |lam| = 0.9 at angle -2 rad, INSIDE the region the
#: failing test samples (|lam| <= 0.9, |angle| <= 2)
LAM = cmath.rect(0.9, -2.0)
BBAR = 1.0

if __name__ == "__main__":
    A64 = order2(TAU, LAM)
    A32 = tuple(f32(a) for a in A64)
    print(f"offending mode: Abar = {LAM:.9f}  |Abar| = {abs(LAM):.6f}  "
          f"Bbar = {BBAR}")
    print(f"float64 coefficients: A0 = {A64[0]:.9f}  A1 = {A64[1]:.9f}")
    print(f"float32 coefficients: A0 = {A32[0]:.9f}  A1 = {A32[1]:.9f}")
    zs = roots2(A64)
    print(f"exact companion roots: {[f'{z:.6f}' for z in zs]}")
    print(f"max companion radius : {max(abs(z) for z in zs):.10f}  "
          f"-> {'UNSTABLE' if max(abs(z) for z in zs) > 1 else 'stable'}")
    print()
    print(f"{'C':>6} {'|H^C|':>14}")
    for C in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        print(f"{C:>6} {transition_norm(A64, C):14.4e}")
    print()
    _, _, c0, c1, _ = coefficients(TAU, EPS)
    drive = [c0*BBAR*(1.0 if t == 0 else 0.0)
             + (c1*BBAR*(1.0 if t == 1 else 0.0)) for t in range(LENGTH)]
    drive = [f32(d) for d in drive]
    print(f"drive finite: {first_nonfinite(drive) is None}, "
          f"max |drive| = {max(abs(d) for d in drive):.4e}")
    s32 = seq(A32, drive, f32)
    s64 = seq(A64, drive)
    ns, n64 = first_nonfinite(s32), first_nonfinite(s64)
    print(f"SEQUENTIAL float32: first nonfinite token = {ns}, "
          f"max |state| before = "
          f"{max(abs(v) for v in s32[:ns or LENGTH]):.4e}")
    print(f"SEQUENTIAL float64: first nonfinite token = {n64}")
    print()
    print(f"{'C':>6} {'first nonfinite':>16} {'max |state| before':>20}")
    for C in (1, 2, 4, 8, 16, 32, 64):
        values = block(A32, drive, C, f32)
        index = first_nonfinite(values)
        head = values[:index or LENGTH]
        print(f"{C:>6} {str(index):>16} "
              f"{max(abs(v) for v in head) if head else 0.0:20.4e}")
    print()
    print("CONCLUSION: the model itself diverges at this mode -- the "
          "sequential path goes nonfinite too -- so this cell must be "
          "rejected by GATE 1 (stability), not measured by GATE 2.")
