"""Why applying H^C in float32 costs precision at the stable cell."""
A = (2.896015538, -2.792430279, 0.896414343)   # tau=1000, eps=1/4, lam=0.9


def power(C):
    H = [[A[0], A[1], A[2]], [1, 0, 0], [0, 1, 0]]
    R = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
    for _ in range(C):
        R = [[sum(H[i][k] * R[k][j] for k in range(3)) for j in range(3)]
             for i in range(3)]
    return R


print(f"{"C":>5} {"max|H^C|":>12} {"eps32*|H^C|":>14}")
for C in (1, 16, 32, 64, 128, 256):
    m = max(abs(v) for row in power(C) for v in row)
    print(f"{C:>5} {m:12.4e} {1.1920929e-07 * m:14.3e}")
