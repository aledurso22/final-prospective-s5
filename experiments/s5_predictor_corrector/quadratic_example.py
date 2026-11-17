"""Fixed real quadratic, SEPARATE from S5: the paper's mechanism in isolation.

h(x) = 1/2 x^T A x, A = diag(eigs) in [m, L_h], real, fixed, no input.
Hadjisavvas et al. (arXiv:2506.15632) Eq. (6) / (7) with constant
coefficients (alpha, theta, beta):

    y_k     = x_k + alpha (x_k - x_{k-1}) - theta (grad(x_k) - grad(x_{k-1}))
    (6) x_{k+1} = y_k - beta grad(x_k)       heavy ball + Hessian correction
    (7) x_{k+1} = y_k - beta grad(y_k)       Nesterov-type

Per eigenvalue lam this is EXACTLY the S5 per-mode algebra with
K -> lam, mu -> alpha, zeta -> theta, eta -> beta and zero input, so the
asymptotic rate is the largest root modulus over the spectrum. Faster
minimization here is NOT evidence about sequence representations: S5's
residual is complex, input-driven and changes every token.

    python -m experiments.s5_predictor_corrector.quadratic_example
"""
import json

import numpy as onp

from s5.predictor_corrector import roots


def rate(alpha, theta, beta, eigs, corrector):
    P, Lc = 1 - beta * eigs, alpha - theta * eigs
    a1, a2 = (P + Lc, -Lc) if corrector == "old" else (P * (1 + Lc), -P * Lc)
    z1, z2 = roots(a1 + 0j, a2 + 0j)
    return float(onp.max(onp.maximum(abs(z1), abs(z2))))


def iterate(alpha, theta, beta, eigs, corrector, tol=1e-8, kmax=100000):
    x = onp.ones_like(eigs); xp = x.copy()
    f0 = 0.5 * onp.sum(eigs * x * x)
    for k in range(kmax):
        g, gp = eigs * x, eigs * xp
        y = x + alpha * (x - xp) - theta * (g - gp)
        xn = y - beta * (g if corrector == "old" else eigs * y)
        xp, x = x, xn
        f = 0.5 * onp.sum(eigs * x * x)
        if not onp.isfinite(f) or f > 1e30:
            return None
        if f < tol * f0:
            return k + 1
    return None


def main():
    eigs = onp.geomspace(1.0, 100.0, 50)            # condition number 100
    beta = 1.0 / eigs.max()
    rows = []
    for alpha in (0.0, 0.5, 0.8, 0.9, 0.95):
        for theta_frac in (0.0, 0.5):
            theta = theta_frac * beta
            row = dict(alpha=alpha, theta=theta, beta=beta)
            for c in ("old", "prediction"):
                row[f"rate_{c}"] = rate(alpha, theta, beta, eigs, c)
                row[f"iters_{c}"] = iterate(alpha, theta, beta, eigs, c)
            rows.append(row)
    # a larger step: where the correction point matters most
    beta2 = 1.9 / eigs.max()
    for alpha in (0.5, 0.9):
        row = dict(alpha=alpha, theta=0.0, beta=beta2)
        for c in ("old", "prediction"):
            row[f"rate_{c}"] = rate(alpha, 0.0, beta2, eigs, c)
            row[f"iters_{c}"] = iterate(alpha, 0.0, beta2, eigs, c)
        rows.append(row)
    for r in rows:
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in r.items()}))
    return rows


if __name__ == "__main__":
    import os
    out = "results/s5_predictor_corrector"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "quadratic_example.json"), "w") as f:
        json.dump(main(), f, indent=1)
