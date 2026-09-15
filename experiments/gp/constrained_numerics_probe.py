"""Measure two things before amending anything. Cluster only, read-only.

A. Where the block matrix exponential actually stops being finite, as a
   function of the derived mass mu = T gamma_n rho, in float64 AND float32.
   The declared admissibility box is to be set FROM this measurement, not
   guessed and not widened after a failure.

B. Whether the analytic gradient of the response leaves is correct, by
   comparing the directional derivative against central differences at several
   steps in float64 and float32. A step-independent relative error indicates a
   WRONG GRADIENT; an error that shrinks as the step shrinks (until rounding
   takes over) indicates a mis-specified finite difference.

Prints a table. Draws no conclusion and changes no bound.
"""

import argparse
import json
import math
import os
import sys

import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def build(np, P=4, H=3, L=10, seed=8):
    from jax.scipy.linalg import block_diag
    from s5.ssm_init import make_DPLR_HiPPO
    rs = onp.random.RandomState(seed)
    a = np.asarray(-onp.exp(rs.uniform(-3, -1, P)) + 1j * rs.uniform(-2, 2, P))
    b = np.asarray(rs.randn(P, H) + 1j * rs.randn(P, H))
    x = np.asarray(rs.randn(L, H))
    return a, b, x, P


def part_a(jax, np):
    from s5.gp_fixed import mass_block_zoh, mass_scan
    a, b, x, P = build(np)

    def f(eta, zeta):
        z = mass_block_zoh(a, b, 5.0, np.exp(eta) * np.ones(P),
                           np.exp(zeta) * np.ones(P))
        return np.sum(np.abs(mass_scan(z["A_bar"], z["B_bar"], x)) ** 2)

    rows = []
    print("  %-10s %-10s %-12s %-10s %-10s" %
          ("gamma_n", "rho", "mu", "value", "grad"))
    for g_n in (1e2, 1.0, 1e-1, 1e-2):
        for rho in (1 - 1e-4, 0.75, 1e-1, 1e-2, 1e-3, 1e-4):
            mu = 5.0 * g_n * rho
            try:
                val = float(f(math.log(g_n), math.log(rho)))
                gr = jax.grad(f, argnums=(0, 1))(math.log(g_n), math.log(rho))
                gok = all(onp.isfinite(float(v)) for v in gr)
            except Exception as exc:                      # noqa: BLE001
                val, gok = float("nan"), False
                print(f"    exception at {g_n},{rho}: {type(exc).__name__}")
            vok = bool(onp.isfinite(val))
            rows.append(dict(gamma_n=g_n, rho=rho, mu=mu, value_finite=vok,
                             grad_finite=bool(gok), value=val))
            print("  %-10.3g %-10.5g %-12.4g %-10s %-10s"
                  % (g_n, rho, mu, vok, gok))
    good = [r["mu"] for r in rows if r["value_finite"] and r["grad_finite"]]
    bad = [r["mu"] for r in rows if not (r["value_finite"] and r["grad_finite"])]
    print(f"  smallest mu that WORKED : {min(good) if good else None}")
    print(f"  largest  mu that FAILED : {max(bad) if bad else None}")
    return rows


def part_b(jax, np, label):
    """Directional derivative vs central differences at several steps."""
    from s5.rawat_s5 import RESPONSE_PARAM_NAMES, init_substrate_ssm
    from s5.ssm_init import make_DPLR_HiPPO
    from jax.scipy.linalg import block_diag
    H, ssm, blocks, L = 4, 8, 2, 16
    blk = ssm // blocks
    Lam, _, _, V, _ = make_DPLR_HiPPO(blk)
    blk //= 2
    P = ssm // 2
    Lam, V = Lam[:blk], V[:, :blk]
    Vc = V.conj().T
    Lam = (Lam * np.ones((blocks, blk))).ravel()
    kw = dict(H=H, P=P, Lambda_re_init=Lam.real, Lambda_im_init=Lam.imag,
              V=block_diag(*([V] * blocks)), Vinv=block_diag(*([Vc] * blocks)),
              C_init="trunc_standard_normal", discretization="zoh",
              dt_min=0.001, dt_max=0.1, conj_sym=True, bidirectional=False)
    mod = init_substrate_ssm("gp_learned_response", **kw)()
    x = np.asarray(onp.random.RandomState(0).randn(L, H))
    v = mod.init(__import__("jax").random.PRNGKey(0), x)
    p = v["params"]

    def loss(params):
        return np.sum(mod.apply({"params": params}, x) ** 2)

    base = float(loss(p))
    g = jax.grad(loss)(p)
    print(f"  [{label}] loss = {base:.6f}")
    out = []
    rs = onp.random.RandomState(1)
    for name in RESPONSE_PARAM_NAMES:
        d = rs.randn(*onp.asarray(p[name]).shape)
        d = np.asarray(d / d.std(), dtype=p[name].dtype)
        ana = float(np.sum(g[name] * d))
        row = dict(leaf=name, analytic=ana, label=label, steps={})
        line = []
        for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4):
            plus = dict(p); plus[name] = p[name] + h * d
            minus = dict(p); minus[name] = p[name] - h * d
            fd = (float(loss(plus)) - float(loss(minus))) / (2 * h)
            rel = abs(ana - fd) / max(abs(fd), 1e-12)
            row["steps"][f"{h:.0e}"] = dict(fd=fd, rel=rel)
            line.append(f"{h:.0e}:{rel:.2e}")
        print(f"  [{label}] {name:22s} analytic {ana:12.6f}   "
              + "  ".join(line))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    import jax
    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {jax.default_backend()!r}")
    jax.config.update("jax_default_matmul_precision", "highest")

    result = {}
    print("=== B. gradient check, FLOAT32 (production dtype) ===")
    import jax.numpy as jnp
    result["grad_f32"] = part_b(jax, jnp, "f32")

    print("\n=== A. finiteness sweep, FLOAT32 ===")
    result["sweep_f32"] = part_a(jax, jnp)

    print("\n(enabling x64 for the float64 comparisons)")
    jax.config.update("jax_enable_x64", True)
    print("\n=== B. gradient check, FLOAT64 ===")
    result["grad_f64"] = part_b(jax, jnp, "f64")
    print("\n=== A. finiteness sweep, FLOAT64 ===")
    result["sweep_f64"] = part_a(jax, jnp)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"\nwrote {args.out}")
    print("NUMERICS_PROBE_DONE")


if __name__ == "__main__":
    main()
