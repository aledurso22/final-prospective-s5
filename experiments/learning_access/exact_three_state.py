"""1. The exact three-state learning-access example.

    z_{k+1} = (rho+theta) z_k + x_k
    b_{k+1} = beta b_k + v_k
    s_{k+1} = a s_k + (1-a)(z_{k+1} + theta b_{k+1})

    w_k = (s_k - a s_{k-1})/(1-a) = z_k + theta b_k

Two gates:
  INFERENCE-PERFECT   w_k = z_k at theta = 0, to numerical precision, for
                      randomized x, v, z0, b0.
  TANGENT-WRONG       d w_k/dtheta|_0 = d z_k/dtheta|_0 + b_k, verified by
                      closed form, autodiff and finite differences.

The gap between the two is exactly b_k: a physical direction that is invisible
in the nominal output and indispensable for learning.
"""
import jax
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.learning_access._common import (AQUA, BLUE, MUTED, ORANGE,
                                                 SURFACE, YELLOW, outdir,
                                                 save, style)
from prospective.three_state import (ThreeStateConfig, dw_dtheta_analytic,
                                     dz_dtheta_analytic, numpy_rollout,
                                     prospective_inverse, rollout, w_of)

SEED, T, N_RANDOM = 20260910, 40, 200


def main():
    rng = np.random.default_rng(SEED)
    d = outdir("1_exact_three_state")
    base = ThreeStateConfig()

    # ---- gate 1: inference-perfect over randomized histories -------------
    worst_w, worst_ident = 0.0, 0.0
    for _ in range(N_RANDOM):
        cfg = ThreeStateConfig(rho=base.rho, beta=base.beta, a=base.a,
                               z0=float(rng.normal()), b0=float(rng.normal() * 4))
        x, v = rng.normal(size=T), rng.normal(size=T)
        out = rollout(cfg, 0.0, x, v)
        w = np.asarray(prospective_inverse(cfg, out["s"]))
        worst_w = max(worst_w, float(np.max(np.abs(w - np.asarray(out["z"])[1:]))))
        # and the identity w = z + theta b at a non-zero theta
        th = float(rng.uniform(-0.2, 0.2))
        o = rollout(cfg, th, x, v)
        w2 = np.asarray(prospective_inverse(cfg, o["s"]))
        ident = np.max(np.abs(w2 - (np.asarray(o["z"])[1:] + th * np.asarray(o["b"])[1:])))
        worst_ident = max(worst_ident, float(ident))
    print(f"  INFERENCE gate over {N_RANDOM} randomized histories")
    print(f"    max |w - z| at theta=0        : {worst_w:.3e}")
    print(f"    max |w - (z + theta b)|       : {worst_ident:.3e}")

    # ---- gate 2: the tangent, three independent ways ---------------------
    cfg = base
    x, v = rng.normal(size=T), rng.normal(size=T)
    ana = dw_dtheta_analytic(cfg, x, v)
    auto = np.asarray(jax.jacobian(lambda t: w_of(cfg, t, x, v))(0.0))
    eps = 1e-6
    fd = (np.asarray(w_of(cfg, eps, x, v)) - np.asarray(w_of(cfg, -eps, x, v))) / (2 * eps)

    err_auto = float(np.max(np.abs(auto - ana)))
    err_fd = float(np.max(np.abs(fd - ana)))
    print(f"\n  TANGENT gate  dw/dtheta|_0 = dz/dtheta|_0 + b")
    print(f"    max |autodiff - analytic|     : {err_auto:.3e}")
    print(f"    max |finite diff - analytic|  : {err_fd:.3e}")

    out0 = numpy_rollout(cfg, 0.0, x, v)
    dz = dz_dtheta_analytic(cfg, x)[1:]
    b = out0["b"][1:]
    gap = float(np.max(np.abs((ana - dz) - b)))
    print(f"    max |(dw - dz) - b|           : {gap:.3e}   <- the gap IS b")

    ok = (worst_w < 1e-12 and worst_ident < 1e-12 and err_auto < 1e-9
          and err_fd < 1e-6 and gap < 1e-12)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} - nominally exact, tangent "
          f"contaminated by the inaccessible b")

    results = dict(max_abs_w_minus_z=worst_w, max_abs_identity=worst_ident,
                   tangent_err_autodiff=err_auto, tangent_err_finite_diff=err_fd,
                   tangent_gap_minus_b=gap, verdict_pass=bool(ok),
                   n_random_histories=N_RANDOM, T=T)

    # ---- figure ---------------------------------------------------------
    k = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), facecolor=SURFACE)
    w0 = np.asarray(w_of(cfg, 0.0, x, v))
    axes[0].plot(k, out0["z"][1:], color="#0b0b0b", lw=2.6, label="intended z")
    axes[0].plot(k, w0, color=AQUA, lw=1.5, ls=(0, (2, 2)), label="realized w")
    style(axes[0], "Nominal inference at theta = 0: exact overlap", "step",
          "value", legend=True)
    axes[0].text(0.96, 0.06, f"max|w-z| = {worst_w:.1e}",
                 transform=axes[0].transAxes, ha="right", fontsize=8.5)

    axes[1].plot(k, dz, color="#0b0b0b", lw=2.6, label=r"intended $\partial_\theta z$")
    axes[1].plot(k, ana, color=ORANGE, lw=1.8, ls=(0, (4, 2)),
                 label=r"implemented $\partial_\theta w$")
    style(axes[1], "Parameter tangents disagree", "step", "tangent", legend=True)

    axes[2].plot(k, ana - dz, color=YELLOW, lw=2.6,
                 label=r"$\partial_\theta w - \partial_\theta z$")
    axes[2].plot(k, b, color=BLUE, lw=1.5, ls=(0, (2, 2)), label="hidden b")
    style(axes[2], "The difference is exactly the hidden mode b", "step",
          "value", legend=True)

    fig.suptitle("Inference-perfect, learning-wrong: the gap between the two "
                 "tangents is the inaccessible physical mode",
                 color="#0b0b0b", fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = f"{d}/fig_exact_three_state.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    cfg_d = dict(seed=SEED, T=T, n_random=N_RANDOM, rho=base.rho, beta=base.beta,
                 a=base.a, z0=base.z0, b0=base.b0)
    print("wrote", save("1_exact_three_state", cfg_d, results), "and", path)
    return results


if __name__ == "__main__":
    main()
