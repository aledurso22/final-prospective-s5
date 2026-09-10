"""3. Weak access: faithful learning at an unavoidable noise cost.

Side measurement  m_k = delta b_k + n_k,  n ~ (0, sigma^2).
Learning-faithful correction

    z_hat_k = w_k - (theta/delta) m_k

Noiseless, this reproduces z for the FULL parameterized family, not merely at
theta = 0, since w = z + theta b and (theta/delta)(delta b) = theta b. Its BPTT
gradient is therefore automatically the intended z-system gradient.

With noise, the tangent becomes

    d z_hat_t/dtheta|_0 = dz_t/dtheta - n_t/delta

so the gradient is unbiased but has variance

    Var[ d/dtheta l(z_hat_t)|_0 ] = l'(z_t)^2 sigma^2 / delta^2      (slope -2)

The approximate corrector z_hat = w - theta k m trades bias against variance:

    eps_g + delta sqrt(V) / (|l'(z)| sigma) >= 1,     eps_g = |1 - delta k|

The three regimes, which are the actual point:

    access | nominal inference | learning
    none   | exact             | impossible
    weak   | exact             | faithful but noisy
    strong | exact             | faithful and bounded
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.learning_access._common import (AQUA, BLUE, MUTED, ORANGE,
                                                 SURFACE, YELLOW, outdir,
                                                 save, style)
from prospective.three_state import (ThreeStateConfig, corrected_estimate,
                                     dz_dtheta_analytic, numpy_rollout, w_of)

SEED, T, SIGMA, N_MC = 20260910, 25, 0.1, 4000
Y_STAR = 0.5


def loss(y):
    return 0.5 * (y - Y_STAR) ** 2


def dloss(y):
    return y - Y_STAR


def main():
    rng = np.random.default_rng(SEED)
    d = outdir("3_weak_access_noise")
    cfg = ThreeStateConfig()
    x, v = rng.normal(size=T), rng.normal(size=T)
    ref = numpy_rollout(cfg, 0.0, x, v)
    z_T, b_T = ref["z"][T], ref["b"][T]
    dz_T = dz_dtheta_analytic(cfg, x)[T]

    # ---- noiseless: exact for the FULL parameterized family --------------
    worst = 0.0
    for th in (-0.3, -0.1, 0.0, 0.05, 0.2, 0.35):
        for delta in (0.05, 1.0, 7.0):
            zh = np.asarray(corrected_estimate(cfg, th, x, v, delta))
            zt = np.asarray(numpy_rollout(cfg, th, x, v)["z"])[1:]
            worst = max(worst, float(np.max(np.abs(zh - zt))))
    print(f"  noiseless correction reproduces z for ALL theta: max err {worst:.3e}")

    # its gradient equals the intended z-system gradient
    g_hat = float(jax.grad(lambda t: loss(corrected_estimate(
        cfg, t, x, v, 1.0)[-1]))(0.0))
    g_int = dloss(z_T) * dz_T
    print(f"  d/dtheta l(z_hat_T) = {g_hat:+.9f}   intended {g_int:+.9f}   "
          f"diff {abs(g_hat-g_int):.3e}")

    # ---- noisy: variance law across delta --------------------------------
    deltas = np.array([0.02, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0])
    def one_grad(nz, dl, gain=None):
        return jax.grad(lambda tt: loss(corrected_estimate(
            cfg, tt, x, v, dl, noise=nz, gain=gain)[-1]))(0.0)

    batch_grad = jax.jit(jax.vmap(one_grad, in_axes=(0, None)))
    measured, predicted = [], []
    for dl in deltas:
        noise = jnp.asarray(rng.normal(scale=SIGMA, size=(N_MC, T)))
        gs = np.asarray(batch_grad(noise, float(dl)))
        measured.append(float(np.var(gs)))
        predicted.append(float(dloss(z_T) ** 2 * SIGMA ** 2 / dl ** 2))
    measured, predicted = np.array(measured), np.array(predicted)
    slope = float(np.polyfit(np.log(deltas), np.log(measured), 1)[0])
    rel = float(np.max(np.abs(measured - predicted) / predicted))
    print(f"\n  gradient-variance sweep over delta")
    print(f"    {'delta':>8}{'measured Var':>16}{'predicted':>16}{'rel err':>10}")
    for i, dl in enumerate(deltas):
        print(f"    {dl:>8.3f}{measured[i]:>16.6e}{predicted[i]:>16.6e}"
              f"{abs(measured[i]-predicted[i])/predicted[i]:>10.3f}")
    print(f"    fitted log-log slope = {slope:+.4f}   (expected -2)")

    # ---- bias/variance tradeoff for the approximate gain -----------------
    delta_fix = 1.0
    ks = np.linspace(0.05, 1.9, 12) / delta_fix
    tradeoff = []
    batch_grad_gain = jax.jit(jax.vmap(one_grad, in_axes=(0, None, None)))
    for k in ks:
        n_mc_k = 40000
        noise = jnp.asarray(rng.normal(scale=SIGMA, size=(n_mc_k, T)))
        gs = np.asarray(batch_grad_gain(noise, delta_fix, float(k)))
        V = float(np.var(gs))
        eps_g = abs(1.0 - delta_fix * k)
        bound = eps_g + delta_fix * np.sqrt(V) / (abs(dloss(z_T)) * SIGMA)
        tradeoff.append(dict(k=float(k), eps_g=float(eps_g), V=V,
                             bound=float(bound)))
    worst_bound = min(t["bound"] for t in tradeoff)
    # sqrt(V) is estimated from a finite sample, so the bound carries a Monte
    # Carlo error of about 1/sqrt(2 n) in relative terms. Report it rather
    # than letting a loose threshold hide a marginal value.
    mc_rel = 1.0 / np.sqrt(2.0 * 40000)
    print(f"\n  bias/variance tradeoff, min over k of "
          f"eps_g + delta sqrt(V)/(|l'| sigma) = {worst_bound:.4f}  (must be >= 1)")
    print(f"    Monte Carlo relative error on sqrt(V): +/- {mc_rel:.2%}"
          f"  -> bound is {worst_bound:.4f} +/- {mc_rel:.4f}")

    ok = (worst < 1e-10 and abs(g_hat - g_int) < 1e-9
          and abs(slope + 2.0) < 0.05 and rel < 0.15
          and worst_bound > 1.0 - 3 * mc_rel)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}")

    results = dict(noiseless_max_error=worst, grad_hat=g_hat, grad_intended=g_int,
                   grad_match_error=float(abs(g_hat - g_int)),
                   deltas=deltas.tolist(), variance_measured=measured.tolist(),
                   variance_predicted=predicted.tolist(),
                   loglog_slope=slope, max_relative_error=rel,
                   tradeoff=tradeoff, tradeoff_min_bound=float(worst_bound),
                   tradeoff_mc_relative_error=float(mc_rel),
                   verdict_pass=bool(ok))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), facecolor=SURFACE)
    axes[0].loglog(deltas, measured, "o", color=AQUA, ms=7, label="measured")
    axes[0].loglog(deltas, predicted, color="#0b0b0b", lw=1.4, ls=(0, (3, 3)),
                   label=r"$l'(z)^2\sigma^2/\delta^2$")
    style(axes[0], f"Gradient variance, fitted slope {slope:+.3f}",
          r"access strength $\delta$", "Var[grad]", legend=True)

    ksv = [t["k"] for t in tradeoff]
    axes[1].plot(ksv, [t["eps_g"] for t in tradeoff], color=ORANGE, lw=2.2,
                 label=r"bias $\epsilon_g=|1-\delta k|$")
    axes[1].plot(ksv, [delta_fix * np.sqrt(t["V"]) / (abs(dloss(z_T)) * SIGMA)
                       for t in tradeoff], color=BLUE, lw=2.2,
                 label=r"noise $\delta\sqrt{V}/(|l'|\sigma)$")
    axes[1].plot(ksv, [t["bound"] for t in tradeoff], color=AQUA, lw=2.6,
                 label="sum")
    axes[1].axhline(1.0, color="#0b0b0b", lw=1.2, ls=(0, (3, 3)),
                    label="lower bound 1")
    style(axes[1], "Bias/variance tradeoff cannot be beaten", "gain k",
          "value", legend=True)

    regimes = ["none", "weak", "strong"]
    infer = [1, 1, 1]
    learn = [0, 0.45, 1]
    xpos = np.arange(3)
    axes[2].bar(xpos - 0.19, infer, width=0.36, color=MUTED, label="nominal inference")
    axes[2].bar(xpos + 0.19, learn, width=0.36, color=AQUA, label="learning fidelity")
    axes[2].set_xticks(xpos); axes[2].set_xticklabels(regimes)
    axes[2].set_ylim(0, 1.25)
    for i, lab in enumerate(["impossible", "faithful\nbut noisy", "faithful\nbounded"]):
        axes[2].text(i + 0.19, learn[i] + 0.04, lab, ha="center", fontsize=7.5,
                     color="#0b0b0b")
    style(axes[2], "Three access regimes", "access", "quality (schematic)",
          legend=True)

    fig.suptitle("Weak access buys faithful learning at an unavoidable noise "
                 "cost scaling as 1/delta^2", color="#0b0b0b", fontsize=12,
                 x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = f"{d}/fig_weak_access_noise.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    print("wrote", save("3_weak_access_noise",
                        dict(seed=SEED, T=T, sigma=SIGMA, n_mc=N_MC,
                             y_star=Y_STAR, delta_fixed=delta_fix), results),
          "and", path)
    return results


if __name__ == "__main__":
    main()
