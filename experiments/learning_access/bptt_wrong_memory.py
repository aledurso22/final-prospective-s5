"""5. Exact BPTT learns the WRONG memory under restricted access.

Controlled analytic case: rho=0.9, beta=0.8, a=0.7, z0=1, b0=-4, two steps,
zero drives, target y* = 0.95^2 = 0.9025.

    A. LOCAL prospective realization      y = w_2 = z_2 + theta b_2
       inference-perfect at theta = 0, but carries the contamination theta*b

    B. COLLECTIVE learning-faithful       y = z_hat_2 = z_2
       given the b-channel access the port criterion demands

Both are differentiated by the SAME autodiff stack. Analytic predictions:

    d J_local/dtheta|_0 = (0.81 - 0.9025)(2*0.9 - 2.56) = +0.0703  > 0
    d J_coll /dtheta|_0 = (0.81 - 0.9025)(2*0.9)        = -0.1665  < 0

so gradient descent moves them in OPPOSITE directions. Both can reach zero
training loss. The local model does it by exploiting the physical nuisance,
landing at theta = -0.10672 (pole 0.7933) instead of theta = +0.05
(pole 0.95).

The decisive diagnostic is not training loss. It is the loss after the hidden
disturbance is removed.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.learning_access._common import (AQUA, BLUE, MUTED, ORANGE,
                                                 SURFACE, YELLOW, outdir,
                                                 save, style)
from prospective.three_state import ThreeStateConfig, rollout, w_of

SEED = 20260910
RHO, BETA, A = 0.9, 0.8, 0.7
Z0, B0 = 1.0, -4.0
TARGET_POLE = 0.95
STEPS, LR, N_ITERS = 2, 0.03, 12000
# lr chosen for stability: the local objective is a quartic in theta and
# larger steps diverge for large |b0|. Verified 0/40 divergences at this
# setting, versus 15/40 at lr=0.35. Both models use the SAME setting.


def make_task(z0=Z0, b0=B0, x=None, v=None, steps=STEPS):
    cfg = ThreeStateConfig(rho=RHO, beta=BETA, a=A, z0=z0, b0=b0)
    x = jnp.zeros(steps) if x is None else jnp.asarray(x)
    v = jnp.zeros(steps) if v is None else jnp.asarray(v)
    y_star = (TARGET_POLE ** steps) * z0
    return cfg, x, v, y_star


def y_local(cfg, theta, x, v):
    """Local prospective observable: the only thing a local corrector sees."""
    return w_of(cfg, theta, x, v)[-1]


def y_collective(cfg, theta, x, v):
    """Learning-faithful: the b-channel access removes the theta*b term."""
    out = rollout(cfg, theta, x, v)
    return out["z"][-1]


def loss_of(model, cfg, theta, x, v, y_star):
    return 0.5 * (model(cfg, theta, x, v) - y_star) ** 2


def train(model, cfg, x, v, y_star, lr=LR, iters=N_ITERS, theta0=0.0,
          record=True):
    """Exact BPTT (autodiff) gradient descent, run inside lax.scan.

    Returns (theta, history). `record=False` skips the history, which is what
    the randomized sweep uses.
    """
    grad = jax.grad(lambda t: loss_of(model, cfg, t, x, v, y_star))
    lossf = lambda t: loss_of(model, cfg, t, x, v, y_star)

    def step(th, _):
        g = grad(th)
        return th - lr * g, (th, lossf(th), g)

    th_final, traj = jax.lax.scan(step, jnp.asarray(theta0), None, length=iters)
    if not record:
        return float(th_final), None
    ths, ls, gs = (np.asarray(a) for a in traj)
    stride = max(1, iters // 400)
    hist = [(i, float(ths[i]), float(ls[i]), float(gs[i]))
            for i in range(0, iters, stride)]
    hist.append((iters, float(th_final), float(lossf(th_final)),
                 float(grad(th_final))))
    return float(th_final), hist


def diagnostics(model, cfg, theta, x, v, y_star, steps=STEPS):
    """Task-visible diagnostics, including the disturbance-removal test."""
    pole = RHO + theta
    # impulse response of the intended computation, nuisance switched OFF
    n_imp = 30
    clean = ThreeStateConfig(rho=RHO, beta=BETA, a=A, z0=0.0, b0=0.0)
    imp_x = jnp.zeros(n_imp).at[0].set(1.0)
    imp = np.asarray(rollout(clean, theta, imp_x, jnp.zeros(n_imp))["z"])[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        efold = float(-1.0 / np.log(abs(pole))) if 0 < abs(pole) < 1 else np.inf
    # disturbance removal: same trained theta, hidden mode zeroed
    cfg0 = ThreeStateConfig(rho=RHO, beta=BETA, a=A, z0=cfg.z0, b0=0.0)
    y_clean = float(model(cfg0, theta, x, jnp.zeros_like(v)))
    return dict(theta=float(theta), learned_pole=float(pole),
                pole_error=float(abs(pole - TARGET_POLE)),
                efolding_steps=efold,
                train_output=float(model(cfg, theta, x, v)),
                train_loss=float(loss_of(model, cfg, theta, x, v, y_star)),
                output_disturbance_removed=y_clean,
                loss_disturbance_removed=float(0.5 * (y_clean - y_star) ** 2),
                impulse_response=imp.tolist())


def main():
    rng = np.random.default_rng(SEED)
    d = outdir("5_bptt_wrong_memory")
    cfg, x, v, y_star = make_task()

    # ---- analytic gradient check BEFORE training ------------------------
    g_loc = float(jax.grad(lambda t: loss_of(y_local, cfg, t, x, v, y_star))(0.0))
    g_col = float(jax.grad(lambda t: loss_of(y_collective, cfg, t, x, v, y_star))(0.0))
    err0 = 0.81 - 0.9025
    g_loc_pred = err0 * (2 * RHO - 2.56)
    g_col_pred = err0 * (2 * RHO)
    eps = 1e-6
    fd_loc = float((loss_of(y_local, cfg, eps, x, v, y_star)
                    - loss_of(y_local, cfg, -eps, x, v, y_star)) / (2 * eps))
    fd_col = float((loss_of(y_collective, cfg, eps, x, v, y_star)
                    - loss_of(y_collective, cfg, -eps, x, v, y_star)) / (2 * eps))

    print(f"  target pole {TARGET_POLE}, y* = {y_star:.6f}, b_2 = "
          f"{BETA**2*B0:+.4f}")
    print(f"\n  initial gradients at theta = 0")
    print(f"    {'model':<14}{'autodiff':>14}{'finite diff':>14}{'analytic':>14}")
    print(f"    {'LOCAL':<14}{g_loc:>+14.9f}{fd_loc:>+14.9f}{g_loc_pred:>+14.9f}")
    print(f"    {'COLLECTIVE':<14}{g_col:>+14.9f}{fd_col:>+14.9f}{g_col_pred:>+14.9f}")
    signs_ok = g_loc > 0 > g_col
    print(f"    opposite signs as predicted: {signs_ok}")

    # ---- train ----------------------------------------------------------
    th_loc, hist_loc = train(y_local, cfg, x, v, y_star)
    th_col, hist_col = train(y_collective, cfg, x, v, y_star)
    D_loc = diagnostics(y_local, cfg, th_loc, x, v, y_star)
    D_col = diagnostics(y_collective, cfg, th_col, x, v, y_star)

    theta_loc_pred = (0.76 - np.sqrt(0.76 ** 2 + 4 * 0.0925)) / 2
    print(f"\n  after training")
    print(f"    {'quantity':<32}{'LOCAL':>14}{'COLLECTIVE':>14}")
    for key, lab in (("train_loss", "training loss"),
                     ("theta", "learned theta"),
                     ("learned_pole", "learned pole"),
                     ("pole_error", "|pole - 0.95|"),
                     ("efolding_steps", "e-folding horizon (steps)"),
                     ("output_disturbance_removed", "output, nuisance removed"),
                     ("loss_disturbance_removed", "LOSS, nuisance removed")):
        print(f"    {lab:<32}{D_loc[key]:>14.6f}{D_col[key]:>14.6f}")
    print(f"\n    analytic local root  theta = {theta_loc_pred:+.6f} "
          f"(pole {RHO+theta_loc_pred:.6f})")
    print(f"    analytic collective  theta = +0.050000 (pole 0.950000)")

    nuisance_assisted = (D_loc["train_loss"] < 1e-8
                         and D_loc["loss_disturbance_removed"] > 1e-3)
    print(f"\n  both reached zero TRAINING loss: "
          f"{D_loc['train_loss'] < 1e-8 and D_col['train_loss'] < 1e-8}")
    print(f"  local is nuisance-assisted (zero train loss, large clean loss): "
          f"{nuisance_assisted}")

    # ---- randomized sweep over nuisance histories and initial conditions -
    sweep = []
    for _ in range(200):
        z0 = float(rng.uniform(0.5, 1.5))
        b0 = float(rng.normal() * 4.0)
        xr = rng.normal(size=STEPS) * 0.0            # zero drives, as specified
        vr = rng.normal(size=STEPS) * 0.5
        c2, x2, v2, ys2 = make_task(z0=z0, b0=b0, x=xr, v=vr)
        tl, _ = train(y_local, c2, x2, v2, ys2, record=False)
        tc, _ = train(y_collective, c2, x2, v2, ys2, record=False)
        dl = diagnostics(y_local, c2, tl, x2, v2, ys2)
        dc = diagnostics(y_collective, c2, tc, x2, v2, ys2)
        sweep.append(dict(z0=z0, b0=b0,
                          local_pole=dl["learned_pole"],
                          coll_pole=dc["learned_pole"],
                          local_pole_err=dl["pole_error"],
                          coll_pole_err=dc["pole_error"],
                          local_clean_loss=dl["loss_disturbance_removed"],
                          coll_clean_loss=dc["loss_disturbance_removed"],
                          local_finite=bool(np.isfinite(tl)),
                          coll_finite=bool(np.isfinite(tc))))

    # Divergences are reported, never silently averaged away as NaN.
    finite = np.array([s["local_finite"] and s["coll_finite"] for s in sweep])
    n_div = int((~finite).sum())
    lp = np.array([s["local_pole_err"] for s in sweep])[finite]
    cp = np.array([s["coll_pole_err"] for s in sweep])[finite]
    lc = np.array([s["local_clean_loss"] for s in sweep])[finite]
    cc = np.array([s["coll_clean_loss"] for s in sweep])[finite]
    frac = float(np.mean(lp > cp))
    print(f"\n  randomized sweep over {len(sweep)} nuisance histories")
    print(f"    diverged (excluded)  : {n_div}")
    print(f"    median |pole-0.95|   : local {np.median(lp):.4f}   "
          f"collective {np.median(cp):.2e}")
    print(f"    median clean loss    : local {np.median(lc):.4e}  "
          f"collective {np.median(cc):.2e}")
    print(f"    local worse than collective in {frac:.1%} of converged runs")

    ok = (signs_ok and nuisance_assisted and D_col["pole_error"] < 1e-4
          and D_loc["pole_error"] > 0.1 and frac > 0.95)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} - exact BPTT on the local "
          f"realization learns the wrong intended memory")

    results = dict(
        initial_gradients=dict(local_autodiff=g_loc, local_fd=fd_loc,
                               local_analytic=g_loc_pred,
                               collective_autodiff=g_col, collective_fd=fd_col,
                               collective_analytic=g_col_pred,
                               opposite_signs=bool(signs_ok)),
        local=D_loc, collective=D_col,
        analytic_local_theta=float(theta_loc_pred),
        nuisance_assisted=bool(nuisance_assisted),
        sweep_summary=dict(n=len(sweep), n_diverged=n_div,
                           median_local_pole_err=float(np.median(lp)),
                           median_coll_pole_err=float(np.median(cp)),
                           median_local_clean_loss=float(np.median(lc)),
                           median_coll_clean_loss=float(np.median(cc)),
                           fraction_local_worse=frac),
        verdict_pass=bool(ok))
    results["local"].pop("impulse_response")
    results["collective"].pop("impulse_response")

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7), facecolor=SURFACE)
    it = [h[0] for h in hist_loc]
    axes[0].plot(it, [h[1] for h in hist_loc], color=ORANGE, lw=2.4, label="local")
    axes[0].plot(it, [h[1] for h in hist_col], color=AQUA, lw=2.4, label="collective")
    axes[0].axhline(0.05, color="#0b0b0b", lw=1.2, ls=(0, (3, 3)),
                    label=r"intended $\theta$")
    style(axes[0], "Training drives them apart", "iteration", r"$\theta$",
          legend=True)

    axes[1].semilogy(it, [max(h[2], 1e-20) for h in hist_loc], color=ORANGE,
                     lw=2.4, label="local")
    axes[1].semilogy(it, [max(h[2], 1e-20) for h in hist_col], color=AQUA,
                     lw=2.4, ls=(0, (4, 2)), label="collective")
    style(axes[1], "Both reach zero TRAINING loss", "iteration", "training loss",
          legend=True)

    n_imp = 30
    kk = np.arange(1, n_imp + 1)
    axes[2].plot(kk, TARGET_POLE ** (kk - 1), color="#0b0b0b", lw=2.4,
                 label=f"intended, pole {TARGET_POLE}")
    axes[2].plot(kk, D_loc["learned_pole"] ** (kk - 1), color=ORANGE, lw=2.2,
                 ls=(0, (4, 2)), label=f"local, pole {D_loc['learned_pole']:.3f}")
    axes[2].plot(kk, D_col["learned_pole"] ** (kk - 1), color=AQUA, lw=1.6,
                 ls=(0, (2, 2)), label=f"collective, pole {D_col['learned_pole']:.3f}")
    style(axes[2], "Task-visible memory: local is far too short", "step",
          "impulse response", legend=True)

    labels = ["training\nloss", "loss with\nnuisance removed"]
    xp = np.arange(2)
    lv = [max(D_loc["train_loss"], 1e-20), max(D_loc["loss_disturbance_removed"], 1e-20)]
    cv = [max(D_col["train_loss"], 1e-20), max(D_col["loss_disturbance_removed"], 1e-20)]
    axes[3].bar(xp - 0.19, lv, width=0.36, color=ORANGE, label="local")
    axes[3].bar(xp + 0.19, cv, width=0.36, color=AQUA, label="collective")
    axes[3].set_yscale("log"); axes[3].set_xticks(xp)
    axes[3].set_xticklabels(labels, fontsize=8)
    style(axes[3], "The decisive diagnostic", None, "loss", legend=True)

    fig.suptitle("Exact BPTT reaches zero training loss by exploiting the "
                 "physical nuisance, and learns the wrong memory",
                 color="#0b0b0b", fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = f"{d}/fig_bptt_wrong_memory.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    cfg_d = dict(seed=SEED, rho=RHO, beta=BETA, a=A, z0=Z0, b0=B0,
                 steps=STEPS, target_pole=TARGET_POLE, lr=LR, iters=N_ITERS)
    print("wrote", save("5_bptt_wrong_memory", cfg_d, results), "and", path)
    return results


if __name__ == "__main__":
    main()
