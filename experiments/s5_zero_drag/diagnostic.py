"""Is a finite-inertia, ZERO-ordinary-drag S5 mode viable and meaningful?

Local, no training. Reads the actual initialized S5 modes (seeds 301-303,
runner.init_state, identical backbone for every arm) or, with --checkpoint,
trained generalized-arm modes, and evaluates per complex mode, in float64:

  * stability of the IMPLEMENTED discretization (Schur-Cohn, complex coeffs)
    and of the continuous token-time law (complex Routh-Hurwitz), over (T, M)
    at gamma = 0;
  * the zero-drag critical mass and the inertial frequency it implies;
  * pole-zero cancellation, DC gain, low-frequency delay, H-infinity and
    Nyquist gain, impulse/step transients, against Native S5;
  * pole predictions against simulated trajectories;
  * the exact-ZOH realization of the same continuous law, SEPARATELY.

    JAX_PLATFORMS=cpu python -m experiments.s5_zero_drag.diagnostic \
        --out results/s5_zero_drag
"""

import argparse
import glob
import json
import os
import time

import numpy as onp

H = 1.0
T_PROD, M_PROD, GAMMA_PROD = 0.05, 0.025, 1.0
T_GRID = (0.05, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
M_GRID = (0.0, 1e-6, 1e-3, 0.025, 0.1, 1, 10, 100, 1e3, 1e4, 1e5)
FREQS = onp.linspace(0.0, onp.pi, 4097)
COLORS = {"native": "C0", "generalized_init (impl, M=.025, g=1)": "C1",
          "zero_drag (impl, M=.025, g=0)": "C2",
          "ALT exact zero_drag (T=5, M=.1, g=0)": "C3",
          "ALT exact damped (T=5, M=.1, g=1)": "C4"}


def init_modes(seeds=(301, 302, 303)):
    """lambda_bar per mode, exactly as S5SSM.setup forms it (clip, zoh).

    The parameters are the PRODUCTION float32 draw of runner.init_state,
    taken in an x64-off subprocess: with x64 on, jax.random draws different
    log_step and B (Lambda is deterministic). Before 2026-11-17 this function
    drew under x64, i.e. a statistically equivalent but different sample.
    """
    from experiments.s5_predictor_corrector.production_init import (
        production_init_flat)
    out = []
    for seed in seeds:
        flat = production_init_flat("native_matched_s5", seed)
        for key in sorted(flat):
            if key[-1] != "Lambda_re":
                continue
            p = key[:-1]
            lam = (onp.minimum(onp.asarray(flat[key], onp.float64), -1e-4)
                   + 1j * onp.asarray(flat[p + ("Lambda_im",)], onp.float64))
            step = onp.exp(onp.asarray(flat[p + ("log_step",)][..., 0],
                                       onp.float64))
            out.append(dict(seed=seed, layer="/".join(p), Lambda=lam,
                            step=step, lambda_bar=onp.exp(lam * step)))
    return out


def checkpoint_modes(paths):
    """Trained generalized-arm modes: lambda_bar and the learned (T, M, gamma)."""
    from experiments.s5_two_compartment.prospective_drift import (
        _layer_groups, _read)
    from experiments.s5_two_compartment.parameter_sample import leaves
    out = []
    for path in paths:
        for name, g in _layer_groups(_read(path)).items():
            lam = (onp.minimum(onp.asarray(g["Lambda_re"], onp.float64), -1e-4)
                   + 1j * onp.asarray(g["Lambda_im"], onp.float64))
            step = onp.exp(onp.asarray(g["log_step"][:, 0], onp.float64))
            T, rho, gamma, M = leaves(g)
            out.append(dict(seed=path, layer=name, Lambda=lam, step=step,
                            lambda_bar=onp.exp(lam * step), T=T, M=M,
                            gamma=gamma, rho=rho))
    return out


def q(values, pct=(0, 5, 25, 50, 75, 95, 100)):
    v = onp.asarray(values, dtype=onp.float64)
    v = v[onp.isfinite(v)]
    if v.size == 0:
        return None
    return {f"p{p}": float(x) for p, x in zip(pct, onp.percentile(v, pct))}


# ----------------------------------------------------------- responses ---
def implemented(lb, T, M, g):
    from s5.independent_inertia import coefficients_numpy
    return coefficients_numpy(lb, T, M, g)       # per unit b_bar


def exact(lb, T, M, g):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from s5.independent_inertia import exact_zoh_coefficients
    lb = jnp.asarray(onp.atleast_1d(lb), jnp.complex128)
    n = lb.shape[0]
    full = lambda v: jnp.broadcast_to(jnp.asarray(v, jnp.float64), (n,))
    a1, a2, c1, c2 = exact_zoh_coefficients(
        lb, jnp.ones((n, 1), jnp.complex128), full(T), full(M), full(g))
    return (onp.asarray(a1), onp.asarray(a2), onp.asarray(c1[:, 0]),
            onp.asarray(c2[:, 0]))


def native(lb):
    lb = onp.atleast_1d(lb).astype(onp.complex128)
    return lb, onp.zeros_like(lb), onp.ones_like(lb), onp.zeros_like(lb)


def freq_response(coeffs, w=FREQS):
    a1, a2, c1, c2 = (onp.atleast_1d(c)[:, None] for c in coeffs)
    z = onp.exp(1j * w)[None, :]
    return (c1 + c2 / z) / (1 - a1 / z - a2 / z ** 2)


def radius(coeffs):
    a1, a2 = (onp.atleast_1d(c) for c in coeffs[:2])
    root = onp.sqrt(a1 * a1 + 4 * a2)
    return onp.maximum(abs((a1 + root) / 2), abs((a1 - root) / 2))


def simulate(coeffs, x):
    from s5.independent_inertia import recurrence_impulse
    return recurrence_impulse(*coeffs, len(x), x=x)


def metrics(coeffs, lb, n=4000):
    """Per-mode response metrics, each normalized to the mode's own DC gain."""
    H = freq_response(coeffs)
    dc = H[:, 0]
    dc_native = 1.0 / (1.0 - onp.atleast_1d(lb))
    mag = abs(H) / abs(dc)[:, None]
    # delay from the phase slope at the lowest nonzero frequency
    w1 = FREQS[1]
    delay = -onp.angle(H[:, 1] / dc) / w1
    imp = simulate(coeffs, onp.eye(1, n, 0)[0])
    stp = simulate(coeffs, onp.ones(n))
    rad = radius(coeffs)
    return dict(
        radius=rad,
        dc_over_native=abs(dc / dc_native),
        delay_tokens=delay,
        hinf_over_dc=mag.max(1),
        peak_freq=FREQS[mag.argmax(1)],
        nyquist_over_dc=mag[:, -1],
        impulse_peak_over_dc=abs(imp).max(0) / abs(dc),
        step_overshoot=abs(stp).max(0) / abs(dc),
        impulse=imp, step=stp, H=H)


def growth_check(coeffs, n=3000):
    """Late-time log-growth of |impulse| against log(radius) predicted."""
    imp = simulate(coeffs, onp.eye(1, n, 0)[0])
    tail = onp.log(abs(imp[n // 2:]) + 1e-300)
    t = onp.arange(n // 2, n)
    slope = onp.polyfit(t, tail, 1)[0]
    return slope, onp.log(radius(coeffs))


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s5_zero_drag")
    ap.add_argument("--checkpoint", nargs="*", default=[],
                    help="trained generalized checkpoints (msgpack); globs ok")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from s5 import independent_inertia as II

    report = {}
    modes = init_modes()
    lb = onp.concatenate([m["lambda_bar"] for m in modes])
    Lam = onp.concatenate([m["Lambda"] for m in modes])
    report["inventory"] = dict(
        modes=int(lb.size), Lambda_re=sorted(set(onp.round(Lam.real, 6))),
        Lambda_im=sorted(set(onp.round(Lam.imag, 4))),
        abs_lambda_bar=q(abs(lb)), one_minus_abs=q(abs(1 - lb)),
        arg_kappa_deg=q(onp.degrees(onp.angle(1 - lb))))

    # -------- 1. production coefficients vs the closed form, all modes ----
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from s5.discrete_recurrence import generalized_coefficients
    worst = 0.0
    for (T, M, g) in [(T_PROD, M_PROD, GAMMA_PROD), (T_PROD, 0, 0),
                      (T_PROD, 0.3, 0), (3.0, 2.0, 0.5)]:
        n = lb.size
        a1, a2, _, _ = generalized_coefficients(
            jnp.asarray(lb), jnp.ones((n, 1), complex), jnp.full(n, T),
            jnp.full(n, M), jnp.full(n, g))
        c2_, c1_, c0_ = II.operator_polynomial(lb, T, M, g)
        kap = II.stiffness(lb, T)
        closed = ((M + T + g), -2 * (M + T) - g + kap * (1 + T),
                  (M + T) - kap * T)
        worst = max(worst, float(abs(onp.asarray(a1) + c1_ / c2_).max()),
                    float(abs(onp.asarray(a2) + c0_ / c2_).max()),
                    *(float(abs(x - y).max()) for x, y in
                      zip((c2_, c1_, c0_), closed)))
    report["closed_form_max_abs_error"] = worst

    # -------- 2. stability over (T, M) at gamma = 0 -----------------------
    grid = {"T": list(T_GRID), "M": list(M_GRID),
            "implemented_stable": [], "continuous_stable": []}
    for T in T_GRID:
        grid["implemented_stable"].append(
            [int(II.schur_cohn_stable(lb, T, M, 0.0).sum()) for M in M_GRID])
        grid["continuous_stable"].append(
            [int(II.continuous_stable(lb, T, M, 0.0).sum()) if M > 0 else None
             for M in M_GRID])
    report["gamma0_grid_out_of_1152"] = grid

    # minimal damping needed by the implemented scheme, per mode, at fixed M
    def gamma_min(T, M, hi=1e3):
        lo_ok = II.schur_cohn_stable(lb, T, M, 0.0)
        lo, hi_ = onp.zeros_like(abs(lb)), onp.full(abs(lb).shape, hi)
        for _ in range(80):
            mid = 0.5 * (lo + hi_)
            ok = II.schur_cohn_stable(lb, T, M, mid)
            hi_, lo = onp.where(ok, mid, hi_), onp.where(ok, lo, mid)
        return onp.where(lo_ok, 0.0, hi_)
    report["implemented_gamma_min_at_T_prod"] = {
        str(M): q(gamma_min(T_PROD, M)) for M in (0.0, M_PROD, 1.0, 100.0)}

    # continuous critical mass and the inertial frequency at the bound
    Mc = II.zero_drag_critical_mass(lb, T_PROD)
    kap = II.stiffness(lb, T_PROD)
    phi = onp.angle(kap)
    report["continuous_zero_drag_at_T_prod"] = dict(
        critical_mass=q(Mc),
        critical_mass_over_M_prod=q(Mc / M_PROD),
        inertial_freq_at_bound_rad_per_token=q(onp.sqrt(abs(kap) / Mc)),
        closed_form_freq=q(onp.sin(abs(phi)) / (T_PROD * onp.sqrt(onp.cos(phi)))),
        damping_ratio_at_bound=q(onp.sin(abs(phi)) * onp.sqrt(onp.cos(phi)) / 2))
    # resolvable inertia needs omega_n <= 1 rad/token at the bound
    report["T_needed_for_resolvable_inertia_tokens"] = q(
        onp.sin(abs(phi)) / onp.sqrt(onp.cos(phi)))

    # -------- 3. structure: cancellation, DC gain, delay ------------------
    for name, (T, M, g) in {"generalized_init": (T_PROD, M_PROD, GAMMA_PROD),
                            "zero_drag_matched": (T_PROD, M_PROD, 0.0),
                            "zucchet": (T_PROD, 0.0, 0.0),
                            "first_order_point": (T_PROD, 0.0, 1.0)}.items():
        res = abs(II.cancellation_residual(lb, T, M, g)) / abs(
            II.operator_polynomial(lb, T, M, g)[0])
        report.setdefault("cancellation_residual_normalized", {})[name] = q(res)
    report["low_freq_delay_tokens"] = {
        "native": q((1 / (1 - lb)).real - 1),
        "generalized_init": q(II.low_frequency_delay(lb, T_PROD, M_PROD, 1.0)),
        "zero_drag_any_M": q(II.low_frequency_delay(lb, T_PROD, M_PROD, 0.0))}

    # -------- 4. arm comparison over the full inventory --------------------
    arms = {
        "native": native(lb),
        "generalized_init (impl, M=.025, g=1)": implemented(lb, T_PROD, M_PROD, 1.0),
        "zero_drag (impl, M=.025, g=0)": implemented(lb, T_PROD, M_PROD, 0.0),
        "ALT exact zero_drag (T=.05, M=0.5 Mc, g=0)": exact(lb, T_PROD, 0.5 * Mc, 0.0),
        "ALT exact damped (T=.05, M=0.5 Mc, g=1)": exact(lb, T_PROD, 0.5 * Mc, 1.0),
        "ALT exact zero_drag (T=5, M=.1, g=0)": exact(lb, 5.0, 0.1, 0.0),
        "ALT exact damped (T=5, M=.1, g=1)": exact(lb, 5.0, 0.1, 1.0),
    }
    table = {}
    for name, coeffs in arms.items():
        rad = radius(coeffs)
        stable = rad < 1
        entry = {"stable_modes": int(stable.sum()), "radius": q(rad)}
        if stable.any():
            sub = tuple(onp.atleast_1d(c)[stable] for c in coeffs)
            m = metrics(sub, lb[stable], n=2000)
            for k in ("dc_over_native", "delay_tokens", "hinf_over_dc",
                      "peak_freq", "nyquist_over_dc", "impulse_peak_over_dc",
                      "step_overshoot"):
                entry[k] = q(m[k])
        table[name] = entry
    report["arms_over_inventory"] = table

    # -------- 5. pole predictions against trajectories ---------------------
    checks = []
    pick = onp.argsort(abs(lb))[[-1, lb.size // 2, 0]]
    for name in ("generalized_init (impl, M=.025, g=1)",
                 "zero_drag (impl, M=.025, g=0)",
                 "ALT exact zero_drag (T=5, M=.1, g=0)"):
        sub = tuple(onp.atleast_1d(c)[pick] for c in arms[name])
        slope, pred = growth_check(sub, n=1500)
        checks.append({"arm": name, "measured_log_growth": slope.tolist(),
                       "predicted_log_radius": pred.tolist()})
    report["pole_vs_trajectory"] = checks

    # -------- 6. trained modes, if given -----------------------------------
    paths = [p for pat in args.checkpoint for p in sorted(glob.glob(pat))]
    if paths:
        tm = checkpoint_modes(paths)
        lt = onp.concatenate([m["lambda_bar"] for m in tm])
        T, M, g = (onp.concatenate([m[k] for m in tm]) for k in ("T", "M", "gamma"))
        report["trained"] = dict(
            checkpoints=paths,
            learned_radius=q(radius(implemented(lt, T, M, g))),
            zero_drag_same_T_M_stable=int(II.schur_cohn_stable(lt, T, M, 0 * g).sum()),
            continuous_zero_drag_same_T_M_stable=int(
                II.continuous_stable(lt, T, M, 0 * g).sum()),
            critical_mass_over_learned_M=q(II.zero_drag_critical_mass(lt, T) / M),
            arg_kappa_deg=q(onp.degrees(onp.angle(1 - lt))),
            cancellation_residual=q(abs(II.cancellation_residual(lt, T, M, g))))
    else:
        report["trained"] = "no checkpoints given; run with --checkpoint"

    with open(os.path.join(args.out, "diagnostic.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    figures(lb, arms, args.out)
    print(json.dumps(report, indent=1, default=float))


def figures(lb, arms, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phi = onp.degrees(onp.angle(1 - lb))
    pick = {"slowest |λ̄|": int(onp.argmax(abs(lb))),
            "least oscillatory (min arg κ)": int(onp.argmin(phi)),
            "median": int(onp.argsort(abs(lb))[lb.size // 2])}
    show = ["native", "generalized_init (impl, M=.025, g=1)",
            "zero_drag (impl, M=.025, g=0)",
            "ALT exact zero_drag (T=5, M=.1, g=0)",
            "ALT exact damped (T=5, M=.1, g=1)"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for col, (label, i) in enumerate(pick.items()):
        for name in show:
            c = tuple(onp.atleast_1d(v)[[i]] for v in arms[name])
            Hm = freq_response(c)[0]
            dc = abs(Hm[0])
            imp = simulate(c, onp.eye(1, 600, 0)[0])[:, 0]
            stp = simulate(c, onp.ones(600))[:, 0]
            clip = lambda v: onp.clip(v, -50, 50)
            unstable = radius(c)[0] >= 1
            style = dict(lw=1, ls="--" if unstable else "-",
                         color=COLORS[name])
            tag = f" [UNSTABLE, ρ={radius(c)[0]:.3f}]" if unstable else ""
            axes[0, col].plot(clip(imp.real / dc), label=name + tag, **style)
            axes[1, col].plot(clip(stp.real / dc), **style)
            if not unstable:   # |H(e^iw)| is not a response for rho >= 1
                axes[2, col].loglog(FREQS[1:], abs(Hm[1:]) / dc, **style)
        axes[0, col].set_title(f"{label}: |λ̄|={abs(lb[i]):.4f}, arg κ={phi[i]:.0f}°")
        axes[0, col].set_ylabel("impulse / DC (Re, clipped ±50)")
        axes[1, col].set_ylabel("step / DC (Re, clipped ±50)")
        axes[2, col].set_ylabel("|H| / DC"); axes[2, col].set_xlabel("rad / token")
    for ax in axes[0]:
        ax.legend(fontsize=6)
    fig.suptitle("Impulse, step and frequency response per unit DC gain. "
                 "Dashed = unstable (frequency response omitted). "
                 "ALT = exact-ZOH scheme, not the implemented discretization.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "responses.png"), dpi=110)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"elapsed {time.time() - t0:.1f}s")
