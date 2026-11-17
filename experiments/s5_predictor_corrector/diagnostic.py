"""Timescale calibration vs correction-at-prediction: local 2x2 diagnostic.

No training. Per complex mode of the ACTUAL initialized backbone (seeds
301-303, runner.init_state, identical for every arm), in float64:

                      old corrector    prediction corrector
    legacy  k=T            A                    B
    calibr. k=gamma        C                    D

all at the ORIGINAL (T, rho, gamma) init, M = rho gamma T, against native S5.
Reports poles, margins, residues, DC gain and delay, impulse/step/frequency
responses (full [-pi, pi] band: the modes are complex), l1 gain over 16,000
steps, pole-vs-trajectory agreement, a stability grid with stable-old /
unstable-new search, and whole-layer output differences at init. Trained
checkpoints are handled by checkpoint_compare.py, not here.

The initialization is the float32 production draw (production_init.py);
JAX_ENABLE_X64 only affects the float64 analysis.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m \
        experiments.s5_predictor_corrector.diagnostic \
        --out results/s5_predictor_corrector
"""

import argparse
import glob
import json
import os
import time

import numpy as onp

from s5 import predictor_corrector as PC

N_STEPS = 16000
FREQS = onp.linspace(-onp.pi, onp.pi, 8193)
PCT = (0, 5, 25, 50, 75, 95, 100)


def q(values):
    v = onp.asarray(values, float).ravel()
    v = v[onp.isfinite(v)]
    if v.size == 0:
        return None
    return {f"p{p}": float(onp.percentile(v, p)) for p in PCT}


def init_params():
    """(T, M, gamma, rho) exactly as the module forms them at init."""
    import jax.numpy as np
    from s5.generalized_prospective_ssm import response_mass_gamma
    raw = (onp.log(onp.expm1(0.05)), 0.0, onp.log(onp.expm1(1.0)))
    T, M, g, rho = response_mass_gamma(*(np.asarray([v], np.float32) for v in raw))
    return tuple(float(onp.asarray(v)[0]) for v in (T, M, g, rho))


def impulse_and_step(co, n=N_STEPS):
    """h_t per mode and the peak |step response|, via the scalar recurrence."""
    a1, a2, g1, g2 = (onp.asarray(v, complex) for v in co)
    h = onp.empty((n,) + a1.shape, complex)
    s1 = onp.zeros_like(a1); s2 = onp.zeros_like(a1)
    acc = onp.zeros_like(a1); peak = onp.zeros(a1.shape)
    for t in range(n):
        s = a1 * s1 + a2 * s2 + (g1 if t == 0 else 0) + (g2 if t == 1 else 0)
        h[t] = s; s2, s1 = s1, s
        acc += s; peak = onp.maximum(peak, abs(acc))
    return h, peak


def arm_metrics(name, co, lb, native_h, native_H, partner=None):
    a1, a2, g1, g2 = co
    z1, z2, R1, R2 = PC.residues(*co)
    rad = onp.maximum(abs(z1), abs(z2))
    stable = PC.schur_cohn(a1, a2)
    K = 1 - lb
    dc = PC.dc_gain(*co)
    H = PC.transfer(*co, onp.exp(1j * FREQS)[:, None])
    h, step_peak = impulse_and_step(co)
    l1 = abs(h).sum(0)
    disc = abs(a1 * a1 + 4 * a2) / onp.maximum(abs(a1) ** 2, 1e-300)
    # COMPONENT-energy proxy: |R2|^2/(1-|z2|^2) over the sum of both
    # component energies. The slow/fast interference (cross) terms are
    # omitted, so this is NOT the exact share of total impulse-response energy;
    # the exact total sum|h_t|^2 is reported alongside for comparison.
    with onp.errstate(divide="ignore", invalid="ignore"):
        e2 = abs(R2) ** 2 / (1 - abs(z2) ** 2)
        e1 = abs(R1) ** 2 / (1 - abs(z1) ** 2)
        # exact infinite-horizon total: sum_t |R1 z1^t + R2 z2^t|^2
        cross = 2 * (R1 * onp.conj(R2) / (1 - z1 * onp.conj(z2))).real
        total = e1 + e2 + cross
    out = dict(
        stable=int(stable.sum()), n=int(lb.size),
        radius=q(rad), margin=q(1 - rad),
        slow_pole_shift=q(abs(z1 - lb)),
        slow_time_constant_ratio=q(onp.log(abs(lb)) / onp.log(abs(z1))),
        fast_pole_abs=q(abs(z2)),
        residue_ratio_fast_over_slow=q(abs(R2) / abs(R1)),
        fast_component_energy_proxy=q(e2 / (e1 + e2)),
        fast_component_energy_share_exact_denominator=q(e2 / total),
        cross_term_over_total_energy=q(abs(cross) / total),
        discriminant_over_a1sq_min=float(disc.min()),
        dc_gain_rel_err_max=float(onp.max(abs(dc - 1 / K) * abs(K))),
        dc_group_delay=q(PC.group_delay_dc(*co)),
        dc_group_delay_minus_native=q(PC.group_delay_dc(*co)
                                      - PC.group_delay_dc(*PC.native_coefficients(lb))),
        freq_rel_l2_vs_native=q(onp.sqrt((abs(H - native_H) ** 2).mean(0)
                                         / (abs(native_H) ** 2).mean(0))),
        hinf_over_native_hinf=q(abs(H).max(0) / abs(native_H).max(0)),
        impulse_rel_l2_vs_native=q(onp.sqrt((abs(h - native_h) ** 2).sum(0)
                                            / (abs(native_h) ** 2).sum(0))),
        l1_gain_over_native=q(l1 / abs(native_h).sum(0)),
        step_peak_over_dc=q(step_peak / abs(dc)),
        tail_abs_at_16000=q(abs(h[-1])),
    )
    if partner is not None:
        pH, ph = partner
        out["freq_rel_l2_vs_old_corrector"] = q(onp.sqrt(
            (abs(H - pH) ** 2).mean(0) / (abs(pH) ** 2).mean(0)))
        out["impulse_rel_l2_vs_old_corrector"] = q(onp.sqrt(
            (abs(h - ph) ** 2).sum(0) / (abs(ph) ** 2).sum(0)))
    return out, H, h


def by_abs_K(lb, T, M, g):
    """Where the calibrated and corrected variants differ from native."""
    W = onp.exp(1j * FREQS)[:, None]
    rel = lambda a, b: onp.sqrt((abs(a - b) ** 2).mean(0) / (abs(b) ** 2).mean(0))
    rows = []
    K = abs(1 - lb)
    for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 2.0)):
        s = (K >= lo) & (K < hi)
        nat = PC.transfer(*PC.native_coefficients(lb[s]), W)
        row = dict(abs_K=[lo, hi], n=int(s.sum()))
        Hs = {}
        for v in ("C", "D"):
            co = PC.scalar_coefficients(lb[s], T, M, g, *PC.VARIANTS[v])
            Hs[v] = PC.transfer(*co, W)
            z, _ = PC.roots(co[0], co[1])
            row[f"{v}_freq_rel_vs_native_median"] = float(onp.median(rel(Hs[v], nat)))
            row[f"{v}_slow_time_constant_ratio_median"] = float(
                onp.median(onp.log(abs(lb[s])) / onp.log(abs(z))))
        row["D_vs_C_freq_rel_median"] = float(onp.median(rel(Hs["D"], Hs["C"])))
        rows.append(row)
    return rows


def scaled_step_equivalence(modes, T, M, g):
    """Is variant A native S5 with the step shrunk by T/gamma? (ZOH: the DC
    gain -B/Lambda does not depend on the step, so no normalization.)"""
    lb = onp.concatenate([m["lambda_bar"] for m in modes])
    Lam = onp.concatenate([m["Lambda"] for m in modes])
    step = onp.concatenate([m["step"] for m in modes])
    lb_s = onp.exp(Lam * step * T / g)
    W = onp.exp(1j * FREQS)[:, None]
    A = PC.transfer(*PC.scalar_coefficients(lb, T, M, g, "legacy", "old"), W)
    # B_bar ratio between the two steps: B_bar = (lambda_bar - 1)/Lambda B
    native_s = PC.transfer(*PC.native_coefficients(lb_s), W) * (lb_s - 1) / (lb - 1)
    native = PC.transfer(*PC.native_coefficients(lb), W)
    rel = lambda a, b: onp.sqrt((abs(a - b) ** 2).mean(0) / (abs(b) ** 2).mean(0))
    K = abs(1 - lb)
    return dict(A_vs_native_scaled_step=q(rel(A, native_s)),
                A_vs_native=q(rel(A, native)),
                A_vs_native_scaled_step_absK_lt_0_1=q(rel(A, native_s)[K < 0.1]),
                note="per mode, unit B_tilde; transfer of the scaled-step mode "
                     "carries its own B_bar")


def pole_vs_trajectory(co, n=3000):
    """Log-growth slope of |h_t| over the tail vs log of the spectral radius."""
    h, _ = impulse_and_step(co, n)
    z1, z2 = PC.roots(co[0], co[1])
    rad = onp.maximum(abs(z1), abs(z2))
    t0, t1 = n // 2, n - 1
    ok = (abs(h[t1]) > 1e-250) & (abs(h[t0]) > 1e-250)
    slope = (onp.log(abs(h[t1][ok])) - onp.log(abs(h[t0][ok]))) / (t1 - t0)
    return float(onp.max(abs(slope - onp.log(rad[ok])))) if ok.any() else None


def stability_grid(lb):
    Ts = (0.05, 0.5, 1, 2, 5, 10, 50)
    gs = (0.01, 0.1, 0.5, 1, 2, 10, 100)
    rhos = (1e-4, 0.5, 1.0)
    rows = []
    for scale in PC.TARGET_SCALES:
        for T in Ts:
            for g in gs:
                for r in rhos:
                    M = r * g * T
                    so = PC.schur_cohn(*PC.scalar_coefficients(lb, T, M, g, scale, "old")[:2])
                    sn = PC.schur_cohn(*PC.scalar_coefficients(lb, T, M, g, scale, "prediction")[:2])
                    rows.append(dict(scale=scale, T=T, gamma=g, rho=r,
                                     old_stable=int(so.sum()),
                                     new_stable=int(sn.sum()),
                                     old_ok_new_bad=int((so & ~sn).sum()),
                                     new_ok_old_bad=int((sn & ~so).sum())))
    return rows


def random_search(n=2_000_000, seed=1):
    """Stable-old / unstable-new over random in-disc modes and parameters."""
    rng = onp.random.default_rng(seed)
    lb = rng.uniform(0, 1, n) ** 0.3 * onp.exp(1j * rng.uniform(-onp.pi, onp.pi, n))
    T = 10 ** rng.uniform(-3, 3, n); g = 10 ** rng.uniform(-3, 3, n)
    M = rng.uniform(1e-4, 1, n) * g * T
    out = {}
    for scale in PC.TARGET_SCALES:
        o = PC.scalar_coefficients(lb, T, M, g, scale, "old")
        nn = PC.scalar_coefficients(lb, T, M, g, scale, "prediction")
        so, sn = PC.schur_cohn(o[0], o[1]), PC.schur_cohn(nn[0], nn[1])
        k = PC.target_scale(T, g, scale)
        _, eta, _ = PC.residual_coefficients(T, M, g, k)
        P = abs(1 - eta * (1 - lb))
        bad = so & ~sn
        out[scale] = dict(samples=n, old_stable=float(so.mean()),
                          new_stable=float(sn.mean()),
                          old_ok_new_bad=int(bad.sum()),
                          old_ok_new_bad_with_absP_lt_1=int((bad & (P < 1)).sum()),
                          new_ok_old_bad=int((sn & ~so).sum()),
                          old_ok_new_bad_abs_lb_gt_0_9=int((bad & (abs(lb) > 0.9)).sum()))
    return out


def layer_outputs(seeds, L=2000):
    """Every production SSM layer at its ACTUAL float32 init (x64-off draw),
    evaluated in float64: ||y_variant - y_native|| / ||y_native|| and
    ||y_new - y_old||, all variants sharing the same parameters."""
    import jax
    import jax.numpy as np
    from experiments.s5_three_arm_full import runner as R
    from experiments.s5_predictor_corrector.production_init import (
        production_init_flat)
    from flax.traverse_util import unflatten_dict
    from s5.ssm import S5SSM
    jax.config.update("jax_enable_x64", True)
    kw = R.ssm_kwargs("generalized_prospective_s5")
    rows = []
    nrm = lambda a: float(onp.linalg.norm(a))
    for seed in seeds:
        flat = production_init_flat("generalized_prospective_s5", seed)
        layers = sorted({k[:-1] for k in flat if k[-1] == "Lambda_re"})
        x = jax.random.normal(jax.random.PRNGKey(seed + 1000), (L, kw["H"]))
        acc = {}
        for layer in layers:
            leaf = {k[-1]: np.asarray(v, np.float64) for k, v in flat.items()
                    if k[:-1] == layer}
            p = {"params": unflatten_dict({(k,): v for k, v in leaf.items()})}
            native_p = {"params": {k: v for k, v in p["params"].items()
                                   if not k.startswith("generalized_")}}
            Dx = onp.asarray(leaf["D"]) * onp.asarray(x)
            ys = {v: onp.asarray(PC.PredictorCorrectorS5SSM(
                **kw, scale=sc, corrector=co).apply(p, x)) - Dx
                for v, (sc, co) in PC.VARIANTS.items()}
            ys["native"] = onp.asarray(S5SSM(**kw).apply(native_p, x)) - Dx
            for k, v in ys.items():
                acc.setdefault(k, []).append(v)
        ssm = {k: onp.stack(v) for k, v in acc.items()}
        rows.append(dict(
            seed=seed, n_layers=len(layers),
            **{f"{v}_vs_native": nrm(ssm[v] - ssm["native"]) / nrm(ssm["native"])
               for v in PC.VARIANTS},
            B_vs_A=nrm(ssm["B"] - ssm["A"]) / nrm(ssm["A"]),
            D_vs_C=nrm(ssm["D"] - ssm["C"]) / nrm(ssm["C"]),
            ssm_rms={k: nrm(v) / onp.sqrt(v.size) for k, v in ssm.items()}))
    return rows


def figures(lb, arms_H, arms_h, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = {"slowest mode": int(onp.argmax(abs(lb))),
             "median |K| mode": int(onp.argsort(abs(1 - lb))[lb.size // 2]),
             "fastest-rotating mode": int(onp.argmax(abs(onp.angle(lb))))}
    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    styles = {"native": dict(c="k", lw=2.5, alpha=.35), "A": dict(c="C1"),
              "B": dict(c="C1", ls="--"), "C": dict(c="C0"),
              "D": dict(c="C0", ls="--")}
    for j, (title, i) in enumerate(picks.items()):
        for name, H in arms_H.items():
            h = arms_h[name][:, i]
            n = min(len(h), 400)
            ax[0, j].plot(abs(h[:n]), label=name, **styles[name])
            ax[1, j].plot(abs(onp.cumsum(h)[:n]), **styles[name])
            ax[2, j].semilogy(FREQS, abs(H[:, i]), **styles[name])
        ax[0, j].set_title(f"{title}: lambda_bar={lb[i]:.4f}")
        ax[0, j].set_ylabel("|impulse|"); ax[1, j].set_ylabel("|step|")
        ax[2, j].set_ylabel("|H(e^iw)|"); ax[2, j].set_xlabel("w (rad/token)")
    ax[0, 0].legend()
    fig.suptitle("Native vs A(k=T,old) B(k=T,pred) C(k=gamma,old) D(k=gamma,pred) "
                 "at the original init (T=.05, rho=.5, gamma=1)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "responses.png"), dpi=110)


def analyse(lb, T, M, g, with_responses=True):
    nat = PC.native_coefficients(lb)
    nH = PC.transfer(*nat, onp.exp(1j * FREQS)[:, None])
    nh, _ = impulse_and_step(nat)
    res = {"native": dict(radius=q(abs(lb)), margin=q(1 - abs(lb)),
                          dc_group_delay=q(PC.group_delay_dc(*nat)),
                          l1_gain_over_dc=q(abs(nh).sum(0) * abs(1 - lb)))}
    Hs, hs = {"native": nH}, {"native": nh}
    for v, (scale, corr) in PC.VARIANTS.items():
        co = PC.scalar_coefficients(lb, T, M, g, scale, corr)
        partner = None
        if corr == "prediction":
            partner = (Hs["A" if scale == "legacy" else "C"],
                       hs["A" if scale == "legacy" else "C"])
        res[v], Hs[v], hs[v] = arm_metrics(v, co, lb, nh, nH, partner)
        if corr == "prediction":     # only the old-corrector partners are
            hs[v] = hs[v][:400]      # needed in full; keep memory bounded
        res[v]["pole_vs_trajectory_max_err"] = pole_vs_trajectory(co)
        k = PC.target_scale(T, g, scale)
        mu, eta, zeta = PC.residual_coefficients(T, M, g, k)
        res[v]["coefficients"] = dict(k=float(onp.mean(k)), mu=q(mu), eta=q(eta),
                                      zeta=q(zeta), eta_over_1_minus_mu=q(eta / (1 - mu)))
    return res, Hs, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s5_predictor_corrector")
    ap.add_argument("--skip-layer", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    from experiments.s5_zero_drag.diagnostic import init_modes
    seeds = (301, 302, 303)
    modes = init_modes(seeds)
    lb = onp.concatenate([m["lambda_bar"] for m in modes])
    T, M, g, rho = init_params()
    report = dict(
        config=dict(seeds=seeds, n_modes=int(lb.size), h=1.0, T=T, M=M,
                    gamma=g, rho=rho, n_steps=N_STEPS,
                    variants={k: dict(scale=s, corrector=c)
                              for k, (s, c) in PC.VARIANTS.items()}),
        inventory=dict(abs_lambda_bar=q(abs(lb)), abs_K=q(abs(1 - lb)),
                       arg_lambda_bar=q(onp.angle(lb))))
    res, Hs, hs = analyse(lb, T, M, g)
    report["init"] = res
    report["by_abs_K"] = by_abs_K(lb, T, M, g)
    report["A_as_native_with_scaled_step"] = scaled_step_equivalence(modes, T, M, g)
    report["stability_grid"] = stability_grid(lb)
    report["random_search"] = random_search()
    # reference point: k = gamma, gamma = 2, M = T (rho = 1/2)
    co = PC.scalar_coefficients(lb, T, T, 2.0, "calibrated", "old")
    x = onp.random.default_rng(0).normal(size=2000)
    nat = PC.simulate(*PC.native_coefficients(lb), x)
    new = PC.simulate(*PC.scalar_coefficients(lb, T, T, 2.0, "calibrated", "prediction"), x)
    report["reference_point_gamma2_M_eq_T"] = dict(
        old_max_rel_err_vs_native=float(abs(PC.simulate(*co, x) - nat).max() / abs(nat).max()),
        new_rel_l2_vs_native=float(onp.linalg.norm(new - nat) / onp.linalg.norm(nat)))
    if not args.skip_layer:
        report["layer_outputs_at_init"] = layer_outputs(seeds)
    report["trained"] = ("not evaluated here; use "
                         "experiments.s5_predictor_corrector.checkpoint_compare")
    figures(lb, Hs, hs, args.out)
    report["seconds"] = time.time() - t0
    with open(os.path.join(args.out, "diagnostic.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("config", "inventory", "init",
                                             "random_search",
                                             "reference_point_gamma2_M_eq_T")},
                     indent=1)[:20000])
    if "layer_outputs_at_init" in report:
        print(json.dumps(report["layer_outputs_at_init"], indent=1))
    print("seconds", report["seconds"])


if __name__ == "__main__":
    main()
