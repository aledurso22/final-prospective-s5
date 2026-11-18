"""Read-only physical-input diagnostics for proximal positive gradient memory.

No datasets, checkpoints or accuracy selection are used. The native backbone
is drawn in a separate float32 subprocess, then independent SciPy reference
equations are evaluated in float64. Every initialized mode in all six layers
is retained. Sampled frequency maxima are estimates, not certified H-infinity
bounds. Finite impulse/step metrics use every token, including interference.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from experiments.s5_predictor_corrector.production_init import production_init_flat


def quant(values):
    values = np.asarray(values, float).ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return dict(zip(("min", "p5", "median", "p95", "max"),
                    map(float, np.percentile(values, [0, 5, 50, 95, 100]))))


def positive_real_margin(omega, fast, slow, weight, mu):
    """Real part of mu + K(i omega)/(i omega), continued to omega=0."""
    omega = np.asarray(omega, float)
    return mu - (1-weight)*fast/(1+(fast*omega)**2) - weight*slow/(1+(slow*omega)**2)


def reference(kappa, variant="mix", fast_ratio=.1, slow_ratio=1.,
              slow_weight=4/9, alpha=1., proximal_ratio=.5):
    """Independent physical-input state matrix, scaled state, exact ZOH.

    For mix, state=(s,h*v_fast,h*v_slow). For P state=(s,h*v,h*q).
    The input is b=Delta*B*u; thus native transfer is 1/(p+kappa).
    """
    if not np.isfinite(kappa) or kappa.real <= 0:
        raise ValueError("kappa must be finite with positive real part")
    if alpha <= 0 or proximal_ratio < 0 or not np.isfinite(alpha+proximal_ratio):
        raise ValueError("invalid horizon or proximal ratio")
    h = alpha/abs(kappa)
    mu = proximal_ratio*h
    R = 1+mu*kappa
    effective = kappa/R
    if variant == "native":
        A = np.asarray([[-kappa]], complex)
        b = np.asarray([1.], complex)
    elif variant == "proximal":
        A = np.asarray([[0, 1/h, 0], [-2*effective, -1/h, 1/h],
                        [effective, 0, -1/h]], complex)
        b = np.asarray([0, 2, -1], complex)/R
    elif variant == "mix":
        if not (0 < fast_ratio < slow_ratio and 0 <= slow_weight <= 1):
            raise ValueError("require 0<fast<slow and weight in [0,1]")
        fast, slow = fast_ratio*h, slow_ratio*h
        A = np.asarray([[0, (1-slow_weight)/h, slow_weight/h],
                        [-h*effective/fast, -1/fast, 0],
                        [-h*effective/slow, 0, -1/slow]], complex)
        b = np.asarray([0, h/fast, h/slow], complex)/R
    else:
        raise ValueError(variant)
    aug = np.zeros((len(A)+1, len(A)+1), complex)
    aug[:-1, :-1], aug[:-1, -1] = A, b
    discrete = expm(aug)
    return A, b, discrete[:-1, :-1], discrete[:-1, -1], h


def modal_residues(A, b):
    """C*(pI-A)^-1*b residues, C selecting s; coordinates do not matter."""
    poles, right = np.linalg.eig(A)
    residues = right[0, :] * np.linalg.solve(right, b)
    order = np.argsort(-poles.real)
    separation = np.inf if len(poles) == 1 else min(
        abs(poles[i]-poles[j]) for i in range(len(poles)) for j in range(i))
    return poles[order], residues[order], float(np.linalg.cond(right)), float(separation)


def finite_responses(Ad, bd, kappa, steps=16000):
    """Exact finite-horizon sums by recurrence, not modal energy proxies."""
    state = np.asarray(bd, complex).copy()
    native_pole = np.exp(-kappa)
    native = -np.expm1(-kappa)/kappa
    step = np.zeros_like(kappa)
    step_peak = np.zeros_like(kappa.real)
    energy = np.zeros_like(kappa.real)
    native_energy = np.zeros_like(kappa.real)
    error_energy = np.zeros_like(kappa.real)
    l1 = np.zeros_like(kappa.real)
    peak_impulse = np.zeros_like(kappa.real)
    first = state[:, 0].copy()
    for _ in range(steps):
        value = state[:, 0]
        step += value
        step_peak = np.maximum(step_peak, abs(step))
        energy += abs(value)**2
        native_energy += abs(native)**2
        error_energy += abs(value-native)**2
        l1 += abs(value)
        peak_impulse = np.maximum(peak_impulse, abs(value))
        last = value.copy()
        state = np.einsum("nij,nj->ni", Ad, state)
        native *= native_pole
    return dict(impulse_energy=energy,
                impulse_energy_over_native=energy/native_energy,
                impulse_relative_l2_vs_native=np.sqrt(error_energy/native_energy),
                impulse_l1=l1, impulse_peak=peak_impulse,
                impulse_first_abs=abs(first), impulse_last_abs=abs(last),
                step_peak_over_dc=step_peak*abs(kappa),
                step_final_relative_error=abs(step*kappa-1))


def frequency_peak(Ad, bd, kappa):
    """Full signed band plus pole-centered adaptive mesh, no real-mode shortcut."""
    poles, right = np.linalg.eig(Ad)
    residue = right[0] * np.linalg.solve(right, bd)
    grid = [np.linspace(-np.pi, np.pi, 4097)]
    for pole in poles:
        width = max(-np.log(abs(pole)), 1e-10)
        grid.append(np.angle(pole)+width*np.linspace(-24, 24, 1001))
    frequencies = (np.concatenate(grid)+np.pi)%(2*np.pi)-np.pi
    phase = np.exp(-1j*frequencies[:, None])
    response = np.sum(residue/(1-poles*phase), axis=1)
    winner = int(np.argmax(abs(response)))
    native_pole = np.exp(-kappa)
    native_b = -np.expm1(-kappa)/kappa
    native_peak = abs(native_b)/(1-abs(native_pole))
    return float(abs(response[winner])), float(frequencies[winner]), float(native_peak)


def initialized_layers(seed):
    """Float32 RNG subprocess, then the same float32 JAX Lambda/Delta algebra."""
    import jax.numpy as jnp
    flat = production_init_flat("native_matched_s5", seed)
    layers = []
    for key in sorted(flat):
        if key[-1] != "Lambda_re":
            continue
        prefix = key[:-1]
        re = jnp.asarray(flat[key], jnp.float32)
        im = jnp.asarray(flat[prefix+("Lambda_im",)], jnp.float32)
        delta = jnp.exp(jnp.asarray(flat[prefix+("log_step",)], jnp.float32)[:, 0])
        kappa = np.asarray(-(jnp.clip(re, None, -1e-4)+1j*im)*delta, np.complex128)
        B = np.asarray(flat[prefix+("B",)], float)
        B = (B[..., 0]+1j*B[..., 1])*np.asarray(delta)[:, None]
        C1 = np.asarray(flat[prefix+("C1",)], float)
        C2 = np.asarray(flat[prefix+("C2",)], float)
        c_norm = np.sqrt(np.sum(abs(C1[..., 0]+1j*C1[..., 1])**2, axis=0)
                         +np.sum(abs(C2[..., 0]+1j*C2[..., 1])**2, axis=0))
        layers.append(dict(layer="/".join(prefix), kappa=kappa,
                           b_norm=np.linalg.norm(B, axis=1), c_norm=c_norm))
    if len(layers) != 6 or sum(len(x["kappa"]) for x in layers) != 384:
        raise ValueError("expected full six-layer, 384-mode production backbone")
    return layers


def inspect_variant(layers, name, kwargs, steps):
    kappa = np.concatenate([x["kappa"] for x in layers])
    coupling = np.concatenate([x["b_norm"]*x["c_norm"] for x in layers])
    blocks = [reference(k, **kwargs) for k in kappa]
    Ad = np.stack([x[2] for x in blocks]); bd = np.stack([x[3] for x in blocks])
    poles, residues, conditions, separations = [], [], [], []
    dc_error, decay_ratio, delay, peak, peak_frequency, native_peak = [], [], [], [], [], []
    for k, (A, b, ad, drive, h) in zip(kappa, blocks):
        p, r, cond, sep = modal_residues(A, b)
        poles.append(p); residues.append(r); conditions.append(cond); separations.append(sep)
        dc = -np.linalg.solve(A, b)[0]
        dc_error.append(abs(dc*k-1))
        # H'(0)=-C*A^-2*b; -H'(0)/H(0) is complex low-frequency lag.
        delay.append(float((np.linalg.solve(A, np.linalg.solve(A, b))[0]/dc).real))
        decay_ratio.append(float(k.real/-np.max(p.real)))
        fpeak, fwhere, npeak = frequency_peak(ad, drive, k)
        peak.append(fpeak); peak_frequency.append(fwhere); native_peak.append(npeak)
    poles, residues = np.asarray(poles), np.asarray(residues)
    radius = np.exp(np.max(poles.real, axis=1))
    finite = finite_responses(Ad, bd, kappa, steps)
    relative_residue = abs(residues)/np.sum(abs(residues), axis=1)[:, None]
    # Each component's finite energy excludes interference; explicitly a proxy.
    discrete_poles = np.exp(poles)
    discrete_residues = residues*np.expm1(poles)/poles
    geom = -np.expm1(2*steps*poles.real)/(-np.expm1(2*poles.real))
    components = abs(discrete_residues)**2*geom
    fractions = components/finite["impulse_energy"][:, None]
    mode_records = []
    offset = 0
    for layer in layers:
        count = len(layer["kappa"])
        ids = slice(offset, offset+count)
        mode_records.append(dict(layer=layer["layer"], n_modes=count,
            kappa_real=kappa[ids].real.tolist(), kappa_imag=kappa[ids].imag.tolist(),
            continuous_poles_real=poles[ids].real.tolist(),
            continuous_poles_imag=poles[ids].imag.tolist(),
            continuous_residues_real=residues[ids].real.tolist(),
            continuous_residues_imag=residues[ids].imag.tolist(),
            continuous_residue_abs_share=relative_residue[ids].tolist(),
            component_energy_over_exact_total=fractions[ids].tolist(),
            radius=radius[ids].tolist(), dc_delay=delay[ids],
            sampled_peak_frequency=peak_frequency[ids],
            sampled_frequency_peak=np.asarray(peak)[ids].tolist(),
            finite_response={k: v[ids].tolist() for k, v in finite.items()}))
        offset += count
    return dict(name=name, parameters=kwargs, n_modes=len(kappa),
                continuous_unstable=int(np.sum(poles.real.max(1)>=0)),
                exact_zoh_unstable=int(np.sum(radius>=1)),
                min_continuous_decay_rate=float(-poles.real.max()),
                radius=quant(radius), pole_separation=quant(separations),
                modal_basis_condition=quant(conditions),
                continuous_dc_relative_error_max=float(max(dc_error)),
                continuous_dc_delay=quant(delay),
                slow_decay_time_over_native=quant(decay_ratio),
                sampled_peak_gain_over_native_peak=quant(np.asarray(peak)/native_peak),
                coupled_peak_gain_proxy=quant(coupling*np.asarray(peak)),
                slowest_pole_residue_abs_share=quant(relative_residue[:, 0]),
                auxiliary_component_energy_over_exact_total=quant(fractions[:, 1:].sum(1)),
                finite_response={k: quant(v) for k, v in finite.items()}, layers=mode_records)


def sensitivity_variants():
    variants = [("native", dict(variant="native")),
                ("proximal_alpha1", dict(variant="proximal")),
                ("mix_declared", dict(variant="mix"))]
    for fast in (.05, .1, .25):
        for slow in (1., 2., 4.):
            if fast == .1 and slow == 1.:
                continue
            weight = (.5-fast)/(slow-fast)
            variants.append((f"sensitivity_fast{fast}_slow{slow}",
                dict(variant="mix", fast_ratio=fast, slow_ratio=slow, slow_weight=weight)))
    return variants


def compact_report(report, raw_path, raw_sha256):
    """Preserve every layer's summaries without repeating full modal arrays."""
    result = {k: v for k, v in report.items() if k != "variants"}
    result["raw_evidence"] = dict(path=str(raw_path), sha256=raw_sha256,
                                  note="Full modal arrays; gitignored, reproducible from this diagnostic.")
    result["variants"] = []
    for original in report["variants"]:
        variant = {k: v for k, v in original.items() if k != "layers"}
        layers = []
        for row in original["layers"]:
            layer = {"layer": row["layer"], "n_modes": row["n_modes"]}
            def small_quant(value):
                q = quant(value)
                return None if q is None else {k: q[k] for k in ("min", "median", "max")}
            for key in ("kappa_real", "kappa_imag", "radius", "dc_delay",
                        "sampled_frequency_peak"):
                layer[key] = small_quant(row[key])
            for key in ("continuous_poles_real", "continuous_residue_abs_share"):
                array = np.asarray(row[key])
                layer[key] = [small_quant(array[:, i]) for i in range(array.shape[1])]
            layer["auxiliary_component_energy_over_exact_total"] = small_quant(
                np.asarray(row["component_energy_over_exact_total"])[:, 1:].sum(1))
            layer["finite_response"] = {k: small_quant(row["finite_response"][k]) for k in
                ("impulse_energy_over_native", "impulse_relative_l2_vs_native",
                 "step_peak_over_dc", "step_final_relative_error")}
            layers.append(layer)
        variant["layer_summaries"] = layers
        result["variants"].append(variant)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out", default="results/s5_proximal_memory_mix/initial_diagnostic_raw.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[301, 302])
    ap.add_argument("--steps", type=int, default=16000)
    args = ap.parse_args()
    destination = Path(args.out)
    raw_destination = Path(args.raw_out)
    for path in (destination, raw_destination):
        if path.exists():
            raise FileExistsError(path)
    if destination.resolve() == raw_destination.resolve():
        raise ValueError("summary and raw output paths must differ")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    started = time.perf_counter()
    records = []
    for seed in args.seeds:
        layers = initialized_layers(seed)
        for name, kwargs in sensitivity_variants():
            row = inspect_variant(layers, name, kwargs, args.steps)
            row["seed"] = seed
            records.append(row)
            print(seed, name, "unstable", row["continuous_unstable"],
                  "decay/native median", row["slow_decay_time_over_native"]["median"],
                  "peak/native median", row["sampled_peak_gain_over_native_peak"]["median"], flush=True)
    report = dict(scope="Read-only initialized-mode diagnostics; no training or accuracy-based selection.",
        initialization="Separate float32 production RNG subprocess; float32 JAX kappa algebra; float64 SciPy analysis.",
        input="Physical continuous b=Delta*B*u, held constant over each token; zero initial dynamic state.",
        frequency_scope="Signed [-pi,pi] sampled frequency maxima with pole-centered adaptive refinement; not certified H-infinity bounds.",
        residue_scope="Auxiliary energy divided by exact total retains denominator interference; numerator component energies omit cross terms, can exceed one.",
        coupling_scope="B/C norm product is a per-mode coupling proxy, not whole-layer gain; bidirectional readouts pooled in norm.",
        stability_scope="Continuous and exact-ZOH poles from independent float64 reference; production float32 checked separately by core tests/smoke.",
        seeds=args.seeds, steps=args.steps, wall_seconds=time.perf_counter()-started, variants=records)
    raw_destination.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = (json.dumps(report, separators=(",", ":"), allow_nan=False)+"\n").encode()
    raw_destination.write_bytes(raw_bytes)
    summary = compact_report(report, raw_destination, hashlib.sha256(raw_bytes).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, separators=(",", ":"), allow_nan=False)+"\n")
    print("wrote", destination, flush=True)


if __name__ == "__main__":
    main()
