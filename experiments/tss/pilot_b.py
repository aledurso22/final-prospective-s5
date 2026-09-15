"""Part B: credit assignment on ONE frozen forward model.

The forward model, its parameters, the data and the loss are held fixed. Only
the way the error is propagated in TIME changes. That is what makes this the
experiment that could support a credit-assignment claim; changing the forward
model belongs to Part A and must not be mixed in here.

Network: two layers of four retained-compartment cells, arranged SPATIALLY
FEEDFORWARD -

    layer 1 drive:  f1 = W1 x + b1            (no dependence on any state)
    layer 2 drive:  f2 = W2 tanh(s1) + b2

so the error coupling is strictly triangular and there is no backward feedback
loop. That is deliberate: Section 7 of the causal-error note exhibits a stable
forward node whose reciprocal error loop is UNSTABLE, and this pilot is not
entitled to assume error-dynamics stability from forward stability. The witness
is reported alongside the results.

THE EXACT DRIVE ADJOINT IS OBTAINED WITHOUT A NEW DERIVATION. Each layer's
per-step drive carries an additive perturbation d_t, held over the step exactly
as the drive is. Then

    rho_exact(t) = dJ / d(d_t)

is the exact discrete drive adjoint, and because r_t is held over the same
step, the identity G_W = sum_t rho_t r_t^T is EXACT for this discretization -
which is checked numerically rather than assumed. Every approximation then
differs from the reference ONLY in the temporal filter, which isolates what is
being compared.
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from s5 import causal_error as CE                                  # noqa: E402
from s5.tss_cells import DT, symmetric_reference                   # noqa: E402

N_UNITS = 4
N_LAYERS = 2
D_IN = 4
SEQ_LEN = 64
N_TRAJ = 32
N_SUB = 2
#: predeclared input bandwidths, in rad/step. Reported in full; the comparison
#: is NOT made at whichever bandwidth flatters either approximation.
BANDWIDTHS = (0.05, 0.15, 0.40)
#: predeclared approximation timescales, as fractions of the forward model's
#: FAST timescale t_- . delta = eps for the reciprocal's strictly proper tail.
EPS_FRACTIONS = (0.25, 0.5, 1.0)
#: norm-matched update sizes for the loss-change metric
STEP_NORMS = (1e-3, 1e-2)
#: below this reference-gradient norm a cosine is undefined and the ABSOLUTE
#: error is reported instead of a ratio
NEAR_ZERO = 1e-12


# ----------------------------------------------------------- the network ---
def init_params(seed):
    rng = onp.random.RandomState(seed)
    def g(shape, fan_in):
        return (rng.randn(*shape) / onp.sqrt(fan_in)).astype(onp.float64)
    return dict(W1=g((N_UNITS, D_IN), D_IN), b1=onp.zeros(N_UNITS),
                W2=g((N_UNITS, N_UNITS), N_UNITS), b2=onp.zeros(N_UNITS),
                z0_1=onp.zeros((2, N_UNITS)), z0_2=onp.zeros((2, N_UNITS)))


def _cell_step(z, f, gamma, T, M, n_sub=N_SUB, dt=DT):
    """One input step of the retained cell with the drive `f` held constant."""
    h = dt / n_sub

    def vf(z):
        s, p = z[0], z[1]
        R = s - f
        return jnp.stack([(p - T * R) / M,
                          -(gamma / M) * p + (gamma * T / M - 1.0) * R])

    def one(z, _):
        k1 = vf(z); k2 = vf(z + 0.5 * h * k1)
        k3 = vf(z + 0.5 * h * k2); k4 = vf(z + h * k3)
        return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

    z, _ = jax.lax.scan(one, z, None, length=n_sub)
    return z


def forward(params, xs, d1, d2, coef):
    """Both layers over one trajectory. `d1`, `d2` are drive perturbations.

    Layer 2 consumes layer 1's output HELD over the step, which is what makes
    `G_W = sum_t rho_t r_t^T` exact for this cascade.
    """
    gamma, T, M = coef["gamma"], coef["T"], coef["M"]

    def l1(z, inp):
        x, d = inp
        f = params["W1"] @ x + params["b1"] + d
        z = _cell_step(z, f, gamma, T, M)
        return z, z[0]

    _, s1 = jax.lax.scan(l1, params["z0_1"], (xs, d1))

    def l2(z, inp):
        s1t, d = inp
        f = params["W2"] @ jnp.tanh(s1t) + params["b2"] + d
        z = _cell_step(z, f, gamma, T, M)
        return z, z[0]

    _, s2 = jax.lax.scan(l2, params["z0_2"], (s1, d2))
    return s1, s2


def loss(params, xs, ys, d1, d2, coef):
    _, s2 = forward(params, xs, d1, d2, coef)
    return 0.5 * jnp.mean(jnp.sum((s2 - ys) ** 2, axis=-1))


def zero_drives():
    return (jnp.zeros((SEQ_LEN, N_UNITS)), jnp.zeros((SEQ_LEN, N_UNITS)))


# --------------------------------------------------------------- the data -
def band_limited(rng, n, length, dim, omega):
    """Unit-variance noise with spectral support |w| <= omega rad/step."""
    w = onp.fft.rfftfreq(length) * 2.0 * onp.pi
    keep = (w <= omega).astype(float)
    z = rng.randn(n, length, dim)
    Z = onp.fft.rfft(z, axis=1) * keep[None, :, None]
    out = onp.fft.irfft(Z, n=length, axis=1)
    sd = out.std(axis=(0, 1), keepdims=True)
    return (out / onp.maximum(sd, 1e-12)).astype(onp.float64)


# ------------------------------------------------- exact reference pieces --
def exact_drive_adjoints(params, xs, ys, coef):
    """rho_exact per layer: dJ/d(drive perturbation). The reference."""
    d1, d2 = zero_drives()
    g = jax.grad(loss, argnums=(3, 4))(params, xs, ys, d1, d2, coef)
    return g[0], g[1]


def exact_param_grad(params, xs, ys, coef):
    d1, d2 = zero_drives()
    return jax.grad(loss)(params, xs, ys, d1, d2, coef)


def teaching_inputs(params, xs, ys, coef, rho2):
    """k per layer, given a layer-2 teaching variable.

    k2 = c2 = dJ/ds2 (layer 2's drive has no state dependence, so A2 = 0).
    k1 = diag(tanh'(s1)) W2^T rho2, the transposed instantaneous Jacobian of
    layer 2's drive with respect to layer 1's state. Spatial weight transport
    is still required; temporal causality does not remove it.
    """
    s1, s2 = forward(params, xs, *zero_drives(), coef)
    c2 = (s2 - ys) / s2.shape[0]
    dphi = 1.0 - jnp.tanh(s1) ** 2
    k1 = dphi * (rho2 @ params["W2"])
    return k1, c2, s1, s2


def assemble_grads(params, xs, s1, rho1, rho2):
    """G_W = sum_t rho_t r_t^T, exact for this cascade's discretization."""
    return dict(W1=rho1.T @ xs, b1=jnp.sum(rho1, axis=0),
                W2=rho2.T @ jnp.tanh(s1), b2=jnp.sum(rho2, axis=0))


def exact_backward_filter(k, coef, n_sub=N_SUB):
    """H_A[k] computed EXACTLY, by running H backward in time.

    H_A(p) = H(-p), so filtering the time-reversed signal with the stable
    causal H and reversing again gives the anticausal adjoint filter. This is
    the continuous-time target; comparing it with `rho_exact` separates the
    filter-approximation error from the continuous-versus-discrete gap.
    """
    gamma, T, M = coef["gamma"], coef["T"], coef["M"]
    kr = k[::-1]

    def step(z, f):
        z = _cell_step(z, f, gamma, T, M, n_sub)
        return z, z[0]

    _, sr = jax.lax.scan(step, jnp.zeros((2, k.shape[1])), kr)
    return sr[::-1]


# ------------------------------------------------------------- the metrics -
def compare(g_hat, g_ref, blocks=("W1", "b1", "W2", "b2")):
    flat_h = onp.concatenate([onp.asarray(g_hat[b]).ravel() for b in blocks])
    flat_r = onp.concatenate([onp.asarray(g_ref[b]).ravel() for b in blocks])
    nr, nh = onp.linalg.norm(flat_r), onp.linalg.norm(flat_h)
    out = dict(ref_norm=float(nr), approx_norm=float(nh),
               abs_error=float(onp.linalg.norm(flat_h - flat_r)))
    if nr < NEAR_ZERO or nh < NEAR_ZERO:
        out.update(cosine=None, relative_error=None,
                   note="reference or approximation near zero; cosine "
                        "undefined, absolute error reported instead")
    else:
        out.update(cosine=float(flat_h @ flat_r / (nr * nh)),
                   relative_error=float(out["abs_error"] / nr))
    out["per_block"] = {}
    for b in blocks:
        a = onp.asarray(g_hat[b]).ravel(); r = onp.asarray(g_ref[b]).ravel()
        n_r, n_a = onp.linalg.norm(r), onp.linalg.norm(a)
        out["per_block"][b] = dict(
            ref_norm=float(n_r), abs_error=float(onp.linalg.norm(a - r)),
            cosine=(None if n_r < NEAR_ZERO or n_a < NEAR_ZERO
                    else float(a @ r / (n_r * n_a))),
            relative_error=(None if n_r < NEAR_ZERO
                            else float(onp.linalg.norm(a - r) / n_r)))
    return out


def loss_change(params, xs, ys, coef, g, step_norm,
                blocks=("W1", "b1", "W2", "b2")):
    """Actual objective change after a small NORM-MATCHED descent step."""
    flat = onp.concatenate([onp.asarray(g[b]).ravel() for b in blocks])
    n = onp.linalg.norm(flat)
    if n < NEAR_ZERO:
        return None
    d1, d2 = zero_drives()
    base = float(loss(params, xs, ys, d1, d2, coef))
    new = dict(params)
    for b in blocks:
        new[b] = params[b] - step_norm * onp.asarray(g[b]) / n
    return float(loss(new, xs, ys, d1, d2, coef)) - base


def jsonable(o):
    if isinstance(o, onp.integer):
        return int(o)
    if isinstance(o, (onp.floating, float)):
        return float(o)
    if isinstance(o, complex):
        return [float(o.real), float(o.imag)]
    if isinstance(o, onp.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, int, str)) or o is None:
        return o
    return str(o)


def write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/tss_pilot")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--seeds", default="100,101,102")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)
    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")

    coef = symmetric_reference()
    gamma, T, M = coef["gamma"], coef["T"], coef["M"]
    from s5.tss_cells import q_roots
    t_minus, t_plus = q_roots(gamma, T, M)
    seeds = [int(s) for s in args.seeds.split(",")]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id, "part_b")
    os.makedirs(out, exist_ok=True)

    # ---- the audit, BEFORE any comparison
    audit = dict(
        coefficients=coef, q_roots=dict(t_minus=t_minus, t_plus=t_plus),
        circuit_inequality_M_le_gamma_T=bool(0 < M <= gamma * T),
        recurrent_witness=CE.recurrent_witness(),
        spatial_arrangement=("feedforward: layer 1's drive has no state "
                             "dependence and layer 2's depends only on layer "
                             "1, so the error coupling is triangular and the "
                             "Section 7 instability cannot arise here. "
                             "Recurrent error dynamics are NOT validated by "
                             "this pilot."),
        initialization_derivative=(
            "the reference differentiates z0_1 and z0_2; NEITHER causal "
            "approximation supplies an initial-state derivative, because its "
            "error states start at zero while the exact adjoint has a TERMINAL "
            "condition. The asymmetry is reported, and the gradient comparison "
            "is restricted to W1, b1, W2, b2."),
        boundary_prescription=(
            "causal error states start at zero; the exact adjoint is "
            "terminal-valued, so the two disagree near the sequence ends by "
            "construction. Interior-window metrics are reported separately."),
        filters={})
    for frac in EPS_FRACTIONS:
        eps = frac * t_minus
        delta = eps
        entry = dict(eps=eps, delta=delta, eps_fraction_of_t_minus=frac,
                     peak_gain=CE.peak_gain_bound(gamma, T, M, eps))
        for kind in ("moment", "reciprocal"):
            entry[kind] = dict(
                poles=CE.filter_poles(kind, gamma, T, M, eps, delta),
                band_error={str(Om): CE.measured_band_error(
                    kind, gamma, T, M, eps, delta, Om) for Om in BANDWIDTHS})
        entry["moment"]["remainder"] = {
            str(Om): CE.remainder_bound(gamma, T, M, eps, Om)
            for Om in BANDWIDTHS}
        audit["filters"][str(frac)] = entry
    print("[audit] circuit inequality M <= gamma*T:",
          audit["circuit_inequality_M_le_gamma_T"])
    print("[audit] recurrent witness: forward pole "
          f"{audit['recurrent_witness']['forward_pole']:.6f}, reciprocal "
          f"error pole {audit['recurrent_witness']['reciprocal_error_pole']:+.6f}")
    for frac, e in audit["filters"].items():
        print(f"[audit] eps={e['eps']:.3f} ({frac} t-)  peak gain bound "
              f"{e['peak_gain']['bound']:.3e}  moment poles stable="
              f"{e['moment']['poles']['stable']}  reciprocal stable="
              f"{e['reciprocal']['poles']['stable']}")

    status = dict(part="B", run_id=run_id, out=out, backend=backend,
                  seeds=seeds, n_units=N_UNITS, n_layers=N_LAYERS,
                  n_trajectories=N_TRAJ, seq_len=SEQ_LEN,
                  bandwidths=list(BANDWIDTHS),
                  eps_fractions=list(EPS_FRACTIONS),
                  step_norms=list(STEP_NORMS), audit=audit,
                  results=[], incomplete=[])
    write(os.path.join(out, "status.json"), status)

    rows = []
    for seed in seeds:
        params = {k: jnp.asarray(v) for k, v in init_params(seed).items()}
        rng = onp.random.RandomState(seed + 5)
        for Om in BANDWIDTHS:
            if time.time() > deadline - args.reserve_s:
                status["incomplete"].append(
                    f"seed {seed} bandwidth {Om}: budget")
                break
            xs_all = jnp.asarray(band_limited(rng, N_TRAJ, SEQ_LEN, D_IN, Om))
            ys_all = jnp.asarray(band_limited(rng, N_TRAJ, SEQ_LEN, N_UNITS,
                                              Om))

            # --- reference, and its independent cross-checks
            per_traj_ref, per_traj_hat = [], {}
            g_ref_sum = None
            for i in range(N_TRAJ):
                xs, ys = xs_all[i], ys_all[i]
                r1, r2 = exact_drive_adjoints(params, xs, ys, coef)
                k1, c2, s1, s2 = teaching_inputs(params, xs, ys, coef, r2)
                g_id = assemble_grads(params, xs, s1, r1, r2)
                g_ref = exact_param_grad(params, xs, ys, coef)
                per_traj_ref.append((g_ref, g_id, r1, r2, k1, c2, s1, xs, ys))
                g_ref_sum = (g_ref if g_ref_sum is None else
                             jax.tree_util.tree_map(lambda a, b: a + b,
                                                    g_ref_sum, g_ref))

            # identity check: sum_t rho_t r_t^T must reproduce dJ/dW exactly
            ident = compare(per_traj_ref[0][1], per_traj_ref[0][0])
            # forward-mode cross-check of the reference, including z0
            xs0, ys0 = per_traj_ref[0][7], per_traj_ref[0][8]
            d1, d2 = zero_drives()
            fwd = jax.jacfwd(loss)(params, xs0, ys0, d1, d2, coef)
            fwd_vs_rev = {b: float(onp.max(onp.abs(
                onp.asarray(fwd[b]) - onp.asarray(per_traj_ref[0][0][b]))))
                for b in params}
            # central finite differences on random directions
            fd = []
            rs = onp.random.RandomState(1234)
            for b in ("W1", "W2", "z0_1", "z0_2"):
                v = rs.randn(*onp.asarray(params[b]).shape)
                v /= onp.linalg.norm(v)
                h = 1e-6
                pp = dict(params); pp[b] = params[b] + h * v
                pm = dict(params); pm[b] = params[b] - h * v
                num = (float(loss(pp, xs0, ys0, d1, d2, coef))
                       - float(loss(pm, xs0, ys0, d1, d2, coef))) / (2 * h)
                ana = float(onp.sum(onp.asarray(per_traj_ref[0][0][b]) * v))
                fd.append(dict(block=b, finite_difference=num, analytic=ana,
                               rel=abs(num - ana) / max(abs(ana), 1e-12)))

            # --- the two approximations, at every predeclared eps
            band = dict(bandwidth=Om, seed=seed,
                        drive_adjoint_identity=ident,
                        forward_vs_reverse_max_abs=fwd_vs_rev,
                        finite_difference_checks=fd, approximations={})
            for frac in EPS_FRACTIONS:
                eps = frac * t_minus
                delta = eps
                for kind in ("moment", "reciprocal"):
                    A, B, w = (CE.moment_matched_system(gamma, T, M, eps)
                               if kind == "moment"
                               else CE.reciprocal_system(gamma, T, M, eps,
                                                         delta))
                    cos_list, rel_list, g_sum = [], [], None
                    rho_cos = []
                    for (g_ref, _gid, r1, r2, k1, c2, s1, xs, ys) in per_traj_ref:
                        # layer 2 first (k2 = c2), then layer 1 through the
                        # transposed Jacobian of layer 2's drive
                        rh2 = CE.run_filter(A, B, w, onp.asarray(c2))
                        dphi = 1.0 - onp.tanh(onp.asarray(s1)) ** 2
                        kk1 = dphi * (rh2 @ onp.asarray(params["W2"]))
                        rh1 = CE.run_filter(A, B, w, kk1)
                        g_hat = assemble_grads(params, xs, s1,
                                               jnp.asarray(rh1),
                                               jnp.asarray(rh2))
                        m = compare(g_hat, g_ref)
                        cos_list.append(m["cosine"])
                        rel_list.append(m["relative_error"])
                        g_sum = (g_hat if g_sum is None else
                                 jax.tree_util.tree_map(lambda a, b: a + b,
                                                        g_sum, g_hat))
                        a = onp.asarray(rh2).ravel()
                        b_ = onp.asarray(r2).ravel()
                        if onp.linalg.norm(a) > NEAR_ZERO \
                                and onp.linalg.norm(b_) > NEAR_ZERO:
                            rho_cos.append(float(a @ b_ / (onp.linalg.norm(a)
                                                           * onp.linalg.norm(b_))))
                    good = [c for c in cos_list if c is not None]
                    key = f"{kind}_eps{frac}"
                    entry = dict(
                        kind=kind, eps=eps, delta=delta,
                        mean_cosine=(float(onp.mean(good)) if good else None),
                        min_cosine=(float(onp.min(good)) if good else None),
                        negative_cosine_fraction=(
                            float(onp.mean([c < 0 for c in good]))
                            if good else None),
                        mean_relative_error=(
                            float(onp.mean([r for r in rel_list
                                            if r is not None]))
                            if any(r is not None for r in rel_list) else None),
                        mean_layer2_rho_cosine=(float(onp.mean(rho_cos))
                                                if rho_cos else None),
                        batch=compare(
                            jax.tree_util.tree_map(lambda v: v / N_TRAJ, g_sum),
                            jax.tree_util.tree_map(lambda v: v / N_TRAJ,
                                                   g_ref_sum)),
                        loss_change={})
                    gb = jax.tree_util.tree_map(lambda v: v / N_TRAJ, g_sum)
                    gr = jax.tree_util.tree_map(lambda v: v / N_TRAJ,
                                                g_ref_sum)
                    for sn in STEP_NORMS:
                        entry["loss_change"][str(sn)] = dict(
                            approximation=loss_change(params, xs_all[0],
                                                      ys_all[0], coef, gb, sn),
                            reference=loss_change(params, xs_all[0],
                                                  ys_all[0], coef, gr, sn))
                    band["approximations"][key] = entry
                    print(f"[seed {seed}] Om={Om:<5} {kind:<10} eps={eps:5.2f}"
                          f"  cos={entry['mean_cosine']}"
                          f"  rel={entry['mean_relative_error']}"
                          f"  neg={entry['negative_cosine_fraction']}")
            rows.append(band)
            status["results"] = rows
            write(os.path.join(out, "results.json"), rows)
            write(os.path.join(out, "status.json"), status)

    status["wall_s"] = time.time() - t0
    status["complete"] = (len(rows) == len(seeds) * len(BANDWIDTHS)
                          and not status["incomplete"])
    write(os.path.join(out, "status.json"), status)
    print(f"[*] wall {status['wall_s']:.0f}s  rows {len(rows)}/"
          f"{len(seeds) * len(BANDWIDTHS)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[INCOMPLETE] {m}")
        print(f"TSS_B_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"TSS_B_STATUS=COMPLETE out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
