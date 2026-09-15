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
from s5.checkpointing import provenance                            # noqa: E402
from s5.tss_cells import (DT, assert_numeric_pytree, q_roots,      # noqa: E402
                          reference_metadata, symmetric_reference)

N_UNITS = 4
N_LAYERS = 2
D_IN = 4
SEQ_LEN = 64
N_TRAJ = 32
N_SUB = 2
#: Predeclared input bands, in rad/step, as INTERVALS. R4: a single cutoff
#: applied as a DFT mask on a 64-sample window kept only DC below 0.0982
#: rad/step. Frequencies are now drawn continuously inside each band, and every
#: band's lowest frequency completes at least about a third of a period in the
#: window. All three are reported; the comparison is NOT made at whichever band
#: flatters either approximation.
BANDS = ((0.03, 0.08), (0.12, 0.25), (0.35, 0.60))
#: sinusoids summed per channel
N_TONES = 4
#: predeclared approximation timescales, as fractions of the forward model's
#: FAST timescale t_- . delta = eps for the reciprocal's strictly proper tail.
EPS_FRACTIONS = (0.25, 0.5, 1.0)
#: norm-matched update sizes for the loss-change metric
STEP_NORMS = (1e-3, 1e-2)
#: below this reference-gradient norm a cosine is undefined and the ABSOLUTE
#: error is reported instead of a ratio
NEAR_ZERO = 1e-12

#: R8, predeclared CORRECTNESS tolerances on the REFERENCE actually used in the
#: reported comparison. Failing any of these invalidates the reference, so the
#: run reports FAILED and refuses dependent interpretation. They are not
#: scientific criteria.
MAX_IDENTITY_REL = 1e-9        # G_W = sum_t rho_t r_t^T against jax.grad
MAX_FWD_REV_ABS = 1e-9         # forward-mode against reverse-mode
MAX_FD_REL = 1e-5              # central differences against the analytic value

#: R8, predeclared SCIENTIFIC rule, committed before execution.
#:   usable          : at EVERY band and EVERY eps, mean cosine > 0, the
#:                     negative-cosine fraction is EXACTLY 0.0, and the
#:                     norm-matched update decreases the batch objective at
#:                     BOTH declared step norms, for EVERY seed.
#:   preferable      : additionally, lower mean relative error than the
#:                     GLE-inspired baseline at EVERY band, not at a chosen one.
#: An unfavourable verdict leaves the execution status PASS.
NEGATIVE_COSINE_TOLERANCE = 0.0


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


def cell_rollout(f_seq, coef, n_units=N_UNITS):
    """The EXECUTED discrete drive -> state map of one layer, zero initial
    state. Linear in `f`, which is what makes `discrete_transpose` exact."""
    gamma, T, M = coef["gamma"], coef["T"], coef["M"]

    def step(z, f):
        z = _cell_step(z, f, gamma, T, M)
        return z, z[0]

    _, s = jax.lax.scan(step, jnp.zeros((2, n_units)), f_seq)
    return s


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


def batch_loss(params, xs, ys, d1, d2, coef):
    """R3: THE objective. Every gradient and every loss change refers to this
    mean over all trajectories.

    The previous revision averaged gradients over 32 trajectories and then
    measured the resulting update against trajectory 0's loss alone. An exact
    gradient of a batch mean need not decrease one member's loss, so that
    mismatch could have rejected the exact reference itself.
    """
    per = jax.vmap(loss, in_axes=(None, 0, 0, None, None, None))(
        params, xs, ys, d1, d2, coef)
    return jnp.mean(per)


def zero_drives():
    return (jnp.zeros((SEQ_LEN, N_UNITS)), jnp.zeros((SEQ_LEN, N_UNITS)))


# --------------------------------------------------------------- the data -
def band_signals(rng, n, length, dim, band):
    """Random-phase sinusoids at CONTINUOUS frequencies drawn inside `band`.

    R4. The previous revision masked the rFFT of white noise at |w| <= Omega.
    For 64 samples at dt = 1 the first nonzero DFT bin is 2*pi/64 = 0.0982
    rad/step, so the declared Omega = 0.05 band retained ONLY DC and every
    "slow" trajectory was constant in time; Omega = 0.15 kept a single bin.
    Those are not varying slow signals, and boundary transients would have
    dominated the comparison they were meant to probe.

    Frequencies are now drawn continuously inside each predeclared band, which
    is a property of the GENERATING PROCESS and is not the same thing as the
    DFT of a finite observation window. The drawn frequencies and the realized
    temporal variance are both recorded.
    """
    lo, hi = band
    freqs = rng.uniform(lo, hi, size=(n, dim, N_TONES))
    phase = rng.uniform(0.0, 2.0 * onp.pi, size=(n, dim, N_TONES))
    t = onp.arange(length)[None, None, None, :]
    x = onp.cos(freqs[..., None] * t + phase[..., None]).sum(axis=2)
    x = onp.transpose(x, (0, 2, 1))                     # (n, length, dim)
    sd = x.std(axis=(0, 1), keepdims=True)
    return (x / onp.maximum(sd, 1e-12)).astype(onp.float64), freqs


def signal_report(x, freqs, band):
    return dict(band=list(band), n_tones=int(N_TONES),
                frequency_min=float(freqs.min()),
                frequency_max=float(freqs.max()),
                temporal_variance=float(onp.mean(onp.var(x, axis=1))),
                note=("band of the GENERATING process; the DFT of a 64-sample "
                      "window is not the same object"))


def spectral_centroid(k):
    """Where the adjoint DRIVE actually has its energy. `k` is (L, ...).

    R4: limiting the input band does not bound the bandwidth of the error
    signal after a nonlinear layer, so the drive's own spectrum is measured
    rather than inherited from the input's.
    """
    k = onp.asarray(k)
    flat = k.reshape(k.shape[0], -1)
    F = onp.abs(onp.fft.rfft(flat, axis=0)) ** 2
    w = onp.fft.rfftfreq(flat.shape[0]) * 2.0 * onp.pi
    tot = F.sum(axis=0) + 1e-30
    cen = (F * w[:, None]).sum(axis=0) / tot
    cum = onp.cumsum(F, axis=0) / tot[None, :]
    idx = onp.argmax(cum >= 0.9, axis=0)
    return dict(centroid=float(onp.mean(cen)),
                f90=float(onp.mean(w[idx])),
                max_grid_frequency=float(w[-1]),
                note="measured on the executed drive, not assumed from the "
                     "input band")


# ------------------------------------------------- exact reference pieces --
#: R9: jitted and vmapped ONCE. Per-trajectory Python loops calling jax.grad
#: were the dominant cost of the previous revision's inner loops.
_per_traj_drive_adj = jax.jit(jax.vmap(
    jax.grad(loss, argnums=(3, 4)), in_axes=(None, 0, 0, None, None, None)))
_per_traj_param_grad = jax.jit(jax.vmap(
    jax.grad(loss, argnums=0), in_axes=(None, 0, 0, None, None, None)))
_batch_param_grad = jax.jit(jax.grad(batch_loss, argnums=0))
_batch_loss = jax.jit(batch_loss)
#: NOTE the in_axes: `forward` takes (params, xs, d1, d2, coef) - it has no
#: `ys` argument, because the targets enter only through the loss. An earlier
#: revision vmapped it with six entries and called it with `ys`, which the
#: production probe could not catch because it never reaches Part B.
_forward_batch = jax.jit(jax.vmap(forward, in_axes=(None, 0, None, None, None)))


def references(params, xs, ys, coef):
    """Everything the comparison is measured against, in four jitted calls."""
    d1, d2 = zero_drives()
    r1, r2 = _per_traj_drive_adj(params, xs, ys, d1, d2, coef)
    g_per = _per_traj_param_grad(params, xs, ys, d1, d2, coef)
    g_batch = _batch_param_grad(params, xs, ys, d1, d2, coef)
    s1, s2 = _forward_batch(params, xs, d1, d2, coef)
    return dict(rho1=r1, rho2=r2, g_per=g_per, g_batch=g_batch, s1=s1, s2=s2)


def assemble_grads(params, xs, s1, rho1, rho2):
    """G_W = sum_t rho_t r_t^T, exact for this cascade's discretization.

    Batched over trajectories: `xs`, `s1`, `rho*` carry a leading batch axis.
    """
    return dict(W1=onp.einsum("btu,btd->bud", rho1, xs),
                b1=rho1.sum(axis=1),
                W2=onp.einsum("btu,btd->bud", rho2, onp.tanh(s1)),
                b2=rho2.sum(axis=1))


def teaching_inputs(params, ys, s1, s2, rho2):
    """k per layer. k2 = dJ/ds2; k1 = diag(tanh'(s1)) W2^T rho2.

    Spatial weight transport is still required: temporal causality does not
    remove the transposed Jacobian coupling.
    """
    c2 = (onp.asarray(s2) - onp.asarray(ys)) / s2.shape[1]
    dphi = 1.0 - onp.tanh(onp.asarray(s1)) ** 2
    k1 = dphi * (onp.asarray(rho2) @ onp.asarray(params["W2"]))
    return k1, c2


# ------------------------------------------------------------- the metrics -
def _flat(g, blocks, i=None):
    return onp.concatenate([
        onp.asarray(g[b])[i].ravel() if i is not None
        else onp.asarray(g[b]).ravel() for b in blocks])


def compare(g_hat, g_ref, blocks=("W1", "b1", "W2", "b2"), i=None):
    fh, fr = _flat(g_hat, blocks, i), _flat(g_ref, blocks, i)
    nr, nh = onp.linalg.norm(fr), onp.linalg.norm(fh)
    out = dict(ref_norm=float(nr), approx_norm=float(nh),
               abs_error=float(onp.linalg.norm(fh - fr)))
    if nr < NEAR_ZERO or nh < NEAR_ZERO:
        out.update(cosine=None, relative_error=None, near_zero=True,
                   note="reference or approximation near zero; the cosine is "
                        "UNDEFINED, which is not a numerical failure")
    else:
        out.update(cosine=float(fh @ fr / (nr * nh)),
                   relative_error=float(out["abs_error"] / nr),
                   near_zero=False)
    out["finite"] = bool(onp.all(onp.isfinite(fh))
                         and onp.all(onp.isfinite(fr)))
    return out


def window_errors(hat, ref, name):
    """R5: teaching-variable error split into predeclared windows.

    The causal filters start from zero states while the exact adjoint is
    terminal-valued, so disagreement at the ends is structural. Reporting only
    a full-window number folds that boundary effect into the comparison.
    These are DIAGNOSTICS on the teaching variable; they are not the gradient
    of an interior-only loss, and truncating a gradient sum is not claimed to
    be one.
    """
    hat, ref = onp.asarray(hat), onp.asarray(ref)
    L = hat.shape[-2]
    wins = dict(start=slice(0, L // 4), interior=slice(L // 4, 3 * L // 4),
                end=slice(3 * L // 4, L))
    out = {}
    for wname, sl in wins.items():
        a, b = hat[..., sl, :].ravel(), ref[..., sl, :].ravel()
        na, nb = onp.linalg.norm(a), onp.linalg.norm(b)
        out[wname] = dict(
            ref_norm=float(nb),
            relative_error=(None if nb < NEAR_ZERO
                            else float(onp.linalg.norm(a - b) / nb)),
            cosine=(None if na < NEAR_ZERO or nb < NEAR_ZERO
                    else float(a @ b / (na * nb))))
    return {name: out}


def loss_change(params, xs, ys, coef, g, step_norm,
                blocks=("W1", "b1", "W2", "b2")):
    """R3: the change in THE batch objective after a norm-matched step."""
    flat = _flat(g, blocks)
    n = onp.linalg.norm(flat)
    if n < NEAR_ZERO:
        return None
    d1, d2 = zero_drives()
    base = float(_batch_loss(params, xs, ys, d1, d2, coef))
    new = dict(params)
    for b in blocks:
        new[b] = params[b] - step_norm * onp.asarray(g[b]) / n
    return float(_batch_loss(new, xs, ys, d1, d2, coef)) - base


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


# ------------------------------------------------- reference validation ----
def validate_reference(params, xs, ys, coef, refs):
    """R8: gate the reference ACTUALLY used, not a fixture on another seed.

    Three independent routes must agree before any approximation is compared
    against them: the drive-adjoint contraction, forward versus reverse mode,
    and central finite differences including the initial states.
    """
    d1, d2 = zero_drives()
    g_id = assemble_grads(params, onp.asarray(xs), refs["s1"], refs["rho1"],
                          refs["rho2"])
    ident = compare(jax.tree_util.tree_map(lambda v: v.mean(axis=0), g_id),
                    jax.tree_util.tree_map(lambda v: v.mean(axis=0),
                                           refs["g_per"]))
    fwd = jax.jacfwd(batch_loss)(params, xs, ys, d1, d2, coef)
    fwd_rev = {b: float(onp.max(onp.abs(onp.asarray(fwd[b])
                                        - onp.asarray(refs["g_batch"][b]))))
               for b in params}
    fd = []
    rs = onp.random.RandomState(1234)
    for b in ("W1", "W2", "z0_1", "z0_2"):
        v = rs.randn(*onp.asarray(params[b]).shape)
        v /= onp.linalg.norm(v)
        h = 1e-6
        pp = dict(params); pp[b] = params[b] + h * v
        pm = dict(params); pm[b] = params[b] - h * v
        num = (float(_batch_loss(pp, xs, ys, d1, d2, coef))
               - float(_batch_loss(pm, xs, ys, d1, d2, coef))) / (2 * h)
        ana = float(onp.sum(onp.asarray(refs["g_batch"][b]) * v))
        fd.append(dict(block=b, finite_difference=num, analytic=ana,
                       relative=abs(num - ana) / max(abs(ana), 1e-12)))
    ok_ident = (ident["relative_error"] is not None
                and ident["relative_error"] < MAX_IDENTITY_REL)
    ok_fwd = max(fwd_rev.values()) < MAX_FWD_REV_ABS
    ok_fd = max(r["relative"] for r in fd) < MAX_FD_REL
    finite = bool(onp.all(onp.isfinite(_flat(refs["g_batch"],
                                             ("W1", "b1", "W2", "b2")))))
    return dict(drive_adjoint_identity=ident,
                forward_vs_reverse_max_abs=fwd_rev,
                finite_difference_checks=fd, reference_finite=finite,
                tolerances=dict(identity_rel=MAX_IDENTITY_REL,
                                fwd_rev_abs=MAX_FWD_REV_ABS,
                                fd_rel=MAX_FD_REL),
                passed=bool(ok_ident and ok_fwd and ok_fd and finite))


def _tfirst(a):
    return onp.moveaxis(onp.asarray(a), 1, 0)          # (n,L,u) -> (L,n,u)


def _bfirst(a):
    return onp.moveaxis(onp.asarray(a), 0, 1)          # (L,n,u) -> (n,L,u)


def evaluate_band(params, xs, ys, coef, filters, seed, band, freqs):
    """One (seed, band): references, both approximations, every declared eps."""
    refs = references(params, xs, ys, coef)
    val = validate_reference(params, xs, ys, coef, refs)
    s1 = onp.asarray(refs["s1"]); s2 = onp.asarray(refs["s2"])
    k1_x, c2 = teaching_inputs(params, onp.asarray(ys), s1, s2, refs["rho2"])
    xs_np = onp.asarray(xs)
    W2 = onp.asarray(params["W2"])
    dphi = 1.0 - onp.tanh(s1) ** 2
    out = dict(seed=seed, band=list(band), reference_validation=val,
               input=signal_report(xs_np, freqs, band),
               drive_spectrum=dict(
                   layer2_k=spectral_centroid(_tfirst(c2)),
                   layer1_k=spectral_centroid(_tfirst(k1_x))),
               approximations={})
    if not val["passed"]:
        return out
    for (kind, frac), (Ad, Bd, w, eps, delta) in filters.items():
        rh2 = _bfirst(CE.run_filter_d(Ad, Bd, w, _tfirst(c2)))
        kk1 = dphi * (rh2 @ W2)
        rh1 = _bfirst(CE.run_filter_d(Ad, Bd, w, _tfirst(kk1)))
        g_hat = assemble_grads(params, xs_np, s1, rh1, rh2)
        per = [compare(g_hat, refs["g_per"], i=i) for i in range(xs_np.shape[0])]
        cos = [m["cosine"] for m in per if m["cosine"] is not None]
        rel = [m["relative_error"] for m in per
               if m["relative_error"] is not None]
        undefined = sum(1 for m in per if m["cosine"] is None)
        nonfinite = sum(1 for m in per if not m["finite"])
        g_mean = {b: onp.asarray(g_hat[b]).mean(axis=0)
                  for b in ("W1", "b1", "W2", "b2")}
        entry = dict(
            kind=kind, eps=eps, delta=delta, eps_fraction=frac,
            mean_cosine=(float(onp.mean(cos)) if cos else None),
            min_cosine=(float(onp.min(cos)) if cos else None),
            negative_cosine_fraction=(float(onp.mean([c < 0 for c in cos]))
                                      if cos else None),
            mean_relative_error=(float(onp.mean(rel)) if rel else None),
            n_undefined_cosine=undefined, n_nonfinite=nonfinite,
            batch=compare(g_mean, refs["g_batch"]),
            teaching_variable_windows={},
            loss_change={})
        entry["teaching_variable_windows"].update(
            window_errors(rh2, refs["rho2"], "layer2"))
        entry["teaching_variable_windows"].update(
            window_errors(rh1, refs["rho1"], "layer1"))
        for sn in STEP_NORMS:
            entry["loss_change"][str(sn)] = dict(
                approximation=loss_change(params, xs, ys, coef, g_mean, sn),
                reference=loss_change(params, xs, ys, coef,
                                      refs["g_batch"], sn))
        out["approximations"][f"{kind}_eps{frac}"] = entry
        print(f"[seed {seed}] band {band} {kind:<10} eps={eps:5.2f}"
              f"  cos={entry['mean_cosine']}"
              f"  rel={entry['mean_relative_error']}"
              f"  neg={entry['negative_cosine_fraction']}")
    return out


def build_filters(coef, t_minus):
    """R9: discretize every declared filter ONCE."""
    gamma, T, M = coef["gamma"], coef["T"], coef["M"]
    out = {}
    for frac in EPS_FRACTIONS:
        eps = frac * t_minus
        for kind in ("moment", "reciprocal"):
            Ad, Bd, w = CE.discretize(kind, gamma, T, M, eps, eps)
            out[(kind, frac)] = (Ad, Bd, w, eps, eps)
    return out


def preflight(params, coef, filters, seeds, status):
    """R9: MEASURE one (seed, band) end to end, then project the rest."""
    rng = onp.random.RandomState(0)
    xs, fx = band_signals(rng, N_TRAJ, SEQ_LEN, D_IN, BANDS[0])
    ys, _ = band_signals(rng, N_TRAJ, SEQ_LEN, N_UNITS, BANDS[0])
    xs, ys = jnp.asarray(xs), jnp.asarray(ys)
    t0 = time.time()
    evaluate_band(params, xs, ys, coef, filters, seeds[0], BANDS[0], fx)
    first_s = time.time() - t0
    t1 = time.time()
    evaluate_band(params, xs, ys, coef, filters, seeds[0], BANDS[0], fx)
    steady_s = time.time() - t1
    n = len(seeds) * len(BANDS)
    total = first_s + max(n - 1, 0) * steady_s
    status["preflight"] = dict(
        first_band_s=first_s, steady_band_s=steady_s,
        n_band_evaluations=n, projected_total_s=total,
        scope=("one full (seed, band) evaluation measured with compilation and "
               "again without it; the first is counted once and the steady "
               "cost for the rest. Filters are discretized once, references "
               "are jitted and vmapped over trajectories."))
    print(f"PREFLIGHT_B_PROJECTED_TOTAL_S={total:.1f} "
          f"(first {first_s:.1f}s, steady {steady_s:.1f}s x {n - 1})")
    return total


def verdict(rows):
    """R8: the predeclared scientific rule, applied mechanically."""
    keys = set()
    for r in rows:
        keys |= set(r.get("approximations", {}))
    out = {}
    for k in sorted(keys):
        entries = [r["approximations"][k] for r in rows
                   if k in r.get("approximations", {})]
        cos = [e["mean_cosine"] for e in entries]
        neg = [e["negative_cosine_fraction"] for e in entries]
        rel = [e["mean_relative_error"] for e in entries]
        dec = []
        for e in entries:
            for sn in STEP_NORMS:
                dec.append(e["loss_change"][str(sn)]["approximation"])
        usable = bool(
            entries
            and all(c is not None and c > 0 for c in cos)
            and all(nn is not None and nn <= NEGATIVE_COSINE_TOLERANCE
                    for nn in neg)
            and all(d is not None and d < 0 for d in dec))
        out[k] = dict(usable=usable,
                      mean_cosine_range=[min(c for c in cos if c is not None),
                                         max(c for c in cos if c is not None)]
                      if any(c is not None for c in cos) else None,
                      worst_negative_cosine_fraction=(
                          max(nn for nn in neg if nn is not None)
                          if any(nn is not None for nn in neg) else None),
                      mean_relative_error_range=(
                          [min(r for r in rel if r is not None),
                           max(r for r in rel if r is not None)]
                          if any(r is not None for r in rel) else None),
                      all_step_norms_decrease=bool(
                          dec and all(d is not None and d < 0 for d in dec)))
    # preferability, per eps, moment against the GLE-inspired baseline
    pref = {}
    for frac in EPS_FRACTIONS:
        mk, rk = f"moment_eps{frac}", f"reciprocal_eps{frac}"
        me = [r["approximations"][mk]["mean_relative_error"] for r in rows
              if mk in r.get("approximations", {})]
        re = [r["approximations"][rk]["mean_relative_error"] for r in rows
              if rk in r.get("approximations", {})]
        ok = bool(me and re and len(me) == len(re)
                  and all(a is not None and b is not None and a < b
                          for a, b in zip(me, re)))
        pref[str(frac)] = dict(
            moment_lower_relative_error_at_every_band_and_seed=ok,
            moment=me, gle_inspired=re)
    return dict(per_filter=out, preferability=pref,
                rule=("usable: mean cosine > 0, negative-cosine fraction "
                      "exactly 0, and the norm-matched update decreases the "
                      "BATCH objective at both step norms, at every band and "
                      "eps and for every seed. preferable: additionally lower "
                      "mean relative error than the GLE-inspired baseline at "
                      "every band, not at a chosen one."),
                note=("an unfavourable verdict is a RESULT and leaves the "
                      "execution status PASS"))


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/tss_pilot")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--seeds", default="100,101,102")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)
    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    left = lambda: deadline - time.time() - args.reserve_s        # noqa: E731
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")

    coef = assert_numeric_pytree(symmetric_reference(), "Part B coef")
    gamma, T, M = coef["gamma"], coef["T"], coef["M"]
    t_minus, t_plus = q_roots(gamma, T, M)
    seeds = [int(s) for s in args.seeds.split(",")]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id, "part_b")
    os.makedirs(out, exist_ok=True)

    # ---- the audit, BEFORE any comparison
    audit = dict(
        coefficients=coef, metadata=reference_metadata(),
        q_roots=dict(t_minus=t_minus, t_plus=t_plus),
        circuit_inequality_M_le_gamma_T=bool(0 < M <= gamma * T),
        recurrent_witness=CE.recurrent_witness(),
        spatial_arrangement=(
            "feedforward: layer 1's drive has no state dependence and layer "
            "2's depends only on layer 1, so the error coupling is triangular "
            "and the Section 7 instability cannot arise here. Recurrent error "
            "dynamics are NOT validated by this pilot."),
        baseline_label=(
            "the reciprocal arm is a GLE-INSPIRED approximation - our "
            "regularized inverse E_eps cascaded with R_delta - and NOT a "
            "verbatim TSS or GLE learning rule. The unregularized 1/H shares "
            "the adjoint's phase on the Fourier axis; the executed product "
            "generally does not, so no exact-phase claim is made."),
        band_error_caveat=(
            "measured band errors are SAMPLED maxima on a finite grid, not "
            "proven continuum suprema"),
        initialization_derivative=(
            "the reference differentiates z0_1 and z0_2; NEITHER causal "
            "approximation supplies an initial-state derivative, because its "
            "error states start at zero while the exact adjoint has a TERMINAL "
            "condition. The initial states are FIXED here, so this limitation "
            "is recorded and is not a launch blocker; the gradient comparison "
            "is restricted to W1, b1, W2, b2."),
        boundary_prescription=(
            "causal error states start at zero and the exact adjoint is "
            "terminal-valued, so they disagree near the sequence ends by "
            "construction. Start/interior/end windows are reported for BOTH "
            "layers' teaching variables as diagnostics; they are not the "
            "gradient of an interior-only loss."),
        filters={})
    for frac in EPS_FRACTIONS:
        eps = frac * t_minus
        entry = dict(eps=eps, delta=eps, eps_fraction_of_t_minus=frac,
                     peak_gain=CE.peak_gain_bound(gamma, T, M, eps))
        for kind in ("moment", "reciprocal"):
            entry[kind] = dict(
                poles=CE.filter_poles(kind, gamma, T, M, eps, eps),
                sampled_band_error={
                    str(b): CE.measured_band_error(kind, gamma, T, M, eps, eps,
                                                   b[1]) for b in BANDS})
        entry["moment"]["remainder"] = {
            str(b): CE.remainder_bound(gamma, T, M, eps, b[1]) for b in BANDS}
        audit["filters"][str(frac)] = entry
    print("[audit] circuit inequality M <= gamma*T:",
          audit["circuit_inequality_M_le_gamma_T"])
    print("[audit] recurrent witness: forward pole "
          f"{audit['recurrent_witness']['forward_pole']:.6f}, GLE-inspired "
          f"error pole {audit['recurrent_witness']['reciprocal_error_pole']:+.6f}")
    for frac, e in audit["filters"].items():
        print(f"[audit] eps={e['eps']:.3f} ({frac} t-)  peak gain bound "
              f"{e['peak_gain']['bound']:.3e}  moment stable="
              f"{e['moment']['poles']['stable']}  GLE-inspired stable="
              f"{e['reciprocal']['poles']['stable']}")

    status = dict(part="B", run_id=run_id, out=out, backend=backend,
                  seeds=seeds, n_units=N_UNITS, n_layers=N_LAYERS,
                  n_trajectories=N_TRAJ, seq_len=SEQ_LEN,
                  bands=[list(b) for b in BANDS], n_tones=N_TONES,
                  eps_fractions=list(EPS_FRACTIONS),
                  step_norms=list(STEP_NORMS), audit=audit,
                  provenance=provenance(), results=[], incomplete=[])
    write(os.path.join(out, "status.json"), status)

    filters = build_filters(coef, t_minus)
    p0 = {k: jnp.asarray(v) for k, v in init_params(seeds[0]).items()}
    proj = preflight(p0, coef, filters, seeds, status)
    write(os.path.join(out, "status.json"), status)
    if args.preflight_only:
        print(f"TSS_B_STATUS=PREFLIGHT_ONLY out={out}")
        return 0
    if proj > left():
        status["incomplete"].append(
            f"projected {proj:.0f}s > remaining {left():.0f}s; Part B NOT "
            f"started. No band, seed or eps was dropped.")
        write(os.path.join(out, "status.json"), status)
        print(f"[!] {status['incomplete'][-1]}")
        print(f"TSS_B_STATUS=INCOMPLETE out={out}")
        return 3

    rows = []
    for seed in seeds:
        params = {k: jnp.asarray(v) for k, v in init_params(seed).items()}
        rng = onp.random.RandomState(seed + 5)
        for band in BANDS:
            if left() < 5:
                status["incomplete"].append(f"seed {seed} band {band}: budget")
                break
            xs, fx = band_signals(rng, N_TRAJ, SEQ_LEN, D_IN, band)
            ys, _ = band_signals(rng, N_TRAJ, SEQ_LEN, N_UNITS, band)
            row = evaluate_band(params, jnp.asarray(xs), jnp.asarray(ys), coef,
                                filters, seed, band, fx)
            rows.append(row)
            status["results"] = rows
            write(os.path.join(out, "results.json"), rows)
            write(os.path.join(out, "status.json"), status)
            # R8: a failed reference invalidates everything measured against it
            if not row["reference_validation"]["passed"]:
                status["failed"] = (
                    f"seed {seed} band {band}: the REFERENCE failed its "
                    f"predeclared correctness tolerances; dependent "
                    f"interpretation is refused. "
                    f"{row['reference_validation']}")
                write(os.path.join(out, "status.json"), status)
                print(f"[FAIL] {status['failed']}")
                print(f"TSS_B_STATUS=FAILED out={out}")
                return 4

    status["verdict"] = verdict(rows)
    status["wall_s"] = time.time() - t0
    status["complete"] = (len(rows) == len(seeds) * len(BANDS)
                          and not status["incomplete"])
    write(os.path.join(out, "status.json"), status)
    print("\n  filter                usable   cosine range        rel range")
    for k, v in status["verdict"]["per_filter"].items():
        cr = v["mean_cosine_range"]; rr = v["mean_relative_error_range"]
        print(f"  {k:<20}{str(v['usable']):<9}"
              f"{('[%.4f, %.4f]' % tuple(cr)) if cr else 'n/a':<20}"
              f"{('[%.3e, %.3e]' % tuple(rr)) if rr else 'n/a'}")
    print(f"[*] wall {status['wall_s']:.0f}s  rows {len(rows)}/"
          f"{len(seeds) * len(BANDS)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[INCOMPLETE] {m}")
        print(f"TSS_B_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"TSS_B_STATUS=COMPLETE out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
