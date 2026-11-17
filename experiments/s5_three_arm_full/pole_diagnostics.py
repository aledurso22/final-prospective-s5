"""Per-arm recurrence poles, from live parameters, for training telemetry.

Two entry points:

* `radius_array(params, arm)` -- jnp, jit-able: the maximum spectral radius of
  each arm's ACTUAL recurrence (native lambda_bar; Zucchet; generalized A;
  calibrated C/D through the predictor-corrector coefficients). Used every
  step inside the fused telemetry.
* `pole_summary(params, arm)` -- numpy float64, once per epoch: Schur-Cohn
  stability, margins, repeated-root proximity, |P|, eta|K|, slow/fast poles,
  decay times and rotations, the slow time constant relative to the native
  pole of the same backbone, residues, T/gamma and eta/(1-mu).

PRODUCTION PRECISION. `pole_summary` evaluates every mode twice:
  * in the precision the model itself uses (float32 parameters, complex64
    coefficients built by the SAME functions as the module, roots by the SAME
    `companion_roots` the factored scan uses), and
  * in float64, from the same float32 parameter values.
Each mode is then classified, per variant:
  * "stable":            both radii <= 1 - BOUNDARY_TOL;
  * "numerical_boundary": the float32 radius is within BOUNDARY_TOL of 1, or
                          float32 and float64 disagree about |z| < 1;
  * "unstable":          float64 radius > 1 + BOUNDARY_TOL (a real
                          instability of the equations, not rounding).
Nothing is clipped or repaired; the counts are recorded.

Nothing here changes any parameter or the training computation.
"""

import jax
import jax.numpy as jnp
import numpy as onp
from flax.traverse_util import flatten_dict

from s5.discrete_recurrence import (companion_radius, generalized_coefficients,
                                    zucchet_coefficients)
from s5.ssm import discretize_zoh

#: arm -> (target scale, corrector) for the predictor-corrector arms;
#: the generalized arm A is evaluated through its ORIGINAL production
#: coefficients (generalized_coefficients), not through this table
CALIBRATED_ARMS = {"calibrated_old_s5": ("calibrated", "old"),
                   "calibrated_prediction_s5": ("calibrated", "prediction")}
SECOND_ORDER_ARMS = ("generalized_prospective_s5",) + tuple(CALIBRATED_ARMS)
RHO_MIN = 1e-4
#: |radius - 1| below this is a numerical-boundary band: ~84 float32 ulps at 1
BOUNDARY_TOL = 1e-5


def _layers(params):
    flat = flatten_dict(params)
    return [(k[:-1], flat) for k in sorted(flat) if k[-1] == "Lambda_re"]


def _tmg(flat, prefix, xp):
    sp = jax.nn.softplus if xp is jnp else (lambda v: onp.logaddexp(0.0, v))
    sig = jax.nn.sigmoid if xp is jnp else (lambda v: 1.0 / (1.0 + onp.exp(-v)))
    T = sp(flat[prefix + ("generalized_T_raw",)])
    rho = RHO_MIN + (1.0 - RHO_MIN) * sig(flat[prefix + ("generalized_rho_raw",)])
    gamma = sp(flat[prefix + ("generalized_gamma_raw",)])
    return T, rho * gamma * T, gamma, rho


def layer_coefficients(flat, prefix, arm):
    """(lambda_bar, a1, a2) of one layer, built exactly as the module builds
    them (jnp; production precision when x64 is off). Native: a1 = lambda_bar,
    a2 = 0."""
    from s5 import predictor_corrector as PC
    lam = (jnp.clip(flat[prefix + ("Lambda_re",)], None, -1e-4)
           + 1j * flat[prefix + ("Lambda_im",)])
    step = jnp.exp(flat[prefix + ("log_step",)][..., 0])
    b = jnp.zeros((lam.shape[0], 1), dtype=lam.dtype)
    abar, bbar = discretize_zoh(lam, b, step)
    if arm == "native_matched_s5":
        return abar, abar, jnp.zeros_like(abar)
    if arm == "zucchet_prospective_s5":
        raw = flat[prefix + ("prospective_T_raw",)]
        coeff = zucchet_coefficients(abar, bbar, jax.nn.softplus(raw))
    elif arm in CALIBRATED_ARMS:
        T, M, gamma, _ = _tmg(flat, prefix, jnp)
        coeff = PC.pc_coefficients(abar, bbar, T, M, gamma, *CALIBRATED_ARMS[arm])
    else:   # generalized (and, as before, matched-lag): production map
        T, M, gamma, _ = _tmg(flat, prefix, jnp)
        coeff = generalized_coefficients(abar, bbar, T, M, gamma)
    return abar, coeff[0], coeff[1]


def radius_array(params, arm):
    """Max spectral radius over all layers and modes, as a jnp scalar."""
    radii = []
    for prefix, flat in _layers(params):
        abar, a1, a2 = layer_coefficients(flat, prefix, arm)
        if arm == "native_matched_s5":
            radii.append(jnp.max(jnp.abs(abar)))
        else:
            radii.append(jnp.max(companion_radius(a1, a2)))
    return jnp.max(jnp.stack(radii))


def production_radii(params, arm):
    """Per-mode radius in the model's own precision, via the factored scan's
    `companion_roots` (what the production scan actually multiplies by)."""
    from s5.factored_recurrence import companion_roots
    out = []
    for prefix, flat in _layers(params):
        abar, a1, a2 = layer_coefficients(flat, prefix, arm)
        if arm == "native_matched_s5":
            out.append(jnp.abs(abar))
        else:
            r1, r2 = companion_roots(a1, a2)
            out.append(jnp.maximum(jnp.abs(r1), jnp.abs(r2)))
    return onp.concatenate([onp.asarray(jax.device_get(r)) for r in out])


def classify(radius_production, radius_f64, tol=BOUNDARY_TOL):
    """'stable' / 'numerical_boundary' / 'unstable' per mode (see module doc)."""
    rp = onp.asarray(radius_production, onp.float64)
    r64 = onp.asarray(radius_f64, onp.float64)
    unstable = r64 > 1.0 + tol
    boundary = ~unstable & ((onp.abs(rp - 1.0) <= tol) | (onp.abs(r64 - 1.0) <= tol)
                            | ((rp < 1.0) != (r64 < 1.0)))
    stable = ~unstable & ~boundary
    return stable, boundary, unstable


def _q(v):
    v = onp.asarray(v, float).ravel()
    v = v[onp.isfinite(v)]
    if v.size == 0:
        return None
    return {k: float(x) for k, x in zip(
        ("min", "p5", "p50", "p95", "max"), onp.percentile(v, (0, 5, 50, 95, 100)))}


def pole_summary(params, arm):
    """Host-side float64 pole report for one parameter tree (call per epoch)."""
    from s5 import predictor_corrector as PC
    host = jax.device_get(params)
    acc = {}

    def add(key, value):
        acc.setdefault(key, []).append(onp.ravel(value))

    for prefix, flat in _layers(host):
        f = {k: onp.asarray(v, onp.float64) for k, v in flat.items()
             if k[:-1] == prefix}
        lam = (onp.minimum(f[prefix + ("Lambda_re",)], -1e-4)
               + 1j * f[prefix + ("Lambda_im",)])
        delta = onp.exp(f[prefix + ("log_step",)][..., 0])
        lb = onp.exp(lam * delta)
        r_native = onp.abs(lb)
        add("backbone_radius", r_native)
        if arm == "native_matched_s5":
            add("radius64", r_native)
        add("log_Delta", onp.log(delta))
        if arm not in SECOND_ORDER_ARMS:
            if arm == "native_matched_s5":
                add("radius", r_native)
                add("decay_tokens", -1.0 / onp.log(r_native))
                add("rotation", onp.abs(onp.angle(lb)))
            continue
        T, M, gamma, rho = _tmg(f, prefix, onp)
        if arm in CALIBRATED_ARMS:
            scale, corr = CALIBRATED_ARMS[arm]
        else:
            scale, corr = "legacy", "old"
        co = PC.scalar_coefficients(lb, T, M, gamma, scale, corr)
        if arm == "generalized_prospective_s5":   # the production map itself
            prod = generalized_coefficients(lb, onp.ones((lb.size, 1)), T, M, gamma)
            co = (prod[0], prod[1], prod[2][:, 0], prod[3][:, 0])
        z1, z2, R1, R2 = PC.residues(*co)
        r1, r2 = onp.abs(z1), onp.abs(z2)
        k = PC.target_scale(T, gamma, scale)
        mu, eta, _ = PC.residual_coefficients(T, M, gamma, k)
        with onp.errstate(divide="ignore", invalid="ignore"):
            add("stable", PC.schur_cohn(co[0], co[1]))
            add("radius", onp.maximum(r1, r2))
            add("radius64", onp.maximum(r1, r2))
            add("slow_decay_tokens", -1.0 / onp.log(r1))
            add("slow_rotation", onp.abs(onp.angle(z1)))
            add("slow_tau_over_native", onp.log(r_native) / onp.log(r1))
            add("fast_radius", r2)
            add("residue_fast_over_slow", onp.abs(R2) / onp.abs(R1))
            add("disc_ratio", onp.abs(co[0] ** 2 + 4 * co[1]) / onp.abs(co[0]) ** 2)
            add("abs_P", onp.abs(1 - eta * (1 - lb)))
            add("eta_absK", eta * onp.abs(1 - lb))
            add("eta_over_1_minus_mu", eta / (1 - mu))
            add("dc_delay", PC.group_delay_dc(*co))
            add("T_over_gamma", T / gamma)
            add("T", T); add("gamma", gamma); add("rho", rho)
    cat = {k: onp.concatenate(v) for k, v in acc.items()}
    out = {"arm": arm, "n_modes": int(cat["backbone_radius"].size)}
    if "radius64" in cat:
        rp = production_radii(params, arm)
        ok, edge, bad = classify(rp, cat["radius64"])
        out["production_precision"] = {
            "dtype": str(onp.asarray(jax.device_get(
                jax.tree_util.tree_leaves(params)[0])).dtype),
            "boundary_tol": BOUNDARY_TOL,
            "stable": int(ok.sum()), "numerical_boundary": int(edge.sum()),
            "unstable": int(bad.sum()),
            "max_radius_production": float(rp.max()),
            "max_radius_float64": float(cat["radius64"].max()),
            "min_margin_production": float(1 - rp.max()),
            "margin_production": _q(1 - rp),
            "max_abs_radius_difference_production_vs_float64": float(
                onp.max(onp.abs(rp - cat["radius64"])))}
    if "stable" in cat:
        sc = cat["stable"].astype(bool)
        out["stable"] = int(sc.sum())
        out["unstable"] = int((~sc).sum())
        out["min_margin"] = float(1 - cat["radius"].max())
        out["min_discriminant_ratio"] = float(cat["disc_ratio"].min())
        out["max_abs_P"] = float(cat["abs_P"].max())
        # response statistics are meaningful only for stable modes
        for k in ("dc_delay", "residue_fast_over_slow"):
            cat[k + "_stable_modes_only"] = cat.pop(k)[sc]
    for k, v in cat.items():
        if k not in ("stable", "radius64"):
            out[k] = _q(v)
    return out
