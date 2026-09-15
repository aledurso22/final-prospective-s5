"""The four Part-A temporal architectures, on one shared encoder and readout.

Everything outside the temporal layer is held fixed across arms: the same input
encoder, the same readout shape, the same two heads and the same loss. What
differs is the temporal operator, which is the point of the comparison.

STATE BUDGET (real temporal coordinates per arm, the declared primary budget of
16, and the unit counts that follow from it):

    arm 1  ideal_prospective        0   (stateless fixed point), 16 units
    arm 2  tss_finite_adaptation   16   (8 units x 2 states)
    arm 3  memory_then_prospective 16   (16 leaky units x 1) + 0 processing
    arm 4  retained_compartment    16   (8 units x 2 states)

The ideal control is NOT artificially enlarged to look state-matched: having no
driven temporal state after its residual transient is the property it exists to
exhibit. It is given 16 units so its parameter count is not the smallest of the
set, which makes it a stronger control rather than a weaker one.

PARAMETER COUNTS DIFFER AT THIS BUDGET and are reported per arm. State matching
and parameter matching are not claimed simultaneously; if a prospective
advantage appears, a parameter-matched comparison is a prerequisite before any
capacity-efficiency claim.

The readout sees ONLY the temporal state, never the raw input. Feeding the
input forward would let the readout reconstruct the current signal without the
temporal layer and erase the fast/slow trade-off the task exists to create.
"""

import jax
import jax.numpy as jnp

from .tss_cells import (DT, IDEAL_ITERS, drive, equivalent_adaptation,
                        ideal_fixed_point, leaky_vector_field,
                        project_contraction, retained_vector_field,
                        rollout_ode, symmetric_reference, tss_vector_field)

ARMS = ("ideal_prospective", "tss_finite_adaptation",
        "memory_then_prospective", "retained_compartment")

D_ENC = 16          # shared encoder width
D_HID = 16          # readout hidden width
STATE_BUDGET = 16   # declared real temporal coordinates

#: units per arm, fixed by the state budget and the arm's states-per-unit
UNITS = dict(ideal_prospective=16, tss_finite_adaptation=8,
             memory_then_prospective=16, retained_compartment=8)
#: recurrent gain at initialization, common to every arm
REC_GAIN = 0.9
#: RK4 substeps per input step; the step-refinement check is in the tests
N_SUB = 2


def temporal_state_count(arm):
    """Real temporal coordinates actually carried, derived from the law."""
    n = UNITS[arm]
    if arm == "ideal_prospective":
        return dict(driven=0, per_unit=0, units=n, processing=0, total=0)
    if arm in ("tss_finite_adaptation", "retained_compartment"):
        return dict(driven=2 * n, per_unit=2, units=n, processing=0,
                    total=2 * n)
    return dict(driven=n, per_unit=1, units=n, processing=0, total=n,
                note="the prospective processing stage is a stateless fixed "
                     "point and adds no coordinate")


def coefficients(T=None):
    """The declared coefficient sectors for arms 2 and 4, and the twin map.

    Arm 4 takes the symmetric circuit point (gamma = T, M = 3T^2/4). Arm 2 is
    given the SAME two poles, so the arms differ only in the prospective zero
    (f + T f' against f + 2T f') rather than in who was handed more memory.
    """
    ref = symmetric_reference() if T is None else symmetric_reference(T)
    tau_m, eps, tau_p_ours = equivalent_adaptation(ref["gamma"], ref["T"],
                                                  ref["M"])
    return dict(
        retained=ref,
        retained_twin=dict(tau_m=tau_m, eps=eps, tau_p=tau_p_ours),
        # TSS's own matching prescription: the prospective time constant equals
        # the membrane one. Poles shared with arm 4 by construction.
        tss=dict(tau_m=tau_m, eps=eps, tau_p=tau_m, matched=True),
        # arm 3's memory leak is set to arm 4's SLOW pole, so no arm starts
        # with a longer intrinsic memory than another
        memory=dict(tau_mem=tau_m))


# ------------------------------------------------------------ parameters ---
def _glorot(key, shape, gain=1.0):
    fan_in, fan_out = shape[-1], shape[0]
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return std * jax.random.normal(key, shape)


def _block(key, n_units, d_in):
    kw, ku = jax.random.split(key)
    return dict(W=REC_GAIN * jax.random.normal(kw, (n_units, n_units))
                / (n_units ** 0.5),
                U=_glorot(ku, (n_units, d_in)),
                b=jnp.zeros((n_units,)))


def init_params(arm, seed, n_classes=8):
    """Shared tensors from ONE key, arm-specific tensors from another.

    Every tensor whose shape does not depend on the arm is drawn from the same
    key in the same order, so the encoder and both output heads are identical
    across arms at the same seed. `shared_tensors` records which ones actually
    matched, rather than asserting it.
    """
    ks = jax.random.PRNGKey(seed)
    k_enc, k_ro, k_sig, k_cls = jax.random.split(ks, 4)
    ka = jax.random.PRNGKey(seed + 7919)
    n = UNITS[arm]
    n_out = n

    p = dict(
        enc_W=_glorot(k_enc, (D_ENC, 11)), enc_b=jnp.zeros((D_ENC,)),
        head_sig_W=_glorot(k_sig, (1, D_HID)), head_sig_b=jnp.zeros((1,)),
        head_cls_W=_glorot(k_cls, (n_classes, D_HID)),
        head_cls_b=jnp.zeros((n_classes,)))

    if arm == "memory_then_prospective":
        k1, k2, k3 = jax.random.split(ka, 3)
        p["mem"] = _block(k1, n, D_ENC)
        kp, kv, kx = jax.random.split(k2, 3)
        p["pros"] = dict(
            W=project_contraction(REC_GAIN * jax.random.normal(kp, (n, n))
                                  / (n ** 0.5)),
            Vm=_glorot(kv, (n, n)), Vx=_glorot(kx, (n, D_ENC)),
            b=jnp.zeros((n,)))
        del k3
    else:
        p["cell"] = _block(ka, n, D_ENC)
        if arm == "ideal_prospective":
            p["cell"]["W"] = project_contraction(p["cell"]["W"])

    p["ro_W"] = _glorot(k_ro, (D_HID, n_out))
    p["ro_b"] = jnp.zeros((D_HID,))
    return p


def project_params(arm, p):
    """Re-impose the contraction cap after every optimizer update.

    Applied ONLY where a fixed point is solved, because uniqueness of that
    solution is part of those arms' definition. It is not imposed on the arms
    that integrate an ODE, which need no such condition.
    """
    if arm == "ideal_prospective":
        p = dict(p, cell=dict(p["cell"], W=project_contraction(p["cell"]["W"])))
    elif arm == "memory_then_prospective":
        p = dict(p, pros=dict(p["pros"],
                              W=project_contraction(p["pros"]["W"])))
    return p


def shared_tensor_report(params_by_arm):
    """Which tensors are bit-identical across arms at this seed."""
    import numpy as onp
    names = set.intersection(*[set(p) for p in params_by_arm.values()])
    out = {}
    for nm in sorted(names):
        vals = [onp.asarray(p[nm]) for p in params_by_arm.values()
                if not isinstance(p[nm], dict)]
        if not vals or any(v.shape != vals[0].shape for v in vals):
            out[nm] = "shape differs across arms"
            continue
        out[nm] = bool(all(onp.array_equal(v, vals[0]) for v in vals))
    return out


def parameter_count(p):
    import numpy as onp
    def walk(d):
        t = 0
        for v in d.values():
            t += walk(v) if isinstance(v, dict) else int(onp.asarray(v).size)
        return t
    return walk(p)


# --------------------------------------------------------------- forward ---
def encode(p, xs):
    return jnp.tanh(xs @ p["enc_W"].T + p["enc_b"])


def readout(p, s_out):
    h = jnp.tanh(jnp.tanh(s_out) @ p["ro_W"].T + p["ro_b"])
    return (h @ p["head_sig_W"].T + p["head_sig_b"])[..., 0], \
        h @ p["head_cls_W"].T + p["head_cls_b"]


def temporal(arm, p, es, cfg, n_sub=N_SUB, iters=IDEAL_ITERS):
    """Run one arm's temporal layer over one encoded sequence `es` (L, D_ENC).

    Returns (s_out (L, n_out), diagnostics).
    """
    if arm == "ideal_prospective":
        def step(_, e):
            s, r = ideal_fixed_point(p["cell"], e, iters)
            return _, (s, r)
        _, (ss, res) = jax.lax.scan(step, 0.0, es)
        return ss, dict(fixed_point_residual=jnp.max(res))

    if arm == "tss_finite_adaptation":
        c = cfg["tss"]
        vf = tss_vector_field(p["cell"], c["tau_m"], c["eps"], c["tau_p"])
        z0 = jnp.zeros((2, UNITS[arm]))
        zs, _ = rollout_ode(vf, z0, es, n_sub)
        return zs[:, 0, :], dict(fixed_point_residual=jnp.asarray(0.0))

    if arm == "retained_compartment":
        c = cfg["retained"]
        vf = retained_vector_field(p["cell"], c["gamma"], c["T"], c["M"])
        z0 = jnp.zeros((2, UNITS[arm]))
        zs, _ = rollout_ode(vf, z0, es, n_sub)
        return zs[:, 0, :], dict(fixed_point_residual=jnp.asarray(0.0))

    if arm == "memory_then_prospective":
        vf = leaky_vector_field(p["mem"], cfg["memory"]["tau_mem"])
        s0 = jnp.zeros((UNITS[arm],))
        sm, _ = rollout_ode(vf, s0, es, n_sub)

        # stateless prospective processing on top: s_p = f_p(s_p, .)
        def pros(sm_t, e_t):
            def it(s, _):
                return (p["pros"]["W"] @ jnp.tanh(s)
                        + p["pros"]["Vm"] @ jnp.tanh(sm_t)
                        + p["pros"]["Vx"] @ e_t + p["pros"]["b"]), None
            s = jnp.zeros((UNITS[arm],))
            s, _ = jax.lax.scan(it, s, None, length=iters)
            r = jnp.max(jnp.abs(
                p["pros"]["W"] @ jnp.tanh(s)
                + p["pros"]["Vm"] @ jnp.tanh(sm_t)
                + p["pros"]["Vx"] @ e_t + p["pros"]["b"] - s))
            return s, r
        sp, res = jax.vmap(pros)(sm, es)
        return sp, dict(fixed_point_residual=jnp.max(res))

    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def forward(arm, p, xs, cfg, n_sub=N_SUB, iters=IDEAL_ITERS):
    """One sequence in, (signal prediction, class logits, diagnostics) out."""
    es = encode(p, xs)
    s_out, diag = temporal(arm, p, es, cfg, n_sub, iters)
    y_sig, y_cls = readout(p, s_out)
    return y_sig, y_cls, diag


def batched_forward(arm, p, xs, cfg, n_sub=N_SUB, iters=IDEAL_ITERS):
    f = lambda x: forward(arm, p, x, cfg, n_sub, iters)      # noqa: E731
    y_sig, y_cls, diag = jax.vmap(f)(xs)
    return y_sig, y_cls, jax.tree_util.tree_map(jnp.max, diag)
