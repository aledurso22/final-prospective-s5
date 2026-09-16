"""The UNCHANGED shell with the two-sided candidate and its comparators.

Task, objective R = w (W k - v) k^T, 38-parameter positive source gate,
embeddings, readout W q, exact rank-one steps and output timing are those of
the completed adaptive-memory study, imported rather than re-implemented.
Existing rules (adaptive_delta, gated_delta, momentum_delta) are delegated to
`experiments.adaptive_memory.model.rollout` unchanged. The attribution control
`heavy_ball_same_mass` IS the existing `adaptive_inertial` law and tree; only
its declared initialization differs (protocol s3).

No oracle feature, alternative readout, additional memory matrix or new
derivative approximation is introduced. Memory carry: candidate 128 = (W, Z),
exactly the existing two-state candidate.
"""

import math

import jax
import jax.numpy as jnp
import numpy as onp

from experiments.adaptive_memory import dynamics as AD
from experiments.adaptive_memory import gate as AG
from experiments.adaptive_memory import model as AM
from experiments.nested_memory.task import N_KEYS, N_VALUES, WRITE

from . import dynamics as MD

D_K = D_V = AM.D_K
COMMON = ("key_raw", "value_table", "readout_W", "readout_b")
GATE = ("gate_u", "gate_b")
#: executed rule each arm delegates to, when it is an existing law
DELEGATE = {"adaptive_delta": "adaptive_delta",
            "heavy_ball_same_mass": "adaptive_inertial",
            "gated_delta": "gated_delta", "momentum_delta": "momentum_delta"}


def init_params(rule, seed, init_coeffs=None, dtype=onp.float32):
    """Common tensors from the SAME stream and order as the completed study,
    so they are bit-identical across arms at a seed; response and gate leaves
    are appended from nothing."""
    ic = init_coeffs or {}
    if rule in ("gated_delta", "momentum_delta"):
        return AM.init_params(rule, seed, dtype=dtype)
    if rule == "adaptive_delta":
        return AM.init_params("adaptive_delta", seed,
                              init_coeffs=dict(raw_eta=ic["raw_eta"]),
                              dtype=dtype)
    if rule == "heavy_ball_same_mass":
        return AM.init_params("adaptive_inertial", seed,
                              init_coeffs=dict(raw_eta=ic["raw_eta"],
                                               raw_tau=ic["raw_tau"]),
                              dtype=dtype)
    base = AM.init_params("adaptive_delta", seed,
                          init_coeffs=dict(raw_eta=ic["raw_eta"]), dtype=dtype)
    if rule == "gp_two_sided":
        return dict(base, raw_tau=jnp.asarray([ic["raw_tau"]], dtype=dtype),
                    raw_r=jnp.asarray([ic.get("raw_r", 0.0)], dtype=dtype))
    if rule == "tss_eq17":
        return dict(base, raw_T=jnp.asarray([ic["raw_T"]], dtype=dtype))
    raise ValueError(f"unknown rule {rule!r}")


def delta_to_two_sided(p_delta, tau0=MD.TAU0):
    """Documented map from a saved first-order delta tree to the candidate.

    Copies EVERY common, gate and eta leaf unchanged; adds raw_tau = log(tau0)
    and raw_r = 0 in the leaves' dtype. The auxiliary carry is not a
    parameter: it is zero at every episode boundary by the rollout. The
    resulting candidate equals the delta model's function exactly (checked).
    """
    dt = p_delta["raw_eta"].dtype
    missing = [k for k in COMMON + GATE + ("raw_eta",) if k not in p_delta]
    extra = [k for k in p_delta if k not in COMMON + GATE + ("raw_eta",)]
    if missing or extra:
        raise ValueError(f"not a first-order delta tree: missing {missing}, "
                         f"unexpected {extra}")
    return dict(p_delta, raw_tau=jnp.asarray([math.log(tau0)], dtype=dt),
                raw_r=jnp.zeros((1,), dtype=dt))


def _shell(p, ep, dtype):
    ep = AM.sanitize_episode(ep)
    key_id, val_id, event = ep["key_id"], ep["val_id"], ep["event"]
    k_all, k_valid = AD.safe_normalize(p["key_raw"])
    keys = k_all[key_id].astype(dtype)
    valid = k_valid[key_id].astype(dtype)
    vsel = jnp.maximum(val_id, 0)
    has_v = (val_id >= 0).astype(dtype)
    is_write = (event == WRITE).astype(dtype) * valid
    vals = (p["value_table"][vsel] * has_v[:, None]).astype(dtype) \
        * is_write[:, None]
    a = AG.source_weight_table(p, jnp.asarray(AM.H32, dtype=dtype),
                               jnp.asarray(AM.H8, dtype=dtype))
    return key_id, vsel, keys, vals, is_write, a


def rollout(rule, p, ep, dtype=None, carry0=None):
    """One episode; the readout is W q only. Carries reset per episode."""
    if dtype is None:
        dtype = p["key_raw"].dtype
    if rule in DELEGATE:
        out = AM.rollout(DELEGATE[rule], p, ep, dtype=dtype, carry0=carry0)
        return dict(out, coeff=out.get("coeff", {}))
    key_id, vsel, keys, vals, is_write, a = _shell(p, ep, dtype)
    L = key_id.shape[0]
    zeros = (jnp.zeros((D_V, D_K), dtype=dtype),
             jnp.zeros((D_V, D_K), dtype=dtype))
    c0 = zeros if carry0 is None else carry0
    for c in c0:
        if c.dtype != dtype:
            raise TypeError(f"carry dtype {c.dtype} != executed {dtype}")

    if rule == "gp_two_sided":
        c = MD.two_sided_response(p)
        eta, tau, rho = c["eta"][0], c["tau"][0], c["rho"][0]
        zero = jnp.zeros((), dtype=dtype)
        F = AD.expm2(MD.two_sided_generator(eta, tau, rho, a))
        F_idle = AD.expm2(MD.two_sided_generator(eta, tau, rho, zero))
        Fw = F[key_id, vsel]
        F_t = jnp.where(is_write[:, None, None] > 0, Fw,
                        jnp.broadcast_to(F_idle, Fw.shape))
        a0 = jnp.exp(-AD.H / tau)

        def step(carry, t):
            W, Z = AD.two_state_step(carry[0], carry[1], keys[t], vals[t],
                                     F_t[t], a0)
            return (W, Z), (p["readout_W"] @ (W @ keys[t]) + p["readout_b"],
                            jnp.sqrt(jnp.sum(W ** 2)),
                            jnp.sqrt(jnp.sum(Z ** 2)))
    elif rule == "tss_eq17":
        c = MD.tss_eq17_response(p)
        eta, T = c["eta"][0], c["T"][0]
        w_t = jnp.where(is_write > 0, a[key_id, vsel], jnp.zeros_like(is_write))

        def step(carry, t):
            W, f = MD.eq17_step(carry[0], carry[1], keys[t], vals[t], w_t[t],
                                eta, T)
            return (W, f), (p["readout_W"] @ (W @ keys[t]) + p["readout_b"],
                            jnp.sqrt(jnp.sum(W ** 2)),
                            jnp.sqrt(jnp.sum(f ** 2)))
    else:
        raise ValueError(f"unknown rule {rule!r}")
    carry, (logits, w_norm, aux_norm) = jax.lax.scan(step, c0, jnp.arange(L))
    return dict(logits=logits, w_norm=w_norm, aux_norm=aux_norm,
                final_carry=carry, dtype=dtype, weights=a, coeff=c)


def parameter_counts(rule, p):
    flat = {k: int(onp.asarray(v).size) for k, v in p.items()}
    common = sum(v for k, v in flat.items() if k in COMMON)
    return dict(total=sum(flat.values()), common=common,
                extra=sum(flat.values()) - common, per_leaf=flat,
                carry_real_numbers=MD.CARRY[rule])
