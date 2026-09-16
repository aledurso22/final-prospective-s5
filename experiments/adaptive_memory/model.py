"""The UNCHANGED small shell, with five memory rules inside it.

The task generator, the feature/readout shell, the readout timing, the query
information restrictions and the state budgets are exactly those of the
completed nested-memory study: `d_k = d_v = 8`, 32 keys, 8 values, a 32x8 raw
key table normalized in use, an 8x8 value table at the one-hot basis, an 8x8
readout at the identity with zero bias, and the query read AFTER advancing its
interval. The common tensors are drawn from the same stream in the same order,
so they are bit-identical across arms at a seed.

The two literature arms are imported from the completed study rather than
re-implemented, so their source-pinned update, gate maps and official
initializers are unchanged by construction.

Expected trainable totals on the unchanged 392-parameter shell: 433, 432, 431,
432 (TSS), 392 (ideal equilibrium), 480, 569. These are COUNTED, not assumed,
and no inert parameter is added to equalize them. The ideal-equilibrium
reference deliberately has NO gate leaves: a strictly positive scalar source
weight changes neither its constraint nor its update, so gate parameters there
would have identically zero gradients.
"""

import jax
import jax.numpy as jnp
import numpy as onp

from experiments.nested_memory import dynamics as ND
from experiments.nested_memory import model as NM
from experiments.nested_memory.task import IDLE, N_KEYS, N_VALUES, QUERY, WRITE

from . import dynamics as AD
from . import gate as AG

D_K = D_V = 8
H32 = AG.contrast_matrix(N_KEYS)
H8 = AG.contrast_matrix(N_VALUES)
#: arms that carry the 38-parameter source gate
GATED_RULES = ("adaptive_prospective", "adaptive_inertial", "adaptive_delta",
               "tss_prospective")
NEW_RULES = GATED_RULES + ("ideal_projection",)


def init_params(rule, seed, init_coeffs=None, dtype=onp.float32):
    """Common tensors from the SAME stream as the completed study.

    `init_coeffs` supplies the calibrated raw response scalars; the gate is
    always initialized at `u = b = 0`, i.e. a source weight of exactly one on
    every write. Response and gate tensors are appended after the common draw
    and drawn from nothing, so they cannot perturb the common tables or the
    data order.
    """
    if rule in ("gated_delta", "momentum_delta"):
        return NM.init_params(rule, seed, dtype=dtype)
    rng = onp.random.RandomState(seed)                 # identical common draw
    p = dict(key_raw=rng.randn(N_KEYS, D_K).astype(dtype),
             value_table=onp.eye(N_VALUES, D_V, dtype=dtype),
             readout_W=onp.eye(N_VALUES, D_V, dtype=dtype),
             readout_b=onp.zeros(N_VALUES, dtype=dtype))
    ic = init_coeffs or {}
    if rule == "adaptive_prospective":
        p |= dict(raw_nu=onp.asarray([ic["raw_nu"]], dtype=dtype),
                  raw_tau=onp.asarray([ic["raw_tau"]], dtype=dtype),
                  raw_rho=onp.asarray([ic["raw_rho"]], dtype=dtype))
    elif rule == "adaptive_inertial":
        p |= dict(raw_eta=onp.asarray([ic["raw_eta"]], dtype=dtype),
                  raw_tau=onp.asarray([ic["raw_tau"]], dtype=dtype))
    elif rule == "adaptive_delta":
        p |= dict(raw_eta=onp.asarray([ic["raw_eta"]], dtype=dtype))
    elif rule == "tss_prospective":
        p |= dict(raw_tau_m=onp.asarray([ic["raw_tau_m"]], dtype=dtype),
                  raw_ratio=onp.asarray([ic["raw_ratio"]], dtype=dtype))
    elif rule == "ideal_projection":
        pass                       # 392 common parameters and nothing else
    else:
        raise ValueError(f"unknown rule {rule!r}")
    if rule in GATED_RULES:
        p |= dict(gate_u=onp.zeros(N_KEYS - 1, dtype=dtype),
                  gate_b=onp.zeros(N_VALUES - 1, dtype=dtype))
    return {k: jnp.asarray(v, dtype=dtype) for k, v in p.items()}


def response_tables(rule, p, dtype):
    """Per-(key, value) responses, computed ONCE outside the scan and vmap.

    The gate depends only on categorical identities, so every possible write
    response is a 32x8 table plus one idle response. That is a response for
    each POSSIBLE token, indexed causally by what is actually observed; it is
    not an observation of a future token.
    """
    if rule == "ideal_projection":
        return dict(weights=None, coeff={})
    h32 = jnp.asarray(H32, dtype=dtype)
    h8 = jnp.asarray(H8, dtype=dtype)
    a = AG.source_weight_table(p, h32, h8)                      # (32, 8)
    zero = jnp.zeros((), dtype=dtype)
    if rule == "adaptive_prospective":
        c = AD.response(p)
        nu, tau, rho = c["nu"][0], c["tau"][0], c["rho"][0]
        F = AD.expm2(AD.prospective_generator(nu, tau, rho, a))
        F_idle = AD.expm2(AD.prospective_generator(nu, tau, rho, zero))
        return dict(F=F, F_idle=F_idle, a0=jnp.exp(-AD.H / tau),
                    weights=a, coeff=c)
    if rule == "adaptive_inertial":
        c = AD.inertial_response(p)
        eta, tau = c["eta"][0], c["tau"][0]
        F = AD.expm2(AD.inertial_generator(eta, tau, a))
        F_idle = AD.expm2(AD.inertial_generator(eta, tau, zero))
        return dict(F=F, F_idle=F_idle, a0=jnp.exp(-AD.H / tau),
                    weights=a, coeff=c)
    if rule == "adaptive_delta":
        eta = jnp.exp(p["raw_eta"])[0]
        return dict(beta=-jnp.expm1(-AD.H * eta * a), weights=a,
                    coeff=dict(eta=eta))
    if rule == "tss_prospective":
        c = AD.tss_response(p)
        M, T = c["M"][0], c["T"][0]
        F = AD.expm2(AD.tss_generator(M, T, a))
        F_idle = AD.expm2(AD.tss_generator(M, T, zero))
        return dict(F=F, F_idle=F_idle, b0=AD.H / M, weights=a, coeff=c)
    return dict(weights=None, coeff={})


def rollout(rule, p, ep, dtype=None, carry0=None):
    """One episode. Returns query logits and diagnostics.

    The readout is `W q` and only `W q`: the auxiliary is never read, alone or
    mixed. Carries persist across the sequence and reset only between
    episodes.
    """
    if dtype is None:
        dtype = p["key_raw"].dtype
    key_id, val_id, event = ep["key_id"], ep["val_id"], ep["event"]
    if rule in ("gated_delta", "momentum_delta"):
        out = NM.rollout(rule, p, ep, NM.constants_for(rule, dtype=dtype),
                         dtype=dtype, carry0=carry0)
        return dict(out, weights=None)

    k_all, k_valid = AD.safe_normalize(p["key_raw"])
    keys = k_all[key_id].astype(dtype)
    valid = k_valid[key_id].astype(dtype)
    vsel = jnp.maximum(val_id, 0)
    has_v = (val_id >= 0).astype(dtype)
    is_write = (event == WRITE).astype(dtype) * valid
    vals = (p["value_table"][vsel] * has_v[:, None]).astype(dtype) \
        * is_write[:, None]
    tab = response_tables(rule, p, dtype)

    if rule == "ideal_projection":
        # Full-strength minimum-change projection on writes; W held otherwise.
        def step(W, t):
            W = AD.projection_step(W, keys[t], vals[t], is_write[t])
            return W, (p["readout_W"] @ (W @ keys[t]) + p["readout_b"],
                       jnp.sqrt(jnp.sum(W ** 2)),
                       jnp.zeros((), dtype=dtype))

        W0 = (jnp.zeros((D_V, D_K), dtype=dtype) if carry0 is None
              else carry0[0])
        W, (logits, w_norm, aux_norm) = jax.lax.scan(
            step, W0, jnp.arange(key_id.shape[0]))
        return dict(logits=logits, w_norm=w_norm, aux_norm=aux_norm,
                    final_carry=(W,), dtype=dtype, weights=None, coeff={})

    if rule == "tss_prospective":
        Fw = tab["F"][key_id, vsel]
        F_t = jnp.where(is_write[:, None, None] > 0, Fw,
                        jnp.broadcast_to(tab["F_idle"], Fw.shape))
        b0 = tab["b0"]

        def step(carry, t):
            W, P = AD.tss_two_state_step(carry[0], carry[1], keys[t], vals[t],
                                         F_t[t], b0)
            return (W, P), (p["readout_W"] @ (W @ keys[t]) + p["readout_b"],
                            jnp.sqrt(jnp.sum(W ** 2)), jnp.sqrt(jnp.sum(P ** 2)))

        c0 = ((jnp.zeros((D_V, D_K), dtype=dtype),
               jnp.zeros((D_V, D_K), dtype=dtype)) if carry0 is None
              else carry0)
        for c in c0:
            if c.dtype != dtype:
                raise TypeError(f"carry dtype {c.dtype} != executed {dtype}")
        carry, (logits, w_norm, aux_norm) = jax.lax.scan(
            step, c0, jnp.arange(key_id.shape[0]))
        return dict(logits=logits, w_norm=w_norm, aux_norm=aux_norm,
                    final_carry=carry, dtype=dtype, weights=tab["weights"],
                    coeff=tab["coeff"])

    if rule == "adaptive_delta":
        beta_t = jnp.where(is_write > 0, tab["beta"][key_id, vsel],
                           jnp.zeros_like(is_write))

        def step(W, t):
            W = AD.delta_step(W, keys[t], vals[t], beta_t[t])
            return W, (p["readout_W"] @ (W @ keys[t]) + p["readout_b"],
                       jnp.sqrt(jnp.sum(W ** 2)),
                       jnp.zeros((), dtype=dtype))

        W0 = (jnp.zeros((D_V, D_K), dtype=dtype) if carry0 is None
              else carry0[0])
        W, (logits, w_norm, aux_norm) = jax.lax.scan(
            step, W0, jnp.arange(key_id.shape[0]))
        return dict(logits=logits, w_norm=w_norm, aux_norm=aux_norm,
                    final_carry=(W,), dtype=dtype, weights=tab["weights"],
                    coeff=tab["coeff"])

    Fw = tab["F"][key_id, vsel]                              # (L, 2, 2)
    F_t = jnp.where(is_write[:, None, None] > 0, Fw,
                    jnp.broadcast_to(tab["F_idle"], Fw.shape))
    a0 = tab["a0"]

    def step(carry, t):
        W, Z = AD.two_state_step(carry[0], carry[1], keys[t], vals[t],
                                 F_t[t], a0)
        return (W, Z), (p["readout_W"] @ (W @ keys[t]) + p["readout_b"],
                        jnp.sqrt(jnp.sum(W ** 2)), jnp.sqrt(jnp.sum(Z ** 2)))

    c0 = ((jnp.zeros((D_V, D_K), dtype=dtype),
           jnp.zeros((D_V, D_K), dtype=dtype)) if carry0 is None else carry0)
    for c in c0:
        if c.dtype != dtype:
            raise TypeError(f"carry dtype {c.dtype} != executed dtype {dtype}")
    carry, (logits, w_norm, aux_norm) = jax.lax.scan(
        step, c0, jnp.arange(key_id.shape[0]))
    return dict(logits=logits, w_norm=w_norm, aux_norm=aux_norm,
                final_carry=carry, dtype=dtype, weights=tab["weights"],
                coeff=tab["coeff"])


def parameter_counts(rule, p):
    flat = {k: int(onp.asarray(v).size) for k, v in p.items()}
    common = sum(v for k, v in flat.items()
                 if k in ("key_raw", "value_table", "readout_W", "readout_b"))
    return dict(total=sum(flat.values()), common=common,
                extra=sum(flat.values()) - common, per_leaf=flat,
                carry_real_numbers=AD.CARRY[rule])
