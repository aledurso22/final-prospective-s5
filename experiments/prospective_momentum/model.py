"""Rollouts, checkpoint conversion and counts for the continuation study.

The two new rules reuse the pinned Momentum DeltaNet shell line for line:
the common input contract (`sanitize_episode`), key normalization, value
masking, gate features, the official gate block, readout `W k` and output
timing are those of `experiments.nested_memory.model.rollout`, which the
native arm executes through the completed studies' delegation chain. Only the
per-token update differs. Existing rules are delegated unchanged to
`experiments.meta_delta.model.rollout`.
"""

import jax
import jax.numpy as jnp
import numpy as onp

from experiments.adaptive_memory import model as AM
from experiments.meta_delta import model as MM
from experiments.nested_memory import dynamics as NMD
from experiments.nested_memory import model as NM
from experiments.nested_memory.task import WRITE

from . import dynamics as PD

D_K, D_V = NM.D_K, NM.D_V
COMMON = MM.COMMON
MOMENTUM_LEAVES = ("key_raw", "value_table", "readout_W", "readout_b",
                   "a_proj", "b_proj", "m_proj", "e_proj", "A_log", "Mu_log",
                   "dt_bias", "mu_bias", "log_factor")


def rollout(rule, p, ep, dtype=None, carry0=None):
    if rule not in ("prospective_momentum", "gain_momentum"):
        return MM.rollout(rule, p, ep, dtype=dtype, carry0=carry0)
    if dtype is None:
        dtype = p["key_raw"].dtype
    ep = AM.sanitize_episode(ep)                      # the common contract
    key_id, val_id, event = ep["key_id"], ep["val_id"], ep["event"]
    # --- the same source operations as nested_memory.model.rollout's shell
    # (same helpers, same order). Equal SOURCE is not a proof of bitwise-equal
    # arithmetic in another compilation context; nesting is checked numerically.
    k_all, k_valid = NMD.safe_normalize(p["key_raw"])   # the SAME helper
    keys = k_all[key_id].astype(dtype)
    valid = k_valid[key_id]
    has_v = (val_id >= 0).astype(dtype)
    vals = (p["value_table"][jnp.maximum(val_id, 0)]
            * has_v[:, None]).astype(dtype)
    mask = ((event == WRITE).astype(dtype)) * valid.astype(dtype)
    gx = NM.gate_features(key_id, val_id, event).astype(dtype)
    alpha, beta, mu, eta = NM._momentum_gates(p, gx)
    if rule == "prospective_momentum":
        scalar, step_fn = p["kappa"][0], PD.prospective_step
        coeff = dict(kappa=p["kappa"][0])
    else:
        scalar, step_fn = PD.executed_gain(p), PD.gain_step
        coeff = dict(raw_log_g=p["log_g"][0], g=scalar)

    def step(carry, t):
        carry = step_fn(carry, keys[t], vals[t], mask[t], alpha[t], beta[t],
                        mu[t], eta[t], scalar)
        W = carry[0]
        logits = p["readout_W"] @ (W @ keys[t]) + p["readout_b"]
        return carry, (logits, jnp.sqrt(jnp.sum(W ** 2)),
                       jnp.sqrt(jnp.sum(carry[1] ** 2)))

    z = jnp.zeros((D_V, D_K), dtype=dtype)
    carry = (z, z) if carry0 is None else carry0
    for c in carry:
        if c.dtype != dtype:
            raise TypeError(f"carry dtype {c.dtype} != executed dtype {dtype}")
    carry, (logits, w_norm, aux_norm) = jax.lax.scan(
        step, carry, jnp.arange(key_id.shape[0]))
    return dict(logits=logits, w_norm=w_norm, aux_norm=aux_norm,
                gates=(alpha, beta, mu, eta), final_carry=carry, dtype=dtype,
                weights=None, coeff=coeff)


def convert_momentum(p_native, rule):
    """Documented map from a saved native Momentum DeltaNet tree.

    Copies EVERY native leaf unchanged and adds one scalar in the leaves'
    dtype: kappa = 0 (prospective) or log_g = 0 (gain). Refuses a tree that is
    not exactly the native leaf set."""
    missing = [k for k in MOMENTUM_LEAVES if k not in p_native]
    extra = [k for k in p_native if k not in MOMENTUM_LEAVES]
    if missing or extra:
        raise ValueError(f"not a native Momentum DeltaNet tree: missing "
                         f"{missing}, unexpected {extra}")
    dt = p_native["A_log"].dtype
    if rule == "momentum_delta":
        return dict(p_native)
    return dict(p_native, **{PD.EXTRA_LEAF[rule]: jnp.zeros((1,), dtype=dt)})


def parameter_counts(rule, p):
    flat = {k: int(onp.asarray(v).size) for k, v in p.items()}
    common = sum(v for k, v in flat.items() if k in COMMON)
    return dict(total=sum(flat.values()), common=common,
                extra=sum(flat.values()) - common, per_leaf=flat,
                carry_real_numbers=PD.CARRY[rule])
