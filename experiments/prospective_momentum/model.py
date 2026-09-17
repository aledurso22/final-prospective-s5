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
from . import filtered as FL
from . import ordinary as OD

D_K, D_V = NM.D_K, NM.D_V
COMMON = MM.COMMON
MOMENTUM_LEAVES = ("key_raw", "value_table", "readout_W", "readout_b",
                   "a_proj", "b_proj", "m_proj", "e_proj", "A_log", "Mu_log",
                   "dt_bias", "mu_bias", "log_factor")


#: rules executed by THIS module's shell; every other rule is delegated
SHELL_RULES = ("prospective_momentum", "gain_momentum", OD.ORDINARY,
               FL.FILTERED)


def rollout(rule, p, ep, dtype=None, carry0=None):
    if rule not in SHELL_RULES:
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
    elif rule == FL.FILTERED:
        # the master law applied to a residual-processing state, executed in
        # its coefficient form; the Momentum update below it is unchanged. The
        # coefficients are formed ONCE here and RETURNED, so acceptance can
        # classify exactly the rounded values this program multiplied by.
        scalar, step_fn = FL.coefficients(p), FL.filtered_step
        coeff = dict(zip(FL.COEFF_NAMES, scalar))
    elif rule == OD.ORDINARY:
        # same shell, same gates; only the per-token update differs. Its carry
        # adds the previous residual (192 real numbers, counted as such).
        scalar, step_fn = p["kappa"][0], OD.ordinary_step
        coeff = dict(kappa=p["kappa"][0])
    else:
        scalar, step_fn = PD.executed_gain(p), PD.gain_step
        coeff = dict(raw_log_g=p["log_g"][0], g=scalar)

    def step(carry, t):
        carry = step_fn(carry, keys[t], vals[t], mask[t], alpha[t], beta[t],
                        mu[t], eta[t], scalar)
        W = carry[0]
        logits = p["readout_W"] @ (W @ keys[t]) + p["readout_b"]
        outs = (logits, jnp.sqrt(jnp.sum(W ** 2)),
                jnp.sqrt(jnp.sum(carry[1] ** 2)))
        if rule == FL.FILTERED:
            # processing-state diagnostic: max |entry| of y, y_prev, R_prev
            # (max-abs cannot overflow while the entries are finite)
            outs = outs + (jnp.maximum(jnp.maximum(
                jnp.max(jnp.abs(carry[2])), jnp.max(jnp.abs(carry[3]))),
                jnp.max(jnp.abs(carry[4]))),)
        return carry, outs

    z = jnp.zeros((D_V, D_K), dtype=dtype)
    n_carry = {OD.ORDINARY: 3, FL.FILTERED: 5}.get(rule, 2)
    carry = (z,) * n_carry if carry0 is None else carry0
    if len(carry) != n_carry:
        raise ValueError(f"{rule} needs {n_carry} carry matrices, "
                         f"got {len(carry)}")
    for c in carry:
        if c.dtype != dtype:
            raise TypeError(f"carry dtype {c.dtype} != executed dtype {dtype}")
    carry, outs = jax.lax.scan(step, carry, jnp.arange(key_id.shape[0]))
    logits, w_norm, aux_norm = outs[:3]
    ret = dict(logits=logits, w_norm=w_norm, aux_norm=aux_norm,
               gates=(alpha, beta, mu, eta), final_carry=carry, dtype=dtype,
               weights=None, coeff=coeff)
    if rule == FL.FILTERED:
        ret["proc_max_abs"] = outs[3]
    return ret


def add_extension(p_native, rule):
    """The documented map from a saved native Momentum tree to an extension.
    `ordinary_prospective` takes the same single scalar kappa = 0 as the
    candidate, so all three start as exactly the native function.
    `filtered_processing` is the exception: it starts on the literal TSS
    boundary at T0, which is a different function from native; its native
    point is set explicitly by the study that wants it."""
    return convert_momentum(p_native, rule)


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
    if rule == FL.FILTERED:
        # three stored scalars, at the declared start M = gamma = 0, T = T0:
        # literal TSS, NOT the native function. The native point of this
        # family is (M, gamma, T) = (0, h, 0) and is applied explicitly.
        return dict(p_native, **FL.initial_leaves(dt))
    leaf = (PD.EXTRA_LEAF[rule] if rule in PD.EXTRA_LEAF
            else ("kappa" if rule == OD.ORDINARY else None))
    if leaf is None:
        raise ValueError(f"no extension leaf for {rule!r}")
    return dict(p_native, **{leaf: jnp.zeros((1,), dtype=dt)})


def parameter_counts(rule, p):
    flat = {k: int(onp.asarray(v).size) for k, v in p.items()}
    common = sum(v for k, v in flat.items() if k in COMMON)
    return dict(total=sum(flat.values()), common=common,
                extra=sum(flat.values()) - common, per_leaf=flat,
                carry_real_numbers=PD.CARRY[rule])
