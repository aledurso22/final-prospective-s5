"""One small feature/readout shell, five memory rules inside it.

Everything outside the memory update is identical across arms and identically
initialized at a given seed: a 32x8 raw key table, an 8x8 value table, and an
8x8 readout with bias. What differs is the inner update, which is the point.

WHAT THIS SHELL OMITS, for every arm alike: short convolutions, output
corrections, output gating/RMS normalization, multiple heads, value expansion
and any large-model block. An implementation of a published RULE inside this
shell is **not** a reproduction of that paper's complete model, and the absence
of those extras is a common-shell adaptation, not a claim they are
unnecessary.

GATE PARAMETERIZATION is taken from the official implementation pinned in
`docs/NESTED_MEMORY_PROTOCOL.md`, not paraphrased from the papers. The only
declared departures are the shell omissions above and the gate input, which
here is the token's own event description rather than a backbone hidden state.
"""

import math

import jax
import jax.numpy as jnp
import numpy as onp

from . import dynamics as D
from .task import IDLE, N_KEYS, N_VALUES, QUERY, WRITE

D_K = D_V = 8
#: gate input: key one-hot (32) + observed value one-hot (8) + event flags (3)
GATE_DIM = N_KEYS + N_VALUES + 3

# ---- official constants, pinned at MomentumDeltaNet commit c6e77fa2
A_MIN_INIT, A_MAX_INIT = 0.0, 16.0
M_MIN_INIT, M_MAX_INIT = 0.0, 16.0
DT_MIN, DT_MAX, DT_INIT_FLOOR = 0.001, 0.1, 1e-4
FACTOR_SCALE = 4.0
TAU_FACTOR = 1
#: official constructor default `min_log_mu: float = -2.`
MIN_LOG_MU = -2.0


def _inv_softplus(x):
    return x + onp.log(-onp.expm1(-x))


def _linear_init(rng, n_in, n_out):
    """PyTorch `nn.Linear(bias=False)` default: U(-1/sqrt(in), 1/sqrt(in))."""
    b = 1.0 / math.sqrt(n_in)
    return rng.uniform(-b, b, size=(n_out, n_in))


def init_params(rule, seed, dtype=onp.float32):
    """Common tensors from ONE stream, gate tensors appended after.

    Additive gate parameters are drawn from a separate stream so they cannot
    change the common tables or the data order, which is what makes the arms
    comparable at a seed.
    """
    rng = onp.random.RandomState(seed)
    p = dict(
        key_raw=rng.randn(N_KEYS, D_K).astype(dtype),      # normalized in use
        value_table=onp.eye(N_VALUES, D_V, dtype=dtype),   # one-hot basis
        readout_W=onp.eye(N_VALUES, D_V, dtype=dtype),     # identity
        readout_b=onp.zeros(N_VALUES, dtype=dtype))
    g = onp.random.RandomState(seed + 991)
    if rule == "gated_delta":
        A = g.uniform(A_MIN_INIT, A_MAX_INIT, size=(1,))
        dt = onp.clip(onp.exp(g.uniform(math.log(DT_MIN), math.log(DT_MAX),
                                        size=(1,))), DT_INIT_FLOOR, None)
        p |= dict(a_proj=_linear_init(g, GATE_DIM, 1).astype(dtype),
                  b_proj=_linear_init(g, GATE_DIM, 1).astype(dtype),
                  A_log=onp.log(A).astype(dtype),
                  dt_bias=_inv_softplus(dt).astype(dtype))
    elif rule == "momentum_delta":
        A = g.uniform(A_MIN_INIT, A_MAX_INIT, size=(1,))
        Mu = g.uniform(M_MIN_INIT, M_MAX_INIT, size=(1,))
        dt = onp.clip(onp.exp(g.uniform(math.log(DT_MIN), math.log(DT_MAX),
                                        size=(1,))), DT_INIT_FLOOR, None)
        inv_dt = _inv_softplus(dt)
        p |= dict(a_proj=_linear_init(g, GATE_DIM, 1).astype(dtype),
                  b_proj=_linear_init(g, GATE_DIM, 1).astype(dtype),
                  m_proj=_linear_init(g, GATE_DIM, 1).astype(dtype),
                  e_proj=_linear_init(g, GATE_DIM, 1).astype(dtype),
                  A_log=onp.log(A).astype(dtype),
                  Mu_log=onp.log(Mu).astype(dtype),
                  dt_bias=inv_dt.astype(dtype),
                  mu_bias=inv_dt.astype(dtype),
                  log_factor=onp.log(g.uniform(0.0, FACTOR_SCALE,
                                               size=(1,))).astype(dtype))
    # every leaf explicitly typed: a weakly typed leaf retraces a step that
    # consumes its own optimizer output
    return {k: jnp.asarray(v, dtype=dtype) for k, v in p.items()}


def gate_features(key_id, val_id, event):
    """The token's own event description. NO oracle state, no query label, no
    future key and no family identifier."""
    k1 = jax.nn.one_hot(key_id, N_KEYS)
    has_v = (val_id >= 0).astype(k1.dtype)
    v1 = jax.nn.one_hot(jnp.maximum(val_id, 0), N_VALUES) * has_v[..., None]
    ev = jax.nn.one_hot(event, 3)
    return jnp.concatenate([k1, v1, ev], axis=-1)


def _gated_delta_gates(p, x):
    """`log_alpha = -exp(A_log) softplus(a + dt_bias)`, `beta = sigmoid(b)`."""
    a = x @ p["a_proj"][0]
    b = x @ p["b_proj"][0]
    log_alpha = -jnp.exp(p["A_log"][0]) * jax.nn.softplus(a + p["dt_bias"][0])
    return jnp.exp(log_alpha), jax.nn.sigmoid(b)


def _momentum_gates(p, x):
    """The official Momentum DeltaNet gate block, including the mu clamp."""
    tau = math.sqrt(GATE_DIM / TAU_FACTOR)
    a = x @ p["a_proj"][0]
    b = x @ p["b_proj"][0]
    m = x @ p["m_proj"][0]
    e = (x @ p["e_proj"][0]) / tau
    log_alpha = -jnp.exp(p["A_log"][0]) * jax.nn.softplus(a + p["dt_bias"][0])
    log_mu = -jnp.exp(p["Mu_log"][0]) * jax.nn.softplus(m + p["mu_bias"][0])
    log_mu = jnp.maximum(log_mu, MIN_LOG_MU)          # official clamp_min_
    eta = jnp.tanh(e) + 1.0                            # range (0, 2)
    beta = jax.nn.sigmoid(b)
    theta = jnp.arctan(eta * jnp.exp(p["log_factor"][0]))
    beta = (jnp.sin(theta) ** 2) * beta
    log_alpha = jnp.log(jnp.cos(theta) ** 2) + log_alpha
    return jnp.exp(log_alpha), beta, jnp.exp(log_mu), eta


def rollout(rule, p, ep, const, dtype=jnp.float32, carry0=None):
    """One episode through the shell. Returns query logits and diagnostics.

    The query is read AFTER advancing its interval, so it sees the autonomous
    motion during that interval. Carries persist across the whole sequence and
    are reset only between episodes.
    """
    key_id, val_id, event = ep["key_id"], ep["val_id"], ep["event"]
    k_all, k_valid = D.safe_normalize(p["key_raw"])
    keys = k_all[key_id]                                   # (L, d_k)
    valid = k_valid[key_id]
    has_v = (val_id >= 0).astype(dtype)
    vals = p["value_table"][jnp.maximum(val_id, 0)] * has_v[:, None]
    mask = ((event == WRITE).astype(dtype)) * valid        # zero-key => no write
    gx = gate_features(key_id, val_id, event).astype(dtype)

    if rule in ("gated_delta", "momentum_delta"):
        if rule == "gated_delta":
            alpha, beta = _gated_delta_gates(p, gx)
            gates = (alpha, beta)
        else:
            alpha, beta, mu, eta = _momentum_gates(p, gx)
            gates = (alpha, beta, mu, eta)
    else:
        gates = ()

    F = jnp.asarray(const["F"], dtype=dtype) if "F" in const else None
    a0 = jnp.asarray(const.get("a0", 0.0), dtype=dtype)
    b0 = jnp.asarray(const.get("b0", 0.0), dtype=dtype)
    beta_fixed = jnp.asarray(const.get("beta", 0.0), dtype=dtype)

    def step(carry, t):
        k, v, m = keys[t], vals[t], mask[t]
        if rule in ("prospective_memory", "inertial_memory"):
            carry = D.two_state_step(carry, k, v, m, F, a0, b0)
        elif rule == "delta_matched_write":
            carry = D.delta_step(carry, k, v, m, beta_fixed)
        elif rule == "gated_delta":
            carry = D.gated_delta_step(carry, k, v, m, gates[0][t], gates[1][t])
        else:
            carry = D.momentum_delta_step(carry, k, v, m, gates[0][t],
                                          gates[1][t], gates[2][t], gates[3][t])
        W = carry[0]
        logits = p["readout_W"] @ (W @ k) + p["readout_b"]
        aux = jnp.sqrt(jnp.sum(carry[1] ** 2)) if len(carry) > 1 else 0.0
        return carry, (logits, jnp.sqrt(jnp.sum(W ** 2)), aux)

    carry = D.init_carry(rule, D_V, D_K, dtype) if carry0 is None else carry0
    carry, (logits, w_norm, aux_norm) = jax.lax.scan(
        step, carry, jnp.arange(key_id.shape[0]))
    return dict(logits=logits, w_norm=w_norm, aux_norm=aux_norm,
                gates=gates, final_carry=carry)


def constants_for(rule):
    if rule == "prospective_memory":
        return D.prospective_constants()
    if rule == "inertial_memory":
        return D.inertial_constants()
    if rule == "delta_matched_write":
        return dict(beta=D.beta_match())
    return {}


def parameter_counts(rule, p):
    flat = {k: onp.asarray(v).size for k, v in p.items()}
    common = sum(v for k, v in flat.items()
                 if k in ("key_raw", "value_table", "readout_W", "readout_b"))
    return dict(total=sum(flat.values()), common=common,
                gate=sum(flat.values()) - common, per_leaf=flat,
                carry_real_numbers=D.carry_size(rule, D_V, D_K))
