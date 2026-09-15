"""ONE post-update projection for the learned response leaves.

Why this is a shared module and not a per-runner helper
------------------------------------------------------
The recall study's first execution clipped the response in the FORWARD pass
but never projected the RAW parameter after an optimizer update. Above the
upper bound `u = log(0.9999)` the forward clip makes `d rho / d eta = 0`, so
the task gradient through that leaf is exactly zero and nothing brings it back:
AdamW's decoupled decay shrinks `eta` toward 0, which is AWAY from a negative
upper bound. The forward law stays admissible, so the failure is silent, and it
reversed that study's only consistent positive result.

The Speech Commands trainer had a projection, but its name whitelist predated
the rho-only parameterization: it recognized `log_response_gamma` and
`log_response_rho` and would NOT have projected `log_response_rho_only`. Two
independent whitelists is how that hazard recurs, so there is now one.

Projection acts on PARAMETERS only. Optimizer state is deliberately untouched:
the momentum that proposed an inadmissible step is left in place, so the policy
is a feasible-set projection and not a hidden optimizer reset.
"""

import jax
import jax.numpy as jnp

from .rawat_s5 import (LOG_GAMMA_BOUNDS, LOG_RHO_BOUNDS,
                       RESPONSE_PARAM_NAMES, RHO_ONLY_PARAM_NAME)

#: every learned response leaf, with its declared RAW-log interval
RESPONSE_LEAF_BOUNDS = {
    "log_response_gamma": LOG_GAMMA_BOUNDS,     # superseded gp_learned_response
    "log_response_rho": LOG_RHO_BOUNDS,         # superseded gp_learned_response
    RHO_ONLY_PARAM_NAME: LOG_RHO_BOUNDS,        # current rho-only arms
}

#: leaf names that are response parameters, in declaration order
ALL_RESPONSE_LEAF_NAMES = tuple(RESPONSE_LEAF_BOUNDS)


def leaf_name(path):
    return path[-1].key if hasattr(path[-1], "key") else str(path[-1])


def bounds_for(name, dtype=None):
    """The declared interval, cast to the LEAF's dtype when one is given.

    The leaves are created float32 even under x64, so comparing a float32 leaf
    against the float64 endpoint `log(0.01)` is a representation mismatch, not
    an admissibility question. Casting first keeps the interval itself
    unchanged.
    """
    lo, hi = RESPONSE_LEAF_BOUNDS[name]
    if dtype is None:
        return lo, hi
    return (jnp.asarray(lo, dtype=dtype), jnp.asarray(hi, dtype=dtype))


def project_response_leaves(params):
    """Clip every raw response leaf back into its declared interval."""
    def fix(path, v):
        name = leaf_name(path)
        if name not in RESPONSE_LEAF_BOUNDS:
            return v
        lo, hi = bounds_for(name, v.dtype)
        return jnp.clip(v, lo, hi)

    return jax.tree_util.tree_map_with_path(fix, params)


def response_leaves(tree, names=None):
    """The response leaves of a tree, as a list. Empty for every other arm."""
    from flax.traverse_util import flatten_dict
    want = RESPONSE_LEAF_BOUNDS if names is None else set(names)
    return [v for k, v in flatten_dict(tree).items() if k[-1] in want]


def projection_telemetry(raw, projected, grads=None, updates=None,
                         names=None):
    """Make the bound interaction OBSERVABLE rather than inferred.

    `n_projected` counts (entry, update) EVENTS - one per raw coordinate the
    projection had to move on this step. It is not a count of distinct modes,
    distinct steps, or time spent at the boundary. `max_overshoot` is the
    PROPOSED pre-projection excursion, i.e. the step the optimizer offered and
    the projection rejected; it is never evidence that the forward model used
    an inadmissible coefficient, because the forward clip precedes it.
    """
    from flax.traverse_util import flatten_dict
    want = RESPONSE_LEAF_BOUNDS if names is None else set(names)
    flat_raw = {k: v for k, v in flatten_dict(raw).items() if k[-1] in want}
    if not flat_raw:
        z = jnp.asarray(0.0)
        return dict(n_projected=jnp.asarray(0), max_overshoot=z,
                    response_grad_norm=z, response_update_norm=z)
    flat_pro = {k: v for k, v in flatten_dict(projected).items()
                if k[-1] in want}
    moved, over = [], []
    for k, a in flat_raw.items():
        lo, hi = bounds_for(k[-1], a.dtype)
        moved.append(jnp.sum(jnp.abs(a - flat_pro[k]) > 0))
        over.append(jnp.max(jnp.maximum(jnp.maximum(a - hi, lo - a), 0.0)))

    def norm(tree):
        vs = response_leaves(tree, want) if tree is not None else []
        if not vs:
            return jnp.asarray(0.0)
        return jnp.sqrt(sum(jnp.sum(v.astype(jnp.float32) ** 2) for v in vs))

    return dict(n_projected=jnp.sum(jnp.stack(moved)),
                max_overshoot=jnp.max(jnp.stack(over)),
                response_grad_norm=norm(grads),
                response_update_norm=norm(updates))
