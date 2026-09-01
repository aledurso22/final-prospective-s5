"""Causal prospective correction operators for S5 block outputs.

These operators deliberately act *after* an S5 state-space scan.  They do not
change the SSM transition, discretization, or associative scan.
"""

import math
from typing import Optional

from flax import linen as nn
import jax.numpy as jnp


def apply_parallel(sequence, alpha, reset_mask=None):
    """Apply a causal first-order lead to an entire sequence.

    Computes ``y[t] = x[t] + alpha * (x[t] - x[t-1])``.  At the first
    element, and wherever ``reset_mask[t]`` is true, the previous value is
    defined to be the current value.  The correction is therefore zero at a
    segment boundary.

    Args:
        sequence: Array shaped ``(length, ...)``.
        alpha: Scalar lead coefficient (or an array broadcastable to a token).
        reset_mask: Optional boolean array shaped ``(length,)``.  True marks
            the first token of a packed segment.
    """
    if sequence.ndim < 1:
        raise ValueError("sequence must have a leading sequence dimension")
    if sequence.shape[0] == 0:
        return sequence

    previous = jnp.concatenate((sequence[:1], sequence[:-1]), axis=0)
    if reset_mask is not None:
        reset_mask = jnp.asarray(reset_mask, dtype=bool)
        if reset_mask.ndim != 1 or reset_mask.shape[0] != sequence.shape[0]:
            raise ValueError("reset_mask must have shape (sequence_length,)")
        broadcast_shape = (reset_mask.shape[0],) + (1,) * (sequence.ndim - 1)
        previous = jnp.where(reset_mask.reshape(broadcast_shape), sequence, previous)

    return sequence + alpha * (sequence - previous)


def step(token, cache: Optional[jnp.ndarray], alpha, reset=False):
    """Apply the same lead to one streaming token.

    The cache is the preceding uncorrected token.  ``None`` and ``reset=True``
    both give a zero correction.  Returns ``(corrected_token, new_cache)``.
    """
    if cache is None:
        previous = token
    else:
        previous = jnp.where(jnp.asarray(reset, dtype=bool), token, cache)
    return token + alpha * (token - previous), token


class ProspectiveLead(nn.Module):
    """Swappable Flax wrapper around the normal prospective lead operator."""

    alpha: float = 0.0
    learned: bool = False
    alpha_max: float = 1.0

    def setup(self):
        if self.alpha_max <= 0.0:
            raise ValueError("alpha_max must be positive")
        if not 0.0 <= self.alpha <= self.alpha_max:
            raise ValueError("alpha must lie in [0, alpha_max]")

        if self.learned:
            # A sigmoid keeps the learned scalar strictly within the requested
            # interval.  An exact alpha=0 baseline remains available through
            # the fixed-alpha path.
            ratio = min(max(self.alpha / self.alpha_max, 1e-4), 1.0 - 1e-4)
            initial_logit = math.log(ratio / (1.0 - ratio))
            self.alpha_logit = self.param(
                "alpha_logit", lambda _key: jnp.asarray(initial_logit, dtype=jnp.float32)
            )

    def effective_alpha(self):
        if self.learned:
            return self.alpha_max * nn.sigmoid(self.alpha_logit)
        return jnp.asarray(self.alpha)

    def apply_parallel(self, sequence, reset_mask=None):
        return apply_parallel(sequence, self.effective_alpha(), reset_mask)

    def step(self, token, cache=None, reset=False):
        return step(token, cache, self.effective_alpha(), reset)

    def __call__(self, sequence, reset_mask=None):
        return self.apply_parallel(sequence, reset_mask)
