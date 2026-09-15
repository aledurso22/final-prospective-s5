"""Causal S5 stack with a tokenwise query readout, shared by every arm.

No temporal pooling, no bidirectionality, no temporal normalization, no
attention, and no readout access to earlier activations: the logits at the
query token are a function of the recurrent carry and that token's own input.
Normalization is tokenwise LayerNorm in EVERY arm, so nothing normalizes across
time or batch. The ordinary nonlinear blocks and residual connections are the
shared `SequenceLayer`.
"""

import jax.numpy as np
from flax import linen as nn

from .seq_model import StackedEncoderModel


class RecallModel(nn.Module):
    ssm: nn.Module
    d_model: int
    n_layers: int
    d_output: int
    query_index: int

    def setup(self):
        self.encoder = StackedEncoderModel(
            ssm=self.ssm, d_model=self.d_model, n_layers=self.n_layers,
            activation="half_glu1", dropout=0.0, training=False,
            prenorm=True, batchnorm=False, bn_momentum=0.9, step_rescale=1.0)
        self.readout_norm = nn.LayerNorm()
        self.readout = nn.Dense(self.d_output)

    def __call__(self, x, integration_timesteps=None):
        h = self.encoder(x, integration_timesteps)      # (L, d_model)
        z = self.readout(self.readout_norm(h))          # tokenwise, causal
        return z[self.query_index]                      # logits at the query

    def all_tokens(self, x, integration_timesteps=None):
        h = self.encoder(x, integration_timesteps)
        return self.readout(self.readout_norm(h))


BatchRecallModel = nn.vmap(
    RecallModel, in_axes=(0, 0), out_axes=0,
    variable_axes={"params": None}, split_rngs={"params": False})


def parameter_count(params):
    import jax
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
