"""Token-wise dual-head model for the cue/recall task.

Deliberately NOT `ClassificationModel`: no pooling over time, no temporal
batch normalization, no future tokens in features, no teacher forcing. The
encoder is the existing `StackedEncoderModel`, unchanged, so every mechanism
is compared through the same stack.

Two heads read the SAME per-token representation:
    recall head   -> 8 logits at every token (scored only at the query token)
    current head  -> 1 logit at every token (scored at every token)

Native feedthrough `D` and the residual paths are left exactly as they are in
all arms. They make the current-input head easier; that is part of the
architecture under test, not something to remove to flatter one mechanism.
"""

import jax
from flax import linen as nn

from .seq_model import StackedEncoderModel


class TokenwiseDualHead(nn.Module):
    ssm: nn.Module
    d_model: int
    n_layers: int
    n_recall: int = 8
    activation: str = "half_glu1"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = True
    batchnorm: bool = False          # LayerNorm, per the frozen spec
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        self.encoder = StackedEncoderModel(
            ssm=self.ssm, d_model=self.d_model, n_layers=self.n_layers,
            activation=self.activation, dropout=self.dropout,
            training=self.training, prenorm=self.prenorm,
            batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale)
        self.recall_head = nn.Dense(self.n_recall)
        self.current_head = nn.Dense(1)

    def __call__(self, x, integration_timesteps):
        h = self.encoder(x, integration_timesteps)      # (L, d_model)
        return self.recall_head(h), self.current_head(h)[..., 0]


BatchTokenwiseDualHead = nn.vmap(
    TokenwiseDualHead,
    in_axes=(0, 0), out_axes=0,
    variable_axes={"params": None, "dropout": None, "batch_stats": None,
                   "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name="batch")
