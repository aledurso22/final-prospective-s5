"""Rawat et al. Appendix E.3 Speech Commands architecture, reused by every arm.

Verbatim from E.3 (arXiv:2609.04134v1):

    "A real input projection maps d_in in {1,20} to n in {32,64} units per
    layer. The L in {2,4,6} pre-normalized residual blocks use this value of n,
    conjugate symmetry, eight HiPPO blocks, ZOH discretization, and a half-GLU
    pointwise map. The readout normalizes the final n-dimensional block state
    and projects it to 64 dimensions; that representation is mean-pooled and
    classified by a 64 -> 256 -> 10 GELU MLP with dropout 0.1. The reported
    runs use batch pre-normalization and are unidirectional."

Resolved, declared differences are listed in `docs/RAWAT_BASELINE_MAP.md`; the
two that matter here are recorded in the code as well:

* "half-GLU" does not say half_glu1 or half_glu2. We use ``half_glu1``, the
  repository default. DECLARED DIFFERENCE, identical on every arm.
* The paper gives dropout 0.1 for the classification MLP and does not state a
  block dropout. We set block dropout to 0.0 and apply 0.1 in the MLP only.
  DECLARED DIFFERENCE, identical on every arm.

Because an identical choice is applied to every arm, these affect the absolute
reproduction of the published number, not the comparison between arms.
"""

import jax.numpy as np
from flax import linen as nn

from .seq_model import StackedEncoderModel, masked_meanpool


class RawatClassifier(nn.Module):
    """Input projection -> L residual S5 blocks -> norm -> 64 -> pool -> MLP."""

    ssm: nn.Module
    d_model: int
    n_layers: int
    d_output: int = 10
    readout_width: int = 64
    mlp_hidden: int = 256
    mlp_dropout: float = 0.1
    block_dropout: float = 0.0
    activation: str = "half_glu1"
    training: bool = True
    prenorm: bool = True
    batchnorm: bool = True
    bn_momentum: float = 0.95
    step_rescale: float = 1.0

    def setup(self):
        self.encoder = StackedEncoderModel(
            ssm=self.ssm, d_model=self.d_model, n_layers=self.n_layers,
            activation=self.activation, dropout=self.block_dropout,
            training=self.training, prenorm=self.prenorm,
            batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale)
        self.readout_norm = (
            nn.BatchNorm(use_running_average=not self.training,
                         momentum=self.bn_momentum, axis_name="batch")
            if self.batchnorm else nn.LayerNorm())
        self.readout_proj = nn.Dense(self.readout_width)
        self.mlp_in = nn.Dense(self.mlp_hidden)
        self.mlp_out = nn.Dense(self.d_output)
        self.drop = nn.Dropout(self.mlp_dropout, broadcast_dims=[0],
                               deterministic=not self.training)

    def __call__(self, x, integration_timesteps=None, lengths=None):
        x = self.encoder(x, integration_timesteps)      # (L, d_model)
        x = self.readout_norm(x)
        x = self.readout_proj(x)                        # (L, 64)
        x = (masked_meanpool(x, lengths) if lengths is not None
             else np.mean(x, axis=0))                   # (64,)
        x = nn.gelu(self.mlp_in(x))
        x = self.drop(x)
        return self.mlp_out(x)                          # (d_output,)


BatchRawatClassifier = nn.vmap(
    RawatClassifier,
    in_axes=(0, 0, 0), out_axes=0,
    variable_axes={"params": None, "dropout": None, "batch_stats": None,
                   "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name="batch")


def parameter_report(params):
    """Total and per-group parameter counts, for the manifest."""
    import jax
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    total = sum(int(v.size) for v in flat.values())
    ssm = sum(int(v.size) for k, v in flat.items() if "/seq/" in "/".join(k)
              or k[-2] == "seq")
    return dict(total=total, ssm=ssm, other=total - ssm)
