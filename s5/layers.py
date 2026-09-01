from flax import linen as nn
import jax

from .prospective import ProspectiveLead


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
            prospective_mode (string): "off" reproduces upstream S5 exactly.  "lead"
                                    applies the causal prospective correction
                                    b_pc[t] = b[t] + alpha*(b[t]-b[t-1]) to the SSM
                                    preactivation, after the readout/feedthrough and
                                    before the activation/GLU, dropout and residual.
            prospective_alpha (float32): lead coefficient (initial value if learned)
            prospective_alpha_learned (bool): learn alpha as alpha_max*sigmoid(a)
            prospective_alpha_max (float32): upper bound used by the learned parameterization
            bidirectional (bool):   whether the wrapped SSM is bidirectional.  The
                                    prospective operator is causal and rejects this.
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    prospective_mode: str = "off"
    prospective_alpha: float = 0.0
    prospective_alpha_learned: bool = False
    prospective_alpha_max: float = 1.0
    bidirectional: bool = False

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(step_rescale=self.step_rescale)

        if self.prospective_mode not in ("off", "lead"):
            raise ValueError(
                "prospective_mode must be one of ('off', 'lead'), got "
                "{}".format(self.prospective_mode))

        if self.prospective_mode == "lead":
            if self.bidirectional:
                raise ValueError(
                    "prospective_mode='lead' is causal and requires a "
                    "unidirectional S5; set bidirectional=False.")
            self.prospective = ProspectiveLead(
                alpha=self.prospective_alpha,
                learned=self.prospective_alpha_learned,
                alpha_max=self.prospective_alpha_max,
            )

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)

        # Causal prospective coordinate, applied to the S5 preactivation after
        # the readout and feedthrough and before the activation/GLU, dropout
        # and residual path.  The state recurrence and associative scan above
        # are untouched.
        if self.prospective_mode == "lead":
            x = self.prospective.apply_parallel(x)

        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x
