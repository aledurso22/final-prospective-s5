"""Model and optimizer construction for the WWJ arms.

This module builds on `experiments/s5_three_arm_full/runner.py` WITHOUT
modifying it: the shared configuration, the HiPPO initialization and the
classification model are imported, so width, depth, dropout, batch size,
sequence length, seeds and data handling are literally the production ones.
Only the SSM constructor and the timescale-parameter optimizer group differ.

Optimizer safeguards for the WWJ timescale parameters, declared here:

* `wwj_tau_raw` and `wwj_eps_raw` go in their own group at
  `WWJ_LR_MULTIPLIER * ssm_lr` (0.1x), because they reparameterize the whole
  recurrence and a step that is fine for B or C is not fine for tau;
* no weight decay on them, for the reason already documented in
  `s5/train_helpers.py`: decay pulls the RAW value toward zero, which is an
  arbitrary preferred timescale, not a small one;
* the rest of the optimizer, including the absence of global gradient
  clipping, is exactly the production `noBCdecay` configuration.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as np
import optax
from flax.training import train_state

from dataloaders import speech_commands10 as SC
from experiments.s5_three_arm_full.runner import (BATCH_SIZE, D_MODEL,
                                                  INPUT_DIM, LR, N_LAYERS,
                                                  SEQ_LEN, SHARED_CONFIG,
                                                  SSM_LR, WEIGHT_DECAY,
                                                  ssm_kwargs)
from s5.seq_model import BatchClassificationModel
from s5.train_helpers import map_nested_fn, no_bc_decay_group
from s5.wwj_ssm import (WWJ_DIAGNOSTIC_ARMS, WWJ_PRINCIPAL_ARMS,
                        WWJ_SCIENTIFIC_NAMES, init_wwj_critical_S5SSM,
                        init_wwj_passive_S5SSM)

#: the timescale parameters, and how much slower they learn
WWJ_PARAMETER_KEYS = frozenset({"wwj_tau_raw", "wwj_eps_raw"})
WWJ_LR_MULTIPLIER = 0.1
#: the initialization selected by `init_grid.py` under its declared rule;
#: passed explicitly by every launcher so it is never implicit. The default
#: is small and close to Native, as the FIR selection rule prefers.
DEFAULT_TAU_INIT = 0.05
DEFAULT_EPS_INIT = 0.0625


def wwj_ssm_factory(arm, tau_init=DEFAULT_TAU_INIT, eps_init=DEFAULT_EPS_INIT):
    """The SSM constructor for a WWJ arm, on production S5 kwargs.

    Only the principal arms and the named diagnostic are constructible here.
    The REJECTED mixed-stencil realization is deliberately unreachable: it is
    unstable on the real S5 modes and must never run in production.
    """
    kw = ssm_kwargs("native_matched_s5")     # identical HiPPO/model geometry
    if arm == "wwj_critical_s5":
        return init_wwj_critical_S5SSM(tau_init=tau_init, **kw)
    if arm == "wwj_passive_s5":
        return init_wwj_passive_S5SSM(tau_init=tau_init, eps_init=eps_init,
                                      **kw)
    if arm in WWJ_DIAGNOSTIC_ARMS:
        return WWJ_DIAGNOSTIC_ARMS[arm](tau_init=tau_init, eps_init=eps_init,
                                        **kw)
    if "mixed_stencil" in arm:
        raise ValueError(
            f"{arm} is the REJECTED mixed-stencil realization (unstable on "
            "S5 modes, max companion radius ~1.705); it is a failed ablation "
            "and cannot be constructed here")
    raise ValueError(f"not a WWJ arm: {arm}")


def wwj_model_cls(arm, **kwargs):
    return partial(BatchClassificationModel, ssm=wwj_ssm_factory(arm, **kwargs),
                   d_output=len(SC.WORDS), d_model=D_MODEL, n_layers=N_LAYERS,
                   padded=False, activation="half_glu1",
                   dropout=SHARED_CONFIG.dropout, mode="pool", prenorm=True,
                   batchnorm=True, bn_momentum=0.95)


def wwj_model(arm, training, **kwargs):
    return wwj_model_cls(arm, **kwargs)(training=training)


def wwj_group(key):
    """Four groups: the production three plus a slower `wwj` group."""
    return "wwj" if key in WWJ_PARAMETER_KEYS else no_bc_decay_group(key)


def create_wwj_train_state(arm, seed, tau_init=DEFAULT_TAU_INIT,
                           eps_init=DEFAULT_EPS_INIT, batch_size=BATCH_SIZE,
                           seq_len=SEQ_LEN):
    """The production `noBCdecay` optimizer plus the slower WWJ group."""
    model_cls = wwj_model_cls(arm, tau_init=tau_init, eps_init=eps_init)
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(seed))
    dummy = np.ones((batch_size, seq_len, INPUT_DIM))
    variables = model.init({"params": init_rng, "dropout": dropout_rng},
                           dummy, np.ones((batch_size, seq_len)))
    params = dict(variables["params"])
    tx = optax.multi_transform(
        {
            "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
            "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=SSM_LR),
            "wwj": optax.inject_hyperparams(optax.adam)(
                learning_rate=WWJ_LR_MULTIPLIER * SSM_LR),
            "regular": optax.inject_hyperparams(optax.adamw)(
                learning_rate=LR, weight_decay=WEIGHT_DECAY),
        },
        map_nested_fn(lambda k, _: wwj_group(k)),
    )

    class TrainState(train_state.TrainState):
        batch_stats: Any

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx,
                             batch_stats=variables["batch_stats"])


def set_wwj_learning_rates(state, lr_value, ssm_lr_value):
    """Set every group's rate, the WWJ group at its declared multiplier."""
    inner = state.opt_state.inner_states
    inner["regular"].inner_state.hyperparams["learning_rate"] = np.array(
        lr_value, dtype=np.float32)
    inner["ssm"].inner_state.hyperparams["learning_rate"] = np.array(
        ssm_lr_value, dtype=np.float32)
    inner["wwj"].inner_state.hyperparams["learning_rate"] = np.array(
        WWJ_LR_MULTIPLIER * ssm_lr_value, dtype=np.float32)
    return state


def wwj_diagnostics(state, arm):
    """Per-layer tau, eps, M, k, m, FIR gain and the NATIVE mode radii.

    The WWJ operator is FIR: it adds zeros, not poles, so there is no
    companion spectral radius to report. What matters is how much it
    amplifies (`max_fir_gain`) and that the Native poles -- the only poles in
    the layer -- are where they always were.
    """
    from flax.traverse_util import flatten_dict

    from s5.ssm import discretize_zoh
    from s5.wwj_operator import (frequency_response, k_and_m, max_fir_gain,
                                 native_radii)
    from s5.wwj_ssm import eps_passive_from_raw, tau_from_raw

    flat = flatten_dict(state.params)
    layers = {}
    for key in flat:
        if key[-1] != "wwj_tau_raw":
            continue
        prefix = key[:-1]
        name = "/".join(prefix)
        tau = tau_from_raw(flat[key])
        if prefix + ("wwj_eps_raw",) in flat:
            eps = eps_passive_from_raw(flat[prefix + ("wwj_eps_raw",)])
        else:
            eps = np.asarray(0.25, dtype=np.float32)
        k, m = k_and_m(tau, eps)
        lambda_continuous = (np.clip(flat[prefix + ("Lambda_re",)], None, -1e-4)
                             + 1j * flat[prefix + ("Lambda_im",)])
        b = flat[prefix + ("B",)]
        step = np.exp(flat[prefix + ("log_step",)][:, 0])
        lambda_bar, _ = discretize_zoh(lambda_continuous,
                                       b[..., 0] + 1j * b[..., 1], step)
        radii = native_radii(lambda_bar)
        omega = np.linspace(0.0, np.pi, 9)
        layers[name] = {
            "tau": float(tau), "eps": float(eps),
            "mass": float(eps * tau * tau), "k": float(k), "m": float(m),
            "max_fir_gain": float(max_fir_gain(k, m)),
            "fir_response_abs_on_coarse_grid":
                [float(value) for value in
                 np.abs(frequency_response(k, m, omega))],
            "native_radius_max": float(np.max(radii)),
            "native_radius_min": float(np.min(radii)),
            "native_radius_per_mode": [float(value) for value in radii],
            "gate": (float(jax.nn.sigmoid(flat[prefix + ("wwj_gate_raw",)]))
                     if prefix + ("wwj_gate_raw",) in flat else None),
        }
    return {"arm": arm, "scientific_name": WWJ_SCIENTIFIC_NAMES[arm],
            "operator": "FIR three-tap; recurrent poles are the Native poles "
                        "plus two zeros",
            "layers": layers,
            "max_fir_gain": max((values["max_fir_gain"]
                                 for values in layers.values()),
                                default=float("nan")),
            "max_native_radius": max((values["native_radius_max"]
                                      for values in layers.values()),
                                     default=float("nan"))}
