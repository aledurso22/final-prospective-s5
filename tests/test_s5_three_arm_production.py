import os
import sys

import jax
import numpy as np
from flax.traverse_util import flatten_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.s5_three_arm_full.runner import (ARM_ORDER, BATCH_SIZE,
                                                   model_for, init_state,
                                                   parameter_count,
                                                   recurrent_state_size)


def test_production_models_initialize_and_are_finite():
    for arm in ARM_ORDER:
        state = init_state(arm, 301)
        assert parameter_count(state.params) > 0
        assert all(np.all(np.isfinite(np.asarray(x)))
                   for x in jax.tree_util.tree_leaves(state.params))
        assert model_for(arm, True) is not None


def test_production_state_sizes_are_declared():
    assert recurrent_state_size("native_matched_s5")["real_values_total"] == 128
    assert recurrent_state_size("zucchet_prospective_s5")["real_values_total"] == 128
    assert recurrent_state_size("generalized_prospective_s5")["real_values_total"] == 256
    assert BATCH_SIZE == 32


def test_shared_initial_parameters_are_identical_where_shapes_permit():
    states = {arm: flatten_dict(init_state(arm, 301).params) for arm in ARM_ORDER}
    native = states["native_matched_s5"]
    for arm in ARM_ORDER[1:]:
        common = set(native) & set(states[arm])
        assert common
        for name in common:
            np.testing.assert_array_equal(native[name], states[arm][name])
