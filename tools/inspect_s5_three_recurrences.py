"""Print one shared-token transition for all three recurrence equations."""

import os
import sys

import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5.three_arm_recurrences import (
    generalized_prospective_s5_transition,
    native_matched_s5_transition,
    zucchet_prospective_s5_transition,
)


def main():
    a = jnp.asarray([-0.35 + 0.20j, -0.55 + 0.10j])
    b = jnp.asarray([[0.30 + 0.05j], [0.15 - 0.20j]])
    x = jnp.asarray([0.7])
    native_state = jnp.asarray([0.1 + 0.2j, -0.2 + 0.1j])
    prospective_state = native_state
    generalized_state = jnp.stack((native_state, jnp.zeros_like(native_state)), axis=-1)
    native_bar = jnp.exp(a)
    native_b = (jnp.expm1(a) / a)[:, None] * b
    native_next = native_matched_s5_transition(native_state, x, native_bar, native_b)
    prospective_next, prospective_observed = zucchet_prospective_s5_transition(
        prospective_state, x, a, b, jnp.asarray([0.7, 0.7]))
    generalized_next, generalized_observed = generalized_prospective_s5_transition(
        generalized_state, x, a, b, jnp.asarray([0.35, 0.35]),
        jnp.ones(2), jnp.asarray([0.7, 0.7]))
    print("Native matched S5 state:", native_next)
    print("Zucchet prospective S5 recurrence state:", prospective_next)
    print("Zucchet observed state:", prospective_observed)
    print("Generalized prospective S5 recurrence state:", generalized_next)
    print("Generalized observed state:", generalized_observed)


if __name__ == "__main__":
    main()
