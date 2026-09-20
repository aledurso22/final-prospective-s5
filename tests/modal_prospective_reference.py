"""FROZEN sequential oracle for the modal cascade. TESTS ONLY.

Nothing in `s5/` or any launcher may import this. Independent transcription
of the stage equation as an obvious loop.
"""

import jax.numpy as np


def stage_sequential(values, d, n, h=1.0, reverse=False):
    """u_t = [d/(h+d)]u_{t-1} + [(h+n)/(h+d)]v_t - [n/(h+d)]v_{t-1}."""
    sequence = values[::-1] if reverse else values
    pole = (d / (h + d)).astype(sequence.dtype)
    alpha = ((h + n) / (h + d)).astype(sequence.dtype)
    beta = (-n / (h + d)).astype(sequence.dtype)
    state = np.zeros_like(sequence[0])
    previous = np.zeros_like(sequence[0])
    out = []
    for value in sequence:
        state = pole * state + alpha * value + beta * previous
        out.append(state)
        previous = value
    stacked = np.stack(out, axis=0)
    return stacked[::-1] if reverse else stacked


def cascade_sequential(values, stages, h=1.0, reverse=False):
    out = values
    for d, n in stages:
        out = stage_sequential(out, d, n, h, reverse)
    return out


def native_sequential(lambda_bar, b_bar, inputs):
    state = np.zeros_like(lambda_bar)
    out = []
    for x in inputs:
        state = lambda_bar * state + b_bar @ x
        out.append(state)
    return np.stack(out, axis=0)
