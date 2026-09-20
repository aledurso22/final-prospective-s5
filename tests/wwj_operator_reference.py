"""FROZEN sequential oracle for the PRINCIPAL WWJ architecture. TESTS ONLY.

Nothing in `s5/` and no launcher may import this. It is an independent
transcription: the Native recurrence as an explicit loop, the augmented
three-block realization as an explicit loop, and the three-tap written out
again from the derivation.
"""

import jax.numpy as np


def native_states_sequential(lambda_bar, b_bar, inputs):
    """s_{t+1} = Abar s_t + Bbar x_t with s_{-1} = 0, by an obvious loop."""
    state = np.zeros_like(lambda_bar)
    out = []
    for x in inputs:
        state = lambda_bar * state + b_bar @ x
        out.append(state)
    return np.stack(out, axis=0)


def three_tap_sequential(states, k, m):
    """z_t = (1+k+m)s_t - (k+2m)s_{t-1} + m s_{t-2}, zero prehistory."""
    zero = np.zeros_like(states[0])
    s1 = s2 = zero
    out = []
    for s in states:
        out.append((1.0 + k + m) * s - (k + 2.0 * m) * s1 + m * s2)
        s1, s2 = s, s1
    return np.stack(out, axis=0)


def augmented_sequential(lambda_bar, b_bar, inputs, k, m):
    """The augmented recurrent realization, run literally:

    q_{t+1} = [[Abar,0,0],[I,0,0],[0,I,0]] q_t + [Bbar x_t; 0; 0]
    z_t     = [(1+k+m)I, -(k+2m)I, m I] q_t
    """
    zero = np.zeros_like(lambda_bar)
    q0 = q1 = q2 = zero
    out = []
    for x in inputs:
        q0, q1, q2 = lambda_bar * q0 + b_bar @ x, q0, q1
        out.append((1.0 + k + m) * q0 - (k + 2.0 * m) * q1 + m * q2)
    return np.stack(out, axis=0)


def wwj_sequential(lambda_bar, b_bar, inputs, k, m, reverse=False):
    sequence = inputs[::-1] if reverse else inputs
    states = three_tap_sequential(
        native_states_sequential(lambda_bar, b_bar, sequence), k, m)
    return states[::-1] if reverse else states
