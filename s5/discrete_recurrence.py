"""Shared scan and coefficient helpers for finite-difference S5 recurrences."""

import jax
import jax.numpy as np


def target_map(lambda_bar, b_bar, response, h=1.0):
    k = response.astype(lambda_bar.dtype) / h
    return 1.0 + k * (lambda_bar - 1.0), k[..., None] * b_bar


def zucchet_coefficients(lambda_bar, b_bar, response, h=1.0):
    F, G = target_map(lambda_bar, b_bar, response, h)
    k = response.astype(lambda_bar.dtype) / h
    return ((1.0 - k) + (1.0 + k) * lambda_bar,
            (k - 1.0) - k * lambda_bar,
            (1.0 + k)[..., None] * b_bar, -k[..., None] * b_bar)


def generalized_coefficients(lambda_bar, b_bar, response, mass, gamma, h=1.0):
    F, G = target_map(lambda_bar, b_bar, response, h)
    q = mass + h * (gamma + response)
    alpha, beta, delta = mass / q, h * h / q, h * response / q
    return ((1.0 + alpha - beta) + (beta + delta) * F,
            -alpha - delta * F,
            (beta + delta)[..., None] * G, -delta[..., None] * G)


def _block_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    c00 = A_j[..., 0, 0] * A_i[..., 0, 0] + A_j[..., 0, 1] * A_i[..., 1, 0]
    c01 = A_j[..., 0, 0] * A_i[..., 0, 1] + A_j[..., 0, 1] * A_i[..., 1, 1]
    c10 = A_j[..., 1, 0] * A_i[..., 0, 0] + A_j[..., 1, 1] * A_i[..., 1, 0]
    c11 = A_j[..., 1, 0] * A_i[..., 0, 1] + A_j[..., 1, 1] * A_i[..., 1, 1]
    A_out = np.stack([np.stack([c00, c01], -1), np.stack([c10, c11], -1)], -2)
    d0 = A_j[..., 0, 0] * b_i[..., 0] + A_j[..., 0, 1] * b_i[..., 1] + b_j[..., 0]
    d1 = A_j[..., 1, 0] * b_i[..., 0] + A_j[..., 1, 1] * b_i[..., 1] + b_j[..., 1]
    return A_out, np.stack([d0, d1], -1)


def companion_radius(a1, a2):
    root = np.sqrt(a1 * a1 + 4.0 * a2)
    return np.maximum(np.abs((a1 + root) / 2.0), np.abs((a1 - root) / 2.0))


def scan_companion(a1, a2, c1, c2, inputs, reverse=False):
    sequence = inputs[::-1] if reverse else inputs
    previous = np.concatenate((np.zeros_like(sequence[:1]), sequence[:-1]), axis=0)
    drive = (jax.vmap(lambda x: c1 @ x)(sequence) +
             jax.vmap(lambda x: c2 @ x)(previous))
    n = sequence.shape[0]
    A = np.zeros((n, a1.shape[0], 2, 2), dtype=a1.dtype)
    A = A.at[:, :, 0, 0].set(a1)
    A = A.at[:, :, 0, 1].set(a2)
    A = A.at[:, :, 1, 0].set(1.0)
    b = np.stack((drive, np.zeros_like(drive)), axis=-1)
    _, states = jax.lax.associative_scan(_block_operator, (A, b))
    states = states[:, :, 0]
    return states[::-1] if reverse else states
