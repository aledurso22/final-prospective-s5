import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5.generalized_prospective_ssm import _block_operator


def _reference(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, (A_j @ b_i[..., None])[..., 0] + b_j


@pytest.mark.parametrize("dtype", (jnp.complex64, jnp.complex128))
def test_scalar_block_operator_matches_matrix_multiplication(dtype):
    A_i = jnp.asarray([[[1.0, 0.2], [-0.1, 0.8]]], dtype=dtype)
    A_j = jnp.asarray([[[0.7, -0.3], [0.4, 0.6]]], dtype=dtype)
    b_i = jnp.asarray([[0.2 + 0.1j, 0.4 - 0.2j]], dtype=dtype)
    b_j = jnp.asarray([[0.1 - 0.3j, -0.2 + 0.4j]], dtype=dtype)
    got = _block_operator((A_i, b_i), (A_j, b_j))
    want = _reference((A_i, b_i), (A_j, b_j))
    for left, right in zip(got, want):
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-6)


def test_scalar_block_operator_is_associative():
    dtype = jnp.complex128
    values = []
    for scale in (1.0, 0.8, 0.6):
        values.append((jnp.asarray([[[scale + 0.1j, 0.2],
                                    [-0.1j, scale - 0.05j]]], dtype=dtype),
                       jnp.asarray([[[0.2 + scale * 0.1j],
                                     [0.3 - scale * 0.1j]]], dtype=dtype)))
    left = _block_operator(_block_operator(values[0], values[1]), values[2])
    right = _block_operator(values[0], _block_operator(values[1], values[2]))
    for left_value, right_value in zip(left, right):
        np.testing.assert_allclose(left_value, right_value, rtol=1e-10,
                                   atol=1e-10)


def test_scalar_block_operator_gradients_match_matrix_reference():
    dtype = jnp.complex128
    A_i = jnp.asarray([[[1.0 + 0.1j, 0.2 - 0.1j],
                        [-0.1 + 0.2j, 0.8 - 0.05j]]], dtype=dtype)
    A_j = jnp.asarray([[[0.7 - 0.2j, -0.3 + 0.1j],
                        [0.4 + 0.2j, 0.6 - 0.1j]]], dtype=dtype)
    b_i = jnp.asarray([[0.2 + 0.1j, 0.4 - 0.2j]], dtype=dtype)
    b_j = jnp.asarray([[0.1 - 0.3j, -0.2 + 0.4j]], dtype=dtype)

    def scalar_operator(a_i, a_j, c_i, c_j):
        A_out, b_out = _block_operator((a_i, c_i), (a_j, c_j))
        return jnp.real(A_out).sum() + jnp.real(b_out).sum()

    def reference_operator(a_i, a_j, c_i, c_j):
        A_out, b_out = _reference((a_i, c_i), (a_j, c_j))
        return jnp.real(A_out).sum() + jnp.real(b_out).sum()

    got = jax.grad(scalar_operator, argnums=(0, 1, 2, 3))(
        A_i, A_j, b_i, b_j)
    want = jax.grad(reference_operator, argnums=(0, 1, 2, 3))(
        A_i, A_j, b_i, b_j)
    for left, right in zip(got, want):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)
