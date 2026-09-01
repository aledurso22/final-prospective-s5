"""Shared helper that builds a real (small) S5 model.

Mirrors the construction in ``s5/train.py`` exactly, so the tests exercise the
genuine S5SSM, its discretization and its associative scan -- not a stub.
"""

import jax.numpy as np
from jax.scipy.linalg import block_diag

from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO


def make_ssm_init_fn(d_model=8, ssm_size=8, blocks=2, conj_sym=True,
                     bidirectional=False, C_init="trunc_standard_normal",
                     discretization="zoh", dt_min=0.001, dt_max=0.1,
                     clip_eigs=False):
    """Build an ``init_S5SSM`` partial the same way ``s5/train.py`` does."""
    block_size = int(ssm_size / blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)

    if conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))

    ssm_init_fn = init_S5SSM(
        H=d_model, P=ssm_size,
        Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
        V=V, Vinv=Vinv,
        C_init=C_init, discretization=discretization,
        dt_min=dt_min, dt_max=dt_max,
        conj_sym=conj_sym, clip_eigs=clip_eigs,
        bidirectional=bidirectional,
    )
    return ssm_init_fn, ssm_size
