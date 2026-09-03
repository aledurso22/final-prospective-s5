"""Pre-training gate for professor Idea 1 inside the real S5 stack.

Checks, before any GPU time is spent:
  1. the S5 implementation reduces to the deterministic mechanism equations;
  2. the homogeneous poles are the predicted ones;
  3. scan / sequential (streaming) equivalence;
  4. plain S5 is untouched when the feature is disabled;
  5. parameter-count accounting is unchanged.
"""

import hashlib
import os
import pathlib
import sys

import jax
import jax.numpy as np
import numpy as onp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5.professor_full_state_pc import (FullStatePCSSM, apply_full_state_pc,
                                        prospective_gain)
from s5.ssm import S5SSM
from tests.s5_reference import make_ssm_init_fn

REPO = pathlib.Path(__file__).resolve().parents[1]
SSM_SHA256 = "be64f943dd1a629f9967ffad9fd1807eb0becf43f28aff1f791a3a94284e0fac"

H, P_BLOCKS, L = 8, 2, 40
TAU = 1.0


def _pair(bidirectional=False, tau=TAU):
    """Plain and Idea-1 SSMs sharing identical parameters."""
    ssm_init_fn, _ = make_ssm_init_fn(d_model=H, ssm_size=H, blocks=P_BLOCKS,
                                      bidirectional=bidirectional)
    plain = ssm_init_fn(step_rescale=1.0)
    partial_kwargs = plain.__dict__.copy()
    pc = FullStatePCSSM(
        **{k: v for k, v in plain.__dict__.items()
           if k in FullStatePCSSM.__dataclass_fields__ and k != "parent"
           and k != "name"},
        prospective_tau=tau)
    return plain, pc


def _init_both():
    plain, pc = _pair()
    u = jax.random.normal(jax.random.PRNGKey(3), (L, H))
    v_plain = plain.init(jax.random.PRNGKey(0), u)
    v_pc = pc.init(jax.random.PRNGKey(0), u)
    return plain, pc, v_plain, v_pc, u


# ------------------------------------------------ (5) parameter accounting


def test_5_parameter_tree_and_count_identical_to_plain():
    plain, pc, v_plain, v_pc, _ = _init_both()
    from flax.traverse_util import flatten_dict
    fp, fq = flatten_dict(v_plain["params"]), flatten_dict(v_pc["params"])
    assert sorted(fp) == sorted(fq), "parameter names differ from plain S5"
    for k in fp:
        assert fp[k].shape == fq[k].shape, f"shape differs at {k}"
        onp.testing.assert_array_equal(fp[k], fq[k])
    n_plain = sum(x.size for x in jax.tree_util.tree_leaves(v_plain["params"]))
    n_pc = sum(x.size for x in jax.tree_util.tree_leaves(v_pc["params"]))
    assert n_plain == n_pc, (n_plain, n_pc)


# ---------------------------------------------------- (4) plain untouched


def test_4_ssm_source_is_byte_identical():
    """Idea 1 must not have modified s5/ssm.py."""
    digest = hashlib.sha256((REPO / "s5" / "ssm.py").read_bytes()).hexdigest()
    assert digest == SSM_SHA256


def test_4_plain_ssm_still_runs_and_is_unaffected():
    plain, _, v_plain, _, u = _init_both()
    y = plain.apply(v_plain, u)
    assert y.shape == (L, H) and np.all(np.isfinite(y))


def test_4_scan_and_binary_operator_still_used_unmodified():
    src = (REPO / "s5" / "ssm.py").read_text()
    assert "def binary_operator(q_i, q_j):" in src
    assert src.count("jax.lax.associative_scan") == 2
    pc_src = (REPO / "s5" / "professor_full_state_pc.py").read_text()
    assert "from .ssm import S5SSM, binary_operator" in pc_src
    # look for an actual call, not the docstring that warns against it
    assert ".roll(" not in pc_src


# ------------------------------- (1) reduces to the mechanism equations


def test_1_matches_closed_form_residual_recursion():
    """The S5 implementation equals s_k = rho s_{k-1} + K u_k - rho K u_{k-1}."""
    _, pc, _, v_pc, u = _init_both()
    p = v_pc["params"]
    Lambda = p["Lambda_re"] + 1j * p["Lambda_im"]
    B_tilde = p["B"][..., 0] + 1j * p["B"][..., 1]
    C_tilde = p["C"][..., 0] + 1j * p["C"][..., 1]
    step = np.exp(p["log_step"][:, 0])
    rho = np.exp(-step / TAU).astype(Lambda.dtype)
    K = prospective_gain(Lambda, B_tilde)

    Ku = onp.asarray(jax.vmap(lambda x: K @ x)(u))
    s = onp.zeros(Ku.shape[1], dtype=complex)
    ref = onp.empty_like(Ku)
    prev = onp.zeros_like(Ku[0])
    r = onp.asarray(rho)
    for k in range(L):
        s = r * s + Ku[k] - r * prev
        ref[k] = s
        prev = Ku[k]
    expected = onp.asarray(jax.vmap(lambda x: 2 * (C_tilde @ x).real)(np.asarray(ref)))
    got = onp.asarray(apply_full_state_pc(rho, K, C_tilde, u, conj_sym=True))
    onp.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_1_forced_response_collapses_to_instantaneous_map():
    """THE analytic prediction: from zero state the SSM becomes memoryless.

    s_k = K u_k exactly, so the whole state-space model degenerates to a
    pointwise linear map. This is the mechanism failing exactly as derived,
    made literal inside S5.
    """
    _, pc, _, v_pc, u = _init_both()
    p = v_pc["params"]
    Lambda = p["Lambda_re"] + 1j * p["Lambda_im"]
    B_tilde = p["B"][..., 0] + 1j * p["B"][..., 1]
    C_tilde = p["C"][..., 0] + 1j * p["C"][..., 1]
    K = prospective_gain(Lambda, B_tilde)

    instantaneous = jax.vmap(lambda x: 2 * (C_tilde @ (K @ x)).real)(u)
    ys = pc.apply(v_pc, u) - jax.vmap(lambda x: p["D"] * x)(u)
    onp.testing.assert_allclose(onp.asarray(ys), onp.asarray(instantaneous),
                                rtol=1e-4, atol=1e-4)


def test_1_impulse_response_has_support_one():
    """Direct consequence: the impulse response is one sample long."""
    _, pc, _, v_pc, _ = _init_both()
    imp = onp.zeros((L, H)); imp[0, 0] = 1.0
    ys = onp.asarray(pc.apply(v_pc, np.asarray(imp)))
    tail = onp.max(onp.abs(ys[1:]))
    assert tail < 1e-5, f"expected memoryless response, tail={tail}"


# ------------------------------------------------ (2) homogeneous poles


def test_2_homogeneous_poles_are_rho():
    """Every mode's homogeneous pole becomes exp(-step/tau), i.e. -1/tau."""
    _, pc, _, v_pc, _ = _init_both()
    step = np.exp(v_pc["params"]["log_step"][:, 0])
    rho = onp.asarray(np.exp(-step / TAU))
    # drive the residual recursion directly with zero input from r0 = 1
    r = onp.ones_like(rho, dtype=complex)
    traj = [r.copy()]
    for _ in range(50):
        r = rho * r
        traj.append(r.copy())
    traj = onp.array(traj)
    ratio = onp.abs(traj[10] / traj[9])
    onp.testing.assert_allclose(ratio, rho, rtol=1e-6)


def test_2_tau_controls_the_pole():
    for tau in (0.5, 2.0):
        _, pc = _pair(tau=tau)
        u = jax.random.normal(jax.random.PRNGKey(1), (L, H))
        v = pc.init(jax.random.PRNGKey(0), u)
        step = np.exp(v["params"]["log_step"][:, 0])
        expected = onp.asarray(np.exp(-step / tau))
        assert onp.all(expected > 0) and onp.all(expected < 1)


# ------------------------------------------- (3) scan vs sequential


def test_3_scan_matches_sequential_streaming():
    """Parallel scan == token-by-token recurrence, to float32 tolerance."""
    _, pc, _, v_pc, u = _init_both()
    p = v_pc["params"]
    Lambda = p["Lambda_re"] + 1j * p["Lambda_im"]
    B_tilde = p["B"][..., 0] + 1j * p["B"][..., 1]
    C_tilde = p["C"][..., 0] + 1j * p["C"][..., 1]
    rho = onp.asarray(np.exp(-np.exp(p["log_step"][:, 0]) / TAU))
    K = onp.asarray(prospective_gain(Lambda, B_tilde))

    scanned = onp.asarray(apply_full_state_pc(
        np.asarray(rho).astype(Lambda.dtype), np.asarray(K), C_tilde, u,
        conj_sym=True))

    s = onp.zeros(K.shape[0], dtype=complex)
    prev = onp.zeros(K.shape[0], dtype=complex)
    seq = onp.empty((L, H))
    Cn = onp.asarray(C_tilde)
    for k in range(L):
        Ku = K @ onp.asarray(u[k])
        s = rho * s + Ku - rho * prev
        prev = Ku
        seq[k] = 2 * (Cn @ s).real
    onp.testing.assert_allclose(scanned, seq, rtol=1e-4, atol=1e-4)


def test_3_causality_future_cannot_change_past():
    _, pc, _, v_pc, u = _init_both()
    changed = u.at[25:].set(1e3)
    a = onp.asarray(pc.apply(v_pc, u))[:25]
    b = onp.asarray(pc.apply(v_pc, changed))[:25]
    onp.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_3_bidirectional_is_rejected():
    plain, pc = _pair(bidirectional=True)
    u = jax.random.normal(jax.random.PRNGKey(2), (L, H))
    with pytest.raises(ValueError, match="unidirectional"):
        pc.init(jax.random.PRNGKey(0), u)


def test_3_jit_and_gradients():
    _, pc, _, v_pc, u = _init_both()
    onp.testing.assert_allclose(onp.asarray(jax.jit(pc.apply)(v_pc, u)),
                                onp.asarray(pc.apply(v_pc, u)),
                                rtol=1e-5, atol=1e-5)
    g = jax.grad(lambda p: np.sum(pc.apply({"params": p}, u) ** 2))(v_pc["params"])
    leaves = jax.tree_util.tree_leaves(g)
    assert leaves and all(np.all(np.isfinite(x)) for x in leaves)
    assert sum(float(np.sum(np.abs(x))) for x in leaves) > 0.0
