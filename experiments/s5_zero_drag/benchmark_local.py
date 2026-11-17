"""Local CPU cost of one SSM layer, forward + backward, float32.

Production width (H=96, P=64, bidirectional), shorter sequence. A relative
measurement on this machine only; the cluster benchmark is the reference.

    JAX_PLATFORMS=cpu python -m experiments.s5_zero_drag.benchmark_local
"""
import time

import jax
import jax.numpy as np

from experiments.s5_three_arm_full import runner as R
from s5 import independent_inertia as II
from s5.generalized_prospective_ssm import GeneralizedProspectiveS5SSM
from s5.ssm import S5SSM

L, BATCH, REPEATS = 4000, 2, 5


def bench(module, x):
    params = jax.vmap(module.init, in_axes=(None, 0))(
        jax.random.PRNGKey(0), x)
    params = jax.tree_util.tree_map(lambda v: v[0], params)
    f = jax.jit(jax.grad(lambda p: np.sum(
        jax.vmap(lambda xb: module.apply(p, xb))(x) ** 2)))
    t0 = time.perf_counter(); jax.block_until_ready(f(params)); compile_s = time.perf_counter() - t0
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); jax.block_until_ready(f(params)); times.append(time.perf_counter() - t0)
    return compile_s, sorted(times)[len(times) // 2]


def exact_setup_cost():
    lam = np.exp((-0.5 + 1j * np.linspace(-20, 0, 64)) * 0.01).astype(np.complex64)
    b = np.ones((64, 96), np.complex64)
    full = lambda v: np.full(64, v, np.float32)
    f = jax.jit(jax.grad(lambda m: np.sum(np.abs(np.stack(
        II.exact_zoh_coefficients(lam, b, full(5.0), m, full(0.0))[:2])) ** 2)))
    jax.block_until_ready(f(full(0.1)))
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        jax.block_until_ready(f(full(0.1)))
    return (time.perf_counter() - t0) / REPEATS


def main():
    kw = R.ssm_kwargs("generalized_prospective_s5")
    x = jax.random.normal(jax.random.PRNGKey(1), (BATCH, L, kw["H"]))
    arms = {
        "native S5": S5SSM(**kw),
        "generalized, sequential (completed run's default)":
            GeneralizedProspectiveS5SSM(**kw, implementation="sequential"),
        "generalized, factored (results run)":
            GeneralizedProspectiveS5SSM(**kw, implementation="factored"),
        "independent M,gamma, factored, damped":
            II.IndependentInertiaS5SSM(**kw, implementation="factored"),
        "independent M,gamma, factored, gamma=0":
            II.IndependentInertiaS5SSM(**kw, zero_damping=True,
                                       implementation="factored"),
    }
    base = None
    for name, mod in arms.items():
        c, t = bench(mod, x)
        base = base or t
        print(f"{name:52s} compile {c:6.1f}s  step {t*1e3:8.1f} ms  x{t/base:5.2f} native")
    print(f"exact-ZOH coefficient setup (64 modes, 3x3 expm, fwd+bwd): {exact_setup_cost()*1e3:.2f} ms")


if __name__ == "__main__":
    print(f"L={L} batch={BATCH} H=96 P=64 bidirectional float32, {jax.devices()[0]}")
    main()
