"""Local CPU cost of one production-width layer, forward + backward, float32.

Relative numbers on this machine only; the cluster benchmark is the reference.

    JAX_PLATFORMS=cpu python -m experiments.s5_predictor_corrector.benchmark_local
"""
import jax

from experiments.s5_three_arm_full import runner as R
from experiments.s5_zero_drag.benchmark_local import BATCH, L, bench
from s5 import predictor_corrector as PC
from s5.generalized_prospective_ssm import GeneralizedProspectiveS5SSM
from s5.ssm import S5SSM


def main():
    kw = R.ssm_kwargs("generalized_prospective_s5")
    x = jax.random.normal(jax.random.PRNGKey(1), (BATCH, L, kw["H"]))
    arms = {"native S5": S5SSM(**kw),
            "generalized (production), factored":
                GeneralizedProspectiveS5SSM(**kw, implementation="factored")}
    for impl in ("factored", "companion", "sequential"):
        for v, (scale, corr) in PC.VARIANTS.items():
            arms[f"{v} ({scale}, {corr}), {impl}"] = PC.PredictorCorrectorS5SSM(
                **kw, scale=scale, corrector=corr, implementation=impl)
    base = None
    print(f"L={L} batch={BATCH} H={kw['H']} P={kw['P']} bidirectional")
    for name, mod in arms.items():
        compile_s, run_s = bench(mod, x)
        base = base or run_s
        print(f"{name:45s} compile {compile_s:6.2f}s  fwd+bwd {run_s*1e3:8.1f} ms"
              f"  ({run_s/base:5.1f}x native)")


if __name__ == "__main__":
    main()
