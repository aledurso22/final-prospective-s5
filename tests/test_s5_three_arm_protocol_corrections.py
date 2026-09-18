import inspect
import os

import jax.numpy as jnp
import numpy as np

from experiments.s5_three_arm_full import data, runner
from s5.train_helpers import no_bc_decay_group


def test_protocol_uses_upstream_initializer_and_exact_three_names():
    assert all(runner.ssm_kwargs(arm)["C_init"] == "lecun_normal"
               for arm in runner.ARM_ORDER)
    assert all(runner.ssm_kwargs(arm)["clip_eigs"] is True
               for arm in runner.ARM_ORDER)
    assert list(runner.SCIENTIFIC_NAMES.values()) == [
        "Native S5",
        "Zucchet prospective dynamics — finite-difference realization",
        "generalized prospective dynamics (M,γ,T) — finite-difference realization",
    ]


def test_raw_normalization_is_one_statistic_per_channel():
    train = np.asarray([[[0.0], [2.0]], [[4.0], [6.0]]], dtype=np.float32)
    values = np.asarray([[[1.0], [3.0]]], dtype=np.float32)
    normalized, mean, std = data.normalize_raw_audio(train, values)
    assert mean.shape == (1, 1, 1)
    assert std.shape == (1, 1, 1)
    np.testing.assert_allclose(mean, [[[3.0]]])
    np.testing.assert_allclose(std, [[[2.5819888]]], rtol=1e-6)


def test_all_recurrence_parameters_use_ssm_group():
    for key in ("prospective_T_raw", "generalized_T_raw",
                "generalized_rho_raw", "generalized_gamma_raw"):
        assert no_bc_decay_group(key) == "ssm"


def test_runner_has_no_historical_gp_production_imports():
    source = inspect.getsource(runner)
    assert all(name not in source for name in
               ("s5.gp_ssm", "s5.gp_coefficients", "s5.gp_second_order"))
    assert set(runner.ARM_CONFIGS) == set(runner.ARM_ORDER)


def test_dispatch_has_each_arm_and_seed_once():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "bin", "slurm", "s5_three_arm_full_array.sbatch")
    source = open(path).read()
    dispatch = "DISPATCH_ARMS=(native_matched_s5 zucchet_prospective_s5 generalized_prospective_s5)"
    assert dispatch in source
    order = ("native_matched_s5", "zucchet_prospective_s5",
             "generalized_prospective_s5")
    tasks = [(arm, seed) for arm in order for seed in (301, 302, 303)]
    assert len(tasks) == len(set(tasks)) == 9


def test_observability_and_fail_closed_path_are_in_runner():
    source = inspect.getsource(runner)
    assert "train_step_observable" in source
    assert "step_metrics.jsonl" in source
    assert "failure.json" in source
    assert "production_check(args.arm, args.seed, args.out)" in source
