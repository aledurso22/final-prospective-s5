"""Scoped proximal-state pilot integration, with production float32 semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from experiments.s5_proximal_history import runner as P
from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full import step_telemetry as ST

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def float32_only():
    assert not jax.config.read("jax_enable_x64")


@pytest.fixture
def tiny(monkeypatch):
    for name, value in {"D_MODEL":8,"SSM_SIZE_BASE":16,"BLOCKS":2,
                        "N_LAYERS":2,"SEQ_LEN":32,"BATCH_SIZE":2}.items():
        monkeypatch.setattr(R,name,value)


def test_scoped_restore_even_after_error():
    hs = {k:getattr(H,k) for k in ("ARM","ALPHA","NormalizedGradientHistoryS5SSM",
          "production_transitions","history_pole_summary","protocol_identity","registered_history")}
    rs = {k:getattr(R,k) for k in ("RUNNABLE_ARMS","ssm_factory","radius_array",
          "pole_summary","production_check","ARM_CONFIGS","SCIENTIFIC_NAMES")}
    old_st = ST.radius_array
    with pytest.raises(RuntimeError,match="intentional"):
        with P.configured(1):
            with H.registered_history():
                assert H.ARM == P.ARM and H.ALPHA == 1
                assert P.ARM in R.RUNNABLE_ARMS
                factory=R.ssm_factory(P.ARM)
                assert factory.keywords["proximal_ratio"] == .5
                assert factory.keywords["history_alpha"] == 1
                assert R.ARM_CONFIGS[P.ARM]["recurrence_constructor"] == "ProximalGradientHistoryS5SSM"
                raise RuntimeError("intentional")
    assert all(getattr(H,k) is v for k,v in hs.items())
    assert all(getattr(R,k) is v for k,v in rs.items())
    assert ST.radius_array is old_st


@pytest.mark.parametrize("alpha,ratio",[(0,.5),(-1,.5),(float("nan"),.5),(1,-1),(1,float("inf"))])
def test_config_refusals(alpha,ratio):
    with pytest.raises(ValueError):
        with P.configured(alpha,ratio):
            pytest.fail("invalid configuration entered")


def test_identity_binds_alpha_ratio_and_sources(tmp_path):
    (tmp_path/"manifest.json").write_text("{}")
    with P.configured(1,.5):
        first=H.protocol_identity(tmp_path,301)
    with P.configured(.1,.5):
        second=H.protocol_identity(tmp_path,301)
    with P.configured(1,1):
        third=H.protocol_identity(tmp_path,301)
    assert first["alpha"] == 1 and first["proximal_ratio"] == .5
    assert first["arm"] == P.ARM and first["batch_size"] == 16 and first["epochs"] == 15
    assert first["outer_optimizer"] == "unchanged"
    assert "s5/proximal_history.py" in first["source_sha256"]
    assert first != second and first != third
    for wrong in (second,third):
        with P.configured(1):
            with pytest.raises(ValueError,match="protocol"):
                H.check_smoke_gate({"protocol_identity":wrong},first)


def test_native_parameters_and_optimizer_are_identical(tiny):
    native=R.init_state("native_matched_s5",301)
    with P.configured(1):
        with H.registered_history():
            state=R.init_state(P.ARM,301)
    a,b=flatten_dict(native.params),flatten_dict(state.params)
    assert set(a)==set(b)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    ta,tb=jax.tree_util.tree_flatten(native.opt_state),jax.tree_util.tree_flatten(state.opt_state)
    assert ta[1]==tb[1]
    for x,y in zip(ta[0],tb[0]):np.testing.assert_array_equal(x,y)


def test_actual_cpu_smoke_updates_and_checkpoint(tiny,tmp_path):
    rng=np.random.default_rng(5)
    train=(rng.normal(size=(8,R.SEQ_LEN,1)).astype(np.float32),np.arange(8)%10)
    val=(rng.normal(size=(4,R.SEQ_LEN,1)).astype(np.float32),np.arange(4)%10)
    with P.configured(1):
        with H.registered_history():
            state=R.init_state(P.ARM,301)
            poles=R.pole_summary(state.params,P.ARM)
            assert poles["proximal_ratio"] == .5
            assert poles["n_modes"] == 16 and poles["production_precision"]["unstable"] == 0
            assert float(H.history_radius(state.params)) == pytest.approx(
                poles["production_precision"]["max_radius_production"],abs=2e-6)
            model=R.model_for(P.ARM,True)
            R.production_check(P.ARM,301,str(tmp_path),
                (jnp.asarray(train[0][:2]),jnp.asarray(train[1][:2])),state,model,epochs=15)
            smoke=R.full_path_smoke(P.ARM,301,{"train":train,"val":val},
                str(tmp_path),state,model,epochs=15)
            assert smoke["checkpoint_restored"] and smoke["telemetry_check"]["values_agree"]
            assert smoke["code_identifier"] == P.ARM
            assert smoke["training_steps"] == 2
            assert smoke["poles_after_smoke_steps"]["production_precision"]["unstable"] == 0
            assert smoke["poles_after_smoke_steps"]["proximal_ratio"] == .5


def test_launcher_syntax_and_dryrun(tmp_path):
    scripts=[ROOT/"bin/run_experiments"/f"{s}_s5_proximal_history.sh" for s in ("cluster","submit")]
    for path in scripts:subprocess.run(["bash","-n",str(path)],check=True)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    env={**os.environ,"EXPECTED_COMMIT":sha,"DRY_RUN":"1","STAGE":"train",
         "SLURM_JOB_ID":"987654","CUDA_VISIBLE_DEVICES":"GPU-example",
         "PROSPECTIVE_VENV":sys.prefix,
         "PROSPECTIVE_RUNS":str(tmp_path),"S5_THREE_ARM_DATA":str(tmp_path),
         "JAX_ENABLE_X64":"0"}
    for path in scripts:
        done=subprocess.run(["bash",str(path)],cwd=ROOT,env=env,capture_output=True,text=True)
        assert done.returncode == 0,done.stdout+done.stderr
        assert "alpha1/301 alpha1/302 alpha0p1/301" in done.stdout or "alpha=1 seed301" in done.stdout
    assert not (tmp_path/"s5-proximal-history").exists()
    done=subprocess.run(["bash",str(scripts[0])],cwd=ROOT,
        env={**env,"CUDA_VISIBLE_DEVICES":"0,1"},capture_output=True,text=True)
    assert done.returncode != 0 and "exactly one" in done.stderr


def test_main_smoke_writes_bound_protocol_and_preserves_old_runner(tiny,tmp_path,monkeypatch):
    rng=np.random.default_rng(6)
    data={"train":(rng.normal(size=(8,32,1)).astype(np.float32),np.arange(8)%10),
          "val":(rng.normal(size=(4,32,1)).astype(np.float32),np.arange(4)%10)}
    cache=tmp_path/"data";cache.mkdir();(cache/"manifest.json").write_text("{}")
    requested=[]
    def load(path,splits):
        requested.append(splits)
        return data,{}
    monkeypatch.setattr(R.EXPERIMENT_DATA,"load_official_raw",load)
    out=tmp_path/"smoke"
    P.main(["--alpha","1","--seed","301","--out",str(out),
            "--data-cache",str(cache),"--smoke"])
    smoke=json.loads((out/"smoke_result.json").read_text())
    assert smoke["code_identifier"] == P.ARM
    assert smoke["proximal_ratio"] == .5 and smoke["history_alpha"] == 1
    assert smoke["backend"] == "cpu" and smoke["checkpoint_restored"]
    assert smoke["protocol_identity"]["alpha"] == 1
    assert smoke["protocol_identity"]["proximal_ratio"] == .5
    assert requested == [("train","val")]
    assert H.ARM == "gradient_history_s5" and P.ARM not in R.RUNNABLE_ARMS


def test_compile_profile_is_abstract_and_restores_registration(tiny,tmp_path):
    from experiments.s5_gradient_history import memory_profile as MP
    old_arms=MP.ARMS
    out=tmp_path/"profile.json"
    P.memory_profile(1,.5,301,out)
    report=json.loads(out.read_text())
    assert report["arm"] == P.ARM and report["history_alpha"] == 1
    assert report["proximal_ratio"] == .5
    assert report["training_executed"] is False and report["data_read"] is False
    assert len(report["updates"]) == 2
    assert all(row["memory"]["buffer_accounting_total_bytes"]>0 for row in report["updates"])
    assert "s5/proximal_history.py" in report["source_sha256"]
    assert MP.ARMS is old_arms and P.ARM not in R.RUNNABLE_ARMS
