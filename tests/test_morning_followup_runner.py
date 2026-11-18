"""Fresh paired-seed entry point: unchanged dynamics and strict GPU smoke gate."""
import copy
import json

import jax
import numpy as np
import pytest

from experiments.s5_morning_followup import runner as F
from experiments.s5_three_arm_full import runner as R
from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_history import runner as P


@pytest.fixture(autouse=True)
def float32_only():
    assert not jax.config.read("jax_enable_x64")


@pytest.fixture
def tiny(monkeypatch):
    for key, value in {"D_MODEL":4,"SSM_SIZE_BASE":8,"BLOCKS":1,
                       "N_LAYERS":1,"SEQ_LEN":16,"BATCH_SIZE":2}.items():
        monkeypatch.setattr(R,key,value)


def test_registration_restored_on_failure():
    old = (R.SEEDS,R.GENERALIZED_IMPLEMENTATION,R.RUNNABLE_ARMS,H.ARM,H.ALPHA)
    with pytest.raises(ValueError,match="intentional"):
        with F.configured(P.ARM):
            assert 304 in R.SEEDS and 305 in R.SEEDS
            assert R.GENERALIZED_IMPLEMENTATION == "factored"
            assert P.ARM in R.RUNNABLE_ARMS
            raise ValueError("intentional")
    assert old == (R.SEEDS,R.GENERALIZED_IMPLEMENTATION,R.RUNNABLE_ARMS,H.ARM,H.ALPHA)


@pytest.mark.parametrize("seed",(304,305))
def test_fresh_seeds_share_parameters_and_optimizer(tiny,seed):
    result=F.matched_initialization(seed)
    assert result["native_proximal_parameters_identical"]
    assert result["native_proximal_optimizer_identical"]
    assert result["native_corrector_shared_parameters_identical"]


@pytest.mark.parametrize("arm",F.ARMS)
def test_actual_new_seed_smoke_uses_original_runner(tiny,tmp_path,monkeypatch,arm):
    rng=np.random.default_rng(7)
    data={"train":(rng.normal(size=(8,16,1)).astype(np.float32),np.arange(8)%10),
          "val":(rng.normal(size=(4,16,1)).astype(np.float32),np.arange(4)%10)}
    cache=tmp_path/"data";cache.mkdir();(cache/"manifest.json").write_text("{}")
    loads=[]
    def load(path,splits):
        loads.append(splits)
        return data,{}
    monkeypatch.setattr(R.EXPERIMENT_DATA,"load_official_raw",load)
    smoke=tmp_path/"smoke"
    F.main(["--arm",arm,"--seed","304","--out",str(smoke),
            "--data-cache",str(cache),"--smoke"])
    report=json.loads((smoke/"smoke_result.json").read_text())
    production=json.loads((smoke/"production_check.json").read_text())
    assert report["seed"]==304 and report["training_steps"]==2
    assert report["epochs_requested"]==15 and report["checkpoint_restored"]
    assert report["code_identifier"]==arm
    assert report["backend"]=="cpu" and loads==[("train","val")]
    expected_scan="history_stationary_block" if arm==P.ARM else "factored" if arm==F.CORRECTOR else None
    assert report["scan_implementation"]==expected_scan
    with F.configured(arm):
        identity=F.protocol_identity(cache,304,arm)
        assert report["protocol_identity"]==identity
        with pytest.raises(ValueError,match="GPU smoke"):
            F.check_smoke_gate(report,production,identity)
        # Exercise all remaining gate checks with a declared GPU report.
        pretend=copy.deepcopy(report);pretend["backend"]="gpu"
        F.check_smoke_gate(pretend,production,identity)
        for field,value in (("seed",305),("training_steps",1),("scan_implementation","wrong")):
            wrong=copy.deepcopy(pretend);wrong[field]=value
            with pytest.raises(ValueError):F.check_smoke_gate(wrong,production,identity)
        wrong=copy.deepcopy(production);wrong["x64_enabled"]=True
        with pytest.raises(ValueError,match="float32"):
            F.check_smoke_gate(pretend,wrong,identity)
        wrong=copy.deepcopy(production);wrong["initial_poles"]["production_precision"]["unstable"]=1
        with pytest.raises(ValueError,match="unstable"):
            F.check_smoke_gate(pretend,wrong,identity)
    with pytest.raises(SystemExit):
        F.main(["--arm",arm,"--seed","304","--out",str(smoke),
                "--data-cache",str(cache),"--smoke"])
    assert R.SEEDS==(301,302,303) and P.ARM not in R.RUNNABLE_ARMS


def test_identity_changes_with_arm_horizon_seed_and_sources(tmp_path):
    (tmp_path/"manifest.json").write_text("{}")
    identities=[]
    for arm,seed,alpha in ((P.ARM,304,1),(P.ARM,305,1),(P.ARM,304,.1),(F.NATIVE,304,1),(F.CORRECTOR,304,1)):
        with F.configured(arm,alpha):
            identities.append(F.protocol_identity(tmp_path,seed,arm,alpha))
    assert len({json.dumps(i,sort_keys=True) for i in identities})==5
    for identity in identities:
        assert identity["epochs"]==15 and identity["batch_size"]==16
        assert "experiments/s5_morning_followup/runner.py" in identity["source_sha256"]


def test_profile_supports_new_seed_without_data(tiny,tmp_path):
    path=tmp_path/"profile.json"
    F.main(["--arm",P.ARM,"--seed","305","--out",str(path),"--profile"])
    report=json.loads(path.read_text())
    assert report["arm"]==P.ARM and report["history_alpha"]==1
    assert report["training_executed"] is False and report["data_read"] is False
    assert R.SEEDS==(301,302,303) and P.ARM not in R.RUNNABLE_ARMS
