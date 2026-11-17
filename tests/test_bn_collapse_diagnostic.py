"""bn_collapse_diagnostic on a real (tiny) training run, float32, x64 off.

Trains generalized_prospective_s5 seed 302 for 3 epochs on 32 synthetic
examples with the production runner, lays it out exactly as the 2x2 launcher
does, then checks: bitwise reproduction of the recorded validation result,
recalibration mechanics, preservation of the run, and the output guards.

    python -m pytest tests/test_bn_collapse_diagnostic.py -q     (~3 min CPU)
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from flax import serialization

from dataloaders import speech_commands10 as SC
from experiments.s5_predictor_corrector import bn_collapse_diagnostic as BD
from experiments.s5_three_arm_full import runner as R

ARM, SEED = "generalized_prospective_s5", 302


@pytest.fixture(autouse=True)
def _require_float32():
    assert not jax.config.read("jax_enable_x64"), "run without JAX_ENABLE_X64"


def _cache(root):
    rng = onp.random.RandomState(0)
    os.makedirs(root)
    counts = {}
    for split, n in (("train", 32), ("val", 16)):
        onp.save(os.path.join(root, f"{split}_x.npy"),
                 rng.normal(size=(n, 16000, 1)).astype(onp.float32))
        onp.save(os.path.join(root, f"{split}_y.npy"), rng.randint(0, 10, n))
        counts[split] = n
    feat = {s: {"x": SC._sha256_file(os.path.join(root, f"{s}_x.npy")),
                "y": SC._sha256_file(os.path.join(root, f"{s}_y.npy"))} for s in counts}
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump({"split_definition": "official", "representation": "raw_waveform",
                   "shape": [16000, 1], "official_validation_list_sha256": "v",
                   "official_testing_list_sha256": "t", "counts": counts,
                   "feature_sha256": feat}, fh)
    return root


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("bn")
    cache = _cache(str(base / "sc10_official_cache"))
    run_root = base / "s5-timescale-2x2" / "20990101-000000"
    task = run_root / ARM / str(SEED)
    os.makedirs(task)
    with open(run_root / "run_metadata.txt", "w") as fh:
        fh.write(f"authoritative_commit=test\nscan_implementation=factored\n"
                 f"epochs_requested=15\nbatch_size=16\ndata_cache={cache}\n")
    R.GENERALIZED_IMPLEMENTATION = "factored"
    from experiments.s5_three_arm_full.data import load_official_raw
    c = load_official_raw(cache, ("train", "val"))[0]
    R.train_arm(ARM, SEED, {"train": c["train"], "val": c["val"]}, str(task), epochs=3)
    return base, cache, str(task)


@pytest.fixture(scope="module")
def report(run):
    base, cache, task = run
    run_root = os.path.dirname(os.path.dirname(task))
    before = _snapshot(run_root)
    rep = BD.main(["--task", task, "--data-cache", cache,
                   "--out", str(base / "diag"), "--probe-examples", "16"])
    return rep, base, before, _snapshot(run_root)


def _snapshot(root):
    """Every file under the run root: size, mtime and content digest."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[p] = (st.st_size, st.st_mtime_ns, BD.sha256(p))
    return out


def test_reproduces_recorded_validation_bitwise(report):
    rep = report[0]
    for e in ("1", "2", "3"):
        r = rep["epochs"][e]
        assert r["reproduction"]["bitwise_identical"], r["reproduction"]
        assert r["validation"]["saved_stats"]["accuracy"] == r["recorded_validation"]["accuracy"]
    assert rep["reproduce_epoch"] == 2


def test_run_is_preserved_and_output_is_separate(report, run):
    rep, base, before, after = report
    assert rep["preservation"] == {"checkpoints_unchanged": True,
                                   "validation_records_unchanged": True}
    # nothing under the run root was created, removed, rewritten or touched
    assert before == after and len(before) > 5
    out = base / "diag"
    assert (out / "bn_diagnostic.json").exists()
    assert len(os.listdir(out / "recalibrated_stats")) == 6


def test_recalibration_uses_training_data_and_both_dropout_modes(report):
    rep = report[0]
    assert rep["protocol"]["validation_used_for_statistics"] is False
    for e in ("1", "2", "3"):
        v = rep["epochs"][e]["validation"]
        assert v["recal_dropout_off"]["recalibration_examples"] == 32
        cmp = rep["epochs"][e]["stats_comparison"]["recal_dropout_on_vs_recal_dropout_off"]
        # dropout changes downstream statistics, so the two estimates differ
        assert any(abs(l["var_ratio"]["p50"] - 1) > 1e-4 for l in cmp.values())


def test_single_batch_recalibration_equals_batch_statistics(run, tmp_path):
    """With one recalibration batch, the population moments ARE that batch's
    moments, so eval-mode logits on that batch must equal the batch-statistics
    forward (dropout off). Checks the momentum-0 capture end to end."""
    base, cache, task = run
    BD.main(["--task", task, "--data-cache", cache, "--epochs", "2",
             "--reproduce-epoch", "2", "--out", str(tmp_path / "one"),
             "--recal-examples", "16", "--probe-examples", "16"])
    template = R.init_state(ARM, SEED)
    with open(os.path.join(task, "checkpoint_epoch_02.msgpack"), "rb") as fh:
        state = serialization.from_bytes(template, fh.read())
    with open(tmp_path / "one" / "recalibrated_stats" / "epoch_02_recal_dropout_off.msgpack", "rb") as fh:
        stats = serialization.msgpack_restore(fh.read())
    from experiments.s5_three_arm_full.data import load_official_raw
    x, y = load_official_raw(cache, ("train",))[0]["train"]
    idx = next(SC.epoch_batches(len(y), 16, SEED, 1, drop_last=True))
    xb = jnp.asarray(x[idx])
    ones = jnp.ones((16, R.SEQ_LEN))
    eval_logits = R.model_for(ARM, False).apply(
        {"params": state.params, "batch_stats": stats}, xb, ones)
    bs_logits, _ = R.model_cls_for(ARM)(training=True, dropout=0.0).apply(
        {"params": state.params, "batch_stats": state.batch_stats}, xb, ones,
        mutable=["batch_stats"])
    assert onp.allclose(eval_logits, bs_logits, atol=2e-3), float(
        jnp.max(jnp.abs(eval_logits - bs_logits)))


def test_guards(run, tmp_path):
    base, cache, task = run
    run_root = os.path.dirname(os.path.dirname(task))
    with pytest.raises(BD.DiagnosticError, match="inside the run directory"):
        BD.main(["--task", task, "--data-cache", cache, "--out", os.path.join(run_root, "x")])
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep").write_text("x")
    with pytest.raises(BD.DiagnosticError, match="non-empty"):
        BD.main(["--task", task, "--data-cache", cache, "--out", str(busy)])
    with pytest.raises(BD.DiagnosticError, match="missing checkpoint"):
        BD.main(["--task", task, "--data-cache", cache, "--epochs", "2", "4",
                 "--out", str(tmp_path / "o")])
    with pytest.raises(BD.DiagnosticError, match="not the run's"):
        BD.main(["--task", task, "--data-cache", str(tmp_path), "--out", str(tmp_path / "o2")])
