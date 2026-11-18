"""Block pole/impulse audit for P (3x3) and proximal inertial (2x2) arms.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest tests/test_block_audit.py -q
"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from flax.traverse_util import unflatten_dict

from experiments.s5_predictor_corrector.production_init import production_init_flat
from experiments.s5_proximal_inertia import block_audit as BA
from s5 import proximal_inertia as PI

HORIZON = 512


@pytest.fixture(autouse=True)
def float32_only():
    assert not jax.config.read("jax_enable_x64")


def _task(root, arm, seed=301, config=None, selected=12, flat=None, corrupt_selection=False):
    task = root / f"{arm}-dir" / str(seed)
    task.mkdir(parents=True)
    flat = production_init_flat("native_matched_s5", seed) if flat is None else flat
    state = {"step": 0, "params": unflatten_dict(flat), "opt_state": {}, "batch_stats": {}}
    for e in sorted({selected, 15}):
        (task / f"checkpoint_epoch_{e:02d}.msgpack").write_bytes(serialization.msgpack_serialize(state))
    rows = [{"epoch": e, "validation": {"accuracy": (3000 + (50 if e == selected else e)) / 3703,
                                        "cross_entropy": 0.2, "n": 3703}} for e in range(1, 16)]
    (task / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = {"code_identifier": arm, "seed": seed, "epochs_requested": 15,
              "selected_epoch": selected + (1 if corrupt_selection else 0),
              "validation": rows[selected - 1]["validation"], **(config or {})}
    (task / "task_result.json").write_text(json.dumps(result))
    return task


CONFIGS = {"proximal_inertial_s5": {"history_alpha": .1, "proximal_ratio": .5, "mass_ratio": 1.5},
           "proximal_inertial_control_s5": {"history_alpha": .1, "proximal_ratio": .5, "mass_ratio": 1.0},
           "proximal_history_s5": {"history_alpha": 1.0, "proximal_ratio": .5}}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    tasks = [_task(root, arm, config=c) for arm, c in CONFIGS.items()]
    missing = root / "unfinished" / "302"
    missing.mkdir(parents=True)
    rep = BA.main([*sum((["--task", str(t)] for t in tasks), []), "--task", str(missing),
                   "--out", str(root / "audit"), "--horizon", str(HORIZON), "--allow-missing"])
    return rep, root, tasks


def test_every_arm_audited_and_missing_recorded(report):
    rep, root, _ = report
    assert [t["arm"] for t in rep["tasks"]] == list(CONFIGS)
    assert len(rep["unavailable"]) == 1 and "task_result" in rep["unavailable"][0]["reason"]
    assert rep["sources_unchanged"] and rep["test_split_read"] is False
    assert (root / "audit" / "block_audit.json").exists() and (root / "audit" / "summary.txt").exists()
    for t in rep["tasks"]:
        assert set(t["stages"]) == {"reconstructed_init", "selected", "final"}
        assert t["selected_epoch"] == 12
        width = 3 if t["arm"] == "proximal_history_s5" else 2
        for s in t["stages"].values():
            assert s["production_precision_totals"]["unstable"] == 0
            assert s["production_precision_totals"]["nonfinite"] == 0
            assert all(l["block_width"] == width and l["n_modes"] == 64 for l in s["layers"])


def test_control_cancels_and_candidate_releases_the_inertial_mode(report):
    rep = report[0]
    by = {t["arm"]: t for t in rep["tasks"]}
    ctrl = by["proximal_inertial_control_s5"]["stages"]["reconstructed_init"]["layers"]
    cand = by["proximal_inertial_s5"]["stages"]["reconstructed_init"]["layers"]
    for l in ctrl:
        assert l["auxiliary_relative_residue"][0]["p100"] < 1e-9        # exact cancellation
        assert l["component_reconstruction_relative_error_max"] < 1e-8
    assert np.median([l["auxiliary_relative_residue"][0]["p50"] for l in cand]) > 1e-3
    for l in cand:   # alpha = 0.1 keeps the dominant pole near native
        d = l["dominant_decay_over_native_counterfactual"]
        assert 0.9 < d["p50"] < 1.1


def test_identical_weights_give_identical_stages(report):
    t = report[0]["tasks"][0]
    a, b = t["stages"]["reconstructed_init"], t["stages"]["selected"]
    for la, lb in zip(a["layers"], b["layers"]):
        assert la["poles_float64"] == lb["poles_float64"]
        assert la["directions"]["forward"]["summary"] == lb["directions"]["forward"]["summary"]


def test_impulse_matches_production_scan():
    flat = production_init_flat("native_matched_s5", 301)
    prefix = sorted(k[:-1] for k in flat if k[-1] == "Lambda_re")[0]
    group = {k[-1]: np.asarray(v) for k, v in flat.items() if k[:-1] == prefix}
    cfg = CONFIGS["proximal_inertial_s5"]
    kappa, _ = BA.kappa64(group)
    Ad, bd = BA.blocks64("proximal_inertial_s5", cfg, kappa)
    h = BA.impulse(Ad, bd, 300)
    Ad32, bd32 = BA.blocks32("proximal_inertial_s5", cfg, group)
    drive = jnp.zeros((300, 64), jnp.complex64).at[0].set(1)
    got = np.asarray(PI.scan_blocks(jnp.asarray(Ad32), jnp.asarray(bd32), drive))[..., 0]
    assert np.abs(got - h).max() <= 5e-5 * np.abs(h).max()


def test_refusals(tmp_path):
    bad = _task(tmp_path, "proximal_inertial_s5", config=CONFIGS["proximal_inertial_s5"],
                corrupt_selection=True)
    with pytest.raises(BA.AuditError, match="selected_epoch"):
        BA.main(["--task", str(bad), "--out", str(tmp_path / "o"), "--horizon", "64"])
    noconf = _task(tmp_path / "n", "proximal_inertial_s5", config={"history_alpha": .1})
    with pytest.raises(BA.AuditError, match="proximal_ratio"):
        BA.main(["--task", str(noconf), "--out", str(tmp_path / "o2"), "--horizon", "64"])
    busy = tmp_path / "busy"; busy.mkdir(); (busy / "x").write_text("1")
    with pytest.raises(BA.AuditError, match="non-empty"):
        BA.main(["--task", str(noconf), "--out", str(busy)])
