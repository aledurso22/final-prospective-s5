import json

import numpy as np
import pytest

from experiments.s5_three_arm_full.data import validate_official_raw_cache


def _write_cache(path, shape=(2, 16000, 1), representation="raw_waveform"):
    path.mkdir()
    values = np.zeros(shape, dtype=np.float32)
    labels = np.zeros((shape[0],), dtype=np.int64)
    np.save(path / "train_x.npy", values)
    np.save(path / "train_y.npy", labels)
    np.save(path / "val_x.npy", values)
    np.save(path / "val_y.npy", labels)
    manifest = {
        "split_definition": "official", "representation": representation,
        "shape": [16000, 1], "counts": {"train": 2, "val": 2},
        "official_validation_list_sha256": "validation",
        "official_testing_list_sha256": "testing",
        "feature_sha256": {},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_official_cache_requires_raw_shape_and_list_provenance(tmp_path):
    cache = tmp_path / "sc10_official_cache"
    _write_cache(cache)
    manifest = validate_official_raw_cache(cache)
    assert manifest["representation"] == "raw_waveform"

    bad = tmp_path / "bad"
    _write_cache(bad, shape=(2, 40, 13), representation="mfcc")
    with pytest.raises(RuntimeError):
        validate_official_raw_cache(bad)


def test_named_mfcc_random_split_cache_is_rejected(tmp_path):
    cache = tmp_path / "sc10_cache"
    _write_cache(cache)
    with pytest.raises(RuntimeError, match="MFCC/random-split"):
        validate_official_raw_cache(cache)
