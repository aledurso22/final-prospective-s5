"""Official-list raw-audio Speech Commands cache."""

import hashlib
import json
import os

import numpy as np

from dataloaders import speech_commands10 as SC

RAW_LENGTH = 16000


def _digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _read_list(root, name):
    with open(os.path.join(root, name)) as handle:
        return {line.strip() for line in handle if line.strip()}


def normalize_raw_audio(train, values):
    """Apply upstream S5 channel statistics over examples and time."""
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), ddof=1, keepdims=True)
    return ((values - mean) / (std + 1e-5)).astype(np.float32), mean, std


def prepare_official_raw(root, cache_dir):
    """Prepare upstream-style raw audio with official validation/test lists."""
    validation = _read_list(root, "validation_list.txt")
    testing = _read_list(root, "testing_list.txt")
    if validation & testing:
        raise ValueError("official validation/testing lists overlap")
    splits = {"train": [], "val": [], "test": []}
    for label, word in enumerate(SC.WORDS):
        directory = os.path.join(root, word)
        for filename in sorted(f for f in os.listdir(directory) if f.endswith(".wav")):
            relative = f"{word}/{filename}"
            split = "val" if relative in validation else (
                "test" if relative in testing else "train")
            splits[split].append((relative, label))

    os.makedirs(cache_dir, exist_ok=True)
    raw = {}
    manifest = {
        "words": list(SC.WORDS), "split_definition": "official",
        "representation": "raw_waveform", "shape": [RAW_LENGTH, 1],
        "normalization": "upstream_s5_normalize_all_data_train_statistics",
        "official_validation_list_sha256": _digest_bytes(
            "\n".join(sorted(validation)).encode()),
        "official_testing_list_sha256": _digest_bytes(
            "\n".join(sorted(testing)).encode()),
        "counts": {}, "file_list_sha256": {}, "feature_sha256": {},
    }
    for split, items in splits.items():
        manifest["counts"][split] = len(items)
        manifest["file_list_sha256"][split] = _digest_bytes(
            "\n".join(f"{path}\t{label}" for path, label in items).encode())
        values = np.zeros((len(items), RAW_LENGTH, 1), dtype=np.float32)
        labels = np.zeros((len(items),), dtype=np.int64)
        for index, (relative, label) in enumerate(items):
            audio = SC._decode_wav(os.path.join(root, relative))
            if audio.shape[0] < RAW_LENGTH:
                audio = np.pad(audio, (0, RAW_LENGTH - audio.shape[0]))
            values[index, :, 0] = audio[:RAW_LENGTH]
            labels[index] = label
        raw[split] = values, labels

    train = raw["train"][0]
    _, mean, std = normalize_raw_audio(train, train)
    manifest["normalization_mean_sha256"] = _digest_bytes(mean.tobytes())
    manifest["normalization_std_sha256"] = _digest_bytes(std.tobytes())
    for split, (values, labels) in raw.items():
        values, _, _ = normalize_raw_audio(train, values)
        np.save(os.path.join(cache_dir, f"{split}_x.npy"), values)
        np.save(os.path.join(cache_dir, f"{split}_y.npy"), labels)
        manifest["feature_sha256"][split] = {
            "x": SC._sha256_file(os.path.join(cache_dir, f"{split}_x.npy")),
            "y": SC._sha256_file(os.path.join(cache_dir, f"{split}_y.npy")),
        }
    with open(os.path.join(cache_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def validate_official_raw_cache(cache_dir, splits=("train", "val")):
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if os.path.basename(os.path.abspath(cache_dir)) == "sc10_cache":
        raise RuntimeError("refusing MFCC/random-split sc10_cache")
    if (manifest.get("split_definition") != "official"
            or manifest.get("representation") != "raw_waveform"
            or manifest.get("shape") != [RAW_LENGTH, 1]):
        raise RuntimeError("refusing non-official raw-audio cache")
    if not manifest.get("official_validation_list_sha256") or not manifest.get(
            "official_testing_list_sha256"):
        raise RuntimeError("official-list split provenance is missing")
    for split in splits:
        x_path = os.path.join(cache_dir, f"{split}_x.npy")
        y_path = os.path.join(cache_dir, f"{split}_y.npy")
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"missing official cache arrays for {split}")
        values = np.load(x_path, mmap_mode="r")
        labels = np.load(y_path, mmap_mode="r")
        if values.ndim != 3 or values.shape[1:] != (RAW_LENGTH, 1):
            raise RuntimeError(f"invalid raw waveform shape for {split}: {values.shape}")
        if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
            raise RuntimeError(f"invalid label array for {split}")
        if manifest.get("counts", {}).get(split) != int(values.shape[0]):
            raise RuntimeError(f"manifest count mismatch for {split}")
    return manifest


def load_official_raw(cache_dir, splits):
    manifest = validate_official_raw_cache(cache_dir, splits)
    loaded = {}
    for split in splits:
        x_path = os.path.join(cache_dir, f"{split}_x.npy")
        y_path = os.path.join(cache_dir, f"{split}_y.npy")
        got = {"x": SC._sha256_file(x_path), "y": SC._sha256_file(y_path)}
        if got != manifest["feature_sha256"][split]:
            raise RuntimeError(f"raw-audio cache digest mismatch for {split}")
        loaded[split] = (np.load(x_path, mmap_mode="r"), np.load(y_path))
    return loaded, manifest
