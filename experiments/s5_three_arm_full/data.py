"""Official-list Speech Commands cache for the three-arm experiment."""

import hashlib
import json
import os

import numpy as np

from dataloaders import speech_commands10 as SC


def _digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _read_list(root, name):
    with open(os.path.join(root, name)) as handle:
        return {line.strip() for line in handle if line.strip()}


def prepare_official(root, cache_dir):
    """Prepare MFCCs using Speech Commands' official validation/test lists."""
    transform = SC._mfcc_transform()
    torch, torchaudio, mfcc = transform
    validation = _read_list(root, "validation_list.txt")
    testing = _read_list(root, "testing_list.txt")
    overlap = validation & testing
    if overlap:
        raise ValueError(f"official validation/testing overlap: {len(overlap)}")
    splits = {"train": [], "val": [], "test": []}
    for label, word in enumerate(SC.WORDS):
        directory = os.path.join(root, word)
        files = sorted(f for f in os.listdir(directory) if f.endswith(".wav"))
        for filename in files:
            relative = f"{word}/{filename}"
            split = "val" if relative in validation else (
                "test" if relative in testing else "train")
            splits[split].append((relative, label))
    os.makedirs(cache_dir, exist_ok=True)
    raw = {}
    manifest = {"words": list(SC.WORDS), "split_definition": "official",
                "official_validation_list_sha256": _digest_bytes(
                    "\n".join(sorted(validation)).encode()),
                "official_testing_list_sha256": _digest_bytes(
                    "\n".join(sorted(testing)).encode()),
                "mfcc": {"n_mfcc": SC.N_MFCC, "n_fft": SC.N_FFT,
                         "hop_length": SC.HOP, "n_mels": SC.N_MELS,
                         "sample_rate": SC.SAMPLE_RATE}, "counts": {},
                "file_list_sha256": {}, "feature_sha256": {}}
    for split, items in splits.items():
        manifest["counts"][split] = len(items)
        manifest["file_list_sha256"][split] = _digest_bytes(
            "\n".join(f"{path}\t{label}" for path, label in items).encode())
        features = np.zeros((len(items), SC.N_FRAMES, SC.N_MFCC), dtype=np.float32)
        labels = np.zeros((len(items),), dtype=np.int32)
        for index, (relative, label) in enumerate(items):
            waveform = SC._load_wav(torch, torchaudio, os.path.join(root, relative))
            value = mfcc(waveform).numpy().T
            if value.shape != (SC.N_FRAMES, SC.N_MFCC):
                raise ValueError(f"{relative}: unexpected MFCC shape {value.shape}")
            features[index], labels[index] = value, label
        raw[split] = features, labels
    mean = raw["train"][0].reshape(-1, SC.N_MFCC).mean(0)
    std = raw["train"][0].reshape(-1, SC.N_MFCC).std(0)
    std = np.where(std < 1e-8, 1.0, std)
    manifest["standardization"] = {"mean": mean.tolist(), "std": std.tolist()}
    for split, (features, labels) in raw.items():
        features = ((features - mean) / std).astype(np.float32)
        np.save(os.path.join(cache_dir, f"{split}_x.npy"), features)
        np.save(os.path.join(cache_dir, f"{split}_y.npy"), labels)
        manifest["feature_sha256"][split] = {
            "x": SC._sha256_file(os.path.join(cache_dir, f"{split}_x.npy")),
            "y": SC._sha256_file(os.path.join(cache_dir, f"{split}_y.npy"))}
    manifest["torchaudio_version"] = torchaudio.__version__
    with open(os.path.join(cache_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def load_official(cache_dir, splits):
    with open(os.path.join(cache_dir, "manifest.json")) as handle:
        manifest = json.load(handle)
    if manifest.get("split_definition") != "official":
        raise RuntimeError("refusing non-official Speech Commands cache")
    loaded = {}
    for split in splits:
        x_path = os.path.join(cache_dir, f"{split}_x.npy")
        y_path = os.path.join(cache_dir, f"{split}_y.npy")
        got = {"x": SC._sha256_file(x_path), "y": SC._sha256_file(y_path)}
        if got != manifest["feature_sha256"][split]:
            raise RuntimeError(f"official cache digest mismatch for {split}")
        loaded[split] = (np.load(x_path, mmap_mode="r"), np.load(y_path))
    return loaded, manifest
