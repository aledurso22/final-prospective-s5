"""Build a tiny SC10-shaped cache for STRUCTURAL checks only.

Random features with the real shapes and manifest layout. It exercises the
loader contract, the digest verification and the runner entry point without
any Speech Commands download. Nothing produced from this data is evidence
about accuracy; the runner still refuses to train on a CPU backend without
`--allow_cpu`.
"""

import json
import os
import sys

import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders import speech_commands10 as SC            # noqa: E402


def build(cache_dir, n_train=64, n_val=32, n_test=32, seed=0):
    os.makedirs(cache_dir, exist_ok=True)
    rs = onp.random.RandomState(seed)
    manifest = dict(words=list(SC.WORDS), split_seed=seed,
                    fractions=list(SC.SPLIT_FRACTIONS),
                    mfcc=dict(n_mfcc=SC.N_MFCC, n_fft=SC.N_FFT,
                              hop_length=SC.HOP, n_mels=SC.N_MELS,
                              sample_rate=SC.SAMPLE_RATE),
                    torchaudio_version="SYNTHETIC",
                    synthetic=True, counts={}, file_list_sha256={},
                    feature_sha256={})
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        x = rs.randn(n, SC.N_FRAMES, SC.N_MFCC).astype(onp.float32)
        y = rs.randint(0, 10, size=(n,)).astype(onp.int32)
        fp = os.path.join(cache_dir, f"{name}_x.npy")
        lp = os.path.join(cache_dir, f"{name}_y.npy")
        onp.save(fp, x); onp.save(lp, y)
        manifest["counts"][name] = n
        manifest["file_list_sha256"][name] = "SYNTHETIC"
        manifest["feature_sha256"][name] = dict(x=SC._sha256_file(fp),
                                                y=SC._sha256_file(lp))
    manifest["standardization"] = dict(mean=[0.0] * SC.N_MFCC,
                                       std=[1.0] * SC.N_MFCC)
    with open(os.path.join(cache_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


if __name__ == "__main__":
    print(build(sys.argv[1])["counts"])
