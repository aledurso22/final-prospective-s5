"""Speech Commands v0.02, 10-word subset, MFCC features, per Rawat E.4.

Verbatim from Appendix E.4 (arXiv:2609.04134v1):

    "All Speech Commands runs use the 10-word subset of v0.02 with a stratified
    70/15/15 split (split seed 0) and feature-wise standardization from the
    training split. MFCC extraction uses 20 coefficients, a 200-sample FFT
    window, 64 mel bands, and a 100-sample hop, producing 161 frames. ...
    Neither Speech Commands nor Path-X uses data augmentation."

Deliberately SEPARATE from `s5/dataloading.py`, which builds the 35-class task.

Resolved details and declared differences (also in docs/RAWAT_BASELINE_MAP.md):

* The paper does not name the MFCC implementation. We use `torchaudio`'s MFCC
  with `n_fft=200, hop_length=100, n_mels=64, n_mfcc=20` because the stated
  parameters are torchaudio's argument names. If torchaudio is absent we RAISE;
  substituting a different MFCC silently would be an undeclared difference.
* The paper does not give the stratification algorithm. We shuffle each class's
  file list with `numpy.random.RandomState(0)` and cut 70/15/15. DECLARED.
* CAVEAT, recorded because it affects the absolute numbers: a stratified random
  file split is NOT speaker-disjoint, unlike the official
  validation_list/testing_list of Speech Commands. The same speaker can appear
  in train and test, which inflates accuracy relative to the official protocol.
  We reproduce the published protocol as stated; every arm shares the split, so
  the comparison is unaffected, but the absolute value is not comparable to
  official-split literature.

Everything is cached to .npy with a manifest of filenames and SHA-256 digests
so a cluster run can be audited and resumed.
"""

import hashlib
import json
import os

import numpy as onp

#: the canonical 10-word subset
WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
SAMPLE_RATE = 16000
N_MFCC = 20
N_FFT = 200
HOP = 100
N_MELS = 64
N_FRAMES = 161
SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def build_splits(root, seed=SPLIT_SEED, fractions=SPLIT_FRACTIONS):
    """Stratified 70/15/15 file split. Deterministic in `seed` alone."""
    splits = {"train": [], "val": [], "test": []}
    for label, word in enumerate(WORDS):
        d = os.path.join(root, word)
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"expected {d}. Point --data_root at the extracted "
                f"speech_commands_v0.02 directory.")
        files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        idx = onp.random.RandomState(seed).permutation(len(files))
        n = len(files)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        for name, sl in (("train", idx[:n_tr]), ("val", idx[n_tr:n_tr + n_va]),
                         ("test", idx[n_tr + n_va:])):
            splits[name] += [(os.path.join(word, files[i]), label) for i in sl]
    for k in splits:
        splits[k].sort()
    return splits


def _mfcc_transform():
    try:
        import torch
        import torchaudio
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError(
            "torchaudio is required for the published MFCC front end "
            "(20 coefficients, 200-sample FFT window, 64 mel bands, 100-sample "
            "hop). Install it in the cluster environment; do NOT substitute a "
            "different MFCC implementation, which would be an undeclared "
            "difference from the reference recipe.") from exc
    tf = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
        melkwargs=dict(n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS))
    return torch, torchaudio, tf


def _load_wav(torch, torchaudio, path):
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    wav = wav.mean(0)                                     # mono
    if wav.shape[0] < SAMPLE_RATE:                        # pad short clips
        wav = torch.nn.functional.pad(wav, (0, SAMPLE_RATE - wav.shape[0]))
    return wav[:SAMPLE_RATE]


def prepare(root, cache_dir, seed=SPLIT_SEED, limit=None):
    """Extract features once, write .npy plus an auditable manifest."""
    torch, torchaudio, tf = _mfcc_transform()
    os.makedirs(cache_dir, exist_ok=True)
    splits = build_splits(root, seed)
    manifest = dict(words=list(WORDS), split_seed=seed,
                    fractions=list(SPLIT_FRACTIONS),
                    mfcc=dict(n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP,
                              n_mels=N_MELS, sample_rate=SAMPLE_RATE),
                    torchaudio_version=torchaudio.__version__,
                    counts={}, file_list_sha256={}, feature_sha256={})

    raw = {}
    for name, items in splits.items():
        if limit:
            items = items[:limit]
        manifest["counts"][name] = len(items)
        manifest["file_list_sha256"][name] = _sha256_bytes(
            "\n".join(f"{p}\t{y}" for p, y in items).encode())
        feats = onp.zeros((len(items), N_FRAMES, N_MFCC), dtype=onp.float32)
        labels = onp.zeros((len(items),), dtype=onp.int32)
        for i, (rel, y) in enumerate(items):
            wav = _load_wav(torch, torchaudio, os.path.join(root, rel))
            m = tf(wav).numpy().T                          # (frames, n_mfcc)
            if m.shape[0] != N_FRAMES:
                raise ValueError(f"{rel}: got {m.shape[0]} frames, expected "
                                 f"{N_FRAMES}")
            feats[i] = m
            labels[i] = y
        raw[name] = (feats, labels)

    # feature-wise standardization from the TRAINING split only
    mu = raw["train"][0].reshape(-1, N_MFCC).mean(0)
    sd = raw["train"][0].reshape(-1, N_MFCC).std(0)
    sd = onp.where(sd < 1e-8, 1.0, sd)
    manifest["standardization"] = dict(mean=mu.tolist(), std=sd.tolist())

    for name, (feats, labels) in raw.items():
        feats = ((feats - mu) / sd).astype(onp.float32)
        fp = os.path.join(cache_dir, f"{name}_x.npy")
        lp = os.path.join(cache_dir, f"{name}_y.npy")
        onp.save(fp, feats)
        onp.save(lp, labels)
        manifest["feature_sha256"][name] = dict(x=_sha256_file(fp),
                                                y=_sha256_file(lp))
    with open(os.path.join(cache_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load(cache_dir):
    """Load cached arrays and verify their digests against the manifest."""
    with open(os.path.join(cache_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    out = {}
    for name in ("train", "val", "test"):
        fp = os.path.join(cache_dir, f"{name}_x.npy")
        lp = os.path.join(cache_dir, f"{name}_y.npy")
        got = dict(x=_sha256_file(fp), y=_sha256_file(lp))
        want = manifest["feature_sha256"][name]
        if got != want:
            raise RuntimeError(
                f"{name} split digest mismatch: cached data does not match the "
                f"manifest. Re-run prepare(); do not train on unverified data.\n"
                f"  expected {want}\n  got      {got}")
        out[name] = (onp.load(fp, mmap_mode="r"), onp.load(lp))
    return out, manifest


def epoch_batches(n, batch_size, seed, epoch, drop_last=True):
    """Deterministic shuffle for (seed, epoch). Resumable without RNG state:
    the permutation is a pure function of the pair, so restarting an epoch
    reproduces it exactly."""
    perm = onp.random.RandomState((seed * 1_000_003 + epoch) % (2 ** 31 - 1)) \
        .permutation(n)
    end = (n // batch_size) * batch_size if drop_last else n
    for i in range(0, end, batch_size):
        yield perm[i:i + batch_size]
