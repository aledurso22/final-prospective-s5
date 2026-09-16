"""Independently pretrained source checkpoints for the replication: manifest,
checksums, save/restore and the arm-to-source plan.

Brief: PROSPECTIVE_MOMENTUM_INDEPENDENT_REPLICATION_2026_09_17.md.

This does NOT relax or repurpose the completed study's hard-coded validator
(`source.py`, the read-only meta-delta checkpoints). New sources are created
inside the run directory, listed in `sources/manifest.json` and
`sources/SHA256SUMS` (sha256sum -c format), verified by checksum every time
they are restored, and re-verified at finalization.
"""

import hashlib
import json
import os

import jax.numpy as jnp
import numpy as onp

from experiments.meta_delta import calibrate as CAL
from experiments.meta_delta import model as MM

from . import source as SRC

SOURCE_DEV = 500
SOURCE_FINAL = (501, 502, 503)
SOURCE_SEEDS = (SOURCE_DEV,) + SOURCE_FINAL
SOURCE_FAMILIES = ("momentum_delta", "gated_delta", "gp_two_sided",
                   "tss_eq17")
#: arm -> source family (candidate, native and gain share the Momentum source)
SOURCE_OF = {"prospective_momentum": "momentum_delta",
             "momentum_delta": "momentum_delta",
             "gain_momentum": "momentum_delta",
             "gated_delta": "gated_delta", "gp_two_sided": "gp_two_sided",
             "tss_eq17": "tss_eq17"}
#: fixed source recipe: the prior designated slot-B source-training recipe
SOURCE_SLOT = "B"
SOURCE_UPDATES = 200
SOURCE_LR = 0.01
SOURCE_DIRNAME = "sources"


class SourceRefusal(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_key(seed, family):
    return f"src{seed}_{family}"


def source_files(seed, family):
    k = source_key(seed, family)
    return f"{k}.msgpack", f"{k}_opt.msgpack"


def slot_coefficients(family):
    """The ORIGINAL slot-B definition (initial coefficients) for a family."""
    return CAL.configurations()["slots"][f"{family}/{SOURCE_SLOT}"]


def init_source(family, seed):
    """Fresh initialization with the original initializer and slot-B
    coefficients: common tensors come from the same seed across families."""
    return MM.init_params(family, seed, init_coeffs=slot_coefficients(family))


def source_plan(dev_rules, final_rules, selection=None):
    """Which source every continuation receives. Pure; no data."""
    plan = []
    for rule in dev_rules:
        for tag in ("A", "B"):
            plan.append(dict(stage="dev", rule=rule, config=tag,
                             seed=SOURCE_DEV,
                             source=source_key(SOURCE_DEV, SOURCE_OF[rule])))
    for rule in final_rules:
        for seed in SOURCE_FINAL:
            plan.append(dict(stage="final", rule=rule,
                             config=(selection or {}).get(rule),
                             seed=seed,
                             source=source_key(seed, SOURCE_OF[rule])))
    return plan


def sources_dir(out):
    return os.path.join(out, SOURCE_DIRNAME)


def write_manifest(out, manifest):
    """manifest.json (atomic) and SHA256SUMS over every source file."""
    d = sources_dir(out)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "manifest.json.partial")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    os.replace(tmp, os.path.join(d, "manifest.json"))
    lines = []
    for e in manifest["entries"]:
        lines.append(f"{e['sha256']}  {e['file']}")
        lines.append(f"{e['opt_sha256']}  {e['opt_file']}")
    tmp = os.path.join(d, "SHA256SUMS.partial")
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    os.replace(tmp, os.path.join(d, "SHA256SUMS"))


def baseline_hashes(manifest):
    out = {}
    for e in manifest["entries"]:
        out[e["file"]] = e["sha256"]
        out[e["opt_file"]] = e["opt_sha256"]
    return out


def rehash(out, keys):
    d = sources_dir(out)
    return {k: (sha256(os.path.join(d, k))
                if os.path.isfile(os.path.join(d, k)) else None)
            for k in keys}


def find_entry(manifest, seed, family):
    hits = [e for e in manifest["entries"]
            if e["source_seed"] == seed and e["family"] == family]
    if len(hits) != 1:
        raise SourceRefusal(f"{len(hits)} manifest entries for "
                            f"{source_key(seed, family)}; exactly one needed")
    return hits[0]


def restore_source(out, entry):
    """Restore a manifest source after verifying its checksum, leaf set,
    shapes, float32 dtype and finiteness. Nothing is substituted."""
    from flax import serialization
    path = os.path.join(sources_dir(out), entry["file"])
    if not os.path.isfile(path):
        raise SourceRefusal(f"missing source {path}")
    got = sha256(path)
    if got != entry["sha256"]:
        raise SourceRefusal(f"checksum mismatch for {entry['file']}: "
                            f"{got} != {entry['sha256']}")
    tmpl = {k: onp.asarray(v) for k, v in SRC.template(entry["family"]).items()}
    with open(path, "rb") as fh:
        raw = serialization.msgpack_restore(fh.read())
    if set(raw) != set(tmpl):
        raise SourceRefusal(f"{entry['file']}: leaf set {sorted(raw)} != "
                            f"{sorted(tmpl)}")
    p = {}
    for k, t in tmpl.items():
        v = onp.asarray(raw[k])
        if v.shape != t.shape or v.dtype != onp.float32:
            raise SourceRefusal(f"{entry['file']}/{k}: {v.shape} {v.dtype}")
        if not onp.all(onp.isfinite(v)):
            raise SourceRefusal(f"{entry['file']}/{k}: non-finite values")
        p[k] = jnp.asarray(v, dtype=jnp.float32)
    return p
