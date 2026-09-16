"""READ-ONLY restoration of the designated development checkpoints.

Source run: /Users/durso/s5-runs/meta-delta/20260916-222310 (commit 5dc4b07).
For every family the designated source is its DEVELOPMENT slot-B checkpoint
at seed 300. Nothing is chosen by looking at results: the file name is fixed
by the protocol, and it must agree with the saved metadata. A missing or
mismatched source is an explicit refusal, never permission to pick another.

Nothing in the source directory is written. Every source file is hashed at
restoration and re-hashed when the study finishes, including on failure.
"""

import hashlib
import json
import os

import jax.numpy as jnp
import numpy as onp

from experiments.meta_delta import calibrate as CAL
from experiments.meta_delta import model as MM

SOURCE_RUN_ID = "20260916-222310"
SOURCE_TAG, SOURCE_CONFIG, SOURCE_SEED = "dev", "B", 300
#: family whose checkpoint each continuation arm starts from
SOURCE_RULE = {"prospective_momentum": "momentum_delta",
               "momentum_delta": "momentum_delta",
               "gain_momentum": "momentum_delta",
               "gated_delta": "gated_delta", "gp_two_sided": "gp_two_sided",
               "tss_eq17": "tss_eq17"}
SOURCE_FAMILIES = ("momentum_delta", "gated_delta", "gp_two_sided", "tss_eq17")
#: declared reproduction tolerances (protocol s5)
REPRO_CE_REL = 1e-4
REPRO_QUERIES_PER_CATEGORY = 1


class SourceRefusal(RuntimeError):
    pass


def stem(rule):
    return f"{SOURCE_TAG}_{rule}_{SOURCE_CONFIG}_seed{SOURCE_SEED}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(source_dir):
    files = {"status.json": os.path.join(source_dir, "status.json"),
             "selection.json": os.path.join(source_dir, "selection.json")}
    for r in SOURCE_FAMILIES:
        files[r] = os.path.join(source_dir, "params", stem(r) + ".msgpack")
    return files


def hash_sources(source_dir):
    return {k: (sha256(v) if os.path.isfile(v) else None)
            for k, v in source_files(source_dir).items()}


def template(rule):
    """Shape/dtype template ONLY; its values are overwritten on restore."""
    slot = CAL.configurations()["slots"][f"{rule}/{SOURCE_CONFIG}"]
    return MM.init_params(rule, SOURCE_SEED, init_coeffs=slot)


def load_metadata(source_dir):
    files = source_files(source_dir)
    for k in ("status.json", "selection.json"):
        if not os.path.isfile(files[k]):
            raise SourceRefusal(f"missing source metadata {files[k]}")
    with open(files["status.json"]) as fh:
        st = json.load(fh)
    with open(files["selection.json"]) as fh:
        sel = json.load(fh)
    return st, sel


def verify_metadata(rule, st, sel):
    """The designated file must be the saved development run of this family,
    slot B, seed 300, with slot B the saved selection. Returns the dev row."""
    if st.get("run_id") != SOURCE_RUN_ID:
        raise SourceRefusal(f"source run_id {st.get('run_id')!r} is not "
                            f"{SOURCE_RUN_ID}")
    if st.get("dev_seed") != SOURCE_SEED or not st.get("complete"):
        raise SourceRefusal("source status is not the completed run with "
                            f"development seed {SOURCE_SEED}")
    rows = [r for r in st.get("development", [])
            if r.get("rule") == rule and r.get("config") == SOURCE_CONFIG
            and r.get("seed") == SOURCE_SEED
            and r.get("tag") == SOURCE_TAG]
    if len(rows) != 1:
        raise SourceRefusal(f"{rule}: {len(rows)} saved development rows for "
                            f"{stem(rule)}; exactly one required")
    row = rows[0]
    if (f"{row['tag']}_{row['rule']}_{row['config']}_seed{row['seed']}"
            != stem(rule)):
        raise SourceRefusal(f"{rule}: metadata stem does not match file name")
    if sel.get("selected", {}).get(rule) != SOURCE_CONFIG:
        raise SourceRefusal(f"{rule}: saved selection is "
                            f"{sel.get('selected', {}).get(rule)!r}, not "
                            f"{SOURCE_CONFIG}")
    return row


def restore(rule, source_dir):
    """Restore one family's designated checkpoint into float32 leaves.
    Refuses missing files, leaf-set/shape/dtype mismatches and non-finite
    values."""
    from flax import serialization
    path = source_files(source_dir)[rule]
    if not os.path.isfile(path):
        raise SourceRefusal(f"missing designated source checkpoint {path}")
    tmpl = {k: onp.asarray(v) for k, v in template(rule).items()}
    with open(path, "rb") as fh:
        raw = serialization.msgpack_restore(fh.read())
    if set(raw) != set(tmpl):
        raise SourceRefusal(f"{rule}: leaf set {sorted(raw)} != "
                            f"{sorted(tmpl)}")
    out = {}
    for k, t in tmpl.items():
        v = onp.asarray(raw[k])
        if v.shape != t.shape or v.dtype != onp.float32:
            raise SourceRefusal(f"{rule}/{k}: {v.shape} {v.dtype}, expected "
                                f"{t.shape} float32")
        if not onp.all(onp.isfinite(v)):
            raise SourceRefusal(f"{rule}/{k}: non-finite restored values")
        out[k] = jnp.asarray(v, dtype=jnp.float32)
    return out


def reproduction_differences(saved, measured):
    """Compare a re-evaluation with the saved development metrics.

    Accuracies are counts over a fixed validation set: each family/category
    may differ by at most REPRO_QUERIES_PER_CATEGORY queries (a tie flipped by
    a different GPU kernel); cross-entropies within REPRO_CE_REL relative.
    Returns (failures, table). All differences are recorded."""
    fails, table = [], []
    for fam in ("recall", "revision"):
        for cn, cs in saved[fam]["by_category"].items():
            cm = measured[fam]["by_category"][cn]
            n = cs["n"]
            dq = abs(cm["accuracy"] - cs["accuracy"]) * n
            dce = (abs(cm["cross_entropy"] - cs["cross_entropy"])
                   / max(abs(cs["cross_entropy"]), 1e-12))
            table.append(dict(family=fam, category=cn, n=n,
                              saved_accuracy=cs["accuracy"],
                              measured_accuracy=cm["accuracy"],
                              query_difference=dq, saved_ce=cs["cross_entropy"],
                              measured_ce=cm["cross_entropy"],
                              ce_relative_difference=dce))
            if (cm["n"] != n or not onp.isfinite(dq) or not onp.isfinite(dce)
                    or dq > REPRO_QUERIES_PER_CATEGORY + 1e-3
                    or dce > REPRO_CE_REL):
                fails.append(f"{fam}/{cn}: queries {dq:.3f}, CE rel {dce:.2e}")
    for key in ("primary", "retention_revision_untouched", "recall_overall"):
        table.append(dict(metric=key, saved=saved[key], measured=measured[key],
                          difference=measured[key] - saved[key]))
    return fails, table


def jax_leaves_equal(a, b):
    return all(bool(onp.array_equal(onp.asarray(a[k]), onp.asarray(b[k])))
               for k in a) and set(a) == set(b)
