"""Focused checks for the independent-source replication. CLUSTER-ONLY.

These add only what independent sources require: frozen streams, the
arm-to-source plan, source independence, manifest save/restore and refusals,
detection of a changed source, the frozen development selection, and the
screens on the replication's seeds. The model, loss, projection, derivative
and supervisor coverage stay in tests/test_prospective_momentum.py and are
not duplicated here.

Tolerances are the completed study's; nothing new is measured.
"""

import json
import os
import subprocess
import sys

import jax
import numpy as onp
import pytest
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.nested_memory import model as NM                 # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import replication as RP    # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import source as SRC        # noqa: E402
from experiments.prospective_momentum import study as ST          # noqa: E402


# ============================================== 1. frozen streams ===========
def test_streams_are_frozen_and_disjoint_from_every_previous_study():
    assert RP.stream_overlaps() == []
    r = RP.new_ranges()
    assert r["source_train"] == (205_000_000, 205_030_199)
    assert r["continuation_train"] == (225_000_000, 225_030_199)
    assert r["source_validation"] == (240_000_000,) * 2
    assert r["dev_validation"] == (250_000_000,) * 2
    assert r["eval_validation"] == (251_000_000,) * 2
    assert r["heldout"] == (260_000_000,) * 2
    # source training and continuation are separate streams at the same seed
    assert RP.source_stream(501, 7) != RP.continuation_stream(501, 7)
    # the registry really does include the completed study and the fixtures
    prev = RP.previous_ranges()
    assert (80_000_000, 80_000_000 + 999 * 10_000 + 999) in prev
    assert (100_000_000, 100_000_000) in prev
    assert RP.TEST_FIXTURE_RANGE in prev


# ============================================== 2. arm -> source plan =======
def test_every_arm_receives_its_intended_source():
    sel = {r: "B" for r in PD.RULES}
    plan = RS.source_plan(PD.RULES, PD.RULES, sel)
    dev = [p for p in plan if p["stage"] == "dev"]
    fin = [p for p in plan if p["stage"] == "final"]
    assert len(dev) == 12 and len(fin) == 18
    assert {p["seed"] for p in dev} == {RS.SOURCE_DEV}
    assert {p["seed"] for p in fin} == set(RS.SOURCE_FINAL)
    assert RS.SOURCE_DEV not in {p["seed"] for p in fin}
    for seed in RS.SOURCE_FINAL:
        trio = {p["source"] for p in fin if p["seed"] == seed
                and p["rule"] in PD.MOMENTUM_FAMILY}
        assert trio == {RS.source_key(seed, "momentum_delta")}, trio
        for rule in ("gated_delta", "gp_two_sided", "tss_eq17"):
            got = {p["source"] for p in fin if p["seed"] == seed
                   and p["rule"] == rule}
            assert got == {RS.source_key(seed, rule)}
    # sources of different seeds are different objects
    assert len({p["source"] for p in fin}) == 4 * len(RS.SOURCE_FINAL)
    # the completed study's read-only checkpoints are never in this plan
    assert all(not p["source"].startswith("dev_") for p in plan)


def test_sources_are_independently_initialized_across_seeds():
    for fam in RS.SOURCE_FAMILIES:
        trees = {s: RS.init_source(fam, s) for s in RS.SOURCE_SEEDS}
        for i, a in enumerate(RS.SOURCE_SEEDS):
            for b in RS.SOURCE_SEEDS[i + 1:]:
                assert not onp.allclose(onp.asarray(trees[a]["key_raw"]),
                                        onp.asarray(trees[b]["key_raw"]))
    # at one seed the common tensors are shared across families
    a = RS.init_source("momentum_delta", RS.SOURCE_DEV)
    b = RS.init_source("gated_delta", RS.SOURCE_DEV)
    for k in ("key_raw", "value_table", "readout_W", "readout_b"):
        assert onp.array_equal(onp.asarray(a[k]), onp.asarray(b[k]))


def test_planned_work_is_reported_in_full():
    w = RP.planned_work()
    assert w["source_runs"] == 16 and w["development_runs"] == 12
    assert w["final_runs"] == 18 and w["total_runs"] == 46
    assert w["total_updates"] == 16 * 200 + 30 * 200 == 9200


# ============================================== 3. manifest and restore =====
def _tiny_source(tmp_path, family="momentum_delta", seed=500):
    p = RS.init_source(family, seed)
    fname, oname = RS.source_files(seed, family)
    d = RS.sources_dir(str(tmp_path))
    ST.save_tree(os.path.join(d, fname), p)
    ST.save_tree(os.path.join(d, oname), ST.TX.init(p))
    entry = dict(family=family, source_seed=seed,
                 key=RS.source_key(seed, family), file=fname,
                 sha256=RS.sha256(os.path.join(d, fname)), opt_file=oname,
                 opt_sha256=RS.sha256(os.path.join(d, oname)))
    return p, entry


def test_manifest_roundtrip_checksums_and_refusals(tmp_path):
    p, entry = _tiny_source(tmp_path)
    manifest = dict(entries=[entry])
    RS.write_manifest(str(tmp_path), manifest)
    d = RS.sources_dir(str(tmp_path))
    assert json.load(open(os.path.join(d, "manifest.json")))["entries"]
    sums = open(os.path.join(d, "SHA256SUMS")).read().strip().split("\n")
    assert sums[0] == f"{entry['sha256']}  {entry['file']}"    # sha256sum -c
    assert len(sums) == 2
    restored = RS.restore_source(str(tmp_path), entry)
    assert SRC.jax_leaves_equal(restored, p)
    assert RS.find_entry(manifest, 500, "momentum_delta") is entry
    with pytest.raises(RS.SourceRefusal):
        RS.find_entry(manifest, 501, "momentum_delta")
    with pytest.raises(RS.SourceRefusal):
        RS.restore_source(str(tmp_path), dict(entry, file="missing.msgpack"))
    with pytest.raises(RS.SourceRefusal):          # checksum mismatch
        RS.restore_source(str(tmp_path), dict(entry, sha256="0" * 64))
    with open(os.path.join(d, entry["file"]), "r+b") as fh:
        fh.seek(0); head = fh.read(1); fh.seek(0)
        fh.write(bytes([head[0] ^ 0xFF]))
    with pytest.raises(RS.SourceRefusal):          # tampered file
        RS.restore_source(str(tmp_path), entry)


def test_a_changed_source_file_fails_the_finalizer(tmp_path):
    _, entry = _tiny_source(tmp_path)
    manifest = dict(entries=[entry])
    base = RS.baseline_hashes(manifest)
    assert set(base) == {entry["file"], entry["opt_file"]}
    st = dict(source=dict(hashes_at_restore=base))
    rehash = lambda: RS.rehash(str(tmp_path), list(base))   # noqa: E731
    assert ST.finalize(st, 0, "PASS", rehash, lambda x: None) == (0, "PASS")
    with open(os.path.join(RS.sources_dir(str(tmp_path)), entry["file"]),
              "ab") as fh:
        fh.write(b"x")
    st2 = dict(source=dict(hashes_at_restore=base))
    assert ST.finalize(st2, 0, "PASS", rehash, lambda x: None) == (4, "FAILED")
    assert st2["source_unchanged"] is False
    os.remove(os.path.join(RS.sources_dir(str(tmp_path)), entry["file"]))
    st3 = dict(source=dict(hashes_at_restore=base))
    assert ST.finalize(st3, 0, "PASS", rehash, lambda x: None) == (4, "FAILED")


def test_the_source_recipe_is_the_original_slot_b_definition():
    assert (RS.SOURCE_UPDATES, RS.SOURCE_LR, RS.SOURCE_SLOT) == (200, 0.01,
                                                                 "B")
    assert dict(ST.LRS)["B"] == RS.SOURCE_LR
    for fam in RS.SOURCE_FAMILIES:
        slot = RS.slot_coefficients(fam)
        assert slot["rule"] == fam and slot["config"] == "B"
    p = RS.init_source("momentum_delta", 501)
    assert PM.parameter_counts("momentum_delta", p)["total"] == 569
    for rule in ("prospective_momentum", "gain_momentum"):
        q = PM.convert_momentum(p, rule)
        assert PM.parameter_counts(rule, q)["total"] == 570
        assert float(q[PD.EXTRA_LEAF[rule]][0]) == 0.0


# ============================================== 4. selection stays frozen ===
def _dev_rows(primary):
    rows = []
    for rule, per_slot in primary.items():
        for tag, v in per_slot.items():
            rows.append(dict(tag="dev", rule=rule, config=tag,
                             lr=dict(ST.LRS)[tag], seed=RS.SOURCE_DEV,
                             final_validation=dict(primary=v,
                                                   revision_ce=1.0)))
    return rows


def test_development_selection_is_frozen_and_finals_cannot_change_it():
    primary = {r: {"A": 0.50, "B": 0.55} for r in PD.RULES}
    primary["gated_delta"] = {"A": 0.60, "B": 0.55}
    status = dict()
    rows = _dev_rows(primary)
    frozen = RP.freeze_selection(rows, status)
    assert frozen["gated_delta"] == "A"
    assert all(frozen[r] == "B" for r in PD.RULES if r != "gated_delta")
    assert RP.selection_unchanged(status, frozen)
    # final rows and held-out results do not enter the selection
    rows.append(dict(tag="final", rule="gated_delta", config="B", lr=0.01,
                     seed=RS.SOURCE_FINAL[0],
                     final_validation=dict(primary=0.99, revision_ce=0.1)))
    status2 = dict()
    assert RP.freeze_selection(rows, status2) == frozen
    # a later mutation of the selection is detected
    status["selection"]["selected"]["gated_delta"] = "B"
    assert not RP.selection_unchanged(status, frozen)


def test_selection_uses_only_source_500_development_rows():
    rows = _dev_rows({r: {"A": 0.50, "B": 0.55} for r in PD.RULES})
    stray = dict(rows[0], seed=RS.SOURCE_FINAL[0], config="A",
                 final_validation=dict(primary=0.99, revision_ce=0.1))
    status = dict()
    assert RP.freeze_selection(rows + [stray], status) is not None
    assert status["selection"]["selected"][rows[0]["rule"]] == "B"


# ============================================== 5. screens on new seeds =====
def _final_rows(primary, retention=0.60, recall=0.70):
    return [dict(rule=r, seed=s, heldout=dict(
        primary=v, retention_revision_untouched=retention,
        recall_overall=recall))
        for s in RS.SOURCE_FINAL for r, v in primary.items()]


def test_screens_use_the_replication_seeds_and_state_directions():
    base = dict(prospective_momentum=0.60, momentum_delta=0.55,
                gated_delta=0.55, gain_momentum=0.55, gp_two_sided=0.50,
                tss_eq17=0.50)
    rows = _final_rows(base)
    sc = RP.annotate_directions(ST.screen(rows, seeds=RS.SOURCE_FINAL))
    assert sc["literature_screen_passed"] is True
    assert sc["gain_control_comparison_passed"] is True
    for c in sc["literature"]:
        assert c["complete_paired_seeds"] and c["retention_direction"] == \
            "unchanged"
        assert "zero measured loss" in c["safeguard_note"]
    # the completed study's seeds are NOT these seeds: without the new seeds
    # the pairing is incomplete and nothing passes
    assert ST.screen(rows)["literature_screen_passed"] is False
    # a retention loss beyond -1 pp still fails, thresholds unchanged
    worse = _final_rows(base)
    for r in worse:
        if r["rule"] == "prospective_momentum":
            r["heldout"]["retention_revision_untouched"] = 0.58
    sc2 = RP.annotate_directions(ST.screen(worse, seeds=RS.SOURCE_FINAL))
    assert sc2["literature_screen_passed"] is False
    assert sc2["literature"][0]["retention_direction"] == "decreased"


def test_preflight_decision_and_planned_projection_fields():
    assert ST.decide_after_preflight(10.0, False, ["x"], 500)[0] == 4
    assert ST.decide_after_preflight(900.0, False, [], 500)[0] == 3
    assert ST.decide_after_preflight(10.0, False, [], 500) is None


# ============================================== 6. launcher verification ====
def test_launcher_verifies_the_new_source_manifest(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib = os.path.join(repo, "bin", "run_experiments",
                       "prospective_momentum_replication_verify.sh")
    run = tmp_path / "run"
    (run / "sources").mkdir(parents=True)
    (run / "sources" / "a.msgpack").write_bytes(b"hello")
    subprocess.run("sha256sum a.msgpack > SHA256SUMS", shell=True,
                   cwd=str(run / "sources"), check=True)
    script = r"""
set -u
source "$LIB"
START=$(date +%s); DEADLINE=$(( START + 40 )); LOG_DIR="$TMP"
PM_INTEGRITY_RC=0
pm_extra_verify "$RUN" > /dev/null; echo NEW_OK=$PM_INTEGRITY_RC
printf 'x' >> "$RUN/sources/a.msgpack"
PM_INTEGRITY_RC=0
pm_extra_verify "$RUN" > /dev/null; echo NEW_CHANGED=$PM_INTEGRITY_RC
PM_INTEGRITY_RC=1
pm_extra_verify "$RUN" > /dev/null; echo WORST_KEPT=$PM_INTEGRITY_RC
rm -f "$RUN/sources/SHA256SUMS"
PM_INTEGRITY_RC=0
pm_extra_verify "$RUN" > /dev/null; echo NEW_MISSING=$PM_INTEGRITY_RC
"""
    env = dict(os.environ, LIB=lib, RUN=str(run), TMP=str(tmp_path),
               PY=sys.executable)
    r = subprocess.run(["bash", "-c", script], cwd=repo, env=env,
                       capture_output=True, text=True, timeout=60)
    print(r.stdout); print(r.stderr[-2000:])
    out = dict(tok.split("=", 1) for tok in r.stdout.split()
               if "=" in tok and tok.split("=", 1)[0].isupper())
    assert out["NEW_OK"] == "0"
    assert out["NEW_CHANGED"] == "1"
    assert out["WORST_KEPT"] == "1"
    assert out["NEW_MISSING"] == "3"


def test_no_parameter_comes_from_the_completed_studys_checkpoints():
    """The old read-only source may be READ by the unchanged restored-
    checkpoint checks, but never supplies parameters or selection here."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "experiments", "prospective_momentum",
                            "replication.py")).read()
    assert "SRC.restore(" not in src and "SRC.SOURCE_RULE" not in src
    assert "20260916-222310" not in src
    assert "RS.restore_source" in src
