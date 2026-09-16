"""Independent-source replication of prospective Momentum DeltaNet.

Protocol, frozen for review before any execution:
docs/PROSPECTIVE_MOMENTUM_REPLICATION_PROTOCOL.md. NOT AUTHORIZED TO RUN
until cleared. Brief: PROSPECTIVE_MOMENTUM_INDEPENDENT_REPLICATION_2026_09_17.md.

Unchanged from the completed study (`study.py`, reused, not copied): the six
arms, production rollouts, loss, optimizer, projection, compiled train step,
evaluation, validation, identity rule, selection rule, screens and
finalization. What is new is only orchestration:

  * 16 source pretraining runs: sources 500 (development) and 501, 502, 503
    (final), each freshly initialized with the original initializer and
    slot-B coefficients, 200 updates at lr 0.01 on its own source-training
    stream; saved, checksummed, restore-checked and read-only afterwards;
  * 12 development continuations from source 500 (slots A/B), selection by
    the original update-200 rule, frozen before finals;
  * 18 final continuations, each from its own final source;
  * fresh named streams; one common held-out set generated after all finals.

Total 46 runs, 9,200 optimizer updates, all inside one 600-second cap.
"""

import argparse
import copy
import os
import signal
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.meta_delta import study as MS                    # noqa: E402
from experiments.nested_memory import task as TK                  # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import study as ST          # noqa: E402

UPDATES = ST.UPDATES
LRS = ST.LRS
VAL_PER_FAMILY = ST.VAL_PER_FAMILY
HELDOUT_PER_FAMILY = ST.HELDOUT_PER_FAMILY
#: frozen named streams (brief s4)
STREAM = dict(source_train=200_000_000, continuation_train=220_000_000,
              source_validation=240_000_000, dev_validation=250_000_000,
              eval_validation=251_000_000, heldout=260_000_000)
#: small fixed generator seeds used by focused-check fixtures (all < 10,000)
TEST_FIXTURE_RANGE = (0, 9_999)
SEED_SPAN = ST.SEED_SPAN
PLANNED = dict(source_runs=len(RS.SOURCE_SEEDS) * len(RS.SOURCE_FAMILIES),
               development_runs=len(PD.RULES) * len(LRS),
               final_runs=len(PD.RULES) * len(RS.SOURCE_FINAL))
PLANNED["total_runs"] = sum(PLANNED.values())
PLANNED["total_updates"] = (PLANNED["source_runs"] * RS.SOURCE_UPDATES
                            + (PLANNED["development_runs"]
                               + PLANNED["final_runs"]) * UPDATES)


def source_stream(seed, update):
    return STREAM["source_train"] + seed * 10_000 + update


def continuation_stream(seed, update):
    return STREAM["continuation_train"] + seed * 10_000 + update


def new_ranges():
    lo, hi = min(RS.SOURCE_SEEDS), max(RS.SOURCE_SEEDS)
    r = dict(source_train=(source_stream(lo, 0),
                           source_stream(hi, RS.SOURCE_UPDATES - 1)),
             continuation_train=(continuation_stream(lo, 0),
                                 continuation_stream(hi, UPDATES - 1)))
    for k in ("source_validation", "dev_validation", "eval_validation",
              "heldout"):
        r[k] = (STREAM[k], STREAM[k])
    return r


def previous_ranges():
    """Every earlier task-generator stream: nested, adaptive, meta-delta (via
    the completed study's registry), the completed prospective-momentum study
    (train range for all seeds < 1000, validations, held-out) and the small
    fixed fixture seeds of the focused checks."""
    prev = list(ST.previous_streams())
    prev.append((ST.STREAM["train"],
                 ST.STREAM["train"] + (SEED_SPAN - 1) * 10_000 + 999))
    prev += [(ST.STREAM[k], ST.STREAM[k]) for k in
             ("dev_validation", "eval_validation", "heldout")]
    prev.append(TEST_FIXTURE_RANGE)
    return prev


def stream_overlaps():
    new = new_ranges()
    bad, names = [], list(new)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if new[a][0] <= new[b][1] and new[b][0] <= new[a][1]:
                bad.append((a, b))
        for lo, hi in previous_ranges():
            if new[a][0] <= hi and lo <= new[a][1]:
                bad.append((a, (lo, hi)))
    return bad


def planned_work():
    return dict(PLANNED)


# ------------------------------------------------------ source pretraining --
def _summary(m):
    return dict(primary=m["primary"], revision_ce=m["revision_ce"],
                retention=m["retention_revision_untouched"],
                recall=m["recall_overall"])


def pretrain_source(family, seed, src_val_np, out, deadline, reserve_s,
                    status):
    """One source: fresh init, 200 updates, save, checksum, restore check.
    Returns (entry, params) or None if the deadline stops it."""
    t0 = time.time()
    p = RS.init_source(family, seed)
    opt = ST.TX.init(p)
    lr = jnp.asarray(RS.SOURCE_LR, dtype=jnp.float32)
    m0 = ST.evaluate(family, p, src_val_np)
    hist, curve, last = [], [], None
    for u in range(RS.SOURCE_UPDATES):
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"source {RS.source_key(seed, family)} stopped at update {u} "
                f"of {RS.SOURCE_UPDATES}")
            return None
        p, opt, last = ST.host_step(family, p, opt, seed, u, lr, hist,
                                    stream=source_stream)
        if u % 50 == 0 or u == RS.SOURCE_UPDATES - 1:
            curve.append(dict(update=u, **last))
    m = ST.evaluate(family, p, src_val_np)
    bad = None
    if last is None or not all(onp.isfinite(v) for v in last.values()):
        bad = "non-finite final source training scalars"
    elif not ST.all_finite(p) or not ST.all_finite(opt):
        bad = "non-finite source parameters or optimizer state"
    elif not ST.metrics_finite(m):
        bad = "non-finite source validation metric"
    else:
        bad = ST.validate(family, p)
    fname, oname = RS.source_files(seed, family)
    d = RS.sources_dir(out)
    ST.save_tree(os.path.join(d, fname), p)
    ST.save_tree(os.path.join(d, oname), opt)
    entry = dict(
        family=family, source_seed=seed, key=RS.source_key(seed, family),
        file=fname, sha256=RS.sha256(os.path.join(d, fname)),
        opt_file=oname, opt_sha256=RS.sha256(os.path.join(d, oname)),
        initializer=dict(function="experiments.meta_delta.model.init_params",
                         seed=seed, slot=RS.SOURCE_SLOT,
                         init_coeffs=RS.slot_coefficients(family)),
        recipe=dict(updates=RS.SOURCE_UPDATES, lr=RS.SOURCE_LR,
                    batch_per_family=ST.BATCH_PER_FAMILY,
                    loss="query-mean cross-entropy (unweighted)",
                    optimizer="clip_by_global_norm(1.0) + Adam(0.9,0.999,1e-8)",
                    projection="study._project (raw_r for gp_two_sided)"),
        stream=dict(name="source_train", first=source_stream(seed, 0),
                    last=source_stream(seed, RS.SOURCE_UPDATES - 1)),
        validation_stream=STREAM["source_validation"],
        metrics_start=_summary(m0), metrics_end=m, curve=curve,
        extension_summary=(ST.summarize_history(hist, family)
                           if family in ST.EXTRA_GRAD else None),
        params=PM.parameter_counts(family, p), invalid=bad)
    # save/restore reproduction: bitwise leaves and re-evaluated metrics
    try:
        restored = RS.restore_source(out, entry)
        entry["restore_bitwise_equal"] = bool(
            ST.SRC.jax_leaves_equal(restored, p))
        m_r = ST.evaluate(family, restored, src_val_np)
        fails, table = ST.SRC.reproduction_differences(m, m_r)
        entry["restore_reproduction_failures"] = fails
        if not entry["restore_bitwise_equal"] or fails:
            entry["invalid"] = entry["invalid"] or (
                f"save/restore reproduction failed: bitwise="
                f"{entry['restore_bitwise_equal']} failures={fails}")
    except RS.SourceRefusal as e:
        entry["invalid"] = entry["invalid"] or f"restore refused: {e}"
    entry["wall_s"] = time.time() - t0
    return entry, p


def momentum_identity(out, manifest, seed, val_np):
    """Candidate, native and gain start as the SAME function of this seed's
    restored Momentum source (the completed study's identity rule)."""
    entry = RS.find_entry(manifest, seed, "momentum_delta")
    pn = RS.restore_source(out, entry)
    ev = {r: ST.evaluate(r, PM.convert_momentum(pn, r), val_np)
          for r in PD.MOMENTUM_FAMILY}
    return {r: ST.identity_differences(ev["momentum_delta"], ev[r])
            for r in ("prospective_momentum", "gain_momentum")}


# ------------------------------------------------------------- preflight ---
def preflight(src_val_np, val_np, out, status):
    """Times the ACTUAL paths on disposable state: source pretraining steps
    on the source stream, continuation steps on the continuation stream,
    evaluation, validation/report, save + checksum + restore. Initializers
    are pure functions of the seed, so nothing needs resetting afterwards;
    no preflight tree is used later."""
    rows, failures, retraced_any = [], [], False
    step_src, eval_s_rule, report_s_rule = {}, {}, {}
    disposable = {}
    lr_src = jnp.asarray(RS.SOURCE_LR, dtype=jnp.float32)
    for fam in RS.SOURCE_FAMILIES:
        p = RS.init_source(fam, RS.SOURCE_DEV)
        opt = ST.TX.init(p)
        hist = []
        t0 = time.time()
        p, opt, _ = ST.host_step(fam, p, opt, RS.SOURCE_DEV, 0, lr_src, hist,
                                 stream=source_stream)
        compile_s = time.time() - t0
        p, opt, _ = ST.host_step(fam, p, opt, RS.SOURCE_DEV, 1, lr_src, hist,
                                 stream=source_stream)
        n0 = ST.train_step._cache_size()
        t1 = time.time()
        for u in range(2, 7):
            p, opt, measured = ST.host_step(fam, p, opt, RS.SOURCE_DEV, u,
                                            lr_src, hist, stream=source_stream)
        step_src[fam] = (time.time() - t1) / 5.0
        retraced_any |= ST.train_step._cache_size() != n0
        bad = MS.measured_scalar_failures(measured)
        if bad:
            failures.append(f"source {fam}: non-finite scalars {bad}")
        disposable[fam] = p
        rows.append(dict(path="source_pretraining", rule=fam,
                         compile_s_incurred=compile_s, step_s=step_src[fam],
                         measured_scalars=measured))
    # save + checksum + restore, disposable directory
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        t0 = time.time()
        fam = "momentum_delta"
        fname, oname = RS.source_files(0, fam)
        ST.save_tree(os.path.join(tmp, "sources", fname), disposable[fam])
        ST.save_tree(os.path.join(tmp, "sources", oname),
                     ST.TX.init(disposable[fam]))
        e = dict(family=fam, file=fname, sha256=RS.sha256(
            os.path.join(tmp, "sources", fname)))
        RS.sha256(os.path.join(tmp, "sources", oname))
        RS.restore_source(tmp, e)
        save_s = time.time() - t0
    lr = jnp.asarray(LRS[0][1], dtype=jnp.float32)
    for rule in PD.RULES:
        fam = RS.SOURCE_OF[rule]
        p = (PM.convert_momentum(disposable[fam], rule)
             if rule in PD.MOMENTUM_FAMILY else dict(disposable[fam]))
        opt = ST.TX.init(p)
        hist = []
        t0 = time.time()
        p2, opt2, _ = ST.host_step(rule, p, opt, RS.SOURCE_DEV, 0, lr, hist,
                                   stream=continuation_stream)
        compile_s = time.time() - t0
        p2, opt2, _ = ST.host_step(rule, p2, opt2, RS.SOURCE_DEV, 1, lr, hist,
                                   stream=continuation_stream)
        n0 = ST.train_step._cache_size()
        t1 = time.time()
        for u in range(2, 7):
            p2, opt2, measured = ST.host_step(rule, p2, opt2, RS.SOURCE_DEV,
                                              u, lr, hist,
                                              stream=continuation_stream)
        step_s = (time.time() - t1) / 5.0
        retraced_any |= ST.train_step._cache_size() != n0
        t2 = time.time()
        ST.evaluate(rule, p2, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        m = ST.evaluate(rule, p2, val_np)
        eval_s_rule[rule] = time.time() - t3
        t4 = time.time()
        acc_bad = ST.validate(rule, p2)
        ST.coefficient_report(rule, p2, val_np)
        report_s_rule[rule] = time.time() - t4
        bad = MS.measured_scalar_failures(measured)
        if bad:
            acc_bad = f"non-finite measured preflight scalars {bad}"
        elif not (ST.all_finite(p2) and ST.all_finite(opt2)
                  and ST.metrics_finite(m)):
            acc_bad = "non-finite preflight state or metrics"
        if acc_bad:
            failures.append(f"{rule}: {acc_bad}")
        rows.append(dict(path="continuation", rule=rule,
                         compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s,
                         step_s=step_s, eval_s=eval_s_rule[rule],
                         report_s=report_s_rule[rule],
                         acceptance_failure=acc_bad,
                         measured_scalars=measured,
                         params=PM.parameter_counts(rule, p2),
                         carry=PD.CARRY[rule]))
        print(f"[preflight] {rule:<22} src-step "
              f"{step_src[fam] * 1e3:6.2f}ms cont-step {step_s * 1e3:6.2f}ms "
              f"eval {eval_s_rule[rule] * 1e3:6.1f}ms report "
              f"{report_s_rule[rule]:5.2f}s compile {compile_s:4.1f}s")
    # projection of ALL remaining work; compilation already incurred above
    n_seeds = len(RS.SOURCE_SEEDS)
    src_s = sum(n_seeds * (RS.SOURCE_UPDATES * step_src[f]
                           + 3.0 * eval_s_rule[f] + 2.0 * report_s_rule[f]
                           + save_s)
                for f in RS.SOURCE_FAMILIES)
    ident_s = n_seeds * sum(eval_s_rule[r] + save_s
                            for r in PD.MOMENTUM_FAMILY)
    n_cont = len(LRS) + len(RS.SOURCE_FINAL)
    cont_s = 0.0
    for r in PD.RULES:
        step_r = [x["step_s"] for x in rows
                  if x["path"] == "continuation" and x["rule"] == r][0]
        cont_s += n_cont * (UPDATES * step_r + len(ST.VAL_AT) * eval_s_rule[r]
                            + 2.0 * report_s_rule[r] + save_s)
        cont_s += len(RS.SOURCE_FINAL) * 2.0 * eval_s_rule[r]   # held-out
    host_s = 40.0                       # ALLOWANCE, recorded as such
    total = src_s + ident_s + cont_s + host_s
    timing = dict(source_s=src_s, identity_s=ident_s, continuation_s=cont_s,
                  save_checksum_restore_s=save_s, host_allowance_s=host_s,
                  projected_remaining_s=total)
    if not all(onp.isfinite(v) and v >= 0 for v in timing.values()):
        failures.append(f"non-finite or negative timing {timing}")
    status["preflight"] = dict(rows=rows, retraced_any=bool(retraced_any),
                               failures=failures, planned=planned_work(),
                               **timing,
                               note=("disposable state; compilation incurred "
                                     "here is not re-counted"))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f} (sources {src_s:.1f}, "
          f"identity {ident_s:.1f}, continuation {cont_s:.1f}, host "
          f"{host_s:.1f})")
    return total, bool(retraced_any), failures


# ------------------------------------------------------------- selection ---
def freeze_selection(dev_rows, status):
    """The original rule on update-200 DEVELOPMENT rows only; a deep copy is
    kept and checked unchanged before held-out and after the screens."""
    dev_only = [r for r in dev_rows if r.get("tag") == "dev"
                and r.get("seed") == RS.SOURCE_DEV]
    sel, _ = ST.select(dev_only, status)
    if sel is None:
        return None
    frozen = copy.deepcopy(sel)
    status["selection_frozen"] = copy.deepcopy(sel)
    return frozen


def selection_unchanged(status, frozen):
    return (frozen is not None
            and status.get("selection", {}).get("selected") == frozen
            and status.get("selection_frozen") == frozen)


def direction(d):
    if d is None:
        return None
    return "increased" if d > 0 else ("decreased" if d < 0 else "unchanged")


#: R3 (review of 7344e32): the reused screen's default note describes the
#: completed study's SHARED source. This replication replaces it; the screen's
#: comparisons, thresholds and the original default note are untouched.
REPLICATION_NOTE = (
    "Independent-source replication. The three final seeds vary "
    "initialization, source pretraining AND continuation: each has its own "
    "freshly initialized and independently pretrained source per family. The "
    "held-out evaluation set and the development-selected recipe (slot) are "
    "common to all final models by design. Three seeds on this small "
    "associative task are not significance, not a benchmark and not SOTA. "
    "The fixed-coefficient update remains QHM-equivalent at alpha = 1; no "
    "optimizer novelty is claimed. Frozen-token stability is not switching "
    "stability, and each source's occupied sector is reported from its own "
    "learned coefficients.")


def annotate_directions(sc):
    for c in sc["literature"] + [sc["gain_control"], sc["old_generalized"],
                                 sc["tss_eq17"]]:
        c["retention_direction"] = direction(c["retention_difference"])
        c["recall_direction"] = direction(c["recall_difference"])
        c["safeguard_note"] = ("passing the -1 pp safeguard does not mean "
                               "zero measured loss")
    sc["reused_screen_default_note"] = sc["note"]
    sc["note"] = REPLICATION_NOTE
    return sc


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/prospective-momentum-replication")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    args = ap.parse_args()

    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    backend = jax.default_backend()
    if backend != "gpu":
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    if jax.config.read("jax_enable_x64") or jnp.zeros(1).dtype != jnp.float32:
        raise SystemExit("REFUSING: x64 is enabled; the declared production "
                         "setting is float32 with x64 disabled.")
    overlaps = stream_overlaps()
    if overlaps:
        raise SystemExit(f"REFUSING: stream ranges overlap: {overlaps}")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)

    manifest = dict(entries=[], note=("independently initialized and "
                                      "pretrained sources; read-only after "
                                      "creation"))
    status = dict(run_id=run_id, out=out, backend=backend,
                  study="independent-source replication",
                  source_seeds=dict(development=RS.SOURCE_DEV,
                                    final=list(RS.SOURCE_FINAL)),
                  final_seeds=list(RS.SOURCE_FINAL), updates=UPDATES,
                  source_recipe=dict(updates=RS.SOURCE_UPDATES,
                                     lr=RS.SOURCE_LR, slot=RS.SOURCE_SLOT),
                  planned_work=planned_work(),
                  streams=STREAM, stream_ranges=new_ranges(),
                  previous_stream_ranges=previous_ranges(),
                  arms={r: PD.DISPLAY[r] for r in PD.RULES},
                  arm_source_family=RS.SOURCE_OF,
                  learning_rates=dict(LRS),
                  projection_relative_margin=PD.PROJ_REL_MARGIN,
                  production_dtype=dict(x64=False, float_dtype="float32"),
                  heldout_policy=("one common held-out set, generated and "
                                  "hashed only after all final runs; opening "
                                  "persisted before use; never used for "
                                  "selection"),
                  completed_study_reference=(
                      "/Users/durso/s5-runs/prospective-momentum/"
                      "20260917-000431 (verdicts unchanged)"),
                  # R2: no source baseline exists yet; `None` means source
                  # invariance is UNAVAILABLE, never "verified"
                  source=dict(hashes_at_restore=None, manifest=manifest),
                  development=[], final=[], incomplete=[])
    status_path = os.path.join(out, "status.json")

    def persist(st):
        st["wall_s"] = time.time() - t0
        ST.write(status_path, st)

    def rehash():
        """Re-hash exactly the source files recorded so far. Before the first
        source exists the baseline is None: return {} without raising, so the
        finalizer reports invariance as unavailable (R2)."""
        keys = (status.get("source") or {}).get("hashes_at_restore")
        return RS.rehash(out, list(keys)) if keys else {}

    signal.signal(signal.SIGTERM, ST._on_sigterm)
    code, _ = ST.guarded(lambda: run_replication(args, status, manifest, out,
                                                 deadline,
                                                 lambda: persist(status)),
                         status, rehash, persist)
    return code


def run_replication(args, status, manifest, out, deadline, save):
    src_val_np = TK.generate_batch(STREAM["source_validation"],
                                   VAL_PER_FAMILY)
    val_np = TK.generate_batch(STREAM["dev_validation"], VAL_PER_FAMILY)
    eval_val_np = TK.generate_batch(STREAM["eval_validation"], VAL_PER_FAMILY)
    status["task"] = dict(
        structure=TK.structure_check(val_np),
        source_validation_digest=TK.episode_digest(src_val_np),
        dev_validation_digest=TK.episode_digest(val_np),
        eval_validation_digest=TK.episode_digest(eval_val_np),
        source_train_first_batch_digest=TK.episode_digest(
            TK.generate_batch(source_stream(RS.SOURCE_DEV, 0),
                              ST.BATCH_PER_FAMILY)),
        continuation_train_first_batch_digest=TK.episode_digest(
            TK.generate_batch(continuation_stream(RS.SOURCE_DEV, 0),
                              ST.BATCH_PER_FAMILY)))
    save()

    proj, retraced, failures = preflight(src_val_np, val_np, out, status)
    save()
    d = ST.decide_after_preflight(proj, retraced, failures,
                                  deadline - time.time() - args.reserve_s)
    if d is not None:
        code, label, why = d
        (status.__setitem__("failed", why) if code == 4
         else status["incomplete"].append(why))
        print(f"[!] {why}")
        return code, label

    # ---- 16 independent sources, read-only after creation
    status["source"]["identity"] = {}
    for seed in RS.SOURCE_SEEDS:
        for fam in RS.SOURCE_FAMILIES:
            r = pretrain_source(fam, seed, src_val_np, out, deadline,
                                args.reserve_s, status)
            if r is None:
                return 3, "INCOMPLETE"
            entry, _ = r
            manifest["entries"].append(entry)
            RS.write_manifest(out, manifest)
            status["source"]["hashes_at_restore"] = RS.baseline_hashes(
                manifest)
            status["source"]["manifest"] = manifest
            save()
            print(f"[source] {entry['key']:<28} primary "
                  f"{entry['metrics_start']['primary']:.4f} -> "
                  f"{entry['metrics_end']['primary']:.4f} restore_bitwise "
                  f"{entry.get('restore_bitwise_equal')} invalid "
                  f"{entry['invalid']}")
            if entry["invalid"]:
                status["failed"] = f"source {entry['key']}: {entry['invalid']}"
                return 4, "FAILED"
        id_val = val_np if seed == RS.SOURCE_DEV else eval_val_np
        ident = momentum_identity(out, manifest, seed, id_val)
        status["source"]["identity"][str(seed)] = ident
        save()
        if any(ident.values()):
            status["failed"] = (f"source {seed}: candidate/native/gain differ "
                                f"at update zero: {ident}")
            return 4, "FAILED"

    def source_for(rule, seed):
        entry = RS.find_entry(manifest, seed, RS.SOURCE_OF[rule])
        p = RS.restore_source(out, entry)             # checksum verified
        if rule in PD.MOMENTUM_FAMILY:
            p = PM.convert_momentum(p, rule)
        return p, f"{entry['file']} sha256={entry['sha256']}"

    # ---- development: source 500 only, both slots
    dev_rows = []
    for rule in PD.RULES:
        for tag, lr in LRS:
            p, label = source_for(rule, RS.SOURCE_DEV)
            r = ST.run_one(rule, tag, lr, RS.SOURCE_DEV, p, val_np, UPDATES,
                           out, deadline, args.reserve_s, status, "dev",
                           stream=continuation_stream, source_label=label)
            if r is None:
                return 3, "INCOMPLETE"
            rec, _ = r
            dev_rows.append(rec)
            status["development"] = dev_rows
            save()
            if rec["invalid"]:
                status["failed"] = f"dev {rule}/{tag}: {rec['invalid']}"
                return 4, "FAILED"
    frozen = freeze_selection(dev_rows, status)
    if frozen is None:
        status["failed"] = "selection could not be formed from dev rows"
        return 4, "FAILED"
    ST.write(os.path.join(out, "selection.json"), status["selection"])
    save()

    # ---- finals: each seed from its OWN independent source
    final_rows, finals = [], {}
    lr_of = dict(LRS)
    for rule in PD.RULES:
        for seed in RS.SOURCE_FINAL:
            p, label = source_for(rule, seed)
            r = ST.run_one(rule, frozen[rule], lr_of[frozen[rule]], seed, p,
                           eval_val_np, UPDATES, out, deadline,
                           args.reserve_s, status, "final",
                           stream=continuation_stream, source_label=label)
            if r is None:
                status["heldout_opened"] = False
                return 3, "INCOMPLETE"
            rec, pf = r
            final_rows.append(rec)
            finals[(rule, seed)] = pf
            status["final"] = final_rows
            save()
            if rec["invalid"]:
                status["failed"] = f"final {rule}/{seed}: {rec['invalid']}"
                return 4, "FAILED"
    if not selection_unchanged(status, frozen):
        status["failed"] = "development selection changed during finals"
        return 4, "FAILED"

    status["heldout_opened"] = True
    status["heldout_opened_at"] = time.time()
    status["heldout_evaluation_complete"] = False
    save()
    held_np = TK.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    status["task"]["heldout_digest"] = TK.episode_digest(held_np)
    for rec in final_rows:
        rec["heldout"] = ST.evaluate(rec["rule"],
                                     finals[(rec["rule"], rec["seed"])],
                                     held_np)
        if not ST.metrics_finite(rec["heldout"]):
            status["failed"] = (f"non-finite held-out {rec['rule']}/"
                                f"{rec['seed']}")
            return 4, "FAILED"
    status["heldout_evaluation_complete"] = True
    status["screen"] = annotate_directions(
        ST.screen(final_rows, seeds=RS.SOURCE_FINAL))
    if not selection_unchanged(status, frozen):
        status["failed"] = "development selection changed after evaluation"
        return 4, "FAILED"
    status["work_completed"] = dict(
        source_runs=len(manifest["entries"]),
        development_runs=len(dev_rows), final_runs=len(final_rows),
        total_runs=len(manifest["entries"]) + len(dev_rows) + len(final_rows),
        total_updates=(len(manifest["entries"]) * RS.SOURCE_UPDATES
                       + (len(dev_rows) + len(final_rows)) * UPDATES))
    sc = status["screen"]
    print(f"[screen] LITERATURE (Momentum AND Gated): "
          f"{sc['literature_screen_passed']}  GAIN CONTROL: "
          f"{sc['gain_control_comparison_passed']}")
    print(f"[screen] old generalized: {sc['old_generalized_comparison_passed']}"
          f"  TSS Eq.(17) direct (applicability-limited): "
          f"{sc['tss_eq17_comparison_passed']}")
    status["complete"] = True
    return 0, "PASS"


if __name__ == "__main__":
    sys.exit(main())
