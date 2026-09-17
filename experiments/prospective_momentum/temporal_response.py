"""Can generalized prospective processing keep literal TSS's immediate
revision benefit while improving later recall and reducing interference with
untouched associations?

Frozen protocol: docs/PROSPECTIVE_TEMPORAL_RESPONSE_PROTOCOL.md. NOT
AUTHORIZED TO RUN until static review clears it. A new hypothesis-driven
study; the completed TSS-containment result is unchanged.

Architecture and equations are those of the TSS-containment study
(`experiments.prospective_momentum.tss_containment`): R_t -> processing state
y -> native Momentum update -> W, executed coefficient form, full BPTT, the
unweighted query cross-entropy, the executed-coefficient gate, the
feasibility repair (gamma >= 0), checkpoint validation and persistence. The
SAME five arms and training steps are reused unchanged. What changes:

* the task: `temporal_task` (declared delays, idle-gap and intervening-write
  conditions, revised and untouched probes at comparable times);
* the metrics: the declared equal-cell weighting of `temporal_task.aggregate`;
* the selection: every family by the SAME unconstrained development ordering,
  plus a separately frozen matched-operating-point rule for generalized
  versus literal TSS;
* a controlled mechanism diagnostic (`temporal_diagnostic`) in the same batch.
"""

import argparse
import copy
import os
import signal
import sys
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.nested_memory.task import FAMILIES               # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import filtered as FL       # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import replication as RP    # noqa: E402
from experiments.prospective_momentum import study as ST          # noqa: E402
from experiments.prospective_momentum import temporal_diagnostic as TD  # noqa
from experiments.prospective_momentum import temporal_task as TT  # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

UPDATES = TC.UPDATES                       # 200
LRS = TC.LRS                               # ("A", 0.003), ("B", 0.01)
VAL_AT = TC.VAL_AT                         # (0, 25, 50, 100, 200)
BATCH_PER_FAMILY = TC.BATCH_PER_FAMILY     # 8
VAL_PER_FAMILY = TC.VAL_PER_FAMILY         # 256
HELDOUT_PER_FAMILY = TC.HELDOUT_PER_FAMILY  # 512
N_DIAG_PER_FAMILY = 32
SOURCE_RUN = TC.SOURCE_RUN
SOURCE_DEV = TC.SOURCE_DEV
SOURCE_FINAL = TC.SOURCE_FINAL
TRAJ32 = TC.TRAJ32
#: fresh streams, asserted disjoint from every range consumed before
STREAM = dict(continuation_train=520_000_000, dev_validation=550_000_000,
              eval_validation=551_000_000, heldout=560_000_000,
              diagnostic=570_000_000)

TSS, GEN, NATIVE, OPERATOR, ANCHOR = TC.TSS, TC.GEN, TC.NATIVE, TC.OPERATOR, \
    TC.ANCHOR
ARMS = TC.ARMS
TRAINED_ARMS = TC.TRAINED_ARMS
LAW_OF, FROZEN_LEAVES, REGIME_OF = TC.LAW_OF, TC.FROZEN_LEAVES, TC.REGIME_OF
ARM_DISPLAY = dict(TC.ARM_DISPLAY)
GEN_MATCHED = "generalized_processing@matched_immediate"
ARM_DISPLAY[GEN_MATCHED] = ("Generalized processing at the development "
                            "checkpoint matched to literal TSS's immediate "
                            "revision (secondary analysis)")
PD.DISPLAY.setdefault(GEN_MATCHED, ARM_DISPLAY[GEN_MATCHED])

#: matched-operating-point band (protocol s4.3), frozen before execution
MATCH_BAND = 0.01
#: deployment feasibility against the native development reference
DEPLOY_MIN_REVISION_GAIN = 0.01

PLANNED = dict(
    trained_runs_development=len(TRAINED_ARMS) * len(LRS),
    development_checkpoints_per_family=len(VAL_AT) * len(LRS) - 1,
    final_trajectories_main=len(TRAINED_ARMS) * len(SOURCE_FINAL),
    final_trajectories_matched_max=len(SOURCE_FINAL),
    frozen_source_evaluations=1 + len(SOURCE_FINAL),
    diagnostic_episodes=2 * N_DIAG_PER_FAMILY)
PLANNED["max_total_updates"] = UPDATES * (
    PLANNED["trained_runs_development"] + PLANNED["final_trajectories_main"]
    + PLANNED["final_trajectories_matched_max"])


def continuation_stream(seed, update):
    return STREAM["continuation_train"] + seed * 10_000 + update


def new_ranges():
    lo, hi = min((SOURCE_DEV,) + SOURCE_FINAL), max((SOURCE_DEV,)
                                                    + SOURCE_FINAL)
    r = dict(continuation_train=(continuation_stream(lo, 0),
                                 continuation_stream(hi, UPDATES - 1)))
    for k in ("dev_validation", "eval_validation", "heldout", "diagnostic"):
        r[k] = (STREAM[k], STREAM[k])
    return r


def previous_ranges():
    prev = list(TC.previous_ranges())
    for lo, hi in TC.new_ranges().values():
        prev.append((lo, hi))
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


# --------------------------------------------------------------- evaluation --
def evaluate_arm(arm, p, eps_np, chunk=128):
    """Declared metrics of `temporal_task.aggregate`, from the SAME compiled
    evaluation batch functions as the containment study; for the processing
    law also the executed coefficient sets and the processing-state
    diagnostic. Only MODEL_INPUTS reach the model."""
    rule = LAW_OF[arm]
    batch_fn = TC.eval_batch_f if rule == FL.FILTERED else ST.eval_batch
    n = eps_np["event"].shape[0]
    cs, ces, wn, an, procs, sink = [], [], [], [], [], []
    for i in range(0, n, chunk):
        sl = {k: jnp.asarray(eps_np[k][i:i + chunk]) for k in TT.MODEL_INPUTS}
        aux = batch_fn(rule, p, sl)
        cs.append(onp.asarray(aux["correct"]))
        ces.append(onp.asarray(aux["ce"]))
        wn.append(onp.asarray(aux["w_norm"]))
        an.append(onp.asarray(aux["aux_norm"]))
        if rule == FL.FILTERED:
            procs.append(onp.asarray(aux["proc_max_abs"]).ravel())
            sink.append(aux)
    m = TT.aggregate(onp.concatenate(cs), onp.concatenate(ces), eps_np)
    wn, an = onp.concatenate(wn), onp.concatenate(an)
    m["state_norms"] = dict(W_frobenius_mean=float(wn.mean()),
                            W_frobenius_max=float(wn.max()),
                            aux_frobenius_mean=float(an.mean()),
                            aux_frobenius_max=float(an.max()))
    sets = None
    if rule == FL.FILTERED:
        proc = onp.concatenate(procs)
        m["processing_state"] = dict(
            max_abs_y_yprev_Rprev=float(proc.max()),
            finite=bool(proc.size and onp.all(onp.isfinite(proc))))
        sets = TC.executed_sets_from(sink)
        m["executed_coefficient_sets"] = sets
    return m, sets


def metrics_finite(m):
    if not TT.metrics_finite(m):
        return False
    return all(onp.isfinite(v) for v in m["state_norms"].values())


def checkpoint_failure(arm, p, opt, source_p, metrics, executed_sets):
    if not ST.all_finite(p):
        return "non-finite parameters"
    if opt is not None and not ST.all_finite(opt):
        return "non-finite optimizer state"
    if not metrics_finite(metrics):
        return "non-finite or empty evaluation cell, aggregate or state norm"
    if LAW_OF[arm] == FL.FILTERED and not (
            metrics.get("processing_state") or {}).get("finite"):
        return "non-finite processing state (y, y_prev, R_prev)"
    changed = TC.frozen_leaf_differences(p, source_p, arm)
    if changed:
        return f"stored constant leaves changed: {changed}"
    return TC.validate(arm, p, executed_sets)


def host_step(arm, p, opt, seed, u, lr, hist):
    """The containment study's steps, on this study's task and streams."""
    rule = LAW_OF[arm]
    eps = ST.to_jax(TT.generate_batch(continuation_stream(seed, u),
                                      BATCH_PER_FAMILY))
    if rule == FL.FILTERED:
        o = TC.train_step_filtered(rule, FROZEN_LEAVES[arm], p, opt, eps, lr)
        p, opt, tel, grads = o[0], o[1], o[8], o[9]
        rec = dict(update=u, n_repaired=int(tel["n_repaired"]),
                   overshoot=float(tel["overshoot"]),
                   jury_min=float(tel["gate_jury_min"]),
                   proc_max_abs=float(tel["gate_proc_max_abs"]),
                   executed_filter_ok=bool(tel["gate_ok"]),
                   executed={k: float(tel["gate_" + k])
                             for k in FL.COEFF_NAMES})
        for k in FL.LEAVES:
            rec[k + "_pre_repair"] = float(tel[k + "_pre"])
            rec[k] = float(tel[k + "_post"])
            rec["grad_" + k] = float(grads[k])
    else:
        o = ST.train_step(rule, p, opt, eps, lr)
        p, opt, tel = o[0], o[1], o[8]
        rec = dict(update=u, n_projected=int(tel["n_projected"]),
                   pre=float(tel["pre"]), post=float(tel["post"]),
                   cap=float(tel["cap"]), bound=float(tel["bound"]),
                   overshoot=float(tel["overshoot"]), grad=float(o[9]),
                   executed_filter_ok=True)
    scalars = dict(zip(ST.SCALAR_NAMES, (float(x) for x in o[2:8])))
    hist.append(rec)
    return p, opt, scalars, rec


def _summary(m):
    """Compact per-checkpoint record: study aggregates and per-family parts
    and delay curves (full cells are kept for endpoints)."""
    return dict(
        primary=m["primary"], revision_ce=m["revision_ce"],
        retention=m["retention_revision_untouched"],
        recall=m["recall_overall"], immediate_revision=m["immediate_revision"],
        later=m["later"], later_parts=m["later_parts"],
        parts={f: m[f]["parts"] for f in FAMILIES},
        by_delay={f: m[f]["by_delay"] for f in FAMILIES})


# ------------------------------------------------------------------ one run --
def run_one(arm, tag, lr_value, seed, source_p, val_np, updates, out,
            deadline, reserve_s, status, stage, source_label, save=None,
            val_at=None, keep_at=(), step_fn=None, step_failure_fn=None):
    """One trajectory of `updates` updates (possibly 0) or the frozen anchor.
    Every checkpoint is evaluated, accepted and persisted before the next
    update; the parameter trees at `keep_at` updates are returned for
    endpoint evaluation. Returns (record, final params, {update: params}) or
    None when the time reserve is reached.

    `step_fn` and `step_failure_fn` are additive hooks (retention-aware
    study): a study may supply its own host step and per-step acceptance with
    the same signatures as `host_step` and `tss_containment.step_failure`.
    The defaults leave this study's behaviour unchanged."""
    rule, regime = LAW_OF[arm], REGIME_OF[arm]
    t0 = time.time()
    p = dict(source_p)
    grid = VAL_AT if val_at is None else tuple(val_at)
    ckpts = tuple(u for u in grid if u <= updates)
    if updates not in ckpts:
        ckpts = ckpts + (updates,)
    missing = [u for u in keep_at if u not in ckpts]
    if missing:
        raise ValueError(f"keep_at {missing} are not checkpoints {ckpts}")
    stem = f"{stage}_{arm}_{tag}_seed{seed}"
    log = status.setdefault("checkpoint_log", [])
    save = save or (lambda: None)
    opt = None if regime == "frozen" else ST.TX.init(p)
    val_hist, hist, curve, kept, bad = [], [], [], {}, None
    for u in range(updates + 1):
        if u in ckpts:
            m, sets = evaluate_arm(arm, p, val_np)
            fail = checkpoint_failure(arm, p, opt, source_p, m, sets)
            pfile = os.path.join(out, "params", f"{stem}_u{u}.msgpack")
            ST.save_tree(pfile, p)
            ofile = None
            if opt is not None:
                ofile = os.path.join(out, "params",
                                     f"{stem}_u{u}_opt.msgpack")
                ST.save_tree(ofile, opt)
            entry = dict(update=u, **_summary(m),
                         state_norms=m["state_norms"],
                         processing_state=m.get("processing_state"),
                         executed_filter=([FL.filter_report(ex)
                                           for ex in sets] if sets else None),
                         accepted=fail is None, failure=fail,
                         params_file=pfile, opt_file=ofile)
            val_hist.append(dict(entry, full=m))
            log.append(dict({k: v for k, v in entry.items()
                             if k not in ("parts", "by_delay")},
                            stage=stage, arm=arm, config=tag, lr=lr_value,
                            seed=seed))
            save()
            if fail:
                bad = f"checkpoint at update {u}: {fail}"
                break
            if u in keep_at:
                kept[u] = dict(p)
        if u == updates:
            break
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{stage}:{arm}/{tag}/seed{seed} stopped at update {u} of "
                f"{updates}")
            return None
        lr = jnp.asarray(lr_value, dtype=jnp.float32)
        p, opt, scalars, rec = (step_fn or host_step)(arm, p, opt, seed, u,
                                                      lr, hist)
        bad = (step_failure_fn or TC.step_failure)(arm, scalars, rec)
        if bad:
            break
        if u % 25 == 0 or u == updates - 1:
            curve.append(dict(update=u, **scalars))
    last = val_hist[-1] if val_hist else None
    if bad is None and not (last and last["update"] == updates):
        bad = "endpoint checkpoint missing"
    rec = dict(
        tag=stage, rule=arm, law=rule, regime=regime,
        display=ARM_DISPLAY[arm], config=tag, lr=lr_value, seed=seed,
        source=source_label, wall_s=time.time() - t0, curve=curve,
        updates=updates,
        validation=[{k: v for k, v in h.items() if k != "full"}
                    for h in val_hist],
        final_validation=(last["full"] if last and bad is None else None),
        params=TC.parameter_counts(arm, p), carry=PD.CARRY[rule],
        coefficient_history=hist,
        frozen_leaf_differences=TC.frozen_leaf_differences(p, source_p, arm),
        invalid=bad)
    return rec, p, kept


# ---------------------------------------------------------------- preflight --
def preflight(sources, val_np, out, status):
    """Times the ACTUAL paths on disposable state: every measured step is
    checked; one full checkpoint (evaluation, acceptance, persistence) and one
    evaluation are timed per arm; the projection covers the worst case of
    every declared trajectory, checkpoint and held-out evaluation."""
    rows, failures, retraced_any = [], [], False
    p0 = sources[SOURCE_DEV]
    lr = jnp.asarray(LRS[0][1], dtype=jnp.float32)
    step_s, ckpt_s, eval_s = {}, {}, {}
    scratch = os.path.join(out, "preflight_disposable")
    for arm, rule, frozen, _ in TRAINED_ARMS:
        p = TC.start_tree(arm, p0)
        opt = ST.TX.init(p)
        hist, acc_bad = [], None
        t0 = time.time()
        p2, opt2, sc, rec = host_step(arm, p, opt, SOURCE_DEV, 0, lr, hist)
        compile_s = time.time() - t0
        acc_bad = acc_bad or TC.step_failure(arm, sc, rec)
        p2, opt2, sc, rec = host_step(arm, p2, opt2, SOURCE_DEV, 1, lr, hist)
        acc_bad = acc_bad or TC.step_failure(arm, sc, rec)
        cache = (TC.train_step_filtered if rule == FL.FILTERED
                 else ST.train_step)
        n0 = cache._cache_size()
        t1 = time.time()
        for u in range(2, 7):
            p2, opt2, sc, rec = host_step(arm, p2, opt2, SOURCE_DEV, u, lr,
                                          hist)
            acc_bad = acc_bad or TC.step_failure(arm, sc, rec)
        step_s[arm] = (time.time() - t1) / 5.0
        retraced_any |= cache._cache_size() != n0
        t2 = time.time()
        evaluate_arm(arm, p2, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        m, sets = evaluate_arm(arm, p2, val_np)
        eval_s[arm] = time.time() - t3
        fail = checkpoint_failure(arm, p2, opt2, p, m, sets)
        ST.save_tree(os.path.join(scratch, f"{arm}.msgpack"), p2)
        ST.save_tree(os.path.join(scratch, f"{arm}_opt.msgpack"), opt2)
        ckpt_s[arm] = time.time() - t3
        acc_bad = acc_bad or fail
        if acc_bad:
            failures.append(f"{arm}: {acc_bad}")
        rows.append(dict(arm=arm, law=rule, compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s,
                         step_s=step_s[arm], checkpoint_s=ckpt_s[arm],
                         evaluation_s=eval_s[arm],
                         measured_steps_checked=len(hist),
                         acceptance_failure=acc_bad,
                         params=TC.parameter_counts(arm, p2),
                         carry=PD.CARRY[rule]))
        print(f"[preflight] {arm:<24} step {step_s[arm] * 1e3:6.2f}ms "
              f"checkpoint {ckpt_s[arm]:5.2f}s eval {eval_s[arm]:5.2f}s "
              f"compile {compile_s:4.1f}s failure {acc_bad}")
    heldout_factor = HELDOUT_PER_FAMILY / VAL_PER_FAMILY
    per_traj = {a: UPDATES * step_s[a] + len(VAL_AT) * ckpt_s[a]
                for a, _, _, _ in TRAINED_ARMS}
    total = 0.0
    for arm, _, _, _ in TRAINED_ARMS:
        total += (len(LRS) + len(SOURCE_FINAL)) * per_traj[arm]
        total += len(SOURCE_FINAL) * heldout_factor * eval_s[arm]
    total += len(SOURCE_FINAL) * (per_traj[GEN]
                                  + heldout_factor * eval_s[GEN])
    anchor = max(ckpt_s.values())
    total += (1 + len(SOURCE_FINAL)) * anchor \
        + len(SOURCE_FINAL) * heldout_factor * max(eval_s.values())
    host_s = 40.0
    total += host_s
    timing = dict(projected_remaining_s=total, host_allowance_s=host_s,
                  heldout_evaluation_factor=heldout_factor)
    if not all(onp.isfinite(v) and v >= 0 for v in timing.values()):
        failures.append(f"non-finite or negative timing {timing}")
    status["preflight"] = dict(
        rows=rows, retraced_any=bool(retraced_any), failures=failures,
        planned=planned_work(), **timing,
        coverage=("projected at the worst case: every development and main "
                  "final trajectory at 200 updates with all five checkpoints, "
                  "the matched-operating-point trajectories as if required, "
                  "every held-out endpoint evaluation scaled from the measured "
                  "evaluation, and the frozen anchor. The mechanism "
                  "diagnostic and start checks have already consumed their "
                  "time before this projection."))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, bool(retraced_any), failures


# ---------------------------------------------------------------- selection --
def checkpoints(dev_rows, arm):
    """Accepted development checkpoints with the timing coordinates; the
    identical update-zero checkpoint is kept once (slot A)."""
    out = []
    for r in dev_rows:
        if r["rule"] != arm:
            continue
        for v in r["validation"]:
            if v["update"] == 0 and r["config"] != LRS[0][0]:
                continue
            out.append(dict(arm=arm, config=r["config"], lr=r["lr"],
                            update=v["update"], primary=v["primary"],
                            revision_ce=v["revision_ce"],
                            retention=v["retention"], recall=v["recall"],
                            immediate_revision=v["immediate_revision"],
                            later=v["later"], accepted=v.get("accepted"),
                            params_file=v.get("params_file")))
    return out


def ineligible(cand):
    keys = ("primary", "revision_ce", "retention", "recall",
            "immediate_revision", "later")
    return [c for c in cand if c.get("accepted") is not True
            or not all(c[k] is not None and onp.isfinite(c[k]) for k in keys)]


def select_main(dev_rows, status):
    """Every trained family by the SAME unconstrained development ordering
    (revision macro accuracy, lower revision CE, fewer updates, lower
    learning rate). Retention and recall are NOT selection constraints here;
    they are held-out screen conditions. Deployment feasibility is recorded
    separately."""
    sel, table = {}, []
    for arm, _, _, _ in TRAINED_ARMS:
        cand = checkpoints(dev_rows, arm)
        if len(cand) != PLANNED["development_checkpoints_per_family"] \
                or ineligible(cand):
            return None, None
        best = sorted(cand, key=TC.order_key)[0]
        sel[arm] = dict(best, diagnostic=False,
                        selected_under=("unconstrained development ordering, "
                                        "identical for every family"))
        table.append(dict(arm=arm, chosen=best, n_candidates=len(cand)))
    nat = sel[NATIVE]
    for arm in (TSS, GEN, OPERATOR):
        s = sel[arm]
        s["feasible"] = bool(
            s["primary"] >= nat["primary"] + DEPLOY_MIN_REVISION_GAIN
            and s["retention"] >= nat["retention"]
            and s["recall"] >= nat["recall"])
    sel[NATIVE]["feasible"] = True
    status["selection"] = dict(
        selected=sel, table=table,
        rule=("each trained family: highest development revision macro "
              "accuracy, then lower revision cross-entropy, then fewer "
              "updates, then lower learning rate, over accepted checkpoints; "
              "no retention or recall constraint in selection"),
        deployment_feasibility=(
            "deployment only: development revision >= native + 1 pp AND "
            "retention >= native AND recall >= native (the selected native "
            "checkpoint's development values)"))
    plan = TC.deployment_plan(sel, status)
    return sel, plan


def select_matched(dev_rows, sel, status):
    """Frozen matched-operating-point rule (protocol s4.3). Reference: the
    SELECTED literal-TSS development checkpoint's immediate revision I_ref.
    Feasible generalized checkpoints: I_ref <= immediate_revision <= I_ref +
    MATCH_BAND. Among them: highest development `later`, then immediate
    revision closest to I_ref, then fewer updates, then lower learning rate.
    No feasible checkpoint: reported as unavailable, never relaxed."""
    ref = sel[TSS]
    i_ref = ref["immediate_revision"]
    cand = checkpoints(dev_rows, GEN)
    feas = [c for c in cand
            if i_ref <= c["immediate_revision"] <= i_ref + MATCH_BAND]
    rec = dict(reference=dict(arm=TSS, config=ref["config"], lr=ref["lr"],
                              update=ref["update"], immediate_revision=i_ref,
                              later=ref["later"]),
               band=[i_ref, i_ref + MATCH_BAND], n_candidates=len(cand),
               n_feasible=len(feas),
               rule=("I_ref <= generalized development immediate revision "
                     "<= I_ref + 0.01; then highest development later, "
                     "then closest immediate revision, then fewer updates, "
                     "then lower learning rate"))
    if not feas:
        rec.update(available=False,
                   reason=("no generalized development checkpoint has "
                           "immediate revision within the frozen band; "
                           "reported, not relaxed"))
        status["matched_operating_point"] = rec
        return None
    best = sorted(feas, key=lambda c: (-c["later"],
                                       c["immediate_revision"] - i_ref,
                                       c["update"], c["lr"]))[0]
    rec.update(available=True, chosen=best,
               same_as_main=bool(best["config"] == sel[GEN]["config"]
                                 and best["update"] == sel[GEN]["update"]))
    status["matched_operating_point"] = rec
    return best


def final_plan(sel, matched):
    """One trajectory per (family, learning rate, final source), run to the
    largest update any endpoint needs at that learning rate; endpoints are
    that trajectory's checkpoints. Deterministic training makes an endpoint
    at update u identical to a trajectory stopped at u."""
    needs = defaultdict(set)
    roles = defaultdict(list)
    for arm, _, _, _ in TRAINED_ARMS:
        s = sel[arm]
        needs[(arm, s["config"], s["lr"])].add(s["update"])
        roles[(arm, s["config"], s["lr"])].append((arm, s["update"]))
    if matched is not None:
        key = (GEN, matched["config"], matched["lr"])
        needs[key].add(matched["update"])
        roles[key].append((GEN_MATCHED, matched["update"]))
    plan = []
    for (arm, config, lr), ups in needs.items():
        plan.append(dict(arm=arm, config=config, lr=lr,
                         updates=max(ups), keep_at=sorted(ups),
                         endpoints=sorted(roles[(arm, config, lr)])))
    return sorted(plan, key=lambda x: (x["arm"], x["config"]))


# ------------------------------------------------------------------- screens --
COMPARISONS = (
    ("generalized_versus_literal_tss", GEN, TSS, "primary"),
    ("generalized_versus_native", GEN, NATIVE, "primary"),
    ("generalized_versus_learned_operator", GEN, OPERATOR,
     "primary (the operator is outside the nonnegative-gamma family)"),
    ("literal_tss_versus_native", TSS, NATIVE, "reference"),
    ("learned_operator_versus_native", OPERATOR, NATIVE, "reference"),
    ("generalized_versus_frozen_source", GEN, ANCHOR, "descriptive"),
)


def _per_seed(rows, arm, fn):
    return {r["seed"]: fn(r["heldout"]) for r in rows if r["rule"] == arm}


def cell_differences(rows, a, b, seeds=SOURCE_FINAL):
    """SECONDARY: paired mean differences per family x kind x delay x
    condition and late cells, plus by-delay curves. Cannot rescue a failed
    aggregate verdict."""
    out = {}
    ra = {r["seed"]: r["heldout"] for r in rows if r["rule"] == a}
    rb = {r["seed"]: r["heldout"] for r in rows if r["rule"] == b}
    if not all(s in ra and s in rb for s in seeds):
        return None
    for fam in FAMILIES:
        cells = {}
        for name in ra[seeds[0]][fam]["cells"]:
            d = [ra[s][fam]["cells"][name]["accuracy"]
                 - rb[s][fam]["cells"][name]["accuracy"] for s in seeds]
            cells[name] = dict(mean=float(onp.mean(d)), per_seed=d)
        by_delay = {}
        for kind in ("revised_probe", "untouched_probe"):
            by_delay[kind] = {}
            for dl in ra[seeds[0]][fam]["by_delay"][kind]:
                d = [ra[s][fam]["by_delay"][kind][dl]
                     - rb[s][fam]["by_delay"][kind][dl] for s in seeds]
                by_delay[kind][dl] = dict(mean=float(onp.mean(d)),
                                          per_seed=d)
        out[fam] = dict(cells=cells, by_delay=by_delay)
    for key in ("immediate_revision", "later"):
        d = [ra[s][key] - rb[s][key] for s in seeds]
        out[key] = dict(mean=float(onp.mean(d)), per_seed=d)
    for key in ra[seeds[0]]["later_parts"]:
        d = [ra[s]["later_parts"][key] - rb[s]["later_parts"][key]
             for s in seeds]
        out[f"later_part/{key}"] = dict(mean=float(onp.mean(d)), per_seed=d)
    return out


def screen(final_rows, seeds=SOURCE_FINAL):
    out = dict(
        rule=("PRIMARY AGGREGATE SCREEN: mean held-out revision difference "
              ">= +1 pp, paired revision differences positive in all final "
              "seeds, mean retention difference >= 0 AND mean recall "
              "difference >= 0"),
        secondary=("delay- and condition-specific differences are secondary "
                   "and cannot rescue a failed aggregate verdict"),
        comparisons=[])
    for name, a, b, kind in COMPARISONS:
        c = ST.compare(final_rows, a, b, seeds)
        c["safeguard_passed_minus_one_pp"] = c.pop("passed")
        c.update(name=name, candidate=a, kind=kind,
                 retention_direction=RP.direction(c["retention_difference"]),
                 recall_direction=RP.direction(c["recall_difference"]))
        c["aggregate_screen_passed"] = bool(
            c["complete_paired_seeds"]
            and c["mean_primary_difference"] >= 0.01
            and c["positive_in_all_seeds"]
            and c["retention_difference"] >= 0
            and c["recall_difference"] >= 0)
        c["secondary_cells"] = cell_differences(final_rows, a, b, seeds)
        out["comparisons"].append(c)
    return out


#: wording of the existing descriptive flag (review R2); its test is unchanged
DESCRIPTIVE_FLAG_MEANING = ("later improved in every seed with no mean "
                            "immediate decrease")
ATTRIBUTION_NOTE = (
    "Even a held-out accuracy match does not match the immediate amplitude c, "
    "the gates, the learned backbone or the actual write matrices. The "
    "trained comparison can support an operating trade-off; it cannot by "
    "itself attribute that trade-off solely to response timing. The frozen-"
    "backbone, equal-first-write diagnostic is the controlled mechanism "
    "comparison and is separate from learned-performance evidence.")


def heldout_immediate_match(diff, band=MATCH_BAND):
    """Whether the held-out immediate-revision difference lies in the ALREADY
    DECLARED band [0, band], per seed and for the mean, reported separately.
    Reporting only: nothing is reselected, widened or suppressed."""
    if diff is None:
        return None
    per = diff["immediate_revision"]["per_seed"]
    mean = diff["immediate_revision"]["mean"]
    return dict(band=[0.0, band],
                per_seed_in_band=[bool(0.0 <= x <= band) for x in per],
                all_seeds_in_band=bool(all(0.0 <= x <= band for x in per)),
                mean_in_band=bool(0.0 <= mean <= band),
                per_seed_difference=per, mean_difference=mean)


def descriptive_flag(diff):
    """The unchanged descriptive flag: mean held-out immediate difference >= 0
    AND later difference > 0 in every seed. It does NOT establish a held-out
    match and does not isolate timing from stronger writing."""
    return bool(diff is not None and diff["immediate_revision"]["mean"] >= 0
                and all(x > 0 for x in diff["later"]["per_seed"]))


def timing_analysis(final_rows, matched, status, seeds=SOURCE_FINAL):
    """SECONDARY, declared (protocol s4.3). For each generalized endpoint
    against the selected literal-TSS endpoint: I and L differences, the
    unchanged descriptive flag, the held-out immediate match against the
    declared band reported separately, and the executed immediate amplitude
    c. The matched endpoint is labelled DEVELOPMENT-matched; whether it stays
    matched on held-out data is measured, not assumed."""
    def amp(rows, arm):
        return {r["seed"]: [ex["c"] for ex in (r.get("heldout", {}).get(
            "executed_coefficient_sets") or [])]
            for r in rows if r["rule"] == arm}

    main = cell_differences(final_rows, GEN, TSS, seeds)
    res = dict(
        main_endpoints=dict(
            label="primary-selected generalized endpoint (not matched)",
            differences=main,
            later_improved_at_no_worse_immediate=descriptive_flag(main),
            descriptive_flag_meaning=DESCRIPTIVE_FLAG_MEANING,
            heldout_immediate_match=heldout_immediate_match(main),
            executed_immediate_amplitude_c=dict(generalized=amp(final_rows,
                                                                GEN),
                                                literal_tss=amp(final_rows,
                                                                TSS))),
        matched_operating_point=status.get("matched_operating_point"))
    if matched is not None:
        mt = cell_differences(final_rows, GEN_MATCHED, TSS, seeds)
        hm = heldout_immediate_match(mt)
        res["matched_endpoints"] = dict(
            label=("DEVELOPMENT-matched generalized endpoint (matched to the "
                   "primary-selected literal-TSS endpoint's development "
                   "immediate revision)"),
            differences=mt,
            later_improved_at_no_worse_immediate=descriptive_flag(mt),
            descriptive_flag_meaning=DESCRIPTIVE_FLAG_MEANING,
            heldout_immediate_match=hm,
            heldout_match_statement=(
                None if hm is None else
                ("development match holds on held-out data in every seed"
                 if hm["all_seeds_in_band"] else
                 "development match does NOT hold on held-out data in every "
                 "seed; reported without reselection, band widening or "
                 "suppression")),
            executed_immediate_amplitude_c=amp(final_rows, GEN_MATCHED))
    res["attribution"] = ATTRIBUTION_NOTE
    res["scope"] = ("secondary: these records describe timing and cannot "
                    "rescue a failed aggregate screen. Immediate revision I "
                    "pools revised queries at actual offsets 1 and 2 (nominal "
                    "delay 1), not exclusively one-token accuracy.")
    return res


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/prospective-temporal-response")
    ap.add_argument("--source_run", default=SOURCE_RUN)
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
        raise SystemExit("REFUSING: x64 is enabled; production is float32.")
    overlaps = stream_overlaps()
    if overlaps:
        raise SystemExit(f"REFUSING: stream ranges overlap: {overlaps}")
    if os.path.realpath(args.out_root).startswith(
            os.path.realpath(args.source_run)):
        raise SystemExit("REFUSING: output inside the read-only source run")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    status = dict(
        run_id=run_id, out=out, backend=backend,
        study=("temporal response: generalized prospective processing versus "
               "literal TSS on the same Momentum backbone"),
        protocol="docs/PROSPECTIVE_TEMPORAL_RESPONSE_PROTOCOL.md",
        source_run=args.source_run,
        source_seeds=dict(development=SOURCE_DEV, final=list(SOURCE_FINAL)),
        final_seeds=list(SOURCE_FINAL), updates=UPDATES,
        checkpoints_at=list(VAL_AT),
        arms={a: ARM_DISPLAY[a] for a, _, _, _ in ARMS},
        arm_law={a: law for a, law, _, _ in ARMS},
        arm_frozen_leaves={a: list(f) for a, _, f, _ in ARMS},
        task=dict(seq_len=TT.SEQ_LEN, delays=list(TT.DELAYS),
                  conditions=list(TT.CONDITIONS), targets=TT.N_TARGETS,
                  selected=TT.N_SELECTED, queries=TT.N_QUERIES,
                  fill_tokens=TT.N_FILL, kinds=list(TT.KINDS),
                  later_delays=list(TT.LATER_DELAYS),
                  immediate_delay=TT.IMMEDIATE_DELAY),
        match_band=MATCH_BAND,
        planned_work=planned_work(), streams=STREAM,
        stream_ranges=new_ranges(), previous_stream_ranges=previous_ranges(),
        learning_rates=dict(LRS),
        production_dtype=dict(x64=False, float_dtype="float32"),
        completed_study_references=[
            SOURCE_RUN,
            "/Users/durso/s5-runs/prospective-tss-containment/20260917-154035"],
        heldout_policy=("one common held-out set, generated only after every "
                        "selection and final endpoint is frozen"),
        source=dict(hashes_at_restore=None), checkpoint_log=[],
        development=[], final_trajectories=[], final=[], incomplete=[])
    status_path = os.path.join(out, "status.json")

    def persist(st):
        st["wall_s"] = time.time() - t0
        ST.write(status_path, st)

    def rehash():
        keys = (status.get("source") or {}).get("hashes_at_restore")
        if not keys:
            return {}
        d = RS.sources_dir(args.source_run)
        return {k: (RS.sha256(os.path.join(d, k))
                    if os.path.isfile(os.path.join(d, k)) else None)
                for k in keys}

    signal.signal(signal.SIGTERM, ST._on_sigterm)
    code, _ = ST.guarded(lambda: run_study(args, status, out, deadline,
                                           lambda: persist(status)),
                         status, rehash, persist)
    return code


def run_study(args, status, out, deadline, save):
    val_np = TT.generate_batch(STREAM["dev_validation"], VAL_PER_FAMILY)
    eval_val_np = TT.generate_batch(STREAM["eval_validation"], VAL_PER_FAMILY)
    diag_np = TT.generate_batch(STREAM["diagnostic"], N_DIAG_PER_FAMILY)
    train0 = TT.generate_batch(continuation_stream(SOURCE_DEV, 0),
                               BATCH_PER_FAMILY)
    checks = dict(dev_validation=TT.structure_check(val_np),
                  eval_validation=TT.structure_check(eval_val_np),
                  diagnostic=TT.structure_check(diag_np),
                  first_training_batch=TT.structure_check(train0))
    status["task_checks"] = dict(
        checks, dev_validation_digest=TT.episode_digest(val_np),
        eval_validation_digest=TT.episode_digest(eval_val_np),
        diagnostic_digest=TT.episode_digest(diag_np))
    bad_task = [n for n, c in checks.items()
                if not (c["oracle_exact"] and c["block_cells_balanced"]
                        and c["query_value_field_is_absent"]
                        and c["idle_value_field_is_absent"]
                        and c["every_query_has_a_label"]
                        and c["no_label_outside_queries"]
                        and c["every_query_has_a_kind"]
                        and c["queries_per_sequence"] == [TT.N_QUERIES]
                        and c["length"] == TT.SEQ_LEN)]
    save()
    if bad_task:
        status["failed"] = f"task structure checks failed: {bad_task}"
        print(f"[!] {status['failed']}")
        return 4, "FAILED"

    try:
        sources = TC.load_sources(args.source_run, status)
    except RS.SourceRefusal as e:
        status["failed"] = f"source refusal: {e}"
        print(f"[!] {status['failed']}")
        return 4, "FAILED"
    print(f"[source] reused {len(sources)} read-only Momentum checkpoints")
    save()

    # start points: storage identity, direct native-point recovery on this
    # task, and executed-coefficient acceptance of the literal-TSS start
    starts, bad = {}, []
    for seed, pn in sources.items():
        tss_p, gen_p = TC.start_tree(TSS, pn), TC.start_tree(GEN, pn)
        same = all(onp.array_equal(onp.asarray(tss_p[k]),
                                   onp.asarray(gen_p[k])) for k in tss_p)
        fails, rows = TC.direct_recovery(pn, val_np)
        _, sets = evaluate_arm(TSS, tss_p, val_np)
        start_fail = TC.validate(TSS, tss_p, sets)
        starts[str(seed)] = dict(
            storage_identity_of_the_two_processing_start_trees=bool(same),
            direct_native_point_recovery_failures=fails,
            direct_native_point_recovery=rows,
            start_acceptance_failure=start_fail)
        if not same or fails or start_fail:
            bad.append(seed)
    status["start_points"] = starts
    save()
    if bad:
        status["failed"] = f"start-point checks failed for seeds {bad}"
        print(f"[!] {status['failed']}")
        return 4, "FAILED"

    # controlled mechanism diagnostic, before any training
    t_diag = time.time()
    diag_fails, diag = TD.run(sources[SOURCE_DEV], diag_np)
    diag["wall_s"] = time.time() - t_diag
    diag["failures"] = diag_fails
    status["mechanism_diagnostic"] = diag
    save()
    print(f"[diagnostic] failures={diag_fails} wall {diag['wall_s']:.1f}s")
    if diag_fails:
        status["failed"] = f"mechanism diagnostic failed: {diag_fails}"
        return 4, "FAILED"

    proj, retraced, failures = preflight(sources, val_np, out, status)
    save()
    d = ST.decide_after_preflight(proj, retraced, failures,
                                  deadline - time.time() - args.reserve_s)
    if d is not None:
        code, label, why = d
        (status.__setitem__("failed", why) if code == 4
         else status["incomplete"].append(why))
        print(f"[!] {why}")
        return code, label

    def tree_of(arm, seed):
        e = status["source"]["entries"][seed]
        return (TC.start_tree(arm, sources[seed]),
                f"{e['file']} sha256={e['sha256']}")

    dev_rows = []
    for arm, _, _, _ in TRAINED_ARMS:
        for tag, lr in LRS:
            p, label = tree_of(arm, SOURCE_DEV)
            r = run_one(arm, tag, lr, SOURCE_DEV, p, val_np, UPDATES, out,
                        deadline, args.reserve_s, status, "dev", label, save)
            if r is None:
                return 3, "INCOMPLETE"
            rec = r[0]
            dev_rows.append(rec)
            status["development"] = dev_rows
            save()
            if rec["invalid"]:
                status["failed"] = f"dev {arm}/{tag}: {rec['invalid']}"
                print(f"[!] {status['failed']}")
                return 4, "FAILED"
    p, label = tree_of(ANCHOR, SOURCE_DEV)
    r = run_one(ANCHOR, "-", 0.0, SOURCE_DEV, p, val_np, 0, out, deadline,
                args.reserve_s, status, "dev", label, save)
    if r is None:
        return 3, "INCOMPLETE"
    dev_rows.append(r[0])
    status["development"] = dev_rows
    if r[0]["invalid"]:
        status["failed"] = f"dev anchor: {r[0]['invalid']}"
        return 4, "FAILED"

    sel, plan = select_main(dev_rows, status)
    if sel is None:
        status["failed"] = ("selection could not be formed: a missing, "
                            "non-finite or unaccepted development checkpoint")
        return 4, "FAILED"
    matched = select_matched(dev_rows, sel, status)
    fplan = final_plan(sel, matched)
    frozen = copy.deepcopy(dict(selection=sel, deployment=plan,
                                matched=matched, final_plan=fplan))
    status["frozen_before_finals"] = copy.deepcopy(frozen)
    status["final_plan"] = fplan
    ST.write(os.path.join(out, "selection.json"), frozen)
    for arm, _, _, _ in TRAINED_ARMS:
        print(f"[selection] {arm:<24} lr {sel[arm]['lr']} update "
              f"{sel[arm]['update']:>3} immediate "
              f"{sel[arm]['immediate_revision']:.4f} later "
              f"{sel[arm]['later']:.4f} deployable={sel[arm]['feasible']}")
    print(f"[matched] available="
          f"{status['matched_operating_point']['available']} chosen="
          f"{matched and (matched['config'], matched['update'])}")
    save()

    final_rows, traj_rows, endpoints = [], [], {}
    for tr in fplan:
        for seed in SOURCE_FINAL:
            p, label = tree_of(tr["arm"], seed)
            r = run_one(tr["arm"], tr["config"], tr["lr"], seed, p,
                        eval_val_np, tr["updates"], out, deadline,
                        args.reserve_s, status, "final", label, save,
                        keep_at=tuple(tr["keep_at"]))
            if r is None:
                status["heldout_opened"] = False
                return 3, "INCOMPLETE"
            rec, _, kept = r
            traj_rows.append(rec)
            status["final_trajectories"] = traj_rows
            save()
            if rec["invalid"]:
                status["failed"] = (f"final {tr['arm']}/{tr['config']}/"
                                    f"{seed}: {rec['invalid']}")
                print(f"[!] {status['failed']}")
                return 4, "FAILED"
            by_update = {v["update"]: v for v in rec["validation"]}
            for role, u in tr["endpoints"]:
                v = by_update[u]
                row = dict(rule=role, law=LAW_OF[tr["arm"]],
                           trained_family=tr["arm"], seed=seed,
                           config=tr["config"], lr=tr["lr"], update=u,
                           endpoint_kind=("zero_update_named_family_endpoint"
                                          if u == 0 else
                                          "trained_named_family_endpoint"),
                           endpoint_params_file=v["params_file"],
                           final_validation_summary={
                               k: v[k] for k in
                               ("primary", "retention", "recall",
                                "immediate_revision", "later")},
                           source=label)
                final_rows.append(row)
                endpoints[(role, seed)] = kept[u]
    for seed in SOURCE_FINAL:
        p, label = tree_of(ANCHOR, seed)
        r = run_one(ANCHOR, "-", 0.0, seed, p, eval_val_np, 0, out, deadline,
                    args.reserve_s, status, "final", label, save)
        if r is None:
            status["heldout_opened"] = False
            return 3, "INCOMPLETE"
        if r[0]["invalid"]:
            status["failed"] = f"final anchor {seed}: {r[0]['invalid']}"
            return 4, "FAILED"
        traj_rows.append(r[0])
        final_rows.append(dict(rule=ANCHOR, law="momentum_delta",
                               trained_family=ANCHOR, seed=seed, config="-",
                               lr=0.0, update=0,
                               endpoint_kind="frozen_source_anchor",
                               endpoint_params_file=r[0]["validation"][0][
                                   "params_file"], source=label))
        endpoints[(ANCHOR, seed)] = r[1]
    status["final"] = final_rows
    now = copy.deepcopy(dict(selection=status["selection"]["selected"],
                             deployment=status["deployment_plan"],
                             matched=matched, final_plan=fplan))
    ref = dict(selection=frozen["selection"], deployment=frozen["deployment"],
               matched=frozen["matched"], final_plan=frozen["final_plan"])
    if now != ref:
        status["failed"] = "selection or final plan changed during finals"
        return 4, "FAILED"

    status["heldout_opened"] = True
    status["heldout_opened_at"] = time.time()
    status["heldout_evaluation_complete"] = False
    save()
    held_np = TT.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    status["task_checks"]["heldout"] = TT.structure_check(held_np)
    status["task_checks"]["heldout_digest"] = TT.episode_digest(held_np)
    for row in final_rows:
        arm = row["trained_family"]
        m, sets = evaluate_arm(arm, endpoints[(row["rule"], row["seed"])],
                               held_np)
        row["heldout"] = m
        if not metrics_finite(m) or (
                LAW_OF[arm] == FL.FILTERED and (
                    not m["processing_state"]["finite"]
                    or any(FL.filter_failure(FL.filter_report(ex))
                           for ex in sets))):
            status["failed"] = (f"non-finite or unaccepted held-out "
                                f"evaluation {row['rule']}/{row['seed']}")
            return 4, "FAILED"
    status["heldout_evaluation_complete"] = True
    main_rows = [r for r in final_rows if r["rule"] != GEN_MATCHED]
    status["screen"] = screen(main_rows)
    status["timing_analysis"] = timing_analysis(final_rows, matched, status)
    status["deployment_outcome"] = TC.deployment_outcome(main_rows, plan)
    rows = dev_rows + traj_rows
    status["work_completed"] = dict(
        development_runs=len([r for r in dev_rows
                              if r["regime"] != "frozen"]),
        final_trajectories=len([r for r in traj_rows
                                if r["regime"] != "frozen"]),
        final_endpoints=len([r for r in final_rows if r["rule"] != ANCHOR]),
        frozen_source_evaluations=len([r for r in rows
                                       if r["regime"] == "frozen"]),
        checkpoints_validated_and_persisted=len(status["checkpoint_log"]),
        total_updates=sum(r["updates"] for r in rows))
    for c in status["screen"]["comparisons"]:
        print(f"[screen] {c['name']:<38} {c['kind'][:9]:<9} "
              f"aggregate_passed={c['aggregate_screen_passed']} mean "
              f"{c['mean_primary_difference']}")
    status["complete"] = True
    return 0, "PASS"


if __name__ == "__main__":
    sys.exit(main())
