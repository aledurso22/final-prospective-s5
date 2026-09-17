"""Same backbone: candidate versus ordinary prospective operator, and
full versus coefficient-only continuation.

Protocol, frozen for review before any execution:
docs/PROSPECTIVE_SAME_BACKBONE_PROTOCOL.md. NOT AUTHORIZED TO RUN until
cleared. Audit: docs/PROSPECTIVE_SAME_BACKBONE_AUDIT.md. Brief:
PROSPECTIVE_SAME_BACKBONE_AND_RETENTION_BRIEF_2026_09_17.md.

Two questions:

1. does the candidate provide anything beyond ordinary first-order
   prospective residual processing on the SAME Momentum memory? (The audit
   proves they are the same law for constant eta and mu; only token-varying
   gates can separate them.)
2. can the revision gain survive while the pretrained backbone is held fixed
   and only the prospective coefficient trains?

Six arms, three laws x two training regimes (the frozen native needs no
training):

    prospective_full    candidate, backbone and kappa train together
    prospective_coeff   candidate, ONLY kappa trains (backbone bitwise fixed)
    ordinary_full       operator, backbone and kappa train together
    ordinary_coeff      operator, ONLY kappa trains
    native_full         native Momentum, backbone trains (no extra scalar)
    native_frozen       the restored source itself: evaluation only, anchor

Sources are the completed replication's independently pretrained Momentum
checkpoints, reused READ-ONLY (seed 500 development, 501-503 final). Full
BPTT and the existing unweighted query cross-entropy are unchanged; freezing
excludes leaves from the optimizer, it does not detach recurrent states or
truncate temporal differentiation.
"""

import argparse
import copy
import json
import os
import signal
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.meta_delta import study as MS                    # noqa: E402
from experiments.nested_memory import task as TK                  # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import ordinary as OD       # noqa: E402
from experiments.prospective_momentum import replication as RP    # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import study as ST          # noqa: E402

# additive registry entries; no completed study's arm list changes
PD.DISPLAY.setdefault(OD.ORDINARY, OD.DISPLAY)
PD.CARRY.setdefault(OD.ORDINARY, OD.CARRY_REALS)
ST.EXTRA_GRAD.setdefault(OD.ORDINARY, "kappa")

UPDATES = ST.UPDATES
LRS = ST.LRS
VAL_AT = ST.VAL_AT
VAL_PER_FAMILY = ST.VAL_PER_FAMILY
HELDOUT_PER_FAMILY = ST.HELDOUT_PER_FAMILY
BATCH_PER_FAMILY = ST.BATCH_PER_FAMILY
#: reused, READ-ONLY independent sources
SOURCE_RUN = "/Users/durso/s5-runs/prospective-momentum-replication/20260917-011842"
SOURCE_DEV = RS.SOURCE_DEV
SOURCE_FINAL = RS.SOURCE_FINAL
SOURCE_FAMILY = "momentum_delta"
#: frozen fresh streams
STREAM = dict(continuation_train=320_000_000, dev_validation=350_000_000,
              eval_validation=351_000_000, heldout=360_000_000)
#: arm id -> (executed law, training regime)
ARMS = (("prospective_full", "prospective_momentum", "full"),
        ("prospective_coeff", "prospective_momentum", "coefficient_only"),
        ("ordinary_full", OD.ORDINARY, "full"),
        ("ordinary_coeff", OD.ORDINARY, "coefficient_only"),
        ("native_full", "momentum_delta", "full"),
        ("native_frozen", "momentum_delta", "frozen"))
TRAINED_ARMS = tuple(a for a in ARMS if a[2] != "frozen")
ARM_DISPLAY = {
    "prospective_full": "Prospective correction, full continuation",
    "prospective_coeff": "Prospective correction, coefficient-only (kappa)",
    "ordinary_full": "Ordinary prospective operator, full continuation",
    "ordinary_coeff": "Ordinary prospective operator, coefficient-only",
    "native_full": "Native Momentum DeltaNet, full continuation",
    "native_frozen": "Native Momentum DeltaNet source, frozen (no training)",
}
# R1 (review of the worktree): the reused `ST.compare` looks up
# `PD.DISPLAY[other]`, and this study's rows are keyed by ARM id, not by law.
# Register the arm ids additively so every declared pair resolves; the
# historical law entries are untouched, and `prospective_full` and
# `prospective_coeff` stay distinct populations.
for _arm, _display in ARM_DISPLAY.items():
    PD.DISPLAY.setdefault(_arm, _display)

LAW_OF = {a: law for a, law, _ in ARMS}
#: laws carrying a trainable extension scalar (native has none)
HAS_EXTENSION = {"prospective_momentum", OD.ORDINARY}
REGIME_OF = {a: reg for a, _, reg in ARMS}
#: the only leaf a coefficient-only arm may change
TRAINABLE_WHEN_FROZEN = ("kappa",)
PLANNED = dict(trained_runs_development=len(TRAINED_ARMS) * len(LRS),
               trained_runs_final=len(TRAINED_ARMS) * len(SOURCE_FINAL),
               frozen_evaluations=1 + len(SOURCE_FINAL))
PLANNED["total_updates"] = (PLANNED["trained_runs_development"]
                            + PLANNED["trained_runs_final"]) * UPDATES
#: literature arms NOT present in this batch (named, per the brief)
ABSENT_LITERATURE = ("gated_delta",)


def continuation_stream(seed, update):
    return STREAM["continuation_train"] + seed * 10_000 + update


def new_ranges():
    lo, hi = min((SOURCE_DEV,) + SOURCE_FINAL), max((SOURCE_DEV,)
                                                    + SOURCE_FINAL)
    r = dict(continuation_train=(continuation_stream(lo, 0),
                                 continuation_stream(hi, UPDATES - 1)))
    for k in ("dev_validation", "eval_validation", "heldout"):
        r[k] = (STREAM[k], STREAM[k])
    return r


def previous_ranges():
    """Everything the generator has consumed, including the replication."""
    prev = list(RP.previous_ranges())
    for lo, hi in RP.new_ranges().values():
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


# ------------------------------------------------- coefficient-only step ---
def freeze_mask(p):
    """1.0 on the trainable extension scalar, 0.0 on every pretrained leaf."""
    return {k: jnp.asarray(1.0 if k in TRAINABLE_WHEN_FROZEN else 0.0,
                           dtype=jnp.asarray(v).dtype)
            for k, v in p.items()}


@partial(jax.jit, static_argnums=(0,))
def train_step_coefficient_only(rule, p, opt, eps, lr):
    """The SAME loss, rollout and full BPTT as the full regime. Only the
    optimizer's input is restricted: the gradient of every pretrained leaf is
    zeroed BEFORE the transformation (so the global-norm clip and Adam see
    only the trainable scalar), and the update is masked again defensively.
    No stop_gradient anywhere: dL/dkappa still flows through the complete
    recurrence."""
    (loss, aux), g = jax.value_and_grad(ST.batch_loss, argnums=1,
                                        has_aux=True)(rule, p, eps)
    mask = freeze_mask(p)
    g_masked = jax.tree_util.tree_map(lambda x, m: x * m, g, mask)
    raw, opt = ST.TX.update(g_masked, opt, p)
    upd = jax.tree_util.tree_map(lambda u, m: -lr * u * m, raw, mask)
    p = optax.apply_updates(p, upd)
    p, tel = ST._project(rule, p)
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    extra = g[ST.EXTRA_GRAD[rule]][0] if rule in ST.EXTRA_GRAD else 0.0
    return (p, opt, loss, acc, optax.global_norm(g), optax.global_norm(upd),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]), tel,
            jnp.asarray(extra, dtype=loss.dtype))


def host_step(arm, p, opt, seed, u, lr, hist):
    rule, regime = LAW_OF[arm], REGIME_OF[arm]
    eps = ST.to_jax(TK.generate_batch(continuation_stream(seed, u),
                                      BATCH_PER_FAMILY))
    step = (train_step_coefficient_only if regime == "coefficient_only"
            else ST.train_step)
    o = step(rule, p, opt, eps, lr)
    p, opt, tel = o[0], o[1], o[8]
    scalars = dict(zip(ST.SCALAR_NAMES, (float(x) for x in o[2:8])))
    hist.append(dict(update=u, n_projected=int(tel["n_projected"]),
                     pre=float(tel["pre"]), post=float(tel["post"]),
                     cap=float(tel["cap"]), bound=float(tel["bound"]),
                     overshoot=float(tel["overshoot"]), grad=float(o[9])))
    return p, opt, scalars


def parameter_counts(arm, p):
    """Stored parameters and, separately, how many of them this regime
    actually trains (review: report them separately)."""
    counts = PM.parameter_counts(LAW_OF[arm], p)
    regime = REGIME_OF[arm]
    if regime == "frozen":
        trainable = 0
    elif regime == "coefficient_only":
        trainable = sum(int(onp.asarray(p[k]).size)
                        for k in TRAINABLE_WHEN_FROZEN if k in p)
    else:
        trainable = counts["total"]
    counts["trainable"] = trainable
    counts["stored"] = counts["total"]
    counts["regime"] = regime
    return counts


def frozen_leaf_differences(p, source_p):
    """Names of pretrained leaves that changed, with their max deviation."""
    out = {}
    for k, v in source_p.items():
        if k in TRAINABLE_WHEN_FROZEN:
            continue
        a, b = onp.asarray(p[k]), onp.asarray(v)
        if not onp.array_equal(a, b):
            out[k] = float(onp.max(onp.abs(a.astype(onp.float64)
                                           - b.astype(onp.float64))))
    return out


# ------------------------------------------------- validity and reports ----
def validate(arm, p):
    rule = LAW_OF[arm]
    if not ST.all_finite(p):
        return "non-finite parameters"
    gr = PD.gate_range_report(p)
    if not gr["gates_valid"]:
        return f"gate range/finiteness failed: {gr}"
    if rule == OD.ORDINARY:
        return OD.transition_failure(OD.transition_report(p))
    return ST.validate(rule, p)


def coefficient_report(arm, p, eps_np=None):
    rule = LAW_OF[arm]
    out = dict(arm=arm, law=rule, regime=REGIME_OF[arm],
               gate_table=PD.gate_range_report(p))
    if rule == OD.ORDINARY:
        out["table_transition"] = OD.transition_report(p)
        out["kappa_stored_directly"] = float(onp.asarray(p["kappa"]).ravel()[0])
        out["carry_note"] = ("carry W, U and the previous residual: "
                             f"{OD.CARRY_REALS} real numbers")
        out["observed_rollout_gates_note"] = (
            "the operator is not in PD.MOMENTUM_FAMILY, so evaluation does "
            "not attach observed-rollout gates to its metrics; its gate "
            "block is the same function of the same parameters, and TABLE "
            "coverage is reported above")
    else:
        out.update(ST.coefficient_report(rule, p, eps_np))
    return out


# --------------------------------------------------------------- one run ---
def run_one(arm, tag, lr_value, seed, source_p, val_np, out, deadline,
            reserve_s, status, stage, source_label):
    """One continuation. `native_frozen` trains nothing and is evaluated once
    (its record carries zero updates and the source's own metrics)."""
    rule, regime = LAW_OF[arm], REGIME_OF[arm]
    t0 = time.time()
    p = dict(source_p)
    if regime == "frozen":
        m = ST.evaluate(rule, p, val_np)
        rec = dict(tag=stage, rule=arm, law=rule, regime=regime,
                   display=ARM_DISPLAY[arm], config=tag, lr=0.0, seed=seed,
                   source=source_label, wall_s=time.time() - t0, curve=[],
                   validation=[], updates=0,
                   start_validation=m, final_validation=m,
                   training_gain=dict(primary=0.0, revision_ce=0.0,
                                      retention=0.0, recall=0.0),
                   params=parameter_counts(arm, p), carry=PD.CARRY[rule],
                   coefficients_final=coefficient_report(arm, p, val_np),
                   extension_history=None, extension_summary=None,
                   frozen_leaf_differences={}, invalid=(
                       None if ST.metrics_finite(m) else "non-finite metric"))
        return rec, p
    lr = jnp.asarray(lr_value, dtype=jnp.float32)
    opt = ST.TX.init(p)
    curve, val_hist, hist, last = [], [], [], None
    for u in range(UPDATES + 1):
        if u in VAL_AT:
            m = ST.evaluate(rule, p, val_np)
            val_hist.append(dict(update=u, primary=m["primary"],
                                 revision_ce=m["revision_ce"],
                                 retention=m["retention_revision_untouched"],
                                 recall=m["recall_overall"],
                                 state_norms=m["state_norms"],
                                 full=m if u in (0, UPDATES) else None))
        if u == UPDATES:
            break
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{stage}:{arm}/{tag}/seed{seed} stopped at update {u} of "
                f"{UPDATES}")
            return None
        p, opt, last = host_step(arm, p, opt, seed, u, lr, hist)
        if u % 50 == 0 or u == UPDATES - 1:
            curve.append(dict(update=u, **last))
    final = val_hist[-1]["full"]
    changed = frozen_leaf_differences(p, source_p)
    bad = None
    if last is None or not all(onp.isfinite(v) for v in last.values()):
        bad = "non-finite final training scalars"
    elif not ST.all_finite(p) or not ST.all_finite(opt):
        bad = "non-finite final parameters or optimizer state"
    elif not ST.metrics_finite(final):
        bad = "non-finite validation metric"
    elif regime == "coefficient_only" and changed:
        bad = f"frozen leaves changed: {changed}"
    else:
        bad = validate(arm, p)
    stem = f"{stage}_{arm}_{tag}_seed{seed}"
    ST.save_tree(os.path.join(out, "params", stem + ".msgpack"), p)
    ST.save_tree(os.path.join(out, "params", stem + "_opt.msgpack"), opt)
    rec = dict(tag=stage, rule=arm, law=rule, regime=regime,
               display=ARM_DISPLAY[arm], config=tag, lr=lr_value, seed=seed,
               source=source_label, wall_s=time.time() - t0, curve=curve,
               updates=UPDATES,
               validation=[{k: v for k, v in h.items() if k != "full"}
                           for h in val_hist],
               start_validation=val_hist[0]["full"], final_validation=final,
               training_gain=dict(
                   primary=final["primary"] - val_hist[0]["primary"],
                   revision_ce=final["revision_ce"]
                   - val_hist[0]["revision_ce"],
                   retention=final["retention_revision_untouched"]
                   - val_hist[0]["retention"],
                   recall=final["recall_overall"] - val_hist[0]["recall"]),
               params=parameter_counts(arm, p), carry=PD.CARRY[rule],
               coefficients_final=coefficient_report(arm, p, val_np),
               extension_history=(hist if rule in HAS_EXTENSION else None),
               extension_summary=(ST.summarize_history(
                   hist, "prospective_momentum")
                   if rule in HAS_EXTENSION else None),
               telemetry_note=("native arms have no extension scalar; their "
                               "projection telemetry is NaN, meaning "
                               "unavailable"),
               frozen_leaf_differences=changed, invalid=bad)
    return rec, p


# ------------------------------------------------------------- preflight ---
def preflight(sources, val_np, status):
    """Times the ACTUAL paths: both training regimes for every trained arm,
    evaluation, validation/reporting, on disposable state."""
    rows, failures, retraced_any = [], [], False
    p0 = sources[SOURCE_DEV]
    lr = jnp.asarray(LRS[0][1], dtype=jnp.float32)
    step_s, eval_s, report_s = {}, {}, {}
    for arm, rule, regime in TRAINED_ARMS:
        p = (PM.add_extension(p0, rule) if rule != "momentum_delta"
             else dict(p0))
        opt = ST.TX.init(p)
        hist = []
        t0 = time.time()
        p2, opt2, _ = host_step(arm, p, opt, SOURCE_DEV, 0, lr, hist)
        compile_s = time.time() - t0
        p2, opt2, _ = host_step(arm, p2, opt2, SOURCE_DEV, 1, lr, hist)
        n0 = (train_step_coefficient_only if regime == "coefficient_only"
              else ST.train_step)._cache_size()
        t1 = time.time()
        for u in range(2, 7):
            p2, opt2, measured = host_step(arm, p2, opt2, SOURCE_DEV, u, lr,
                                           hist)
        step_s[arm] = (time.time() - t1) / 5.0
        retraced_any |= (train_step_coefficient_only
                         if regime == "coefficient_only"
                         else ST.train_step)._cache_size() != n0
        t2 = time.time()
        ST.evaluate(LAW_OF[arm], p2, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        m = ST.evaluate(LAW_OF[arm], p2, val_np)
        eval_s[arm] = time.time() - t3
        t4 = time.time()
        acc_bad = validate(arm, p2)
        coefficient_report(arm, p2, val_np)
        report_s[arm] = time.time() - t4
        bad = MS.measured_scalar_failures(measured)
        if bad:
            acc_bad = f"non-finite measured preflight scalars {bad}"
        elif not (ST.all_finite(p2) and ST.all_finite(opt2)
                  and ST.metrics_finite(m)):
            acc_bad = "non-finite preflight state or metrics"
        elif regime == "coefficient_only":
            changed = frozen_leaf_differences(p2, p)
            if changed:
                acc_bad = f"preflight frozen leaves changed: {changed}"
        if acc_bad:
            failures.append(f"{arm}: {acc_bad}")
        rows.append(dict(arm=arm, law=rule, regime=regime,
                         compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s,
                         step_s=step_s[arm], eval_s=eval_s[arm],
                         report_s=report_s[arm], acceptance_failure=acc_bad,
                         measured_scalars=measured,
                         params=parameter_counts(arm, p2),
                         carry=PD.CARRY[rule]))
        print(f"[preflight] {arm:<20} {regime:<17} step "
              f"{step_s[arm] * 1e3:6.2f}ms eval {eval_s[arm] * 1e3:6.1f}ms "
              f"report {report_s[arm]:5.2f}s compile {compile_s:4.1f}s "
              f"params {parameter_counts(arm, p2)['total']} trainable "
              f"{parameter_counts(arm, p2)['trainable']} carry "
              f"{PD.CARRY[rule]}")
    n_runs = len(LRS) + len(SOURCE_FINAL)
    total = 0.0
    for arm, _, _ in TRAINED_ARMS:
        total += n_runs * (UPDATES * step_s[arm] + len(VAL_AT) * eval_s[arm]
                           + 2.0 * report_s[arm])
        total += len(SOURCE_FINAL) * 2.0 * eval_s[arm]        # held-out
    anchor = max(eval_s.values())
    total += (1 + len(SOURCE_FINAL)) * (anchor + 2.0 * anchor)  # frozen arm
    host_s = 40.0                            # ALLOWANCE, recorded as such
    total += host_s
    timing = dict(projected_remaining_s=total, host_allowance_s=host_s)
    if not all(onp.isfinite(v) and v >= 0 for v in timing.values()):
        failures.append(f"non-finite or negative timing {timing}")
    status["preflight"] = dict(rows=rows, retraced_any=bool(retraced_any),
                               failures=failures, planned=planned_work(),
                               **timing,
                               note=("disposable state; compilation incurred "
                                     "here is not re-counted"))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, bool(retraced_any), failures


# ------------------------------------------------------------- selection ---
def select(dev_rows, status):
    """The unchanged rule, applied per ARM: update-200 development revision
    macro accuracy, then lower revision cross-entropy, then slot A."""
    sel, table = {}, []
    for arm, _, regime in TRAINED_ARMS:
        cand = [r for r in dev_rows if r["rule"] == arm]
        if len(cand) != len(LRS):
            return None, table
        ranked = sorted(cand, key=lambda r: (-r["final_validation"]["primary"],
                                             r["final_validation"]["revision_ce"],
                                             r["config"]))
        sel[arm] = ranked[0]["config"]
        table.append(dict(arm=arm, chosen=ranked[0]["config"],
                          candidates=[dict(config=r["config"], lr=r["lr"],
                                           primary=r["final_validation"]
                                           ["primary"],
                                           revision_ce=r["final_validation"]
                                           ["revision_ce"]) for r in cand]))
    status["selection"] = dict(selected=sel, table=table,
                               rule=("highest revision macro accuracy at "
                                     "update 200, then lower revision CE, "
                                     "then slot A; development only"))
    return sel, table


# ---------------------------------------------------------------- screens ---
#: every comparison is declared here BEFORE execution
COMPARISONS = (
    ("same_backbone", "prospective_full", "ordinary_full",
     "Q1: candidate versus ordinary operator, both fully continued"),
    ("same_backbone_coefficient_only", "prospective_coeff", "ordinary_coeff",
     "Q1 under coefficient-only training"),
    ("versus_continued_native", "prospective_full", "native_full",
     "Q2 control: candidate versus the further-trained native backbone"),
    ("versus_frozen_source", "prospective_full", "native_frozen",
     "Q2 anchor: candidate versus the untrained source"),
    ("ordinary_versus_continued_native", "ordinary_full", "native_full",
     "operator versus the further-trained native backbone"),
    ("coefficient_only_versus_full_candidate", "prospective_coeff",
     "prospective_full", "Q3: does freezing the backbone cost revision?"),
    ("coefficient_only_versus_full_ordinary", "ordinary_coeff",
     "ordinary_full", "Q3 for the operator"),
    ("coefficient_only_versus_frozen_source", "prospective_coeff",
     "native_frozen", "Q3: coefficient-only against the untrained source"),
    # R3: freezing must also be compared with the FURTHER-TRAINED native
    # baseline whose retention the intervention is meant to preserve; both
    # use already planned runs, so no extra training, source or data.
    ("coefficient_only_versus_continued_native", "prospective_coeff",
     "native_full",
     "Q2/Q3: does the frozen-backbone candidate hold up against the "
     "further-trained native baseline?"),
    ("ordinary_coefficient_only_versus_continued_native", "ordinary_coeff",
     "native_full", "the same question for the operator"),
)
#: comparisons that DESCRIBE the training intervention: freezing is not
#: required to increase revision for them to be informative
DESCRIPTIVE_ONLY = ("coefficient_only_versus_full_candidate",
                    "coefficient_only_versus_full_ordinary")


def screen(final_rows, seeds=SOURCE_FINAL):
    """Each comparison separately, with the unchanged safeguard AND the
    stronger descriptive condition (no measured decrease)."""
    out = dict(rule=("mean primary difference >= +1 pp; all paired primary "
                     "differences > 0; mean retention AND recall differences "
                     "each >= -1 pp"),
               descriptive_condition=("no measured decrease: mean retention "
                                      "AND recall differences both >= 0"),
               absent_literature_arms=list(ABSENT_LITERATURE),
               comparisons=[])
    for name, a, b, why in COMPARISONS:
        c = ST.compare(final_rows, a, b, seeds)
        c.update(name=name, candidate=a, question=why,
                 display=ARM_DISPLAY.get(b, b),
                 descriptive_only=bool(name in DESCRIPTIVE_ONLY),
                 retention_direction=RP.direction(c["retention_difference"]),
                 recall_direction=RP.direction(c["recall_difference"]),
                 no_measured_decrease=bool(
                     c["complete_paired_seeds"]
                     and c["retention_difference"] >= 0
                     and c["recall_difference"] >= 0))
        out["comparisons"].append(c)
    out["note"] = (
        "Intervention study on the completed replication's independently "
        "pretrained Momentum sources, reused read-only: NOT another "
        "independent-source replication. The active native comparator is the "
        "continued native Momentum rule and its frozen source anchor; "
        f"literature arms absent from this batch: {list(ABSENT_LITERATURE)}. "
        "No joint literature win can be claimed from it, and the completed "
        "Gated DeltaNet results are historical, not fresh matched "
        "evaluations. For constant eta and mu the candidate and the operator "
        "are the same law (audit s1), so a difference here is attributable "
        "only to token-varying gates. A safeguard pass is not zero loss and "
        "not proof of noninferiority; three seeds on this small task are not "
        "significance, a benchmark or SOTA. Comparisons marked "
        "descriptive_only characterise the training intervention: freezing "
        "the backbone is not required to RAISE revision for them to be "
        "informative. No-measured-decrease alone is never a win: the "
        "revision difference is reported beside it in every comparison.")
    return out


# ------------------------------------------------------------------ main ---
def load_sources(source_run, status):
    """Restore the replication's Momentum sources, READ-ONLY and checksum
    verified. Returns {seed: native tree} or raises SourceRefusal."""
    path = os.path.join(RS.sources_dir(source_run), "manifest.json")
    if not os.path.isfile(path):
        raise RS.SourceRefusal(f"missing source manifest {path}")
    with open(path) as fh:
        manifest = json.load(fh)
    entries, sources = {}, {}
    for seed in (SOURCE_DEV,) + SOURCE_FINAL:
        e = RS.find_entry(manifest, seed, SOURCE_FAMILY)
        sources[seed] = RS.restore_source(source_run, e)     # verifies sha256
        entries[seed] = dict(key=e["key"], file=e["file"], sha256=e["sha256"],
                             metrics_end=e.get("metrics_end", {}).get(
                                 "primary"))
    status["source"] = dict(
        source_run=source_run, reuse="read-only; nothing is written here",
        entries=entries,
        hashes_at_restore={e["file"]: e["sha256"] for e in entries.values()})
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/prospective-same-backbone")
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
        raise SystemExit("REFUSING: x64 is enabled; the declared production "
                         "setting is float32 with x64 disabled.")
    overlaps = stream_overlaps()
    if overlaps:
        raise SystemExit(f"REFUSING: stream ranges overlap: {overlaps}")
    if os.path.realpath(args.out_root).startswith(
            os.path.realpath(args.source_run)):
        raise SystemExit("REFUSING: output inside the read-only source run")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)

    status = dict(run_id=run_id, out=out, backend=backend,
                  study="same backbone: ordinary operator and retention",
                  source_run=args.source_run,
                  source_seeds=dict(development=SOURCE_DEV,
                                    final=list(SOURCE_FINAL)),
                  final_seeds=list(SOURCE_FINAL), updates=UPDATES,
                  arms={a: ARM_DISPLAY[a] for a, _, _ in ARMS},
                  arm_law={a: law for a, law, _ in ARMS},
                  arm_regime={a: reg for a, _, reg in ARMS},
                  trainable_leaf_when_frozen=list(TRAINABLE_WHEN_FROZEN),
                  planned_work=planned_work(), streams=STREAM,
                  stream_ranges=new_ranges(),
                  previous_stream_ranges=previous_ranges(),
                  learning_rates=dict(LRS),
                  projection_relative_margin=PD.PROJ_REL_MARGIN,
                  production_dtype=dict(x64=False, float_dtype="float32"),
                  absent_literature_arms=list(ABSENT_LITERATURE),
                  completed_study_references=[
                      "/Users/durso/s5-runs/prospective-momentum/20260917-000431",
                      SOURCE_RUN],
                  heldout_policy=("one common held-out set, generated and "
                                  "hashed only after all final runs; opening "
                                  "persisted before use"),
                  source=dict(hashes_at_restore=None),
                  development=[], final=[], incomplete=[])
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
    val_np = TK.generate_batch(STREAM["dev_validation"], VAL_PER_FAMILY)
    eval_val_np = TK.generate_batch(STREAM["eval_validation"], VAL_PER_FAMILY)
    status["task"] = dict(
        structure=TK.structure_check(val_np),
        dev_validation_digest=TK.episode_digest(val_np),
        eval_validation_digest=TK.episode_digest(eval_val_np),
        continuation_first_batch_digest=TK.episode_digest(
            TK.generate_batch(continuation_stream(SOURCE_DEV, 0),
                              BATCH_PER_FAMILY)))
    save()

    try:
        sources = load_sources(args.source_run, status)
    except RS.SourceRefusal as e:
        status["failed"] = f"source refusal: {e}"
        print(f"[!] {status['failed']}")
        return 4, "FAILED"
    print(f"[source] reused {len(sources)} read-only Momentum checkpoints "
          f"from {args.source_run}")
    save()

    # all three laws must be the SAME function at kappa = 0 on every source
    ident = {}
    for seed, pn in sources.items():
        base = ST.evaluate("momentum_delta", pn, val_np)
        ident[str(seed)] = {
            law: ST.identity_differences(
                base, ST.evaluate(law, PM.add_extension(pn, law), val_np))
            for law in ("prospective_momentum", OD.ORDINARY)}
    status["update_zero_identity"] = ident
    if any(f for s in ident.values() for f in s.values()):
        status["failed"] = f"laws differ at kappa = 0: {ident}"
        print(f"[!] {status['failed']}")
        return 4, "FAILED"
    save()

    proj, retraced, failures = preflight(sources, val_np, status)
    save()
    d = ST.decide_after_preflight(proj, retraced, failures,
                                  deadline - time.time() - args.reserve_s)
    if d is not None:
        code, label, why = d
        (status.__setitem__("failed", why) if code == 4
         else status["incomplete"].append(why))
        print(f"[!] {why}")
        return code, label

    def start_tree(arm, seed):
        rule = LAW_OF[arm]
        pn = sources[seed]
        p = dict(pn) if rule == "momentum_delta" else PM.add_extension(pn,
                                                                      rule)
        e = status["source"]["entries"][seed]
        return p, f"{e['file']} sha256={e['sha256']}"

    dev_rows = []
    for arm, _, _ in TRAINED_ARMS:
        for tag, lr in LRS:
            p, label = start_tree(arm, SOURCE_DEV)
            r = run_one(arm, tag, lr, SOURCE_DEV, p, val_np, out, deadline,
                        args.reserve_s, status, "dev", label)
            if r is None:
                return 3, "INCOMPLETE"
            rec, _ = r
            dev_rows.append(rec)
            status["development"] = dev_rows
            save()
            if rec["invalid"]:
                status["failed"] = f"dev {arm}/{tag}: {rec['invalid']}"
                return 4, "FAILED"
    p, label = start_tree("native_frozen", SOURCE_DEV)
    rec, _ = run_one("native_frozen", "-", 0.0, SOURCE_DEV, p, val_np, out,
                     deadline, args.reserve_s, status, "dev", label)
    dev_rows.append(rec)
    status["development"] = dev_rows
    sel, _ = select(dev_rows, status)
    if sel is None:
        status["failed"] = "selection could not be formed"
        return 4, "FAILED"
    frozen_sel = copy.deepcopy(sel)
    status["selection_frozen"] = copy.deepcopy(sel)
    ST.write(os.path.join(out, "selection.json"), status["selection"])
    save()

    final_rows, finals = [], {}
    lr_of = dict(LRS)
    for arm, _, regime in ARMS:
        for seed in SOURCE_FINAL:
            p, label = start_tree(arm, seed)
            tag = "-" if regime == "frozen" else frozen_sel[arm]
            r = run_one(arm, tag, (0.0 if regime == "frozen"
                                   else lr_of[frozen_sel[arm]]), seed, p,
                        eval_val_np, out, deadline, args.reserve_s, status,
                        "final", label)
            if r is None:
                status["heldout_opened"] = False
                return 3, "INCOMPLETE"
            rec, pf = r
            final_rows.append(rec)
            finals[(arm, seed)] = pf
            status["final"] = final_rows
            save()
            if rec["invalid"]:
                status["failed"] = f"final {arm}/{seed}: {rec['invalid']}"
                return 4, "FAILED"
    if status["selection"]["selected"] != frozen_sel:
        status["failed"] = "development selection changed during finals"
        return 4, "FAILED"

    status["heldout_opened"] = True
    status["heldout_opened_at"] = time.time()
    status["heldout_evaluation_complete"] = False
    save()
    held_np = TK.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    status["task"]["heldout_digest"] = TK.episode_digest(held_np)
    for rec in final_rows:
        rec["heldout"] = ST.evaluate(rec["law"],
                                     finals[(rec["rule"], rec["seed"])],
                                     held_np)
        if not ST.metrics_finite(rec["heldout"]):
            status["failed"] = (f"non-finite held-out {rec['rule']}/"
                                f"{rec['seed']}")
            return 4, "FAILED"
    status["heldout_evaluation_complete"] = True
    status["screen"] = screen(final_rows)
    if status["selection"]["selected"] != frozen_sel:
        status["failed"] = "development selection changed after evaluation"
        return 4, "FAILED"
    status["work_completed"] = dict(
        development_runs=len([r for r in dev_rows if r["updates"]]),
        final_runs=len([r for r in final_rows if r["updates"]]),
        frozen_evaluations=len([r for r in dev_rows + final_rows
                                if not r["updates"]]),
        total_updates=sum(r["updates"] for r in dev_rows + final_rows))
    for c in status["screen"]["comparisons"]:
        print(f"[screen] {c['name']:<42} passed={c['passed']} "
              f"no_measured_decrease={c['no_measured_decrease']} mean "
              f"{c['mean_primary_difference']}")
    status["complete"] = True
    return 0, "PASS"


if __name__ == "__main__":
    sys.exit(main())
