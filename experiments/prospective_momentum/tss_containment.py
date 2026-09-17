"""Does departing from the exact literal-TSS boundary help, on the same
Momentum backbone?

Specification: docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md (d95266d). Frozen
protocol: docs/PROSPECTIVE_TSS_CONTAINMENT_PROTOCOL.md. Clearance for
implementation: PROSPECTIVE_TSS_IMPLEMENTATION_CLEARANCE_d95266d_2026_09_17.md.
NOT AUTHORIZED TO RUN until the implementation review clears it.

Five arms, one new executed law (`experiments.prospective_momentum.filtered`)
plus the two existing ones:

    tss_processing          master law with M = gamma = 0 STORED CONSTANTS,
                            only T trains: literal TSS at every executed point
    generalized_processing  the same law with M, gamma, T all trainable,
                            STARTED at M = gamma = 0, T = T0 = h, i.e.
                            function-matched to `tss_processing`
    native_full             native Momentum DeltaNet, continued
    operator_full           the learned two-tap residual operator (kappa),
                            a strong comparator OUTSIDE the containment claim
                            (its learned kappa > 1 needs gamma < 0)
    native_frozen           the restored source, evaluation only (anchor)

Sources are the replication's independently pretrained Momentum checkpoints,
reused READ-ONLY (seed 500 development, 501-503 final), with fresh streams.
Full BPTT and the existing unweighted query cross-entropy are unchanged.

Three points of the clearance are implemented here:

1. the executed filter's acceptance is a GATE, not telemetry: the rounded
   coefficients the production step actually executes are formed once
   (`filtered.in_loop_guard` / `filtered.filter_report`), and a run whose
   executed polynomial is not finite and strictly Schur stable is refused;
2. a scientific extension comparison needs a GENERALIZED endpoint against a
   LITERAL-TSS endpoint. Deployment selections are reported separately and
   never relabelled: a native model is never called TSS;
3. diagnostic execution is deterministic and frozen before the finals: every
   family's final endpoint is its best FEASIBLE development checkpoint if one
   exists and otherwise its best unconstrained one, flagged `diagnostic`.
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
from experiments.prospective_momentum import filtered as FL       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import ordinary as OD       # noqa: E402
from experiments.prospective_momentum import replication as RP    # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import same_backbone as SB  # noqa: E402
from experiments.prospective_momentum import study as ST          # noqa: E402

# additive registry entries only; no completed study's arm list changes
# (`filtered` registers its own display and carry at import)
PD.DISPLAY.setdefault(OD.ORDINARY, OD.DISPLAY)
PD.CARRY.setdefault(OD.ORDINARY, OD.CARRY_REALS)
ST.EXTRA_GRAD.setdefault(OD.ORDINARY, "kappa")

#: declared tolerance policy for the two exact-point recovery checks. The
#: identities are exact in real arithmetic; two separately compiled float
#: recurrences are NEVER required to agree bitwise (review R3). Magnitudes are
#: always recorded, passing or failing.
RECOVERY_QUERY_TOL = 1.0        # at most one argmax flip per category
RECOVERY_CE_REL = 1e-3          # relative cross-entropy agreement

UPDATES = ST.UPDATES                       # 200
LRS = ST.LRS                               # ("A", 0.003), ("B", 0.01)
#: development checkpoints; every family gets the same opportunities
VAL_AT = (0, 25, 50, 100, 200)
VAL_PER_FAMILY = ST.VAL_PER_FAMILY
HELDOUT_PER_FAMILY = ST.HELDOUT_PER_FAMILY
BATCH_PER_FAMILY = ST.BATCH_PER_FAMILY
SOURCE_RUN = SB.SOURCE_RUN                 # reused READ-ONLY
SOURCE_DEV = RS.SOURCE_DEV
SOURCE_FINAL = RS.SOURCE_FINAL
SOURCE_FAMILY = "momentum_delta"
#: frozen fresh streams; disjointness from every previous study is asserted
STREAM = dict(continuation_train=420_000_000, dev_validation=450_000_000,
              eval_validation=451_000_000, heldout=460_000_000)

TSS = "tss_processing"
GEN = "generalized_processing"
NATIVE = "native_full"
OPERATOR = "operator_full"
ANCHOR = "native_frozen"
#: arm id -> (law, frozen coefficient leaves, regime)
ARMS = ((TSS, FL.FILTERED, ("fil_M", "fil_gamma"), "full"),
        (GEN, FL.FILTERED, (), "full"),
        (NATIVE, "momentum_delta", (), "full"),
        (OPERATOR, OD.ORDINARY, (), "full"),
        (ANCHOR, "momentum_delta", (), "frozen"))
TRAINED_ARMS = tuple(a for a in ARMS if a[3] != "frozen")
#: the extensions selected against the native baseline's development scores
EXTENSION_ARMS = (TSS, GEN, OPERATOR)
LAW_OF = {a: law for a, law, _, _ in ARMS}
FROZEN_LEAVES = {a: fz for a, _, fz, _ in ARMS}
REGIME_OF = {a: reg for a, _, _, reg in ARMS}
ARM_DISPLAY = {
    TSS: ("Literal TSS residual processing (M = gamma = 0 fixed; T trains)"),
    GEN: ("Master-law residual processing (M, gamma, T train), started on "
          "the literal TSS boundary at T0 = h"),
    NATIVE: "Native Momentum DeltaNet, full continuation",
    OPERATOR: ("Learned two-tap prospective residual operator (kappa), "
               "outside the nonnegative-gamma family"),
    ANCHOR: "Native Momentum DeltaNet source, frozen (no training)",
}
for _arm, _display in ARM_DISPLAY.items():
    PD.DISPLAY.setdefault(_arm, _display)
#: laws whose per-update telemetry is the filter repair
FILTER_ARMS = tuple(a for a, law, _, _ in ARMS if law == FL.FILTERED)
#: literature arms NOT present in this batch
ABSENT_LITERATURE = ("gated_delta",)

PLANNED = dict(trained_runs_development=len(TRAINED_ARMS) * len(LRS),
               trained_runs_final=len(TRAINED_ARMS) * len(SOURCE_FINAL),
               frozen_evaluations=1 + len(SOURCE_FINAL),
               development_checkpoints_per_family=len(VAL_AT) * len(LRS) - 1)
PLANNED["max_total_updates"] = (PLANNED["trained_runs_development"]
                                + PLANNED["trained_runs_final"]) * UPDATES


def recovery_differences(ref, other):
    """Finite-first, tolerance-based comparison of two separately compiled
    recurrences that are mathematically the same function. Returns
    (failures, magnitudes); `failures` empty means agreement within the
    declared tolerances."""
    fails, mags = [], {}
    for fam in TK.FAMILIES:
        for cn in TK.CATEGORIES:
            a, b = ref[fam]["by_category"][cn], other[fam]["by_category"][cn]
            vals = [a["accuracy"], b["accuracy"], a["cross_entropy"],
                    b["cross_entropy"]]
            if not all(v is not None and onp.isfinite(v) for v in vals):
                fails.append(f"{fam}/{cn}: non-finite metric {vals}")
                continue
            dq = abs(a["accuracy"] - b["accuracy"]) * a["n"]
            dce = (abs(a["cross_entropy"] - b["cross_entropy"])
                   / max(abs(a["cross_entropy"]), 1e-12))
            mags[f"{fam}/{cn}"] = dict(query_difference=float(dq),
                                       cross_entropy_relative=float(dce))
            if dq > RECOVERY_QUERY_TOL or dce > RECOVERY_CE_REL:
                fails.append(f"{fam}/{cn}: queries {dq:.3f} (tol "
                             f"{RECOVERY_QUERY_TOL}) CE rel {dce:.2e} (tol "
                             f"{RECOVERY_CE_REL:.0e})")
    return fails, mags


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
    """Every range the generator has consumed, including the same-backbone
    study (which already includes the replication and its predecessors)."""
    prev = list(SB.previous_ranges())
    for lo, hi in SB.new_ranges().values():
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


# ----------------------------------------------------------- train step ----
def leaf_mask(p, frozen):
    return {k: jnp.asarray(0.0 if k in frozen else 1.0,
                           dtype=jnp.asarray(v).dtype) for k, v in p.items()}


@partial(jax.jit, static_argnums=(0, 1))
def train_step_filtered(rule, frozen, p, opt, eps, lr):
    """The SAME loss, rollout and full BPTT. Frozen coefficient leaves are
    removed from the optimizer's input (their gradient is zeroed BEFORE the
    transformation, and the update is masked again), and the repair restores
    them bitwise. No stop_gradient: dL/dT still flows through the complete
    recurrence, and the backbone trains in both processing arms."""
    (loss, aux), g = jax.value_and_grad(ST.batch_loss, argnums=1,
                                        has_aux=True)(rule, p, eps)
    mask = leaf_mask(p, frozen)
    g_masked = jax.tree_util.tree_map(lambda x, m: x * m, g, mask)
    raw, opt = ST.TX.update(g_masked, opt, p)
    upd = jax.tree_util.tree_map(lambda u, m: -lr * u * m, raw, mask)
    p = optax.apply_updates(p, upd)
    p, tel = FL.repair(p, frozen=frozen)
    guard = FL.in_loop_guard(FL.coefficients(p), p["fil_M"].dtype)
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    grads = {k: g[k][0] for k in FL.LEAVES}
    return (p, opt, loss, acc, optax.global_norm(g), optax.global_norm(upd),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]),
            dict(tel, **{"guard_" + k: v for k, v in guard.items()}), grads)


def host_step(arm, p, opt, seed, u, lr, hist):
    """One host step. The filtered arms use this module's masked step and the
    feasibility repair; the existing arms use the unchanged study step."""
    rule = LAW_OF[arm]
    eps = ST.to_jax(TK.generate_batch(continuation_stream(seed, u),
                                      BATCH_PER_FAMILY))
    if rule == FL.FILTERED:
        o = train_step_filtered(rule, FROZEN_LEAVES[arm], p, opt, eps, lr)
        p, opt, tel, grads = o[0], o[1], o[8], o[9]
        rec = dict(update=u, n_repaired=int(tel["n_repaired"]),
                   overshoot=float(tel["overshoot"]),
                   gap_gamma_plus_T=float(tel["gap_gamma_plus_T"]),
                   gap_filter=float(tel["gap_filter"]),
                   A=float(tel["guard_A"]), c1=float(tel["guard_c1"]),
                   c0=float(tel["guard_c0"]),
                   jury_min=float(tel["guard_jury_min"]),
                   executed_filter_ok=bool(tel["guard_ok"]))
        for k in FL.LEAVES:
            rec[k + "_pre"] = float(tel[k + "_pre"])
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


# ------------------------------------------------- validity and reports ----
def validate(arm, p):
    """None if acceptable, else the reason. Nothing is clamped here."""
    rule = LAW_OF[arm]
    if not ST.all_finite(p):
        return "non-finite parameters"
    gr = PD.gate_range_report(p)
    if not gr["gates_valid"]:
        return f"gate range/finiteness failed: {gr}"
    if rule == FL.FILTERED:
        bad = FL.filter_failure(FL.filter_report(p))
        if bad:
            return bad
        if arm == TSS:
            M = float(onp.asarray(p["fil_M"]).ravel()[0])
            gam = float(onp.asarray(p["fil_gamma"]).ravel()[0])
            if not (M == 0.0 and gam == 0.0):
                return (f"literal-TSS arm left its boundary: M={M} "
                        f"gamma={gam} (these leaves are stored constants)")
        # the downstream Momentum block's own frozen-token diagnostic, under
        # the same policy as the native arm: unstable or non-finite fails
        rep = PD.transition_report(p, "momentum_delta")
        bad = {k: v for k, v in rep["classification"].items()
               if k in ("unstable", "nonfinite")}
        return (f"Momentum-block frozen-token transitions {bad}"
                if bad else None)
    if rule == OD.ORDINARY:
        return OD.transition_failure(OD.transition_report(p))
    return ST.validate(rule, p)


def coefficient_report(arm, p, eps_np=None):
    rule = LAW_OF[arm]
    out = dict(arm=arm, law=rule, regime=REGIME_OF[arm],
               frozen_coefficient_leaves=list(FROZEN_LEAVES[arm]),
               gate_table=PD.gate_range_report(p))
    if rule == FL.FILTERED:
        out["executed_filter"] = FL.filter_report(p)
        out["momentum_block_diagnostic"] = PD.transition_report(
            p, "momentum_delta")
        out["carry_note"] = (
            f"executed carry W, U, y, y_prev, R_prev = {FL.CARRY_EXECUTED} "
            f"real numbers; a law with M fixed at zero needs "
            f"{FL.CARRY_MINIMAL_M_ZERO} (y_prev dormant). This "
            "implementation executes the same five-carry step in both "
            "processing arms, so the EXECUTED number is reported for both.")
        out["scope_note"] = (
            "the executed gate certifies the isolated processing filter "
            "only: not the closed-loop memory, not switching across tokens "
            "with varying gates, and not the Momentum block below it. The "
            "coefficient domain is broader than the passive-compartment "
            "domain: nonnegative M, gamma, T implies no physical "
            "realizability.")
    elif rule == OD.ORDINARY:
        out["table_transition"] = OD.transition_report(p)
        out["kappa_stored_directly"] = float(
            onp.asarray(p["kappa"]).ravel()[0])
        out["containment_note"] = (
            "kappa maps to (M, gamma, T) = (0, h(1-kappa), kappa h); "
            "kappa > 1 requires gamma < 0 and is OUTSIDE the "
            "nonnegative-gamma family. This arm is a separate comparator.")
        out["carry_note"] = ("carry W, U and the previous residual: "
                             f"{OD.CARRY_REALS} real numbers")
    else:
        out.update(ST.coefficient_report(rule, p, eps_np))
    return out


def parameter_counts(arm, p):
    counts = PM.parameter_counts(LAW_OF[arm], p)
    frozen = sum(int(onp.asarray(p[k]).size) for k in FROZEN_LEAVES[arm]
                 if k in p)
    counts["stored"] = counts["total"]
    counts["trainable"] = (0 if REGIME_OF[arm] == "frozen"
                           else counts["total"] - frozen)
    counts["frozen_constants"] = frozen
    counts["regime"] = REGIME_OF[arm]
    if LAW_OF[arm] == FL.FILTERED:
        counts["carry_real_numbers_executed"] = FL.CARRY_EXECUTED
        counts["carry_real_numbers_minimal_if_M_zero"] = (
            FL.CARRY_MINIMAL_M_ZERO)
    return counts


def frozen_leaf_differences(p, source_p, arm):
    """Stored constants must not move: a storage invariant, checked bitwise
    (unlike a claim about two separately compiled computations)."""
    out = {}
    for k in FROZEN_LEAVES[arm]:
        a, b = onp.asarray(p[k]), onp.asarray(source_p[k])
        if not onp.array_equal(a, b):
            out[k] = [float(onp.max(onp.abs(a.astype(onp.float64)
                                            - b.astype(onp.float64)))),
                      a.tolist(), b.tolist()]
    return out


# ------------------------------------------------------------ start trees --
def start_tree(arm, native_p):
    """The declared start of each arm. Both processing arms start at
    M = gamma = 0, T = T0: literal TSS, function-matched to each other."""
    rule = LAW_OF[arm]
    if rule == "momentum_delta":
        return dict(native_p)
    return PM.add_extension(native_p, rule)


def native_point_tree(native_p):
    """The native fallback MAP for the processing family: (M, gamma, T) =
    (0, h, 0), at which the law is exactly native Momentum. Used to verify
    value equivalence and to place a native deployment fallback."""
    dt = native_p["A_log"].dtype
    M, gam, T = FL.native_point()
    return dict(native_p, fil_M=jnp.full((1,), M, dtype=dt),
                fil_gamma=jnp.full((1,), gam, dtype=dt),
                fil_T=jnp.full((1,), T, dtype=dt))


# --------------------------------------------------------------- one run ---
def run_one(arm, tag, lr_value, seed, source_p, val_np, updates, out,
            deadline, reserve_s, status, stage, source_label):
    """One continuation of `updates` optimizer updates (0 for the anchor)."""
    rule, regime = LAW_OF[arm], REGIME_OF[arm]
    t0 = time.time()
    p = dict(source_p)
    val_at = tuple(u for u in VAL_AT if u <= updates)
    if updates not in val_at:
        val_at = val_at + (updates,)
    if regime == "frozen":
        m = ST.evaluate(rule, p, val_np)
        rec = dict(tag=stage, rule=arm, law=rule, regime=regime,
                   display=ARM_DISPLAY[arm], config=tag, lr=0.0, seed=seed,
                   source=source_label, wall_s=time.time() - t0, curve=[],
                   validation=[], updates=0, start_validation=m,
                   final_validation=m,
                   training_gain=dict(primary=0.0, revision_ce=0.0,
                                      retention=0.0, recall=0.0),
                   params=parameter_counts(arm, p), carry=PD.CARRY[rule],
                   coefficients_final=coefficient_report(arm, p, val_np),
                   coefficient_history=None, frozen_leaf_differences={},
                   invalid=(None if ST.metrics_finite(m)
                            else "non-finite metric"))
        return rec, p
    lr = jnp.asarray(lr_value, dtype=jnp.float32)
    opt = ST.TX.init(p)
    curve, val_hist, hist, last, gate_bad = [], [], [], None, None
    for u in range(updates + 1):
        if u in val_at:
            m = ST.evaluate(rule, p, val_np)
            val_hist.append(dict(update=u, primary=m["primary"],
                                 revision_ce=m["revision_ce"],
                                 retention=m["retention_revision_untouched"],
                                 recall=m["recall_overall"],
                                 state_norms=m["state_norms"],
                                 full=(m if u in (0, updates) else None)))
        if u == updates:
            break
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{stage}:{arm}/{tag}/seed{seed} stopped at update {u} of "
                f"{updates}")
            return None
        p, opt, last, rec = host_step(arm, p, opt, seed, u, lr, hist)
        if not rec.get("executed_filter_ok", True):
            # clearance s1: an update whose EXECUTED filter polynomial is not
            # finite and strictly stable is refused, not merely reported
            gate_bad = (f"executed filter gate failed at update {u}: "
                        f"jury_min={rec.get('jury_min')} c1={rec.get('c1')} "
                        f"c0={rec.get('c0')} A={rec.get('A')}")
            break
        if u % 25 == 0 or u == updates - 1:
            curve.append(dict(update=u, **last))
    final = val_hist[-1]["full"] if val_hist else None
    changed = frozen_leaf_differences(p, source_p, arm)
    bad = gate_bad
    if bad is not None:
        pass
    elif last is None or not all(onp.isfinite(v) for v in last.values()):
        bad = "non-finite final training scalars"
    elif not ST.all_finite(p) or not ST.all_finite(opt):
        bad = "non-finite final parameters or optimizer state"
    elif final is None or not ST.metrics_finite(final):
        bad = "non-finite validation metric"
    elif changed:
        bad = f"stored constant leaves changed: {changed}"
    else:
        bad = validate(arm, p)
    stem = f"{stage}_{arm}_{tag}_seed{seed}"
    ST.save_tree(os.path.join(out, "params", stem + ".msgpack"), p)
    ST.save_tree(os.path.join(out, "params", stem + "_opt.msgpack"), opt)
    rec = dict(tag=stage, rule=arm, law=rule, regime=regime,
               display=ARM_DISPLAY[arm], config=tag, lr=lr_value, seed=seed,
               source=source_label, wall_s=time.time() - t0, curve=curve,
               updates=updates,
               validation=[{k: v for k, v in h.items() if k != "full"}
                           for h in val_hist],
               start_validation=(val_hist[0]["full"] if val_hist else None),
               final_validation=final,
               training_gain=(dict(
                   primary=final["primary"] - val_hist[0]["primary"],
                   revision_ce=final["revision_ce"]
                   - val_hist[0]["revision_ce"],
                   retention=final["retention_revision_untouched"]
                   - val_hist[0]["retention"],
                   recall=final["recall_overall"] - val_hist[0]["recall"])
                   if final and val_hist else None),
               params=parameter_counts(arm, p), carry=PD.CARRY[rule],
               coefficients_final=coefficient_report(arm, p, val_np),
               coefficient_history=hist, frozen_leaf_differences=changed,
               invalid=bad)
    return rec, p


# ------------------------------------------------------------- preflight ---
def preflight(sources, val_np, status):
    """Times the ACTUAL paths of every trained arm on disposable state."""
    rows, failures, retraced_any = [], [], False
    p0 = sources[SOURCE_DEV]
    lr = jnp.asarray(LRS[0][1], dtype=jnp.float32)
    step_s, eval_s, report_s = {}, {}, {}
    for arm, rule, frozen, _ in TRAINED_ARMS:
        p = start_tree(arm, p0)
        opt = ST.TX.init(p)
        hist = []
        t0 = time.time()
        p2, opt2, _, _ = host_step(arm, p, opt, SOURCE_DEV, 0, lr, hist)
        compile_s = time.time() - t0
        p2, opt2, _, _ = host_step(arm, p2, opt2, SOURCE_DEV, 1, lr, hist)
        cache = (train_step_filtered if rule == FL.FILTERED else ST.train_step)
        n0 = cache._cache_size()
        t1 = time.time()
        for u in range(2, 7):
            p2, opt2, measured, rec = host_step(arm, p2, opt2, SOURCE_DEV, u,
                                                lr, hist)
        step_s[arm] = (time.time() - t1) / 5.0
        retraced_any |= cache._cache_size() != n0
        t2 = time.time()
        ST.evaluate(rule, p2, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        m = ST.evaluate(rule, p2, val_np)
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
        elif not rec.get("executed_filter_ok", True):
            acc_bad = f"executed filter gate failed in preflight: {rec}"
        else:
            ch = frozen_leaf_differences(p2, p, arm)
            if ch:
                acc_bad = f"preflight stored constants changed: {ch}"
        if acc_bad:
            failures.append(f"{arm}: {acc_bad}")
        rows.append(dict(arm=arm, law=rule, regime=REGIME_OF[arm],
                         frozen_coefficient_leaves=list(frozen),
                         compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s,
                         step_s=step_s[arm], eval_s=eval_s[arm],
                         report_s=report_s[arm], acceptance_failure=acc_bad,
                         measured_scalars=measured, last_telemetry=rec,
                         params=parameter_counts(arm, p2),
                         carry=PD.CARRY[rule]))
        print(f"[preflight] {arm:<24} step {step_s[arm] * 1e3:6.2f}ms eval "
              f"{eval_s[arm] * 1e3:6.1f}ms report {report_s[arm]:5.2f}s "
              f"compile {compile_s:4.1f}s params "
              f"{parameter_counts(arm, p2)['stored']} stored / "
              f"{parameter_counts(arm, p2)['trainable']} trainable carry "
              f"{PD.CARRY[rule]} failure {acc_bad}")
    n_runs = len(LRS) + len(SOURCE_FINAL)
    total = 0.0
    for arm, _, _, _ in TRAINED_ARMS:
        total += n_runs * (UPDATES * step_s[arm] + len(VAL_AT) * eval_s[arm]
                           + 2.0 * report_s[arm])
        total += len(SOURCE_FINAL) * eval_s[arm]              # held-out
    anchor = max(eval_s.values())
    total += (1 + 2 * len(SOURCE_FINAL)) * anchor             # frozen arm
    host_s = 40.0                            # ALLOWANCE, recorded as such
    total += host_s
    timing = dict(projected_remaining_s=total, host_allowance_s=host_s)
    if not all(onp.isfinite(v) and v >= 0 for v in timing.values()):
        failures.append(f"non-finite or negative timing {timing}")
    status["preflight"] = dict(rows=rows, retraced_any=bool(retraced_any),
                               failures=failures, planned=planned_work(),
                               **timing,
                               note=("disposable state; compilation incurred "
                                     "here is not re-counted. Final "
                                     "trajectories are projected at the full "
                                     "update count, the worst case."))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, bool(retraced_any), failures


# ------------------------------------------------------------- selection ---
def order_key(c):
    """The single development ordering: revision macro accuracy, then lower
    revision cross-entropy, then fewer updates, then lower learning rate."""
    return (-c["primary"], c["revision_ce"], c["update"], c["lr"])


def checkpoints(dev_rows, arm):
    """Every development checkpoint of one arm, with the identical
    update-zero checkpoint deduplicated (slot A keeps it)."""
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
                            retention=v["retention"], recall=v["recall"]))
    return out


def finite_checkpoints(cand):
    """None if every selection quantity is finite; else the offending rows.
    A non-finite checkpoint is never ranked or selected."""
    bad = [c for c in cand
           if not all(onp.isfinite(c[k]) for k in
                      ("primary", "revision_ce", "retention", "recall"))]
    return bad or None


def select(dev_rows, status):
    """Native first, then each extension under the matched-retention
    constraint, then the deterministic diagnostic rule of clearance s3."""
    nat = checkpoints(dev_rows, NATIVE)
    if len(nat) != PLANNED["development_checkpoints_per_family"] \
            or finite_checkpoints(nat):
        return None, None
    nat_best = sorted(nat, key=order_key)[0]
    r_native, c_native = nat_best["retention"], nat_best["recall"]
    sel = {NATIVE: dict(nat_best, feasible=True, diagnostic=False,
                        selected_under="unconstrained native ordering")}
    table = [dict(arm=NATIVE, chosen=nat_best, n_candidates=len(nat))]
    for arm in EXTENSION_ARMS:
        cand = checkpoints(dev_rows, arm)
        if len(cand) != PLANNED["development_checkpoints_per_family"] \
                or finite_checkpoints(cand):
            return None, None
        feas = [c for c in cand
                if c["retention"] >= r_native and c["recall"] >= c_native]
        if feas:
            best = sorted(feas, key=order_key)[0]
            sel[arm] = dict(best, feasible=True, diagnostic=False,
                            selected_under=("matched-retention feasible set "
                                            "(retention >= R_native AND "
                                            "recall >= C_native)"))
        else:
            best = sorted(cand, key=order_key)[0]
            sel[arm] = dict(best, feasible=False, diagnostic=True,
                            selected_under=("NO feasible checkpoint: best "
                                            "unconstrained checkpoint, "
                                            "carried as a DIAGNOSTIC "
                                            "endpoint, constraint-failing "
                                            "and not a feasible contender"))
        table.append(dict(arm=arm, chosen=sel[arm], n_candidates=len(cand),
                          n_feasible=len(feas)))
    status["selection"] = dict(
        selected=sel, table=table, r_native=r_native, c_native=c_native,
        rule=("native first by revision macro accuracy, then lower revision "
              "cross-entropy, then fewer updates, then lower learning rate; "
              "each extension then over checkpoints with retention >= "
              "R_native AND recall >= C_native (no allowance), by the same "
              "ordering; if none is feasible the best unconstrained "
              "checkpoint is carried as a flagged diagnostic endpoint. "
              "Development data only."))
    plan = deployment_plan(sel, status)
    return sel, plan


def deployment_plan(sel, status):
    """Separate from the scientific comparison: which MODEL a deployment
    selection would use for each extension family, decided on development
    revision only. A fallback is never counted as an improvement, and a
    native model is never labelled TSS."""
    plan = {}
    for arm in EXTENSION_ARMS:
        falls = [dict(kind="native", arm=NATIVE,
                      map=("(M, gamma, T) = (0, h, 0)"
                           if LAW_OF[arm] == FL.FILTERED else "kappa = 0"),
                      primary=sel[NATIVE]["primary"],
                      note=("the selected native checkpoint placed at this "
                            "family's exact native point; labelled native"))]
        if arm == GEN and sel[TSS]["feasible"]:
            falls.append(dict(
                kind="tss", arm=TSS, map="(M, gamma) = (0, 0)",
                primary=sel[TSS]["primary"],
                note=("a GENUINELY selected, feasible literal-TSS checkpoint "
                      "at the exact boundary")))
        best_fb = sorted(falls, key=lambda f: -f["primary"])[0]
        trained_ok = bool(sel[arm]["feasible"]
                          and sel[arm]["primary"] > best_fb["primary"])
        plan[arm] = dict(
            choice=("trained" if trained_ok else best_fb["kind"]),
            evaluate_arm=(arm if trained_ok else best_fb["arm"]),
            trained_development_primary=sel[arm]["primary"],
            trained_feasible=sel[arm]["feasible"],
            fallbacks=falls, best_fallback=best_fb,
            strict_improvement_required=True,
            note=("a trained endpoint is preferred only on STRICTLY higher "
                  "development revision than the best available fallback; a "
                  "selected fallback is a deployment choice, not evidence "
                  "that optimization learned to revert to a subfamily. If "
                  "the literal-TSS family itself has no feasible checkpoint "
                  "there is NO TSS fallback: a native choice is never "
                  "wrapped in an M = gamma = 0 map."))
    status["deployment_plan"] = plan
    return plan


# ---------------------------------------------------------------- screens ---
#: every comparison is declared here BEFORE execution
COMPARISONS = (
    ("extension_versus_literal_tss", GEN, TSS, "scientific",
     "Does departing from the exact TSS boundary help, from the same "
     "starting function?"),
    ("generalized_versus_native", GEN, NATIVE, "scientific",
     "Matched-retention screen of the master-law processing against the "
     "continued native baseline"),
    ("literal_tss_versus_native", TSS, NATIVE, "scientific",
     "Matched-retention screen of literal TSS processing against the "
     "continued native baseline"),
    ("generalized_versus_learned_operator", GEN, OPERATOR, "outside_claim",
     "Against the strong learned two-tap operator, which is OUTSIDE this "
     "nonnegative-gamma family (its learned kappa > 1 needs gamma < 0)"),
    ("learned_operator_versus_native", OPERATOR, NATIVE, "outside_claim",
     "The learned operator against the continued native baseline"),
    ("generalized_versus_frozen_source", GEN, ANCHOR, "descriptive",
     "Anchor: against the untrained source (descriptive only)"),
)


def screen(final_rows, sel, seeds=SOURCE_FINAL):
    out = dict(rule=("mean held-out revision difference >= +1 pp, all paired "
                     "revision differences positive, and mean retention AND "
                     "recall differences both >= 0"),
               safeguard_rule_reported_separately=(
                   "the historical -1 pp safeguard is reported beside it, "
                   "never as this study's criterion"),
               absent_literature_arms=list(ABSENT_LITERATURE),
               comparisons=[])
    for name, a, b, kind, why in COMPARISONS:
        c = ST.compare(final_rows, a, b, seeds)
        c["safeguard_passed_minus_one_pp"] = c.pop("passed")
        fa, fb = sel.get(a, {}), sel.get(b, {})
        constraint_failing = [x for x, s in ((a, fa), (b, fb))
                              if s and s.get("diagnostic")]
        c.update(name=name, candidate=a, kind=kind, question=why,
                 display=ARM_DISPLAY.get(b, b),
                 candidate_display=ARM_DISPLAY.get(a, a),
                 retention_direction=RP.direction(c["retention_difference"]),
                 recall_direction=RP.direction(c["recall_difference"]),
                 no_measured_decrease=bool(
                     c["complete_paired_seeds"]
                     and c["retention_difference"] >= 0
                     and c["recall_difference"] >= 0),
                 constraint_failing_endpoints=constraint_failing,
                 constrained_screen_available=bool(not constraint_failing),
                 endpoints_are_trained_models=True)
        c["promising_matched_retention"] = bool(
            c["complete_paired_seeds"]
            and c["mean_primary_difference"] is not None
            and c["mean_primary_difference"] >= 0.01
            and c["positive_in_all_seeds"] and c["no_measured_decrease"]
            and not constraint_failing)
        if constraint_failing:
            c["availability_note"] = (
                f"the constrained (matched-retention) screen is INFEASIBLE "
                f"here: {constraint_failing} had no development checkpoint "
                "meeting retention >= R_native AND recall >= C_native, so "
                "its endpoint is a flagged diagnostic. The comparison of the "
                "two trained endpoints is still reported, and keeps its "
                "constraint-failing label.")
        out["comparisons"].append(c)
    out["note"] = (
        "Every comparison above is between TRAINED endpoints of the named "
        "families; deployment fallbacks are reported separately in "
        "`deployment_plan` and are never counted as improvements. A native "
        "model is never labelled TSS. The generalized arm starts "
        "function-matched to the literal-TSS arm at T0 = h, so a difference "
        "between them is attributable to the departure from the boundary "
        "plus its own training trajectory, not to a different starting "
        "function. The learned two-tap operator is a separate strong "
        "comparator OUTSIDE this family. No Gated DeltaNet arm is present, "
        f"so no joint literature win follows ({list(ABSENT_LITERATURE)}). "
        "Three seeds on this small associative task are not significance, a "
        "benchmark or SOTA; a screen is a finite-sample condition, not "
        "statistical noninferiority and not a no-loss guarantee.")
    return out


# ------------------------------------------------------------------ main ---
def load_sources(source_run, status):
    """Restore the replication's Momentum sources, READ-ONLY and checksum
    verified (the same verifier as the completed studies)."""
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
                    default="/Users/durso/s5-runs/prospective-tss-containment")
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

    status = dict(
        run_id=run_id, out=out, backend=backend,
        study=("exact TSS containment: literal TSS versus the master-law "
               "residual processing on the same Momentum backbone"),
        specification="docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md",
        protocol="docs/PROSPECTIVE_TSS_CONTAINMENT_PROTOCOL.md",
        source_run=args.source_run,
        source_seeds=dict(development=SOURCE_DEV, final=list(SOURCE_FINAL)),
        final_seeds=list(SOURCE_FINAL), updates=UPDATES,
        checkpoints_at=list(VAL_AT),
        arms={a: ARM_DISPLAY[a] for a, _, _, _ in ARMS},
        arm_law={a: law for a, law, _, _ in ARMS},
        arm_frozen_leaves={a: list(f) for a, _, f, _ in ARMS},
        arm_regime={a: r for a, _, _, r in ARMS},
        filter_constants=dict(h=FL.H, T0=FL.T0, G_MIN=FL.G_MIN,
                              DELTA_FILTER=FL.DELTA_FILTER,
                              policy=("declared numerical gaps; a numerical "
                                      "robustness policy, NOT part of the "
                                      "derivation and NOT the acceptance "
                                      "certificate")),
        acceptance_gate=("finiteness and all three strict Jury conditions of "
                         "the rounded polynomial the production step "
                         "executes, in the loop and at every validation "
                         "point; isolated filter only"),
        carry=dict(filtered_executed=FL.CARRY_EXECUTED,
                   filtered_minimal_if_M_zero=FL.CARRY_MINIMAL_M_ZERO,
                   operator=OD.CARRY_REALS, native=PD.CARRY["momentum_delta"]),
        planned_work=planned_work(), streams=STREAM,
        stream_ranges=new_ranges(), previous_stream_ranges=previous_ranges(),
        learning_rates=dict(LRS),
        production_dtype=dict(x64=False, float_dtype="float32"),
        absent_literature_arms=list(ABSENT_LITERATURE),
        completed_study_references=[
            "/Users/durso/s5-runs/prospective-momentum/20260917-000431",
            SOURCE_RUN,
            "/Users/durso/s5-runs/prospective-same-backbone/20260917-134320"],
        heldout_policy=("one common held-out set, generated and hashed only "
                        "after all final runs and selections are frozen; "
                        "opening persisted before use"),
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
    eval_val_np = TK.generate_batch(STREAM["eval_validation"],
                                    VAL_PER_FAMILY)
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

    # --- declared start points, checked before any training
    starts = {}
    for seed, pn in sources.items():
        tss_p, gen_p = start_tree(TSS, pn), start_tree(GEN, pn)
        same = all(onp.array_equal(onp.asarray(tss_p[k]),
                                   onp.asarray(gen_p[k])) for k in tss_p)
        nat_pt = native_point_tree(pn)
        base = ST.evaluate("momentum_delta", pn, val_np)
        rec = ST.evaluate(FL.FILTERED, nat_pt, val_np)
        fails, mags = recovery_differences(base, rec)
        starts[str(seed)] = dict(
            processing_arms_start_identical=bool(same),
            native_point_recovery=fails,
            native_point_recovery_magnitudes=mags,
            native_point_recovery_strict_identity_diagnostic=(
                ST.identity_differences(base, rec)),
            start_filter=FL.filter_report(tss_p),
            native_point_filter=FL.filter_report(nat_pt),
            operator_start=OD.transition_report(start_tree(OPERATOR, pn)))
    status["start_points"] = dict(
        rows=starts,
        tolerances=dict(query_difference=RECOVERY_QUERY_TOL,
                        cross_entropy_relative=RECOVERY_CE_REL,
                        policy=("finite first, then the declared tolerances; "
                                "bitwise equality is reserved for stored "
                                "leaves and genuinely shared executed "
                                "operations. The stricter update-zero "
                                "identity check of the completed studies is "
                                "recorded as a DIAGNOSTIC only.")),
        note=("both processing arms start at M = gamma = 0, T = T0 = h: "
              "literal TSS, and the two-tap point kappa = 1. That start is "
              "NOT native Momentum. The native point of this family is "
              "(M, gamma, T) = (0, h, 0), whose value equivalence with the "
              "native rule is verified here at the declared identity "
              "tolerances - separately compiled float recurrences are never "
              "required to agree bitwise."))
    bad = [s for s, r in starts.items()
           if not r["processing_arms_start_identical"]
           or r["native_point_recovery"]
           or FL.filter_failure(r["start_filter"])
           or FL.filter_failure(r["native_point_filter"])]
    if bad:
        status["failed"] = f"start-point checks failed for seeds {bad}"
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

    def tree_of(arm, seed):
        pn = sources[seed]
        e = status["source"]["entries"][seed]
        return start_tree(arm, pn), f"{e['file']} sha256={e['sha256']}"

    dev_rows = []
    for arm, _, _, _ in TRAINED_ARMS:
        for tag, lr in LRS:
            p, label = tree_of(arm, SOURCE_DEV)
            r = run_one(arm, tag, lr, SOURCE_DEV, p, val_np, UPDATES, out,
                        deadline, args.reserve_s, status, "dev", label)
            if r is None:
                return 3, "INCOMPLETE"
            rec, _ = r
            dev_rows.append(rec)
            status["development"] = dev_rows
            save()
            if rec["invalid"]:
                status["failed"] = f"dev {arm}/{tag}: {rec['invalid']}"
                print(f"[!] {status['failed']}")
                return 4, "FAILED"
    p, label = tree_of(ANCHOR, SOURCE_DEV)
    rec, _ = run_one(ANCHOR, "-", 0.0, SOURCE_DEV, p, val_np, 0, out,
                     deadline, args.reserve_s, status, "dev", label)
    dev_rows.append(rec)
    status["development"] = dev_rows

    sel, plan = select(dev_rows, status)
    if sel is None:
        status["failed"] = "selection could not be formed"
        return 4, "FAILED"
    frozen_sel = copy.deepcopy(sel)
    frozen_plan = copy.deepcopy(plan)
    status["selection_frozen"] = copy.deepcopy(sel)
    status["deployment_plan_frozen"] = copy.deepcopy(plan)
    ST.write(os.path.join(out, "selection.json"),
             dict(selection=status["selection"], deployment=plan))
    for arm in EXTENSION_ARMS:
        print(f"[selection] {arm:<24} update {sel[arm]['update']:>3} lr "
              f"{sel[arm]['lr']} feasible={sel[arm]['feasible']} "
              f"diagnostic={sel[arm]['diagnostic']} deployment="
              f"{plan[arm]['choice']}")
    save()

    final_rows, finals = [], {}
    for arm, _, _, regime in ARMS:
        for seed in SOURCE_FINAL:
            p, label = tree_of(arm, seed)
            if regime == "frozen":
                tag, lr_value, updates = "-", 0.0, 0
            else:
                tag = frozen_sel[arm]["config"]
                lr_value = frozen_sel[arm]["lr"]
                updates = frozen_sel[arm]["update"]
            r = run_one(arm, tag, lr_value, seed, p, eval_val_np, updates,
                        out, deadline, args.reserve_s, status, "final", label)
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
                print(f"[!] {status['failed']}")
                return 4, "FAILED"
    if status["selection"]["selected"] != frozen_sel \
            or status["deployment_plan"] != frozen_plan:
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
    status["screen"] = screen(final_rows, frozen_sel)
    status["deployment_outcome"] = deployment_outcome(final_rows, frozen_plan)
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
        print(f"[screen] {c['name']:<38} kind={c['kind']:<13} "
              f"promising={c['promising_matched_retention']} "
              f"no_measured_decrease={c['no_measured_decrease']} mean "
              f"{c['mean_primary_difference']}")
    status["complete"] = True
    return 0, "PASS"


def deployment_outcome(final_rows, plan):
    """What each extension family's frozen deployment choice scores on
    held-out data, taken from the ALREADY TRAINED final runs. Reported apart
    from the scientific comparisons; a fallback is a deployment choice."""
    out = {}
    for arm, pl in plan.items():
        src = pl["evaluate_arm"]
        rows = [r for r in final_rows if r["rule"] == src and "heldout" in r]
        mean = (float(onp.mean([r["heldout"]["primary"] for r in rows]))
                if rows else None)
        out[arm] = dict(
            choice=pl["choice"], evaluated_model=src,
            label=("trained " + arm if pl["choice"] == "trained"
                   else ("native Momentum model (NOT literal TSS)"
                         if pl["choice"] == "native"
                         else "literal TSS model")),
            heldout_primary_mean=mean,
            counted_as_improvement=False,
            note=("a selected fallback reuses an already trained baseline "
                  "model without further optimization; it does not show that "
                  "joint optimization learned to revert to a subfamily"))
    return out


if __name__ == "__main__":
    sys.exit(main())
