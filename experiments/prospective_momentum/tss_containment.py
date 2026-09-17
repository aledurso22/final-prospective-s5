"""Does departing from the exact literal-TSS boundary help, on the same
Momentum backbone?

Specification: docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md (d95266d). Frozen
protocol: docs/PROSPECTIVE_TSS_CONTAINMENT_PROTOCOL.md. Clearance:
PROSPECTIVE_TSS_IMPLEMENTATION_CLEARANCE_d95266d_2026_09_17.md. Implementation
review of 7613c86: PROSPECTIVE_TSS_IMPLEMENTATION_REVIEW_7613c86_2026_09_17.md
(R1-R5 and the reporting corrections; dispositions in the protocol s11).
NOT AUTHORIZED TO RUN until the corrected implementation is cleared.

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
    native_frozen           the restored source, evaluation only (anchor)

Sources: the replication's independently pretrained Momentum checkpoints,
READ-ONLY (seed 500 development, 501-503 final), with fresh streams. Full
BPTT and the existing unweighted query cross-entropy are unchanged.

Every selectable checkpoint (updates 0, 25, 50, 100, 200 and each final
endpoint) is VALIDATED AND PERSISTED before it can enter selection: finite
parameters and optimizer state, finite metrics and processing-state
diagnostics, acceptance of every coefficient set the evaluation program
actually executed, frozen-constant invariants and the arm's domain. A single
invalid checkpoint fails the run; it cannot be hidden by a valid endpoint.
Endpoints are NAMED-FAMILY endpoints and may have zero updates (review R1).
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

UPDATES = ST.UPDATES                       # 200
LRS = ST.LRS                               # ("A", 0.003), ("B", 0.01)
#: selectable checkpoints; every family gets the same opportunities
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

#: DECISIVE recovery policy (review R2): direct finite-first function
#: recovery - logits and the W, U carries on representative episodes of every
#: restored source - at the existing float32 trajectory tolerance.
TRAJ32 = 2e-5
RECOVERY_EPISODES_PER_FAMILY = 2
#: aggregate count / cross-entropy differences are RECORDED as a diagnostic
#: only, against the previously declared identity tolerance (zero count
#: difference, ST.IDENTITY_CE_REL relative cross-entropy); they never decide
COUNT_CONSISTENCY_TOL = 1e-6

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
ABSENT_LITERATURE = ("gated_delta",)

PLANNED = dict(trained_runs_development=len(TRAINED_ARMS) * len(LRS),
               named_family_final_endpoints=(len(TRAINED_ARMS)
                                             * len(SOURCE_FINAL)),
               frozen_source_evaluations=1 + len(SOURCE_FINAL),
               development_checkpoints_per_family=len(VAL_AT) * len(LRS) - 1)
PLANNED["max_total_updates"] = (PLANNED["trained_runs_development"]
                                + PLANNED["named_family_final_endpoints"]
                                ) * UPDATES
PLANNED["note"] = ("a final endpoint whose selected update is 0 is a "
                   "zero-update NAMED-FAMILY endpoint, not a frozen-source "
                   "anchor; it is counted with the final endpoints")


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


# ------------------------------------------- loss/eval with executed values --
def _episode_loss_f(rule, p, ep):
    """The unchanged query cross-entropy (same formula as `ST._episode_loss`),
    whose aux also returns the coefficients THIS program executed and the
    processing-state diagnostic."""
    out = PM.rollout(rule, p, ep)
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)
    ce = optax.softmax_cross_entropy(
        out["logits"], jax.nn.one_hot(lab, TK.N_VALUES)) * q
    correct = (jnp.argmax(out["logits"], -1) == lab) * q
    aux = dict(ce=ce, correct=correct.astype(jnp.float32),
               q=q.astype(jnp.float32), w_norm=out["w_norm"],
               aux_norm=out["aux_norm"],
               coeff=tuple(out["coeff"][k] for k in FL.COEFF_NAMES),
               proc_max_abs=out["proc_max_abs"])
    return jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0), aux


def batch_loss_f(rule, p, eps):
    losses, aux = jax.vmap(lambda e: _episode_loss_f(rule, p, e))(eps)
    return jnp.mean(losses), aux


@partial(jax.jit, static_argnums=(0,))
def eval_batch_f(rule, p, eps):
    _, aux = batch_loss_f(rule, p, eps)
    return aux


def executed_sets_from(auxes):
    """Every DISTINCT executed coefficient set across evaluation chunks (a
    chunk of another shape is another compiled program). Values are taken
    from the programs' outputs, never recomputed."""
    seen, sets = set(), []
    for aux in auxes:
        arrs = [onp.asarray(x) for x in aux["coeff"]]
        for i in range(arrs[0].shape[0]):
            vals = tuple(float(a[i]) for a in arrs)
            key = onp.asarray(vals, onp.float64).tobytes()   # bitwise identity
            if key not in seen:
                seen.add(key)
                sets.append(dict(zip(FL.COEFF_NAMES, vals)))
    return sets


def evaluate_arm(arm, p, eps_np):
    """Metrics plus, for the processing law, the executed coefficient sets
    and the processing-state diagnostic of the SAME compiled evaluation."""
    rule = LAW_OF[arm]
    if rule != FL.FILTERED:
        return ST.evaluate(rule, p, eps_np), None
    sink = []
    m = ST.evaluate(rule, p, eps_np, batch_fn=eval_batch_f, aux_sink=sink)
    proc = onp.concatenate([onp.asarray(a["proc_max_abs"]).ravel()
                            for a in sink])
    m["processing_state"] = dict(
        max_abs_y_yprev_Rprev=(float(proc.max()) if proc.size else None),
        finite=bool(proc.size and onp.all(onp.isfinite(proc))),
        meaning=("max |entry| of y, y_prev and R_prev over every token of "
                 "every evaluated episode"))
    sets = executed_sets_from(sink)
    m["executed_coefficient_sets"] = sets
    return m, sets


# ------------------------------------------------------------- train step ---
def leaf_mask(p, frozen):
    return {k: jnp.asarray(0.0 if k in frozen else 1.0,
                           dtype=jnp.asarray(v).dtype) for k, v in p.items()}


@partial(jax.jit, static_argnums=(0, 1))
def train_step_filtered(rule, frozen, p, opt, eps, lr):
    """The SAME loss and full BPTT. Frozen coefficient leaves are removed from
    the optimizer input (gradient zeroed BEFORE the transformation, update
    masked again) and restored bitwise by the repair. The gate reads the
    coefficients THIS update's forward pass executed; the repaired tree's
    coefficients are executed, and gated, by the next forward pass or by the
    checkpoint evaluation that every endpoint has."""
    (loss, aux), g = jax.value_and_grad(batch_loss_f, argnums=1,
                                        has_aux=True)(rule, p, eps)
    guard = FL.in_loop_guard(aux["coeff"])
    mask = leaf_mask(p, frozen)
    g_masked = jax.tree_util.tree_map(lambda x, m: x * m, g, mask)
    raw, opt = ST.TX.update(g_masked, opt, p)
    upd = jax.tree_util.tree_map(lambda u, m: -lr * u * m, raw, mask)
    p = optax.apply_updates(p, upd)
    p, tel = FL.repair(p, frozen=frozen)
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    grads = {k: g[k][0] for k in FL.LEAVES}
    gate = dict(ok=jnp.all(guard["ok"]), jury_min=jnp.min(guard["jury_min"]),
                **{k: guard[k][0] for k in FL.COEFF_NAMES},
                proc_max_abs=jnp.max(aux["proc_max_abs"]))
    return (p, opt, loss, acc, optax.global_norm(g), optax.global_norm(upd),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]),
            dict(tel, **{"gate_" + k: v for k, v in gate.items()}), grads)


def host_step(arm, p, opt, seed, u, lr, hist):
    """One host step. Returns the step's scalars and its telemetry record;
    the CALLER rejects non-finite scalars and a failed gate at THIS step."""
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


def step_failure(arm, scalars, rec):
    """None if THIS step is acceptable (review R3: checked at every step,
    not only the last one)."""
    bad = MS.measured_scalar_failures(scalars)
    if bad:
        return f"non-finite step scalars {bad} at update {rec['update']}"
    if LAW_OF[arm] == FL.FILTERED:
        if not rec["executed_filter_ok"]:
            return (f"executed filter gate failed at update {rec['update']}: "
                    f"{rec['executed']} jury_min={rec['jury_min']}")
        vals = [rec[k] for k in FL.LEAVES] + [rec["grad_" + k]
                                              for k in FL.LEAVES]
        if not all(onp.isfinite(v) for v in vals + [rec["proc_max_abs"]]):
            return f"non-finite coefficient telemetry at update {rec['update']}"
    return None


# ------------------------------------------------- validity and reports ----
def validate(arm, p, executed_sets=None):
    """None if acceptable, else the reason. Nothing is clamped here. For the
    processing law `executed_sets` must be the coefficient sets the
    evaluation program executed; a missing record is itself a failure."""
    rule = LAW_OF[arm]
    if not ST.all_finite(p):
        return "non-finite parameters"
    gr = PD.gate_range_report(p)
    if not gr["gates_valid"]:
        return f"gate range/finiteness failed: {gr}"
    if rule == FL.FILTERED:
        if not executed_sets:
            return "no executed coefficient record for the processing law"
        for ex in executed_sets:
            bad = FL.filter_failure(FL.filter_report(ex))
            if bad:
                return bad
        if arm == TSS:
            M = float(onp.asarray(p["fil_M"]).ravel()[0])
            gam = float(onp.asarray(p["fil_gamma"]).ravel()[0])
            ex_ok = all(ex["M"] == 0.0 and ex["gamma"] == 0.0
                        for ex in executed_sets)
            if not (M == 0.0 and gam == 0.0 and ex_ok):
                return (f"literal-TSS arm left its boundary: stored M={M} "
                        f"gamma={gam}; executed {executed_sets}")
        rep = PD.transition_report(p, "momentum_delta")
        bad = {k: v for k, v in rep["classification"].items()
               if k in ("unstable", "nonfinite")}
        return (f"Momentum-block frozen-token transitions {bad}"
                if bad else None)
    if rule == OD.ORDINARY:
        return OD.transition_failure(OD.transition_report(p))
    return ST.validate(rule, p)


def checkpoint_failure(arm, p, opt, source_p, metrics, executed_sets):
    """The acceptance of ONE selectable checkpoint (review R3)."""
    if not ST.all_finite(p):
        return "non-finite parameters"
    if opt is not None and not ST.all_finite(opt):
        return "non-finite optimizer state"
    if not ST.metrics_finite(metrics):
        return "non-finite evaluation metric or state norm"
    if LAW_OF[arm] == FL.FILTERED:
        ps = metrics.get("processing_state") or {}
        if not ps.get("finite"):
            return f"non-finite processing state (y, y_prev, R_prev): {ps}"
    changed = frozen_leaf_differences(p, source_p, arm)
    if changed:
        return f"stored constant leaves changed: {changed}"
    return validate(arm, p, executed_sets)


def coefficient_report(arm, p, executed_sets=None, eps_np=None):
    rule = LAW_OF[arm]
    out = dict(arm=arm, law=rule, regime=REGIME_OF[arm],
               frozen_coefficient_leaves=list(FROZEN_LEAVES[arm]),
               gate_table=PD.gate_range_report(p))
    if rule == FL.FILTERED:
        out["executed_filter"] = [FL.filter_report(ex)
                                  for ex in (executed_sets or [])]
        out["momentum_block_diagnostic"] = PD.transition_report(
            p, "momentum_delta")
        out["carry_note"] = (
            f"IMPLEMENTED carry W, U, y, y_prev, R_prev = "
            f"{FL.CARRY_EXECUTED} real numbers in both processing arms; "
            f"{FL.CARRY_MINIMAL_M_ZERO} is only the theoretical minimum of a "
            "law with M fixed at zero, not the implemented cost")
        out["scope_note"] = (
            "acceptance classifies the rounded coefficients the evaluation "
            "program executed: a coefficient-polynomial result for the "
            "isolated processing filter, not a theorem about every "
            "floating-point trajectory, the closed-loop memory, switching, "
            "or the Momentum block. The coefficient domain is broader than "
            "the passive-compartment domain.")
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
        counts["carry_real_numbers_implemented"] = FL.CARRY_EXECUTED
        counts["carry_real_numbers_theoretical_minimum_if_M_zero"] = (
            FL.CARRY_MINIMAL_M_ZERO)
    return counts


def frozen_leaf_differences(p, source_p, arm):
    """Stored constants must not move: a storage invariant, checked bitwise."""
    out = {}
    for k in FROZEN_LEAVES[arm]:
        a, b = onp.asarray(p[k]), onp.asarray(source_p[k])
        if not onp.array_equal(a, b):
            out[k] = [float(onp.max(onp.abs(a.astype(onp.float64)
                                            - b.astype(onp.float64)))),
                      a.tolist(), b.tolist()]
    return out


def load_params(path, like):
    """Restore a saved parameter tree and verify its leaf set, shapes and
    dtypes against `like`."""
    from flax import serialization
    with open(path, "rb") as fh:
        raw = serialization.msgpack_restore(fh.read())
    if set(raw) != set(like):
        raise ValueError(f"{path}: leaf set {sorted(raw)} != {sorted(like)}")
    out = {}
    for k, t in like.items():
        v, t = onp.asarray(raw[k]), onp.asarray(t)
        if v.shape != t.shape or v.dtype != t.dtype:
            raise ValueError(f"{path}/{k}: {v.shape} {v.dtype} != "
                             f"{t.shape} {t.dtype}")
        out[k] = jnp.asarray(v)
    return out


# ------------------------------------------------------------ start trees --
def start_tree(arm, native_p):
    rule = LAW_OF[arm]
    if rule == "momentum_delta":
        return dict(native_p)
    return PM.add_extension(native_p, rule)


def native_point_tree(native_p):
    """(M, gamma, T) = (0, h, 0): the processing law's exact native point."""
    dt = native_p["A_log"].dtype
    M, gam, T = FL.native_point()
    return dict(native_p, fil_M=jnp.full((1,), M, dtype=dt),
                fil_gamma=jnp.full((1,), gam, dtype=dt),
                fil_T=jnp.full((1,), T, dtype=dt))


# ------------------------------------------------ direct function recovery --
@partial(jax.jit, static_argnums=(0,))
def rollout_outputs(rule, p, ep):
    out = PM.rollout(rule, p, ep)
    res = dict(logits=out["logits"], W=out["final_carry"][0],
               U=out["final_carry"][1])
    if rule == FL.FILTERED:
        res["coeff"] = tuple(out["coeff"][k] for k in FL.COEFF_NAMES)
    return res


def recovery_episodes(val_np):
    """Representative episodes: the first RECOVERY_EPISODES_PER_FAMILY of
    each task family, fixed before execution."""
    idx = []
    for fi in range(len(TK.FAMILIES)):
        idx += [int(i) for i in onp.flatnonzero(val_np["family"] == fi)
                [:RECOVERY_EPISODES_PER_FAMILY]]
    return idx


def _rel(a, b):
    a, b = onp.asarray(a, onp.float64), onp.asarray(b, onp.float64)
    if not (onp.all(onp.isfinite(a)) and onp.all(onp.isfinite(b))):
        return float("inf")
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def direct_recovery(native_p, val_np):
    """DECISIVE native-point recovery: finite first, then logits and the W, U
    carries of the processing law at (0, h, 0) against the native rule on
    every representative episode, at TRAJ32."""
    rows, fails = [], []
    nat_pt = native_point_tree(native_p)
    for i in recovery_episodes(val_np):
        ep = {k: jnp.asarray(val_np[k][i])
              for k in ("key_id", "val_id", "event", "label")}
        a = rollout_outputs("momentum_delta", native_p, ep)
        b = rollout_outputs(FL.FILTERED, nat_pt, ep)
        errs = dict(logits=_rel(b["logits"], a["logits"]),
                    W=_rel(b["W"], a["W"]), U=_rel(b["U"], a["U"]))
        rows.append(dict(episode=i, relative_errors=errs,
                         executed=FL.coefficient_values(b["coeff"])))
        for k, e in errs.items():
            if not (onp.isfinite(e) and e <= TRAJ32):
                fails.append(f"episode {i} {k}: {e:.3e} (TRAJ32 {TRAJ32})")
    return fails, rows


def count_differences(ref, other):
    """RECORDED DIAGNOSTIC: aggregate per-category differences from INTEGER
    correct counts (review R4). Each count is reconstructed from accuracy x n
    with a documented consistency check; mismatched or non-positive
    denominators and non-finite values are MALFORMED. `identical` uses the
    previously declared identity tolerance (zero count difference and
    ST.IDENTITY_CE_REL relative cross-entropy); it never decides recovery."""
    malformed, per = [], {}
    for fam in TK.FAMILIES:
        for cn in TK.CATEGORIES:
            a, b = ref[fam]["by_category"][cn], other[fam]["by_category"][cn]
            name = f"{fam}/{cn}"
            vals = [a.get("accuracy"), b.get("accuracy"),
                    a.get("cross_entropy"), b.get("cross_entropy")]
            if any(v is None or not onp.isfinite(v) for v in vals):
                malformed.append(f"{name}: non-finite {vals}")
                continue
            na, nb = a.get("n"), b.get("n")
            if not (isinstance(na, int) and isinstance(nb, int)
                    and na == nb and na > 0):
                malformed.append(f"{name}: denominators {na} vs {nb}")
                continue
            ca, cb = a["accuracy"] * na, b["accuracy"] * nb
            ia, ib = int(round(ca)), int(round(cb))
            if (abs(ca - ia) > COUNT_CONSISTENCY_TOL * na
                    or abs(cb - ib) > COUNT_CONSISTENCY_TOL * nb):
                malformed.append(f"{name}: non-integer counts {ca} {cb}")
                continue
            dce = (abs(a["cross_entropy"] - b["cross_entropy"])
                   / max(abs(a["cross_entropy"]), 1e-12))
            per[name] = dict(count_difference=abs(ia - ib),
                             cross_entropy_relative=float(dce), n=na)
    identical = bool(not malformed and all(
        v["count_difference"] == 0
        and v["cross_entropy_relative"] <= ST.IDENTITY_CE_REL
        for v in per.values()))
    return dict(malformed=malformed, per_category=per,
                identical_within_declared_identity_tolerance=identical,
                decisive=False)


# --------------------------------------------------------------- one run ---
def run_one(arm, tag, lr_value, seed, source_p, val_np, updates, out,
            deadline, reserve_s, status, stage, source_label, save=None,
            val_at=None):
    """One named-family endpoint of `updates` optimizer updates (possibly 0),
    or the frozen anchor. Every checkpoint is validated and persisted, and
    its record is written to status before the next update (review R3)."""
    rule, regime = LAW_OF[arm], REGIME_OF[arm]
    t0 = time.time()
    p = dict(source_p)
    grid = VAL_AT if val_at is None else tuple(val_at)
    ckpts = tuple(u for u in grid if u <= updates)
    if updates not in ckpts:
        ckpts = ckpts + (updates,)
    stem = f"{stage}_{arm}_{tag}_seed{seed}"
    log = status.setdefault("checkpoint_log", [])
    save = save or (lambda: None)
    opt = None if regime == "frozen" else ST.TX.init(p)
    curve, val_hist, hist, bad = [], [], [], None
    for u in range(updates + 1):
        if u in ckpts:
            m, sets = evaluate_arm(arm, p, val_np)
            fail = checkpoint_failure(arm, p, opt, source_p, m, sets)
            pfile = os.path.join(out, "params", f"{stem}_u{u}.msgpack")
            ST.save_tree(pfile, p)
            ofile = None
            if opt is not None:
                ofile = os.path.join(out, "params", f"{stem}_u{u}_opt.msgpack")
                ST.save_tree(ofile, opt)
            entry = dict(update=u, primary=m["primary"],
                         revision_ce=m["revision_ce"],
                         retention=m["retention_revision_untouched"],
                         recall=m["recall_overall"],
                         state_norms=m["state_norms"],
                         processing_state=m.get("processing_state"),
                         executed_filter=([FL.filter_report(ex)
                                           for ex in sets] if sets else None),
                         accepted=fail is None, failure=fail,
                         params_file=pfile, opt_file=ofile)
            val_hist.append(dict(entry, full=m))
            log.append(dict(entry, stage=stage, arm=arm, config=tag,
                            lr=lr_value, seed=seed))
            save()
            if fail:
                bad = f"checkpoint at update {u}: {fail}"
                break
        if u == updates:
            break
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{stage}:{arm}/{tag}/seed{seed} stopped at update {u} of "
                f"{updates}")
            return None
        lr = jnp.asarray(lr_value, dtype=jnp.float32)
        p, opt, scalars, rec = host_step(arm, p, opt, seed, u, lr, hist)
        bad = step_failure(arm, scalars, rec)
        if bad:
            break
        if u % 25 == 0 or u == updates - 1:
            curve.append(dict(update=u, **scalars))
    last_ck = val_hist[-1] if val_hist else None
    final = (last_ck["full"] if last_ck and last_ck["update"] == updates
             and bad is None else None)
    if bad is None and final is None:
        bad = "endpoint checkpoint missing"
    first = val_hist[0]["full"] if val_hist else None
    rec = dict(tag=stage, rule=arm, law=rule, regime=regime,
               display=ARM_DISPLAY[arm], config=tag, lr=lr_value, seed=seed,
               source=source_label, wall_s=time.time() - t0, curve=curve,
               updates=updates,
               endpoint_kind=("frozen_source_anchor" if regime == "frozen"
                              else ("zero_update_named_family_endpoint"
                                    if updates == 0
                                    else "trained_named_family_endpoint")),
               validation=[{k: v for k, v in h.items() if k != "full"}
                           for h in val_hist],
               start_validation=first, final_validation=final,
               training_gain=(dict(
                   primary=final["primary"] - first["primary"],
                   revision_ce=final["revision_ce"] - first["revision_ce"],
                   retention=final["retention_revision_untouched"]
                   - first["retention_revision_untouched"],
                   recall=final["recall_overall"] - first["recall_overall"])
                   if final and first else None),
               params=parameter_counts(arm, p), carry=PD.CARRY[rule],
               coefficients_final=coefficient_report(
                   arm, p, (last_ck or {}).get("executed_filter") and [
                       r["executed"] for r in last_ck["executed_filter"]],
                   val_np),
               coefficient_history=hist,
               frozen_leaf_differences=frozen_leaf_differences(p, source_p,
                                                               arm),
               endpoint_params_file=(last_ck or {}).get("params_file"),
               invalid=bad)
    return rec, p


# ------------------------------------------------------------- preflight ---
def preflight(sources, val_np, out, status):
    """Times the ACTUAL paths of every trained arm on disposable state: every
    measured step is checked (not only the last), and one full checkpoint
    (evaluation, acceptance, parameter and optimizer persistence) is timed."""
    rows, failures, retraced_any = [], [], False
    p0 = sources[SOURCE_DEV]
    lr = jnp.asarray(LRS[0][1], dtype=jnp.float32)
    step_s, ckpt_s, heldout_s = {}, {}, {}
    scratch = os.path.join(out, "preflight_disposable")
    for arm, rule, frozen, _ in TRAINED_ARMS:
        p = start_tree(arm, p0)
        opt = ST.TX.init(p)
        hist, acc_bad = [], None
        t0 = time.time()
        p2, opt2, sc, rec = host_step(arm, p, opt, SOURCE_DEV, 0, lr, hist)
        compile_s = time.time() - t0
        acc_bad = acc_bad or step_failure(arm, sc, rec)
        p2, opt2, sc, rec = host_step(arm, p2, opt2, SOURCE_DEV, 1, lr, hist)
        acc_bad = acc_bad or step_failure(arm, sc, rec)
        cache = (train_step_filtered if rule == FL.FILTERED else ST.train_step)
        n0 = cache._cache_size()
        t1 = time.time()
        for u in range(2, 7):
            p2, opt2, sc, rec = host_step(arm, p2, opt2, SOURCE_DEV, u, lr,
                                          hist)
            acc_bad = acc_bad or step_failure(arm, sc, rec)
        step_s[arm] = (time.time() - t1) / 5.0
        retraced_any |= cache._cache_size() != n0
        t2 = time.time()
        evaluate_arm(arm, p2, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        m, sets = evaluate_arm(arm, p2, val_np)
        heldout_s[arm] = time.time() - t3
        fail = checkpoint_failure(arm, p2, opt2, p, m, sets)
        ST.save_tree(os.path.join(scratch, f"{arm}.msgpack"), p2)
        ST.save_tree(os.path.join(scratch, f"{arm}_opt.msgpack"), opt2)
        coefficient_report(arm, p2, sets, val_np)
        ckpt_s[arm] = time.time() - t3
        acc_bad = acc_bad or fail
        if acc_bad:
            failures.append(f"{arm}: {acc_bad}")
        rows.append(dict(arm=arm, law=rule, regime=REGIME_OF[arm],
                         frozen_coefficient_leaves=list(frozen),
                         compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s,
                         step_s=step_s[arm], checkpoint_s=ckpt_s[arm],
                         evaluation_s=heldout_s[arm],
                         acceptance_failure=acc_bad,
                         measured_steps_checked=len(hist),
                         params=parameter_counts(arm, p2),
                         carry=PD.CARRY[rule]))
        print(f"[preflight] {arm:<24} step {step_s[arm] * 1e3:6.2f}ms "
              f"checkpoint {ckpt_s[arm]:5.2f}s compile {compile_s:4.1f}s "
              f"stored {parameter_counts(arm, p2)['stored']} trainable "
              f"{parameter_counts(arm, p2)['trainable']} carry "
              f"{PD.CARRY[rule]} failure {acc_bad}")
    n_runs = len(LRS) + len(SOURCE_FINAL)
    total = 0.0
    for arm, _, _, _ in TRAINED_ARMS:
        total += n_runs * (UPDATES * step_s[arm] + len(VAL_AT) * ckpt_s[arm])
        total += len(SOURCE_FINAL) * heldout_s[arm]           # held-out
    anchor = max(ckpt_s.values())
    total += (1 + 2 * len(SOURCE_FINAL)) * anchor             # frozen arm
    host_s = 40.0                            # ALLOWANCE, recorded as such
    total += host_s
    timing = dict(projected_remaining_s=total, host_allowance_s=host_s)
    if not all(onp.isfinite(v) and v >= 0 for v in timing.values()):
        failures.append(f"non-finite or negative timing {timing}")
    status["preflight"] = dict(
        rows=rows, retraced_any=bool(retraced_any), failures=failures,
        planned=planned_work(), **timing,
        note=("disposable state; compilation incurred here is not "
              "re-counted. Every run is projected at the full update count "
              "with every checkpoint validated and persisted (worst case)."))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, bool(retraced_any), failures


# ------------------------------------------------------------- selection ---
def order_key(c):
    """The single development ordering: revision macro accuracy, then lower
    revision cross-entropy, then fewer updates, then lower learning rate."""
    return (-c["primary"], c["revision_ce"], c["update"], c["lr"])


def checkpoints(dev_rows, arm):
    """Every development checkpoint of one arm, with the identical
    update-zero checkpoint deduplicated (slot A keeps it). Carries the
    acceptance flag and the persisted tree's identity."""
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
                            accepted=v.get("accepted"),
                            params_file=v.get("params_file")))
    return out


def ineligible_checkpoints(cand):
    """None if every checkpoint is finite AND recorded as accepted; else the
    offending rows. Such a checkpoint never enters selection: the selection
    fails instead (review R3)."""
    bad = [c for c in cand
           if c.get("accepted") is not True
           or not all(onp.isfinite(c[k]) for k in
                      ("primary", "revision_ce", "retention", "recall"))]
    return bad or None


def select(dev_rows, status):
    nat = checkpoints(dev_rows, NATIVE)
    if len(nat) != PLANNED["development_checkpoints_per_family"] \
            or ineligible_checkpoints(nat):
        return None, None
    nat_best = sorted(nat, key=order_key)[0]
    r_native, c_native = nat_best["retention"], nat_best["recall"]
    sel = {NATIVE: dict(nat_best, feasible=True, diagnostic=False,
                        selected_under="unconstrained native ordering")}
    table = [dict(arm=NATIVE, chosen=nat_best, n_candidates=len(nat))]
    for arm in EXTENSION_ARMS:
        cand = checkpoints(dev_rows, arm)
        if len(cand) != PLANNED["development_checkpoints_per_family"] \
                or ineligible_checkpoints(cand):
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
              "each extension then over ACCEPTED checkpoints with retention "
              ">= R_native AND recall >= C_native (no allowance), by the same "
              "ordering; if none is feasible the best unconstrained "
              "checkpoint is carried as a flagged diagnostic endpoint. "
              "Development data only."))
    plan = deployment_plan(sel, status)
    return sel, plan


#: declared final tie rule for deployment fallbacks that tie on all four
#: ordering keys: the native model, the simpler one
FALLBACK_TIE_ORDER = {"native": 0, "tss": 1}


def deployment_plan(sel, status):
    """Separate from the scientific comparison: which MODEL a deployment
    selection would use for each extension family, decided on development
    data with the frozen full ordering. A fallback is never counted as an
    improvement, and a native model is never labelled TSS."""
    plan = {}

    def identity(s, arm):
        return dict(arm=arm, executed_family=LAW_OF[arm],
                    config=s.get("config"), lr=s.get("lr"),
                    update=s.get("update"),
                    development_params_file=s.get("params_file"))

    nat = sel[NATIVE]
    for arm in EXTENSION_ARMS:
        if LAW_OF[arm] == FL.FILTERED and arm == GEN:
            nat_map = ("native model placed at this family's exact native "
                       "point (M, gamma, T) = (0, h, 0)")
        elif arm == OPERATOR:
            nat_map = "native model placed at kappa = 0"
        else:
            nat_map = ("EXTERNAL deployment selection of the native model; "
                       "NOT a point of the literal-TSS family")
        falls = [dict(kind="native", map=nat_map,
                      primary=nat["primary"], revision_ce=nat["revision_ce"],
                      update=nat["update"], lr=nat["lr"],
                      identity=identity(nat, NATIVE))]
        if arm == GEN and sel[TSS]["feasible"]:
            t = sel[TSS]
            falls.append(dict(
                kind="tss", map="(M, gamma) = (0, 0): the exact boundary",
                primary=t["primary"], revision_ce=t["revision_ce"],
                update=t["update"], lr=t["lr"], identity=identity(t, TSS)))
        best_fb = sorted(falls, key=lambda f: order_key(f)
                         + (FALLBACK_TIE_ORDER[f["kind"]],))[0]
        trained_ok = bool(sel[arm]["feasible"]
                          and sel[arm]["primary"] > best_fb["primary"])
        choice = "trained" if trained_ok else best_fb["kind"]
        chosen_identity = (identity(sel[arm], arm) if trained_ok
                           else best_fb["identity"])
        plan[arm] = dict(
            choice=choice, evaluate_arm=chosen_identity["arm"],
            executed_family=chosen_identity["executed_family"],
            chosen_checkpoint=chosen_identity,
            parameter_map=("the family's own trained parameters"
                           if trained_ok else best_fb["map"]),
            trained_development_primary=sel[arm]["primary"],
            trained_feasible=sel[arm]["feasible"],
            fallbacks=falls, best_fallback=best_fb,
            fallback_ordering=("the frozen full ordering (primary, CE, "
                               "updates, learning rate), then the declared "
                               "tie rule native before tss"),
            strict_improvement_required=True,
            note=("a trained endpoint is preferred only on STRICTLY higher "
                  "development revision than the best available fallback; a "
                  "selected fallback is a deployment choice, not evidence "
                  "that optimization learned to revert to a subfamily. With "
                  "no feasible literal-TSS checkpoint there is NO TSS "
                  "fallback, and a native model is never labelled TSS."))
    status["deployment_plan"] = plan
    return plan


# ---------------------------------------------------------------- screens ---
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
        kinds = {x: sorted({r.get("endpoint_kind") for r in final_rows
                            if r["rule"] == x}) for x in (a, b)}
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
                 endpoint_kinds=kinds)
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
                "two named-family endpoints is still reported, and keeps its "
                "constraint-failing label.")
        out["comparisons"].append(c)
    out["note"] = (
        "Every comparison above is between NAMED-FAMILY final endpoints (a "
        "selected update of 0 is a zero-update endpoint of that family, "
        "listed in `endpoint_kinds`); deployment fallbacks are reported "
        "separately and never counted as improvements. A native model is "
        "never labelled TSS. The generalized arm starts function-matched to "
        "the literal-TSS arm at T0 = h. The learned two-tap operator is a "
        "separate strong comparator OUTSIDE this family. No Gated DeltaNet "
        f"arm is present ({list(ABSENT_LITERATURE)}), so no joint literature "
        "win follows. Three seeds on this small associative task are not "
        "significance, a benchmark or SOTA; a screen is a finite-sample "
        "condition, not statistical noninferiority and not a no-loss "
        "guarantee.")
    return out


def deployment_outcome(final_rows, plan):
    out = {}
    for arm, pl in plan.items():
        src = pl["evaluate_arm"]
        rows = [r for r in final_rows if r["rule"] == src and "heldout" in r]
        mean = (float(onp.mean([r["heldout"]["primary"] for r in rows]))
                if rows else None)
        out[arm] = dict(
            choice=pl["choice"], evaluated_model=src,
            executed_family=pl["executed_family"],
            chosen_checkpoint=pl["chosen_checkpoint"],
            parameter_map=pl["parameter_map"],
            final_endpoint_files={r["seed"]: r.get("endpoint_params_file")
                                  for r in rows},
            label=("named-family endpoint of " + arm
                   if pl["choice"] == "trained"
                   else ("native Momentum model (NOT literal TSS)"
                         if pl["choice"] == "native"
                         else "literal TSS model")),
            heldout_primary_mean=mean, counted_as_improvement=False,
            note=("a selected fallback reuses an already trained baseline "
                  "model without further optimization; it does not show that "
                  "joint optimization learned to revert to a subfamily"))
    return out


# ------------------------------------------------------------------ main ---
def load_sources(source_run, status):
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
        executed_form=("y_next = a y - b y_prev + c R - d R_prev with "
                       "a = [2M + h(gamma+T) - h^2]/A, b = M/A, "
                       "c = [h^2 + hT]/A, d = hT/A: the same equation"),
        filter_constants=dict(h=FL.H, T0=FL.T0, G_MIN=FL.G_MIN,
                              DELTA_FILTER=FL.DELTA_FILTER,
                              policy=("declared robustness gaps enforced by "
                                      "the repair; not part of the "
                                      "derivation, no rounding guarantee, "
                                      "not the acceptance certificate")),
        acceptance_gate=("every executed M, gamma, T, A, a, b, c, d finite and "
                         "1 - a + b, 1 + a + b, 1 - b > 0 for the rounded "
                         "coefficients each compiled forward pass executed: "
                         "at every update (executed dtype) and at every "
                         "checkpoint (exact rationals); isolated filter "
                         "coefficients only"),
        recovery_policy=dict(
            decisive=("finite-first logits and W, U carries at TRAJ32 on "
                      "representative episodes of every restored source"),
            TRAJ32=TRAJ32, episodes_per_family=RECOVERY_EPISODES_PER_FAMILY,
            aggregate=("integer count and CE differences recorded against "
                       "the previously declared identity tolerance; "
                       "diagnostic only")),
        carry=dict(filtered_implemented=FL.CARRY_EXECUTED,
                   filtered_theoretical_minimum_if_M_zero=(
                       FL.CARRY_MINIMAL_M_ZERO),
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
        source=dict(hashes_at_restore=None), checkpoint_log=[],
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


def start_point_checks(sources, val_np, status):
    """Before any training. Each check is labelled with its actual scope."""
    starts, bad = {}, []
    for seed, pn in sources.items():
        tss_p, gen_p = start_tree(TSS, pn), start_tree(GEN, pn)
        same = all(onp.array_equal(onp.asarray(tss_p[k]),
                                   onp.asarray(gen_p[k])) for k in tss_p)
        fails, rows = direct_recovery(pn, val_np)
        m_nat = ST.evaluate("momentum_delta", pn, val_np)
        m_pt, pt_sets = evaluate_arm(GEN, native_point_tree(pn), val_np)
        m_start, start_sets = evaluate_arm(TSS, tss_p, val_np)
        start_fail = validate(TSS, tss_p, start_sets)
        pt_fail = validate(GEN, native_point_tree(pn), pt_sets)
        starts[str(seed)] = dict(
            storage_identity_of_the_two_processing_start_trees=bool(same),
            direct_native_point_recovery_failures=fails,
            direct_native_point_recovery=rows,
            aggregate_native_point_differences=count_differences(m_nat,
                                                                 m_pt),
            start_acceptance_failure=start_fail,
            start_executed_filter=[FL.filter_report(ex) for ex in start_sets],
            native_point_acceptance_failure=pt_fail)
        if not same or fails or start_fail or pt_fail \
                or starts[str(seed)]["aggregate_native_point_differences"][
                    "malformed"]:
            bad.append(seed)
    status["start_points"] = dict(
        rows=starts,
        scope=dict(
            storage_identity=("the two processing arms' start trees are "
                              "bitwise identical stored trees under the same "
                              "law; this does NOT by itself test literal TSS"),
            direct_native_point_recovery=(
                "DECISIVE: the processing law at (0, h, 0) against the "
                "native rule, logits and W, U carries, finite first, TRAJ32, "
                "on representative episodes of this source"),
            aggregate_native_point_differences=(
                "recorded diagnostic: integer counts and CE against the "
                "previously declared identity tolerance; malformed records "
                "fail, magnitudes never decide"),
            acceptance=("executed-coefficient acceptance of the literal-TSS "
                        "start and of the native point"),
            independent_literal_tss=(
                "the independent literal-TSS reference comparison runs in "
                "the focused checks (tests/test_prospective_tss_containment"
                ".py and the float32 probe), not here")),
        tolerances=dict(TRAJ32=TRAJ32,
                        identity_ce_rel=ST.IDENTITY_CE_REL))
    return bad


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

    bad = start_point_checks(sources, val_np, status)
    save()
    if bad:
        status["failed"] = f"start-point checks failed for seeds {bad}"
        print(f"[!] {status['failed']}")
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
        pn = sources[seed]
        e = status["source"]["entries"][seed]
        return start_tree(arm, pn), f"{e['file']} sha256={e['sha256']}"

    dev_rows = []
    for arm, _, _, _ in TRAINED_ARMS:
        for tag, lr in LRS:
            p, label = tree_of(arm, SOURCE_DEV)
            r = run_one(arm, tag, lr, SOURCE_DEV, p, val_np, UPDATES, out,
                        deadline, args.reserve_s, status, "dev", label, save)
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
    r = run_one(ANCHOR, "-", 0.0, SOURCE_DEV, p, val_np, 0, out, deadline,
                args.reserve_s, status, "dev", label, save)
    if r is None:
        return 3, "INCOMPLETE"
    rec, _ = r
    dev_rows.append(rec)
    status["development"] = dev_rows
    if rec["invalid"]:
        status["failed"] = f"dev anchor: {rec['invalid']}"
        return 4, "FAILED"

    sel, plan = select(dev_rows, status)
    if sel is None:
        status["failed"] = ("selection could not be formed: a missing, "
                            "non-finite or unaccepted development checkpoint")
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
                        out, deadline, args.reserve_s, status, "final", label,
                        save)
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
        m, sets = evaluate_arm(rec["rule"], finals[(rec["rule"],
                                                    rec["seed"])], held_np)
        rec["heldout"] = m
        if not ST.metrics_finite(m) or (
                LAW_OF[rec["rule"]] == FL.FILTERED and (
                    not m["processing_state"]["finite"]
                    or any(FL.filter_failure(FL.filter_report(ex))
                           for ex in sets))):
            status["failed"] = (f"non-finite or unaccepted held-out "
                                f"evaluation {rec['rule']}/{rec['seed']}")
            return 4, "FAILED"
    status["heldout_evaluation_complete"] = True
    status["screen"] = screen(final_rows, frozen_sel)
    status["deployment_outcome"] = deployment_outcome(final_rows, frozen_plan)
    if status["selection"]["selected"] != frozen_sel:
        status["failed"] = "development selection changed after evaluation"
        return 4, "FAILED"
    rows = dev_rows + final_rows
    status["work_completed"] = dict(
        development_runs=len([r for r in dev_rows
                              if r["regime"] != "frozen"]),
        named_family_final_endpoints=len([r for r in final_rows
                                          if r["regime"] != "frozen"]),
        zero_update_named_family_endpoints=len(
            [r for r in rows if r.get("endpoint_kind")
             == "zero_update_named_family_endpoint"]),
        frozen_source_evaluations=len([r for r in rows
                                       if r["regime"] == "frozen"]),
        checkpoints_validated_and_persisted=len(status["checkpoint_log"]),
        total_updates=sum(r["updates"] for r in rows))
    for c in status["screen"]["comparisons"]:
        print(f"[screen] {c['name']:<38} kind={c['kind']:<13} "
              f"promising={c['promising_matched_retention']} "
              f"no_measured_decrease={c['no_measured_decrease']} mean "
              f"{c['mean_primary_difference']}")
    status["complete"] = True
    return 0, "PASS"


if __name__ == "__main__":
    sys.exit(main())
