"""Retention-aware continuation: can generalized processing achieve more
revision than literal TSS when training explicitly discourages damage to
untouched associations and recall?

Frozen protocol: docs/PROSPECTIVE_RETENTION_AWARE_PROTOCOL.md. NOT AUTHORIZED
TO RUN until static review clears it. A new study; every completed result is
unchanged.

Reused unchanged from `temporal_response` and `tss_containment`: the Momentum
backbone and processing equations, full BPTT, coefficient domain and repair,
executed-coefficient gate, the temporal task and its metrics, checkpoint
acceptance and persistence, source verification and reporting.

The ONE change is the training objective, for every trained family:

    L = L_existing
        + lambda * max(0, CE_untouched - CE_untouched_reference)
        + lambda * max(0, CE_recall    - CE_recall_reference)

with lambda in {0, 1}. CE_untouched and CE_recall use the evaluation's own
category weighting (`temporal_task.aggregate`): CE_untouched is the mean of
the revision family's equal-cell untouched-probe CE and its late-untouched
CE; CE_recall is the recall family's equal-cell macro CE. The reference terms
come from a FIXED, read-only native Momentum endpoint of the completed
temporal-response run, matched to the source seed and verified from saved
artifacts, evaluated on the SAME training episodes with no gradient. Task
metadata forms the loss only and never enters the model. The penalties are
training proxies; accuracy preservation is still measured, not assumed.
"""

import argparse
import copy
import json
import os
import signal
import sys
import time
from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.nested_memory.task import FAMILIES, QUERY        # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import filtered as FL       # noqa: E402
from experiments.prospective_momentum import replication as RP    # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import study as ST          # noqa: E402
from experiments.prospective_momentum import temporal_response as TR  # noqa
from experiments.prospective_momentum import temporal_task as TT  # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

#: the previously selected learning rate, for every family and slot
LR = 0.01
#: two LOSS slots replace the two learning-rate slots (slot id, lambda)
LAMBDAS = (("lambda0", 0.0), ("lambda1", 1.0))
LAMBDA_OF = dict(LAMBDAS)
UPDATES = TR.UPDATES
VAL_AT = TR.VAL_AT
BATCH_PER_FAMILY = TR.BATCH_PER_FAMILY
VAL_PER_FAMILY = TR.VAL_PER_FAMILY
HELDOUT_PER_FAMILY = TR.HELDOUT_PER_FAMILY
SOURCE_RUN = TR.SOURCE_RUN
SOURCE_DEV, SOURCE_FINAL = TR.SOURCE_DEV, TR.SOURCE_FINAL
TRAJ32 = TR.TRAJ32
STREAM = dict(continuation_train=620_000_000, dev_validation=650_000_000,
              eval_validation=651_000_000, heldout=660_000_000)

#: the completed temporal-response run whose native endpoints are the
#: references, and the DECLARED endpoint files (from its frozen selection and
#: final plan: native_full, slot B = lr 0.01, update 200). The mapping is
#: re-derived from that run's saved status and must match exactly.
REFERENCE_RUN = ("/Users/durso/s5-runs/prospective-temporal-response/"
                 "20260917-163003")
REFERENCE_CONFIG, REFERENCE_LR, REFERENCE_UPDATE = "B", 0.01, 200
REFERENCE_FILES = {
    500: "dev_native_full_B_seed500_u200.msgpack",
    501: "final_native_full_B_seed501_u200.msgpack",
    502: "final_native_full_B_seed502_u200.msgpack",
    503: "final_native_full_B_seed503_u200.msgpack"}
REFERENCE_METRICS = {"primary": "primary",
                     "retention": "retention_revision_untouched",
                     "recall": "recall_overall",
                     "immediate_revision": "immediate_revision",
                     "later": "later"}

TSS, GEN, NATIVE, OPERATOR, ANCHOR = TR.TSS, TR.GEN, TR.NATIVE, TR.OPERATOR, \
    TR.ANCHOR
EXTENSION_ARMS = (TSS, GEN, OPERATOR)
TRAINED_ARMS = TR.TRAINED_ARMS
LAW_OF, FROZEN_LEAVES = TR.LAW_OF, TR.FROZEN_LEAVES
#: metadata used ONLY by the loss; never passed to the model
META_FIELDS = ("kind", "delay", "condition", "family")
assert not set(META_FIELDS) & set(TT.MODEL_INPUTS)
ROLES = ("reference", "constrained", "unconstrained", "lambda0")


def role_id(arm, role):
    return f"{arm}@{role}"


for _arm, _, _, _ in TR.ARMS:
    for _role in ROLES:
        PD.DISPLAY.setdefault(role_id(_arm, _role),
                              f"{TR.ARM_DISPLAY[_arm]} [{_role}]")

PLANNED = dict(
    development_trajectories=len(TRAINED_ARMS) * len(LAMBDAS),
    development_checkpoints_per_family=len(VAL_AT) * len(LAMBDAS) - 1,
    final_trajectories_max=len(TRAINED_ARMS) * len(LAMBDAS)
    * len(SOURCE_FINAL),
    frozen_source_evaluations=1 + len(SOURCE_FINAL))
PLANNED["max_total_updates"] = UPDATES * (
    PLANNED["development_trajectories"] + PLANNED["final_trajectories_max"])


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
    prev = list(TR.previous_ranges())
    prev += list(TR.new_ranges().values())
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


# ------------------------------------------------------------------ the loss --
def _cell_mean(ce, mask):
    n = jnp.sum(mask)
    return jnp.sum(ce * mask) / jnp.maximum(n, 1.0), n


def category_ce(ce, q, meta):
    """(CE_untouched, CE_recall, smallest contributing cell count) with the
    evaluation's category weighting. `ce` per token (episodes x tokens)."""
    dt = ce.dtype
    fam = meta["family"][:, None]
    cond = meta["condition"][:, None]
    kind, delay = meta["kind"], meta["delay"]
    qf = q.astype(dt)
    counts = []

    def block_part(f, k):
        vals = []
        for d in TT.DELAYS:
            for c in range(len(TT.CONDITIONS)):
                m = qf * ((fam == f) & (cond == c) & (kind == k)
                          & (delay == d)).astype(dt)
                v, n = _cell_mean(ce, m)
                vals.append(v)
                counts.append(n)
        return jnp.mean(jnp.stack(vals))

    def late_part(f, k):
        vals = []
        for c in range(len(TT.CONDITIONS)):
            m = qf * ((fam == f) & (cond == c) & (kind == k)).astype(dt)
            v, n = _cell_mean(ce, m)
            vals.append(v)
            counts.append(n)
        return jnp.mean(jnp.stack(vals))
    rev, rec = FAMILIES.index("revision"), FAMILIES.index("recall")
    ce_untouched = 0.5 * (block_part(rev, 1) + late_part(rev, 3))
    ce_recall = 0.25 * (block_part(rec, 0) + block_part(rec, 1)
                        + late_part(rec, 2) + late_part(rec, 3))
    return ce_untouched, ce_recall, jnp.min(jnp.stack(counts))


def _hinge(x):
    """max(0, x) with a zero derivative for x <= 0 (no tie-splitting)."""
    return jnp.where(x > 0, x, jnp.zeros_like(x))


def ra_batch_loss(rule, lam, p, ref_p, eps, meta):
    """The existing loss plus the declared hinge penalties. `eps` holds ONLY
    model inputs; `meta` is used here only. The reference rollout is
    stop-gradient on its parameters and outputs."""
    if rule == FL.FILTERED:
        base, aux = TC.batch_loss_f(rule, p, eps)
    else:
        base, aux = ST.batch_loss(rule, p, eps)
    _, ref_aux = ST.batch_loss("momentum_delta",
                               jax.lax.stop_gradient(ref_p), eps)
    ref_ce = jax.lax.stop_gradient(ref_aux["ce"])
    q = eps["event"] == QUERY
    cu, cr, n_min = category_ce(aux["ce"], q, meta)
    ru, rr, _ = category_ce(ref_ce, q, meta)
    pu, pr = _hinge(cu - ru), _hinge(cr - rr)
    total = base + lam * pu + lam * pr
    terms = dict(base=base, ce_untouched=cu, ce_recall=cr,
                 reference_untouched=ru, reference_recall=rr,
                 penalty_untouched=pu, penalty_recall=pr,
                 min_cell_count=n_min)
    return total, dict(aux, terms=terms)


@partial(jax.jit, static_argnums=(0, 1, 2))
def train_step_ra(rule, frozen, lam, p, opt, ref_p, eps, meta, lr):
    """The existing training steps with the retention-aware loss. Only `p`
    is differentiated; the processing arms keep the masked frozen leaves,
    repair and forward-pass gate, the others the existing projection."""
    (loss, aux), g = jax.value_and_grad(ra_batch_loss, argnums=2,
                                        has_aux=True)(rule, lam, p, ref_p,
                                                      eps, meta)
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    if rule == FL.FILTERED:
        guard = FL.in_loop_guard(aux["coeff"])
        mask = TC.leaf_mask(p, frozen)
        g_masked = jax.tree_util.tree_map(lambda x, m: x * m, g, mask)
        raw, opt = ST.TX.update(g_masked, opt, p)
        upd = jax.tree_util.tree_map(lambda u, m: -lr * u * m, raw, mask)
        p = optax.apply_updates(p, upd)
        p, tel = FL.repair(p, frozen=frozen)
        gate = dict(ok=jnp.all(guard["ok"]),
                    jury_min=jnp.min(guard["jury_min"]),
                    **{k: guard[k][0] for k in FL.COEFF_NAMES},
                    proc_max_abs=jnp.max(aux["proc_max_abs"]))
        tel = dict(tel, **{"gate_" + k: v for k, v in gate.items()})
        extra = {k: g[k][0] for k in FL.LEAVES}
    else:
        raw, opt = ST.TX.update(g, opt, p)
        upd = jax.tree_util.tree_map(lambda u: -lr * u, raw)
        p = optax.apply_updates(p, upd)
        p, tel = ST._project(rule, p)
        extra = (g[ST.EXTRA_GRAD[rule]][0] if rule in ST.EXTRA_GRAD
                 else jnp.zeros((), dtype=loss.dtype))
    return (p, opt, loss, acc, optax.global_norm(g), optax.global_norm(upd),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]), tel, extra,
            aux["terms"])


def make_host_step(lam, refs):
    """A host step with the `temporal_response.run_one` hook signature."""
    def host_step(arm, p, opt, seed, u, lr, hist):
        rule = LAW_OF[arm]
        batch = TT.generate_batch(continuation_stream(seed, u),
                                  BATCH_PER_FAMILY)
        eps = ST.to_jax(batch)
        meta = {k: jnp.asarray(batch[k]) for k in META_FIELDS}
        o = train_step_ra(rule, FROZEN_LEAVES[arm], lam, p, opt, refs[seed],
                          eps, meta, lr)
        p, opt, tel, extra, terms = o[0], o[1], o[8], o[9], o[10]
        if rule == FL.FILTERED:
            rec = dict(update=u, n_repaired=int(tel["n_repaired"]),
                       jury_min=float(tel["gate_jury_min"]),
                       proc_max_abs=float(tel["gate_proc_max_abs"]),
                       executed_filter_ok=bool(tel["gate_ok"]),
                       executed={k: float(tel["gate_" + k])
                                 for k in FL.COEFF_NAMES})
            for k in FL.LEAVES:
                rec[k] = float(tel[k + "_post"])
                rec["grad_" + k] = float(extra[k])
        else:
            rec = dict(update=u, n_projected=int(tel["n_projected"]),
                       grad=float(extra), executed_filter_ok=True)
        rec["lambda"] = lam
        rec["terms"] = {k: float(v) for k, v in terms.items()}
        scalars = dict(zip(ST.SCALAR_NAMES, (float(x) for x in o[2:8])))
        hist.append(rec)
        return p, opt, scalars, rec
    return host_step


def step_failure_ra(arm, scalars, rec):
    """The existing per-step acceptance plus finite loss terms and non-empty
    loss cells, checked at EVERY step."""
    bad = TC.step_failure(arm, scalars, rec)
    if bad:
        return bad
    terms = rec.get("terms") or {}
    missing = [k for k in ("base", "ce_untouched", "ce_recall",
                           "reference_untouched", "reference_recall",
                           "penalty_untouched", "penalty_recall",
                           "min_cell_count") if k not in terms]
    if missing:
        return f"missing loss terms {missing} at update {rec['update']}"
    nonfinite = [k for k, v in terms.items() if not onp.isfinite(v)]
    if nonfinite:
        return f"non-finite loss terms {nonfinite} at update {rec['update']}"
    if terms["min_cell_count"] <= 0:
        return f"empty loss cell at update {rec['update']}"
    return None


# -------------------------------------------------------------- references --
class ReferenceRefusal(RuntimeError):
    pass


def reference_mapping(ref_run, doc):
    """Derive seed -> reference endpoint from the completed run's SAVED status
    only, and require it to equal the declared files. Scores never choose a
    reference; any mismatch refuses."""
    if not (doc.get("complete") is True and not doc.get("failed")
            and doc.get("study_status") == "PASS"):
        raise ReferenceRefusal("reference run is not a complete PASS")
    try:
        sel = doc["selection"]["selected"][NATIVE]
        frz = doc["frozen_before_finals"]["selection"][NATIVE]
    except (KeyError, TypeError) as e:
        raise ReferenceRefusal(f"reference selection record missing: {e!r}")
    for k in ("config", "lr", "update", "params_file"):
        if sel.get(k) != frz.get(k):
            raise ReferenceRefusal(f"native selection changed after freezing "
                                   f"({k}: {sel.get(k)} vs {frz.get(k)})")
    if (sel["config"], sel["lr"], sel["update"]) != (
            REFERENCE_CONFIG, REFERENCE_LR, REFERENCE_UPDATE):
        raise ReferenceRefusal(f"native selection {sel['config']}/"
                               f"{sel['lr']}/{sel['update']} is not the "
                               "declared reference recipe")

    def expected(seed):
        return os.path.realpath(os.path.join(ref_run, "params",
                                             REFERENCE_FILES[seed]))
    out = {}
    if os.path.realpath(sel["params_file"]) != expected(SOURCE_DEV):
        raise ReferenceRefusal(f"development reference {sel['params_file']} "
                               f"is not {expected(SOURCE_DEV)}")
    out[SOURCE_DEV] = dict(path=expected(SOURCE_DEV), stage="development",
                           stream="dev_validation",
                           recorded={k: sel[k] for k in REFERENCE_METRICS})
    rows = [r for r in doc.get("final", []) if r.get("rule") == NATIVE]
    for seed in SOURCE_FINAL:
        rs = [r for r in rows if r.get("seed") == seed]
        if len(rs) != 1:
            raise ReferenceRefusal(f"{len(rs)} native final rows for seed "
                                   f"{seed}")
        r = rs[0]
        if (r.get("config"), r.get("lr"), r.get("update")) != (
                sel["config"], sel["lr"], sel["update"]) or r.get(
                "endpoint_kind") != "trained_named_family_endpoint":
            raise ReferenceRefusal(f"native final row for seed {seed} does "
                                   "not follow the frozen selection")
        if os.path.realpath(r["endpoint_params_file"]) != expected(seed):
            raise ReferenceRefusal(f"final reference for seed {seed} is "
                                   f"{r['endpoint_params_file']}, not "
                                   f"{expected(seed)}")
        out[seed] = dict(path=expected(seed), stage="final",
                         stream="eval_validation",
                         recorded={k: r["final_validation_summary"][k]
                                   for k in REFERENCE_METRICS})
    return out


def load_references(ref_run, sources, status):
    """Restore, hash and verify the four references. Their recorded
    validation metrics must be reproduced on the completed study's own
    validation streams (used for verification only). Returns
    ({seed: tree}, {hash key: sha256})."""
    spath = os.path.join(ref_run, "status.json")
    if not os.path.isfile(spath):
        raise ReferenceRefusal(f"missing {spath}")
    hashes = {f"reference:{os.path.realpath(spath)}": RS.sha256(spath)}
    with open(spath) as fh:
        doc = json.load(fh)
    mapping = reference_mapping(ref_run, doc)
    refs = {}
    for seed, m in mapping.items():
        if not os.path.isfile(m["path"]):
            raise ReferenceRefusal(f"missing reference {m['path']}")
        hashes[f"reference:{m['path']}"] = RS.sha256(m["path"])
        try:
            p = TC.load_params(m["path"], sources[seed])
        except ValueError as e:
            raise ReferenceRefusal(str(e))
        if not ST.all_finite(p):
            raise ReferenceRefusal(f"non-finite reference {m['path']}")
        refs[seed] = p
    batches = {k: TT.generate_batch(TR.STREAM[k], VAL_PER_FAMILY)
               for k in ("dev_validation", "eval_validation")}
    repro = {}
    for seed, m in mapping.items():
        got, _ = TR.evaluate_arm(NATIVE, refs[seed], batches[m["stream"]])
        repro[str(seed)] = {}
        for k, want in m["recorded"].items():
            g = got[REFERENCE_METRICS[k]]
            ok = bool(g is not None and onp.isfinite(g)
                      and abs(g - want) <= TRAJ32 * max(abs(want), 1.0))
            repro[str(seed)][k] = dict(recorded=want, reproduced=g, ok=ok)
            if not ok:
                raise ReferenceRefusal(f"reference {seed} does not reproduce "
                                       f"its recorded {k}: {g} vs {want}")
    status["reference"] = dict(
        run=ref_run, files={str(s): m["path"] for s, m in mapping.items()},
        stages={str(s): m["stage"] for s, m in mapping.items()},
        hashes=hashes, reproduction=repro,
        note=("fixed, read-only native endpoints of the completed "
              "temporal-response run, matched by source seed from its saved "
              "frozen selection and final rows; never chosen by score"))
    return refs, hashes


# --------------------------------------------------------------- selection --
def checkpoints_ra(dev_rows, arm):
    out = []
    for r in dev_rows:
        if r["rule"] != arm:
            continue
        for v in r["validation"]:
            if v["update"] == 0 and r["config"] != LAMBDAS[0][0]:
                continue
            out.append(dict(arm=arm, config=r["config"],
                            lam=LAMBDA_OF[r["config"]], lr=r["lr"],
                            update=v["update"], primary=v["primary"],
                            revision_ce=v["revision_ce"],
                            retention=v["retention"], recall=v["recall"],
                            immediate_revision=v["immediate_revision"],
                            later=v["later"], accepted=v.get("accepted"),
                            params_file=v.get("params_file")))
    return out


def order_ra(c):
    """Revision, lower revision CE, fewer updates, then lower lambda."""
    return (-c["primary"], c["revision_ce"], c["update"], c["lam"])


def select_ra(dev_rows, status):
    """Continued-native reference first; then, per extension, the constrained
    endpoint (retention AND recall >= the native reference, or None =
    infeasible), plus the unconstrained endpoint and the lambda = 0 control,
    both descriptive."""
    nat = checkpoints_ra(dev_rows, NATIVE)
    n_expected = PLANNED["development_checkpoints_per_family"]
    if len(nat) != n_expected or TR.ineligible(nat):
        return None, None
    ref = sorted(nat, key=order_ra)[0]
    r_nat, c_nat = ref["retention"], ref["recall"]
    endpoints = {NATIVE: dict(
        reference=ref, unconstrained=ref,
        lambda0=sorted([c for c in nat if c["lam"] == 0.0],
                       key=order_ra)[0])}
    deploy = {NATIVE: dict(ref, feasible=True, diagnostic=False)}
    table = [dict(arm=NATIVE, reference=ref, n_candidates=len(nat))]
    for arm in EXTENSION_ARMS:
        cand = checkpoints_ra(dev_rows, arm)
        if len(cand) != n_expected or TR.ineligible(cand):
            return None, None
        feas = [c for c in cand
                if c["retention"] >= r_nat and c["recall"] >= c_nat]
        cons = sorted(feas, key=order_ra)[0] if feas else None
        unc = sorted(cand, key=order_ra)[0]
        l0 = sorted([c for c in cand if c["lam"] == 0.0], key=order_ra)[0]
        endpoints[arm] = dict(constrained=cons, unconstrained=unc,
                              lambda0=l0)
        deploy[arm] = dict(cons if cons else unc, feasible=bool(cons),
                           diagnostic=not cons)
        table.append(dict(arm=arm, n_candidates=len(cand),
                          n_feasible=len(feas), constrained=cons,
                          infeasible=cons is None))
    status["selection"] = dict(
        native_reference=ref, r_native=r_nat, c_native=c_nat,
        endpoints=endpoints, table=table,
        rule=("native: highest development revision, then lower revision "
              "CE, fewer updates, lower lambda. Each extension, constrained: "
              "the same ordering over accepted checkpoints with retention >= "
              "R_native AND recall >= C_native; none = INFEASIBLE, recorded, "
              "no fallback counted as improvement. Unconstrained endpoints "
              "and lambda = 0 controls use the same ordering and are "
              "descriptive."))
    plan = TC.deployment_plan(deploy, status)
    return endpoints, plan


def final_plan_ra(endpoints):
    """One trajectory per (family, lambda slot, final seed), run to the
    largest update any of its endpoints needs."""
    needs, roles = defaultdict(set), defaultdict(set)
    for arm, eps in endpoints.items():
        for role, c in eps.items():
            if c is None:
                continue
            key = (arm, c["config"])
            needs[key].add(c["update"])
            roles[key].add((role_id(arm, role), c["update"]))
    return sorted([dict(arm=a, config=cfg, lam=LAMBDA_OF[cfg], lr=LR,
                        updates=max(ups), keep_at=sorted(ups),
                        endpoints=sorted(roles[(a, cfg)]))
                   for (a, cfg), ups in needs.items()],
                  key=lambda t: (t["arm"], t["config"]))


# ----------------------------------------------------------------- screens --
PRIMARY = (("generalized_versus_literal_tss", GEN, TSS),
           ("generalized_versus_native", GEN, NATIVE),
           ("generalized_versus_learned_operator", GEN, OPERATOR))


def _compare(rows, name, a, b, kind, seeds=SOURCE_FINAL):
    c = ST.compare(rows, a, b, seeds)
    c["safeguard_passed_minus_one_pp"] = c.pop("passed")
    c.update(name=name, candidate=a, kind=kind,
             retention_direction=RP.direction(c["retention_difference"]),
             recall_direction=RP.direction(c["recall_difference"]))
    c["aggregate_screen_passed"] = bool(
        c["complete_paired_seeds"] and c["mean_primary_difference"] >= 0.01
        and c["positive_in_all_seeds"] and c["retention_difference"] >= 0
        and c["recall_difference"] >= 0)
    c["secondary_cells"] = TR.cell_differences(rows, a, b, seeds)
    return c


def screen_ra(rows, endpoints):
    """PRIMARY verdicts on constrained endpoints (native: its reference);
    DESCRIPTIVE comparisons of unconstrained endpoints, lambda = 0 controls,
    each extension against its own lambda = 0 control, and references."""
    def primary_role(arm):
        return "reference" if arm == NATIVE else "constrained"
    out = dict(rule=("mean held-out revision difference >= +1 pp, positive "
                     "paired revision differences in all three seeds, mean "
                     "retention AND recall differences >= 0"),
               primary=[], descriptive=[])
    for name, a, b in PRIMARY:
        missing = [x for x in (a, b)
                   if endpoints[x].get(primary_role(x)) is None]
        if missing:
            out["primary"].append(dict(
                name=name, candidate=role_id(a, primary_role(a)),
                against=role_id(b, primary_role(b)), kind="primary",
                available=False, aggregate_screen_passed=False,
                reason=(f"constrained selection INFEASIBLE for {missing}; "
                        "no verdict, and no fallback counts as an "
                        "improvement")))
            continue
        c = _compare(rows, name, role_id(a, primary_role(a)),
                     role_id(b, primary_role(b)), "primary")
        c["available"] = True
        out["primary"].append(c)
    desc = []
    for name, a, b in PRIMARY:
        desc.append((f"{name}@unconstrained", role_id(a, "unconstrained"),
                     role_id(b, "unconstrained")))
        desc.append((f"{name}@lambda0", role_id(a, "lambda0"),
                     role_id(b, "lambda0")))
    for arm in EXTENSION_ARMS:
        if endpoints[arm]["constrained"] is not None:
            desc.append((f"{arm}@constrained_versus_own_lambda0",
                         role_id(arm, "constrained"),
                         role_id(arm, "lambda0")))
    for arm in (TSS, OPERATOR):
        if endpoints[arm]["constrained"] is not None:
            desc.append((f"{arm}@constrained_versus_native_reference",
                         role_id(arm, "constrained"),
                         role_id(NATIVE, "reference")))
    desc.append(("generalized_unconstrained_versus_frozen_source",
                 role_id(GEN, "unconstrained"), ANCHOR))
    for name, a, b in desc:
        out["descriptive"].append(_compare(rows, name, a, b, "descriptive"))
    out["note"] = ("A generalized improvement over its own lambda = 0 control "
                   "is insufficient: the central verdict is generalized "
                   "against equally trained literal TSS. The CE penalties "
                   "are training proxies; accuracy preservation is measured. "
                   "Descriptive comparisons cannot rescue a primary verdict.")
    return out


def deployment_outcome_ra(rows, plan):
    out = {}
    for arm, pl in plan.items():
        src = (role_id(NATIVE, "reference") if pl["evaluate_arm"] == NATIVE
               else role_id(pl["evaluate_arm"], "constrained"))
        rs = [r for r in rows if r["rule"] == src and "heldout" in r]
        out[arm] = dict(choice=pl["choice"], evaluated_model=src,
                        heldout_primary_mean=(float(onp.mean(
                            [r["heldout"]["primary"] for r in rs]))
                            if rs else None),
                        counted_as_improvement=False)
    return out


# --------------------------------------------------------------- preflight --
def _cache_size():
    return train_step_ra._cache_size()


def preflight_ra(sources, refs, val_np, out, status, step_factory=None,
                 evaluate=None, checkpoint_failure=None, save_tree=None,
                 cache_size=None, start_tree=None):
    """Measures and validates EVERY (family, lambda slot) executable on
    disposable production state before any training (review R1): the actual
    host step (reference forward, synchronization and per-step acceptance),
    its initial call, its steady step time and retrace behaviour, plus a full
    checkpoint (evaluation, acceptance, persistence). A failure or retrace in
    either slot is recorded under that slot and cannot be hidden by the other.
    The projection uses the slot-specific measurements; both compilations are
    already incurred here and are not charged again. The keyword arguments
    exist only so the orchestration can be checked with stubs; production
    uses the defaults."""
    step_factory = step_factory or make_host_step
    evaluate = evaluate or TR.evaluate_arm
    checkpoint_failure = checkpoint_failure or TR.checkpoint_failure
    save_tree = save_tree or ST.save_tree
    cache_size = cache_size or _cache_size
    start_tree = start_tree or TC.start_tree
    rows, failures, retraced_any = [], [], False
    p0 = sources[SOURCE_DEV]
    lr = jnp.asarray(LR, dtype=jnp.float32)
    scratch = os.path.join(out, "preflight_disposable")
    cost = {}
    for arm, rule, frozen, _ in TRAINED_ARMS:
        for tag, lam in LAMBDAS:
            step = step_factory(lam, refs)
            p = start_tree(arm, p0)
            opt = ST.TX.init(p)
            hist, acc_bad = [], None
            t0 = time.time()
            p2, opt2, sc, rec = step(arm, p, opt, SOURCE_DEV, 0, lr, hist)
            warm_s = time.time() - t0
            acc_bad = acc_bad or step_failure_ra(arm, sc, rec)
            p2, opt2, sc, rec = step(arm, p2, opt2, SOURCE_DEV, 1, lr, hist)
            acc_bad = acc_bad or step_failure_ra(arm, sc, rec)
            n0 = cache_size()
            t1 = time.time()
            for u in range(2, 7):
                p2, opt2, sc, rec = step(arm, p2, opt2, SOURCE_DEV, u, lr,
                                         hist)
                acc_bad = acc_bad or step_failure_ra(arm, sc, rec)
            step_s = (time.time() - t1) / 5.0
            retraced = bool(cache_size() != n0)
            evaluate(arm, p2, val_np)                      # compile, once
            t3 = time.time()
            m, sets = evaluate(arm, p2, val_np)
            eval_s = time.time() - t3
            fail = checkpoint_failure(arm, p2, opt2, p, m, sets)
            save_tree(os.path.join(scratch, f"{arm}_{tag}.msgpack"), p2)
            save_tree(os.path.join(scratch, f"{arm}_{tag}_opt.msgpack"), opt2)
            ckpt_s = time.time() - t3
            acc_bad = acc_bad or fail
            timings = dict(warmup_s=warm_s, step_s=step_s,
                           evaluation_s=eval_s, checkpoint_s=ckpt_s)
            bad_t = {k: v for k, v in timings.items()
                     if not (onp.isfinite(v) and v >= 0)}
            if bad_t:
                acc_bad = acc_bad or f"non-finite or negative timing {bad_t}"
            if acc_bad:
                failures.append(f"{arm}/{tag}: {acc_bad}")
            retraced_any |= retraced
            cost[(arm, tag)] = timings
            rows.append(dict(arm=arm, config=tag, lam=lam, **timings,
                             retraced=retraced,
                             measured_steps_checked=len(hist),
                             last_terms=rec.get("terms"),
                             acceptance_failure=acc_bad))
            print(f"[preflight] {arm:<24} {tag:<7} warm {warm_s:5.1f}s "
                  f"step {step_s * 1e3:6.2f}ms checkpoint {ckpt_s:5.2f}s "
                  f"retraced {retraced} failure {acc_bad}")
    factor = HELDOUT_PER_FAMILY / VAL_PER_FAMILY
    total = 0.0
    for arm, _, _, _ in TRAINED_ARMS:
        for tag, _ in LAMBDAS:
            c = cost[(arm, tag)]
            traj = UPDATES * c["step_s"] + len(VAL_AT) * c["checkpoint_s"]
            total += traj                                        # development
            total += len(SOURCE_FINAL) * traj                    # finals
        worst_eval = max(cost[(arm, t)]["evaluation_s"] for t, _ in LAMBDAS)
        total += len(ROLES) * len(SOURCE_FINAL) * factor * worst_eval
    worst_ckpt = max(c["checkpoint_s"] for c in cost.values())
    worst_eval = max(c["evaluation_s"] for c in cost.values())
    total += (1 + len(SOURCE_FINAL)) * worst_ckpt + len(SOURCE_FINAL) * \
        factor * worst_eval                                      # anchor
    host_s = 40.0
    total += host_s
    if not onp.isfinite(total) or total < 0:
        failures.append(f"non-finite or negative projection {total}")
    status["preflight"] = dict(
        rows=rows, retraced_any=bool(retraced_any), failures=failures,
        projected_remaining_s=total, host_allowance_s=host_s,
        slots_measured=[[r["arm"], r["config"]] for r in rows],
        coverage=("both lambda slots of all four families measured and "
                  "validated on disposable production state (actual host "
                  "step with the reference forward, initial call, steady "
                  "step, retrace, full checkpoint). Projection from those "
                  "slot-specific costs at the worst case: both slots in "
                  "development and for every final seed, all five "
                  "checkpoints each, every endpoint role's held-out "
                  "evaluation (worst slot), the anchor and a 40 s host "
                  "allowance. Both compilations were incurred here and are "
                  "not charged again."))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, bool(retraced_any), failures


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/prospective-retention-aware")
    ap.add_argument("--source_run", default=SOURCE_RUN)
    ap.add_argument("--reference_run", default=REFERENCE_RUN)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    args = ap.parse_args()
    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    if jax.default_backend() != "gpu":
        raise SystemExit("REFUSING: backend is not 'gpu'.")
    if jax.config.read("jax_enable_x64") or jnp.zeros(1).dtype != jnp.float32:
        raise SystemExit("REFUSING: production is float32 with x64 off.")
    if stream_overlaps():
        raise SystemExit(f"REFUSING: stream overlaps {stream_overlaps()}")
    if os.path.realpath(args.reference_run) != os.path.realpath(
            REFERENCE_RUN):
        raise SystemExit("REFUSING: the reference run is declared, not "
                         "substitutable")
    for root in (args.source_run, args.reference_run):
        if os.path.realpath(args.out_root).startswith(os.path.realpath(root)):
            raise SystemExit("REFUSING: output inside a read-only run")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    status = dict(
        run_id=run_id, out=out, backend=jax.default_backend(),
        study=("retention-aware continuation: generalized versus literal TSS "
               "under an explicit preservation penalty"),
        protocol="docs/PROSPECTIVE_RETENTION_AWARE_PROTOCOL.md",
        source_run=args.source_run, reference_run=args.reference_run,
        source_seeds=dict(development=SOURCE_DEV, final=list(SOURCE_FINAL)),
        final_seeds=list(SOURCE_FINAL), updates=UPDATES,
        checkpoints_at=list(VAL_AT), learning_rate=LR,
        lambda_slots=dict(LAMBDAS),
        loss=("L_existing + lambda max(0, CE_untouched - CE_untouched_ref) + "
              "lambda max(0, CE_recall - CE_recall_ref), evaluation category "
              "weighting, reference without gradient on the same episodes"),
        planned_work=dict(PLANNED), streams=STREAM,
        stream_ranges=new_ranges(), previous_stream_ranges=previous_ranges(),
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
        got = {}
        for k in keys:
            path = (k[len("reference:"):] if k.startswith("reference:")
                    else os.path.join(d, k))
            got[k] = RS.sha256(path) if os.path.isfile(path) else None
        return got

    signal.signal(signal.SIGTERM, ST._on_sigterm)
    code, _ = ST.guarded(lambda: run_study(args, status, out, deadline,
                                           lambda: persist(status)),
                         status, rehash, persist)
    return code


def run_study(args, status, out, deadline, save):
    val_np = TT.generate_batch(STREAM["dev_validation"], VAL_PER_FAMILY)
    eval_val_np = TT.generate_batch(STREAM["eval_validation"], VAL_PER_FAMILY)
    train0 = TT.generate_batch(continuation_stream(SOURCE_DEV, 0),
                               BATCH_PER_FAMILY)
    checks = dict(dev_validation=TT.structure_check(val_np),
                  eval_validation=TT.structure_check(eval_val_np),
                  first_training_batch=TT.structure_check(train0))
    status["task_checks"] = checks
    bad = [n for n, c in checks.items()
           if not (c["oracle_exact"] and c["block_cells_balanced"]
                   and c["queries_per_sequence"] == [TT.N_QUERIES])]
    save()
    if bad:
        status["failed"] = f"task checks failed: {bad}"
        return 4, "FAILED"

    try:
        sources = TC.load_sources(args.source_run, status)
    except RS.SourceRefusal as e:
        status["failed"] = f"source refusal: {e}"
        return 4, "FAILED"
    try:
        refs, ref_hashes = load_references(args.reference_run, sources,
                                           status)
    except ReferenceRefusal as e:
        status["failed"] = f"reference refusal: {e}"
        print(f"[!] {status['failed']}")
        return 4, "FAILED"
    status["source"]["hashes_at_restore"].update(ref_hashes)
    print(f"[reference] {len(refs)} native references verified and "
          "reproduced")
    save()

    starts, bad = {}, []
    for seed, pn in sources.items():
        tss_p, gen_p = TC.start_tree(TSS, pn), TC.start_tree(GEN, pn)
        same = all(onp.array_equal(onp.asarray(tss_p[k]),
                                   onp.asarray(gen_p[k])) for k in tss_p)
        fails, _ = TC.direct_recovery(pn, val_np)
        _, sets = TR.evaluate_arm(TSS, tss_p, val_np)
        start_fail = TC.validate(TSS, tss_p, sets)
        starts[str(seed)] = dict(storage_identity=bool(same),
                                 recovery_failures=fails,
                                 start_acceptance_failure=start_fail)
        if not same or fails or start_fail:
            bad.append(seed)
    status["start_points"] = starts
    save()
    if bad:
        status["failed"] = f"start-point checks failed for seeds {bad}"
        return 4, "FAILED"

    proj, retraced, failures = preflight_ra(sources, refs, val_np, out,
                                            status)
    save()
    d = ST.decide_after_preflight(proj, retraced, failures,
                                  deadline - time.time() - args.reserve_s)
    if d is not None:
        code, label, why = d
        (status.__setitem__("failed", why) if code == 4
         else status["incomplete"].append(why))
        print(f"[!] {why}")
        return code, label

    steps = {tag: make_host_step(lam, refs) for tag, lam in LAMBDAS}

    def tree_of(arm, seed):
        e = status["source"]["entries"][seed]
        return (TC.start_tree(arm, sources[seed]),
                f"{e['file']} sha256={e['sha256']}")

    dev_rows = []
    for arm, _, _, _ in TRAINED_ARMS:
        for tag, lam in LAMBDAS:
            p, label = tree_of(arm, SOURCE_DEV)
            r = TR.run_one(arm, tag, LR, SOURCE_DEV, p, val_np, UPDATES, out,
                           deadline, args.reserve_s, status, "dev", label,
                           save, None, (), steps[tag], step_failure_ra)
            if r is None:
                return 3, "INCOMPLETE"
            dev_rows.append(r[0])
            status["development"] = dev_rows
            save()
            if r[0]["invalid"]:
                status["failed"] = f"dev {arm}/{tag}: {r[0]['invalid']}"
                return 4, "FAILED"
    p, label = tree_of(ANCHOR, SOURCE_DEV)
    r = TR.run_one(ANCHOR, "-", 0.0, SOURCE_DEV, p, val_np, 0, out, deadline,
                   args.reserve_s, status, "dev", label, save)
    if r is None or r[0]["invalid"]:
        status["failed"] = "dev anchor evaluation failed"
        return 4, "FAILED"
    dev_rows.append(r[0])

    endpoints, plan = select_ra(dev_rows, status)
    if endpoints is None:
        status["failed"] = ("selection could not be formed: a missing, "
                            "non-finite or unaccepted development checkpoint")
        return 4, "FAILED"
    fplan = final_plan_ra(endpoints)
    frozen = copy.deepcopy(dict(endpoints=endpoints, deployment=plan,
                                final_plan=fplan))
    status["frozen_before_finals"] = copy.deepcopy(frozen)
    status["final_plan"] = fplan
    ST.write(os.path.join(out, "selection.json"), frozen)
    for arm, eps in endpoints.items():
        print(f"[selection] {arm:<24} " + " ".join(
            f"{role}={'INFEASIBLE' if c is None else (c['config'], c['update'])}"
            for role, c in eps.items()))
    save()

    final_rows, traj_rows, kept_trees = [], [], {}
    for tr in fplan:
        for seed in SOURCE_FINAL:
            p, label = tree_of(tr["arm"], seed)
            r = TR.run_one(tr["arm"], tr["config"], LR, seed, p, eval_val_np,
                           tr["updates"], out, deadline, args.reserve_s,
                           status, "final", label, save, None,
                           tuple(tr["keep_at"]), steps[tr["config"]],
                           step_failure_ra)
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
                return 4, "FAILED"
            by_u = {v["update"]: v for v in rec["validation"]}
            for rid, u in tr["endpoints"]:
                final_rows.append(dict(
                    rule=rid, trained_family=tr["arm"], seed=seed,
                    config=tr["config"], lam=tr["lam"], update=u,
                    endpoint_kind=("zero_update_named_family_endpoint"
                                   if u == 0 else
                                   "trained_named_family_endpoint"),
                    endpoint_params_file=by_u[u]["params_file"],
                    source=label))
                kept_trees[(rid, seed)] = kept[u]
    for seed in SOURCE_FINAL:
        p, label = tree_of(ANCHOR, seed)
        r = TR.run_one(ANCHOR, "-", 0.0, seed, p, eval_val_np, 0, out,
                       deadline, args.reserve_s, status, "final", label, save)
        if r is None or r[0]["invalid"]:
            status["failed"] = f"final anchor {seed} failed"
            return 4, "FAILED"
        final_rows.append(dict(rule=ANCHOR, trained_family=ANCHOR, seed=seed,
                               config="-", lam=None, update=0,
                               endpoint_kind="frozen_source_anchor",
                               source=label))
        kept_trees[(ANCHOR, seed)] = r[1]
    status["final"] = final_rows
    if copy.deepcopy(dict(endpoints=status["selection"]["endpoints"],
                          deployment=status["deployment_plan"],
                          final_plan=fplan)) != frozen:
        status["failed"] = "selection or plan changed during finals"
        return 4, "FAILED"

    status["heldout_opened"] = True
    status["heldout_evaluation_complete"] = False
    save()
    held_np = TT.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    status["task_checks"]["heldout"] = TT.structure_check(held_np)
    for row in final_rows:
        arm = row["trained_family"]
        m, sets = TR.evaluate_arm(arm, kept_trees[(row["rule"], row["seed"])],
                                  held_np)
        row["heldout"] = m
        if not TR.metrics_finite(m) or (
                LAW_OF[arm] == FL.FILTERED and (
                    not m["processing_state"]["finite"]
                    or any(FL.filter_failure(FL.filter_report(ex))
                           for ex in sets))):
            status["failed"] = (f"non-finite or unaccepted held-out "
                                f"evaluation {row['rule']}/{row['seed']}")
            return 4, "FAILED"
    status["heldout_evaluation_complete"] = True
    status["screen"] = screen_ra(final_rows, endpoints)
    status["deployment_outcome"] = deployment_outcome_ra(final_rows, plan)
    rows = dev_rows + traj_rows
    status["work_completed"] = dict(
        development_trajectories=len([r for r in dev_rows
                                      if r["regime"] != "frozen"]),
        final_trajectories=len([r for r in traj_rows
                                if r["regime"] != "frozen"]),
        final_endpoints=len([r for r in final_rows if r["rule"] != ANCHOR]),
        checkpoints_validated_and_persisted=len(status["checkpoint_log"]),
        total_updates=sum(r["updates"] for r in rows))
    for c in status["screen"]["primary"]:
        print(f"[screen] {c['name']:<38} available={c.get('available')} "
              f"passed={c['aggregate_screen_passed']} mean "
              f"{c.get('mean_primary_difference')}")
    status["complete"] = True
    return 0, "PASS"


if __name__ == "__main__":
    sys.exit(main())
