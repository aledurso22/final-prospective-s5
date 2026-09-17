"""Prospective correction of a trained Momentum DeltaNet memory: one bounded
continuation comparison.

Protocol, frozen for review before any execution:
docs/PROSPECTIVE_MOMENTUM_PROTOCOL.md. NOT AUTHORIZED TO RUN until cleared.

Six arms, each continuing from a designated DEVELOPMENT slot-B checkpoint of
the completed meta-delta run (READ ONLY):

    prospective_momentum  candidate, native momentum checkpoint + kappa = 0
    momentum_delta        native continuation of the same checkpoint
    gain_momentum         same checkpoint + gain-only control, log_g = 0
    gated_delta           Gated DeltaNet, its own checkpoint
    gp_two_sided          completed generalized rule, unchanged, own checkpoint
    tss_eq17              TSS Eq. (17) direct, applicability-limited, own

Outer optimizer state reset identically; all parameters trained; 200 updates;
slots A (lr 0.003) / B (lr 0.01) for every family on development seed 400;
final seeds 401-403 each restart from the designated source. One 600-second
cap for everything. Execution status is separate from every performance
verdict.
"""

import argparse
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

from experiments.meta_delta import dynamics as MDD                # noqa: E402
from experiments.meta_delta import study as MS                    # noqa: E402
from experiments.nested_memory import task as TK                  # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import source as SRC        # noqa: E402

DEV_SEED = 400
FINAL_SEEDS = (401, 402, 403)
UPDATES = 200
BATCH_PER_FAMILY = MS.BATCH_PER_FAMILY
VAL_PER_FAMILY = MS.VAL_PER_FAMILY
HELDOUT_PER_FAMILY = MS.HELDOUT_PER_FAMILY
VAL_AT = (0, 100, 200)
LRS = (("A", 0.003), ("B", 0.01))
#: new named streams (brief s9)
STREAM = dict(train=80_000_000, dev_validation=90_000_000,
              eval_validation=91_000_000, heldout=100_000_000)
SEED_SPAN = 1000                  # seeds are < 1000 in every study's layout
#: identity of arms 1-3 at update zero (protocol s5)
IDENTITY_CE_REL = 2e-5
#: the same transformation object as the completed study; lr applied per call
TX = MS.TX


def train_stream_seed(seed, update):
    return STREAM["train"] + seed * 10_000 + update


def previous_streams():
    """Every stream the task generator has consumed in earlier studies, as
    closed integer ranges (single seeds are ranges of length one)."""
    tr = [(a, b) for a, b in MS.PREVIOUS_TRAIN_RANGES]
    tr.append((MS.STREAM["train"],
               MS.STREAM["train"] + (SEED_SPAN - 1) * 10_000 + 999))
    pts = list(MS.PREVIOUS_STREAMS) + [MS.STREAM[k] for k in
                                       ("dev_validation", "eval_validation",
                                        "heldout")]
    return tr + [(s, s) for s in pts]


def new_streams():
    seeds = (DEV_SEED,) + FINAL_SEEDS
    lo = train_stream_seed(min(seeds), 0)
    hi = train_stream_seed(max(seeds), UPDATES - 1)
    return dict(train=(lo, hi),
                dev_validation=(STREAM["dev_validation"],) * 2,
                eval_validation=(STREAM["eval_validation"],) * 2,
                heldout=(STREAM["heldout"],) * 2)


def stream_overlaps():
    """Pairs of overlapping ranges; empty when all are disjoint."""
    new = new_streams()
    bad = []
    names = list(new)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if new[a][0] <= new[b][1] and new[b][0] <= new[a][1]:
                bad.append((a, b))
        for (lo, hi) in previous_streams():
            if new[a][0] <= hi and lo <= new[a][1]:
                bad.append((a, (lo, hi)))
    return bad


def to_jax(batch):
    return MS.to_jax(batch)


# ------------------------------------------------------------ compiled -----
def _episode_loss(rule, p, ep, with_gates=False):
    out = PM.rollout(rule, p, ep)
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)
    ce = optax.softmax_cross_entropy(
        out["logits"], jax.nn.one_hot(lab, TK.N_VALUES)) * q
    correct = (jnp.argmax(out["logits"], -1) == lab) * q
    aux = dict(ce=ce, correct=correct.astype(jnp.float32),
               q=q.astype(jnp.float32), w_norm=out["w_norm"],
               aux_norm=out["aux_norm"])
    if with_gates and rule in PD.MOMENTUM_FAMILY:
        # the gates RETURNED by this rollout (observed-rollout coverage, R4)
        aux["gates"] = tuple(out["gates"])
    return jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0), aux


def batch_loss(rule, p, eps, with_gates=False):
    losses, aux = jax.vmap(lambda e: _episode_loss(rule, p, e, with_gates))(
        eps)
    return jnp.mean(losses), aux


def _project(rule, p):
    """Post-update projection from UPDATED parameters; optimizer untouched.

    For the old generalized arm the telemetry exports the REAL proposal
    raw_r, its projection cap (the completed study's log_rho_upper, margin
    included) and the un-margined log bound -log1p(-1/x), x = eta tau L
    (+inf for x <= 1); all are logarithms of rho."""
    if rule == "gp_two_sided":
        pre = p["raw_r"][0]
        p, t = MDD.project_two_sided(p)
        dt = p["raw_r"].dtype
        eta, tau = jnp.exp(p["raw_eta"]), jnp.exp(p["raw_tau"])
        cap = MDD.log_rho_upper(eta, tau, dt).astype(dt)[0]
        x = (eta * tau * MDD.GATE_BOUND_L)[0]
        xs = jnp.where(x > 1, x, 2.0)
        bound = jnp.where(x > 1, -jnp.log1p(-1.0 / xs), jnp.inf).astype(dt)
        return p, dict(n_projected=t["n_projected"], pre=pre,
                       post=p["raw_r"][0], cap=cap, bound=bound,
                       overshoot=t["max_overshoot"])
    return PD.project(p)


EXTRA_GRAD = {"prospective_momentum": "kappa", "gain_momentum": "log_g",
              "gp_two_sided": "raw_r"}


@partial(jax.jit, static_argnums=(0,))
def train_step(rule, p, opt, eps, lr):
    (loss, aux), g = jax.value_and_grad(batch_loss, argnums=1, has_aux=True)(
        rule, p, eps)
    raw, opt = TX.update(g, opt, p)
    upd = jax.tree_util.tree_map(lambda u: -lr * u, raw)
    p = optax.apply_updates(p, upd)
    p, tel = _project(rule, p)
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    extra_g = (g[EXTRA_GRAD[rule]][0] if rule in EXTRA_GRAD
               else jnp.zeros((), dtype=loss.dtype))
    return (p, opt, loss, acc, optax.global_norm(g), optax.global_norm(upd),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]), tel, extra_g)


@partial(jax.jit, static_argnums=(0,))
def eval_batch(rule, p, eps):
    _, aux = batch_loss(rule, p, eps, with_gates=True)
    return aux


SCALAR_NAMES = MS.SCALAR_NAMES


def host_step(rule, p, opt, seed, u, lr, hist, stream=None):
    """ONE shared host step, used by training AND by the preflight timing, so
    the measured cost includes every host synchronization of a real update."""
    # `stream` (default: this study's continuation stream) lets the
    # independent-source replication reuse this exact step on its own named
    # streams; the completed study's behaviour is unchanged.
    stream = train_stream_seed if stream is None else stream
    eps = to_jax(TK.generate_batch(stream(seed, u),
                                   BATCH_PER_FAMILY))
    o = train_step(rule, p, opt, eps, lr)
    p, opt, tel = o[0], o[1], o[8]
    scalars = dict(zip(SCALAR_NAMES, (float(x) for x in o[2:8])))
    rec = dict(update=u, n_projected=int(tel["n_projected"]),
               pre=float(tel["pre"]), post=float(tel["post"]),
               cap=float(tel["cap"]), bound=float(tel["bound"]),
               overshoot=float(tel["overshoot"]),
               grad=float(o[9]))
    hist.append(rec)
    return p, opt, scalars


def evaluate(rule, p, eps_np, chunk=128, batch_fn=None, aux_sink=None):
    # `batch_fn` and `aux_sink` are additive hooks (TSS containment study):
    # a study may evaluate with its own compiled batch function and receive
    # every chunk's aux, e.g. the coefficients that exact compiled program
    # executed. Defaults leave every completed study's evaluation unchanged.
    n = eps_np["event"].shape[0]
    cat, fam = eps_np["category"], eps_np["family"]
    cs, ces, qs, wn, an, gs = [], [], [], [], [], []
    for i in range(0, n, chunk):
        sl = {k: jnp.asarray(v[i:i + chunk]) for k, v in eps_np.items()
              if k in ("key_id", "val_id", "event", "label")}
        aux = (eval_batch if batch_fn is None else batch_fn)(rule, p, sl)
        if aux_sink is not None:
            aux_sink.append(aux)
        cs.append(onp.asarray(aux["correct"]))
        ces.append(onp.asarray(aux["ce"]))
        qs.append(onp.asarray(aux["q"]))
        wn.append(onp.asarray(aux["w_norm"]))
        an.append(onp.asarray(aux["aux_norm"]))
        if "gates" in aux:
            gs.append([onp.asarray(x) for x in aux["gates"]])
    c = onp.concatenate(cs); ce = onp.concatenate(ces)
    q = onp.concatenate(qs).astype(bool)
    wn = onp.concatenate(wn); an = onp.concatenate(an)
    out = {}
    for fi, fname in enumerate(TK.FAMILIES):
        fm = (fam == fi)[:, None] & q
        per = {}
        for ci, cn in enumerate(TK.CATEGORIES):
            sel = fm & (cat == ci)
            per[cn] = dict(accuracy=float(c[sel].mean()) if sel.any() else None,
                           cross_entropy=float(ce[sel].mean())
                           if sel.any() else None, n=int(sel.sum()))
        macro = [per[cn]["accuracy"] for cn in TK.CATEGORIES
                 if per[cn]["accuracy"] is not None]
        out[fname] = dict(accuracy=float(c[fm].mean()),
                          cross_entropy=float(ce[fm].mean()),
                          macro_accuracy=float(onp.mean(macro)),
                          by_category=per, n=int(fm.sum()))
    out["primary"] = out["revision"]["macro_accuracy"]
    out["revision_ce"] = out["revision"]["cross_entropy"]
    out["retention_revision_untouched"] = float(onp.mean([
        out["revision"]["by_category"]["middle_untouched"]["accuracy"],
        out["revision"]["by_category"]["late_untouched"]["accuracy"]]))
    out["recall_overall"] = out["recall"]["accuracy"]
    out["state_norms"] = dict(
        W_frobenius_mean=float(wn.mean()), W_frobenius_max=float(wn.max()),
        aux_frobenius_mean=float(an.mean()), aux_frobenius_max=float(an.max()),
        aux_meaning=("Q (momentum carry)" if rule in PD.MOMENTUM_FAMILY
                     else "second carry, or 0 for single-carry rules"))
    if gs:
        gates = [onp.concatenate([g[i] for g in gs]) for i in range(4)]
        out["observed_rollout_gates"] = PD.observed_gate_report(
            rule, gates, eps_np["event"],
            kappa=(float(onp.asarray(p["kappa"]).ravel()[0])
                   if "kappa" in p else None),
            executed_g=(float(onp.asarray(PD.executed_gain(p)))
                        if "log_g" in p else None))
    return out


all_finite = MS.all_finite


def metrics_finite(m):
    """Task metrics (the completed study's check) AND the added computed
    quantities: state norms and observed-rollout gates (review R3.4)."""
    if not MS.metrics_finite(m):
        return False
    sn = m.get("state_norms")
    if sn is None:
        return False
    vals = [v for v in sn.values() if not isinstance(v, str)]
    if not all(onp.isfinite(float(v)) for v in vals):
        return False
    og = m.get("observed_rollout_gates")
    return og is None or bool(og.get("finite"))
write = MS.write
save_tree = MS.save_tree


# ------------------------------------------------- validity and reports ----
def validate(rule, p):
    """None if acceptable, else a reason. Nothing is clamped here."""
    if not all_finite(p):
        return "non-finite parameters"
    if rule in PD.MOMENTUM_FAMILY:
        gr = PD.gate_range_report(p)
        if not gr["gates_valid"]:
            return f"gate range/finiteness failed: {gr}"
        rep = PD.transition_report(p, rule)
        if rule == "momentum_delta":
            bad = {k: v for k, v in rep["classification"].items()
                   if k in ("unstable", "nonfinite")}
            return (f"native frozen-token transitions {bad} (diagnosis "
                    f"required; no tolerance change)" if bad else None)
        return PD.transition_failure(rep)
    if rule in ("gp_two_sided", "tss_eq17"):
        return MS.validate_coefficients(rule, p)
    return None


def coefficient_report(rule, p, eps_np=None):
    if rule in PD.MOMENTUM_FAMILY:
        # TABLE coverage here; OBSERVED-ROLLOUT gates are in each evaluation's
        # `observed_rollout_gates` (returned by the actual rollouts, R4)
        out = dict(rule=rule, table_transition=PD.transition_report(p, rule),
                   gate_table=PD.gate_range_report(p))
        if rule == "prospective_momentum":
            out["kappa_stored_directly"] = float(
                onp.asarray(p["kappa"]).ravel()[0])
        if rule == "gain_momentum":
            lg = float(onp.asarray(p["log_g"]).ravel()[0])
            out.update(raw_log_g=lg,
                       executed_g=float(onp.asarray(PD.executed_gain(p))),
                       reference_g_f64=float(onp.exp(onp.float64(lg))))
        return out
    out = MS.coefficient_report(rule, p, eps_np)
    if rule in ("gp_two_sided", "tss_eq17"):
        # raw logarithmic leaves kept SEPARATE from exponentiated coefficients
        out["raw_log_leaves"] = {k: float(onp.asarray(v).ravel()[0])
                                 for k, v in p.items() if k.startswith("raw_")}
        out["exponentiated"] = {k[len("raw_"):]: float(
            onp.exp(onp.asarray(v, onp.float64)).ravel()[0])
            for k, v in p.items() if k.startswith("raw_")}
        for k in list(out):
            if k.startswith("raw_") and k != "raw_log_leaves":
                out.pop(k)
    return out


PARAMETERIZATION = {"prospective_momentum": "kappa (direct value)",
                    "gain_momentum": "log_g (logarithm of the gain)",
                    "gp_two_sided": "raw_r (logarithm of rho)"}


def summarize_history(hist, rule):
    """Projection summary with margins in the parameterization's own units.

    kappa: relative margin 1 - post/cap to the projection cap.
    log_g: log slack cap - post, and relative margin 1 - exp(post - cap) of
           the executed gain to the gain cap.
    raw_r: log slack cap - post.
    `cap` includes the projection margin; `bound` is the un-margined bound
    (for log parameterizations, a log bound)."""
    if not hist:
        return {}
    pre = onp.array([h["pre"] for h in hist])
    post = onp.array([h["post"] for h in hist])
    cap = onp.array([h["cap"] for h in hist])
    bound = onp.array([h["bound"] for h in hist])
    out = dict(parameterization=PARAMETERIZATION[rule],
               n_projection_events=int(sum(h["n_projected"] for h in hist)),
               first_proposal=float(pre[0]), last_value=float(post[-1]),
               min_post=float(post.min()), max_post=float(post.max()),
               min_projection_cap=float(cap.min()),
               min_unmargined_bound=float(bound.min()),
               max_overshoot=float(max(h["overshoot"] for h in hist)))
    fin = onp.isfinite(cap)
    if rule == "prospective_momentum":
        ok = fin & (cap > 0)
        out["min_relative_margin_to_cap"] = (
            float(onp.min(1.0 - post[ok] / cap[ok])) if ok.any() else None)
    else:
        out["min_log_slack_to_cap"] = (float(onp.min(cap[fin] - post[fin]))
                                       if fin.any() else None)
        if rule == "gain_momentum":
            out["min_relative_gain_margin_to_cap"] = (
                float(onp.min(1.0 - onp.exp(post[fin] - cap[fin])))
                if fin.any() else None)
    return out


# --------------------------------------------------------------- one run ---
def run_one(rule, tag, lr_value, seed, source_p, val_np, updates, out,
            deadline, reserve_s, status, stage, stream=None,
            source_label=None):
    lr = jnp.asarray(lr_value, dtype=jnp.float32)
    p = dict(source_p)                        # a restart from the source
    opt = TX.init(p)                          # optimizer state reset
    curve, val_hist, hist = [], [], []
    t0 = time.time()
    last = None
    for u in range(updates + 1):
        if u in VAL_AT:
            m = evaluate(rule, p, val_np)
            val_hist.append(dict(update=u, primary=m["primary"],
                                 revision_ce=m["revision_ce"],
                                 retention=m["retention_revision_untouched"],
                                 recall=m["recall_overall"],
                                 state_norms=m["state_norms"],
                                 full=m if u in (0, updates) else None))
        if u == updates:
            break
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{stage}:{rule}/{tag}/seed{seed} stopped at update {u} of "
                f"{updates}")
            return None
        p, opt, last = host_step(rule, p, opt, seed, u, lr, hist,
                                 stream=stream)
        if u % 50 == 0 or u == updates - 1:
            curve.append(dict(update=u, **last))
    final = val_hist[-1]["full"]
    bad = None
    if last is None or not all(onp.isfinite(v) for v in last.values()):
        bad = "non-finite final training scalars"
    elif not all_finite(p) or not all_finite(opt):
        bad = "non-finite final parameters or optimizer state"
    elif not metrics_finite(final):
        bad = "non-finite validation metric"
    else:
        bad = validate(rule, p)
    stem = f"{stage}_{rule}_{tag}_seed{seed}"
    save_tree(os.path.join(out, "params", stem + ".msgpack"), p)
    save_tree(os.path.join(out, "params", stem + "_opt.msgpack"), opt)
    rec = dict(tag=stage, rule=rule, display=PD.DISPLAY[rule], config=tag,
               lr=lr_value, seed=seed,
               source=(source_label if source_label is not None
                       else SRC.stem(SRC.SOURCE_RULE[rule])),
               wall_s=time.time() - t0, curve=curve,
               validation=[{k: v for k, v in h.items() if k != "full"}
                           for h in val_hist],
               start_validation=val_hist[0]["full"],
               final_validation=final,
               training_gain=dict(
                   primary=final["primary"] - val_hist[0]["primary"],
                   revision_ce=final["revision_ce"]
                   - val_hist[0]["revision_ce"],
                   retention=final["retention_revision_untouched"]
                   - val_hist[0]["retention"],
                   recall=final["recall_overall"] - val_hist[0]["recall"]),
               params=PM.parameter_counts(rule, p), carry=PD.CARRY[rule],
               coefficients_final=coefficient_report(rule, p, val_np),
               extension_history=(hist if rule in EXTRA_GRAD else None),
               extension_summary=summarize_history(hist, rule)
               if rule in EXTRA_GRAD else None,
               invalid=bad)
    return rec, p


# --------------------------------------------------------- source stage ----
def identity_differences(ref, other):
    """Arms 1-3 at update zero: identical per-category accuracy counts and
    cross-entropy within IDENTITY_CE_REL relative. Returns failures."""
    fails = []
    for fam in TK.FAMILIES:
        for cn in TK.CATEGORIES:
            a, b = ref[fam]["by_category"][cn], other[fam]["by_category"][cn]
            dq = abs(a["accuracy"] - b["accuracy"]) * a["n"]
            dce = (abs(a["cross_entropy"] - b["cross_entropy"])
                   / max(abs(a["cross_entropy"]), 1e-12))
            if not (onp.isfinite(dq) and onp.isfinite(dce)) or dq > 0.5 \
                    or dce > IDENTITY_CE_REL:
                fails.append(f"{fam}/{cn}: queries {dq:.3f} CE rel {dce:.2e}")
    return fails


def source_stage(source_dir, val_dev_np, status):
    """Restore, verify, reproduce, convert. Returns (sources, failure)."""
    rep = dict(source_dir=source_dir, hashes_at_restore=SRC.hash_sources(
        source_dir))
    status["source"] = rep
    try:
        st, sel = SRC.load_metadata(source_dir)
        restored = {}
        for fam in SRC.SOURCE_FAMILIES:
            row = SRC.verify_metadata(fam, st, sel)
            p = SRC.restore(fam, source_dir)
            m = evaluate(fam, p, val_dev_np)
            fails, table = SRC.reproduction_differences(row["final_validation"],
                                                        m)
            rep[fam] = dict(file=SRC.stem(fam) + ".msgpack",
                            reproduction=table, reproduction_failures=fails,
                            params=PM.parameter_counts(fam, p))
            print(f"[source] {fam:<22} reproduced primary "
                  f"{m['primary']:.4f} (saved "
                  f"{row['final_validation']['primary']:.4f}) failures "
                  f"{len(fails)}")
            if fails:
                return None, f"{fam}: saved development metrics not " \
                             f"reproduced: {fails}"
            restored[fam] = p
    except SRC.SourceRefusal as e:
        return None, f"source refusal: {e}"
    sources = {}
    for rule in PD.RULES:
        fam = SRC.SOURCE_RULE[rule]
        sources[rule] = (PM.convert_momentum(restored[fam], rule)
                         if rule in PD.MOMENTUM_FAMILY else restored[fam])
        why = validate(rule, sources[rule])
        if why:
            return None, f"{rule} at the source: {why}"
    rep["momentum_source_transitions"] = {
        r: PD.transition_report(sources[r], r) for r in PD.MOMENTUM_FAMILY}
    rep["momentum_source_gate_table"] = PD.gate_range_report(
        sources["prospective_momentum"])
    return sources, None



# ------------------------------------------------------------- preflight ---
def preflight(sources, val_np, status):
    rows, total, failures, retraced_any = [], 0.0, [], False
    n_runs = len(LRS) + len(FINAL_SEEDS)
    for rule in PD.RULES:
        lr = jnp.asarray(LRS[0][1], dtype=jnp.float32)
        p = dict(sources[rule])
        opt = TX.init(p)
        hist = []
        t0 = time.time()
        p2, opt2, _ = host_step(rule, p, opt, DEV_SEED, 0, lr, hist)
        compile_s = time.time() - t0
        p2, opt2, _ = host_step(rule, p2, opt2, DEV_SEED, 1, lr, hist)
        n0 = train_step._cache_size()
        t1 = time.time()
        for u in range(2, 7):
            p2, opt2, measured = host_step(rule, p2, opt2, DEV_SEED, u, lr,
                                           hist)
        step_s = (time.time() - t1) / 5.0
        retraced = train_step._cache_size() != n0
        t2 = time.time()
        evaluate(rule, p2, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        m = evaluate(rule, p2, val_np)
        eval_s = time.time() - t3
        t4 = time.time()
        acc_bad = validate(rule, p2)
        coefficient_report(rule, p2, val_np)
        report_s = time.time() - t4
        bad_scalars = MS.measured_scalar_failures(measured)
        if bad_scalars:
            acc_bad = f"non-finite measured preflight scalars {bad_scalars}"
        elif not (all_finite(p2) and all_finite(opt2) and metrics_finite(m)):
            acc_bad = "non-finite preflight state or metrics"
        if acc_bad:
            failures.append(f"{rule}: {acc_bad}")
        arm_s = (n_runs * (UPDATES * step_s + len(VAL_AT) * eval_s
                           + 2.0 * report_s)
                 + len(FINAL_SEEDS) * 2.0 * eval_s)
        timing = dict(step_s=step_s, eval_s=eval_s, report_s=report_s,
                      arm_remaining_s=arm_s)
        if not all(onp.isfinite(v) and v >= 0 for v in timing.values()):
            failures.append(f"{rule}: non-finite or negative timing {timing}")
        total += arm_s
        retraced_any |= bool(retraced)
        rows.append(dict(rule=rule, compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s, **timing,
                         retraced=bool(retraced), acceptance_failure=acc_bad,
                         measured_scalars=measured, extension_history=hist,
                         params=PM.parameter_counts(rule, p2),
                         carry=PD.CARRY[rule]))
        print(f"[preflight] {rule:<22} compile {compile_s:5.1f}s step "
              f"{step_s * 1e3:7.2f}ms eval {eval_s * 1e3:7.1f}ms report "
              f"{report_s:5.2f}s arm {arm_s:6.1f}s params "
              f"{PM.parameter_counts(rule, p2)['total']} carry "
              f"{PD.CARRY[rule]}")
    host_s = 40.0                       # ALLOWANCE, recorded as such
    total += host_s
    status["preflight"] = dict(rows=rows, host_allowance_s=host_s,
                               projected_remaining_s=total,
                               retraced_any=retraced_any, failures=failures,
                               runs_projected=n_runs * len(PD.RULES),
                               note=("measured on the actual continuation "
                                     "path from each restored source; its "
                                     "updates are discarded"))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, retraced_any, failures


decide_after_preflight = MS.decide_after_preflight


# ------------------------------------------------------------- selection ---
def select(dev_rows, status):
    sel, table = {}, []
    for rule in PD.RULES:
        cand = [r for r in dev_rows if r["rule"] == rule]
        if len(cand) != len(LRS):
            return None, table
        ranked = sorted(cand, key=lambda r: (-r["final_validation"]["primary"],
                                             r["final_validation"]["revision_ce"],
                                             r["config"]))
        sel[rule] = ranked[0]["config"]
        table.append(dict(rule=rule, chosen=ranked[0]["config"],
                          candidates=[dict(config=r["config"], lr=r["lr"],
                                           primary=r["final_validation"]
                                           ["primary"],
                                           revision_ce=r["final_validation"]
                                           ["revision_ce"]) for r in cand]))
    status["selection"] = dict(selected=sel, table=table,
                               rule=("highest revision macro accuracy, then "
                                     "lower revision CE, then slot A"))
    return sel, table


# ---------------------------------------------------------------- screens ---
def compare(final_rows, cand, other, seeds=None):
    def per_seed(rule, key):
        return {r["seed"]: r["heldout"][key] for r in final_rows
                if r["rule"] == rule}

    def mean(rule, key):
        return float(onp.mean(list(per_seed(rule, key).values())))
    a, b = per_seed(cand, "primary"), per_seed(other, "primary")
    paired = {s: a[s] - b[s] for s in sorted(a) if s in b}
    seeds = FINAL_SEEDS if seeds is None else tuple(seeds)
    complete = (set(paired) == set(seeds)
                and len(a) == len(seeds) and len(b) == len(seeds))
    dm = (mean(cand, "primary") - mean(other, "primary")) if complete else None
    d_ret = (mean(cand, "retention_revision_untouched")
             - mean(other, "retention_revision_untouched")) if complete else None
    d_rec = (mean(cand, "recall_overall")
             - mean(other, "recall_overall")) if complete else None
    safe = bool(complete and d_ret >= -0.01 and d_rec >= -0.01)
    positive = bool(complete and all(v > 0 for v in paired.values()))
    return dict(against=other, display=PD.DISPLAY[other],
                mean_primary_difference=dm, paired_primary_differences=paired,
                positive_in_all_seeds=positive, retention_difference=d_ret,
                recall_difference=d_rec, retention_and_recall_safeguards_met=safe,
                complete_paired_seeds=complete,
                passed=bool(complete and dm >= 0.01 and positive and safe))


def screen(final_rows, seeds=None):
    """Separate verdicts; none substitutes for another.

    * literature: the candidate against BOTH Momentum DeltaNet and Gated
      DeltaNet continuations (joint);
    * gain control: the same rule against the gain-only control, reported
      independently;
    * old generalized rule and TSS Eq. (17): reported separately, neither
      substitutes for the literature screen.
    No favourable category substitutes for the primary metric."""
    cand = "prospective_momentum"
    lit = [compare(final_rows, cand, o, seeds) for o in PD.LITERATURE]
    gain = compare(final_rows, cand, "gain_momentum", seeds)
    old = compare(final_rows, cand, "gp_two_sided", seeds)
    tss = compare(final_rows, cand, "tss_eq17", seeds)
    return dict(
        candidate=cand,
        rule=("mean primary difference >= +1 pp; all three paired primary "
              "differences > 0; mean retention AND recall differences each "
              ">= -1 pp"),
        literature=lit, literature_screen_passed=all(c["passed"] for c in lit),
        gain_control=gain, gain_control_comparison_passed=gain["passed"],
        old_generalized=old, old_generalized_comparison_passed=old["passed"],
        tss_eq17=tss, tss_eq17_comparison_passed=tss["passed"],
        tss_label=PD.DISPLAY["tss_eq17"],
        note=("Development continuation screen. Three continuation seeds "
              "measure new training-stream variation conditional on ONE "
              "source per family, not independent pretrained models. Not "
              "significance, not SOTA, and not a benchmark result. The "
              "fixed-coefficient update is QHM-equivalent at alpha = 1; no "
              "optimizer novelty is claimed."))


# ------------------------------------------------------------ finalization ---
class Terminated(BaseException):
    """Raised by the SIGTERM handler so the finalizer runs within the
    launcher's kill grace (review R1/R2)."""


def _on_sigterm(signum, frame):
    raise Terminated(f"signal {signum}")


SEVERITY = {0: 0, 3: 1, 4: 2}
LABEL = {0: "PASS", 3: "INCOMPLETE", 4: "FAILED"}


def _worse(code, new_code):
    return new_code if SEVERITY[new_code] > SEVERITY[code] else code


def finalize(status, code, label, rehash, persist):
    """ONE finalizer for success, ordinary failure and runtime exceptions.

    * the computation's own outcome is kept as `computation_status`;
    * source invariance is re-verified: a CHANGED or missing source forces
      FAILED/4; an UNAVAILABLE verification (no restore baseline, or the
      re-hash raised) can never yield PASS and never claims invariance
      (INCOMPLETE/3 unless already worse);
    * the original failure reason is preserved; integrity and persistence
      problems are APPENDED under their own keys;
    * the status actually persisted carries the returned (code, label).
    Returns (code, label)."""
    status["computation_status"] = label
    status["computation_exit"] = code
    before = (status.get("source") or {}).get("hashes_at_restore")
    problems = []
    try:
        after = rehash()
        status["source_hashes_at_finish"] = after
        if before is None:
            status["source_unchanged"] = None
            status["integrity_verified"] = False
            problems.append((3, "no restore-time source hashes; invariance "
                                "not verified"))
        elif after != before or any(v is None for v in after.values()):
            status["source_unchanged"] = False
            status["integrity_verified"] = False
            problems.append((4, "source changed or missing at finish"))
        else:
            status["source_unchanged"] = True
            status["integrity_verified"] = True
    except Exception as e:                         # recorded, never hidden
        status["source_unchanged"] = None
        status["integrity_verified"] = False
        problems.append((3, f"source re-hash failed: {e!r}; invariance not "
                            "verified"))
    for c, why in problems:
        status.setdefault("integrity_failures", []).append(why)
        code = _worse(code, c)
    if code != status["computation_exit"]:
        label = LABEL[code]
        key = "failed" if code == 4 else None
        if key and not status.get("failed"):
            status["failed"] = "; ".join(w for _, w in problems)
        if code == 3:
            status.setdefault("incomplete", []).append(
                "; ".join(w for _, w in problems))
    status["study_status"] = label
    status["study_exit"] = code
    try:
        persist(status)
    except Exception as e:
        print(f"[!] status could not be persisted: {e!r}")
        code, label = 4, "FAILED"
    print(f"SOURCE_UNCHANGED={status.get('source_unchanged')}")
    print(f"PROSPECTIVE_MOMENTUM_STATUS={label}")
    return code, label


def guarded(body, status, rehash, persist):
    """Run `body() -> (code, label)`; any runtime exception or termination
    after this point still reaches `finalize`, with the original failure
    reason preserved and the exception appended."""
    import traceback
    try:
        code, label = body()
    except Terminated as e:
        status.setdefault("runtime_failures", []).append(
            f"terminated: {e}")
        status.setdefault("incomplete", []).append(
            "terminated by the watchdog before completion")
        code, label = 3, "INCOMPLETE"
        if status.get("failed"):
            code, label = 4, "FAILED"
    except Exception as e:
        status.setdefault("runtime_failures", []).append(
            dict(error=repr(e), traceback=traceback.format_exc()))
        if not status.get("failed"):
            status["failed"] = f"runtime exception {e!r}"
        code, label = 4, "FAILED"
    return finalize(status, code, label, rehash, persist)


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/prospective-momentum")
    ap.add_argument("--source_dir",
                    default="/Users/durso/s5-runs/meta-delta/20260916-222310")
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
            os.path.realpath(args.source_dir)):
        raise SystemExit("REFUSING: output inside the read-only source")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)

    status = dict(run_id=run_id, out=out, backend=backend, dev_seed=DEV_SEED,
                  final_seeds=list(FINAL_SEEDS), updates=UPDATES,
                  streams=STREAM, stream_ranges=new_streams(),
                  previous_stream_ranges=previous_streams(),
                  arms={r: PD.DISPLAY[r] for r in PD.RULES},
                  sources={r: SRC.stem(SRC.SOURCE_RULE[r]) for r in PD.RULES},
                  learning_rates=dict(LRS),
                  projection_relative_margin=PD.PROJ_REL_MARGIN,
                  production_dtype=dict(x64=False, float_dtype="float32"),
                  heldout_policy=("held-out episodes are GENERATED and hashed "
                                  "only at final evaluation, after all final "
                                  "runs finish; the opening is persisted "
                                  "BEFORE evaluation"),
                  terminal_verdict_note=("study_status is this process's "
                                         "verdict; the launcher adds "
                                         "`terminal` (integrity re-check, "
                                         "digest) as the final verdict"),
                  development=[], final=[], incomplete=[])
    status_path = os.path.join(out, "status.json")

    def persist(st):
        st["wall_s"] = time.time() - t0
        write(status_path, st)

    signal.signal(signal.SIGTERM, _on_sigterm)
    code, _ = guarded(lambda: run_study(args, status, out, deadline,
                                        lambda: persist(status)),
                      status, lambda: SRC.hash_sources(args.source_dir),
                      persist)
    return code


def run_study(args, status, out, deadline, save):
    val_dev_src = TK.generate_batch(MS.STREAM["dev_validation"],
                                    MS.VAL_PER_FAMILY)
    val_np = TK.generate_batch(STREAM["dev_validation"], VAL_PER_FAMILY)
    eval_val_np = TK.generate_batch(STREAM["eval_validation"], VAL_PER_FAMILY)
    status["task"] = dict(structure=TK.structure_check(val_np),
                          dev_validation_digest=TK.episode_digest(val_np),
                          eval_validation_digest=TK.episode_digest(eval_val_np))
    save()

    sources, why = source_stage(args.source_dir, val_dev_src, status)
    if why:
        status["failed"] = why
        print(f"[!] {why}")
        return 4, "FAILED"
    # arms 1-3 must be the same function at update zero
    ident = {r: evaluate(r, sources[r], val_np) for r in PD.MOMENTUM_FAMILY}
    id_fail = {r: identity_differences(ident["momentum_delta"], ident[r])
               for r in ("prospective_momentum", "gain_momentum")}
    status["update_zero_identity"] = dict(
        failures=id_fail, primary={r: ident[r]["primary"] for r in ident},
        revision_ce={r: ident[r]["revision_ce"] for r in ident})
    if any(id_fail.values()):
        status["failed"] = f"arms 1-3 differ at update zero: {id_fail}"
        print(f"[!] {status['failed']}")
        return 4, "FAILED"
    save()

    proj, retraced, failures = preflight(sources, val_np, status)
    d = decide_after_preflight(proj, retraced, failures,
                               deadline - time.time() - args.reserve_s)
    if d is not None:
        code, label, why = d
        (status.__setitem__("failed", why) if code == 4
         else status["incomplete"].append(why))
        print(f"[!] {why}")
        return code, label

    dev_rows = []
    for rule in PD.RULES:
        for tag, lr in LRS:
            r = run_one(rule, tag, lr, DEV_SEED, sources[rule], val_np,
                        UPDATES, out, deadline, args.reserve_s, status, "dev")
            if r is None:
                return 3, "INCOMPLETE"
            rec, _ = r
            dev_rows.append(rec)
            status["development"] = dev_rows
            save()
            if rec["invalid"]:
                status["failed"] = f"dev {rule}/{tag}: {rec['invalid']}"
                return 4, "FAILED"
    sel, _ = select(dev_rows, status)
    write(os.path.join(out, "selection.json"), status["selection"])

    final_rows, finals = [], {}
    lr_of = dict(LRS)
    for rule in PD.RULES:
        for seed in FINAL_SEEDS:
            r = run_one(rule, sel[rule], lr_of[sel[rule]], seed, sources[rule],
                        eval_val_np, UPDATES, out, deadline, args.reserve_s,
                        status, "final")
            if r is None:
                status["heldout_opened"] = False
                return 3, "INCOMPLETE"
            rec, p = r
            final_rows.append(rec)
            finals[(rule, seed)] = p
            status["final"] = final_rows
            save()
            if rec["invalid"]:
                status["failed"] = f"final {rule}/{seed}: {rec['invalid']}"
                return 4, "FAILED"

    # R1.4: the opening is persisted BEFORE the data are generated and used
    status["heldout_opened"] = True
    status["heldout_opened_at"] = time.time()
    status["heldout_evaluation_complete"] = False
    save()
    held_np = TK.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    status["task"]["heldout_digest"] = TK.episode_digest(held_np)
    for rec in final_rows:
        rec["heldout"] = evaluate(rec["rule"], finals[(rec["rule"],
                                                      rec["seed"])], held_np)
        if not metrics_finite(rec["heldout"]):
            status["failed"] = f"non-finite held-out {rec['rule']}/{rec['seed']}"
            return 4, "FAILED"
    status["heldout_evaluation_complete"] = True
    status["screen"] = screen(final_rows)
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
