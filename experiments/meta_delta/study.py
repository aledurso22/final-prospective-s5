"""Generalized prospective memory around the delta boundary: bounded screen.

Protocol, frozen for review before any execution:
docs/META_DELTA_PROTOCOL.md. NOT AUTHORIZED TO RUN until the coordinator
clears it.

Six arms, cold start from identical common initializations (protocol s2):
    gp_two_sided          candidate; starts EXACTLY as adaptive_delta
    adaptive_delta        first-order control
    heavy_ball_same_mass  attribution control: same gamma and M, T Rdot removed
    tss_eq17              ordinary-prospective reference, actual TSS Eq. (17)
    gated_delta           Gated DeltaNet rule (literature)
    momentum_delta        Momentum DeltaNet rule (literature, primary)

Two slots per family (lr 0.003 / 0.01) on development seed 300; selection by
the unchanged rule; three final seeds 301/302/303; held-out opened only after
all final runs. One 600-second cap. Execution status is separate from every
performance verdict.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.adaptive_memory import gate as AG           # noqa: E402
from experiments.adaptive_memory import model as AM          # noqa: E402
from experiments.meta_delta import calibrate as CAL          # noqa: E402
from experiments.meta_delta import dynamics as MD            # noqa: E402
from experiments.meta_delta import model as MM               # noqa: E402
from experiments.nested_memory import task as TK             # noqa: E402

DEV_SEED = 300
FINAL_SEEDS = (301, 302, 303)
UPDATES = 200
BATCH_PER_FAMILY = 8
VAL_PER_FAMILY = 256
HELDOUT_PER_FAMILY = 512
VAL_AT = (0, 100, 200)
GRAD_CLIP = 1.0
#: fresh, disjoint named streams. Earlier studies used 0 / 1e6 / 7e6 / 9e6
#: (nested) and 2e7+ / 3.0e7 / 3.1e7 / 4e7 (adaptive).
STREAM = dict(train=50_000_000, dev_validation=60_000_000,
              eval_validation=61_000_000, heldout=70_000_000)
PREVIOUS_STREAMS = (0, 1_000_000, 7_000_000, 9_000_000,
                    30_000_000, 31_000_000, 40_000_000)
PREVIOUS_TRAIN_RANGES = ((1_000_000, 1_000_000 + 999 * 10_000 + 999),
                         (20_000_000, 20_000_000 + 999 * 10_000 + 999))

#: one transformation for all arms; lr applied dynamically (one compilation)
TX = optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                 optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8))


def train_stream_seed(seed, update):
    return STREAM["train"] + seed * 10_000 + update


def to_jax(batch):
    return {k: jnp.asarray(v) for k, v in batch.items()
            if k in ("key_id", "val_id", "event", "label")}


def _episode_loss(rule, p, ep):
    out = MM.rollout(rule, p, ep)
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)
    ce = optax.softmax_cross_entropy(
        out["logits"], jax.nn.one_hot(lab, TK.N_VALUES)) * q
    correct = (jnp.argmax(out["logits"], -1) == lab) * q
    return (jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0),
            dict(ce=ce, correct=correct.astype(jnp.float32),
                 q=q.astype(jnp.float32), w_norm=out["w_norm"],
                 aux_norm=out["aux_norm"]))


def batch_loss(rule, p, eps):
    losses, aux = jax.vmap(lambda e: _episode_loss(rule, p, e))(eps)
    return jnp.mean(losses), aux


@partial(jax.jit, static_argnums=(0,))
def train_step(rule, p, opt, eps, lr):
    (loss, aux), g = jax.value_and_grad(batch_loss, argnums=1, has_aux=True)(
        rule, p, eps)
    raw, opt = TX.update(g, opt, p)
    upd = jax.tree_util.tree_map(lambda u: -lr * u, raw)
    p = optax.apply_updates(p, upd)
    # post-update domain projection of raw_r from the UPDATED eta and tau;
    # optimizer state untouched; a no-op for every arm without raw_r
    p, tel = MD.project_two_sided(p)
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    return (p, opt, loss, acc, optax.global_norm(g), optax.global_norm(upd),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]), tel)


@partial(jax.jit, static_argnums=(0,))
def eval_batch(rule, p, eps):
    _, aux = batch_loss(rule, p, eps)
    return aux


def evaluate(rule, p, eps_np, chunk=128):
    n = eps_np["event"].shape[0]
    cat, fam = eps_np["category"], eps_np["family"]
    cs, ces, qs = [], [], []
    for i in range(0, n, chunk):
        sl = {k: jnp.asarray(v[i:i + chunk]) for k, v in eps_np.items()
              if k in ("key_id", "val_id", "event", "label")}
        aux = eval_batch(rule, p, sl)
        cs.append(onp.asarray(aux["correct"]))
        ces.append(onp.asarray(aux["ce"]))
        qs.append(onp.asarray(aux["q"]))
    c = onp.concatenate(cs); ce = onp.concatenate(ces)
    q = onp.concatenate(qs).astype(bool)
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
    return out


def all_finite(tree):
    return bool(all(onp.all(onp.isfinite(onp.asarray(v)))
                    for v in jax.tree_util.tree_leaves(tree)
                    if onp.issubdtype(onp.asarray(v).dtype, onp.inexact)))


def metrics_finite(m):
    vals = []
    for fam in TK.FAMILIES:
        vals += [m[fam]["accuracy"], m[fam]["cross_entropy"],
                 m[fam]["macro_accuracy"]]
        for cn in TK.CATEGORIES:
            cc = m[fam]["by_category"][cn]
            vals += [v for v in (cc["accuracy"], cc["cross_entropy"])
                     if v is not None]
    return bool(onp.all(onp.isfinite(onp.asarray(vals, dtype=float))))


# ------------------------------------------------ coefficients and domain ---
def validate_coefficients(rule, p):
    """Executed learned coefficients against EACH arm's own declared domain.
    None if acceptable, else a reason. Nothing is clamped here."""
    def sc(v):
        return float(onp.asarray(v).ravel()[0])
    if rule == "gp_two_sided":
        rep = MD.domain_report(p)
        return None if rep["passed"] else f"two-sided domain failed: {rep}"
    if rule == "adaptive_delta":
        eta = sc(jnp.exp(p["raw_eta"]))
        return None if (onp.isfinite(eta) and eta > 0) else f"eta={eta}"
    if rule == "heavy_ball_same_mass":
        eta, tau = sc(jnp.exp(p["raw_eta"])), sc(jnp.exp(p["raw_tau"]))
        ok = all(onp.isfinite(v) and v > 0 for v in (eta, tau))
        return None if ok else f"heavy ball eta={eta} tau={tau}"
    if rule == "tss_eq17":
        eta, T = sc(jnp.exp(p["raw_eta"])), sc(jnp.exp(p["raw_T"]))
        ok = all(onp.isfinite(v) and v > 0 for v in (eta, T))
        return None if ok else f"Eq.17 eta={eta} T={T}"
    return None


def coefficient_report(rule, p):
    out = dict(rule=rule)
    if "gate_u" in p:
        a = onp.asarray(AG.source_weight_table(
            p, jnp.asarray(AM.H32, dtype=p["key_raw"].dtype),
            jnp.asarray(AM.H8, dtype=p["key_raw"].dtype)))
        out["source_weight"] = dict(min=float(a.min()),
                                    median=float(onp.median(a)),
                                    max=float(a.max()))
    if rule == "gp_two_sided":
        out["domain"] = MD.domain_report(p)
    elif rule in ("adaptive_delta", "heavy_ball_same_mass", "tss_eq17"):
        out.update({k: float(onp.exp(onp.asarray(v, onp.float64)).ravel()[0])
                    for k, v in p.items() if k.startswith("raw_")})
    return out


def jsonable(o):
    if isinstance(o, onp.integer):
        return int(o)
    if isinstance(o, (onp.floating, float)):
        return float(o)
    if isinstance(o, onp.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, int, str)) or o is None:
        return o
    return str(o)


def write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)
    os.replace(tmp, path)


def save_tree(path, tree):
    from flax import serialization
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(serialization.to_bytes(
            jax.tree_util.tree_map(lambda v: onp.asarray(v), tree)))


# --------------------------------------------------------------- one run ---
def run_one(slot, seed, val_np, updates, out, deadline, reserve_s, status, tag):
    rule = slot["rule"]
    lr = jnp.asarray(slot["lr"], dtype=jnp.float32)
    p = MM.init_params(rule, seed, init_coeffs=slot)
    opt = TX.init(p)
    curve, val_hist = [], []
    tel = dict(events=0, max_overshoot=0.0, min_log_margin=float("inf"))
    t0 = time.time()
    last = None
    for u in range(updates + 1):
        if u in VAL_AT:
            m = evaluate(rule, p, val_np)
            val_hist.append(dict(update=u, primary=m["primary"],
                                 revision_ce=m["revision_ce"],
                                 retention=m["retention_revision_untouched"],
                                 recall=m["recall_overall"],
                                 full=m if u == updates else None))
        if u == updates:
            break
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{tag}:{rule}/{slot['config']}/seed{seed} stopped at update "
                f"{u} of {updates}")
            return None
        eps = to_jax(TK.generate_batch(train_stream_seed(seed, u),
                                       BATCH_PER_FAMILY))
        (p, opt, loss, acc, gn, un, wn, an, t) = train_step(rule, p, opt, eps,
                                                             lr)
        tel["events"] += int(t["n_projected"])
        tel["max_overshoot"] = max(tel["max_overshoot"],
                                   float(t["max_overshoot"]))
        tel["min_log_margin"] = min(tel["min_log_margin"],
                                    float(t["log_margin"]))
        last = [float(loss), float(acc), float(gn), float(un), float(wn),
                float(an)]
        if u % 50 == 0:
            curve.append(dict(update=u, loss=last[0], train_acc=last[1],
                              grad_norm=last[2], update_norm=last[3],
                              w_norm=last[4], aux_norm=last[5]))
    final = val_hist[-1]["full"]
    bad = None
    if last is None or not all(onp.isfinite(last)):
        bad = "non-finite final training scalars"
    elif not all_finite(p) or not all_finite(opt):
        bad = "non-finite final parameters or optimizer state"
    elif not metrics_finite(final):
        bad = "non-finite validation metric"
    else:
        bad = validate_coefficients(rule, p)
    stem = f"{tag}_{rule}_{slot['config']}_seed{seed}"
    save_tree(os.path.join(out, "params", stem + ".msgpack"), p)
    save_tree(os.path.join(out, "params", stem + "_opt.msgpack"), opt)
    rec = dict(tag=tag, rule=rule, display=MD.DISPLAY[rule],
               config=slot["config"], lr=slot["lr"], seed=seed,
               wall_s=time.time() - t0, curve=curve,
               validation=[{k: v for k, v in h.items() if k != "full"}
                           for h in val_hist],
               final_validation=final,
               training_gain=dict(
                   primary=final["primary"] - val_hist[0]["primary"],
                   revision_ce=final["revision_ce"]
                   - val_hist[0]["revision_ce"]),
               params=MM.parameter_counts(rule, p), carry=MD.CARRY[rule],
               initialization=slot,
               coefficients_final=coefficient_report(rule, p),
               projection_telemetry=tel, invalid=bad)
    return rec, p


# ------------------------------------------------------------- preflight ---
def preflight(cfg, val_np, status):
    eps = to_jax(TK.generate_batch(train_stream_seed(DEV_SEED, 0),
                                   BATCH_PER_FAMILY))
    rows, total, failures, retraced_any = [], 0.0, [], False
    n_runs = 2 + len(FINAL_SEEDS)
    for rule in MD.RULES:
        slot = cfg["slots"][f"{rule}/A"]
        lr = jnp.asarray(slot["lr"], dtype=jnp.float32)
        p = MM.init_params(rule, DEV_SEED, init_coeffs=slot)
        opt = TX.init(p)
        t0 = time.time()
        o = train_step(rule, p, opt, eps, lr)
        jax.block_until_ready(o[2])
        compile_s = time.time() - t0
        p2, opt2 = o[0], o[1]
        o = train_step(rule, p2, opt2, eps, lr)
        p2, opt2 = o[0], o[1]
        jax.block_until_ready(o[2])
        n0 = train_step._cache_size()
        t1 = time.time()
        for _ in range(5):
            batch = to_jax(TK.generate_batch(train_stream_seed(DEV_SEED, 1),
                                             BATCH_PER_FAMILY))
            o = train_step(rule, p2, opt2, batch, lr)
            p2, opt2 = o[0], o[1]
            _ = [float(x) for x in o[2:8]] + [int(o[8]["n_projected"])]
        step_s = (time.time() - t1) / 5.0
        retraced = train_step._cache_size() != n0
        t2 = time.time()
        evaluate(rule, p2, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        m = evaluate(rule, p2, val_np)
        eval_s = time.time() - t3
        acc_bad = (None if (all_finite(p2) and all_finite(opt2)
                            and metrics_finite(m))
                   else "non-finite preflight state or metrics")
        acc_bad = acc_bad or validate_coefficients(rule, p2)
        if acc_bad:
            failures.append(f"{rule}: {acc_bad}")
        arm_s = (n_runs * (UPDATES * step_s + len(VAL_AT) * eval_s)
                 + len(FINAL_SEEDS) * 2.0 * eval_s)
        timing = dict(step_s=step_s, eval_s=eval_s, arm_remaining_s=arm_s)
        if not all(onp.isfinite(v) and v >= 0 for v in timing.values()):
            failures.append(f"{rule}: non-finite or negative timing {timing}")
        total += arm_s
        retraced_any |= bool(retraced)
        rows.append(dict(rule=rule, compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s, **timing,
                         retraced=bool(retraced), acceptance_failure=acc_bad,
                         params=MM.parameter_counts(rule, p),
                         carry=MD.CARRY[rule]))
        print(f"[preflight] {rule:<22} compile {compile_s:5.1f}s step "
              f"{step_s * 1e3:7.2f}ms eval {eval_s * 1e3:7.1f}ms arm "
              f"{arm_s:6.1f}s params {MM.parameter_counts(rule, p)['total']} "
              f"carry {MD.CARRY[rule]}")
    host_s = 40.0                       # ALLOWANCE, recorded as such
    total += host_s
    status["preflight"] = dict(rows=rows, host_allowance_s=host_s,
                               projected_remaining_s=total,
                               retraced_any=retraced_any, failures=failures,
                               runs_projected=n_runs * len(MD.RULES))
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, retraced_any, failures


def decide_after_preflight(proj, retraced, failures, left_s):
    """Invalid numerical state: FAILED (4). Retrace or over budget: INCOMPLETE
    (3). None: the batch may start."""
    if failures:
        return 4, "FAILED", f"preflight acceptance/timing failed: {failures}"
    if proj is None or not onp.isfinite(proj) or proj < 0:
        return 4, "FAILED", f"non-finite or negative projection {proj!r}"
    if retraced:
        return 3, "INCOMPLETE", "retrace during preflight timing; not started"
    if proj > left_s:
        return 3, "INCOMPLETE", (f"projected {proj:.0f}s > remaining "
                                 f"{left_s:.0f}s; not started, nothing "
                                 f"reduced")
    return None


# ------------------------------------------------------------- selection ---
def select(dev_rows, status):
    sel, table = {}, []
    for rule in MD.RULES:
        cand = [r for r in dev_rows if r["rule"] == rule]
        if len(cand) != 2:
            return None, table
        ranked = sorted(cand, key=lambda r: (-r["final_validation"]["primary"],
                                             r["final_validation"]["revision_ce"],
                                             r["config"]))
        sel[rule] = ranked[0]["config"]
        table.append(dict(rule=rule, chosen=ranked[0]["config"],
                          candidates=[dict(config=r["config"], lr=r["lr"],
                                           primary=r["final_validation"]["primary"],
                                           revision_ce=r["final_validation"]
                                           ["revision_ce"]) for r in cand]))
    status["selection"] = dict(selected=sel, table=table,
                               rule=("highest revision macro accuracy, then "
                                     "lower revision CE, then configuration A"))
    return sel, table


# ---------------------------------------------------------------- screens ---
def _compare(final_rows, cand, other):
    def mean(rule, key):
        return float(onp.mean([r["heldout"][key] for r in final_rows
                               if r["rule"] == rule]))

    def per_seed(rule, key):
        return {r["seed"]: r["heldout"][key] for r in final_rows
                if r["rule"] == rule}
    a, b = per_seed(cand, "primary"), per_seed(other, "primary")
    paired = {s: a[s] - b[s] for s in sorted(a) if s in b}
    complete = set(paired) == set(FINAL_SEEDS)
    dm = (mean(cand, "primary") - mean(other, "primary")) if complete else None
    d_ret = (mean(cand, "retention_revision_untouched")
             - mean(other, "retention_revision_untouched")) if complete else None
    d_rec = (mean(cand, "recall_overall")
             - mean(other, "recall_overall")) if complete else None
    safe = bool(complete and d_ret >= -0.01 and d_rec >= -0.01)
    positive = bool(complete and all(v > 0 for v in paired.values()))
    return dict(against=other, display=MD.DISPLAY[other],
                mean_primary_difference=dm, paired_primary_differences=paired,
                positive_in_all_seeds=positive, retention_difference=d_ret,
                recall_difference=d_rec, retention_safeguard_met=safe,
                complete_paired_seeds=complete,
                passed=bool(complete and dm >= 0.01 and positive and safe))


def screen(final_rows):
    """THREE separate verdicts; none substitutes for another."""
    cand = "gp_two_sided"
    lit = [_compare(final_rows, cand, o) for o in MD.LITERATURE]
    ordn = [_compare(final_rows, cand, o) for o in MD.ORDINARY_PROSPECTIVE]
    attr = _compare(final_rows, cand, "heavy_ball_same_mass")
    delta = _compare(final_rows, cand, "adaptive_delta")
    return dict(
        candidate=cand,
        literature=lit, literature_screen_passed=all(c["passed"] for c in lit),
        ordinary_prospectivity=ordn,
        ordinary_prospectivity_screen_passed=all(c["passed"] for c in ordn),
        attribution_T_Rdot=attr,
        T_Rdot_attribution_passed=attr["passed"],
        versus_delta_descriptive=delta,
        note=("Literature, ordinary-prospectivity and T Rdot attribution are "
              "separate verdicts. A gain over delta alone is reported but "
              "attributes nothing to T Rdot. Development screen only; not "
              "significance and not SOTA."))


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/meta-delta")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)

    def finish(code, label):
        status["wall_s"] = time.time() - t0
        write(os.path.join(out, "status.json"), status)
        print(f"META_DELTA_STATUS={label} out={out}")
        return code

    cfg = CAL.configurations()
    val_np = TK.generate_batch(STREAM["dev_validation"], VAL_PER_FAMILY)
    eval_val_np = TK.generate_batch(STREAM["eval_validation"], VAL_PER_FAMILY)
    held_np = TK.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    status = dict(run_id=run_id, out=out, backend=backend, dev_seed=DEV_SEED,
                  final_seeds=list(FINAL_SEEDS), updates=UPDATES,
                  streams=STREAM, arms={r: MD.DISPLAY[r] for r in MD.RULES},
                  calibration=cfg,
                  task=dict(structure=TK.structure_check(val_np),
                            dev_validation_digest=TK.episode_digest(val_np),
                            heldout_digest=TK.episode_digest(held_np)),
                  development=[], final=[], incomplete=[])
    write(os.path.join(out, "status.json"), status)

    proj, retraced, failures = preflight(cfg, val_np, status)
    d = decide_after_preflight(proj, retraced, failures,
                               deadline - time.time() - args.reserve_s)
    if d is not None:
        code, label, why = d
        (status.__setitem__("failed", why) if code == 4
         else status["incomplete"].append(why))
        print(f"[!] {why}")
        return finish(code, label)

    dev_rows = []
    for rule in MD.RULES:
        for tag, _ in CAL.LRS:
            r = run_one(cfg["slots"][f"{rule}/{tag}"], DEV_SEED, val_np,
                        UPDATES, out, deadline, args.reserve_s, status, "dev")
            if r is None:
                return finish(3, "INCOMPLETE")
            rec, _ = r
            if rec["invalid"]:
                status["failed"] = f"dev {rule}/{tag}: {rec['invalid']}"
                return finish(4, "FAILED")
            dev_rows.append(rec)
            status["development"] = dev_rows
            write(os.path.join(out, "status.json"), status)
    sel, _ = select(dev_rows, status)
    write(os.path.join(out, "selection.json"), status["selection"])

    final_rows, finals = [], {}
    for rule in MD.RULES:
        for seed in FINAL_SEEDS:
            r = run_one(cfg["slots"][f"{rule}/{sel[rule]}"], seed, eval_val_np,
                        UPDATES, out, deadline, args.reserve_s, status,
                        "final")
            if r is None:
                status["heldout_opened"] = False
                return finish(3, "INCOMPLETE")
            rec, p = r
            if rec["invalid"]:
                status["failed"] = f"final {rule}/{seed}: {rec['invalid']}"
                return finish(4, "FAILED")
            final_rows.append(rec)
            finals[(rule, seed)] = p
            status["final"] = final_rows
            write(os.path.join(out, "status.json"), status)

    for rec in final_rows:
        rec["heldout"] = evaluate(rec["rule"], finals[(rec["rule"],
                                                      rec["seed"])], held_np)
        if not metrics_finite(rec["heldout"]):
            status["failed"] = f"non-finite held-out {rec['rule']}/{rec['seed']}"
            return finish(4, "FAILED")
    status["heldout_opened"] = True
    status["screen"] = screen(final_rows)
    sc = status["screen"]
    print(f"[screen] LITERATURE: {sc['literature_screen_passed']}  "
          f"ORDINARY-PROSPECTIVITY: {sc['ordinary_prospectivity_screen_passed']}"
          f"  T Rdot ATTRIBUTION: {sc['T_Rdot_attribution_passed']}")
    status["complete"] = True
    return finish(0, "PASS")


if __name__ == "__main__":
    sys.exit(main())
