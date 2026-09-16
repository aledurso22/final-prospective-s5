"""Checks, preflight, training and evaluation for the nested-memory study.

One process, one deadline. Execution status is reported separately from the
scientific outcome: a completed unfavourable comparison is PASS.
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

from experiments.nested_memory import dynamics as D          # noqa: E402
from experiments.nested_memory import model as MD            # noqa: E402
from experiments.nested_memory import task as TK             # noqa: E402

SEEDS = (100, 101, 102)
UPDATES = 200
BATCH_PER_FAMILY = 8                  # batch 16 total
VAL_PER_FAMILY = 256
HELDOUT_PER_FAMILY = 512
VAL_AT = (0, 100, 200)
LR = 3e-3
GRAD_CLIP = 1.0
#: named RNG streams; recorded rather than derived from registration order
STREAM = dict(initialization=0, train=1_000_000, validation=7_000_000,
              heldout=9_000_000)

_TX = None


def get_tx():
    global _TX
    if _TX is None:
        _TX = optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                          optax.adam(LR, b1=0.9, b2=0.999, eps=1e-8))
    return _TX


def train_stream_seed(seed, update):
    return STREAM["train"] + seed * 10_000 + update


def to_jax(batch):
    return {k: jnp.asarray(v) for k, v in batch.items()
            if k in ("key_id", "val_id", "event", "label", "category",
                     "family")}


def _episode_loss(rule, p, ep, const):
    out = MD.rollout(rule, p, ep, const)
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)
    ce = optax.softmax_cross_entropy(out["logits"], jax.nn.one_hot(
        lab, TK.N_VALUES)) * q
    correct = (jnp.argmax(out["logits"], -1) == lab) * q
    return (jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0),
            dict(ce=ce, correct=correct.astype(jnp.float32), q=q.astype(
                jnp.float32), w_norm=out["w_norm"], aux_norm=out["aux_norm"]))


def batch_loss(rule, p, eps, const):
    f = jax.vmap(lambda e: _episode_loss(rule, p, e, const))
    losses, aux = f(eps)
    return jnp.mean(losses), aux


@partial(jax.jit, static_argnums=(0, 1))
def train_step(rule, tx, p, opt, eps, const):
    (loss, aux), g = jax.value_and_grad(batch_loss, argnums=1, has_aux=True)(
        rule, p, eps, const)
    upd, opt = tx.update(g, opt, p)
    p = optax.apply_updates(p, upd)
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    return (p, opt, loss, acc, optax.global_norm(g),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]))


@partial(jax.jit, static_argnums=(0,))
def eval_batch(rule, p, eps, const):
    _, aux = batch_loss(rule, p, eps, const)
    return aux


def evaluate(rule, p, eps_np, const, chunk=128):
    """Per-family, per-category accuracy and cross entropy."""
    n = eps_np["event"].shape[0]
    corr = onp.zeros(n, dtype=object)
    cat = eps_np["category"]; fam = eps_np["family"]
    all_c, all_ce, all_q = [], [], []
    for i in range(0, n, chunk):
        sl = {k: jnp.asarray(v[i:i + chunk]) for k, v in eps_np.items()
              if k in ("key_id", "val_id", "event", "label")}
        aux = eval_batch(rule, p, sl, const)
        all_c.append(onp.asarray(aux["correct"]))
        all_ce.append(onp.asarray(aux["ce"]))
        all_q.append(onp.asarray(aux["q"]))
    c = onp.concatenate(all_c); ce = onp.concatenate(all_ce)
    q = onp.concatenate(all_q).astype(bool)
    out = {}
    for fi, fname in enumerate(TK.FAMILIES):
        fm = (fam == fi)[:, None] & q
        per_cat = {}
        for ci, cname in enumerate(TK.CATEGORIES):
            sel = fm & (cat == ci)
            per_cat[cname] = dict(
                accuracy=float(c[sel].mean()) if sel.any() else None,
                cross_entropy=float(ce[sel].mean()) if sel.any() else None,
                n=int(sel.sum()))
        macro = [per_cat[cn]["accuracy"] for cn in TK.CATEGORIES
                 if per_cat[cn]["accuracy"] is not None]
        out[fname] = dict(
            accuracy=float(c[fm].mean()), cross_entropy=float(ce[fm].mean()),
            macro_accuracy=float(onp.mean(macro)), by_category=per_cat,
            n=int(fm.sum()))
    # PRIMARY: revision-family macro accuracy, equal weight per category
    out["primary"] = out["revision"]["macro_accuracy"]
    out["retention_revision_untouched"] = float(onp.mean([
        out["revision"]["by_category"]["middle_untouched"]["accuracy"],
        out["revision"]["by_category"]["late_untouched"]["accuracy"]]))
    out["recall_overall"] = out["recall"]["accuracy"]
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
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)


def save_tree(path, tree):
    from flax import serialization
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(serialization.to_bytes(
            jax.tree_util.tree_map(lambda v: onp.asarray(v), tree)))


def all_finite(tree):
    return bool(all(onp.all(onp.isfinite(onp.asarray(v)))
                    for v in jax.tree_util.tree_leaves(tree)))


def metrics_finite(m):
    """Every reported scalar, including per-category cross entropies."""
    vals = []
    for fam in TK.FAMILIES:
        vals += [m[fam]["accuracy"], m[fam]["cross_entropy"],
                 m[fam]["macro_accuracy"]]
        for cn in TK.CATEGORIES:
            c = m[fam]["by_category"][cn]
            vals += [v for v in (c["accuracy"], c["cross_entropy"])
                     if v is not None]
    vals += [m["primary"], m["retention_revision_untouched"],
             m["recall_overall"]]
    return bool(onp.all(onp.isfinite(onp.asarray(vals, dtype=float))))


def gate_report(rule, p, eps):
    """Trained gate distributions and boundary occupancy, for the record."""
    if rule not in ("gated_delta", "momentum_delta"):
        return None
    x = MD.gate_features(jnp.asarray(eps["key_id"]).reshape(-1),
                         jnp.asarray(eps["val_id"]).reshape(-1),
                         jnp.asarray(eps["event"]).reshape(-1))
    x = x.astype(p["key_raw"].dtype)

    def st(v):
        v = onp.asarray(v).ravel()
        return dict(min=float(v.min()), median=float(onp.median(v)),
                    max=float(v.max()), mean=float(v.mean()))

    if rule == "gated_delta":
        a, b = MD._gated_delta_gates(p, x)
        return dict(alpha=st(a), beta=st(b))
    a, b, mu, eta = MD._momentum_gates(p, x)
    at_clamp = float(onp.mean(
        onp.abs(onp.log(onp.asarray(mu)) - MD.MIN_LOG_MU) < 1e-6))
    return dict(alpha=st(a), beta=st(b), mu=st(mu), eta=st(eta),
                min_log_mu=MD.MIN_LOG_MU,
                fraction_of_tokens_at_the_mu_clamp=at_clamp)


def config_hash(rule, seed):
    h = hashlib.sha256()
    h.update(repr((rule, seed, UPDATES, BATCH_PER_FAMILY, LR, GRAD_CLIP,
                   D.M_REF, D.GAMMA_REF, D.T_REF, D.H_REF,
                   MD.MIN_LOG_MU, TK.SEQ_LEN)).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------- preflight ---
def preflight(val_eps, status, seeds):
    """Synchronized steady-state timings for ALL FIVE arms, plus evaluation.

    Compilation is timed separately and counted once per arm; the step is timed
    only after feeding the step its own optimizer output, so a retrace cannot
    be read as slow physics. The projection covers training, the three
    validation passes per run, the held-out pass and the remaining
    compilations - not the training step alone.
    """
    tx = get_tx()
    eps = to_jax(TK.generate_batch(train_stream_seed(seeds[0], 0),
                                   BATCH_PER_FAMILY))
    rows, total, incurred, retraced_any = [], 0.0, 0.0, False
    for rule in D.RULES:
        const = MD.constants_for(rule)
        p = MD.init_params(rule, seeds[0])
        opt = tx.init(p)
        t0 = time.time()
        p2, opt2, loss, acc, gn, wn, an = train_step(rule, tx, p, opt, eps, const)
        jax.block_until_ready(loss)
        compile_s = time.time() - t0
        # consume our own output once before timing
        p2, opt2, loss, acc, gn, wn, an = train_step(rule, tx, p2, opt2, eps, const)
        jax.block_until_ready(loss)
        n_before = train_step._cache_size()
        t1 = time.time()
        for _ in range(5):
            p2, opt2, loss, acc, gn, wn, an = train_step(rule, tx, p2, opt2, eps,
                                                     const)
        jax.block_until_ready(loss)
        step_s = (time.time() - t1) / 5.0
        retraced = train_step._cache_size() != n_before
        t2 = time.time()
        evaluate(rule, p, val_eps, const)
        jax.block_until_ready(p["key_raw"])
        eval_compile_s = time.time() - t2
        t3 = time.time()
        evaluate(rule, p, val_eps, const)
        jax.block_until_ready(p["key_raw"])
        eval_s = time.time() - t3
        # R3: compilation that has ALREADY HAPPENED here is incurred, not
        # outstanding. The jitted functions and the cached optimizer transform
        # persist in this process and every arm's training and evaluation
        # shapes are identical to the ones just compiled, so no further
        # compilation is outstanding for them. Counting it again compares
        # already-spent time against the clock that has already advanced past
        # it, and can refuse a batch that fits.
        incurred += compile_s + eval_compile_s
        arm_s = len(seeds) * (UPDATES * step_s + len(VAL_AT) * eval_s
                              + 2.0 * eval_s)
        total += arm_s
        retraced_any |= bool(retraced)
        rows.append(dict(rule=rule, display=D.DISPLAY[rule],
                         compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s,
                         step_s=step_s, eval_s=eval_s,
                         arm_remaining_s=arm_s,
                         retraced_during_timing=bool(retraced),
                         params=MD.parameter_counts(rule, p),
                         carry=D.CARRY[rule]))
        print(f"[preflight] {D.DISPLAY[rule]:<48} compile {compile_s:5.1f}s"
              f" (+eval {eval_compile_s:4.1f}s)  step {step_s*1e3:7.2f}ms"
              f"  eval {eval_s*1e3:7.1f}ms  arm {arm_s:6.1f}s"
              f"  params {MD.parameter_counts(rule, p)['total']}"
              f"  carry {D.CARRY[rule]}")
        if retraced:
            print(f"[!] {rule}: RETRACE during step timing; projection "
                  f"unreliable")
        del p2, opt2
    # An ALLOWANCE, not a measurement: episode generation, metrics assembly
    # and serialization. Labelled as such.
    host_s = 45.0
    total += host_s
    status["preflight"] = dict(
        rows=rows, host_allowance_s=host_s,
        incurred_compilation_s=incurred, projected_remaining_s=total,
        retraced_any=retraced_any,
        scope=("PROJECTED REMAINING work only: seeds x updates, three "
               "validation passes per run, the held-out pass and a host "
               "allowance. Compilation already performed in this preflight is "
               "reported separately as incurred; every training and evaluation "
               "shape is identical to one already compiled in this process, so "
               "no further compilation is outstanding."))
    print(f"PREFLIGHT_INCURRED_COMPILATION_S={incurred:.1f}")
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    if retraced_any:
        print("[!] a retrace was detected during step timing; the projection "
              "is not trustworthy and the batch will not be started")
    return total, retraced_any


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/nested-memory")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--updates", type=int, default=UPDATES)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    left = lambda: deadline - time.time() - args.reserve_s        # noqa: E731
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'. The "
                         f"model study requires a GPU.")
    seeds = [int(s) for s in args.seeds.split(",")]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)

    # fixed, shared evaluation sets from named streams
    val_np = TK.generate_batch(STREAM["validation"], VAL_PER_FAMILY)
    held_np = TK.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    struct = TK.structure_check(val_np)
    const_p = D.prospective_constants()

    status = dict(
        run_id=run_id, out=out, backend=backend, seeds=seeds,
        updates=args.updates, batch_per_family=BATCH_PER_FAMILY,
        lr=LR, grad_clip=GRAD_CLIP, streams=STREAM,
        arms={r: D.DISPLAY[r] for r in D.RULES},
        law_metadata={r: MD.constants_metadata(r) for r in D.RULES},
        coefficients=dict(M=D.M_REF, gamma=D.GAMMA_REF, T=D.T_REF, h=D.H_REF,
                          F=const_p["F"].tolist(), a0=const_p["a0"],
                          b0=const_p["b0"],
                          beta_match=D.beta_match(),
                          ordinary_gradient_beta=D.ordinary_gradient_beta(),
                          admissible_M_le_gamma_T=bool(
                              D.M_REF <= D.GAMMA_REF * D.T_REF)),
        gate_source=dict(
            repo="https://github.com/HuuYuLong/MomentumDeltaNet",
            commit="c6e77fa261fb0c002fae1a14b6209a5b28d2edc9",
            min_log_mu=MD.MIN_LOG_MU,
            note=("min_log_mu = -2 is the OFFICIAL constructor default, which "
                  "resolves the -1/-2 ambiguity between the paper's passages "
                  "from a pinned configuration rather than by preference")),
        heldout_policy=(
            "held-out episodes are generated and hashed BEFORE training from "
            "their own named stream; their EVALUATION is DEFERRED until every "
            "declared configuration finishes. This is deferred evaluation, "
            "not data first created after training."),
        resume_policy=(
            "config hashes are recorded for identification only. No "
            "resume-by-hash path is implemented, so none is promised. "
            "Completed artifacts are preserved and are never overwritten: "
            "each run writes under its own run id."),
        diagnostic_scope=(
            "collected: state and auxiliary norms, gradient norms, trained "
            "gate distributions and mu-clamp occupancy, parameter and carry "
            "counts, per-category metrics, final parameters AND optimizer "
            "state. Anything absent is stated rather than inferred, and no "
            "run is repeated merely to fill a reporting field."),
        task=dict(structure=struct,
                  validation_digest=TK.episode_digest(val_np),
                  heldout_digest=TK.episode_digest(held_np),
                  oracle_exact=TK.oracle_is_exact(val_np)),
        config_hashes={f"{r}/{s}": config_hash(r, s)
                       for r in D.RULES for s in seeds},
        results=[], validation=[], incomplete=[])
    print(f"[*] out={out} backend={backend} seeds={seeds}")
    print(f"[*] queries/seq {struct['queries_per_sequence']}  "
          f"query value field absent: {struct['query_value_field_is_absent']}  "
          f"oracle exact: {status['task']['oracle_exact']}")
    print(f"[*] beta_match={D.beta_match():.7f}  a0={const_p['a0']:.7f}  "
          f"b0={const_p['b0']:.7f}  min_log_mu={MD.MIN_LOG_MU}")
    write(os.path.join(out, "status.json"), status)

    proj, retraced = preflight(val_np, status, seeds)
    write(os.path.join(out, "status.json"), status)
    if retraced:
        status["incomplete"].append(
            "a retrace was detected during preflight step timing; the "
            "projection is untrustworthy and the batch was NOT started")
        write(os.path.join(out, "status.json"), status)
        print(f"NESTED_STATUS=INCOMPLETE out={out}")
        return 3
    if args.preflight_only:
        print(f"NESTED_STATUS=PREFLIGHT_ONLY out={out}")
        return 0
    if proj > left():
        status["incomplete"].append(
            f"projected {proj:.0f}s > remaining {left():.0f}s; the batch was "
            f"NOT started. No seed, update count or competitor was reduced.")
        write(os.path.join(out, "status.json"), status)
        print(f"[!] {status['incomplete'][-1]}")
        print(f"NESTED_STATUS=INCOMPLETE out={out}")
        return 3

    tx = get_tx()
    rows, finals = [], {}
    for rule in D.RULES:
        const = MD.constants_for(rule)
        for seed in seeds:
            if left() < 15:
                status["incomplete"].append(f"{rule}/{seed} not started")
                continue
            p = MD.init_params(rule, seed)
            opt = tx.init(p)
            curve, val_hist = [], []
            t_run = time.time()
            for u in range(args.updates + 1):
                if u in VAL_AT:
                    m = evaluate(rule, p, val_np, const)
                    val_hist.append(dict(update=u, **m))
                    print(f"  [{rule}/{seed}] update {u:>3}  "
                          f"primary={m['primary']:.4f}  "
                          f"recall={m['recall_overall']:.4f}")
                if u == args.updates:
                    break
                eps = to_jax(TK.generate_batch(train_stream_seed(seed, u),
                                               BATCH_PER_FAMILY))
                p, opt, loss, acc, gn, wn, an = train_step(rule, tx, p, opt, eps,
                                                       const)
                if u % 50 == 0:
                    curve.append(dict(update=u, loss=float(loss),
                                      train_acc=float(acc),
                                      grad_norm=float(gn),
                                      w_norm=float(wn), aux_norm=float(an)))
            wall = time.time() - t_run
            # R5: the sampled curve is not the whole run. Final parameters,
            # the last step's scalars and every reported validation metric are
            # checked too, so a NaN in an unsampled tail cannot reach a
            # nominal complete status.
            bad = None
            if not all(onp.isfinite([c["loss"] for c in curve])):
                bad = "non-finite sampled training loss"
            elif not all(onp.isfinite([float(loss), float(acc), float(gn),
                                       float(wn), float(an)])):
                bad = "non-finite final training scalars"
            elif not all_finite(p):
                bad = "non-finite final parameters"
            elif not all(metrics_finite(v) for v in val_hist):
                bad = "non-finite validation metric"
            if bad is not None:
                status["failed"] = f"{rule}/{seed}: {bad}"
                write(os.path.join(out, "status.json"), status)
                print(f"[FAIL] {status['failed']}")
                print(f"NESTED_STATUS=FAILED out={out}")
                return 4
            save_tree(os.path.join(out, "params", f"{rule}_seed{seed}.msgpack"),
                      p)
            save_tree(os.path.join(out, "params",
                                   f"{rule}_seed{seed}_opt.msgpack"), opt)
            finals[(rule, seed)] = p
            rows.append(dict(rule=rule, display=D.DISPLAY[rule], seed=seed,
                             wall_s=wall, curve=curve, validation=val_hist,
                             config_hash=config_hash(rule, seed),
                             params=MD.parameter_counts(rule, p),
                             carry=D.CARRY[rule], law=D.LAW_METADATA[rule],
                             gates=gate_report(rule, p, val_np)))
            status["results"] = rows
            write(os.path.join(out, "results.json"), rows)
            write(os.path.join(out, "status.json"), status)

    complete = (len(rows) == len(D.RULES) * len(seeds)
                and not status["incomplete"])
    # HELD-OUT is opened only after every declared configuration finished
    if complete:
        for r in rows:
            const = MD.constants_for(r["rule"])
            r["heldout"] = evaluate(r["rule"], finals[(r["rule"], r["seed"])],
                                    held_np, const)
        status["results"] = rows
        status["heldout_opened"] = True
        for r in rows:
            if not metrics_finite(r["heldout"]):
                status["failed"] = (f"{r['rule']}/{r['seed']}: non-finite "
                                    f"held-out metric")
                write(os.path.join(out, "status.json"), status)
                print(f"NESTED_STATUS=FAILED out={out}")
                return 4
    else:
        status["heldout_opened"] = False
        status["incomplete"].append(
            "held-out evaluation NOT opened: an incomplete batch has no "
            "comparative success verdict")

    status["wall_s"] = time.time() - t0
    status["complete"] = complete
    write(os.path.join(out, "results.json"), rows)
    write(os.path.join(out, "status.json"), status)

    if complete:
        print("\n  arm                                          primary  "
              "retention  recall")
        for rule in D.RULES:
            rs = [r for r in rows if r["rule"] == rule]
            pr = onp.mean([r["heldout"]["primary"] for r in rs])
            rt = onp.mean([r["heldout"]["retention_revision_untouched"]
                           for r in rs])
            rc = onp.mean([r["heldout"]["recall_overall"] for r in rs])
            print(f"  {D.DISPLAY[rule]:<44}{pr:9.4f}{rt:11.4f}{rc:8.4f}")
    print(f"[*] wall {status['wall_s']:.0f}s  rows {len(rows)}/"
          f"{len(D.RULES) * len(seeds)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[INCOMPLETE] {m}")
        print(f"NESTED_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"NESTED_STATUS=COMPLETE out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
