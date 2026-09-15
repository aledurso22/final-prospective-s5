"""Part A: fast tracking and delayed recall, four temporal laws, full BPTT.

One process, one deadline. This part tests whether a temporal FORWARD model is
useful under a common accurate optimizer. It is not a credit-assignment result:
every arm is trained with exact BPTT through its own discrete updates, so a
better score here says something about the forward law and nothing about any
local learning rule.
"""

import argparse
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

import tasks.dual_recall as TK                                     # noqa: E402
from s5 import tss_models as TM                                    # noqa: E402
from s5.tss_cells import (DT, T_HORIZON, equivalent_adaptation,    # noqa: E402
                          retained_vector_field, rollout_ode,
                          symmetric_reference, tss_gamma_equivalent)

SEEDS = (100, 101, 102)
UPDATES = 300
BATCH = 16
#: One learning rate, declared before execution and identical for every arm.
#: No development budget was spent on any arm, so the budget is equal at zero;
#: choosing a rate for the new model alone is exactly what this avoids.
LR = 3e-3
GRAD_CLIP = 1.0


def loss_terms(arm, p, xs, y_sig, y_cls, q_idx, cfg):
    """Mean current-signal MSE plus mean recall cross entropy at query steps.

    The two terms are averaged SEPARATELY and then summed, so the 63 non-query
    steps cannot drown out the single recall decision.
    """
    pred_sig, logits, diag = TM.batched_forward(arm, p, xs, cfg)
    mse = jnp.mean((pred_sig - y_sig) ** 2)
    rows = jnp.arange(xs.shape[0])
    q_logits = logits[rows, q_idx]
    ce = jnp.mean(optax.softmax_cross_entropy(
        q_logits, jax.nn.one_hot(y_cls, TK.N_CLASSES)))
    acc = jnp.mean(jnp.argmax(q_logits, -1) == y_cls)
    return mse + ce, dict(mse=mse, ce=ce, acc=acc,
                          residual=diag["fixed_point_residual"])


@partial(jax.jit, static_argnums=(0, 1))
def train_step(arm, tx, p, opt, xs, y_sig, y_cls, q_idx, cfg):
    (loss, aux), g = jax.value_and_grad(loss_terms, argnums=1, has_aux=True)(
        arm, p, xs, y_sig, y_cls, q_idx, cfg)
    upd, opt = tx.update(g, opt, p)
    p = optax.apply_updates(p, upd)
    p = TM.project_params(arm, p)
    return p, opt, loss, aux, optax.global_norm(g)


@partial(jax.jit, static_argnums=(0,))
def eval_all(arm, p, xs, y_sig, y_cls, q_idx, cfg):
    pred_sig, logits, diag = TM.batched_forward(arm, p, xs, cfg)
    rows = jnp.arange(xs.shape[0])
    q_logits = logits[rows, q_idx]
    return dict(mse=jnp.mean((pred_sig - y_sig) ** 2),
                ce=jnp.mean(optax.softmax_cross_entropy(
                    q_logits, jax.nn.one_hot(y_cls, TK.N_CLASSES))),
                correct=(jnp.argmax(q_logits, -1) == y_cls),
                logits=q_logits,
                residual=diag["fixed_point_residual"])


def evaluate(arm, p, batch, cfg):
    xs, y_sig, y_cls, q_idx, meta = batch
    out = eval_all(arm, p, jnp.asarray(xs), jnp.asarray(y_sig),
                   jnp.asarray(y_cls), jnp.asarray(q_idx), cfg)
    correct = onp.asarray(out["correct"])
    by_delay = {int(d): float(onp.mean(correct[meta["delay"] == d]))
                for d in TK.DELAYS}
    return dict(signal_mse=float(out["mse"]), recall_ce=float(out["ce"]),
                recall_accuracy=float(onp.mean(correct)),
                recall_by_delay=by_delay,
                combined=float(out["mse"]) + float(out["ce"]),
                fixed_point_residual=float(out["residual"]),
                logits=onp.asarray(out["logits"]))


def interventions(arm, p, base, cfg, seed):
    """Cue shuffle and distractor replacement, on matched regenerated batches."""
    shuffled = TK.shuffled_cue(base, seed)
    replaced = TK.replaced_distractors(base, seed + 1)
    b = evaluate(arm, p, base, cfg)
    s = evaluate(arm, p, shuffled, cfg)
    r = evaluate(arm, p, replaced, cfg)
    return dict(
        base_recall=b["recall_accuracy"],
        follows_shuffled_cue=s["recall_accuracy"],
        recall_after_distractor_replacement=r["recall_accuracy"],
        cue_logit_l2=float(onp.sqrt(onp.mean(
            onp.sum((s["logits"] - b["logits"]) ** 2, axis=-1)))),
        distractor_logit_l2=float(onp.sqrt(onp.mean(
            onp.sum((r["logits"] - b["logits"]) ** 2, axis=-1)))),
        note=("follows_shuffled_cue scores the NEW cue class: a model that "
              "actually reads the cue should stay high, one that memorized "
              "the distractor stream should not"))


def jsonable(o):
    if isinstance(o, onp.integer):
        return int(o)
    if isinstance(o, onp.floating):
        return float(o)
    if isinstance(o, onp.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


def write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)


def preflight(cfg, status):
    """Measure compile and steady step cost for EVERY arm, then project.

    Compile and steady cost are measured separately and only the steady part is
    scaled by the update count. Preflight state is discarded; the comparative
    runs re-initialize.
    """
    rng = onp.random.RandomState(0)
    xs, y_sig, y_cls, q_idx, _ = TK.generate(rng, BATCH)
    args = (jnp.asarray(xs), jnp.asarray(y_sig), jnp.asarray(y_cls),
            jnp.asarray(q_idx))
    rows, total = [], 0.0
    for arm in TM.ARMS:
        p = TM.init_params(arm, SEEDS[0])
        tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR))
        opt = tx.init(p)
        t0 = time.time()
        p2, opt2, loss, aux, gn = train_step(arm, tx, p, opt, *args, cfg)
        loss.block_until_ready()
        compile_s = time.time() - t0
        t1 = time.time()
        for _ in range(3):
            p2, opt2, loss, aux, gn = train_step(arm, tx, p2, opt2, *args, cfg)
        loss.block_until_ready()
        step_s = (time.time() - t1) / 3.0
        arm_s = len(SEEDS) * (compile_s * 0 + UPDATES * step_s) + compile_s
        total += arm_s
        rows.append(dict(arm=arm, compile_s=compile_s, step_s=step_s,
                         arm_total_s=arm_s,
                         params=TM.parameter_count(p),
                         state=TM.temporal_state_count(arm)))
        print(f"[preflight] {arm:24s} compile {compile_s:6.1f}s  "
              f"step {step_s * 1e3:7.2f}ms  arm {arm_s:6.1f}s  "
              f"params {TM.parameter_count(p):5d}  "
              f"states {TM.temporal_state_count(arm)['total']:3d}")
        del p2, opt2
    host_s = 45.0          # evaluation, interventions and probes
    total += host_s
    status["preflight"] = dict(rows=rows, host_allowance_s=host_s,
                               projected_total_s=total,
                               scope="compile once per arm plus seeds x updates")
    print(f"PREFLIGHT_A_PROJECTED_TOTAL_S={total:.1f}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/tss_pilot")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--updates", type=int, default=UPDATES)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    left = lambda: deadline - time.time() - args.reserve_s        # noqa: E731

    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    cfg = TM.coefficients()
    seeds = [int(s) for s in args.seeds.split(",")]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id, "part_a")
    os.makedirs(out, exist_ok=True)

    eval_batch = TK.balanced_eval_set()
    struct = TK.structure_check(eval_batch)
    leak = TK.leakage_probe(eval_batch)
    ref = symmetric_reference()
    tss_eq = tss_gamma_equivalent(cfg["tss"]["tau_m"], cfg["tss"]["eps"],
                                  cfg["tss"]["tau_p"])

    status = dict(
        part="A", run_id=run_id, out=out, backend=backend, seeds=seeds,
        arms=list(TM.ARMS), updates=args.updates, batch=BATCH, lr=LR,
        grad_clip=GRAD_CLIP, dt=DT, clock="one input step = one unit of time",
        coefficients=cfg, symmetric_reference=ref,
        state_budget=TM.STATE_BUDGET,
        state_counts={a: TM.temporal_state_count(a) for a in TM.ARMS},
        task=dict(structure=struct, leakage_probe=leak,
                  seq_len=TK.SEQ_LEN, delays=list(TK.DELAYS),
                  cue_steps=list(TK.CUE_STEPS), signal_tau=TK.SIGNAL_TAU),
        sector_audit=dict(
            tss_matched_point_maps_to=tss_eq,
            claim=("TSS's matching prescription tau_p = tau_m maps to gamma = "
                   "0 under (MAP), so M <= gamma T fails for any M > 0: arms 2 "
                   "and 4 are two coefficient sectors of ONE family, and the "
                   "circuit inequality excludes the TSS-matched sector. They "
                   "are not equivalent realizations and not different model "
                   "classes."),
            shared_poles=("arms 2 and 4 are given the same Q roots, so they "
                          "differ only in the prospective zero: f + T f' "
                          "against f + 2T f'")),
        interpretation=dict(
            what_this_tests="usefulness of the temporal FORWARD model under a "
                            "common accurate optimizer (exact BPTT)",
            what_this_does_not_test="any credit-assignment claim; Part B is "
                                    "the experiment that can support one",
            parameter_vs_state=("the state budget is matched, parameter counts "
                                "are NOT; both are reported and no "
                                "capacity-efficiency claim follows from this "
                                "batch"),
            seeds="three seeds identify a development signal, not a "
                  "superiority theorem"),
        results=[], incomplete=[])
    print(f"[*] out={out} backend={backend} seeds={seeds}")
    print(f"[*] task structure: {struct}")
    print(f"[*] leakage probe (must be near {leak['chance']:.3f}): "
          f"{leak['accuracy']:.4f}")
    print(f"[*] TSS matched point maps to gamma={tss_eq['gamma']:.6g}, "
          f"admissible under M<=gamma*T: {tss_eq['admissible']}")
    write(os.path.join(out, "status.json"), status)

    proj = preflight(cfg, status)
    write(os.path.join(out, "status.json"), status)
    if args.preflight_only:
        print(f"TSS_A_STATUS=PREFLIGHT_ONLY out={out}")
        return 0
    if proj > left():
        status["incomplete"].append(
            f"projected {proj:.0f}s > remaining {left():.0f}s; Part A NOT "
            f"started. No arm, seed or update count was reduced.")
        write(os.path.join(out, "status.json"), status)
        print(f"[!] {status['incomplete'][-1]}")
        print(f"TSS_A_STATUS=INCOMPLETE out={out}")
        return 3

    rows = []
    init_by_arm = {}
    for seed in seeds:
        if left() < 20:
            status["incomplete"].append(f"seed {seed} not started (budget)")
            break
        rng = onp.random.RandomState(seed)
        # ONE ordered stream of training batches, shared by every arm
        stream = [TK.generate(rng, BATCH) for _ in range(args.updates)]
        for arm in TM.ARMS:
            if left() < 10:
                status["incomplete"].append(f"seed {seed} arm {arm}: budget")
                continue
            p = TM.init_params(arm, seed)
            init_by_arm.setdefault(seed, {})[arm] = p
            tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                             optax.adam(LR))
            opt = tx.init(p)
            curve, t_a = [], time.time()
            for i, (xs, ysig, ycls, q, _m) in enumerate(stream):
                p, opt, loss, aux, gn = train_step(
                    arm, tx, p, opt, jnp.asarray(xs), jnp.asarray(ysig),
                    jnp.asarray(ycls), jnp.asarray(q), cfg)
                if i % 25 == 0 or i == args.updates - 1:
                    curve.append(dict(update=i, loss=float(loss),
                                      mse=float(aux["mse"]),
                                      ce=float(aux["ce"]),
                                      train_acc=float(aux["acc"]),
                                      grad_norm=float(gn)))
            wall = time.time() - t_a
            ev = evaluate(arm, p, eval_batch, cfg)
            iv = interventions(arm, p, eval_batch, cfg, seed)
            ev.pop("logits", None)
            row = dict(seed=seed, arm=arm, wall_s=wall, curve=curve,
                       params=TM.parameter_count(p),
                       state_counts=TM.temporal_state_count(arm),
                       **ev, interventions=iv)
            rows.append(row)
            status["results"] = rows
            write(os.path.join(out, "results.json"), rows)
            write(os.path.join(out, "status.json"), status)
            print(f"[seed {seed}] {arm:24s} mse={ev['signal_mse']:.4f}  "
                  f"recall={ev['recall_accuracy']:.4f}  "
                  f"({'/'.join('%.2f' % ev['recall_by_delay'][d] for d in TK.DELAYS)})"
                  f"  combined={ev['combined']:.4f}  {wall:.0f}s")

    if init_by_arm:
        s0 = sorted(init_by_arm)[0]
        status["shared_tensors_at_seed_%d" % s0] = TM.shared_tensor_report(
            init_by_arm[s0])

    # paired differences, per seed, against every other arm
    by = {(r["seed"], r["arm"]): r for r in rows}
    pairs = {}
    for a in TM.ARMS:
        for b in TM.ARMS:
            if a >= b:
                continue
            d = [dict(seed=s,
                      recall_pp=100.0 * (by[(s, a)]["recall_accuracy"]
                                         - by[(s, b)]["recall_accuracy"]),
                      signal_mse=(by[(s, a)]["signal_mse"]
                                  - by[(s, b)]["signal_mse"]),
                      combined=(by[(s, a)]["combined"] - by[(s, b)]["combined"]))
                 for s in seeds if (s, a) in by and (s, b) in by]
            if d:
                pairs[f"{a} - {b}"] = dict(
                    per_seed=d,
                    mean_recall_pp=float(onp.mean([x["recall_pp"] for x in d])),
                    mean_signal_mse=float(onp.mean([x["signal_mse"]
                                                    for x in d])),
                    mean_combined=float(onp.mean([x["combined"] for x in d])))
    status["paired_differences"] = pairs
    status["wall_s"] = time.time() - t0
    status["complete"] = (len(rows) == len(seeds) * len(TM.ARMS)
                          and not status["incomplete"])
    write(os.path.join(out, "status.json"), status)

    print("\n  arm                       recall   d8/d16/d32        signal MSE")
    for a in TM.ARMS:
        rs = [r for r in rows if r["arm"] == a]
        if not rs:
            print(f"  {a:24s} NOT COMPLETED")
            continue
        print(f"  {a:24s} {onp.mean([r['recall_accuracy'] for r in rs]):.4f}   "
              + "/".join("%.2f" % onp.mean([r["recall_by_delay"][d] for r in rs])
                         for d in TK.DELAYS)
              + f"      {onp.mean([r['signal_mse'] for r in rs]):.4f}")
    print(f"[*] wall {status['wall_s']:.0f}s  rows {len(rows)}/"
          f"{len(seeds) * len(TM.ARMS)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[INCOMPLETE] {m}")
        print(f"TSS_A_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"TSS_A_STATUS=COMPLETE out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
