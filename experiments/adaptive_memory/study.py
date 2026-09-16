"""Adaptive-memory comparison: calibration, checks, selection, training, eval.

Seven arms after the 16 September 2026 ordinary-prospective amendment:
14 development runs on seed 200, 21 final runs on seeds 201/202/203,
35 runs and 7,000 optimizer updates in one 600-second cap.

One process, one 600-second deadline. Execution status (PASS/INCOMPLETE/
FAILED) is reported separately from the performance screen: a completed
unfavourable comparison is execution PASS with the screen FAILED.
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

from experiments.adaptive_memory import calibrate as CAL     # noqa: E402
from experiments.adaptive_memory import dynamics as AD       # noqa: E402
from experiments.adaptive_memory import gate as AG           # noqa: E402
from experiments.adaptive_memory import model as AM          # noqa: E402
from experiments.nested_memory import task as TK             # noqa: E402

DEV_SEED = 200
FINAL_SEEDS = (201, 202, 203)
UPDATES = 200
BATCH_PER_FAMILY = 8
VAL_PER_FAMILY = 256
HELDOUT_PER_FAMILY = 512
VAL_AT = (0, 100, 200)
GRAD_CLIP = 1.0
#: disjoint named streams; the completed study used 0 / 1e6 / 7e6 / 9e6
STREAM = dict(train=20_000_000, dev_validation=30_000_000,
              eval_validation=31_000_000, heldout=40_000_000)

#: ONE transformation object for the whole batch. `optax.adam(lr)` is exactly
#: `chain(scale_by_adam(), scale(-lr))`, so holding the rate OUT of the chain
#: and applying `-lr` in the step keeps the update identical while making the
#: rate a DYNAMIC argument. Both declared rates (0.003, 0.01) therefore share a
#: single compilation per arm, so preflight does not silently omit a second
#: compilation for every configuration-B slot.
TX = optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                 optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8))


def train_stream_seed(seed, update):
    return STREAM["train"] + seed * 10_000 + update


def to_jax(batch):
    return {k: jnp.asarray(v) for k, v in batch.items()
            if k in ("key_id", "val_id", "event", "label")}


def _episode_loss(rule, p, ep):
    out = AM.rollout(rule, p, ep)
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
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    return (p, opt, loss, acc, optax.global_norm(g), optax.global_norm(upd),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]))


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
        cs.append(onp.asarray(aux["correct"])); ces.append(onp.asarray(aux["ce"]))
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
                    for v in jax.tree_util.tree_leaves(tree)))


def metrics_finite(m):
    vals = []
    for fam in TK.FAMILIES:
        vals += [m[fam]["accuracy"], m[fam]["cross_entropy"],
                 m[fam]["macro_accuracy"]]
        for cn in TK.CATEGORIES:
            c = m[fam]["by_category"][cn]
            vals += [v for v in (c["accuracy"], c["cross_entropy"])
                     if v is not None]
    return bool(onp.all(onp.isfinite(onp.asarray(vals, dtype=float))))


def coefficient_report(rule, p, eps_np=None):
    """Executed response coefficients, gate distribution and derived scalars."""
    out = dict(rule=rule)
    if rule in AM.GATED_RULES:
        h32 = jnp.asarray(AM.H32, dtype=p["key_raw"].dtype)
        h8 = jnp.asarray(AM.H8, dtype=p["key_raw"].dtype)
        a = onp.asarray(AG.source_weight_table(p, h32, h8))
        out["source_weight"] = dict(
            min=float(a.min()), median=float(onp.median(a)),
            max=float(a.max()), mean=float(a.mean()),
            fraction_below_one=float(onp.mean(a < 1.0)))
        out["raw"] = {k: float(onp.asarray(p[k]).ravel()[0]) for k in p
                      if k.startswith("raw_")}
    if rule == "adaptive_prospective":
        c = AD.response(p)
        out.update({k: float(onp.asarray(c[k]).ravel()[0]) for k in
                    ("nu", "tau", "rho", "eta", "kappa", "gamma", "M", "T")})
        out["delta_gamma_T_minus_M"] = out["gamma"] * out["T"] - out["M"]
        out["admissible_0_lt_M_lt_gamma_T"] = bool(
            0.0 < out["M"] < out["gamma"] * out["T"])
    elif rule == "adaptive_inertial":
        c = AD.inertial_response(p)
        out.update({k: float(onp.asarray(c[k]).ravel()[0])
                    for k in ("eta", "tau")})
    elif rule == "adaptive_delta":
        # F2: the EXECUTED transform at the executed dtype, not a host
        # NumPy exp of a float-converted raw value. A finite logarithm does
        # not establish a finite executed coefficient.
        out["eta"] = float(jnp.exp(p["raw_eta"]).ravel()[0])
    elif rule == "tss_prospective":
        c = AD.tss_response(p)
        out.update({k: float(onp.asarray(c[k]).ravel()[0]) for k in
                    ("tau_m", "epsilon", "ratio", "M", "T", "gamma")})
        out["adaptation_faster_than_membrane"] = bool(
            out["epsilon"] < out["tau_m"])
        out["outside_the_generalized_sector"] = True
    elif rule == "ideal_projection":
        out["write_strength"] = 1.0
        out["note"] = ("fixed by the equality constraint; no learned response "
                       "scalar, no relaxation factor and no gate leaves")
    elif eps_np is not None:
        from experiments.nested_memory.study import gate_report
        out["gates"] = gate_report(rule, p, eps_np)
    return out


def validate_coefficients(rule, p):
    """F2. Enforce the executed constraints on the LEARNED coefficients.

    The initialized A/B slots are checked separately in the focused checks.
    This is the host-side checkpoint gate: after training, the derived
    physical coefficients must still be finite and must still satisfy the
    sector their own arm declares, BEFORE the checkpoint is accepted for
    selection or evaluation. A violation is a recorded failed run, never a
    silent PASS with a report flag that says otherwise.

    Each arm is checked against ITS OWN constraint. TSS has `gamma = 0` by
    construction and is deliberately outside the generalized candidate's
    `M < gamma T` sector; it is never checked against it.

    Returns `None` when the checkpoint is acceptable, else a reason string.
    """
    def scalars(d):
        return {k: float(jnp.asarray(v).ravel()[0]) for k, v in d.items()}

    if rule == "adaptive_prospective":
        c = scalars(AD.response(p))
        if not all(onp.isfinite(v) for v in c.values()):
            return f"non-finite derived response coefficients: {c}"
        for k in ("nu", "tau", "M", "gamma", "T", "eta", "kappa"):
            if not c[k] > 0.0:
                return f"derived {k} = {c[k]!r} is not positive: {c}"
        if not 0.0 < c["rho"] < 1.0:
            return f"derived rho = {c['rho']!r} left (0, 1): {c}"
        if not c["M"] < c["gamma"] * c["T"]:
            return (f"admissibility violated: M = {c['M']!r} is not below "
                    f"gamma*T = {c['gamma'] * c['T']!r}: {c}")
    elif rule == "adaptive_inertial":
        c = scalars(AD.inertial_response(p))
        if not all(onp.isfinite(v) for v in c.values()):
            return f"non-finite derived inertial coefficients: {c}"
        if not (c["eta"] > 0.0 and c["tau"] > 0.0):
            return f"derived inertial coefficients not positive: {c}"
    elif rule == "adaptive_delta":
        eta = float(jnp.exp(p["raw_eta"]).ravel()[0])
        if not (onp.isfinite(eta) and eta > 0.0):
            return f"derived eta = {eta!r} is not finite and positive"
    elif rule == "tss_prospective":
        c = scalars(AD.tss_response(p))
        if not all(onp.isfinite(v) for v in c.values()):
            return f"non-finite derived TSS coefficients: {c}"
        if not c["tau_m"] > 0.0:
            return f"derived tau_m = {c['tau_m']!r} is not positive: {c}"
        if not 0.0 < c["epsilon"] < c["tau_m"]:
            return (f"finite adaptation is not strictly faster than the "
                    f"membrane: epsilon = {c['epsilon']!r}, "
                    f"tau_m = {c['tau_m']!r}")
        if not (c["M"] > 0.0 and c["T"] > 0.0):
            return f"derived TSS M or T is not positive: {c}"
        if c["gamma"] != 0.0:
            return (f"TSS gamma must remain exactly 0 by construction, got "
                    f"{c['gamma']!r}")
        # deliberately NOT checked against M < gamma*T: this reference lies
        # outside the generalized candidate's sector, and that is intended
    return None


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


def config_hash(rule, slot, seed):
    h = hashlib.sha256()
    h.update(repr((rule, slot["config"], slot["lr"], seed, UPDATES,
                   BATCH_PER_FAMILY, GRAD_CLIP, TK.SEQ_LEN, STREAM,
                   sorted((k, float(v)) for k, v in slot.items()
                          if isinstance(v, (int, float))))).encode())
    return h.hexdigest()[:16]


def run_one(slot, seed, val_np, updates, out, deadline, reserve_s, status, tag):
    """Train one configuration from scratch. Returns (record, params) or None."""
    rule = slot["rule"]
    lr = jnp.asarray(slot["lr"], dtype=jnp.float32)
    p = AM.init_params(rule, seed, init_coeffs=slot)
    opt = TX.init(p)
    curve, val_hist = [], []
    t0 = time.time()
    loss = acc = gn = un = wn = an = jnp.zeros(())
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
                f"{tag}:{rule}/{slot['config']}/seed{seed} stopped at "
                f"update {u} of {updates}; NOT reduced by choice")
            return None
        eps = to_jax(TK.generate_batch(train_stream_seed(seed, u),
                                       BATCH_PER_FAMILY))
        p, opt, loss, acc, gn, un, wn, an = train_step(rule, p, opt, eps, lr)
        if u % 50 == 0:
            curve.append(dict(update=u, loss=float(loss), train_acc=float(acc),
                              grad_norm=float(gn), update_norm=float(un),
                              w_norm=float(wn), aux_norm=float(an)))
    final = val_hist[-1]["full"]
    bad = None
    if not all(onp.isfinite([c["loss"] for c in curve])):
        bad = "non-finite sampled training loss"
    elif not all(onp.isfinite([float(loss), float(acc), float(gn), float(un),
                               float(wn), float(an)])):
        bad = "non-finite final training scalars"
    elif not all_finite(p):
        bad = "non-finite final parameters"
    elif not metrics_finite(final):
        bad = "non-finite validation metric"
    else:
        # F2: the learned coefficients, not the initialized fixture
        bad = validate_coefficients(rule, p)
    stem = f"{tag}_{rule}_{slot['config']}_seed{seed}"
    save_tree(os.path.join(out, "params", stem + ".msgpack"), p)
    save_tree(os.path.join(out, "params", stem + "_opt.msgpack"), opt)
    rec = dict(
        tag=tag, rule=rule, display=AD.DISPLAY[rule], config=slot["config"],
        lr=slot["lr"], seed=seed, wall_s=time.time() - t0, curve=curve,
        validation=[{k: v for k, v in h.items() if k != "full"}
                    for h in val_hist],
        final_validation=final,
        training_gain=dict(
            primary=final["primary"] - val_hist[0]["primary"],
            revision_ce=final["revision_ce"] - val_hist[0]["revision_ce"],
            recall=final["recall_overall"] - val_hist[0]["recall"]),
        params=AM.parameter_counts(rule, p), carry=AD.CARRY[rule],
        config_hash=config_hash(rule, slot, seed),
        initialization=slot,
        coefficients_final=coefficient_report(rule, p, val_np),
        invalid=bad)
    return rec, p


# ------------------------------------------------------------- preflight ---
def preflight(cfg, val_np, status):
    """Project the WHOLE declared batch: 35 runs, all evaluations, both splits.

    Compilation already performed here is reported as incurred; every training
    and evaluation shape in the batch is identical to one compiled here, so no
    further compilation is outstanding. A retrace during step timing makes the
    projection untrustworthy and refuses the batch.
    """
    eps = to_jax(TK.generate_batch(train_stream_seed(DEV_SEED, 0),
                                   BATCH_PER_FAMILY))
    rows, total, incurred, retraced_any = [], 0.0, 0.0, False
    n_runs_per_rule = 2 + len(FINAL_SEEDS)          # 2 development + 3 final
    # 7 arms x 5 runs = 35 runs, 7,000 optimizer updates
    for rule in AD.RULES:
        slot = cfg["slots"][f"{rule}/A"]
        lr = jnp.asarray(slot["lr"], dtype=jnp.float32)
        p = AM.init_params(rule, DEV_SEED, init_coeffs=slot)
        opt = TX.init(p)
        t0 = time.time()
        out5 = train_step(rule, p, opt, eps, lr)
        jax.block_until_ready(out5[2])
        compile_s = time.time() - t0
        p2, opt2 = out5[0], out5[1]
        out5 = train_step(rule, p2, opt2, eps, lr)     # consume own output
        jax.block_until_ready(out5[2])
        p2, opt2 = out5[0], out5[1]
        n_before = train_step._cache_size()
        t1 = time.time()
        for _ in range(5):
            out5 = train_step(rule, p2, opt2, eps, lr)
            p2, opt2 = out5[0], out5[1]
        jax.block_until_ready(out5[2])
        step_s = (time.time() - t1) / 5.0
        retraced = train_step._cache_size() != n_before
        t2 = time.time()
        evaluate(rule, p, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        evaluate(rule, p, val_np)
        eval_s = time.time() - t3
        incurred += compile_s + eval_compile_s
        # per rule: 5 runs x (200 steps + 3 validation passes)
        #         + 3 final held-out passes at twice the validation size
        arm_s = (n_runs_per_rule * (UPDATES * step_s + len(VAL_AT) * eval_s)
                 + len(FINAL_SEEDS) * 2.0 * eval_s)
        total += arm_s
        retraced_any |= bool(retraced)
        rows.append(dict(rule=rule, display=AD.DISPLAY[rule],
                         compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s,
                         step_s=step_s, eval_s=eval_s, arm_remaining_s=arm_s,
                         runs=n_runs_per_rule,
                         retraced_during_timing=bool(retraced),
                         params=AM.parameter_counts(rule, p),
                         carry=AD.CARRY[rule]))
        print(f"[preflight] {AD.DISPLAY[rule]:<48} compile {compile_s:5.1f}s"
              f" (+eval {eval_compile_s:4.1f}s)  step {step_s*1e3:7.2f}ms"
              f"  eval {eval_s*1e3:7.1f}ms  arm {arm_s:6.1f}s"
              f"  params {AM.parameter_counts(rule, p)['total']}"
              f"  carry {AD.CARRY[rule]}")
        if retraced:
            print(f"[!] {rule}: RETRACE during step timing")
        del p2, opt2
    host_s = 45.0                     # ALLOWANCE, not a measurement
    total += host_s
    status["preflight"] = dict(
        rows=rows, host_allowance_s=host_s, incurred_compilation_s=incurred,
        projected_remaining_s=total, retraced_any=retraced_any,
        runs_projected=n_runs_per_rule * len(AD.RULES),
        scope=("PROJECTED REMAINING work: 14 development + 21 final runs at "
               f"{UPDATES} updates, three validation passes per run, 21 "
               "held-out passes and a host allowance. Compilation already "
               "performed here is reported separately as incurred; every "
               "shape in the batch is identical to one compiled here."))
    print(f"PREFLIGHT_INCURRED_COMPILATION_S={incurred:.1f}")
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, retraced_any


# ------------------------------------------------------------- selection ---
def select(dev_rows, status):
    """Highest revision macro accuracy, then lower revision CE, then A.

    Development validation only. Held-out results are not open here, and no
    category-specific preference enters.
    """
    sel, table = {}, []
    for rule in AD.RULES:
        cand = [r for r in dev_rows if r["rule"] == rule]
        if len(cand) != 2:
            return None, table
        ranked = sorted(cand, key=lambda r: (-r["final_validation"]["primary"],
                                             r["final_validation"]["revision_ce"],
                                             r["config"]))
        sel[rule] = ranked[0]["config"]
        table.append(dict(
            rule=rule, display=AD.DISPLAY[rule], chosen=ranked[0]["config"],
            candidates=[dict(config=r["config"], lr=r["lr"],
                             primary=r["final_validation"]["primary"],
                             revision_ce=r["final_validation"]["revision_ce"],
                             retention=r["final_validation"][
                                 "retention_revision_untouched"],
                             recall=r["final_validation"]["recall_overall"])
                        for r in cand],
            rule_applied=("highest revision macro accuracy, then lower "
                          "revision CE, then configuration A")))
    status["selection"] = dict(
        selected=sel, table=table,
        basis="development seed 200, update-200 checkpoint, 256 validation "
              "sequences per family",
        note="frozen before any final seed is trained; held-out not opened")
    return sel, table


# ---------------------------------------------------------------- screen ---
def screen(final_rows, status):
    """The predeclared screens. Reported, never loosened.

    TWO independent verdicts, plus attribution:

    * the literature screen, against Momentum DeltaNet (primary) and Gated
      DeltaNet;
    * the ordinary-prospectivity extension screen, against the TSS
      finite-adaptation reference and the ideal prospective equilibrium.

    A win on one CANNOT substitute for a loss on the other, in either
    direction. Both are reported whatever they say.
    """
    def mean(rule, key):
        return float(onp.mean([r["heldout"][key] for r in final_rows
                               if r["rule"] == rule]))

    def per_seed(rule, key):
        return {r["seed"]: r["heldout"][key] for r in final_rows
                if r["rule"] == rule}

    cand = "adaptive_prospective"

    def compare(other):
        dm = mean(cand, "primary") - mean(other, "primary")
        a, b = per_seed(cand, "primary"), per_seed(other, "primary")
        paired = {s: a[s] - b[s] for s in sorted(a) if s in b}
        d_ret = mean(cand, "retention_revision_untouched") \
            - mean(other, "retention_revision_untouched")
        d_rec = mean(cand, "recall_overall") - mean(other, "recall_overall")
        # R2. NEITHER metric may regress by more than one point. The earlier
        # `not (d_ret < -0.01 and d_rec < -0.01)` rejected a candidate only
        # when BOTH regressed, so -20 points of retention with unchanged
        # recall would have passed. The coordinator has resolved the original
        # "or" wording; the one-point threshold is unchanged.
        safe = (d_ret >= -0.01) and (d_rec >= -0.01)
        # every declared paired seed must be present: an empty or partial
        # `all()` must never produce a comparative verdict
        complete_pairs = (set(paired) == set(FINAL_SEEDS))
        ok = (complete_pairs and dm >= 0.01
              and all(v > 0 for v in paired.values()) and safe)
        return dict(
            against=other, display=AD.DISPLAY[other],
            mean_primary_candidate=mean(cand, "primary"),
            mean_primary_other=mean(other, "primary"),
            mean_primary_difference=dm, meets_plus_one_point=bool(dm >= 0.01),
            paired_primary_differences=paired,
            positive_in_all_seeds=bool(all(v > 0 for v in paired.values())),
            retention_difference=d_ret, recall_difference=d_rec,
            retention_safeguard_met=bool(safe),
            retention_within_one_point=bool(d_ret >= -0.01),
            recall_within_one_point=bool(d_rec >= -0.01),
            paired_seeds=sorted(paired), complete_paired_seeds=complete_pairs,
            passed=bool(ok))

    lit = [compare(o) for o in ("momentum_delta", "gated_delta")]
    ordn = [compare(o) for o in AD.ORDINARY_PROSPECTIVE]
    res = dict(candidate=cand, comparisons=lit,
               ordinary_prospectivity=ordn,
               literature_screen_passed=bool(all(c["passed"] for c in lit)),
               ordinary_prospectivity_screen_passed=bool(
                   all(c["passed"] for c in ordn)))

    # attribution: the prospective derivative, against the equally gated
    # controls. Kept SEPARATE from both screens.
    res["attribution"] = []
    for other in ("adaptive_inertial", "adaptive_delta"):
        a, b = per_seed(cand, "primary"), per_seed(other, "primary")
        paired = {s: a[s] - b[s] for s in sorted(a) if s in b}
        d_ret = mean(cand, "retention_revision_untouched") \
            - mean(other, "retention_revision_untouched")
        d_rec = mean(cand, "recall_overall") - mean(other, "recall_overall")
        complete_pairs = (set(paired) == set(FINAL_SEEDS))
        res["attribution"].append(dict(
            against=other, display=AD.DISPLAY[other],
            paired_primary_differences=paired,
            positive_in_all_seeds=bool(all(v > 0 for v in paired.values())),
            retention_difference=d_ret, recall_difference=d_rec,
            complete_paired_seeds=complete_pairs,
            exceeds=bool(complete_pairs
                         and all(v > 0 for v in paired.values())
                         and d_ret >= -0.01 and d_rec >= -0.01)))
    # R2. Attribution is reported INDEPENDENTLY of the literature screen, as
    # promised. An advantage over the equally source-gated inertial control
    # stays interpretable even when a stronger literature method wins; it does
    # NOT imply competitive performance and does not establish unique
    # causation.
    res["prospective_term_credited"] = bool(res["attribution"][0]["exceeds"])
    res["attribution_is_independent_of_the_literature_screen"] = True

    res["per_arm_means"] = {
        r: dict(primary=mean(r, "primary"),
                retention=mean(r, "retention_revision_untouched"),
                recall=mean(r, "recall_overall"),
                revision_ce=mean(r, "revision_ce"),
                per_seed_primary=per_seed(r, "primary"))
        for r in AD.RULES}
    res["note"] = (
        "TWO SEPARATE VERDICTS. The literature screen and the "
        "ordinary-prospectivity extension screen are independent; neither "
        "substitutes for the other. Both are bounded development screens "
        "against freshly trained, equally selected baselines - not "
        "statistical significance and not a state-of-the-art claim. Every "
        "arm here is newly trained in this batch, so no previously reported "
        "score is used as a control. The inertial and prospective arms are "
        "independently calibrated and trained, so their difference compares "
        "optimized rule families and is not a term-removal ablation of one "
        "trained trajectory. Neither ordinary reference is assumed memoryless "
        "or assumed to score worse.")
    status["screen"] = res
    return res


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/adaptive-memory")
    ap.add_argument("--run_id", default=None)
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
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)

    # ---- calibration, before any data is touched -------------------------
    try:
        cfg = CAL.configurations()
    except (FloatingPointError, OverflowError, ValueError,
            ArithmeticError) as exc:
        # R1. An arithmetic failure in the host solve is recorded as a FAILED
        # CHECK with its exception type, NOT as evidence that the physical
        # initialization is impossible. The two are different findings and are
        # labelled differently.
        kind = ("initialization-design obstruction"
                if isinstance(exc, FloatingPointError)
                else "host arithmetic failure in the calibration solve")
        write(os.path.join(out, "status.json"),
              dict(run_id=run_id,
                   failed=f"calibration: {kind}: {type(exc).__name__}: {exc}",
                   failure_class=kind, exception=type(exc).__name__,
                   note="a host arithmetic failure is a defect in the solver, "
                        "not a demonstration that the declared initialization "
                        "cannot exist"))
        print(f"[FAIL] calibration {kind}: {type(exc).__name__}: {exc}")
        print(f"ADAPTIVE_STATUS=FAILED out={out}")
        return 4
    if not cfg["reference_recovery"]["passed"]:
        write(os.path.join(out, "status.json"),
              dict(run_id=run_id, calibration=cfg,
                   failed="the short prospective slot did not recover "
                          "nu = 4/3 within 1e-8"))
        print(f"[FAIL] reference recovery: {cfg['reference_recovery']}")
        print(f"ADAPTIVE_STATUS=FAILED out={out}")
        return 4

    val_np = TK.generate_batch(STREAM["dev_validation"], VAL_PER_FAMILY)
    eval_val_np = TK.generate_batch(STREAM["eval_validation"], VAL_PER_FAMILY)
    held_np = TK.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    struct = TK.structure_check(val_np)

    status = dict(
        run_id=run_id, out=out, backend=backend, dev_seed=DEV_SEED,
        final_seeds=list(FINAL_SEEDS), updates=args.updates,
        batch_per_family=BATCH_PER_FAMILY, grad_clip=GRAD_CLIP,
        budget_s=args.budget_s, reserve_s=args.reserve_s, streams=STREAM,
        stream_disjointness=dict(
            previous_study=dict(common=0, train_base=1_000_000,
                                validation=7_000_000, heldout=9_000_000),
            this_study_train_range=[train_stream_seed(min(DEV_SEED,
                                                          *FINAL_SEEDS), 0),
                                    train_stream_seed(max(DEV_SEED,
                                                          *FINAL_SEEDS),
                                                      args.updates)],
            checked="no overlap between any two named streams here, and none "
                    "with the completed study's logged streams"),
        arms={r: AD.DISPLAY[r] for r in AD.RULES},
        calibration=cfg,
        gate=dict(parameters=(TK.N_KEYS - 1) + (TK.N_VALUES - 1),
                  form="a = 2 * sigmoid((H32 u)_k + (H8 b)_v), strictly "
                       "positive, initialized at u = b = 0 so a = 1 exactly",
                  contrast=dict(H32=AG.check_contrast(AM.H32),
                                H8=AG.check_contrast(AM.H8))),
        task=dict(structure=struct,
                  dev_validation_digest=TK.episode_digest(val_np),
                  eval_validation_digest=TK.episode_digest(eval_val_np),
                  heldout_digest=TK.episode_digest(held_np),
                  oracle_exact=TK.oracle_is_exact(val_np)),
        heldout_policy=(
            "held-out episodes are generated and hashed BEFORE training from "
            "their own named stream; their EVALUATION is DEFERRED until all "
            "all 21 final runs finish. Deferred evaluation, not data created "
            "after training."),
        selection_policy=(
            "two slots per family, one development seed, the same update "
            "count and the same validation set. The axes searched differ "
            "across families: this is an EQUAL SELECTION BUDGET, not "
            "exhaustive or identical hyperparameter tuning."),
        development=[], final=[], incomplete=[])
    print(f"[*] out={out} backend={backend}")
    print(f"[*] beta_star={cfg['beta_star']:.10f}  "
          f"nu_A={cfg['slots']['adaptive_prospective/A']['nu']:.12f} "
          f"(|.-4/3|={cfg['reference_recovery']['absolute_error']:.2e})")
    for k in sorted(cfg["slots"]):
        s = cfg["slots"][k]
        extra = " ".join(f"{n}={s[n]:.6g}" for n in ("nu", "eta", "tau", "rho")
                         if n in s)
        print(f"    {k:<28} lr={s['lr']:<6g} {extra}")
    print(f"[*] queries/seq {struct['queries_per_sequence']}  "
          f"oracle exact: {status['task']['oracle_exact']}")
    write(os.path.join(out, "status.json"), status)

    proj, retraced = preflight(cfg, val_np, status)
    write(os.path.join(out, "status.json"), status)
    if retraced:
        status["incomplete"].append(
            "a retrace was detected during preflight step timing; the "
            "projection is untrustworthy and the batch was NOT started")
        write(os.path.join(out, "status.json"), status)
        print(f"ADAPTIVE_STATUS=INCOMPLETE out={out}")
        return 3
    if args.preflight_only:
        print(f"ADAPTIVE_STATUS=PREFLIGHT_ONLY out={out}")
        return 0
    if proj > left():
        status["incomplete"].append(
            f"projected {proj:.0f}s > remaining {left():.0f}s; the batch was "
            f"NOT started. No arm, slot, seed or update count was reduced.")
        write(os.path.join(out, "status.json"), status)
        print(f"[!] {status['incomplete'][-1]}")
        print(f"ADAPTIVE_STATUS=INCOMPLETE out={out}")
        return 3

    # ---- stage 1: 14 development runs ------------------------------------
    dev_rows = []
    for rule in AD.RULES:
        for tag in ("A", "B"):
            slot = cfg["slots"][f"{rule}/{tag}"]
            r = run_one(slot, DEV_SEED, val_np, args.updates, out, deadline,
                        args.reserve_s, status, "dev")
            if r is None:
                break
            rec, _ = r
            if rec["invalid"]:
                status["failed"] = f"dev {rule}/{tag}: {rec['invalid']}"
                write(os.path.join(out, "status.json"), status)
                print(f"[FAIL] {status['failed']}")
                print(f"ADAPTIVE_STATUS=FAILED out={out}")
                return 4
            dev_rows.append(rec)
            status["development"] = dev_rows
            write(os.path.join(out, "development.json"), dev_rows)
            write(os.path.join(out, "status.json"), status)
            print(f"  [dev {rule}/{tag}] primary="
                  f"{rec['final_validation']['primary']:.4f}  "
                  f"revCE={rec['final_validation']['revision_ce']:.4f}  "
                  f"({rec['wall_s']:.0f}s)")

    if len(dev_rows) != 2 * len(AD.RULES):
        status["incomplete"].append(
            f"only {len(dev_rows)} of {2 * len(AD.RULES)} development runs "
            f"finished; no "
            "selection was frozen and no final seed was trained")
        status["heldout_opened"] = False
        status["wall_s"] = time.time() - t0
        write(os.path.join(out, "status.json"), status)
        print(f"ADAPTIVE_STATUS=INCOMPLETE out={out}")
        return 3

    sel, table = select(dev_rows, status)
    write(os.path.join(out, "selection.json"), status["selection"])
    write(os.path.join(out, "status.json"), status)
    print("\n[selection]")
    for row in table:
        print(f"  {row['display']:<48} -> {row['chosen']}")

    # ---- stage 2: 21 final runs, freshly trained from scratch -----------
    final_rows, finals = [], {}
    for rule in AD.RULES:
        slot = cfg["slots"][f"{rule}/{sel[rule]}"]
        for seed in FINAL_SEEDS:
            r = run_one(slot, seed, eval_val_np, args.updates, out, deadline,
                        args.reserve_s, status, "final")
            if r is None:
                break
            rec, p = r
            if rec["invalid"]:
                status["failed"] = f"final {rule}/seed{seed}: {rec['invalid']}"
                write(os.path.join(out, "status.json"), status)
                print(f"[FAIL] {status['failed']}")
                print(f"ADAPTIVE_STATUS=FAILED out={out}")
                return 4
            final_rows.append(rec)
            finals[(rule, seed)] = p
            status["final"] = final_rows
            write(os.path.join(out, "final.json"), final_rows)
            write(os.path.join(out, "status.json"), status)
            print(f"  [final {rule}/{sel[rule]}/seed{seed}] primary="
                  f"{rec['final_validation']['primary']:.4f}  "
                  f"({rec['wall_s']:.0f}s)")

    complete = (len(final_rows) == len(AD.RULES) * len(FINAL_SEEDS)
                and not status["incomplete"])
    if complete:
        for rec in final_rows:
            rec["heldout"] = evaluate(rec["rule"],
                                      finals[(rec["rule"], rec["seed"])],
                                      held_np)
            if not metrics_finite(rec["heldout"]):
                status["failed"] = (f"{rec['rule']}/seed{rec['seed']}: "
                                    f"non-finite held-out metric")
                write(os.path.join(out, "status.json"), status)
                print(f"ADAPTIVE_STATUS=FAILED out={out}")
                return 4
        status["heldout_opened"] = True
        status["final"] = final_rows
        res = screen(final_rows, status)
        print("\n  arm                                          primary  "
              "retention  recall")
        for rule in AD.RULES:
            rs = [r for r in final_rows if r["rule"] == rule]
            print(f"  {AD.DISPLAY[rule]:<44}"
                  f"{onp.mean([r['heldout']['primary'] for r in rs]):9.4f}"
                  f"{onp.mean([r['heldout']['retention_revision_untouched'] for r in rs]):11.4f}"
                  f"{onp.mean([r['heldout']['recall_overall'] for r in rs]):8.4f}")
        print(f"\n  LITERATURE SCREEN: "
              f"{'PASSED' if res['literature_screen_passed'] else 'FAILED'}")
        for c in res["comparisons"]:
            print(f"    vs {c['display']:<42} "
                  f"dmean={c['mean_primary_difference']:+.4f}"
                  f"  all-seed positive={c['positive_in_all_seeds']}"
                  f"  retention safeguard={c['retention_safeguard_met']}")
        print(f"  ORDINARY-PROSPECTIVITY SCREEN: "
              f"{'PASSED' if res['ordinary_prospectivity_screen_passed'] else 'FAILED'}")
        for c in res["ordinary_prospectivity"]:
            print(f"    vs {c['display']:<42} "
                  f"dmean={c['mean_primary_difference']:+.4f}"
                  f"  all-seed positive={c['positive_in_all_seeds']}"
                  f"  retention safeguard={c['retention_safeguard_met']}")
        print(f"  prospective term credited: "
              f"{res['prospective_term_credited']}  (attribution, separate)")
    else:
        status["heldout_opened"] = False
        status["incomplete"].append(
            "held-out evaluation NOT opened: an incomplete batch has no "
            "comparative verdict, and no partial result licenses changing "
            "the candidate or the selection rule")

    status["wall_s"] = time.time() - t0
    status["complete"] = complete
    write(os.path.join(out, "final.json"), final_rows)
    write(os.path.join(out, "status.json"), status)
    print(f"[*] wall {status['wall_s']:.0f}s  dev {len(dev_rows)}/"
          f"{2 * len(AD.RULES)}  final {len(final_rows)}/"
          f"{len(AD.RULES) * len(FINAL_SEEDS)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[!] {m}")
        print(f"ADAPTIVE_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"ADAPTIVE_STATUS=PASS out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
